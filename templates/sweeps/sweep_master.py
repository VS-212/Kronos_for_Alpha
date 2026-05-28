"""
TEMPLATE: sweep_master.py — General-purpose grid search sweep
Source:  Replaces all individual sweep_*.py templates
Purpose: YAML-driven parameter sweep over any strategy family
Usage:   python -m templates.sweeps.sweep_master --config sweep_config.yaml
Status:  Reference example — adapt to your instrument before production use
"""

import argparse
import importlib
import itertools
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.evaluation.output import load_samples
from src.strategies.core import load_config, load_mamba_sber, report


def _cartesian_product(grid: dict) -> list[dict]:
    """Expand a grid dict into a list of parameter combinations."""
    if not grid:
        return [{}]
    keys = list(grid.keys())
    values = [grid[k] if isinstance(grid[k], list) else [grid[k]] for k in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def _load_strategy(strategy_cfg: dict):
    """Dynamically import a strategy function."""
    mod = importlib.import_module(strategy_cfg["import_path"])
    return getattr(mod, strategy_cfg["function"])


def _resolve_params(fixed: dict, defaults: dict, grid_combo: dict) -> dict:
    """Merge defaults, fixed, and grid-combo params (later overrides earlier)."""
    merged = {}
    if defaults:
        merged.update(defaults)
    if fixed:
        merged.update(fixed)
    merged.update(grid_combo)
    return merged


def _params_str(params: dict) -> str:
    """Compact param display string."""
    parts = []
    for k, v in params.items():
        if isinstance(v, float):
            parts.append(f"{k}={v:.4f}")
        else:
            parts.append(f"{k}={v}")
    return " ".join(parts)


def _run_one_config(
    sname: str,
    func,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    mamba,
    pred_len: int,
    sample_count: int,
    tp_q: float,
    sl_q: float,
    params: dict,
) -> dict | None:
    try:
        tr = func(
            train_df, mamba,
            pred_len=pred_len, sample_count=sample_count,
            tp_q=tp_q, sl_q=sl_q, **params,
        )
    except Exception as e:
        print(f"  [{sname}] {_params_str(params)} → ERROR: {e}")
        return None

    tr_rep = report(tr) if len(tr) > 0 else {"error": "no trades", "n": 0}
    if tr_rep.get("error"):
        return {
            "strategy": sname, "params": _params_str(params),
            "n_tr": 0, "wr_tr": 0, "ret_tr": 0, "sharpe_tr": 0, "mdd_tr": 0, "pf_tr": 0,
            "n_te": 0, "wr_te": 0, "ret_te": 0, "sharpe_te": 0, "mdd_te": 0, "pf_te": 0,
        }

    try:
        te = func(
            test_df, mamba,
            pred_len=pred_len, sample_count=sample_count,
            tp_q=tp_q, sl_q=sl_q, **params,
        )
    except Exception:
        te_rep = {"error": "eval failed", "n": 0, "sharpe": 0,
                  "total_return_pct": 0, "mdd_pct": 0, "win_rate": 0, "profit_factor": 0}
    else:
        te_rep = report(te) if len(te) > 0 else {"error": "no trades", "n": 0,
                                                  "sharpe": 0, "total_return_pct": 0,
                                                  "mdd_pct": 0, "win_rate": 0, "profit_factor": 0}

    return {
        "strategy": sname, "params": _params_str(params),
        "n_tr": tr_rep["n"], "wr_tr": tr_rep["win_rate"],
        "ret_tr": tr_rep["total_return_pct"], "sharpe_tr": tr_rep["sharpe"],
        "mdd_tr": tr_rep["mdd_pct"], "pf_tr": tr_rep["profit_factor"],
        "n_te": te_rep.get("n", 0), "wr_te": te_rep.get("win_rate", 0),
        "ret_te": te_rep.get("total_return_pct", 0), "sharpe_te": te_rep.get("sharpe", 0),
        "mdd_te": te_rep.get("mdd_pct", 0), "pf_te": te_rep.get("profit_factor", 0),
    }


def _print_table(results: list[dict]):
    W = 140
    print("\n" + "=" * W)
    print(f"{'Strategy':<22} {'Params':<35} {'Trn':>5} {'TrWR%':>7} {'TrRet%':>8} "
          f"{'TrSh':>7} {'TrMDD%':>7} {'TrPF':>6} | "
          f"{'TeN':>5} {'TeWR%':>7} {'TeRet%':>8} {'TeSh':>7} {'TeMDD%':>7} {'TePF':>6}")
    print("=" * W)

    for r in results:
        print(f"{r['strategy']:<22} {r['params']:<35} "
              f"{r['n_tr']:>5} {r['wr_tr']:>6.1f}% {r['ret_tr']:>+7.2f} "
              f"{r['sharpe_tr']:>6.2f} {r['mdd_tr']:>+6.2f} {r['pf_tr']:>5.2f} | "
              f"{r['n_te']:>5} {r['wr_te']:>6.1f}% {r['ret_te']:>+7.2f} "
              f"{r['sharpe_te']:>6.2f} {r['mdd_te']:>+6.2f} {r['pf_te']:>5.2f}")

    print("\n" + "=" * W)
    print("BEST PER STRATEGY (by Test Sharpe)")
    print("=" * W)

    strategies_seen = sorted(set(r["strategy"] for r in results))
    for sname in strategies_seen:
        subset = [r for r in results if r["strategy"] == sname and r["n_te"] > 0]
        if not subset:
            print(f"  {sname:<22} — no trades on test")
            continue
        best = max(subset, key=lambda x: (x["sharpe_te"], x["n_te"]))
        print(f"  {sname:<22} {best['params']:<35} "
              f"tr S={best['sharpe_tr']:.2f} R={best['ret_tr']:+.2f}% n={best['n_tr']:>5} | "
              f"te S={best['sharpe_te']:.2f} R={best['ret_te']:+.2f}% n={best['n_te']:>5}")


def main():
    parser = argparse.ArgumentParser(description="Sweep Master — YAML-driven grid search")
    parser.add_argument("--config", type=str, default="sweep_config.yaml",
                        help="Path to YAML config (default: sweep_config.yaml)")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        config_path = Path(__file__).resolve().parent / args.config
    if not config_path.exists():
        print(f"ERROR: Config file not found: {args.config}")
        sys.exit(1)

    with open(config_path) as f:
        sweep_cfg = yaml.safe_load(f)

    samples_path = sweep_cfg["samples_path"]
    if not Path(samples_path).exists():
        print(f"ERROR: Samples file not found: {samples_path}")
        sys.exit(1)

    sc = sweep_cfg.get("samples_config", {})
    pred_len = sc.get("pred_len", 6)
    sample_count = sc.get("sample_count", 5)
    tp_q = sc.get("tp_q", 0.90)
    sl_q = sc.get("sl_q", 0.10)

    train_period = sweep_cfg.get("train_period", "2025")
    test_period = sweep_cfg.get("test_period", "2026")

    defaults = sweep_cfg.get("defaults", {})
    output_cfg = sweep_cfg.get("output", {})

    print(f"Loading samples: {samples_path}")
    df = load_samples(samples_path)
    df["year"] = df["month"].str[:4]
    train_df = df[df["year"] == train_period]
    test_df = df[df["year"] == test_period]
    print(f"Train ({train_period}): {len(train_df)} | Test ({test_period}): {len(test_df)}\n")

    print("Loading Mamba OHLCV...")
    cfg = load_config()
    mamba = load_mamba_sber(cfg)
    print(f"Mamba rows: {len(mamba)}\n")

    all_results = []

    for strategy_cfg in sweep_cfg["strategies"]:
        sname = strategy_cfg["name"]
        grid = strategy_cfg.get("grid", {})
        fixed = strategy_cfg.get("fixed", {})
        func = _load_strategy(strategy_cfg)

        combos = _cartesian_product(grid) if grid else [{}]

        if not grid:
            # Fixed params only — baseline run
            params = _resolve_params(fixed, defaults, {})
            print(f"Running {sname} (fixed params: {_params_str(params)})...")
            result = _run_one_config(
                sname, func, train_df, test_df, mamba,
                pred_len, sample_count, tp_q, sl_q, params,
            )
            if result:
                all_results.append(result)
        else:
            print(f"Sweeping {sname} — {len(combos)} config(s)...")
            for i, combo in enumerate(combos):
                params = _resolve_params(fixed, defaults, combo)
                result = _run_one_config(
                    sname, func, train_df, test_df, mamba,
                    pred_len, sample_count, tp_q, sl_q, params,
                )
                if result:
                    all_results.append(result)
                    r = result
                    print(f"  [{i+1}/{len(combos)}] {_params_str(combo):>40s}  "
                          f"Tr n={r['n_tr']:>4} Sh={r['sharpe_tr']:>5.2f} | "
                          f"Te n={r['n_te']:>4} Sh={r['sharpe_te']:>5.2f}")

    if not all_results:
        print("No results generated.")
        sys.exit(1)

    if output_cfg.get("print_table", True):
        _print_table(all_results)

    if output_cfg.get("save_json", False):
        json_path = output_cfg.get("json_path", "sweep_results.json")
        with open(json_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\nResults saved to {json_path}")


if __name__ == "__main__":
    main()
