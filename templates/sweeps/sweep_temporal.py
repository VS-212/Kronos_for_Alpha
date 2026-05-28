"""
TEMPLATE: sweep_temporal.py — Hour-of-day and quarterly diagnostics
Source:  Replaces sweep_hour5.py and sweep_hour_dist.py
Purpose: Per-hour trade metrics + hour×quarter pivot tables per strategy
Usage:   python -m templates.sweeps.sweep_temporal \
           --samples_path /path/to/samples.parquet \
           --strategies S01_BB,S20_OB,S02_BB_MR \
           --mode both
Status:  Reference example — adapt to your instrument before production use
"""

import argparse
import importlib
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.evaluation.output import load_samples
from src.strategies.core import load_config, load_mamba_sber, report


PRED_LEN = 6
SAMPLE_COUNT = 5
TP_Q = 0.90
SL_Q = 0.10


STRATEGY_REGISTRY = {
    "S01_BB": {
        "import_path": "src.strategies.s01_bb",
        "function": "run",
        "params": {"consensus_threshold": 0.6},
    },
    "S02_BB_MR": {
        "import_path": "src.strategies.s02_bb_mr",
        "function": "run",
        "params": {"consensus_threshold": 0.6},
    },
    "S05_BB_BREAKOUT": {
        "import_path": "src.strategies.s05_bb_breakout",
        "function": "run",
        "params": {"consensus_threshold": 0.6},
    },
    "S20_OB": {
        "import_path": "src.strategies.s20_ob",
        "function": "run",
        "params": {"lookback": 48, "move_threshold": 0.005, "max_age": 24, "consensus_threshold": 0.6},
    },
    "S28_VOL_OB": {
        "import_path": "src.strategies.s28_vol_ob",
        "function": "run",
        "params": {"consensus_threshold": 0.6},
    },
    "S34_VWAP_OB": {
        "import_path": "src.strategies.s34_vwap_ob",
        "function": "run",
        "params": {"consensus_threshold": 0.6},
    },
    "S38_LOWVOL_OB": {
        "import_path": "src.strategies.s38_lowvol_ob",
        "function": "run",
        "params": {"consensus_threshold": 0.6},
    },
    "VANILLA": {
        "import_path": "src.strategies.vanilla",
        "function": "run",
        "params": {"consensus_threshold": 0.6},
    },
}


def _load_strategy(name: str):
    """Load strategy function from registry."""
    if name not in STRATEGY_REGISTRY:
        available = ", ".join(STRATEGY_REGISTRY.keys())
        raise ValueError(f"Unknown strategy '{name}'. Available: {available}")
    entry = STRATEGY_REGISTRY[name]
    mod = importlib.import_module(entry["import_path"])
    return getattr(mod, entry["function"]), entry["params"]


def _extract_hour(trades: pd.DataFrame) -> pd.DataFrame:
    """Add hour_of_day column from pred_ts."""
    t = trades.copy()
    t["hour_of_day"] = pd.to_datetime(t["pred_ts"]).dt.hour
    t["quarter"] = trades["quarter"]
    return t


def _compute_hourly_metrics(trades: pd.DataFrame, total_trades: int) -> pd.DataFrame:
    """Compute per-hour metrics: n, trade_pct, win_rate, avg_return, sharpe,
    profit_factor, tp_pct, sl_pct, close_pct."""
    if len(trades) == 0:
        return pd.DataFrame()

    t = _extract_hour(trades)
    rows = []
    for hour, grp in t.groupby("hour_of_day", sort=True):
        r = report(grp)
        n = len(grp)
        rows.append({
            "hour": hour,
            "n": n,
            "trade_pct": n / max(total_trades, 1) * 100,
            "win_rate": r.get("win_rate", 0) * 100,
            "avg_return": r.get("avg_return_pct", 0),
            "sharpe": r.get("sharpe", 0),
            "profit_factor": r.get("profit_factor", 0),
            "tp_pct": r.get("tp_rate", 0) * 100,
            "sl_pct": r.get("sl_rate", 0) * 100,
            "close_pct": r.get("close_rate", 0) * 100,
        })
    return pd.DataFrame(rows).set_index("hour")


def _compute_pivot(trades: pd.DataFrame) -> pd.DataFrame:
    """Hour × quarter pivot of Sharpe ratio."""
    t = _extract_hour(trades)
    rows = []
    for hour, grp in t.groupby("hour_of_day", sort=True):
        for q, qgrp in grp.groupby("quarter", sort=True):
            r = report(qgrp)
            rows.append({
                "hour": hour,
                "quarter": q,
                "n": r.get("n", 0),
                "sharpe": r.get("sharpe", 0),
                "profit_factor": r.get("profit_factor", 0),
                "win_rate": r.get("win_rate", 0) * 100,
            })
    piv = pd.DataFrame(rows)
    if piv.empty:
        return piv
    return piv.pivot(index="hour", columns="quarter", values="sharpe")


def _print_hourly_table(name: str, hourly: pd.DataFrame, total: int):
    """Pretty-print a per-hour metrics table."""
    if hourly.empty:
        print(f"\n  {name}: NO TRADES")
        return

    W = 100
    print(f"\n{'─' * W}")
    print(f"  {name}  (total trades: {total})")
    print(f"{'─' * W}")
    print(f"{'Hour':>5} {'N':>5} {'Trd%':>6} {'WR%':>6} {'AvgRet%':>8} "
          f"{'Sharpe':>7} {'PF':>6} {'TP%':>6} {'SL%':>6} {'CLS%':>6}")
    print(f"{'─' * W}")
    for idx, row in hourly.iterrows():
        print(f"{idx:>5} {int(row['n']):>5} {row['trade_pct']:>5.1f}% "
              f"{row['win_rate']:>5.1f}% {row['avg_return']:>+7.3f} "
              f"{row['sharpe']:>6.2f} {row['profit_factor']:>5.2f} "
              f"{row['tp_pct']:>5.1f}% {row['sl_pct']:>5.1f}% {row['close_pct']:>5.1f}%")


def _print_pivot(name: str, pivot: pd.DataFrame):
    """Pretty-print hour × quarter pivot table."""
    if pivot.empty:
        print(f"\n  {name}: NO TRADES — no pivot")
        return

    W = 110
    print(f"\n{'─' * W}")
    print(f"  {name}  —  Hour × Quarter Sharpe Pivot")
    print(f"{'─' * W}")
    cols = sorted(pivot.columns.tolist())
    header = f"{'Hour':>5}"
    for c in cols:
        header += f" {c:>10}"
    print(header)
    print(f"{'─' * W}")
    for idx in sorted(pivot.index):
        line = f"{idx:>5}"
        for c in cols:
            val = pivot.loc[idx, c] if c in pivot.columns else float("nan")
            if pd.isna(val):
                line += f" {'—':>10}"
            else:
                line += f" {val:>10.2f}"
        print(line)


def main():
    parser = argparse.ArgumentParser(description="Temporal Diagnostics — hour & quarter analysis")
    parser.add_argument("--samples_path", type=str, required=True,
                        help="Path to samples parquet file")
    parser.add_argument("--strategies", type=str,
                        default="S01_BB,S20_OB,S02_BB_MR",
                        help="Comma-separated strategy names (from registry)")
    parser.add_argument("--mode", type=str, default="both",
                        choices=["pivot", "flat", "both"],
                        help="Output mode: pivot (hour×quarter), flat (per-hour), or both")
    parser.add_argument("--train_period", type=str, default="2025",
                        help="Year string to filter (default: 2025)")
    parser.add_argument("--test_period", type=str, default="2026",
                        help="Year string to filter (default: 2026)")
    args = parser.parse_args()

    samples_path = args.samples_path
    if not Path(samples_path).exists():
        print(f"ERROR: Samples file not found: {samples_path}")
        sys.exit(1)

    strategy_names = [s.strip() for s in args.strategies.split(",") if s.strip()]

    print(f"Loading samples: {samples_path}")
    df = load_samples(samples_path)
    df["year"] = df["month"].str[:4]
    train_df = df[df["year"] == args.train_period]
    test_df = df[df["year"] == args.test_period]
    print(f"  Train ({args.train_period}): {len(train_df)}  Test ({args.test_period}): {len(test_df)}")

    print("Loading Mamba OHLCV...")
    cfg = load_config()
    mamba = load_mamba_sber(cfg)
    print(f"  Mamba rows: {len(mamba)}")

    for sname in strategy_names:
        try:
            func, params = _load_strategy(sname)
        except ValueError as e:
            print(f"SKIP: {e}")
            continue

        print(f"\nRunning {sname} on test ({args.test_period})...")
        test_trades = func(
            test_df, mamba,
            pred_len=PRED_LEN, sample_count=SAMPLE_COUNT,
            tp_q=TP_Q, sl_q=SL_Q, **params,
        )
        if len(test_trades) == 0:
            print(f"  {sname}: 0 trades on test period")
            continue

        r_all = report(test_trades)
        total_trades = len(test_trades)
        print(f"  {sname} test: n={total_trades} sharpe={r_all['sharpe']:.2f} "
              f"ret={r_all['total_return_pct']:+.2f}% wr={r_all['win_rate']*100:.1f}%")

        if args.mode in ("flat", "both"):
            hourly = _compute_hourly_metrics(test_trades, total_trades)
            _print_hourly_table(sname, hourly, total_trades)

        if args.mode in ("pivot", "both"):
            pivot = _compute_pivot(test_trades)
            _print_pivot(sname, pivot)

    print()


if __name__ == "__main__":
    main()
