"""
M-CALIBRATE: Hyperparameter sweep runner
Contract: config.yaml + Kronos model → pred_len sweep → T/top_p/sc sweep → best config JSON
Status: ✅ ready
"""

import itertools
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.core.kronos import KronosModel
from src.data.cache import DataCache
from src.data.moex import MOEXDataSource
from src.evaluation.evaluate import evaluate


def _random_indices(df_len, lookback, pred_len, n, seed=42):
    """Pick n random window starts, spread across available range."""
    max_start = df_len - lookback - pred_len
    if max_start <= 0:
        return []
    rng = np.random.default_rng(seed)
    n = min(n, max_start)
    return sorted(rng.choice(max_start, n, replace=False).tolist())


def load_config():
    config_path = Path(__file__).parent / "config.yaml"
    global_path = Path(__file__).parent.parent.parent / "config" / "global.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    with open(global_path) as f:
        global_cfg = yaml.safe_load(f)
    cfg["global"] = global_cfg
    return cfg


def fetch_data(data_source, cache, cfg):
    ticker = cfg["calibration_ticker"]
    cache_key = DataCache.key(
        ticker,
        5,
        cfg["global"]["data"]["start_date"],
        cfg["global"]["data"]["end_date"],
    )

    df = cache.get(cache_key)
    if df is not None:
        print(f"Cache hit: {cache_key}")
        return df

    df = data_source.fetch_candles(
        ticker,
        interval=5,
        start=cfg["global"]["data"]["start_date"],
        end=cfg["global"]["data"]["end_date"],
    )
    if df is not None and not df.empty:
        cache.put(cache_key, df)
    return df


def pass_1_pred_len_sweep(model, df, cfg, eval_indices):
    results = []
    lookback = cfg["global"]["data"]["lookback_candles"]

    fixed_T = cfg["sweep"]["pass_1"]["fixed"]["temperature"]
    fixed_top_p = cfg["sweep"]["pass_1"]["fixed"]["top_p"]
    fixed_sc = cfg["sweep"]["pass_1"]["fixed"]["sample_count"]

    for pred_len in cfg["sweep"]["pass_1"]["pred_len"]:
        print(f"\nPass 1: pred_len={pred_len}")
        all_metrics = []

        for i in eval_indices:
            x_df = df.iloc[i : i + lookback].copy()
            prev_close = float(x_df.iloc[-1]["close"])
            y_actual = df.iloc[i + lookback : i + lookback + pred_len].copy()

            try:
                samples = model.predict_samples(
                    x_df,
                    pred_len=pred_len,
                    T=fixed_T,
                    top_p=fixed_top_p,
                    sample_count=fixed_sc,
                )
                metrics = evaluate(y_actual, samples, prev_close=prev_close)
                if metrics:
                    all_metrics.append(metrics)
            except Exception as e:
                print(f"  Error at idx {i}: {e}")
                continue

        if all_metrics:
            avg_metrics = {}
            for k in all_metrics[0].keys():
                vals = [m[k] for m in all_metrics if not np.isnan(m.get(k, np.nan))]
                avg_metrics[k] = float(np.mean(vals)) if vals else 0.0
            avg_metrics["pred_len"] = pred_len
            results.append(avg_metrics)
            print(
                f"  direction_accuracy={avg_metrics['direction_accuracy']:.3f}, "
                f"expectancy={avg_metrics['expectancy']:.6f}"
            )

    return results


def pass_2_param_sweep(model, df, cfg, best_pred_len, eval_indices):
    results = []
    lookback = cfg["global"]["data"]["lookback_candles"]

    temps = cfg["sweep"]["pass_2"]["temperature"]
    top_ps = cfg["sweep"]["pass_2"]["top_p"]
    scs = cfg["sweep"]["pass_2"]["sample_count"]

    for T_val, top_p_val, sc_val in itertools.product(temps, top_ps, scs):
        print(
            f"\nPass 2: T={T_val}, top_p={top_p_val}, "
            f"sample_count={sc_val}, pred_len={best_pred_len}"
        )
        all_metrics = []

        for i in eval_indices:
            x_df = df.iloc[i : i + lookback].copy()
            prev_close = float(x_df.iloc[-1]["close"])
            y_actual = df.iloc[i + lookback : i + lookback + best_pred_len].copy()

            try:
                samples = model.predict_samples(
                    x_df,
                    pred_len=best_pred_len,
                    T=T_val,
                    top_p=top_p_val,
                    sample_count=sc_val,
                )
                metrics = evaluate(y_actual, samples, prev_close=prev_close)
                if metrics:
                    all_metrics.append(metrics)
            except Exception as e:
                continue

        if all_metrics:
            avg_metrics = {}
            for k in all_metrics[0].keys():
                vals = [m[k] for m in all_metrics if not np.isnan(m.get(k, np.nan))]
                avg_metrics[k] = float(np.mean(vals)) if vals else 0.0
            avg_metrics["T"] = T_val
            avg_metrics["top_p"] = top_p_val
            avg_metrics["sample_count"] = sc_val
            avg_metrics["pred_len"] = best_pred_len
            results.append(avg_metrics)
            print(
                f"  consensus_dir_acc={avg_metrics.get('consensus_dir_acc', 0):.3f}  "
                f"dir_acc={avg_metrics.get('direction_accuracy', 0):.3f}  "
                f"cons_rate={avg_metrics.get('consensus_rate', 0):.3f}  "
                f"expectancy={avg_metrics['expectancy']:.6f}  "
                f"unf_exp={avg_metrics.get('unfiltered_expectancy', 0):.6f}"
            )

    return results


def main():
    cfg = load_config()
    device = cfg["global"]["model"].get("device", "cuda")

    print("=" * 60)
    print("CALIBRATION PASS 0 — Inference parameter sweep")
    print(f"Model: {cfg['global']['model']['name']}")
    print(f"Ticker: {cfg['calibration_ticker']}")
    print("=" * 60)

    model = KronosModel(
        model_name=cfg["global"]["model"]["name"],
        tokenizer_name=cfg["global"]["model"]["tokenizer"],
        device=device,
        max_context=cfg["global"]["model"]["max_context"],
        session_filter=cfg["global"]["execution"]["session_filter"],
        main_session_start=cfg["global"]["session"]["main_session_start"],
        main_session_end=cfg["global"]["session"]["main_session_end"],
    )
    print("Loading model...")
    model.load()

    data_source = MOEXDataSource(
        board=cfg["global"]["universe"]["board"],
        market=cfg["global"]["universe"]["market"],
        engine=cfg["global"]["universe"]["engine"],
    )
    cache_dir = cfg["global"]["data"]["cache_dir"]
    cache = DataCache(cache_dir)

    print("\nFetching SBER data...")
    df = fetch_data(data_source, cache, cfg)
    if df is None or len(df) < 2048:
        print(f"ERROR: Not enough data for SBER")
        return

    if "begin" in df.columns:
        df = df.set_index("begin")
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    session_start = cfg["global"]["session"]["main_session_start"]
    session_end = cfg["global"]["session"]["main_session_end"]
    start_t = pd.Timestamp(session_start).time()
    end_t = pd.Timestamp(session_end).time()
    df = df[(df.index.time >= start_t) & (df.index.time <= end_t)]
    if df.empty:
        print("ERROR: No data after main-session filter")
        return

    if hasattr(df.index, "tz") and df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    print(f"Main-session candles: {len(df)}")

    # ── Pre-compute evaluation indices (shared across all passes) ──
    lookback = cfg["global"]["data"]["lookback_candles"]
    n_timestamps = cfg.get("n_timestamps", 50)
    seed = cfg.get("seed", 42)
    max_pred_len = max(cfg["sweep"]["pass_1"]["pred_len"])
    eval_indices = _random_indices(len(df), lookback, max_pred_len, n_timestamps, seed)
    if not eval_indices:
        print("ERROR: No valid evaluation windows available")
        return
    print(f"Evaluation windows: {len(eval_indices)} (pred_len≤{max_pred_len})")

    # ── Pass 1: pred_len sweep ──
    print("\n── Pass 1: pred_len sweep ──")
    p1_results = pass_1_pred_len_sweep(model, df, cfg, eval_indices)

    if not p1_results:
        print("ERROR: Pass 1 produced no results")
        return

    best_p1 = max(p1_results, key=lambda r: (r.get("consensus_dir_acc", 0), r.get("expectancy", 0)))

    best_pred_len = int(best_p1["pred_len"])
    print(
        f"\nBest pred_len: {best_pred_len} "
        f"(expectancy={best_p1['expectancy']:.6f}, "
        f"consensus_dir_acc={best_p1['consensus_dir_acc']:.3f}, "
        f"consensus_rate={best_p1['consensus_rate']:.3f})"
    )

    # ── Pass 2: T, top_p, sample_count sweep ──
    print("\n── Pass 2: T x top_p x sample_count sweep ──")
    p2_results = pass_2_param_sweep(model, df, cfg, best_pred_len, eval_indices)

    output = {
        "timestamp": datetime.now().isoformat(),
        "model": cfg["global"]["model"]["name"],
        "n_windows": len(eval_indices),
        "pass_1": sorted(p1_results, key=lambda r: r["expectancy"], reverse=True),
        "pass_2": sorted(p2_results, key=lambda r: r["expectancy"], reverse=True)
        if p2_results
        else [],
        "best_config": None,
    }

    if p2_results:
        best_p2 = max(p2_results, key=lambda r: (r["consensus_dir_acc"], r["expectancy"]))
        output["best_config"] = {
            "pred_len": best_p2["pred_len"],
            "temperature": best_p2["T"],
            "top_p": best_p2["top_p"],
            "sample_count": best_p2["sample_count"],
            "expectancy": best_p2["expectancy"],
            "consensus_dir_acc": best_p2["consensus_dir_acc"],
            "consensus_rate": best_p2["consensus_rate"],
            "direction_accuracy": best_p2["direction_accuracy"],
        }

    result_path = Path(__file__).parent / "results.json"
    with open(result_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {result_path}")

    if output["best_config"]:
        bc = output["best_config"]
        print(f"\nBEST INFERENCE CONFIG:")
        print(f"  pred_len={bc['pred_len']}")
        print(f"  T={bc['temperature']}")
        print(f"  top_p={bc['top_p']}")
        print(f"  sample_count={bc['sample_count']}")
        print(f"  consensus_dir_acc={bc['consensus_dir_acc']:.3f}")
        print(f"  consensus_rate={bc['consensus_rate']:.3f}")
        print(f"  expectancy={bc['expectancy']:.6f}")
        print(f"  direction_accuracy={bc['direction_accuracy']:.3f}")


if __name__ == "__main__":
    main()
