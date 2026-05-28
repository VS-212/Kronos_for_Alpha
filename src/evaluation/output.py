"""
M-OUTPUT: Compact result serialization
Contract: windows list + config → parquet files + summary JSON
Status: ✅ ready
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd


def save_samples(
    ticker: str,
    windows: list[dict],
    pred_len: int,
    sample_count: int,
    config: dict,
    output_dir: str,
):
    """Save per-window prediction samples + actuals to a compact parquet.

    Each row = one prediction window.

    Columns:
        window_id       — sequential int
        pred_ts        — datetime of first predicted candle
        month          — "YYYY-MM" grouping key
        prev_close     — entry reference price
        config_T       — temperature used
        config_top_p   — top_p used
        samples_blob   — bytes: (sample_count, pred_len) float32 close paths
        actual_blob    — bytes: (pred_len, 4) float32 [open, high, low, close]

    The blob format keeps storage tiny (~700 KB for 3K windows)
    and reconstruction is a single np.frombuffer + reshape call.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for w in windows:
        rows.append(
            {
                "window_id": w["window_id"],
                "pred_ts": w["pred_ts"],
                "month": w["month"],
                "prev_close": np.float32(w["prev_close"]),
                "samples_blob": np.asarray(w["samples"], dtype=np.float32).tobytes(),
                "actual_blob": np.asarray(w["actuals"], dtype=np.float32).tobytes(),
                "config_T": np.float32(config.get("T", 1.0)),
                "config_top_p": np.float32(config.get("top_p", 0.9)),
            }
        )

    df = pd.DataFrame(rows)
    df["window_id"] = df["window_id"].astype(np.uint32)

    path = out_dir / f"{ticker}_samples_pl{pred_len}_sc{sample_count}.parquet"
    df.to_parquet(path, index=False)
    return path


def load_samples(path: str) -> pd.DataFrame:
    """Load samples parquet and return raw DataFrame (blobs unmodified)."""
    return pd.read_parquet(path)


def reconstruct(blob: bytes, *shape) -> np.ndarray:
    """Deserialize a numpy array from bytes."""
    return np.frombuffer(blob, dtype=np.float32).reshape(shape)


def compute_monthly_metrics(
    df: pd.DataFrame,
    pred_len: int,
    sample_count: int,
) -> pd.DataFrame:
    """Compute evaluate() metrics per month from a samples parquet.

    Returns a DataFrame with one row per month containing all
    the metrics from src.evaluation.evaluate.evaluate().
    """
    from src.evaluation.evaluate import evaluate

    monthly = []
    for month, group in df.groupby("month", sort=True):
        all_metrics = []
        for _, row in group.iterrows():
            close_only = reconstruct(row["samples_blob"], sample_count, pred_len)
            # Stored only close (95% of metrics use it).
            # Pad to (S, N, 6) — [O,H,L,C,V,A] — for evaluate(); unused dims = 0.
            samples_6d = np.zeros((sample_count, pred_len, 6), dtype=np.float32)
            samples_6d[:, :, 3] = close_only

            actuals = reconstruct(row["actual_blob"], pred_len, 4)
            actual_df = pd.DataFrame(actuals, columns=["open", "high", "low", "close"])
            metrics = evaluate(actual_df, samples_6d, prev_close=float(row["prev_close"]))
            if metrics:
                all_metrics.append(metrics)

        if all_metrics:
            avg = {}
            keys = all_metrics[0].keys()
            for k in keys:
                vals = [m[k] for m in all_metrics if not np.isnan(m.get(k, np.nan))]
                avg[k] = float(np.mean(vals)) if vals else 0.0
            avg["month"] = month
            avg["n_windows"] = len(all_metrics)
            monthly.append(avg)

    return pd.DataFrame(monthly)


def save_monthly_metrics(
    ticker: str,
    metrics_df: pd.DataFrame,
    config: dict,
    output_dir: str,
) -> Path:
    """Save monthly metrics to parquet."""
    pred_len = config.get("pred_len", 6)
    sc = config.get("sample_count", 5)
    path = (
        Path(output_dir)
        / f"{ticker}_monthly_pl{pred_len}_sc{sc}_T{config['T']}_P{config['top_p']}.parquet"
    )
    metrics_df.to_parquet(path, index=False)
    return path


def trade_summary(
    df: pd.DataFrame,
    pred_len: int,
    sample_count: int,
    tp_quantile: float = 0.80,
    sl_quantile: float = 0.20,
) -> dict:
    """Per-TRADE breakdown (not per-step).

    One trade = one window (full pred_len horizon).
    Entry = open T (first predicted candle). Exit = first TP/SL hit (via HIGH/LOW),
    or close at T+pred_len. SL checked first (conservative).
    Only windows with consensus (4/5 agree) on first step enter.
    """
    tp_total = sl_total = close_total = 0
    w_total = l_total = 0
    sum_return = 0.0
    sum_abs_return = 0.0
    entered = 0

    for _, row in df.iterrows():
        close_only = reconstruct(row["samples_blob"], sample_count, pred_len)
        actuals = reconstruct(row["actual_blob"], pred_len, 4)
        prev_close = float(row["prev_close"])
        if prev_close == 0:
            continue

        # Consensus on entry step (step 0): 4/5 agree on direction
        s_up = np.mean(close_only[:, 0] > prev_close)
        s_dn = np.mean(close_only[:, 0] < prev_close)
        if s_up < 0.8 and s_dn < 0.8:
            continue
        pred_dir = 1 if s_up >= 0.8 else -1

        entry_price = float(actuals[0, 0])

        # Q80 / Q20 boundaries per step from sample distribution
        close_q80 = np.quantile(close_only, tp_quantile, axis=0)
        close_q20 = np.quantile(close_only, sl_quantile, axis=0)

        exit_step = pred_len - 1
        exit_reason = "close"
        exit_price = float(actuals[-1, 3])
        for i in range(pred_len):
            hi, lo = float(actuals[i, 1]), float(actuals[i, 2])
            if pred_dir == 1:  # long
                tp_lvl = close_q80[i] if close_q80[i] > prev_close else np.inf
                sl_lvl = close_q20[i] if close_q20[i] < prev_close else -np.inf
                if lo <= sl_lvl:
                    exit_step = i
                    exit_reason = "sl"
                    exit_price = sl_lvl
                    break
                if hi >= tp_lvl:
                    exit_step = i
                    exit_reason = "tp"
                    exit_price = tp_lvl
                    break
            else:  # short
                tp_lvl = close_q20[i] if close_q20[i] < prev_close else -np.inf
                sl_lvl = close_q80[i] if close_q80[i] > prev_close else np.inf
                if hi >= sl_lvl:
                    exit_step = i
                    exit_reason = "sl"
                    exit_price = sl_lvl
                    break
                if lo <= tp_lvl:
                    exit_step = i
                    exit_reason = "tp"
                    exit_price = tp_lvl
                    break

        trade_return = pred_dir * (exit_price - entry_price) / entry_price
        sum_return += trade_return
        sum_abs_return += abs(trade_return)
        entered += 1

        if trade_return > 0:
            w_total += 1
        else:
            l_total += 1

        if exit_reason == "tp":
            tp_total += 1
        elif exit_reason == "sl":
            sl_total += 1
        else:
            close_total += 1

    return {
        "total_windows": len(df),
        "entered": entered,
        "tp_hits": tp_total,
        "sl_hits": sl_total,
        "closes": close_total,
        "tp_rate": tp_total / max(entered, 1),
        "sl_rate": sl_total / max(entered, 1),
        "close_rate": close_total / max(entered, 1),
        "wins": w_total,
        "losses": l_total,
        "win_rate": w_total / max(w_total + l_total, 1),
        "total_return_pct": sum_return * 100,
        "avg_trade_return_pct": sum_return / max(entered, 1) * 100,
    }


def save_summary(
    tickers: list[str],
    all_configs: list[dict],
    output_dir: str,
):
    """Save cross-ticker summary JSON."""
    summary = {
        "tickers": tickers,
        "configs": all_configs,
    }
    path = Path(output_dir) / "summary.json"
    with open(path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    return path
