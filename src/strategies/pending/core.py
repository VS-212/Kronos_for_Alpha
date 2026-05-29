"""
M-STRATEGY-CORE: Shared strategy engine
Contract: config, data loading, trade simulation, reporting → trades DataFrame
Status: ✅ ready
"""

"""Shared core for all strategies: data loading, TP/SL simulation, reporting."""
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.evaluation.output import reconstruct
from src.signals.atoms import boundaries, consensus, dispersion
from src.signals.volatility import compute_atr


def load_config() -> dict:
    with open(Path(__file__).resolve().parent.parent.parent / "config" / "global.yaml") as f:
        return yaml.safe_load(f)


def load_mamba_sber(cfg: dict) -> pd.DataFrame:
    """Load SBER OHLCV from Mamba. Filter to main session first, then compute BB."""
    mamba_path = cfg["data"]["mamba_path"]
    raw = pd.read_parquet(mamba_path)
    ts = pd.to_datetime(raw["timestamp"])
    raw = raw.set_index(ts)
    cols = [f"SBER_{c}" for c in ["open", "high", "low", "close", "volume"]]
    df = raw[cols].copy()
    df.columns = ["open", "high", "low", "close", "volume"]
    df.index.name = "timestamp"

    df["is_main_session"] = (df.index.hour * 60 + df.index.minute >= 600) & (
        df.index.hour * 60 + df.index.minute <= 1120
    )
    dates_full = np.array([d.toordinal() for d in df.index.date], dtype=np.int32)
    df["is_day_start"] = np.concatenate([[True], np.diff(dates_full).astype(bool)])
    df = df[df["is_main_session"]].copy()

    df["sma20"] = df["close"].rolling(20).mean()
    df["std20"] = df["close"].rolling(20).std(ddof=1)
    df["bb_upper"] = df["sma20"] + 2 * df["std20"]
    df["bb_lower"] = df["sma20"] - 2 * df["std20"]
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["sma20"]

    return df


def load_mamba_full(cfg: dict) -> pd.DataFrame:
    """Load SBER OHLCV from Mamba — ALL sessions, no filtering."""
    mamba_path = cfg["data"]["mamba_path"]
    raw = pd.read_parquet(mamba_path)
    ts = pd.to_datetime(raw["timestamp"])
    raw = raw.set_index(ts)
    cols = [f"SBER_{c}" for c in ["open", "high", "low", "close", "volume"]]
    df = raw[cols].copy()
    df.columns = ["open", "high", "low", "close", "volume"]
    df.index.name = "timestamp"
    df["is_main_session"] = (df.index.hour * 60 + df.index.minute >= 600) & (
        df.index.hour * 60 + df.index.minute <= 1120
    )
    dates_ord = np.array([d.toordinal() for d in df.index.date], dtype=np.int32)
    df["is_day_start"] = np.concatenate([[True], np.diff(dates_ord).astype(bool)])
    return df


def lookup_bb(mamba: pd.DataFrame, pred_ts: pd.Timestamp) -> dict:
    """Get BB values at the candle just before pred_ts."""
    mask = mamba.index < pred_ts
    if mask.sum() == 0:
        return {"sma20": np.nan, "bb_upper": np.nan, "bb_lower": np.nan, "bb_width": np.nan}
    row = mamba[mask].iloc[-1]
    return {
        "sma20": float(row["sma20"]),
        "bb_upper": float(row["bb_upper"]),
        "bb_lower": float(row["bb_lower"]),
        "bb_width": float(row["bb_width"]),
        "last_close": float(row["close"]),
    }


def lookup_mamba_window(mamba: pd.DataFrame, pred_ts: pd.Timestamp, lookback: int = 100) -> dict:
    """Get last N candles of OHLCV before pred_ts."""
    mask = mamba.index < pred_ts
    if mask.sum() == 0:
        return {"has_data": False}
    window = mamba[mask].iloc[-lookback:]
    dates = window.index.date
    is_day_start = np.concatenate([[True], np.diff(dates).astype(bool)])
    ims = (
        window["is_main_session"].values
        if "is_main_session" in window.columns
        else np.ones(len(window), dtype=bool)
    )
    return {
        "has_data": True,
        "open": window["open"].values.astype(np.float64),
        "high": window["high"].values.astype(np.float64),
        "low": window["low"].values.astype(np.float64),
        "close": window["close"].values.astype(np.float64),
        "volume": window["volume"].values.astype(np.float64),
        "is_day_start": is_day_start,
        "is_main_session": ims,
    }


def _simulate_trade(
    close_only,
    actuals,
    prev_close,
    pred_dir,
    pred_len,
    tp_q=0.90,
    sl_q=0.10,
    mamba=None,
    pred_ts=None,
    atr_tp_mult=None,
    atr_sl_mult=None,
    save_first_n=0,
    bypass_consensus=False,
    consensus_threshold=0.8,
    dispersion_cap=None,
    exit_on_flip=False,
    belief_blob=None,
    belief_early_exit_threshold=1.5,
) -> dict:
    if not bypass_consensus:
        cons = consensus(close_only, prev_close, threshold=consensus_threshold)
        if not cons["has_consensus"][0]:
            return None

    entry_price = float(actuals[0, 0])

    if dispersion_cap is not None:
        disp = dispersion(close_only, prev_close)
        if disp["mean_std"] > dispersion_cap:
            return None

    if atr_tp_mult is not None and mamba is not None and pred_ts is not None:
        w = lookup_mamba_window(mamba, pred_ts, lookback=200)
        if w["has_data"]:
            atr_val = compute_atr(w["high"], w["low"], w["close"])["atr"]
            tp = entry_price + pred_dir * atr_tp_mult * atr_val
            sl = entry_price - pred_dir * atr_sl_mult * atr_val
            bnd = {
                "tp_long": np.full(pred_len, tp),
                "sl_long": np.full(pred_len, sl),
                "tp_short": np.full(pred_len, tp),
                "sl_short": np.full(pred_len, sl),
            }
        else:
            bnd = boundaries(close_only, prev_close, tp_q=tp_q, sl_q=sl_q)
    else:
        bnd = boundaries(close_only, prev_close, tp_q=tp_q, sl_q=sl_q)

    exit_step = pred_len - 1
    exit_reason = "close"
    exit_price = float(actuals[-1, 3])

    for i in range(pred_len):
        if i < save_first_n:
            continue
        hi, lo = float(actuals[i, 1]), float(actuals[i, 2])
        if pred_dir == 1:
            if lo <= bnd["sl_long"][i]:
                exit_step = i
                exit_reason = "sl"
                exit_price = float(bnd["sl_long"][i])
                break
            if hi >= bnd["tp_long"][i]:
                exit_step = i
                exit_reason = "tp"
                exit_price = float(bnd["tp_long"][i])
                break
        else:
            if hi >= bnd["sl_short"][i]:
                exit_step = i
                exit_reason = "sl"
                exit_price = float(bnd["sl_short"][i])
                break
            if lo <= bnd["tp_short"][i]:
                exit_step = i
                exit_reason = "tp"
                exit_price = float(bnd["tp_short"][i])
                break

        if exit_on_flip:
            remaining = close_only[:, i:]
            cons_r = consensus(remaining, entry_price, threshold=0.6)
            if cons_r["has_consensus"][0] and cons_r["consensus_dir"][0] != pred_dir:
                exit_step = i
                exit_reason = "flip"
                exit_price = float(actuals[i, 3])
                break

        # ── Belief-based early exit ──
        if belief_blob is not None:
            from src.evaluation.output import reconstruct
            beliefs = reconstruct(belief_blob, *close_only.shape[:2], 4)
            if i >= 2:
                ent_init = float(np.mean(beliefs[:, :min(3, 1), 1]))  # entropy_s1 at step 0
                ent_curr = float(np.mean(beliefs[:, i, 1]))
                if ent_curr > max(ent_init * belief_early_exit_threshold, 2.0):
                    exit_step = i
                    exit_reason = "belief_exit"
                    exit_price = float(actuals[i, 3])
                    break

    trade_return = pred_dir * (exit_price - entry_price) / entry_price
    return {
        "direction": pred_dir,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "exit_step": exit_step,
        "exit_reason": exit_reason,
        "return": trade_return,
    }


def _run_ict_strategy(
    df,
    mamba,
    pred_len,
    sample_count,
    tp_q,
    sl_q,
    detector_fn,
    lookback,
    atr_tp_mult=None,
    atr_sl_mult=None,
    save_first_n=0,
    consensus_threshold=0.8,
    **detect_kwargs,
):
    """Generic runner for any ICT-based strategy."""
    trades = []
    for _, row in df.iterrows():
        close_only = reconstruct(row["samples_blob"], sample_count, pred_len)
        actuals = reconstruct(row["actual_blob"], pred_len, 4)
        prev_close = float(row["prev_close"])
        if prev_close == 0:
            continue

        cons = consensus(close_only, prev_close, threshold=consensus_threshold)
        if not cons["has_consensus"][0]:
            continue
        pred_dir = int(cons["consensus_dir"][0])

        w = lookup_mamba_window(mamba, row["pred_ts"], lookback)
        if not w["has_data"]:
            continue

        sig = detector_fn(
            **{k: w[k] for k in ["open", "high", "low", "close", "volume"]},
            lookback=lookback,
            **detect_kwargs,
        )
        if sig["signal"] == "none":
            continue
        if not (
            (sig["signal"] == "bullish" and pred_dir == 1)
            or (sig["signal"] == "bearish" and pred_dir == -1)
        ):
            continue

        trade = _simulate_trade(
            close_only,
            actuals,
            prev_close,
            pred_dir,
            pred_len,
            tp_q,
            sl_q,
            mamba=mamba,
            pred_ts=row["pred_ts"],
            atr_tp_mult=atr_tp_mult,
            atr_sl_mult=atr_sl_mult,
            save_first_n=save_first_n,
            consensus_threshold=consensus_threshold,
        )
        if trade is not None:
            trade.update(
                {
                    "window_id": int(row["window_id"]),
                    "pred_ts": row["pred_ts"],
                    "month": row["month"],
                    "quarter": f"{row['month'][:4]}-Q{(int(row['month'][5:]) - 1) // 3 + 1}",
                }
            )
            trades.append(trade)
    return pd.DataFrame(trades)


def _enrich_trade(trade, row):
    """Add window_id, pred_ts, month, quarter to a trade dict."""
    trade.update(
        {
            "window_id": int(row["window_id"]),
            "pred_ts": row["pred_ts"],
            "month": row["month"],
            "quarter": f"{row['month'][:4]}-Q{(int(row['month'][5:]) - 1) // 3 + 1}",
        }
    )
    return trade


def report(trades: pd.DataFrame) -> dict:
    if len(trades) == 0:
        return {"error": "no trades", "n": 0}
    n = len(trades)
    tp_hits = int((trades["exit_reason"] == "tp").sum())
    sl_hits = int((trades["exit_reason"] == "sl").sum())
    closes = int((trades["exit_reason"] == "close").sum())
    wins = int((trades["return"] > 0).sum())
    losses = int((trades["return"] <= 0).sum())
    total_ret = trades["return"].sum()
    avg_ret = trades["return"].mean()
    std_ret = trades["return"].std()
    equity = (1 + trades["return"]).cumprod()
    peak = equity.cummax()
    drawdown = (equity - peak) / peak
    mdd = float(drawdown.min())
    sharpe = avg_ret / max(std_ret, 1e-10) * np.sqrt(252 * 6 / 6)
    win_rate = wins / max(n, 1)
    avg_win = float(trades.loc[trades["return"] > 0, "return"].mean()) if wins else 0.0
    avg_loss = float(trades.loc[trades["return"] <= 0, "return"].mean()) if losses else 0.0
    sum_wins = avg_win * wins if wins else 0.0
    sum_losses = abs(avg_loss * losses) if losses else 0.0
    pf = sum_wins / max(sum_losses, 1e-10)
    return {
        "n": n,
        "tp_hits": tp_hits,
        "sl_hits": sl_hits,
        "closes": closes,
        "tp_rate": tp_hits / max(n, 1),
        "sl_rate": sl_hits / max(n, 1),
        "close_rate": closes / max(n, 1),
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "total_return_pct": total_ret * 100,
        "avg_return_pct": avg_ret * 100,
        "std_return_pct": std_ret * 100,
        "avg_win_pct": avg_win * 100,
        "avg_loss_pct": avg_loss * 100,
        "sharpe": sharpe,
        "mdd_pct": mdd * 100,
        "profit_factor": pf,
    }


def quarterly_breakdown(trades: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for q, grp in trades.groupby("quarter", sort=True):
        r = report(grp)
        r["quarter"] = q
        rows.append(r)
    return pd.DataFrame(rows).set_index("quarter")
