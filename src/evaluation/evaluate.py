"""
M-EVALUATE: Per-window evaluation metrics
Contract: actual_df + samples + prev_close → 14 metrics dict per window
Status: ✅ ready
"""

import numpy as np
import pandas as pd

try:
    from scipy.stats import pearsonr, spearmanr
except ImportError:
    spearmanr = None
    pearsonr = None


def evaluate(
    actual_df: pd.DataFrame,
    samples: np.ndarray,
    prev_close: float,
    tp_quantile: float = 0.80,
    sl_quantile: float = 0.20,
) -> dict[str, float]:
    """
    Calibration metrics from Kronos per-sample predictions.

    Args:
        actual_df: Ground truth OHLCV DataFrame (future candles).
        samples: Per-sample predictions (sample_count, pred_len, 6).
        prev_close: Close of LAST input candle (entry reference).
        tp_quantile: Quantile for TP boundary (default 0.80).
        sl_quantile: Quantile for SL boundary (default 0.20).

    Returns:
        Dict of {metric_name: float}.
    """
    if samples is None or samples.size == 0:
        return {}

    S, N, F = samples.shape
    if N == 0:
        return {}

    N_eval = min(N, len(actual_df))
    if N_eval == 0:
        return {}

    actual_c = actual_df["close"].values[:N_eval]
    prev_close = float(prev_close)
    if prev_close == 0:
        return {}

    # ── Predicted aggregates ──
    mean_close = np.mean(samples[:, :N_eval, 3], axis=0)
    mean_high = np.mean(samples[:, :N_eval, 1], axis=0)
    mean_low = np.mean(samples[:, :N_eval, 2], axis=0)

    close_q80 = np.quantile(samples[:, :N_eval, 3], tp_quantile, axis=0)
    close_q20 = np.quantile(samples[:, :N_eval, 3], sl_quantile, axis=0)

    actual_return = (actual_c - prev_close) / prev_close
    pred_return = (mean_close - prev_close) / prev_close

    results = {}

    # ── 1. Direction accuracy ──
    actual_dir = np.sign(actual_return)
    pred_dir = np.sign(pred_return)
    non_flat = pred_dir != 0
    if non_flat.sum() > 0:
        results["direction_accuracy"] = float(np.mean(actual_dir[non_flat] == pred_dir[non_flat]))
    else:
        results["direction_accuracy"] = 0.0
    results["flat_prediction_rate"] = float(1.0 - non_flat.mean())

    # ── 1b. Consensus metrics (4/5 samples agree on direction) ──
    consensus_mask: np.ndarray | slice = slice(None)
    consensus_dir: np.ndarray | None = None
    if S == 5:
        sample_up = np.mean(samples[:, :N_eval, 3] > prev_close, axis=0)
        sample_down = np.mean(samples[:, :N_eval, 3] < prev_close, axis=0)
        consensus_up = sample_up >= 0.8
        consensus_down = sample_down >= 0.8
        consensus_mask = consensus_up | consensus_down
        consensus_dir = np.where(consensus_up, 1, -1)
        results["consensus_rate"] = float(consensus_mask.mean())
        results["consensus_strength"] = float(np.mean(np.maximum(sample_up, sample_down)))
        if consensus_mask.sum() > 0:
            results["consensus_dir_acc"] = float(
                np.mean(actual_dir[consensus_mask] == consensus_dir[consensus_mask])
            )
        else:
            results["consensus_dir_acc"] = 0.0
    else:
        results["consensus_rate"] = 0.0
        results["consensus_strength"] = 0.0
        results["consensus_dir_acc"] = 0.0

    # ── 2. Direction Sharpe (PnL if trading in predicted direction) ──
    dir_pnl = pred_dir * actual_return
    results["direction_sharpe"] = float(np.mean(dir_pnl) / (np.std(dir_pnl) + 1e-8) * np.sqrt(104))

    # ── 3. Return correlation (Pearson) ──
    if pearsonr is not None:
        mask = ~np.isnan(actual_return) & ~np.isnan(pred_return)
        if mask.sum() > 2:
            r_val, _ = pearsonr(actual_return[mask], pred_return[mask])
            results["return_correlation"] = float(r_val)
        else:
            results["return_correlation"] = 0.0
    else:
        results["return_correlation"] = 0.0

    # ── 4. IC Rank (Spearman) ──
    if spearmanr is not None:
        mask = ~np.isnan(actual_return) & ~np.isnan(pred_return)
        if mask.sum() > 2:
            ic, _ = spearmanr(actual_return[mask], pred_return[mask])
            results["ic_rank"] = float(ic)
        else:
            results["ic_rank"] = 0.0
    else:
        results["ic_rank"] = 0.0

    # ── 5. Bias (systematic over/under-predict) ──
    results["bias"] = float(np.mean(pred_return - actual_return))

    # ── 6. MAE (mean absolute error in return space) ──
    results["mae"] = float(np.mean(np.abs(pred_return - actual_return)))

    # ── 7. Prediction variance (avg sample diversity) ──
    pred_std = np.std(samples[:, :N_eval, 3], axis=0)
    results["prediction_volatility"] = float(np.mean(pred_std / prev_close))

    # ── 8. TP / SL events — unfiltered (all steps) vs consensus-filtered ──
    actual_h = actual_df["high"].values[:N_eval]
    actual_l = actual_df["low"].values[:N_eval]

    def _tp_sl_stats(mask):
        tp_h = sl_h = w_cnt = l_cnt = total = 0
        w_sum = l_sum = 0.0
        for i in range(N_eval):
            if isinstance(mask, np.ndarray) and not mask[i]:
                continue
            total += 1
            hi, lo = float(actual_h[i]), float(actual_l[i])
            if pred_dir[i] == 1:
                tp_lvl = close_q80[i] if close_q80[i] > prev_close else np.inf
                sl_lvl = close_q20[i] if close_q20[i] < prev_close else -np.inf
                if lo <= sl_lvl:
                    sl_h += 1
                elif hi >= tp_lvl:
                    tp_h += 1
            elif pred_dir[i] == -1:
                tp_lvl = close_q20[i] if close_q20[i] < prev_close else -np.inf
                sl_lvl = close_q80[i] if close_q80[i] > prev_close else np.inf
                if hi >= sl_lvl:
                    sl_h += 1
                elif lo <= tp_lvl:
                    tp_h += 1
            r = actual_return[i]
            if r > 0:
                w_sum += abs(r)
                w_cnt += 1
            elif r < 0:
                l_sum += abs(r)
                l_cnt += 1
        th = tp_h + sl_h
        tpr = tp_h / max(th, 1)
        aw = w_sum / max(w_cnt, 1)
        al = l_sum / max(l_cnt, 1)
        exp = tpr * aw - (sl_h / max(th, 1)) * al
        return total, tpr, sl_h / max(th, 1), tp_h / max(th, 1), exp, w_sum, l_sum, w_cnt, l_cnt

    unf_total, unf_tpr, unf_slr, unf_ratio, unf_exp, *_ = _tp_sl_stats(slice(None))
    results["unfiltered_trades"] = float(unf_total)
    results["unfiltered_expectancy"] = float(unf_exp)

    if S == 5:
        c_total, c_tpr, c_slr, c_ratio, c_exp, c_ws, c_ls, c_wc, c_lc = _tp_sl_stats(consensus_mask)
        results["consensus_trades"] = float(c_total)
        results["expectancy"] = float(c_exp)

        if consensus_mask.sum() > 0 and consensus_dir is not None:
            cons_ret = consensus_dir[consensus_mask] * actual_return[consensus_mask]
            results["consensus_sharpe"] = float(
                np.mean(cons_ret) / (np.std(cons_ret) + 1e-8) * np.sqrt(104)
            )
        else:
            results["consensus_sharpe"] = 0.0
    else:
        results["consensus_trades"] = 0.0
        results["expectancy"] = unf_exp
        results["consensus_sharpe"] = 0.0

    # ── 9. Max drawdown (always-in-predicted-direction) ──
    cumulative = np.cumprod(1 + dir_pnl)
    peak = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - peak) / peak
    results["max_drawdown_simple"] = float(np.min(drawdown))

    return results
