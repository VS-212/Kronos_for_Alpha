"""
M-SIGNAL-ATOMS: Composable signal atoms from Kronos samples
Contract: (sample_count, pred_len) close-price array + prev_close → dict of derived signals
Status: ✅ ready
"""

"""Composable signal atoms from Kronos sample predictions.

Each atom takes a (sample_count, pred_len) close-price array + prev_close
and returns a dict of derived signals. Atoms are pure numpy — no pandas.
"""

import numpy as np


def direction(close_samples: np.ndarray, prev_close: float) -> dict:
    """Per-sample direction at each step.

    Args:
        close_samples: (sample_count, pred_len) float32
        prev_close: scalar entry reference

    Returns:
        dir_sign: (sample_count, pred_len) int8 — +1 if up, -1 if down, 0 if flat
        dir_return: (sample_count, pred_len) — fractional return per sample/step
    """
    ret = (close_samples - prev_close) / prev_close
    return {
        "dir_sign": np.sign(ret, dtype=np.int8),
        "dir_return": ret,
    }


def consensus(close_samples: np.ndarray, prev_close: float, threshold: float = 0.8) -> dict:
    """Direction consensus across samples.

    Consensus = fraction of samples that agree on direction (up/down).

    Returns:
        consensus_up: (pred_len,) float — fraction of samples predicting up
        consensus_dn: (pred_len,) float — fraction predicting down
        has_consensus: (pred_len,) bool — True where either side >= threshold
        consensus_dir: (pred_len,) int8 — +1 (up) or -1 (down) where has_consensus, else 0
        consensus_strength: (pred_len,) float — max(up, dn)
    """
    S = close_samples.shape[0]
    up = np.mean(close_samples > prev_close, axis=0)
    dn = np.mean(close_samples < prev_close, axis=0)
    has_cons = (up >= threshold) | (dn >= threshold)
    c_dir = np.where(up >= threshold, 1, np.where(dn >= threshold, -1, 0)).astype(np.int8)
    return {
        "consensus_up": up,
        "consensus_dn": dn,
        "has_consensus": has_cons,
        "consensus_dir": c_dir,
        "consensus_strength": np.maximum(up, dn),
    }


def boundaries(
    close_samples: np.ndarray, prev_close: float, tp_q: float = 0.80, sl_q: float = 0.20
) -> dict:
    """TP/SL price boundaries from sample quantiles.

    Returns:
        q80: (pred_len,) — upper quantile per step
        q20: (pred_len,) — lower quantile per step
        tp_long: (pred_len,) — TP for long trades (q80, capped above prev_close)
        sl_long: (pred_len,) — SL for long trades (q20, capped below prev_close)
        tp_short: (pred_len,) — TP for short trades (q20, capped below prev_close)
        sl_short: (pred_len,) — SL for short trades (q80, capped above prev_close)
    """
    q80 = np.quantile(close_samples, tp_q, axis=0)
    q20 = np.quantile(close_samples, sl_q, axis=0)

    tp_long = np.where(q80 > prev_close, q80, np.inf)
    sl_long = np.where(q20 < prev_close, q20, -np.inf)
    tp_short = np.where(q20 < prev_close, q20, -np.inf)
    sl_short = np.where(q80 > prev_close, q80, np.inf)

    return {
        "q80": q80,
        "q20": q20,
        "tp_long": tp_long,
        "sl_long": sl_long,
        "tp_short": tp_short,
        "sl_short": sl_short,
    }


def dispersion(close_samples: np.ndarray, prev_close: float) -> dict:
    """Sample dispersion (uncertainty) metrics.

    Returns:
        std_return: (pred_len,) — std of returns across samples per step
        mean_std: float — average of std_return
        range_pct: (pred_len,) — (max - min) / prev_close per step
    """
    ret = (close_samples - prev_close) / prev_close
    std_r = np.std(ret, axis=0)
    rng = (np.max(close_samples, axis=0) - np.min(close_samples, axis=0)) / prev_close
    return {
        "std_return": std_r,
        "mean_std": float(np.mean(std_r)),
        "range_pct": rng,
    }


def trend_strength(close_samples: np.ndarray, prev_close: float) -> dict:
    """Magnitude of directional agreement across samples (last step).

    Returns:
        trend_strength: scalar — |mean(sign(pred_return))|, 0..1
        mean_return: scalar — average predicted return (last step)
    """
    S = close_samples.shape[0]
    last_close = close_samples[:, -1]
    ret = (last_close - prev_close) / prev_close
    dirs = np.sign(ret)
    mean_ret = float(np.mean(ret))
    strength = float(np.abs(np.mean(dirs)))
    return {
        "trend_strength": strength,
        "mean_return": mean_ret,
    }


def linearity(close_samples: np.ndarray) -> dict:
    """R² of linear trend per sample path, averaged.

    High linearity = sample paths are straight lines (steady trend).
    Low linearity = sample paths are curved / noisy.

    Returns:
        r2_per_sample: (sample_count,) — R² per sample
        mean_r2: float — average R² across samples
    """
    S, N = close_samples.shape
    x = np.arange(N, dtype=np.float32)
    x_mean = x.mean()
    x_centered = x - x_mean
    denom = (x_centered**2).sum()

    r2s = np.zeros(S, dtype=np.float32)
    for s in range(S):
        y = close_samples[s]
        y_mean = y.mean()
        slope = (x_centered * (y - y_mean)).sum() / denom
        y_pred = y_mean + slope * x_centered
        ss_res = ((y - y_pred) ** 2).sum()
        ss_tot = ((y - y_mean) ** 2).sum()
        r2s[s] = 1.0 - ss_res / max(ss_tot, 1e-10)

    return {
        "r2_per_sample": r2s,
        "mean_r2": float(r2s.mean()),
    }


def asymmetry(close_samples: np.ndarray, prev_close: float) -> dict:
    """Return asymmetry: magnitude-weighted direction bias.

    Instead of counting votes (consensus), weights each sample's vote
    by its predicted return. Catches cases where minority direction
    has much larger magnitude than majority.

    Returns:
        net_return: scalar — mean return across all samples and steps
        up_magnitude: scalar — mean of positive returns
        dn_magnitude: scalar — mean of abs(negative returns)
        asymmetry_ratio: scalar — up_magnitude / dn_magnitude (> 1 = bullish)
        direction: int — +1 if net_return > 0, -1 otherwise
    """
    ret = (close_samples - prev_close) / prev_close
    net_return = float(np.mean(ret))

    up = ret[ret > 0]
    dn = ret[ret < 0]
    up_mag = float(np.mean(up)) if len(up) > 0 else 0.0
    dn_mag = float(np.mean(np.abs(dn))) if len(dn) > 0 else 0.0
    asym_ratio = up_mag / max(dn_mag, 1e-10)

    return {
        "net_return": net_return,
        "up_magnitude": up_mag,
        "dn_magnitude": dn_mag,
        "asymmetry_ratio": asym_ratio,
        "asymmetry_dir": 1 if net_return > 0 else -1,
    }


def expectancy(close_samples: np.ndarray, prev_close: float) -> dict:
    """Risk-adjusted return expectation.

    Returns:
        mean_return: (pred_len,) — average return per step
        sharpe_step: (pred_len,) — mean/std per step
        sharpe_avg: float — mean of sharpe_step
    """
    ret = (close_samples - prev_close) / prev_close
    mu = np.mean(ret, axis=0)
    sigma = np.std(ret, axis=0) + 1e-10
    sr = mu / sigma
    return {
        "mean_return": mu,
        "sharpe_step": sr,
        "sharpe_avg": float(np.mean(sr)),
    }
