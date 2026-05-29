"""
M-METRICS: Financial strategy evaluation metrics
Contract: returns array, equity array, pred/actual arrays → metrics dict
Status: ✅ ready
"""

import numpy as np

try:
    from scipy.stats import norm, pearsonr, spearmanr
except ImportError:
    spearmanr = pearsonr = norm = None

BARS_PER_DAY = 52
TRADING_DAYS = 252
PERIODS_ANN = BARS_PER_DAY * TRADING_DAYS


def sharpe_ratio(returns: np.ndarray, rf: float = 0.0, periods: int = PERIODS_ANN) -> float:
    if len(returns) < 2:
        return 0.0
    excess = returns - rf / periods
    return np.sqrt(periods) * np.mean(excess) / (np.std(excess) + 1e-8)


def max_drawdown(equity: np.ndarray) -> float:
    if len(equity) < 2:
        return 0.0
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    return float(np.min(dd))


def profit_factor(returns: np.ndarray) -> float:
    gross_profit = returns[returns > 0].sum()
    gross_loss = abs(returns[returns < 0].sum())
    return gross_profit / (gross_loss + 1e-8)


def win_rate(returns: np.ndarray) -> float:
    if len(returns) == 0:
        return 0.0
    return float(np.mean(returns > 0))


def calmar_ratio(returns: np.ndarray, equity: np.ndarray) -> float:
    ann_ret = np.mean(returns) * PERIODS_ANN
    mdd = max_drawdown(equity)
    return ann_ret / (abs(mdd) + 1e-8)


def direction_accuracy(actual: np.ndarray, pred: np.ndarray) -> float:
    act_dir = np.sign(actual)
    pred_dir = np.sign(pred)
    non_flat = pred_dir != 0
    if non_flat.sum() == 0:
        return 0.0
    return float(np.mean(act_dir[non_flat] == pred_dir[non_flat]))


def direction_sharpe(actual: np.ndarray, pred: np.ndarray) -> float:
    pnl = np.sign(pred) * actual
    return sharpe_ratio(pnl)


def return_correlation(actual: np.ndarray, pred: np.ndarray) -> float:
    if pearsonr is None:
        return 0.0
    mask = ~np.isnan(actual) & ~np.isnan(pred)
    if mask.sum() < 3:
        return 0.0
    r_val, _ = pearsonr(actual[mask], pred[mask])
    return float(r_val)


def ic_rank(actual: np.ndarray, pred: np.ndarray) -> float:
    if spearmanr is None:
        return 0.0
    mask = ~np.isnan(actual) & ~np.isnan(pred)
    if mask.sum() < 3:
        return 0.0
    ic, _ = spearmanr(actual[mask], pred[mask])
    return float(ic)


def bias(actual: np.ndarray, pred: np.ndarray) -> float:
    if len(actual) == 0:
        return 0.0
    return float(np.mean(pred - actual))


def mae(actual: np.ndarray, pred: np.ndarray) -> float:
    if len(actual) == 0:
        return 0.0
    return float(np.mean(np.abs(pred - actual)))


def prediction_volatility(samples: np.ndarray, prev_close: float) -> float:
    pred_std = np.std(samples[:, :, 3], axis=0)
    return float(np.mean(pred_std / max(prev_close, 1e-8)))


def _central_skewness(x: np.ndarray) -> float:
    """Central skewness E[(x-μ)³] / σ³."""
    mu = np.mean(x)
    s = np.std(x, ddof=0)
    if s < 1e-12:
        return 0.0
    return float(np.mean((x - mu) ** 3) / s ** 3)


def _central_kurtosis(x: np.ndarray) -> float:
    """Central excess kurtosis E[(x-μ)⁴] / σ⁴ - 3."""
    mu = np.mean(x)
    s = np.std(x, ddof=0)
    if s < 1e-12:
        return 0.0
    return float(np.mean((x - mu) ** 4) / s ** 4) - 3.0


def psr(returns: np.ndarray, target_sharpe: float = 0.0) -> float:
    """Probabilistic Sharpe Ratio (Egozcue & Ziemba 2009).

    Uses period-level Sharpe (non-annualized) with central skewness/kurtosis.
    """
    if norm is None:
        return 0.0
    n = len(returns)
    if n < 2:
        return 0.0
    mu = np.mean(returns)
    sigma = np.std(returns, ddof=0)
    if sigma < 1e-12:
        return 1.0 if mu >= target_sharpe else 0.0
    sr_period = mu / sigma
    s3 = _central_skewness(returns)
    s4 = _central_kurtosis(returns)
    var_est = (1.0 + 0.5 * sr_period ** 2 - s3 * sr_period + s4 * sr_period ** 2 / 4.0) / (n - 1) if n > 1 else 1.0
    z = (sr_period - target_sharpe) / (np.sqrt(var_est) + 1e-8)
    return float(norm.cdf(z))


def dsharpe_ratio(returns: np.ndarray, num_trials: int = 1) -> float:
    """Deflated Sharpe Ratio (Bailey & López de Prado 2012).

    Adjusts PSR for multiple testing across num_trials strategies.
    Default num_trials=1 returns PSR (no deflation).
    """
    if norm is None:
        return 0.0
    if num_trials < 2:
        return psr(returns)
    n = len(returns)
    if n < 2:
        return 0.0
    mu = np.mean(returns)
    sigma = np.std(returns, ddof=0)
    if sigma < 1e-12:
        return 0.5
    sr_period = mu / sigma
    s3 = _central_skewness(returns)
    s4 = _central_kurtosis(returns)
    var_est = (1.0 + 0.5 * sr_period ** 2 - s3 * sr_period + s4 * sr_period ** 2 / 4.0) / (n - 1)
    gamma = 0.5772156649  # Euler-Mascheroni constant
    z_max = (1 - gamma) * norm.ppf(1 - 1.0 / num_trials) + gamma * norm.ppf(1 - 1.0 / (num_trials * np.e))
    z = (sr_period * np.sqrt(n - 1) - z_max) / (np.sqrt(var_est * (n - 1)) + 1e-8)
    return float(norm.cdf(z))


def sortino_ratio(returns: np.ndarray, rf: float = 0.0, periods: int = PERIODS_ANN) -> float:
    """Sortino ratio — RMS of negative deviations (semi-deviation), not std of downside."""
    if len(returns) < 2:
        return 0.0
    excess = returns - rf / periods
    downside = np.minimum(excess, 0.0)
    downside_risk = np.sqrt(np.mean(downside ** 2))
    if downside_risk < 1e-8:
        return 0.0
    return np.sqrt(periods) * np.mean(excess) / downside_risk


def avg_return(returns: np.ndarray, periods: int = PERIODS_ANN) -> float:
    """Annualized average return."""
    if len(returns) == 0:
        return 0.0
    return float(np.mean(returns) * periods)


def n_trades(returns: np.ndarray) -> int:
    """Number of non-zero return observations (active bars, not trade entries)."""
    return int(np.sum(returns != 0))


def trade_pct(returns: np.ndarray) -> float:
    """Fraction of periods with active trades."""
    if len(returns) == 0:
        return 0.0
    return float(n_trades(returns) / len(returns))


class StrategyMetrics:
    def __init__(
        self,
        name: str = "",
        total_trades: int = 0,
        long_trades: int = 0,
        short_trades: int = 0,
        win_rate: float = 0.0,
        profit_factor: float = 0.0,
        sharpe_ratio: float = 0.0,
        max_drawdown: float = 0.0,
        total_return: float = 0.0,
        avg_return: float = 0.0,
        avg_duration: float = 0.0,
        calmar_ratio: float = 0.0,
        psr_score: float = 0.0,
        dsr_score: float = 0.0,
    ):
        self.name = name
        self.total_trades = total_trades
        self.long_trades = long_trades
        self.short_trades = short_trades
        self.long_pct = long_trades / max(total_trades, 1)
        self.short_pct = short_trades / max(total_trades, 1)
        self.win_rate = win_rate
        self.profit_factor = profit_factor
        self.sharpe_ratio = sharpe_ratio
        self.max_drawdown = max_drawdown
        self.total_return = total_return
        self.avg_return = avg_return
        self.avg_duration = avg_duration
        self.calmar_ratio = calmar_ratio
        self.psr_score = psr_score
        self.dsr_score = dsr_score

    @classmethod
    def from_trades(cls, trades, equity, name=""):
        n = len(trades)
        if n == 0 or len(equity) < 2:
            return cls(name=name)
        pnls = np.array([t.pnl_pct for t in trades])
        sides = np.array([getattr(t, "side", "LONG") for t in trades])
        total_ret = equity[-1] / equity[0] - 1.0
        durations = np.array([t.duration for t in trades])
        long_n = int(np.sum(sides == "LONG"))
        short_n = int(np.sum(sides == "SHORT"))
        return cls(
            name=name,
            total_trades=n,
            long_trades=long_n,
            short_trades=short_n,
            win_rate=win_rate(pnls),
            profit_factor=profit_factor(pnls),
            sharpe_ratio=sharpe_ratio(pnls),
            max_drawdown=max_drawdown(equity),
            total_return=total_ret,
            avg_return=float(np.mean(pnls)),
            avg_duration=float(np.mean(durations)),
            calmar_ratio=calmar_ratio(pnls, equity),
            psr_score=psr(pnls),
            dsr_score=dsharpe_ratio(pnls),
        )

    def __repr__(self) -> str:
        return (
            f"{self.name}: trades={self.total_trades} "
            f"L={self.long_pct:.0%}/S={self.short_pct:.0%} "
            f"WR={self.win_rate:.1%} PF={self.profit_factor:.2f} "
            f"Sharpe={self.sharpe_ratio:.2f} "
            f"MaxDD={self.max_drawdown:.1%} Ret={self.total_return:.2%}"
        )


def evaluate_model(
    actual: np.ndarray,
    pred: np.ndarray,
    samples: np.ndarray,
    prev_close: float,
) -> dict:
    actual_r = (actual[:, 3] - prev_close) / prev_close
    pred_r = (pred[:, 3] - prev_close) / prev_close

    metrics = {
        "direction_accuracy": direction_accuracy(actual_r, pred_r),
        "direction_sharpe": direction_sharpe(actual_r, pred_r),
        "return_correlation": return_correlation(actual_r, pred_r),
        "ic_rank": ic_rank(actual_r, pred_r),
        "bias": bias(actual_r, pred_r),
        "mae": mae(actual_r, pred_r),
        "prediction_volatility": prediction_volatility(samples, prev_close),
    }

    dir_pnl = np.sign(pred_r) * actual_r
    cumulative = np.cumprod(1 + dir_pnl)
    peak = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - peak) / peak
    metrics["max_drawdown_simple"] = float(np.min(drawdown))

    return metrics
