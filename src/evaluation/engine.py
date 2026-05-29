"""
M-ENGINE: Backtest execution engine
Contract: signals array + data dict → metrics dict, per_bar array
Status: ✅ ready
"""

import numpy as np

from src.evaluation.metrics import (
    avg_return,
    bias,
    calmar_ratio,
    direction_accuracy,
    direction_sharpe,
    dsharpe_ratio,
    ic_rank,
    mae,
    max_drawdown,
    n_trades,
    profit_factor,
    psr,
    return_correlation,
    sharpe_ratio,
    sortino_ratio,
    trade_pct,
    win_rate,
)
from src.evaluation.simulation import simulate_trade

STRAT_METRICS = [
    ("sharpe",           "Sharpe (ann)",     lambda r, e, a: sharpe_ratio(r)),
    ("sortino",          "Sortino (ann)",    lambda r, e, a: sortino_ratio(r)),
    ("max_dd",           "Max Drawdown",     lambda r, e, a: max_drawdown(e)),
    ("profit_factor",    "Profit Factor",    lambda r, e, a: profit_factor(r[r != 0]) if (r != 0).sum() > 0 else 0.0),
    ("win_rate",         "Win Rate",         lambda r, e, a: win_rate(r[r != 0]) if (r != 0).sum() > 0 else 0.0),
    ("calmar",           "Calmar",           lambda r, e, a: calmar_ratio(r[r != 0], e) if (r != 0).sum() > 0 else 0.0),
    ("avg_return",       "Avg Return (ann)", lambda r, e, a: avg_return(r)),
    ("total_return",     "Total Return",     lambda r, e, a: float(e[-1] / e[0] - 1)),
    ("aver_ret",         "AverRet",          lambda r, e, a: float(a.get("total_pnl", r.sum()) / max(a["n_long"] + a["n_short"], 1))),
    ("trade_pct",        "Trade %",          lambda r, e, a: trade_pct(r)),
    ("psr",              "PSR",              lambda r, e, a: psr(r[r != 0]) if (r != 0).sum() > 1 else 0.0),
    ("dsr",              "DSR",              lambda r, e, a: dsharpe_ratio(r[r != 0]) if (r != 0).sum() > 1 else 0.0),
]

PRED_METRICS = [
    ("dir_acc",          "Direction Acc",    lambda r, e, a: direction_accuracy(a["actual"], a["pred"])),
    ("dir_sharpe",       "Direction Sharpe", lambda r, e, a: direction_sharpe(a["actual"], a["pred"])),
    ("return_corr",      "Return Corr",      lambda r, e, a: return_correlation(a["actual"], a["pred"])),
    ("ic_rank",          "IC (rank)",        lambda r, e, a: ic_rank(a["actual"], a["pred"])),
    ("bias",             "Bias",             lambda r, e, a: bias(a["actual"], a["pred"])),
    ("mae",              "MAE",              lambda r, e, a: mae(a["actual"], a["pred"])),
]

TRADE_METRICS = [
    ("n_trades",         "N trades",         lambda r, e, a: int(n_trades(r))),
    ("n_long",           "N long",           lambda r, e, a: a.get("n_long", 0)),
    ("n_short",          "N short",          lambda r, e, a: a.get("n_short", 0)),
    ("n_bar_active",     "Active bars",      lambda r, e, a: int((r != 0).sum())),
]

ANNUAL_BARS = 6 * 24 * 252  # 10min bars ≈ 36288


def compute_all_metrics(per_bar, cumul, actual_ret, pred_ret, signals):
    extra = {
        "n_long": int((signals == 1).sum()),
        "n_short": int((signals == -1).sum()),
        "total_pnl": float(per_bar.sum()),
        "actual": actual_ret,
        "pred": pred_ret,
    }
    m = {}
    for key, _, fn in STRAT_METRICS + PRED_METRICS + TRADE_METRICS:
        m[key] = float(fn(per_bar, cumul, extra))
    return m


def get_tp_sl(i, sig, entry_close, g_tp, g_sl):
    ec = entry_close[i]
    if sig == 1:
        tp = g_tp[i] if g_tp[i] > ec else np.inf
        sl = g_sl[i] if g_sl[i] < ec else -np.inf
    else:
        tp = g_sl[i] if g_sl[i] < ec else -np.inf
        sl = g_tp[i] if g_tp[i] > ec else np.inf
    return tp, sl


def run_backtest(signals, name, data, get_tp_sl_fn=get_tp_sl, pred_ret_ref=None, verbose=True):
    N = data['N']
    raw = data['raw']
    LK = data['LK']
    PL = data['PL']
    COMM = data['COMM']
    actual_ret = data['actual_ret']
    pred_ret = pred_ret_ref if pred_ret_ref is not None else data['pred_ret']

    per_bar = np.zeros(len(raw))
    n_pos = np.zeros(len(raw), dtype=int)

    for i in range(N):
        sig = signals[i]
        if sig == 0:
            continue
        tp, sl = get_tp_sl_fn(i, sig, data['entry_close'], data['g_tp'], data['g_sl'])
        exit_bar, ret, holding = simulate_trade(
            i, sig, tp, sl,
            data['entry_open'], data['raw'], data['LK'], data['PL'],
        )
        if exit_bar is None or ret == 0.0:
            continue
        ret -= COMM
        for s in range(holding):
            bi = i + LK + s
            if bi >= len(raw):
                break
            per_bar[bi] += ret / holding
            n_pos[bi] += 1

    mask = n_pos > 0
    per_bar[mask] = per_bar[mask] / n_pos[mask]
    cumul = np.cumprod(1 + per_bar)

    m = compute_all_metrics(per_bar, cumul, actual_ret, pred_ret, signals)
    m["label"] = name

    if verbose:
        print(f"  {name:<35s}  ret={m['total_return']:>7.2%}  "
              f"Sharpe={m['sharpe']:>7.3f}  DD={m['max_dd']:>7.2%}  "
              f"WR={m['win_rate']:>6.2%}  PF={m['profit_factor']:>5.3f}  "
              f"AverRet={m['aver_ret']:>7.4%}  "
              f"trades={int(m['n_trades']):>5d}  bars={int(m['n_bar_active']):>5d}")
    return m, per_bar


def run_backtest_custom(signals, name, get_tp_sl_fn, data, pred_ret_ref=None, verbose=True):
    N = data['N']
    raw = data['raw']
    LK = data['LK']
    PL = data['PL']
    COMM = data['COMM']
    actual_ret = data['actual_ret']
    pred_ret = pred_ret_ref if pred_ret_ref is not None else data['pred_ret']

    per_bar = np.zeros(len(raw))
    n_pos = np.zeros(len(raw), dtype=int)

    for i in range(N):
        sig = signals[i]
        if sig == 0:
            continue
        tp, sl = get_tp_sl_fn(i, sig, data['entry_close'], data['g_tp'], data['g_sl'])
        exit_bar, ret, holding = simulate_trade(
            i, sig, tp, sl,
            data['entry_open'], data['raw'], data['LK'], data['PL'],
        )
        if exit_bar is None or ret == 0.0:
            continue
        ret -= COMM
        for s in range(holding):
            bi = i + LK + s
            if bi >= len(raw):
                break
            per_bar[bi] += ret / holding
            n_pos[bi] += 1

    mask = n_pos > 0
    per_bar[mask] = per_bar[mask] / n_pos[mask]
    cumul = np.cumprod(1 + per_bar)

    m = compute_all_metrics(per_bar, cumul, actual_ret, pred_ret, signals)
    m["label"] = name

    if verbose:
        print(f"  {name:<35s}  ret={m['total_return']:>7.2%}  "
              f"Sharpe={m['sharpe']:>7.3f}  DD={m['max_dd']:>7.2%}  "
              f"WR={m['win_rate']:>6.2%}  PF={m['profit_factor']:>5.3f}  "
              f"AverRet={m['aver_ret']:>7.4%}  "
              f"trades={int(m['n_trades']):>5d}  bars={int(m['n_bar_active']):>5d}")
    return m, per_bar
