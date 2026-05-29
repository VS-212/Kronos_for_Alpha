"""
M-REGIME: BB regime classification + temporal analysis
Contract: data dict + per_bar arrays → regime tables, filtered signals
Status: ✅ ready
"""

from datetime import datetime

import numpy as np

__all__ = [
    "apply_filters",
    "build_regime_masks",
    "champion_breakdown",
    "classify_regimes",
    "compute_bar_metadata",
    "compute_quarterly_breakdown",
    "compute_raw_bb_mapping",
    "top5_15h_16h",
]


def compute_bar_metadata(ts):
    """Returns (bar_hour, bar_dow) arrays for each bar index."""
    n_raw = len(ts)
    bar_hour = np.zeros(n_raw, dtype=int)
    bar_dow = np.zeros(n_raw, dtype=int)
    for bi, t in enumerate(ts):
        dt = datetime.fromisoformat(str(t))
        bar_hour[bi] = dt.hour
        bar_dow[bi] = dt.weekday()
    return bar_hour, bar_dow


def compute_raw_bb_mapping(bb_width, bb_pct, wf_width, bb_mom, N, LK, n_raw):
    """Map per-window BB values to raw bar indices with -1 shift (no lookahead).

    Each raw bar bi receives the BB regime from the window that closed at bi-1,
    avoiding any forward-looking bias.
    """
    raw_bb_width = np.zeros(n_raw)
    raw_bb_pct = np.zeros(n_raw)
    raw_wf_width = np.zeros(n_raw)
    raw_bb_mom = np.zeros(n_raw, dtype=bool)
    for bi in range(n_raw):
        wi = bi - LK
        if 0 <= wi < N:
            raw_bb_width[bi] = bb_width[wi]
            raw_bb_pct[bi] = bb_pct[wi]
            raw_wf_width[bi] = wf_width[wi]
            raw_bb_mom[bi] = bb_mom[wi]
        else:
            raw_bb_width[bi] = bb_width[0] if N > 0 else 0.0
            raw_bb_pct[bi] = 0.5
            raw_wf_width[bi] = wf_width[0] if N > 0 else 0.001
            raw_bb_mom[bi] = True
    return raw_bb_width, raw_bb_pct, raw_wf_width, raw_bb_mom


def classify_regimes(active_h1, pb_champ, raw_bb_width, raw_wf_width, raw_bb_pct, raw_bb_mom, label="Strategy"):
    """Classify each bar into BB regime.

    Regimes are: {high_vol, low_vol} x {oversold, overbought, middle} x {expanding, contracting}
    Prints formatted table with TotRet, PF, AverRet per regime.
    Returns regimes dict mapping regime name -> list of returns.
    """
    regimes = {}
    for bi in np.where(active_h1)[0]:
        rr = pb_champ[bi]
        vol_reg = "high_vol" if raw_bb_width[bi] >= raw_wf_width[bi] else "low_vol"
        zone = "oversold" if raw_bb_pct[bi] < 0.3 else ("overbought" if raw_bb_pct[bi] > 0.7 else "middle")
        mom = "expanding" if raw_bb_mom[bi] else "contracting"
        reg = f"{vol_reg}_{zone}_{mom}"
        regimes.setdefault(reg, []).append(rr)

    print(f"\n  {'═' * 100}")
    print(f"  {label} — H1 2026 by BB regime")
    print(f"  {'═' * 100}")
    print(f"  {'Regime':<35} {'Bars':<7} {'TotRet':<10} {'PF':<8} {'AverRet':<10}")
    print(f"  {'─' * 70}")
    for reg in sorted(regimes.keys(), key=lambda k: -abs(sum(regimes[k]))):
        rr = np.array(regimes[reg])
        n = len(rr)
        if n < 5:
            continue
        tot = float(rr.sum())
        pos = rr[rr > 0].sum()
        neg = abs(rr[rr < 0].sum())
        pf = float(pos / max(neg, 1e-12))
        ar = float(rr.mean())
        print(f"  {reg:<35} {n:<7} {tot:>8.4%}  {pf:>6.4f}  {ar:>8.5%}")
    return regimes


def build_regime_masks(raw_bb_width, raw_wf_width, raw_bb_pct, raw_bb_mom, N, LK, n_raw):
    """Returns reg_bad, reg_best (per-window bool arrays), reg_bad4 (4-regime filter).

    reg_bad:   high_vol + oversold + expanding  (bad for longs)
    reg_best:  high_vol + middle + contracting   (best regime)
    reg_bad4:  4 specific regimes to filter out (overbought cntr, oversold cntr, oversold exp, low_vol_oversold_exp)
    """
    reg_bad = np.zeros(N, dtype=bool)
    reg_best = np.zeros(N, dtype=bool)
    for i in range(N):
        pos = i + LK - 1
        if pos >= n_raw:
            continue
        hv = raw_bb_width[pos] >= raw_wf_width[pos]
        os = raw_bb_pct[pos] < 0.3
        ex = raw_bb_mom[pos]
        md = (not os) and raw_bb_pct[pos] < 0.7
        reg_bad[i] = hv and os and ex
        reg_best[i] = hv and md and (not ex)

    reg_bad4 = np.zeros(N, dtype=bool)
    for i in range(N):
        pos = i + LK - 1
        if pos >= n_raw:
            continue
        hv = raw_bb_width[pos] >= raw_wf_width[pos]
        os = raw_bb_pct[pos] < 0.3
        ob = raw_bb_pct[pos] > 0.7
        ex = raw_bb_mom[pos]
        cntr = not ex
        reg_bad4[i] = (hv and ob and cntr) or (hv and os and cntr) or (hv and os and ex) or (not hv and os and ex)

    return reg_bad, reg_best, reg_bad4


def compute_quarterly_breakdown(pb, bars_per_qtr, sorted_qtrs):
    """Per-quarter: AverRet, PF, WinRate, n_active.

    Returns list of (qtr_label, aver_ret, pf, win_rate, n_active) tuples,
    including 'overall' as the last row.
    """
    rows = []
    for qtr in sorted_qtrs:
        mask = bars_per_qtr[qtr]
        active = pb[mask] != 0
        n_active = int(active.sum())
        if n_active < 5:
            continue
        rtrn = pb[mask][active]
        pos_sum = rtrn[rtrn > 0].sum()
        neg_sum = abs(rtrn[rtrn < 0].sum())
        pf = float(pos_sum / max(neg_sum, 1e-12))
        ar = float(rtrn.mean())
        wr = float((rtrn > 0).sum() / max(n_active, 1))
        rows.append((qtr, ar, pf, wr, n_active))
    active_all = pb != 0
    if active_all.sum() > 5:
        ra = pb[active_all]
        pos_sum = ra[ra > 0].sum()
        neg_sum = abs(ra[ra < 0].sum())
        pf = float(pos_sum / max(neg_sum, 1e-12))
        ar = float(ra.mean())
        wr = float((ra > 0).sum() / max(int(active_all.sum()), 1))
        rows.append(("overall", ar, pf, wr, int(active_all.sum())))
    return rows


def apply_filters(sig, data, has_bbpct=False, has_bbwidth=False, has_bbmom=False,
                  has_conftrend=False, has_rsiextreme=False, has_vol=False, has_mcbig=False):
    """Apply BB/confidence/RSI/volatility filters to a signal array.

    Parameters
    ----------
    sig : np.ndarray of int (N,) — signal values: -1, 0, 1
    data : dict with keys: bb_pct, bb_width, wf_width, bb_mom, conf_trend,
        rsi14, vol_at_close, wf_vol, mc_std (all per-window arrays of length N)
    has_* : bool — which filters to apply

    Returns filtered signal copy.
    """
    N = len(sig)
    out = sig.copy()
    bb_pct = data.get("bb_pct")
    bb_width = data.get("bb_width")
    wf_width = data.get("wf_width")
    bb_mom = data.get("bb_mom")
    conf_trend = data.get("conf_trend")
    rsi14 = data.get("rsi14")
    vol_at_close = data.get("vol_at_close")
    wf_vol = data.get("wf_vol")
    mc_std = data.get("mc_std")
    for i in range(N):
        if out[i] == 0:
            continue
        if has_bbpct and ((out[i] == 1 and bb_pct[i] > 0.3) or (out[i] == -1 and bb_pct[i] < 0.7)):
            out[i] = 0
        if has_bbwidth and bb_width[i] < wf_width[i]:
            out[i] = 0
        if has_bbmom and not bb_mom[i]:
            out[i] = 0
        if has_conftrend and not conf_trend[i]:
            out[i] = 0
        if has_rsiextreme and ((out[i] == 1 and rsi14[i] > 70) or (out[i] == -1 and rsi14[i] < 30)):
            out[i] = 0
        if has_vol and vol_at_close[i] < wf_vol[i]:
            out[i] = 0
        if has_mcbig and i > 10 and mc_std[i] < np.median(mc_std[:i]):
            out[i] = 0
    return out


def champion_breakdown(labels, per_bars, ts, results, champion_label,
                       champion_data):
    """Orchestrator: hour breakdown, day-of-week, regime breakdown,
    regime filter, 4-regime filter, top 5 comparison.

    Parameters
    ----------
    labels : list of str — strategy names
    per_bars : list of np.ndarray — per-bar returns for each strategy
    ts : np.ndarray — timestamps for raw bars
    results : list — appended with regime-filtered strategy results
    champion_label : str — label of the champion strategy to analyze
    champion_data : dict — all required context:
        raw, N, LK, bb_width, bb_pct, wf_width, bb_mom, h1_2026_mask,
        get_tp_sl_no_tp (callable), run_backtest_custom (callable),
        base_signals (dict), pred_ret, wf_q90, wf_q10,
        rsi14, conf_trend, vol_at_close, wf_vol, mc_std,
        sorted_qtrs (list of str), bars_per_qtr (dict str->np.ndarray)
    """
    raw = champion_data["raw"]
    N = champion_data["N"]
    LK = champion_data["LK"]
    bb_width = champion_data["bb_width"]
    bb_pct = champion_data["bb_pct"]
    wf_width = champion_data["wf_width"]
    bb_mom = champion_data["bb_mom"]
    h1_2026_mask = champion_data["h1_2026_mask"]
    get_tp_sl_no_tp = champion_data["get_tp_sl_no_tp"]
    run_backtest_custom = champion_data["run_backtest_custom"]
    base_signals = champion_data["base_signals"]
    pred_ret = champion_data["pred_ret"]
    wf_q90 = champion_data["wf_q90"]
    wf_q10 = champion_data["wf_q10"]
    rsi14 = champion_data["rsi14"]
    conf_trend = champion_data["conf_trend"]
    vol_at_close = champion_data["vol_at_close"]
    wf_vol = champion_data["wf_vol"]
    mc_std = champion_data["mc_std"]
    sorted_qtrs = champion_data["sorted_qtrs"]
    bars_per_qtr = champion_data["bars_per_qtr"]

    n_raw = len(raw)
    filter_data = {"bb_pct": bb_pct, "bb_width": bb_width, "wf_width": wf_width,
                   "bb_mom": bb_mom, "conf_trend": conf_trend, "rsi14": rsi14,
                   "vol_at_close": vol_at_close, "wf_vol": wf_vol, "mc_std": mc_std}

    idx = labels.index(champion_label) if champion_label in labels else -1
    if idx < 0:
        print(f"  Champion label '{champion_label}' not found.")
        return
    pb_champ = per_bars[idx]

    bar_hour, bar_dow = compute_bar_metadata(ts)
    active_h1 = h1_2026_mask & (pb_champ != 0)
    bar_returns = pb_champ[active_h1]
    bar_hours = bar_hour[active_h1]
    bar_dows = bar_dow[active_h1]

    DOW_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri"]

    print(f"\n  {'═' * 80}")
    print(f"  {champion_label} — H1 2026 by hour")
    print(f"  {'═' * 80}")
    print(f"  {'Hour':<6} {'Bars':<7} {'TotRet':<9} {'PF':<8} {'AverRet':<10}")
    print(f"  {'─' * 40}")
    for hr in sorted(set(bar_hours)):
        mask = bar_hours == hr
        n = int(mask.sum())
        if n < 5:
            continue
        rr = bar_returns[mask]
        tot = float(rr.sum())
        pos = rr[rr > 0].sum()
        neg = abs(rr[rr < 0].sum())
        pf = float(pos / max(neg, 1e-12))
        ar = float(rr.mean())
        print(f"  {hr:>2}h   {n:<7} {tot:>7.4%}  {pf:>6.4f}  {ar:>8.5%}")

    print(f"\n  {'═' * 80}")
    print(f"  {champion_label} — H1 2026 by day of week")
    print(f"  {'═' * 80}")
    print(f"  {'Day':<6} {'Bars':<7} {'TotRet':<9} {'PF':<8} {'AverRet':<10}")
    print(f"  {'─' * 40}")
    for dw in range(5):
        mask = bar_dows == dw
        n = int(mask.sum())
        if n < 5:
            continue
        rr = bar_returns[mask]
        tot = float(rr.sum())
        pos = rr[rr > 0].sum()
        neg = abs(rr[rr < 0].sum())
        pf = float(pos / max(neg, 1e-12))
        ar = float(rr.mean())
        print(f"  {DOW_NAMES[dw]:<6} {n:<7} {tot:>7.4%}  {pf:>6.4f}  {ar:>8.5%}")

    raw_bb_width, raw_bb_pct, raw_wf_width, raw_bb_mom = compute_raw_bb_mapping(
        bb_width, bb_pct, wf_width, bb_mom, N, LK, n_raw)

    classify_regimes(active_h1, pb_champ, raw_bb_width, raw_wf_width,
                     raw_bb_pct, raw_bb_mom, label=champion_label)

    reg_bad, reg_best, reg_bad4 = build_regime_masks(
        raw_bb_width, raw_wf_width, raw_bb_pct, raw_bb_mom, N, LK, n_raw)

    sig_w = np.zeros(N, dtype=int)
    sig_w[pred_ret > wf_q90] = 1
    sig_w[pred_ret < wf_q10] = -1

    s = sig_w.copy()
    s[reg_bad] = 0
    r, _pb = run_backtest_custom(s, "WF+skipBadReg noTP", get_tp_sl_no_tp,
                                 pred_ret_ref=pred_ret)
    results.append(r)
    per_bars.append(_pb)

    s = np.zeros(N, dtype=int)
    for i in range(N):
        if reg_best[i] and pred_ret[i] > wf_q90[i]:
            s[i] = 1
        if reg_best[i] and pred_ret[i] < wf_q10[i]:
            s[i] = -1
    r, _pb = run_backtest_custom(s, "WF+onlyBestReg noTP", get_tp_sl_no_tp,
                                 pred_ret_ref=pred_ret)
    results.append(r)
    per_bars.append(_pb)

    s = apply_filters(sig_w, filter_data, has_bbpct=True)
    s[reg_bad] = 0
    r, _pb = run_backtest_custom(s, "WF+BB%B+skipBadReg noTP", get_tp_sl_no_tp,
                                 pred_ret_ref=pred_ret)
    results.append(r)
    per_bars.append(_pb)

    TOP5 = [
        ("WF+BB%B+BBmom noTP",           "WF",    True,  False, True,  False, False, False, False),
        ("WF+BB%B+BBmom+vol noTP",       "WF",    True,  False, True,  False, False, True,  False),
        ("WF+BB%B+BBmom+mcBig noTP",     "WF",    True,  False, True,  False, False, False, True),
        ("WF+BB%B+BBmom+confTrend noTP", "WF",    True,  False, True,  True,  False, False, False),
        ("H5<0.4+BBwidth+BBmom noTP",     "H5<0.4", False, True,  True,  False, False, False, False),
    ]

    print(f"\n  {'═' * 140}")
    print(f"  4-REGIME FILTER — top 5 strategies: before vs after")
    print(f"  Filtered: high_vol_overbought/oversold_contracting, high_vol_oversold_expanding, low_vol_oversold_expanding")
    print(f"  {'═' * 140}")
    for label, base_key, hbbp, hbbw, hbm, hct, hre, hv, hmc in TOP5:
        base_sig = base_signals[base_key].copy()
        sig0 = apply_filters(base_sig, filter_data, has_bbpct=hbbp,
                             has_bbwidth=hbbw, has_bbmom=hbm,
                             has_conftrend=hct, has_rsiextreme=hre,
                             has_vol=hv, has_mcbig=hmc)
        sig1 = sig0.copy()
        sig1[reg_bad4] = 0
        r0, pb0 = run_backtest_custom(sig0, label, get_tp_sl_no_tp,
                                      pred_ret_ref=pred_ret, verbose=False)
        r1, pb1 = run_backtest_custom(sig1, f"{label}+4regFilter",
                                      get_tp_sl_no_tp, pred_ret_ref=pred_ret,
                                      verbose=False)
        results.append(r1)
        per_bars.append(pb1)

        q0 = compute_quarterly_breakdown(pb0, bars_per_qtr, sorted_qtrs)
        q1 = compute_quarterly_breakdown(pb1, bars_per_qtr, sorted_qtrs)
        n0d = {k: v for k, *v in list(q0)}
        n1d = {k: v for k, *v in list(q1)}

        print(f"\n  ┌─ {label}")
        print(f"  │ {'Quarter':<12} {'AverRet bef':>10} {'AverRet aft':>10} {'Δ AverRet':>10}   "
              f"{'PF bef':>8} {'PF aft':>8} {'Δ PF':>8}   "
              f"{'WR bef':>7} {'WR aft':>7} {'Δ WR':>7}")
        print(f"  ├─{'─' * 98}")
        for qtr in sorted_qtrs + (["overall"] if "overall" in n0d else []):
            if qtr not in n0d:
                continue
            ar0, pf0, wr0, _ = n0d[qtr]
            ar1, pf1, wr1, _ = n1d.get(qtr, (0, 0, 0, 0))
            dar = ar1 - ar0
            dpf = pf1 - pf0
            dwr = wr1 - wr0
            print(f"  │ {qtr:<12} {ar0:>10.6%} {ar1:>10.6%} {dar:>+10.6%}   "
                  f"{pf0:>8.4f} {pf1:>8.4f} {dpf:>+8.4f}   "
                  f"{wr0:>6.2%} {wr1:>6.2%} {dwr:>+6.2%}")
        ov0 = n0d.get("overall", (0, 0, 0, 0))
        ov1 = n1d.get("overall", (0, 0, 0, 0))
        if ov0[0] != 0:
            print(f"  │ {'─' * 98}")
            print(f"  │ SHARPE: {r0['sharpe']:.3f} → {r1['sharpe']:.3f}  ({r1['sharpe'] - r0['sharpe']:+5.3f})  |  "
                  f"TRADES: {int(r0['n_trades'])} → {int(r1['n_trades'])}  ({int(r1['n_trades']) - int(r0['n_trades']):+4d})")
        print(f"  └─")


def top5_15h_16h(labels, per_bars, ts, bars_per_qtr, sorted_qtrs, bar_hour,
                 TOP5, base_signals, data, get_tp_sl_no_tp, pred_ret,
                 bar_dow, run_backtest_custom):
    """AverRet & PF at 15h/16h by day-of-week per quarter for top 5 strategies.

    Parameters
    ----------
    labels : list of str
    per_bars : list of np.ndarray
    ts : np.ndarray — timestamps
    bars_per_qtr : dict str->np.ndarray
    sorted_qtrs : list of str
    bar_hour : np.ndarray — hour per raw bar
    bar_dow : np.ndarray — day-of-week per raw bar
    TOP5 : list of (label, base_key, *filters) tuples
    base_signals : dict str->np.ndarray
    data : dict with 'bb_pct', 'bb_width', etc. for apply_filters
    get_tp_sl_no_tp : callable
    pred_ret : np.ndarray
    run_backtest_custom : callable
    """
    DOW_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri"]
    hr15_16_mask = (bar_hour == 15) | (bar_hour == 16)
    bar_qtr = np.array([f"{str(t)[:4]}-Q{(int(str(t)[5:7]) - 1) // 3 + 1}" for t in ts])

    print(f"\n  {'═' * 140}")
    print(f"  TOP 5 — AverRet & PF at 15h/16h, by day-of-week, per quarter")
    print(f"  {'═' * 140}")
    for label, base_key, hbbp, hbbw, hbm, hct, hre, hv, hmc in TOP5:
        base_sig = base_signals[base_key].copy()
        sig = apply_filters(base_sig, data, has_bbpct=hbbp, has_bbwidth=hbbw,
                            has_bbmom=hbm, has_conftrend=hct,
                            has_rsiextreme=hre, has_vol=hv, has_mcbig=hmc)
        _, pb = run_backtest_custom(sig, label, get_tp_sl_no_tp,
                                    pred_ret_ref=pred_ret, verbose=False)
        active_idx = np.where((pb != 0) & hr15_16_mask)[0]
        if len(active_idx) == 0:
            continue
        bar_vals = pb[active_idx]

        print(f"\n  ┌─ {label}")
        print(f"  │ {'Day':<6} {'Quarter':<10} {'AverRet':>10} {'PF':>8} {'Bars':<6}")
        print(f"  ├─{'─' * 45}")
        for dw in range(5):
            for qtr in sorted_qtrs:
                dw_mask = bar_dow[active_idx] == dw
                qtr_mask = bar_qtr[active_idx] == qtr
                mask = dw_mask & qtr_mask
                n = int(mask.sum())
                if n < 3:
                    continue
                rr = bar_vals[mask]
                ar = float(rr.mean())
                pos = rr[rr > 0].sum()
                neg = abs(rr[rr < 0].sum())
                pf = float(pos / max(neg, 1e-12))
                print(f"  │ {DOW_NAMES[dw]:<6} {qtr:<10} {ar:>9.6%}  {pf:>7.4f}  {n:<5d}")
        print(f"  │ {'─' * 45}")
        for dw in range(5):
            mask = bar_dow[active_idx] == dw
            n = int(mask.sum())
            if n < 3:
                continue
            rr = bar_vals[mask]
            ar = float(rr.mean())
            pos = rr[rr > 0].sum()
            neg = abs(rr[rr < 0].sum())
            pf = float(pos / max(neg, 1e-12))
            print(f"  │ {DOW_NAMES[dw]:<6} {'all':<10} {ar:>9.6%}  {pf:>7.4f}  {n:<5d}")
        print(f"  └─")
