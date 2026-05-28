"""
TEMPLATE: Run all strategies on current samples and print metrics.
Source: kronos-artifact/scripts/run_backtest.py
Purpose: Reference example for multi-strategy backtest comparison script
Usage: Adapt to your instrument/asset before production use.
Status: Example — not production code
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.output import load_samples, reconstruct
from src.signals.atoms import consensus
from src.strategies.core import load_config, load_mamba_sber, _simulate_trade, _enrich_trade, report
from src.strategies.vanilla import run as run_vanilla
from src.strategies.s01_bb_consensus import run as run_s1
from src.strategies.s02_bb_mean_rev import run as run_s2
from src.strategies.s05_bb_narrow import run as run_s5
from src.strategies.s20_order_block import run as run_s20
from src.strategies.s28_vol_ob import run as run_s28
from src.strategies.s34_vwap_ob import run as run_s34
from src.strategies.s38_low_vol_ob import run as run_s38

PRED_LEN = 6
SAMPLE_COUNT = 5
SAMP_PATH = "/tmp/opencode/SBER_samples_pl6_sc5.parquet"

strategies = [
    ("Vanilla ct=0.8",      lambda d, m, pl: run_vanilla(d, m, pl, SAMPLE_COUNT, consensus_threshold=0.8)),
    ("Vanilla ct=1.0",      lambda d, m, pl: run_vanilla(d, m, pl, SAMPLE_COUNT, consensus_threshold=1.0)),
    ("S1 BB+Cons ct=0.8",  lambda d, m, pl: run_s1(d, m, pl, SAMPLE_COUNT, consensus_threshold=0.8)),
    ("S2 BB MR   ct=1.0",  lambda d, m, pl: run_s2(d, m, pl, SAMPLE_COUNT, consensus_threshold=1.0)),
    ("S5 BB Narr ct=0.8",  lambda d, m, pl: run_s5(d, m, pl, SAMPLE_COUNT, consensus_threshold=0.8)),
    ("S20 OB     ct=1.0",  lambda d, m, pl: run_s20(d, m, pl, SAMPLE_COUNT, consensus_threshold=1.0)),
    ("S28 VolOB  ct=1.0",  lambda d, m, pl: run_s28(d, m, pl, SAMPLE_COUNT, consensus_threshold=1.0)),
    ("S34 VWAP+OB ct=1.0", lambda d, m, pl: run_s34(d, m, pl, SAMPLE_COUNT, consensus_threshold=1.0)),
    ("S38 LoVolOB ct=1.0", lambda d, m, pl: run_s38(d, m, pl, SAMPLE_COUNT, consensus_threshold=1.0)),
]

def main():
    cfg = load_config()
    mamba = load_mamba_sber(cfg)
    df = load_samples(SAMP_PATH)
    ts = pd.to_datetime(df["pred_ts"])

    train = df[(ts >= "2025-01-01") & (ts < "2026-01-01")].copy()
    test  = df[ts >= "2026-01-01"].copy()

    print(f"Train: {len(train)} | Test: {len(test)}")
    print()

    # Full metrics table
    hdr = f"{'Strategy':<22} {'Tr n':>5} {'Tr Sh':>8} {'Tr PF':>7} {'Te n':>5} {'Te Sh':>8} {'Te PF':>7} {'Te WR':>7} {'Te Ret%':>8}"
    print(hdr)
    print("-" * 90)
    for name, func in strategies:
        tr = func(train, mamba, PRED_LEN)
        te = func(test, mamba, PRED_LEN)
        tr_r = report(tr) if tr is not None and len(tr) > 0 else {}
        te_r = report(te) if te is not None and len(te) > 0 else {}
        print(f"{name:<22} {tr_r.get('n',0):>5d} {tr_r.get('sharpe',0):>8.2f} {tr_r.get('profit_factor',0):>7.2f} "
              f"{te_r.get('n',0):>5d} {te_r.get('sharpe',0):>8.2f} {te_r.get('profit_factor',0):>7.2f} "
              f"{te_r.get('win_rate',0):>6.1%} {te_r.get('total_return_pct',0):>+7.2f}%")

if __name__ == "__main__":
    main()
