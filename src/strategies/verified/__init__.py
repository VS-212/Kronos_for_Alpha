"""
Verified strategies — validated on SBER_preds_pl12_sc5.npy + feats_test_raw.npy

Champions:
  - WF baseline (Sharpe 6.91)
  - WF+BB%B noTP (Sharpe 19.70)
  - WF+BB%B+BBmom noTP (Sharpe 22.91)
  - WF+BB%B+BBmom+rollWR noTP (Sharpe 23.51)

Usage:
  from src.strategies.verified import s01_wf, s02_bb_pct, s03_bb_mom, s04_bb_rollwr
  metrics, per_bar = s01_wf.run(data)

Status: ready
"""

from . import s01_wf as s01_wf
from . import s02_bb_pct as s02_bb_pct
from . import s03_bb_mom as s03_bb_mom
from . import s04_bb_rollwr as s04_bb_rollwr
