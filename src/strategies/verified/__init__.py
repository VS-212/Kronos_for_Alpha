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

import importlib
import os

from . import s01_wf as s01_wf
from . import s02_bb_pct as s02_bb_pct
from . import s03_bb_mom as s03_bb_mom
from . import s04_bb_rollwr as s04_bb_rollwr

_VERIFIED_DIR = os.path.dirname(os.path.abspath(__file__))

def discover_verified():
    result = []
    for fname in sorted(os.listdir(_VERIFIED_DIR)):
        if fname.startswith("s") and fname.endswith(".py") and fname != "__init__.py":
            mod_name = fname[:-3]
            mod_path = os.path.join(_VERIFIED_DIR, fname)
            result.append((mod_name, mod_path))
    return result

def get(name):
    mod_name = name.replace("-", "_").replace(".py", "")
    for fname in sorted(os.listdir(_VERIFIED_DIR)):
        if fname.startswith("s") and fname.endswith(".py") and fname != "__init__.py":
            if fname[:-3] == mod_name or fname == mod_name:
                spec = importlib.util.spec_from_file_location(mod_name, os.path.join(_VERIFIED_DIR, fname))
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                return mod
    raise ImportError(f"Verified strategy '{name}' not found")
