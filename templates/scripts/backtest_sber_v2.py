"""
Single-asset backtest: SBER predictions with TP/SL from MC distribution.
Strangler Fig — delegates to modular orchestrator (run_sber.py).
"""
import subprocess, sys, os
script = os.path.join(os.path.dirname(__file__), "run_sber.py")
sys.exit(subprocess.call([sys.executable, script]))
