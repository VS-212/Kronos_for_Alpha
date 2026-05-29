"""Run a single SBER backtest with custom parameters.

Known failures:
  - "KeyError: 'pred_ret'" → signals.npy not aligned with data (same N required)
  - "ValueError: could not broadcast" → signal shape != (N,)
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.data.loader_sber import load_sber_data
from src.evaluation.engine import run_backtest_custom
from src.evaluation.simulation import get_tp_sl, get_tp_sl_no_sl, get_tp_sl_no_tp
from src.strategies.registry import add_strategy


def build_wf_signal(data, q_long: float = 0.90, q_short: float = 0.10):
    pred_ret = data["pred_ret"]
    N = data["N"]
    sig = np.zeros(N, dtype=np.int32)
    sig[pred_ret > data["wf_q90" if q_long >= 0.90 else "wf_q10"]] = 1
    sig[pred_ret < data["wf_q10" if q_short <= 0.10 else "wf_q90"]] = -1
    return sig


TP_SL_FUNCS = {
    "default": get_tp_sl,
    "no_tp": get_tp_sl_no_tp,
    "no_sl": get_tp_sl_no_sl,
}


def main():
    p = argparse.ArgumentParser(description="Run single SBER backtest")
    g = p.add_argument_group("Data")
    g.add_argument("--data", help="Path to .npy or .npz with signal array (shape N,)")
    g.add_argument("--name", default="custom", help="Strategy name for output")

    g = p.add_argument_group("Signal (if no --data)")
    g.add_argument("--strategy", choices=["wf"], help="Built-in signal from data")
    g.add_argument("--wf-q-long", type=float, default=0.90, help="WF quantile for long (default 0.90)")
    g.add_argument("--wf-q-short", type=float, default=0.10, help="WF quantile for short (default 0.10)")

    g = p.add_argument_group("Parameters")
    g.add_argument("--pl", type=int, default=12, help="Profit horizon in bars (default 12)")
    g.add_argument("--lk", type=int, default=500, help="Lookback window size (default 500)")
    g.add_argument("--comm", type=float, default=0.0, help="Commission per trade (default 0.0)")
    g.add_argument("--tp-q", type=float, default=0.80, help="TP quantile (default 0.80)")
    g.add_argument("--sl-q", type=float, default=0.20, help="SL quantile (default 0.20)")
    g.add_argument("--tp-sl", choices=list(TP_SL_FUNCS), default="default", help="TP/SL mode")

    g = p.add_argument_group("Output")
    g.add_argument("--json", action="store_true", help="Output as JSON (machine-readable)")
    g.add_argument("--register", action="store_true", help="Append result to registry.json")

    args = p.parse_args()

    if not args.data and not args.strategy:
        p.error("Provide --data (signals.npy) or --strategy wf")

    data = load_sber_data()
    data["LK"] = args.lk
    data["PL"] = args.pl
    data["COMM"] = args.comm
    data["TP_Q"] = args.tp_q
    data["SL_Q"] = args.sl_q

    if args.data:
        signals = np.load(args.data)
    else:
        signals = build_wf_signal(data, args.wf_q_long, args.wf_q_short)

    tp_sl_fn = TP_SL_FUNCS[args.tp_sl]

    metrics, per_bar = run_backtest_custom(signals, args.name, tp_sl_fn, data, verbose=not args.json)

    if args.json:
        print(json.dumps({k: float(v) if isinstance(v, np.floating) else int(v) if isinstance(v, np.integer) else v
                          for k, v in metrics.items()}, indent=2))
    else:
        print(f"\n{'='*60}")
        print(f"  {args.name}")
        print(f"{'='*60}")
        for k in ["sharpe", "sortino", "max_dd", "profit_factor", "win_rate",
                   "calmar", "avg_return", "total_return", "psr", "dir_acc", "n_trades"]:
            v = metrics.get(k)
            if v is not None:
                pct = k in ("max_dd", "avg_return", "total_return", "win_rate", "dir_acc", "trade_pct")
                print(f"  {k:20s}  {v:+.4f}" if not pct else f"  {k:20s}  {v:+.2%}")
        print(f"{'='*60}")
        print(f"  PL={args.pl}  LK={args.lk}  COMM={args.comm}  TP/SL={args.tp_sl}")
        print(f"{'='*60}\n")

    if args.register:
        entry = {
            "name": args.name,
            "asset": "SBER",
            "metrics": {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                        for k, v in metrics.items()},
            "params": {
                "pl": args.pl, "lk": args.lk, "comm": args.comm,
                "tp_q": args.tp_q, "sl_q": args.sl_q, "tp_sl_mode": args.tp_sl,
            },
        }
        add_strategy(entry)
        print(f"  → registered in registry.json")


if __name__ == "__main__":
    main()
