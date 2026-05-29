"""Compare two SBER backtest strategies side-by-side.

Usage:
  python -m src.cli.compare --ref "WF+BB%B+BBmom+rollWR noTP" --test "my-strategy"
  python -m src.cli.compare --ref signals1.npy --test signals2.npy
  python -m src.cli.compare --ref "WF+BB%B+BBmom+rollWR noTP" --test signals.npy
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
from src.strategies.registry import lookup_strategy

METRICS_LAYOUT = [
    ("sharpe", "f"),
    ("sortino", "f"),
    ("max_dd", "pct"),
    ("profit_factor", "f"),
    ("win_rate", "pct"),
    ("calmar", "f"),
    ("avg_return", "pct"),
    ("total_return", "pct"),
    ("aver_ret", "pct"),
    ("trade_pct", "pct"),
    ("psr", "f"),
    ("dsr", "f"),
    ("dir_acc", "pct"),
    ("ic_rank", "f"),
    ("n_trades", "d"),
    ("n_long", "d"),
    ("n_short", "d"),
]

TP_SL_FUNCS = {
    "default": get_tp_sl,
    "no_tp": get_tp_sl_no_tp,
    "no_sl": get_tp_sl_no_sl,
}


def _resolve_strategy(ref, data_cache):
    """Resolve a strategy argument to (signals, name)."""
    if ref.endswith(".npy"):
        return np.load(ref), Path(ref).stem
    info = lookup_strategy(ref)
    if info:
        return None, ref  # lazy: will run via data
    return None, ref


def _resolve_and_run(ref, args, data_cache):
    signals, name = _resolve_strategy(ref, data_cache)
    if signals is not None:
        data = data_cache["data"]
        tp_sl_fn = data_cache["tp_sl_fn"]
        metrics, _ = run_backtest_custom(signals, name, tp_sl_fn, data, verbose=False)
        return metrics, signals
    info = lookup_strategy(ref)
    if info:
        return info.get("metrics", {}), None
    return None, None


def _format(v, fmt):
    if v is None:
        return "  N/A"
    v = float(v)
    if fmt == "pct":
        return f"{v:>+9.2%}"
    elif fmt == "d":
        return f"{int(v):>9d}"
    else:
        return f"{v:>+9.4f}"


def _delta(v1, v2, fmt):
    if v1 is None or v2 is None:
        return ""
    v1, v2 = float(v1), float(v2)
    if fmt == "d":
        return f"({int(v2 - v1):+d})"
    elif fmt == "pct":
        d = v2 - v1
        if abs(d) < 0.0001:
            return ""
        return f"({d:+.2%})"
    else:
        d = v2 - v1
        if abs(d) < 0.0001:
            return ""
        return f"({d:+.4f})"


def main():
    p = argparse.ArgumentParser(description="Compare two SBER backtest strategies")
    p.add_argument("--ref", required=True, help="Reference: strategy name (registry) or signals.npy")
    p.add_argument("--test", required=True, help="Test: strategy name (registry) or signals.npy")
    p.add_argument("--pl", type=int, default=12)
    p.add_argument("--lk", type=int, default=500)
    p.add_argument("--comm", type=float, default=0.0)
    p.add_argument("--tp-q", type=float, default=0.80)
    p.add_argument("--sl-q", type=float, default=0.20)
    p.add_argument("--tp-sl", choices=list(TP_SL_FUNCS), default="default")
    p.add_argument("--json", action="store_true", help="JSON output")
    args = p.parse_args()

    data = load_sber_data()
    data["LK"] = args.lk
    data["PL"] = args.pl
    data["COMM"] = args.comm
    data["TP_Q"] = args.tp_q
    data["SL_Q"] = args.sl_q
    tp_sl_fn = TP_SL_FUNCS[args.tp_sl]

    data_cache = {"data": data, "tp_sl_fn": tp_sl_fn}

    metrics_ref = _resolve_and_run(args.ref, args, data_cache)[0]
    metrics_test = _resolve_and_run(args.test, args, data_cache)[0]

    if metrics_ref is None and metrics_test is None:
        print("Both strategies not found in registry.", file=sys.stderr)
        sys.exit(1)

    if args.json:
        result = {
            "ref": {"name": args.ref, "metrics": metrics_ref},
            "test": {"name": args.test, "metrics": metrics_test},
        }
        print(json.dumps(result, indent=2))
        return

    # Human-readable table
    name_ref = args.ref if len(args.ref) <= 40 else args.ref[:37] + "..."
    name_test = args.test if len(args.test) <= 40 else args.test[:37] + "..."
    width = max(len(name_ref), len(name_test), 10)
    print()
    print(f"  {'Metric':<20s}  {'REF':>{width}s}  {'TEST':>{width}s}  Δ")
    print(f"  {'─'*20}  {'─'*width}  {'─'*width}  {'─'*10}")
    for key, fmt in METRICS_LAYOUT:
        v1 = metrics_ref.get(key) if metrics_ref else None
        v2 = metrics_test.get(key) if metrics_test else None
        f1 = _format(v1, fmt)
        f2 = _format(v2, fmt)
        d = _delta(v1, v2, fmt)
        print(f"  {key:<20s}  {f1:>{width}s}  {f2:>{width}s}  {d}")
    print()


if __name__ == "__main__":
    main()
