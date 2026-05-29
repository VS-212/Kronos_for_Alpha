"""
M-REGISTRY: Unified strategy registry — lookup, import, export.

Python API:
    list_strategies(asset, min_sharpe, path) -> list[dict]
    lookup_strategy(name_or_id, path) -> dict | None
    import_from_csv(path, output_path) -> int
    export_registry(input_path, output_path) -> None
    discover_verified(path) -> list[tuple[str, str]]

CLI:
    python -m src.strategies.registry --top N [--asset X] [--min-sharpe Y]
    python -m src.strategies.registry --lookup NAME
    python -m src.strategies.registry --import-csv PATH
    python -m src.strategies.registry --export-json IN_PATH OUT_PATH
    python -m src.strategies.registry --stats

Status: ✅ ready
"""

import csv
import json
import os
import sys

REGISTRY_DIR = os.path.dirname(os.path.abspath(__file__))
REGISTRY_PATH = os.path.join(REGISTRY_DIR, "registry.json")
VERIFIED_DIR = os.path.join(REGISTRY_DIR, "verified")

METRICS_KEYS = [
    "sharpe", "sortino", "max_dd", "profit_factor", "win_rate",
    "calmar", "avg_return", "total_return", "aver_ret", "trade_pct",
    "psr", "dsr", "n_trades", "n_long", "n_short", "n_bar_active",
]

STRAT_METRICS_ORDER = [
    "sharpe", "sortino", "max_dd", "profit_factor", "win_rate",
    "calmar", "avg_return", "total_return",
]

TRADE_METRICS_ORDER = [
    "aver_ret", "trade_pct", "psr", "dsr",
    "n_trades", "n_long", "n_short", "n_bar_active",
]


def _load(path=None):
    path = path or REGISTRY_PATH
    if not os.path.exists(path):
        return {"meta": {"format_version": "1.0", "generated": None, "source": None}, "strategies": {}}
    with open(path) as f:
        return json.load(f)


def _save(registry, path=None):
    path = path or REGISTRY_PATH
    with open(path, "w") as f:
        json.dump(registry, f, indent=2)
    return path


def list_strategies(asset=None, min_sharpe=None, max_dd=None, min_wr=None, path=None):
    registry = _load(path)
    results = []
    for name, info in registry.get("strategies", {}).items():
        m = info.get("metrics", {})
        if asset and info.get("asset") != asset:
            continue
        if min_sharpe is not None and m.get("sharpe", -999) < min_sharpe:
            continue
        if max_dd is not None and m.get("max_dd", 999) > max_dd:
            continue
        if min_wr is not None and m.get("win_rate", -1) < min_wr:
            continue
        results.append({"name": name, **info})
    return sorted(results, key=lambda r: r.get("metrics", {}).get("sharpe", 0), reverse=True)


def lookup_strategy(name_or_id, path=None):
    registry = _load(path)
    strategies = registry.get("strategies", {})
    if name_or_id in strategies:
        return {"name": name_or_id, **strategies[name_or_id]}
    for name, info in strategies.items():
        if info.get("id") == name_or_id:
            return {"name": name, **info}
    return None


def import_from_csv(path, output_path=None):
    registry = _load(output_path)
    if "strategies" not in registry:
        registry["strategies"] = {}
    count = 0
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get("strategy", "").strip()
            if not name:
                continue
            metrics = {}
            for key in METRICS_KEYS:
                raw = row.get(key, "").strip()
                try:
                    metrics[key] = float(raw) if raw else None
                except (ValueError, TypeError):
                    metrics[key] = None
            entry = {
                "id": None,
                "asset": None,
                "params": {},
                "data_source": None,
                "code_path": None,
                "status": "known",
                "metrics": metrics,
            }
            if name in registry["strategies"]:
                existing = registry["strategies"][name]
                existing["metrics"] = metrics
                if existing.get("asset"):
                    entry["asset"] = existing["asset"]
                if existing.get("params"):
                    entry["params"] = existing["params"]
            registry["strategies"][name] = entry
            count += 1
    registry["meta"] = {
        "format_version": "1.0",
        "generated": None,
        "source": os.path.abspath(path),
    }
    _save(registry, output_path)
    return count


def export_registry(input_path=None, output_path=None):
    registry = _load(input_path)
    out = output_path or (input_path or REGISTRY_PATH)
    _save(registry, out)
    return out


def discover_verified(path=None):
    base = path or VERIFIED_DIR
    if not os.path.isdir(base):
        return []
    results = []
    for fname in sorted(os.listdir(base)):
        if fname.startswith("s") and fname.endswith(".py") and fname != "__init__.py":
            mod_name = fname[:-3]
            results.append((mod_name, os.path.join(base, fname)))
    return results


def _print_markdown_table(rows):
    headers = ["Strategy", "Sharpe", "Sortino", "MaxDD", "PF", "WR", "Trades"]
    col_w = max(len(r.get("name", "")) for r in rows) + 2
    col_w = max(col_w, 12)
    print(f"  {'Strategy':<{col_w}s} {'Sharpe':>8s} {'Sortino':>8s} {'MaxDD':>8s} {'PF':>8s} {'WR':>8s} {'Trades':>8s}")
    print(f"  {'─' * col_w} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")
    for r in rows:
        m = r.get("metrics", {})
        name = r.get("name", "")
        sharpe = m.get("sharpe")
        sortino = m.get("sortino")
        maxdd = m.get("max_dd")
        pf = m.get("profit_factor")
        wr = m.get("win_rate")
        nt = m.get("n_trades")
        def fmt(v, pct=False):
            if v is None:
                return "  N/A"
            if pct:
                return f"{v:>7.2%}"
            if isinstance(v, float):
                return f"{v:>8.4f}"
            return f"{int(v):>8d}"
        print(f"  {name:<{col_w}s} {fmt(sharpe)} {fmt(sortino)} {fmt(maxdd)} {fmt(pf)} {fmt(wr, pct=True)} {fmt(nt)}")


def _print_stats(registry):
    strategies = registry.get("strategies", {})
    if not strategies:
        print("  No strategies in registry.")
        return
    sharpe_vals = [s["metrics"].get("sharpe") for s in strategies.values() if s["metrics"].get("sharpe") is not None]
    n_with_metrics = len(sharpe_vals)
    n_total = len(strategies)
    statuses = {}
    for s in strategies.values():
        st = s.get("status", "unknown")
        statuses[st] = statuses.get(st, 0) + 1
    print(f"\n  Registry stats:")
    print(f"    Total strategies:    {n_total}")
    print(f"    With metrics:        {n_with_metrics}")
    print(f"    By status:           {statuses}")
    if sharpe_vals:
        print(f"    Sharpe range:        {min(sharpe_vals):.2f} – {max(sharpe_vals):.2f}")
        print(f"    Sharpe mean:         {sum(sharpe_vals)/len(sharpe_vals):.2f}")
        print(f"    Sharpe median:       {sorted(sharpe_vals)[len(sharpe_vals)//2]:.2f}")


def _print_detailed(info):
    name = info.get("name", "?")
    m = info.get("metrics", {})
    print(f"\n  Strategy: {name}")
    print(f"  ├─ Status:  {info.get('status', '?')}")
    print(f"  ├─ Asset:   {info.get('asset', '?')}")
    print(f"  ├─ Params:  {info.get('params', {})}")
    print(f"  ├─ ID:      {info.get('id', '?')}")
    print(f"  └─ Path:    {info.get('code_path', '?')}")
    print(f"\n  Metrics:")
    for key in STRAT_METRICS_ORDER + TRADE_METRICS_ORDER:
        v = m.get(key)
        if v is not None:
            if key in ("win_rate", "avg_return", "total_return", "aver_ret", "trade_pct"):
                print(f"    {key:20s}  {v:>.4%}")
            elif key in ("n_trades", "n_long", "n_short", "n_bar_active"):
                print(f"    {key:20s}  {int(v):>d}")
            else:
                print(f"    {key:20s}  {v:>.4f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Strategy Registry CLI")
    parser.add_argument("--top", type=int, default=None, help="Show top N strategies by Sharpe")
    parser.add_argument("--asset", type=str, default=None, help="Filter by asset")
    parser.add_argument("--min-sharpe", type=float, default=None, help="Minimum Sharpe filter")
    parser.add_argument("--max-dd", type=float, default=None, help="Maximum drawdown filter")
    parser.add_argument("--min-wr", type=float, default=None, help="Minimum win rate filter")
    parser.add_argument("--lookup", type=str, default=None, help="Lookup strategy by name or ID")
    parser.add_argument("--import-csv", type=str, default=None, help="Import metrics from CSV")
    parser.add_argument("--export-json", type=str, default=None, nargs="*", help="Export registry to JSON (IN_PATH [OUT_PATH])")
    parser.add_argument("--discover", action="store_true", help="Scan verified/ directory")
    parser.add_argument("--stats", action="store_true", help="Show registry statistics")
    args = parser.parse_args()

    if args.import_csv:
        count = import_from_csv(args.import_csv)
        print(f"Imported {count} strategies from {args.import_csv}")
        sys.exit(0)

    if args.discover:
        discovered = discover_verified()
        if discovered:
            print(f"Discovered {len(discovered)} verified strategies:")
            for mod_name, fpath in discovered:
                print(f"  {mod_name:20s}  {fpath}")
        else:
            print("No verified strategies found.")
        sys.exit(0)

    if args.export_json is not None:
        paths = args.export_json
        inp = paths[0] if paths else None
        out = paths[1] if len(paths) > 1 else None
        result = export_registry(inp, out)
        print(f"Exported registry to {result}")
        sys.exit(0)

    if args.stats:
        registry = _load()
        _print_stats(registry)
        sys.exit(0)

    if args.lookup:
        info = lookup_strategy(args.lookup)
        if info:
            _print_detailed(info)
        else:
            print(f"Strategy '{args.lookup}' not found.")
            sys.exit(1)
        sys.exit(0)

    results = list_strategies(asset=args.asset, min_sharpe=args.min_sharpe, max_dd=args.max_dd, min_wr=args.min_wr)
    if args.top:
        results = results[:args.top]
    if results:
        _print_markdown_table(results)
        print(f"\n  {len(results)} strategies shown")
    else:
        print("No strategies match the filters.")
