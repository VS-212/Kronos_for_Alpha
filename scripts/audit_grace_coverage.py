"""M-GRACE-AUDIT: Cross-reference src/ modules vs all GRACE artifacts.

Produces manifest.json with per-layer module coverage and issue list.
Usage:
    python scripts/audit_grace_coverage.py [--output path] [--json]

Output (manifest.json):
    {
      "summary": { counts... },
      "layers": { "evaluation": { modules: [...issues...] }, ... },
      "orphan_structure": { zero_edge_nodes: [...], missing_kg_nodes: [...] }
    }
"""

import argparse
import json
import os
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
DOCS_DIR = PROJECT_ROOT / "docs"
AGENTS_MD = PROJECT_ROOT / "AGENTS.md"
CONTRACTS_MD = DOCS_DIR / "module-contracts.md"
KG_XML = DOCS_DIR / "knowledge-graph.xml"
DP_XML = DOCS_DIR / "development-plan.xml"
VM_XML = DOCS_DIR / "verification-plan.xml"

EXCLUDE_INIT = {"__init__.py"}


def _scan_src_modules():
    """Return {filepath: None} for all .py files under src/ excluding __init__.py."""
    modules = {}
    for py_file in SRC_DIR.rglob("*.py"):
        if py_file.name in EXCLUDE_INIT:
            continue
        rel = str(py_file.relative_to(PROJECT_ROOT))
        modules[rel] = None
    return modules


def _parse_agents_table():
    """Parse AGENTS.md table: return {m_id: file_path}."""
    text = AGENTS_MD.read_text()
    entries = {}
    for m in re.finditer(r"^\|\s*(M-\w[\w-]*)\s*\|\s*([^\|]+?)\s*\|[^\|]*\|$", text, re.MULTILINE):
        m_id = m.group(1)
        fpath = m.group(2).strip()
        if fpath and fpath != "—":
            entries[m_id] = fpath
    return entries


def _parse_contracts():
    """Parse module-contracts.md: return set of M-XXX with contracts."""
    text = CONTRACTS_MD.read_text()
    return set(re.findall(r"^### (M-\w[\w-]*):", text, re.MULTILINE))


def _safe_parse_xml(path):
    """Parse XML with fallback to regex extraction on malformed files."""
    if not path.exists():
        return None
    try:
        return ET.parse(path)
    except ET.ParseError:
        # Fall back to regex extraction for malformed XML
        import re as _re
        text = path.read_text(errors="replace")
        root = _re.sub(r"<[^>]+>", lambda m: m.group(0), text)  # keep tags as-is for regex extraction
        return None  # signal: use regex fallback


def _parse_xml_nodes(path):
    """Parse KG or DP XML for <node id=M-XXX file=...>."""
    if not path.exists():
        return {}
    tree = _safe_parse_xml(path)
    if tree is not None:
        nodes = {}
        for el in tree.iter():
            if el.tag.endswith("node"):
                nid = el.get("id", "")
                if nid.startswith("M-"):
                    nodes[nid] = {
                        "file": el.get("file", ""),
                        "name": el.get("name", ""),
                        "category": el.get("category", el.get("status", "")),
                        "status": el.get("status", el.get("phase", "")),
                    }
        return nodes
    # Fallback: regex
    nodes = {}
    text = path.read_text(errors="replace")
    for m in re.finditer(r'id="(M-[^"]+)".*?file="([^"]+)"', text):
        nid = m.group(1)
        if nid not in nodes:
            nodes[nid] = {"file": m.group(2), "name": "", "category": "", "status": ""}
    return nodes


def _parse_xml_module_entries(path):
    """Parse development-plan.xml <module id=M-XXX file=...>."""
    if not path.exists():
        return {}
    tree = _safe_parse_xml(path)
    if tree is not None:
        modules = {}
        for el in tree.iter():
            if el.tag.endswith("module"):
                mid = el.get("id", "")
                if mid.startswith("M-"):
                    modules[mid] = {
                        "file": el.get("file", ""),
                        "name": el.get("name", ""),
                        "status": el.get("status", ""),
                        "contract": el.get("contract", ""),
                        "depends_on": el.get("depends_on", ""),
                    }
        return modules
    # Fallback: regex
    modules = {}
    text = path.read_text(errors="replace")
    for m in re.finditer(r'id="(M-[^"]+)".*?file="([^"]+)"', text):
        mid = m.group(1)
        if mid not in modules:
            modules[mid] = {"file": m.group(2), "name": "", "status": "", "contract": "", "depends_on": ""}
    return modules


def _parse_xml_edges(path):
    """Parse KG XML for edges: return {source_id: set(target_id)}."""
    if not path.exists():
        return defaultdict(set)
    tree = _safe_parse_xml(path)
    if tree is not None:
        edges = defaultdict(set)
        for el in tree.iter():
            if el.tag.endswith("edge"):
                src = el.get("source", "")
                tgt = el.get("target", "")
                if src and tgt:
                    edges[src].add(tgt)
        return edges
    # Fallback: regex
    edges = defaultdict(set)
    text = path.read_text(errors="replace")
    for m in re.finditer(r'source="(M-[^"]+)".*?target="(M-[^"]+)"', text):
        edges[m.group(1)].add(m.group(2))
    return edges


def _parse_vm_verifications(path):
    """Parse verification-plan.xml: return {module_id: [V-M-xxx-id, ...]}."""
    if not path.exists():
        return {}
    tree = _safe_parse_xml(path)
    vm_map = defaultdict(list)
    if tree is not None:
        for el in tree.iter():
            if el.tag.endswith("verification"):
                vid = el.get("id", "")
                mod = el.get("module", "")
                if vid.startswith("V-M-") and mod.startswith("M-"):
                    vm_map[mod].append(vid)
            elif el.tag.endswith("verification_suite"):
                vs_id = el.get("id", "")
                if vs_id and vs_id.startswith("V-M-"):
                    module_id = "M-" + vs_id.split("V-M-", 1)[1]
                    vm_map[module_id].append(vs_id)
        return vm_map
    # Fallback: regex (only for well-formed verification suites)
    text = path.read_text(errors="replace")
    # Match verification suites: <verification_suite id="V-M-XXX">
    for m in re.finditer(r'<verification_suite\s+id="(V-M-[A-Za-z0-9-]+)"', text):
        vid = m.group(1)
        module_id = "M-" + vid.split("V-M-", 1)[1]
        vm_map[module_id].append(vid)
    # Match individual verifications with module attribute
    for m in re.finditer(r'<verification\s+id="(V-M-[^"]+)".*?module="(M-[^"]+)"', text):
        vid, mod = m.group(1), m.group(2)
        vm_map[mod].append(vid)
    return vm_map


def _infer_module_id(filepath):
    """Heuristic: map file path to expected M-XXX ID."""
    mapping = {
        "src/data/fetcher.py": "M-FETCH",
        "src/data/preprocess.py": "M-PREPROCESS",
        "src/data/dataset.py": "M-DATASET",
        "src/data/loader_sber.py": "M-LOAD-SBER",
        "src/data/base.py": "M-DATA-BASE",
        "src/data/cache.py": "M-DATA-CACHE",
        "src/core/kronos/tokenizer.py": "M-KRONOS-TOKENIZER",
        "src/core/kronos/model.py": "M-KRONOS-MODEL",
        "src/core/kronos/modules.py": "M-KRONOS-MODULES",
        "src/core/kronos/predictor.py": "M-KRONOS-PREDICTOR",
        "src/core/registry.py": "M-KRONOS-REGISTRY",
        "src/evaluation/metrics.py": "M-METRICS",
        "src/evaluation/engine.py": "M-ENGINE",
        "src/evaluation/simulation.py": "M-SIM",
        "src/evaluation/quarterly.py": "M-QUARTERLY",
        "src/evaluation/regime.py": "M-REGIME",
        "src/evaluation/calibrate.py": "M-CALIBRATE",
        "src/evaluation/evaluate.py": "M-EVALUATE",
        "src/evaluation/output.py": "M-OUTPUT",
        "src/evaluation/walk_forward.py": "M-WALK-FORWARD",
        "src/evaluation/backtest.py": "M-BACKTEST",
        "src/signals/filters.py": "M-FILTERS",
        "src/signals/atoms.py": "M-SIGNAL-ATOMS",
        "src/signals/bars.py": "M-SIGNAL-BARS",
        "src/signals/bollinger.py": "M-SIGNAL-BOLLINGER",
        "src/signals/divergence.py": "M-SIGNAL-DIV",
        "src/signals/fractal.py": "M-SIGNAL-FRACTAL",
        "src/signals/ict.py": "M-SIGNAL-ICT",
        "src/signals/volatility.py": "M-SIGNAL-VOL",
        "src/signals/vwap.py": "M-SIGNAL-VWAP",
        "src/strategies/verified/s01_wf.py": "M-STRATEGY-WF",
        "src/strategies/verified/s02_bb_pct.py": "M-STRATEGY-BBPCT",
        "src/strategies/verified/s03_bb_mom.py": "M-STRATEGY-BBMOM",
        "src/strategies/verified/s04_bb_rollwr.py": "M-STRATEGY-BBROLLWR",
        "src/strategies/pending/core.py": "M-STRATEGY-CORE",
        "src/strategies/pending/vanilla.py": "M-STRATEGY-VANILLA",
        "src/strategies/pending/s01_bb.py": "M-STRATEGY-S01",
        "src/strategies/pending/s02_bb_mr.py": "M-STRATEGY-S02",
        "src/strategies/pending/s05_bb_breakout.py": "M-STRATEGY-S05",
        "src/strategies/pending/s20_ob.py": "M-STRATEGY-S20",
        "src/strategies/pending/s28_vol_ob.py": "M-STRATEGY-S28",
        "src/strategies/pending/s34_vwap_ob.py": "M-STRATEGY-S34",
        "src/strategies/pending/s38_lowvol_ob.py": "M-STRATEGY-S38",
        "src/strategies/registry.py": "M-REGISTRY",
        "src/cli/backtest.py": "M-CLI-BT",
        "src/cli/compare.py": "M-CLI-CMP",
    }
    return mapping.get(filepath)


def _layer_name(filepath):
    """Extract layer name from file path: 'src/{layer}/...'."""
    parts = Path(filepath).parts
    if len(parts) >= 2 and parts[0] == "src":
        if len(parts) >= 3:
            return f"{parts[1]}/{parts[2]}" if parts[1] in ("core", "strategies") else parts[1]
        return parts[1]
    return "root"


def _priority(module_info):
    """Assign priority to a module's coverage state."""
    missing = module_info.get("missing", [])
    if "kg_node" in missing:
        return "critical"
    if "contract" in missing and "vm" in missing:
        return "high"
    if len(missing) >= 3:
        return "high"
    if len(missing) >= 1:
        return "medium"
    return "low"


def run_audit():
    """Run full coverage audit, return dict."""
    src_modules = _scan_src_modules()
    agents_table = _parse_agents_table()
    contracts = _parse_contracts()
    kg_nodes = _parse_xml_nodes(KG_XML)
    kg_edges = _parse_xml_edges(KG_XML)
    dp_modules = _parse_xml_module_entries(DP_XML)
    vm_coverage = _parse_vm_verifications(VM_XML)

    # Build reverse: filepath → inferred M-ID
    file_to_mid = {}
    for fpath in src_modules:
        mid = _infer_module_id(fpath)
        if mid:
            file_to_mid[fpath] = mid

    # Per-module analysis
    layers = defaultdict(lambda: {"directory": "", "modules": [], "issue_count": 0, "unit_count": 0})
    all_missing_kg = []
    zero_edge_nodes = []
    agents_orphans = []

    for fpath in sorted(src_modules):
        mid = file_to_mid.get(fpath)
        layer = _layer_name(fpath)

        if not layers[layer]["directory"]:
            parts = Path(fpath).parts
            dir_path = os.path.join(*parts[:3]) if len(parts) > 3 else os.path.join(*parts[:2])
            layers[layer]["directory"] = dir_path

        issues = []
        missing = []

        # KG node check
        has_kg = mid in kg_nodes if mid else False
        if not has_kg:
            missing.append("kg_node")
            all_missing_kg.append({"file": fpath, "expected_id": mid})
            issues.append({"type": "no_kg_node", "severity": "critical", "detail": f"expected {mid}"})

        # Contract check
        has_contract = mid in contracts if mid else False
        if not has_contract and mid:
            issues.append({"type": "no_contract", "severity": "high", "detail": f"no M-XXX contract in module-contracts.md"})
            missing.append("contract")

        # VM check
        has_vm = mid in vm_coverage if mid else False
        if not has_vm and mid:
            issues.append({"type": "no_verification", "severity": "high", "detail": "no V-M-XXX entry in verification-plan.xml"})
            missing.append("vm")

        # AGENTS table check
        has_agents = mid in agents_table if mid else False
        if not has_agents and mid:
            issues.append({"type": "no_agents_row", "severity": "medium", "detail": "not in AGENTS.md table"})
            missing.append("agents_row")

        # KG edges check (incoming + outgoing)
        edge_count = len(kg_edges.get(mid, set()))
        has_edges = edge_count > 0
        if has_kg and not has_edges and mid:
            issues.append({"type": "zero_edges", "severity": "high", "detail": "node has 0 edges in KG"})

        module_info = {
            "file": fpath,
            "expected_id": mid,
            "layer": layer,
            "has_kg_node": has_kg,
            "has_contract": has_contract,
            "has_vm": has_vm,
            "has_agents_row": has_agents,
            "kg_edge_count": edge_count,
            "missing": missing,
            "issues": issues,
            "priority": _priority({"missing": missing}),
        }

        layers[layer]["modules"].append(module_info)
        layers[layer]["unit_count"] += 1
        if issues:
            layers[layer]["issue_count"] += len(issues)

    # Orphan analysis
    all_kg_ids = set(kg_nodes.keys())
    all_file_ids = set(file_to_mid.values())
    # Build reverse: M-ID → files
    mid_to_files = defaultdict(list)
    for f, m in file_to_mid.items():
        mid_to_files[m].append(f)
    for mid in all_kg_ids:
        if mid not in all_file_ids:
            # Might be a supporting module (config, docs, ci) that's not in src/
            if mid not in ("M-CONFIG", "M-DOCS", "M-CI", "M-INFRA"):
                agents_orphans.append({"id": mid, "file": kg_nodes[mid]["file"]})
        if len(kg_edges.get(mid, set())) == 0:
            zero_edge_nodes.append({"id": mid, "file": kg_nodes[mid].get("file", ""), "name": kg_nodes[mid].get("name", "")})

    summary = {
        "total_py_files": len(src_modules),
        "total_mapped": len(file_to_mid),
        "with_kg_node": sum(1 for m in file_to_mid.values() if m in kg_nodes),
        "with_contract": sum(1 for m in file_to_mid.values() if m in contracts),
        "with_vm": sum(1 for m in file_to_mid.values() if m in vm_coverage),
        "with_agents_row": sum(1 for m in file_to_mid.values() if m in agents_table),
        "total_kg_nodes": len(kg_nodes),
        "total_kg_edges": sum(len(e) for e in kg_edges.values()),
        "total_vm_entries": sum(len(v) for v in vm_coverage.values()),
        "total_contracts": len(contracts),
        "orphan_nodes": len(agents_orphans),
        "zero_edge_nodes": len(zero_edge_nodes),
        "modules_with_issues": sum(1 for layer in layers.values() for m in layer["modules"] if m["issues"]),
        "critical_issues": sum(1 for layer in layers.values() for m in layer["modules"] for i in m["issues"] if i["severity"] == "critical"),
        "high_issues": sum(1 for layer in layers.values() for m in layer["modules"] for i in m["issues"] if i["severity"] == "high"),
        "layers_total": len(layers),
    }

    return {
        "meta": {
            "tool": "audit_grace_coverage.py",
            "project_root": str(PROJECT_ROOT),
            "version": "1.0",
        },
        "summary": summary,
        "layers": dict(layers),
        "orphan_structure": {
            "zero_edge_nodes": zero_edge_nodes,
            "orphan_kg_nodes_no_file": agents_orphans,
        },
    }


def main():
    p = argparse.ArgumentParser(description="GRACE Coverage Audit")
    p.add_argument("--output", "-o", default=None, help="Output JSON file (default: stdout)")
    p.add_argument("--json", action="store_true", default=True, help="JSON output")
    p.add_argument("--summary", action="store_true", help="Print only summary table (human-readable)")
    args = p.parse_args()

    result = run_audit()

    if args.summary:
        s = result["summary"]
        print(f"\n  GRACE Coverage Audit")
        print(f"  {'─' * 50}")
        print(f"  Python files (src/):         {s['total_py_files']:>5d}")
        print(f"  Mapped to M-XXX:            {s['total_mapped']:>5d}")
        print(f"  With KG node:               {s['with_kg_node']:>5d}  / {s['total_kg_nodes']} total")
        print(f"  With contract:              {s['with_contract']:>5d}  / {s['total_contracts']} total")
        print(f"  With verification (V-M-):   {s['with_vm']:>5d}  / {s['total_vm_entries']} total")
        print(f"  With AGENTS.md row:         {s['with_agents_row']:>5d}")
        print(f"  Modules with issues:        {s['modules_with_issues']:>5d}")
        print(f"  Critical / High issues:     {s['critical_issues']} / {s['high_issues']}")
        print(f"  Orphan KG nodes (no file):  {s['orphan_nodes']:>5d}")
        print(f"  Zero-edge nodes:            {s['zero_edge_nodes']:>5d}")

        for lid, layer in sorted(result["layers"].items()):
            issues = sum(1 for m in layer["modules"] for _ in m["issues"])
            if issues:
                print(f"\n  [{lid}]  {layer['directory']}  ({layer['unit_count']} modules, {issues} issues)")
                for m in layer["modules"]:
                    if m["issues"]:
                        sev = max(i["severity"] for i in m["issues"])
                        flag = "🔴" if sev == "critical" else "🟡" if sev == "high" else "⚪"
                        print(f"    {flag} {m['file']}")
                        for i in m["issues"]:
                            print(f"       └─ {i['type']}: {i['detail']}")
        print()
        return

    json_str = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json_str)
        print(f"Manifest written to {args.output}")
        print(f"  {result['summary']['critical_issues']} critical, {result['summary']['high_issues']} high issues")
    else:
        print(json_str)


if __name__ == "__main__":
    main()
