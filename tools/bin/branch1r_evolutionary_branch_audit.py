#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, TextIO


SCHEMA_VERSION = "branch1r_evolutionary_branch_audit_v1"
TREE_SCHEMA_VERSION = "branch1r_tree_v1"
GATE_SCHEMA_VERSION = "branch1r_three_layer_gate_v1"
NO_CLAIMS = (
    "BRANCH-1R is offline reporting and interpretation only. It does not show intelligence, emergence, "
    "consciousness, broad cognition, memory selection, prediction selection, solved lineage diversity, "
    "general lineage repair, beneficial adaptive mechanism, culture, superintelligence, or general open-ended evolution."
)

DESCRIPTOR_FAMILIES = (
    "genome",
    "behavior",
    "ecology",
    "reproduction",
    "cognition",
)
NUMERIC_DESCRIPTOR_FIELDS = {
    "behavior": ("behavior_score", "mean_drive_jitter", "mean_aggression_bias", "mean_exploration_bias"),
    "ecology": ("ecology_score", "patch_score", "mean_energy", "mean_health", "mean_injury"),
    "reproduction": ("reproduction_score", "mean_offspring_count", "mean_first_reproduction_step"),
    "cognition": ("cognition_score", "mean_memory_dim", "gru_fraction", "mean_prediction_error"),
}


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return int(default)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    return out if math.isfinite(out) else float(default)


def _stable_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _branch_hash(payload: Mapping[str, Any]) -> str:
    digest = hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()
    return digest[:10]


def _entropy_effective(counts: Mapping[str, int]) -> float:
    total = float(sum(max(0, int(v)) for v in counts.values()))
    if total <= 0.0:
        return 0.0
    entropy = 0.0
    for raw in counts.values():
        count = max(0, int(raw))
        if count <= 0:
            continue
        share = float(count) / total
        entropy -= share * math.log(share)
    return math.exp(entropy) if entropy > 0.0 else 0.0


def _top1_share(counts: Mapping[str, int]) -> float:
    total = float(sum(max(0, int(v)) for v in counts.values()))
    if total <= 0.0:
        return 0.0
    return float(max(max(0, int(v)) for v in counts.values())) / total


def _descriptor_signature(row: Mapping[str, Any], family: str) -> str:
    for key in (f"{family}_signature", f"{family}_descriptor", f"{family}_strategy"):
        value = str(row.get(key, "") or "")
        if value:
            return value
    numeric: dict[str, float] = {}
    for key in NUMERIC_DESCRIPTOR_FIELDS.get(family, ()):
        if key in row and row.get(key, "") not in ("", None):
            numeric[key] = round(_safe_float(row.get(key), 0.0), 6)
    return _stable_json(numeric) if numeric else ""


def _descriptor_payload(row: Mapping[str, Any]) -> dict[str, str]:
    payload = {family: _descriptor_signature(row, family) for family in DESCRIPTOR_FAMILIES}
    strategy = str(row.get("strategy_id") or row.get("strategy_signature") or row.get("strategy") or "")
    if not strategy:
        strategy_base = {
            "behavior": payload.get("behavior", ""),
            "ecology": payload.get("ecology", ""),
            "reproduction": payload.get("reproduction", ""),
            "cognition": payload.get("cognition", ""),
        }
        strategy = "strategy_" + _branch_hash(strategy_base)[:8] if any(strategy_base.values()) else ""
    payload["strategy"] = strategy
    return payload


def _descriptor_delta(
    parent: Mapping[str, str],
    child: Mapping[str, str],
    *,
    required_families: int,
) -> tuple[bool, list[str]]:
    changed = [
        family
        for family in DESCRIPTOR_FAMILIES
        if str(parent.get(family, "") or "") and str(child.get(family, "") or "") and parent.get(family) != child.get(family)
    ]
    return len(changed) >= int(required_families), changed


def _event_type(row: Mapping[str, Any]) -> str:
    return str(row.get("event_type") or row.get("event") or "").strip()


def _agent_id(row: Mapping[str, Any], *names: str) -> str:
    for name in names:
        value = str(row.get(name, "") or "")
        if value and value not in {"-1", "0.0"}:
            return str(_safe_int(value, 0)) if value.replace(".", "", 1).lstrip("-").isdigit() else value
    return ""


def _founder_id(row: Mapping[str, Any], *, child: bool = False, parent: bool = False) -> str:
    keys: list[str] = []
    if child:
        keys.extend(("child_founder_ancestry_id", "child_biological_lineage_id", "child_lineage_id"))
    if parent:
        keys.extend(("parent_founder_ancestry_id", "parent_biological_lineage_id", "parent_lineage_id"))
    keys.extend(("founder_ancestry_id", "biological_lineage_id", "lineage_id"))
    for key in keys:
        value = str(row.get(key, "") or "")
        if value:
            return value
    return ""


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_no}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"JSONL row must be an object at {path}:{line_no}")
            rows.append(payload)
    return rows


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _load_event_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".csv":
        return [dict(row) for row in _read_csv(path)]
    return _read_jsonl(path)


def load_branch_observability_rows(root: Path) -> list[dict[str, Any]]:
    """Load BRANCH-2 compact birth edges plus lifecycle descriptor rows."""
    edge_path = Path(root) / "branch_observability_birth_edges.jsonl"
    lifecycle_path = Path(root) / "branch_observability_agent_lifecycle.jsonl"
    rows: list[dict[str, Any]] = []
    if edge_path.exists():
        rows.extend(_read_jsonl(edge_path))
    if lifecycle_path.exists():
        for row in _read_jsonl(lifecycle_path):
            if str(row.get("event_type", "") or ""):
                rows.append(row)
                continue
            agent_id = _agent_id(row, "agent_id")
            if not agent_id:
                continue
            out = dict(row)
            parent_ids = row.get("parent_ids", [])
            try:
                parent_ids_list = list(parent_ids) if not isinstance(parent_ids, str) else json.loads(parent_ids)
            except Exception:
                parent_ids_list = []
            out["event_type"] = "founder" if not parent_ids_list else "lifecycle"
            out["generation"] = _safe_int(row.get("birth_generation", 0), 0)
            out["agent_id"] = agent_id
            out["founder_ancestry_id"] = row.get("founder_ancestry_id", "")
            out["biological_lineage_id"] = row.get("biological_lineage_id", "")
            out["genome_signature"] = row.get("final_genome_signature") or row.get("genome_signature", "")
            rows.append(out)
    return rows


def analyze_event_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    min_persistence_generations: int = 2,
    min_living_descendants: int = 2,
    min_successful_births: int = 1,
    required_descriptor_families: int = 2,
) -> dict[str, Any]:
    """Build a deterministic reporting-only branch tree from explicit event rows."""
    ordered = sorted(
        (dict(row) for row in rows),
        key=lambda row: (
            _safe_int(row.get("generation", 0), 0),
            _safe_int(row.get("tick", 0), 0),
            str(row.get("event_id", "") or ""),
            _agent_id(row, "child_agent_id", "child_id", "agent_id"),
            _agent_id(row, "parent_agent_id", "parent_id"),
        ),
    )
    coverage = {
        "schema_version": SCHEMA_VERSION,
        "event_rows": len(ordered),
        "parent_child_edge_rows": 0,
        "descriptor_rows": 0,
        "descriptor_families_seen": [],
        "missing_parent_child_edges": False,
        "missing_descriptors": False,
    }
    branch_defs: dict[str, dict[str, Any]] = {}
    agent_branch: dict[str, str] = {}
    agent_descriptor: dict[str, dict[str, str]] = {}
    branch_alive: dict[int, Counter[str]] = defaultdict(Counter)
    branch_births_by_generation: dict[int, Counter[str]] = defaultdict(Counter)
    strategy_alive: dict[int, Counter[str]] = defaultdict(Counter)
    branch_children: dict[str, list[str]] = defaultdict(list)

    def ensure_root(founder: str, generation: int, agent_id: str, descriptor: Mapping[str, str]) -> str:
        root_key = founder or agent_id or "unknown_founder"
        branch_id = "br_root_" + _branch_hash({"founder": root_key})[:8]
        if branch_id not in branch_defs:
            branch_defs[branch_id] = {
                "schema_version": TREE_SCHEMA_VERSION,
                "branch_id": branch_id,
                "parent_branch_id": "",
                "founder_ancestry_id": root_key,
                "origin_generation": int(generation),
                "origin_agent_id": agent_id,
                "origin_birth_event_id": "",
                "origin_mutation_event_id": "",
                "defining_descriptor_deltas": [],
                "descriptor": dict(descriptor),
                "first_seen_generation": int(generation),
                "last_seen_generation": int(generation),
                "living_descendant_count_max": 0,
                "successful_birth_count": 0,
                "extinction_generation": "",
                "child_branch_ids": [],
                "strategy_descriptor_summary": {"strategy_id": descriptor.get("strategy", "")},
                "evidence_coverage_status": "root_founder_reporting_only",
                "confidence_classification": "root_branch",
            }
        return branch_id

    for row in ordered:
        generation = _safe_int(row.get("generation", 0), 0)
        event_type = _event_type(row)
        descriptor = _descriptor_payload(row)
        descriptor_seen = any(descriptor.get(family) for family in DESCRIPTOR_FAMILIES)
        if descriptor_seen:
            coverage["descriptor_rows"] += 1
        if event_type in {"founder", "founder_birth", "initial_agent", "alive_snapshot"}:
            aid = _agent_id(row, "agent_id", "child_agent_id", "child_id")
            if not aid:
                continue
            founder = _founder_id(row, child=True) or aid
            bid = ensure_root(founder, generation, aid, descriptor)
            agent_branch.setdefault(aid, bid)
            agent_descriptor[aid] = dict(descriptor)
            branch_alive[generation][bid] += 1
            if descriptor.get("strategy"):
                strategy_alive[generation][descriptor["strategy"]] += 1
            continue
        if event_type in {"birth_success", "successful_birth", "birth"}:
            parent_id = _agent_id(row, "parent_agent_id", "parent_id")
            child_id = _agent_id(row, "child_agent_id", "child_id", "agent_id")
            if not parent_id or not child_id:
                continue
            coverage["parent_child_edge_rows"] += 1
            parent_founder = _founder_id(row, parent=True) or parent_id
            parent_descriptor = agent_descriptor.get(parent_id, _descriptor_payload(row))
            parent_branch = agent_branch.get(parent_id) or ensure_root(parent_founder, generation, parent_id, parent_descriptor)
            branch_births_by_generation[generation][parent_branch] += 1
            branch_defs[parent_branch]["successful_birth_count"] += 1
            split, deltas = _descriptor_delta(
                branch_defs[parent_branch].get("descriptor", {}),
                descriptor,
                required_families=required_descriptor_families,
            )
            if split:
                child_branch = "br_" + _branch_hash(
                    {
                        "parent_branch_id": parent_branch,
                        "founder_ancestry_id": _founder_id(row, child=True) or branch_defs[parent_branch]["founder_ancestry_id"],
                        "origin_generation": generation,
                        "origin_agent_id": child_id,
                        "deltas": deltas,
                    }
                )
                if child_branch not in branch_defs:
                    branch_defs[child_branch] = {
                        "schema_version": TREE_SCHEMA_VERSION,
                        "branch_id": child_branch,
                        "parent_branch_id": parent_branch,
                        "founder_ancestry_id": _founder_id(row, child=True) or branch_defs[parent_branch]["founder_ancestry_id"],
                        "origin_generation": int(generation),
                        "origin_agent_id": child_id,
                        "origin_birth_event_id": str(row.get("event_id", "") or ""),
                        "origin_mutation_event_id": str(row.get("mutation_event_id", "") or ""),
                        "defining_descriptor_deltas": deltas,
                        "descriptor": dict(descriptor),
                        "first_seen_generation": int(generation),
                        "last_seen_generation": int(generation),
                        "living_descendant_count_max": 0,
                        "successful_birth_count": 0,
                        "extinction_generation": "",
                        "child_branch_ids": [],
                        "strategy_descriptor_summary": {"strategy_id": descriptor.get("strategy", "")},
                        "evidence_coverage_status": "explicit_parent_child_and_descriptor",
                        "confidence_classification": "pending_persistence",
                    }
                    branch_children[parent_branch].append(child_branch)
                assigned = child_branch
            else:
                assigned = parent_branch
            agent_branch[child_id] = assigned
            agent_descriptor[child_id] = dict(descriptor)
            branch_alive[generation][assigned] += 1
            if descriptor.get("strategy"):
                strategy_alive[generation][descriptor["strategy"]] += 1

    descriptor_families = sorted(
        family
        for family in DESCRIPTOR_FAMILIES
        if any(str(desc.get(family, "") or "") for desc in agent_descriptor.values())
    )
    coverage["descriptor_families_seen"] = descriptor_families
    coverage["missing_parent_child_edges"] = coverage["parent_child_edge_rows"] == 0
    coverage["missing_descriptors"] = len(descriptor_families) < int(required_descriptor_families)

    for generation, counts in sorted(branch_alive.items()):
        for branch_id, count in counts.items():
            branch_defs[branch_id]["last_seen_generation"] = max(
                int(branch_defs[branch_id]["last_seen_generation"]),
                int(generation),
            )
            branch_defs[branch_id]["living_descendant_count_max"] = max(
                int(branch_defs[branch_id]["living_descendant_count_max"]),
                int(count),
            )
    for parent, children in branch_children.items():
        branch_defs[parent]["child_branch_ids"] = sorted(children)

    final_generation = max(branch_alive.keys(), default=0)
    for branch in branch_defs.values():
        persistence = int(branch["last_seen_generation"]) - int(branch["first_seen_generation"]) + 1
        branch["persistence_generations"] = persistence
        if not branch.get("parent_branch_id"):
            branch["confidence_classification"] = "root_branch"
            continue
        persistent = (
            persistence >= int(min_persistence_generations)
            and (
                int(branch.get("living_descendant_count_max", 0)) >= int(min_living_descendants)
                or int(branch.get("successful_birth_count", 0)) >= int(min_successful_births)
            )
        )
        if persistent:
            branch["confidence_classification"] = "persistent_branch" if branch.get("parent_branch_id") else "root_branch"
        else:
            branch["confidence_classification"] = "transient_descriptor_split"
            branch["extinction_generation"] = (
                int(branch["last_seen_generation"]) + 1 if int(branch["last_seen_generation"]) < final_generation else ""
            )

    generation_rows = []
    branch_event_rows = []
    for generation in sorted(set(branch_alive) | set(branch_births_by_generation) | set(strategy_alive)):
        alive_counts = Counter(
            {
                bid: count
                for bid, count in branch_alive.get(generation, Counter()).items()
                if branch_defs.get(bid, {}).get("confidence_classification") in {"persistent_branch", "root_branch"}
            }
        )
        birth_counts = Counter(
            {
                bid: count
                for bid, count in branch_births_by_generation.get(generation, Counter()).items()
                if branch_defs.get(bid, {}).get("confidence_classification") in {"persistent_branch", "root_branch"}
            }
        )
        generation_rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "generation": int(generation),
                "active_descendant_branches": len([v for v in alive_counts.values() if int(v) > 0]),
                "effective_descendant_branches": round(_entropy_effective(alive_counts), 6),
                "reproducing_branches": len([v for v in birth_counts.values() if int(v) > 0]),
                "top1_branch_share": round(_top1_share(alive_counts), 6),
                "active_strategies": len([v for v in strategy_alive.get(generation, Counter()).values() if int(v) > 0]),
                "effective_strategies": round(_entropy_effective(strategy_alive.get(generation, Counter())), 6),
            }
        )
    for branch in sorted(branch_defs.values(), key=lambda b: (int(b["origin_generation"]), str(b["branch_id"]))):
        if branch.get("parent_branch_id"):
            branch_event_rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "generation": int(branch["origin_generation"]),
                    "event_type": "branch_split",
                    "branch_id": branch["branch_id"],
                    "parent_branch_id": branch["parent_branch_id"],
                    "confidence_classification": branch["confidence_classification"],
                }
            )

    tree = {
        "schema_version": TREE_SCHEMA_VERSION,
        "branch_count": len(branch_defs),
        "persistent_branch_count": sum(
            1 for branch in branch_defs.values() if branch.get("confidence_classification") == "persistent_branch"
        ),
        "root_branch_count": sum(1 for branch in branch_defs.values() if branch.get("confidence_classification") == "root_branch"),
        "branches": sorted(branch_defs.values(), key=lambda b: (int(b["origin_generation"]), str(b["branch_id"]))),
    }
    classifications = _classify_three_layer(tree, generation_rows, coverage)
    return {
        "schema_version": SCHEMA_VERSION,
        "coverage": coverage,
        "tree": tree,
        "branches_by_generation": generation_rows,
        "branch_events": branch_event_rows,
        "strategy_descriptors": _strategy_rows(strategy_alive),
        "three_layer_gate": classifications,
        "no_claims": NO_CLAIMS,
    }


def _strategy_rows(strategy_alive: Mapping[int, Counter[str]]) -> list[dict[str, Any]]:
    rows = []
    for generation, counts in sorted(strategy_alive.items()):
        total = sum(int(v) for v in counts.values())
        for strategy, count in sorted(counts.items()):
            rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "generation": int(generation),
                    "strategy_id": str(strategy),
                    "alive_count": int(count),
                    "alive_share": round(float(count) / float(total), 6) if total else 0.0,
                }
            )
    return rows


def _classify_three_layer(
    tree: Mapping[str, Any],
    generation_rows: list[Mapping[str, Any]],
    coverage: Mapping[str, Any],
) -> dict[str, Any]:
    if coverage.get("missing_parent_child_edges") or coverage.get("missing_descriptors"):
        classification = "insufficient_branch_data"
    elif not generation_rows:
        classification = "insufficient_branch_data"
    else:
        final = generation_rows[-1]
        active_branches = _safe_int(final.get("active_descendant_branches", 0), 0)
        active_strategies = _safe_int(final.get("active_strategies", 0), 0)
        if active_branches <= 1 and active_strategies <= 1:
            classification = "strategy_collapse"
        elif active_branches <= 1:
            classification = "branch_collapse"
        elif active_strategies <= 1:
            classification = "founder_diverse_but_strategy_converged"
        else:
            classification = "healthy_branching_search"
    return {
        "schema_version": GATE_SCHEMA_VERSION,
        "classification": classification,
        "ancestry_layer": "available_if_founder_lineage_rows_present",
        "descendant_branch_layer": "available" if not coverage.get("missing_parent_child_edges") else "missing_parent_child_edges",
        "strategy_layer": "available" if not coverage.get("missing_descriptors") else "missing_descriptor_families",
        "branch_count": int(tree.get("branch_count", 0)),
        "persistent_branch_count": int(tree.get("persistent_branch_count", 0)),
        "final_generation": int(generation_rows[-1]["generation"]) if generation_rows else -1,
    }


def _open_zip_member_text(zf: zipfile.ZipFile, name: str) -> TextIO:
    return zf.open(name, "r")  # type: ignore[return-value]


def _read_last_jsonl_from_zip(zf: zipfile.ZipFile, name: str) -> dict[str, Any] | None:
    last: dict[str, Any] | None = None
    with zf.open(name, "r") as raw:
        for bline in raw:
            text = bline.decode("utf-8").strip()
            if not text:
                continue
            payload = json.loads(text)
            if isinstance(payload, dict):
                last = payload
    return last


def _read_last_csv_rows_from_zip(zf: zipfile.ZipFile, name: str) -> list[dict[str, str]]:
    with zf.open(name, "r") as raw:
        text = raw.read().decode("utf-8")
    rows = list(csv.DictReader(text.splitlines()))
    if not rows:
        return []
    max_gen = max(_safe_int(row.get("generation", 0), 0) for row in rows)
    return [row for row in rows if _safe_int(row.get("generation", 0), 0) == max_gen]


def _compact_strategy_count(descriptor: Mapping[str, Any] | None) -> tuple[int, float]:
    if not isinstance(descriptor, Mapping):
        return 0, 0.0
    lineages = descriptor.get("lineages", [])
    if not isinstance(lineages, list):
        return 0, 0.0
    counts: Counter[str] = Counter()
    for lineage in lineages:
        if not isinstance(lineage, Mapping):
            continue
        payload = _descriptor_payload(lineage)
        strategy = payload.get("strategy", "")
        if strategy:
            counts[strategy] += _safe_int(lineage.get("population_count", 1), 1)
    return len(counts), round(_entropy_effective(counts), 6)


def analyze_archive_zip(path: Path, *, family: str, max_runs: int = 9999) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with zipfile.ZipFile(path) as zf:
        names = sorted(zf.namelist())
        run_dirs = sorted({name.rsplit("/metrics/", 1)[0] for name in names if "/metrics/lineage_summary.csv" in name})
        for run_dir in run_dirs[: int(max_runs)]:
            lineage_summary = f"{run_dir}/metrics/lineage_summary.csv"
            descriptor_path = f"{run_dir}/metrics/lineage_descriptors.jsonl"
            event_path = f"{run_dir}/metrics/lineage_reproduction_funnel_events.jsonl"
            final_rows = _read_last_csv_rows_from_zip(zf, lineage_summary) if lineage_summary in names else []
            final_counts: dict[str, int] = {}
            if final_rows and "dominant_lineage_population_share" in final_rows[-1]:
                summary = final_rows[-1]
                active = _safe_int(summary.get("active_lineages", 0), 0)
                top = _safe_float(summary.get("dominant_lineage_population_share", 0.0), 0.0)
                final_counts = {"top": int(round(top * 1000)), "other": max(0, int(round((1.0 - top) * 1000)))}
                if active <= 1:
                    final_counts = {"top": 1000}
            else:
                final_counts = {
                    str(row.get("lineage_id") or row.get("dominant_lineage_id") or idx): _safe_int(
                        row.get("population_count", row.get("active_lineages", 0)),
                        0,
                    )
                    for idx, row in enumerate(final_rows)
                }
            descriptor = _read_last_jsonl_from_zip(zf, descriptor_path) if descriptor_path in names else None
            active_strategy_count, effective_strategy_count = _compact_strategy_count(descriptor)
            has_parent_child_events = event_path in names
            branch_status = "available" if has_parent_child_events else "insufficient_branch_data"
            classification = "insufficient_branch_data" if not has_parent_child_events else "available_not_expanded_in_archive_index"
            rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "family": family,
                    "source_archive": str(path),
                    "run_dir_member": run_dir,
                    "run_id": run_dir.rsplit("/", 1)[-1],
                    "founder_active_count": len([v for v in final_counts.values() if int(v) > 0]),
                    "founder_effective_count": round(_entropy_effective(final_counts), 6),
                    "founder_top1_share": round(_top1_share(final_counts), 6),
                    "descendant_branch_status": branch_status,
                    "descendant_branch_count": "",
                    "effective_descendant_branches": "",
                    "active_strategy_count": active_strategy_count,
                    "effective_strategy_count": effective_strategy_count,
                    "classification": classification,
                    "coverage_gap": "" if has_parent_child_events else "parent_child_birth_event_rows_absent_or_pruned",
                }
            )
    index = {
        "schema_version": SCHEMA_VERSION,
        "source_archive": str(path),
        "family": family,
        "run_count": len(rows),
        "branch_reconstruction_status": "available" if any(r["descendant_branch_status"] == "available" for r in rows) else "insufficient_branch_data",
        "no_claims": NO_CLAIMS,
    }
    return rows, index


def write_analysis_outputs(result: Mapping[str, Any], output_dir: Path) -> dict[str, str]:
    paths = {
        "tree": output_dir / "branch1r_tree.json",
        "branches_by_generation": output_dir / "branch1r_branches_by_generation.csv",
        "branch_events": output_dir / "branch1r_branch_events.jsonl",
        "strategy_descriptors": output_dir / "branch1r_strategy_descriptors.csv",
        "metric_coverage": output_dir / "branch1r_metric_coverage.json",
        "three_layer_gate": output_dir / "branch1r_three_layer_gate.json",
    }
    _write_json(paths["tree"], result["tree"])
    _write_csv(
        paths["branches_by_generation"],
        result["branches_by_generation"],
        [
            "schema_version",
            "generation",
            "active_descendant_branches",
            "effective_descendant_branches",
            "reproducing_branches",
            "top1_branch_share",
            "active_strategies",
            "effective_strategies",
        ],
    )
    paths["branch_events"].parent.mkdir(parents=True, exist_ok=True)
    paths["branch_events"].write_text(
        "".join(_stable_json(row) + "\n" for row in result["branch_events"]),
        encoding="utf-8",
    )
    _write_csv(
        paths["strategy_descriptors"],
        result["strategy_descriptors"],
        ["schema_version", "generation", "strategy_id", "alive_count", "alive_share"],
    )
    _write_json(paths["metric_coverage"], result["coverage"])
    _write_json(paths["three_layer_gate"], result["three_layer_gate"])
    return {key: str(value) for key, value in paths.items()}


def write_contracts(output_dir: Path) -> dict[str, str]:
    contracts = {
        "branch1r_branch_definition.json": {
            "schema_version": SCHEMA_VERSION,
            "founder_ancestry": "inherited root biological lineage identity",
            "descendant_evolutionary_branch": "persistent descendant group with parent-child continuity and material multi-family descriptor divergence",
            "strategy_identity": "reporting label for persistent behavior/ecology/reproduction/cognition pattern independent of founder ancestry",
            "branch_creation_rule": {
                "required_descriptor_family_deltas": 2,
                "minimum_persistence_generations": 2,
                "minimum_living_descendants_or_births": "living_descendants>=2 or successful_births>=1",
                "determinism": "canonical event ordering and hash-derived IDs; no RNG",
            },
        },
        "branch1r_branch_schema.json": {
            "schema_version": TREE_SCHEMA_VERSION,
            "fields": [
                "branch_id",
                "parent_branch_id",
                "founder_ancestry_id",
                "origin_generation",
                "origin_agent_id",
                "origin_birth_event_id",
                "origin_mutation_event_id",
                "defining_descriptor_deltas",
                "persistence_generations",
                "living_descendant_count_max",
                "successful_birth_count",
                "extinction_generation",
                "child_branch_ids",
                "strategy_descriptor_summary",
                "evidence_coverage_status",
                "confidence_classification",
            ],
        },
        "branch1r_gate_contract.json": {
            "schema_version": GATE_SCHEMA_VERSION,
            "layer_a": "founder ancestry concentration",
            "layer_b": "descendant branch diversity",
            "layer_c": "strategy diversity",
            "classifications": [
                "healthy_branching_search",
                "founder_concentrated_but_branching",
                "founder_diverse_but_strategy_converged",
                "branch_stagnation",
                "branch_collapse",
                "strategy_collapse",
                "reproduction_collapse_before_branching",
                "baseline_mechanics_confound",
                "insufficient_branch_data",
                "integrity_invalid",
            ],
            "not_a_single_score": True,
            "mechanics_effect": "none_reporting_only",
        },
        "branch1r_metric_dictionary.json": {
            "schema_version": SCHEMA_VERSION,
            "founder_metrics": ["active founder ancestries", "effective founder ancestries", "top1 founder share"],
            "branch_metrics": [
                "active descendant branches",
                "effective descendant branches",
                "reproducing branches",
                "branch splits",
                "branch extinctions",
                "top1 branch share",
            ],
            "strategy_metrics": ["active strategies", "effective strategies", "strategy entropy proxy"],
        },
    }
    written: dict[str, str] = {}
    for name, payload in contracts.items():
        path = output_dir / name
        _write_json(path, payload)
        written[name] = str(path)
    return written


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BRANCH-1R offline branch-aware lineage audit.")
    parser.add_argument("--events", type=Path, action="append", default=[], help="JSONL/CSV event file with parent-child rows.")
    parser.add_argument("--archive-zip", type=Path, action="append", default=[], help="Runtime archive zip to index conservatively.")
    parser.add_argument("--archive-family", action="append", default=[], help="Family label for the corresponding --archive-zip.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-runs", type=int, default=9999)
    parser.add_argument("--write-contracts", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Any] = {}
    if args.write_contracts:
        outputs.update(write_contracts(args.output_dir))
    if args.events:
        rows: list[dict[str, Any]] = []
        for path in args.events:
            rows.extend(_load_event_rows(path))
        result = analyze_event_rows(rows)
        outputs.update(write_analysis_outputs(result, args.output_dir))
    if args.archive_zip:
        families = list(args.archive_family or [])
        all_rows: list[dict[str, Any]] = []
        indexes = []
        for idx, archive in enumerate(args.archive_zip):
            family = families[idx] if idx < len(families) else archive.stem
            rows, index = analyze_archive_zip(archive, family=family, max_runs=int(args.max_runs))
            all_rows.extend(rows)
            indexes.append(index)
        _write_csv(
            args.output_dir / "branch1r_existing_evidence_results.csv",
            all_rows,
            [
                "schema_version",
                "family",
                "source_archive",
                "run_dir_member",
                "run_id",
                "founder_active_count",
                "founder_effective_count",
                "founder_top1_share",
                "descendant_branch_status",
                "descendant_branch_count",
                "effective_descendant_branches",
                "active_strategy_count",
                "effective_strategy_count",
                "classification",
                "coverage_gap",
            ],
        )
        _write_json(args.output_dir / "branch1r_existing_evidence_index.json", {"schema_version": SCHEMA_VERSION, "archives": indexes})
        outputs["existing_evidence_results"] = str(args.output_dir / "branch1r_existing_evidence_results.csv")
        outputs["existing_evidence_index"] = str(args.output_dir / "branch1r_existing_evidence_index.json")
    print(json.dumps({"ok": True, "outputs": outputs}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
