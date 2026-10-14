#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping


SCHEMA_VERSION = "branch4_niche_strategy_audit_v1"
STRATEGY_SCHEMA_VERSION = "branch4_niche_strategy_signature_v1"
NO_CLAIMS = (
    "BRANCH-4 is a bounded reporting and method-readiness package for branch-linked niche exposure "
    "and deterministic strategy compression. It does not show intelligence, emergence, consciousness, "
    "broad cognition, memory selection, prediction selection, solved lineage diversity, general lineage "
    "repair, beneficial adaptive mechanism, culture, superintelligence, or general open-ended evolution."
)

RUN_ID_PART_RE = re.compile(r"__(?P<key>context|init|init_arm|setting)_(?P<value>.*?)(?=__|$)")
SEED_RE = re.compile(r"__seed(?P<seed>\d+)")
PROFILE_THRESHOLDS: dict[str, dict[str, Any]] = {
    "conservative": {
        "min_observed_ticks": 16,
        "min_agents": 8,
        "min_persistent_strategies": 2,
        "min_reproducing_strategies": 2,
        "min_branch_strategy_links": 2,
        "branch1r": {
            "min_persistence_generations": 3,
            "min_living_descendants": 3,
            "min_successful_births": 2,
            "required_descriptor_families": 3,
        },
    },
    "nominal": {
        "min_observed_ticks": 8,
        "min_agents": 6,
        "min_persistent_strategies": 2,
        "min_reproducing_strategies": 1,
        "min_branch_strategy_links": 2,
        "branch1r": {
            "min_persistence_generations": 2,
            "min_living_descendants": 2,
            "min_successful_births": 1,
            "required_descriptor_families": 2,
        },
    },
    "permissive": {
        "min_observed_ticks": 1,
        "min_agents": 2,
        "min_persistent_strategies": 1,
        "min_reproducing_strategies": 1,
        "min_branch_strategy_links": 1,
        "branch1r": {
            "min_persistence_generations": 1,
            "min_living_descendants": 1,
            "min_successful_births": 1,
            "required_descriptor_families": 1,
        },
    },
}


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_branch1r() -> Any:
    path = _project_root() / "tools/bin/branch1r_evolutionary_branch_audit.py"
    spec = importlib.util.spec_from_file_location("branch1r_evolutionary_branch_audit", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _stable_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _stable_hash(payload: Any, n: int = 12) -> str:
    return hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()[: int(n)]


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value in ("", None):
            return float(default)
        out = float(value)
        return out if math.isfinite(out) else float(default)
    except Exception:
        return float(default)


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return int(default)


def _sum_values(value: Any) -> float:
    if isinstance(value, Mapping):
        return sum(_num(v, 0.0) for v in value.values())
    return 0.0


def _counter_total(value: Any) -> int:
    if isinstance(value, Mapping):
        return sum(_int(v, 0) for v in value.values())
    return 0


def _dominant(counter: Mapping[str, Any] | None) -> tuple[str, float, int]:
    counts = {str(k): _int(v, 0) for k, v in dict(counter or {}).items() if _int(v, 0) > 0}
    total = sum(counts.values())
    if total <= 0:
        return "", 0.0, 0
    key, count = sorted(counts.items(), key=lambda item: (-int(item[1]), str(item[0])))[0]
    return key, round(float(count) / float(total), 6), len(counts)


def _bin(value: float, cuts: Iterable[float], prefix: str) -> str:
    idx = 0
    for cut in cuts:
        if float(value) > float(cut):
            idx += 1
    return f"{prefix}_{idx}"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
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
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    try:
        csv.field_size_limit(sys.maxsize)
    except Exception:
        pass
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _sha256_file(path: Path) -> str:
    if not path.exists():
        return ""
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _effective_count(values: Iterable[Any]) -> float:
    counts = Counter(str(v) for v in values if str(v or ""))
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    entropy = 0.0
    for count in counts.values():
        p = float(count) / float(total)
        entropy -= p * math.log(p)
    return round(float(math.exp(entropy)), 6)


def _parse_run_id(run_id: str) -> dict[str, str]:
    out = {"setting": "", "context": "", "init_arm": "", "seed": ""}
    for match in RUN_ID_PART_RE.finditer(str(run_id)):
        key = "init_arm" if match.group("key") == "init" else match.group("key")
        out[key] = match.group("value")
    seed = SEED_RE.search(str(run_id))
    if seed:
        out["seed"] = str(int(seed.group("seed")))
    return out


def _run_label(run_dir: Path) -> str:
    meta = _run_metadata(run_dir)
    if str(meta.get("run_id", "")):
        return str(meta["run_id"])
    summary = _read_json(run_dir / "branch_observability" / "branch_observability_run_summary.json")
    return str(summary.get("run_id") or run_dir.name)


def _run_metadata(run_dir: Path) -> dict[str, Any]:
    run_dir = Path(run_dir)
    for parent in list(run_dir.parents)[:6]:
        summary_path = parent / "summary.json"
        summary = _read_json(summary_path)
        for row in summary.get("runs", []) or []:
            if not isinstance(row, Mapping):
                continue
            row_run_dir = str(row.get("run_dir", ""))
            runtime_run_id = str(row.get("runtime_run_id", ""))
            if row_run_dir == str(run_dir) or runtime_run_id == run_dir.name:
                axes = dict(row.get("axes", {}) or {})
                return {
                    "run_id": str(row.get("run_id", "") or runtime_run_id or run_dir.name),
                    "runtime_run_id": runtime_run_id,
                    "context": str(axes.get("context", "")),
                    "init_arm": str(axes.get("init_arm", "")),
                    "seed": str(row.get("seed", "")),
                    "run_tag": str(row.get("run_tag", "")),
                }
    return {}


def discover_run_dirs(roots: Iterable[Path]) -> list[Path]:
    found: dict[str, Path] = {}
    for root in roots:
        root = Path(root)
        if (root / "branch_observability" / "branch_observability_niche_lifecycle.jsonl").exists():
            found[str(root.resolve())] = root
            continue
        for path in root.rglob("branch_observability_niche_lifecycle.jsonl"):
            if path.parent.name != "branch_observability":
                continue
            run_dir = path.parent.parent
            found[str(run_dir.resolve())] = run_dir
    return [found[key] for key in sorted(found)]


def niche_signature(row: Mapping[str, Any], *, profile: str = "nominal") -> dict[str, Any]:
    thresholds = PROFILE_THRESHOLDS[profile]
    ticks = max(0, _int(row.get("total_observed_ticks"), 0))
    dominant_zone, dominant_share, zone_count = _dominant(row.get("zone_exposure_counts") if isinstance(row.get("zone_exposure_counts"), Mapping) else {})
    shelter_fraction = _num(row.get("shelter_cover_exposure_ticks"), 0.0) / max(1.0, float(ticks))
    food_rate = _sum_values(row.get("food_collected_by_zone")) / max(1.0, float(ticks))
    damage_rate = _sum_values(row.get("damage_by_zone")) / max(1.0, float(ticks))
    birth_rate = _num(row.get("successful_birth_rate"), 0.0)
    movement_rate = _num(row.get("movement_action_rate"), 0.0)
    gather_rate = _num(row.get("gather_action_rate"), 0.0)
    transition_rate = _num(row.get("transition_rate"), 0.0)
    missing = tuple(sorted(str(x) for x in row.get("missing_data_coverage", []) or []))
    if ticks < int(thresholds["min_observed_ticks"]) or not dominant_zone:
        payload: tuple[str, ...] = (
            STRATEGY_SCHEMA_VERSION,
            profile,
            "insufficient_data",
            _bin(float(ticks), (0, 4, 8, 16), "observed_ticks"),
            "missing_" + _stable_hash(missing, 8),
        )
        status = "insufficient_data"
    else:
        payload = (
            STRATEGY_SCHEMA_VERSION,
            profile,
            "dominant_zone=" + dominant_zone,
            _bin(dominant_share, (0.34, 0.50, 0.75, 0.90), "dominant_share"),
            _bin(float(zone_count), (1, 2, 4, 8), "zone_count"),
            _bin(movement_rate, (0.05, 0.20, 0.45, 0.70), "move_rate"),
            _bin(gather_rate, (0.01, 0.05, 0.12, 0.25), "gather_rate"),
            _bin(transition_rate, (0.01, 0.05, 0.15, 0.35), "transition_rate"),
            _bin(shelter_fraction, (0.05, 0.20, 0.50, 0.80), "shelter_cover_frac"),
            _bin(food_rate, (0.001, 0.01, 0.05, 0.20), "food_gain_rate"),
            _bin(damage_rate, (0.001, 0.01, 0.05, 0.20), "damage_rate"),
        )
        status = "available"
    strategy_id = "eco_strategy_" + _stable_hash(payload, 12)
    return {
        "schema_version": STRATEGY_SCHEMA_VERSION,
        "profile": profile,
        "strategy_id": strategy_id,
        "strategy_status": status,
        "signature_payload": "|".join(payload),
        "total_observed_ticks": ticks,
        "dominant_zone_class": dominant_zone,
        "dominant_zone_share": dominant_share,
        "zone_count": zone_count,
        "movement_action_rate": round(movement_rate, 6),
        "gather_action_rate": round(gather_rate, 6),
        "transition_rate": round(transition_rate, 6),
        "shelter_cover_fraction": round(shelter_fraction, 6),
        "food_collected_per_tick": round(food_rate, 6),
        "damage_per_tick": round(damage_rate, 6),
        "successful_birth_rate": round(birth_rate, 6),
    }


def _merged_agent_rows(run_dir: Path) -> list[dict[str, Any]]:
    bo_root = run_dir / "branch_observability"
    lifecycle = {_int(row.get("agent_id"), -1): dict(row) for row in _read_jsonl(bo_root / "branch_observability_agent_lifecycle.jsonl")}
    niche = {_int(row.get("agent_id"), -1): dict(row) for row in _read_jsonl(bo_root / "branch_observability_niche_lifecycle.jsonl")}
    run_id = _run_label(run_dir)
    axes = _parse_run_id(run_id)
    axes.update({key: value for key, value in _run_metadata(run_dir).items() if key in {"context", "init_arm", "seed"} and str(value)})
    rows = []
    for agent_id in sorted(set(lifecycle) | set(niche)):
        if agent_id < 0:
            continue
        row = dict(lifecycle.get(agent_id, {}))
        row.update(niche.get(agent_id, {}))
        row["run_dir"] = str(run_dir)
        row["run_id"] = run_id
        row.update(axes)
        rows.append(row)
    return rows


def _branch1r_rows(run_dir: Path, strategy_by_agent: Mapping[int, str]) -> list[dict[str, Any]]:
    bo_root = run_dir / "branch_observability"
    lifecycle = _read_jsonl(bo_root / "branch_observability_agent_lifecycle.jsonl")
    edges = _read_jsonl(bo_root / "branch_observability_birth_edges.jsonl")
    by_agent = {_int(row.get("agent_id"), -1): dict(row) for row in lifecycle}
    rows: list[dict[str, Any]] = []
    for aid, row in sorted(by_agent.items()):
        if aid < 0:
            continue
        if list(row.get("parent_ids", []) or []):
            continue
        strategy = str(strategy_by_agent.get(aid, ""))
        rows.append(
            {
                "event_type": "founder",
                "generation": _int(row.get("birth_generation"), 0),
                "tick": _int(row.get("birth_tick"), 0),
                "agent_id": aid,
                "founder_ancestry_id": row.get("founder_ancestry_id", ""),
                "biological_lineage_id": row.get("biological_lineage_id", ""),
                "behavior_signature": "behavior_" + strategy,
                "ecology_signature": "ecology_" + strategy,
                "reproduction_signature": "reproduction_" + _bin(_num(row.get("successful_births"), 0.0), (0, 1, 2), "births"),
                "cognition_signature": "cognition_reporting_neutral",
                "strategy_id": strategy,
            }
        )
    for edge in edges:
        child_id = _int(edge.get("child_agent_id"), -1)
        parent_id = _int(edge.get("parent_agent_id", edge.get("primary_parent_agent_id")), -1)
        child = by_agent.get(child_id, {})
        strategy = str(strategy_by_agent.get(child_id, ""))
        rows.append(
            {
                "event_type": "birth_success",
                "generation": _int(edge.get("generation"), 0),
                "tick": _int(edge.get("tick"), 0),
                "event_id": edge.get("birth_event_id", ""),
                "parent_agent_id": parent_id,
                "child_agent_id": child_id,
                "parent_biological_lineage_id": edge.get("parent_biological_lineage_id", ""),
                "child_biological_lineage_id": edge.get("child_biological_lineage_id", ""),
                "child_founder_ancestry_id": edge.get("child_founder_ancestry_id", child.get("founder_ancestry_id", "")),
                "behavior_signature": "behavior_" + strategy,
                "ecology_signature": "ecology_" + strategy,
                "reproduction_signature": "reproduction_" + _bin(_num(child.get("successful_births"), 0.0), (0, 1, 2), "births"),
                "cognition_signature": "cognition_reporting_neutral",
                "strategy_id": strategy,
            }
        )
    return rows


def analyze_runs(run_dirs: Iterable[Path], *, profile: str = "nominal") -> dict[str, Any]:
    branch1r = _load_branch1r()
    raw_rows: list[dict[str, Any]] = []
    compressed_rows: list[dict[str, Any]] = []
    branch_map_rows: list[dict[str, Any]] = []
    generation_rows: list[dict[str, Any]] = []
    matrix_rows: list[dict[str, Any]] = []
    sensitivity_source_rows: list[dict[str, Any]] = []
    branch_results: dict[str, Any] = {}
    for run_dir in sorted(Path(p) for p in run_dirs):
        run_id = _run_label(run_dir)
        axes = _parse_run_id(run_id)
        axes.update({key: value for key, value in _run_metadata(run_dir).items() if key in {"context", "init_arm", "seed"} and str(value)})
        merged = _merged_agent_rows(run_dir)
        strategy_by_agent: dict[int, str] = {}
        for row in merged:
            sig = niche_signature(row, profile=profile)
            aid = _int(row.get("agent_id"), -1)
            if aid >= 0:
                strategy_by_agent[aid] = str(sig["strategy_id"])
            sensitivity_source = dict(row)
            sensitivity_source["run_id"] = run_id
            sensitivity_source.update(axes)
            sensitivity_source_rows.append(sensitivity_source)
            raw_rows.append(
                {
                    "run_id": run_id,
                    "run_dir": str(run_dir),
                    "agent_id": aid,
                    "founder_ancestry_id": row.get("founder_ancestry_id", ""),
                    "biological_lineage_id": row.get("biological_lineage_id", ""),
                    "birth_generation": row.get("birth_generation", ""),
                    "final_generation": row.get("final_generation", ""),
                    "raw_strategy_id": row.get("strategy_id", ""),
                    **sig,
                    **axes,
                }
            )
        counts = Counter(strategy_by_agent.values())
        births_by_strategy: Counter[str] = Counter()
        for row in merged:
            sid = strategy_by_agent.get(_int(row.get("agent_id"), -1), "")
            if sid:
                births_by_strategy[sid] += _counter_total(row.get("successful_births_by_zone"))
        for strategy_id, count in sorted(counts.items()):
            compressed_rows.append(
                {
                    "run_id": run_id,
                    "strategy_id": strategy_id,
                    "agent_count": int(count),
                    "agent_share": round(float(count) / max(1.0, float(len(strategy_by_agent))), 6),
                    "successful_births_attributed": int(births_by_strategy.get(strategy_id, 0)),
                    "reproducing": int(births_by_strategy.get(strategy_id, 0) > 0),
                    **axes,
                }
            )
        b_rows = _branch1r_rows(run_dir, strategy_by_agent)
        b_result = branch1r.analyze_event_rows(b_rows, **PROFILE_THRESHOLDS[profile]["branch1r"])
        branch_results[run_id] = b_result
        branch_lookup: dict[str, str] = {}
        for branch in b_result.get("tree", {}).get("branches", []) or []:
            branch_lookup[str(branch.get("strategy_descriptor_summary", {}).get("strategy_id", ""))] = str(branch.get("branch_id", ""))
        for strategy_id in sorted(counts):
            branch_map_rows.append(
                {
                    "run_id": run_id,
                    "strategy_id": strategy_id,
                    "branch_id": branch_lookup.get(strategy_id, ""),
                    "agent_count": int(counts[strategy_id]),
                    "successful_births_attributed": int(births_by_strategy.get(strategy_id, 0)),
                    "branch_reconstruction_status": b_result.get("three_layer_gate", {}).get("descendant_branch_layer", ""),
                    "three_layer_classification": b_result.get("three_layer_gate", {}).get("classification", ""),
                    **axes,
                }
            )
        for row in b_result.get("strategy_descriptors", []) or []:
            generation_rows.append({"run_id": run_id, **axes, **dict(row)})
        coverage = b_result.get("coverage", {})
        matrix_rows.append(
            {
                "run_id": run_id,
                "run_dir": str(run_dir),
                "agent_rows": len(merged),
                "compressed_strategy_count": len(counts),
                "effective_strategy_count": _effective_count(strategy_by_agent.values()),
                "reproducing_strategy_count": len([k for k, v in births_by_strategy.items() if int(v) > 0]),
                "birth_edge_rows": int(coverage.get("parent_child_edge_rows", 0) or 0),
                "missing_parent_child_edges": bool(coverage.get("missing_parent_child_edges", False)),
                "missing_descriptors": bool(coverage.get("missing_descriptors", False)),
                "classification": b_result.get("three_layer_gate", {}).get("classification", ""),
                "niche_sha256": _sha256_file(run_dir / "branch_observability" / "branch_observability_niche_lifecycle.jsonl"),
                **axes,
            }
        )
    coverage = _coverage_summary(raw_rows, compressed_rows, branch_map_rows, matrix_rows, profile=profile)
    sensitivity = compression_sensitivity(sensitivity_source_rows)
    gate = three_layer_gate(coverage, matrix_rows, profile=profile)
    return {
        "raw_individual_signatures": raw_rows,
        "compressed_strategies": compressed_rows,
        "persistent_strategies": [row for row in compressed_rows if int(row.get("agent_count", 0)) >= int(PROFILE_THRESHOLDS[profile]["min_agents"])],
        "branch_strategy_map": branch_map_rows,
        "strategy_by_generation": generation_rows,
        "observability_matrix": matrix_rows,
        "strategy_compression_sensitivity": sensitivity,
        "strategy_coverage": coverage,
        "three_layer_gate": gate,
        "branch_results": branch_results,
    }


def _coverage_summary(
    raw_rows: list[Mapping[str, Any]],
    compressed_rows: list[Mapping[str, Any]],
    branch_map_rows: list[Mapping[str, Any]],
    matrix_rows: list[Mapping[str, Any]],
    *,
    profile: str,
) -> dict[str, Any]:
    run_count = len({str(row.get("run_id", "")) for row in raw_rows})
    raw_ids = {str(row.get("raw_strategy_id", "")) for row in raw_rows if str(row.get("raw_strategy_id", ""))}
    compressed_ids = {str(row.get("strategy_id", "")) for row in raw_rows if str(row.get("strategy_id", ""))}
    available = [row for row in raw_rows if row.get("strategy_status") == "available"]
    return {
        "schema_version": SCHEMA_VERSION,
        "profile": profile,
        "run_count": run_count,
        "agent_rows": len(raw_rows),
        "available_agent_rows": len(available),
        "available_agent_row_share": round(float(len(available)) / max(1.0, float(len(raw_rows))), 6),
        "raw_strategy_ids": len(raw_ids),
        "compressed_strategy_ids": len(compressed_ids),
        "compression_ratio": round(float(len(compressed_ids)) / max(1.0, float(len(raw_ids))), 6),
        "matrix_contexts": sorted({str(row.get("context", "")) for row in matrix_rows if str(row.get("context", ""))}),
        "matrix_init_arms": sorted({str(row.get("init_arm", "")) for row in matrix_rows if str(row.get("init_arm", ""))}),
        "matrix_seeds": sorted({str(row.get("seed", "")) for row in matrix_rows if str(row.get("seed", ""))}),
        "runs_with_parent_child_edges": sum(1 for row in matrix_rows if not bool(row.get("missing_parent_child_edges", True))),
        "runs_with_descriptor_coverage": sum(1 for row in matrix_rows if not bool(row.get("missing_descriptors", True))),
        "runs_with_two_or_more_strategies": sum(1 for row in matrix_rows if _int(row.get("compressed_strategy_count"), 0) >= 2),
        "runs_with_reproducing_strategy": sum(1 for row in matrix_rows if _int(row.get("reproducing_strategy_count"), 0) >= 1),
        "branch_strategy_links": len([row for row in branch_map_rows if str(row.get("branch_id", ""))]),
        "no_claims": NO_CLAIMS,
    }


def compression_sensitivity(raw_rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    by_run: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in raw_rows:
        by_run[str(row.get("run_id", ""))].append(row)
    for profile in ("conservative", "nominal", "permissive"):
        strategy_ids = []
        for run_rows in by_run.values():
            for row in run_rows:
                strategy_ids.append(niche_signature(row, profile=profile)["strategy_id"])
        rows.append(
            {
                "profile": profile,
                "agent_rows": len(raw_rows),
                "compressed_strategy_ids": len(set(strategy_ids)),
                "effective_strategy_ids": _effective_count(strategy_ids),
                "row_order_independent": True,
                "uses_python_hash": False,
                "excludes_genome_capacity_lifetime_health_raw_offspring": True,
            }
        )
    return rows


def three_layer_gate(coverage: Mapping[str, Any], matrix_rows: list[Mapping[str, Any]], *, profile: str) -> dict[str, Any]:
    thresholds = PROFILE_THRESHOLDS[profile]
    blockers = []
    if int(coverage.get("run_count", 0)) <= 0:
        blockers.append("no_run_dirs_with_niche_lifecycle_rows")
    if float(coverage.get("available_agent_row_share", 0.0)) < 0.80:
        blockers.append("insufficient_niche_observation_coverage")
    if int(coverage.get("compressed_strategy_ids", 0)) >= max(1, int(coverage.get("agent_rows", 0)) * 0.50):
        blockers.append("compressed_strategy_ids_still_near_individual_labels")
    if int(coverage.get("runs_with_parent_child_edges", 0)) < max(1, int(coverage.get("run_count", 0)) // 2):
        blockers.append("parent_child_edge_coverage_too_sparse")
    if int(coverage.get("runs_with_two_or_more_strategies", 0)) <= 0:
        blockers.append("no_run_separates_multiple_ecological_strategies")
    if int(coverage.get("runs_with_reproducing_strategy", 0)) <= 0:
        blockers.append("no_strategy_has_reproduction_attribution")
    if int(coverage.get("branch_strategy_links", 0)) < int(thresholds["min_branch_strategy_links"]):
        blockers.append("branch_to_strategy_mapping_incomplete")
    classification = "reporting_strategy_method_ready" if not blockers else "reporting_strategy_not_ready"
    return {
        "schema_version": SCHEMA_VERSION,
        "profile": profile,
        "classification": classification,
        "method_ready": not blockers,
        "blockers": blockers,
        "run_count": coverage.get("run_count", 0),
        "agent_rows": coverage.get("agent_rows", 0),
        "compressed_strategy_ids": coverage.get("compressed_strategy_ids", 0),
        "coverage": dict(coverage),
        "matrix_classifications": sorted(Counter(str(row.get("classification", "")) for row in matrix_rows).items()),
        "no_claims": NO_CLAIMS,
    }


def write_outputs(result: Mapping[str, Any], output_dir: Path) -> dict[str, str]:
    paths = {
        "raw_individual_signatures": output_dir / "branch4_raw_individual_signatures.csv",
        "compressed_strategies": output_dir / "branch4_compressed_strategies.csv",
        "persistent_strategies": output_dir / "branch4_persistent_strategies.csv",
        "branch_strategy_map": output_dir / "branch4_branch_strategy_map.csv",
        "strategy_by_generation": output_dir / "branch4_strategy_by_generation.csv",
        "strategy_coverage": output_dir / "branch4_strategy_coverage.json",
        "three_layer_gate": output_dir / "branch4_three_layer_gate.json",
        "observability_matrix": output_dir / "branch4_observability_matrix.csv",
        "strategy_compression_sensitivity": output_dir / "branch4_strategy_compression_sensitivity.csv",
        "strategy_readiness_decision": output_dir / "branch4_strategy_readiness_decision.json",
    }
    _write_csv(
        paths["raw_individual_signatures"],
        result["raw_individual_signatures"],
        [
            "run_id",
            "context",
            "init_arm",
            "seed",
            "agent_id",
            "founder_ancestry_id",
            "biological_lineage_id",
            "birth_generation",
            "final_generation",
            "raw_strategy_id",
            "strategy_id",
            "strategy_status",
            "dominant_zone_class",
            "dominant_zone_share",
            "zone_count",
            "movement_action_rate",
            "gather_action_rate",
            "transition_rate",
            "shelter_cover_fraction",
            "food_collected_per_tick",
            "damage_per_tick",
            "successful_birth_rate",
            "signature_payload",
            "run_dir",
        ],
    )
    _write_csv(paths["compressed_strategies"], result["compressed_strategies"], ["run_id", "context", "init_arm", "seed", "strategy_id", "agent_count", "agent_share", "successful_births_attributed", "reproducing"])
    _write_csv(paths["persistent_strategies"], result["persistent_strategies"], ["run_id", "context", "init_arm", "seed", "strategy_id", "agent_count", "agent_share", "successful_births_attributed", "reproducing"])
    _write_csv(paths["branch_strategy_map"], result["branch_strategy_map"], ["run_id", "context", "init_arm", "seed", "strategy_id", "branch_id", "agent_count", "successful_births_attributed", "branch_reconstruction_status", "three_layer_classification"])
    _write_csv(paths["strategy_by_generation"], result["strategy_by_generation"], ["run_id", "context", "init_arm", "seed", "schema_version", "generation", "strategy_id", "alive_count", "alive_share"])
    _write_json(paths["strategy_coverage"], result["strategy_coverage"])
    _write_json(paths["three_layer_gate"], result["three_layer_gate"])
    _write_csv(paths["observability_matrix"], result["observability_matrix"], ["run_id", "context", "init_arm", "seed", "agent_rows", "compressed_strategy_count", "effective_strategy_count", "reproducing_strategy_count", "birth_edge_rows", "missing_parent_child_edges", "missing_descriptors", "classification", "niche_sha256", "run_dir"])
    _write_csv(paths["strategy_compression_sensitivity"], result["strategy_compression_sensitivity"], ["profile", "agent_rows", "compressed_strategy_ids", "effective_strategy_ids", "row_order_independent", "uses_python_hash", "excludes_genome_capacity_lifetime_health_raw_offspring"])
    _write_json(paths["strategy_readiness_decision"], result["three_layer_gate"])
    return {key: str(value) for key, value in paths.items()}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit BRANCH-4 branch-linked niche strategies from compact run outputs.")
    parser.add_argument("--runs-root", action="append", default=[], help="Run root or campaign root to scan. May be repeated.")
    parser.add_argument("--run-dir", action="append", default=[], help="Explicit single run directory. May be repeated.")
    parser.add_argument("--output-dir", required=True, help="Evidence output directory.")
    parser.add_argument("--profile", choices=sorted(PROFILE_THRESHOLDS), default="nominal")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    roots = [Path(p) for p in args.runs_root] + [Path(p) for p in args.run_dir]
    run_dirs = discover_run_dirs(roots)
    result = analyze_runs(run_dirs, profile=str(args.profile))
    paths = write_outputs(result, Path(args.output_dir))
    print(json.dumps({"schema_version": SCHEMA_VERSION, "run_dirs": len(run_dirs), "paths": paths, "decision": result["three_layer_gate"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
