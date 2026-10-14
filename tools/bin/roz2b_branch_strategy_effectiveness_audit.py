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


SCHEMA_VERSION = "roz2b_branch_strategy_effectiveness_audit_v1"
NO_CLAIMS = (
    "ROZ-2B is a bounded branch-aware effectiveness decision for a default-off static Patch World "
    "niche substrate. It does not show intelligence, emergence, consciousness, broad cognition, "
    "adaptive memory, prediction selection, solved lineage diversity, or general open-ended evolution."
)
RUN_RE = re.compile(
    r"__arm_(?P<arm>.*?)__context_(?P<context>.*?)__init_(?P<init_arm>.*?)__seed(?P<seed>\d+)"
)
KEY_METRICS = (
    "alive",
    "deaths",
    "successful_births",
    "failed_births",
    "birth_attempts",
    "food_collected",
    "damage",
    "lineage_active_count",
    "lineage_top1_alive_share",
    "lineage_effective_alive_lineages",
    "final_effective_alive_lineages",
    "final_top1_alive_share",
    "collapse_generation_observed",
)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_branch4() -> Any:
    path = _project_root() / "tools/bin/branch4_niche_strategy_audit.py"
    spec = importlib.util.spec_from_file_location("branch4_niche_strategy_audit", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


branch4 = _load_branch4()


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


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    try:
        csv.field_size_limit(sys.maxsize)
    except Exception:
        pass
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


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


def _stable_hash_path(path: Path) -> str:
    if not path.exists():
        return ""
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _parse_run_id(run_id: str) -> dict[str, str]:
    match = RUN_RE.search(str(run_id))
    if not match:
        return {"arm": "", "context": "", "init_arm": "", "seed": ""}
    return {
        "arm": match.group("arm"),
        "context": match.group("context"),
        "init_arm": match.group("init_arm"),
        "seed": str(int(match.group("seed"))),
    }


def _run_label(run_dir: Path) -> str:
    return str(branch4._run_label(Path(run_dir)))


def _run_dirs(paths: Iterable[str]) -> list[Path]:
    return branch4.discover_run_dirs([Path(p) for p in paths])


def _final_metrics(run_dir: Path) -> dict[str, Any]:
    metrics = _read_csv(run_dir / "metrics.csv")
    if not metrics:
        return {"metrics_row_count": 0}
    final = metrics[-1]
    out: dict[str, Any] = {"metrics_row_count": len(metrics)}
    for key in KEY_METRICS:
        out[key] = final.get(key, "")
    out["population_time_area"] = round(sum(_num(row.get("alive"), 0.0) for row in metrics), 6)
    out["births_area"] = round(sum(_num(row.get("successful_births"), 0.0) for row in metrics), 6)
    out["death_area"] = round(sum(_num(row.get("deaths"), 0.0) for row in metrics), 6)
    out["final_generation"] = _int(final.get("generation"), len(metrics) - 1)
    return out


def _load_path2_summary(run_dir: Path) -> dict[str, Any]:
    candidates = [
        run_dir / "metrics" / "lineage_reproduction_funnel_run_summary.json",
        run_dir / "lineage_reproduction_funnel_run_summary.json",
    ]
    for path in candidates:
        if path.exists():
            return _read_json(path)
    return {}


def _cell_key(row: Mapping[str, Any], *, include_arm: bool = False) -> tuple[str, str, str] | tuple[str, str, str, str]:
    if include_arm:
        return (str(row.get("arm", "")), str(row.get("context", "")), str(row.get("init_arm", "")), str(row.get("seed", "")))
    return (str(row.get("context", "")), str(row.get("init_arm", "")), str(row.get("seed", "")))


def _median(values: Iterable[float]) -> float:
    vals = sorted(float(v) for v in values if math.isfinite(float(v)))
    if not vals:
        return 0.0
    mid = len(vals) // 2
    if len(vals) % 2:
        return round(vals[mid], 6)
    return round((vals[mid - 1] + vals[mid]) / 2.0, 6)


def _generation_areas(rows: list[Mapping[str, Any]]) -> dict[str, dict[str, float]]:
    grouped: dict[str, dict[int, list[Mapping[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        run_id = str(row.get("run_id", ""))
        if not run_id:
            continue
        grouped[run_id][_int(row.get("generation"), 0)].append(row)
    out: dict[str, dict[str, float]] = {}
    for run_id, by_gen in grouped.items():
        strategy_area = 0.0
        reproducing_area = 0.0
        top_share_area = 0.0
        for gen_rows in by_gen.values():
            live = [_num(row.get("alive_count"), 0.0) for row in gen_rows]
            strategy_area += float(len([x for x in live if x > 0]))
            reproducing_area += float(len([x for x in live if x >= 2]))
            total = sum(live)
            top_share_area += max(live) / total if total > 0 else 0.0
        out[run_id] = {
            "strategy_generation_area": round(strategy_area, 6),
            "reproducing_strategy_generation_area": round(reproducing_area, 6),
            "top_strategy_share_area": round(top_share_area, 6),
            "generation_rows": float(len(by_gen)),
        }
    return out


def _compact_generation_rows(rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        run_id = str(row.get("run_id", ""))
        if not run_id:
            continue
        grouped[(run_id, _int(row.get("generation"), 0))].append(row)
    out: list[dict[str, Any]] = []
    for (run_id, generation), gen_rows in sorted(grouped.items()):
        parsed = _parse_run_id(run_id)
        alive = [_num(row.get("alive_count"), 0.0) for row in gen_rows]
        total_alive = sum(alive)
        out.append(
            {
                "run_id": run_id,
                **parsed,
                "generation": generation,
                "strategy_count": len([value for value in alive if value > 0.0]),
                "reproducing_strategy_proxy_count": len([value for value in alive if value >= 2.0]),
                "alive_count_sum": round(total_alive, 6),
                "top_strategy_alive_share": round(max(alive) / total_alive, 6) if total_alive > 0.0 else 0.0,
            }
        )
    return out


def _dist(rows: list[Mapping[str, Any]], strategy_id: str) -> dict[str, float]:
    counts: Counter[str] = Counter()
    for row in rows:
        if str(row.get("strategy_id", "")) == strategy_id:
            zone = str(row.get("dominant_zone_class", ""))
            if zone:
                counts[zone] += max(1, _int(row.get("total_observed_ticks"), 1))
    total = sum(counts.values())
    if total <= 0:
        return {}
    return {key: float(value) / float(total) for key, value in sorted(counts.items())}


def _jsd(a: Mapping[str, float], b: Mapping[str, float]) -> float:
    keys = set(a) | set(b)
    if not keys:
        return 0.0

    def kl(p: Mapping[str, float], q: Mapping[str, float]) -> float:
        out = 0.0
        for key in keys:
            pv = float(p.get(key, 0.0))
            qv = float(q.get(key, 0.0))
            if pv > 0.0 and qv > 0.0:
                out += pv * math.log(pv / qv, 2)
        return out

    mid = {key: (float(a.get(key, 0.0)) + float(b.get(key, 0.0))) / 2.0 for key in keys}
    return round((kl(a, mid) + kl(b, mid)) / 2.0, 6)


def _specialization(raw_rows: list[Mapping[str, Any]], compressed_rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    raw_by_run: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in raw_rows:
        raw_by_run[str(row.get("run_id", ""))].append(row)
    compressed_by_run: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in compressed_rows:
        compressed_by_run[str(row.get("run_id", ""))].append(row)
    out: list[dict[str, Any]] = []
    for run_id, strategies in sorted(compressed_by_run.items()):
        parsed = _parse_run_id(run_id)
        persistent = [row for row in strategies if _int(row.get("agent_count"), 0) >= 2]
        reproducing = [row for row in persistent if _int(row.get("successful_births_attributed"), 0) > 0]
        dists = [_dist(raw_by_run.get(run_id, []), str(row.get("strategy_id", ""))) for row in persistent]
        pairs = []
        for idx, left in enumerate(dists):
            for right in dists[idx + 1 :]:
                pairs.append(_jsd(left, right))
        dominant_zones = {
            str(max(dist.items(), key=lambda item: item[1])[0])
            for dist in dists
            if dist
        }
        out.append(
            {
                "run_id": run_id,
                **parsed,
                "persistent_strategy_count": len(persistent),
                "reproducing_persistent_strategy_count": len(reproducing),
                "distinct_dominant_zones": len(dominant_zones),
                "mean_pairwise_zone_jsd": _median(pairs) if pairs else 0.0,
                "formula": "median pairwise Jensen-Shannon divergence across persistent strategy dominant-zone distributions",
            }
        )
    return out


def _build_run_rows(run_dirs: list[Path], nominal: Mapping[str, Any], conservative: Mapping[str, Any], permissive: Mapping[str, Any]) -> list[dict[str, Any]]:
    matrix_by_run = {str(row.get("run_id", "")): row for row in nominal.get("observability_matrix", [])}
    areas = _generation_areas(list(nominal.get("strategy_by_generation", []) or []))
    cons_matrix = {str(row.get("run_id", "")): row for row in conservative.get("observability_matrix", [])}
    perm_matrix = {str(row.get("run_id", "")): row for row in permissive.get("observability_matrix", [])}
    rows: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        run_id = _run_label(run_dir)
        parsed = _parse_run_id(run_id)
        final = _final_metrics(run_dir)
        matrix = matrix_by_run.get(run_id, {})
        area = areas.get(run_id, {})
        births = max(1.0, _num(final.get("births_area"), 0.0))
        pop_area = max(1.0, _num(final.get("population_time_area"), 0.0))
        rows.append(
            {
                "run_id": run_id,
                **parsed,
                "run_dir": str(run_dir),
                "status": "completed" if (run_dir / "metrics.csv").exists() else "missing_metrics",
                "classification": matrix.get("classification", ""),
                "classification_conservative": cons_matrix.get(run_id, {}).get("classification", ""),
                "classification_permissive": perm_matrix.get(run_id, {}).get("classification", ""),
                "agent_rows": matrix.get("agent_rows", ""),
                "compressed_strategy_count": matrix.get("compressed_strategy_count", ""),
                "effective_strategy_count": matrix.get("effective_strategy_count", ""),
                "reproducing_strategy_count": matrix.get("reproducing_strategy_count", ""),
                "niche_sha256": matrix.get("niche_sha256", ""),
                "metrics_sha256": _stable_hash_path(run_dir / "metrics.csv"),
                "branch_niche_sha256": _stable_hash_path(run_dir / "branch_observability" / "branch_observability_niche_lifecycle.jsonl"),
                "strategy_generation_area": area.get("strategy_generation_area", 0.0),
                "reproducing_strategy_generation_area": area.get("reproducing_strategy_generation_area", 0.0),
                "strategy_area_per_100_births": round(100.0 * _num(area.get("strategy_generation_area"), 0.0) / births, 6),
                "strategy_area_per_1k_agent_gen": round(1000.0 * _num(area.get("strategy_generation_area"), 0.0) / pop_area, 6),
                "births_per_agent_generation": round(_num(final.get("births_area"), 0.0) / pop_area, 6),
                **final,
            }
        )
    return rows


def _paired(active: list[Mapping[str, Any]], control: list[Mapping[str, Any]], metric: str) -> list[dict[str, Any]]:
    control_by_key = {_cell_key(row): row for row in control}
    out = []
    for row in active:
        match = control_by_key.get(_cell_key(row))
        if not match:
            continue
        a = _num(row.get(metric), 0.0)
        c = _num(match.get(metric), 0.0)
        out.append(
            {
                "context": row.get("context", ""),
                "init_arm": row.get("init_arm", ""),
                "seed": row.get("seed", ""),
                "metric": metric,
                "active": round(a, 6),
                "control": round(c, 6),
                "delta": round(a - c, 6),
                "relative_delta": round((a - c) / max(abs(c), 1.0), 6),
            }
        )
    return out


def _compare_exact(active: list[Mapping[str, Any]], control: list[Mapping[str, Any]], metrics: Iterable[str]) -> dict[str, Any]:
    mismatches = []
    control_by_key = {_cell_key(row): row for row in control}
    for row in active:
        match = control_by_key.get(_cell_key(row))
        if not match:
            mismatches.append({"key": list(_cell_key(row)), "metric": "_match", "active": row.get("run_id", ""), "control": ""})
            continue
        for metric in metrics:
            if str(row.get(metric, "")) != str(match.get(metric, "")):
                mismatches.append(
                    {
                        "key": list(_cell_key(row)),
                        "metric": metric,
                        "active": row.get(metric, ""),
                        "control": match.get(metric, ""),
                    }
                )
    return {"checked_pairs": len(active), "mismatch_count": len(mismatches), "mismatches": mismatches[:100]}


def _paired_summary(rows: list[Mapping[str, Any]], *, metric: str) -> dict[str, Any]:
    deltas = [_num(row.get("delta"), 0.0) for row in rows if str(row.get("metric", "")) == metric]
    rels = [_num(row.get("relative_delta"), 0.0) for row in rows if str(row.get("metric", "")) == metric]
    return {
        "metric": metric,
        "pairs": len(deltas),
        "median_delta": _median(deltas),
        "median_relative_delta": _median(rels),
        "improved_count": sum(1 for value in deltas if value > 0.0),
        "worse_count": sum(1 for value in deltas if value < 0.0),
    }


def _decision(run_rows: list[dict[str, Any]], specialization_rows: list[dict[str, Any]], repeat_parity: Mapping[str, Any]) -> dict[str, Any]:
    true_flat_active = [r for r in run_rows if r["arm"] == "active_roz2a" and r["context"] == "true_flat_noop"]
    true_flat_reporting = [r for r in run_rows if r["arm"] == "reporting_only" and r["context"] == "true_flat_noop"]
    flat_compare = _compare_exact(true_flat_active, true_flat_reporting, KEY_METRICS)
    niche_active = [r for r in run_rows if r["arm"] == "active_roz2a" and r["context"] == "roz2a_niche_context"]
    niche_reporting = [r for r in run_rows if r["arm"] == "reporting_only" and r["context"] == "roz2a_niche_context"]
    pair_rows = []
    for metric in ("strategy_area_per_100_births", "strategy_area_per_1k_agent_gen", "reproducing_strategy_generation_area", "births_per_agent_generation"):
        pair_rows.extend(_paired(niche_active, niche_reporting, metric))
    branch_summary = _paired_summary(pair_rows, metric="strategy_area_per_100_births")
    strategy_summary = _paired_summary(pair_rows, metric="strategy_area_per_1k_agent_gen")
    birth_summary = _paired_summary(pair_rows, metric="births_per_agent_generation")
    active_specs = {
        _cell_key(row): row
        for row in specialization_rows
        if row.get("arm") == "active_roz2a" and row.get("context") == "roz2a_niche_context"
    }
    reporting_specs = {
        _cell_key(row): row
        for row in specialization_rows
        if row.get("arm") == "reporting_only" and row.get("context") == "roz2a_niche_context"
    }
    spec_deltas = []
    for key, row in active_specs.items():
        ctrl = reporting_specs.get(key)
        if ctrl:
            spec_deltas.append(_num(row.get("mean_pairwise_zone_jsd"), 0.0) - _num(ctrl.get("mean_pairwise_zone_jsd"), 0.0))
    flat_pass = flat_compare["mismatch_count"] == 0 and len(true_flat_active) == 10
    repeat_pass = bool(repeat_parity.get("passed", False)) if repeat_parity else False
    viability_pass = abs(float(birth_summary["median_relative_delta"])) <= 0.15
    branch_pass = float(branch_summary["median_relative_delta"]) >= 0.15
    strategy_pass = float(strategy_summary["median_relative_delta"]) >= 0.10 and _median(spec_deltas) > 0.0
    blockers = []
    if not flat_pass:
        blockers.append("true_flat_active_reporting_parity_failed")
    if repeat_parity and not repeat_pass:
        blockers.append("repeat_parity_failed")
    if not viability_pass:
        blockers.append("population_or_birth_confound")
    preliminary = "CLEAN-NEUTRAL"
    if blockers:
        preliminary = "INTEGRITY-INVALID" if "repeat_parity_failed" in blockers or not flat_pass else "HARMFUL-OR-CONFOUNDED"
    elif branch_pass and strategy_pass:
        preliminary = "EFFECTIVE-DIRECTIONAL"
    final = {
        "EFFECTIVE-DIRECTIONAL": "ROZ-2B-A",
        "CLEAN-NEUTRAL": "ROZ-2B-B",
        "HARMFUL-OR-CONFOUNDED": "ROZ-2B-C",
        "INTEGRITY-INVALID": "ROZ-2B-E",
    }[preliminary]
    return {
        "schema_version": SCHEMA_VERSION,
        "control_disposition": "CONTROL-C",
        "flat_neutrality": flat_compare,
        "repeat_parity": dict(repeat_parity),
        "branch_direction": branch_summary,
        "strategy_direction": strategy_summary,
        "birth_confound": birth_summary,
        "median_specialization_delta": _median(spec_deltas),
        "integrity_blockers": blockers,
        "preliminary_effectiveness_classification": preliminary,
        "final_decision": final,
        "horizon_extension": "HORIZON-EXTENSION-NOT-RUN" if preliminary != "EFFECTIVE-DIRECTIONAL" else "HORIZON-EXTENSION-ELIGIBLE",
        "no_claims": NO_CLAIMS,
    }


def _repeat_parity(main_rows: list[dict[str, Any]], repeat_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not repeat_rows:
        return {"schema_version": SCHEMA_VERSION, "passed": False, "reason": "no_repeat_rows"}
    main_by_run = {row["run_id"]: row for row in main_rows}
    mismatches = []
    for row in repeat_rows:
        main = main_by_run.get(row["run_id"])
        if not main:
            mismatches.append({"run_id": row["run_id"], "metric": "_match", "main": "", "repeat": row["run_dir"]})
            continue
        for metric in ("metrics_sha256", "branch_niche_sha256", "classification", "compressed_strategy_count", "effective_strategy_count"):
            if str(main.get(metric, "")) != str(row.get(metric, "")):
                mismatches.append({"run_id": row["run_id"], "metric": metric, "main": main.get(metric, ""), "repeat": row.get(metric, "")})
    return {
        "schema_version": SCHEMA_VERSION,
        "checked_repeats": len(repeat_rows),
        "mismatch_count": len(mismatches),
        "passed": len(mismatches) == 0,
        "mismatches": mismatches[:100],
    }


def analyze(main_roots: list[str], repeat_roots: list[str], output_dir: Path) -> dict[str, Any]:
    main_dirs = _run_dirs(main_roots)
    repeat_dirs = _run_dirs(repeat_roots) if repeat_roots else []
    nominal = branch4.analyze_runs(main_dirs, profile="nominal")
    conservative = branch4.analyze_runs(main_dirs, profile="conservative")
    permissive = branch4.analyze_runs(main_dirs, profile="permissive")
    repeat_nominal = branch4.analyze_runs(repeat_dirs, profile="nominal") if repeat_dirs else {
        "observability_matrix": [],
        "strategy_by_generation": [],
    }
    repeat_conservative = branch4.analyze_runs(repeat_dirs, profile="conservative") if repeat_dirs else {
        "observability_matrix": [],
        "strategy_by_generation": [],
    }
    repeat_permissive = branch4.analyze_runs(repeat_dirs, profile="permissive") if repeat_dirs else {
        "observability_matrix": [],
        "strategy_by_generation": [],
    }
    main_rows = _build_run_rows(main_dirs, nominal, conservative, permissive)
    repeat_rows = _build_run_rows(repeat_dirs, repeat_nominal, repeat_conservative, repeat_permissive) if repeat_dirs else []
    specialization = _specialization(nominal["raw_individual_signatures"], nominal["compressed_strategies"])
    repeat = _repeat_parity(main_rows, repeat_rows)
    decision = _decision(main_rows, specialization, repeat)
    pair_rows = []
    for metric in ("strategy_area_per_100_births", "strategy_area_per_1k_agent_gen", "reproducing_strategy_generation_area", "births_per_agent_generation"):
        active = [r for r in main_rows if r["arm"] == "active_roz2a"]
        reporting = [r for r in main_rows if r["arm"] == "reporting_only"]
        pair_rows.extend(_paired(active, reporting, metric))
    threshold_rows = [
        {
            "run_id": row["run_id"],
            "arm": row["arm"],
            "context": row["context"],
            "init_arm": row["init_arm"],
            "seed": row["seed"],
            "classification_conservative": row["classification_conservative"],
            "classification_nominal": row["classification"],
            "classification_permissive": row["classification_permissive"],
            "compressed_strategy_count": row["compressed_strategy_count"],
            "effective_strategy_count": row["effective_strategy_count"],
        }
        for row in main_rows
    ]
    zone_rows = [
        {
            "run_id": row["run_id"],
            "arm": row["arm"],
            "context": row["context"],
            "init_arm": row["init_arm"],
            "seed": row["seed"],
            "alive": row.get("alive", ""),
            "successful_births": row.get("successful_births", ""),
            "deaths": row.get("deaths", ""),
            "population_time_area": row.get("population_time_area", ""),
            "births_area": row.get("births_area", ""),
        }
        for row in main_rows
    ]
    outputs = {
        "campaign_matrix": output_dir / "roz2b_campaign_matrix.csv",
        "branch_persistence_summary": output_dir / "roz2b_branch_persistence_summary.csv",
        "branch_metrics_by_generation": output_dir / "roz2b_branch_metrics_by_generation.csv",
        "strategy_persistence_summary": output_dir / "roz2b_strategy_persistence_summary.csv",
        "strategy_metrics_by_generation": output_dir / "roz2b_strategy_metrics_by_generation.csv",
        "branch_strategy_specialization": output_dir / "roz2b_branch_strategy_specialization.csv",
        "zone_demand_metrics": output_dir / "roz2b_zone_demand_metrics.csv",
        "population_confounds": output_dir / "roz2b_population_confounds.csv",
        "threshold_sensitivity": output_dir / "roz2b_threshold_sensitivity.csv",
        "repeat_parity": output_dir / "roz2b_repeat_parity.json",
        "checkpoint_parity": output_dir / "roz2b_checkpoint_parity.json",
        "primary_decision": output_dir / "roz2b_primary_effectiveness_decision.json",
        "horizon_extension": output_dir / "roz2b_horizon_extension_decision.json",
        "final_decision": output_dir / "roz2b_final_decision.json",
    }
    base_fields = [
        "run_id",
        "arm",
        "context",
        "init_arm",
        "seed",
        "status",
        "classification",
        "classification_conservative",
        "classification_permissive",
        "agent_rows",
        "compressed_strategy_count",
        "effective_strategy_count",
        "reproducing_strategy_count",
        "strategy_generation_area",
        "reproducing_strategy_generation_area",
        "strategy_area_per_100_births",
        "strategy_area_per_1k_agent_gen",
        "births_per_agent_generation",
        "alive",
        "successful_births",
        "deaths",
        "population_time_area",
        "births_area",
        "metrics_sha256",
        "branch_niche_sha256",
        "run_dir",
    ]
    _write_csv(outputs["campaign_matrix"], main_rows, base_fields)
    _write_csv(outputs["branch_persistence_summary"], main_rows, base_fields)
    _write_csv(outputs["strategy_persistence_summary"], main_rows, base_fields)
    gen_rows = _compact_generation_rows(nominal["strategy_by_generation"])
    _write_csv(
        outputs["branch_metrics_by_generation"],
        gen_rows,
        ["run_id", "arm", "context", "init_arm", "seed", "generation", "strategy_count", "reproducing_strategy_proxy_count", "alive_count_sum", "top_strategy_alive_share"],
    )
    _write_csv(
        outputs["strategy_metrics_by_generation"],
        gen_rows,
        ["run_id", "arm", "context", "init_arm", "seed", "generation", "strategy_count", "reproducing_strategy_proxy_count", "alive_count_sum", "top_strategy_alive_share"],
    )
    _write_csv(
        outputs["branch_strategy_specialization"],
        specialization,
        ["run_id", "arm", "context", "init_arm", "seed", "persistent_strategy_count", "reproducing_persistent_strategy_count", "distinct_dominant_zones", "mean_pairwise_zone_jsd", "formula"],
    )
    _write_csv(outputs["zone_demand_metrics"], zone_rows, ["run_id", "arm", "context", "init_arm", "seed", "alive", "successful_births", "deaths", "population_time_area", "births_area"])
    _write_csv(outputs["population_confounds"], pair_rows, ["context", "init_arm", "seed", "metric", "active", "control", "delta", "relative_delta"])
    _write_csv(outputs["threshold_sensitivity"], threshold_rows, ["run_id", "arm", "context", "init_arm", "seed", "classification_conservative", "classification_nominal", "classification_permissive", "compressed_strategy_count", "effective_strategy_count"])
    _write_json(outputs["repeat_parity"], repeat)
    _write_json(
        outputs["checkpoint_parity"],
        {"schema_version": SCHEMA_VERSION, "required": False, "status": "not_run", "reason": "ROZ-2B changed registry/control composition only; no checkpointed state boundary code was modified."},
    )
    _write_json(outputs["primary_decision"], decision)
    _write_json(
        outputs["horizon_extension"],
        {
            "schema_version": SCHEMA_VERSION,
            "status": decision["horizon_extension"],
            "run": decision["horizon_extension"] == "HORIZON-EXTENSION-ELIGIBLE",
            "reason": "predeclared extension runs only after EFFECTIVE-DIRECTIONAL primary result",
        },
    )
    _write_json(outputs["final_decision"], decision)
    return {"schema_version": SCHEMA_VERSION, "main_run_count": len(main_dirs), "repeat_run_count": len(repeat_dirs), "decision": decision, "outputs": {k: str(v) for k, v in outputs.items()}}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze ROZ-2B branch and strategy effectiveness outputs.")
    parser.add_argument("--runs-root", action="append", default=[], help="Main campaign root. May be repeated.")
    parser.add_argument("--repeat-runs-root", action="append", default=[], help="Repeat parity root. May be repeated.")
    parser.add_argument("--output-dir", required=True, help="Compact evidence output directory.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    payload = analyze(list(args.runs_root or []), list(args.repeat_runs_root or []), Path(args.output_dir))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
