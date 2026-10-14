#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping


SCHEMA_VERSION = 1
TOP1_THRESHOLD = 0.8
EFFECTIVE_COLLAPSE_THRESHOLD = 1.5
RUN_ID_RE = re.compile(r"__context_(?P<context>.+?)__init_(?P<init_arm>.+?)__seed(?P<seed>\d+)$")
NO_CLAIMS = (
    "OES-LITGATE3 is a bounded, gate-attached open-ended-search continuation under the LIT-1 "
    "lineage-integrity gate. It does not show intelligence, emergence, consciousness, broad cognition, "
    "memory selection, prediction selection, solved lineage diversity, lineage repair, beneficial adaptive "
    "mechanism, culture, or superintelligence. It does not add direct cognition reward, memory reward, "
    "prediction reward, lineage rescue, diversity reward, protected reproduction, archive reseeding, or "
    "survival subsidy."
)

KNOWN_CAVEAT_CONTEXT = "wdl_delayed_hazard_full_stack"
KNOWN_CAVEAT_INIT = "founder_balanced"
KNOWN_CAVEAT_SEED = 202
EARLY_WARNING_GENERATION_CUTOFF = 24
RUNAWAY_FINAL_TOP1_THRESHOLD = 0.9

GATE_FIELDS = [
    "run_id",
    "context",
    "init_arm",
    "seed",
    "status",
    "generation_count",
    "top1_dominance_generation",
    "effective_lineage_collapse_generation",
    "final_top1_post_refill_alive_share",
    "final_effective_post_refill_alive_lineages",
    "min_effective_post_refill_alive_lineages",
    "max_top1_post_refill_alive_share",
    "final_selected_parent_effective_lineages",
    "final_successful_birth_effective_lineages",
    "final_top1_selected_parent_share",
    "final_top1_successful_birth_share",
    "final_top1_refill_amplification_ratio",
    "event_file_present",
    "scan_v4_sampled",
    "scan_event_rows_read",
    "scan_event_schema_versions_seen",
    "scan_exposure_rows",
    "scan_attempt_rows",
    "warning_stage",
    "unknowns",
    "run_dir",
    "phase1_output_dir",
]

WARNING_FIELDS = [
    "run_id",
    "context",
    "init_arm",
    "seed",
    "top1_dominance_generation",
    "effective_lineage_collapse_generation",
    "warning_stage",
    "crossing_top1_post_refill_alive_share",
    "crossing_effective_post_refill_alive_lineages",
    "crossing_top1_selected_parent_share",
    "crossing_top1_scan_exposure_share",
    "crossing_top1_attempt_share",
    "crossing_top1_successful_birth_share",
    "crossing_top1_refill_amplification_ratio",
    "scan_v4_sampled",
    "unknowns",
]


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


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


def _safe_float(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return 0.0
    return out if math.isfinite(out) else 0.0


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return int(default)


def _parse_run_id(run_id: str) -> dict[str, Any]:
    match = RUN_ID_RE.search(run_id)
    if not match:
        raise ValueError(f"cannot parse run_id: {run_id}")
    return {
        "context": match.group("context"),
        "init_arm": match.group("init_arm"),
        "seed": int(match.group("seed")),
    }


def _phase1_dir_for_run(phase1_root: Path, run_dir: Path) -> Path:
    exact = phase1_root / run_dir.name
    if exact.is_dir():
        return exact
    matches = sorted(phase1_root.glob(f"*_{run_dir.name}"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected exactly one Phase 1 output directory for run {run_dir.name}, found {len(matches)}"
        )
    return matches[0]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _effective(rows: Iterable[Mapping[str, Any]], share_field: str) -> float:
    shares = [_safe_float(row.get(share_field, 0.0)) for row in rows]
    entropy = -sum(share * math.log(share) for share in shares if share > 0.0)
    return math.exp(entropy) if entropy > 0.0 else 0.0


def _generation_rows(rows: Iterable[Mapping[str, Any]]) -> dict[int, list[Mapping[str, Any]]]:
    out: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        out[_safe_int(row.get("generation", 0))].append(row)
    return out


def _top_row(rows: list[Mapping[str, Any]], field: str = "post_refill_alive_share") -> Mapping[str, Any]:
    return max(rows, key=lambda row: (_safe_float(row.get(field, 0.0)), str(row.get("lineage_id", ""))))


def _top1_generation(gens: Mapping[int, list[Mapping[str, Any]]]) -> int:
    for generation, rows in sorted(gens.items()):
        if _safe_float(_top_row(rows).get("post_refill_alive_share", 0.0)) >= TOP1_THRESHOLD:
            return int(generation)
    return -1


def _effective_generation(gens: Mapping[int, list[Mapping[str, Any]]]) -> int:
    for generation, rows in sorted(gens.items()):
        if _effective(rows, "post_refill_alive_share") <= EFFECTIVE_COLLAPSE_THRESHOLD:
            return int(generation)
    return -1


def _stage_from_rows(
    *,
    compact_rows: list[Mapping[str, Any]],
    scan_rows: list[Mapping[str, Any]] | None,
    generation: int,
) -> tuple[str, dict[str, Any], list[str]]:
    compact_by_gen = _generation_rows(compact_rows)
    compact_top = _top_row(compact_by_gen[generation])
    top_lineage = str(compact_top.get("lineage_id", ""))
    unknowns: list[str] = []
    scan_top: Mapping[str, Any] | None = None
    if scan_rows:
        scan_gen = _generation_rows(scan_rows).get(generation, [])
        scan_matches = [row for row in scan_gen if str(row.get("lineage_id", "")) == top_lineage]
        if scan_matches:
            scan_top = scan_matches[0]
    else:
        unknowns.extend(["scan_exposure_stage", "scan_attempt_stage"])

    values = {
        "top_lineage_id": top_lineage,
        "top1_post_refill_alive_share": _safe_float(compact_top.get("post_refill_alive_share", 0.0)),
        "effective_post_refill_alive_lineages": _effective(compact_by_gen[generation], "post_refill_alive_share"),
        "top1_selected_parent_share": _safe_float(compact_top.get("selected_parent_share", 0.0)),
        "top1_successful_birth_share": _safe_float(compact_top.get("offspring_share", 0.0)),
        "top1_refill_amplification_ratio": _safe_float(compact_top.get("refill_amplification_ratio", 0.0)),
        "top1_scan_exposure_share": _safe_float(scan_top.get("scan_exposure_share", 0.0)) if scan_top else 0.0,
        "top1_attempt_share": _safe_float(scan_top.get("attempt_share", 0.0)) if scan_top else 0.0,
    }

    if scan_top and values["top1_scan_exposure_share"] >= TOP1_THRESHOLD:
        stage = "scan-exposure dominance"
    elif scan_top and values["top1_attempt_share"] >= TOP1_THRESHOLD:
        stage = "attempt dominance"
    elif values["top1_selected_parent_share"] >= TOP1_THRESHOLD:
        stage = "selected-parent dominance"
    elif values["top1_successful_birth_share"] >= TOP1_THRESHOLD:
        stage = "successful-birth dominance"
    elif values["top1_refill_amplification_ratio"] > 1.1:
        stage = "refill amplification"
    else:
        stage = "mixed late dominance"
    return stage, values, unknowns


def _scan_cell_from_phase1(result: Mapping[str, Any], phase1_root: Path) -> dict[str, Any]:
    run_id = str(result["run_id"])
    run_dir = Path(str(result["run_dir"]))
    phase1_dir = _phase1_dir_for_run(phase1_root, run_dir)
    summary = _read_json(phase1_dir / "lit1_lineage_funnel_phase1_summary.json")
    by_gen = _read_csv(phase1_dir / "lit1_lineage_funnel_phase1_by_generation.csv")
    event_summary = summary.get("event_summary", {}) or {}
    counts = event_summary.get("event_type_counts", {}) or {}
    parsed = _parse_run_id(run_id)
    return {
        "run_id": run_id,
        "context": parsed["context"],
        "init_arm": parsed["init_arm"],
        "seed": parsed["seed"],
        "phase1_dir": str(phase1_dir),
        "event_file_present": bool(event_summary.get("event_file_present", False)),
        "event_rows_read": int(event_summary.get("event_rows_read", 0)),
        "event_schema_versions_seen": [str(v) for v in event_summary.get("event_schema_versions_seen", [])],
        "unsupported_event_schema_versions": [
            str(v) for v in event_summary.get("unsupported_event_schema_versions", [])
        ],
        "scan_exposure_rows": int(counts.get("reproduction_scan_exposure", 0)),
        "scan_attempt_rows": int(counts.get("reproduction_scan_attempt_recorded", 0)),
        "by_generation": by_gen,
        "run_dir": str(run_dir),
    }


def _compact_cell_from_phase1(
    result: Mapping[str, Any],
    compact_phase1_root: Path,
    scan_by_run_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    run_id = str(result["run_id"])
    run_dir = Path(str(result["run_dir"]))
    parsed = _parse_run_id(run_id)
    phase1_dir = _phase1_dir_for_run(compact_phase1_root, run_dir)
    summary = _read_json(phase1_dir / "lit1_lineage_funnel_phase1_summary.json")
    coverage = _read_json(phase1_dir / "lit1_lineage_funnel_phase1_metric_coverage.json")
    by_gen_rows = _read_csv(phase1_dir / "lit1_lineage_funnel_phase1_by_generation.csv")
    gens = _generation_rows(by_gen_rows)
    top1_gen = _top1_generation(gens)
    effective_gen = _effective_generation(gens)
    final_gen = max(gens)
    final_rows = gens[final_gen]
    final_top = _top_row(final_rows)
    top1_values = [_safe_float(_top_row(rows).get("post_refill_alive_share", 0.0)) for rows in gens.values()]
    effective_values = [_effective(rows, "post_refill_alive_share") for rows in gens.values()]
    scan_cell = scan_by_run_id.get(run_id)
    scan_rows = list(scan_cell.get("by_generation", [])) if scan_cell else None
    warning_stage = ""
    warning_values: dict[str, Any] = {}
    unknowns: list[str] = []
    if top1_gen >= 0:
        warning_stage, warning_values, unknowns = _stage_from_rows(
            compact_rows=by_gen_rows,
            scan_rows=scan_rows,
            generation=top1_gen,
        )
    elif not scan_cell:
        unknowns = ["scan_exposure_stage", "scan_attempt_stage"]
    return {
        "run_id": run_id,
        "context": parsed["context"],
        "init_arm": parsed["init_arm"],
        "seed": parsed["seed"],
        "status": str(result.get("status", "")),
        "generation_count": int(summary.get("generation_count", 0)),
        "top1_dominance_generation": top1_gen,
        "effective_lineage_collapse_generation": effective_gen,
        "final_top1_post_refill_alive_share": _safe_float(final_top.get("post_refill_alive_share", 0.0)),
        "final_effective_post_refill_alive_lineages": _effective(final_rows, "post_refill_alive_share"),
        "min_effective_post_refill_alive_lineages": min(effective_values) if effective_values else 0.0,
        "max_top1_post_refill_alive_share": max(top1_values) if top1_values else 0.0,
        "final_selected_parent_effective_lineages": _safe_float(
            final_top.get("selected_parent_effective_lineage_count", 0.0)
        ),
        "final_successful_birth_effective_lineages": _effective(final_rows, "offspring_share"),
        "final_top1_selected_parent_share": _safe_float(final_top.get("selected_parent_share", 0.0)),
        "final_top1_successful_birth_share": _safe_float(final_top.get("offspring_share", 0.0)),
        "final_top1_refill_amplification_ratio": _safe_float(
            final_top.get("refill_amplification_ratio", 0.0)
        ),
        "event_file_present": bool((summary.get("event_summary", {}) or {}).get("event_file_present", False)),
        "scan_v4_sampled": bool(scan_cell),
        "scan_event_rows_read": int((scan_cell or {}).get("event_rows_read", 0)),
        "scan_event_schema_versions_seen": ",".join((scan_cell or {}).get("event_schema_versions_seen", [])),
        "scan_exposure_rows": int((scan_cell or {}).get("scan_exposure_rows", 0)),
        "scan_attempt_rows": int((scan_cell or {}).get("scan_attempt_rows", 0)),
        "warning_stage": warning_stage,
        "warning_values": warning_values,
        "unknowns": ",".join(unknowns),
        "missing_required_metrics": ",".join(
            str(item) for item in coverage.get("missing_required_metrics", [])
        ),
        "run_dir": str(run_dir),
        "phase1_output_dir": str(phase1_dir),
    }


def _campaign_results(summary_path: Path) -> list[Mapping[str, Any]]:
    return list(_read_json(summary_path).get("results", []))


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _repeat_summary(root_a: Path | None, root_b: Path | None) -> dict[str, Any]:
    if not root_a or not root_b:
        return {"performed": False, "passed": False, "reason": "not_requested"}
    files = [
        "metrics/lineage_reproduction_funnel_generation_summary.csv",
        "metrics/lineage_reproduction_funnel_run_summary.json",
        "metrics.csv",
    ]

    def dirs(root: Path) -> list[Path]:
        return sorted(root.glob("children/*/e/*/b*/r/r*"))

    a_dirs = dirs(root_a)
    b_dirs = dirs(root_b)
    checks = []
    passed = len(a_dirs) == len(b_dirs) and len(a_dirs) > 0
    for idx, (a_run, b_run) in enumerate(zip(a_dirs, b_dirs), start=1):
        item = {"cell": f"cell_{idx:02d}", "a_run_dir": str(a_run), "b_run_dir": str(b_run), "file_checks": []}
        for rel in files:
            a_path = a_run / rel
            b_path = b_run / rel
            row = {
                "path": rel,
                "a_exists": a_path.exists(),
                "b_exists": b_path.exists(),
                "a_sha256": _hash_file(a_path) if a_path.exists() else "",
                "b_sha256": _hash_file(b_path) if b_path.exists() else "",
            }
            row["match"] = bool(row["a_sha256"] and row["a_sha256"] == row["b_sha256"])
            passed = passed and row["match"]
            item["file_checks"].append(row)
        item["passed"] = all(row["match"] for row in item["file_checks"])
        checks.append(item)
    return {
        "performed": True,
        "passed": passed,
        "selected_cells": len(checks),
        "roots": {"a": str(root_a), "b": str(root_b)},
        "checks": checks,
    }


def _is_known_caveat(row: Mapping[str, Any]) -> bool:
    return (
        str(row.get("context")) == KNOWN_CAVEAT_CONTEXT
        and str(row.get("init_arm")) == KNOWN_CAVEAT_INIT
        and int(row.get("seed", -1)) == KNOWN_CAVEAT_SEED
    )


def _required_scan_cells() -> set[tuple[str, str, int]]:
    return {
        (KNOWN_CAVEAT_CONTEXT, KNOWN_CAVEAT_INIT, 202),
        (KNOWN_CAVEAT_CONTEXT, KNOWN_CAVEAT_INIT, 101),
        (KNOWN_CAVEAT_CONTEXT, KNOWN_CAVEAT_INIT, 303),
        (KNOWN_CAVEAT_CONTEXT, "legacy", 202),
        ("tuned_full_jungle_recovery", KNOWN_CAVEAT_INIT, 202),
        ("flat_control", KNOWN_CAVEAT_INIT, 202),
    }


def _scan_key(row: Mapping[str, Any]) -> tuple[str, str, int]:
    return (str(row.get("context")), str(row.get("init_arm")), int(row.get("seed", -1)))


def _warning_stage_classification(warning_rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "warning_count": len(warning_rows),
        "classifications": [
            {
                "run_id": row["run_id"],
                "context": row["context"],
                "init_arm": row["init_arm"],
                "seed": row["seed"],
                "known_caveat_cell": _is_known_caveat(row),
                "top1_dominance_generation": row["top1_dominance_generation"],
                "effective_lineage_collapse_generation": row["effective_lineage_collapse_generation"],
                "classification": row["warning_stage"] or "insufficient data",
                "scan_v4_sampled": row["scan_v4_sampled"],
                "supporting_values_at_crossing": {
                    "top1_post_refill_alive_share": row["crossing_top1_post_refill_alive_share"],
                    "effective_post_refill_alive_lineages": row["crossing_effective_post_refill_alive_lineages"],
                    "top1_selected_parent_share": row["crossing_top1_selected_parent_share"],
                    "top1_scan_exposure_share": row["crossing_top1_scan_exposure_share"],
                    "top1_attempt_share": row["crossing_top1_attempt_share"],
                    "top1_successful_birth_share": row["crossing_top1_successful_birth_share"],
                    "top1_refill_amplification_ratio": row["crossing_top1_refill_amplification_ratio"],
                },
                "unknowns": row["unknowns"],
            }
            for row in warning_rows
        ],
        "no_claims": NO_CLAIMS,
    }


def _caveat_assessment(
    *,
    gate_rows: list[Mapping[str, Any]],
    scan_cells: list[Mapping[str, Any]],
    repeat: Mapping[str, Any],
    decision: Mapping[str, Any],
) -> dict[str, Any]:
    warning_cells = [row for row in gate_rows if int(row["top1_dominance_generation"]) >= 0]
    known_warning_cells = [row for row in warning_cells if _is_known_caveat(row)]
    broadened_warning_cells = [row for row in warning_cells if not _is_known_caveat(row)]
    effective_collapse_cells = [
        row for row in gate_rows if int(row["effective_lineage_collapse_generation"]) >= 0
    ]
    early_warning_cells = [
        row for row in warning_cells
        if int(row["top1_dominance_generation"]) < EARLY_WARNING_GENERATION_CUTOFF
    ]
    runaway_warning_cells = [
        row for row in warning_cells
        if _safe_float(row["final_top1_post_refill_alive_share"]) > RUNAWAY_FINAL_TOP1_THRESHOLD
    ]
    known_scan_cells = [
        row for row in scan_cells
        if _scan_key(row) == (KNOWN_CAVEAT_CONTEXT, KNOWN_CAVEAT_INIT, KNOWN_CAVEAT_SEED)
    ]
    neighbor_rows = [
        row for row in gate_rows
        if str(row["context"]) == KNOWN_CAVEAT_CONTEXT
        and str(row["init_arm"]) == KNOWN_CAVEAT_INIT
        and int(row["seed"]) != KNOWN_CAVEAT_SEED
    ]
    neighbor_scan_cells = [
        row for row in scan_cells
        if str(row["context"]) == KNOWN_CAVEAT_CONTEXT
        and str(row["init_arm"]) == KNOWN_CAVEAT_INIT
        and int(row["seed"]) != KNOWN_CAVEAT_SEED
    ]
    stable = (
        len(known_warning_cells) <= 1
        and not broadened_warning_cells
        and not effective_collapse_cells
        and not early_warning_cells
        and not runaway_warning_cells
        and bool(known_scan_cells or not known_warning_cells)
        and bool(repeat.get("passed", False))
        and decision.get("decision") in {"OES-LITGATE3-A", "OES-LITGATE3-B"}
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "known_caveat": {
            "context": KNOWN_CAVEAT_CONTEXT,
            "init_arm": KNOWN_CAVEAT_INIT,
            "seed": KNOWN_CAVEAT_SEED,
        },
        "known_caveat_recurred": bool(known_warning_cells),
        "known_caveat_warning_cells": [row["run_id"] for row in known_warning_cells],
        "broadened_warning_cells": [row["run_id"] for row in broadened_warning_cells],
        "effective_lineage_collapse_cells": [row["run_id"] for row in effective_collapse_cells],
        "early_warning_cells": [row["run_id"] for row in early_warning_cells],
        "runaway_warning_cells": [row["run_id"] for row in runaway_warning_cells],
        "known_caveat_scan_v4_cells": [
            {
                "run_id": row["run_id"],
                "event_rows_read": row["event_rows_read"],
                "scan_exposure_rows": row["scan_exposure_rows"],
                "scan_attempt_rows": row["scan_attempt_rows"],
                "event_schema_versions_seen": row["event_schema_versions_seen"],
            }
            for row in known_scan_cells
        ],
        "neighbor_wdl_founder_cells": [
            {
                "run_id": row["run_id"],
                "seed": row["seed"],
                "top1_dominance_generation": row["top1_dominance_generation"],
                "final_top1_post_refill_alive_share": row["final_top1_post_refill_alive_share"],
                "max_top1_post_refill_alive_share": row["max_top1_post_refill_alive_share"],
                "final_effective_post_refill_alive_lineages": row[
                    "final_effective_post_refill_alive_lineages"
                ],
                "scan_v4_sampled": row["scan_v4_sampled"],
                "scan_exposure_rows": row["scan_exposure_rows"],
                "scan_attempt_rows": row["scan_attempt_rows"],
            }
            for row in sorted(neighbor_rows, key=lambda item: int(item["seed"]))
        ],
        "neighbor_wdl_founder_scan_v4_cells": [
            {
                "run_id": row["run_id"],
                "seed": row["seed"],
                "event_rows_read": row["event_rows_read"],
                "scan_exposure_rows": row["scan_exposure_rows"],
                "scan_attempt_rows": row["scan_attempt_rows"],
            }
            for row in sorted(neighbor_scan_cells, key=lambda item: int(item["seed"]))
        ],
        "repeat_parity_passed": bool(repeat.get("passed", False)),
        "stable_bounded_caveat": stable,
        "acceptance": (
            "accepted_as_bounded_comparison_caveat"
            if stable and known_warning_cells
            else "not_recurred_clean_pass"
            if stable
            else "not_accepted_gate_blocked_or_integrity_failed"
        ),
        "no_claims": NO_CLAIMS,
    }


def _decision(
    *,
    compact_summary: Mapping[str, Any],
    scan_summary: Mapping[str, Any],
    gate_rows: list[Mapping[str, Any]],
    scan_cells: list[Mapping[str, Any]],
    repeat: Mapping[str, Any],
    expected_compact_runs: int,
    expected_scan_runs: int,
) -> dict[str, Any]:
    integrity_failed: list[str] = []
    gate_blocked: list[str] = []
    warnings: list[str] = []
    compact_failures = int(compact_summary.get("failures", 0) or 0)
    scan_failures = int(scan_summary.get("failures", 0) or 0)
    if compact_failures or scan_failures:
        integrity_failed.append("campaign_failures")
    if len(gate_rows) != expected_compact_runs:
        integrity_failed.append("compact_coverage")
    if len(scan_cells) != expected_scan_runs:
        integrity_failed.append("scan_coverage")
    if not all(row.get("status") == "ok" for row in gate_rows):
        integrity_failed.append("compact_status")
    if not repeat.get("passed", False):
        integrity_failed.append("repeat_parity")
    if expected_scan_runs >= 6:
        observed_scan = {_scan_key(row) for row in scan_cells}
        missing_required = sorted(_required_scan_cells() - observed_scan)
        if missing_required:
            integrity_failed.append("scan_required_cells_missing")

    top1_warning_cells = [row for row in gate_rows if int(row["top1_dominance_generation"]) >= 0]
    effective_collapse_cells = [
        row for row in gate_rows if int(row["effective_lineage_collapse_generation"]) >= 0
    ]
    if effective_collapse_cells:
        gate_blocked.append("effective_lineage_collapse")
    if top1_warning_cells:
        warnings.append("top1_dominance")

    warning_run_ids = {str(row["run_id"]) for row in top1_warning_cells}
    for row in scan_cells:
        run_id = str(row["run_id"])
        if row.get("event_schema_versions_seen") and row.get("event_schema_versions_seen") != ["4"]:
            integrity_failed.append("scan_schema_v4_only")
        event_missing = not bool(row.get("event_file_present"))
        scan_rows_missing = not (
            int(row.get("scan_exposure_rows", 0)) > 0
            and int(row.get("scan_attempt_rows", 0)) > 0
        )
        if run_id in warning_run_ids:
            if event_missing:
                gate_blocked.append("warning_scan_v4_missing")
            if scan_rows_missing:
                gate_blocked.append("warning_scan_rows_missing")
        else:
            if event_missing:
                integrity_failed.append("scan_event_files")
            if scan_rows_missing:
                integrity_failed.append("scan_rows")

    if any(not _is_known_caveat(row) for row in top1_warning_cells):
        gate_blocked.append("warning_broadened")
    if any(int(row["top1_dominance_generation"]) < EARLY_WARNING_GENERATION_CUTOFF for row in top1_warning_cells):
        gate_blocked.append("early_top1_warning")
    if any(
        _safe_float(row["final_top1_post_refill_alive_share"]) > RUNAWAY_FINAL_TOP1_THRESHOLD
        for row in top1_warning_cells
    ):
        gate_blocked.append("runaway_top1_warning")

    if integrity_failed:
        decision = "OES-LITGATE3-D"
        gate_result = "integrity_fail"
    elif gate_blocked:
        decision = "OES-LITGATE3-C"
        gate_result = "blocked"
    elif warnings:
        decision = "OES-LITGATE3-B"
        gate_result = "warn_bounded"
    else:
        decision = "OES-LITGATE3-A"
        gate_result = "pass"

    return {
        "schema_version": SCHEMA_VERSION,
        "decision": decision,
        "gate_result": gate_result,
        "failed_criteria": sorted(set(integrity_failed + gate_blocked)),
        "integrity_failed_criteria": sorted(set(integrity_failed)),
        "gate_blocked_criteria": sorted(set(gate_blocked)),
        "warning_criteria": sorted(set(warnings)),
        "top1_warning_cells": [row["run_id"] for row in top1_warning_cells],
        "effective_lineage_collapse_cells": [row["run_id"] for row in effective_collapse_cells],
        "compact_cells": len(gate_rows),
        "scan_v4_cells": len(scan_cells),
        "repeat_parity": repeat,
        "open_search_interpretable": decision in {"OES-LITGATE3-A", "OES-LITGATE3-B"},
        "main_supported_finding": (
            "OES-LITGATE3 produced bounded gate-attached search telemetry under the tested matrix."
            if decision == "OES-LITGATE3-A"
            else "OES-LITGATE3 remains interpretable only with the known WDL founder seed-202 caveat attached."
            if decision == "OES-LITGATE3-B"
            else "OES-LITGATE3 produced a fail-closed gate package, but interpretation is blocked."
        ),
        "main_unsupported_finding": (
            "No intelligence, emergence, broad cognition, memory selection, prediction selection, "
            "solved lineage diversity, lineage repair, or beneficial adaptive mechanism claim is supported."
        ),
        "no_claims": NO_CLAIMS,
    }


def build_report(
    *,
    compact_campaign_summary_path: Path,
    compact_phase1_output_root: Path,
    scan_campaign_summary_path: Path,
    scan_phase1_output_root: Path,
    output_dir: Path,
    expected_compact_runs: int,
    expected_scan_runs: int,
    repeat_a: Path | None = None,
    repeat_b: Path | None = None,
) -> dict[str, Any]:
    compact_summary = _read_json(compact_campaign_summary_path)
    scan_summary = _read_json(scan_campaign_summary_path)
    scan_cells = [
        _scan_cell_from_phase1(result, scan_phase1_output_root)
        for result in sorted(_campaign_results(scan_campaign_summary_path), key=lambda row: int(row.get("index", 0)))
    ]
    scan_by_run_id = {str(row["run_id"]): row for row in scan_cells}
    gate_rows = [
        _compact_cell_from_phase1(result, compact_phase1_output_root, scan_by_run_id)
        for result in sorted(_campaign_results(compact_campaign_summary_path), key=lambda row: int(row.get("index", 0)))
    ]
    repeat = _repeat_summary(repeat_a, repeat_b)
    decision = _decision(
        compact_summary=compact_summary,
        scan_summary=scan_summary,
        gate_rows=gate_rows,
        scan_cells=scan_cells,
        repeat=repeat,
        expected_compact_runs=expected_compact_runs,
        expected_scan_runs=expected_scan_runs,
    )
    warning_rows = [
        {
            "run_id": row["run_id"],
            "context": row["context"],
            "init_arm": row["init_arm"],
            "seed": row["seed"],
            "top1_dominance_generation": row["top1_dominance_generation"],
            "effective_lineage_collapse_generation": row["effective_lineage_collapse_generation"],
            "warning_stage": row["warning_stage"],
            "crossing_top1_post_refill_alive_share": row.get("warning_values", {}).get("top1_post_refill_alive_share", ""),
            "crossing_effective_post_refill_alive_lineages": row.get("warning_values", {}).get("effective_post_refill_alive_lineages", ""),
            "crossing_top1_selected_parent_share": row.get("warning_values", {}).get("top1_selected_parent_share", ""),
            "crossing_top1_scan_exposure_share": row.get("warning_values", {}).get("top1_scan_exposure_share", ""),
            "crossing_top1_attempt_share": row.get("warning_values", {}).get("top1_attempt_share", ""),
            "crossing_top1_successful_birth_share": row.get("warning_values", {}).get("top1_successful_birth_share", ""),
            "crossing_top1_refill_amplification_ratio": row.get("warning_values", {}).get("top1_refill_amplification_ratio", ""),
            "scan_v4_sampled": row["scan_v4_sampled"],
            "unknowns": row["unknowns"],
        }
        for row in gate_rows
        if int(row["top1_dominance_generation"]) >= 0
        or int(row["effective_lineage_collapse_generation"]) >= 0
    ]
    context_groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in gate_rows:
        context_groups[(str(row["context"]), str(row["init_arm"]))].append(row)
    gate_summary = {
        "schema_version": SCHEMA_VERSION,
        "decision": decision,
        "context_summary": [
            {
                "context": context,
                "init_arm": init_arm,
                "runs": len(rows),
                "top1_warning_cells": sum(1 for row in rows if int(row["top1_dominance_generation"]) >= 0),
                "effective_collapse_cells": sum(
                    1 for row in rows if int(row["effective_lineage_collapse_generation"]) >= 0
                ),
                "final_top1_mean": statistics.fmean(
                    [_safe_float(row["final_top1_post_refill_alive_share"]) for row in rows]
                ),
                "final_top1_max": max(_safe_float(row["final_top1_post_refill_alive_share"]) for row in rows),
                "final_effective_min": min(
                    _safe_float(row["final_effective_post_refill_alive_lineages"]) for row in rows
                ),
            }
            for (context, init_arm), rows in sorted(context_groups.items())
        ],
    }
    scan_coverage = {
        "schema_version": SCHEMA_VERSION,
        "expected_scan_runs": expected_scan_runs,
        "scan_runs": len(scan_cells),
        "sampled_cells": [
            {key: row[key] for key in [
                "run_id", "context", "init_arm", "seed", "event_file_present",
                "event_rows_read", "event_schema_versions_seen", "scan_exposure_rows",
                "scan_attempt_rows", "run_dir", "phase1_dir",
            ]}
            for row in scan_cells
        ],
    }
    open_search_summary = {
        "schema_version": SCHEMA_VERSION,
        "interpretation_allowed": decision["open_search_interpretable"],
        "decision": decision["decision"],
        "gate_result": decision["gate_result"],
        "bounded_indicators": {
            "compact_cells": len(gate_rows),
            "contexts": sorted({str(row["context"]) for row in gate_rows}),
            "init_arms": sorted({str(row["init_arm"]) for row in gate_rows}),
            "seeds": sorted({int(row["seed"]) for row in gate_rows}),
            "final_effective_lineage_min": min(
                _safe_float(row["final_effective_post_refill_alive_lineages"]) for row in gate_rows
            ) if gate_rows else 0.0,
            "final_top1_max": max(
                _safe_float(row["final_top1_post_refill_alive_share"]) for row in gate_rows
            ) if gate_rows else 0.0,
        },
        "claim_boundary": NO_CLAIMS,
    }
    warning_stage_classification = _warning_stage_classification(warning_rows)
    caveat_assessment = _caveat_assessment(
        gate_rows=gate_rows,
        scan_cells=scan_cells,
        repeat=repeat,
        decision=decision,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "decision": output_dir / "oes_litgate3_decision.json",
        "gate_summary": output_dir / "oes_litgate3_gate_summary.json",
        "gate_matrix": output_dir / "oes_litgate3_gate_matrix.csv",
        "warning_cells": output_dir / "oes_litgate3_warning_cells.csv",
        "scan_v4_coverage": output_dir / "oes_litgate3_scan_v4_coverage.json",
        "warning_stage_classification": output_dir / "oes_litgate3_warning_stage_classification.json",
        "caveat_assessment": output_dir / "oes_litgate3_caveat_assessment.json",
        "open_search_summary": output_dir / "oes_litgate3_open_search_summary.json",
        "repeat_parity_summary": output_dir / "oes_litgate3_repeat_parity_summary.json",
    }
    _write_json(outputs["decision"], decision)
    _write_json(outputs["gate_summary"], gate_summary)
    _write_csv(outputs["gate_matrix"], gate_rows, GATE_FIELDS)
    _write_csv(outputs["warning_cells"], warning_rows, WARNING_FIELDS)
    _write_json(outputs["scan_v4_coverage"], scan_coverage)
    _write_json(outputs["warning_stage_classification"], warning_stage_classification)
    _write_json(outputs["caveat_assessment"], caveat_assessment)
    _write_json(outputs["open_search_summary"], open_search_summary)
    _write_json(outputs["repeat_parity_summary"], repeat)
    return {
        "outputs": {key: str(path) for key, path in outputs.items()},
        "decision": decision,
        "gate_rows": gate_rows,
        "warning_rows": warning_rows,
        "scan_cells": scan_cells,
        "gate_summary": gate_summary,
        "warning_stage_classification": warning_stage_classification,
        "caveat_assessment": caveat_assessment,
        "open_search_summary": open_search_summary,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build OES-LITGATE3 decision/evidence outputs.")
    parser.add_argument("--compact-campaign-summary", type=Path, required=True)
    parser.add_argument("--compact-phase1-output-root", type=Path, required=True)
    parser.add_argument("--scan-campaign-summary", type=Path, required=True)
    parser.add_argument("--scan-phase1-output-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-compact-runs", type=int, required=True)
    parser.add_argument("--expected-scan-runs", type=int, required=True)
    parser.add_argument("--repeat-a", type=Path)
    parser.add_argument("--repeat-b", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    report = build_report(
        compact_campaign_summary_path=args.compact_campaign_summary,
        compact_phase1_output_root=args.compact_phase1_output_root,
        scan_campaign_summary_path=args.scan_campaign_summary,
        scan_phase1_output_root=args.scan_phase1_output_root,
        output_dir=args.output_dir,
        expected_compact_runs=args.expected_compact_runs,
        expected_scan_runs=args.expected_scan_runs,
        repeat_a=args.repeat_a,
        repeat_b=args.repeat_b,
    )
    print(json.dumps({"decision": report["decision"]["decision"], "outputs": report["outputs"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
