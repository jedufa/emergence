#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

csv.field_size_limit(1024 * 1024 * 1024)

TOP1_THRESHOLD = 0.8
EFFECTIVE_COLLAPSE_THRESHOLD = 1.5
DECISION_NO_CLAIMS = (
    "LRO-3 is a bounded lineage-blind repair comparison and fallback task. It does not show intelligence, "
    "emergence, consciousness, broad cognition, memory selection, prediction selection, solved lineage diversity, "
    "general lineage repair, beneficial adaptive mechanism, culture, superintelligence, or general open-ended "
    "evolution. It does not add direct cognition reward, lineage rescue, diversity reward, protected reproduction, "
    "forced equal reproduction, archive reseeding, or survival subsidy."
)
LRO2_EVIDENCE = Path("evidences/06-open-ended-search/07-lro2-local-reproductive-opportunity-tiny-validation")
DEFAULT_EVIDENCE = Path("evidences/06-open-ended-search/08-lro3-autonomous-comparison")
DEFAULT_REPORT = Path("docs/500_campaigns-decisions-closeouts/520_campaign-reports/300_lineage-reproduction/EIEL-DOC-000403--eiel-lro3-autonomous-lineage-blind-repair-comparison-2026-07-10.md")
RUN_RE = re.compile(r"__setting_(?P<setting>.+?)__context_(?P<context>.+?)__init_(?P<init_arm>.+?)__seed(?P<seed>\d+)$")
EVENT_FLOAT_PATTERNS = {
    name: re.compile(rb'"' + re.escape(name.encode("utf-8")) + rb'":\s*"?(-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)')
    for name in (
        "lro1_opportunity_multiplier",
        "lro1_opportunity_score_parent_site",
        "lro1_opportunity_context_density",
        "lro1_opportunity_context_waste",
        "lro1_opportunity_context_shelter_cover",
        "lro1_opportunity_context_food",
    )
}
EVENT_ENABLE_RE = re.compile(rb'"lro1_reproductive_opportunity_enable":\s*"?([01])')


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fields: list[str] | None = None) -> None:
    rows = list(rows)
    if fields is None:
        fields = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return int(default)


def _last_csv_row(path: Path) -> dict[str, str]:
    last: dict[str, str] = {}
    if not path.exists():
        return last
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            last = dict(row)
    return last


def _parse_run_id(run_id: str) -> dict[str, Any]:
    match = RUN_RE.search(run_id)
    if not match:
        return {"setting": "", "context": "", "init_arm": "", "seed": -1}
    return {
        "setting": match.group("setting"),
        "context": match.group("context"),
        "init_arm": match.group("init_arm"),
        "seed": int(match.group("seed")),
    }


def _effective(shares: Iterable[float]) -> float:
    entropy = -sum(share * math.log(share) for share in shares if share > 0.0)
    return math.exp(entropy) if entropy > 0.0 else 0.0


def _generation_summary(path: Path) -> dict[str, Any]:
    rows = _read_csv(path) if path.exists() else []
    by_gen: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_gen[_int(row.get("generation"))].append(row)
    timeline = []
    top1_gen = -1
    effective_gen = -1
    for gen, gen_rows in sorted(by_gen.items()):
        post = [_float(row.get("post_refill_alive_share")) for row in gen_rows]
        selected = [_float(row.get("selected_parent_share")) for row in gen_rows]
        offspring = [_float(row.get("offspring_share")) for row in gen_rows]
        top1_post = max(post) if post else 0.0
        effective_post = _effective(post)
        item = {
            "generation": int(gen),
            "top1_post_refill_alive_share": top1_post,
            "effective_post_refill_alive_lineages": effective_post,
            "top1_selected_parent_share": max(selected) if selected else 0.0,
            "effective_selected_parent_lineages": _effective(selected),
            "top1_successful_birth_share": max(offspring) if offspring else 0.0,
            "effective_successful_birth_lineages": _effective(offspring),
            "birth_attempts": sum(_float(row.get("birth_attempts")) for row in gen_rows),
            "successful_births": sum(_float(row.get("successful_births")) for row in gen_rows),
            "failed_births": sum(_float(row.get("failed_births")) for row in gen_rows),
        }
        if top1_gen < 0 and top1_post >= TOP1_THRESHOLD:
            top1_gen = int(gen)
        if effective_gen < 0 and effective_post <= EFFECTIVE_COLLAPSE_THRESHOLD:
            effective_gen = int(gen)
        timeline.append(item)
    final = timeline[-1] if timeline else {}
    return {
        "generation_count": len(timeline),
        "top1_dominance_generation": top1_gen,
        "effective_lineage_collapse_generation": effective_gen,
        "final_top1_post_refill_alive_share": final.get("top1_post_refill_alive_share", 0.0),
        "final_effective_post_refill_alive_lineages": final.get("effective_post_refill_alive_lineages", 0.0),
        "final_top1_selected_parent_share": final.get("top1_selected_parent_share", 0.0),
        "final_top1_successful_birth_share": final.get("top1_successful_birth_share", 0.0),
        "timeline_tail": timeline[-5:],
    }


def _child_summaries(root: Path) -> list[Path]:
    out = []
    for path in sorted(root.glob("children/*/*/*/*/summary.json")):
        try:
            if _read_json(path).get("runs"):
                out.append(path)
        except Exception:
            pass
    return out


def _campaign_info(root: Path | None) -> dict[str, Any]:
    if root is None:
        return {"provided": False}
    summary = root / "parallel_campaign_summary.json"
    if not summary.exists():
        return {"provided": True, "exists": False, "runs_root": str(root)}
    payload = _read_json(summary)
    return {
        "provided": True,
        "exists": True,
        "status": payload.get("status"),
        "planned_runs": payload.get("planned_runs"),
        "failures": len(payload.get("failures", []) or []),
        "total_wall_s": payload.get("total_wall_s"),
        "runs_root": payload.get("runs_root", str(root)),
        "campaign_output_profile": payload.get("campaign_output_profile"),
    }


def _run_rows(root: Path | None) -> list[dict[str, Any]]:
    if root is None or not root.exists():
        return []
    rows = []
    for summary_path in _child_summaries(root):
        payload = _read_json(summary_path)
        run = dict(payload["runs"][0])
        run_dir = Path(run["run_dir"])
        final_metrics = _last_csv_row(run_dir / "metrics.csv")
        funnel_path = run_dir / "metrics" / "lineage_reproduction_funnel_run_summary.json"
        funnel = _read_json(funnel_path) if funnel_path.exists() else {}
        gen = _generation_summary(run_dir / "metrics" / "lineage_reproduction_funnel_generation_summary.csv")
        parsed = _parse_run_id(str(run.get("run_id", "")))
        row = {
            "run_id": run.get("run_id", ""),
            "runtime_run_id": run.get("runtime_run_id", ""),
            "status": run.get("status", ""),
            "setting": parsed["setting"] or run.get("axes", {}).get("setting", ""),
            "context": parsed["context"] or run.get("axes", {}).get("context", ""),
            "init_arm": parsed["init_arm"] or run.get("axes", {}).get("init_arm", ""),
            "seed": parsed["seed"] if parsed["seed"] >= 0 else _int(run.get("seed")),
            "run_dir": str(run_dir),
            "metrics_sha256": _sha256(run_dir / "metrics.csv") if (run_dir / "metrics.csv").exists() else "",
            "funnel_summary_sha256": _sha256(funnel_path) if funnel_path.exists() else "",
            "event_schema_version": _int(funnel.get("event_schema_version")),
            "lro1_event_rows": _int(funnel.get("lro1_event_rows")),
            "lro1_enabled_rows": _int(funnel.get("lro1_enabled_rows")),
            "lro1_disabled_rows": _int(funnel.get("lro1_disabled_rows")),
            "lro1_disabled_leakage_rows": _int(funnel.get("lro1_disabled_leakage_rows")),
            "lro1_multiplier_lt1_rows": _int(funnel.get("lro1_multiplier_lt1_rows")),
            "lro1_multiplier_eq1_rows": _int(funnel.get("lro1_multiplier_eq1_rows")),
            "lro1_invalid_context_rows": _int(funnel.get("lro1_invalid_context_rows")),
            "reproduction_scan_exposure_rows": _int(funnel.get("reproduction_scan_exposure_rows")),
            "reproduction_scan_eligible_rows": _int(funnel.get("reproduction_scan_eligible_rows")),
            "reproduction_scan_no_attempt_rows": _int(funnel.get("reproduction_scan_no_attempt_rows")),
            "birth_success_rows": _int(funnel.get("birth_success_rows")),
            "birth_fail_rows": _int(funnel.get("birth_fail_rows")),
            **gen,
        }
        for field in (
            "alive",
            "deaths",
            "birth_attempts",
            "successful_births",
            "failed_births",
            "birth_fail_no_space",
            "lineage_active_count",
            "lineage_effective_alive_lineages",
            "lineage_top1_alive_share",
            "patch_density_mean",
            "patch_density_max",
            "patch_waste_mean",
            "patch_scent_mean",
            "patch_food_amount_mean",
            "patch_cover_mean",
            "patch_shelter_mean",
            "jungle_predator_risk_mean",
            "wdl_delayed_hazard_event_count",
        ):
            row[field] = final_metrics.get(field, "")
        rows.append(row)
    return sorted(rows, key=lambda r: (str(r["setting"]), str(r["context"]), str(r["init_arm"]), int(r["seed"])))


def _extract_float(line: bytes, name: str) -> float | None:
    match = EVENT_FLOAT_PATTERNS[name].search(line)
    if not match:
        return None
    try:
        return float(match.group(1))
    except Exception:
        return None


def _event_stats(path: Path) -> dict[str, Any]:
    stats: dict[str, Any] = {
        "event_jsonl_path": str(path),
        "event_jsonl_bytes": path.stat().st_size,
        "sampled_enabled_rows": 0,
        "sampled_disabled_rows": 0,
        "sampled_disabled_leakage_rows": 0,
        "multiplier_min": None,
        "multiplier_max": None,
        "multiplier_sum": 0.0,
        "score_sum": 0.0,
        "high_density_or_waste_rows": 0,
        "high_density_or_waste_multiplier_sum": 0.0,
        "shelter_food_rows": 0,
        "shelter_food_multiplier_sum": 0.0,
    }
    with path.open("rb") as handle:
        for line in handle:
            match = EVENT_ENABLE_RE.search(line)
            if not match:
                continue
            enabled = match.group(1) == b"1"
            multiplier = _extract_float(line, "lro1_opportunity_multiplier")
            score = _extract_float(line, "lro1_opportunity_score_parent_site")
            if enabled:
                stats["sampled_enabled_rows"] += 1
                if multiplier is not None:
                    stats["multiplier_sum"] += multiplier
                    stats["multiplier_min"] = multiplier if stats["multiplier_min"] is None else min(stats["multiplier_min"], multiplier)
                    stats["multiplier_max"] = multiplier if stats["multiplier_max"] is None else max(stats["multiplier_max"], multiplier)
                if score is not None:
                    stats["score_sum"] += score
                density = _extract_float(line, "lro1_opportunity_context_density") or 0.0
                waste = _extract_float(line, "lro1_opportunity_context_waste") or 0.0
                shelter = _extract_float(line, "lro1_opportunity_context_shelter_cover") or 0.0
                food = _extract_float(line, "lro1_opportunity_context_food") or 0.0
                if multiplier is not None and (density >= 4.0 or waste >= 0.25):
                    stats["high_density_or_waste_rows"] += 1
                    stats["high_density_or_waste_multiplier_sum"] += multiplier
                if multiplier is not None and (shelter >= 0.25 or food >= 0.25):
                    stats["shelter_food_rows"] += 1
                    stats["shelter_food_multiplier_sum"] += multiplier
            else:
                stats["sampled_disabled_rows"] += 1
                if multiplier is not None and abs(multiplier - 1.0) > 1e-12:
                    stats["sampled_disabled_leakage_rows"] += 1
                if score is not None and abs(score) > 1e-12:
                    stats["sampled_disabled_leakage_rows"] += 1
    enabled_rows = int(stats["sampled_enabled_rows"])
    stats["multiplier_mean"] = stats["multiplier_sum"] / enabled_rows if enabled_rows else None
    stats["score_mean"] = stats["score_sum"] / enabled_rows if enabled_rows else None
    stats["high_density_or_waste_multiplier_mean"] = (
        stats["high_density_or_waste_multiplier_sum"] / stats["high_density_or_waste_rows"]
        if stats["high_density_or_waste_rows"]
        else None
    )
    stats["shelter_food_multiplier_mean"] = (
        stats["shelter_food_multiplier_sum"] / stats["shelter_food_rows"] if stats["shelter_food_rows"] else None
    )
    for key in ("multiplier_sum", "score_sum", "high_density_or_waste_multiplier_sum", "shelter_food_multiplier_sum"):
        stats.pop(key, None)
    return stats


def _scan_rows(root: Path | None) -> list[dict[str, Any]]:
    rows = _run_rows(root)
    for row in rows:
        event_path = Path(row["run_dir"]) / "metrics" / "lineage_reproduction_funnel_events.jsonl"
        if event_path.exists():
            row.update(_event_stats(event_path))
    return rows


def _paired(rows: list[dict[str, Any]], candidate_setting: str) -> list[dict[str, Any]]:
    by_key = {(r["setting"], r["context"], r["init_arm"], int(r["seed"])): r for r in rows}
    out = []
    for row in rows:
        if row["setting"] != candidate_setting:
            continue
        base = by_key.get(("baseline_flags_off", row["context"], row["init_arm"], int(row["seed"])))
        if not base:
            continue
        item = {
            "context": row["context"],
            "init_arm": row["init_arm"],
            "seed": row["seed"],
            "baseline_run_id": base["run_id"],
            "candidate_run_id": row["run_id"],
        }
        for field in (
            "alive",
            "deaths",
            "birth_attempts",
            "successful_births",
            "failed_births",
            "birth_fail_no_space",
            "final_top1_post_refill_alive_share",
            "final_effective_post_refill_alive_lineages",
            "final_top1_selected_parent_share",
            "final_top1_successful_birth_share",
            "wdl_delayed_hazard_event_count",
        ):
            item[f"baseline_{field}"] = _float(base.get(field))
            item[f"candidate_{field}"] = _float(row.get(field))
            item[f"delta_{field}"] = _float(row.get(field)) - _float(base.get(field))
        out.append(item)
    return out


def _summary_stats(rows: list[dict[str, Any]], fields: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {"n": len(rows)}
    for field in fields:
        vals = [_float(row.get(field)) for row in rows]
        out[f"{field}_mean"] = statistics.fmean(vals) if vals else 0.0
        out[f"{field}_min"] = min(vals) if vals else 0.0
        out[f"{field}_max"] = max(vals) if vals else 0.0
    return out


def _repeat_parity(matrix_rows: list[dict[str, Any]], repeat_rows: list[dict[str, Any]]) -> dict[str, Any]:
    matrix = {str(row["run_id"]): row for row in matrix_rows}
    keys = ["metrics_sha256", "funnel_summary_sha256", "alive", "successful_births", "failed_births", "birth_attempts"]
    comparisons = []
    passed = bool(repeat_rows)
    for repeat in repeat_rows:
        base = matrix.get(str(repeat["run_id"]))
        mismatches = ["missing_matrix_cell"] if not base else [key for key in keys if str(base.get(key)) != str(repeat.get(key))]
        comparisons.append({"run_id": repeat["run_id"], "matched": bool(base), "mismatches": mismatches, "passed": not mismatches})
        passed = passed and not mismatches
    return {"status": "pass" if passed else "fail", "checked_runs": len(comparisons), "comparisons": comparisons}


def _stage0_outputs(evidence_dir: Path) -> None:
    _write_json(
        evidence_dir / "lro3_stage0_code_path_audit.json",
        {
            "schema_version": "lro3_stage0_code_path_audit_v1",
            "multiplier_application_point": "runner_ticks_reproduction.py computes lro1_try_fields after spawn-space discovery and before the stochastic try-probability draw.",
            "random_draw_path": "self.np_rng.random() is compared to effective_try_prob only when effective_try_prob < 1.0; no extra draw is introduced for neutral flat cells with effective_try_prob == baseline try_prob == 1.0.",
            "affects": {
                "exposure": False,
                "eligibility_pre_spawn": False,
                "spawn_space_search": False,
                "attempt_offer": "Only schema-v4 no-attempt telemetry changes after the try-probability gate.",
                "attempt_probability": True,
                "birth_attempt_counter": "Indirectly, but only for cells where baseline try_prob was below 1.0 or candidate multiplier suppresses a would-be attempt before _birth_attempts_gen increments.",
                "birth_success_after_attempt": False,
            },
            "lineage_identity_in_mechanics": False,
            "flat_control_expected_neutral": True,
            "founder_balanced_and_legacy_share_mechanics": True,
            "wdl_uses_same_candidate_mechanics": True,
            "reporting_adequacy_gate": "Gate B/Gate D boundary: compact metrics are adequate for final outcomes, but scan-v4 sample is required to localize stochastic no-attempt suppression.",
        },
    )
    _write_json(
        evidence_dir / "lro3_stage0_metric_semantics.json",
        {
            "schema_version": "lro3_stage0_metric_semantics_v1",
            "birth_attempts": "Cumulative attempts after spawn-space, try-probability, buffer, and viability prechecks reach the attempt counter.",
            "reproduction_scan_no_attempt_rows": "Schema-v4 localization for candidates suppressed by the try-probability draw before birth_attempts increments.",
            "successful_births": "Children successfully added after child construction and env.add_agent.",
            "failed_births": "Post-attempt failures such as buffer, conception, gestation, make_child, or apply-time no-space.",
            "why_lro2_attempts_could_be_unchanged": "The LRO2 paired summary used compact birth_attempts. Candidate A is meant to suppress the try-probability path before the compact attempt counter, so unchanged attempts do not prove inactivity; scan-v4 no-attempt rows are the correct localizer.",
            "why_successful_births_can_drop_without_attempt_delta": "The candidate changes RNG schedule and no-attempt distribution in active contexts; compact final counters can show fewer successful births even when the post-gate attempt counter is unchanged.",
        },
    )


def _lro2_reanalysis(evidence_dir: Path) -> None:
    compact = _read_csv(LRO2_EVIDENCE / "lro2_compact_matrix.csv")
    scan = _read_csv(LRO2_EVIDENCE / "lro2_scan_v4_sample_summary.csv")
    paired = _read_csv(LRO2_EVIDENCE / "lro2_paired_effects.csv")
    by_scan = {(r["setting"], r["context"], r["init_arm"], _int(r["seed"])): r for r in scan}
    by_cell = []
    for row in compact:
        key = (row["setting"], row["context"], row["init_arm"], _int(row["seed"]))
        scan_row = by_scan.get(key, {})
        by_cell.append(
            {
                "setting": row["setting"],
                "context": row["context"],
                "init_arm": row["init_arm"],
                "seed": row["seed"],
                "alive": row.get("alive", ""),
                "birth_attempts": row.get("birth_attempts", ""),
                "successful_births": row.get("successful_births", ""),
                "failed_births": row.get("failed_births", ""),
                "final_effective_alive_lineages": row.get("final_effective_alive_lineages", ""),
                "final_top1_alive_share": row.get("final_top1_alive_share", ""),
                "scan_v4_sampled": bool(scan_row),
                "scan_exposure_rows": scan_row.get("reproduction_scan_exposure_rows", ""),
                "scan_no_attempt_rows": scan_row.get("reproduction_scan_no_attempt_rows", ""),
                "scan_eligible_rows": scan_row.get("reproduction_scan_eligible_rows", ""),
                "multiplier_min": scan_row.get("multiplier_min", ""),
                "multiplier_max": scan_row.get("multiplier_max", ""),
                "multiplier_mean": scan_row.get("multiplier_mean", ""),
                "disabled_leakage_rows": scan_row.get("lro1_disabled_leakage_rows", ""),
            }
        )
    by_context = []
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in paired:
        groups[(row["context"], row["init_arm"])].append(row)
    for (context, init_arm), rows in sorted(groups.items()):
        by_context.append(
            {
                "context": context,
                "init_arm": init_arm,
                "paired_cells": len(rows),
                "delta_birth_attempts_mean": statistics.fmean(_float(r["delta_birth_attempts"]) for r in rows),
                "delta_successful_births_mean": statistics.fmean(_float(r["delta_successful_births"]) for r in rows),
                "delta_failed_births_mean": statistics.fmean(_float(r["delta_failed_births"]) for r in rows),
                "delta_effective_alive_lineages_mean": statistics.fmean(_float(r["delta_final_effective_alive_lineages"]) for r in rows),
                "delta_top1_alive_share_mean": statistics.fmean(_float(r["delta_final_top1_alive_share"]) for r in rows),
            }
        )
    _write_csv(evidence_dir / "lro3_lro2_reanalysis_by_cell.csv", by_cell)
    _write_csv(evidence_dir / "lro3_lro2_reanalysis_by_context.csv", by_context)
    _write_json(
        evidence_dir / "lro3_lro2_funnel_localization.json",
        {
            "schema_version": "lro3_lro2_funnel_localization_v1",
            "answer_birth_attempts_unchanged": "LRO2 compact birth_attempts is a post-try-gate counter; Candidate A activity is visible in schema-v4 enabled rows, multipliers, and no-attempt localization rather than in compact attempt totals.",
            "mean_successful_birth_delta": statistics.fmean(_float(r["delta_successful_births"]) for r in paired),
            "worst_effective_delta_cells": sorted(
                paired,
                key=lambda row: _float(row["delta_final_effective_alive_lineages"]),
            )[:5],
            "scan_v4_sample_cells": len(scan),
            "disabled_leakage_rows": sum(_int(row.get("lro1_disabled_leakage_rows")) for row in scan),
            "enabled_rows": sum(_int(row.get("lro1_enabled_rows")) for row in scan),
            "interpretation": "Candidate A was active and measurable, but LRO2 did not support benefit; longer bounded comparison is required before selection or rejection.",
        },
    )
    _write_json(
        evidence_dir / "lro3_lro2_risk_assessment.json",
        {
            "schema_version": "lro3_lro2_risk_assessment_v1",
            "candidate_a_prior": "method_ready_but_not_scientifically_positive",
            "risks": [
                "global or near-global birth suppression without lineage-search improvement",
                "WDL founder-balanced seed-202 caveat broadening",
                "compact-only interpretation missing pre-attempt no-attempt localization",
                "large schema-v4 event output if sampled too broadly",
            ],
            "go_gate": "Proceed to Candidate A longer bounded comparison with compact main matrix, sampled schema-v4 traces, and repeat parity.",
        },
    )


def _candidate_a_decision(matrix_rows: list[dict[str, Any]], scan_rows: list[dict[str, Any]], repeat: dict[str, Any], compact_info: dict[str, Any], scan_info: dict[str, Any]) -> dict[str, Any]:
    candidate = "candidate_a_local_patch_carrying_capacity_v1"
    paired = _paired(matrix_rows, candidate)
    pressure = [row for row in paired if row["context"] != "flat_control"]
    wdl_founder = [row for row in paired if row["context"] == "wdl_delayed_hazard_full_stack" and row["init_arm"] == "founder_balanced"]
    integrity_failures = []
    if compact_info.get("status") != "complete" or compact_info.get("failures") != 0 or len(matrix_rows) != 80:
        integrity_failures.append("compact_matrix_incomplete")
    if scan_info.get("status") != "complete" or scan_info.get("failures") != 0 or len(scan_rows) != 8:
        integrity_failures.append("scan_v4_sample_incomplete")
    if repeat.get("status") != "pass":
        integrity_failures.append("repeat_parity_failed")
    if sum(_int(row.get("lro1_disabled_leakage_rows")) for row in scan_rows) != 0:
        integrity_failures.append("disabled_control_leakage")
    if not any(_int(row.get("lro1_enabled_rows")) > 0 for row in scan_rows):
        integrity_failures.append("enabled_lro_rows_missing")
    if any(_int(row.get("lro1_invalid_context_rows")) > 0 for row in scan_rows):
        integrity_failures.append("invalid_lro_context_rows")
    for row in scan_rows:
        if row["setting"] == candidate and row.get("multiplier_min") not in ("", None):
            if _float(row.get("multiplier_min")) < 0.35 or _float(row.get("multiplier_max")) > 1.0:
                integrity_failures.append("multiplier_bounds")
    flat_pairs = [row for row in paired if row["context"] == "flat_control"]
    flat_neutral = all(abs(_float(row.get("delta_successful_births"))) < 1e-12 and abs(_float(row.get("delta_birth_attempts"))) < 1e-12 for row in flat_pairs)
    if not flat_neutral:
        integrity_failures.append("flat_control_not_neutral")
    pressure_summary = _summary_stats(
        pressure,
        [
            "delta_successful_births",
            "delta_birth_attempts",
            "delta_final_effective_post_refill_alive_lineages",
            "delta_final_top1_post_refill_alive_share",
            "delta_wdl_delayed_hazard_event_count",
        ],
    )
    wdl_summary = _summary_stats(
        wdl_founder,
        [
            "delta_successful_births",
            "delta_final_effective_post_refill_alive_lineages",
            "delta_final_top1_post_refill_alive_share",
            "delta_wdl_delayed_hazard_event_count",
        ],
    )
    warnings = [row for row in matrix_rows if _int(row.get("top1_dominance_generation"), -1) >= 0]
    collapses = [row for row in matrix_rows if _int(row.get("effective_lineage_collapse_generation"), -1) >= 0]
    candidate_collapses = [row for row in collapses if row["setting"] == candidate]
    scientific_blockers = []
    if candidate_collapses:
        scientific_blockers.append("candidate_effective_lineage_collapse")
    severe_birth_suppression = pressure_summary["delta_successful_births_mean"] < -8.0
    effective_degrades = pressure_summary["delta_final_effective_post_refill_alive_lineages_mean"] < -0.15
    top1_worsens = pressure_summary["delta_final_top1_post_refill_alive_share_mean"] > 0.03
    if integrity_failures or scientific_blockers:
        code = "LRO3-A4"
        if integrity_failures:
            interpretation = "Candidate A cannot be interpreted because required integrity gates failed."
        else:
            interpretation = "Candidate A fails closed because candidate-specific effective-lineage collapse appeared in the longer comparison; Candidate B fallback is not run from a fail-closed A4 branch."
    elif severe_birth_suppression or (effective_degrades and top1_worsens):
        code = "LRO3-A3"
        interpretation = "Candidate A is harmful or broadly worsens search-quality indicators; reject and pivot to Candidate B."
    elif effective_degrades or top1_worsens or abs(pressure_summary["delta_final_effective_post_refill_alive_lineages_mean"]) < 0.10:
        code = "LRO3-A2"
        interpretation = "Candidate A is safe but neutral/inconclusive; do not adopt, pivot to Candidate B."
    else:
        code = "LRO3-A1"
        interpretation = "Candidate A shows bounded improvement and qualifies for robustness continuation."
    return {
        "schema_version": "lro3_candidate_a_decision_v1",
        "decision": code,
        "interpretation": interpretation,
        "integrity_failures": sorted(set(integrity_failures)),
        "scientific_blockers": sorted(set(scientific_blockers)),
        "flat_control_neutral": flat_neutral,
        "pressure_summary": pressure_summary,
        "wdl_founder_balanced_summary": wdl_summary,
        "top1_warning_cells": [row["run_id"] for row in warnings],
        "effective_collapse_cells": [row["run_id"] for row in collapses],
        "candidate_effective_collapse_cells": [row["run_id"] for row in candidate_collapses],
        "severe_birth_suppression": severe_birth_suppression,
        "effective_degrades": effective_degrades,
        "top1_worsens": top1_worsens,
        "paired_cells": len(paired),
        "no_claims": DECISION_NO_CLAIMS,
    }


def _markdown(summary: dict[str, Any]) -> str:
    a = summary.get("candidate_a_decision", {"decision": "not_run"})
    top = summary.get("consolidated_decision", {"decision": "LRO-3-D"})
    lines = [
        "# EIEL LRO-3 Autonomous Lineage-Blind Repair Comparison",
        "",
        f"Final top-level decision: **{top['decision']}**.",
        f"Candidate A decision: **{a.get('decision', 'not_run')}**.",
        "",
        "## Purpose",
        "",
        "LRO-3 tests whether the LRO-2 default-off local reproductive opportunity mechanism actually improves bounded lineage-search indicators over a longer comparison, and whether a fallback is needed.",
        "",
        "## Prior state",
        "",
        "LRO-2 reached method readiness but not benefit: the tiny paired matrix had unchanged compact birth attempts, fewer successful births on average, and slightly lower effective alive lineages.",
        "",
        "## Stage 0 audit",
        "",
        "Candidate A is applied after spawn-space discovery and before the stochastic try-probability draw. Compact `birth_attempts` is a post-gate counter; schema-v4 scan/no-attempt telemetry is required to localize pre-attempt suppression.",
        "",
        "## LRO-2 reanalysis",
        "",
        "The LRO-2 evidence supports active/default-off measurement, not scientific success. Candidate A activity is visible in sampled enabled rows and bounded multipliers; disabled leakage stayed zero.",
        "",
        "## Reporting and integration gate",
        "",
        "Gate B/D: implementation is active and lineage-blind, with a narrow interpretation constraint. Compact matrix outcomes are adequate for final direction; scan-v4 sample is required for mechanism localization.",
        "",
        "## Candidate A comparison",
        "",
    ]
    compact = summary.get("campaigns", {}).get("candidate_a_compact", {})
    scan = summary.get("campaigns", {}).get("candidate_a_scan_v4_sample", {})
    repeat = summary.get("candidate_a_repeat_parity", {})
    lines.extend(
        [
            f"- Compact matrix: `{compact.get('status', 'not_run')}`, planned `{compact.get('planned_runs', '')}`, failures `{compact.get('failures', '')}`, root `{compact.get('runs_root', '')}`.",
            f"- Scan-v4 sample: `{scan.get('status', 'not_run')}`, planned `{scan.get('planned_runs', '')}`, failures `{scan.get('failures', '')}`, root `{scan.get('runs_root', '')}`.",
            f"- Repeat parity: `{repeat.get('status', 'not_run')}`, checked `{repeat.get('checked_runs', '')}`.",
            "",
            "## Candidate A decision",
            "",
            f"{a.get('interpretation', 'Candidate A has not yet been evaluated.')}",
            "",
        ]
    )
    if a.get("pressure_summary"):
        ps = a["pressure_summary"]
        lines.extend(
            [
                f"- Pressure-context mean delta successful births: `{ps.get('delta_successful_births_mean', 0.0):.3f}`.",
                f"- Pressure-context mean delta effective post-refill lineages: `{ps.get('delta_final_effective_post_refill_alive_lineages_mean', 0.0):.6f}`.",
                f"- Pressure-context mean delta top1 post-refill share: `{ps.get('delta_final_top1_post_refill_alive_share_mean', 0.0):.6f}`.",
                "",
            ]
        )
    if a.get("scientific_blockers") or a.get("integrity_failures"):
        lines.extend(
            [
                f"- Integrity failures: `{', '.join(a.get('integrity_failures') or []) or 'none'}`.",
                f"- Scientific blockers: `{', '.join(a.get('scientific_blockers') or []) or 'none'}`.",
                f"- Candidate effective-collapse cells: `{len(a.get('candidate_effective_collapse_cells') or [])}`.",
                "",
            ]
        )
    lines.extend(
        [
            "## Candidate B implementation",
            "",
            summary.get("candidate_b_status", "Skipped unless Candidate A is safely negative, neutral, or loses robustness."),
            "",
            "## Consolidated decision",
            "",
            top.get("interpretation", ""),
            "",
            "## Win/Loss",
            "",
            summary.get("win_loss", "Pending final branch completion."),
            "",
            "## Simple explanation",
            "",
            summary.get("simple_explanation", "I tested whether a local ecological reproduction gate helps lineage-search conditions before considering a fallback. This is pending final branch completion."),
            "",
            "## Known limitations",
            "",
            "- This is bounded lineage-search telemetry, not an open-endedness result.",
            "- Compact cells do not retain full schema-v4 event streams; sampled scan-v4 cells are used for mechanism localization.",
            "- Candidate decisions are limited to the configured horizons, seeds, contexts, and default-off knobs.",
            "",
            "## No-claims statement",
            "",
            DECISION_NO_CLAIMS,
            "",
            "## Next step",
            "",
            summary.get("recommended_next_step", "Complete the next safe LRO-3 branch."),
            "",
        ]
    )
    return "\n".join(lines)


def _write_package(summary: dict[str, Any], evidence_dir: Path, report_path: Path, registry_paths: list[Path]) -> Path:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    markdown = _markdown(summary)
    report_path.write_text(markdown, encoding="utf-8")
    (evidence_dir / report_path.name).write_text(markdown, encoding="utf-8")
    zip_path = evidence_dir / "08-lro3-autonomous-comparison.evidence.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        added: set[str] = set()
        for path in sorted(evidence_dir.iterdir()):
            if path == zip_path or not path.is_file() or path.suffix == ".zip":
                continue
            zf.write(path, arcname=path.name)
            added.add(path.name)
        for registry in registry_paths:
            if registry.exists() and registry.name not in added:
                zf.write(registry, arcname=registry.name)
                added.add(registry.name)
        if report_path.exists() and report_path.name not in added:
            zf.write(report_path, arcname=report_path.name)
    return zip_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Build LRO-3 autonomous comparison evidence/report package.")
    parser.add_argument("--evidence-dir", default=str(DEFAULT_EVIDENCE))
    parser.add_argument("--report-path", default=str(DEFAULT_REPORT))
    parser.add_argument("--candidate-a-compact-root", default="")
    parser.add_argument("--candidate-a-scan-root", default="")
    parser.add_argument("--candidate-a-repeat-root", default="")
    parser.add_argument("--candidate-a-registry", default="_spec/registry/v1/lro3_candidate_a_longer_bounded_comparison.yaml")
    args = parser.parse_args()

    evidence_dir = Path(args.evidence_dir).resolve()
    report_path = Path(args.report_path).resolve()
    evidence_dir.mkdir(parents=True, exist_ok=True)
    _stage0_outputs(evidence_dir)
    _lro2_reanalysis(evidence_dir)

    compact_root = Path(args.candidate_a_compact_root).resolve() if args.candidate_a_compact_root else None
    scan_root = Path(args.candidate_a_scan_root).resolve() if args.candidate_a_scan_root else None
    repeat_root = Path(args.candidate_a_repeat_root).resolve() if args.candidate_a_repeat_root else None
    compact_rows = _run_rows(compact_root)
    scan_rows = _scan_rows(scan_root)
    repeat_rows = _run_rows(repeat_root)
    repeat = _repeat_parity(compact_rows, repeat_rows) if repeat_rows else {"status": "not_run", "checked_runs": 0}
    compact_info = _campaign_info(compact_root)
    scan_info = _campaign_info(scan_root)
    repeat_info = _campaign_info(repeat_root)
    candidate_a_decision = (
        _candidate_a_decision(compact_rows, scan_rows, repeat, compact_info, scan_info)
        if compact_rows or scan_rows or repeat_rows
        else {"decision": "not_run", "interpretation": "Candidate A longer comparison has not been run yet."}
    )
    paired = _paired(compact_rows, "candidate_a_local_patch_carrying_capacity_v1")
    if compact_rows:
        _write_csv(evidence_dir / "lro3_candidate_a_matrix.csv", compact_rows)
    if paired:
        _write_csv(evidence_dir / "lro3_candidate_a_paired_effects.csv", paired)
    if scan_rows:
        _write_csv(evidence_dir / "lro3_candidate_a_scan_v4_summary.csv", scan_rows)
    _write_json(evidence_dir / "lro3_candidate_a_repeat_parity.json", repeat)
    _write_json(evidence_dir / "lro3_candidate_a_decision.json", candidate_a_decision)
    _write_json(
        evidence_dir / "lro3_candidate_a_gate_summary.json",
        {
            "schema_version": "lro3_candidate_a_gate_summary_v1",
            "compact": compact_info,
            "scan_v4_sample": scan_info,
            "repeat": repeat_info,
            "reporting_gate": "implementation_active_measurable_with_scan_v4_required_for_no_attempt_localization",
            "candidate_a_decision": candidate_a_decision,
        },
    )
    top_decision = {
        "decision": "LRO-3-D",
        "interpretation": "LRO-3 is not complete until Candidate A is evaluated and Candidate B fallback is either executed or explicitly skipped by rule.",
    }
    win_loss = "Pending. Candidate A comparison evidence is incomplete."
    simple = (
        "I am testing whether the local reproductive opportunity gate helps lineage-search conditions before "
        "falling back to offspring dispersal. The task is not complete until the executed branch has a decision."
    )
    next_step = "Run or finish the next LRO-3 branch."
    if candidate_a_decision.get("decision") == "LRO3-A1":
        top_decision = {
            "decision": "LRO-3-A",
            "interpretation": "Candidate A passed the longer bounded comparison under the current decision rule; Candidate B fallback was not run.",
        }
        win_loss = "Win for Candidate A bounded comparison; no broad open-endedness claim is made."
        simple = (
            "I tested whether the local reproduction-opportunity gate helped keep lineage search healthier. "
            "It fit the EIEL goal by changing local opportunity without lineage protection. Under this rule it "
            "improved the bounded indicators enough to continue with Candidate A robustness in a future slice."
        )
        next_step = "Run bounded Candidate A robustness only; keep the feature default-off."
    elif candidate_a_decision.get("decision") in {"LRO3-A2", "LRO3-A3"}:
        top_decision = {
            "decision": "LRO-3-D",
            "interpretation": "Candidate A is safe negative/neutral; Candidate B fallback is required before top-level closure.",
        }
        win_loss = "Loss for Candidate A selection; fallback branch is required."
        simple = (
            "I tested whether reducing reproduction probability in crowded or poor places improved lineage-search "
            "conditions. It did not pass the Candidate A rule, so the next branch should try the different idea: "
            "bounded offspring dispersal instead of more probability tuning."
        )
        next_step = "Implement and tiny-validate default-off Candidate B offspring dispersal fallback."
    elif candidate_a_decision.get("decision") == "LRO3-A4":
        top_decision = {
            "decision": "LRO-3-D",
            "interpretation": "Candidate A hit a fail-closed scientific/integrity branch; Candidate B fallback is deferred until the collapse evidence is reviewed or the gate is redesigned.",
        }
        win_loss = "Loss for Candidate A selection; fail-closed branch, so no immediate Candidate B fallback was run."
        simple = (
            "I tested whether reducing reproduction probability in crowded or poor places helped keep multiple "
            "lineage paths active. It did not: one tuned-Jungle seed collapsed under Candidate A in both setup "
            "arms while the matching baselines stayed healthier. Because that is a fail-closed result, I stopped "
            "instead of adding a second new mechanism on top of a confusing failure."
        )
        next_step = "Review the Candidate A collapse cells and decide whether to redesign the comparison gate before implementing Candidate B."
    summary = {
        "schema_version": "lro3_autonomous_comparison_summary_v1",
        "campaigns": {
            "candidate_a_compact": compact_info,
            "candidate_a_scan_v4_sample": scan_info,
            "candidate_a_repeat": repeat_info,
        },
        "candidate_a_decision": candidate_a_decision,
        "candidate_a_repeat_parity": repeat,
        "candidate_b_status": "Not executed in this package build.",
        "consolidated_decision": top_decision,
        "win_loss": win_loss,
        "simple_explanation": simple,
        "recommended_next_step": next_step,
        "no_claims": DECISION_NO_CLAIMS,
    }
    _write_json(evidence_dir / "lro3_consolidated_decision.json", summary)
    _write_json(
        evidence_dir / "lro3_runtime_archive_manifest.json",
        {
            "schema_version": "lro3_runtime_archive_manifest_v1",
            "candidate_a_compact_root": str(compact_root) if compact_root else "",
            "candidate_a_scan_root": str(scan_root) if scan_root else "",
            "candidate_a_repeat_root": str(repeat_root) if repeat_root else "",
            "archive_paths": [],
            "note": "Archive paths are filled during closeout after runtime cleanup.",
        },
    )
    zip_path = _write_package(summary, evidence_dir, report_path, [Path(args.candidate_a_registry)])
    print(
        json.dumps(
            {
                "summary": str(evidence_dir / "lro3_consolidated_decision.json"),
                "report": str(report_path),
                "evidence_dir": str(evidence_dir),
                "zip": str(zip_path),
                "zip_sha256": _sha256(zip_path),
                "candidate_a_decision": candidate_a_decision.get("decision"),
                "top_level_decision": top_decision.get("decision"),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
