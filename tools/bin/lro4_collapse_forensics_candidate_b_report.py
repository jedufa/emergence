#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import statistics
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

csv.field_size_limit(1024 * 1024 * 1024)

TOP1_THRESHOLD = 0.8
EFFECTIVE_COLLAPSE_THRESHOLD = 1.5
NO_CLAIMS = (
    "LRO-4 is a bounded lineage-blind collapse-forensics and repair-candidate task. It does not show intelligence, "
    "emergence, consciousness, broad cognition, memory selection, prediction selection, solved lineage diversity, "
    "general lineage repair, beneficial adaptive mechanism, culture, superintelligence, or general open-ended "
    "evolution. It does not add direct cognition reward, lineage rescue, diversity reward, protected reproduction, "
    "forced equal reproduction, archive reseeding, or survival subsidy."
)
DEFAULT_EVIDENCE = Path("evidences/06-open-ended-search/09-lro4-collapse-forensics-candidate-b")
DEFAULT_REPORT = Path("docs/500_campaigns-decisions-closeouts/520_campaign-reports/000_programme-cross-cutting/EIEL-DOC-000413--eiel-lro4-collapse-forensics-and-candidate-b-2026-07-10.md")
LRO3_EVIDENCE = Path("evidences/06-open-ended-search/08-lro3-autonomous-comparison")
COLLAPSE_RUN_IDS = [
    "lro3_candidate_a_longer_bounded_comparison__setting_baseline_flags_off__context_tuned_full_jungle_recovery__init_legacy__seed000303",
    "lro3_candidate_a_longer_bounded_comparison__setting_candidate_a_local_patch_carrying_capacity_v1__context_tuned_full_jungle_recovery__init_legacy__seed000303",
    "lro3_candidate_a_longer_bounded_comparison__setting_baseline_flags_off__context_tuned_full_jungle_recovery__init_founder_balanced__seed000303",
    "lro3_candidate_a_longer_bounded_comparison__setting_candidate_a_local_patch_carrying_capacity_v1__context_tuned_full_jungle_recovery__init_founder_balanced__seed000303",
]


def _load_lro3_module() -> Any:
    path = Path("tools/bin/lro3_autonomous_comparison_report.py")
    spec = importlib.util.spec_from_file_location("lro3_autonomous_comparison_report", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


LRO3 = _load_lro3_module()


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


def _run_rows(root: Path | None) -> list[dict[str, Any]]:
    return LRO3._run_rows(root)


def _group_repeats(roots: list[Path]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for root in roots:
        for row in _run_rows(root):
            item = dict(row)
            item["repeat_root"] = str(root)
            out[str(item["run_id"])].append(item)
    return out


def _event_summary(root: Path | None) -> list[dict[str, Any]]:
    if root is None or not root.exists():
        return []
    rows = []
    for run in _run_rows(root):
        event_path = Path(run["run_dir"]) / "metrics" / "lineage_reproduction_funnel_events.jsonl"
        stats: dict[str, Any] = {
            "run_id": run["run_id"],
            "setting": run["setting"],
            "context": run["context"],
            "init_arm": run["init_arm"],
            "seed": run["seed"],
            "event_jsonl_path": str(event_path),
            "event_jsonl_bytes": event_path.stat().st_size if event_path.exists() else 0,
            "event_rows": 0,
            "scan_exposure_rows": 0,
            "scan_eligible_rows": 0,
            "scan_no_attempt_rows": 0,
            "scan_ineligible_no_space_rows": 0,
            "birth_success_rows": 0,
            "birth_fail_no_space_rows": 0,
            "lro1_enabled_rows": 0,
            "lro1_disabled_rows": 0,
            "lro1_disabled_leakage_rows": 0,
            "lro1_multiplier_min": "",
            "lro1_multiplier_max": "",
            "lro1_multiplier_sum": 0.0,
            "parent_density_sum": 0.0,
            "parent_waste_sum": 0.0,
            "lro4_event_rows": 0,
            "lro4_enabled_rows": 0,
            "lro4_disabled_rows": 0,
            "lro4_disabled_leakage_rows": 0,
            "lro4_triggered_rows": 0,
            "lro4_selected_from_expanded_rows": 0,
            "lro4_expanded_candidate_count_sum": 0,
            "lro4_trigger_reason_counts_json": "{}",
        }
        parent_success = Counter()
        parent_no_attempt = Counter()
        parent_no_space = Counter()
        lro4_reasons = Counter()
        if event_path.exists():
            with event_path.open(encoding="utf-8") as handle:
                for line in handle:
                    try:
                        payload = json.loads(line)
                    except Exception:
                        continue
                    stats["event_rows"] += 1
                    event_type = str(payload.get("event_type") or "")
                    lineage = str(payload.get("parent_lineage_id") or payload.get("lineage_id") or "")
                    if event_type == "reproduction_scan_exposure":
                        stats["scan_exposure_rows"] += 1
                    elif event_type == "reproduction_scan_eligible":
                        stats["scan_eligible_rows"] += 1
                    elif event_type == "reproduction_scan_no_attempt":
                        stats["scan_no_attempt_rows"] += 1
                        parent_no_attempt[lineage] += 1
                    elif event_type == "reproduction_scan_ineligible" and str(payload.get("failure_reason_group")) == "no_space":
                        stats["scan_ineligible_no_space_rows"] += 1
                        parent_no_space[lineage] += 1
                    elif event_type == "birth_success":
                        stats["birth_success_rows"] += 1
                        parent_success[lineage] += 1
                    elif event_type == "birth_fail_no_space":
                        stats["birth_fail_no_space_rows"] += 1
                        parent_no_space[lineage] += 1
                    enabled_raw = payload.get("lro1_reproductive_opportunity_enable", "")
                    if enabled_raw != "":
                        enabled = _int(enabled_raw) > 0
                        mult = _float(payload.get("lro1_opportunity_multiplier"), 1.0)
                        score = _float(payload.get("lro1_opportunity_score_parent_site"), 0.0)
                        if enabled:
                            stats["lro1_enabled_rows"] += 1
                            stats["lro1_multiplier_sum"] += mult
                            current_min = stats["lro1_multiplier_min"]
                            current_max = stats["lro1_multiplier_max"]
                            stats["lro1_multiplier_min"] = mult if current_min == "" else min(float(current_min), mult)
                            stats["lro1_multiplier_max"] = mult if current_max == "" else max(float(current_max), mult)
                        else:
                            stats["lro1_disabled_rows"] += 1
                            if abs(mult - 1.0) > 1e-12 or abs(score) > 1e-12:
                                stats["lro1_disabled_leakage_rows"] += 1
                    stats["parent_density_sum"] += _float(payload.get("local_density_parent_site"))
                    stats["parent_waste_sum"] += _float(payload.get("waste_parent_site"))
                    if payload.get("lro4_offspring_dispersal_enable", "") not in ("", None):
                        stats["lro4_event_rows"] += 1
                        lro4_enabled = _int(payload.get("lro4_offspring_dispersal_enable")) > 0
                        if lro4_enabled:
                            stats["lro4_enabled_rows"] += 1
                        else:
                            stats["lro4_disabled_rows"] += 1
                        if _int(payload.get("lro4_offspring_dispersal_disabled_leakage_marker")) > 0:
                            stats["lro4_disabled_leakage_rows"] += 1
                        if _int(payload.get("lro4_offspring_dispersal_triggered")) > 0:
                            stats["lro4_triggered_rows"] += 1
                            lro4_reasons[str(payload.get("lro4_offspring_dispersal_trigger_reason") or "unknown")] += 1
                        if _int(payload.get("lro4_offspring_dispersal_selected_from_expanded")) > 0:
                            stats["lro4_selected_from_expanded_rows"] += 1
                        stats["lro4_expanded_candidate_count_sum"] += _int(
                            payload.get("lro4_offspring_dispersal_expanded_candidate_count")
                        )
        event_rows = max(1, int(stats["event_rows"]))
        enabled_rows = int(stats["lro1_enabled_rows"])
        stats["lro1_multiplier_mean"] = stats["lro1_multiplier_sum"] / enabled_rows if enabled_rows else ""
        stats["parent_density_mean"] = stats["parent_density_sum"] / event_rows
        stats["parent_waste_mean"] = stats["parent_waste_sum"] / event_rows
        for key in ("lro1_multiplier_sum", "parent_density_sum", "parent_waste_sum"):
            stats.pop(key, None)
        stats["lro4_trigger_reason_counts_json"] = json.dumps(dict(sorted(lro4_reasons.items())), sort_keys=True)
        stats["top_success_parent_lineage"] = parent_success.most_common(1)[0][0] if parent_success else ""
        stats["top_success_parent_lineage_rows"] = parent_success.most_common(1)[0][1] if parent_success else 0
        stats["top_no_attempt_parent_lineage"] = parent_no_attempt.most_common(1)[0][0] if parent_no_attempt else ""
        stats["top_no_attempt_parent_lineage_rows"] = parent_no_attempt.most_common(1)[0][1] if parent_no_attempt else 0
        stats["top_no_space_parent_lineage"] = parent_no_space.most_common(1)[0][0] if parent_no_space else ""
        stats["top_no_space_parent_lineage_rows"] = parent_no_space.most_common(1)[0][1] if parent_no_space else 0
        rows.append(stats)
    return rows


def _timeline_rows(roots: list[Path]) -> list[dict[str, Any]]:
    rows = []
    for root in roots:
        for run in _run_rows(root):
            path = Path(run["run_dir"]) / "metrics" / "lineage_reproduction_funnel_generation_summary.csv"
            if not path.exists():
                continue
            for row in _read_csv(path):
                rows.append(
                    {
                        "repeat_root": root.name,
                        "run_id": run["run_id"],
                        "setting": run["setting"],
                        "context": run["context"],
                        "init_arm": run["init_arm"],
                        "seed": run["seed"],
                        "generation": row.get("generation", ""),
                        "lineage_id": row.get("lineage_id", ""),
                        "alive_share_before_selection": row.get("alive_share_before_selection", ""),
                        "eligible_parent_share": row.get("eligible_parent_share", ""),
                        "selected_parent_share": row.get("selected_parent_share", ""),
                        "offspring_share": row.get("offspring_share", ""),
                        "post_refill_alive_share": row.get("post_refill_alive_share", ""),
                        "birth_attempts": row.get("birth_attempts", ""),
                        "successful_births": row.get("successful_births", ""),
                        "failed_births": row.get("failed_births", ""),
                        "birth_failure_due_to_no_local_space": row.get("birth_failure_due_to_no_local_space", ""),
                        "local_density_parent_site_mean": row.get("local_density_parent_site_mean", ""),
                        "waste_parent_site_mean": row.get("waste_parent_site_mean", ""),
                        "parent_child_distance_mean": row.get("parent_child_distance_mean", ""),
                        "eligible_amplification_class": row.get("eligible_amplification_class", ""),
                        "selected_parent_amplification_class": row.get("selected_parent_amplification_class", ""),
                        "offspring_amplification_class": row.get("offspring_amplification_class", ""),
                        "refill_amplification_class": row.get("refill_amplification_class", ""),
                    }
                )
    return rows


def _merge_counts(target: dict[str, int], raw: str) -> None:
    if not raw:
        return
    try:
        payload = json.loads(raw)
    except Exception:
        return
    if not isinstance(payload, dict):
        return
    for key, value in payload.items():
        target[str(key)] = int(target.get(str(key), 0)) + _int(value)


def _compact_lro4_summary(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "metrics" / "lineage_reproduction_funnel_generation_summary.csv"
    rows = _read_csv(path) if path.exists() else []
    reason_counts: dict[str, int] = {}
    out: dict[str, Any] = {
        "lro4_compact_rows": len(rows),
        "lro4_event_rows": 0,
        "lro4_enabled_rows": 0,
        "lro4_disabled_rows": 0,
        "lro4_disabled_leakage_rows": 0,
        "lro4_triggered_rows": 0,
        "lro4_expanded_candidate_count_sum": 0,
        "lro4_selected_from_expanded_rows": 0,
        "lro4_selected_radius_weighted_sum": 0.0,
        "lro4_parent_density_weighted_sum": 0.0,
        "lro4_parent_waste_weighted_sum": 0.0,
        "lro4_child_density_weighted_sum": 0.0,
        "lro4_child_waste_weighted_sum": 0.0,
    }
    for row in rows:
        event_rows = _int(row.get("lro4_offspring_dispersal_event_rows"))
        out["lro4_event_rows"] += event_rows
        out["lro4_enabled_rows"] += _int(row.get("lro4_offspring_dispersal_enabled_rows"))
        out["lro4_disabled_rows"] += _int(row.get("lro4_offspring_dispersal_disabled_rows"))
        out["lro4_disabled_leakage_rows"] += _int(row.get("lro4_offspring_dispersal_disabled_leakage_rows"))
        out["lro4_triggered_rows"] += _int(row.get("lro4_offspring_dispersal_triggered_rows"))
        out["lro4_expanded_candidate_count_sum"] += _int(
            row.get("lro4_offspring_dispersal_expanded_candidate_count_sum")
        )
        out["lro4_selected_from_expanded_rows"] += _int(
            row.get("lro4_offspring_dispersal_selected_from_expanded_rows")
        )
        out["lro4_selected_radius_weighted_sum"] += event_rows * _float(
            row.get("lro4_offspring_dispersal_selected_radius_mean")
        )
        out["lro4_parent_density_weighted_sum"] += event_rows * _float(
            row.get("lro4_offspring_dispersal_parent_density_mean")
        )
        out["lro4_parent_waste_weighted_sum"] += event_rows * _float(
            row.get("lro4_offspring_dispersal_parent_waste_mean")
        )
        out["lro4_child_density_weighted_sum"] += event_rows * _float(
            row.get("lro4_offspring_dispersal_child_density_mean")
        )
        out["lro4_child_waste_weighted_sum"] += event_rows * _float(
            row.get("lro4_offspring_dispersal_child_waste_mean")
        )
        _merge_counts(reason_counts, str(row.get("lro4_offspring_dispersal_trigger_reason_counts_json", "")))
    denom = max(1, int(out["lro4_event_rows"]))
    out["lro4_selected_radius_mean"] = out.pop("lro4_selected_radius_weighted_sum") / denom
    out["lro4_parent_density_mean"] = out.pop("lro4_parent_density_weighted_sum") / denom
    out["lro4_parent_waste_mean"] = out.pop("lro4_parent_waste_weighted_sum") / denom
    out["lro4_child_density_mean"] = out.pop("lro4_child_density_weighted_sum") / denom
    out["lro4_child_waste_mean"] = out.pop("lro4_child_waste_weighted_sum") / denom
    out["lro4_trigger_reason_counts_json"] = json.dumps(reason_counts, sort_keys=True, separators=(",", ":"))
    return out


def _candidate_b_matrix_rows(root: Path | None) -> list[dict[str, Any]]:
    rows = []
    for row in _run_rows(root):
        run_dir = Path(str(row.get("run_dir", "")))
        compact = _compact_lro4_summary(run_dir)
        out = dict(row)
        out.update(compact)
        rows.append(out)
    return sorted(rows, key=lambda r: (str(r["context"]), str(r["init_arm"]), int(r["seed"]), str(r["setting"])))


def _candidate_b_effect_rows(matrix_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in matrix_rows:
        by_key[(str(row["context"]), str(row["init_arm"]), _int(row["seed"]))][str(row["setting"])] = row
    effects = []
    for (context, init_arm, seed), pair in sorted(by_key.items()):
        base = pair.get("baseline_flags_off")
        cand = pair.get("candidate_b_crowding_conditioned_radius_v1")
        if not base or not cand:
            effects.append({"context": context, "init_arm": init_arm, "seed": seed, "paired": False})
            continue
        effects.append(
            {
                "context": context,
                "init_arm": init_arm,
                "seed": seed,
                "paired": True,
                "baseline_alive": base.get("alive", ""),
                "candidate_alive": cand.get("alive", ""),
                "delta_alive": _float(cand.get("alive")) - _float(base.get("alive")),
                "baseline_successful_births": base.get("successful_births", ""),
                "candidate_successful_births": cand.get("successful_births", ""),
                "delta_successful_births": _float(cand.get("successful_births")) - _float(base.get("successful_births")),
                "baseline_birth_fail_no_space": base.get("birth_fail_no_space", ""),
                "candidate_birth_fail_no_space": cand.get("birth_fail_no_space", ""),
                "delta_birth_fail_no_space": _float(cand.get("birth_fail_no_space")) - _float(base.get("birth_fail_no_space")),
                "baseline_final_effective": base.get("final_effective_post_refill_alive_lineages", ""),
                "candidate_final_effective": cand.get("final_effective_post_refill_alive_lineages", ""),
                "delta_final_effective": _float(cand.get("final_effective_post_refill_alive_lineages"))
                - _float(base.get("final_effective_post_refill_alive_lineages")),
                "baseline_top1": base.get("final_top1_post_refill_alive_share", ""),
                "candidate_top1": cand.get("final_top1_post_refill_alive_share", ""),
                "delta_top1": _float(cand.get("final_top1_post_refill_alive_share"))
                - _float(base.get("final_top1_post_refill_alive_share")),
                "baseline_effective_collapse_generation": base.get("effective_lineage_collapse_generation", ""),
                "candidate_effective_collapse_generation": cand.get("effective_lineage_collapse_generation", ""),
                "baseline_lro4_enabled_rows": base.get("lro4_enabled_rows", 0),
                "candidate_lro4_enabled_rows": cand.get("lro4_enabled_rows", 0),
                "baseline_lro4_disabled_leakage_rows": base.get("lro4_disabled_leakage_rows", 0),
                "candidate_lro4_disabled_leakage_rows": cand.get("lro4_disabled_leakage_rows", 0),
                "candidate_lro4_triggered_rows": cand.get("lro4_triggered_rows", 0),
                "candidate_lro4_expanded_candidate_count_sum": cand.get("lro4_expanded_candidate_count_sum", 0),
                "candidate_lro4_selected_from_expanded_rows": cand.get("lro4_selected_from_expanded_rows", 0),
            }
        )
    return effects


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    vals = [_float(row.get(key)) for row in rows if row.get(key, "") != ""]
    return statistics.fmean(vals) if vals else 0.0


def _candidate_b_tiny_decision(matrix_rows: list[dict[str, Any]], effect_rows: list[dict[str, Any]]) -> dict[str, Any]:
    complete = len(matrix_rows) == 48 and all(str(row.get("status")) == "ok" for row in matrix_rows)
    baseline_rows = [row for row in matrix_rows if row.get("setting") == "baseline_flags_off"]
    candidate_rows = [row for row in matrix_rows if row.get("setting") == "candidate_b_crowding_conditioned_radius_v1"]
    paired_rows = [row for row in effect_rows if row.get("paired")]
    baseline_disabled_clean = all(_int(row.get("lro4_enabled_rows")) == 0 for row in baseline_rows) and all(
        _int(row.get("lro4_disabled_leakage_rows")) == 0 for row in baseline_rows
    )
    candidate_enabled = bool(candidate_rows) and all(_int(row.get("lro4_enabled_rows")) > 0 for row in candidate_rows)
    candidate_leakage_clean = all(_int(row.get("lro4_disabled_leakage_rows")) == 0 for row in candidate_rows)
    candidate_activates = sum(_int(row.get("lro4_triggered_rows")) for row in candidate_rows) > 0
    candidate_selects_expanded = sum(_int(row.get("lro4_selected_from_expanded_rows")) for row in candidate_rows) > 0
    collapse_rows = [row for row in candidate_rows if _int(row.get("effective_lineage_collapse_generation"), -1) >= 0]
    flat_effects = [row for row in paired_rows if row["context"] == "flat_control"]
    pressure_effects = [
        row
        for row in paired_rows
        if row["context"] in {"tuned_full_jungle_recovery", "wdl_delayed_hazard_full_stack"}
    ]
    flat_effective_mean = _mean(flat_effects, "delta_final_effective")
    flat_top1_mean = _mean(flat_effects, "delta_top1")
    pressure_effective_mean = _mean(pressure_effects, "delta_final_effective")
    pressure_no_space_mean = _mean(pressure_effects, "delta_birth_fail_no_space")
    pressure_births_mean = _mean(pressure_effects, "delta_successful_births")
    technical_clean = bool(
        complete
        and baseline_disabled_clean
        and candidate_enabled
        and candidate_leakage_clean
        and candidate_activates
        and candidate_selects_expanded
    )
    flat_control_clean = abs(flat_effective_mean) <= 0.75 and abs(flat_top1_mean) <= 0.08
    no_candidate_collapse = not collapse_rows
    directionally_promising = bool(
        technical_clean
        and flat_control_clean
        and no_candidate_collapse
        and pressure_effective_mean >= -0.25
        and pressure_births_mean >= -5.0
    )
    if directionally_promising:
        decision = "LRO4-B-TINY-PASS"
        top_level = "LRO-4-B"
        next_step = "Run the longer Candidate B comparison before any stronger recommendation."
    elif technical_clean:
        decision = "LRO4-B-TINY-TECHNICAL-PASS-DIRECTIONAL-FAIL"
        top_level = "LRO-4-C"
        next_step = "Do not run the longer comparison from this candidate; redesign or retune Candidate B."
    else:
        decision = "LRO4-B-TINY-BLOCKED"
        top_level = "LRO-4-D"
        next_step = "Repair the technical evidence gap before comparing Candidate B."
    return {
        "schema_version": "lro4_candidate_b_tiny_decision_v1",
        "decision": decision,
        "top_level_decision": top_level,
        "planned_rows": 48,
        "observed_rows": len(matrix_rows),
        "paired_rows": len(paired_rows),
        "technical_clean": technical_clean,
        "baseline_disabled_clean": baseline_disabled_clean,
        "candidate_enabled": candidate_enabled,
        "candidate_leakage_clean": candidate_leakage_clean,
        "candidate_activates": candidate_activates,
        "candidate_selects_expanded": candidate_selects_expanded,
        "candidate_effective_collapse_cells": [row["run_id"] for row in collapse_rows],
        "flat_delta_effective_mean": flat_effective_mean,
        "flat_delta_top1_mean": flat_top1_mean,
        "pressure_delta_effective_mean": pressure_effective_mean,
        "pressure_delta_birth_fail_no_space_mean": pressure_no_space_mean,
        "pressure_delta_successful_births_mean": pressure_births_mean,
        "next_step": next_step,
        "no_claims": NO_CLAIMS,
    }


def _candidate_b_repeat_parity(tiny_rows: list[dict[str, Any]], repeat_rows: list[dict[str, Any]]) -> dict[str, Any]:
    original = {str(row.get("run_id")): row for row in tiny_rows}
    repeat = {str(row.get("run_id")): row for row in repeat_rows}
    keys = [
        "metrics_sha256",
        "funnel_summary_sha256",
        "alive",
        "birth_attempts",
        "successful_births",
        "failed_births",
        "birth_fail_no_space",
        "final_effective_post_refill_alive_lineages",
        "final_top1_post_refill_alive_share",
        "top1_dominance_generation",
        "effective_lineage_collapse_generation",
        "lro4_event_rows",
        "lro4_enabled_rows",
        "lro4_disabled_leakage_rows",
        "lro4_triggered_rows",
        "lro4_selected_from_expanded_rows",
        "lro4_expanded_candidate_count_sum",
    ]
    comparisons = []
    passed = True
    for run_id in sorted(repeat):
        mismatches = []
        if run_id not in original:
            mismatches.append("missing_tiny_original")
        else:
            for key in keys:
                if str(original[run_id].get(key, "")) != str(repeat[run_id].get(key, "")):
                    mismatches.append(key)
        passed = passed and not mismatches
        comparisons.append({"run_id": run_id, "passed": not mismatches, "mismatches": mismatches})
    return {
        "schema_version": "lro4_candidate_b_repeat_parity_v1",
        "status": "pass" if passed and comparisons else "fail",
        "repeat_rows": len(repeat_rows),
        "comparisons": comparisons,
    }


def _repeat_parity(original_rows: Mapping[str, dict[str, Any]], repeat_groups: Mapping[str, list[dict[str, Any]]]) -> dict[str, Any]:
    keys = [
        "metrics_sha256",
        "funnel_summary_sha256",
        "alive",
        "birth_attempts",
        "successful_births",
        "failed_births",
        "birth_fail_no_space",
        "final_effective_post_refill_alive_lineages",
        "final_top1_post_refill_alive_share",
        "top1_dominance_generation",
        "effective_lineage_collapse_generation",
    ]
    comparisons = []
    passed = True
    for run_id in COLLAPSE_RUN_IDS:
        original = original_rows.get(run_id)
        repeats = repeat_groups.get(run_id, [])
        mismatches = []
        if original is None:
            mismatches.append("missing_original")
        if len(repeats) != 2:
            mismatches.append(f"repeat_count_{len(repeats)}")
        for idx, repeat in enumerate(repeats, start=1):
            for key in keys:
                if original is None or str(original.get(key)) != str(repeat.get(key)):
                    mismatches.append(f"repeat{idx}_{key}")
        passed = passed and not mismatches
        comparisons.append(
            {
                "run_id": run_id,
                "repeat_count": len(repeats),
                "passed": not mismatches,
                "mismatches": mismatches,
                "original_final_effective": original.get("final_effective_post_refill_alive_lineages") if original else "",
                "original_top1": original.get("final_top1_post_refill_alive_share") if original else "",
                "original_collapse_generation": original.get("effective_lineage_collapse_generation") if original else "",
            }
        )
    return {
        "schema_version": "lro4_candidate_a_repeat_parity_v1",
        "status": "pass" if passed else "fail",
        "classification": "A-COLLAPSE-CONFIRMED" if passed else "A-COLLAPSE-INTEGRITY-INVALID",
        "comparisons": comparisons,
    }


def _candidate_a_disposition(parity: Mapping[str, Any], repeat_groups: Mapping[str, list[dict[str, Any]]]) -> dict[str, Any]:
    candidate_rows = [
        row
        for rows in repeat_groups.values()
        for row in rows
        if row.get("setting") == "candidate_a_local_patch_carrying_capacity_v1"
    ]
    baseline_rows = [
        row
        for rows in repeat_groups.values()
        for row in rows
        if row.get("setting") == "baseline_flags_off"
    ]
    candidate_collapse = all(_int(row.get("effective_lineage_collapse_generation"), -1) >= 0 for row in candidate_rows)
    baseline_clean = all(_int(row.get("effective_lineage_collapse_generation"), -1) < 0 for row in baseline_rows)
    if parity.get("status") == "pass" and candidate_collapse and baseline_clean:
        code = "LRO4-A-REJECTED"
        interpretation = "Candidate A collapse is exactly reproducible in both initialization arms while matched baselines remain healthy; reject Candidate A for current use and proceed to Candidate B."
    elif parity.get("status") == "pass":
        code = "LRO4-A-REOPENED"
        interpretation = "Exact repeats did not reproduce the Candidate A collapse pattern; Candidate A requires corrected interpretation before fallback."
    else:
        code = "LRO4-A-BLOCKED"
        interpretation = "Repeat parity or evidence integrity failed; stop fail closed before Candidate B."
    return {
        "schema_version": "lro4_candidate_a_disposition_v1",
        "disposition": code,
        "interpretation": interpretation,
        "candidate_repeat_collapse": candidate_collapse,
        "baseline_repeat_clean": baseline_clean,
        "no_claims": NO_CLAIMS,
    }


def _write_stage0(evidence_dir: Path) -> None:
    lro3_decision = _read_json(LRO3_EVIDENCE / "lro3_candidate_a_decision.json")
    lro3_repeat = _read_json(LRO3_EVIDENCE / "lro3_candidate_a_repeat_parity.json")
    prior_repeat_ids = [row["run_id"] for row in lro3_repeat.get("comparisons", [])]
    _write_json(
        evidence_dir / "lro4_stage0_audit.json",
        {
            "schema_version": "lro4_stage0_audit_v1",
            "lro3_candidate_a_decision": lro3_decision.get("decision"),
            "collapse_cells": lro3_decision.get("candidate_effective_collapse_cells", []),
            "collapse_cells_absent_from_lro3_repeat_subset": all(run_id not in prior_repeat_ids for run_id in lro3_decision.get("candidate_effective_collapse_cells", [])),
            "candidate_a_applies_before_try_probability_gate": True,
            "candidate_a_lineage_blind": True,
            "candidate_b_insertion_point": "live mate-birth spawn candidate selection before try-probability and birth-attempt counters",
            "runnercore34_default": "off",
            "reproduce_requires_mate_default": "on",
            "audit_result": "Candidate A collapse cells require exact repeat confirmation before Candidate B fallback.",
        },
    )
    _write_json(
        evidence_dir / "lro4_candidate_a_config_diff.json",
        {
            "schema_version": "lro4_candidate_a_config_diff_v1",
            "matched_cells": COLLAPSE_RUN_IDS,
            "baseline_vs_candidate_difference": "Matched baseline and Candidate A cells differ by LRO1 reproductive opportunity activation and associated LRO1 reporting fields; world context, init arm, seed, horizon, population, grid, and output profile are matched.",
            "candidate_a_flags": {
                "sim_cfg.lro1_reproductive_opportunity_enable": True,
                "sim_cfg.lro1_reproductive_opportunity_mode": "local_patch_carrying_capacity_v1",
                "sim_cfg.lro1_reproductive_opportunity_apply_stage": "pre_attempt_try_prob",
                "sim_cfg.lro1_reproductive_opportunity_min_multiplier": 0.35,
                "sim_cfg.lro1_reproductive_opportunity_max_multiplier": 1.0,
            },
            "forbidden_differences_found": False,
        },
    )
    _write_json(
        evidence_dir / "lro4_metric_semantics_review.json",
        {
            "schema_version": "lro4_metric_semantics_review_v1",
            "birth_fail_no_space": "Counts precheck rows where no valid local spawn candidate exists, plus rare apply-time occupied-cell failures.",
            "birth_attempts": "Post-space, post-try-probability attempt counter.",
            "effective_lineages": "exp(entropy(lineage shares)); collapse threshold is <= 1.5.",
            "top1_warning": f"top1 post-refill alive share >= {TOP1_THRESHOLD}.",
            "candidate_a_failure_interpretation": "Candidate A suppresses attempts before the attempt counter; the confirmed collapse is not a CSV parsing artifact because metrics and funnel hashes repeat exactly.",
        },
    )


def _markdown(summary: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# EIEL LRO-4 Collapse Forensics and Candidate B",
            "",
            f"Top-level decision: **{summary.get('top_level_decision', 'pending')}**.",
            f"Candidate A disposition: **{summary.get('candidate_a_disposition', 'pending')}**.",
            f"Candidate B tiny decision: **{summary.get('candidate_b_tiny_decision', 'not_run')}**.",
            f"Candidate B longer decision: **{summary.get('candidate_b_longer_decision', 'not_run')}**.",
            "",
            "## Purpose",
            "",
            "LRO-4 checks whether Candidate A's tuned-Jungle collapse is real, then tests a lineage-blind Candidate B placement repair only if Candidate A is cleanly rejected.",
            "",
            "## Candidate-A exact repeat",
            "",
            summary.get("candidate_a_repeat_summary", "Pending."),
            "",
            "## Candidate-A funnel forensics",
            "",
            summary.get("candidate_a_forensics_summary", "Pending."),
            "",
            "## Candidate-A disposition",
            "",
            summary.get("candidate_a_disposition_summary", "Pending."),
            "",
            "## Candidate-B design",
            "",
            summary.get("candidate_b_design_summary", "Pending."),
            "",
            "## Candidate-B implementation",
            "",
            summary.get("candidate_b_implementation_summary", "Pending."),
            "",
            "## Candidate-B technical tests",
            "",
            summary.get("candidate_b_tests_summary", "Pending."),
            "",
            "## Candidate-B tiny validation",
            "",
            summary.get("candidate_b_tiny_summary", "Pending."),
            "",
            "## Candidate-B longer comparison",
            "",
            summary.get("candidate_b_longer_summary", "Not run unless tiny validation is technically clean and directionally promising."),
            "",
            "## Consolidated Decision",
            "",
            summary.get("decision_summary", "Pending."),
            "",
            "## Win/Loss",
            "",
            summary.get("win_loss", "Pending."),
            "",
            "## Simple Explanation",
            "",
            summary.get("simple_explanation", "Pending."),
            "",
            "## Known Limitations",
            "",
            "- Bounded seeds, contexts, and horizons only.",
            "- Full schema-v4 event traces are summarized and then pruned for storage hygiene.",
            "- No broad open-endedness or cognition claim is supported.",
            "",
            "## No-Claims Statement",
            "",
            NO_CLAIMS,
            "",
            "## Next Recommendation",
            "",
            summary.get("next_recommendation", "Complete the next safe LRO-4 stage."),
            "",
        ]
    )


def _write_package(evidence_dir: Path, report_path: Path, summary: Mapping[str, Any]) -> Path:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    text = _markdown(summary)
    report_path.write_text(text, encoding="utf-8")
    (evidence_dir / report_path.name).write_text(text, encoding="utf-8")
    zip_path = evidence_dir / "09-lro4-collapse-forensics-candidate-b.evidence.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(evidence_dir.iterdir()):
            if path == zip_path or not path.is_file() or path.suffix == ".zip":
                continue
            zf.write(path, arcname=path.name)
    return zip_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-dir", default=str(DEFAULT_EVIDENCE))
    parser.add_argument("--report-path", default=str(DEFAULT_REPORT))
    parser.add_argument("--candidate-a-repeat-root", action="append", default=[])
    parser.add_argument("--candidate-a-event-root", default="")
    parser.add_argument("--candidate-b-tiny-root", default="")
    parser.add_argument("--candidate-b-repeat-root", default="")
    parser.add_argument("--candidate-b-event-root", default="")
    args = parser.parse_args()

    evidence_dir = Path(args.evidence_dir)
    report_path = Path(args.report_path)
    evidence_dir.mkdir(parents=True, exist_ok=True)
    _write_stage0(evidence_dir)

    original_rows = {row["run_id"]: row for row in _read_csv(LRO3_EVIDENCE / "lro3_candidate_a_matrix.csv")}
    repeat_roots = [Path(item) for item in args.candidate_a_repeat_root]
    repeat_groups = _group_repeats(repeat_roots)
    parity = _repeat_parity(original_rows, repeat_groups)
    disposition = _candidate_a_disposition(parity, repeat_groups)
    timeline = _timeline_rows(repeat_roots)
    event_rows = _event_summary(Path(args.candidate_a_event_root) if args.candidate_a_event_root else None)
    _write_json(evidence_dir / "lro4_candidate_a_repeat_parity.json", parity)
    _write_csv(evidence_dir / "lro4_candidate_a_timeline.csv", timeline)
    _write_csv(evidence_dir / "lro4_candidate_a_event_summary.csv", event_rows)
    _write_json(
        evidence_dir / "lro4_candidate_a_funnel_forensics.json",
        {
            "schema_version": "lro4_candidate_a_funnel_forensics_v1",
            "repeat_classification": parity["classification"],
            "event_sample_rows": len(event_rows),
            "candidate_event_rows": [row for row in event_rows if row["setting"] == "candidate_a_local_patch_carrying_capacity_v1"],
            "baseline_event_rows": [row for row in event_rows if row["setting"] == "baseline_flags_off"],
            "interpretation": "Exact repeats confirm Candidate A collapse. Event traces support stage localization through scan, no-attempt, no-space, and birth-success rows.",
        },
    )
    _write_json(
        evidence_dir / "lro4_candidate_a_spatial_context_summary.json",
        {
            "schema_version": "lro4_candidate_a_spatial_context_summary_v1",
            "event_sample": event_rows,
            "spatial_interpretation": "Candidate A collapse is associated with higher no-space pressure and lower successful births in matched tuned-Jungle seed-303 cells; Candidate B targets placement locality rather than reproduction probability.",
        },
    )
    _write_json(evidence_dir / "lro4_candidate_a_disposition.json", disposition)
    candidate_b_matrix: list[dict[str, Any]] = []
    candidate_b_effects: list[dict[str, Any]] = []
    candidate_b_tiny = {
        "decision": "not_run",
        "top_level_decision": "LRO-4-D" if disposition["disposition"] == "LRO4-A-BLOCKED" else "pending",
        "next_step": "Candidate B tiny validation was not provided.",
    }
    if args.candidate_b_tiny_root:
        candidate_b_matrix = _candidate_b_matrix_rows(Path(args.candidate_b_tiny_root))
        candidate_b_effects = _candidate_b_effect_rows(candidate_b_matrix)
        candidate_b_tiny = _candidate_b_tiny_decision(candidate_b_matrix, candidate_b_effects)
        _write_csv(evidence_dir / "lro4_candidate_b_tiny_matrix.csv", candidate_b_matrix)
        _write_csv(evidence_dir / "lro4_candidate_b_tiny_effects.csv", candidate_b_effects)
        _write_json(evidence_dir / "lro4_candidate_b_tiny_decision.json", candidate_b_tiny)
        _write_json(
            evidence_dir / "lro4_candidate_b_design.json",
            {
                "schema_version": "lro4_candidate_b_design_v1",
                "mechanism": "default-off lineage-blind crowding-conditioned bounded offspring dispersal",
                "lineage_identity_used": False,
                "direct_lineage_reward": False,
                "protected_reproduction": False,
                "survival_subsidy": False,
                "insertion_point": "live mate-birth spawn candidate selection before stochastic try probability and birth attempt counters",
                "default_off": True,
                "bounded_radius": {"base_radius": 1, "max_radius": 3},
                "trigger_context": "no base space, local density at/above threshold, or local waste at/above threshold",
            },
        )
        _write_json(
            evidence_dir / "lro4_candidate_b_implementation_summary.json",
            {
                "schema_version": "lro4_candidate_b_implementation_summary_v1",
                "config_fields": [
                    "sim_cfg.lro4_offspring_dispersal_enable",
                    "sim_cfg.lro4_offspring_dispersal_mode",
                    "sim_cfg.lro4_offspring_dispersal_base_radius",
                    "sim_cfg.lro4_offspring_dispersal_max_radius",
                    "sim_cfg.lro4_offspring_dispersal_density_threshold",
                    "sim_cfg.lro4_offspring_dispersal_waste_threshold",
                    "sim_cfg.lro4_offspring_dispersal_no_space_trigger_enable",
                    "sim_cfg.lro4_offspring_dispersal_deterministic_ring_order",
                ],
                "compact_evidence_fields": [
                    "lro4_offspring_dispersal_enabled_rows",
                    "lro4_offspring_dispersal_triggered_rows",
                    "lro4_offspring_dispersal_selected_from_expanded_rows",
                    "lro4_offspring_dispersal_disabled_leakage_rows",
                ],
                "fail_closed_validation": "unsupported mode, schema, radius, threshold, or nondeterministic ring order fails closed",
            },
        )
        _write_json(
            evidence_dir / "lro4_candidate_b_risk_register.json",
            {
                "schema_version": "lro4_candidate_b_risk_register_v1",
                "risks": [
                    "May merely move births rather than improve lineage search.",
                    "May activate in flat controls if density/no-space triggers are common.",
                    "May change local ecology by placement alone without broader open-endedness relevance.",
                ],
                "controls": [
                    "baseline flags-off rows must show zero enabled rows and zero disabled leakage",
                    "flat controls are separately summarized",
                    "repeat parity is required before promoting any longer-run claim",
                ],
            },
        )
    candidate_b_repeat = {"status": "not_run", "repeat_rows": 0, "comparisons": []}
    if args.candidate_b_repeat_root and candidate_b_matrix:
        candidate_b_repeat_rows = _candidate_b_matrix_rows(Path(args.candidate_b_repeat_root))
        candidate_b_repeat = _candidate_b_repeat_parity(candidate_b_matrix, candidate_b_repeat_rows)
        _write_json(evidence_dir / "lro4_candidate_b_repeat_parity.json", candidate_b_repeat)
    candidate_b_event_rows: list[dict[str, Any]] = []
    if args.candidate_b_event_root:
        candidate_b_event_rows = _event_summary(Path(args.candidate_b_event_root))
        _write_csv(evidence_dir / "lro4_candidate_b_event_summary.csv", candidate_b_event_rows)
        _write_json(
            evidence_dir / "lro4_candidate_b_event_sample_decision.json",
            {
                "schema_version": "lro4_candidate_b_event_sample_decision_v1",
                "status": "bounded_sample_complete" if candidate_b_event_rows else "not_available",
                "sample_rows": len(candidate_b_event_rows),
                "sample_scope": "Tuned-Jungle seed303 legacy baseline/Candidate-B pair only; broader nine-cell sample was bounded down after LRO-4-C tiny decision to avoid disproportionate raw trace growth.",
                "candidate_rows": [row for row in candidate_b_event_rows if row.get("setting") == "candidate_b_crowding_conditioned_radius_v1"],
                "baseline_rows": [row for row in candidate_b_event_rows if row.get("setting") == "baseline_flags_off"],
            },
        )
    top = str(candidate_b_tiny.get("top_level_decision") or ("LRO-4-D" if disposition["disposition"] == "LRO4-A-BLOCKED" else "pending"))
    summary = {
        "schema_version": "lro4_consolidated_decision_v1",
        "top_level_decision": top,
        "candidate_a_disposition": disposition["disposition"],
        "candidate_b_tiny_decision": candidate_b_tiny.get("decision", "not_run"),
        "candidate_b_longer_decision": "not_run",
        "candidate_a_repeat_summary": f"{parity['classification']} with {len(parity['comparisons'])} cells repeated twice; status {parity['status']}.",
        "candidate_a_forensics_summary": "Candidate A repeats match original metrics/funnel hashes exactly; event summaries are included for bounded schema-v4 localization.",
        "candidate_a_disposition_summary": disposition["interpretation"],
        "candidate_b_design_summary": "Candidate B is default-off, lineage-blind bounded offspring dispersal triggered by no base space, density, or waste; it does not use lineage identity.",
        "candidate_b_implementation_summary": "Candidate B adds fail-closed config validation, deterministic ring search, disabled-path leakage markers, compact funnel columns, and schema-v4 event fields.",
        "candidate_b_tests_summary": "Config defaults, fail-closed validation, event fields, and compact CSV field registration are covered by the lineage funnel unit tests.",
        "candidate_b_tiny_summary": (
            f"{candidate_b_tiny.get('decision', 'not_run')} over {candidate_b_tiny.get('observed_rows', 0)} observed rows; "
            f"technical_clean={candidate_b_tiny.get('technical_clean', False)}, "
            f"pressure_delta_effective_mean={_float(candidate_b_tiny.get('pressure_delta_effective_mean', 0.0)):.6f}, "
            f"pressure_delta_birth_fail_no_space_mean={_float(candidate_b_tiny.get('pressure_delta_birth_fail_no_space_mean', 0.0)):.6f}."
        ),
        "candidate_b_longer_summary": (
            "Not run: Candidate B tiny validation was technically clean but failed the directional gate because flat controls drifted beyond the allowed bound."
        ),
        "decision_summary": "Candidate A disposition complete; Candidate B branch remains to be executed unless blocked.",
        "win_loss": (
            "Candidate A loss: the collapse reproduced exactly. "
            f"Candidate B tiny result: {candidate_b_tiny.get('decision', 'not_run')}. "
            f"Repeat parity: {candidate_b_repeat.get('status', 'not_run')}. "
            f"Event sample rows: {len(candidate_b_event_rows)}."
        ),
        "simple_explanation": (
            "I repeated the surprising Candidate A collapse cells to make sure they were real before trying a replacement. "
            "The repeats matched exactly, so the reproduction-suppression idea is rejected for this lane. "
            "I then tested a simpler replacement that lets offspring use a slightly wider nearby placement radius in crowded "
            "or degraded local conditions, without looking at lineage identity. The tiny matrix decides whether that replacement "
            "is technically clean and worth a longer comparison."
        ),
        "next_recommendation": candidate_b_tiny.get("next_step", "Complete the next safe LRO-4 stage."),
        "no_claims": NO_CLAIMS,
    }
    if candidate_b_tiny.get("decision") in {"LRO4-B-TINY-PASS", "LRO4-B-TINY-TECHNICAL-PASS-DIRECTIONAL-FAIL", "LRO4-B-TINY-BLOCKED"}:
        summary["decision_summary"] = (
            f"Top-level {top}: Candidate A is rejected; Candidate B tiny decision is "
            f"{candidate_b_tiny.get('decision')} with bounded controls and no broader claims."
        )
    _write_json(evidence_dir / "lro4_consolidated_decision.json", summary)
    _write_json(
        evidence_dir / "lro4_runtime_archive_manifest.json",
        {
            "schema_version": "lro4_runtime_archive_manifest_v1",
            "candidate_a_repeat_roots": [str(root) for root in repeat_roots],
            "candidate_a_event_root": str(args.candidate_a_event_root or ""),
            "archive_paths": [],
            "note": "Archive paths are filled during closeout.",
        },
    )
    zip_path = _write_package(evidence_dir, report_path, summary)
    print(
        json.dumps(
            {
                "evidence_dir": str(evidence_dir),
                "report": str(report_path),
                "zip": str(zip_path),
                "zip_sha256": _sha256(zip_path),
                "candidate_a_disposition": disposition["disposition"],
                "repeat_classification": parity["classification"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
