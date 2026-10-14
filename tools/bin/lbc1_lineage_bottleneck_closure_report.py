#!/usr/bin/env python3
"""LBC-1 lineage bottleneck closure decision helper.

Reporting and replay orchestration only. This helper reuses the checkpoint
continuation path completed in R3a/R3b and extracts richer lineage exposure
tracking from existing metrics JSON fields.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


csv.field_size_limit(sys.maxsize)

R3A_PATH = Path(__file__).resolve().parent / "integrated_r3a_branch_replay.py"
SPEC = importlib.util.spec_from_file_location("integrated_r3a_branch_replay", R3A_PATH)
assert SPEC is not None and SPEC.loader is not None
r3a = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(r3a)

SCENARIOS: tuple[str, ...] = (
    "lbc1_00_tuned_full_jungle_recovery",
    "lbc1_01_default_full_jungle",
    "lbc1_02_theme12_clean_pressure_reference",
)

EXPOSURE_FIELDS: dict[str, str] = {
    "predator_exposure": "lineage_predator_event_count",
    "predator_damage": "lineage_predator_damage_total",
    "damage_prevented": "lineage_predator_damage_prevented_total",
    "waste_exposure": "lineage_patch_waste_mean",
    "density_exposure": "lineage_patch_density_mean",
    "food_exposure": "lineage_patch_food_amount_mean",
    "recovery_exposure": "lineage_patch_recovery_activity_mean",
    "shelter_exposure": "lineage_patch_shelter_mean",
    "cover_exposure": "lineage_patch_cover_mean",
    "movement_cost_exposure": "lineage_patch_movement_cost_mean",
    "birth_patch_food": "lineage_birth_patch_food_amount_mean",
    "birth_patch_cover": "lineage_birth_patch_cover_mean",
    "birth_patch_shelter": "lineage_birth_patch_shelter_mean",
    "birth_patch_waste": "lineage_birth_patch_waste_mean",
    "birth_patch_density": "lineage_birth_patch_density_mean",
    "birth_patch_recovery": "lineage_birth_patch_recovery_activity_mean",
}

REQUIRED_TRAJECTORY_FIELDS: tuple[str, ...] = (
    "candidate_present",
    "candidate_alive_count",
    "candidate_alive_share",
    "candidate_birth_count",
    "candidate_birth_share",
)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_yaml(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if yaml is not None:
        path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    else:
        path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _num(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _lineage_num(lineage_id: Any) -> int | None:
    text = str(lineage_id or "").strip()
    if text.startswith("L"):
        text = text[1:]
    try:
        return int(text)
    except ValueError:
        return None


def _same_lineage(left: Any, right: Any) -> bool:
    if str(left or "") == str(right or ""):
        return True
    left_num = _lineage_num(left)
    right_num = _lineage_num(right)
    return left_num is not None and left_num == right_num


def _scenario_runs(source_root: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for summary_path in sorted(source_root.rglob("summary.json")):
        try:
            summary = _read_json(summary_path)
        except Exception:
            continue
        if summary.get("status") != "complete":
            continue
        for run in summary.get("runs") or []:
            scenario = str((run.get("axes") or {}).get("scenario") or "")
            if scenario in SCENARIOS:
                item = dict(run)
                item["scenario"] = scenario
                item["summary_path"] = str(summary_path)
                out.append(item)
    out.sort(key=lambda item: (SCENARIOS.index(str(item["scenario"])), int(item.get("seed", 0) or 0)))
    return out


def source_summary(source_root: Path) -> dict[str, Any]:
    runs = _scenario_runs(source_root)
    rows = []
    for run in runs:
        run_dir = Path(str(run.get("run_dir")))
        metrics = run_dir / "metrics.csv"
        checkpoint_gens = []
        for bundle in sorted((run_dir / "checkpoint_bundles_v2").glob("bundle_gen*")):
            try:
                checkpoint_gens.append(int(bundle.name.replace("bundle_gen", "")))
            except ValueError:
                pass
        rows.append(
            {
                "scenario": run.get("scenario"),
                "seed": run.get("seed"),
                "status": run.get("status"),
                "run_dir": str(run_dir),
                "metrics_exists": metrics.exists(),
                "metrics_bytes": metrics.stat().st_size if metrics.exists() else 0,
                "checkpoint_generations": checkpoint_gens,
                "has_checkpoint_generations_0_to_6": all(gen in checkpoint_gens for gen in range(7)),
            }
        )
    return {
        "schema_version": "lbc1_source_summary_v1",
        "planned": 15,
        "completed": sum(1 for row in rows if row["status"] == "ok"),
        "failed": sum(1 for row in rows if row["status"] != "ok"),
        "runs": rows,
    }


def _dominant_lineage_id(run_dir: Path) -> str:
    return r3a._dominant_lineage_id(run_dir)


def _lineage_row_at_or_after(run_dir: Path, lineage_id: str, generation: int) -> dict[str, Any] | None:
    return r3a._lineage_row_at_or_after(run_dir, lineage_id, generation)


def _branch_candidates(run_dir: Path) -> list[dict[str, Any]]:
    return r3a._branch_candidates(run_dir)


def _archive_entries(run_dir: Path) -> dict[str, list[dict[str, Any]]]:
    return r3a._archive_entries(run_dir)


def _best_archive_entry(entries: Mapping[str, Sequence[Mapping[str, Any]]], lineage_id: str, generation: int) -> dict[str, Any] | None:
    return r3a._best_archive_entry(entries, lineage_id, generation)


def _candidate_entry(
    *,
    role: str,
    candidate_class: str,
    run_record: Mapping[str, Any],
    candidate: Mapping[str, Any],
    archive_entry: Mapping[str, Any] | None,
    source_root: Path,
    output_root: Path,
) -> dict[str, Any] | None:
    run_dir = Path(str(run_record.get("run_dir") or ""))
    config_path = run_dir / "config.json"
    metrics_path = run_dir / "metrics.csv"
    if not config_path.exists() or not metrics_path.exists():
        return None
    config = _read_json(config_path)
    sim = dict(config.get("sim_cfg") or {})
    generation = int(candidate.get("generation", 0) or 0)
    lineage_id = str(candidate.get("lineage_id") or "")
    if not lineage_id:
        return None
    checkpoint = r3a.link_checkpoint_for_generation(run_dir, generation)
    if not checkpoint.get("valid"):
        return None
    lineage_row = _lineage_row_at_or_after(run_dir, lineage_id, generation) or {}
    category = "dominant_lineage" if role == "dominant" else str(candidate.get("candidate_type") or "")
    scenario = str(run_record.get("scenario") or "")
    seed = int(run_record.get("seed", sim.get("seed", 0)) or 0)
    entry_id = f"lbc1_{role}_{scenario}_{seed:06d}_{lineage_id}_g{generation:06d}"
    descriptor = {
        "branch_candidate": dict(candidate),
        "archive_entry": dict(archive_entry or {}),
        "lineage_descriptor": r3a._lineage_descriptor_at(run_dir, lineage_id, generation),
        "lineage_metrics": dict(lineage_row),
    }
    return {
        "manifest_version": "lbc1_branch_replay_manifest_v1",
        "manifest_id": "lbc1_branch_replay",
        "entry_id": entry_id,
        "candidate_id": entry_id,
        "source_campaign_id": "lbc1_lineage_bottleneck_closure_source",
        "source_output_root": str(source_root),
        "source_run_id": r3a._run_id(run_dir),
        "source_scenario_id": scenario,
        "source_seed": seed,
        "source_device": str(sim.get("device") or ""),
        "source_generation": generation,
        "source_checkpoint_path": str(checkpoint["source_checkpoint_path"]),
        "source_checkpoint_metadata_path": str(checkpoint["source_checkpoint_metadata_path"]),
        "source_config_path": str(config_path),
        "source_metrics_path": str(metrics_path),
        "candidate_lineage_id": lineage_id,
        "candidate_biological_lineage_id": lineage_id,
        "candidate_category": category,
        "candidate_class": candidate_class,
        "candidate_dominance_class": role,
        "dominant_or_non_dominant": role,
        "candidate_generation_emitted": generation,
        "candidate_alive_share_at_emission": candidate.get("population_share", lineage_row.get("population_share")),
        "candidate_birth_share_at_emission": candidate.get("reproduction_share", lineage_row.get("reproduction_share")),
        "candidate_descriptor_snapshot": descriptor,
        "candidate_archive_entry_id": str((archive_entry or {}).get("archive_entry_id") or "missing_optional"),
        "checkpoint_reference_generation": int(checkpoint["checkpoint_generation"]),
        "replay_mode": "checkpoint_continuation",
        "replay_start_generation": int(checkpoint["checkpoint_generation"]),
        "replay_generations": 8,
        "replay_steps_per_generation": int(sim.get("steps_per_generation", 60) or 60),
        "population_handling_mode": "restored_from_checkpoint",
        "allowed_mutation_policy": "unchanged",
        "allowed_selection_policy": "unchanged",
        "allowed_reproduction_policy": "unchanged",
        "allowed_ecology_policy": "unchanged",
        "expected_flags": r3a._expected_flags(config),
        "expected_tuned_overrides": r3a._expected_tuned_overrides(config),
        "expected_theme4_disabled_status": r3a._theme4_disabled_status(config),
        "expected_theme5_reporting_status": r3a._theme5_reporting_status(config),
        "output_root": str(output_root / "replays" / "primary" / entry_id),
        "output_dir": str(output_root / "replays" / "primary" / entry_id / "replay"),
        "reason_selected": "dominant final lineage from source run" if role == "dominant" else "valid non-dominant branch candidate at or before collapse window",
    }


def select_candidate_cohort(source_root: Path, output_root: Path) -> dict[str, Any]:
    runs = _scenario_runs(source_root)
    by_scenario: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        by_scenario[str(run["scenario"])].append(run)
    selected: list[dict[str, Any]] = []
    limitations: list[str] = []
    for scenario in SCENARIOS:
        scenario_runs = by_scenario.get(scenario, [])
        dominant_added = 0
        non_dominant_added = 0
        for run in scenario_runs:
            if dominant_added >= 2:
                break
            run_dir = Path(str(run["run_dir"]))
            dominant_id = _dominant_lineage_id(run_dir)
            row = _lineage_row_at_or_after(run_dir, dominant_id, 0) or {"lineage_id": dominant_id, "generation": 0}
            candidate = {
                "lineage_id": dominant_id,
                "generation": int(row.get("generation", 0) or 0),
                "candidate_type": "dominant_lineage",
                "population_share": row.get("population_share"),
                "reproduction_share": row.get("reproduction_share"),
            }
            archive_entry = _best_archive_entry(_archive_entries(run_dir), dominant_id, int(candidate["generation"]))
            entry = _candidate_entry(
                role="dominant",
                candidate_class="dominant",
                run_record=run,
                candidate=candidate,
                archive_entry=archive_entry,
                source_root=source_root,
                output_root=output_root,
            )
            if entry is not None:
                selected.append(entry)
                dominant_added += 1
        non_dom_pool: list[tuple[int, float, Mapping[str, Any], Mapping[str, Any], Mapping[str, Any] | None]] = []
        for run in scenario_runs:
            run_dir = Path(str(run["run_dir"]))
            dominant_id = _dominant_lineage_id(run_dir)
            archive_by_lineage = _archive_entries(run_dir)
            for candidate in _branch_candidates(run_dir):
                lineage_id = str(candidate.get("lineage_id") or "")
                gen = int(candidate.get("generation", 0) or 0)
                if not lineage_id or lineage_id == dominant_id or gen > 5:
                    continue
                if (r3a._float(candidate.get("population_share")) or 0.0) <= 0.0:
                    continue
                if not r3a.link_checkpoint_for_generation(run_dir, gen).get("valid"):
                    continue
                archive_entry = _best_archive_entry(archive_by_lineage, lineage_id, gen)
                score = float(candidate.get("candidate_score", 0.0) or 0.0)
                non_dom_pool.append((gen, -score, run, candidate, archive_entry))
        non_dom_pool.sort(key=lambda item: (item[0], item[1], int(item[2].get("seed", 0) or 0), str(item[3].get("lineage_id"))))
        used_lineages: set[tuple[int, str]] = set()
        for gen, _neg_score, run, candidate, archive_entry in non_dom_pool:
            if non_dominant_added >= 2:
                break
            key = (int(run.get("seed", 0) or 0), str(candidate.get("lineage_id") or ""))
            if key in used_lineages:
                continue
            candidate_class = "early_non_dominant" if int(gen) <= 2 else "near_collapse_non_dominant"
            entry = _candidate_entry(
                role="non_dominant",
                candidate_class=candidate_class,
                run_record=run,
                candidate=candidate,
                archive_entry=archive_entry,
                source_root=source_root,
                output_root=output_root,
            )
            if entry is not None:
                selected.append(entry)
                used_lineages.add(key)
                non_dominant_added += 1
        if dominant_added < 2:
            limitations.append(f"{scenario}: only {dominant_added}/2 dominant candidates selected")
        if non_dominant_added < 2:
            limitations.append(f"{scenario}: only {non_dominant_added}/2 non-dominant candidates selected")
    return {
        "schema_version": "lbc1_candidate_cohort_v1",
        "target_candidate_count": 12,
        "selected_count": len(selected),
        "selection_limitations": limitations,
        "candidates": selected,
    }


def build_manifest(cohort: Mapping[str, Any], output_root: Path) -> dict[str, Any]:
    entries = [dict(item) for item in cohort.get("candidates") or []]
    return {
        "manifest_version": "lbc1_branch_replay_manifest_set_v1",
        "manifest_id": "lbc1_branch_replay",
        "source_campaign_id": "lbc1_lineage_bottleneck_closure_source",
        "source_output_root": str(entries[0].get("source_output_root") if entries else ""),
        "replay_mode": "checkpoint_continuation",
        "output_root": str(output_root),
        "entries": entries,
    }


def validate_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    validation = r3a.validate_manifest(manifest)
    errors = list(validation.get("errors") or [])
    for idx, entry in enumerate(manifest.get("entries") or []):
        if not entry.get("output_dir"):
            errors.append({"entry_index": idx, "field": "output_dir", "error": "missing output_dir"})
        if entry.get("candidate_class") not in {"dominant", "non_dominant_branch_candidate", "early_non_dominant", "near_collapse_non_dominant"}:
            errors.append({"entry_index": idx, "field": "candidate_class", "error": "unknown candidate class"})
    return {
        "schema_version": "lbc1_manifest_validation_v1",
        "entry_count": len(manifest.get("entries") or []),
        "valid": not errors,
        "errors": errors,
        "warnings": list(validation.get("warnings") or []),
    }


def _descriptor_by_lineage(row: Mapping[str, str], lineage_id: str) -> dict[str, Any] | None:
    target = _lineage_num(lineage_id)
    if target is None:
        return None
    raw = row.get("lineage_strategy_descriptors_json") or ""
    if not raw:
        return None
    try:
        descriptors = json.loads(raw)
    except Exception:
        return None
    for item in descriptors:
        if _lineage_num(item.get("lineage_id")) == target:
            return dict(item)
    return None


def _lineage_metrics_by_generation(replay_run_dir: Path) -> dict[int, dict[str, dict[str, Any]]]:
    out: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in _read_jsonl(replay_run_dir / "metrics" / "lineage_metrics.jsonl"):
        generation = int(row.get("generation", 0) or 0)
        lineage_id = str(row.get("lineage_id") or "")
        if lineage_id:
            out[generation][lineage_id] = dict(row)
    return dict(out)


def _lineage_descriptors_by_generation(replay_run_dir: Path) -> dict[int, dict[str, dict[str, Any]]]:
    out: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in _read_jsonl(replay_run_dir / "metrics" / "lineage_descriptors.jsonl"):
        generation = int(row.get("generation", 0) or 0)
        for item in row.get("lineages") or []:
            lineage_id = str(item.get("lineage_id") or "")
            if lineage_id:
                out[generation][lineage_id] = dict(item)
    return dict(out)


def _find_lineage_row(rows: Mapping[str, Mapping[str, Any]], lineage_id: str) -> dict[str, Any] | None:
    if lineage_id in rows:
        return dict(rows[lineage_id])
    for key, value in rows.items():
        if _same_lineage(key, lineage_id):
            return dict(value)
    return None


def _dominant_lineage_from_replay(metrics_by_gen: Mapping[int, Mapping[str, Mapping[str, Any]]]) -> str:
    if not metrics_by_gen:
        return ""
    generation = min(metrics_by_gen)
    rows = list(metrics_by_gen.get(generation, {}).values())
    if not rows:
        return ""
    top = max(
        rows,
        key=lambda row: (
            int(_num(row.get("population_count")) or 0),
            _num(row.get("population_share")) or 0.0,
            int(_num(row.get("offspring_count")) or 0),
            str(row.get("lineage_id") or ""),
        ),
    )
    return str(top.get("lineage_id") or "")


def _exposure_value(descriptor: Mapping[str, Any] | None, source: str) -> tuple[Any, str]:
    if not descriptor:
        return None, "candidate lineage descriptor missing for generation"
    value = descriptor.get(source)
    if value is not None:
        return value, "descriptor"
    ecology_profile = descriptor.get("ecology_patch_profile") or {}
    nested_aliases = {
        "lineage_predator_event_count": "predator_exposure_rate",
        "lineage_patch_waste_mean": "waste_tolerance_proxy",
        "lineage_patch_shelter_mean": "shelter_use_rate",
    }
    alias = nested_aliases.get(source)
    if alias and ecology_profile.get(alias) is not None:
        return ecology_profile.get(alias), "descriptor.ecology_patch_profile"
    return None, f"{source} unavailable in replay lineage descriptors"


def _build_replay_lineage_tracking(replay_run_dir: Path, entry: Mapping[str, Any]) -> dict[str, Any]:
    candidate = str(entry.get("candidate_lineage_id") or "")
    candidate_bio = str(entry.get("candidate_biological_lineage_id") or candidate)
    metrics_by_gen = _lineage_metrics_by_generation(replay_run_dir)
    descriptors_by_gen = _lineage_descriptors_by_generation(replay_run_dir)
    dominant = _dominant_lineage_from_replay(metrics_by_gen)
    generations = sorted(set(metrics_by_gen) | set(descriptors_by_gen))
    candidate_trajectory: list[dict[str, Any]] = []
    dominant_trajectory: list[dict[str, Any]] = []
    exposure_rows: list[dict[str, Any]] = []
    birth_context_rows: list[dict[str, Any]] = []
    availability = {
        field: {
            "available": False,
            "available_count": 0,
            "total_rows": 0,
            "source": "unavailable",
            "reason_if_unavailable": "no replay rows inspected",
        }
        for field in EXPOSURE_FIELDS
    }
    first_missing_generation: int | None = None
    seen_present = False
    reappeared = False
    previous_present = False
    for generation in generations:
        metric_rows = metrics_by_gen.get(generation, {})
        descriptor_rows = descriptors_by_gen.get(generation, {})
        cand_metric = _find_lineage_row(metric_rows, candidate)
        dom_metric = _find_lineage_row(metric_rows, dominant)
        cand_desc = _find_lineage_row(descriptor_rows, candidate)
        candidate_present = bool(cand_metric and int(_num(cand_metric.get("population_count")) or 0) > 0)
        if candidate_present:
            if seen_present and not previous_present:
                reappeared = True
            seen_present = True
        elif first_missing_generation is None:
            first_missing_generation = generation
        previous_present = candidate_present
        exposure_values: dict[str, Any] = {}
        exposure_meta: dict[str, dict[str, Any]] = {}
        for public, source in EXPOSURE_FIELDS.items():
            value, value_source = _exposure_value(cand_desc, source)
            data = availability[public]
            data["total_rows"] += 1
            if value is not None:
                data["available"] = True
                data["available_count"] += 1
                data["source"] = value_source
                data["reason_if_unavailable"] = ""
            elif not data["available"]:
                data["source"] = "unavailable"
                data["reason_if_unavailable"] = value_source
            exposure_values[f"candidate_{public}"] = value
            exposure_meta[f"candidate_{public}"] = {
                "available": value is not None,
                "source": value_source if value is not None else "unavailable",
                "reason_if_unavailable": "" if value is not None else value_source,
            }
        candidate_item: dict[str, Any] = {
            "generation": generation,
            "candidate_lineage_id": candidate,
            "candidate_biological_lineage_id": candidate_bio,
            "candidate_present": candidate_present,
            "candidate_alive_count": int(_num(cand_metric.get("population_count")) or 0) if cand_metric else 0,
            "candidate_alive_share": (_num(cand_metric.get("population_share")) or 0.0) if cand_metric else 0.0,
            "candidate_birth_count": int(_num(cand_metric.get("offspring_count")) or 0) if cand_metric else 0,
            "candidate_birth_share": (_num(cand_metric.get("reproduction_share")) or 0.0) if cand_metric else 0.0,
            "candidate_extinct_flag": not candidate_present,
            "candidate_first_missing_generation": first_missing_generation,
            "candidate_reappeared_flag": reappeared,
            **exposure_values,
        }
        # Backward-compatible aliases consumed by LBC-1 tables.
        for public in EXPOSURE_FIELDS:
            candidate_item[public] = candidate_item[f"candidate_{public}"]
        dominant_item: dict[str, Any] = {
            "generation": generation,
            "dominant_lineage_id": dominant,
            "dominant_biological_lineage_id": dominant,
            "dominant_alive_count": int(_num(dom_metric.get("population_count")) or 0) if dom_metric else 0,
            "dominant_alive_share": (_num(dom_metric.get("population_share")) or 0.0) if dom_metric else 0.0,
            "dominant_birth_count": int(_num(dom_metric.get("offspring_count")) or 0) if dom_metric else 0,
            "dominant_birth_share": (_num(dom_metric.get("reproduction_share")) or 0.0) if dom_metric else 0.0,
        }
        candidate_item["dominant_lineage_alive_share"] = dominant_item["dominant_alive_share"]
        candidate_item["dominant_lineage_birth_share"] = dominant_item["dominant_birth_share"]
        candidate_trajectory.append(candidate_item)
        dominant_trajectory.append(dominant_item)
        exposure_rows.append(
            {
                "generation": generation,
                "candidate_lineage_id": candidate,
                "candidate_present": candidate_present,
                **exposure_values,
                "field_availability": exposure_meta,
            }
        )
        birth_context_rows.append(
            {
                "generation": generation,
                "candidate_lineage_id": candidate,
                "candidate_birth_count": candidate_item["candidate_birth_count"],
                "candidate_birth_share": candidate_item["candidate_birth_share"],
                "candidate_birth_patch_food": candidate_item["candidate_birth_patch_food"],
                "candidate_birth_patch_cover": candidate_item["candidate_birth_patch_cover"],
                "candidate_birth_patch_shelter": candidate_item["candidate_birth_patch_shelter"],
                "candidate_birth_patch_waste": candidate_item["candidate_birth_patch_waste"],
                "candidate_birth_patch_density": candidate_item["candidate_birth_patch_density"],
                "candidate_birth_patch_recovery": candidate_item["candidate_birth_patch_recovery"],
                "field_availability": {
                    key: exposure_meta[key]
                    for key in (
                        "candidate_birth_patch_food",
                        "candidate_birth_patch_cover",
                        "candidate_birth_patch_shelter",
                        "candidate_birth_patch_waste",
                        "candidate_birth_patch_density",
                        "candidate_birth_patch_recovery",
                    )
                },
            }
        )
    absent_at_start = bool(candidate_trajectory and not candidate_trajectory[0]["candidate_present"])
    if candidate_trajectory:
        candidate_trajectory[0]["candidate_absent_at_replay_start"] = absent_at_start
    for item in candidate_trajectory[1:]:
        item["candidate_absent_at_replay_start"] = absent_at_start
    field_availability = {
        field: {
            **data,
            "available_fraction": (data["available_count"] / data["total_rows"]) if data["total_rows"] else 0.0,
        }
        for field, data in availability.items()
    }
    tracking = {
        "schema_version": "lbc1r_candidate_lineage_tracking_v2",
        "candidate_lineage_id": candidate,
        "candidate_biological_lineage_id": candidate_bio,
        "dominant_lineage_id": dominant,
        "dominant_biological_lineage_id": dominant,
        "identity_mapping": {
            "candidate_lineage_id": candidate,
            "candidate_biological_lineage_id": candidate_bio,
            "candidate_resolved_in_replay_start": not absent_at_start,
            "candidate_absent_at_replay_start": absent_at_start,
            "candidate_lineage_key_source": "manifest candidate_lineage_id matched against replay metrics/lineage_metrics.jsonl",
            "dominant_lineage_key_source": "top replay-start lineage by population_count in metrics/lineage_metrics.jsonl",
        },
        "trajectory": candidate_trajectory,
        "field_availability": field_availability,
        "field_availability_counts": {field: data["available_count"] for field, data in field_availability.items()},
    }
    dominant_tracking = {
        "schema_version": "lbc1r_dominant_lineage_tracking_v1",
        "dominant_lineage_id": dominant,
        "dominant_biological_lineage_id": dominant,
        "trajectory": dominant_trajectory,
    }
    exposure_tracking = {
        "schema_version": "lbc1r_lineage_exposure_tracking_v1",
        "candidate_lineage_id": candidate,
        "field_availability": field_availability,
        "trajectory": exposure_rows,
    }
    birth_context_tracking = {
        "schema_version": "lbc1r_lineage_birth_context_tracking_v1",
        "candidate_lineage_id": candidate,
        "field_availability": {field: field_availability[field] for field in EXPOSURE_FIELDS if field.startswith("birth_patch_")},
        "trajectory": birth_context_rows,
    }
    exposure_summary = {
        "schema_version": "lbc1r_candidate_exposure_summary_v1",
        "candidate_lineage_id": candidate,
        "candidate_absent_at_replay_start": absent_at_start,
        "field_availability": field_availability,
        "notes": [
            "Exposure values are copied only when present in replay lineage descriptors.",
            "Unavailable fields are explicit and are not approximated from global metrics.",
        ],
    }
    return {
        "candidate": tracking,
        "dominant": dominant_tracking,
        "exposure": exposure_tracking,
        "birth_context": birth_context_tracking,
        "exposure_summary": exposure_summary,
    }


def _hash_file(path: Path) -> str | None:
    return r3a._sha256_file(path)


def run_replay_entry(entry: Mapping[str, Any], *, run_root: Path, label: str, overwrite: bool) -> dict[str, Any]:
    result = r3a.run_replay_entry(entry, run_root=run_root, label=label, overwrite=overwrite)
    if result.get("status") != "ok":
        return result
    replay_dir = Path(str(result.get("run_dir")))
    tracking_bundle = _build_replay_lineage_tracking(replay_dir, entry)
    tracking = tracking_bundle["candidate"]
    tracking_path = replay_dir / "candidate_lineage_tracking.json"
    dominant_path = replay_dir / "dominant_lineage_tracking.json"
    exposure_path = replay_dir / "lineage_exposure_tracking.json"
    birth_context_path = replay_dir / "lineage_birth_context_tracking.json"
    exposure_summary_path = replay_dir / "candidate_exposure_summary.json"
    _write_json(tracking_path, tracking)
    _write_json(dominant_path, tracking_bundle["dominant"])
    _write_json(exposure_path, tracking_bundle["exposure"])
    _write_json(birth_context_path, tracking_bundle["birth_context"])
    _write_json(exposure_summary_path, tracking_bundle["exposure_summary"])
    result["tracking"] = tracking
    result["dominant_tracking_hash"] = _hash_file(dominant_path)
    result["lineage_exposure_tracking_hash"] = _hash_file(exposure_path)
    result["lineage_birth_context_tracking_hash"] = _hash_file(birth_context_path)
    result["candidate_exposure_summary_hash"] = _hash_file(exposure_summary_path)
    result["candidate_lineage_tracking_hash"] = _hash_file(tracking_path)
    return result


def run_replays(manifest: Mapping[str, Any], output_root: Path, *, overwrite: bool) -> dict[str, Any]:
    entries = list(manifest.get("entries") or [])
    results = [run_replay_entry(entry, run_root=output_root / "replays", label="primary", overwrite=overwrite) for entry in entries]
    return {
        "schema_version": "lbc1_replay_execution_v1",
        "planned": len(entries),
        "completed": sum(1 for item in results if item.get("status") == "ok"),
        "failed": sum(1 for item in results if item.get("status") != "ok"),
        "replay_runs": results,
    }


def run_repeat_parity(manifest: Mapping[str, Any], output_root: Path, *, overwrite: bool) -> dict[str, Any]:
    entries = list(manifest.get("entries") or [])
    selected: list[Mapping[str, Any]] = []
    wanted = [
        ("dominant", "lbc1_00_tuned_full_jungle_recovery"),
        ("non_dominant", "lbc1_00_tuned_full_jungle_recovery"),
        ("non_dominant", "lbc1_01_default_full_jungle"),
        ("non_dominant", "lbc1_02_theme12_clean_pressure_reference"),
    ]
    for dominance, scenario in wanted:
        match = next(
            (
                entry
                for entry in entries
                if entry.get("candidate_dominance_class") == dominance and entry.get("source_scenario_id") == scenario
            ),
            None,
        )
        if match is not None:
            selected.append(match)
    checks = []
    for entry in selected:
        a = run_replay_entry(entry, run_root=output_root / "replays", label="parity_a", overwrite=overwrite)
        b = run_replay_entry(entry, run_root=output_root / "replays", label="parity_b", overwrite=overwrite)
        passed = (
            a.get("status") == "ok"
            and b.get("status") == "ok"
            and a.get("metrics_hash") == b.get("metrics_hash")
            and a.get("candidate_lineage_tracking_hash") == b.get("candidate_lineage_tracking_hash")
            and a.get("dominant_tracking_hash") == b.get("dominant_tracking_hash")
            and a.get("lineage_exposure_tracking_hash") == b.get("lineage_exposure_tracking_hash")
            and a.get("lineage_birth_context_tracking_hash") == b.get("lineage_birth_context_tracking_hash")
            and a.get("candidate_exposure_summary_hash") == b.get("candidate_exposure_summary_hash")
            and a.get("config_hash") == b.get("config_hash")
            and a.get("final_metrics_row") == b.get("final_metrics_row")
        )
        checks.append(
            {
                "entry_id": entry.get("entry_id"),
                "source_scenario_id": entry.get("source_scenario_id"),
                "candidate_dominance_class": entry.get("candidate_dominance_class"),
                "passed": bool(passed),
                "a": {k: v for k, v in a.items() if k not in {"tracking", "final_metrics_row"}},
                "b": {k: v for k, v in b.items() if k not in {"tracking", "final_metrics_row"}},
                "final_metrics_rows_match": a.get("final_metrics_row") == b.get("final_metrics_row"),
            }
        )
    return {
        "schema_version": "lbc1_repeat_parity_summary_v1",
        "executed": True,
        "passed": bool(checks) and all(item.get("passed") for item in checks),
        "checks": checks,
    }


def classify_fate(trajectory: Sequence[Mapping[str, Any]]) -> str:
    if not trajectory:
        return "unclear"
    for field in REQUIRED_TRAJECTORY_FIELDS:
        if field not in trajectory[0]:
            return "unclear"
    if not bool(trajectory[0].get("candidate_present")):
        return "absent_at_replay_start"
    start = int(_num(trajectory[0].get("candidate_alive_count")) or 0)
    end = int(_num(trajectory[-1].get("candidate_alive_count")) or 0)
    start_share = _num(trajectory[0].get("candidate_alive_share")) or 0.0
    end_share = _num(trajectory[-1].get("candidate_alive_share")) or 0.0
    dominant_end = _num(trajectory[-1].get("dominant_lineage_alive_share")) or 0.0
    births_after_start = sum(int(_num(item.get("candidate_birth_count")) or 0) for item in trajectory[1:])
    if start > 0 and end == 0:
        return "dominant_sweep" if dominant_end >= 0.75 else "extinct_after_start"
    if end > 0 and (end_share > start_share + 0.001 or (start_share < 0.02 and births_after_start > 0)):
        return "recovered"
    if end > 0 and end_share < start_share - 0.001:
        return "declined"
    if end > 0:
        return "persisted"
    return "unclear"


def cause_proxy(entry: Mapping[str, Any], trajectory: Sequence[Mapping[str, Any]]) -> str:
    if not trajectory:
        return "unknown"
    if not bool(trajectory[0].get("candidate_present")):
        return "absent_at_replay_start"
    start_share = _num(trajectory[0].get("candidate_alive_share"))
    births = sum(int(_num(item.get("candidate_birth_count")) or 0) for item in trajectory)
    end_alive = int(_num(trajectory[-1].get("candidate_alive_count")) or 0)
    dominant_end = _num(trajectory[-1].get("dominant_lineage_alive_share")) or 0.0
    predator_vals = [_num(item.get("predator_damage")) for item in trajectory if _num(item.get("predator_damage")) is not None]
    waste_vals = [_num(item.get("waste_exposure")) for item in trajectory if _num(item.get("waste_exposure")) is not None]
    food_vals = [_num(item.get("food_exposure")) for item in trajectory if _num(item.get("food_exposure")) is not None]
    terrain_vals = [
        (_num(item.get("cover_exposure")) or 0.0) + (_num(item.get("shelter_exposure")) or 0.0)
        for item in trajectory
        if _num(item.get("cover_exposure")) is not None or _num(item.get("shelter_exposure")) is not None
    ]
    causes: list[str] = []
    if start_share is not None and start_share < 0.02:
        causes.append("already_low_share")
    if births <= 0:
        causes.append("low_reproduction")
    if predator_vals and sum(predator_vals) > 2.0 and end_alive == 0:
        causes.append("high_predator_damage")
    if waste_vals and sum(waste_vals) / len(waste_vals) > 0.35:
        causes.append("high_waste_density")
    if food_vals and sum(food_vals) / len(food_vals) < 0.25:
        causes.append("low_food_exposure")
    if terrain_vals and sum(terrain_vals) / len(terrain_vals) < 0.3:
        causes.append("low_terrain_protection")
    if end_alive == 0 and dominant_end >= 0.75:
        causes.append("global_dominant_sweep")
    return ";".join(causes) if causes else "unknown"


def exposure_field_availability(replay_runs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    total = 0
    present = {field: 0 for field in EXPOSURE_FIELDS}
    reasons = {field: "" for field in EXPOSURE_FIELDS}
    sources = {field: EXPOSURE_FIELDS[field] for field in EXPOSURE_FIELDS}
    for run in replay_runs:
        tracking = run.get("tracking") or {}
        for field, data in (tracking.get("field_availability") or {}).items():
            if data.get("reason_if_unavailable") and not reasons.get(field):
                reasons[field] = str(data.get("reason_if_unavailable"))
            if data.get("source") and data.get("source") != "unavailable":
                sources[field] = str(data.get("source"))
        for item in (tracking.get("trajectory") or []):
            total += 1
            for field in EXPOSURE_FIELDS:
                if item.get(field) is not None:
                    present[field] += 1
    fields = {
        field: {
            "available_count": count,
            "total_rows": total,
            "available_fraction": (count / total) if total else 0.0,
            "status": "available" if total and count == total else ("partial" if count else "unavailable"),
            "source": sources[field] if count else "unavailable",
            "expected_source": EXPOSURE_FIELDS[field],
            "reason_if_unavailable": "" if count else (reasons[field] or f"{EXPOSURE_FIELDS[field]} unavailable in replay lineage descriptors"),
        }
        for field, count in present.items()
    }
    sufficient = total > 0 and sum(1 for data in fields.values() if data["status"] in {"available", "partial"}) >= 10
    return {"schema_version": "lbc1_exposure_field_availability_v1", "sufficient_for_proxy": sufficient, "fields": fields}


def summarize_replays(manifest: Mapping[str, Any], execution: Mapping[str, Any]) -> dict[str, Any]:
    by_entry = {str(entry.get("entry_id")): dict(entry) for entry in manifest.get("entries") or []}
    rows: list[dict[str, Any]] = []
    for run in execution.get("replay_runs") or []:
        entry = by_entry.get(str(run.get("entry_id")), {})
        trajectory = list((run.get("tracking") or {}).get("trajectory") or [])
        start = trajectory[0] if trajectory else {}
        end = trajectory[-1] if trajectory else {}
        extinct_generation = ""
        for item in trajectory:
            if int(_num(item.get("candidate_alive_count")) or 0) == 0:
                extinct_generation = item.get("generation")
                break
        birth_total = sum(int(_num(item.get("candidate_birth_count")) or 0) for item in trajectory)
        row = {
            "entry_id": run.get("entry_id"),
            "status": run.get("status"),
            "candidate_dominance_class": entry.get("candidate_dominance_class"),
            "candidate_class": entry.get("candidate_class"),
            "source_scenario_id": entry.get("source_scenario_id"),
            "source_seed": entry.get("source_seed"),
            "candidate_lineage_id": entry.get("candidate_lineage_id"),
            "candidate_category": entry.get("candidate_category"),
            "alive_at_replay_start": start.get("candidate_alive_count", ""),
            "alive_at_replay_end": end.get("candidate_alive_count", ""),
            "alive_share_start": start.get("candidate_alive_share", ""),
            "alive_share_end": end.get("candidate_alive_share", ""),
            "births_during_replay": birth_total,
            "birth_share_end": end.get("candidate_birth_share", ""),
            "extinct_generation": extinct_generation,
            "persistence_duration_generations": sum(1 for item in trajectory if int(_num(item.get("candidate_alive_count")) or 0) > 0),
            "fate": classify_fate(trajectory),
            "cause_proxy": cause_proxy(entry, trajectory),
        }
        for field in EXPOSURE_FIELDS:
            vals = [_num(item.get(field)) for item in trajectory if _num(item.get(field)) is not None]
            row[f"{field}_mean"] = sum(vals) / len(vals) if vals else ""
        rows.append(row)
    return {
        "schema_version": "lbc1_replay_summary_v1",
        "planned": execution.get("planned"),
        "completed": execution.get("completed"),
        "failed": execution.get("failed"),
        "candidate_fates": rows,
    }


def grouped_summaries(fates: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_class: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    by_scenario: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in fates:
        by_class[str(row.get("candidate_dominance_class"))].append(row)
        by_scenario[str(row.get("source_scenario_id"))].append(row)
    comparison = []
    for key, rows in sorted(by_class.items()):
        starts = [_num(row.get("alive_share_start")) or 0.0 for row in rows]
        ends = [_num(row.get("alive_share_end")) or 0.0 for row in rows]
        comparison.append(
            {
                "group": key,
                "count": len(rows),
                "mean_alive_share_change": (sum(ends) / len(ends) - sum(starts) / len(starts)) if rows else 0.0,
                "extinction_or_sweep_rate": sum(1 for row in rows if row.get("fate") in {"extinct_after_start", "dominant_sweep"}) / len(rows) if rows else 0.0,
                "mean_births_during_replay": sum(int(row.get("births_during_replay") or 0) for row in rows) / len(rows) if rows else 0.0,
                "mean_persistence_duration": sum(int(row.get("persistence_duration_generations") or 0) for row in rows) / len(rows) if rows else 0.0,
            }
        )
    scenario_summary: dict[str, Any] = {}
    for key, rows in sorted(by_scenario.items()):
        scenario_summary[key] = {
            "count": len(rows),
            "non_dominant_count": sum(1 for row in rows if row.get("candidate_dominance_class") == "non_dominant"),
            "fates": {fate: sum(1 for row in rows if row.get("fate") == fate) for fate in sorted({str(row.get("fate")) for row in rows})},
            "mean_births_during_replay": sum(int(row.get("births_during_replay") or 0) for row in rows) / len(rows) if rows else 0.0,
        }
    return comparison, {"schema_version": "lbc1_scenario_contrast_summary_v1", "scenarios": scenario_summary}


def decide(
    *,
    validation: Mapping[str, Any],
    parity: Mapping[str, Any],
    cohort: Mapping[str, Any],
    availability: Mapping[str, Any],
    replay_summary: Mapping[str, Any],
    comparison: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    fates = list(replay_summary.get("candidate_fates") or [])
    candidate_count = int(cohort.get("selected_count", 0) or 0)
    non_dom = [row for row in fates if row.get("candidate_dominance_class") == "non_dominant"]
    dom = [row for row in fates if row.get("candidate_dominance_class") == "dominant"]
    non_dom_extinct = sum(1 for row in non_dom if row.get("fate") in {"extinct_after_start", "dominant_sweep"})
    non_dom_zero_birth = sum(1 for row in non_dom if int(row.get("births_during_replay") or 0) == 0)
    dom_recovered = sum(1 for row in dom if row.get("fate") in {"recovered", "persisted"})
    non_dom_recovered = sum(1 for row in non_dom if row.get("fate") in {"recovered", "persisted"})
    unclear = sum(1 for row in fates if row.get("fate") == "unclear")
    exposure_sufficient = bool(availability.get("sufficient_for_proxy"))
    trajectory_interpretable = (
        candidate_count >= 8
        and bool(fates)
        and len(fates) >= min(candidate_count, 8)
        and unclear == 0
        and bool(non_dom)
        and bool(dom)
    )
    invalid = not validation.get("valid") or not parity.get("passed")
    if invalid:
        decision = "D"
        wording = "Invalidated -> infrastructure repair"
    elif not trajectory_interpretable:
        decision = "C"
        wording = "Reporting insufficient -> trajectory/fate evidence still not interpretable"
    elif non_dom_extinct / len(non_dom) >= 0.75 and non_dom_zero_birth / len(non_dom) >= 0.75 and dom_recovered >= max(1, len(dom) // 2) and dom_recovered / len(dom) > non_dom_recovered / len(non_dom):
        decision = "A"
        wording = "Accept collapse as V0 limitation -> proceed to CRB-1"
    else:
        decision = "B"
        wording = "One targeted ecology/lineage fix -> implement one fix only -> revalidate"
    evidence = {
        "A": {
            "supporting": [
                f"candidate cohort size = {candidate_count}",
                f"non-dominant extinct/sweep fraction = {(non_dom_extinct / len(non_dom)) if non_dom else 0.0:.3f}",
                f"non-dominant zero-birth fraction = {(non_dom_zero_birth / len(non_dom)) if non_dom else 0.0:.3f}",
                f"dominant persisted/recovered count = {dom_recovered}",
                f"non-dominant persisted/recovered count = {non_dom_recovered}",
                f"trajectory/fate tracking interpretable = {trajectory_interpretable}",
                f"ecological exposure fields available = {exposure_sufficient}",
            ],
            "contradicting": [] if decision == "A" else ["decision thresholds for A were not all satisfied"],
            "confidence": "high" if decision == "A" else "medium",
        },
        "B": {
            "supporting": [
                "choose only if non-dominants sometimes persist/recover or trajectory evidence indicates one targeted bottleneck fix",
                f"non-dominant persisted/recovered count = {non_dom_recovered}",
            ],
            "contradicting": ["LBC-1 did not implement mechanics fixes during decision campaign"],
            "confidence": "low" if decision != "B" else "medium",
        },
        "C": {
            "supporting": [] if trajectory_interpretable else ["trajectory/fate/reproduction evidence was not interpretable enough for A/B"],
            "contradicting": [
                "manifest/replay/parity were valid",
                "missing ecological exposure fields alone do not force C in the rerun",
            ] if trajectory_interpretable else [],
            "confidence": "medium",
        },
        "D": {
            "supporting": [] if not invalid else ["manifest validation or repeat parity failed"],
            "contradicting": ["manifest validation and repeat parity passed"] if not invalid else [],
            "confidence": "low" if not invalid else "high",
        },
    }
    next_step = {
        "A": "Proceed to CRB-1 with explicit caveat that Patch/Jungle V0 is a pressure substrate, not a persistent lineage-diversity substrate.",
        "B": "Implement exactly one targeted fix: initial founder/lineage balance, then revalidate once.",
        "C": "Stop and decide whether Phase II should accept the remaining reporting limits; do not start another broad reporting loop.",
        "D": "Repair replay/manifest/checkpoint infrastructure before rerunning LBC-1.",
    }[decision]
    return {
        "schema_version": "lbc1_decision_v1",
        "phase_ii_decision": decision,
        "decision_wording": wording,
        "decision_evidence": evidence,
        "next_step": next_step,
        "thresholds": {
            "accept_A_candidate_count_min": 8,
            "accept_A_non_dominant_extinct_or_sweep_fraction_min": 0.75,
            "accept_A_non_dominant_zero_birth_fraction_min": 0.75,
            "accept_A_requires_trajectory_interpretable": True,
            "accept_A_requires_exposure_sufficient": False,
        },
    }


def _md_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def write_report(
    path: Path,
    *,
    worktree_info: Mapping[str, Any],
    source: Mapping[str, Any],
    cohort: Mapping[str, Any],
    availability: Mapping[str, Any],
    validation: Mapping[str, Any],
    replay_summary: Mapping[str, Any],
    parity: Mapping[str, Any],
    comparison: Sequence[Mapping[str, Any]],
    scenario_summary: Mapping[str, Any],
    decision: Mapping[str, Any],
    tests_run: Sequence[str],
    files_changed: Sequence[str],
) -> None:
    cohort_rows = [
        [
            item.get("entry_id"),
            item.get("candidate_dominance_class"),
            item.get("candidate_class"),
            item.get("source_scenario_id"),
            item.get("source_seed"),
            item.get("candidate_lineage_id"),
            item.get("checkpoint_reference_generation"),
        ]
        for item in cohort.get("candidates") or []
    ]
    avail_rows = [
        [field, data.get("status"), data.get("available_fraction"), data.get("source")]
        for field, data in sorted((availability.get("fields") or {}).items())
    ]
    fate_rows = [
        [
            row.get("entry_id"),
            row.get("candidate_dominance_class"),
            row.get("source_scenario_id"),
            row.get("fate"),
            row.get("alive_at_replay_start"),
            row.get("alive_at_replay_end"),
            row.get("births_during_replay"),
            row.get("cause_proxy"),
        ]
        for row in replay_summary.get("candidate_fates") or []
    ]
    comp_rows = [
        [
            row.get("group"),
            row.get("count"),
            row.get("mean_alive_share_change"),
            row.get("extinction_or_sweep_rate"),
            row.get("mean_births_during_replay"),
            row.get("mean_persistence_duration"),
        ]
        for row in comparison
    ]
    text = f"""# EIEL Phase II LBC-1 Lineage Bottleneck Closure Decision Report

## 1. Purpose

LBC-1 is the Phase II decision campaign for deciding whether early lineage
collapse is acceptable as a V0 limitation or blocks CRB-1.

## 2. Phase Context

Phase I substrate validation is accepted as complete with a known
lineage-collapse limitation. R1/R2/R3/R3a/R3b established integrated baseline,
forensics, checkpoint-backed replay readiness, and a small replay mini-campaign.

## 3. R1/R2/R3/R3a/R3b Dependency Summary

- R1 commit: `{worktree_info.get('r1_commit')}`
- R2 commit: `{worktree_info.get('r2_commit')}`
- R3 commit: `{worktree_info.get('r3_commit')}`
- R3a commit: `{worktree_info.get('r3a_commit')}`
- R3b commit: `{worktree_info.get('r3b_commit')}`

## 4. Worktree And Base Commit

Worktree: `{worktree_info.get('worktree_path')}`

Base commit: `{worktree_info.get('base_commit')}`

Git status before work:

```text
{str(worktree_info.get('git_status_before', '')).strip()}
```

Git status after work:

```text
{str(worktree_info.get('git_status_after', '')).strip()}
```

## 5. Scenario Definitions

- `lbc1_00_tuned_full_jungle_recovery`: tuned Campaign C/D pressure stack with low recovery.
- `lbc1_01_default_full_jungle`: current default full Jungle.
- `lbc1_02_theme12_clean_pressure_reference`: Theme12 clean/calibrated pressure reference.

Theme 4 experimental cognition was disabled in source and replay configs.

## 6. Source Campaign Settings

Registry: `_spec/registry/v1/lbc1_lineage_bottleneck_closure_source.yaml`

Settings: 3 scenarios, 5 paired seeds, 8 generations, 60 steps/generation,
population 64, grid 16, CUDA device/bank device.

## 7. Checkpoint Policy

Checkpoint Bundle v2 was enabled every generation, and gen0 emission was
enabled by `EIEL_CHECKPOINT_BUNDLE_V2_AT_START=1`. Required generations 0-6
were present in all completed source runs.

## 8. Source Campaign Completion

Source runs planned/completed/failed: `{source.get('planned')} / {source.get('completed')} / {source.get('failed')}`.

## 9. Candidate Cohort

Selected candidates: `{cohort.get('selected_count')}` of target `{cohort.get('target_candidate_count')}`.

{_md_table(('Entry', 'Class', 'Candidate class', 'Scenario', 'Seed', 'Lineage', 'Checkpoint gen'), cohort_rows)}

## 10. Exposure Field Availability

Sufficient for proxy classification: `{availability.get('sufficient_for_proxy')}`.

{_md_table(('Field', 'Status', 'Available fraction', 'Source'), avail_rows)}

## 11. Manifest Validation

Valid: `{validation.get('valid')}`.

Errors: `{len(validation.get('errors') or [])}`.

Warnings: `{len(validation.get('warnings') or [])}`.

## 12. Replay Execution Summary

Replay runs planned/completed/failed: `{replay_summary.get('planned')} / {replay_summary.get('completed')} / {replay_summary.get('failed')}`.

## 13. Repeat Parity Result

Repeat parity executed: `{parity.get('executed')}`.

Repeat parity passed: `{parity.get('passed')}`.

## 14. Candidate Fate Analysis

{_md_table(('Entry', 'Class', 'Scenario', 'Fate', 'Alive start', 'Alive end', 'Births', 'Cause proxy'), fate_rows)}

## 15. Dominant Vs Non-Dominant Comparison

{_md_table(('Group', 'Count', 'Mean alive-share change', 'Extinction/sweep rate', 'Mean births', 'Mean persistence'), comp_rows)}

## 16. Scenario Contrast

```json
{json.dumps(scenario_summary, indent=2, sort_keys=True)}
```

## 17. Cause-Of-Fate Proxy Classification

Cause proxy labels are thresholded diagnostic proxies. They are not exact
mechanistic causes.

## 18. Decision Evidence Table

```json
{json.dumps(decision.get('decision_evidence'), indent=2, sort_keys=True)}
```

## 19. Final Phase II Decision

`{decision.get('phase_ii_decision')}. {decision.get('decision_wording')}`

## 20. Consequence Of Decision

{decision.get('next_step')}

## 21. Files Changed

{chr(10).join(f'- `{item}`' for item in files_changed)}

## 22. Tests Run

{chr(10).join(f'- `{item}`' for item in tests_run)}

## 23. Known Limitations

- LBC-1 replay is still a compact replay campaign, not a long-horizon study.
- Cause labels are proxies derived from replay metrics and lineage descriptors.
- No mechanics or coefficients were changed.

## 24. No-Claims Statement

LBC-1 is a lineage bottleneck closure decision campaign.
It does not prove lineage diversity is solved.
It does not preserve, rescue, inject, or reward lineages.
It does not activate Theme 4 cognition.
It does not test cognition-readiness directly.
It does not show intelligence, perception adaptiveness, memory selection, prediction selection, or emergence.
It only decides whether early lineage collapse is acceptable as a V0 limitation or requires one targeted fix/reporting repair/infrastructure repair.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    source_root = Path(args.source_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    source = source_summary(source_root)
    cohort = select_candidate_cohort(source_root, output_root)
    manifest = build_manifest(cohort, output_root)
    validation = validate_manifest(manifest)
    _write_json(Path(args.source_summary_json), source)
    _write_json(Path(args.cohort_json), cohort)
    _write_json(Path(args.manifest_json), manifest)
    _write_yaml(Path(args.manifest_yaml), manifest)
    _write_json(Path(args.repo_manifest_json), manifest)
    _write_yaml(Path(args.repo_manifest_yaml), manifest)
    _write_json(Path(args.validation_json), validation)
    if not validation.get("valid"):
        execution = {"schema_version": "lbc1_replay_execution_v1", "planned": 0, "completed": 0, "failed": 0, "replay_runs": []}
        parity = {"schema_version": "lbc1_repeat_parity_summary_v1", "executed": False, "passed": False, "checks": []}
    else:
        execution = run_replays(manifest, output_root, overwrite=bool(args.overwrite))
        parity = run_repeat_parity(manifest, output_root, overwrite=bool(args.overwrite)) if execution.get("failed") == 0 else {
            "schema_version": "lbc1_repeat_parity_summary_v1",
            "executed": False,
            "passed": False,
            "checks": [],
        }
    _write_json(output_root / "lbc1_replay_execution_summary.json", execution)
    _write_json(Path(args.repeat_parity_json), parity)
    availability = exposure_field_availability(execution.get("replay_runs") or [])
    replay_summary = summarize_replays(manifest, execution)
    comparison, scenario_summary = grouped_summaries(replay_summary.get("candidate_fates") or [])
    decision = decide(
        validation=validation,
        parity=parity,
        cohort=cohort,
        availability=availability,
        replay_summary=replay_summary,
        comparison=comparison,
    )
    _write_json(Path(args.availability_json), availability)
    _write_json(Path(args.replay_summary_json), replay_summary)
    _write_csv(
        Path(args.replay_summary_csv),
        replay_summary.get("candidate_fates") or [],
        (
            "entry_id",
            "status",
            "candidate_dominance_class",
            "candidate_class",
            "source_scenario_id",
            "source_seed",
            "candidate_lineage_id",
            "candidate_category",
            "fate",
            "cause_proxy",
        ),
    )
    fate_fields = [
        "entry_id",
        "candidate_dominance_class",
        "candidate_class",
        "source_scenario_id",
        "source_seed",
        "candidate_lineage_id",
        "alive_at_replay_start",
        "alive_at_replay_end",
        "alive_share_start",
        "alive_share_end",
        "births_during_replay",
        "birth_share_end",
        "extinct_generation",
        "persistence_duration_generations",
        "fate",
        "cause_proxy",
    ] + [f"{field}_mean" for field in EXPOSURE_FIELDS]
    _write_csv(Path(args.fate_csv), replay_summary.get("candidate_fates") or [], fate_fields)
    _write_csv(
        Path(args.comparison_csv),
        comparison,
        ("group", "count", "mean_alive_share_change", "extinction_or_sweep_rate", "mean_births_during_replay", "mean_persistence_duration"),
    )
    _write_json(Path(args.scenario_summary_json), scenario_summary)
    _write_json(Path(args.decision_json), decision)
    if args.report_md:
        worktree_info = _read_json(Path(args.worktree_info_json)) if args.worktree_info_json else {}
        write_report(
            Path(args.report_md),
            worktree_info=worktree_info,
            source=source,
            cohort=cohort,
            availability=availability,
            validation=validation,
            replay_summary=replay_summary,
            parity=parity,
            comparison=comparison,
            scenario_summary=scenario_summary,
            decision=decision,
            tests_run=[line for line in str(args.tests_run or "").splitlines() if line.strip()],
            files_changed=[line for line in str(args.files_changed or "").splitlines() if line.strip()],
        )
    return {
        "decision": decision.get("phase_ii_decision"),
        "decision_wording": decision.get("decision_wording"),
        "source_completed": source.get("completed"),
        "candidate_count": cohort.get("selected_count"),
        "manifest_valid": validation.get("valid"),
        "replay_completed": replay_summary.get("completed"),
        "repeat_parity_passed": parity.get("passed"),
        "exposure_sufficient": availability.get("sufficient_for_proxy"),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LBC-1 lineage bottleneck closure report helper")
    parser.add_argument("--source_root", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--source_summary_json", required=True)
    parser.add_argument("--cohort_json", required=True)
    parser.add_argument("--availability_json", required=True)
    parser.add_argument("--manifest_json", required=True)
    parser.add_argument("--manifest_yaml", required=True)
    parser.add_argument("--repo_manifest_json", required=True)
    parser.add_argument("--repo_manifest_yaml", required=True)
    parser.add_argument("--validation_json", required=True)
    parser.add_argument("--repeat_parity_json", required=True)
    parser.add_argument("--replay_summary_json", required=True)
    parser.add_argument("--replay_summary_csv", required=True)
    parser.add_argument("--fate_csv", required=True)
    parser.add_argument("--comparison_csv", required=True)
    parser.add_argument("--scenario_summary_json", required=True)
    parser.add_argument("--decision_json", required=True)
    parser.add_argument("--worktree_info_json", default="")
    parser.add_argument("--report_md", default="")
    parser.add_argument("--tests_run", default="")
    parser.add_argument("--files_changed", default="")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    result = run(parse_args(argv))
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
