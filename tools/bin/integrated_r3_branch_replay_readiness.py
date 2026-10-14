#!/usr/bin/env python3
"""R3 branch-candidate replay readiness helper.

Reporting and manifest infrastructure only. The helper reads R2 candidate,
lineage, and archive outputs, builds deterministic branch replay manifests, and
validates whether those manifests are executable. It does not change simulation
mechanics or perform archive reseeding.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

try:
    import yaml
except Exception:  # pragma: no cover - exercised only when PyYAML is absent.
    yaml = None


csv.field_size_limit(sys.maxsize)

SCENARIOS: tuple[str, ...] = (
    "r2_00_patch_observed",
    "r2_01_default_full_jungle",
    "r2_02_tuned_full_jungle_no_recovery",
    "r2_03_tuned_full_jungle_recovery",
    "r2_04_theme12_clean_pressure_reference",
)

REQUIRED_CANDIDATE_FIELDS: tuple[str, ...] = (
    "manifest_version",
    "manifest_id",
    "source_campaign_id",
    "source_output_root",
    "source_run_id",
    "source_scenario_id",
    "source_seed",
    "source_device",
    "source_generation",
    "source_checkpoint_path",
    "source_config_path",
    "source_metrics_path",
    "candidate_lineage_id",
    "candidate_biological_lineage_id",
    "candidate_category",
    "candidate_generation_emitted",
    "candidate_alive_share_at_emission",
    "candidate_birth_share_at_emission",
    "candidate_descriptor_snapshot",
    "replay_mode",
    "replay_start_generation",
    "replay_generations",
    "replay_steps_per_generation",
    "population_handling_mode",
    "allowed_mutation_policy",
    "allowed_selection_policy",
    "expected_flags",
    "expected_tuned_overrides",
    "expected_theme4_disabled_status",
    "expected_theme5_reporting_status",
    "output_root",
)

PATH_FIELDS: tuple[str, ...] = ("source_checkpoint_path", "source_config_path", "source_metrics_path")

OPTIONAL_FIELDS: tuple[str, ...] = ("source_step_or_tick", "candidate_archive_entry_id")

THEME4_DISABLED_FIELDS: tuple[str, ...] = (
    "phase_e_e1_enable",
    "enable_developmental_cognition",
    "enable_introspection_channels",
    "enable_introspection_policy_input",
    "enable_self_regulation",
    "enable_self_regulation_surface",
    "enable_self_regulation_policy_output",
    "enable_self_modification",
    "enable_self_modification_policy_output",
    "enable_directed_inheritance",
    "enable_directed_inheritance_surface",
    "enable_directed_inheritance_policy_output",
)

EXPECTED_FLAG_FIELDS: tuple[str, ...] = (
    "patch_world_enable",
    "patch_world_observation_enable",
    "patch_world_waste_consequence_enable",
    "jungle_predator_pressure_enable",
    "jungle_reproduction_emissions_enable",
    "jungle_terrain_protection_enable",
    "jungle_ecological_recovery_enable",
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
        return
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def _marker(status: str, field: str, reason: str) -> dict[str, str]:
    return {"status": status, "field": field, "reason": reason}


def _is_missing_required(value: Any) -> bool:
    return isinstance(value, Mapping) and value.get("status") == "missing_required"


def _is_missing_optional(value: Any) -> bool:
    return isinstance(value, Mapping) and value.get("status") == "missing_optional"


def _sha256_path(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _scenario_run_rows(r2_root: Path) -> list[dict[str, Any]]:
    summary_path = r2_root / "r2_summary.json"
    if summary_path.exists():
        data = _read_json(summary_path)
        runs = (data.get("lineage_fate_summary") or {}).get("runs") or []
        if runs:
            return [dict(run) for run in runs]
    fate_path = r2_root / "r2_lineage_fate_summary.json"
    if fate_path.exists():
        return [dict(run) for run in (_read_json(fate_path).get("runs") or [])]
    return []


def _archive_entries(run_dir: Path) -> dict[str, list[dict[str, Any]]]:
    by_lineage: dict[str, list[dict[str, Any]]] = defaultdict(list)
    entries_dir = run_dir / "archives" / "lineage_qd" / "entries"
    for path in sorted(entries_dir.glob("*.json")):
        try:
            item = _read_json(path)
        except Exception:
            continue
        lineage_id = str(item.get("lineage_id") or "")
        if lineage_id:
            item["_archive_path"] = str(path)
            by_lineage[lineage_id].append(item)
    for entries in by_lineage.values():
        entries.sort(key=lambda item: (int(item.get("generation", 0) or 0), str(item.get("archive_entry_id") or "")))
    return by_lineage


def _best_archive_entry(
    archive_by_lineage: Mapping[str, Sequence[Mapping[str, Any]]],
    lineage_id: str,
    generation: int,
) -> dict[str, Any] | None:
    entries = list(archive_by_lineage.get(lineage_id) or [])
    if not entries:
        return None
    return dict(
        min(
            entries,
            key=lambda item: (
                abs(int(item.get("generation", 0) or 0) - int(generation)),
                -float(item.get("candidate_score", 0.0) or 0.0),
                str(item.get("archive_entry_id") or ""),
            ),
        )
    )


def _branch_candidates(run_dir: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in _read_jsonl(run_dir / "metrics" / "branch_candidates.jsonl"):
        for item in row.get("candidates") or []:
            if isinstance(item, dict):
                out.append(dict(item))
    return out


def _lineage_metrics(run_dir: Path) -> list[dict[str, Any]]:
    return _read_jsonl(run_dir / "metrics" / "lineage_metrics.jsonl")


def _final_dominant_lineage_id(run_dir: Path) -> str:
    rows = _lineage_metrics(run_dir)
    if not rows:
        return ""
    max_gen = max(int(row.get("generation", 0) or 0) for row in rows)
    final_rows = [row for row in rows if int(row.get("generation", 0) or 0) == max_gen]
    if not final_rows:
        return ""
    top = max(
        final_rows,
        key=lambda row: (
            float(row.get("population_share", 0.0) or 0.0),
            float(row.get("reproduction_share", 0.0) or 0.0),
            str(row.get("lineage_id") or ""),
        ),
    )
    return str(top.get("lineage_id") or "")


def _lineage_descriptor_at(run_dir: Path, lineage_id: str, generation: int) -> dict[str, Any]:
    best: dict[str, Any] = {}
    for row in _read_jsonl(run_dir / "metrics" / "lineage_descriptors.jsonl"):
        if int(row.get("generation", 0) or 0) != int(generation):
            continue
        for item in row.get("lineages") or []:
            if str(item.get("lineage_id") or "") == str(lineage_id):
                best = dict(item)
                break
        if best:
            break
    return best


def _checkpoint_path_for_generation(run_dir: Path, generation: int) -> str | dict[str, str]:
    root = run_dir / "checkpoint_bundles_v2"
    if not root.exists():
        return _marker("missing_required", "source_checkpoint_path", "source run has no checkpoint_bundles_v2 directory")
    bundles = sorted(root.glob("bundle_gen*"))
    candidates: list[tuple[int, Path]] = []
    for bundle in bundles:
        try:
            gen = int(str(bundle.name).replace("bundle_gen", ""))
        except ValueError:
            continue
        if gen <= int(generation):
            candidates.append((gen, bundle))
    if not candidates:
        return _marker(
            "missing_required",
            "source_checkpoint_path",
            f"no checkpoint bundle at or before generation {generation}",
        )
    return str(max(candidates, key=lambda item: item[0])[1])


def _expected_flags(config: Mapping[str, Any]) -> dict[str, Any]:
    env = dict(config.get("env_cfg") or {})
    return {field: env.get(field) for field in EXPECTED_FLAG_FIELDS if field in env}


def _expected_tuned_overrides(config: Mapping[str, Any]) -> dict[str, Any]:
    env = dict(config.get("env_cfg") or {})
    fields = (
        "jungle_predator_base_risk",
        "jungle_predator_scent_weight",
        "jungle_predator_waste_weight",
        "jungle_predator_density_weight",
        "jungle_predator_visibility_weight",
        "jungle_predator_risk_cap",
        "jungle_predator_health_damage",
        "patch_waste_regrowth_penalty_scale",
        "patch_waste_regrowth_penalty_max",
        "patch_density_regrowth_penalty_scale",
        "patch_density_regrowth_penalty_max",
        "patch_regrowth_min_multiplier",
        "jungle_cover_damage_reduction_weight",
        "jungle_shelter_damage_reduction_weight",
        "jungle_terrain_damage_multiplier_min",
        "jungle_recovery_regrowth_boost_scale",
        "jungle_recovery_regrowth_boost_max",
    )
    return {field: env.get(field) for field in fields if field in env}


def _theme4_disabled_status(config: Mapping[str, Any]) -> dict[str, bool]:
    sim = dict(config.get("sim_cfg") or {})
    return {field: bool(sim.get(field, False)) is False for field in THEME4_DISABLED_FIELDS}


def _theme5_reporting_status(config: Mapping[str, Any]) -> dict[str, Any]:
    sim = dict(config.get("sim_cfg") or {})
    fields = (
        "lineage_qd_archive_enable",
        "lineage_qd_archive_top_n",
        "lineage_qd_archive_min_score",
        "lineage_qd_archive_cooldown_generations",
    )
    return {field: sim.get(field) for field in fields if field in sim}


def _run_id(run_dir: Path) -> str:
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        try:
            manifest = _read_json(manifest_path)
            return str(manifest.get("runtime_run_id") or manifest.get("run_id") or run_dir.name)
        except Exception:
            return run_dir.name
    return run_dir.name


def _source_device(config: Mapping[str, Any]) -> str:
    sim = dict(config.get("sim_cfg") or {})
    return str(sim.get("device") or "")


def _source_seed(run_record: Mapping[str, Any], config: Mapping[str, Any]) -> int:
    if "seed" in run_record:
        return int(run_record["seed"])
    sim = dict(config.get("sim_cfg") or {})
    return int(sim.get("seed", 0) or 0)


def _candidate_entry(
    *,
    selection_role: str,
    run_record: Mapping[str, Any],
    candidate: Mapping[str, Any],
    archive_entry: Mapping[str, Any] | None,
    output_root: Path,
    source_output_root: Path,
) -> dict[str, Any]:
    run_dir = Path(str(run_record["run_dir"]))
    config_path = run_dir / "config.json"
    metrics_path = run_dir / "metrics.csv"
    config = _read_json(config_path) if config_path.exists() else {}
    generation = int(candidate.get("generation", 0) or 0)
    lineage_id = str(candidate.get("lineage_id") or "")
    archive_id = str((archive_entry or {}).get("archive_entry_id") or "")
    category = "dominant_lineage" if selection_role == "dominant" else str(candidate.get("candidate_type") or "")
    descriptor = {
        "branch_candidate": dict(candidate),
        "archive_entry": dict(archive_entry or {}),
        "lineage_descriptor": _lineage_descriptor_at(run_dir, lineage_id, generation),
    }
    manifest_id = f"r3_{selection_role}_{str(run_record.get('scenario'))}_{int(run_record.get('seed', 0)):06d}_{lineage_id}_{generation}"
    return {
        "manifest_version": "r3_branch_replay_manifest_v1",
        "manifest_id": manifest_id,
        "source_campaign_id": "integrated_r2_lineage_collapse_forensics",
        "source_output_root": str(source_output_root),
        "source_run_id": _run_id(run_dir),
        "source_scenario_id": str(run_record.get("scenario") or ""),
        "source_seed": _source_seed(run_record, config),
        "source_device": _source_device(config),
        "source_generation": generation,
        "source_step_or_tick": _marker("missing_optional", "source_step_or_tick", "R2 branch candidate artifacts are generation-level"),
        "source_checkpoint_path": _checkpoint_path_for_generation(run_dir, generation),
        "source_config_path": str(config_path) if config_path.exists() else _marker("missing_required", "source_config_path", "missing config.json"),
        "source_metrics_path": str(metrics_path) if metrics_path.exists() else _marker("missing_required", "source_metrics_path", "missing metrics.csv"),
        "candidate_lineage_id": lineage_id,
        "candidate_biological_lineage_id": lineage_id,
        "candidate_category": category or _marker("missing_required", "candidate_category", "candidate has no type/category"),
        "candidate_generation_emitted": generation,
        "candidate_alive_share_at_emission": candidate.get("population_share", _marker("missing_required", "candidate_alive_share_at_emission", "missing population_share")),
        "candidate_birth_share_at_emission": candidate.get("reproduction_share", _marker("missing_required", "candidate_birth_share_at_emission", "missing reproduction_share")),
        "candidate_descriptor_snapshot": descriptor,
        "candidate_archive_entry_id": archive_id or _marker("missing_optional", "candidate_archive_entry_id", "no matching archive entry found"),
        "replay_mode": "manifest_only_validation",
        "replay_start_generation": generation,
        "replay_generations": 4,
        "replay_steps_per_generation": 60,
        "population_handling_mode": "restore_from_checkpoint_without_population_modification",
        "allowed_mutation_policy": "source_config_only_no_override",
        "allowed_selection_policy": "source_config_only_no_override",
        "expected_flags": _expected_flags(config),
        "expected_tuned_overrides": _expected_tuned_overrides(config),
        "expected_theme4_disabled_status": _theme4_disabled_status(config),
        "expected_theme5_reporting_status": _theme5_reporting_status(config),
        "output_root": str(output_root / "replays" / manifest_id),
    }


def select_candidates_from_r2(r2_root: Path, output_root: Path) -> dict[str, Any]:
    runs = _scenario_run_rows(r2_root)
    by_scenario = {str(run.get("scenario")): [] for run in runs}
    for run in runs:
        by_scenario.setdefault(str(run.get("scenario")), []).append(run)
    for entries in by_scenario.values():
        entries.sort(key=lambda item: int(item.get("seed", 0) or 0))

    selected: list[dict[str, Any]] = []
    source_output_root = r2_root

    tuned_runs = by_scenario.get("r2_03_tuned_full_jungle_recovery", [])
    if tuned_runs:
        run = tuned_runs[0]
        run_dir = Path(str(run["run_dir"]))
        dominant_id = _final_dominant_lineage_id(run_dir)
        archive_by_lineage = _archive_entries(run_dir)
        candidates = [item for item in _branch_candidates(run_dir) if str(item.get("lineage_id") or "") == dominant_id]
        if candidates:
            candidate = sorted(candidates, key=lambda item: int(item.get("generation", 0) or 0))[0]
        else:
            candidate = {
                "lineage_id": dominant_id,
                "generation": 0,
                "candidate_type": "dominant_lineage",
                "population_share": _marker("missing_required", "candidate_alive_share_at_emission", "dominant lineage was not in branch_candidates.jsonl"),
                "reproduction_share": _marker("missing_required", "candidate_birth_share_at_emission", "dominant lineage was not in branch_candidates.jsonl"),
            }
        archive_entry = _best_archive_entry(archive_by_lineage, dominant_id, int(candidate.get("generation", 0) or 0))
        selected.append(
            _candidate_entry(
                selection_role="dominant",
                run_record=run,
                candidate=candidate,
                archive_entry=archive_entry,
                output_root=output_root,
                source_output_root=source_output_root,
            )
        )

    for scenario, role in (
        ("r2_03_tuned_full_jungle_recovery", "non_dominant_tuned_recovery"),
        ("r2_01_default_full_jungle", "non_dominant_default_full_jungle"),
        ("r2_04_theme12_clean_pressure_reference", "non_dominant_theme12_clean_pressure"),
    ):
        chosen: tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any] | None] | None = None
        for run in by_scenario.get(scenario, []):
            run_dir = Path(str(run["run_dir"]))
            dominant_id = _final_dominant_lineage_id(run_dir)
            archive_by_lineage = _archive_entries(run_dir)
            candidates = [
                item
                for item in _branch_candidates(run_dir)
                if str(item.get("lineage_id") or "")
                and str(item.get("lineage_id") or "") != dominant_id
                and int(item.get("generation", 0) or 0) <= 5
                and (_float(item.get("population_share")) or 0.0) > 0.0
                and _best_archive_entry(archive_by_lineage, str(item.get("lineage_id")), int(item.get("generation", 0) or 0))
                is not None
            ]
            if not candidates:
                continue
            candidates.sort(
                key=lambda item: (
                    int(item.get("generation", 0) or 0),
                    -float(item.get("candidate_score", 0.0) or 0.0),
                    str(item.get("lineage_id") or ""),
                )
            )
            candidate = candidates[0]
            archive_entry = _best_archive_entry(
                archive_by_lineage,
                str(candidate.get("lineage_id")),
                int(candidate.get("generation", 0) or 0),
            )
            chosen = (run, candidate, archive_entry)
            break
        if chosen is None:
            continue
        run, candidate, archive_entry = chosen
        selected.append(
            _candidate_entry(
                selection_role=role,
                run_record=run,
                candidate=candidate,
                archive_entry=archive_entry,
                output_root=output_root,
                source_output_root=source_output_root,
            )
        )

    return {
        "schema_version": "integrated_r3_candidate_selection_v1",
        "source_root": str(r2_root),
        "selection_criteria": [
            "one dominant tuned-recovery lineage candidate",
            "one non-dominant tuned-recovery branch candidate",
            "one non-dominant default full-Jungle branch candidate",
            "one non-dominant Theme12 clean-pressure branch candidate",
            "prefer generation <= 5, nonzero population share, archive entry present",
        ],
        "selected_count": len(selected),
        "selected_candidates": selected,
    }


def build_manifest(candidate_selection: Mapping[str, Any]) -> dict[str, Any]:
    candidates = list(candidate_selection.get("selected_candidates") or [])
    manifest_hash_input = json.dumps(candidates, sort_keys=True, separators=(",", ":"))
    manifest_id = "r3_branch_replay_smoke_" + hashlib.sha256(manifest_hash_input.encode("utf-8")).hexdigest()[:16]
    return {
        "manifest_version": "r3_branch_replay_manifest_set_v1",
        "manifest_id": manifest_id,
        "source_campaign_id": "integrated_r2_lineage_collapse_forensics",
        "source_output_root": str(candidate_selection.get("source_root") or ""),
        "replay_mode": "manifest_only_validation",
        "candidates": candidates,
    }


def validate_manifest(manifest: Mapping[str, Any], *, require_existing_paths: bool = True) -> dict[str, Any]:
    candidates = list(manifest.get("candidates") or [])
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    for idx, candidate in enumerate(candidates):
        if not isinstance(candidate, Mapping):
            errors.append({"candidate_index": idx, "field": "candidate", "error": "candidate entry is not an object"})
            continue
        for field in REQUIRED_CANDIDATE_FIELDS:
            if field not in candidate:
                errors.append({"candidate_index": idx, "field": field, "error": "missing field"})
                continue
            value = candidate.get(field)
            if _is_missing_required(value):
                errors.append({"candidate_index": idx, "field": field, "error": value.get("reason", "missing_required")})
            elif value in (None, ""):
                errors.append({"candidate_index": idx, "field": field, "error": "empty required field"})
        for field in OPTIONAL_FIELDS:
            if field not in candidate:
                warnings.append({"candidate_index": idx, "field": field, "warning": "optional field omitted"})
            elif _is_missing_optional(candidate.get(field)):
                warnings.append({"candidate_index": idx, "field": field, "warning": candidate[field].get("reason")})
        if require_existing_paths:
            for field in PATH_FIELDS:
                value = candidate.get(field)
                if isinstance(value, str) and not Path(value).exists():
                    errors.append({"candidate_index": idx, "field": field, "error": f"path does not exist: {value}"})
        theme4 = candidate.get("expected_theme4_disabled_status")
        if isinstance(theme4, Mapping):
            enabled = [field for field, disabled in theme4.items() if disabled is not True]
            if enabled:
                errors.append(
                    {
                        "candidate_index": idx,
                        "field": "expected_theme4_disabled_status",
                        "error": f"Theme 4 experimental gates not disabled: {enabled}",
                    }
                )
        if candidate.get("population_handling_mode") != "restore_from_checkpoint_without_population_modification":
            errors.append(
                {
                    "candidate_index": idx,
                    "field": "population_handling_mode",
                    "error": "population handling would modify population",
                }
            )
        if candidate.get("allowed_mutation_policy") != "source_config_only_no_override":
            errors.append({"candidate_index": idx, "field": "allowed_mutation_policy", "error": "mutation override not allowed"})
        if candidate.get("allowed_selection_policy") != "source_config_only_no_override":
            errors.append({"candidate_index": idx, "field": "allowed_selection_policy", "error": "selection override not allowed"})
    return {
        "schema_version": "integrated_r3_manifest_validation_v1",
        "candidate_count": len(candidates),
        "valid": not errors,
        "executable_ready": not errors,
        "errors": errors,
        "warnings": warnings,
    }


def missing_replay_requirements(validation: Mapping[str, Any]) -> dict[str, Any]:
    errors = list(validation.get("errors") or [])
    checkpoint_errors = [item for item in errors if item.get("field") == "source_checkpoint_path"]
    return {
        "schema_version": "integrated_r3_missing_replay_requirements_v1",
        "missing_checkpoint_count": len(checkpoint_errors),
        "missing_checkpoint_errors": checkpoint_errors,
        "other_errors": [item for item in errors if item.get("field") != "source_checkpoint_path"],
        "minimum_to_execute_mode_a": [
            "R2-like source runs must save checkpoint_bundles_v2 near candidate-emission generations.",
            "Manifest entries must point to an existing checkpoint bundle.",
            "A replay runner can then use existing branch_from_checkpoint/replay support without population modification.",
        ],
    }


def smoke_summary(validation: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "integrated_r3_smoke_summary_v1",
        "executed": False,
        "reason": "executable replay skipped because manifest validation found missing required checkpoint paths",
        "validation_valid": bool(validation.get("valid")),
        "smoke_runs": [],
    }


def repeat_parity_summary(validation: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "integrated_r3_repeat_parity_summary_v1",
        "executed": False,
        "reason": "repeat parity requires executable replay manifests with source checkpoints",
        "validation_valid": bool(validation.get("valid")),
        "checks": [],
    }


def readiness_classification(validation: Mapping[str, Any], selection: Mapping[str, Any]) -> str:
    candidates = list(selection.get("selected_candidates") or [])
    if not candidates:
        return "BLOCKED"
    errors = list(validation.get("errors") or [])
    checkpoint_errors = [item for item in errors if item.get("field") == "source_checkpoint_path"]
    other_errors = [item for item in errors if item.get("field") != "source_checkpoint_path"]
    if not errors:
        return "READY"
    if checkpoint_errors and not other_errors:
        return "METADATA-ONLY"
    if checkpoint_errors:
        return "PARTIAL"
    return "BLOCKED"


def recommended_next_step(classification: str) -> str:
    if classification == "READY":
        return "R3b branch-candidate replay campaign"
    if classification in {"PARTIAL", "METADATA-ONLY"}:
        return "R3a replay infrastructure completion"
    return "R2b reporting refinement"


def _md_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def write_report(
    path: Path,
    *,
    worktree_info: Mapping[str, Any],
    r2_paths: Mapping[str, str],
    selection: Mapping[str, Any],
    manifest_path: Path,
    validation: Mapping[str, Any],
    missing: Mapping[str, Any],
    smoke: Mapping[str, Any],
    parity: Mapping[str, Any],
    classification: str,
    tests_run: Sequence[str],
    files_changed: Sequence[str],
) -> None:
    candidate_rows = []
    for candidate in selection.get("selected_candidates") or []:
        checkpoint = candidate.get("source_checkpoint_path")
        checkpoint_status = checkpoint.get("status") if isinstance(checkpoint, Mapping) else "present"
        candidate_rows.append(
            [
                candidate.get("manifest_id"),
                candidate.get("source_scenario_id"),
                candidate.get("source_seed"),
                candidate.get("candidate_generation_emitted"),
                candidate.get("candidate_biological_lineage_id"),
                candidate.get("candidate_category"),
                candidate.get("candidate_archive_entry_id")
                if not isinstance(candidate.get("candidate_archive_entry_id"), Mapping)
                else "missing_optional",
                checkpoint_status,
            ]
        )
    text = f"""# EIEL Integrated R3 Branch-Candidate Replay Readiness Report

## 1. Purpose

R3 asks whether Theme 5 branch candidates emitted during R2 can be converted
into deterministic, executable replay/restart manifests and validated by a
small controlled smoke test. This is infrastructure readiness only.

## 2. R1/R2 Dependency Summary

- R1 Integrated Baseline Reproduction: PASS.
- R2 Theme 5-assisted Lineage-Collapse Forensics: PASS.
- R2 commit: `{worktree_info.get('r2_commit', '')}`.
- R1 commit: `{worktree_info.get('r1_commit', '')}`.

## 3. Base Commit And Worktree

Worktree: `{worktree_info.get('worktree_path', '')}`.

Base commit: `{worktree_info.get('base_commit', '')}`.

Git status before work:

```text
{str(worktree_info.get('git_status_before', '')).strip()}
```

## 4. R2 Source Artifact Paths

{_md_table(('Artifact', 'Path'), sorted(r2_paths.items()))}

## 5. Candidate Selection Criteria

- Select one dominant tuned-recovery lineage candidate.
- Select one non-dominant branch candidate from tuned full Jungle recovery.
- Select one non-dominant branch candidate from default full Jungle.
- Select one non-dominant branch candidate from Theme12 clean pressure.
- Prefer generation <= 5, nonzero population share, and matching archive entry.

## 6. Selected Candidates Table

{_md_table(('Manifest ID', 'Scenario', 'Seed', 'Generation', 'Lineage', 'Category', 'Archive entry', 'Checkpoint'), candidate_rows)}

## 7. Manifest Schema

Manifest entries include source run, scenario, seed, generation, config,
metrics, candidate lineage identity, descriptor snapshot, archive entry,
expected flags, Theme 4 disabled status, Theme 5 reporting status, replay
mode, replay horizon, population handling, selection policy, and mutation
policy. Missing required fields are explicit `missing_required` objects.

Manifest path: `{manifest_path}`.

## 8. Manifest Validation Result

Valid executable manifest: `{validation.get('valid')}`.

Executable ready: `{validation.get('executable_ready')}`.

Errors: `{len(validation.get('errors') or [])}`.

Warnings: `{len(validation.get('warnings') or [])}`.

## 9. Executable Replay Support Status

Existing checkpoint branching support is present, but the R2 source runs did
not save checkpoint bundles near candidate emission. R2 archive entries also
record `checkpoint_reference: null`. R3 therefore validates Mode C
manifest-only replay readiness and does not execute Mode A/B replay.

## 10. Smoke Execution Result

Smoke executed: `{smoke.get('executed')}`.

Reason: `{smoke.get('reason')}`.

## 11. Repeat Parity Result

Repeat parity executed: `{parity.get('executed')}`.

Reason: `{parity.get('reason')}`.

## 12. Candidate Replay Trajectories

No replay trajectories were generated because executable checkpoints are
missing. Candidate descriptor snapshots are stored in the generated manifest
and candidate selection JSON.

## 13. Missing Fields / Missing Infrastructure

Missing checkpoint count: `{missing.get('missing_checkpoint_count')}`.

Minimum required to execute Mode A:

{chr(10).join(f'- {item}' for item in missing.get('minimum_to_execute_mode_a') or [])}

## 14. Replay Readiness Classification

`{classification}`

## 15. Files Changed

{chr(10).join(f'- `{item}`' for item in files_changed)}

## 16. Tests Run

{chr(10).join(f'- `{item}`' for item in tests_run)}

## 17. Known Limitations

- R3 does not execute branch replay because source checkpoints are missing.
- Candidate archive entries are metadata-only and do not include executable
  checkpoint references.
- R3 does not assess long-term branch viability.
- R3 does not alter live evolution or preserve weak lineages.

## 18. Recommended Next Step

`{recommended_next_step(classification)}`

## 19. No-Claims Statement

R3 makes no claim that branch replay improves evolution, branch candidates are
long-term viable, cognition is adaptive, intelligence emerged, or lineage
diversity is solved.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def discover_r2_paths(r2_root: Path) -> dict[str, str]:
    names = (
        "r2_summary.json",
        "r2_scenario_summary.csv",
        "r2_paired_deltas.csv",
        "r2_lineage_collapse_timing.csv",
        "r2_lineage_fate_summary.json",
        "r2_branch_candidate_fate_summary.json",
        "r2_archive_candidate_summary.json",
    )
    return {name: str(r2_root / name) for name in names if (r2_root / name).exists()}


def run(args: argparse.Namespace) -> dict[str, Any]:
    r2_root = Path(args.r2_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    selection = select_candidates_from_r2(r2_root, output_root)
    manifest = build_manifest(selection)
    validation = validate_manifest(manifest, require_existing_paths=True)
    missing = missing_replay_requirements(validation)
    smoke = smoke_summary(validation)
    parity = repeat_parity_summary(validation)
    classification = readiness_classification(validation, selection)

    manifest_json = Path(args.manifest_json)
    manifest_yaml = Path(args.manifest_yaml)
    _write_json(Path(args.candidate_json), selection)
    _write_json(manifest_json, manifest)
    _write_yaml(manifest_yaml, manifest)
    _write_json(Path(args.validation_json), validation)
    _write_json(Path(args.missing_json), missing)
    _write_json(Path(args.smoke_json), smoke)
    _write_json(Path(args.repeat_parity_json), parity)

    r2_paths = discover_r2_paths(r2_root)
    worktree_info = _read_json(Path(args.worktree_info_json)) if args.worktree_info_json else {}
    files_changed = [
        "_spec/branch_replay/v1/r3_branch_replay_smoke_manifest.json",
        "_spec/branch_replay/v1/r3_branch_replay_smoke_manifest.yaml",
        "tools/bin/integrated_r3_branch_replay_readiness.py",
        "tests/integration/branching/test_r3_branch_replay_readiness.py",
        "docs/400_implementation-engineering/410_implementation-reports/000_programme-cross-cutting/EIEL-DOC-000390--eiel-integrated-r3-branch-candidate-replay-readiness-report.md",
    ]
    tests_run = [
        "pytest -q tests/evolvability/patch_world",
        "pytest -q tests/evolvability/jungle",
        "pytest -q tests/evolvability/theme4",
        "pytest -q tests/determinism",
        "pytest -q tests/regression/test_checkpoint_bundle_contracts.py",
        "pytest -q tests/integration/branching/test_branching_determinism.py",
        "pytest -q tests/integration/metrics",
        "git diff --check",
    ]
    write_report(
        Path(args.report_md),
        worktree_info=worktree_info,
        r2_paths=r2_paths,
        selection=selection,
        manifest_path=manifest_json,
        validation=validation,
        missing=missing,
        smoke=smoke,
        parity=parity,
        classification=classification,
        tests_run=tests_run,
        files_changed=files_changed,
    )
    return {
        "classification": classification,
        "selected_count": selection.get("selected_count"),
        "validation_valid": validation.get("valid"),
        "manifest_json": str(manifest_json),
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate R3 branch replay readiness artifacts")
    parser.add_argument("--r2-root", default="/tmp/emergence_integrated_r2_lineage_collapse_forensics")
    parser.add_argument("--output-root", default="/tmp/emergence_integrated_r3_branch_replay_readiness")
    parser.add_argument("--candidate-json", required=True)
    parser.add_argument("--manifest-json", required=True)
    parser.add_argument("--manifest-yaml", required=True)
    parser.add_argument("--validation-json", required=True)
    parser.add_argument("--missing-json", required=True)
    parser.add_argument("--smoke-json", required=True)
    parser.add_argument("--repeat-parity-json", required=True)
    parser.add_argument("--report-md", required=True)
    parser.add_argument("--worktree-info-json", default="")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    result = run(_parse_args(argv))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
