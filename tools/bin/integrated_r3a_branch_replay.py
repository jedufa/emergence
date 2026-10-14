#!/usr/bin/env python3
"""R3a checkpoint-enabled branch replay helper.

This tool is orchestration/reporting infrastructure only. It links Theme 5
branch candidates from checkpoint-enabled source runs to checkpoint_bundles_v2
and optionally executes deterministic Mode A checkpoint continuations. It does
not modify population membership, selection, reproduction, mutation, ecology,
or cognition behavior.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


csv.field_size_limit(sys.maxsize)

SCENARIOS: tuple[str, ...] = (
    "r3a_00_tuned_full_jungle_recovery",
    "r3a_01_default_full_jungle",
    "r3a_02_theme12_clean_pressure_reference",
)

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

TUNED_OVERRIDE_FIELDS: tuple[str, ...] = (
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
    "jungle_recovery_fertility_enable",
    "jungle_resource_spread_enable",
)

REQUIRED_ENTRY_FIELDS: tuple[str, ...] = (
    "manifest_version",
    "manifest_id",
    "entry_id",
    "source_campaign_id",
    "source_output_root",
    "source_run_id",
    "source_scenario_id",
    "source_seed",
    "source_device",
    "source_generation",
    "source_checkpoint_path",
    "source_checkpoint_metadata_path",
    "source_config_path",
    "source_metrics_path",
    "candidate_lineage_id",
    "candidate_biological_lineage_id",
    "candidate_category",
    "candidate_generation_emitted",
    "candidate_alive_share_at_emission",
    "candidate_birth_share_at_emission",
    "candidate_descriptor_snapshot",
    "candidate_archive_entry_id",
    "replay_mode",
    "replay_start_generation",
    "replay_generations",
    "replay_steps_per_generation",
    "population_handling_mode",
    "allowed_mutation_policy",
    "allowed_selection_policy",
    "allowed_reproduction_policy",
    "allowed_ecology_policy",
    "expected_flags",
    "expected_tuned_overrides",
    "expected_theme4_disabled_status",
    "expected_theme5_reporting_status",
    "output_root",
)

PATH_FIELDS: tuple[str, ...] = (
    "source_checkpoint_path",
    "source_checkpoint_metadata_path",
    "source_config_path",
    "source_metrics_path",
)

FORBIDDEN_INTERVENTION_KEYS: tuple[str, ...] = (
    "lineage_protection",
    "lineage_rescue",
    "diversity_reward",
    "archive_reseed",
    "candidate_injection",
    "population_quota",
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


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
    return parsed if math.isfinite(parsed) else None


def _sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_tree(path: Path) -> str | None:
    if not path.exists():
        return None
    if path.is_file():
        return _sha256_file(path)
    h = hashlib.sha256()
    for item in sorted(p for p in path.rglob("*") if p.is_file()):
        rel = item.relative_to(path).as_posix()
        h.update(rel.encode("utf-8"))
        h.update(b"\0")
        with item.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        h.update(b"\0")
    return h.hexdigest()


def _is_truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on", "enabled"}
    return bool(value)


def _summary_runs(source_root: Path) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for summary_path in sorted(source_root.rglob("summary.json")):
        try:
            summary = _read_json(summary_path)
        except Exception:
            continue
        for row in summary.get("runs") or []:
            scenario = str((row.get("axes") or {}).get("scenario") or "")
            if scenario in SCENARIOS and str(row.get("status")) == "ok":
                item = dict(row)
                item["scenario"] = scenario
                item["summary_path"] = str(summary_path)
                runs.append(item)
    runs.sort(key=lambda item: (SCENARIOS.index(str(item["scenario"])), int(item.get("seed", 0) or 0), str(item.get("run_dir"))))
    return runs


def _run_id(run_dir: Path) -> str:
    manifest = run_dir / "manifest.json"
    if manifest.exists():
        try:
            data = _read_json(manifest)
            return str(data.get("runtime_run_id") or data.get("run_id") or run_dir.name)
        except Exception:
            pass
    return run_dir.name


def _archive_entries(run_dir: Path) -> dict[str, list[dict[str, Any]]]:
    by_lineage: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in sorted((run_dir / "archives" / "lineage_qd" / "entries").glob("*.json")):
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
    candidates: list[dict[str, Any]] = []
    for row in _read_jsonl(run_dir / "metrics" / "branch_candidates.jsonl"):
        generation = int(row.get("generation", row.get("gen", 0)) or 0)
        for item in row.get("candidates") or []:
            if isinstance(item, dict):
                out = dict(item)
                out.setdefault("generation", generation)
                candidates.append(out)
    return candidates


def _lineage_rows(run_dir: Path) -> list[dict[str, Any]]:
    return _read_jsonl(run_dir / "metrics" / "lineage_metrics.jsonl")


def _lineage_row_at_or_after(run_dir: Path, lineage_id: str, generation: int) -> dict[str, Any] | None:
    rows = [
        row
        for row in _lineage_rows(run_dir)
        if str(row.get("lineage_id") or "") == str(lineage_id)
        and int(row.get("generation", 0) or 0) >= int(generation)
    ]
    if not rows:
        return None
    rows.sort(key=lambda row: int(row.get("generation", 0) or 0))
    return dict(rows[0])


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


def _dominant_lineage_id(run_dir: Path, *, generation: int | None = None) -> str:
    rows = _lineage_rows(run_dir)
    if not rows:
        return ""
    if generation is None:
        generation = max(int(row.get("generation", 0) or 0) for row in rows)
    selected = [row for row in rows if int(row.get("generation", 0) or 0) == int(generation)]
    if not selected:
        return ""
    top = max(
        selected,
        key=lambda row: (
            float(row.get("population_share", 0.0) or 0.0),
            float(row.get("reproduction_share", 0.0) or 0.0),
            str(row.get("lineage_id") or ""),
        ),
    )
    return str(top.get("lineage_id") or "")


def link_checkpoint_for_generation(run_dir: Path, generation: int, *, allow_prior: bool = True) -> dict[str, Any]:
    root = run_dir / "checkpoint_bundles_v2"
    if not root.exists():
        return {"valid": False, "reason": "source run has no checkpoint_bundles_v2 directory"}
    bundles: list[tuple[int, Path]] = []
    for bundle in sorted(root.glob("bundle_gen*")):
        if not bundle.is_dir():
            continue
        try:
            gen = int(bundle.name.replace("bundle_gen", ""))
        except ValueError:
            continue
        if gen == int(generation) or (allow_prior and gen <= int(generation)):
            bundles.append((gen, bundle))
    if not bundles:
        return {"valid": False, "reason": f"no checkpoint bundle at or before generation {generation}"}
    gen, bundle = max(bundles, key=lambda item: item[0])
    metadata = bundle / "metadata.json"
    return {
        "valid": metadata.exists(),
        "reason": "" if metadata.exists() else "checkpoint metadata missing",
        "checkpoint_generation": int(gen),
        "source_checkpoint_path": str(bundle),
        "source_checkpoint_metadata_path": str(metadata),
    }


def _expected_flags(config: Mapping[str, Any]) -> dict[str, Any]:
    env = dict(config.get("env_cfg") or {})
    return {field: env.get(field) for field in EXPECTED_FLAG_FIELDS if field in env}


def _expected_tuned_overrides(config: Mapping[str, Any]) -> dict[str, Any]:
    env = dict(config.get("env_cfg") or {})
    return {field: env.get(field) for field in TUNED_OVERRIDE_FIELDS if field in env}


def _theme4_disabled_status(config: Mapping[str, Any]) -> dict[str, bool]:
    sim = dict(config.get("sim_cfg") or {})
    return {field: bool(sim.get(field, False)) is False for field in THEME4_DISABLED_FIELDS}


def _theme5_reporting_status(config: Mapping[str, Any]) -> dict[str, Any]:
    sim = dict(config.get("sim_cfg") or {})
    fields = (
        "lineage_observability_enable",
        "lineage_qd_archive_enable",
        "lineage_qd_archive_top_n",
        "lineage_qd_archive_min_score",
        "lineage_qd_archive_cooldown_generations",
    )
    return {field: sim.get(field) for field in fields if field in sim}


def _entry_for_candidate(
    *,
    role: str,
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
    pop = dict(config.get("pop_cfg") or {})
    env = dict(config.get("env_cfg") or {})
    generation = int(candidate.get("generation", 0) or 0)
    lineage_id = str(candidate.get("lineage_id") or "")
    if not lineage_id:
        return None
    checkpoint = link_checkpoint_for_generation(run_dir, generation)
    if not checkpoint.get("valid"):
        return None
    lineage_row = _lineage_row_at_or_after(run_dir, lineage_id, generation) or {}
    category = "dominant_lineage" if role == "dominant" else str(candidate.get("candidate_type") or "")
    alive_share = candidate.get("population_share", lineage_row.get("population_share"))
    birth_share = candidate.get("reproduction_share", lineage_row.get("reproduction_share"))
    archive_id = str((archive_entry or {}).get("archive_entry_id") or "")
    scenario = str(run_record.get("scenario") or "")
    seed = int(run_record.get("seed", sim.get("seed", 0)) or 0)
    entry_id = f"r3a_{role}_{scenario}_{seed:06d}_{lineage_id}_g{generation:06d}"
    descriptor = {
        "branch_candidate": dict(candidate),
        "archive_entry": dict(archive_entry or {}),
        "lineage_descriptor": _lineage_descriptor_at(run_dir, lineage_id, generation),
        "lineage_metrics": dict(lineage_row),
    }
    return {
        "manifest_version": "r3a_branch_replay_manifest_v1",
        "manifest_id": "r3a_branch_replay_smoke",
        "entry_id": entry_id,
        "source_campaign_id": "integrated_r3a_checkpoint_enabled_branch_source",
        "source_output_root": str(source_root),
        "source_run_id": _run_id(run_dir),
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
        "candidate_generation_emitted": generation,
        "candidate_alive_share_at_emission": alive_share,
        "candidate_birth_share_at_emission": birth_share,
        "candidate_descriptor_snapshot": descriptor,
        "candidate_archive_entry_id": archive_id or "missing_optional",
        "checkpoint_reference_generation": int(checkpoint["checkpoint_generation"]),
        "replay_mode": "checkpoint_continuation",
        "replay_start_generation": int(checkpoint["checkpoint_generation"]),
        "replay_generations": 4,
        "replay_steps_per_generation": int(sim.get("steps_per_generation", 60) or 60),
        "population_handling_mode": "restored_from_checkpoint",
        "allowed_mutation_policy": "unchanged",
        "allowed_selection_policy": "unchanged",
        "allowed_reproduction_policy": "unchanged",
        "allowed_ecology_policy": "unchanged",
        "expected_flags": _expected_flags(config),
        "expected_tuned_overrides": _expected_tuned_overrides(config),
        "expected_theme4_disabled_status": _theme4_disabled_status(config),
        "expected_theme5_reporting_status": _theme5_reporting_status(config),
        "source_run_settings": {
            "generations": int(sim.get("generations", 0) or 0),
            "steps_per_generation": int(sim.get("steps_per_generation", 0) or 0),
            "population_size": int(pop.get("population_size", 0) or 0),
            "grid_width": int(env.get("width", 0) or 0),
            "grid_height": int(env.get("height", 0) or 0),
            "policy_tick_stride": int(sim.get("policy_tick_stride", 1) or 1),
            "bank_device": str(sim.get("bank_device") or ""),
        },
        "output_root": str(output_root / "replays" / entry_id),
    }


def select_candidates(source_root: Path, output_root: Path) -> dict[str, Any]:
    runs = _summary_runs(source_root)
    by_scenario: dict[str, list[dict[str, Any]]] = {scenario: [] for scenario in SCENARIOS}
    for run in runs:
        by_scenario.setdefault(str(run["scenario"]), []).append(run)
    for values in by_scenario.values():
        values.sort(key=lambda item: int(item.get("seed", 0) or 0))

    selected: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []

    tuned_runs = by_scenario.get("r3a_00_tuned_full_jungle_recovery", [])
    if tuned_runs:
        run = tuned_runs[0]
        run_dir = Path(str(run["run_dir"]))
        final_dominant = _dominant_lineage_id(run_dir)
        archive_by_lineage = _archive_entries(run_dir)
        row = _lineage_row_at_or_after(run_dir, final_dominant, 0) or {"lineage_id": final_dominant, "generation": 0}
        candidate = {
            "lineage_id": final_dominant,
            "generation": int(row.get("generation", 0) or 0),
            "candidate_type": "dominant_lineage",
            "population_share": row.get("population_share"),
            "reproduction_share": row.get("reproduction_share"),
        }
        archive_entry = _best_archive_entry(archive_by_lineage, final_dominant, int(candidate["generation"]))
        entry = _entry_for_candidate(
            role="dominant",
            run_record=run,
            candidate=candidate,
            archive_entry=archive_entry,
            source_root=source_root,
            output_root=output_root,
        )
        if entry is not None:
            selected.append(entry)
        else:
            missing.append({"scenario": str(run["scenario"]), "role": "dominant", "missing_candidate_reason": "dominant candidate could not be linked to checkpoint"})
    else:
        missing.append({"scenario": "r3a_00_tuned_full_jungle_recovery", "role": "dominant", "missing_candidate_reason": "source run missing"})

    for scenario, role in (
        ("r3a_00_tuned_full_jungle_recovery", "non_dominant_tuned_recovery"),
        ("r3a_01_default_full_jungle", "non_dominant_default_full_jungle"),
        ("r3a_02_theme12_clean_pressure_reference", "non_dominant_theme12_clean_pressure"),
    ):
        chosen: dict[str, Any] | None = None
        for run in by_scenario.get(scenario, []):
            run_dir = Path(str(run["run_dir"]))
            dominant = _dominant_lineage_id(run_dir)
            archive_by_lineage = _archive_entries(run_dir)
            candidates = [
                item
                for item in _branch_candidates(run_dir)
                if str(item.get("lineage_id") or "")
                and str(item.get("lineage_id") or "") != dominant
                and int(item.get("generation", 0) or 0) <= 5
                and (_float(item.get("population_share")) or 0.0) > 0.0
                and link_checkpoint_for_generation(run_dir, int(item.get("generation", 0) or 0)).get("valid")
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
            chosen = _entry_for_candidate(
                role=role,
                run_record=run,
                candidate=candidate,
                archive_entry=archive_entry,
                source_root=source_root,
                output_root=output_root,
            )
            if chosen is not None:
                break
        if chosen is None:
            missing.append({"scenario": scenario, "role": role, "missing_candidate_reason": "no non-dominant branch candidate linked to checkpoint"})
        else:
            selected.append(chosen)

    return {
        "schema_version": "integrated_r3a_candidate_selection_v1",
        "source_root": str(source_root),
        "selection_criteria": [
            "one dominant tuned recovery candidate",
            "one non-dominant tuned recovery branch candidate",
            "one non-dominant default full-Jungle branch candidate",
            "one non-dominant Theme12 clean-pressure branch candidate",
            "generation <= 5",
            "nonzero alive share at emission",
            "checkpoint bundle at or before emission generation",
        ],
        "selected_count": len(selected),
        "selected_candidates": selected,
        "missing_candidates": missing,
    }


def build_manifest(selection: Mapping[str, Any]) -> dict[str, Any]:
    entries = list(selection.get("selected_candidates") or [])
    stable = json.dumps(entries, sort_keys=True, separators=(",", ":"))
    manifest_id = "r3a_branch_replay_smoke_" + hashlib.sha256(stable.encode("utf-8")).hexdigest()[:16]
    out_entries = []
    for entry in entries:
        item = dict(entry)
        item["manifest_id"] = manifest_id
        out_entries.append(item)
    return {
        "manifest_version": "r3a_branch_replay_manifest_set_v1",
        "manifest_id": manifest_id,
        "source_campaign_id": "integrated_r3a_checkpoint_enabled_branch_source",
        "source_output_root": str(selection.get("source_root") or ""),
        "replay_mode": "checkpoint_continuation",
        "entries": out_entries,
    }


def validate_manifest(manifest: Mapping[str, Any], *, require_existing_paths: bool = True) -> dict[str, Any]:
    entries = list(manifest.get("entries") or manifest.get("candidates") or [])
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    for idx, entry in enumerate(entries):
        if not isinstance(entry, Mapping):
            errors.append({"entry_index": idx, "field": "entry", "error": "entry is not an object"})
            continue
        for field in REQUIRED_ENTRY_FIELDS:
            if field not in entry:
                errors.append({"entry_index": idx, "field": field, "error": "missing field"})
                continue
            if entry.get(field) in (None, ""):
                errors.append({"entry_index": idx, "field": field, "error": "empty required field"})
        if require_existing_paths:
            for field in PATH_FIELDS:
                value = entry.get(field)
                if not isinstance(value, str) or not Path(value).exists():
                    errors.append({"entry_index": idx, "field": field, "error": f"path does not exist: {value}"})
        if entry.get("replay_mode") != "checkpoint_continuation":
            errors.append({"entry_index": idx, "field": "replay_mode", "error": "only checkpoint_continuation is allowed"})
        if entry.get("population_handling_mode") != "restored_from_checkpoint":
            errors.append({"entry_index": idx, "field": "population_handling_mode", "error": "population must be restored from checkpoint"})
        for policy in (
            "allowed_mutation_policy",
            "allowed_selection_policy",
            "allowed_reproduction_policy",
            "allowed_ecology_policy",
        ):
            if entry.get(policy) != "unchanged":
                errors.append({"entry_index": idx, "field": policy, "error": "policy must be unchanged"})
        theme4 = entry.get("expected_theme4_disabled_status")
        if isinstance(theme4, Mapping):
            enabled = [field for field, disabled in theme4.items() if disabled is not True]
            if enabled:
                errors.append({"entry_index": idx, "field": "expected_theme4_disabled_status", "error": f"Theme 4 gates enabled: {enabled}"})
        else:
            errors.append({"entry_index": idx, "field": "expected_theme4_disabled_status", "error": "missing Theme 4 disabled status"})
        config_path = Path(str(entry.get("source_config_path") or ""))
        if config_path.exists():
            config = _read_json(config_path)
            env = dict(config.get("env_cfg") or {})
            for key, expected in dict(entry.get("expected_flags") or {}).items():
                if env.get(key) != expected:
                    errors.append({"entry_index": idx, "field": f"expected_flags.{key}", "error": f"source config has {env.get(key)!r}, expected {expected!r}"})
            for section_name in ("sim_cfg", "env_cfg", "pop_cfg"):
                for key, value in dict(config.get(section_name) or {}).items():
                    lowered = str(key).lower()
                    if any(token in lowered for token in FORBIDDEN_INTERVENTION_KEYS) and _is_truthy(value):
                        errors.append({"entry_index": idx, "field": f"{section_name}.{key}", "error": "lineage intervention flag is enabled"})
        if str(entry.get("candidate_archive_entry_id") or "") == "missing_optional":
            warnings.append({"entry_index": idx, "field": "candidate_archive_entry_id", "warning": "no matching archive entry found"})
    return {
        "schema_version": "integrated_r3a_manifest_validation_v1",
        "entry_count": len(entries),
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
    }


def _knob_args(values: Mapping[str, Any], *, prefix: str) -> list[str]:
    args: list[str] = []
    for key in sorted(values):
        value = values[key]
        if isinstance(value, bool):
            text = "true" if value else "false"
        else:
            text = str(value)
        args.extend(["--knob", f"{prefix}.{key}={text}"])
    return args


def _replay_cli(entry: Mapping[str, Any], *, runs_dir: Path, run_id: str) -> list[str]:
    config = _read_json(Path(str(entry["source_config_path"])))
    sim = dict(config.get("sim_cfg") or {})
    pop = dict(config.get("pop_cfg") or {})
    env = dict(config.get("env_cfg") or {})
    width = int(env.get("width", 16) or 16)
    height = int(env.get("height", width) or width)
    if width != height:
        raise ValueError(f"R3a replay expects square grid, got {width}x{height}")
    args = [
        sys.executable,
        "-m",
        "engine.main",
        "--resume_from",
        str(entry["source_checkpoint_path"]),
        "--generations",
        str(int(entry.get("replay_generations", 4) or 4)),
        "--steps",
        str(int(entry.get("replay_steps_per_generation", sim.get("steps_per_generation", 60)) or 60)),
        "--pop",
        str(int(pop.get("population_size", 64) or 64)),
        "--grid_size",
        str(width),
        "--policy_tick_stride",
        str(int(sim.get("policy_tick_stride", 1) or 1)),
        "--device",
        str(entry.get("source_device") or sim.get("device") or "cuda"),
        "--bank_device",
        str(sim.get("bank_device") or entry.get("source_device") or "cuda"),
        "--seed",
        str(int(entry.get("source_seed", sim.get("seed", 0)) or 0)),
        "--runs_dir",
        str(runs_dir),
        "--run_id",
        run_id,
        "--run_tag",
        str(entry.get("entry_id") or run_id),
        "--allow_existing_run_dir",
        "--checkpoint_bundle_v2_period_gen",
        "0",
        "--checkpoint_rotate",
        "0",
    ]
    args.extend(_knob_args(dict(entry.get("expected_flags") or {}), prefix="env_cfg"))
    args.extend(_knob_args(dict(entry.get("expected_tuned_overrides") or {}), prefix="env_cfg"))
    for field in THEME4_DISABLED_FIELDS:
        args.extend(["--knob", f"sim_cfg.{field}=false"])
    args.extend(["--knob", "sim_cfg.behavior_log_enable=false"])
    return args


def _final_metrics_row(metrics_path: Path) -> dict[str, str]:
    rows = _read_csv(metrics_path)
    return dict(rows[-1]) if rows else {}


def _tracking_from_replay(
    replay_run_dir: Path,
    *,
    candidate_lineage_id: str,
    dominant_lineage_id: str,
) -> dict[str, Any]:
    by_generation: dict[int, dict[str, Any]] = {}
    for row in _read_jsonl(replay_run_dir / "metrics" / "lineage_metrics.jsonl"):
        generation = int(row.get("generation", 0) or 0)
        item = by_generation.setdefault(generation, {"generation": generation})
        lineage = str(row.get("lineage_id") or "")
        if lineage == candidate_lineage_id:
            item.update(
                {
                    "candidate_alive_count": row.get("population_count"),
                    "candidate_alive_share": row.get("population_share"),
                    "candidate_birth_count": row.get("offspring_count"),
                    "candidate_birth_share": row.get("reproduction_share"),
                    "candidate_extinct_flag": False,
                }
            )
        if lineage == dominant_lineage_id:
            item.update(
                {
                    "dominant_lineage_id": dominant_lineage_id,
                    "dominant_alive_count": row.get("population_count"),
                    "dominant_alive_share": row.get("population_share"),
                    "dominant_birth_count": row.get("offspring_count"),
                    "dominant_birth_share": row.get("reproduction_share"),
                }
            )
    for item in by_generation.values():
        item.setdefault("candidate_alive_count", 0)
        item.setdefault("candidate_alive_share", 0.0)
        item.setdefault("candidate_birth_count", 0)
        item.setdefault("candidate_birth_share", 0.0)
        item.setdefault("candidate_extinct_flag", True)
    trajectory = [by_generation[g] for g in sorted(by_generation)]
    return {
        "schema_version": "integrated_r3a_candidate_lineage_tracking_v1",
        "candidate_lineage_id": candidate_lineage_id,
        "dominant_lineage_id": dominant_lineage_id,
        "trajectory": trajectory,
    }


def run_replay_entry(entry: Mapping[str, Any], *, run_root: Path, label: str, overwrite: bool) -> dict[str, Any]:
    entry_id = str(entry.get("entry_id") or "entry")
    runs_dir = run_root / label / entry_id
    run_id = "replay"
    replay_run_dir = runs_dir / run_id
    if overwrite and runs_dir.exists():
        shutil.rmtree(runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)
    cmd = _replay_cli(entry, runs_dir=runs_dir, run_id=run_id)
    proc = subprocess.run(cmd, cwd=str(_project_root()), check=False, capture_output=True, text=True)
    (runs_dir / "console.txt").write_text((proc.stdout or "") + (proc.stderr or ""), encoding="utf-8")
    if proc.returncode != 0:
        return {
            "entry_id": entry_id,
            "status": "failed",
            "returncode": proc.returncode,
            "run_dir": str(replay_run_dir),
            "console_tail": ((proc.stdout or "") + (proc.stderr or ""))[-4000:],
        }
    source_run_dir = Path(str(entry["source_metrics_path"])).parent
    dominant = _dominant_lineage_id(source_run_dir, generation=int(entry.get("source_generation", 0) or 0))
    tracking = _tracking_from_replay(
        replay_run_dir,
        candidate_lineage_id=str(entry["candidate_lineage_id"]),
        dominant_lineage_id=dominant,
    )
    tracking_path = replay_run_dir / "candidate_lineage_tracking.json"
    _write_json(tracking_path, tracking)
    metadata = {
        "schema_version": "integrated_r3a_replay_metadata_v1",
        "manifest_id": entry.get("manifest_id"),
        "manifest_entry_id": entry_id,
        "source_checkpoint_path": entry.get("source_checkpoint_path"),
        "source_run_id": entry.get("source_run_id"),
        "candidate_lineage_id": entry.get("candidate_lineage_id"),
        "dominant_lineage_id": dominant,
        "replay_start_generation": entry.get("replay_start_generation"),
        "replay_end_generation": int(entry.get("replay_start_generation", 0) or 0)
        + int(entry.get("replay_generations", 0) or 0),
        "scenario_id": entry.get("source_scenario_id"),
        "seed": entry.get("source_seed"),
        "device": entry.get("source_device"),
        "config_hash": _sha256_file(replay_run_dir / "config.json"),
        "checkpoint_hash": _sha256_tree(Path(str(entry["source_checkpoint_path"]))),
    }
    metadata_path = replay_run_dir / "replay_metadata.json"
    _write_json(metadata_path, metadata)
    final_metrics = _final_metrics_row(replay_run_dir / "metrics.csv")
    _write_json(replay_run_dir / "final_metrics_row.json", final_metrics)
    return {
        "entry_id": entry_id,
        "status": "ok",
        "returncode": 0,
        "run_dir": str(replay_run_dir),
        "metrics_hash": _sha256_file(replay_run_dir / "metrics.csv"),
        "replay_metadata_hash": _sha256_file(metadata_path),
        "candidate_lineage_tracking_hash": _sha256_file(tracking_path),
        "config_hash": _sha256_file(replay_run_dir / "config.json"),
        "final_metrics_row": final_metrics,
        "tracking": tracking,
    }


def run_smokes(manifest: Mapping[str, Any], *, output_root: Path, overwrite: bool) -> dict[str, Any]:
    entries = list(manifest.get("entries") or [])
    results = [run_replay_entry(entry, run_root=output_root / "replays", label="primary", overwrite=overwrite) for entry in entries]
    return {
        "schema_version": "integrated_r3a_smoke_summary_v1",
        "executed": True,
        "planned": len(entries),
        "completed": sum(1 for item in results if item.get("status") == "ok"),
        "failed": sum(1 for item in results if item.get("status") != "ok"),
        "smoke_runs": results,
    }


def run_repeat_parity(manifest: Mapping[str, Any], *, output_root: Path, overwrite: bool) -> dict[str, Any]:
    entries = list(manifest.get("entries") or [])
    dominant = [entry for entry in entries if str(entry.get("candidate_category")) == "dominant_lineage"]
    nondominant = [entry for entry in entries if str(entry.get("candidate_category")) != "dominant_lineage"]
    selected = []
    if dominant:
        selected.append(dominant[0])
    if nondominant:
        selected.append(nondominant[0])
    checks: list[dict[str, Any]] = []
    for entry in selected:
        a = run_replay_entry(entry, run_root=output_root / "replays", label="parity_a", overwrite=overwrite)
        b = run_replay_entry(entry, run_root=output_root / "replays", label="parity_b", overwrite=overwrite)
        match = (
            a.get("metrics_hash") == b.get("metrics_hash")
            and a.get("replay_metadata_hash") == b.get("replay_metadata_hash")
            and a.get("candidate_lineage_tracking_hash") == b.get("candidate_lineage_tracking_hash")
            and a.get("config_hash") == b.get("config_hash")
            and a.get("final_metrics_row") == b.get("final_metrics_row")
        )
        checks.append(
            {
                "entry_id": entry.get("entry_id"),
                "candidate_category": entry.get("candidate_category"),
                "a": {k: v for k, v in a.items() if k not in {"tracking", "final_metrics_row"}},
                "b": {k: v for k, v in b.items() if k not in {"tracking", "final_metrics_row"}},
                "final_metrics_rows_match": a.get("final_metrics_row") == b.get("final_metrics_row"),
                "passed": bool(match and a.get("status") == "ok" and b.get("status") == "ok"),
            }
        )
    return {
        "schema_version": "integrated_r3a_repeat_parity_summary_v1",
        "executed": True,
        "checks": checks,
        "passed": bool(checks) and all(item.get("passed") for item in checks),
    }


def _smoke_compact(smoke: Mapping[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for run in smoke.get("smoke_runs") or []:
        tracking = run.get("tracking") or {}
        trajectory = list(tracking.get("trajectory") or [])
        out.append(
            {
                "entry_id": run.get("entry_id"),
                "status": run.get("status"),
                "run_dir": run.get("run_dir"),
                "candidate_alive_trajectory": [item.get("candidate_alive_count") for item in trajectory],
                "candidate_alive_share_trajectory": [item.get("candidate_alive_share") for item in trajectory],
                "candidate_birth_trajectory": [item.get("candidate_birth_count") for item in trajectory],
                "candidate_alive_at_end": bool(trajectory and not trajectory[-1].get("candidate_extinct_flag")),
                "notes": "checkpoint-continuation smoke only; not long-term viability evidence",
            }
        )
    return out


def readiness_classification(validation: Mapping[str, Any], smoke: Mapping[str, Any], parity: Mapping[str, Any]) -> str:
    if not validation.get("valid"):
        return "METADATA-ONLY"
    if not smoke.get("executed") or smoke.get("failed"):
        return "PARTIAL"
    if not parity.get("executed") or not parity.get("passed"):
        return "PARTIAL"
    return "READY"


def _md_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def write_report(
    path: Path,
    *,
    worktree_info: Mapping[str, Any],
    source_registry: str,
    selection: Mapping[str, Any],
    manifest_json: Path,
    validation: Mapping[str, Any],
    smoke: Mapping[str, Any],
    parity: Mapping[str, Any],
    classification: str,
    tests_run: Sequence[str],
    files_changed: Sequence[str],
) -> None:
    candidate_rows = []
    for entry in selection.get("selected_candidates") or []:
        candidate_rows.append(
            [
                entry.get("entry_id"),
                entry.get("source_scenario_id"),
                entry.get("source_seed"),
                entry.get("candidate_generation_emitted"),
                entry.get("candidate_biological_lineage_id"),
                entry.get("candidate_category"),
                entry.get("checkpoint_reference_generation"),
                Path(str(entry.get("source_checkpoint_path"))).name,
            ]
        )
    linkage_rows = [
        [
            entry.get("entry_id"),
            entry.get("candidate_generation_emitted"),
            entry.get("checkpoint_reference_generation"),
            "present" if Path(str(entry.get("source_checkpoint_path"))).exists() else "missing",
            "present" if Path(str(entry.get("source_checkpoint_metadata_path"))).exists() else "missing",
        ]
        for entry in selection.get("selected_candidates") or []
    ]
    smoke_rows = [
        [
            item.get("entry_id"),
            item.get("status"),
            item.get("candidate_alive_trajectory"),
            item.get("candidate_birth_trajectory"),
            item.get("candidate_alive_at_end"),
        ]
        for item in _smoke_compact(smoke)
    ]
    parity_rows = [
        [item.get("entry_id"), item.get("candidate_category"), item.get("passed")]
        for item in parity.get("checks") or []
    ]
    text = f"""# EIEL Integrated R3a Checkpoint-Enabled Branch Replay Report

## 1. Purpose

R3a tests whether checkpoint-enabled R2-style source runs can link Theme 5
branch candidates to checkpoint_bundles_v2, generate valid executable
manifests, and run deterministic Mode A checkpoint-continuation replay smokes.

## 2. R1/R2/R3 Dependency Summary

- R1 commit: `{worktree_info.get('r1_commit', '')}`
- R2 commit: `{worktree_info.get('r2_commit', '')}`
- R3 commit: `{worktree_info.get('r3_commit', '')}`
- R3 status dependency: COMPLETE, METADATA-ONLY because source checkpoints were missing.

## 3. Worktree Path And Base Commit

Worktree: `{worktree_info.get('worktree_path', '')}`

Base commit: `{worktree_info.get('base_commit', '')}`

Git status before work:

```text
{str(worktree_info.get('git_status_before', '')).strip()}
```

Git status after work:

```text
{str(worktree_info.get('git_status_after', '')).strip()}
```

## 4. Source Campaign Registry And Settings

Registry: `{source_registry}`

Settings: 3 scenarios, seed 101, 8 generations, 60 steps per generation,
population 64, grid 16, CUDA device/bank device. Checkpoint Bundle v2 was
enabled every generation and gen0 emission was enabled by
`EIEL_CHECKPOINT_BUNDLE_V2_AT_START=1`.

## 5. Checkpoint Policy

The source runs use existing checkpoint_bundles_v2 infrastructure only. No
new checkpoint schema or simulation behavior was added. Checkpoints are linked
by source run, scenario, seed, candidate generation, bundle generation, bundle
path, metadata path, source config, and source metrics path.

## 6. Source Campaign Completion

Source runs selected: `{len(selection.get('selected_candidates') or [])}` candidates from
`{selection.get('source_root')}`.

Missing candidate records: `{len(selection.get('missing_candidates') or [])}`.

## 7. Checkpoint Bundle Availability

Required source checkpoint paths were present for all selected candidates:
`{validation.get('valid')}`.

## 8. Candidate Selection Table

{_md_table(('Entry', 'Scenario', 'Seed', 'Candidate gen', 'Lineage', 'Category', 'Checkpoint gen', 'Bundle'), candidate_rows)}

## 9. Candidate-To-Checkpoint Linkage Table

{_md_table(('Entry', 'Candidate gen', 'Checkpoint gen', 'Bundle path', 'Metadata'), linkage_rows)}

## 10. Manifest Paths

- `{manifest_json}`
- `_spec/branch_replay/v1/r3a_branch_replay_smoke_manifest.json`
- `_spec/branch_replay/v1/r3a_branch_replay_smoke_manifest.yaml`

## 11. Manifest Validation Result

Valid: `{validation.get('valid')}`

Errors: `{len(validation.get('errors') or [])}`

Warnings: `{len(validation.get('warnings') or [])}`

## 12. Replay Mode Implementation Status

Mode A checkpoint-continuation replay is implemented by
`tools/bin/integrated_r3a_branch_replay.py` through the existing
`engine.main --resume_from <checkpoint_bundles_v2/...>` path. Population,
lineage membership, selection, reproduction, mutation, and ecology are not
modified.

## 13. Replay Smoke Results

Executed: `{smoke.get('executed')}`

Completed: `{smoke.get('completed')}`

Failed: `{smoke.get('failed')}`

{_md_table(('Entry', 'Status', 'Alive trajectory', 'Birth trajectory', 'Alive at end'), smoke_rows)}

## 14. Repeat Parity Results

Executed: `{parity.get('executed')}`

Passed: `{parity.get('passed')}`

{_md_table(('Entry', 'Category', 'Passed'), parity_rows)}

## 15. Candidate Lineage Tracking Summary

Replay outputs include `candidate_lineage_tracking.json`,
`replay_metadata.json`, `final_metrics_row.json`, and `metrics.csv` for each
successful smoke run.

## 16. Missing Infrastructure

None if classification is READY. If not READY, inspect
`r3a_branch_replay_manifest_validation.json`, `r3a_smoke_summary.json`, and
`r3a_repeat_parity_summary.json`.

## 17. Readiness Classification

`{classification}`

## 18. Files Changed

{chr(10).join(f'- `{item}`' for item in files_changed)}

## 19. Tests Run

{chr(10).join(f'- `{item}`' for item in tests_run)}

## 20. Known Limitations

- R3a smoke replays are only 3-4 generation infrastructure checks.
- Candidate persistence during smoke is not long-term viability evidence.
- Candidate archive entries remain reporting artifacts, not active interventions.
- Theme 4 cognition remains disabled.

## 21. Recommended Next Step

`{'R3b branch-candidate replay campaign' if classification == 'READY' else 'complete replay runner / parity support'}`

## 22. No-Claims Statement

R3a validates checkpoint-backed branch replay infrastructure only.
It does not show that branch candidates are viable, adaptive, cognitively meaningful, or selected.
It does not show that lineage diversity is solved.
It does not activate Theme 4 cognition.
It does not alter Patch/Jungle mechanics.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    source_root = Path(args.source_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    selection = select_candidates(source_root, output_root)
    manifest = build_manifest(selection)
    validation = validate_manifest(manifest)

    _write_json(Path(args.candidate_json), selection)
    for path in (Path(args.manifest_json), Path(args.repo_manifest_json)):
        _write_json(path, manifest)
    for path in (Path(args.manifest_yaml), Path(args.repo_manifest_yaml)):
        _write_yaml(path, manifest)
    _write_json(Path(args.validation_json), validation)

    smoke: dict[str, Any]
    parity: dict[str, Any]
    if bool(args.run_replays) and validation.get("valid"):
        smoke = run_smokes(manifest, output_root=output_root, overwrite=bool(args.overwrite))
        if int(smoke.get("failed", 0) or 0) == 0:
            parity = run_repeat_parity(manifest, output_root=output_root, overwrite=bool(args.overwrite))
        else:
            parity = {
                "schema_version": "integrated_r3a_repeat_parity_summary_v1",
                "executed": False,
                "passed": False,
                "reason": "smoke failures block repeat parity",
                "checks": [],
            }
    else:
        smoke = {
            "schema_version": "integrated_r3a_smoke_summary_v1",
            "executed": False,
            "planned": len(manifest.get("entries") or []),
            "completed": 0,
            "failed": 0,
            "reason": "replay execution not requested or manifest invalid",
            "smoke_runs": [],
        }
        parity = {
            "schema_version": "integrated_r3a_repeat_parity_summary_v1",
            "executed": False,
            "passed": False,
            "reason": "replay execution not requested or manifest invalid",
            "checks": [],
        }
    _write_json(Path(args.smoke_json), smoke)
    _write_json(Path(args.repeat_parity_json), parity)
    _write_json(
        output_root / "r3a_smoke_compact.json",
        {"schema_version": "integrated_r3a_smoke_compact_v1", "runs": _smoke_compact(smoke)},
    )

    classification = readiness_classification(validation, smoke, parity)
    if args.report_md:
        worktree_info = _read_json(Path(args.worktree_info_json)) if args.worktree_info_json else {}
        write_report(
            Path(args.report_md),
            worktree_info=worktree_info,
            source_registry=args.source_registry,
            selection=selection,
            manifest_json=Path(args.manifest_json),
            validation=validation,
            smoke=smoke,
            parity=parity,
            classification=classification,
            tests_run=[item for item in str(args.tests_run or "").splitlines() if item.strip()],
            files_changed=[item for item in str(args.files_changed or "").splitlines() if item.strip()],
        )
    return {
        "classification": classification,
        "selected_count": selection.get("selected_count"),
        "manifest_valid": validation.get("valid"),
        "smoke_completed": smoke.get("completed"),
        "parity_passed": parity.get("passed"),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="R3a checkpoint branch replay helper")
    parser.add_argument("--source_root", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--candidate_json", required=True)
    parser.add_argument("--manifest_json", required=True)
    parser.add_argument("--manifest_yaml", required=True)
    parser.add_argument("--repo_manifest_json", required=True)
    parser.add_argument("--repo_manifest_yaml", required=True)
    parser.add_argument("--validation_json", required=True)
    parser.add_argument("--smoke_json", required=True)
    parser.add_argument("--repeat_parity_json", required=True)
    parser.add_argument("--source_registry", default="_spec/registry/v1/integrated_r3a_checkpoint_enabled_branch_source.yaml")
    parser.add_argument("--worktree_info_json", default="")
    parser.add_argument("--report_md", default="")
    parser.add_argument("--tests_run", default="")
    parser.add_argument("--files_changed", default="")
    parser.add_argument("--run_replays", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    result = run(parse_args(argv))
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0 if result.get("classification") in {"READY", "PARTIAL", "METADATA-ONLY"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
