#!/usr/bin/env python3
"""Founder spatial separation lineage persistence probe helper.

This helper writes harness registries, summarizes completed harness batches,
and packages small evidence artifacts. It does not run simulations directly and
does not alter simulation mechanics.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import zipfile
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence
import sys

try:
    import yaml  # type: ignore
except Exception as exc:  # pragma: no cover
    raise RuntimeError("PyYAML is required") from exc


csv.field_size_limit(sys.maxsize)

SEEDS = (43, 101, 202, 303, 404, 505, 606, 707)
PARITY_SEED = 43
GENERATIONS = 36
STEPS = 80
POPULATION = 64
GRID_SIZE = 16
DEVICE = "cuda"
FOUNDER_GROUP_COUNT = 8
FOUNDER_MIN_PER_GROUP = 8
FOUNDER_SPATIAL_MIN_DISTANCE = 5

REGISTRY_PATH = Path("_spec/registry/v1/founder_spatial_lineage_persistence_probe_20260704.yaml")
PARITY_REGISTRY_PATH = Path(
    "_spec/registry/v1/founder_spatial_lineage_persistence_probe_parity_20260704.yaml"
)
REPORT_DOC = Path("docs/500_campaigns-decisions-closeouts/520_campaign-reports/300_lineage-reproduction/EIEL-DOC-000335--eiel-founder-spatial-lineage-persistence-probe-2026-07-04.md")
EVIDENCE_DIR = Path("evidences/05-theme-5-lineage-diversity")
EVIDENCE_NOTE = EVIDENCE_DIR / "03-founder-spatial-lineage-persistence-probe.md"
EVIDENCE_ZIP = EVIDENCE_DIR / "03-founder-spatial-lineage-persistence-probe.evidence.zip"


def _knobs(*items: str) -> list[str]:
    out: list[str] = []
    for item in items:
        out.extend(["--knob", item])
    return out


COGNITION_OFF_KNOBS = (
    "sim_cfg.phase_e_e1_enable=false",
    "sim_cfg.enable_developmental_cognition=false",
    "sim_cfg.enable_introspection_channels=false",
    "sim_cfg.enable_introspection_policy_input=false",
    "sim_cfg.enable_self_regulation=false",
    "sim_cfg.enable_self_regulation_surface=false",
    "sim_cfg.enable_self_regulation_policy_output=false",
    "sim_cfg.enable_self_modification=false",
    "sim_cfg.enable_self_modification_policy_output=false",
    "sim_cfg.enable_directed_inheritance=false",
    "sim_cfg.enable_directed_inheritance_surface=false",
    "sim_cfg.enable_directed_inheritance_policy_output=false",
    "sim_cfg.enable_cognitive_trait_registry=false",
    "sim_cfg.enable_cognitive_status_metrics=false",
    "sim_cfg.enable_cognitive_budget_ledger=false",
    "sim_cfg.episode0_fitness_weight=0.0",
    "sim_cfg.episode0_pred_err_fitness_weight=0.0",
    "sim_cfg.episode0_thrashing_fitness_weight=0.0",
    "sim_cfg.episode0_drive_jitter_fitness_weight=0.0",
    "sim_cfg.episode0_selfharm_fitness_weight=0.0",
)

PATCH_JUNGLE_BASE = _knobs(
    "env_cfg.patch_world_enable=true",
    "env_cfg.patch_world_observation_enable=true",
    "env_cfg.patch_world_waste_consequence_enable=true",
    "env_cfg.jungle_predator_pressure_enable=true",
    "env_cfg.jungle_reproduction_emissions_enable=true",
    "env_cfg.jungle_juvenile_predator_vulnerability_enable=false",
    "env_cfg.jungle_terrain_protection_enable=true",
    "env_cfg.jungle_ecological_recovery_enable=true",
    "env_cfg.patch_movement_cost_apply_enable=false",
    "env_cfg.patch_cover_default=0.25",
    "env_cfg.patch_shelter_default=0.15",
    "env_cfg.patch_shelter_from_cover_enable=false",
)

THEME12_BASE = _knobs(
    "env_cfg.patch_world_enable=true",
    "env_cfg.patch_world_observation_enable=true",
    "env_cfg.patch_world_waste_consequence_enable=true",
    "env_cfg.jungle_predator_pressure_enable=true",
    "env_cfg.jungle_reproduction_emissions_enable=true",
    "env_cfg.jungle_juvenile_predator_vulnerability_enable=false",
    "env_cfg.jungle_terrain_protection_enable=true",
    "env_cfg.jungle_ecological_recovery_enable=true",
    "env_cfg.patch_movement_cost_apply_enable=false",
    "env_cfg.patch_cover_default=0.25",
    "env_cfg.patch_shelter_default=0.15",
    "env_cfg.patch_shelter_from_cover_enable=false",
    "env_cfg.patch_waste_regrowth_penalty_scale=0.35",
    "env_cfg.patch_density_regrowth_penalty_scale=0.35",
    "env_cfg.jungle_predator_risk_cap=0.045",
    "env_cfg.jungle_predator_health_damage=0.22",
    "env_cfg.jungle_recovery_regrowth_boost_scale=0.08",
)

ARM_FLAGS = {
    "legacy": _knobs(
        "sim_cfg.lineage_founder_balance_enable=false",
        "sim_cfg.lineage_founder_spatial_separation_enable=false",
    ),
    "founder_balance": _knobs(
        "sim_cfg.lineage_founder_balance_enable=true",
        f"sim_cfg.lineage_founder_group_count={FOUNDER_GROUP_COUNT}",
        f"sim_cfg.lineage_founder_min_agents_per_group={FOUNDER_MIN_PER_GROUP}",
        "sim_cfg.lineage_founder_assignment_mode=deterministic_round_robin",
        "sim_cfg.lineage_founder_balance_schema_version=1",
        "sim_cfg.lineage_founder_spatial_separation_enable=false",
    ),
    "founder_spatial": _knobs(
        "sim_cfg.lineage_founder_balance_enable=false",
        "sim_cfg.lineage_founder_spatial_separation_enable=true",
        f"sim_cfg.lineage_founder_spatial_group_count={FOUNDER_GROUP_COUNT}",
        f"sim_cfg.lineage_founder_spatial_min_distance={FOUNDER_SPATIAL_MIN_DISTANCE}",
        "sim_cfg.lineage_founder_spatial_assignment_mode=grid_quadrants",
        "sim_cfg.lineage_founder_spatial_use_patch_basins=true",
        "sim_cfg.lineage_founder_spatial_schema_version=1",
    ),
}

CONTEXT_FLAGS = {
    "clean_patch_jungle": PATCH_JUNGLE_BASE,
    "theme12_clean_ecology": THEME12_BASE,
}


def _cli_defaults() -> list[list[str]]:
    rows = [
        ["--generations", str(GENERATIONS)],
        ["--steps", str(STEPS)],
        ["--pop", str(POPULATION)],
        ["--grid_size", str(GRID_SIZE)],
        ["--policy_tick_stride", "1"],
        ["--device", DEVICE],
        ["--bank_device", DEVICE],
        ["--checkpoint_bundle_v2_period_gen", "1"],
        ["--checkpoint_rotate", "0"],
    ]
    for knob in (
        "sim_cfg.behavior_log_enable=false",
        "sim_cfg.policy_deterministic_actions=true",
        "sim_cfg.checkpoint_bundle_v2_strict_replay_mode=true",
        "sim_cfg.lineage_observability_enable=true",
        "sim_cfg.lineage_qd_archive_enable=true",
        "sim_cfg.lineage_qd_archive_top_n=8",
        "sim_cfg.lineage_qd_archive_min_score=0.25",
        "sim_cfg.lineage_qd_archive_cooldown_generations=0",
        "sim_cfg.lineage_collapse_forensics_enable=true",
        "sim_cfg.lineage_collapse_forensics_schema_version=1",
        "sim_cfg.device=cuda",
        "sim_cfg.bank_device=cuda",
        *COGNITION_OFF_KNOBS,
    ):
        rows.append(["--knob", knob])
    return rows


def _registry_doc(*, parity: bool = False) -> dict[str, Any]:
    axes: dict[str, Any] = {
        "context": {
            "values": list(CONTEXT_FLAGS.keys()) if not parity else ["clean_patch_jungle"],
            "flags": CONTEXT_FLAGS,
        },
        "init_arm": {
            "values": list(ARM_FLAGS.keys()),
            "flags": ARM_FLAGS,
        },
    }
    matrix_order = ["context", "init_arm"]
    if parity:
        axes["repeat"] = {"values": ["a", "b"], "flags": {"a": [], "b": []}}
        matrix_order.append("repeat")
    matrix_order.append("seed")
    name = "founder_spatial_lineage_persistence_probe_parity_20260704" if parity else "founder_spatial_lineage_persistence_probe_20260704"
    return {
        "harness_api": 1,
        "version": 1,
        "name": name,
        "description": "Founder spatial separation lineage persistence probe.",
        "intent": {
            "question": "Does default-off founder spatial/ecological separation improve live lineage persistence and reduce founder collision?",
            "safe_claim_boundary": "Initialization-only diagnostic; no lineage protection, rescue, reseeding, rewards, cognition, intelligence, or emergence claim.",
        },
        "matrix_order": matrix_order,
        "axes": axes,
        "seeds": [PARITY_SEED] if parity else list(SEEDS),
        "cli_defaults": _cli_defaults(),
        "expected_artifacts": ["manifest.json", "run_config.csv", "config.json", "metrics.csv"],
        "naming": {
            "run_tag_prefix": name,
            "run_id_format": (
                f"{name}__context_{{context}}__arm_{{init_arm}}__repeat_{{repeat}}__seed{{seed:06d}}"
                if parity
                else f"{name}__context_{{context}}__arm_{{init_arm}}__seed{{seed:06d}}"
            ),
            "run_tag_format": (
                f"{name}__context_{{context}}__arm_{{init_arm}}__repeat_{{repeat}}"
                if parity
                else f"{name}__context_{{context}}__arm_{{init_arm}}"
            ),
        },
    }


def write_registries() -> None:
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY_PATH.write_text(yaml.safe_dump(_registry_doc(), sort_keys=False), encoding="utf-8")
    PARITY_REGISTRY_PATH.write_text(yaml.safe_dump(_registry_doc(parity=True), sort_keys=False), encoding="utf-8")
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    for path in (REGISTRY_PATH, PARITY_REGISTRY_PATH):
        target = EVIDENCE_DIR / path.name
        target.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _f(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def _i(value: Any, default: int = -1) -> int:
    try:
        return int(float(value))
    except Exception:
        return default


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else ""


def _latest_batches(runs_root: Path, registry_name: str) -> list[Path]:
    out = []
    for manifest_path in sorted(runs_root.rglob("manifest.json")):
        manifest = _read_json(manifest_path)
        if str(manifest.get("registry", {}).get("name", "")) == registry_name:
            out.append(manifest_path.parent)
    return sorted(out, key=lambda p: p.stat().st_mtime)


def _select_batch(runs_root: Path, registry_name: str, explicit: str = "") -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()
    batches = _latest_batches(runs_root, registry_name)
    if not batches:
        raise FileNotFoundError(f"No batch found for registry {registry_name!r} under {runs_root}")
    return batches[-1]


def _last_metrics_row(run_dir: Path) -> dict[str, str]:
    rows = _read_csv(run_dir / "metrics.csv")
    return rows[-1] if rows else {}


def _run_records(batch_dir: Path) -> list[dict[str, Any]]:
    summary = _read_json(batch_dir / "summary.json")
    matrix_by_runtime = {row.get("runtime_run_id", ""): row for row in _read_csv(batch_dir / "matrix.csv")}
    records = []
    for run in summary.get("runs", []) or []:
        if not isinstance(run, Mapping):
            continue
        run_dir = Path(str(run.get("run_dir", "")))
        runtime_id = str(run.get("runtime_run_id", "") or run_dir.name)
        matrix = matrix_by_runtime.get(runtime_id, {})
        final = _last_metrics_row(run_dir)
        spatial_init = _read_json(run_dir / "metrics" / "lineage_founder_spatial_initialization.json")
        spatial_rows = _read_jsonl(run_dir / "metrics" / "lineage_founder_spatial_survival.jsonl")
        archive_rows = _read_jsonl(run_dir / "archives" / "lineage_qd" / "archive_index.jsonl")
        ok = str(run.get("status", "")).lower() == "ok" or _i(run.get("returncode"), -1) == 0
        records.append(
            {
                "run_id": str(run.get("run_id", matrix.get("run_id", ""))),
                "runtime_run_id": runtime_id,
                "run_dir": str(run_dir),
                "ok": ok,
                "exit_code": _i(run.get("returncode", run.get("exit_code")), -1),
                "context": str((run.get("axes") or {}).get("context", matrix.get("context", ""))),
                "init_arm": str((run.get("axes") or {}).get("init_arm", matrix.get("init_arm", ""))),
                "repeat": str((run.get("axes") or {}).get("repeat", matrix.get("repeat", ""))),
                "seed": _i(run.get("seed", matrix.get("seed")), -1),
                "metrics_hash": _hash(run_dir / "metrics.csv"),
                "config_hash": _hash(run_dir / "config.json"),
                "final_alive": _f(final.get("alive")),
                "final_effective_alive_lineages": _f(final.get("final_effective_alive_lineages", final.get("lineage_effective_alive_lineages"))),
                "final_alive_lineage_entropy": _f(final.get("final_alive_lineage_entropy", final.get("lineage_alive_share_entropy"))),
                "final_top1_alive_share": _f(final.get("final_top1_alive_share", final.get("lineage_top1_alive_share"))),
                "first_effective_lineage_collapse_generation": _i(final.get("first_effective_lineage_collapse_generation"), -1),
                "first_single_alive_lineage_generation": _i(final.get("first_single_alive_lineage_generation"), -1),
                "founder_collision_index": _f(spatial_init.get("founder_collision_index")),
                "founder_spatial_enabled": bool(spatial_init.get("enabled", False)),
                "founder_spatial_min_distance_satisfied": bool(spatial_init.get("min_distance_satisfied", True)),
                "founder_spatial_warning": str(spatial_init.get("warning", "") or ""),
                "archive_entries": len(archive_rows),
                "archive_lineages": len({str(r.get("lineage_id", "")) for r in archive_rows if r.get("lineage_id")}),
                "checkpoint_backed_archive_entries": sum(
                    1
                    for r in archive_rows
                    if bool((r.get("branch_replay_eligibility") or {}).get("eligible"))
                ),
                "live_vs_archive_boundary": all(
                    bool((r.get("live_population_context") or {}).get("live_diversity_not_archive_diversity"))
                    for r in archive_rows
                )
                if archive_rows
                else True,
                "founder_spatial_survival_rows": len(spatial_rows),
            }
        )
    return records


def _mean(values: Sequence[float]) -> float:
    return float(mean(values)) if values else 0.0


def _group_summary(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        groups[(str(record.get("context", "")), str(record.get("init_arm", "")))].append(record)
    rows = []
    for (context, arm), items in sorted(groups.items()):
        rows.append(
            {
                "context": context,
                "init_arm": arm,
                "planned": len(items),
                "completed": sum(1 for x in items if x.get("ok")),
                "failed": sum(1 for x in items if not x.get("ok")),
                "mean_final_alive": round(_mean([_f(x.get("final_alive")) for x in items]), 4),
                "mean_final_effective_alive_lineages": round(
                    _mean([_f(x.get("final_effective_alive_lineages")) for x in items]), 4
                ),
                "mean_final_alive_lineage_entropy": round(
                    _mean([_f(x.get("final_alive_lineage_entropy")) for x in items]), 4
                ),
                "mean_final_top1_alive_share": round(_mean([_f(x.get("final_top1_alive_share")) for x in items]), 4),
                "mean_collapse_generation": round(
                    _mean(
                        [
                            _f(x.get("first_effective_lineage_collapse_generation"))
                            for x in items
                            if _i(x.get("first_effective_lineage_collapse_generation"), -1) >= 0
                        ]
                    ),
                    4,
                ),
                "collapse_observed_count": sum(
                    1 for x in items if _i(x.get("first_effective_lineage_collapse_generation"), -1) >= 0
                ),
                "mean_founder_collision_index": round(_mean([_f(x.get("founder_collision_index")) for x in items]), 6),
                "archive_entries": sum(_i(x.get("archive_entries"), 0) for x in items),
                "archive_lineages": sum(_i(x.get("archive_lineages"), 0) for x in items),
                "checkpoint_backed_archive_entries": sum(_i(x.get("checkpoint_backed_archive_entries"), 0) for x in items),
            }
        )
    return rows


def _deltas(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_key = {(str(r.get("context")), str(r.get("init_arm"))): r for r in rows}
    out = []
    for context in sorted({str(r.get("context")) for r in rows}):
        legacy = by_key.get((context, "legacy"), {})
        for arm in ("founder_balance", "founder_spatial"):
            arm_row = by_key.get((context, arm), {})
            if not legacy or not arm_row:
                continue
            out.append(
                {
                    "context": context,
                    "comparison": f"{arm}_minus_legacy",
                    "delta_mean_final_effective_alive_lineages": round(
                        _f(arm_row.get("mean_final_effective_alive_lineages"))
                        - _f(legacy.get("mean_final_effective_alive_lineages")),
                        4,
                    ),
                    "delta_mean_final_alive_lineage_entropy": round(
                        _f(arm_row.get("mean_final_alive_lineage_entropy"))
                        - _f(legacy.get("mean_final_alive_lineage_entropy")),
                        4,
                    ),
                    "delta_mean_final_top1_alive_share": round(
                        _f(arm_row.get("mean_final_top1_alive_share")) - _f(legacy.get("mean_final_top1_alive_share")),
                        4,
                    ),
                    "delta_collapse_observed_count": _i(arm_row.get("collapse_observed_count"), 0)
                    - _i(legacy.get("collapse_observed_count"), 0),
                    "delta_mean_founder_collision_index": round(
                        _f(arm_row.get("mean_founder_collision_index")) - _f(legacy.get("mean_founder_collision_index")),
                        6,
                    ),
                }
            )
    return out


def _parity_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[str, str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        groups[(str(record.get("context", "")), str(record.get("init_arm", "")), _i(record.get("seed"), -1))].append(record)
    checks = []
    for key, items in sorted(groups.items()):
        if len(items) != 2:
            checks.append({"context": key[0], "init_arm": key[1], "seed": key[2], "passed": False, "reason": "expected two repeats"})
            continue
        metrics_hashes = {str(x.get("metrics_hash", "")) for x in items}
        checks.append(
            {
                "context": key[0],
                "init_arm": key[1],
                "seed": key[2],
                "passed": len(metrics_hashes) == 1,
                "metrics_hashes": sorted(metrics_hashes),
                "run_dirs": [str(x.get("run_dir", "")) for x in items],
            }
        )
    return {
        "schema_version": "founder_spatial_lineage_persistence_parity_v1",
        "planned_checks": len(checks),
        "passed_checks": sum(1 for x in checks if x.get("passed")),
        "failed_checks": sum(1 for x in checks if not x.get("passed")),
        "checks": checks,
    }


def _decision(group_rows: Sequence[Mapping[str, Any]], deltas: Sequence[Mapping[str, Any]], parity: Mapping[str, Any]) -> dict[str, Any]:
    if any(_i(r.get("failed"), 0) for r in group_rows) or _i(parity.get("failed_checks"), 0) > 0:
        return {"decision": "Decision 6", "label": "invalidated", "reason": "failed runs or repeat parity failure"}
    spatial = [d for d in deltas if str(d.get("comparison")) == "founder_spatial_minus_legacy"]
    if not spatial:
        return {"decision": "Decision 5", "label": "reporting insufficient", "reason": "missing spatial-vs-legacy deltas"}
    improved = [
        d
        for d in spatial
        if _f(d.get("delta_mean_final_effective_alive_lineages")) > 0.25
        and _f(d.get("delta_mean_final_alive_lineage_entropy")) >= 0.0
        and _f(d.get("delta_mean_final_top1_alive_share")) <= 0.0
    ]
    worse = [
        d
        for d in spatial
        if _f(d.get("delta_mean_final_effective_alive_lineages")) < -0.25
        or _f(d.get("delta_mean_final_top1_alive_share")) > 0.05
    ]
    if len(improved) == len(spatial):
        return {"decision": "Decision 1", "label": "supported future arm", "reason": "spatial improved live lineage persistence in both contexts"}
    if improved and worse:
        return {"decision": "Decision 4", "label": "mixed/substrate-dependent", "reason": "spatial effects differ by context"}
    if improved:
        return {"decision": "Decision 4", "label": "mixed/substrate-dependent", "reason": "spatial improvement appears context-bound"}
    if any(_f(d.get("delta_mean_final_effective_alive_lineages")) <= 0.0 for d in spatial):
        return {"decision": "Decision 2", "label": "no improvement", "reason": "spatial did not improve live effective lineage persistence"}
    return {"decision": "Decision 3", "label": "pseudo-diversity", "reason": "metrics do not show robust live-persistence improvement"}


def _metric_integrity(
    *,
    main_batch: Path,
    parity_batch: Path,
    main_records: Sequence[Mapping[str, Any]],
    parity_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    checks = []
    for record in list(main_records) + list(parity_records):
        run_dir = Path(str(record.get("run_dir", "")))
        checks.append(
            {
                "run_id": str(record.get("run_id", "")),
                "run_dir": str(run_dir),
                "ok": bool(record.get("ok")),
                "metrics_csv_exists": (run_dir / "metrics.csv").exists(),
                "config_json_exists": (run_dir / "config.json").exists(),
                "lineage_summary_exists": (run_dir / "metrics" / "lineage_summary.csv").exists(),
                "founder_spatial_survival_rows": _i(record.get("founder_spatial_survival_rows"), 0),
                "metrics_sha256": str(record.get("metrics_hash", "")),
            }
        )
    return {
        "schema_version": "founder_spatial_metric_integrity_v1",
        "main_batch": str(main_batch),
        "parity_batch": str(parity_batch),
        "main_runs": len(main_records),
        "parity_runs": len(parity_records),
        "all_runs_ok": all(bool(r.get("ok")) for r in list(main_records) + list(parity_records)),
        "all_metrics_present": all(bool(c["metrics_csv_exists"]) for c in checks),
        "all_configs_present": all(bool(c["config_json_exists"]) for c in checks),
        "checks": checks,
    }


def _no_intervention_audit() -> dict[str, Any]:
    return {
        "schema_version": "founder_spatial_no_intervention_audit_v1",
        "audit_scope": "registry and helper configuration",
        "lineage_protection": False,
        "lineage_rescue": False,
        "archive_reseeding": False,
        "archive_parent_selection": False,
        "post_initialization_lineage_quota": False,
        "reward_shaping_enabled": False,
        "cognition_claim_or_trait_campaign_enabled": False,
        "episode0_reward_weights": {
            "sim_cfg.episode0_fitness_weight": 0.0,
            "sim_cfg.episode0_pred_err_fitness_weight": 0.0,
            "sim_cfg.episode0_thrashing_fitness_weight": 0.0,
            "sim_cfg.episode0_drive_jitter_fitness_weight": 0.0,
            "sim_cfg.episode0_selfharm_fitness_weight": 0.0,
        },
        "passive_archive_enabled": True,
        "archive_claim_boundary": "metadata-only archive; archive diversity is not live diversity and must not influence live selection",
        "founder_balance_boundary": "initialization-only founder label assignment",
        "founder_spatial_boundary": "initialization-only founder placement diagnostic",
    }


def _founder_geometry_summary(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        groups[(str(record.get("context", "")), str(record.get("init_arm", "")))].append(record)
    rows = []
    for (context, arm), items in sorted(groups.items()):
        rows.append(
            {
                "context": context,
                "init_arm": arm,
                "runs": len(items),
                "spatial_enabled_count": sum(1 for x in items if bool(x.get("founder_spatial_enabled"))),
                "mean_founder_collision_index": round(_mean([_f(x.get("founder_collision_index")) for x in items]), 6),
                "min_distance_satisfied_count": sum(
                    1 for x in items if bool(x.get("founder_spatial_min_distance_satisfied"))
                ),
                "warning_count": sum(1 for x in items if str(x.get("founder_spatial_warning", "") or "")),
            }
        )
    return rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def _render_report(summary: Mapping[str, Any]) -> str:
    decision = summary.get("decision", {})
    lines = [
        "# EIEL Founder Spatial Lineage Persistence Probe",
        "",
        "## Boundary",
        "",
        "This is an initialization-only lineage diagnostic. It does not implement lineage protection, rescue, reseeding, reward shaping, cognition, intelligence, emergence, culture, tools, or agriculture claims.",
        "",
        "## Campaign",
        "",
        f"- Main batch: `{summary.get('main_batch_dir')}`",
        f"- Repeat-parity batch: `{summary.get('parity_batch_dir')}`",
        f"- Runtime root: `{summary.get('runs_root')}`",
        f"- Device: `{DEVICE}`",
        f"- Matrix: 3 init arms x 2 clean ecology contexts x 8 seeds, plus 3 repeat-parity checks.",
        f"- Generations/steps/pop/grid: `{GENERATIONS}` / `{STEPS}` / `{POPULATION}` / `{GRID_SIZE}`",
        "",
        "## Group Summary",
        "",
        "| Context | Init arm | Completed | Mean alive | Mean effective live lineages | Mean entropy | Mean top1 share | Collapse count | Mean founder collision | Archive entries |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary.get("group_summary", []):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("context", "")),
                    str(row.get("init_arm", "")),
                    f"{row.get('completed')}/{row.get('planned')}",
                    str(row.get("mean_final_alive")),
                    str(row.get("mean_final_effective_alive_lineages")),
                    str(row.get("mean_final_alive_lineage_entropy")),
                    str(row.get("mean_final_top1_alive_share")),
                    str(row.get("collapse_observed_count")),
                    str(row.get("mean_founder_collision_index")),
                    str(row.get("archive_entries")),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Spatial-vs-Legacy Deltas", ""])
    if summary.get("deltas"):
        lines.extend(
            [
                "| Context | Comparison | Delta effective live lineages | Delta entropy | Delta top1 share | Delta collapse count | Delta founder collision |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in summary.get("deltas", []):
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row.get("context", "")),
                        str(row.get("comparison", "")),
                        str(row.get("delta_mean_final_effective_alive_lineages")),
                        str(row.get("delta_mean_final_alive_lineage_entropy")),
                        str(row.get("delta_mean_final_top1_alive_share")),
                        str(row.get("delta_collapse_observed_count")),
                        str(row.get("delta_mean_founder_collision_index")),
                    ]
                )
                + " |"
            )
    else:
        lines.append("No deltas were available.")
    parity = summary.get("repeat_parity", {})
    lines.extend(
        [
            "",
            "## Repeat Parity",
            "",
            f"- Checks passed: `{parity.get('passed_checks')}/{parity.get('planned_checks')}`",
            f"- Failed checks: `{parity.get('failed_checks')}`",
            "",
            "## Archive Boundary",
            "",
            f"- Total archive entries: `{summary.get('archive_reporting', {}).get('total_archive_entries')}`",
            f"- Checkpoint-backed archive entries: `{summary.get('archive_reporting', {}).get('checkpoint_backed_archive_entries')}`",
            f"- Live-vs-archive boundary preserved: `{summary.get('archive_reporting', {}).get('live_vs_archive_boundary_preserved')}`",
            "",
            "## Decision",
            "",
            f"- Final decision: `{decision.get('decision')}` (`{decision.get('label')}`)",
            f"- Reason: {decision.get('reason')}",
            "",
            "## Evidence",
            "",
            f"- Summary JSON: `{EVIDENCE_DIR / 'founder_spatial_lineage_persistence_summary.json'}`",
            f"- Run summary CSV: `{EVIDENCE_DIR / 'founder_spatial_run_summary.csv'}`",
            f"- Group summary CSV: `{EVIDENCE_DIR / 'founder_spatial_group_summary.csv'}`",
            f"- Repeat parity JSON: `{EVIDENCE_DIR / 'founder_spatial_repeat_parity_summary.json'}`",
            f"- Metric integrity JSON: `{EVIDENCE_DIR / 'founder_spatial_metric_integrity.json'}`",
            f"- No-intervention audit JSON: `{EVIDENCE_DIR / 'founder_spatial_no_intervention_audit.json'}`",
            f"- Founder geometry JSON: `{EVIDENCE_DIR / 'founder_spatial_founder_geometry_summary.json'}`",
            f"- Evidence ZIP: `{EVIDENCE_ZIP}`",
        ]
    )
    return "\n".join(lines) + "\n"


def summarize(args: argparse.Namespace) -> dict[str, Any]:
    runs_root = Path(args.runs_root).expanduser().resolve()
    main_batch = _select_batch(runs_root, "founder_spatial_lineage_persistence_probe_20260704", args.main_batch)
    parity_batch = _select_batch(runs_root, "founder_spatial_lineage_persistence_probe_parity_20260704", args.parity_batch)
    main_records = _run_records(main_batch)
    parity_records = _run_records(parity_batch)
    group_rows = _group_summary(main_records)
    deltas = _deltas(group_rows)
    parity = _parity_summary(parity_records)
    metric_integrity = _metric_integrity(
        main_batch=main_batch,
        parity_batch=parity_batch,
        main_records=main_records,
        parity_records=parity_records,
    )
    no_intervention_audit = _no_intervention_audit()
    founder_geometry = _founder_geometry_summary(main_records)
    archive_reporting = {
        "total_archive_entries": sum(_i(r.get("archive_entries"), 0) for r in main_records),
        "archive_lineages_sum_by_run": sum(_i(r.get("archive_lineages"), 0) for r in main_records),
        "checkpoint_backed_archive_entries": sum(_i(r.get("checkpoint_backed_archive_entries"), 0) for r in main_records),
        "live_vs_archive_boundary_preserved": all(bool(r.get("live_vs_archive_boundary")) for r in main_records),
    }
    campaign_manifest = {
        "schema_version": "founder_spatial_lineage_persistence_campaign_manifest_v1",
        "registries": [str(REGISTRY_PATH), str(PARITY_REGISTRY_PATH)],
        "main_batch_manifest": str(main_batch / "manifest.json"),
        "main_batch_matrix": str(main_batch / "matrix.csv"),
        "main_batch_summary": str(main_batch / "summary.json"),
        "parity_batch_manifest": str(parity_batch / "manifest.json"),
        "parity_batch_matrix": str(parity_batch / "matrix.csv"),
        "parity_batch_summary": str(parity_batch / "summary.json"),
        "generations": GENERATIONS,
        "steps": STEPS,
        "population": POPULATION,
        "grid_size": GRID_SIZE,
        "device": DEVICE,
        "seeds": list(SEEDS),
        "parity_seed": PARITY_SEED,
    }
    summary = {
        "schema_version": "founder_spatial_lineage_persistence_summary_v1",
        "runs_root": str(runs_root),
        "main_batch_dir": str(main_batch),
        "parity_batch_dir": str(parity_batch),
        "planned_main_runs": len(main_records),
        "completed_main_runs": sum(1 for x in main_records if x.get("ok")),
        "failed_main_runs": sum(1 for x in main_records if not x.get("ok")),
        "planned_parity_runs": len(parity_records),
        "completed_parity_runs": sum(1 for x in parity_records if x.get("ok")),
        "failed_parity_runs": sum(1 for x in parity_records if not x.get("ok")),
        "run_summary": main_records,
        "group_summary": group_rows,
        "deltas": deltas,
        "repeat_parity": parity,
        "archive_reporting": archive_reporting,
        "metric_integrity": metric_integrity,
        "no_intervention_audit": no_intervention_audit,
        "founder_geometry_summary": founder_geometry,
        "campaign_manifest": campaign_manifest,
    }
    summary["decision"] = _decision(group_rows, deltas, parity)
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    _write_csv(EVIDENCE_DIR / "founder_spatial_run_summary.csv", main_records)
    _write_csv(EVIDENCE_DIR / "founder_spatial_group_summary.csv", group_rows)
    _write_csv(EVIDENCE_DIR / "founder_spatial_deltas.csv", deltas)
    (EVIDENCE_DIR / "founder_spatial_repeat_parity_summary.json").write_text(
        json.dumps(parity, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (EVIDENCE_DIR / "founder_spatial_decision.json").write_text(
        json.dumps(summary["decision"], indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (EVIDENCE_DIR / "founder_spatial_archive_reporting.json").write_text(
        json.dumps(archive_reporting, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (EVIDENCE_DIR / "founder_spatial_metric_integrity.json").write_text(
        json.dumps(metric_integrity, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (EVIDENCE_DIR / "founder_spatial_no_intervention_audit.json").write_text(
        json.dumps(no_intervention_audit, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (EVIDENCE_DIR / "founder_spatial_founder_geometry_summary.json").write_text(
        json.dumps(founder_geometry, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (EVIDENCE_DIR / "founder_spatial_campaign_manifest.json").write_text(
        json.dumps(campaign_manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (EVIDENCE_DIR / "founder_spatial_lineage_persistence_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    REPORT_DOC.write_text(_render_report(summary), encoding="utf-8")
    EVIDENCE_NOTE.write_text(
        "Founder spatial separation lineage persistence probe evidence.\n\n"
        f"- Report: `{REPORT_DOC}`\n"
        f"- Summary: `{EVIDENCE_DIR / 'founder_spatial_lineage_persistence_summary.json'}`\n"
        f"- Main batch: `{main_batch}`\n"
        f"- Repeat-parity batch: `{parity_batch}`\n"
        f"- Decision: `{summary['decision']['decision']}` (`{summary['decision']['label']}`)\n",
        encoding="utf-8",
    )
    write_zip()
    return summary


def write_zip() -> None:
    include = [
        REPORT_DOC,
        EVIDENCE_NOTE,
        REGISTRY_PATH,
        PARITY_REGISTRY_PATH,
        EVIDENCE_DIR / "founder_spatial_lineage_persistence_summary.json",
        EVIDENCE_DIR / "founder_spatial_run_summary.csv",
        EVIDENCE_DIR / "founder_spatial_group_summary.csv",
        EVIDENCE_DIR / "founder_spatial_deltas.csv",
        EVIDENCE_DIR / "founder_spatial_repeat_parity_summary.json",
        EVIDENCE_DIR / "founder_spatial_decision.json",
        EVIDENCE_DIR / "founder_spatial_archive_reporting.json",
        EVIDENCE_DIR / "founder_spatial_metric_integrity.json",
        EVIDENCE_DIR / "founder_spatial_no_intervention_audit.json",
        EVIDENCE_DIR / "founder_spatial_founder_geometry_summary.json",
        EVIDENCE_DIR / "founder_spatial_campaign_manifest.json",
    ]
    EVIDENCE_ZIP.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(EVIDENCE_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in include:
            if path.exists():
                zf.write(path, path.as_posix())


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("write-registries")
    p_sum = sub.add_parser("summarize")
    p_sum.add_argument("--runs-root", required=True)
    p_sum.add_argument("--main-batch", default="")
    p_sum.add_argument("--parity-batch", default="")
    args = parser.parse_args(argv)
    if args.command == "write-registries":
        write_registries()
        print(str(REGISTRY_PATH))
        print(str(PARITY_REGISTRY_PATH))
        return 0
    if args.command == "summarize":
        summary = summarize(args)
        print(json.dumps({"decision": summary["decision"], "evidence_zip": str(EVIDENCE_ZIP)}, sort_keys=True))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
