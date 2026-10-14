#!/usr/bin/env python3
"""Run the MUTRAD-0C mutation-distance baseline/readiness diagnostic.

This is reporting infrastructure only. It enables the existing lineage
reproduction funnel event stream, reads MUTRAD-0 parent-child genome-distance
fields, and writes compact evidence summaries.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
ENGINE_ROOT = REPO_ROOT / "engine"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(ENGINE_ROOT) not in sys.path:
    sys.path.insert(0, str(ENGINE_ROOT))

from env.env_config import EnvConfig  # noqa: E402
from evolution.population_pkg.population_config import PopConfig  # noqa: E402
from simulation.metrics_pkg.metrics_logger import MetricsLogger  # noqa: E402
from simulation.runner import SimulationRunner  # noqa: E402
from simulation.runner_pkg.runner_config import SimConfig  # noqa: E402


NO_CLAIMS = (
    "MUTRAD-0C-BASE is a baseline measurement/readiness task for environmental mutation volatility. "
    "It does not show intelligence, emergence, consciousness, memory selection, prediction selection, "
    "general cognition, open-endedness, or superintelligence. It does not add direct cognition reward, "
    "direct memory reward, direct prediction reward, diversity reward, lineage rescue, protected "
    "reproduction, mutation weather, local mutagenic hotspots, archive reseeding, or survival subsidy. "
    "Any future positive result may support only bounded improvement in heritable novelty/search "
    "conditions under tested configurations."
)


@dataclass(frozen=True)
class ContextDefinition:
    context_id: str
    label: str
    description: str
    env_overrides: tuple[tuple[str, bool | int | float | str], ...] = ()


TUNED_JUNGLE = (
    ("jungle_predator_scent_weight", 0.02),
    ("jungle_predator_waste_weight", 0.02),
    ("jungle_predator_density_weight", 0.02),
    ("jungle_predator_visibility_weight", 0.02),
    ("jungle_predator_risk_cap", 0.05),
    ("jungle_predator_health_damage", 0.25),
    ("patch_waste_regrowth_penalty_scale", 0.5),
    ("patch_waste_regrowth_penalty_max", 0.5),
    ("patch_density_regrowth_penalty_scale", 0.5),
    ("patch_density_regrowth_penalty_max", 0.5),
    ("patch_regrowth_min_multiplier", 0.1),
    ("jungle_cover_damage_reduction_weight", 0.4),
    ("jungle_shelter_damage_reduction_weight", 0.4),
    ("jungle_terrain_damage_multiplier_min", 0.15),
)


CONTEXTS: tuple[ContextDefinition, ...] = (
    ContextDefinition(
        context_id="flat_control",
        label="Flat/control baseline",
        description="Legacy flat world with Patch/Jungle/WDL substrate disabled.",
    ),
    ContextDefinition(
        context_id="patch_jungle_collapse_prone",
        label="Patch/Jungle collapse-prone",
        description="Full Jungle pressure without ecological recovery.",
        env_overrides=(
            ("patch_world_enable", True),
            ("patch_world_observation_enable", True),
            ("patch_world_waste_consequence_enable", True),
            ("jungle_predator_pressure_enable", True),
            ("jungle_reproduction_emissions_enable", True),
            ("jungle_terrain_protection_enable", True),
            ("jungle_ecological_recovery_enable", False),
            ("patch_cover_default", 0.25),
            ("patch_shelter_default", 0.15),
            ("patch_shelter_from_cover_enable", False),
        ),
    ),
    ContextDefinition(
        context_id="patch_jungle_recovery",
        label="Tuned full Jungle recovery",
        description="Tuned full Jungle pressure with bounded ecological recovery.",
        env_overrides=(
            ("patch_world_enable", True),
            ("patch_world_observation_enable", True),
            ("patch_world_waste_consequence_enable", True),
            ("jungle_predator_pressure_enable", True),
            ("jungle_reproduction_emissions_enable", True),
            ("jungle_terrain_protection_enable", True),
            ("jungle_ecological_recovery_enable", True),
            ("patch_cover_default", 0.25),
            ("patch_shelter_default", 0.15),
            ("patch_shelter_from_cover_enable", False),
            ("jungle_recovery_regrowth_boost_scale", 0.05),
            ("jungle_recovery_regrowth_boost_max", 0.1),
            *TUNED_JUNGLE,
        ),
    ),
    ContextDefinition(
        context_id="wdl_delayed_hazard",
        label="WDL delayed-hazard context",
        description="Patch World plus WDL delayed hazard cue/active schedule.",
        env_overrides=(
            ("patch_world_enable", True),
            ("patch_world_observation_enable", True),
            ("wdl_delayed_hazard_enable", True),
            ("wdl_delayed_hazard_observation_enable", True),
            ("wdl_delayed_hazard_delay_steps", 4),
            ("wdl_delayed_hazard_cue_duration_steps", 2),
            ("wdl_delayed_hazard_active_duration_steps", 3),
            ("wdl_delayed_hazard_event_period_steps", 12),
            ("wdl_delayed_hazard_radius", 1),
            ("wdl_delayed_hazard_intensity", 0.10),
        ),
    ),
)


DISTANCE_FIELDS = (
    "parent_child_genome_distance_mean",
    "parent_child_genome_distance_max",
    "parent_child_genome_scalar_distance_mean",
    "parent_child_genome_memory_field_delta",
    "parent_child_genome_prediction_field_delta",
    "parent_child_genome_module_gate_delta",
)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in ("", None):
            return float(default)
        out = float(value)
    except Exception:
        return float(default)
    return out if math.isfinite(out) else float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value in ("", None):
            return int(default)
        return int(float(value))
    except Exception:
        return int(default)


def _mean(values: Iterable[Any]) -> float:
    vals = [_safe_float(v) for v in values if v not in ("", None)]
    return float(sum(vals) / len(vals)) if vals else 0.0


def _quantile(values: Iterable[Any], q: float) -> float:
    vals = sorted(_safe_float(v) for v in values if v not in ("", None))
    if not vals:
        return 0.0
    if len(vals) == 1:
        return float(vals[0])
    idx = max(0.0, min(1.0, float(q))) * (len(vals) - 1)
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return float(vals[lo])
    frac = idx - lo
    return float(vals[lo] * (1.0 - frac) + vals[hi] * frac)


def _effective_count(counts: Mapping[str, int]) -> float:
    total = float(sum(max(0, int(v)) for v in counts.values()))
    if total <= 0.0:
        return 0.0
    entropy = 0.0
    for count in counts.values():
        p = max(0.0, float(count)) / total
        if p > 0.0:
            entropy -= p * math.log(p)
    return float(math.exp(entropy))


def _top_share(counts: Mapping[str, int]) -> float:
    total = float(sum(max(0, int(v)) for v in counts.values()))
    if total <= 0.0:
        return 0.0
    return float(max(max(0, int(v)) for v in counts.values()) / total)


def _sha256(path: Path) -> str:
    if not path.exists() or not path.is_file():
        return ""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _iter_events(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "metrics" / "lineage_reproduction_funnel_events.jsonl"
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _read_last_generation_summary(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "metrics" / "lineage_reproduction_funnel_generation_summary.csv"
    if not path.exists() or path.stat().st_size <= 0:
        return {}
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}
    generations = [_safe_int(row.get("generation", -1), -1) for row in rows]
    last_gen = max(generations)
    final = [row for row in rows if _safe_int(row.get("generation", -1), -1) == last_gen]
    alive_counts = {str(row.get("lineage_id") or ""): _safe_int(row.get("post_refill_alive_count", 0), 0) for row in final}
    alive_counts = {k: v for k, v in alive_counts.items() if k and v > 0}
    offspring_counts = {str(row.get("lineage_id") or ""): _safe_int(row.get("successful_births", 0), 0) for row in final}
    offspring_counts = {k: v for k, v in offspring_counts.items() if k and v > 0}
    return {
        "final_generation": int(last_gen),
        "effective_alive_lineages_final": _effective_count(alive_counts),
        "top1_alive_share_final": _top_share(alive_counts),
        "offspring_effective_lineages_final_generation": _effective_count(offspring_counts),
        "top1_offspring_share_final_generation": _top_share(offspring_counts),
        "final_generation_birth_success_count": int(sum(offspring_counts.values())),
    }


def _genome_payload(genome: Any) -> dict[str, Any]:
    if genome is None:
        return {}
    payload = getattr(genome, "c1_payload", None)
    if isinstance(payload, Mapping):
        return dict(payload)
    if hasattr(genome, "to_dict"):
        try:
            data = genome.to_dict()
            return dict(data) if isinstance(data, Mapping) else {}
        except Exception:
            return {}
    return {}


def _count_payload_list(genome: Any, field: str) -> int:
    payload = _genome_payload(genome)
    items = payload.get(field, [])
    if isinstance(items, list):
        return int(len(items))
    return _safe_int(getattr(genome, f"{field}_count", 0), 0)


def _structural_audit(sim: SimulationRunner) -> dict[str, Any]:
    agents = list(getattr(sim.pop, "agents", {}).values())
    genomes = [getattr(agent, "genome", None) for agent in agents]
    modules = [_count_payload_list(genome, "modules") for genome in genomes]
    connections = [_count_payload_list(genome, "connections") for genome in genomes]
    c1_applied = [
        str(getattr(genome, "c1_operator_applied", "") or _genome_payload(genome).get("genes", {}).get("c1_operator_applied", "") or "")
        for genome in genomes
    ]
    c1_noop = [
        _safe_int(getattr(genome, "c1_operator_noop_count", _genome_payload(genome).get("genes", {}).get("c1_operator_noop_count", 0)), 0)
        for genome in genomes
    ]
    c2_modules = [
        _safe_int(getattr(genome, "c2_matched_modules", _genome_payload(genome).get("genes", {}).get("c2_matched_modules", 0)), 0)
        for genome in genomes
    ]
    c2_connections = [
        _safe_int(getattr(genome, "c2_matched_connections", _genome_payload(genome).get("genes", {}).get("c2_matched_connections", 0)), 0)
        for genome in genomes
    ]
    return {
        "final_agent_count": int(len(agents)),
        "final_module_count_mean": _mean(modules),
        "final_module_count_max": int(max(modules) if modules else 0),
        "final_connection_count_mean": _mean(connections),
        "final_connection_count_max": int(max(connections) if connections else 0),
        "final_genomes_with_modules": int(sum(1 for value in modules if value > 0)),
        "final_genomes_with_connections": int(sum(1 for value in connections if value > 0)),
        "final_c1_operator_applied_nonempty_count": int(sum(1 for value in c1_applied if value.strip())),
        "final_c1_operator_noop_total": int(sum(c1_noop)),
        "final_c2_matched_modules_total": int(sum(c2_modules)),
        "final_c2_matched_connections_total": int(sum(c2_connections)),
    }


def summarize_cell_events(events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    births = [dict(row) for row in events if str(row.get("event_type") or "") == "birth_success"]
    failures = [dict(row) for row in events if str(row.get("event_type") or "").startswith("birth_fail")]
    child_lineage_counts: dict[str, int] = {}
    parent_child_distances: list[float] = []
    for row in births:
        lineage = str(row.get("child_lineage_id") or row.get("lineage_id") or row.get("parent_lineage_id") or "")
        if lineage:
            child_lineage_counts[lineage] = child_lineage_counts.get(lineage, 0) + 1
        parent_child_distances.append(_safe_float(row.get("parent_child_distance", 0.0), 0.0))

    summary: dict[str, Any] = {
        "event_jsonl_rows": int(len(events)),
        "birth_success_count": int(len(births)),
        "failed_birth_count_if_available": int(len(failures)),
        "offspring_count": int(len(births)),
        "offspring_effective_lineages": _effective_count(child_lineage_counts),
        "top1_offspring_share_if_available": _top_share(child_lineage_counts),
        "parent_child_distance_q50": _quantile(parent_child_distances, 0.50),
        "parent_child_distance_q90": _quantile(parent_child_distances, 0.90),
        "parent_child_distance_max": max(parent_child_distances) if parent_child_distances else 0.0,
    }
    for field in DISTANCE_FIELDS:
        values = [_safe_float(row.get(field, 0.0), 0.0) for row in births]
        summary[f"{field}_event_mean"] = _mean(values)
        summary[f"{field}_event_q90"] = _quantile(values, 0.90)
        summary[f"{field}_event_max"] = max(values) if values else 0.0
        summary[f"{field}_present_birth_rows"] = int(sum(1 for row in births if row.get(field, "") not in ("", None)))
    summary["memory_dim_delta_event_count"] = int(
        sum(1 for row in births if _safe_float(row.get("parent_child_genome_memory_field_delta", 0.0), 0.0) > 0.0)
    )
    summary["pred_lr_delta_event_count"] = int(
        sum(1 for row in births if _safe_float(row.get("parent_child_genome_prediction_field_delta", 0.0), 0.0) > 0.0)
    )
    summary["module_gate_delta_event_count"] = int(
        sum(1 for row in births if _safe_float(row.get("parent_child_genome_module_gate_delta", 0.0), 0.0) > 0.0)
    )
    return summary


def _apply_common_env(env_cfg: EnvConfig) -> None:
    env_cfg.reproduce_requires_mate = False
    env_cfg.reproduce_try_prob = 1.0
    env_cfg.reproduce_cooldown_steps = 1
    env_cfg.reproduce_parent_energy_share = 0.1
    env_cfg.reproduce_min_energy_buffer = 0.0
    env_cfg.food_spawn_per_step = max(0.0, float(getattr(env_cfg, "food_spawn_per_step", 0.0) or 0.0))


def _configure_cell(
    *,
    run_dir: Path,
    context: ContextDefinition,
    seed: int,
    generations: int,
    steps: int,
    population: int,
    grid: int,
    device: str,
) -> SimulationRunner:
    env_cfg = EnvConfig(width=int(grid), height=int(grid))
    _apply_common_env(env_cfg)
    for key, value in context.env_overrides:
        setattr(env_cfg, key, value)
    pop_cfg = PopConfig(population_size=int(population), parent_fraction=0.5, cull_fraction=0.25)
    sim_cfg = SimConfig(
        seed=int(seed),
        generations=int(generations),
        steps_per_generation=int(steps),
        device=str(device),
        bank_device=str(device),
        behavior_log_enable=False,
        policy_deterministic_actions=True,
    )
    sim_cfg.phase_e_e1_enable = False
    sim_cfg.epi_enable = False
    sim_cfg.checkpoint_bundle_v2_period_gen = 0
    sim_cfg.enable_cognitive_trait_registry = False
    sim_cfg.enable_cognitive_status_metrics = False
    sim_cfg.enable_cognitive_budget_ledger = False
    sim_cfg.lineage_reproduction_funnel_reporting_enable = True
    sim_cfg.lineage_reproduction_funnel_write_events = True
    sim_cfg.lineage_reproduction_funnel_event_schema_version = 4
    sim_cfg.lineage_reproduction_funnel_write_scan_action_events = True
    return SimulationRunner(
        env_cfg=env_cfg,
        pop_cfg=pop_cfg,
        sim_cfg=sim_cfg,
        logger=MetricsLogger(str(run_dir)),
        run_dir=str(run_dir),
        perf=False,
    )


def _run_cell(
    *,
    raw_root: Path,
    context: ContextDefinition,
    seed: int,
    generations: int,
    steps: int,
    population: int,
    grid: int,
    device: str,
) -> dict[str, Any]:
    run_id = f"{context.context_id}_seed{int(seed)}"
    run_dir = raw_root / run_id
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    sim = _configure_cell(
        run_dir=run_dir,
        context=context,
        seed=int(seed),
        generations=int(generations),
        steps=int(steps),
        population=int(population),
        grid=int(grid),
        device=str(device),
    )
    sim.run(start_gen=0, skip_init=False)
    try:
        sim.logger.close()
    except Exception:
        pass
    events = _iter_events(run_dir)
    event_summary = summarize_cell_events(events)
    generation_summary = _read_last_generation_summary(run_dir)
    run_summary = _read_json(run_dir / "metrics" / "lineage_reproduction_funnel_run_summary.json")
    structural = _structural_audit(sim)
    return {
        "context_id": context.context_id,
        "context_label": context.label,
        "seed": int(seed),
        "run_dir": str(run_dir),
        "completed": True,
        "generations": int(generations),
        "steps_per_generation": int(steps),
        "population": int(population),
        "grid": int(grid),
        "device": str(device),
        "metrics_sha256": _sha256(run_dir / "metrics.csv"),
        "config_sha256": _sha256(run_dir / "config.json"),
        "lineage_funnel_events_sha256": _sha256(
            run_dir / "metrics" / "lineage_reproduction_funnel_events.jsonl"
        ),
        "lineage_funnel_run_summary_sha256": _sha256(
            run_dir / "metrics" / "lineage_reproduction_funnel_run_summary.json"
        ),
        "run_summary_birth_success_rows": _safe_int(run_summary.get("birth_success_rows", 0), 0),
        "run_summary_birth_fail_rows": _safe_int(run_summary.get("birth_fail_rows", 0), 0),
        "event_schema_version": _safe_int(run_summary.get("event_schema_version", 0), 0),
        **event_summary,
        **generation_summary,
        **structural,
    }


def _aggregate_contexts(cells: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in cells:
        grouped.setdefault(str(row.get("context_id") or ""), []).append(row)
    out: list[dict[str, Any]] = []
    for context_id, rows in sorted(grouped.items()):
        birth_counts = [_safe_int(row.get("birth_success_count", 0), 0) for row in rows]
        out.append(
            {
                "context_id": context_id,
                "context_label": str(rows[0].get("context_label") or context_id),
                "cell_count": int(len(rows)),
                "seed_count": int(len({int(row.get("seed", 0)) for row in rows})),
                "total_birth_success_count": int(sum(birth_counts)),
                "min_birth_success_count": int(min(birth_counts) if birth_counts else 0),
                "failed_birth_count_if_available": int(
                    sum(_safe_int(row.get("failed_birth_count_if_available", 0), 0) for row in rows)
                ),
                "parent_child_genome_distance_mean": _mean(
                    row.get("parent_child_genome_distance_mean_event_mean", 0.0) for row in rows
                ),
                "parent_child_genome_distance_q90_mean": _mean(
                    row.get("parent_child_genome_distance_mean_event_q90", 0.0) for row in rows
                ),
                "parent_child_genome_distance_max": max(
                    (_safe_float(row.get("parent_child_genome_distance_max_event_max", 0.0), 0.0) for row in rows),
                    default=0.0,
                ),
                "parent_child_genome_scalar_distance_mean": _mean(
                    row.get("parent_child_genome_scalar_distance_mean_event_mean", 0.0) for row in rows
                ),
                "parent_child_genome_memory_field_delta_mean": _mean(
                    row.get("parent_child_genome_memory_field_delta_event_mean", 0.0) for row in rows
                ),
                "parent_child_genome_prediction_field_delta_mean": _mean(
                    row.get("parent_child_genome_prediction_field_delta_event_mean", 0.0) for row in rows
                ),
                "parent_child_genome_module_gate_delta_mean": _mean(
                    row.get("parent_child_genome_module_gate_delta_event_mean", 0.0) for row in rows
                ),
                "memory_dim_delta_event_count": int(
                    sum(_safe_int(row.get("memory_dim_delta_event_count", 0), 0) for row in rows)
                ),
                "pred_lr_delta_event_count": int(
                    sum(_safe_int(row.get("pred_lr_delta_event_count", 0), 0) for row in rows)
                ),
                "module_gate_delta_event_count": int(
                    sum(_safe_int(row.get("module_gate_delta_event_count", 0), 0) for row in rows)
                ),
                "effective_alive_lineages_final_mean": _mean(
                    row.get("effective_alive_lineages_final", 0.0) for row in rows
                ),
                "top1_alive_share_final_mean": _mean(row.get("top1_alive_share_final", 0.0) for row in rows),
                "offspring_effective_lineages_mean": _mean(row.get("offspring_effective_lineages", 0.0) for row in rows),
                "top1_offspring_share_mean": _mean(
                    row.get("top1_offspring_share_if_available", 0.0) for row in rows
                ),
                "final_module_count_max": int(
                    max((_safe_int(row.get("final_module_count_max", 0), 0) for row in rows), default=0)
                ),
                "final_connection_count_max": int(
                    max((_safe_int(row.get("final_connection_count_max", 0), 0) for row in rows), default=0)
                ),
                "final_c1_operator_applied_nonempty_count": int(
                    sum(_safe_int(row.get("final_c1_operator_applied_nonempty_count", 0), 0) for row in rows)
                ),
                "final_c2_matched_modules_total": int(
                    sum(_safe_int(row.get("final_c2_matched_modules_total", 0), 0) for row in rows)
                ),
            }
        )
    return out


def classify_metrics(cells: Sequence[Mapping[str, Any]], context_summary: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    missing_field_cells = []
    for row in cells:
        births = _safe_int(row.get("birth_success_count", 0), 0)
        if births <= 0:
            continue
        missing = [
            field
            for field in DISTANCE_FIELDS
            if _safe_int(row.get(f"{field}_present_birth_rows", 0), 0) < births
        ]
        if missing:
            missing_field_cells.append(
                {"context_id": row.get("context_id"), "seed": row.get("seed"), "missing_fields": missing}
            )
    total_births = int(sum(_safe_int(row.get("birth_success_count", 0), 0) for row in cells))
    cells_with_births = int(sum(1 for row in cells if _safe_int(row.get("birth_success_count", 0), 0) > 0))
    module_delta_total = int(sum(_safe_int(row.get("module_gate_delta_event_count", 0), 0) for row in cells))
    memory_delta_total = int(sum(_safe_int(row.get("memory_dim_delta_event_count", 0), 0) for row in cells))
    pred_delta_total = int(sum(_safe_int(row.get("pred_lr_delta_event_count", 0), 0) for row in cells))
    max_distance = max(
        (_safe_float(row.get("parent_child_genome_distance_max_event_max", 0.0), 0.0) for row in cells),
        default=0.0,
    )
    mean_distance = _mean(row.get("parent_child_genome_distance_mean_event_mean", 0.0) for row in cells)
    structural_payload_active = any(
        _safe_int(row.get("final_module_count_max", 0), 0) > 0
        or _safe_int(row.get("final_connection_count_max", 0), 0) > 0
        or _safe_int(row.get("final_c2_matched_modules_total", 0), 0) > 0
        or _safe_int(row.get("final_c2_matched_connections_total", 0), 0) > 0
        for row in cells
    )
    c1_noop_annotation_count = int(sum(_safe_int(row.get("final_c1_operator_noop_total", 0), 0) for row in cells))
    c1_annotation_nonempty_count = int(
        sum(_safe_int(row.get("final_c1_operator_applied_nonempty_count", 0), 0) for row in cells)
    )
    return {
        "schema_version": "mutrad0c_metric_classification_v1",
        "total_cell_count": int(len(cells)),
        "context_count": int(len(context_summary)),
        "total_birth_success_count": total_births,
        "cells_with_birth_success": cells_with_births,
        "cells_without_birth_success": int(len(cells) - cells_with_births),
        "missing_distance_field_cell_count": int(len(missing_field_cells)),
        "missing_distance_field_cells": missing_field_cells,
        "parent_child_genome_distance_mean_over_cells": mean_distance,
        "parent_child_genome_distance_max_observed": max_distance,
        "memory_dim_delta_event_count": memory_delta_total,
        "pred_lr_delta_event_count": pred_delta_total,
        "module_gate_delta_event_count": module_delta_total,
        "structural_payload_active_in_final_population": bool(structural_payload_active),
        "c1_operator_annotation_nonempty_count": c1_annotation_nonempty_count,
        "c1_operator_noop_annotation_count": c1_noop_annotation_count,
        "mutation_appears_narrow_local_by_reporting_proxy": bool(
            total_births > 0
            and not missing_field_cells
            and module_delta_total == 0
            and not structural_payload_active
            and max_distance <= 0.25
        ),
    }


def decide(classification: Mapping[str, Any]) -> dict[str, Any]:
    total_cells = _safe_int(classification.get("total_cell_count", 0), 0)
    births = _safe_int(classification.get("total_birth_success_count", 0), 0)
    cells_without_births = _safe_int(classification.get("cells_without_birth_success", 0), 0)
    missing = _safe_int(classification.get("missing_distance_field_cell_count", 0), 0)
    narrow = bool(classification.get("mutation_appears_narrow_local_by_reporting_proxy", False))
    if births <= 0 or cells_without_births == total_cells:
        decision = "MUTRAD-0C-BASE-D"
        recommendation = "Do not proceed; rerun with interpretable births or repair config gaps."
    elif missing > 0:
        decision = "MUTRAD-0C-BASE-B"
        recommendation = "Repair or extend reporting before changing mutation mechanics."
    elif narrow:
        decision = "MUTRAD-0C-BASE-A"
        recommendation = "Proceed to a default-off scalar-only mutation weather pilot; do not add hotspots."
    else:
        decision = "MUTRAD-0C-BASE-C"
        recommendation = "Current mutation reporting is sufficient and not narrow/local enough to justify weather yet."
    return {
        "schema_version": "mutrad0c_decision_v1",
        "decision": decision,
        "mutrad1_recommendation": recommendation,
        "no_claims": NO_CLAIMS,
        "decision_inputs": dict(classification),
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _parse_seeds(value: str) -> tuple[int, ...]:
    seeds = tuple(int(part.strip()) for part in str(value).split(",") if part.strip())
    if not seeds:
        raise argparse.ArgumentTypeError("at least one seed is required")
    return seeds


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--evidence-dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seeds", type=_parse_seeds, default=(101, 202, 303))
    parser.add_argument("--generations", type=int, default=6)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--population", type=int, default=64)
    parser.add_argument("--grid", type=int, default=16)
    args = parser.parse_args()

    raw_root = Path(args.raw_root)
    evidence_dir = Path(args.evidence_dir)
    raw_root.mkdir(parents=True, exist_ok=True)
    evidence_dir.mkdir(parents=True, exist_ok=True)

    cells: list[dict[str, Any]] = []
    for context in CONTEXTS:
        for seed in args.seeds:
            cells.append(
                _run_cell(
                    raw_root=raw_root,
                    context=context,
                    seed=int(seed),
                    generations=int(args.generations),
                    steps=int(args.steps),
                    population=int(args.population),
                    grid=int(args.grid),
                    device=str(args.device),
                )
            )

    context_summary = _aggregate_contexts(cells)
    classification = classify_metrics(cells, context_summary)
    decision = decide(classification)
    manifest = {
        "schema_version": "mutrad0c_manifest_v1",
        "raw_root": str(raw_root),
        "evidence_dir": str(evidence_dir),
        "device": str(args.device),
        "seeds": list(args.seeds),
        "generations": int(args.generations),
        "steps_per_generation": int(args.steps),
        "population": int(args.population),
        "grid": int(args.grid),
        "contexts": [
            {
                "context_id": context.context_id,
                "label": context.label,
                "description": context.description,
                "env_overrides": list(context.env_overrides),
            }
            for context in CONTEXTS
        ],
        "mechanics_policy": (
            "reporting-only baseline: no mutation weather, no local hotspots, no lineage rescue, "
            "no protected reproduction, no reward changes"
        ),
    }
    _write_csv(evidence_dir / "mutrad0c_cells.csv", cells)
    _write_json(evidence_dir / "mutrad0c_cells.json", cells)
    _write_csv(evidence_dir / "mutrad0c_context_summary.csv", context_summary)
    _write_json(evidence_dir / "mutrad0c_context_summary.json", context_summary)
    _write_json(evidence_dir / "mutrad0c_metric_classification.json", classification)
    _write_json(evidence_dir / "mutrad0c_decision_summary.json", decision)
    _write_json(evidence_dir / "mutrad0c_manifest.json", manifest)
    return 0 if str(decision["decision"]) != "MUTRAD-0C-BASE-D" else 2


if __name__ == "__main__":
    raise SystemExit(main())
