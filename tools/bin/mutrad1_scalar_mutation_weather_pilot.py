#!/usr/bin/env python3
from __future__ import annotations

"""Run the tiny MUTRAD-1 scalar mutation weather pilot."""

import argparse
import csv
import hashlib
import json
import math
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
ENGINE_ROOT = REPO_ROOT / "engine"
for root in (REPO_ROOT, ENGINE_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from env.env_config import EnvConfig  # noqa: E402
from evolution.population_pkg.population_config import PopConfig  # noqa: E402
from simulation.metrics_pkg.metrics_logger import MetricsLogger  # noqa: E402
from simulation.runner import SimulationRunner  # noqa: E402
from simulation.runner_pkg.runner_config import SimConfig  # noqa: E402


NO_CLAIMS = (
    "MUTRAD-1 is a default-off scalar mutation-weather pilot for bounded environmental mutation volatility. "
    "It does not show intelligence, emergence, consciousness, memory selection, prediction selection, "
    "general cognition, open-endedness, or superintelligence. It does not add direct cognition reward, "
    "direct memory reward, direct prediction reward, diversity reward, lineage rescue, protected reproduction, "
    "local mutagenic hotspots, structural mutation weather, archive reseeding, or survival subsidy. Any positive "
    "result supports only bounded evidence that scalar mutation weather can safely increase heritable novelty "
    "under tested configurations."
)


@dataclass(frozen=True)
class Context:
    context_id: str
    label: str
    overrides: tuple[tuple[str, bool | int | float | str], ...] = ()


@dataclass(frozen=True)
class Arm:
    arm_id: str
    label: str
    weather_enabled: bool = False
    profile: str = ""
    fixed_multiplier: float = 1.0
    fixture_generations: str = ""


CONTEXTS = (
    Context("flat_control", "Flat/control baseline"),
    Context(
        "patch_jungle_representative",
        "Patch/Jungle representative",
        (
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
            ("jungle_predator_risk_cap", 0.05),
            ("jungle_predator_health_damage", 0.25),
        ),
    ),
    Context(
        "wdl_delayed_hazard",
        "WDL delayed-hazard representative",
        (
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


ARMS = (
    Arm("baseline_current_mutation", "Baseline current mutation"),
    Arm(
        "fixed_high_scalar_mutation_control",
        "Fixed-high scalar mutation control",
        weather_enabled=True,
        fixed_multiplier=10.0,
    ),
    Arm(
        "safe_pulse_v0",
        "safe_pulse_v0 scalar weather",
        weather_enabled=True,
        profile="safe_pulse_v0",
        fixture_generations="[1]",
    ),
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


def _mean(values: Sequence[Any]) -> float:
    vals = [_safe_float(v) for v in values if v not in ("", None)]
    return float(sum(vals) / len(vals)) if vals else 0.0


def _effective_count(counts: Mapping[str, int]) -> float:
    total = float(sum(max(0, int(v)) for v in counts.values()))
    if total <= 0:
        return 0.0
    entropy = 0.0
    for count in counts.values():
        p = max(0.0, float(count)) / total
        if p > 0:
            entropy -= p * math.log(p)
    return float(math.exp(entropy))


def _top_share(counts: Mapping[str, int]) -> float:
    total = float(sum(max(0, int(v)) for v in counts.values()))
    if total <= 0:
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


def _normalized_events_sha256(run_dir: Path) -> str:
    rows = []
    for row in _iter_events(run_dir):
        clean = dict(row)
        clean.pop("run_id", None)
        rows.append(clean)
    blob = json.dumps(rows, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


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


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _read_final_generation(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "metrics" / "lineage_reproduction_funnel_generation_summary.csv"
    if not path.exists() or path.stat().st_size <= 0:
        return {}
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}
    last_gen = max(_safe_int(row.get("generation", -1), -1) for row in rows)
    final = [row for row in rows if _safe_int(row.get("generation", -1), -1) == last_gen]
    alive = {
        str(row.get("lineage_id") or ""): _safe_int(row.get("post_refill_alive_count", 0), 0)
        for row in final
    }
    offspring = {
        str(row.get("lineage_id") or ""): _safe_int(row.get("successful_births", 0), 0)
        for row in final
    }
    alive = {k: v for k, v in alive.items() if k and v > 0}
    offspring = {k: v for k, v in offspring.items() if k and v > 0}
    return {
        "final_generation": int(last_gen),
        "effective_alive_lineages_final": _effective_count(alive),
        "top1_alive_share_final": _top_share(alive),
        "effective_reproducing_lineages_if_available": _effective_count(offspring),
        "top1_offspring_share_final_generation": _top_share(offspring),
    }


def _configure_cell(
    *,
    run_dir: Path,
    context: Context,
    arm: Arm,
    seed: int,
    generations: int,
    steps: int,
    population: int,
    grid: int,
    device: str,
) -> SimulationRunner:
    env_cfg = EnvConfig(width=int(grid), height=int(grid))
    env_cfg.reproduce_requires_mate = False
    env_cfg.reproduce_try_prob = 1.0
    env_cfg.reproduce_cooldown_steps = 1
    env_cfg.reproduce_parent_energy_share = 0.1
    env_cfg.reproduce_min_energy_buffer = 0.0
    for key, value in context.overrides:
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
    sim_cfg.mutation_weather_enabled = bool(arm.weather_enabled)
    sim_cfg.mutation_weather_profile = str(arm.profile)
    sim_cfg.mutation_weather_seed_salt = "mutrad1-pilot"
    sim_cfg.mutation_weather_max_scalar_multiplier = 10.0
    sim_cfg.mutation_weather_apply_to_scalar_sigma = True
    sim_cfg.mutation_weather_apply_to_mutation_probability = False
    sim_cfg.mutation_weather_structural_enabled = False
    sim_cfg.mutation_weather_fixture_generations = str(arm.fixture_generations)
    sim_cfg._mutation_weather_fixture_generations = tuple(
        int(part.strip())
        for part in str(arm.fixture_generations).replace("[", "").replace("]", "").split(",")
        if part.strip()
    )
    sim_cfg.mutation_weather_fixed_scalar_multiplier = float(arm.fixed_multiplier)
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
    context: Context,
    arm: Arm,
    seed: int,
    repeat_index: int,
    generations: int,
    steps: int,
    population: int,
    grid: int,
    device: str,
) -> dict[str, Any]:
    run_id = f"{arm.arm_id}_{context.context_id}_seed{int(seed)}"
    if repeat_index:
        run_id += f"_repeat{int(repeat_index)}"
    run_dir = raw_root / run_id
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    try:
        sim = _configure_cell(
            run_dir=run_dir,
            context=context,
            arm=arm,
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
        run_failed = 0
        error = ""
    except Exception as exc:
        run_failed = 1
        error = str(exc)
    events = _iter_events(run_dir)
    births = [row for row in events if str(row.get("event_type") or "") == "birth_success"]
    failures = [row for row in events if str(row.get("event_type") or "").startswith("birth_fail")]
    weather_on = [
        row for row in births if _safe_int(row.get("child_mutation_weather_active", row.get("mutation_weather_active", 0)), 0) > 0
    ]
    child_lineage_counts: dict[str, int] = {}
    for row in births:
        lineage = str(row.get("child_lineage_id") or row.get("lineage_id") or "")
        if lineage:
            child_lineage_counts[lineage] = int(child_lineage_counts.get(lineage, 0)) + 1
    summary = _read_json(run_dir / "metrics" / "lineage_reproduction_funnel_run_summary.json")
    final = _read_final_generation(run_dir)
    distances = [_safe_float(row.get("parent_child_genome_distance_mean", 0.0), 0.0) for row in births]
    scalar_distances = [_safe_float(row.get("parent_child_genome_scalar_distance_mean", 0.0), 0.0) for row in births]
    return {
        "arm_id": arm.arm_id,
        "arm_label": arm.label,
        "context_id": context.context_id,
        "context_label": context.label,
        "seed": int(seed),
        "repeat_index": int(repeat_index),
        "run_dir": str(run_dir),
        "run_failure_count": int(run_failed),
        "run_error": error,
        "birth_success_count": int(len(births)),
        "offspring_affected_by_weather_count": int(len(weather_on)),
        "weather_active_generation_count": _safe_int(summary.get("weather_active_generation_count", 0), 0),
        "weather_event_count": _safe_int(summary.get("weather_event_count", 0), 0),
        "parent_child_genome_distance_mean": _mean(distances),
        "parent_child_genome_distance_max": max((_safe_float(row.get("parent_child_genome_distance_max", 0.0), 0.0) for row in births), default=0.0),
        "parent_child_genome_scalar_distance_mean": _mean(scalar_distances),
        "parent_child_genome_distance_mean_weather_off": _safe_float(summary.get("parent_child_genome_distance_mean_weather_off", 0.0), 0.0),
        "parent_child_genome_distance_mean_weather_on": _safe_float(summary.get("parent_child_genome_distance_mean_weather_on", 0.0), 0.0),
        "parent_child_genome_distance_max_weather_off": _safe_float(summary.get("parent_child_genome_distance_max_weather_off", 0.0), 0.0),
        "parent_child_genome_distance_max_weather_on": _safe_float(summary.get("parent_child_genome_distance_max_weather_on", 0.0), 0.0),
        "memory_field_delta_event_count": int(
            sum(1 for row in births if _safe_float(row.get("parent_child_genome_memory_field_delta", 0.0), 0.0) > 0.0)
        ),
        "prediction_field_delta_event_count": int(
            sum(1 for row in births if _safe_float(row.get("parent_child_genome_prediction_field_delta", 0.0), 0.0) > 0.0)
        ),
        "module_gate_delta_event_count": int(
            sum(1 for row in births if _safe_float(row.get("parent_child_genome_module_gate_delta", 0.0), 0.0) > 0.0)
        ),
        "memory_field_delta_events_weather_on": _safe_int(summary.get("memory_field_delta_events_weather_on", 0), 0),
        "prediction_field_delta_events_weather_on": _safe_int(summary.get("prediction_field_delta_events_weather_on", 0), 0),
        "module_gate_delta_events_weather_on": _safe_int(summary.get("module_gate_delta_events_weather_on", 0), 0),
        "effective_alive_lineages_final": _safe_float(final.get("effective_alive_lineages_final", 0.0), 0.0),
        "effective_reproducing_lineages_if_available": _safe_float(
            final.get("effective_reproducing_lineages_if_available", 0.0), 0.0
        ),
        "offspring_effective_lineages": _effective_count(child_lineage_counts),
        "top1_alive_share_final": _safe_float(final.get("top1_alive_share_final", 0.0), 0.0),
        "top1_offspring_share_if_available": _top_share(child_lineage_counts),
        "offspring_survival_proxy": _safe_float(final.get("effective_alive_lineages_final", 0.0), 0.0),
        "birth_failure_count_if_available": int(len(failures)),
        "lineage_funnel_events_sha256": _sha256(run_dir / "metrics" / "lineage_reproduction_funnel_events.jsonl"),
        "lineage_funnel_events_normalized_sha256": _normalized_events_sha256(run_dir),
        "metrics_sha256": _sha256(run_dir / "metrics.csv"),
    }


def _aggregate(rows: Sequence[Mapping[str, Any]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for row in rows:
        if _safe_int(row.get("repeat_index", 0), 0) != 0:
            continue
        grouped.setdefault(tuple(row.get(key, "") for key in keys), []).append(row)
    out = []
    for group_key, group_rows in sorted(grouped.items()):
        record = {key: group_key[idx] for idx, key in enumerate(keys)}
        record.update(
            {
                "cell_count": len(group_rows),
                "run_failure_count": sum(_safe_int(row.get("run_failure_count", 0), 0) for row in group_rows),
                "birth_success_count": sum(_safe_int(row.get("birth_success_count", 0), 0) for row in group_rows),
                "offspring_affected_by_weather_count": sum(
                    _safe_int(row.get("offspring_affected_by_weather_count", 0), 0) for row in group_rows
                ),
                "weather_active_generation_count": sum(
                    _safe_int(row.get("weather_active_generation_count", 0), 0) for row in group_rows
                ),
                "weather_event_count": sum(_safe_int(row.get("weather_event_count", 0), 0) for row in group_rows),
                "parent_child_genome_distance_mean": _mean(
                    [row.get("parent_child_genome_distance_mean", 0.0) for row in group_rows]
                ),
                "parent_child_genome_distance_max": max(
                    (_safe_float(row.get("parent_child_genome_distance_max", 0.0), 0.0) for row in group_rows),
                    default=0.0,
                ),
                "parent_child_genome_scalar_distance_mean": _mean(
                    [row.get("parent_child_genome_scalar_distance_mean", 0.0) for row in group_rows]
                ),
                "parent_child_genome_distance_mean_weather_on": _mean(
                    [row.get("parent_child_genome_distance_mean_weather_on", 0.0) for row in group_rows]
                ),
                "parent_child_genome_distance_mean_weather_off": _mean(
                    [row.get("parent_child_genome_distance_mean_weather_off", 0.0) for row in group_rows]
                ),
                "module_gate_delta_event_count": sum(
                    _safe_int(row.get("module_gate_delta_event_count", 0), 0) for row in group_rows
                ),
                "module_gate_delta_events_weather_on": sum(
                    _safe_int(row.get("module_gate_delta_events_weather_on", 0), 0) for row in group_rows
                ),
                "effective_alive_lineages_final": _mean(
                    [row.get("effective_alive_lineages_final", 0.0) for row in group_rows]
                ),
                "top1_alive_share_final": _mean([row.get("top1_alive_share_final", 0.0) for row in group_rows]),
                "top1_offspring_share_if_available": _mean(
                    [row.get("top1_offspring_share_if_available", 0.0) for row in group_rows]
                ),
                "birth_failure_count_if_available": sum(
                    _safe_int(row.get("birth_failure_count_if_available", 0), 0) for row in group_rows
                ),
            }
        )
        out.append(record)
    return out


def _decision(arm_summary: Sequence[Mapping[str, Any]], parity: Mapping[str, Any]) -> dict[str, Any]:
    by_arm = {str(row.get("arm_id")): row for row in arm_summary}
    baseline = by_arm.get("baseline_current_mutation", {})
    fixed = by_arm.get("fixed_high_scalar_mutation_control", {})
    pulse = by_arm.get("safe_pulse_v0", {})
    failures = sum(_safe_int(row.get("run_failure_count", 0), 0) for row in arm_summary)
    baseline_affected = _safe_int(baseline.get("offspring_affected_by_weather_count", 0), 0)
    pulse_affected = _safe_int(pulse.get("offspring_affected_by_weather_count", 0), 0)
    births = sum(_safe_int(row.get("birth_success_count", 0), 0) for row in arm_summary)
    module_deltas = sum(_safe_int(row.get("module_gate_delta_event_count", 0), 0) for row in arm_summary)
    baseline_scalar = _safe_float(baseline.get("parent_child_genome_scalar_distance_mean", 0.0), 0.0)
    pulse_on = _safe_float(pulse.get("parent_child_genome_distance_mean_weather_on", 0.0), 0.0)
    fixed_scalar = _safe_float(fixed.get("parent_child_genome_scalar_distance_mean", 0.0), 0.0)
    deterministic = bool(parity.get("passed", False))
    if failures or births <= 0 or module_deltas > 0:
        decision = "MUTRAD-1-C"
        recommendation = "Disable and redesign before scaling."
    elif baseline_affected != 0 or not deterministic:
        decision = "MUTRAD-1-C"
        recommendation = "Repair baseline safety or replay determinism before scaling."
    elif pulse_affected > 0 and pulse_on > baseline_scalar and fixed_scalar >= baseline_scalar:
        decision = "MUTRAD-1-A"
        recommendation = "Proceed later to a bounded MUTRAD-1R validation."
    elif pulse_affected > 0:
        decision = "MUTRAD-1-B"
        recommendation = "Implementation is safe, but the pilot was weakly informative."
    else:
        decision = "MUTRAD-1-D"
        recommendation = "Stop at code/reporting; pilot conditions did not activate interpretable weather."
    return {
        "schema_version": "mutrad1_decision_v1",
        "decision": decision,
        "next_recommendation": recommendation,
        "baseline_weather_affected_offspring": baseline_affected,
        "pulse_weather_affected_offspring": pulse_affected,
        "total_birth_success_count": births,
        "run_failure_count": failures,
        "module_gate_delta_event_count": module_deltas,
        "baseline_scalar_distance_mean": baseline_scalar,
        "fixed_scalar_distance_mean": fixed_scalar,
        "pulse_weather_on_distance_mean": pulse_on,
        "determinism_parity": dict(parity),
        "no_claims": NO_CLAIMS,
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


def _parse_seeds(text: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in str(text).split(",") if part.strip())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--evidence-dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seeds", type=_parse_seeds, default=(101, 202))
    parser.add_argument("--generations", type=int, default=2)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--population", type=int, default=12)
    parser.add_argument("--grid", type=int, default=8)
    args = parser.parse_args()

    raw_root = Path(args.raw_root)
    evidence_dir = Path(args.evidence_dir)
    raw_root.mkdir(parents=True, exist_ok=True)
    evidence_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for arm in ARMS:
        for context in CONTEXTS:
            for seed in args.seeds:
                rows.append(
                    _run_cell(
                        raw_root=raw_root,
                        context=context,
                        arm=arm,
                        seed=int(seed),
                        repeat_index=0,
                        generations=int(args.generations),
                        steps=int(args.steps),
                        population=int(args.population),
                        grid=int(args.grid),
                        device=str(args.device),
                    )
                )
    repeat = _run_cell(
        raw_root=raw_root,
        context=CONTEXTS[0],
        arm=ARMS[2],
        seed=int(args.seeds[0]),
        repeat_index=1,
        generations=int(args.generations),
        steps=int(args.steps),
        population=int(args.population),
        grid=int(args.grid),
        device=str(args.device),
    )
    rows.append(repeat)

    base_repeat = [
        row
        for row in rows
        if row["arm_id"] == "safe_pulse_v0"
        and row["context_id"] == "flat_control"
        and int(row["seed"]) == int(args.seeds[0])
        and int(row["repeat_index"]) == 0
    ][0]
    parity = {
        "schema_version": "mutrad1_single_cell_replay_parity_v1",
        "arm_id": "safe_pulse_v0",
        "context_id": "flat_control",
        "seed": int(args.seeds[0]),
        "passed": bool(
            base_repeat.get("lineage_funnel_events_normalized_sha256") == repeat.get("lineage_funnel_events_normalized_sha256")
            and base_repeat.get("metrics_sha256") == repeat.get("metrics_sha256")
        ),
        "events_normalized_sha256_match": base_repeat.get("lineage_funnel_events_normalized_sha256")
        == repeat.get("lineage_funnel_events_normalized_sha256"),
        "metrics_sha256_match": base_repeat.get("metrics_sha256") == repeat.get("metrics_sha256"),
    }
    context_arm_summary = _aggregate(rows, ("arm_id", "context_id"))
    arm_summary = _aggregate(rows, ("arm_id",))
    decision = _decision(arm_summary, parity)
    manifest = {
        "schema_version": "mutrad1_pilot_manifest_v1",
        "raw_root": str(raw_root),
        "evidence_dir": str(evidence_dir),
        "contexts": [context.__dict__ for context in CONTEXTS],
        "arms": [arm.__dict__ for arm in ARMS],
        "seeds": list(args.seeds),
        "generations": int(args.generations),
        "steps_per_generation": int(args.steps),
        "population": int(args.population),
        "grid": int(args.grid),
        "device": str(args.device),
        "mechanics_policy": "default-off scalar sigma multiplier only; no hotspots, structural weather, rewards, rescue, or survival subsidy",
    }
    _write_csv(evidence_dir / "mutrad1_cells.csv", rows)
    _write_json(evidence_dir / "mutrad1_cells.json", rows)
    _write_csv(evidence_dir / "mutrad1_context_arm_summary.csv", context_arm_summary)
    _write_json(evidence_dir / "mutrad1_context_arm_summary.json", context_arm_summary)
    _write_csv(evidence_dir / "mutrad1_arm_summary.csv", arm_summary)
    _write_json(evidence_dir / "mutrad1_arm_summary.json", arm_summary)
    _write_json(evidence_dir / "mutrad1_replay_parity.json", parity)
    _write_json(evidence_dir / "mutrad1_decision_summary.json", decision)
    _write_json(evidence_dir / "mutrad1_manifest.json", manifest)
    return 0 if str(decision["decision"]) in {"MUTRAD-1-A", "MUTRAD-1-B", "MUTRAD-1-D"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
