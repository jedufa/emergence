from __future__ import annotations

import hashlib
import json
from dataclasses import fields
from pathlib import Path

import numpy as np

from env.env_config import EnvConfig
from evolution.population_pkg.population_config import PopConfig
from simulation.checkpoint_bundle_v2 import checkpoint_bundle_v2_apply, checkpoint_bundle_v2_load, checkpoint_bundle_v2_save
from simulation.memory_action_window import (
    EVENTS_FILENAME,
    GENERATION_SUMMARY_FILENAME,
    RUN_SUMMARY_FILENAME,
    SCHEMA_FILENAME,
    MEMORY_ACTION_WINDOW_SCHEMA_VERSION,
    memory_action_window_schema,
)
from simulation.metrics_pkg.noop_loggers import NoOpMetricsLogger
from simulation.runner import SimulationRunner
from simulation.runner_pkg.runner_config import SimConfig


def _env_cfg() -> EnvConfig:
    cfg = EnvConfig(width=1, height=1, view_radius=0)
    cfg.enable_biomes = False
    cfg.food_spawn_per_step = 0
    cfg.max_food = 0
    cfg.patch_world_enable = True
    cfg.patch_world_observation_enable = True
    cfg.wdl_delayed_hazard_enable = True
    cfg.wdl_delayed_hazard_observation_enable = True
    cfg.wdl_delayed_hazard_delay_steps = 4
    cfg.wdl_delayed_hazard_cue_duration_steps = 2
    cfg.wdl_delayed_hazard_active_duration_steps = 3
    cfg.wdl_delayed_hazard_event_period_steps = 12
    cfg.wdl_delayed_hazard_radius = 0
    cfg.wdl_delayed_hazard_intensity = 0.01
    return cfg


def _sim_cfg(
    seed: int,
    *,
    enabled: bool = False,
    run_id: str = "fixture",
    sample_tick_period: int = 1,
    sample_first_n_generations: int = 0,
) -> SimConfig:
    cfg = SimConfig(
        seed=int(seed),
        generations=1,
        steps_per_generation=7,
        device="cpu",
        bank_device="cpu",
        behavior_log_enable=False,
        policy_deterministic_actions=True,
        policy_tick_stride=1,
    )
    cfg.phase_e_e1_enable = False
    cfg.epi_enable = False
    cfg.checkpoint_bundle_v2_period_gen = 0
    cfg.enable_cognitive_trait_registry = False
    cfg.enable_cognitive_status_metrics = False
    cfg.enable_cognitive_budget_ledger = False
    if enabled:
        cfg.memory_action_window_instrumentation_enable = True
        cfg.memory_action_window_emit_jsonl = True
        cfg.memory_action_window_include_lineage = True
        cfg.memory_action_window_include_memory_state = True
        cfg.memory_action_window_run_id = str(run_id)
        cfg.memory_action_window_trait_arm = "memory_normal"
        cfg.memory_action_window_world_context = "fixture_wdl_enabled"
        cfg.memory_action_window_init_arm = "init_legacy"
        cfg.memory_action_window_sample_tick_period = int(sample_tick_period)
        cfg.memory_action_window_sample_generation_period = 1
        cfg.memory_action_window_sample_first_n_generations = int(sample_first_n_generations)
    return cfg


def _run(
    tmp_path: Path,
    *,
    seed: int = 123,
    enabled: bool = False,
    run_id: str = "fixture",
    sample_tick_period: int = 1,
    sample_first_n_generations: int = 0,
) -> SimulationRunner:
    sim = SimulationRunner(
        env_cfg=_env_cfg(),
        pop_cfg=PopConfig(population_size=1),
        sim_cfg=_sim_cfg(
            seed,
            enabled=enabled,
            run_id=run_id,
            sample_tick_period=sample_tick_period,
            sample_first_n_generations=sample_first_n_generations,
        ),
        logger=NoOpMetricsLogger(),
        run_dir=str(tmp_path),
        perf=False,
    )
    sim.run()
    return sim


def _fingerprint(sim: SimulationRunner) -> dict[str, object]:
    alive = sorted(sim.pop.alive_agents(), key=lambda a: int(a.agent_id))
    return {
        "alive_ids": [int(a.agent_id) for a in alive],
        "health": [round(float(a.health), 8) for a in alive],
        "energy": [round(float(a.energy), 8) for a in alive],
        "pos_y": np.asarray(sim.env.agent_pos_y[:4], dtype=np.int32).tolist(),
        "pos_x": np.asarray(sim.env.agent_pos_x[:4], dtype=np.int32).tolist(),
        "wdl_damage": round(float(getattr(sim, "_wdl_delayed_hazard_damage_total_gen", 0.0)), 8),
    }


def _jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_memory_action_window_default_off_no_files_and_equivalent_state(tmp_path: Path) -> None:
    baseline = _run(tmp_path / "baseline", seed=321, enabled=False)
    repeated = _run(tmp_path / "repeated", seed=321, enabled=False)

    assert _fingerprint(baseline) == _fingerprint(repeated)
    assert not (tmp_path / "baseline" / "memory_action_window").exists()
    assert not (tmp_path / "repeated" / "memory_action_window").exists()


def test_memory_action_window_enabled_emits_required_phases_and_schema(tmp_path: Path) -> None:
    _run(tmp_path / "enabled", seed=123, enabled=True)
    out = tmp_path / "enabled" / "memory_action_window"

    assert (out / SCHEMA_FILENAME).exists()
    assert (out / EVENTS_FILENAME).exists()
    assert (out / GENERATION_SUMMARY_FILENAME).exists()
    assert (out / RUN_SUMMARY_FILENAME).exists()

    schema = json.loads((out / SCHEMA_FILENAME).read_text(encoding="utf-8"))
    assert schema["schema_version"] == MEMORY_ACTION_WINDOW_SCHEMA_VERSION
    rows = _jsonl(out / EVENTS_FILENAME)
    phases = {str(row["phase"]) for row in rows}
    assert {"cue", "hidden_gap", "active_hazard"}.issubset(phases)

    required = set(memory_action_window_schema()["required_fields"])
    for row in rows:
        assert required.issubset(row)
        assert row["run_id"] == "fixture"
        assert row["trait_arm"] == "memory_normal"
        assert isinstance(row["action_histogram"], dict)
        assert isinstance(row["lineage_context"], dict)
        assert isinstance(row["memory_summary"], dict)
        assert isinstance(row["exposure_damage_summary"], dict)


def test_memory_action_window_enabled_output_is_deterministic(tmp_path: Path) -> None:
    _run(tmp_path / "a", seed=456, enabled=True, run_id="parity")
    _run(tmp_path / "b", seed=456, enabled=True, run_id="parity")
    a = (tmp_path / "a" / "memory_action_window" / EVENTS_FILENAME).read_bytes()
    b = (tmp_path / "b" / "memory_action_window" / EVENTS_FILENAME).read_bytes()
    assert hashlib.sha256(a).hexdigest() == hashlib.sha256(b).hexdigest()


def test_memory_action_window_sampling_limits_events_and_preserves_state(tmp_path: Path) -> None:
    full = _run(tmp_path / "full", seed=654, enabled=True, run_id="sampled")
    sampled = _run(
        tmp_path / "sampled",
        seed=654,
        enabled=True,
        run_id="sampled",
        sample_tick_period=2,
        sample_first_n_generations=1,
    )

    assert _fingerprint(full) == _fingerprint(sampled)

    full_rows = _jsonl(tmp_path / "full" / "memory_action_window" / EVENTS_FILENAME)
    sampled_rows = _jsonl(tmp_path / "sampled" / "memory_action_window" / EVENTS_FILENAME)
    assert 0 < len(sampled_rows) < len(full_rows)

    run_summary = json.loads(
        (tmp_path / "sampled" / "memory_action_window" / RUN_SUMMARY_FILENAME).read_text(encoding="utf-8")
    )
    assert run_summary["diagnostic_sampling"]["sample_tick_period"] == 2
    assert run_summary["diagnostic_sampling"]["sample_first_n_generations"] == 1


def test_memory_action_window_does_not_add_default_config_fields() -> None:
    env_fields = {field.name for field in fields(EnvConfig)}
    sim_fields = {field.name for field in fields(SimConfig)}
    forbidden = {
        "memory_action_window_instrumentation_enable",
        "memory_action_window_emit_jsonl",
        "memory_action_window_include_lineage",
        "memory_action_window_include_memory_state",
        "memory_action_window_sample_tick_period",
        "memory_action_window_sample_generation_period",
        "memory_action_window_sample_first_n_generations",
    }
    assert forbidden.isdisjoint(env_fields)
    assert forbidden.isdisjoint(sim_fields)


def test_memory_action_window_checkpoint_safe_stateless_derivation(tmp_path: Path) -> None:
    sim = SimulationRunner(
        env_cfg=_env_cfg(),
        pop_cfg=PopConfig(population_size=1),
        sim_cfg=_sim_cfg(789, enabled=True),
        logger=NoOpMetricsLogger(),
        run_dir=str(tmp_path / "source"),
        perf=False,
    )
    sim.pop.init_random()
    sim._ensure_bank()
    sim._place_agents()
    sim.env.patch_world_step(0)

    bundle = tmp_path / "source" / "checkpoint_bundles_v2" / "bundle_gen000000"
    checkpoint_bundle_v2_save(sim, str(bundle), generation_index=0, strict_replay_mode=True)
    payload = checkpoint_bundle_v2_load(str(bundle), strict=True, expected_commit_hash=None)

    restored = SimulationRunner(
        env_cfg=_env_cfg(),
        pop_cfg=PopConfig(population_size=1),
        sim_cfg=_sim_cfg(789, enabled=True),
        logger=NoOpMetricsLogger(),
        run_dir=str(tmp_path / "restored"),
        perf=False,
    )
    checkpoint_bundle_v2_apply(restored, payload, restore_rng=True)

    schema = json.loads((tmp_path / "source" / "memory_action_window" / SCHEMA_FILENAME).read_text(encoding="utf-8"))
    assert schema["checkpoint_mode"] == "stateless_derived_from_checkpointed_runner_env_population_state"
    assert np.array_equal(sim.env.patch_delayed_hazard_timer, restored.env.patch_delayed_hazard_timer)
    assert np.array_equal(sim.env.patch_delayed_hazard_event_id, restored.env.patch_delayed_hazard_event_id)
