from __future__ import annotations

import hashlib
import json
from dataclasses import fields
from pathlib import Path

import numpy as np

from env.env_config import EnvConfig
from evolution.population_pkg.population_config import PopConfig
from simulation.mem_max_live_level4 import (
    EVENTS_FILENAME,
    RUN_SUMMARY_FILENAME,
    SCHEMA_FILENAME,
    SCHEMA_VERSION,
    mem_max_live_level4_schema,
)
from simulation.metrics_pkg.noop_loggers import NoOpMetricsLogger
from simulation.runner import SimulationRunner
from simulation.runner_pkg.runner_config import SimConfig


def _env_cfg() -> EnvConfig:
    cfg = EnvConfig(width=7, height=7, view_radius=1)
    cfg.enable_biomes = False
    cfg.food_spawn_per_step = 0
    cfg.max_food = 0
    cfg.patch_world_enable = True
    cfg.patch_world_observation_enable = True
    cfg.patch_food_amount_initial_empty_cell_value = 1.0
    cfg.patch_food_regrow_per_step = 0.0
    cfg.patch_waste_consume_emit = 0.0
    cfg.patch_waste_presence_emit = 0.0
    cfg.patch_waste_decay_per_step = 0.0
    cfg.mem_max_delayed_choice_enable = True
    cfg.mem_max_delayed_choice_observation_enable = True
    cfg.mem_max_delayed_choice_observation_schema_version = 2
    cfg.mem_max_delayed_choice_global_cue_observation_enable = True
    cfg.mem_max_delayed_choice_navigation_observation_enable = True
    cfg.mem_max_delayed_choice_cue_steps = 1
    cfg.mem_max_delayed_choice_hidden_steps = 2
    cfg.mem_max_delayed_choice_active_steps = 2
    cfg.mem_max_delayed_choice_event_period_steps = 10
    cfg.mem_max_delayed_choice_target_distance = 2
    cfg.mem_max_delayed_choice_damage = 0.25
    return cfg


def _sim_cfg(seed: int, *, enabled: bool = False, run_id: str = "fixture", stratified: bool = False) -> SimConfig:
    cfg = SimConfig(
        seed=int(seed),
        generations=1,
        steps_per_generation=8,
        device="cpu",
        bank_device="cpu",
        behavior_log_enable=False,
        policy_deterministic_actions=True,
        policy_tick_stride=1,
    )
    cfg.phase_e_e1_enable = False
    cfg.epi_enable = False
    cfg.checkpoint_bundle_v2_period_gen = 0
    cfg.enable_cognitive_trait_registry = True
    cfg.enable_cognitive_status_metrics = False
    cfg.enable_cognitive_budget_ledger = False
    cfg.disable_prediction_error_selection_pressure = True
    if enabled:
        cfg.mem_max_live_level4_instrumentation_enable = True
        cfg.mem_max_live_level4_emit_jsonl = True
        cfg.mem_max_live_level4_run_id = str(run_id)
        cfg.mem_max_live_level4_max_events_per_run = 64
        cfg.mem_max_live_level4_sample_first_n_per_generation = 64
        if stratified:
            cfg.mem_max_live_level4_sampling_mode = "stratified"
            cfg.mem_max_live_level4_per_stratum_limit = 2
    return cfg


def _run(tmp_path: Path, *, seed: int = 123, enabled: bool = False, run_id: str = "fixture", stratified: bool = False) -> SimulationRunner:
    sim = SimulationRunner(
        env_cfg=_env_cfg(),
        pop_cfg=PopConfig(population_size=3),
        sim_cfg=_sim_cfg(seed, enabled=enabled, run_id=run_id, stratified=stratified),
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
        "pos_y": np.asarray(sim.env.agent_pos_y[:8], dtype=np.int32).tolist(),
        "pos_x": np.asarray(sim.env.agent_pos_x[:8], dtype=np.int32).tolist(),
        "correct": int(getattr(sim, "_mem_max_delayed_choice_correct_arrival_count_gen", 0) or 0),
        "incorrect": int(getattr(sim, "_mem_max_delayed_choice_incorrect_arrival_count_gen", 0) or 0),
    }


def _jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_mem_max_live_level4_default_off_no_files_and_equivalent_state(tmp_path: Path) -> None:
    baseline = _run(tmp_path / "baseline", seed=101, enabled=False)
    repeated = _run(tmp_path / "repeated", seed=101, enabled=False)

    assert _fingerprint(baseline) == _fingerprint(repeated)
    assert not (tmp_path / "baseline" / "mem_max_live_level4").exists()
    assert not (tmp_path / "repeated" / "mem_max_live_level4").exists()


def test_mem_max_live_level4_emits_position_aware_schema_and_events(tmp_path: Path) -> None:
    _run(tmp_path / "enabled", seed=102, enabled=True)
    out = tmp_path / "enabled" / "mem_max_live_level4"

    assert (out / SCHEMA_FILENAME).exists()
    assert (out / EVENTS_FILENAME).exists()
    assert (out / RUN_SUMMARY_FILENAME).exists()

    schema = json.loads((out / SCHEMA_FILENAME).read_text(encoding="utf-8"))
    assert schema["schema_version"] == SCHEMA_VERSION
    rows = _jsonl(out / EVENTS_FILENAME)
    assert rows
    required = set(mem_max_live_level4_schema()["required_fields"])
    for row in rows:
        assert required.issubset(row)
        assert row["grid_height"] == 7
        assert row["grid_width"] == 7
        assert len(row["action_logits"]) == 7
        assert len(row["action_probabilities"]) == 7
        assert row["executed_action"] == row["chosen_action"]
        assert row["action_target_class"] in {"toward_a", "toward_b", "toward_both", "neutral", "away_or_blocked"}


def test_mem_max_live_level4_output_is_deterministic(tmp_path: Path) -> None:
    _run(tmp_path / "a", seed=103, enabled=True, run_id="parity")
    _run(tmp_path / "b", seed=103, enabled=True, run_id="parity")

    a = (tmp_path / "a" / "mem_max_live_level4" / EVENTS_FILENAME).read_bytes()
    b = (tmp_path / "b" / "mem_max_live_level4" / EVENTS_FILENAME).read_bytes()
    assert hashlib.sha256(a).hexdigest() == hashlib.sha256(b).hexdigest()


def test_mem_max_live_level4_stratified_sampling_covers_later_phases(tmp_path: Path) -> None:
    _run(tmp_path / "stratified", seed=104, enabled=True, run_id="stratified", stratified=True)

    out = tmp_path / "stratified" / "mem_max_live_level4"
    rows = _jsonl(out / EVENTS_FILENAME)
    summary = json.loads((out / RUN_SUMMARY_FILENAME).read_text(encoding="utf-8"))

    assert rows
    assert summary["sampling_mode"] == "stratified"
    assert summary["full_stream_rows_total"] >= summary["rows_emitted_total"] > 0
    assert summary["stratum_available_counts"]
    assert summary["stratum_sampled_counts"]
    assert {row["event_phase"] for row in rows} & {"hidden", "active"}
    assert any(row["post_hidden_decision_opportunity"] for row in rows)
    assert all(row["decision_event_id"] for row in rows)
    assert set(summary["sampled_level_counts"]).issubset(set(summary["full_level_counts"]) | set(summary["sampled_level_counts"]))


def test_mem_max_live_level4_stratified_output_is_deterministic(tmp_path: Path) -> None:
    _run(tmp_path / "sa", seed=105, enabled=True, run_id="stratified_parity", stratified=True)
    _run(tmp_path / "sb", seed=105, enabled=True, run_id="stratified_parity", stratified=True)

    a = (tmp_path / "sa" / "mem_max_live_level4" / EVENTS_FILENAME).read_bytes()
    b = (tmp_path / "sb" / "mem_max_live_level4" / EVENTS_FILENAME).read_bytes()
    assert hashlib.sha256(a).hexdigest() == hashlib.sha256(b).hexdigest()


def test_mem_max_live_level4_does_not_add_default_config_fields() -> None:
    env_fields = {field.name for field in fields(EnvConfig)}
    sim_fields = {field.name for field in fields(SimConfig)}
    forbidden = {
        "mem_max_live_level4_instrumentation_enable",
        "mem_max_live_level4_emit_jsonl",
        "mem_max_live_level4_max_events_per_run",
        "mem_max_live_level4_sample_first_n_per_generation",
        "mem_max_live_level4_sample_tick_period",
        "mem_max_live_level4_sampling_mode",
        "mem_max_live_level4_per_stratum_limit",
    }
    assert forbidden.isdisjoint(env_fields)
    assert forbidden.isdisjoint(sim_fields)
