from __future__ import annotations

from pathlib import Path

import numpy as np

from env.env_config import EnvConfig
from evolution.population_pkg.population_config import PopConfig
from simulation.mem_max_pre_cue_checkpoint import execute_paired_branch_replay, roundtrip_snapshot
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


def _sim_cfg(tmp_path: Path) -> SimConfig:
    cfg = SimConfig(
        seed=23023,
        generations=1,
        steps_per_generation=6,
        device="cpu",
        bank_device="cpu",
        behavior_log_enable=False,
        policy_deterministic_actions=True,
        policy_tick_stride=1,
    )
    cfg.phase_e_e1_enable = False
    cfg.epi_enable = False
    cfg.enable_cognitive_trait_registry = True
    cfg.enable_cognitive_status_metrics = False
    cfg.enable_cognitive_budget_ledger = False
    cfg.disable_prediction_error_selection_pressure = True
    cfg.force_memory_disabled = False
    cfg.reset_memory_each_tick = False
    cfg.force_prediction_zero_lr = False
    cfg.reset_prediction_each_tick = False
    cfg.prediction_pressure_scenario = False
    cfg.memory_pressure_scenario = False
    cfg.mem_max_pre_cue_checkpoint_enable = True
    cfg.mem_max_pre_cue_checkpoint_max_snapshots = 1
    cfg.mem_max_pre_cue_checkpoint_dir = str(tmp_path / "snapshots")
    cfg.mem_max_live_level4_instrumentation_enable = True
    return cfg


def _make_sim(tmp_path: Path) -> SimulationRunner:
    sim = SimulationRunner(
        env_cfg=_env_cfg(),
        pop_cfg=PopConfig(population_size=4),
        sim_cfg=_sim_cfg(tmp_path),
        logger=NoOpMetricsLogger(),
        run_dir=str(tmp_path / "run"),
        perf=False,
    )
    sim.pop.init_random()
    sim.env.reset()
    for idx, agent in enumerate(sim.pop.alive_agents()):
        sim.env.add_agent(
            int(agent.agent_id),
            (3, 1 + int(idx)),
            slot=int(agent.slot),
            slot_token=int(agent.slot_token),
        )
    return sim


def test_mem_max_23_pre_cue_snapshot_roundtrip_and_noninterference(tmp_path: Path) -> None:
    sim = _make_sim(tmp_path)
    before_timer = np.asarray(sim.env.mem_max_delayed_choice_timer).copy()

    sim.env.patch_world_step(0)

    recorder = sim.mem_max_pre_cue_checkpoint_recorder
    assert recorder is not None
    assert len(recorder.snapshots) == 1
    snap = recorder.snapshots[0]
    assert snap["live_noninterference"] is True
    assert snap["clone_identity_match"] is True
    assert snap["no_aliasing"]["passed"] is True
    assert snap["roundtrip"]["matches_metadata"] is True
    assert np.array_equal(before_timer, np.full_like(before_timer, -1))
    assert int(np.sum(sim.env.mem_max_delayed_choice_cue_a + sim.env.mem_max_delayed_choice_cue_b)) == 1
    assert roundtrip_snapshot(snap["snapshot_path"])["matches_metadata"] is True


def test_mem_max_23_paired_replay_uses_same_snapshot_and_forced_treatments(tmp_path: Path) -> None:
    sim = _make_sim(tmp_path)
    sim.env.patch_world_step(0)
    snap = sim.mem_max_pre_cue_checkpoint_recorder.snapshots[0]

    paired = execute_paired_branch_replay(snap["snapshot_path"])

    assert paired["shared_state_sha256"] == snap["state_sha256"]
    assert len(paired["branches"]) == 2
    assert {row["forced_correct_target"] for row in paired["branches"]} == {1, 2}
    assert paired["hidden_window_executed_ticks"] == 4
    for row in paired["branches"]:
        assert row["decision_abs_step"] == 3
        assert row["focal_agent_id"] >= 0
        assert len(row["probs"]) == 7
        assert sum(row["probs"]) > 0.99
