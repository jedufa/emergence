from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np

from env.env_config import EnvConfig
from env.gridworld_pkg.patch_world import apply_respec0b_resource_gather, initialize_respec0b_resource_class_array
from evolution.population_pkg.population_config import PopConfig
from simulation.metrics_pkg.noop_loggers import NoOpMetricsLogger
from simulation.respec_n1_zone_telemetry import ZONE_A, ZONE_B, finish_generation, layout_digest
from simulation.runner import SimulationRunner
from simulation.runner_pkg.runner_config import SimConfig


def _fake_env(*, telemetry: bool) -> SimpleNamespace:
    cfg = EnvConfig(width=4, height=4)
    cfg.patch_world_enable = True
    cfg.respec0b_resource_specialization_enable = True
    cfg.respec0b_resource_class_mode = "vertical_split_ab"
    cfg.respec_n1_zone_telemetry_enable = bool(telemetry)
    env = SimpleNamespace(cfg=cfg)
    env.grid = np.ones((4, 4), dtype=np.int8)
    env.patch_food_amount = np.ones((4, 4), dtype=np.float32)
    env.respec0b_resource_class = initialize_respec0b_resource_class_array(cfg, shape=(4, 4))
    env.respec0b_resource_active_y = np.full((8,), -1, dtype=np.int32)
    env.respec0b_resource_active_x = np.full((8,), -1, dtype=np.int32)
    env.respec0b_resource_active_class = np.zeros((8,), dtype=np.int32)
    env.respec0b_resource_handling_progress = np.zeros((8,), dtype=np.float32)
    return env


def test_respec_n1_layout_digest_uses_authoritative_resource_class_array() -> None:
    env = _fake_env(telemetry=True)
    digest = layout_digest(env)
    zones = {row["zone"]: row for row in digest["zones"]}

    assert zones[ZONE_A]["cell_count"] == 8
    assert zones[ZONE_B]["cell_count"] == 8
    assert digest["layout"]["classes_sha256"]


def test_respec_n1_gather_events_are_default_off_and_passive() -> None:
    ids = np.asarray([3], dtype=np.int32)
    actions = np.asarray([4], dtype=np.int32)
    ys = np.asarray([1], dtype=np.int32)
    xs = np.asarray([0], dtype=np.int32)
    food = np.zeros((1,), dtype=np.float32)
    axes = np.asarray([0.75], dtype=np.float32)

    off_env = _fake_env(telemetry=False)
    off_result = apply_respec0b_resource_gather(off_env, ids, actions, ys, xs, food.copy(), axes)
    assert not hasattr(off_env, "respec_n1_resource_gather_events_last")

    on_env = _fake_env(telemetry=True)
    on_food = food.copy()
    on_result = apply_respec0b_resource_gather(on_env, ids, actions, ys, xs, on_food, axes)
    events = getattr(on_env, "respec_n1_resource_gather_events_last", [])

    assert on_result == off_result
    assert events
    assert events[0]["agent_id"] == 3
    assert events[0]["zone"] == ZONE_A
    assert events[0]["success"] == 1


def test_respec_n1_tiny_simulation_writes_compact_zone_summary(tmp_path) -> None:
    env_cfg = EnvConfig(width=8, height=8)
    env_cfg.patch_world_enable = True
    env_cfg.patch_world_observation_enable = True
    env_cfg.respec0b_resource_specialization_enable = True
    env_cfg.respec0b_resource_class_mode = "vertical_split_ab"
    env_cfg.respec_n1_zone_telemetry_enable = True
    pop_cfg = PopConfig(population_size=6)
    sim_cfg = SimConfig(seed=123, generations=1, steps_per_generation=2, device="cpu", bank_device="cpu", behavior_log_enable=False)
    sim_cfg.policy_deterministic_actions = True
    sim_cfg.resource_handling_representation_enable = True
    sim_cfg.resource_handling_representation_reporting_enable = True
    sim_cfg.checkpoint_bundle_v2_period_gen = 0

    runner = SimulationRunner(env_cfg, pop_cfg, sim_cfg, NoOpMetricsLogger(), str(tmp_path), perf=False)
    runner.run()

    summary_path = tmp_path / "metrics" / "respec_n1_zone_run_summary.json"
    rows_path = tmp_path / "metrics" / "respec_n1_zone_generation_summary.csv"
    payload = json.loads(summary_path.read_text(encoding="utf-8"))

    assert rows_path.exists()
    assert payload["schema_version"] == "respec_n1_zone_telemetry_v1"
    assert payload["layout"]["zones"]
    assert payload["arm_zone"]


def test_respec_n1_birth_context_uses_parent_zone_and_selected_parent_arm(tmp_path) -> None:
    env = _fake_env(telemetry=True)
    env.agent_pos_y = np.zeros((16,), dtype=np.int32)
    env.agent_pos_x = np.zeros((16,), dtype=np.int32)
    runner = SimpleNamespace(env=env, env_cfg=env.cfg, run_dir=str(tmp_path))
    parent = SimpleNamespace(
        agent_id=7,
        genome=SimpleNamespace(resource_handling_axis=0.75),
        stats=SimpleNamespace(lineage_id="lineage:7"),
    )

    finish_generation(
        runner,
        generation=0,
        alive_after=[],
        selected_parent_agents=[parent],
        birth_contexts=[
            {
                "parent_id": 7,
                "parent_y": 1,
                "parent_x": 0,
                "parent_lineage_id": "lineage:7",
            }
        ],
    )

    payload = json.loads((tmp_path / "metrics" / "respec_n1_zone_run_summary.json").read_text(encoding="utf-8"))
    keyed = {(row["arm"], row["zone"]): row for row in payload["arm_zone"]}

    assert keyed[("a_favored", ZONE_A)]["births"] == 1
    assert keyed[("a_favored", ZONE_A)]["selected_parent_events"] == 1
