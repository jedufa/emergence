from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from env.env_config import EnvConfig
from env.gridworld import GridWorld
from env.gridworld_pkg.constants import ACT_GATHER, ACT_MOVE_E, ACT_MOVE_N, ACT_MOVE_W, ACT_REST
from env.gridworld_pkg.jungle import compute_predator_risk_maps, validate_jungle_config
from env.gridworld_pkg.patch_world import (
    PREDRISK_FIXED1_ZONE_EXPOSED_RICH,
    PREDRISK_FIXED1_ZONE_SAFE_COVERED,
    initialize_patch_world_arrays,
    predrisk_fixed1_map_digest,
    predrisk_fixed1_resource_risk_budget_contract,
    validate_patch_world_arrays,
)
from simulation.predrisk_fixed1_probe import (
    ARM_NAMES,
    choose_actions,
    controller_contract,
    policy_hashes,
    record_post_env,
    write_generation_summary,
)
from simulation.rng_manager import RNGManager
from simulation.runner_pkg.runner_config import SimConfig


def _env_cfg() -> EnvConfig:
    cfg = EnvConfig(width=5, height=5, view_radius=1)
    cfg.enable_biomes = False
    cfg.food_spawn_per_step = 0
    cfg.max_food = 0
    cfg.food_energy = 9.0
    cfg.patch_world_enable = True
    cfg.patch_world_observation_enable = True
    cfg.patch_food_regrow_per_step = 0.0
    cfg.patch_food_gather_amount = 1.0
    cfg.patch_movement_cost_apply_enable = True
    cfg.predrisk_fixed1_enable = True
    cfg.jungle_predator_pressure_enable = True
    cfg.jungle_predator_base_risk = 0.0
    cfg.jungle_predator_scent_weight = 0.05
    cfg.jungle_predator_waste_weight = 0.05
    cfg.jungle_predator_visibility_weight = 0.05
    cfg.jungle_predator_cover_protection_weight = 0.03
    cfg.jungle_predator_shelter_protection_weight = 0.03
    cfg.jungle_predator_risk_cap = 0.10
    cfg.jungle_terrain_protection_enable = True
    return cfg


def _sim_cfg(arm: str = "risk_exploit", *, telemetry: bool = False) -> SimConfig:
    cfg = SimConfig()
    cfg.policy_deterministic_actions = True
    cfg.predrisk_fixed1_probe_enable = True
    cfg.predrisk_fixed1_probe_arm = arm
    cfg.predrisk_fixed1_telemetry_enable = telemetry
    return cfg


def _runner(tmp_path: Path, arm: str = "risk_exploit", *, telemetry: bool = False) -> SimpleNamespace:
    cfg = _env_cfg()
    env = GridWorld(cfg, seed=123, run_dir=None, rng=RNGManager(global_seed=123).env_np)
    env.grid.fill(0)
    for name in (
        "patch_food_amount",
        "patch_fertility",
        "patch_cover",
        "patch_shelter",
        "patch_movement_cost",
        "patch_scent",
        "patch_waste",
        "predrisk_fixed1_zone_class",
    ):
        getattr(env, name).fill(0.0)
    env.patch_movement_cost.fill(1.0)
    env.add_agent(0, (2, 2))
    return SimpleNamespace(sim_cfg=_sim_cfg(arm, telemetry=telemetry), env_cfg=cfg, env=env, run_dir=str(tmp_path))


def _agent(health: float = 100.0) -> SimpleNamespace:
    return SimpleNamespace(agent_id=0, energy=100.0, health=health)


def test_predrisk_fixed1_defaults_are_disabled() -> None:
    env_cfg = EnvConfig()
    sim_cfg = SimConfig()

    assert env_cfg.predrisk_fixed1_enable is False
    assert env_cfg.predrisk_fixed1_control_mode == "correlated_tradeoff"
    assert sim_cfg.predrisk_fixed1_probe_enable is False
    assert sim_cfg.predrisk_fixed1_probe_arm == ""
    assert sim_cfg.predrisk_fixed1_telemetry_enable is False


def test_predrisk_fixed1_world_contract_and_budget_have_real_tradeoff() -> None:
    cfg = _env_cfg()
    arrays = initialize_patch_world_arrays(cfg, np.zeros((8, 8), dtype=np.int8))
    digest = predrisk_fixed1_map_digest(arrays["predrisk_fixed1_zone_class"])
    budget = predrisk_fixed1_resource_risk_budget_contract(cfg, classes=arrays["predrisk_fixed1_zone_class"])
    risk = compute_predator_risk_maps(
        SimpleNamespace(cfg=cfg, grid=np.zeros((8, 8)), agent_grid=np.zeros((8, 8), dtype=np.int32), **arrays)
    )["risk"]

    assert digest["exposed_rich_cell_count"] == digest["safe_covered_cell_count"]
    assert budget["exposed_has_greater_food_opportunity"] is True
    assert budget["safe_has_lower_expected_damage"] is True
    assert budget["no_zone_dominates_both"] is True
    exposed = arrays["predrisk_fixed1_zone_class"] == PREDRISK_FIXED1_ZONE_EXPOSED_RICH
    safe = arrays["predrisk_fixed1_zone_class"] == PREDRISK_FIXED1_ZONE_SAFE_COVERED
    assert float(np.mean(risk[exposed])) > float(np.mean(risk[safe]))


@pytest.mark.parametrize(
    "control",
    [
        "true_flat",
        "resource_only",
        "predator_only",
        "cover_only",
        "label_only_sham",
        "uncorrelated_risk_resource",
        "budget_matched_uniform",
        "correlated_tradeoff",
    ],
)
def test_predrisk_fixed1_all_controls_validate_and_initialize(control: str) -> None:
    cfg = _env_cfg()
    cfg.predrisk_fixed1_control_mode = control
    validate_jungle_config(cfg)
    arrays = initialize_patch_world_arrays(cfg, np.zeros((8, 8), dtype=np.int8))

    assert arrays["predrisk_fixed1_zone_class"].shape == (8, 8)
    assert np.isfinite(arrays["patch_food_amount"]).all()


def test_predrisk_fixed1_fixed_mlp_probe_contract_and_hashes_are_frozen() -> None:
    contract = controller_contract()
    hashes = policy_hashes()

    assert tuple(contract["arms"]) == ARM_NAMES
    assert set(hashes) == set(ARM_NAMES)
    assert len(set(hashes.values())) == len(ARM_NAMES)
    assert "predrisk_fixed1_zone_class" in contract["hidden_state_excluded_from_action"]
    assert "jungle_predator_risk" in contract["hidden_state_excluded_from_action"]


def test_predrisk_fixed1_risk_and_safe_probes_choose_contrasting_actions(tmp_path: Path) -> None:
    risk_runner = _runner(tmp_path, "risk_exploit")
    safe_runner = _runner(tmp_path, "safe_cover")
    for runner in (risk_runner, safe_runner):
        runner.env.patch_food_amount[2, 3] = np.float32(1.0)
        runner.env.patch_cover[2, 3] = np.float32(0.0)
        runner.env.patch_shelter[2, 3] = np.float32(0.0)
        runner.env.patch_food_amount[2, 1] = np.float32(0.35)
        runner.env.patch_cover[2, 1] = np.float32(0.9)
        runner.env.patch_shelter[2, 1] = np.float32(0.9)

    risk = choose_actions(risk_runner, [_agent()], np.asarray([ACT_REST], dtype=np.int32))
    safe = choose_actions(safe_runner, [_agent()], np.asarray([ACT_REST], dtype=np.int32))

    assert risk.tolist() == [ACT_MOVE_E]
    assert safe.tolist() == [ACT_MOVE_W]


def test_predrisk_fixed1_gather_and_damage_response_are_construct_valid(tmp_path: Path) -> None:
    risk_runner = _runner(tmp_path, "risk_exploit")
    safe_runner = _runner(tmp_path, "safe_cover")
    risk_runner.env.patch_food_amount[2, 2] = np.float32(1.0)
    safe_runner.env.patch_food_amount[2, 2] = np.float32(0.25)

    risk = choose_actions(risk_runner, [_agent()], np.asarray([ACT_REST], dtype=np.int32))
    safe = choose_actions(safe_runner, [_agent(health=20.0)], np.asarray([ACT_GATHER], dtype=np.int32))

    assert risk.tolist() == [ACT_GATHER]
    assert safe.tolist() == [ACT_REST]


def test_predrisk_fixed1_action_does_not_depend_on_hidden_zone_label(tmp_path: Path) -> None:
    runner = _runner(tmp_path, "balanced_generalist")
    runner.env.patch_food_amount[1, 2] = np.float32(0.75)
    runner.env.predrisk_fixed1_zone_class[2, 2] = PREDRISK_FIXED1_ZONE_EXPOSED_RICH
    first = choose_actions(runner, [_agent()], np.asarray([ACT_REST], dtype=np.int32)).copy()

    runner.env.predrisk_fixed1_zone_class[2, 2] = PREDRISK_FIXED1_ZONE_SAFE_COVERED
    second = choose_actions(runner, [_agent()], np.asarray([ACT_REST], dtype=np.int32)).copy()

    assert first.tolist() == second.tolist()
    assert first.tolist() == [ACT_MOVE_N]


def test_predrisk_fixed1_telemetry_writes_compact_summary(tmp_path: Path) -> None:
    runner = _runner(tmp_path, "risk_exploit", telemetry=True)
    runner.env.predrisk_fixed1_zone_class[2, 2] = PREDRISK_FIXED1_ZONE_EXPOSED_RICH
    runner.env.patch_food_amount[2, 2] = np.float32(1.0)
    actions = choose_actions(runner, [_agent()], np.asarray([ACT_REST], dtype=np.int32))
    record_post_env(runner, agent_ids=np.asarray([0], dtype=np.int32), actions=actions, food_gained=np.asarray([1.0]))
    runner._eligible_parent_agents_gen = [_agent()]
    runner._selected_parent_agents_gen = [_agent()]
    runner._lineage_reproduction_funnel_birth_contexts_gen = [{"parent_id": 0}]
    runner._jungle_predator_event_contexts_gen = [
        {"y": 2.0, "x": 2.0, "raw_damage": 0.25, "effective_damage": 0.10, "damage_prevented": 0.15}
    ]

    write_generation_summary(runner, generation=0)

    rows = list(csv.DictReader((tmp_path / "metrics" / "predrisk_fixed1_generation_summary.csv").open()))
    contract = json.loads((tmp_path / "metrics" / "predrisk_fixed1_probe_contract.json").read_text(encoding="utf-8"))
    assert rows[0]["exposed_rich_gather_attempts"] == "1.0"
    assert rows[0]["exposed_rich_available_resource_agent_ticks"] == "1.0"
    assert rows[0]["exposed_rich_food_gain"] == "1.0"
    assert rows[0]["exposed_rich_effective_predator_damage_cost"] == "0.1"
    assert rows[0]["exposed_rich_predator_damage_prevented"] == "0.15"
    assert rows[0]["exposed_rich_eligible_parent_events_after_zone_use"] == "1.0"
    assert rows[0]["exposed_rich_selected_parent_events_after_zone_use"] == "1.0"
    assert rows[0]["exposed_rich_successful_births_after_zone_use"] == "1.0"
    assert contract["arm"] == "risk_exploit"
    assert contract["zone_label_use"] == "telemetry only"
    assert "successful gather in the zone" in contract["recent_zone_use_definition"]
    assert contract["net_return_fields"]["effective_predator_damage_cost"] == "<zone>_effective_predator_damage_cost"


def test_predrisk_fixed1_checkpoint_array_validation_preserves_zone_class() -> None:
    cfg = _env_cfg()
    arrays = initialize_patch_world_arrays(cfg, np.zeros((6, 6), dtype=np.int8))
    restored = validate_patch_world_arrays(arrays, shape=(6, 6), cfg=cfg)

    assert "predrisk_fixed1_zone_class" in restored
    assert np.array_equal(restored["predrisk_fixed1_zone_class"], arrays["predrisk_fixed1_zone_class"])
