from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from env.gridworld_pkg.constants import ACT_ATTACK, ACT_MOVE_N, FOOD
from simulation.runner_pkg.runner_action_opportunity import (
    record_policy_action_execution_consequences,
    record_policy_action_family_decisions,
)


def _agent(agent_id: int, *, y: int = 1, x: int = 1) -> SimpleNamespace:
    return SimpleNamespace(
        agent_id=agent_id,
        energy=1.0,
        health=1.0,
        injury=0.0,
        alive=True,
        stats=SimpleNamespace(survival_steps=0),
        _pos=(y, x),
    )


def _runner(*, enabled: bool = True, attack_target: bool = False) -> tuple[SimpleNamespace, list[SimpleNamespace]]:
    agent0 = _agent(0)
    alive = [agent0]
    agent_grid = np.full((3, 3), -1, dtype=np.int32)
    agent_grid[1, 1] = 0
    pos_y = np.asarray([1, 0], dtype=np.int32)
    pos_x = np.asarray([1, 1], dtype=np.int32)
    if attack_target:
        target = _agent(1, y=0, x=1)
        alive.append(target)
        agent_grid[0, 1] = 1
    grid = np.zeros((3, 3), dtype=np.int8)
    grid[1, 1] = FOOD
    runner = SimpleNamespace(
        run_id="rep2s-test-run",
        sim_cfg=SimpleNamespace(
            structural_policy_action_family_mapping_reporting_enable=enabled,
            structural_decision_margin_event_sample_limit=8,
            structural_variant_identity_reporting_enable=False,
        ),
        env_cfg=SimpleNamespace(attack_damage=0.2),
        env=SimpleNamespace(
            agent_grid=agent_grid,
            grid=grid,
            agent_pos_y=pos_y,
            agent_pos_x=pos_x,
        ),
        _eiel_fdp_cur_gen=2,
        _eiel_fdp_cur_tick=7,
    )
    return runner, alive


def test_rep2s_reporting_is_default_off() -> None:
    runner, alive = _runner(enabled=False)
    record_policy_action_family_decisions(
        runner,
        alive=alive,
        row_indices=[0],
        logits=np.zeros((1, 7), dtype=np.float32),
        actions=np.asarray([ACT_MOVE_N], dtype=np.int32),
        sample_noise=np.zeros((1, 7), dtype=np.float32),
        policy_kind="mlp",
    )
    assert not hasattr(runner, "_structural_policy_action_family_total")


def test_rep2s_decision_event_is_joinable_and_deterministic_without_rng_use() -> None:
    runner, alive = _runner(enabled=True)
    rng_state_before = np.random.get_state()
    record_policy_action_family_decisions(
        runner,
        alive=alive,
        row_indices=[0],
        logits=np.asarray([[3.0, 0.0, 0.0, 0.0, 2.5, 0.5, -1.0]], dtype=np.float32),
        actions=np.asarray([ACT_MOVE_N], dtype=np.int32),
        sample_noise=np.asarray([[0.1, 0.2, 0.3, 0.4, 0.0, 0.0, 0.0]], dtype=np.float32),
        policy_kind="mlp",
    )
    rng_state_after = np.random.get_state()
    assert all(np.array_equal(a, b) if isinstance(a, np.ndarray) else a == b for a, b in zip(rng_state_before, rng_state_after))
    total = runner._structural_policy_action_family_total
    assert total["decision_count"] == 1
    event = total["decision_event_sample"][0]
    assert event["decision_event_id"]
    assert event["chosen_family"] == "movement"
    assert event["sample_noise_present"] is True
    assert sum(total["family_probability_mass_sum"].values()) == pytest.approx(1.0)


def test_rep2s_execution_join_records_movement_consequence() -> None:
    runner, alive = _runner(enabled=True)
    record_policy_action_family_decisions(
        runner,
        alive=alive,
        row_indices=[0],
        logits=np.asarray([[3.0, 0.0, 0.0, 0.0, 2.5, 0.5, -1.0]], dtype=np.float32),
        actions=np.asarray([ACT_MOVE_N], dtype=np.int32),
        sample_noise=None,
        policy_kind="mlp",
    )
    decision_id = runner._structural_policy_action_family_total["decision_event_sample"][0]["decision_event_id"]
    record_policy_action_execution_consequences(
        runner,
        alive=alive,
        agent_ids_np=np.asarray([0], dtype=np.int32),
        actions_np=np.asarray([ACT_MOVE_N], dtype=np.int32),
        pre_energy=np.asarray([1.0], dtype=np.float32),
        pre_health=np.asarray([1.0], dtype=np.float32),
        pre_injury=np.asarray([0.0], dtype=np.float32),
        pre_pos_y=np.asarray([1], dtype=np.int32),
        pre_pos_x=np.asarray([1], dtype=np.int32),
        post_pos_y=np.asarray([0], dtype=np.int32),
        post_pos_x=np.asarray([1], dtype=np.int32),
        new_energy=np.asarray([0.99], dtype=np.float32),
        new_health=np.asarray([1.0], dtype=np.float32),
        food_gained=np.asarray([0.0], dtype=np.float32),
        new_injury=np.asarray([0.0], dtype=np.float32),
    )
    total = runner._structural_policy_action_family_total
    assert total["execution_join_count"] == 1
    event = total["execution_event_sample"][0]
    assert event["decision_event_id"] == decision_id
    assert event["executed_family"] == "movement"
    assert event["consequence_status"] == "observed"
    assert event["consequence_success"] is True


def test_rep2s_attack_consequence_fails_closed_without_per_attacker_effect() -> None:
    runner, alive = _runner(enabled=True, attack_target=True)
    record_policy_action_family_decisions(
        runner,
        alive=alive,
        row_indices=[0],
        logits=np.asarray([[0.0, 0.0, 0.0, 0.0, -2.0, -2.0, 4.0]], dtype=np.float32),
        actions=np.asarray([ACT_ATTACK], dtype=np.int32),
        sample_noise=None,
        policy_kind="mlp",
    )
    record_policy_action_execution_consequences(
        runner,
        alive=alive,
        agent_ids_np=np.asarray([0], dtype=np.int32),
        actions_np=np.asarray([ACT_ATTACK], dtype=np.int32),
        pre_energy=np.asarray([1.0], dtype=np.float32),
        pre_health=np.asarray([1.0], dtype=np.float32),
        pre_injury=np.asarray([0.0], dtype=np.float32),
        pre_pos_y=np.asarray([1], dtype=np.int32),
        pre_pos_x=np.asarray([1], dtype=np.int32),
        post_pos_y=np.asarray([1], dtype=np.int32),
        post_pos_x=np.asarray([1], dtype=np.int32),
        new_energy=np.asarray([0.95], dtype=np.float32),
        new_health=np.asarray([1.0], dtype=np.float32),
        food_gained=np.asarray([0.0], dtype=np.float32),
        new_injury=np.asarray([0.0], dtype=np.float32),
    )
    total = runner._structural_policy_action_family_total
    assert total["execution_join_count"] == 1
    event = total["execution_event_sample"][0]
    assert event["executed_family"] == "attack"
    assert event["execution_accepted"] is True
    assert event["consequence_status"] == "not_evaluable"
    assert total["consequence_not_evaluable_family_counts"]["attack"] == 1


def test_rep2t_stride_slots_each_receive_joinable_decision() -> None:
    runner, alive = _runner(enabled=True)
    runner.sim_cfg.structural_action_consequence_telemetry_reporting_enable = True
    record_policy_action_family_decisions(
        runner,
        alive=alive,
        row_indices=[0],
        logits=np.asarray([[3.0, 0.0, 0.0, 0.0, 2.5, 0.5, -1.0]], dtype=np.float32),
        actions=np.asarray([[ACT_MOVE_N, 4]], dtype=np.int32),
        sample_noise=np.zeros((1, 2, 7), dtype=np.float32),
        policy_kind="mlp",
        action_source_class="neural_policy_stride",
    )
    assert runner._structural_policy_action_family_total["decision_count"] == 2
    assert len(runner._structural_policy_action_family_pending[0]) == 2
    record_policy_action_execution_consequences(
        runner,
        alive=alive,
        agent_ids_np=np.asarray([0], dtype=np.int32),
        actions_np=np.asarray([ACT_MOVE_N], dtype=np.int32),
        pre_energy=np.asarray([1.0], dtype=np.float32),
        pre_health=np.asarray([1.0], dtype=np.float32),
        pre_injury=np.asarray([0.0], dtype=np.float32),
        pre_pos_y=np.asarray([1], dtype=np.int32),
        pre_pos_x=np.asarray([1], dtype=np.int32),
        post_pos_y=np.asarray([0], dtype=np.int32),
        post_pos_x=np.asarray([1], dtype=np.int32),
        new_energy=np.asarray([0.99], dtype=np.float32),
        new_health=np.asarray([1.0], dtype=np.float32),
        food_gained=np.asarray([0.0], dtype=np.float32),
        new_injury=np.asarray([0.0], dtype=np.float32),
    )
    record_policy_action_execution_consequences(
        runner,
        alive=alive,
        agent_ids_np=np.asarray([0], dtype=np.int32),
        actions_np=np.asarray([4], dtype=np.int32),
        pre_energy=np.asarray([1.0], dtype=np.float32),
        pre_health=np.asarray([1.0], dtype=np.float32),
        pre_injury=np.asarray([0.0], dtype=np.float32),
        pre_pos_y=np.asarray([0], dtype=np.int32),
        pre_pos_x=np.asarray([1], dtype=np.int32),
        post_pos_y=np.asarray([0], dtype=np.int32),
        post_pos_x=np.asarray([1], dtype=np.int32),
        new_energy=np.asarray([1.5], dtype=np.float32),
        new_health=np.asarray([1.0], dtype=np.float32),
        food_gained=np.asarray([0.5], dtype=np.float32),
        new_injury=np.asarray([0.0], dtype=np.float32),
    )
    total = runner._structural_policy_action_family_total
    assert total["execution_join_count"] == 2
    assert total["execution_missing_decision_count"] == 0
    assert total["source_class_counts"]["neural_policy_stride_refill_mlp"] >= 1
    assert total["source_class_counts"]["neural_policy_stride_cached_mlp"] >= 1


def test_rep2u_stale_prefetch_is_not_true_action_substitution() -> None:
    runner, alive = _runner(enabled=True)
    runner.sim_cfg.structural_action_consequence_telemetry_reporting_enable = True
    record_policy_action_family_decisions(
        runner,
        alive=alive,
        row_indices=[0],
        logits=np.asarray([[3.0, 0.0, 0.0, 0.0, 2.5, 0.5, -1.0]], dtype=np.float32),
        actions=np.asarray([[ACT_MOVE_N, 4]], dtype=np.int32),
        sample_noise=np.zeros((1, 2, 7), dtype=np.float32),
        policy_kind="mlp",
        action_source_class="neural_policy_stride",
    )
    record_policy_action_execution_consequences(
        runner,
        alive=alive,
        agent_ids_np=np.asarray([0], dtype=np.int32),
        actions_np=np.asarray([ACT_MOVE_N], dtype=np.int32),
        pre_energy=np.asarray([1.0], dtype=np.float32),
        pre_health=np.asarray([1.0], dtype=np.float32),
        pre_injury=np.asarray([0.0], dtype=np.float32),
        pre_pos_y=np.asarray([1], dtype=np.int32),
        pre_pos_x=np.asarray([1], dtype=np.int32),
        post_pos_y=np.asarray([0], dtype=np.int32),
        post_pos_x=np.asarray([1], dtype=np.int32),
        new_energy=np.asarray([0.99], dtype=np.float32),
        new_health=np.asarray([1.0], dtype=np.float32),
        food_gained=np.asarray([0.0], dtype=np.float32),
        new_injury=np.asarray([0.0], dtype=np.float32),
    )
    assert len(runner._structural_policy_action_family_pending[0]) == 1
    record_policy_action_family_decisions(
        runner,
        alive=alive,
        row_indices=[0],
        logits=np.asarray([[0.0, 0.0, 0.0, 0.0, 5.0, 0.5, -1.0]], dtype=np.float32),
        actions=np.asarray([[4, 4]], dtype=np.int32),
        sample_noise=np.zeros((1, 2, 7), dtype=np.float32),
        policy_kind="mlp",
        action_source_class="neural_policy_stride",
    )
    record_policy_action_execution_consequences(
        runner,
        alive=alive,
        agent_ids_np=np.asarray([0], dtype=np.int32),
        actions_np=np.asarray([4], dtype=np.int32),
        pre_energy=np.asarray([1.0], dtype=np.float32),
        pre_health=np.asarray([1.0], dtype=np.float32),
        pre_injury=np.asarray([0.0], dtype=np.float32),
        pre_pos_y=np.asarray([0], dtype=np.int32),
        pre_pos_x=np.asarray([1], dtype=np.int32),
        post_pos_y=np.asarray([0], dtype=np.int32),
        post_pos_x=np.asarray([1], dtype=np.int32),
        new_energy=np.asarray([1.5], dtype=np.float32),
        new_health=np.asarray([1.0], dtype=np.float32),
        food_gained=np.asarray([0.5], dtype=np.float32),
        new_injury=np.asarray([0.0], dtype=np.float32),
    )
    total = runner._structural_policy_action_family_total
    assert total["prefetched_not_applied_count"] == 1
    assert total["stale_prefetch_decision_count"] == 1
    assert total["true_action_substitution_count"] == 0
    assert total["action_substitution_count"] == 0
    assert total["chosen_applied_same_count"] == 2


def test_rep2t_attack_consequence_uses_attacker_tape_when_available() -> None:
    runner, alive = _runner(enabled=True, attack_target=True)
    runner.sim_cfg.structural_action_consequence_telemetry_reporting_enable = True
    runner.env.last_attack_consequence_events = [
        {
            "schema_version": "attacker_consequence_event_v1",
            "attacker_id": 0,
            "target_id": 1,
            "attack_accepted": True,
            "attack_rejected": False,
            "rejection_reason": "",
            "attempted_damage": 0.75,
            "applied_damage": 0.75,
            "target_health_before": 1.0,
            "target_health_after": 0.25,
            "target_killed_by_event": False,
            "consequence_evaluable": True,
            "not_evaluable_reason": "",
        }
    ]
    record_policy_action_family_decisions(
        runner,
        alive=alive,
        row_indices=[0],
        logits=np.asarray([[0.0, 0.0, 0.0, 0.0, -2.0, -2.0, 4.0]], dtype=np.float32),
        actions=np.asarray([ACT_ATTACK], dtype=np.int32),
        sample_noise=None,
        policy_kind="mlp",
    )
    record_policy_action_execution_consequences(
        runner,
        alive=alive,
        agent_ids_np=np.asarray([0], dtype=np.int32),
        actions_np=np.asarray([ACT_ATTACK], dtype=np.int32),
        pre_energy=np.asarray([1.0], dtype=np.float32),
        pre_health=np.asarray([1.0], dtype=np.float32),
        pre_injury=np.asarray([0.0], dtype=np.float32),
        pre_pos_y=np.asarray([1], dtype=np.int32),
        pre_pos_x=np.asarray([1], dtype=np.int32),
        post_pos_y=np.asarray([1], dtype=np.int32),
        post_pos_x=np.asarray([1], dtype=np.int32),
        new_energy=np.asarray([0.95], dtype=np.float32),
        new_health=np.asarray([1.0], dtype=np.float32),
        food_gained=np.asarray([0.0], dtype=np.float32),
        new_injury=np.asarray([0.0], dtype=np.float32),
    )
    event = runner._structural_policy_action_family_total["execution_event_sample"][0]
    assert event["consequence_status"] == "observed"
    assert event["consequence_success"] is True
    assert runner._structural_attack_consequence_events[0]["target_id"] == 1
