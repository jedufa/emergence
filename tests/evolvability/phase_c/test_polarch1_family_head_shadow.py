from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from env.gridworld_pkg.constants import ACT_ATTACK, ACT_GATHER, ACT_MOVE_N, FOOD
from evolution.policy_action_family import (
    FACTORIZED_SHADOW_SCHEMA_VERSION,
    factorized_family_shadow_payload,
    family_head_contract,
    within_family_head_contract,
)
from simulation.runner_pkg.runner_action_opportunity import record_policy_action_family_decisions


def _agent(agent_id: int, *, y: int = 1, x: int = 1, memory_dim: int = 0) -> SimpleNamespace:
    return SimpleNamespace(
        agent_id=agent_id,
        energy=1.0,
        health=1.0,
        injury=0.0,
        alive=True,
        slot=agent_id,
        slot_token=100 + agent_id,
        genome=SimpleNamespace(memory_dim=memory_dim),
        stats=SimpleNamespace(survival_steps=0),
        _pos=(y, x),
    )


def _runner(*, shadow: bool = False, record_events: bool = False, attack_target: bool = False) -> tuple[SimpleNamespace, list[SimpleNamespace]]:
    agent0 = _agent(0, memory_dim=8 if attack_target else 0)
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
        run_id="polarch1-test-run",
        sim_cfg=SimpleNamespace(
            structural_policy_action_family_mapping_reporting_enable=False,
            structural_action_consequence_telemetry_reporting_enable=False,
            policy_family_head_shadow_enable=shadow,
            policy_family_head_shadow_record_events=record_events,
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
        _eiel_fdp_cur_gen=3,
        _eiel_fdp_cur_tick=11,
    )
    return runner, alive


def test_polarch1_pure_factorization_exactly_reconstructs_primitives() -> None:
    logits = [3.0, 0.0, -1.0, 0.5, 2.5, -0.25, 1.0]
    payload = factorized_family_shadow_payload(logits, valid_actions=range(7), chosen_action_id=ACT_MOVE_N)

    assert payload["schema_version"] == FACTORIZED_SHADOW_SCHEMA_VERSION
    assert payload["shadow_authority"] is False
    assert payload["random_draws"] == 0
    assert payload["factorization_evaluable"] is True
    assert payload["max_abs_reconstruction_error"] <= 1e-12
    assert sum(payload["primitive_probabilities"]) == pytest.approx(1.0)
    assert sum(payload["family_probabilities"].values()) == pytest.approx(1.0)
    assert sum(payload["reconstructed_primitive_probabilities"]) == pytest.approx(1.0)
    for actual, reconstructed in zip(payload["primitive_probabilities"], payload["reconstructed_primitive_probabilities"]):
        assert reconstructed == pytest.approx(actual)
    assert abs(payload["entropy_decomposition_residual"]) <= 1e-12
    assert payload["selected_movement_direction"] == "MOVE_N"


def test_polarch1_invalid_family_and_singleton_conditionals_are_stable() -> None:
    payload = factorized_family_shadow_payload([0.0, 0.0, 0.0, 0.0, 4.0, 1.0, -3.0], valid_actions=[ACT_GATHER])

    assert payload["family_validity"]["movement"] is False
    assert payload["family_probabilities"]["movement"] == pytest.approx(0.0)
    assert payload["singleton_conditional_probabilities"]["gather"] == pytest.approx(1.0)
    assert payload["within_family_conditional_probabilities"]["gather"]["GATHER"] == pytest.approx(1.0)
    assert payload["reconstructed_primitive_probabilities"][ACT_GATHER] == pytest.approx(1.0)


def test_polarch1_contracts_expose_stable_future_head_ids_without_parameters() -> None:
    family = family_head_contract()
    within = within_family_head_contract()

    assert family["family_heads"]["movement"]["family_head_id"] == "movement"
    assert within["within_family_heads"]["movement"]["within_family_head_id"] == "movement.direction"
    assert within["within_family_heads"]["attack"]["target_mode"] == "environment_resolved"
    assert within["within_family_heads"]["movement"]["parameter_block"] == "absent"


def test_polarch1_shadow_is_default_off() -> None:
    runner, alive = _runner(shadow=False)
    record_policy_action_family_decisions(
        runner,
        alive=alive,
        row_indices=[0],
        logits=np.asarray([[3.0, 0.0, 0.0, 0.0, 2.5, 0.5, -1.0]], dtype=np.float32),
        actions=np.asarray([ACT_MOVE_N], dtype=np.int32),
        policy_kind="mlp",
    )

    assert not hasattr(runner, "_policy_family_head_shadow_total")
    assert not hasattr(runner, "_structural_policy_action_family_total")


def test_polarch1_shadow_enabled_records_no_authority_summary_without_rng_use() -> None:
    runner, alive = _runner(shadow=True)
    rng_state_before = np.random.get_state()
    record_policy_action_family_decisions(
        runner,
        alive=alive,
        row_indices=[0],
        logits=np.asarray([[3.0, 0.0, 0.0, 0.0, 2.5, 0.5, -1.0]], dtype=np.float32),
        actions=np.asarray([ACT_MOVE_N], dtype=np.int32),
        policy_kind="mlp",
    )
    rng_state_after = np.random.get_state()

    assert all(np.array_equal(a, b) if isinstance(a, np.ndarray) else a == b for a, b in zip(rng_state_before, rng_state_after))
    summary = runner._policy_family_head_shadow_total
    assert summary["decision_count"] == 1
    assert summary["evaluable_count"] == 1
    assert summary["random_draws"] == 0
    assert summary["action_authority_changes"] == 0
    assert summary["max_reconstruction_error"] <= 1e-12
    event = runner._structural_policy_action_family_total["decision_event_sample"][0]
    assert event["family_shadow_status"] == "evaluable"
    assert event["family_shadow_authority"] is False
    assert "family_shadow_payload" not in event


def test_polarch1_stride_refill_and_cached_slots_share_originating_refill_shadow() -> None:
    runner, alive = _runner(shadow=True, record_events=True)
    record_policy_action_family_decisions(
        runner,
        alive=alive,
        row_indices=[0],
        logits=np.asarray([[3.0, 0.0, 0.0, 0.0, 2.5, 0.5, -1.0]], dtype=np.float32),
        actions=np.asarray([[ACT_MOVE_N, ACT_GATHER]], dtype=np.int32),
        sample_noise=np.zeros((1, 2, 7), dtype=np.float32),
        policy_kind="mlp",
        action_source_class="neural_policy_stride",
    )

    events = runner._structural_policy_action_family_total["decision_event_sample"]
    assert events[0]["family_shadow_stride_mode"] == "stride_refill"
    assert events[1]["family_shadow_stride_mode"] == "stride_cached"
    assert events[1]["family_shadow_originating_refill_event_id"] == events[0]["decision_event_id"]
    assert events[0]["family_shadow_distribution_id"] == events[1]["family_shadow_distribution_id"]
    summary = runner._policy_family_head_shadow_total
    assert summary["stride_refill_count"] == 1
    assert summary["stride_cached_count"] == 1
    assert summary["cache_application_count"] == 1


def test_polarch1_gru_source_and_memory_dimension_are_covered_by_same_shadow_contract() -> None:
    runner, alive = _runner(shadow=True, attack_target=True)
    record_policy_action_family_decisions(
        runner,
        alive=alive,
        row_indices=[0],
        logits=np.asarray([[0.0, 0.0, 0.0, 0.0, -2.0, -2.0, 4.0]], dtype=np.float32),
        actions=np.asarray([ACT_ATTACK], dtype=np.int32),
        policy_kind="gru",
    )

    event = runner._structural_policy_action_family_total["decision_event_sample"][0]
    assert event["policy_kind"] == "gru"
    assert event["family_shadow_status"] == "evaluable"
    assert event["family_shadow_stride_mode"] == "direct"
    assert runner._policy_family_head_shadow_total["policy_kind_counts"]["gru"] == 1
