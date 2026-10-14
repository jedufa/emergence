"""Executable source audit for the Tensor World C3 preparation boundary."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch

from engine.tensor_world.c3_contract import (
    C3_ACTION_COUNT_V1,
    C3_COMPOSED_OBSERVATION_WIDTH_V1,
    C3_DRIVE_WIDTH_V1,
    C3_OBSERVATION_CHANNELS_V1,
    C3_OBSERVATION_COLUMNS_V1,
    C3_OBSERVATION_REGISTRY_SHA256_V1,
    C3_PATCH_OBSERVATION_WIDTH_V1,
    C3_POLICY_HIDDEN_WIDTH_V1,
    C3_POLICY_PACKED_WIDTH_V1,
    C3_SOURCE_AUDIT_SCHEMA_V1,
    C3_SOURCE_FIELD_GROUPS,
    C3_SOURCE_MANIFEST_PATH,
    C3_SOURCE_MANIFEST_SHA256,
    C3_SOURCE_ORDER,
    C3_SOURCE_REFERENCES,
    validate_c3_source_audit,
)
from engine.tensor_world.c3_fixture import make_c3_fixture_v1
from engine.tensor_world.c3_mechanics import (
    TensorWorldC3MechanicsConfigV1,
    apply_c3_chemistry_oracle_v1,
    apply_c3_active_mechanics_oracle_v1,
    capture_c3_pre_action_v1,
)
from engine.tensor_world.c3_backend import (
    TensorWorldC3BackendV1,
    _active_mechanics_cuda_v1,
    _validate_c3_active_mechanics_state_v1,
    _sync_compiled_policy_v1,
    _validate_c3_noise_samples_v1,
)
from engine.tensor_world.c3_native import (
    EVENT_STATE_FIELDS_V1,
    FULL_TICK_OUTPUT_FIELDS_V1,
    FULL_TICK_STATE_FIELDS_V1,
    TOP_INPUT_FIELDS_V1,
    c3_fixed_mlp_policy_cpu_oracle_v1,
    c3_observation_cpu_oracle_v1,
    compare_c3_identity_surfaces_v1,
    compare_c3_observation_channels_v1,
    make_c3_persistent_native_state_v1,
)
from engine.tensor_world.contracts import TensorWorldContractError, TickRequestV1
from engine.tensor_world.c3_state import (
    C3_STATE_SCHEMA_VERSION_V1,
    c3_state_tensor_names_v1,
)
from tools.scripts.run_tensor_world_c3_preparation import (
    _compare_bundles,
    _frozen_budget_context,
    _structural_residency,
)


ROOT = Path(__file__).resolve().parents[1]


def test_c3_source_audit_is_current_and_classified() -> None:
    receipt = validate_c3_source_audit(ROOT)
    assert receipt["schema_version"] == C3_SOURCE_AUDIT_SCHEMA_V1
    assert receipt["source_manifest_sha256"] == C3_SOURCE_MANIFEST_SHA256
    assert receipt["manifest_field_count"] >= 160
    assert receipt["classified_field_count"] >= 170
    assert set(C3_SOURCE_FIELD_GROUPS) == {
        "C3_REQUIRED",
        "C4_TRANSACTIONAL",
        "C5_DETERMINISM_ONLY",
        "C6_EXTERNAL_ONLY",
        "INACTIVE_ON_TARGET_ROUTE",
    }
    assert len(C3_SOURCE_ORDER) >= 10


def test_c3_source_abi_constants_are_frozen() -> None:
    assert C3_PATCH_OBSERVATION_WIDTH_V1 == 74
    assert C3_COMPOSED_OBSERVATION_WIDTH_V1 == 83
    assert C3_POLICY_HIDDEN_WIDTH_V1 == 64
    assert C3_ACTION_COUNT_V1 == 7
    assert C3_POLICY_PACKED_WIDTH_V1 == 9_991
    assert C3_DRIVE_WIDTH_V1 == 4


def test_c3_observation_registry_is_source_bound() -> None:
    assert len(C3_OBSERVATION_CHANNELS_V1) == 9
    assert len(C3_OBSERVATION_COLUMNS_V1) == 83
    assert C3_OBSERVATION_COLUMNS_V1[0] == "legacy_occupancy_or_food[dy=-1,dx=-1]"
    assert C3_OBSERVATION_COLUMNS_V1[-1] == "agent_health_over_start_health"
    assert C3_OBSERVATION_REGISTRY_SHA256_V1 == (
        "e8a5eac546d09f302db52503a14a9ca40597e3e6bc9e488d29d9e626035f176f"
    )


def test_c3_source_references_and_machine_artifact_bind_same_manifest() -> None:
    manifest = ROOT / C3_SOURCE_MANIFEST_PATH
    reference = C3_SOURCE_REFERENCES["state_manifest"]
    assert reference["path"] == str(C3_SOURCE_MANIFEST_PATH)
    assert reference["sha256"] == C3_SOURCE_MANIFEST_SHA256

    artifact = ROOT / "_artifacts/tensor_world_c0_c7_v3/c3/tensor_world_C3_source_audit.json"
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    assert payload["schema_version"] == C3_SOURCE_AUDIT_SCHEMA_V1
    assert payload["status"] == "PREPARATION_ONLY"
    assert payload["gate_authorization"] == "HOLD_BRIDGE_EVIDENCE_ACCESS"
    assert payload["source_manifest"]["path"] == str(C3_SOURCE_MANIFEST_PATH)
    assert payload["source_manifest"]["sha256"] == C3_SOURCE_MANIFEST_SHA256
    assert manifest.is_file()


def test_c3_fixture_owns_explicit_agent_and_transaction_partitions() -> None:
    bundle, _ = make_c3_fixture_v1(batch=1, capacity=8, active=4, width=8)
    bundle.validate()
    assert bundle.schema_version == C3_STATE_SCHEMA_VERSION_V1
    assert bundle.agent.prev_drive_delta.shape == (1, 8, 4)
    assert bundle.agent.age.dtype == torch.int32
    assert bundle.agent.survival_steps.dtype == torch.int32
    assert bundle.agent.energy_collected.dtype == torch.float32
    assert bundle.agent.energy_collected_exact.dtype == torch.float64
    assert bundle.genome.policy_use_gru.shape == (1, 8)
    assert bundle.genome.genome_drive_sp_hunger.shape == (1, 8)
    assert bundle.genome.genome_drive_sp_pain.shape == (1, 8)
    assert bundle.genome.genome_drive_sp_fatigue.shape == (1, 8)
    assert bundle.genome.genome_drive_sp_fear.shape == (1, 8)
    assert bundle.transactions.observations.shape == (1, 8, 83)
    assert bundle.policy_runtime.action_cache_actions.shape == (1, 8, 2)
    assert bundle.lifecycle.death_candidate.shape == (1, 8)
    assert bundle.adaptation.dev_stage_initialized.dtype == torch.bool
    assert bundle.adaptation.dev_stage_initialized.shape == (1, 8)
    assert len(c3_state_tensor_names_v1()) >= 70


def test_c3_observation_cpu_oracle_materializes_source_channel_order() -> None:
    bundle, config = make_c3_fixture_v1(batch=1, capacity=4, active=1, width=4)
    bundle.base.core.agents.pos_y[0, 0] = 0
    bundle.base.core.agents.pos_x[0, 0] = 0
    bundle.base.core.agents.energy[0, 0] = 80.0
    bundle.base.core.agents.health[0, 0] = 6.0
    bundle.base.core.world.food_amount[0, 0, 0] = 0.75
    bundle.base.core.world.cover[0, 0, 0] = 0.8
    bundle.base.core.world.shelter[0, 0, 0] = 0.6
    bundle.base.core.world.scent[0, 0, 0] = 0.25
    bundle.base.core.world.waste[0, 0, 0] = 0.5
    bundle.base.core.world.density[0, 0, 0] = 2.0
    bundle.base.core.world.movement_cost[0, 0, 0] = 2.0
    bundle.base.wdl.cue[0, 0, 0] = 0.9
    observations = c3_observation_cpu_oracle_v1(bundle, config)
    assert observations.shape == (1, 4, 83)
    assert torch.equal(observations[0, 1], torch.zeros(83))
    assert torch.allclose(
        observations[0, 0, [4, 13, 22, 31, 40, 49, 58, 67, 76]],
        torch.tensor([-1.0, 0.75, 0.8, 0.6, 0.25, 0.5, 2.0 / 9.0, 0.5, 0.9]),
        rtol=0.0,
        atol=1.0e-6,
    )
    assert torch.allclose(
        observations[0, 0, 81:],
        torch.tensor([0.5, 0.5]),
        rtol=0.0,
        atol=1.0e-6,
    )


def test_c3_persistent_native_projection_is_source_ordered_and_rng_fail_closed() -> None:
    bundle, _ = make_c3_fixture_v1(batch=1, capacity=8, active=4, width=8)
    state = make_c3_persistent_native_state_v1(bundle, ticks=2)
    assert len(TOP_INPUT_FIELDS_V1) == 58
    assert len(FULL_TICK_STATE_FIELDS_V1) == 29
    assert len(FULL_TICK_OUTPUT_FIELDS_V1) == 18
    assert len(EVENT_STATE_FIELDS_V1) == 30
    assert state.tensors["active_agent_ids"].shape == (1, 4)
    assert state.tensors["mlp_pack"].shape == (1, 8, C3_POLICY_PACKED_WIDTH_V1)
    assert state.tensors["observations"].shape == (1, 2, 8, 83)
    assert state.full_tick_state[20].shape == (1, 2, 8, 4)
    with pytest.raises(TensorWorldContractError, match="C5 PCG64"):
        state.validate(require_rng=True)

    bound = make_c3_persistent_native_state_v1(
        bundle,
        ticks=2,
        rng_state={
            "runner_pcg64_state_words": state.full_tick_state[16],
            "runner_pcg64_buffer_words": state.full_tick_state[17],
            "shared_env_policy_pcg64_state_words": state.full_tick_state[18],
            "shared_env_policy_pcg64_buffer_words": state.full_tick_state[19],
        },
    )
    bound.validate(require_rng=True)

    with pytest.raises(TensorWorldContractError, match="PCG64 state words"):
        make_c3_persistent_native_state_v1(
            bundle,
            ticks=1,
            rng_state={
                "runner_pcg64_state_words": torch.zeros((1, 4), dtype=torch.int64),
                "runner_pcg64_buffer_words": state.full_tick_state[17],
                "shared_env_policy_pcg64_state_words": state.full_tick_state[18],
                "shared_env_policy_pcg64_buffer_words": state.full_tick_state[19],
            },
        )


def test_c3_fixed_mlp_cpu_policy_oracle_matches_source_pack_contract() -> None:
    bundle, _ = make_c3_fixture_v1(batch=1, capacity=8, active=4, width=8)
    observations = torch.arange(8 * 83, dtype=torch.float32).reshape(1, 8, 83) / 100.0
    logits, actions = c3_fixed_mlp_policy_cpu_oracle_v1(
        observations,
        bundle.base.policy.packed_weights,
        bundle.agent.drives,
        bundle.genome.aggression_bias,
        bundle.genome.action_temperature,
        bundle.genome.drive_logit_gain,
    )
    assert logits.shape == (1, 8, 7)
    assert actions.shape == (1, 8)
    assert logits.dtype == torch.float32
    assert actions.dtype == torch.int32
    assert torch.equal(actions, torch.argmax(logits, dim=-1).to(torch.int32))
    assert torch.isfinite(logits).all()
    pack = bundle.base.policy.packed_weights[0, 0]
    observation = observations[0, 0]
    first = torch.tanh(
        pack[: 64 * 83].reshape(64, 83) @ observation
        + pack[64 * 83 : 64 * 83 + 64]
    )
    second_start = 64 * 83 + 64
    second = torch.tanh(
        pack[second_start : second_start + 64 * 64].reshape(64, 64) @ first
        + pack[second_start + 64 * 64 : second_start + 64 * 64 + 64]
    )
    output_start = second_start + 64 * 64 + 64
    explicit = (
        pack[output_start : output_start + 7 * 64].reshape(7, 64) @ second
        + pack[output_start + 7 * 64 : output_start + 7 * 64 + 7]
    )
    explicit[6] += bundle.genome.aggression_bias[0, 0]
    explicit /= bundle.genome.action_temperature[0, 0].clamp_min(0.05)
    gain = bundle.genome.drive_logit_gain[0, 0].clamp_min(0.0)
    explicit[4] += gain * 1.2 * bundle.agent.drives[0, 0, 0]
    explicit[5] += gain * (0.8 * bundle.agent.drives[0, 0, 1] + bundle.agent.drives[0, 0, 2])
    explicit[6] -= gain * (0.8 * bundle.agent.drives[0, 0, 1] + bundle.agent.drives[0, 0, 3])
    assert torch.allclose(logits[0, 0], explicit, rtol=0.0, atol=1e-6)


def test_c3_fixed_mlp_policy_oracle_applies_fixture_gumbel_and_dead_rows() -> None:
    bundle, _ = make_c3_fixture_v1(batch=1, capacity=8, active=4, width=8)
    observations = torch.zeros(1, 8, 83)
    logits, actions = c3_fixed_mlp_policy_cpu_oracle_v1(
        observations,
        bundle.base.policy.packed_weights,
        bundle.agent.drives,
        bundle.genome.aggression_bias,
        bundle.genome.action_temperature,
        bundle.genome.drive_logit_gain,
        alive=bundle.base.core.agents.alive,
        action_uniforms=bundle.base.core.rng.action_uniforms,
    )
    assert torch.equal(logits[0, 4:], torch.zeros(4, 7))
    assert torch.equal(actions[0, 4:], torch.full((4,), -1, dtype=torch.int32))
    uniforms = bundle.base.core.rng.action_uniforms.to(torch.float32).clamp(1.0e-6, 1.0 - 1.0e-6)
    expected = torch.argmax(
        logits[0, :4] - torch.log(-torch.log(uniforms[0, :4])), dim=-1
    ).to(torch.int32)
    assert torch.equal(actions[0, :4], expected)


def test_c3_active_mechanics_emits_death_candidates_without_c4_slot_mutation() -> None:
    bundle, _ = make_c3_fixture_v1(batch=1, capacity=8, active=4, width=8)
    bundle.base.core.agents.energy[0, 0] = 0.0
    bundle.base.core.agents.health[0, 1] = 0.0
    bundle.base.core.transactions.actions[0, :4] = 5
    before_alive = bundle.base.core.agents.alive.clone()
    before_ids = bundle.base.core.agents.agent_id.clone()
    capture_c3_pre_action_v1(bundle)
    apply_c3_active_mechanics_oracle_v1(
        bundle,
        TensorWorldC3MechanicsConfigV1(),
    )
    assert bool(bundle.lifecycle.death_candidate[0, 0])
    assert int(bundle.lifecycle.death_reason[0, 0]) == 1
    assert bool(bundle.lifecycle.death_candidate[0, 1])
    assert int(bundle.lifecycle.death_reason[0, 1]) == 2
    assert torch.equal(bundle.base.core.agents.alive, before_alive)
    assert torch.equal(bundle.base.core.agents.agent_id, before_ids)


def test_c3_active_mechanics_binds_transaction_surfaces_and_action_history() -> None:
    bundle, _ = make_c3_fixture_v1(batch=1, capacity=8, active=4, width=8)
    observations = torch.arange(8 * 83, dtype=torch.float32).reshape(1, 8, 83)
    logits = torch.arange(8 * 7, dtype=torch.float32).reshape(1, 8, 7)
    bundle.base.transactions.observations.copy_(observations)
    bundle.base.transactions.logits.copy_(logits)
    bundle.base.core.transactions.actions[0, :4] = torch.tensor([4, 5, 6, 0])
    capture_c3_pre_action_v1(bundle)
    _active_mechanics_cuda_v1(bundle, TensorWorldC3MechanicsConfigV1())

    assert torch.equal(bundle.transactions.observations, observations)
    assert torch.equal(bundle.transactions.logits, logits)
    assert torch.equal(
        bundle.transactions.row_actions[0, :4],
        torch.tensor([4, 5, 6, 0], dtype=torch.int32),
    )
    assert torch.equal(bundle.agent.last_action[0, :4], torch.tensor([4, 5, 6, 0]))
    assert torch.equal(bundle.agent.attacks[0, :4], torch.tensor([0, 0, 1, 0], dtype=torch.int32))
    assert torch.equal(
        bundle.policy_runtime.action_cache_actions[0, :4],
        torch.tensor([[4, 4], [5, 5], [6, 6], [0, 0]], dtype=torch.int32),
    )
    assert torch.equal(
        bundle.policy_runtime.action_cache_token[0, :4],
        torch.arange(4, dtype=torch.int64),
    )
    assert torch.equal(bundle.policy_runtime.action_cache_cursor[0, :4], torch.ones(4, dtype=torch.int32))
    assert torch.equal(bundle.policy_runtime.action_cache_next[0, :4], torch.tensor([4, 5, 6, 0], dtype=torch.int32))

    bundle.base.core.transactions.actions[0, :4] = torch.tensor([5, 4, 6, 1])
    capture_c3_pre_action_v1(bundle)
    _active_mechanics_cuda_v1(bundle, TensorWorldC3MechanicsConfigV1())
    assert torch.equal(bundle.agent.last_last_action[0, :4], torch.tensor([4, 5, 6, 0]))
    assert torch.equal(bundle.agent.last_action[0, :4], torch.tensor([5, 4, 6, 1]))
    assert torch.equal(bundle.agent.action_switches[0, :4], torch.tensor([1, 1, 0, 1]))
    assert torch.equal(bundle.agent.action_oscillations[0, :4], torch.tensor([0, 0, 0, 0]))


def test_c3_scalar_and_resident_mechanics_match_transaction_surfaces() -> None:
    scalar, _ = make_c3_fixture_v1(batch=1, capacity=8, active=4, width=8)
    resident, _ = make_c3_fixture_v1(batch=1, capacity=8, active=4, width=8)
    actions = torch.tensor([5, 4, 6, 0], dtype=torch.int64)
    for bundle in (scalar, resident):
        bundle.base.core.transactions.actions[0, :4] = actions
        bundle.base.core.agents.survival_steps[0, :4] = torch.tensor(
            [0, 2, 5, 11], dtype=torch.int64
        )
        bundle.base.core.agents.energy[0, :4] = torch.tensor(
            [1.0, 3.0, 40.0, 80.0]
        )
        capture_c3_pre_action_v1(bundle)
    config = TensorWorldC3MechanicsConfigV1(
        enable_rest_charge=True,
        life_history_mode=1,
        life_history_maturity_start_ticks=2,
        life_history_maturity_rise_ticks=3,
        life_history_fertility_decline_start_ticks=5,
        life_history_fertility_decline_ticks=4,
        life_history_elder_fertility_floor=0.2,
        life_history_senescence_start_ticks=3,
        life_history_senescence_ramp_ticks=2,
        life_history_max_age_ticks=10,
    )
    apply_c3_active_mechanics_oracle_v1(scalar, config)
    _active_mechanics_cuda_v1(resident, config)
    assert torch.equal(scalar.transactions.row_actions, resident.transactions.row_actions)
    assert torch.equal(scalar.transactions.hot_counters, resident.transactions.hot_counters)
    assert torch.equal(scalar.lifecycle.death_candidate, resident.lifecycle.death_candidate)
    assert torch.equal(scalar.lifecycle.death_reason, resident.lifecycle.death_reason)
    assert torch.allclose(scalar.agent.drives, resident.agent.drives, rtol=0.0, atol=1e-6)
    assert torch.allclose(scalar.base.core.agents.health, resident.base.core.agents.health, rtol=0.0, atol=1e-6)
    assert torch.allclose(scalar.agent.last_energy_delta, resident.agent.last_energy_delta, rtol=0.0, atol=1e-6)
    assert torch.allclose(scalar.agent.last_health_delta, resident.agent.last_health_delta, rtol=0.0, atol=1e-6)
    assert torch.allclose(scalar.agent.last_injury_delta, resident.agent.last_injury_delta, rtol=0.0, atol=1e-6)
    assert torch.equal(scalar.agent.last_action, resident.agent.last_action)
    assert torch.equal(scalar.agent.last_last_action, resident.agent.last_last_action)
    assert torch.equal(scalar.agent.action_switches, resident.agent.action_switches)
    assert torch.equal(scalar.agent.action_oscillations, resident.agent.action_oscillations)
    assert torch.equal(scalar.agent.attacks, resident.agent.attacks)
    assert torch.equal(scalar.agent.rest_streak, resident.agent.rest_streak)
    assert torch.allclose(scalar.lifecycle.maturity_factor, resident.lifecycle.maturity_factor, rtol=0.0, atol=1e-6)
    assert torch.allclose(scalar.lifecycle.fertility_factor, resident.lifecycle.fertility_factor, rtol=0.0, atol=1e-6)
    assert torch.allclose(scalar.lifecycle.senescence_factor, resident.lifecycle.senescence_factor, rtol=0.0, atol=1e-6)
    assert torch.equal(scalar.agent.survival_steps, resident.agent.survival_steps)
    assert torch.equal(scalar.agent.age, resident.agent.age)
    assert torch.equal(scalar.base.core.agents.age, torch.tensor([[1, 1, 1, 1, 0, 0, 0, 0]], dtype=torch.int64))
    assert torch.equal(scalar.agent.energy_collected, resident.agent.energy_collected)
    assert torch.equal(scalar.agent.energy_collected_exact, resident.agent.energy_collected_exact)
    assert torch.equal(scalar.policy_runtime.action_cache_actions, resident.policy_runtime.action_cache_actions)
    assert torch.equal(scalar.policy_runtime.action_cache_token, resident.policy_runtime.action_cache_token)


@pytest.mark.parametrize("batch", (1, 4, 48))
def test_c3_c4_disabled_multitick_cpu_parity(batch: int) -> None:
    """Exercise the C3 agent surface across worlds without C4 slot mutation."""

    scalar, _ = make_c3_fixture_v1(batch=batch, capacity=8, active=4, width=8)
    resident, _ = make_c3_fixture_v1(batch=batch, capacity=8, active=4, width=8)
    config = TensorWorldC3MechanicsConfigV1(
        enable_rest_charge=True,
        life_history_mode=1,
        life_history_maturity_start_ticks=2,
        life_history_maturity_rise_ticks=3,
        life_history_fertility_decline_start_ticks=5,
        life_history_fertility_decline_ticks=4,
        life_history_elder_fertility_floor=0.2,
        life_history_senescence_start_ticks=3,
        life_history_senescence_ramp_ticks=2,
        life_history_max_age_ticks=10,
    )
    base_actions = torch.arange(batch * 8, dtype=torch.int64).reshape(batch, 8)
    for tick in range(8):
        actions = (base_actions + tick).remainder(7)
        for bundle in (scalar, resident):
            bundle.base.core.transactions.actions.copy_(actions)
            capture_c3_pre_action_v1(bundle)
        apply_c3_active_mechanics_oracle_v1(scalar, config)
        _active_mechanics_cuda_v1(resident, config)
        assert torch.equal(scalar.transactions.row_actions, resident.transactions.row_actions)
        assert torch.equal(scalar.transactions.hot_counters, resident.transactions.hot_counters)
        assert torch.equal(scalar.lifecycle.death_candidate, resident.lifecycle.death_candidate)
        assert torch.equal(scalar.lifecycle.death_reason, resident.lifecycle.death_reason)
        assert torch.equal(scalar.agent.survival_steps, resident.agent.survival_steps)
        assert torch.equal(scalar.agent.age, resident.agent.age)
        assert torch.allclose(scalar.base.core.agents.energy, resident.base.core.agents.energy, rtol=0.0, atol=1e-6)
        assert torch.allclose(scalar.base.core.agents.health, resident.base.core.agents.health, rtol=0.0, atol=1e-6)
        assert torch.allclose(scalar.agent.drives, resident.agent.drives, rtol=0.0, atol=1e-6)
        assert torch.equal(scalar.agent.last_action, resident.agent.last_action)
        assert torch.equal(scalar.agent.last_last_action, resident.agent.last_last_action)
        assert torch.equal(scalar.agent.action_switches, resident.agent.action_switches)
        assert torch.equal(scalar.agent.action_oscillations, resident.agent.action_oscillations)
        assert torch.equal(scalar.agent.attacks, resident.agent.attacks)
        assert torch.equal(scalar.agent.rest_streak, resident.agent.rest_streak)
        assert torch.allclose(scalar.lifecycle.maturity_factor, resident.lifecycle.maturity_factor, rtol=0.0, atol=1e-6)
        assert torch.allclose(scalar.lifecycle.fertility_factor, resident.lifecycle.fertility_factor, rtol=0.0, atol=1e-6)
        assert torch.allclose(scalar.lifecycle.senescence_factor, resident.lifecycle.senescence_factor, rtol=0.0, atol=1e-6)
        assert torch.equal(scalar.policy_runtime.action_cache_actions, resident.policy_runtime.action_cache_actions)
        assert torch.equal(scalar.policy_runtime.action_cache_token, resident.policy_runtime.action_cache_token)


def test_c3_nonzero_drive_noise_scalar_resident_parity_clamps_drives() -> None:
    scalar, _ = make_c3_fixture_v1(batch=1, capacity=8, active=4, width=8)
    resident, _ = make_c3_fixture_v1(batch=1, capacity=8, active=4, width=8)
    actions = torch.tensor([6, 4, 5, 0], dtype=torch.int64)
    samples = torch.tensor(
        [[[-2.0, 1.5, 0.5, -0.25], [0.4, -0.8, 1.2, -1.1], [0.1, 0.2, -0.3, 0.4], [0.8, -0.7, 0.6, -0.5], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]]
    )
    for bundle in (scalar, resident):
        bundle.base.core.transactions.actions[0, :4] = actions
        bundle.agent.drive_noise[0, :4] = 0.75
        bundle.transactions.drive_noise_samples.copy_(samples)
        bundle.transactions.drive_noise_samples_valid[0, :4] = True
        capture_c3_pre_action_v1(bundle)
    config = TensorWorldC3MechanicsConfigV1()
    apply_c3_active_mechanics_oracle_v1(scalar, config)
    _active_mechanics_cuda_v1(resident, config)
    assert torch.all((resident.agent.drives[0, :4] >= 0.0) & (resident.agent.drives[0, :4] <= 1.0))
    assert torch.allclose(scalar.agent.drives, resident.agent.drives, rtol=0.0, atol=1e-6)
    assert torch.allclose(scalar.agent.drive_volatility, resident.agent.drive_volatility, rtol=0.0, atol=1e-6)
    assert torch.allclose(scalar.agent.drive_jitter, resident.agent.drive_jitter, rtol=0.0, atol=1e-6)


def test_c3_chemistry_oracle_requires_materialized_c5_draws() -> None:
    bundle, _ = make_c3_fixture_v1(batch=1, capacity=8, active=4, width=8)
    with pytest.raises(TensorWorldContractError, match="chemistry noise sample is missing"):
        apply_c3_chemistry_oracle_v1(
            bundle,
            TensorWorldC3MechanicsConfigV1(enable_chemistry=True),
        )

    bundle.transactions.chemistry_noise_samples_valid[0, :4] = True
    bundle.base.core.agents.energy[0, :4] = 80.0
    bundle.base.core.agents.health[0, :4] = 6.0
    bundle.agent.injury[0, :4] = 0.5
    before = bundle.adaptation.chemistry.clone()
    apply_c3_chemistry_oracle_v1(
        bundle,
        TensorWorldC3MechanicsConfigV1(enable_chemistry=True),
    )
    assert torch.all(torch.isfinite(bundle.adaptation.chemistry))
    assert not torch.equal(bundle.adaptation.chemistry[0, :4], before[0, :4])
    assert torch.equal(bundle.adaptation.chemistry[0, 4:], before[0, 4:])


def test_c3_resident_noise_validation_fails_closed_for_active_rows() -> None:
    bundle, _ = make_c3_fixture_v1(batch=1, capacity=8, active=4, width=8)
    config = TensorWorldC3MechanicsConfigV1(enable_chemistry=True)
    bundle.agent.drive_noise[0, 0] = 0.25
    with pytest.raises(TensorWorldContractError, match="drive mechanics"):
        _validate_c3_noise_samples_v1(bundle, config)

    bundle.transactions.drive_noise_samples_valid[0, 0] = True
    bundle.transactions.drive_noise_samples[0, 0, 0] = float("nan")
    with pytest.raises(TensorWorldContractError, match="drive noise samples"):
        _validate_c3_noise_samples_v1(bundle, config)

    bundle.agent.drive_noise[0, 0] = 0.0
    bundle.transactions.drive_noise_samples[0, 0, 0] = 0.0
    with pytest.raises(TensorWorldContractError, match="chemistry"):
        _validate_c3_noise_samples_v1(bundle, config)

    bundle.transactions.chemistry_noise_samples_valid[0, :4] = True
    _validate_c3_noise_samples_v1(bundle, config)

    bundle.genome.drive_noise[0, 1] = 0.5
    _sync_compiled_policy_v1(bundle)
    with pytest.raises(TensorWorldContractError, match="drive mechanics"):
        _validate_c3_noise_samples_v1(bundle, TensorWorldC3MechanicsConfigV1())


def test_c3_active_mechanics_state_validation_matches_source_fail_closed_boundary() -> None:
    bundle, _ = make_c3_fixture_v1(batch=1, capacity=8, active=4, width=8)
    capture_c3_pre_action_v1(bundle)
    _validate_c3_active_mechanics_state_v1(bundle)

    bundle.agent.drive_inertia[0, 0] = -0.01
    with pytest.raises(TensorWorldContractError, match="drive_inertia"):
        _validate_c3_active_mechanics_state_v1(bundle)

    bundle.agent.drive_inertia[0, 0] = 0.25
    bundle.base.core.agents.alive[0, 0] = False
    with pytest.raises(TensorWorldContractError, match="partitions"):
        _validate_c3_active_mechanics_state_v1(bundle)

    bundle.base.core.agents.alive[0, 0] = True
    bundle.agent.drive_volatility[0, 0] = float("nan")
    with pytest.raises(TensorWorldContractError, match="drive_volatility"):
        _validate_c3_active_mechanics_state_v1(bundle)


def test_c3_identity_surface_comparator_is_slot_permutation_invariant() -> None:
    expected_ids = torch.tensor([[10, 20, -1, -1]], dtype=torch.int64)
    actual_ids = torch.tensor([[20, -1, 10, -1]], dtype=torch.int64)
    expected = {
        "observations": torch.tensor([[[1.0, 2.0], [3.0, 4.0], [0.0, 0.0], [0.0, 0.0]]]),
        "logits": torch.tensor([[0.5, 0.75, 0.0, 0.0]]),
    }
    actual = {
        "observations": torch.tensor([[[3.0, 4.0], [0.0, 0.0], [1.0, 2.0], [0.0, 0.0]]]),
        "logits": torch.tensor([[0.75, 0.0, 0.5, 0.0]]),
    }
    assert compare_c3_identity_surfaces_v1(expected_ids, expected, actual_ids, actual) == {}
    actual["logits"][0, 0] = 0.25
    assert compare_c3_identity_surfaces_v1(expected_ids, expected, actual_ids, actual) == {
        "logits": ((0, 20),),
    }

    with pytest.raises(TensorWorldContractError, match="identity-set mismatch"):
        compare_c3_identity_surfaces_v1(
            expected_ids,
            expected,
            torch.tensor([[20, -1, 30, -1]], dtype=torch.int64),
            actual,
        )


def test_c3_identity_comparator_covers_policy_action_movement_and_occupancy() -> None:
    """Compare C3 agent-facing surfaces by biological ID, not resident slot."""

    expected_ids = torch.tensor([[101, 202, -1, -1]], dtype=torch.int64)
    actual_ids = torch.tensor([[202, -1, 101, -1]], dtype=torch.int64)
    expected = {
        "observations": torch.tensor(
            [
                [
                    [1.0, 2.0, 3.0],
                    [4.0, 5.0, 6.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                ]
            ]
        ),
        "logits": torch.tensor(
            [
                [
                    [0.1, 0.2],
                    [0.3, 0.4],
                    [0.0, 0.0],
                    [0.0, 0.0],
                ]
            ]
        ),
        "row_actions": torch.tensor([[4, 5, -1, -1]], dtype=torch.int32),
        "actions_by_slot": torch.tensor([[4, 5, -1, -1]], dtype=torch.int32),
        "pos_y": torch.tensor([[7, 9, -1, -1]], dtype=torch.int64),
        "pos_x": torch.tensor([[3, 1, -1, -1]], dtype=torch.int64),
        "occupied": torch.tensor([[True, True, False, False]]),
    }
    actual = {
        "observations": torch.tensor(
            [
                [
                    [4.0, 5.0, 6.0],
                    [0.0, 0.0, 0.0],
                    [1.0, 2.0, 3.0],
                    [0.0, 0.0, 0.0],
                ]
            ]
        ),
        "logits": torch.tensor(
            [
                [
                    [0.3, 0.4],
                    [0.0, 0.0],
                    [0.1, 0.2],
                    [0.0, 0.0],
                ]
            ]
        ),
        "row_actions": torch.tensor([[5, -1, 4, -1]], dtype=torch.int32),
        "actions_by_slot": torch.tensor([[5, -1, 4, -1]], dtype=torch.int32),
        "pos_y": torch.tensor([[9, -1, 7, -1]], dtype=torch.int64),
        "pos_x": torch.tensor([[1, -1, 3, -1]], dtype=torch.int64),
        "occupied": torch.tensor([[True, False, True, False]]),
    }

    assert compare_c3_identity_surfaces_v1(
        expected_ids, expected, actual_ids, actual
    ) == {}

    actual["pos_x"][0, 0] = 2
    assert compare_c3_identity_surfaces_v1(
        expected_ids, expected, actual_ids, actual
    ) == {"pos_x": ((0, 202),)}


def test_c3_observation_comparator_reports_channel_and_identity() -> None:
    expected_ids = torch.tensor([[101, 202, -1]], dtype=torch.int64)
    actual_ids = torch.tensor([[202, -1, 101]], dtype=torch.int64)
    expected = torch.zeros(1, 3, 83)
    actual = torch.zeros(1, 3, 83)
    expected[0, 0, 18] = 0.25
    actual[0, 2, 18] = 0.25
    actual[0, 0, 22] = 0.5
    assert compare_c3_observation_channels_v1(
        expected_ids, expected, actual_ids, actual
    ) == {
        "patch_cover": ((0, 202, "patch_cover[dy=0,dx=0]"),),
    }


def test_c3_preparation_runner_preserves_delta_and_residency_boundaries() -> None:
    assert _structural_residency()["passed"] is True
    expected, _ = make_c3_fixture_v1(batch=1, capacity=8, active=4, width=8)
    actual, _ = make_c3_fixture_v1(batch=1, capacity=8, active=4, width=8)
    assert _compare_bundles(expected, actual) == {}
    actual.agent.drives[0, 0, 0] += 1.0e-3
    actual.base.core.agents.energy[0, 0] += 1.0
    mismatches = _compare_bundles(expected, actual)
    assert "agent.drives" in mismatches
    assert "base.core.agents.energy" in mismatches


def test_c3_preparation_runner_reconstructs_c1_c2_budget() -> None:
    budget = _frozen_budget_context()
    assert budget["c1_b48_median_ms"] == pytest.approx(10.225152015686035)
    assert budget["c2_b48_cumulative_median_ms"] == pytest.approx(15.557759761845)
    assert budget["remaining_budget_after_c2_ms"] > 0.0
    assert budget["c1_source_decision"] == "TW-C1-CONTRACT-RECONCILED-PASS-C2"
    assert budget["c2_source_decision"] == "TW-C2-STRONG-GO-REVIEW"


@pytest.mark.skipif(
    not torch.cuda.is_available() or not os.environ.get("TENSOR_WORLD_C2_PREBUILT"),
    reason="receipt-bound C2 CUDA extension is not configured",
)
def test_c3_backend_smoke_preserves_resident_boundary() -> None:
    source, _ = make_c3_fixture_v1(batch=1, capacity=8, active=4, width=8)
    backend = TensorWorldC3BackendV1(
        device="cuda:0", extension_path=os.environ["TENSOR_WORLD_C2_PREBUILT"]
    )
    handle = backend.initialize({"bundle": source, "fixture_mode": True})
    backend.advance(handle, TickRequestV1())
    backend.advance(handle, TickRequestV1())
    exported = backend.export_state(handle)["bundle"]
    assert exported.base.tick == 2
    assert exported.schema_version == C3_STATE_SCHEMA_VERSION_V1
    assert backend.residency_counters(handle)["host_callback_count"] == 0
    backend.close(handle)
