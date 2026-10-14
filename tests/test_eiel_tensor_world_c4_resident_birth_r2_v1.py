"""Focused CPU contract and B1 CUDA parity for executable C4 R2 births."""

from __future__ import annotations

import copy
from dataclasses import fields, is_dataclass
from pathlib import Path

import pytest
import torch

from engine.tensor_world.c4_fixture import make_c4_fixture_v1
from engine.tensor_world.c4_population import (
    TensorWorldC4BirthConfigV1,
    commit_c4_ordinary_births_oracle_v1,
    plan_c4_ordinary_births_oracle_v1,
)
from engine.tensor_world.c4_resident_birth_r2 import (
    C4_RESIDENT_BIRTH_R2_DECISION_V1,
    C4_RESIDENT_BIRTH_R2_SCHEMA_VERSION_V1,
    TensorWorldC4ResidentBirthR2StagingV1,
    _half_tensors,
    _inherit_tensors,
    _minus_one_tensors,
    _one_tensors,
    _primary_tensors,
    _zero_tensors,
    c4_resident_birth_r2_source_receipt_v1,
    execute_c4_resident_birth_r2_cuda_v1,
    make_c4_resident_birth_r2_staging_v1,
)
from engine.tensor_world.contracts import TensorWorldContractError


ROOT = Path(__file__).resolve().parents[1]


def _move_dataclass_tensors(value: object, device: torch.device) -> object:
    clone = copy.deepcopy(value)

    def move(owner: object) -> None:
        if not is_dataclass(owner):
            return
        for field in fields(owner):
            item = getattr(owner, field.name)
            if isinstance(item, torch.Tensor):
                setattr(owner, field.name, item.to(device=device))
            elif is_dataclass(item):
                move(item)

    move(clone)
    return clone


def _cpu_staging() -> TensorWorldC4ResidentBirthR2StagingV1:
    slots = (1, 4)
    return TensorWorldC4ResidentBirthR2StagingV1(
        spawn_uniforms=torch.zeros((1, 4, 3), dtype=torch.float64),
        parent_slots=torch.full(slots, -1, dtype=torch.int32),
        child_slots=torch.full(slots, -1, dtype=torch.int32),
        child_pos_y=torch.full(slots, -1, dtype=torch.int32),
        child_pos_x=torch.full(slots, -1, dtype=torch.int32),
        child_energy=torch.zeros(slots),
        planned_count=torch.zeros(1, dtype=torch.int64),
        decision_parent_slots=torch.full(slots, -1, dtype=torch.int64),
        decision_adjacent_counts=torch.zeros(slots, dtype=torch.int32),
        decision_reason_codes=torch.full(slots, -1, dtype=torch.int32),
        decision_count=torch.zeros(1, dtype=torch.int64),
        planned_repro_cooldown=torch.zeros(slots, dtype=torch.int32),
        spawn_rng_draws_added=torch.zeros(1, dtype=torch.int64),
        spawn_rng_semantic_draws_added=torch.zeros(1, dtype=torch.int64),
        child_agent_ids=torch.full(slots, -1, dtype=torch.int64),
        queue_cursor=torch.zeros(1, dtype=torch.int64),
        event_sequence=torch.zeros(1, dtype=torch.int64),
        status=torch.zeros(1, dtype=torch.int32),
    )


def _poison_free_slots(bundle) -> None:
    free = slice(3, bundle.capacity)
    core = bundle.base.base.core
    c2 = bundle.base.base
    a = bundle.base.agent
    core.transactions.actions[:, free] = 6
    core.transactions.moved[:, free] = True
    core.transactions.gathered[:, free] = True
    core.transactions.food_gained[:, free] = 9.0
    core.transactions.observations[:, free] = 9.0
    core.transactions.logits[:, free] = 9.0
    c2.eed.active_y[:, free] = 7
    c2.eed.active_x[:, free] = 7
    c2.eed.active_class[:, free] = 2
    c2.eed.handling_progress[:, free] = 9.0
    c2.eed.handling_axis[:, free] = 9.0
    c2.policy.packed_weights[:, free] = -99.0
    c2.effects.pre_policy_health_delta[:, free] = 9.0
    c2.effects.post_action_health_delta[:, free] = 9.0
    c2.effects.hazard_flags[:, free] = 3
    c2.world_events.gather_amount[:, free] = 9.0
    c2.world_events.gather_resource_class[:, free] = 2
    c2.transactions.observations[:, free] = 9.0
    c2.transactions.logits[:, free] = 9.0
    a.age[:, free] = 99
    a.survival_steps[:, free] = 99
    a.energy_collected[:, free] = 99.0
    a.energy_collected_exact[:, free] = 99.0
    c2.rng.jungle_uniforms[:, free] = 0.123


def test_c4_r2_receipt_binds_sources_and_reset_semantics() -> None:
    receipt = c4_resident_birth_r2_source_receipt_v1()
    assert receipt["schema_version"] == C4_RESIDENT_BIRTH_R2_SCHEMA_VERSION_V1
    assert receipt["decision"] == C4_RESIDENT_BIRTH_R2_DECISION_V1
    assert receipt["route"] == "R2_B1_CUDA_ORDINARY_BIRTH_ONLY_NO_MUTATION"
    assert receipt["gate_claim"] is False
    assert receipt["cpu_fallback"] is False
    assert receipt["mutation"] == "fail_closed"
    assert receipt["mate_or_recombination"] == "fail_closed"
    assert "neutral_axis" in receipt["handling_axis_newborn"]
    assert "preserve" in receipt["jungle_uniforms_newborn"]
    assert "preserve" in receipt["physical_identity_projection"]
    for source in receipt["sources"].values():
        assert len(source["sha256"]) == 64
        assert (ROOT / source["path"]).is_file()


def test_c4_r2_staging_rejects_cpu_and_unsupported_modes() -> None:
    with pytest.raises(TensorWorldContractError, match="CPU fallback is forbidden"):
        _cpu_staging().validate(batch=1, capacity=4)
    with pytest.raises(TensorWorldContractError, match="CPU fallback is forbidden"):
        make_c4_resident_birth_r2_staging_v1(batch=1, capacity=4, device="cpu")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_c4_r2_b1_oracle_parity_poison_reset_and_atomic_failure(tmp_path: Path) -> None:
    cpu_bundle, _ = make_c4_fixture_v1(batch=1, capacity=8, active=3, width=8)
    _poison_free_slots(cpu_bundle)
    cuda_bundle = _move_dataclass_tensors(cpu_bundle, torch.device("cuda"))
    scalar_rng_before = [
        getattr(cuda_bundle.scalar_rng, field.name).clone()
        for field in fields(cuda_bundle.scalar_rng)
        if field.name != "schema_version"
    ]
    spawn_tape = torch.zeros((1, 8, 3), dtype=torch.float64)
    config = TensorWorldC4BirthConfigV1(allow_mutation=False)
    plan_c4_ordinary_births_oracle_v1(cpu_bundle, config, spawn_uniforms=spawn_tape)
    commit_c4_ordinary_births_oracle_v1(cpu_bundle, config)

    staging = make_c4_resident_birth_r2_staging_v1(batch=1, capacity=8, device="cuda")
    staging.spawn_uniforms.copy_(spawn_tape.to(device="cuda"))
    staging.event_sequence.copy_(cuda_bundle.events.active_count)
    with pytest.raises(TensorWorldContractError, match="mutation is unsupported"):
        execute_c4_resident_birth_r2_cuda_v1(
            cuda_bundle,
            staging,
            TensorWorldC4BirthConfigV1(allow_mutation=True),
        )
    with pytest.raises(TensorWorldContractError, match="mate/recombination"):
        execute_c4_resident_birth_r2_cuda_v1(
            cuda_bundle, staging, config, recombination=True
        )
    execute_c4_resident_birth_r2_cuda_v1(
        cuda_bundle, staging, config, build_directory=tmp_path / "build"
    )
    torch.cuda.synchronize()
    assert staging.status.cpu().tolist() == [0]
    assert staging.planned_count.cpu().tolist() == [2]
    for expected, actual in zip(
        _primary_tensors(cpu_bundle), _primary_tensors(cuda_bundle), strict=True
    ):
        torch.testing.assert_close(actual.cpu(), expected, equal_nan=True)
    for expected, actual in zip(
        _inherit_tensors(cpu_bundle), _inherit_tensors(cuda_bundle), strict=True
    ):
        torch.testing.assert_close(actual.cpu(), expected, equal_nan=True)

    children = staging.child_slots[0, :2].cpu().tolist()
    for tensor in _zero_tensors(cuda_bundle):
        assert torch.count_nonzero(tensor[0, children]).item() == 0
    for tensor in _minus_one_tensors(cuda_bundle):
        assert torch.all(tensor[0, children] == -1).item()
    for tensor in _half_tensors(cuda_bundle):
        torch.testing.assert_close(
            tensor[0, children], torch.full_like(tensor[0, children], 0.5)
        )
    for tensor in _one_tensors(cuda_bundle):
        torch.testing.assert_close(
            tensor[0, children], torch.ones_like(tensor[0, children])
        )
    assert torch.all(cuda_bundle.base.base.eed.handling_axis[0, children] == 0).item()
    assert torch.all(
        cuda_bundle.base.base.rng.jungle_uniforms[0, children] == 0.123
    ).item()
    for before, field in zip(
        scalar_rng_before,
        (
            getattr(cuda_bundle.scalar_rng, item.name)
            for item in fields(cuda_bundle.scalar_rng)
            if item.name != "schema_version"
        ),
        strict=True,
    ):
        torch.testing.assert_close(field, before, equal_nan=True)

    failed, _ = make_c4_fixture_v1(batch=1, capacity=8, active=3, width=8)
    failed_cuda = _move_dataclass_tensors(failed, torch.device("cuda"))
    failed_cuda.events.active_count.fill_(failed_cuda.events.capacity)
    failed_staging = make_c4_resident_birth_r2_staging_v1(
        batch=1, capacity=8, device="cuda"
    )
    failed_staging.event_sequence.copy_(failed_cuda.events.active_count)
    before = [
        tensor.clone()
        for group in (
            _primary_tensors(failed_cuda), _inherit_tensors(failed_cuda),
            _zero_tensors(failed_cuda), _minus_one_tensors(failed_cuda),
            _half_tensors(failed_cuda), _one_tensors(failed_cuda),
        )
        for tensor in group
    ]
    execute_c4_resident_birth_r2_cuda_v1(
        failed_cuda, failed_staging, config, build_directory=tmp_path / "build"
    )
    torch.cuda.synchronize()
    assert failed_staging.status.cpu().tolist() == [6]
    after = [
        tensor
        for group in (
            _primary_tensors(failed_cuda), _inherit_tensors(failed_cuda),
            _zero_tensors(failed_cuda), _minus_one_tensors(failed_cuda),
            _half_tensors(failed_cuda), _one_tensors(failed_cuda),
        )
        for tensor in group
    ]
    for expected, actual in zip(before, after, strict=True):
        torch.testing.assert_close(actual, expected, equal_nan=True)

    sequence_failed = _move_dataclass_tensors(failed, torch.device("cuda"))
    sequence_staging = make_c4_resident_birth_r2_staging_v1(
        batch=1, capacity=8, device="cuda"
    )
    sequence_staging.event_sequence.fill_(1)
    sequence_before = [tensor.clone() for tensor in _primary_tensors(sequence_failed)]
    execute_c4_resident_birth_r2_cuda_v1(
        sequence_failed, sequence_staging, config, build_directory=tmp_path / "build"
    )
    torch.cuda.synchronize()
    assert sequence_staging.status.cpu().tolist() == [7]
    for expected, actual in zip(
        sequence_before, _primary_tensors(sequence_failed), strict=True
    ):
        torch.testing.assert_close(actual, expected, equal_nan=True)

    occupancy_failed = _move_dataclass_tensors(failed, torch.device("cuda"))
    empty_y, empty_x = torch.nonzero(
        failed.base.base.core.occupancy.agent_id_grid[0] == -1
    )[0].tolist()
    occupancy_failed.base.base.core.occupancy.slot_grid[0, empty_y, empty_x] = 0
    occupancy_staging = make_c4_resident_birth_r2_staging_v1(
        batch=1, capacity=8, device="cuda"
    )
    occupancy_before = [tensor.clone() for tensor in _primary_tensors(occupancy_failed)]
    execute_c4_resident_birth_r2_cuda_v1(
        occupancy_failed, occupancy_staging, config, build_directory=tmp_path / "build"
    )
    torch.cuda.synchronize()
    assert occupancy_staging.status.cpu().tolist() == [4]
    for expected, actual in zip(
        occupancy_before, _primary_tensors(occupancy_failed), strict=True
    ):
        torch.testing.assert_close(actual, expected, equal_nan=True)

    energy_failed = _move_dataclass_tensors(failed, torch.device("cuda"))
    energy_staging = make_c4_resident_birth_r2_staging_v1(
        batch=1, capacity=8, device="cuda"
    )
    energy_before = [tensor.clone() for tensor in _primary_tensors(energy_failed)]
    execute_c4_resident_birth_r2_cuda_v1(
        energy_failed,
        energy_staging,
        TensorWorldC4BirthConfigV1(allow_mutation=False, max_energy=0.0),
        build_directory=tmp_path / "build",
    )
    torch.cuda.synchronize()
    assert energy_staging.status.cpu().tolist() == [1]
    for expected, actual in zip(
        energy_before, _primary_tensors(energy_failed), strict=True
    ):
        torch.testing.assert_close(actual, expected, equal_nan=True)
