"""Focused contract, B1 parity, slot-reuse reset, and rollback for C4 R3."""

from __future__ import annotations

import copy
from dataclasses import fields, is_dataclass
from pathlib import Path

import pytest
import torch

from engine.tensor_world.c4_fixture import make_c4_fixture_v1
from engine.tensor_world.c4_population import (
    TensorWorldC4BirthConfigV1,
    apply_c4_death_settlement_oracle_v1,
    commit_c4_ordinary_births_oracle_v1,
    plan_c4_ordinary_births_oracle_v1,
)
from engine.tensor_world.c4_resident_birth_r2 import (
    _half_tensors,
    _inherit_tensors,
    _minus_one_tensors,
    _one_tensors,
    _primary_tensors,
    _zero_tensors,
)
from engine.tensor_world.c4_resident_turnover_r3 import (
    C4_RESIDENT_TURNOVER_R3_DECISION_V1,
    C4_RESIDENT_TURNOVER_R3_SCHEMA_VERSION_V1,
    _death_authoritative,
    c4_resident_turnover_r3_source_receipt_v1,
    execute_c4_resident_turnover_r3_cuda_v1,
    make_c4_resident_turnover_r3_staging_v1,
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


def _all_authoritative(bundle) -> list[torch.Tensor]:
    return [
        *_primary_tensors(bundle),
        *_death_authoritative(bundle),
        *_inherit_tensors(bundle),
        *_zero_tensors(bundle),
        *_minus_one_tensors(bundle),
        *_half_tensors(bundle),
        *_one_tensors(bundle),
    ]


def _poison_reused_slot(bundle, slot: int) -> None:
    core = bundle.base.base.core
    c2 = bundle.base.base
    agent = bundle.base.agent
    core.transactions.actions[:, slot] = 6
    core.transactions.moved[:, slot] = True
    core.transactions.gathered[:, slot] = True
    core.transactions.food_gained[:, slot] = 9.0
    core.transactions.observations[:, slot] = 9.0
    core.transactions.logits[:, slot] = 9.0
    c2.eed.active_y[:, slot] = 7
    c2.eed.active_x[:, slot] = 7
    c2.eed.active_class[:, slot] = 2
    c2.eed.handling_progress[:, slot] = 9.0
    c2.eed.handling_axis[:, slot] = 9.0
    c2.effects.pre_policy_health_delta[:, slot] = 9.0
    c2.effects.post_action_health_delta[:, slot] = 9.0
    c2.effects.hazard_flags[:, slot] = 3
    c2.world_events.gather_amount[:, slot] = 9.0
    c2.world_events.gather_resource_class[:, slot] = 2
    c2.transactions.observations[:, slot] = 9.0
    c2.transactions.logits[:, slot] = 9.0
    agent.age[:, slot] = 99
    agent.survival_steps[:, slot] = 99
    agent.energy_collected[:, slot] = 99.0
    agent.energy_collected_exact[:, slot] = 99.0
    c2.rng.jungle_uniforms[:, slot] = 0.123
    bundle.population.repro_cooldown[:, slot] = 17
    bundle.population.planned_repro_cooldown[:, slot] = 17


def _slot_reuse_fixture():
    bundle, _ = make_c4_fixture_v1(batch=1, capacity=8, active=3, width=8)
    # Physical slots 3..7 remain empty but are deliberately absent from the
    # declared queue, making the newly deceased slot the first reusable row.
    bundle.population.free_queue.fill_(-1)
    bundle.population.free_queue_count.zero_()
    _poison_reused_slot(bundle, 1)
    bundle.base.base.core.agents.health[0, 1] = 0.0
    bundle.base.lifecycle.death_candidate[0, 1] = True
    return bundle


def test_c4_r3_receipt_is_bounded_and_source_bound() -> None:
    receipt = c4_resident_turnover_r3_source_receipt_v1()
    assert receipt["schema_version"] == C4_RESIDENT_TURNOVER_R3_SCHEMA_VERSION_V1
    assert receipt["decision"] == C4_RESIDENT_TURNOVER_R3_DECISION_V1
    assert receipt["route"] == "R3_B1_DEATH_RELEASE_ORDINARY_BIRTH_SLOT_REUSE"
    assert receipt["authoritative_commit"] == (
        "one_status_guarded_multi_kernel_commit_phase"
    )
    assert receipt["event_order"] == "death_rows_then_birth_rows"
    assert receipt["rung_admission"] == (
        "death_and_birth_and_newly_dead_child_slot_required"
    )
    assert receipt["gate_claim"] is False
    for source in receipt["sources"].values():
        assert len(source["sha256"]) == 64
        assert (ROOT / source["path"]).is_file()


def test_c4_r3_rejects_cpu_staging_without_fallback() -> None:
    with pytest.raises(TensorWorldContractError, match="requires CUDA"):
        make_c4_resident_turnover_r3_staging_v1(
            batch=1, capacity=8, grid_size=8, device="cpu"
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_c4_r3_b1_exact_parity_slot_reset_and_failed_birth_rollback(
    tmp_path: Path,
) -> None:
    cpu_bundle = _slot_reuse_fixture()
    cuda_bundle = _move_dataclass_tensors(cpu_bundle, torch.device("cuda"))
    dead_id = int(cpu_bundle.base.base.core.agents.agent_id[0, 1])
    jungle_before = cuda_bundle.base.base.rng.jungle_uniforms.clone()
    static_before = (
        cuda_bundle.base.base.core.identity.domain_world_ids.clone(),
        cuda_bundle.base.base.core.identity.slot_ids.clone(),
    )
    scalar_rng_before = [
        getattr(cuda_bundle.scalar_rng, field.name).clone()
        for field in fields(cuda_bundle.scalar_rng)
        if field.name != "schema_version"
    ]
    spawn_tape = torch.zeros((1, 8, 3), dtype=torch.float64)
    config = TensorWorldC4BirthConfigV1(allow_mutation=False, generation=7)

    apply_c4_death_settlement_oracle_v1(cpu_bundle)
    plan_c4_ordinary_births_oracle_v1(cpu_bundle, config, spawn_uniforms=spawn_tape)
    commit_c4_ordinary_births_oracle_v1(cpu_bundle, config)

    staging = make_c4_resident_turnover_r3_staging_v1(
        batch=1, capacity=8, grid_size=8, device="cuda"
    )
    staging.birth.spawn_uniforms.copy_(spawn_tape.to(device="cuda"))
    staging.birth.event_sequence.copy_(cuda_bundle.events.active_count)
    with pytest.raises(TensorWorldContractError, match="mutation is unsupported"):
        execute_c4_resident_turnover_r3_cuda_v1(
            cuda_bundle, staging, TensorWorldC4BirthConfigV1(allow_mutation=True)
        )
    with pytest.raises(TensorWorldContractError, match="mate/recombination"):
        execute_c4_resident_turnover_r3_cuda_v1(
            cuda_bundle, staging, config, recombination=True
        )
    execute_c4_resident_turnover_r3_cuda_v1(
        cuda_bundle, staging, config, build_directory=tmp_path / "build"
    )
    torch.cuda.synchronize()
    assert staging.death_status.cpu().tolist() == [0]
    assert staging.birth.status.cpu().tolist() == [0]
    assert staging.death_slots[0, 0].item() == 1
    assert staging.birth.child_slots[0, 0].item() == 1
    child_id = int(staging.birth.child_agent_ids[0, 0])
    assert child_id > dead_id
    assert int(cuda_bundle.base.base.core.agents.agent_id[0, 1]) == child_id
    assert cuda_bundle.events.typed_event_type[0, :2].cpu().tolist() == [4, 1]
    assert cuda_bundle.events.typed_generation[0, :2].cpu().tolist() == [0, 7]
    assert int(cuda_bundle.events.typed_founder_id[0, 0]) == int(
        cpu_bundle.events.typed_founder_id[0, 0]
    )
    for expected, actual in zip(
        _all_authoritative(cpu_bundle), _all_authoritative(cuda_bundle), strict=True
    ):
        torch.testing.assert_close(actual.cpu(), expected, equal_nan=True)
    assert int(cuda_bundle.population.planned_repro_cooldown[0, 1]) == 0
    torch.testing.assert_close(cuda_bundle.base.base.rng.jungle_uniforms, jungle_before)
    torch.testing.assert_close(
        cuda_bundle.base.base.core.identity.domain_world_ids, static_before[0]
    )
    torch.testing.assert_close(cuda_bundle.base.base.core.identity.slot_ids, static_before[1])
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

    failed = _slot_reuse_fixture()
    failed_cuda = _move_dataclass_tensors(failed, torch.device("cuda"))
    failed_cuda.events.active_count.fill_(failed_cuda.events.capacity - 1)
    failed_staging = make_c4_resident_turnover_r3_staging_v1(
        batch=1, capacity=8, grid_size=8, device="cuda"
    )
    failed_staging.birth.event_sequence.copy_(failed_cuda.events.active_count)
    before = [tensor.clone() for tensor in _all_authoritative(failed_cuda)]
    execute_c4_resident_turnover_r3_cuda_v1(
        failed_cuda, failed_staging, config, build_directory=tmp_path / "build"
    )
    torch.cuda.synchronize()
    assert failed_staging.death_status.cpu().tolist() == [0]
    assert failed_staging.birth.status.cpu().tolist() == [6]
    for expected, actual in zip(before, _all_authoritative(failed_cuda), strict=True):
        torch.testing.assert_close(actual, expected, equal_nan=True)

    # R3 is not a general turnover coordinator: its executable admission is
    # exactly death + birth + reuse of a slot released by this transaction.
    rejected_cases = []
    zero_death, _ = make_c4_fixture_v1(batch=1, capacity=8, active=3, width=8)
    rejected_cases.append((zero_death, config))
    rejected_cases.append(
        (_slot_reuse_fixture(), TensorWorldC4BirthConfigV1(max_births=0))
    )
    old_prefix = _slot_reuse_fixture()
    old_prefix.population.free_queue[0, 0] = 3
    old_prefix.population.free_queue_count[0] = 1
    rejected_cases.append((old_prefix, TensorWorldC4BirthConfigV1(max_births=1)))
    for case_index, (rejected, rejected_config) in enumerate(rejected_cases):
        rejected_cuda = _move_dataclass_tensors(rejected, torch.device("cuda"))
        rejected_staging = make_c4_resident_turnover_r3_staging_v1(
            batch=1, capacity=8, grid_size=8, device="cuda"
        )
        rejected_staging.birth.event_sequence.copy_(rejected_cuda.events.active_count)
        rejected_before = [tensor.clone() for tensor in _all_authoritative(rejected_cuda)]
        execute_c4_resident_turnover_r3_cuda_v1(
            rejected_cuda,
            rejected_staging,
            rejected_config,
            build_directory=tmp_path / "build",
        )
        torch.cuda.synchronize()
        assert rejected_staging.death_status.cpu().tolist() == [25], case_index
        assert rejected_staging.birth.status.cpu().tolist() == [24], case_index
        for expected, actual in zip(
            rejected_before, _all_authoritative(rejected_cuda), strict=True
        ):
            torch.testing.assert_close(actual, expected, equal_nan=True)
