"""Focused R0/R1 contract and CUDA parity for resident C4 death settlement."""

from __future__ import annotations

import copy
from dataclasses import fields, is_dataclass
from pathlib import Path

import pytest
import torch

from engine.tensor_world.c4_fixture import make_c4_fixture_v1
from engine.tensor_world.c4_population import apply_c4_death_settlement_oracle_v1
from engine.tensor_world.c4_resident_death import (
    C4_RESIDENT_DEATH_DECISION_V1,
    C4_RESIDENT_DEATH_SCHEMA_VERSION_V1,
    C4_RESIDENT_DEATH_STAGING_SCHEMA_VERSION_V1,
    TensorWorldC4ResidentDeathStagingV1,
    _authoritative_tensors,
    c4_resident_death_native_source_manifest_v1,
    execute_c4_resident_death_transaction_cuda_v1,
    make_c4_resident_death_staging_v1,
)
from engine.tensor_world.contracts import TensorWorldContractError


ROOT = Path(__file__).resolve().parents[1]


def _cpu_staging(*, batch: int = 2, capacity: int = 4):
    slots = (batch, capacity)
    return TensorWorldC4ResidentDeathStagingV1(
        death_agent_ids=torch.full(slots, -1, dtype=torch.int64),
        death_slots=torch.full(slots, -1, dtype=torch.int64),
        death_reason_codes=torch.zeros(slots, dtype=torch.int32),
        death_count=torch.zeros(batch, dtype=torch.int64),
        survivor_agent_ids=torch.full(slots, -1, dtype=torch.int64),
        survivor_slots=torch.full(slots, -1, dtype=torch.int32),
        survivor_count=torch.zeros(batch, dtype=torch.int64),
        status=torch.zeros(batch, dtype=torch.int32),
    )


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


def test_c4_resident_death_manifest_is_bounded_and_no_gate() -> None:
    manifest = c4_resident_death_native_source_manifest_v1()
    assert manifest["schema_version"] == C4_RESIDENT_DEATH_SCHEMA_VERSION_V1
    assert manifest["staging_schema_version"] == (
        C4_RESIDENT_DEATH_STAGING_SCHEMA_VERSION_V1
    )
    assert manifest["decision"] == C4_RESIDENT_DEATH_DECISION_V1
    assert manifest["route"] == "R0_R1_CUDA_DEATH_ONLY"
    assert manifest["caller_owned_fixed_capacity_staging"] is True
    assert manifest["cpu_fallback"] is False
    assert manifest["python_world_or_agent_loops"] is False
    assert manifest["gate_claim"] is False
    assert set(manifest["sources"]) == {
        "adapter",
        "binding",
        "kernel",
        "architecture_b_lifecycle",
        "architecture_b_typed_events",
    }
    for source in manifest["sources"].values():
        assert len(source["sha256"]) == 64
        assert (ROOT / source["path"]).is_file()


def test_c4_resident_death_staging_rejects_cpu_without_fallback() -> None:
    with pytest.raises(TensorWorldContractError, match="CPU fallback is forbidden"):
        _cpu_staging().validate(batch=2, capacity=4)
    with pytest.raises(TensorWorldContractError, match="CPU fallback is forbidden"):
        make_c4_resident_death_staging_v1(
            batch=2, capacity=4, device=torch.device("cpu")
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_c4_resident_death_b1_oracle_parity(tmp_path: Path) -> None:
    cpu_bundle, _ = make_c4_fixture_v1(batch=1, capacity=6, active=3, width=8)
    cpu_bundle.base.base.core.agents.health[0, 1] = 0.0
    cpu_bundle.base.lifecycle.death_candidate[0, 1] = True
    cuda_bundle = _move_dataclass_tensors(cpu_bundle, torch.device("cuda"))
    apply_c4_death_settlement_oracle_v1(cpu_bundle)

    staging = make_c4_resident_death_staging_v1(
        batch=1, capacity=6, device=torch.device("cuda")
    )
    execute_c4_resident_death_transaction_cuda_v1(
        cuda_bundle,
        staging,
        build_directory=tmp_path / "build",
    )
    torch.cuda.synchronize()
    assert staging.status.cpu().tolist() == [0]
    assert staging.death_agent_ids[:, :1].cpu().tolist() == [[2]]
    assert staging.death_slots[:, :1].cpu().tolist() == [[1]]
    for expected, actual in zip(
        _authoritative_tensors(cpu_bundle),
        _authoritative_tensors(cuda_bundle),
        strict=True,
    ):
        torch.testing.assert_close(actual.cpu(), expected, equal_nan=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_c4_resident_death_batched_parity_and_fail_closed_atomicity(tmp_path: Path) -> None:
    cpu_bundle, _ = make_c4_fixture_v1(batch=2, capacity=8, active=4, width=8)
    cpu_bundle.base.base.core.agents.energy[0, 1] = 0.0
    cpu_bundle.base.base.core.agents.health[0, 2] = 0.0
    cpu_bundle.base.lifecycle.death_candidate[0, 1:3] = True
    cpu_bundle.base.base.core.agents.health[1, 3] = 0.0
    cpu_bundle.base.lifecycle.death_candidate[1, 3] = True
    cuda_bundle = _move_dataclass_tensors(cpu_bundle, torch.device("cuda"))
    apply_c4_death_settlement_oracle_v1(cpu_bundle)

    staging = make_c4_resident_death_staging_v1(
        batch=2, capacity=8, device=torch.device("cuda")
    )
    execute_c4_resident_death_transaction_cuda_v1(
        cuda_bundle,
        staging,
        build_directory=tmp_path / "build",
    )
    torch.cuda.synchronize()
    assert staging.status.cpu().tolist() == [0, 0]
    assert staging.death_agent_ids[:, :2].cpu().tolist() == [[2, 3], [4, -1]]
    for expected, actual in zip(
        _authoritative_tensors(cpu_bundle),
        _authoritative_tensors(cuda_bundle),
        strict=True,
    ):
        torch.testing.assert_close(actual.cpu(), expected, equal_nan=True)

    # One invalid world suppresses the commit in every world.
    failed_cpu, _ = make_c4_fixture_v1(batch=2, capacity=6, active=3, width=8)
    failed_cpu.base.lifecycle.death_candidate[0, 0] = True
    failed_cpu.population.free_queue[1, 1] = failed_cpu.population.free_queue[1, 0]
    failed_cuda = _move_dataclass_tensors(failed_cpu, torch.device("cuda"))
    before = [tensor.clone() for tensor in _authoritative_tensors(failed_cuda)]
    failed_staging = make_c4_resident_death_staging_v1(
        batch=2, capacity=6, device=torch.device("cuda")
    )
    execute_c4_resident_death_transaction_cuda_v1(
        failed_cuda,
        failed_staging,
        build_directory=tmp_path / "build",
    )
    torch.cuda.synchronize()
    assert failed_staging.status.cpu().tolist() == [0, 4]
    for expected, actual in zip(before, _authoritative_tensors(failed_cuda), strict=True):
        torch.testing.assert_close(actual, expected, equal_nan=True)

    failed_cuda.events = None
    with pytest.raises(TensorWorldContractError, match="event boundary is fail-closed"):
        execute_c4_resident_death_transaction_cuda_v1(failed_cuda, failed_staging)
