"""Static and deferred CUDA tests for the isolated R6 all-live preflight."""

from __future__ import annotations

import copy
from dataclasses import fields, is_dataclass
import os
from pathlib import Path

import pytest
import torch

from engine.tensor_world.c4_mating import (
    commit_c4_mate_births_oracle_v1,
    plan_c4_mate_births_oracle_v1,
)
from engine.tensor_world.c4_population import apply_c4_death_settlement_oracle_v1
from engine.tensor_world.c4_resident_collision_r6 import (
    make_c4_resident_collision_r6_boundary_v1,
    map_c4_resident_collision_r6_to_r5_abi_v1,
)
from engine.tensor_world.c4_resident_preflight_r6 import (
    C4_RESIDENT_PREFLIGHT_R6_DECISION_V1,
    C4_RESIDENT_PREFLIGHT_R6_SCHEMA_VERSION_V1,
    C4_RESIDENT_PREFLIGHT_R6_STATUS_V1,
    c4_resident_preflight_r6_source_receipt_v1,
    execute_c4_resident_preflight_r6_cuda_v1,
    make_c4_resident_preflight_r6_staging_v1,
)
from engine.tensor_world.contracts import TensorWorldContractError


ROOT = Path(__file__).resolve().parents[1]
KERNEL = ROOT / "engine/tensor_world/native/tensor_world_c4_resident_preflight_r6.cu"
BINDING = ROOT / (
    "engine/tensor_world/native/tensor_world_c4_resident_preflight_r6_binding.cpp"
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


def _authoritative(boundary) -> list[torch.Tensor]:
    groups = map_c4_resident_collision_r6_to_r5_abi_v1(
        boundary.bundle, boundary.rng_context
    )
    return [tensor for tensors in groups.values() for tensor in tensors]


def test_c4_r6_preflight_receipt_binds_frozen_r5_without_gate_claim() -> None:
    receipt = c4_resident_preflight_r6_source_receipt_v1()

    assert receipt["schema_version"] == C4_RESIDENT_PREFLIGHT_R6_SCHEMA_VERSION_V1
    assert receipt["route"] == "R6_B4_ALL_LIVE_PREFLIGHT_THEN_FROZEN_R5"
    assert receipt["decision"] == C4_RESIDENT_PREFLIGHT_R6_DECISION_V1
    assert receipt["batch"] == 4
    assert receipt["caller_owned_status"] == "contiguous_cuda_int32_B"
    assert receipt["atomicity"] == (
        "device_status_merged_before_any_authoritative_commit"
    )
    assert receipt["r5_source_modified"] is False
    assert receipt["runtime_fault_atomicity"] == (
        "not_claimed_for_multi_kernel_launch_faults"
    )
    assert receipt["gate_claim"] is False
    assert C4_RESIDENT_PREFLIGHT_R6_STATUS_V1 == {
        0: "ok",
        50: "invalid_or_duplicate_free_queue",
        51: "live_identity_or_id_map_inconsistent",
        52: "occupancy_or_patch_mirror_inconsistent",
        53: "batch_global_peer_failed",
    }
    assert all(len(row["sha256"]) == 64 for row in receipt["sources"].values())


def test_c4_r6_staging_and_cpu_route_fail_before_native_load(monkeypatch) -> None:
    with pytest.raises(TensorWorldContractError, match="bounded to B4"):
        make_c4_resident_preflight_r6_staging_v1(
            batch=1, capacity=8, grid_size=8, streams=2, device="cpu"
        )
    with pytest.raises(TensorWorldContractError, match="CPU fallback"):
        make_c4_resident_preflight_r6_staging_v1(
            batch=4, capacity=8, grid_size=8, streams=2, device="cpu"
        )

    boundary = make_c4_resident_collision_r6_boundary_v1(
        "b4_fragmentation_lineage_reuse"
    )

    def forbidden_loader(**_kwargs):
        raise AssertionError("native loader must not run")

    monkeypatch.setattr(
        "engine.tensor_world.c4_resident_preflight_r6."
        "load_tensor_world_c4_resident_preflight_r6_extension",
        forbidden_loader,
    )
    with pytest.raises(TensorWorldContractError, match="requires a CUDA"):
        execute_c4_resident_preflight_r6_cuda_v1(
            boundary.bundle,
            object(),
            boundary.config,
            rng_context=boundary.rng_context,
        )


def test_c4_r6_native_source_preflights_all_live_state_before_guarded_commit() -> None:
    source = KERNEL.read_text(encoding="utf-8")
    binding = BINDING.read_text(encoding="utf-8")

    assert '#include "tensor_world_c4_resident_population_r5.cu"' in source
    assert "__global__ void all_live_semantic_preflight" in source
    assert "for (int64_t slot = 0; slot < live.capacity; ++slot)" in source
    assert "live.agent_id_to_slot[offset + id] != slot" in source
    assert "live.occupancy_agent[cell] != id" in source
    assert "live.patch_agent[cell] != static_cast<int32_t>(id)" in source
    assert "live.free_queue[offset + prior] == slot" in source
    assert "__global__ void merge_preflight_status" in source
    preflight = source.index("r6::all_live_semantic_preflight<<<")
    r5_plan = source.index("r5::prepare_post_death_shadow<<<")
    merge = source.index("r6::merge_preflight_status<<<")
    r5_merge = source.index("r5::merge_all_status<<<")
    commit = source.index("r5::commit_deaths<<<")
    assert preflight < r5_plan < merge < r5_merge < commit
    assert "population_transaction" in binding


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_c4_r6_cuda_positive_parity_and_all_live_failure_atomicity(
    tmp_path: Path,
) -> None:
    build_dir = Path(os.environ.get("EIEL_C4_R6_PREFLIGHT_BUILD_DIR", tmp_path / "build"))
    cpu = make_c4_resident_collision_r6_boundary_v1(
        "b4_fragmentation_lineage_reuse"
    )
    cuda = copy.deepcopy(cpu)
    cuda.bundle = _move_dataclass_tensors(cpu.bundle, torch.device("cuda"))
    cuda.rng_context = _move_dataclass_tensors(
        cpu.rng_context, torch.device("cuda")
    )
    apply_c4_death_settlement_oracle_v1(
        cpu.bundle, event_sequence=cpu.rng_context.event_sequence
    )
    plan_c4_mate_births_oracle_v1(
        cpu.bundle, cpu.config, rng_context=cpu.rng_context
    )
    commit_c4_mate_births_oracle_v1(
        cpu.bundle, cpu.config, rng_context=cpu.rng_context
    )
    staging = make_c4_resident_preflight_r6_staging_v1(
        batch=4, capacity=8, grid_size=8, streams=2, device="cuda"
    )
    execute_c4_resident_preflight_r6_cuda_v1(
        cuda.bundle,
        staging,
        cuda.config,
        rng_context=cuda.rng_context,
        build_directory=build_dir,
    )
    torch.cuda.synchronize()
    assert staging.status.cpu().tolist() == [0, 0, 0, 0]
    for expected, actual in zip(_authoritative(cpu), _authoritative(cuda), strict=True):
        torch.testing.assert_close(actual.cpu(), expected, equal_nan=True)

    expected_status = {
        "duplicate_free_queue": [53, 53, 53, 50],
        "conflicting_occupancy": [53, 53, 52, 53],
        "conflicting_identity_map": [53, 53, 53, 51],
    }
    for case, statuses in expected_status.items():
        failed = make_c4_resident_collision_r6_boundary_v1(case)
        failed.bundle = _move_dataclass_tensors(failed.bundle, torch.device("cuda"))
        failed.rng_context = _move_dataclass_tensors(
            failed.rng_context, torch.device("cuda")
        )
        failed_staging = make_c4_resident_preflight_r6_staging_v1(
            batch=4, capacity=8, grid_size=8, streams=2, device="cuda"
        )
        before = [tensor.clone() for tensor in _authoritative(failed)]
        execute_c4_resident_preflight_r6_cuda_v1(
            failed.bundle,
            failed_staging,
            failed.config,
            rng_context=failed.rng_context,
            build_directory=build_dir,
        )
        torch.cuda.synchronize()
        assert failed_staging.status.cpu().tolist() == statuses
        assert all(
            value != 0
            for value in failed_staging.population.mutation.birth.status.cpu().tolist()
        )
        for expected, actual in zip(before, _authoritative(failed), strict=True):
            torch.testing.assert_close(actual, expected, equal_nan=True)
