from __future__ import annotations

from copy import deepcopy
from inspect import getsource

import pytest
import torch

from engine.perf.cuda_architecture_b_resident_reproduction_exposure_sidecar_v1 import (
    CELL_STATUS_ATTEMPT_CAPACITY_EXCEEDED,
    CELL_STATUS_OK,
    CudaResidentReproductionExposureError,
    CudaResidentReproductionExposureSidecarV1,
    PATH_ORDINARY_LIVE_BIRTH,
    ReproductionExposureExistingCountsV1,
    ResidentReproductionExposureShapeV1,
    STORAGE_MODE_COMPACT_COMPLETE,
    allocate_cuda_resident_reproduction_exposure_sidecar_v1,
    project_resident_reproduction_exposure_v1,
)


def _append_success(
    owner: CudaResidentReproductionExposureSidecarV1,
    *,
    cell: int,
    generation: int,
    parent_0: int,
    parent_1: int,
) -> None:
    attempt = int(owner.attempt_cursors[cell].item())
    snapshot = int(owner.snapshot_cursors[cell].item())
    owner.snapshot_paths[cell, snapshot : snapshot + 2] = (
        PATH_ORDINARY_LIVE_BIRTH
    )
    owner.snapshot_generations[cell, snapshot : snapshot + 2] = generation
    owner.snapshot_ticks[cell, snapshot : snapshot + 2] = 3
    owner.snapshot_eligible_counts[cell, snapshot : snapshot + 2] = 1
    owner.snapshot_eligible_parent_ids[cell, snapshot, 0] = parent_0
    owner.snapshot_eligible_parent_ids[cell, snapshot + 1, 0] = parent_1
    owner.snapshot_cursors[cell] = snapshot + 2
    owner.attempt_paths[cell, attempt] = PATH_ORDINARY_LIVE_BIRTH
    owner.attempt_generations[cell, attempt] = generation
    owner.attempt_ticks[cell, attempt] = 3
    owner.attempt_eligible_snapshot_refs[cell, attempt] = torch.tensor(
        (snapshot, snapshot + 1),
        dtype=torch.int64,
        device=owner.device,
    )
    owner.attempt_eligible_counts[cell, attempt] = 1
    owner.attempt_selected_parent_ids[cell, attempt] = torch.tensor(
        (parent_0, parent_1),
        dtype=torch.int64,
        device=owner.device,
    )
    owner.attempt_success[cell, attempt] = True
    owner.attempt_failure_codes[cell, attempt] = 0
    owner.attempt_commit_ordinals[cell, attempt] = attempt
    owner.attempt_child_agent_ids[cell, attempt] = 10_000 + attempt
    owner.attempt_cursors[cell] = attempt + 1


def test_shape_and_cpu_fallback_fail_closed() -> None:
    with pytest.raises(
        CudaResidentReproductionExposureError,
        match="B/E/S/C capacities",
    ):
        ResidentReproductionExposureShapeV1(
            batch_cells=1,
            attempt_capacity=0,
            snapshot_capacity=1,
            eligible_capacity=2,
        )
    with pytest.raises(
        CudaResidentReproductionExposureError,
        match="explicit CUDA device",
    ):
        allocate_cuda_resident_reproduction_exposure_sidecar_v1(
            ResidentReproductionExposureShapeV1(2, 3, 2, 4),
            cell_enabled=(True, False),
            device="cpu",
        )


def test_existing_count_contract_rejects_internal_contradiction() -> None:
    with pytest.raises(
        CudaResidentReproductionExposureError,
        match="attempts must equal",
    ):
        ReproductionExposureExistingCountsV1(
            attempts=(2,),
            successes=(1,),
            failures=(0,),
        )


def test_segment_drain_copies_checkpoint_rows_into_resident_storage_directly() -> None:
    source = getsource(
        CudaResidentReproductionExposureSidecarV1.drain_segment
    )

    assert (
        "getattr(self, name).copy_(value, non_blocking=False)"
        in source
    )
    assert "value.to(device=self.device)" not in source


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_projection_derives_contributions_only_from_success_and_restores() -> None:
    shape = ResidentReproductionExposureShapeV1(2, 3, 2, 4)
    owner = allocate_cuda_resident_reproduction_exposure_sidecar_v1(
        shape,
        cell_enabled=(True, False),
        device="cuda",
    )
    owner.snapshot_paths[0, :2] = PATH_ORDINARY_LIVE_BIRTH
    owner.snapshot_generations[0, :2] = 7
    owner.snapshot_ticks[0, :2] = 11
    owner.snapshot_eligible_counts[0, :2] = 2
    owner.snapshot_eligible_parent_ids[0, :2, :2] = torch.tensor(
        ((101, 103), (102, 103)), dtype=torch.int64, device="cuda"
    )
    owner.snapshot_cursors[0] = 2

    owner.attempt_paths[0, :2] = PATH_ORDINARY_LIVE_BIRTH
    owner.attempt_generations[0, :2] = 7
    owner.attempt_ticks[0, :2] = 11
    owner.attempt_eligible_snapshot_refs[0, :2] = torch.tensor(
        ((0, 1), (0, 1)), dtype=torch.int64, device="cuda"
    )
    owner.attempt_eligible_counts[0, :2] = 2
    owner.attempt_selected_parent_ids[0, :2] = torch.tensor(
        ((101, 102), (101, 103)), dtype=torch.int64, device="cuda"
    )
    owner.attempt_success[0, 0] = True
    owner.attempt_failure_codes[0, :2] = torch.tensor(
        (0, 9), dtype=torch.int32, device="cuda"
    )
    owner.attempt_cursors[0] = 2
    owner.attempt_commit_ordinals[0, 0] = 0
    owner.attempt_child_agent_ids[0, 0] = 104
    owner.cell_status[0] = CELL_STATUS_OK
    owner.validate_exact()

    checkpoint = owner.export_checkpoint_projection()
    projected = project_resident_reproduction_exposure_v1(
        checkpoint,
        existing_counts=ReproductionExposureExistingCountsV1(
            attempts=(2, 0),
            successes=(1, 0),
            failures=(1, 0),
        ),
    )
    assert projected.cells[0].selected_parent_counts == {101: 2, 102: 1, 103: 1}
    assert projected.cells[0].parent_contributions == {101: 1, 102: 1}
    assert projected.attempts[0].primary_eligible_snapshot_ref == 0
    assert projected.attempts[0].mate_eligible_snapshot_ref == 1
    assert projected.cells[1].enabled is False
    assert projected.cells[1].attempt_count == 0

    restored = CudaResidentReproductionExposureSidecarV1.from_checkpoint_projection(
        checkpoint,
        device="cuda",
    )
    assert restored is not owner
    assert restored.export_checkpoint_projection()["projection_sha256"] == (
        checkpoint["projection_sha256"]
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_validation_rejects_disabled_activity_and_checkpoint_tampering() -> None:
    shape = ResidentReproductionExposureShapeV1(1, 2, 1, 2)
    owner = allocate_cuda_resident_reproduction_exposure_sidecar_v1(
        shape,
        cell_enabled=(False,),
        device="cuda",
    )
    owner.attempt_cursors[0] = 1
    with pytest.raises(
        CudaResidentReproductionExposureError,
        match="disabled cell",
    ):
        owner.validate_exact()
    owner.attempt_cursors[0] = 0

    checkpoint = owner.export_checkpoint_projection()
    damaged = deepcopy(checkpoint)
    damaged["tensors"]["snapshot_ticks"][0, 0] = 4
    with pytest.raises(
        CudaResidentReproductionExposureError,
        match="coordinates or counts",
    ):
        project_resident_reproduction_exposure_v1(damaged)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_validation_rejects_selected_parent_outside_referenced_snapshot() -> None:
    shape = ResidentReproductionExposureShapeV1(1, 1, 1, 2)
    owner = allocate_cuda_resident_reproduction_exposure_sidecar_v1(
        shape,
        cell_enabled=(True,),
        device="cuda",
    )
    owner.snapshot_paths[0, 0] = PATH_ORDINARY_LIVE_BIRTH
    owner.snapshot_generations[0, 0] = 0
    owner.snapshot_ticks[0, 0] = 1
    owner.snapshot_eligible_counts[0, 0] = 2
    owner.snapshot_eligible_parent_ids[0, 0] = torch.tensor(
        (3, 4), dtype=torch.int64, device="cuda"
    )
    owner.snapshot_cursors[0] = 1
    owner.attempt_paths[0, 0] = PATH_ORDINARY_LIVE_BIRTH
    owner.attempt_generations[0, 0] = 0
    owner.attempt_ticks[0, 0] = 1
    owner.attempt_eligible_snapshot_refs[0, 0] = 0
    owner.attempt_eligible_counts[0, 0] = 2
    owner.attempt_selected_parent_ids[0, 0] = torch.tensor(
        (3, 99), dtype=torch.int64, device="cuda"
    )
    owner.attempt_success[0, 0] = True
    owner.attempt_failure_codes[0, 0] = 0
    owner.attempt_commit_ordinals[0, 0] = 0
    owner.attempt_child_agent_ids[0, 0] = 100
    owner.attempt_cursors[0] = 1
    with pytest.raises(
        CudaResidentReproductionExposureError,
        match="role-specific eligible set",
    ):
        owner.validate_exact()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_segment_drain_reuses_capacity_and_matches_fresh_owner_exactly() -> None:
    shape = ResidentReproductionExposureShapeV1(1, 2, 4, 2)
    uninterrupted = allocate_cuda_resident_reproduction_exposure_sidecar_v1(
        shape,
        cell_enabled=(True,),
        device="cuda",
    )
    _append_success(
        uninterrupted,
        cell=0,
        generation=0,
        parent_0=11,
        parent_1=13,
    )
    resident_storage = {
        name: (id(value), value.data_ptr(), value.untyped_storage().data_ptr())
        for name, value in uninterrupted.tensor_map().items()
    }
    first = uninterrupted.drain_segment()
    assert first.segment_output["cells"][0]["attempt_count"] == 1
    assert int(uninterrupted.attempt_cursors[0].item()) == 0
    assert int(uninterrupted.snapshot_cursors[0].item()) == 0
    assert resident_storage == {
        name: (id(value), value.data_ptr(), value.untyped_storage().data_ptr())
        for name, value in uninterrupted.tensor_map().items()
    }

    fresh = CudaResidentReproductionExposureSidecarV1.from_checkpoint_projection(
        first.continuation_checkpoint,
        device="cuda",
    )
    for owner in (uninterrupted, fresh):
        _append_success(
            owner,
            cell=0,
            generation=1,
            parent_0=17,
            parent_1=19,
        )

    continued = uninterrupted.drain_segment()
    restored = fresh.drain_segment()

    assert continued.segment_output == restored.segment_output
    assert (
        continued.continuation_checkpoint["projection_sha256"]
        == restored.continuation_checkpoint["projection_sha256"]
    )
    continuity = continued.continuation_checkpoint["continuity"]
    assert continuity["segment_count"] == 2
    assert continuity["cumulative_attempt_counts"] == [2]
    assert continuity["cumulative_success_counts"] == [2]
    assert continuity["cumulative_snapshot_counts"] == [4]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_path_instrumentation_is_checkpointed_and_disabled_path_stays_zero() -> None:
    shape = ResidentReproductionExposureShapeV1(
        1, 2, 4, 2, registered_path_codes=(7, 19)
    )
    owner = allocate_cuda_resident_reproduction_exposure_sidecar_v1(
        shape,
        cell_enabled=(True,),
        device="cuda",
    )
    owner.path_rng_counter_states[0, 1] = 3
    owner.path_counter_write_counts[0, 1] = 1
    owner.path_event_row_write_counts[0, 1] = 2
    owner.path_rng_state_write_counts[0, 1] = 1

    drained = owner.drain_segment()
    rows = drained.segment_output["cells"][0]["path_instrumentation"]

    assert rows == [
        {
            "path_code": 7,
            "rng_counter_state": 0,
            "counter_write_count": 0,
            "event_row_write_count": 0,
            "rng_state_write_count": 0,
        },
        {
            "path_code": 19,
            "rng_counter_state": 3,
            "counter_write_count": 1,
            "event_row_write_count": 2,
            "rng_state_write_count": 1,
        },
    ]
    restored = CudaResidentReproductionExposureSidecarV1.from_checkpoint_projection(
        drained.continuation_checkpoint,
        device="cuda",
    )
    assert torch.equal(
        restored.path_rng_counter_states.cpu(), torch.tensor([[0, 3]])
    )
    assert torch.equal(
        restored.path_event_row_write_counts.cpu(), torch.tensor([[0, 2]])
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_segment_drain_keeps_overflow_sticky_and_other_cells_reusable() -> None:
    shape = ResidentReproductionExposureShapeV1(2, 1, 2, 2)
    owner = allocate_cuda_resident_reproduction_exposure_sidecar_v1(
        shape,
        cell_enabled=(True, True),
        device="cuda",
    )
    _append_success(
        owner,
        cell=0,
        generation=0,
        parent_0=101,
        parent_1=103,
    )
    _append_success(
        owner,
        cell=1,
        generation=0,
        parent_0=201,
        parent_1=203,
    )
    owner.cell_status[0] = CELL_STATUS_ATTEMPT_CAPACITY_EXCEEDED

    first = owner.drain_segment()

    assert int(owner.cell_status[0].item()) == (
        CELL_STATUS_ATTEMPT_CAPACITY_EXCEEDED
    )
    assert int(owner.cell_status[1].item()) == CELL_STATUS_OK
    assert torch.equal(owner.attempt_cursors.cpu(), torch.tensor((0, 0)))
    _append_success(
        owner,
        cell=1,
        generation=1,
        parent_0=211,
        parent_1=213,
    )
    second = owner.drain_segment()

    assert first.segment_output["cells"][0]["attempt_count"] == 1
    assert second.segment_output["cells"][0]["attempt_count"] == 0
    assert second.segment_output["cells"][1]["attempt_count"] == 1
    continuity = second.continuation_checkpoint["continuity"]
    assert continuity["cumulative_attempt_counts"] == [1, 2]
    assert continuity["cumulative_success_counts"] == [1, 2]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_compact_complete_projects_every_attempt_without_snapshot_rows() -> None:
    shape = ResidentReproductionExposureShapeV1(
        batch_cells=1,
        attempt_capacity=2,
        snapshot_capacity=1,
        eligible_capacity=4,
        storage_mode=STORAGE_MODE_COMPACT_COMPLETE,
        aggregate_capacity=3,
    )
    owner = allocate_cuda_resident_reproduction_exposure_sidecar_v1(
        shape,
        cell_enabled=(True,),
        device="cuda",
    )
    owner.attempt_paths[0, 0] = PATH_ORDINARY_LIVE_BIRTH
    owner.attempt_generations[0, 0] = 5
    owner.attempt_ticks[0, 0] = 17
    owner.attempt_eligible_counts[0, 0] = torch.tensor(
        (3, 2), dtype=torch.int64, device="cuda"
    )
    owner.attempt_selected_parent_ids[0, 0] = torch.tensor(
        (2_147_483_645, 2_147_483_646),
        dtype=torch.int64,
        device="cuda",
    )
    owner.attempt_success[0, 0] = True
    owner.attempt_failure_codes[0, 0] = 0
    owner.attempt_commit_ordinals[0, 0] = 0
    owner.attempt_child_agent_ids[0, 0] = 19
    owner.attempt_cursors[0] = 1
    owner.aggregate_agent_seen[0] = True
    owner.aggregate_agent_ids[0] = torch.tensor(
        (2_147_483_645, 2_147_483_646, 2_147_483_647),
        dtype=torch.int64,
        device="cuda",
    )
    owner.aggregate_primary_opportunity_counts[0] = 1
    owner.aggregate_mate_opportunity_counts[0, :2] = 1
    owner.aggregate_seen_counts[0] = 3

    checkpoint = owner.export_checkpoint_projection()
    projected = project_resident_reproduction_exposure_v1(checkpoint)

    assert projected.shape.storage_mode == STORAGE_MODE_COMPACT_COMPLETE
    assert projected.snapshots == ()
    assert projected.attempts[0].generation == 5
    assert projected.attempts[0].tick == 17
    assert projected.attempts[0].primary_eligible_count == 3
    assert projected.attempts[0].mate_eligible_count == 2
    assert projected.cells[0].primary_eligibility_opportunities == {
        2_147_483_645: 1,
        2_147_483_646: 1,
        2_147_483_647: 1,
    }
    assert projected.cells[0].mate_eligibility_opportunities == {
        2_147_483_645: 1,
        2_147_483_646: 1,
    }
    drained = owner.drain_segment()
    assert drained.segment_output["cells"][0][
        "primary_eligibility_opportunity_counts"
    ] == [
        {"agent_id": 2_147_483_645, "count": 1},
        {"agent_id": 2_147_483_646, "count": 1},
        {"agent_id": 2_147_483_647, "count": 1},
    ]
    assert "eligible_snapshot_refs" not in drained.segment_output["cells"][0][
        "attempts"
    ][0]
    assert int(owner.snapshot_cursors[0].item()) == 0
    assert int(owner.aggregate_seen_counts[0].item()) == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_qualification_attempt_capacity_is_default_off_and_sticky_after_drain() -> None:
    owner = allocate_cuda_resident_reproduction_exposure_sidecar_v1(
        ResidentReproductionExposureShapeV1(2, 4, 8, 4),
        cell_enabled=(True, True),
        device="cuda",
    )
    assert owner.qualification_attempt_capacity_enabled.tolist() == [False, False]
    assert owner.qualification_effective_attempt_capacity.tolist() == [4, 4]

    owner.arm_qualification_attempt_capacity_v1(
        target_cell_index=0,
        effective_capacity=0,
    )
    owner.cell_status[0] = CELL_STATUS_ATTEMPT_CAPACITY_EXCEEDED
    owner.drain_segment()

    assert owner.qualification_attempt_capacity_enabled.tolist() == [True, False]
    assert owner.qualification_effective_attempt_capacity.tolist() == [0, 4]
    assert owner.cell_status.tolist() == [
        CELL_STATUS_ATTEMPT_CAPACITY_EXCEEDED,
        CELL_STATUS_OK,
    ]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_failed_no_mate_attempt_allows_zero_mate_eligibility() -> None:
    shape = ResidentReproductionExposureShapeV1(1, 1, 2, 1)
    owner = allocate_cuda_resident_reproduction_exposure_sidecar_v1(
        shape,
        cell_enabled=(True,),
        device="cuda",
    )
    owner.snapshot_paths[0, :2] = PATH_ORDINARY_LIVE_BIRTH
    owner.snapshot_generations[0, :2] = 0
    owner.snapshot_ticks[0, :2] = 1
    owner.snapshot_eligible_counts[0] = torch.tensor(
        (1, 0), dtype=torch.int64, device="cuda"
    )
    owner.snapshot_eligible_parent_ids[0, 0, 0] = 3
    owner.snapshot_cursors[0] = 2
    owner.attempt_paths[0, 0] = PATH_ORDINARY_LIVE_BIRTH
    owner.attempt_generations[0, 0] = 0
    owner.attempt_ticks[0, 0] = 1
    owner.attempt_eligible_snapshot_refs[0, 0] = torch.tensor(
        (0, 1), dtype=torch.int64, device="cuda"
    )
    owner.attempt_eligible_counts[0, 0] = torch.tensor(
        (1, 0), dtype=torch.int64, device="cuda"
    )
    owner.attempt_selected_parent_ids[0, 0, 0] = 3
    owner.attempt_failure_codes[0, 0] = 9
    owner.attempt_cursors[0] = 1

    owner.validate_exact()
