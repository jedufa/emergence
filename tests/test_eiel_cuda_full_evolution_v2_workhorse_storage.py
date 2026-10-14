from __future__ import annotations

import pytest

from engine.campaigns.cuda_full_evolution_v2_workhorse_resources import (
    WorkhorseStorageAccountingError,
    workhorse_checkpoint_storage_accounting_v2,
)
from engine.perf.cuda_full_evolution_v2.workhorse import (
    BACKEND_FUSED_CTA_TEMPORAL_V2,
    FullEvolutionCudaWorkhorseContractV2,
)
from eiel_composition.workhorse_v2_materialize import (
    _apply_workhorse_storage_accounting_v2,
    _baseline,
)


def test_acs2_baseline_declares_only_actual_deviations() -> None:
    requirement_ids = {
        row["requirement_id"] for row in _baseline(acs_slots=2)["deviations"]
    }
    assert "checkpoint.acs_two_slot" not in requirement_ids
    assert "runtime.production_cuda" in requirement_ids


@pytest.mark.parametrize("acs_slots", (0, 1))
def test_non_acs2_baseline_declares_checkpoint_deviation(acs_slots: int) -> None:
    requirement_ids = {
        row["requirement_id"]
        for row in _baseline(acs_slots=acs_slots)["deviations"]
    }
    assert "checkpoint.acs_two_slot" in requirement_ids


@pytest.mark.parametrize(
    ("shape_id", "batch", "payload_bytes"),
    (
        ("w64h64p1024c2048", 1088, 416_956_416),
        ("w128h80p2560c5120", 2496, 2_390_409_216),
    ),
)
def test_large_route_two_checkpoint_acs2_accounting_is_payload_exact(
    shape_id: str, batch: int, payload_bytes: int
) -> None:
    contract = FullEvolutionCudaWorkhorseContractV2(
        shape_id=shape_id,
        backend_id=BACKEND_FUSED_CTA_TEMPORAL_V2,
        B=batch,
        temporal_k=64,
        total_ticks=50_000,
        start_next_tick=0,
    )
    accounting = workhorse_checkpoint_storage_accounting_v2(
        contract=contract,
        checkpoint_ticks=(25_000, 50_000),
        acs_checkpoint_slots=2,
        dispatch_count=784,
    )

    assert accounting["checkpoint_payload_bytes"] == payload_bytes
    assert accounting["checkpoint_count"] == 2
    assert accounting["physical_checkpoint_payload_copies"] == 2
    assert accounting["absolute_tick_projection_payload_copies"] == 2
    assert accounting["indexed_world_payload_copies"] == 0
    assert (
        accounting["checkpoint_durable_payload_bytes_upper_bound"]
        == 4 * payload_bytes
    )
    assert accounting["acs_pending_capture_files_upper_bound"] == 2
    assert (
        accounting["acs_pending_capture_file_bytes_upper_bound"]
        == 2 * payload_bytes
    )
    assert accounting["disk_bytes_upper_bound"] > 6 * payload_bytes
    assert accounting["horizon_dependent_resident_bytes"] == 0


def test_one_terminal_checkpoint_uses_only_one_of_two_scratch_slots() -> None:
    contract = FullEvolutionCudaWorkhorseContractV2(
        shape_id="w128h80p2560c5120",
        backend_id=BACKEND_FUSED_CTA_TEMPORAL_V2,
        B=2496,
        temporal_k=64,
        total_ticks=64,
        start_next_tick=100,
    )
    accounting = workhorse_checkpoint_storage_accounting_v2(
        contract=contract,
        checkpoint_ticks=(164,),
        acs_checkpoint_slots=2,
        dispatch_count=1,
    )
    assert accounting["checkpoint_count"] == 1
    assert accounting["acs_pending_capture_files_upper_bound"] == 1
    assert (
        accounting["acs_pending_capture_file_bytes_upper_bound"]
        == accounting["checkpoint_payload_bytes"]
    )

    spec = {
        "execution": {"budgets": {"disk_bytes": 1}},
        "artifact_lifecycle": {
            "storage": {
                "total_max_bytes": 1,
                "scratch_max_bytes": 1,
                "checkpoint_max_bytes": 1,
                "durable_max_bytes": 1,
                "emergency_reserve_bytes": 128 * 1024**2,
            }
        },
    }
    _apply_workhorse_storage_accounting_v2(spec, accounting)
    storage = spec["artifact_lifecycle"]["storage"]
    assert spec["execution"]["budgets"]["disk_bytes"] == storage[
        "total_max_bytes"
    ]
    assert storage["total_max_bytes"] == (
        accounting["acs_pending_capture_file_bytes_upper_bound"]
        + accounting["checkpoint_bytes_upper_bound"]
        + accounting["durable_noncheckpoint_bytes_upper_bound"]
        + storage["emergency_reserve_bytes"]
    )
    assert storage["scratch_max_bytes"] == accounting[
        "acs_pending_capture_file_bytes_upper_bound"
    ]
    assert storage["checkpoint_max_bytes"] == accounting[
        "checkpoint_bytes_upper_bound"
    ]
    assert storage["durable_max_bytes"] == accounting[
        "durable_noncheckpoint_bytes_upper_bound"
    ]


@pytest.mark.parametrize(
    "ticks",
    (
        (),
        (25_000,),
        (25_000, 25_000, 50_000),
        (0, 50_000),
    ),
)
def test_storage_accounting_rejects_inexact_checkpoint_schedule(
    ticks: tuple[int, ...]
) -> None:
    contract = FullEvolutionCudaWorkhorseContractV2(
        shape_id="w64h64p1024c2048",
        backend_id=BACKEND_FUSED_CTA_TEMPORAL_V2,
        B=1088,
        temporal_k=64,
        total_ticks=50_000,
    )
    with pytest.raises(WorkhorseStorageAccountingError, match="exact absolute"):
        workhorse_checkpoint_storage_accounting_v2(
            contract=contract,
            checkpoint_ticks=ticks,
            acs_checkpoint_slots=2,
            dispatch_count=784,
        )
