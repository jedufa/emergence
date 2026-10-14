from __future__ import annotations

import hashlib
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import pytest

from engine.campaigns.canonical import canonical_sha256
from engine.campaigns.resident_prepared_owner_resource_audit_v1 import (
    PREPARE_ONLY_STAGES,
    CudaMemorySampleV1,
    LIFETIME_ANALYSIS_SCHEMA_VERSION,
    ResidentPreparedOwnerAuditStageRecorderV1,
    ResidentPreparedOwnerResourceAuditRequestV1,
    inventory_resident_prepared_owner_storages_v1,
    serialize_preparation_projection_v1,
    validate_prepared_owner_resource_audit_receipt_v1,
)

_DIGEST = "a" * 64
_PROJECTION_DIGEST = canonical_sha256({"prepared": True})
_HARDWARE = {
    "device_index": 0,
    "device_name": "test-cuda",
    "device_uuid": "GPU-test",
    "compute_capability": [7, 5],
    "total_memory_bytes": 10_000,
    "cuda_runtime_version": "test-cuda-runtime",
    "torch_version": "test-torch",
}


class _Storage:
    def __init__(self, pointer: int, size: int) -> None:
        self._pointer = pointer
        self._size = size

    def data_ptr(self) -> int:
        return self._pointer

    def nbytes(self) -> int:
        return self._size


class _Tensor:
    def __init__(self, storage: _Storage, device: str = "cuda:0") -> None:
        self._storage = storage
        self.device = device

    def untyped_storage(self) -> _Storage:
        return self._storage


@dataclass
class _Owner:
    primary: _Tensor
    nested: object


def _request(**changes: object) -> ResidentPreparedOwnerResourceAuditRequestV1:
    values: dict[str, object] = {
        "audit_id": "audit-1",
        "provider_id": "provider-1",
        "profile_id": "profile-1",
        "campaign_id": "campaign-1",
        "plan_sha256": _DIGEST,
        "qualification_key_sha256": _DIGEST,
        "execution_route_sha256": _DIGEST,
        "route_qualification": {
            "authority_state": "route_qualified",
            "receipt_path": "/tmp/test-route-qualification.json",
            "receipt_sha256": _DIGEST,
            "artifact_conformance_exact": True,
        },
        "candidate_authority_sha256": _DIGEST,
        "artifact_sha256s": {"native_extension": _DIGEST},
        "configuration_sha256": _DIGEST,
        "resource_estimator_id": "test-estimator-v1",
        "conservative_vram_projection_bytes": 900,
        "runtime_assumptions": {},
        "measurement_authorization_sha256": _DIGEST,
        "hardware_identity": _HARDWARE,
        "source_projection_root_sha256": _PROJECTION_DIGEST,
        "ordered_cell_ids_sha256": _DIGEST,
        "batch_request_sha256": _DIGEST,
        "batch_cells": 72,
        "resident_segment_generations": 10,
        "physical_window_generations": 2,
        "ticks_per_generation": 80,
        "resident_capacity": 256,
        "initial_population": 128,
        "agent_id_allocation_generation_count": 131_072,
        "declared_vram_budget_bytes": 1_000,
        "minimum_physical_headroom_fraction": 0.10,
        "minimum_residual_headroom_bytes": 0,
    }
    values.update(changes)
    return ResidentPreparedOwnerResourceAuditRequestV1(**values)  # type: ignore[arg-type]


def _sample(
    stage: str,
    index: int,
    *,
    allocated: int = 100,
    reserved: int = 120,
    peak_allocated: int | None = None,
    peak_reserved: int | None = None,
    free: int = 8_000,
) -> CudaMemorySampleV1:
    return CudaMemorySampleV1(
        stage=stage,
        elapsed_ns=index * 10,
        allocated_bytes=allocated,
        reserved_bytes=reserved,
        peak_allocated_bytes=allocated if peak_allocated is None else peak_allocated,
        peak_reserved_bytes=reserved if peak_reserved is None else peak_reserved,
        free_bytes=free,
        total_bytes=10_000,
    )


def _complete_recorder(
    tmp_path: Path,
    *,
    request: ResidentPreparedOwnerResourceAuditRequestV1 | None = None,
    peak_allocated: int = 700,
    peak_reserved: int = 800,
    minimum_free: int = 2_000,
    teardown_allocated: int = 100,
    teardown_reserved: int = 120,
    teardown_free: int = 8_000,
) -> ResidentPreparedOwnerAuditStageRecorderV1:
    recorder = ResidentPreparedOwnerAuditStageRecorderV1(request or _request())
    for index, stage in enumerate(PREPARE_ONLY_STAGES):
        if stage == "process_baseline":
            sample = _sample(stage, index)
        elif stage == "teardown":
            sample = _sample(
                stage,
                index,
                allocated=teardown_allocated,
                reserved=teardown_reserved,
                free=teardown_free,
            )
        else:
            sample = _sample(
                stage,
                index,
                allocated=min(peak_allocated, 500),
                reserved=min(peak_reserved, 600),
                peak_allocated=peak_allocated,
                peak_reserved=peak_reserved,
                free=minimum_free,
            )
        recorder.record_sample(sample)
    recorder.record_storage_inventory(
        inventory_resident_prepared_owner_storages_v1(
            {"owner": _Tensor(_Storage(7, 500))}
        )
    )
    clocks = iter((100, 145))
    serialization = serialize_preparation_projection_v1(
        {"prepared": True},
        projection_sha256=_PROJECTION_DIGEST,
        destination=tmp_path / "projection.bin",
        format_id="test-json-v1",
        serializer=lambda _value, path: path.write_bytes(b"projection"),
        clock_ns=lambda: next(clocks),
    )
    recorder.record_preparation_serialization(serialization)
    return recorder


def test_inventory_dealiases_views_and_handles_cycles() -> None:
    shared = _Storage(17, 400)
    second = _Storage(29, 250)
    cycle: list[object] = []
    cycle.append(cycle)
    owner = _Owner(
        primary=_Tensor(shared),
        nested={
            "aliases": [_Tensor(shared), _Tensor(second, "cpu")],
            "cycle": cycle,
        },
    )

    inventory = inventory_resident_prepared_owner_storages_v1(owner)

    assert inventory.tensor_view_count == 3
    assert inventory.unique_storage_count == 2
    assert inventory.retained_storage_bytes == 650
    assert inventory.device_storage_bytes == {"cpu": 250, "cuda:0": 400}
    assert inventory.storage_rows[0]["view_count"] == 2
    assert inventory.storage_rows[0]["nbytes"] == 400


def test_inventory_omits_zero_byte_tensor_views() -> None:
    owner = {
        "empty": _Tensor(_Storage(0, 0), "cpu"),
        "retained": _Tensor(_Storage(17, 400), "cuda:0"),
    }

    inventory = inventory_resident_prepared_owner_storages_v1(owner)

    assert inventory.tensor_view_count == 1
    assert inventory.unique_storage_count == 1
    assert inventory.retained_storage_bytes == 400
    assert inventory.device_storage_bytes == {"cuda:0": 400}
    assert inventory.storage_rows == (
        {
            "alias_group_id": "cuda:0:11",
            "data_ptr": 17,
            "storage_data_ptr": 17,
            "device": "cuda:0",
            "nbytes": 400,
            "storage_nbytes": 400,
            "dtype": "unknown",
            "shape": None,
            "encoding": "dense_tensor_storage_v1",
            "paths": ["$['retained']"],
            "view_count": 1,
            "logical_bytes_total": 0,
        },
    )
    assert inventory.tensor_view_rows[0]["path"] == "$['retained']"
    assert inventory.tensor_view_rows[0]["alias_group_id"] == "cuda:0:11"
    assert inventory.tensor_view_rows[0]["storage_nbytes"] == 400


def test_serialization_binds_bytes_hash_and_elapsed_time(tmp_path: Path) -> None:
    clocks = iter((11, 41))
    destination = tmp_path / "projection.bin"

    result = serialize_preparation_projection_v1(
        {"value": 1},
        projection_sha256=_DIGEST,
        destination=destination,
        format_id="fake-v1",
        serializer=lambda _value, path: path.write_bytes(b"abc"),
        clock_ns=lambda: next(clocks),
    )

    assert result.serialized_bytes == 3
    assert result.serialized_sha256 == hashlib.sha256(b"abc").hexdigest()
    assert result.serialization_time_ns == 30
    with pytest.raises(FileExistsError):
        serialize_preparation_projection_v1(
            {},
            projection_sha256=_DIGEST,
            destination=destination,
            format_id="fake-v1",
            serializer=lambda _value, path: path.write_bytes(b"replacement"),
        )


def test_receipt_proves_prepare_only_budget_headroom_and_teardown(
    tmp_path: Path,
) -> None:
    recorder = _complete_recorder(tmp_path)

    receipt = recorder.finalize()
    payload = receipt.payload()

    assert payload["request_sha256"] == receipt.request.request_sha256
    decision = dict(payload["decision"])
    lifetime = decision.pop("lifetime_analysis")
    assert lifetime["schema_version"] == LIFETIME_ANALYSIS_SCHEMA_VERSION
    assert lifetime["peak_reserved_stage"] == "serialized_projection"
    assert lifetime["peak_process_delta_bytes"] == 680
    assert lifetime["retained_cuda_storage_bytes"] == 500
    assert lifetime["cuda_retained_to_peak_gap_bytes"] == 180
    assert lifetime["cuda_retained_to_peak_gap_bytes_per_cell_floor"] == 2
    assert lifetime["device_storage_bytes"] == {"cuda:0": 500}
    assert lifetime["top_retained_storage_rows"][0]["nbytes"] == 500
    assert len(lifetime["stage_deltas"]) == len(PREPARE_ONLY_STAGES)
    assert decision == {
        "measured_owner_peak_bytes": 680,
        "retained_owner_storage_bytes": 500,
        "retained_cuda_storage_bytes": 500,
        "peak_process_allocated_delta_bytes": 600,
        "peak_process_reserved_delta_bytes": 680,
        "minimum_observed_free_bytes": 2_000,
        "required_physical_headroom_bytes": 1_000,
        "residual_headroom_bytes": 1_000,
        "minimum_residual_headroom_bytes": 0,
        "declared_vram_budget_bytes": 1_000,
        "within_declared_budget": True,
        "conservative_vram_projection_bytes": 900,
        "within_conservative_projection": True,
        "physical_headroom_ok": True,
        "residual_headroom_ok": True,
        "teardown_to_baseline": {
            "allocated_delta_bytes": 0,
            "reserved_delta_bytes": 0,
            "free_delta_bytes": 0,
            "tolerance_bytes": 0,
            "allocator_cache_non_authoritative": True,
            "ok": True,
        },
        "prepare_only_evidence_scope": (
            "structural_zero_dispatch_zero_tick_counter_v1"
        ),
        "prepare_only": True,
        "ok": True,
    }
    assert payload["executor_call_count"] == 0
    assert payload["scientific_tick_count"] == 0
    assert payload["receipt_sha256"] == canonical_sha256(receipt.unsigned_payload())


@pytest.mark.parametrize(
    ("changes", "recorder_changes", "failed_field"),
    [
        (
            {"conservative_vram_projection_bytes": 600},
            {},
            "within_conservative_projection",
        ),
        ({}, {"minimum_free": 999}, "physical_headroom_ok"),
        (
            {"minimum_residual_headroom_bytes": 1_001},
            {},
            "residual_headroom_ok",
        ),
        (
            {},
            {"teardown_allocated": 101, "teardown_free": 7_999},
            "teardown_to_baseline",
        ),
    ],
)
def test_decision_fails_closed_for_each_resource_obligation(
    tmp_path: Path,
    changes: dict[str, object],
    recorder_changes: dict[str, int],
    failed_field: str,
) -> None:
    request = _request(**changes)
    recorder = _complete_recorder(
        tmp_path,
        request=request,
        **recorder_changes,
    )

    decision = recorder.finalize().decision

    assert decision["ok"] is False
    if failed_field == "teardown_to_baseline":
        assert decision[failed_field]["ok"] is False
    else:
        assert decision[failed_field] is False


def test_recorder_rejects_out_of_order_and_non_prepare_only_work(
    tmp_path: Path,
) -> None:
    recorder = ResidentPreparedOwnerAuditStageRecorderV1(_request())
    with pytest.raises(ValueError, match="expected stage"):
        recorder.record_sample(_sample("prepared_owner", 0))

    recorder = _complete_recorder(tmp_path)
    recorder.record_executor_call()
    recorder.record_scientific_ticks(1)
    with pytest.raises(ValueError, match="counters record"):
        recorder.finalize()


def test_request_rejects_unbound_or_invalid_shapes() -> None:
    with pytest.raises(ValueError, match="artifact_sha256s"):
        _request(artifact_sha256s={})
    with pytest.raises(ValueError, match="initial_population"):
        _request(initial_population=257)
    request = _request(
        conservative_vram_projection_bytes=1_001,
        declared_vram_budget_bytes=1_000,
    )
    assert request.conservative_vram_projection_bytes == 1_001
    assert request.declared_vram_budget_bytes == 1_000
    with pytest.raises(ValueError, match="headroom_fraction"):
        _request(minimum_physical_headroom_fraction=1.0)


def test_receipt_validator_requires_exact_bindings_and_hardware(
    tmp_path: Path,
) -> None:
    receipt = _complete_recorder(tmp_path).finalize().payload()
    request = receipt["request"]
    expected = {
        key: request[key]
        for key in (
            "provider_id",
            "profile_id",
            "campaign_id",
            "plan_sha256",
            "qualification_key_sha256",
            "execution_route_sha256",
            "route_qualification",
            "artifact_sha256s",
            "configuration_sha256",
            "resource_estimator_id",
            "conservative_vram_projection_bytes",
            "runtime_assumptions",
            "measurement_authorization_sha256",
            "hardware_identity",
            "ordered_cell_ids_sha256",
            "batch_request_sha256",
            "dimensions",
            "policy",
        )
    }

    promoted = validate_prepared_owner_resource_audit_receipt_v1(
        receipt,
        expected_request_bindings=expected,
        live_hardware_identity=_HARDWARE,
    )

    assert promoted["vram_bytes_upper_bound"] == 900
    assert promoted["measured_owner_peak_bytes"] == 680
    with pytest.raises(ValueError, match="hardware identity"):
        validate_prepared_owner_resource_audit_receipt_v1(
            receipt,
            expected_request_bindings=expected,
            live_hardware_identity={**_HARDWARE, "device_uuid": "GPU-other"},
        )

    empty_inventory = deepcopy(receipt)
    empty_inventory["storage_inventory"] = {}
    empty_inventory["receipt_sha256"] = canonical_sha256(
        {
            key: value
            for key, value in empty_inventory.items()
            if key != "receipt_sha256"
        }
    )
    with pytest.raises(ValueError, match="storage inventory"):
        validate_prepared_owner_resource_audit_receipt_v1(
            empty_inventory,
            expected_request_bindings=expected,
            live_hardware_identity=_HARDWARE,
        )

    inconsistent_inventory = deepcopy(receipt)
    inconsistent_inventory["storage_inventory"]["storage_rows"][0][
        "nbytes"
    ] += 1
    inconsistent_inventory["receipt_sha256"] = canonical_sha256(
        {
            key: value
            for key, value in inconsistent_inventory.items()
            if key != "receipt_sha256"
        }
    )
    with pytest.raises(ValueError, match="storage row|storage inventory"):
        validate_prepared_owner_resource_audit_receipt_v1(
            inconsistent_inventory,
            expected_request_bindings=expected,
            live_hardware_identity=_HARDWARE,
        )

    tampered_sample = deepcopy(receipt)
    tampered_sample["stage_samples"][2]["peak_reserved_bytes"] += 1
    tampered_sample["receipt_sha256"] = canonical_sha256(
        {
            key: value
            for key, value in tampered_sample.items()
            if key != "receipt_sha256"
        }
    )
    with pytest.raises(ValueError, match="decision differs"):
        validate_prepared_owner_resource_audit_receipt_v1(
            tampered_sample,
            expected_request_bindings=expected,
            live_hardware_identity=_HARDWARE,
        )

    tampered_decision = deepcopy(receipt)
    tampered_decision["decision"]["measured_owner_peak_bytes"] += 1
    tampered_decision["receipt_sha256"] = canonical_sha256(
        {
            key: value
            for key, value in tampered_decision.items()
            if key != "receipt_sha256"
        }
    )
    with pytest.raises(ValueError, match="decision differs"):
        validate_prepared_owner_resource_audit_receipt_v1(
            tampered_decision,
            expected_request_bindings=expected,
            live_hardware_identity=_HARDWARE,
        )

    Path(
        receipt["preparation_serialization"]["serialized_path"]
    ).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="serialization"):
        validate_prepared_owner_resource_audit_receipt_v1(
            receipt,
            expected_request_bindings=expected,
            live_hardware_identity=_HARDWARE,
        )
