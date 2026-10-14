from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import gc
import hashlib
import json
import os
from pathlib import Path
import struct

import pytest
import torch

import engine.campaigns.cuda_full_evolution_v2_acs_adapter as adapter
import engine.perf.cuda_full_evolution_v2.workhorse.authority as authority
import engine.perf.cuda_full_evolution_v2.workhorse.layout as workhorse_layout
from engine.campaigns.acs_native_detached_slots_v1 import (
    ACSNativeDetachedSlotError,
    ACSNativeDetachedSlotsV1,
    validate_detached_capture_host_only_v1,
)
from engine.perf.cuda_full_evolution_v2.workhorse import (
    BACKEND_FUSED_CTA_TEMPORAL_V2,
    SUPPORTED_WORKHORSE_SHAPES_V2,
    FullEvolutionCudaWorkhorseContractV2,
)
from engine.perf.cuda_full_evolution_v2.workhorse.layout import workhorse_layout_v2
from engine.perf.cuda_full_evolution_v2.contract import (
    CudaFullEvolutionNativeContractV2,
)
from engine.perf.cuda_full_evolution_v2.layout import state_layout_v2
from engine.perf.cuda_full_evolution_v2.runtime import (
    CudaFullEvolutionCheckpointCaptureSourceV1,
    CudaFullEvolutionCheckpointV2,
)


def _contract(
    *, B: int = 2, total_ticks: int = 20, shape_index: int = 0
) -> FullEvolutionCudaWorkhorseContractV2:
    return FullEvolutionCudaWorkhorseContractV2(
        shape_id=SUPPORTED_WORKHORSE_SHAPES_V2[shape_index].shape_id,
        backend_id=BACKEND_FUSED_CTA_TEMPORAL_V2,
        B=B,
        temporal_k=1,
        total_ticks=total_ticks,
    )


def _state(contract: FullEvolutionCudaWorkhorseContractV2, *, tick: int) -> torch.Tensor:
    row_bytes = workhorse_layout_v2(contract.shape).state_bytes_per_world
    raw = bytearray(contract.B * row_bytes)
    for world in range(contract.B):
        header = [0] * 32
        header[0] = 1000 + world
        header[1] = tick
        header[4] = 0
        header[13] = 2000 + world
        header[14] = 4
        struct.pack_into("<32Q", raw, world * row_bytes, *header)
    return torch.frombuffer(raw, dtype=torch.uint8).clone().reshape(contract.B, row_bytes)


def _fake_metadata(**values) -> dict:
    headers = values["state_headers"]
    tick = values["checkpoint_at_tick"]
    assert all(header[1] == tick and header[4] == 0 and header[14] == 4 for header in headers)
    body = {
        "schema_version": "eiel_full_evolution_cuda_workhorse_checkpoint_bundle_v2",
        "time_authority": "absolute_tick_v1",
        "contract": values["contract"].to_dict(),
        "checkpoint_at_tick": tick,
        "checkpoint_identity": {"checkpoint_identity_sha256": "9" * 64},
        "canonical_state": {
            "file_name": "canonical_state.bin",
            "bytes": values["state_nbytes"],
            "sha256": values["state_sha256"],
        },
        "world_uids": [int(header[0]) for header in headers],
        "rng_keys": [int(header[13]) for header in headers],
    }
    return {
        **body,
        "bundle_body_sha256": hashlib.sha256(
            json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
    }


def _capture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    ordinal: int,
    tick: int,
) -> tuple[dict, torch.Tensor]:
    monkeypatch.setattr(
        adapter,
        "workhorse_checkpoint_metadata_from_state_identity_v2",
        _fake_metadata,
    )
    monkeypatch.setattr(
        authority,
        "workhorse_checkpoint_metadata_from_state_identity_v2",
        _fake_metadata,
    )

    contract = _contract(total_ticks=max(20, tick))
    state = _state(contract, tick=tick)
    artifact_path = tmp_path / "artifact.so"
    if not artifact_path.exists():
        artifact_path.write_bytes(b"workhorse-artifact")
    event_path = tmp_path / f"event-{ordinal}.json"
    event_path.write_text("{}\n", encoding="ascii")
    capture = adapter.capture_workhorse_checkpoint_to_file_v1(
        state,
        tmp_path / f"checkpoint-{ordinal}",
        contract=contract,
        checkpoint_at_tick=tick,
        artifact_path=artifact_path,
        event_sink_receipt_path=event_path,
        attempt_authority_binding={"attempt": f"attempt-{ordinal}"},
        chunk_bytes=64 * 1024,
        require_native_pool=False,
    )
    return capture, state


def test_workhorse_capture_is_host_only_and_two_slot_fifo_is_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first_capture, first_state = _capture(
        tmp_path, monkeypatch, ordinal=1, tick=10
    )
    second_capture, second_state = _capture(
        tmp_path, monkeypatch, ordinal=2, tick=20
    )
    for capture in (first_capture, second_capture):
        validation = validate_detached_capture_host_only_v1(capture)
        assert validation["tensor_free"] is True
        assert validation["callable_free"] is True

    pipeline = ACSNativeDetachedSlotsV1(slots=2)
    first = pipeline.begin_capture(compute_frontier=10)
    second = pipeline.begin_capture(compute_frontier=20)
    pipeline.submit_capture(first, first_capture)
    pipeline.submit_capture(second, second_capture)
    assert pipeline.telemetry()["queue_occupancy_high_water"] == 2

    first_result = pipeline.result(first)
    first_path = Path(first_result["path"])
    assert (first_path / "canonical_state.bin").read_bytes() == bytes(
        first_state.numpy()
    )
    assert first_result["worker_full_state_bytes_materialized"] == 0
    pipeline.acknowledge_durable(first)

    second_result = pipeline.result(second)
    second_path = Path(second_result["path"])
    assert (second_path / "canonical_state.bin").read_bytes() == bytes(
        second_state.numpy()
    )
    pipeline.acknowledge_durable(second)
    pipeline.close()

    telemetry = pipeline.telemetry()
    assert telemetry["slot_states"] == ["FREE", "FREE"]
    assert telemetry["durable_checkpoint_frontier"] == 20
    assert telemetry["queued_cuda_references"] == 0
    assert telemetry["queued_cuda_callbacks"] == 0
    assert telemetry["capture_pool_count"] == 1
    assert telemetry["capture_pool_model"] == "one_shared_two_slab_pool_not_per_slot"


def test_workhorse_capture_rejects_changed_file_identity_and_quarantines(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    capture, _state_tensor = _capture(
        tmp_path, monkeypatch, ordinal=1, tick=10
    )
    capture_path = Path(capture["capture_path"])
    capture_path.chmod(0o600)
    with capture_path.open("ab") as handle:
        handle.write(b"tamper")

    pipeline = ACSNativeDetachedSlotsV1(slots=1)
    token = pipeline.begin_capture(compute_frontier=10)
    pipeline.submit_capture(token, capture)
    with pytest.raises(ACSNativeDetachedSlotError, match="worker failed"):
        pipeline.result(token)
    telemetry = pipeline.telemetry()
    assert telemetry["failure_latched"] is True
    assert telemetry["queued_cuda_references"] == 0
    assert telemetry["failure_capture_paths"]
    pipeline.abort()


def test_workhorse_postpublication_readback_failure_quarantines_canonical_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    capture, _state_tensor = _capture(
        tmp_path, monkeypatch, ordinal=1, tick=10
    )
    destination = Path(capture["destination"])
    original_read = adapter._read_small_file

    def corrupt_metadata(path: Path) -> bytes:
        if path.name == "checkpoint.json":
            return b"corrupt"
        return original_read(path)

    monkeypatch.setattr(adapter, "_read_small_file", corrupt_metadata)
    with pytest.raises(
        adapter.CudaFullEvolutionV2ACSError,
        match="publication was quarantined",
    ) as raised:
        adapter.finalize_workhorse_checkpoint_capture_v1(capture)
    assert not destination.exists()
    failure_path = Path(raised.value.publication_failure_path)
    assert failure_path.is_dir()
    assert failure_path.name.startswith(f"{destination.name}.publication.failed.")
    assert Path(capture["capture_path"]).is_file()


def test_workhorse_capture_descriptor_rejects_tensor_or_callback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    capture, _state_tensor = _capture(
        tmp_path, monkeypatch, ordinal=1, tick=10
    )
    with pytest.raises(ACSNativeDetachedSlotError, match="device object"):
        validate_detached_capture_host_only_v1({**capture, "owner": torch.zeros(1)})
    with pytest.raises(ACSNativeDetachedSlotError, match="executable state"):
        validate_detached_capture_host_only_v1({**capture, "finalize": lambda: None})


@pytest.mark.parametrize("shape_index", [0, 1])
def test_workhorse_coordinator_executes_w64_and_w128_with_one_reused_pool(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    shape_index: int,
) -> None:
    monkeypatch.setattr(
        adapter,
        "workhorse_checkpoint_metadata_from_state_identity_v2",
        _fake_metadata,
    )
    contract = _contract(B=2, total_ticks=30, shape_index=shape_index)
    artifact = tmp_path / "artifact.so"
    artifact.write_bytes(b"workhorse-artifact")
    first_event = tmp_path / "event-10.json"
    second_event = tmp_path / "event-20.json"
    third_event = tmp_path / "event-30.json"
    first_event.write_text("{}\n", encoding="ascii")
    second_event.write_text("{}\n", encoding="ascii")
    third_event.write_text("{}\n", encoding="ascii")
    coordinator = adapter.CudaFullEvolutionV2WorkhorseACSCoordinatorV1(
        slots=2,
        checkpoint_root=tmp_path / "checkpoints",
        artifact_path=artifact,
        attempt_authority_binding={"attempt": "fixture"},
        chunk_bytes=64 * 1024,
        require_native_pool=False,
    )
    first_state = _state(contract, tick=10)
    second_state = _state(contract, tick=20)
    third_state = _state(contract, tick=30)
    first = coordinator.capture_workhorse_checkpoint(
        first_state,
        contract=contract,
        checkpoint_at_tick=10,
        event_sink_receipt_path=first_event,
    )
    second = coordinator.capture_workhorse_checkpoint(
        second_state,
        contract=contract,
        checkpoint_at_tick=20,
        event_sink_receipt_path=second_event,
    )
    first_result = coordinator.settle_checkpoint(first)
    assert coordinator.telemetry()["checkpoint_receipts"] == []
    with pytest.raises(
        adapter.CudaFullEvolutionV2ACSError,
        match="external commit/acknowledgement",
    ):
        coordinator.settle_checkpoint(second)
    with pytest.raises(
        adapter.CudaFullEvolutionV2ACSError,
        match="capacity requires the FIFO head",
    ):
        coordinator.capture_workhorse_checkpoint(
            third_state,
            contract=contract,
            checkpoint_at_tick=30,
            event_sink_receipt_path=third_event,
        )
    assert not (tmp_path / "checkpoints/checkpoint-next-tick-000000000030").exists()
    with pytest.raises(
        adapter.CudaFullEvolutionV2ACSError,
        match="external commit and acknowledgement",
    ):
        coordinator.close()
    coordinator.acknowledge_checkpoint(first)
    third = coordinator.capture_workhorse_checkpoint(
        third_state,
        contract=contract,
        checkpoint_at_tick=30,
        event_sink_receipt_path=third_event,
    )
    second_result = coordinator.settle_checkpoint(second)
    coordinator.acknowledge_checkpoint(second)
    third_result = coordinator.settle_checkpoint(third)
    coordinator.acknowledge_checkpoint(third)
    coordinator.close()

    assert Path(first_result["state_path"]).read_bytes() == bytes(
        first_state.numpy()
    )
    assert Path(second_result["state_path"]).read_bytes() == bytes(
        second_state.numpy()
    )
    assert Path(third_result["state_path"]).read_bytes() == bytes(
        third_state.numpy()
    )
    second_metadata = json.loads(
        Path(second_result["manifest_path"]).read_text(encoding="ascii")
    )
    monkeypatch.setattr(
        adapter,
        "inspect_workhorse_checkpoint_bundle_v2",
        lambda *_args, **_kwargs: (
            second_metadata,
            Path(second_result["state_path"]),
        ),
    )
    selected = adapter.reopen_workhorse_checkpoint_world_v1(
        second_result["manifest_path"], world_index=1
    )
    row_bytes = workhorse_layout_v2(contract.shape).state_bytes_per_world
    assert selected["selected_contract"]["B"] == 1
    assert selected["selected_world_uid"] == 1001
    assert selected["selected_rng_key"] == 2001
    assert selected["canonical_state"] == bytes(second_state.numpy())[row_bytes:]
    telemetry = coordinator.telemetry()
    assert telemetry["queue_occupancy_high_water"] == 2
    assert telemetry["capture_pool"]["route_capture_count"] == 3
    assert telemetry["capture_pool"]["slab_count"] == 2
    assert telemetry["queued_cuda_references"] == 0
    assert telemetry["writer_alive"] is False


@pytest.mark.parametrize(
    ("artifact_environment", "expected_shape_id"),
    (
        (
            "EIEL_FULL_EVOLUTION_WORKHORSE_V2_W64_PREBUILT",
            "w64h64p1024c2048",
        ),
        (
            "EIEL_FULL_EVOLUTION_WORKHORSE_V2_W128_PREBUILT",
            "w128h80p2560c5120",
        ),
    ),
)
def test_workhorse_physical_two_slot_capture_releases_owner_before_fresh_restore(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    artifact_environment: str,
    expected_shape_id: str,
) -> None:
    """Physical transport sentinel; scheduler/semantic authority is tested separately."""

    artifact_value = os.environ.get(artifact_environment)
    if not artifact_value or not torch.cuda.is_available():
        pytest.skip(f"requires CUDA and {artifact_environment}")

    from engine.perf.cuda_full_evolution_v2.workhorse.probe import (
        _allocate_owner,
        _initialize_owner,
        _load,
        _tensor_bytes,
        _torch,
    )

    physical_torch = _torch()
    artifact = Path(artifact_value).expanduser().resolve()
    loaded = _load(artifact, physical_torch)
    assert loaded.shape.shape_id == expected_shape_id
    assert loaded.backend.backend_id == BACKEND_FUSED_CTA_TEMPORAL_V2
    contract = FullEvolutionCudaWorkhorseContractV2(
        shape_id=loaded.shape.shape_id,
        backend_id=loaded.backend.backend_id,
        B=2,
        temporal_k=1,
        total_ticks=20,
    )
    monkeypatch.setattr(
        adapter,
        "workhorse_checkpoint_metadata_from_state_identity_v2",
        _fake_metadata,
    )
    monkeypatch.setattr(
        authority,
        "workhorse_checkpoint_metadata_from_state_identity_v2",
        _fake_metadata,
    )

    def reopen_transport_fixture(
        metadata_path: str | Path, **_kwargs: object
    ) -> tuple[dict, bytes]:
        path = Path(metadata_path)
        return (
            json.loads(path.read_text(encoding="ascii")),
            (path.parent / "canonical_state.bin").read_bytes(),
        )

    monkeypatch.setattr(
        authority,
        "reopen_workhorse_checkpoint_bundle_v2",
        reopen_transport_fixture,
    )
    event_receipt = tmp_path / "transport-only-event-receipt.json"
    event_receipt.write_text("{}\n", encoding="ascii")
    coordinator = adapter.CudaFullEvolutionV2WorkhorseACSCoordinatorV1(
        slots=2,
        checkpoint_root=tmp_path / "checkpoints",
        artifact_path=artifact,
        attempt_authority_binding={"technical_transport_sentinel": True},
        chunk_bytes=64 * 1024,
        require_native_pool=True,
    )

    owner = _allocate_owner(physical_torch, loaded, contract.B, contract.temporal_k)
    _initialize_owner(
        loaded,
        owner,
        world_uid_base=810_000,
        start_tick=19,
        rng_key=901,
        fixture=False,
    )
    expected_states = [_tensor_bytes(owner.state)]
    synchronous_paths = [
        authority.write_workhorse_checkpoint_bundle_v2(
            tmp_path / "synchronous-checkpoint-19",
            contract=contract,
            checkpoint_at_tick=19,
            state_bytes=expected_states[-1],
            artifact_path=artifact,
            event_sink_receipt_path=event_receipt,
            attempt_authority_binding={"technical_transport_sentinel": True},
        )
    ]
    tokens = [coordinator.capture_workhorse_checkpoint(
        owner.state,
        contract=contract,
        checkpoint_at_tick=19,
        event_sink_receipt_path=event_receipt,
    )]
    _initialize_owner(
        loaded,
        owner,
        world_uid_base=810_000,
        start_tick=20,
        rng_key=901,
        fixture=False,
    )
    expected_states.append(_tensor_bytes(owner.state))
    synchronous_paths.append(
        authority.write_workhorse_checkpoint_bundle_v2(
            tmp_path / "synchronous-checkpoint-20",
            contract=contract,
            checkpoint_at_tick=20,
            state_bytes=expected_states[-1],
            artifact_path=artifact,
            event_sink_receipt_path=event_receipt,
            attempt_authority_binding={"technical_transport_sentinel": True},
        )
    )
    expected_scratch = _tensor_bytes(owner.scratch)
    tokens.append(coordinator.capture_workhorse_checkpoint(
        owner.state,
        contract=contract,
        checkpoint_at_tick=20,
        event_sink_receipt_path=event_receipt,
    ))
    assert coordinator.telemetry()["queue_occupancy_high_water"] == 2

    del owner
    gc.collect()
    physical_torch.cuda.empty_cache()
    physical_torch.cuda.synchronize()
    for token in tokens:
        coordinator.settle_checkpoint(token)
        coordinator.acknowledge_checkpoint(token)
    coordinator.close()
    telemetry = coordinator.telemetry()
    assert telemetry["queued_cuda_references"] == 0
    assert telemetry["writer_alive"] is False
    assert telemetry["capture_pool"]["native_pool"] is True
    assert telemetry["capture_pool"]["pinned"] is True
    assert telemetry["capture_pool"]["route_capture_count"] == 2
    assert len(telemetry["checkpoint_receipts"]) == 2
    for receipt, expected_state, synchronous_path in zip(
        telemetry["checkpoint_receipts"],
        expected_states,
        synchronous_paths,
        strict=True,
    ):
        assert Path(receipt["state_path"]).read_bytes() == expected_state
        synchronous_metadata, synchronous_state = (
            authority.reopen_workhorse_checkpoint_bundle_v2(synchronous_path)
        )
        detached_metadata, detached_state = (
            authority.reopen_workhorse_checkpoint_bundle_v2(
                receipt["manifest_path"]
            )
        )
        assert detached_metadata == synchronous_metadata
        assert detached_state == synchronous_state == expected_state
        assert receipt["worker_full_state_bytes_materialized"] == 0
        assert receipt["worker_cuda_references"] == 0

    fresh = _allocate_owner(physical_torch, loaded, contract.B, contract.temporal_k)
    restored_cpu = physical_torch.frombuffer(
        bytearray(expected_states[-1]), dtype=physical_torch.uint8
    ).reshape(tuple(fresh.state.shape))
    fresh.state.copy_(restored_cpu)
    loaded.module.restore_derived(fresh.state, fresh.scratch)
    physical_torch.cuda.synchronize()
    assert _tensor_bytes(fresh.state) == expected_states[-1]
    assert _tensor_bytes(fresh.scratch) == expected_scratch
    del fresh
    gc.collect()
    physical_torch.cuda.empty_cache()
    physical_torch.cuda.synchronize()


def test_shared_metadata_law_binds_artifact_ack_attempt_world_and_rng(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = _contract(B=2, total_ticks=10)
    artifact = tmp_path / "workhorse.so"
    artifact.write_bytes(b"artifact")
    artifact_sha = hashlib.sha256(b"artifact").hexdigest()
    manifest = {
        "native_shape": {
                "state_bytes_per_world": workhorse_layout_v2(
                    contract.shape
                ).state_bytes_per_world,
        },
        "identity": {"identity_sha256": "1" * 64},
        "sealed_artifact_identity": {
            "sealed_artifact_identity_sha256": "2" * 64,
        },
        "manifest_sha256": "3" * 64,
    }
    attempt_binding = {"schema_version": "attempt-fixture"}
    attempt_view = {
        "binding": attempt_binding,
        "contract": contract,
        "binding_sha256": "4" * 64,
        "lineage_sha256": "5" * 64,
        "start_next_tick": contract.start_next_tick,
        "target_next_tick": contract.start_next_tick + contract.total_ticks,
    }
    headers = []
    for world in range(contract.B):
        header = [0] * 32
        header[0] = 1000 + world
        header[1] = 10
        header[13] = 2000 + world
        header[14] = 4
        headers.append(tuple(header))
    rng_identity = {
        "schema_version": "eiel_workhorse_rng_identity_v2",
        "rng_epoch": 4,
        "world_uids": [1000, 1001],
        "rng_keys": [2000, 2001],
    }
    sink = {
        "artifact_sha256": artifact_sha,
        "manifest_sha256": manifest["manifest_sha256"],
        "world_uids": [1000, 1001],
        "rng_identity": rng_identity,
        "rng_identity_sha256": authority.canonical_sha256(rng_identity),
        "dispatch_identity_sha256": "6" * 64,
    }

    monkeypatch.setattr(
        authority,
        "validate_workhorse_prebuilt_v2",
        lambda _path: (artifact, tmp_path / "manifest.json", deepcopy(manifest)),
    )
    monkeypatch.setattr(authority, "_event_attempt_view_v2", lambda *_a, **_k: deepcopy(attempt_view))

    def validate_sink(_path, **expected):
        assert expected["expected_artifact_identity_sha256"] == "1" * 64
        assert expected["expected_end_next_absolute_tick"] == 10
        assert expected["expected_attempt_authority_binding_sha256"] == "4" * 64
        assert expected["expected_post_state_sha256"] == "7" * 64
        return deepcopy(sink)

    monkeypatch.setattr(authority, "validate_workhorse_event_sink_receipt_v2", validate_sink)
    monkeypatch.setattr(
        authority,
        "workhorse_governing_contract_bindings_v2",
        lambda: {"governing": "fixture"},
    )
    monkeypatch.setattr(
        workhorse_layout,
        "workhorse_checkpoint_identity_v2",
        lambda *_a, **_k: {"checkpoint_identity_sha256": "8" * 64},
    )

    metadata = authority.workhorse_checkpoint_metadata_from_state_identity_v2(
        contract=contract,
        checkpoint_at_tick=10,
        state_nbytes=contract.B
        * workhorse_layout_v2(contract.shape).state_bytes_per_world,
        state_sha256="7" * 64,
        state_headers=headers,
        artifact_path=artifact,
        event_sink_receipt_path=tmp_path / "sink.json",
        attempt_authority_binding=attempt_binding,
    )
    assert metadata["world_uids"] == [1000, 1001]
    assert metadata["rng_keys"] == [2000, 2001]
    assert metadata["event_sink_ack"] == sink
    assert metadata["attempt_authority_binding"] == attempt_binding
    assert metadata["canonical_state"]["sha256"] == "7" * 64
    assert metadata["checkpoint_host_staging_slabs"] == 2
    assert metadata["checkpoint_device_staging_bytes"] == 0


def _native_checkpoint(
    *, tick: int = 10, B: int = 2
) -> tuple[CudaFullEvolutionCheckpointV2, dict[str, object]]:
    contract = CudaFullEvolutionNativeContractV2(B=B, total_ticks=max(20, tick))
    row_bytes = state_layout_v2().canonical_checkpoint_bytes_per_world
    raw = bytes((index * 29 + 7) % 256 for index in range(B * row_bytes))
    state_sha256 = hashlib.sha256(raw).hexdigest()
    layout_sha256 = hashlib.sha256(
        json.dumps(
            {
                "dtype": "uint8",
                "shape": [B, row_bytes],
                "nbytes": len(raw),
                "content_sha256": state_sha256,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    ).hexdigest()
    checkpoint = CudaFullEvolutionCheckpointV2(
        contract=contract,
        canonical_state=raw,
        canonical_state_shape=(B, row_bytes),
        canonical_state_nbytes=len(raw),
        canonical_state_sha256=state_sha256,
        canonical_state_layout_sha256=layout_sha256,
        initial_next_absolute_tick=0,
        advanced_ticks=tick,
        acknowledged_event_next_tick=tick,
        last_sink_receipt_sha256="1" * 64,
        world_uids=tuple(1000 + index for index in range(B)),
        rng_keys=tuple(2000 + index for index in range(B)),
        rng_semantic_epoch=3,
        artifact_path=Path("/tmp/native-v2-artifact.so"),
        artifact_sha256="2" * 64,
        manifest_sha256="3" * 64,
        device_identity={
            "index": 0,
            "name": "fixture",
            "uuid": "GPU-FIXTURE",
            "pci_domain_id": 0,
            "pci_bus_id": 1,
            "pci_device_id": 0,
            "total_memory_bytes": 8 * 1024**3,
            "multiprocessor_count": 1,
            "sm": "sm_75",
            "cuda_runtime": "fixture",
        },
        population_fixture_id=None,
        metadata_seal_sha256="0" * 64,
    )
    runtime_checkpoint = {
        "schema_version": checkpoint.schema_version,
        "time_authority": checkpoint.time_authority,
        "contract": contract.to_dict(),
        "canonical_state_shape": list(checkpoint.canonical_state_shape),
        "canonical_state_nbytes": checkpoint.canonical_state_nbytes,
        "canonical_state_sha256": checkpoint.canonical_state_sha256,
        "canonical_state_layout_sha256": checkpoint.canonical_state_layout_sha256,
        "initial_next_absolute_tick": checkpoint.initial_next_absolute_tick,
        "advanced_ticks": checkpoint.advanced_ticks,
        "acknowledged_event_next_tick": checkpoint.acknowledged_event_next_tick,
        "last_sink_receipt_sha256": checkpoint.last_sink_receipt_sha256,
        "world_uids": list(checkpoint.world_uids),
        "rng_keys": list(checkpoint.rng_keys),
        "rng_semantic_epoch": checkpoint.rng_semantic_epoch,
        "artifact_sha256": checkpoint.artifact_sha256,
        "manifest_sha256": checkpoint.manifest_sha256,
        "device_identity": dict(checkpoint.device_identity),
        "population_fixture_id": checkpoint.population_fixture_id,
    }
    metadata_seal = hashlib.sha256(
        json.dumps(
            runtime_checkpoint,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    checkpoint = replace(checkpoint, metadata_seal_sha256=metadata_seal)
    runtime_checkpoint["metadata_seal_sha256"] = metadata_seal
    return checkpoint, runtime_checkpoint


def test_native_capture_publishes_exact_host_only_bundle_and_zero_worker_materialization(
    tmp_path: Path,
) -> None:
    checkpoint, runtime_checkpoint = _native_checkpoint()
    artifact = tmp_path / "artifact.so"
    artifact.write_bytes(b"native-artifact")
    checkpoint = replace(checkpoint, artifact_path=artifact.resolve())
    destination = tmp_path / "checkpoint-tick-000000000010"
    capture = adapter.capture_native_checkpoint_to_file_v1(
        checkpoint,
        destination,
        runtime_checkpoint=runtime_checkpoint,
        artifact_path=artifact,
        checkpoint_at_tick=10,
        chunk_bytes=64 * 1024,
        require_native_pool=False,
    )
    assert validate_detached_capture_host_only_v1(capture)["tensor_free"] is True
    pipeline = ACSNativeDetachedSlotsV1(slots=1)
    token = pipeline.begin_capture(compute_frontier=10)
    pipeline.submit_capture(token, capture)
    result = dict(pipeline.result(token))
    assert result["checkpoint_id"].startswith("cuda-native-checkpoint-tick-000000000010-")
    assert result["worker_full_state_bytes_materialized"] == 0
    assert result["worker_cuda_references"] == 0
    published = Path(result["path"])
    assert (published / "canonical_state.bin").read_bytes() == checkpoint.canonical_state
    metadata = json.loads((published / "metadata.json").read_text(encoding="ascii"))
    completion = json.loads(
        (published / "bundle_complete.json").read_text(encoding="ascii")
    )
    assert metadata["runtime_checkpoint"] == runtime_checkpoint
    assert completion["checkpoint_id"] == result["checkpoint_id"]
    assert completion["canonical_state_sha256"] == checkpoint.canonical_state_sha256
    reopened = adapter.reopen_native_checkpoint_bundle_v1(published)
    assert reopened == checkpoint
    selected, projection = adapter.reopen_native_checkpoint_world_v1(
        published, world_uid=checkpoint.world_uids[1]
    )
    row_bytes = checkpoint.canonical_state_shape[1]
    assert selected.contract.B == 1
    assert selected.world_uids == (checkpoint.world_uids[1],)
    assert selected.rng_keys == (checkpoint.rng_keys[1],)
    assert selected.canonical_state == checkpoint.canonical_state[row_bytes:]
    assert projection["source_world_count"] == 2
    assert projection["selected_world_index"] == 1
    assert projection["next_absolute_tick"] == 10
    assert not Path(capture["capture_path"]).exists()
    pipeline.acknowledge_durable(token)
    pipeline.close()


def test_native_capture_binds_compact_science_frontier_into_acs_bundle(
    tmp_path: Path,
) -> None:
    checkpoint, runtime_checkpoint = _native_checkpoint()
    artifact = tmp_path / "artifact.so"
    artifact.write_bytes(b"native-artifact-with-compact-frontier")
    checkpoint = replace(checkpoint, artifact_path=artifact.resolve())
    frontier_body = {
        "schema_version": (
            "eiel_cuda_v2_compact_science_continuation_frontier_v1"
        ),
        "next_absolute_tick": 10,
        "native_world_control_set_sha256": "a" * 64,
        "counter_frontier": {"next_absolute_tick": 10},
        "dispatch_artifacts": [],
        "snapshot_artifacts": [],
    }
    frontier = {
        **frontier_body,
        "frontier_sha256": hashlib.sha256(
            adapter._canonical_json(frontier_body)
        ).hexdigest(),
    }
    destination = tmp_path / "checkpoint-tick-000000000010"
    capture = adapter.capture_native_checkpoint_to_file_v1(
        checkpoint,
        destination,
        runtime_checkpoint=runtime_checkpoint,
        artifact_path=artifact,
        checkpoint_at_tick=10,
        compact_science_frontier=frontier,
        chunk_bytes=64 * 1024,
        require_native_pool=False,
    )
    assert capture["compact_science_frontier"] == frontier
    result = adapter.finalize_native_checkpoint_capture_v1(capture)
    assert result["compact_science_frontier"] == frontier
    metadata = json.loads(
        (Path(result["path"]) / "metadata.json").read_text(encoding="ascii")
    )
    # CRCA's absolute-tick checkpoint metadata has an exact closed shape.
    # The compact frontier is bound by the immutable capture descriptor,
    # detached result, and provider checkpoint record instead of being added
    # as an unrecognised native-bundle metadata key.
    assert "compact_science_frontier" not in metadata
    assert adapter.reopen_native_checkpoint_bundle_v1(Path(result["path"])) == checkpoint


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_native_direct_resident_capture_uses_pinned_pool_without_full_host_state(
    tmp_path: Path,
) -> None:
    if adapter.streamed._native_slab_pool_type() is None:
        pytest.skip("prebuilt ACS native slab pool is unavailable")
    checkpoint, _runtime_checkpoint = _native_checkpoint()
    layout = state_layout_v2()
    resident = torch.zeros(
        (checkpoint.contract.B, layout.state_bytes_per_world),
        dtype=torch.uint8,
        device="cuda",
    )
    canonical = torch.frombuffer(
        bytearray(checkpoint.canonical_state), dtype=torch.uint8
    ).reshape(checkpoint.canonical_state_shape)
    resident[:, : layout.canonical_checkpoint_bytes_per_world].copy_(
        canonical.to("cuda")
    )
    artifact = tmp_path / "artifact.so"
    artifact.write_bytes(b"native-direct-artifact")
    source = CudaFullEvolutionCheckpointCaptureSourceV1(
        contract=checkpoint.contract,
        resident_state=resident,
        canonical_bytes_per_world=layout.canonical_checkpoint_bytes_per_world,
        initial_next_absolute_tick=checkpoint.initial_next_absolute_tick,
        advanced_ticks=checkpoint.advanced_ticks,
        acknowledged_event_next_tick=checkpoint.acknowledged_event_next_tick,
        last_sink_receipt_sha256=checkpoint.last_sink_receipt_sha256,
        world_uids=checkpoint.world_uids,
        rng_keys=checkpoint.rng_keys,
        rng_semantic_epoch=checkpoint.rng_semantic_epoch,
        artifact_path=artifact.resolve(),
        artifact_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),
        manifest_sha256=checkpoint.manifest_sha256,
        device_identity=checkpoint.device_identity,
        population_fixture_id=checkpoint.population_fixture_id,
    )
    capture = adapter.capture_native_resident_checkpoint_to_file_v1(
        source,
        tmp_path / "direct-checkpoint",
        artifact_path=artifact,
        checkpoint_at_tick=checkpoint.acknowledged_event_next_tick,
        chunk_bytes=64 * 1024,
        require_native_pool=True,
    )
    assert validate_detached_capture_host_only_v1(capture)["tensor_free"] is True
    assert capture["source_device_type"] == "cuda_resident_canonical_rows_v1"
    assert capture["producer_full_state_host_bytes_materialized"] == 0
    assert capture["producer_compacted_cuda_bytes_materialized"] == 0
    assert capture["native_slab_pool"]["pinned"] is True
    assert capture["native_slab_pool"]["route_capture_count"] == checkpoint.contract.B
    result = adapter.finalize_native_checkpoint_capture_v1(capture)
    assert result["worker_cuda_references"] == 0
    assert result["producer_full_state_host_bytes_materialized"] == 0
    reopened = adapter.reopen_native_checkpoint_bundle_v1(result["path"])
    assert reopened.canonical_state == checkpoint.canonical_state
    assert reopened.world_uids == checkpoint.world_uids
    assert reopened.acknowledged_event_next_tick == 10


def test_native_capture_authority_seal_rejects_nested_runtime_mutation(
    tmp_path: Path,
) -> None:
    checkpoint, runtime_checkpoint = _native_checkpoint()
    capture = adapter.capture_native_checkpoint_to_file_v1(
        checkpoint,
        tmp_path / "checkpoint",
        runtime_checkpoint=runtime_checkpoint,
        artifact_path=checkpoint.artifact_path,
        checkpoint_at_tick=10,
        require_native_pool=False,
    )
    changed = deepcopy(capture)
    changed["runtime_checkpoint"]["advanced_ticks"] = 11  # type: ignore[index]
    with pytest.raises(adapter.CudaFullEvolutionV2ACSError, match="authority seal"):
        adapter.finalize_native_checkpoint_capture_v1(changed)
    assert Path(capture["capture_path"]).exists()


def test_native_postpublication_readback_failure_quarantines_canonical_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint, runtime_checkpoint = _native_checkpoint()
    destination = tmp_path / "checkpoint"
    capture = adapter.capture_native_checkpoint_to_file_v1(
        checkpoint,
        destination,
        runtime_checkpoint=runtime_checkpoint,
        artifact_path=checkpoint.artifact_path,
        checkpoint_at_tick=10,
        require_native_pool=False,
    )
    original_read = adapter._read_small_file

    def corrupt_metadata(path: Path) -> bytes:
        if path.name == "metadata.json":
            return b"corrupt"
        return original_read(path)

    monkeypatch.setattr(adapter, "_read_small_file", corrupt_metadata)
    with pytest.raises(
        adapter.CudaFullEvolutionV2ACSError,
        match="publication was quarantined",
    ) as raised:
        adapter.finalize_native_checkpoint_capture_v1(capture)
    assert not destination.exists()
    failure_path = Path(raised.value.publication_failure_path)
    assert failure_path.is_dir()
    assert failure_path.name.startswith(f"{destination.name}.publication.failed.")
    assert Path(capture["capture_path"]).is_file()


def test_native_capture_fails_closed_when_destination_is_claimed_before_worker_publish(
    tmp_path: Path,
) -> None:
    checkpoint, runtime_checkpoint = _native_checkpoint()
    destination = tmp_path / "checkpoint"
    capture = adapter.capture_native_checkpoint_to_file_v1(
        checkpoint,
        destination,
        runtime_checkpoint=runtime_checkpoint,
        artifact_path=checkpoint.artifact_path,
        checkpoint_at_tick=10,
        require_native_pool=False,
    )
    destination.mkdir()
    pipeline = ACSNativeDetachedSlotsV1(slots=1)
    token = pipeline.begin_capture(compute_frontier=10)
    pipeline.submit_capture(token, capture)
    with pytest.raises(ACSNativeDetachedSlotError, match="worker failed"):
        pipeline.result(token)
    assert pipeline.telemetry()["failure_latched"] is True
    pipeline.abort()
