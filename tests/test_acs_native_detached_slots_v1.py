from __future__ import annotations

from collections.abc import Callable, Mapping
import json
from dataclasses import replace
import os
from pathlib import Path
from types import SimpleNamespace
import threading
import time

import pytest
import torch

import engine.campaigns.acs_native_detached_slots_v1 as detached
import engine.campaigns.providers as providers
from engine.campaigns.artifact_lifecycle import (
    build_sealed_artifact_inventory_row_v1,
)
from engine.campaigns.acs_native_aggregate_checkpoint_v1 import (
    load_acs_native_aggregate_checkpoint,
)
from engine.campaigns.acs_native_detached_slots_v1 import (
    ACSNativeDetachedSlotError,
    ACSNativeDetachedSlotsV1,
    capture_acs_native_aggregate_payload_v1,
    capture_acs_native_aggregate_payload_from_cursor_v1,
    execute_detached_slots_focused_probe_v1,
)
from engine.campaigns.qualification import (
    validate_execution_route_acs_focused_evidence_v1,
)
from engine.campaigns.providers import ProductionEngineLifecycleV1, ResidentBatchRequestV1
from engine.perf.cuda_architecture_b_direct_resident_patch_v1 import (
    CheckpointProjectionCursorV1,
    DetachedCellProjectionV1,
    DETACHED_CELL_PROJECTION_SCHEMA_VERSION_V1,
)
from tests.test_acs_native_aggregate_checkpoint_v1 import _projection


def _capture(tmp_path: Path, index: int) -> dict[str, object]:
    cell_ids = ("cell-0", "cell-1")
    digests = {cell_id: "a" * 64 for cell_id in cell_ids}
    identities = {
        cell_id: {
            "snapshot_digest_sha256": digests[cell_id],
            "source_projection_sha256": digests[cell_id],
            "snapshot_id": f"snapshot-{cell_id}",
        }
        for cell_id in cell_ids
    }
    observers = {
        cell_id: {"event_frontier": 17 + cell_index, "cell_id": cell_id}
        for cell_index, cell_id in enumerate(cell_ids)
    }
    return capture_acs_native_aggregate_payload_v1(
        (_projection(0), _projection(1)),
        tmp_path / f"checkpoint-{index}.acsaggregate",
        cell_ids=cell_ids,
        generation=4,
        tick=index,
        absolute_next_tick=160 * (index + 1),
        batch_id="detached-test",
        segment_ordinal=index + 1,
        source_projection_sha256s=digests,
        architecture_b_identities=identities,
        observer_continuations=observers,
        chunk_bytes=7,
    )


def _cursor_spool_capture_v2(
    tmp_path: Path,
    *,
    name: str = "cursor-v2.acsaggregate",
) -> dict[str, object]:
    projections = (_projection(0), _projection(1))
    cell_ids = ("cell-0", "cell-1")

    def factory(index: int, cell_id: str) -> DetachedCellProjectionV1:
        projection = projections[index]
        return DetachedCellProjectionV1(
            schema_version=DETACHED_CELL_PROJECTION_SCHEMA_VERSION_V1,
            cell_id=cell_id,
            cell_index=index,
            projection=projection,
            compact_result_rows=(
                {
                    "cell_id": cell_id,
                    "metrics": {"population.alive": 10 + index},
                },
            ),
            terminal_generation=4,
            terminal_tick=160,
            live_projection_bytes=sum(
                int(tensor.numel()) * int(tensor.element_size())
                for tensor in projection["tensors"].values()  # type: ignore[union-attr]
            ),
        )

    cursor = CheckpointProjectionCursorV1.begin(
        owner=object(),
        coordinate=160,
        ordered_cell_ids=cell_ids,
        factory=factory,
    )
    identities = {
        cell_id: {
            "snapshot_digest_sha256": "a" * 64,
            "source_projection_sha256": "a" * 64,
            "snapshot_id": f"snapshot-{cell_id}",
        }
        for cell_id in cell_ids
    }
    observers = {
        cell_id: {"event_frontier": 17 + index, "cell_id": cell_id}
        for index, cell_id in enumerate(cell_ids)
    }
    result = capture_acs_native_aggregate_payload_from_cursor_v1(
        cursor,
        tmp_path / name,
        cell_ids=cell_ids,
        generation=4,
        tick=160,
        absolute_next_tick=320,
        batch_id="detached-cursor-v2-test",
        segment_ordinal=1,
        chunk_bytes=7,
        authority_factory=lambda cell_id, _projection: {
            "source_projection_sha256": "a" * 64,
            "architecture_b_identity": identities[cell_id],
            "observer_continuation": observers[cell_id],
            "mechanics_checkpoint_digest": "b" * 64,
            "canonical_checkpoint_storage_digest": "c" * 64,
        },
    )
    return result


def _wait_until(predicate: Callable[[], bool], *, timeout_seconds: float = 5.0) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    assert predicate()


def test_detached_slots_publish_one_aggregate_without_worker_tensor_state(
    tmp_path: Path,
) -> None:
    pipeline = ACSNativeDetachedSlotsV1(slots=1)
    try:
        token = pipeline.begin_capture(compute_frontier=160)
        capture = _capture(tmp_path, 0)
        assert capture["schema_version"] == detached.DETACHED_CAPTURE_SCHEMA_VERSION_V1
        assert not any(isinstance(value, torch.Tensor) for value in capture.values())
        pipeline.submit_capture(token, capture)
        receipt = pipeline.result(token)
        published = Path(str(receipt["path"]))
        assert published.is_file()
        assert receipt["cell_count"] == 2
        assert receipt["detached_writer_wall_time_ns"] >= 0
        assert (
            receipt["detached_slot_reserved_perf_ns"]
            <= receipt["detached_capture_submitted_perf_ns"]
            <= receipt["detached_writer_started_perf_ns"]
            < receipt["detached_writer_finished_perf_ns"]
        )
        restored = load_acs_native_aggregate_checkpoint(
            published,
            cell_id="cell-1",
            expected_source_projection_sha256="a" * 64,
        )
        assert torch.equal(
            restored["projection"]["tensors"]["active"],
            _projection(1)["tensors"]["active"],  # type: ignore[index]
        )
        pipeline.acknowledge_durable(token)
        telemetry = pipeline.telemetry()
        assert telemetry["slot_states"] == ["FREE"]
        assert telemetry["compute_frontier"] == 160
        assert telemetry["durable_checkpoint_frontier"] == 160
        assert telemetry["capture_pool_count"] == 1
        assert telemetry["worker_accepts_cuda_objects"] is False
        assert telemetry["worker_accepts_callbacks"] is False
        assert telemetry["worker_tensor_device_policy"] == "forced_cpu_context_v1"
    finally:
        pipeline.close()


def test_detached_worker_forces_cpu_default_device_for_host_settlement(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    capture = _capture(tmp_path, 0)
    observed_devices: list[str] = []
    real_finalize = detached._finalize_capture_v1

    def observe_finalize(
        value: Mapping[str, object],
        *,
        cancel_event: threading.Event | None = None,
    ) -> dict[str, object]:
        observed_devices.append(str(torch.get_default_device()))
        assert torch.zeros(1).device.type == "cpu"
        return real_finalize(value, cancel_event=cancel_event)

    monkeypatch.setattr(detached, "_finalize_capture_v1", observe_finalize)
    prior_default = str(torch.get_default_device())
    torch.set_default_device("meta")
    pipeline = ACSNativeDetachedSlotsV1(slots=1)
    try:
        token = pipeline.begin_capture(compute_frontier=160)
        pipeline.submit_capture(token, capture)
        pipeline.result(token)
        pipeline.acknowledge_durable(token)
        assert observed_devices == ["cpu"]
    finally:
        pipeline.close()
        torch.set_default_device(None if prior_default == "cpu" else prior_default)


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="requires an authorized CUDA device for the legacy default-type leak",
)
def test_detached_worker_fences_legacy_cuda_default_tensor_type(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    capture = _capture(tmp_path, 0)
    observed_devices: list[str] = []
    real_finalize = detached._finalize_capture_v1

    def observe_finalize(
        value: Mapping[str, object],
        *,
        cancel_event: threading.Event | None = None,
    ) -> dict[str, object]:
        observed_devices.append(str(torch.zeros(1).device))
        return real_finalize(value, cancel_event=cancel_event)

    monkeypatch.setattr(detached, "_finalize_capture_v1", observe_finalize)
    prior_default_dtype = torch.get_default_dtype()
    prior_default_device = str(torch.get_default_device())
    torch.set_default_device(None)
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    pipeline = ACSNativeDetachedSlotsV1(slots=1)
    try:
        assert torch.zeros(1).device.type == "cuda"
        token = pipeline.begin_capture(compute_frontier=160)
        pipeline.submit_capture(token, capture)
        pipeline.result(token)
        pipeline.acknowledge_durable(token)
        assert observed_devices == ["cpu"]
    finally:
        pipeline.close()
        torch.set_default_tensor_type(torch.FloatTensor)
        torch.set_default_dtype(prior_default_dtype)
        torch.set_default_device(
            None if prior_default_device == "cpu" else prior_default_device
        )


def test_detached_cursor_spool_v2_is_host_file_only_before_worker_submit(
    tmp_path: Path,
) -> None:
    result = _cursor_spool_capture_v2(tmp_path)
    capture = result["capture"]
    assert capture["schema_version"] == detached.DETACHED_SPOOL_CAPTURE_SCHEMA_VERSION_V2
    detached._assert_no_tensors(capture)  # type: ignore[arg-type]
    detached._assert_no_callables(capture)  # type: ignore[arg-type]

    spool_path = Path(str(capture["spool_path"]))
    materialization_path = Path(str(capture["materialization_path"]))
    destination = Path(str(capture["destination"]))
    assert spool_path.is_file()
    assert capture["spool_bytes"] == spool_path.stat().st_size
    assert (
        capture["producer_spool_started_perf_ns"]
        < capture["producer_spool_completed_perf_ns"]
    )
    assert "capture_path" not in capture
    assert not materialization_path.exists()
    assert not destination.exists()


def test_detached_cursor_spool_v2_restores_exact_one_cell_before_worker_submit(
    tmp_path: Path,
) -> None:
    capture = _cursor_spool_capture_v2(tmp_path)["capture"]
    spool_path = Path(str(capture["spool_path"]))

    restored = detached.load_acs_native_detached_spool_cell_v1(
        capture,
        cell_id="cell-1",
        expected_source_projection_sha256="a" * 64,
    )
    expected = _projection(1)
    assert restored["cell_id"] == "cell-1"
    assert restored["source_projection_sha256"] == "a" * 64
    assert restored["architecture_b_identity"]["snapshot_id"] == "snapshot-cell-1"
    assert restored["observer_continuation"] == {
        "event_frontier": 18,
        "cell_id": "cell-1",
    }
    assert restored["projection"]["metadata"] == expected["metadata"]
    assert restored["projection"]["event_cursor"] == expected["event_cursor"]
    assert torch.equal(
        restored["projection"]["tensors"]["active"],  # type: ignore[index]
        expected["tensors"]["active"],  # type: ignore[index]
    )
    assert spool_path.is_file()


@pytest.mark.parametrize("mutation", ("corruption", "truncation", "stat", "descriptor"))
def test_detached_cursor_spool_v2_mutation_fails_closed(
    mutation: str,
    tmp_path: Path,
) -> None:
    capture = _cursor_spool_capture_v2(tmp_path)["capture"]
    spool_path = Path(str(capture["spool_path"]))
    mutated = dict(capture)

    if mutation == "corruption":
        raw = bytearray(spool_path.read_bytes())
        raw[0] ^= 0xFF
        spool_path.write_bytes(raw)
        mutated["spool_seal"] = dict(capture["spool_seal"])  # type: ignore[arg-type]
        mutated["spool_seal"]["file_identity"] = detached._spool_file_identity_v1(  # type: ignore[index]
            spool_path
        )
        mutated["spool_seal"]["descriptor_sha256"] = (  # type: ignore[index]
            detached._spool_descriptor_digest_v1(mutated)
        )
    elif mutation == "truncation":
        with spool_path.open("r+b") as stream_handle:
            stream_handle.truncate(spool_path.stat().st_size - 1)
        mutated["spool_bytes"] = spool_path.stat().st_size
        mutated["spool_seal"] = dict(capture["spool_seal"])  # type: ignore[arg-type]
        mutated["spool_seal"]["file_identity"] = detached._spool_file_identity_v1(  # type: ignore[index]
            spool_path
        )
        mutated["spool_seal"]["descriptor_sha256"] = (  # type: ignore[index]
            detached._spool_descriptor_digest_v1(mutated)
        )
    elif mutation == "stat":
        observed = spool_path.stat()
        os.utime(
            spool_path,
            ns=(int(observed.st_atime_ns), int(observed.st_mtime_ns) + 1),
        )
    elif mutation == "descriptor":
        mutated["spool_seal"] = dict(capture["spool_seal"])  # type: ignore[arg-type]
        mutated["spool_seal"]["descriptor_sha256"] = "0" * 64  # type: ignore[index]
    else:  # pragma: no cover - parameter set is intentionally closed
        raise AssertionError(f"unknown spool mutation: {mutation}")

    with pytest.raises(ACSNativeDetachedSlotError, match="detached cursor spool"):
        detached.load_acs_native_detached_spool_cell_v1(
            mutated,
            cell_id="cell-0",
            expected_source_projection_sha256="a" * 64,
        )
    assert spool_path.exists()


def test_detached_cursor_spool_v2_worker_materializes_publishes_and_restores(
    tmp_path: Path,
) -> None:
    capture = _cursor_spool_capture_v2(tmp_path)["capture"]
    spool_path = Path(str(capture["spool_path"]))
    materialization_path = Path(str(capture["materialization_path"]))
    destination = Path(str(capture["destination"]))
    pipeline = ACSNativeDetachedSlotsV1(slots=1)
    try:
        token = pipeline.begin_capture(compute_frontier=320)
        pipeline.submit_capture(token, capture)  # type: ignore[arg-type]
        receipt = pipeline.result(token)
        published = Path(str(receipt["path"]))
        assert published == destination
        assert published.is_file()
        restored = load_acs_native_aggregate_checkpoint(
            published,
            cell_id="cell-1",
            expected_source_projection_sha256="a" * 64,
        )
        assert torch.equal(
            restored["projection"]["tensors"]["active"],  # type: ignore[index]
            _projection(1)["tensors"]["active"],  # type: ignore[index]
        )
        assert receipt["detached_capture_path"] == str(materialization_path)
        assert receipt["detached_spool_materialization_seconds"] >= 0
        assert not spool_path.exists()
        assert not materialization_path.exists()
        assert not list(tmp_path.glob("*.spool.tmp"))
        assert not list(tmp_path.glob("*.capture.tmp"))
        assert not list(tmp_path.glob("*.finalize.tmp"))
        pipeline.acknowledge_durable(token)
    finally:
        pipeline.close()


def test_detached_cursor_spool_source_lease_detaches_from_durable_slot_until_release(
    tmp_path: Path,
) -> None:
    capture = dict(_cursor_spool_capture_v2(tmp_path)["capture"])
    capture["source_consumer_required"] = True
    spool_path = Path(str(capture["spool_path"]))
    pipeline = ACSNativeDetachedSlotsV1(slots=1)
    try:
        token = pipeline.begin_capture(compute_frontier=320)
        pipeline.submit_capture(token, capture)
        _wait_until(
            lambda: pipeline.telemetry()["slot_states"][0] == "DURABLE"
        )
        assert pipeline.ready(token) is True
        assert spool_path.is_file()
        assert isinstance(pipeline.result(token), Mapping)
        pipeline.acknowledge_durable(token)
        assert spool_path.is_file()
        telemetry = pipeline.telemetry()
        assert telemetry["queue_occupancy"] == 0
        assert telemetry["detached_source_lease_count"] == 1
        assert telemetry["source_lease_spool_bytes_active"] == capture["spool_bytes"]
        assert telemetry["temporary_capture_files_active"] == 1
        assert telemetry["temporary_capture_file_high_water"] == 1
        successor_token = pipeline.begin_capture(compute_frontier=640)
        pipeline.release_source(token)
        assert not spool_path.exists()
        successor_capture = _capture(tmp_path, 99)
        pipeline.submit_capture(successor_token, successor_capture)
        assert isinstance(pipeline.result(successor_token), Mapping)
        pipeline.acknowledge_durable(successor_token)
        assert pipeline.telemetry()["temporary_capture_files_active"] == 0
    finally:
        pipeline.abort()


def test_detached_source_ack_does_not_invalidate_retained_successor_descriptor(
    tmp_path: Path,
) -> None:
    """Reproduce PREQUAL-4B's stale descriptor after owner handoff.

    The successor importer can finish while its resident checkpoint projector
    still owns the lazy canonical descriptor. Durability acknowledgement must
    free the slot without deleting that descriptor; only explicit owner-source
    release may remove the spool.
    """

    capture = dict(_cursor_spool_capture_v2(tmp_path)["capture"])
    capture["source_consumer_required"] = True
    retained_descriptor = providers._LazyDetachedSpoolProjectionSequence(capture)
    pipeline = ACSNativeDetachedSlotsV1(slots=2)
    try:
        token = pipeline.begin_capture(compute_frontier=1120)
        pipeline.submit_capture(token, capture)
        _wait_until(
            lambda: pipeline.telemetry()["slot_states"][0] == "DURABLE"
        )

        # One successor consumer completes, while the prepared owner's lazy
        # descriptor remains reachable through the second consumer.
        assert isinstance(retained_descriptor[0], Mapping)
        pipeline.result(token)
        pipeline.acknowledge_durable(token)
        assert isinstance(retained_descriptor[1], Mapping)
        pipeline.release_source(token)
        with pytest.raises(ACSNativeDetachedSlotError, match="spool stat failed"):
            retained_descriptor[1]
    finally:
        pipeline.abort()


def test_bounded_source_cursor_factory_rolls_to_replacement_spool(
    tmp_path: Path,
) -> None:
    first = dict(_cursor_spool_capture_v2(tmp_path / "first")["capture"])
    second = dict(_cursor_spool_capture_v2(tmp_path / "second")["capture"])
    cell_ids = tuple(first["header"]["cell_ids"])  # type: ignore[index]
    provider = SimpleNamespace(terminal_projection_digest=lambda _value: "a" * 64)
    store = providers._BoundedSourceProjectionStoreV1(
        provider,
        root=tmp_path / "source-store",
        cell_ids=cell_ids,
        source_projections=providers._LazyDetachedSpoolProjectionSequence(first),
    )
    factory = store.cursor_factory()

    factory.rebind_detached_spool(second)  # type: ignore[attr-defined]
    Path(str(first["spool_path"])).unlink()
    cursor = factory()
    for cell_id in cell_ids:
        detached_source = cursor.next_cell()
        assert detached_source is not None
        assert detached_source.cell_id == cell_id
        assert isinstance(detached_source.require_payload(), Mapping)
        cursor.import_and_release(cell_id, None)
    assert cursor.close().state == "SEALED"


def test_abort_quarantines_detached_owner_source_lease(tmp_path: Path) -> None:
    capture = dict(_cursor_spool_capture_v2(tmp_path)["capture"])
    capture["source_consumer_required"] = True
    spool_path = Path(str(capture["spool_path"]))
    pipeline = ACSNativeDetachedSlotsV1(slots=1)
    token = pipeline.begin_capture(compute_frontier=1120)
    pipeline.submit_capture(token, capture)
    assert isinstance(pipeline.result(token), Mapping)
    pipeline.acknowledge_durable(token)

    with pytest.raises(ACSNativeDetachedSlotError, match="source leases"):
        pipeline.close()
    pipeline.abort()

    assert not spool_path.exists()
    assert pipeline.telemetry()["detached_source_lease_count"] == 0
    assert any(
        path.endswith(".spool.failed")
        for path in pipeline.telemetry()["failure_capture_paths"]
    )


def test_detached_cursor_spool_writer_and_successor_reader_can_overlap(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    capture = dict(_cursor_spool_capture_v2(tmp_path)["capture"])
    capture["source_consumer_required"] = True
    spool_path = Path(str(capture["spool_path"]))
    both_ready = threading.Barrier(2)
    writer_started = threading.Event()
    source_started = threading.Event()
    writer_materialize = detached._materialize_cursor_spool_v2
    source_load = detached.load_acs_native_detached_spool_cell_v1

    def overlap_materialize(
        value: Mapping[str, object],
        *,
        cancel_event: threading.Event | None = None,
    ) -> dict[str, object]:
        writer_started.set()
        both_ready.wait(timeout=5.0)
        return writer_materialize(value, cancel_event=cancel_event)

    def overlap_load(
        value: Mapping[str, object],
        *,
        cell_id: str,
        expected_source_projection_sha256: str | None = None,
    ) -> dict[str, object]:
        source_started.set()
        both_ready.wait(timeout=5.0)
        return source_load(
            value,
            cell_id=cell_id,
            expected_source_projection_sha256=expected_source_projection_sha256,
        )

    monkeypatch.setattr(detached, "_materialize_cursor_spool_v2", overlap_materialize)
    monkeypatch.setattr(
        detached, "load_acs_native_detached_spool_cell_v1", overlap_load
    )
    pipeline = ACSNativeDetachedSlotsV1(slots=1)
    reader_thread: threading.Thread | None = None
    reader_holder: dict[str, object] = {}
    try:
        token = pipeline.begin_capture(compute_frontier=320)
        pipeline.submit_capture(token, capture)
        assert writer_started.wait(5.0)

        def read_successor_cell() -> None:
            try:
                reader_holder["restored"] = detached.load_acs_native_detached_spool_cell_v1(
                    capture,
                    cell_id="cell-1",
                    expected_source_projection_sha256="a" * 64,
                )
            except BaseException as exc:  # pragma: no cover - assertion below reports it
                reader_holder["error"] = exc

        reader_thread = threading.Thread(target=read_successor_cell, daemon=True)
        reader_thread.start()
        assert source_started.wait(5.0)
        reader_thread.join(timeout=5.0)
        assert not reader_thread.is_alive()
        assert "error" not in reader_holder
        restored = reader_holder["restored"]
        assert isinstance(restored, Mapping)
        assert torch.equal(
            restored["projection"]["tensors"]["active"],  # type: ignore[index]
            _projection(1)["tensors"]["active"],  # type: ignore[index]
        )
        _wait_until(
            lambda: pipeline.telemetry()["slot_states"][0] == "DURABLE"
        )
        assert spool_path.is_file()
        pipeline.release_source(token)
        pipeline.result(token)
        pipeline.acknowledge_durable(token)
        assert not spool_path.exists()
    finally:
        if reader_thread is not None:
            reader_thread.join(timeout=1.0)
        pipeline.abort()


def test_detached_cursor_spool_v2_corruption_is_fail_closed_and_quarantined(
    tmp_path: Path,
) -> None:
    capture = _cursor_spool_capture_v2(tmp_path)["capture"]
    spool_path = Path(str(capture["spool_path"]))
    raw = bytearray(spool_path.read_bytes())
    raw[0] ^= 0xFF
    spool_path.write_bytes(raw)

    pipeline = ACSNativeDetachedSlotsV1(slots=1)
    try:
        token = pipeline.begin_capture(compute_frontier=320)
        pipeline.submit_capture(token, capture)  # type: ignore[arg-type]
        with pytest.raises(ACSNativeDetachedSlotError, match="worker failed"):
            pipeline.result(token)
        telemetry = pipeline.telemetry()
        assert telemetry["failure_latched"] is True
        failed_paths = [Path(str(path)) for path in telemetry["failure_capture_paths"]]
        assert len(failed_paths) == 1
        assert failed_paths[0].name.endswith(".spool.failed")
        assert failed_paths[0].is_file()
        assert not spool_path.exists()
        assert not Path(str(capture["materialization_path"])).exists()
        assert not Path(str(capture["destination"])).exists()
        with pytest.raises(ACSNativeDetachedSlotError, match="failure is latched"):
            pipeline.begin_capture(compute_frontier=480)
    finally:
        pipeline.abort()


def test_detached_cursor_spool_v2_worker_failure_is_fail_closed_and_quarantined(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    capture = _cursor_spool_capture_v2(tmp_path)["capture"]

    def fail_finalize(
        _capture: object,
        *,
        cancel_event: threading.Event | None = None,
    ) -> dict[str, object]:
        del cancel_event
        raise RuntimeError("injected detached cursor v2 worker failure")

    monkeypatch.setattr(detached, "_finalize_capture_v1", fail_finalize)
    pipeline = ACSNativeDetachedSlotsV1(slots=1)
    try:
        token = pipeline.begin_capture(compute_frontier=320)
        pipeline.submit_capture(token, capture)  # type: ignore[arg-type]
        with pytest.raises(ACSNativeDetachedSlotError, match="worker failed"):
            pipeline.result(token)
        telemetry = pipeline.telemetry()
        assert telemetry["failure_latched"] is True
        failed_paths = [Path(str(path)) for path in telemetry["failure_capture_paths"]]
        assert len(failed_paths) == 1
        assert failed_paths[0].name.endswith(".spool.failed")
        assert failed_paths[0].is_file()
        assert not Path(str(capture["spool_path"])).exists()
        assert not Path(str(capture["materialization_path"])).exists()
        assert not Path(str(capture["destination"])).exists()
    finally:
        pipeline.abort()


def test_detached_abort_during_blocked_finalize_quarantines_capture_and_never_durable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    capture = _capture(tmp_path, 0)
    entered_finalize = threading.Event()
    real_finalize = detached._finalize_capture_v1

    def blocked_finalize(
        value: Mapping[str, object],
        *,
        cancel_event: threading.Event | None = None,
    ) -> dict[str, object]:
        assert cancel_event is not None
        entered_finalize.set()
        assert cancel_event.wait(timeout=5.0)
        return real_finalize(value, cancel_event=cancel_event)

    monkeypatch.setattr(detached, "_finalize_capture_v1", blocked_finalize)
    pipeline = ACSNativeDetachedSlotsV1(slots=1)
    abort_thread: threading.Thread | None = None
    try:
        token = pipeline.begin_capture(compute_frontier=160)
        pipeline.submit_capture(token, capture)
        assert entered_finalize.wait(timeout=5.0)

        abort_thread = threading.Thread(target=pipeline.abort, daemon=True)
        abort_thread.start()
        abort_thread.join(timeout=5.0)
        assert not abort_thread.is_alive(), "abort must join a blocked writer"

        telemetry = pipeline.telemetry()
        assert telemetry["aborted"] is True
        assert telemetry["writer_alive"] is False
        assert telemetry["failure_capture_paths"]
        quarantined = [
            Path(str(path)) for path in telemetry["failure_capture_paths"]
        ]
        assert any(path.is_file() and path.name.endswith(".capture.failed") for path in quarantined)
        assert not Path(str(capture["destination"])).exists()
        assert not list(tmp_path.glob("*.acsaggregate"))
        with pytest.raises(ACSNativeDetachedSlotError, match="aborted"):
            pipeline.result(token)
    finally:
        if abort_thread is None or abort_thread.is_alive():
            pipeline.abort()


def test_detached_abort_after_atomic_link_quarantines_late_publication(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    capture = _capture(tmp_path, 0)
    destination = Path(str(capture["destination"]))
    publication_linked = threading.Event()
    real_finalize = detached._finalize_capture_v1

    def linked_then_blocked_finalize(
        value: Mapping[str, object],
        *,
        cancel_event: threading.Event | None = None,
    ) -> dict[str, object]:
        assert cancel_event is not None
        result = real_finalize(value, cancel_event=cancel_event)
        assert destination.is_file()
        publication_linked.set()
        assert cancel_event.wait(timeout=5.0)
        return result

    monkeypatch.setattr(
        detached,
        "_finalize_capture_v1",
        linked_then_blocked_finalize,
    )
    pipeline = ACSNativeDetachedSlotsV1(slots=1)
    abort_thread: threading.Thread | None = None
    try:
        token = pipeline.begin_capture(compute_frontier=160)
        pipeline.submit_capture(token, capture)
        assert publication_linked.wait(timeout=5.0)

        abort_thread = threading.Thread(target=pipeline.abort, daemon=True)
        abort_thread.start()
        abort_thread.join(timeout=5.0)
        assert not abort_thread.is_alive(), "abort must fence a linked publication"

        telemetry = pipeline.telemetry()
        quarantined = [
            Path(str(path)) for path in telemetry["failure_capture_paths"]
        ]
        assert telemetry["aborted"] is True
        assert telemetry["writer_alive"] is False
        assert not destination.exists()
        assert any(
            path.is_file() and ".publication.failed" in path.name
            for path in quarantined
        )
        with pytest.raises(ACSNativeDetachedSlotError, match="aborted"):
            pipeline.result(token)
    finally:
        if abort_thread is None or abort_thread.is_alive():
            pipeline.abort()


def test_provider_excludes_cursor_spool_index_from_normal_capture_evidence() -> None:
    source = Path(providers.__file__).read_text(encoding="utf-8")
    block_start = source.index('capture["detached_capture"] = {')
    block_end = source.index("pending_acs_checkpoints.append", block_start)
    block = source[block_start:block_end]
    assert "for key, value in detached_capture.items()" in block
    assert (
        'if key\n'
        '                        not in {\n'
        '                            "header",\n'
        '                            "spool_index",\n'
        '                            "source_projection_sha256s",\n'
        '                        }'
        in block
    )


def test_provider_binds_worker_interval_into_capture_measurement() -> None:
    source = Path(providers.__file__).read_text(encoding="utf-8")
    block_start = source.index(
        'capture_measurement = capture.get("capture_barrier_measurement")'
    )
    block_end = source.index("checkpoint_set = (", block_start)
    block = source[block_start:block_end]
    for field in (
        "detached_capture_submitted_perf_ns",
        "detached_writer_started_perf_ns",
        "detached_writer_finished_perf_ns",
    ):
        assert field in block


def test_detached_cursor_capture_keeps_one_cell_live_and_preserves_dynamic_shapes(
    tmp_path: Path,
) -> None:
    first = _projection(0)
    second = _projection(1)
    second["tensors"]["active"] = torch.arange(8, dtype=torch.float32).reshape(2, 4)  # type: ignore[index]
    projections = (first, second)
    cell_ids = ("cell-0", "cell-1")

    def factory(index: int, cell_id: str) -> DetachedCellProjectionV1:
        projection = projections[index]
        return DetachedCellProjectionV1(
            schema_version=DETACHED_CELL_PROJECTION_SCHEMA_VERSION_V1,
            cell_id=cell_id,
            cell_index=index,
            projection=projection,
            compact_result_rows=(
                {
                    "cell_id": cell_id,
                    "metrics": {"population.alive": 10 + index},
                },
            ),
            terminal_generation=4,
            terminal_tick=160,
            live_projection_bytes=sum(
                int(tensor.numel()) * int(tensor.element_size())
                for tensor in projection["tensors"].values()  # type: ignore[union-attr]
            ),
        )

    cursor = CheckpointProjectionCursorV1.begin(
        owner=object(),
        coordinate=160,
        ordered_cell_ids=cell_ids,
        factory=factory,
    )
    identities = {
        cell_id: {
            "snapshot_digest_sha256": "a" * 64,
            "source_projection_sha256": "a" * 64,
            "snapshot_id": f"snapshot-{cell_id}",
        }
        for cell_id in cell_ids
    }
    observers = {
        cell_id: {"event_frontier": 17 + index, "cell_id": cell_id}
        for index, cell_id in enumerate(cell_ids)
    }

    result = capture_acs_native_aggregate_payload_from_cursor_v1(
        cursor,
        tmp_path / "cursor.acsaggregate",
        cell_ids=cell_ids,
        generation=4,
        tick=160,
        absolute_next_tick=320,
        batch_id="detached-cursor-test",
        segment_ordinal=1,
        chunk_bytes=7,
        authority_factory=lambda cell_id, _projection: {
            "source_projection_sha256": "a" * 64,
            "architecture_b_identity": identities[cell_id],
            "observer_continuation": observers[cell_id],
            "mechanics_checkpoint_digest": "b" * 64,
            "canonical_checkpoint_storage_digest": "c" * 64,
        },
    )
    capture = result["capture"]
    assert capture["header"]["field_schema_mode"] == "per_cell_v1"
    assert result["cursor_receipt"]["max_live_tensor_bearing_projections"] == 1
    assert capture["capture_pool_count"] == 1
    assert result["compact_result_rows"] == (
        {
            "cell_id": "cell-0",
            "metrics": {"population.alive": 10},
        },
        {
            "cell_id": "cell-1",
            "metrics": {"population.alive": 11},
        },
    )

    pipeline = ACSNativeDetachedSlotsV1(slots=1)
    try:
        token = pipeline.begin_capture(compute_frontier=320)
        pipeline.submit_capture(token, capture)
        receipt = pipeline.result(token)
        restored = load_acs_native_aggregate_checkpoint(
            Path(str(receipt["path"])),
            cell_id="cell-1",
            expected_source_projection_sha256="a" * 64,
        )
        assert torch.equal(
            restored["projection"]["tensors"]["active"],  # type: ignore[index]
            second["tensors"]["active"],  # type: ignore[index]
        )
        pipeline.acknowledge_durable(token)
    finally:
        pipeline.close()


def test_provider_fences_device_before_detached_cursor_capture() -> None:
    source = Path(providers.__file__).read_text(encoding="utf-8")
    physical_capture = source.index("build_direct_resident_physical_capture_v1(")
    physical_fence = source.rindex(
        "_synchronize_detached_acs_device_boundary_v1(",
        0,
        physical_capture,
    )
    assert physical_fence < physical_capture
    block_start = source.rindex(
        "tick_native_detached_acs\n                    and projection_cursor is not None"
    )
    block_end = source.index("capture_completed_perf_ns", block_start)
    block = source[block_start:block_end]
    assert block.index("_synchronize_detached_acs_device_boundary_v1(") < block.index(
        "capture_acs_native_aggregate_payload_from_cursor_v1("
    )


def test_direct_physical_host_finalizer_emits_bounded_writer_timing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        detached,
        "_finalize_capture_v1",
        lambda capture, *, cancel_event=None: {
            "status": "complete",
            "value": capture["value"],
        },
    )
    receipt = detached.finalize_direct_resident_physical_capture_v1(
        {"value": 7}
    )
    assert receipt["status"] == "complete"
    assert receipt["value"] == 7
    assert (
        receipt["detached_writer_started_perf_ns"]
        <= receipt["detached_writer_finished_perf_ns"]
    )
    assert receipt["detached_writer_wall_time_ns"] >= 0
    assert receipt["detached_writer_cpu_time_ns"] >= 0


def test_native_two_slab_copy_uses_events_not_per_chunk_stream_sync() -> None:
    source_path = (
        Path(providers.__file__).resolve().parents[1]
        / "native_runtime"
        / "native"
        / "acs_native_streamed_checkpoint_v1.cpp"
    )
    source = source_path.read_text(encoding="utf-8")
    start = source.index("std::uint64_t SlabPool::copy_tensor_chunks")
    end = source.index("pybind11::dict SlabPool::telemetry", start)
    block = source[start:end]
    assert ".copy_(source_slice, true)" in block
    assert '.attr("record")(stream)' in block
    assert '.attr("synchronize")()' in block
    assert "current_stream(source.device).synchronize" not in block
    assert ".copy_(source_slice, false)" not in block


def test_provider_drops_detached_tensor_aliases_before_owner_release_or_reuse() -> None:
    source = Path(providers.__file__).read_text(encoding="utf-8")
    comment_start = source.index("# The worker owns only the file path and scalar header;")
    block_start = source.rfind("source_projection_acs_token = acs_capture_token", 0, comment_start)
    block_end = source.index(
        "                if segment_end_next_tick == deliberate_interruption_tick:",
        block_start,
    )
    block = source[block_start:block_end]
    assert block.index("del immutable_source_projections") < block.index(
        "owner_to_close = resident_owner"
    )
    assert block.index("del detached_capture") < block.index(
        "owner_to_close = resident_owner"
    )


def test_streamed_resident_owner_never_reopens_consumed_source_descriptors() -> None:
    class ConsumedSourceDescriptors:
        def __iter__(self):
            raise AssertionError("consumed ACS source descriptors were revisited")

    prepared = SimpleNamespace(source_projections=ConsumedSourceDescriptors())

    assert providers._resident_owner_compact_source_projections_v1(
        prepared,
        bounded_streamed_preparation=True,
    ) == ()


def test_nonstreamed_resident_owner_preserves_compact_source_views() -> None:
    projections = ({"cell_id": "cell-a"}, {"cell_id": "cell-b"})
    prepared = SimpleNamespace(source_projections=projections)

    assert providers._resident_owner_compact_source_projections_v1(
        prepared,
        bounded_streamed_preparation=False,
    ) == projections


def test_provider_captures_declared_checkpoints_and_transfers_logical_owner_epochs() -> None:
    source = Path(providers.__file__).read_text(encoding="utf-8")
    assert "full_checkpoint_boundary = publish_checkpoint" in source
    assert "full_checkpoint_boundary = publish_checkpoint or owner_transition" not in source
    assert "transfer_owner_epoch_at_boundary(" in source
    assert "physical_owner_reused" in source

    reserve_start = source.index(
        "if acs_pipeline is not None and full_checkpoint_boundary:"
    )
    reserve_end = source.index("if segment_execution_observer is not None:", reserve_start)
    reserve_block = source[reserve_start:reserve_end]
    assert "acs_capture_token = acs_pipeline.begin_capture(" in reserve_block

    capture_start = source.index("source_consumer_required = bool(")
    capture_end = source.index("pending_acs_checkpoints.append", capture_start)
    capture_block = source[capture_start:capture_end]
    assert "segment_end_next_tick != request.total_ticks" in capture_block
    assert '"source_consumer_required"' in capture_block
    assert "acs_pipeline.submit_capture(" in capture_block

    prepare_start = source.index("if prepare_owner:")
    prepare_call = source.index("self._prepare_resident_segment_owner_v1(", prepare_start)
    release_call = source.index(
        "source_released_perf_ns = acs_pipeline.release_source(",
        prepare_call,
    )
    cursor_capture_call = source.index(
        "capture_acs_native_aggregate_payload_from_cursor_v1(",
        prepare_call,
    )
    assert prepare_call < cursor_capture_call < release_call


def test_cuda_detached_cursor_copy_has_explicit_producer_to_copy_stream_handoff() -> None:
    source = Path(detached.__file__).read_text(encoding="utf-8")
    copy_sites = [
        source.index("def _aggregate_capture_header("),
        source.index("capture_acs_native_aggregate_payload_from_cursor_v1("),
    ]
    for site in copy_sites:
        block_end = source.find("\ndef ", site + 1)
        block = source[site:] if block_end < 0 else source[site:block_end]
        assert "_record_producer_event_v1(" in block
        assert "producer_event=producer_event" in block


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or detached.streamed._native_slab_pool_type() is None,
    reason="requires an authorized CUDA device and the native slab extension",
)
def test_cuda_detached_cursor_capture_preserves_same_owner_continuation(
    tmp_path: Path,
) -> None:
    """Exercise the detached CUDA copy seam with one owner across two cells."""

    device = torch.device("cuda")
    owner_tensor = torch.empty(128, dtype=torch.float32, device=device)
    cell_ids = ("cell-0", "cell-1")

    def factory(index: int, cell_id: str) -> DetachedCellProjectionV1:
        # Both cells deliberately expose the same resident tensor.  The second
        # mutation models a nonterminal same-owner continuation after the first
        # detached capture has released its projection mapping.
        owner_tensor.fill_(float(index + 1))
        projection = {
            "schema_version": "cxxbackbone10_checkpoint_projection_v2",
            "metadata": {
                "world": "patch",
                "native_runtime_fixed_policy_family": "patch",
                "owner_continuation": True,
                "cell_index": index,
            },
            "event_cursor": 17 + index,
            "tensors": {"active": owner_tensor},
        }
        return DetachedCellProjectionV1(
            schema_version=DETACHED_CELL_PROJECTION_SCHEMA_VERSION_V1,
            cell_id=cell_id,
            cell_index=index,
            projection=projection,
            terminal_generation=4,
            terminal_tick=160,
            live_projection_bytes=int(owner_tensor.numel())
            * int(owner_tensor.element_size()),
        )

    cursor = CheckpointProjectionCursorV1.begin(
        owner=owner_tensor,
        coordinate=160,
        ordered_cell_ids=cell_ids,
        factory=factory,
    )
    identities = {
        cell_id: {
            "snapshot_digest_sha256": "a" * 64,
            "source_projection_sha256": "a" * 64,
            "snapshot_id": f"snapshot-{cell_id}",
        }
        for cell_id in cell_ids
    }
    capture_result = capture_acs_native_aggregate_payload_from_cursor_v1(
        cursor,
        tmp_path / "cuda-same-owner.acsaggregate",
        cell_ids=cell_ids,
        generation=4,
        tick=160,
        absolute_next_tick=320,
        batch_id="cuda-same-owner-test",
        segment_ordinal=1,
        chunk_bytes=64,
        require_native_extension=True,
        authority_factory=lambda cell_id, _projection: {
            "source_projection_sha256": "a" * 64,
            "architecture_b_identity": identities[cell_id],
            "observer_continuation": {"cell_id": cell_id, "frontier": 17},
            "mechanics_checkpoint_digest": "b" * 64,
            "canonical_checkpoint_storage_digest": "c" * 64,
        },
    )
    capture = capture_result["capture"]
    assert capture["native_slab_pool"]["slab_count"] == 2  # type: ignore[index]
    assert capture["native_slab_pool"]["pinned"] is True  # type: ignore[index]
    assert capture_result["cursor_receipt"]["max_live_tensor_bearing_projections"] == 1  # type: ignore[index]

    pipeline = ACSNativeDetachedSlotsV1(slots=1)
    try:
        token = pipeline.begin_capture(compute_frontier=320)
        pipeline.submit_capture(token, capture)
        receipt = pipeline.result(token)
        restored_first = load_acs_native_aggregate_checkpoint(
            Path(str(receipt["path"])),
            cell_id="cell-0",
            expected_source_projection_sha256="a" * 64,
        )
        restored_second = load_acs_native_aggregate_checkpoint(
            Path(str(receipt["path"])),
            cell_id="cell-1",
            expected_source_projection_sha256="a" * 64,
        )
        assert torch.equal(
            restored_first["projection"]["tensors"]["active"],  # type: ignore[index]
            torch.ones(128),
        )
        assert torch.equal(
            restored_second["projection"]["tensors"]["active"],  # type: ignore[index]
            torch.full((128,), 2.0),
        )
        pipeline.acknowledge_durable(token)
    finally:
        pipeline.close()


def test_provider_publishes_detached_terminal_rows_to_batch_metrics() -> None:
    source = Path(providers.__file__).read_text(encoding="utf-8")
    block_start = source.index("def finalize_oldest_acs_checkpoint(")
    block_end = source.index("        if tick_native:", block_start)
    block = source[block_start:block_end]
    assert "nonlocal checkpoint_payload, metrics_rows" in block
    assert block.index('capture.get("metrics_rows", ())') < block.index(
        "metrics_rows = projected_rows"
    )
    assert block.index('capture["metrics_rows"] = [') < block.index(
        '"metrics_rows": capture["metrics_rows"]'
    )


def test_streamed_authoritative_source_precedes_fresh_runner_initialization() -> None:
    source = Path(providers.__file__).read_text(encoding="utf-8")
    block_start = source.index("def streamed_request_factory(index: int)")
    block_end = source.index("            prepared = preparer(", block_start)
    block = source[block_start:block_end]
    assert block.index("source_projection = detached_source.require_payload()") < (
        block.index("build = runner_builder(")
    )
    assert '{"initialize_fresh_source": False}' in block
    assert block.index("except BaseException as exc:") < block.index(
        "def release_source(direct_request: Any)"
    )
    assert "fail_source(str(exc))" in block


def test_detached_cursor_production_accepts_preallocated_pageable_cpu_slabs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeNativeCpuSlabPool:
        def __init__(self, chunk_bytes: int, pinned: bool) -> None:
            assert pinned is False
            self.chunk_bytes = chunk_bytes

        def telemetry(self) -> dict[str, object]:
            return {
                "schema_version": "eiel_acs_native_slab_pool_v1",
                "chunk_bytes": self.chunk_bytes,
                "slab_count": 2,
                "pinned": False,
                "preallocated": True,
            }

        def copy_tensor_chunks(self, tensor: torch.Tensor, sink: object) -> int:
            raw = tensor.detach().contiguous().numpy().tobytes()
            count = 0
            for offset in range(0, len(raw), self.chunk_bytes):
                sink.write(raw[offset : offset + self.chunk_bytes])  # type: ignore[attr-defined]
                count += 1
            return count

    monkeypatch.setattr(
        detached.streamed,
        "_native_slab_pool_type",
        lambda: FakeNativeCpuSlabPool,
    )
    cell_ids = ("cell-0", "cell-1")
    projections = (_projection(0), _projection(1))

    def factory(index: int, cell_id: str) -> DetachedCellProjectionV1:
        projection = projections[index]
        return DetachedCellProjectionV1(
            schema_version=DETACHED_CELL_PROJECTION_SCHEMA_VERSION_V1,
            cell_id=cell_id,
            cell_index=index,
            projection=projection,
            terminal_generation=4,
            terminal_tick=160,
            live_projection_bytes=sum(
                int(tensor.numel()) * int(tensor.element_size())
                for tensor in projection["tensors"].values()  # type: ignore[union-attr]
            ),
        )

    cursor = CheckpointProjectionCursorV1.begin(
        owner=object(),
        coordinate=160,
        ordered_cell_ids=cell_ids,
        factory=factory,
    )
    result = capture_acs_native_aggregate_payload_from_cursor_v1(
        cursor,
        tmp_path / "cpu-native.acsaggregate",
        cell_ids=cell_ids,
        generation=4,
        tick=160,
        absolute_next_tick=320,
        batch_id="detached-cpu-native-test",
        segment_ordinal=1,
        chunk_bytes=7,
        require_native_extension=True,
        authority_factory=lambda cell_id, _projection: {
            "source_projection_sha256": "a" * 64,
            "architecture_b_identity": {"cell_id": cell_id},
            "observer_continuation": {"cell_id": cell_id},
            "mechanics_checkpoint_digest": "b" * 64,
            "canonical_checkpoint_storage_digest": "c" * 64,
        },
    )
    assert result["capture"]["native_slab_pool"]["slab_count"] == 2  # type: ignore[index]
    assert result["capture"]["native_slab_pool"]["pinned"] is False  # type: ignore[index]
    assert result["capture"]["native_slab_pool"]["preallocated"] is True  # type: ignore[index]


def test_detached_slots_two_slot_fifo_and_shared_pool(tmp_path: Path) -> None:
    pipeline = ACSNativeDetachedSlotsV1(slots=2)
    try:
        first = pipeline.begin_capture(compute_frontier=160)
        second = pipeline.begin_capture(compute_frontier=320)
        pipeline.submit_capture(first, _capture(tmp_path, 0))
        pipeline.submit_capture(second, _capture(tmp_path, 1))
        with pytest.raises(ACSNativeDetachedSlotError, match="FIFO"):
            pipeline.result(second)
        first_receipt = pipeline.result(first)
        pipeline.acknowledge_durable(first)
        second_receipt = pipeline.result(second)
        pipeline.acknowledge_durable(second)
        assert first_receipt["tick"] == 0
        assert second_receipt["tick"] == 1
        telemetry = pipeline.telemetry()
        assert telemetry["queue_occupancy_high_water"] == 2
        assert telemetry["writer_count"] == 1
        assert telemetry["capture_pool_count"] == 1
        assert telemetry["capture_pool_model"] == "one_shared_two_slab_pool_not_per_slot"
    finally:
        pipeline.close()


@pytest.mark.parametrize("slots", (1, 2))
def test_detached_focused_probe_qualifies_the_detached_route(
    slots: int,
) -> None:
    route = "f" * 64
    evidence = execute_detached_slots_focused_probe_v1(
        execution_route_sha256=route,
        obligations=frozenset(
            {"async_checkpoint_backpressure", "async_checkpoint_failure_latch"}
        ),
        immutable_snapshot_slots=slots,
    )
    covered, validated = validate_execution_route_acs_focused_evidence_v1(
        execution_route_sha256=route,
        covered_obligations={
            "async_checkpoint_backpressure",
            "async_checkpoint_failure_latch",
        },
        route_obligation_results=[evidence],
        immutable_snapshot_slots=slots,
    )
    assert evidence["exact"] is True
    assert covered == {
        "async_checkpoint_backpressure",
        "async_checkpoint_failure_latch",
    }
    assert validated is not None


def test_detached_slots_reject_three_and_executable_worker_metadata(tmp_path: Path) -> None:
    with pytest.raises(ACSNativeDetachedSlotError, match="exactly one or two"):
        ACSNativeDetachedSlotsV1(slots=3)

    pipeline = ACSNativeDetachedSlotsV1(slots=1)
    try:
        token = pipeline.begin_capture(compute_frontier=1)
        capture = _capture(tmp_path, 0)
        capture["callback"] = lambda: None
        with pytest.raises(ACSNativeDetachedSlotError, match="executable state"):
            pipeline.submit_capture(token, capture)
        pipeline.fail_capture(token, "callback crossed detached boundary")
        with pytest.raises(ACSNativeDetachedSlotError, match="worker failed"):
            pipeline.result(token)
        assert pipeline.telemetry()["failure_latched"] is True
    finally:
        pipeline.abort()


def test_detached_worker_failure_latches_and_closes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    pipeline = ACSNativeDetachedSlotsV1(slots=1)
    real_finalize = detached._finalize_capture_v1

    def fail_finalize(
        _capture: object,
        *,
        cancel_event: threading.Event | None = None,
    ) -> dict[str, object]:
        del cancel_event
        raise RuntimeError("injected detached writer failure")

    monkeypatch.setattr(detached, "_finalize_capture_v1", fail_finalize)
    try:
        token = pipeline.begin_capture(compute_frontier=1)
        pipeline.submit_capture(token, _capture(tmp_path, 0))
        with pytest.raises(ACSNativeDetachedSlotError, match="worker failed"):
            pipeline.result(token)
        with pytest.raises(ACSNativeDetachedSlotError, match="failure is latched"):
            pipeline.begin_capture(compute_frontier=2)
        telemetry = pipeline.telemetry()
        assert telemetry["failure_latched"] is True
        assert telemetry["queue_occupancy"] == 1
        assert telemetry["writer_alive"] is False
        failed_paths = telemetry["failure_capture_paths"]
        assert len(failed_paths) == 1
        assert Path(str(failed_paths[0])).is_file()
        assert not list(tmp_path.glob("*.capture.tmp"))
        assert not list(tmp_path.glob("*.acsaggregate"))
    finally:
        monkeypatch.setattr(detached, "_finalize_capture_v1", real_finalize)
        pipeline.abort()


def test_provider_binds_detached_receipt_without_reaccessing_projections(
    tmp_path: Path,
) -> None:
    provider = ProductionEngineLifecycleV1()
    # The fixture intentionally omits a complete lifecycle-event declaration;
    # the injected legacy writer seam allows the provider's existing fail-closed
    # compatibility rule to bind an empty observer cursor for this host test.
    provider._checkpoint_writer = object()
    request = ResidentBatchRequestV1(
        profile_id="acs-native-test",
        batch_id="detached-provider-test",
        cell_ids=("cell-0", "cell-1"),
        generations=4,
        ticks_per_generation=160,
        checkpoint_milestones=(4,),
        parameter_arrays=(),
        seed_arrays=(),
        execution_route_sha256="b" * 64,
        acs_checkpoint_slots=2,
        acs_capture_instrumentation=True,
        native_streamed_checkpoint=True,
    )
    projections = (_projection(0), _projection(1))
    digests = {cell_id: "a" * 64 for cell_id in request.cell_ids}
    provider._layered_checkpoint_digest_v1 = (  # type: ignore[method-assign]
        lambda _projection, _layer: "b" * 64
    )
    authorities = provider._aggregate_checkpoint_authorities_v1(
        request=request,
        source_projections=projections,
        generation=4,
        tick=0,
        absolute_next_tick=None,
        segment_ordinal=1,
        expected_source_projection_digest_sha256s=digests,
    )
    capture = _capture(tmp_path, 0)
    pipeline = ACSNativeDetachedSlotsV1(slots=2)
    try:
        token = pipeline.begin_capture(compute_frontier=160)
        pipeline.submit_capture(token, capture)
        receipt = pipeline.result(token)
        checkpoint_set = provider._aggregate_checkpoint_set_from_detached_receipt_v1(
            request=request,
            receipt=receipt,
            authorities=authorities,
            writer_telemetry={"owner_snapshot_capture_mode": "detached_file_backed_native_v1"},
        )
        assert len(checkpoint_set["bundle_rows"]) == 2
        assert checkpoint_set["bundle_rows"][0]["path"] == checkpoint_set["bundle_rows"][1]["path"]
        assert checkpoint_set["native_checkpoint_receipts"]["cell-1"]["path"] == str(
            tmp_path / "checkpoint-0.acsaggregate"
        )
        pipeline.acknowledge_durable(token)
    finally:
        pipeline.close()


def test_provider_rejects_worker_source_digest_mismatch_before_bundle_binding(
    tmp_path: Path,
) -> None:
    provider = ProductionEngineLifecycleV1()
    provider._checkpoint_writer = object()
    request = ResidentBatchRequestV1(
        profile_id="acs-native-test",
        batch_id="detached-provider-digest-mismatch-test",
        cell_ids=("cell-0", "cell-1"),
        generations=4,
        ticks_per_generation=160,
        checkpoint_milestones=(4,),
        parameter_arrays=(),
        seed_arrays=(),
        execution_route_sha256="b" * 64,
        acs_checkpoint_slots=2,
        acs_capture_instrumentation=True,
        native_streamed_checkpoint=True,
    )
    projections = (_projection(0), _projection(1))
    digests = {cell_id: "a" * 64 for cell_id in request.cell_ids}
    provider._layered_checkpoint_digest_v1 = (  # type: ignore[method-assign]
        lambda _projection, _layer: "b" * 64
    )
    authorities = provider._aggregate_checkpoint_authorities_v1(
        request=request,
        source_projections=projections,
        generation=4,
        tick=0,
        absolute_next_tick=None,
        segment_ordinal=1,
        expected_source_projection_digest_sha256s=digests,
    )
    pipeline = ACSNativeDetachedSlotsV1(slots=2)
    try:
        token = pipeline.begin_capture(compute_frontier=160)
        pipeline.submit_capture(token, _capture(tmp_path, 0))
        receipt = dict(pipeline.result(token))
        cells = dict(receipt["cells"])
        cells["cell-1"] = {
            **dict(cells["cell-1"]),
            "source_projection_sha256": "d" * 64,
        }
        receipt["cells"] = cells
        with pytest.raises(
            providers.CampaignProviderError,
            match="worker/source authority digest differs: cell-1",
        ):
            provider._aggregate_checkpoint_set_from_detached_receipt_v1(
                request=request,
                receipt=receipt,
                authorities=authorities,
                writer_telemetry={
                    "owner_snapshot_capture_mode": "detached_file_backed_native_v1"
                },
            )
        pipeline.acknowledge_durable(token)
    finally:
        pipeline.close()


def test_tick_native_provider_uses_detached_slots_for_two_checkpoint_frontiers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Reuse the provider's strict filesystem harness so the test exercises the
    # real aggregate capture branch, not only the slot state machine.
    from tests.test_eiel_campaign_patch_provider_execution_v1 import (
        _Context,
        _plan,
        _strict_filesystem_provider_harness,
        _tick_request,
    )
    import engine.campaigns.providers as providers

    monkeypatch.setattr(
        providers,
        "_mark_native_streamed_capture_owner_v1",
        lambda _prepared: None,
    )
    provider, _state = _strict_filesystem_provider_harness()
    provider._checkpoint_digest = lambda projection: (
        f"{int(projection['metadata']['generation']):064x}"
    )
    original_request_builder = provider._request_builder
    original_projector = provider._projector
    assert original_request_builder is not None
    assert original_projector is not None

    def request_builder(*args: object, **kwargs: object) -> object:
        request = original_request_builder(*args, **kwargs)
        projection = dict(request.import_projection)  # type: ignore[attr-defined]
        metadata = dict(projection.get("metadata", {}))
        metadata.update(
            {
                "typed_lifecycle_event_next_sequence": 0,
                "typed_lifecycle_event_sequence_base": 0,
                "event_history_frontier": 0,
            }
        )
        projection["metadata"] = metadata
        request.import_projection = projection  # type: ignore[attr-defined]
        return request

    def projector(prepared: object, **kwargs: object) -> object:
        kwargs.pop("generation_base_override", None)
        kwargs.pop("bounded_projection_cursor", None)
        return original_projector(prepared, **kwargs)

    monkeypatch.setattr(provider, "_request_builder", request_builder)
    monkeypatch.setattr(provider, "_projector", projector)
    plan = _plan()
    plan["execution"] = {
        "acs_checkpoint_slots": 2,
        "acs_capture_instrumentation": True,
        "native_streamed_checkpoint": True,
    }
    request = replace(
        _tick_request(total_ticks=320, checkpoints=(160, 320), restore_tick=None),
        acs_checkpoint_slots=2,
        acs_capture_instrumentation=True,
        native_streamed_checkpoint=True,
    )

    result = provider.execute_batch(plan, request, _Context(tmp_path))

    assert result["status"] == "completed"
    telemetry = result["result"]["acs_checkpoint_staging"]
    assert telemetry["capture_mode"] == "detached_file_backed_native_v1"
    assert telemetry["slots"] == 2
    assert telemetry["queue_occupancy_high_water"] == 2
    assert telemetry["worker_accepts_cuda_objects"] is False
    assert telemetry["worker_accepts_callbacks"] is False
    assert telemetry["capture_pool_count"] == 1
    assert telemetry["failure_latched"] is False
    assert telemetry["closed"] is True


def _frontier_checkpoint(path: Path, coordinate: int, identity: str) -> dict[str, object]:
    return {
        "time_authority": "absolute_tick_v1",
        "absolute_tick": coordinate,
        "next_absolute_tick": coordinate,
        "checkpoint_identity_sha256": identity,
        "cell_bundles": [{"cell_id": "cell-0", "path": str(path)}],
    }


def _frontier_release_fixture(
    tmp_path: Path,
) -> tuple[SimpleNamespace, dict[int, Path], dict[int, str]]:
    run_root = tmp_path / "frontier-run"
    batch_root = run_root / "batches" / "batch-0000"
    checkpoint_root = batch_root / "checkpoint_native_aggregate_v1"
    checkpoint_root.mkdir(parents=True)
    coordinates = (2500, 3360, 4480)
    identities = {coordinate: f"{coordinate:064x}" for coordinate in coordinates}
    paths = {
        coordinate: checkpoint_root / f"tick-{coordinate:012d}.acsaggregate"
        for coordinate in coordinates
    }
    for coordinate, path in paths.items():
        path.write_bytes(f"checkpoint-{coordinate}".encode("ascii"))

    inventory = []
    for coordinate, path in paths.items():
        row = build_sealed_artifact_inventory_row_v1(
            run_root=run_root,
            path=path,
            artifact_family="checkpoint.bundle_v2",
            retention_class="R1",
            owner_writer="frontier-test",
            producer_checkpoint_identity_sha256=identities[coordinate],
        )
        observed = path.stat()
        row["producer_sealed_stat"] = {
            "device": int(observed.st_dev),
            "inode": int(observed.st_ino),
            "bytes": int(observed.st_size),
            "mtime_ns": int(observed.st_mtime_ns),
            "ctime_ns": int(observed.st_ctime_ns),
        }
        inventory.append(row)
    manifest_path = batch_root / "batch_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "cell_ids": ["cell-0"],
                "artifact_inventory_complete": True,
                "artifact_inventory": inventory,
            }
        ),
        encoding="utf-8",
    )
    return (
        SimpleNamespace(
            run_root=run_root,
            batch_root=batch_root,
            manifest_path=manifest_path,
        ),
        paths,
        identities,
    )


def test_delayed_async_2500_publication_preserves_live_3360_continuation(
    tmp_path: Path,
) -> None:
    context, paths, identities = _frontier_release_fixture(tmp_path)
    before_publication = {
        str(coordinate): _frontier_checkpoint(
            paths[coordinate], coordinate, identities[coordinate]
        )
        for coordinate in (2500, 3360)
    }
    published_2500 = _frontier_checkpoint(paths[2500], 2500, identities[2500])

    after_2500 = providers._tick_native_catalog_after_async_publication_v1(
        before_publication,
        published_coordinate=2500,
        published_checkpoint=published_2500,
        requested_checkpoint_ticks=(2500, 5000),
    )

    assert after_2500["2500"] == published_2500
    assert after_2500["3360"]["next_absolute_tick"] == 3360
    assert after_2500["3360"]["cell_bundles"][0]["path"] == str(paths[3360])  # type: ignore[index]
    released = providers._release_tick_native_transient_storage_v1(
        context=context,
        prior_checkpoint_catalog=before_publication,
        retained_checkpoint_catalog=after_2500,
        replacement_checkpoint=published_2500,
        requested_checkpoint_ticks=(2500, 5000),
        release_source_import=False,
    )

    assert released == []
    assert paths[2500].is_file()
    assert paths[3360].is_file()
    manifest = json.loads(context.manifest_path.read_text(encoding="utf-8"))
    assert str(paths[3360].relative_to(context.run_root)) in {
        row["path"] for row in manifest["artifact_inventory"]
    }


def test_delayed_async_publication_rebinds_manifest_to_newest_frontier(
    tmp_path: Path,
) -> None:
    paths = {
        coordinate: tmp_path / f"tick-{coordinate}.acsaggregate"
        for coordinate in (2500, 3360)
    }
    checkpoints = {
        str(coordinate): _frontier_checkpoint(
            paths[coordinate], coordinate, f"{coordinate:064x}"
        )
        for coordinate in (2500, 3360)
    }
    recorded: list[int] = []
    context = SimpleNamespace(
        record_checkpoint=lambda checkpoint: recorded.append(
            int(checkpoint["next_absolute_tick"])
        )
    )

    frontier = providers._record_tick_native_async_publication_v1(
        context,
        published_coordinate=2500,
        published_checkpoint=checkpoints["2500"],
        postpublication_catalog=checkpoints,
    )

    assert frontier == 3360
    assert recorded == [2500, 3360]


def test_later_4480_replacement_releases_3360_only_after_source_frontier_moves(
    tmp_path: Path,
) -> None:
    context, paths, identities = _frontier_release_fixture(tmp_path)
    catalog_after_2500 = {
        str(coordinate): _frontier_checkpoint(
            paths[coordinate], coordinate, identities[coordinate]
        )
        for coordinate in (2500, 3360)
    }
    published_4480 = _frontier_checkpoint(paths[4480], 4480, identities[4480])
    catalog_after_4480 = providers._tick_native_catalog_after_async_publication_v1(
        catalog_after_2500,
        published_coordinate=4480,
        published_checkpoint=published_4480,
        requested_checkpoint_ticks=(2500, 5000),
    )

    assert "3360" not in catalog_after_4480
    assert catalog_after_4480["2500"]["next_absolute_tick"] == 2500
    assert catalog_after_4480["4480"] == published_4480
    released = providers._release_tick_native_transient_storage_v1(
        context=context,
        prior_checkpoint_catalog=catalog_after_2500,
        retained_checkpoint_catalog=catalog_after_4480,
        replacement_checkpoint=published_4480,
        requested_checkpoint_ticks=(2500, 5000),
        release_source_import=False,
    )

    assert paths[3360].exists() is False
    assert paths[2500].is_file()
    assert paths[4480].is_file()
    assert [row["coordinate"] for row in released] == [3360]


def test_async_release_rejects_attempt_to_evict_protected_newer_descriptor(
    tmp_path: Path,
) -> None:
    context, paths, identities = _frontier_release_fixture(tmp_path)
    prior_catalog = {
        str(coordinate): _frontier_checkpoint(
            paths[coordinate], coordinate, identities[coordinate]
        )
        for coordinate in (2500, 3360)
    }
    published_2500 = _frontier_checkpoint(paths[2500], 2500, identities[2500])

    with pytest.raises(
        providers.CampaignProviderError,
        match="cannot release checkpoint 3360 newer than replacement 2500",
    ):
        providers._release_tick_native_transient_storage_v1(
            context=context,
            prior_checkpoint_catalog=prior_catalog,
            retained_checkpoint_catalog={},
            replacement_checkpoint=published_2500,
            requested_checkpoint_ticks=(5000,),
            release_source_import=False,
        )

    assert paths[3360].is_file()
    assert paths[2500].is_file()
    manifest = json.loads(context.manifest_path.read_text(encoding="utf-8"))
    assert str(paths[3360].relative_to(context.run_root)) in {
        row["path"] for row in manifest["artifact_inventory"]
    }
