from __future__ import annotations

from dataclasses import FrozenInstanceError, fields, replace
import gzip
import hashlib
import json
from pathlib import Path
import struct
import threading
import time
from typing import Any

import pytest
import torch

from engine.native_runtime import (
    cuda_v2_compact_science_capture_v1 as compact_capture_module,
)
from engine.native_runtime.compact_science_async_publication_v1 import (
    CompactScienceAsyncPublicationError,
    CompactScienceAsyncPublicationV1,
    CompactSciencePublicationTokenV1,
)
from engine.native_runtime.cuda_v2_compact_science_capture_v1 import (
    CompactScienceCaptureError,
    FrozenCompactScienceDispatchHostV1,
    freeze_compact_science_dispatch_host_v1,
    materialize_frozen_compact_science_dispatch_v1,
    publish_frozen_compact_science_dispatch_v1,
    remove_compact_science_stage_v1,
    write_compact_science_dispatch_v1,
)
from engine.perf.cuda_full_evolution_v2.direct_sparse_runtime import (
    COMPACT_SCIENCE_AGENT_METRIC_FIELDS_V1,
    COMPACT_SCIENCE_BIRTH_PROVENANCE_ENCODING_V1,
    CudaFullEvolutionCompactScienceCaptureV1,
)
from engine.perf.cuda_full_evolution_v2.runtime import (
    CudaFullEvolutionSparseEventDispatchV3,
)


CONTROL_SET = "3" * 64


def _capture_fixture() -> tuple[
    CudaFullEvolutionSparseEventDispatchV3,
    CudaFullEvolutionCompactScienceCaptureV1,
    dict[str, dict[str, Any]],
]:
    death_key = 1 << 6
    birth_key = 2 << 6
    packed_events = b"".join(
        (
            struct.pack("<QQQQ", 11, 0, 0, death_key),
            struct.pack("<QQQQ", 12, 7, 9, birth_key),
        )
    )
    dispatch = CudaFullEvolutionSparseEventDispatchV3(
        packed_events=packed_events,
        world_offsets=(0, 2),
        event_counts=(2,),
        final_next_agent_ids=(13,),
        world_uids=(101,),
        start_next_absolute_tick=8,
        end_next_absolute_tick=9,
        tick_count=1,
        drain_elapsed_seconds=0.0,
        count_projection_elapsed_seconds=0.0,
        compaction_and_transfer_elapsed_seconds=0.0,
        content_hash_elapsed_seconds=0.0,
        content_sha256="5" * 64,
        artifact_sha256="6" * 64,
        manifest_sha256="7" * 64,
    )
    metrics = torch.zeros(
        (2, len(COMPACT_SCIENCE_AGENT_METRIC_FIELDS_V1)), dtype=torch.int64
    )
    for index in range(metrics.shape[1]):
        metrics[0, index] = -9 if index == 17 else index + 1
    provenance = torch.zeros((2, 132), dtype=torch.uint8)
    provenance[1, 0] = 1
    provenance[1, 1] = 2
    provenance[1, 128:132] = torch.tensor(
        [0x78, 0x56, 0x34, 0x12], dtype=torch.uint8
    )
    capture = CudaFullEvolutionCompactScienceCaptureV1(
        event_agent_metrics=metrics,
        event_birth_provenance=provenance,
        metric_fields=COMPACT_SCIENCE_AGENT_METRIC_FIELDS_V1,
        birth_provenance_encoding=(
            COMPACT_SCIENCE_BIRTH_PROVENANCE_ENCODING_V1
        ),
        start_next_absolute_tick=8,
        end_next_absolute_tick=9,
        world_uids=(101,),
        world_offsets=(0, 2),
        event_content_sha256="5" * 64,
    )
    cells = {
        "alpha": {
            "science_logical_world_id": 101,
            "science_ownership": {"study_id": "GENERIC-A"},
            "science_world_sha256": {
                "algorithm": "sha256",
                "hex": "8" * 64,
            },
            "science_acquisition_sha256": {
                "algorithm": "sha256",
                "hex": "9" * 64,
            },
        }
    }
    return dispatch, capture, cells


def _frozen_fixture() -> FrozenCompactScienceDispatchHostV1:
    dispatch, capture, cells = _capture_fixture()
    return freeze_compact_science_dispatch_host_v1(
        dispatch=dispatch,
        capture=capture,
        world_cells=cells,
        native_world_control_set_sha256=CONTROL_SET,
    )


def _span(
    frozen: FrozenCompactScienceDispatchHostV1,
    start: int,
) -> FrozenCompactScienceDispatchHostV1:
    return replace(
        frozen,
        start_next_absolute_tick=start,
        end_next_absolute_tick=start + 1,
        tick_count=1,
        event_content_sha256=hashlib.sha256(str(start).encode("ascii")).hexdigest(),
    )


def _synthetic_receipt(
    staging_root: str | Path,
    frozen: FrozenCompactScienceDispatchHostV1,
) -> dict[str, Any]:
    root = Path(staging_root).resolve()
    root.mkdir()
    (root / "manifest.json").write_text("{}", encoding="ascii")
    return {
        "path": str(root),
        "start_next_absolute_tick": frozen.start_next_absolute_tick,
        "end_next_absolute_tick": frozen.end_next_absolute_tick,
        "event_content_sha256": frozen.event_content_sha256,
    }


def _publish_synthetic(
    token: CompactSciencePublicationTokenV1,
    staged_receipt: dict[str, Any],
) -> dict[str, Any]:
    Path(staged_receipt["path"]).rename(token.destination)
    return {**staged_receipt, "path": token.destination}


def _assert_tensor_free(value: object) -> None:
    assert not isinstance(value, torch.Tensor)
    if isinstance(value, tuple):
        for child in value:
            _assert_tensor_free(child)


def test_frozen_dispatch_materializes_exact_birth_and_death_bytes(
    tmp_path: Path,
) -> None:
    frozen = _frozen_fixture()

    receipt = materialize_frozen_compact_science_dispatch_v1(
        tmp_path / "staged", frozen=frozen
    )

    root = Path(receipt["path"])
    events = gzip.decompress((root / receipt["event_file"]).read_bytes())
    death_metrics = gzip.decompress((root / receipt["metric_file"]).read_bytes())
    birth_provenance = gzip.decompress(
        (root / receipt["birth_provenance_file"]).read_bytes()
    )
    expected_death_metrics = b"".join(
        struct.pack("<q" if index == 17 else "<I", -9 if index == 17 else index + 1)
        for index in range(len(COMPACT_SCIENCE_AGENT_METRIC_FIELDS_V1))
    )
    expected_birth_provenance = bytes([0x21] + [0] * 63) + bytes(
        [0x78, 0x56, 0x34, 0x12]
    )

    assert events == frozen.packed_events
    assert death_metrics == expected_death_metrics
    assert birth_provenance == expected_birth_provenance
    assert receipt["death_event_count"] == 1
    assert receipt["birth_event_count"] == 1
    assert receipt["metric_bytes"] == 88
    assert receipt["birth_provenance_bytes"] == 68


def test_stage_materialization_does_not_use_named_tempfiles(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fail_mkstemp(*_args: Any, **_kwargs: Any) -> None:
        pytest.fail("unpublished compact stage must not use named tempfiles")

    monkeypatch.setattr(compact_capture_module.tempfile, "mkstemp", fail_mkstemp)
    receipt = materialize_frozen_compact_science_dispatch_v1(
        tmp_path / "stage-no-tempfiles", frozen=_frozen_fixture()
    )

    root = Path(receipt["path"])
    assert {path.name for path in root.iterdir()} == {
        receipt["event_file"],
        receipt["metric_file"],
        receipt["birth_provenance_file"],
        "manifest.json",
    }


def test_compact_manifest_retains_source_sink_receipt_binding(
    tmp_path: Path,
) -> None:
    dispatch, capture, cells = _capture_fixture()
    source_sink_receipt_sha256 = "a" * 64
    frozen = freeze_compact_science_dispatch_host_v1(
        dispatch=dispatch,
        capture=capture,
        world_cells=cells,
        native_world_control_set_sha256=CONTROL_SET,
        source_sink_receipt_sha256=source_sink_receipt_sha256,
    )

    receipt = materialize_frozen_compact_science_dispatch_v1(
        tmp_path / "staged-with-source-binding", frozen=frozen
    )

    manifest = json.loads(
        (Path(receipt["path"]) / "manifest.json").read_text(encoding="utf-8")
    )
    assert receipt["source_sink_receipt_sha256"] == source_sink_receipt_sha256
    assert manifest["source_sink_receipt_sha256"] == source_sink_receipt_sha256


def test_frozen_dispatch_capsule_is_tensor_free_and_immutable() -> None:
    dispatch, capture, cells = _capture_fixture()
    frozen = freeze_compact_science_dispatch_host_v1(
        dispatch=dispatch,
        capture=capture,
        world_cells=cells,
        native_world_control_set_sha256=CONTROL_SET,
    )
    original_metric_bytes = frozen.event_agent_metrics_i64le
    original_logical_worlds = frozen.logical_worlds_json

    for field in fields(frozen):
        _assert_tensor_free(getattr(frozen, field.name))
    assert all(
        isinstance(getattr(frozen, field.name), (bytes, int, str, tuple))
        for field in fields(frozen)
    )
    with pytest.raises(FrozenInstanceError):
        frozen.event_count = 0  # type: ignore[misc]

    capture.event_agent_metrics.fill_(999)
    cells["alpha"]["science_ownership"]["study_id"] = "MUTATED"
    assert frozen.event_agent_metrics_i64le == original_metric_bytes
    assert frozen.logical_worlds_json == original_logical_worlds
    assert json.loads(frozen.logical_worlds_json)[0]["science_ownership"] == {
        "study_id": "GENERIC-A"
    }


def test_frozen_dispatch_rejects_unbounded_logical_world_metadata() -> None:
    dispatch, capture, cells = _capture_fixture()
    cells["alpha"]["science_ownership"] = {"payload": "x" * (2 * 1024 * 1024)}

    with pytest.raises(
        CompactScienceCaptureError,
        match="logical-world metadata exceeds",
    ):
        freeze_compact_science_dispatch_host_v1(
            dispatch=dispatch,
            capture=capture,
            world_cells=cells,
            native_world_control_set_sha256=CONTROL_SET,
        )


def test_real_stage_publication_is_atomic_and_strips_private_seal(
    tmp_path: Path,
) -> None:
    staged = materialize_frozen_compact_science_dispatch_v1(
        tmp_path / ".stage", frozen=_frozen_fixture()
    )

    published = publish_frozen_compact_science_dispatch_v1(
        staged, destination=tmp_path / "dispatch"
    )

    assert published["path"] == str((tmp_path / "dispatch").resolve())
    assert "_stage_seal_v1" not in published
    assert not (tmp_path / ".stage").exists()
    assert (tmp_path / "dispatch" / "manifest.json").is_file()


def test_parent_fsync_failure_rolls_canonical_publication_back(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    stage_root = tmp_path / ".stage-parent-fsync"
    destination = tmp_path / "dispatch-parent-fsync"
    staged = materialize_frozen_compact_science_dispatch_v1(
        stage_root,
        frozen=_frozen_fixture(),
    )

    def fail_parent_fsync(_descriptor: int) -> None:
        raise OSError("injected publication parent fsync failure")

    monkeypatch.setattr(compact_capture_module.os, "fsync", fail_parent_fsync)
    with pytest.raises(
        CompactScienceCaptureError,
        match="canonical root was rolled back",
    ):
        publish_frozen_compact_science_dispatch_v1(
            staged,
            destination=destination,
        )

    assert not destination.exists()
    assert stage_root.is_dir()
    monkeypatch.undo()
    remove_compact_science_stage_v1(stage_root)


def test_stage_materialization_serializes_same_stage_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cleanup_started = threading.Event()
    cleanup_finished = threading.Event()
    cleanup_errors: list[BaseException] = []
    cleanup_thread: threading.Thread | None = None
    original_exclusive_json = compact_capture_module._write_exclusive_json_v1

    def concurrent_cleanup(path: str | Path) -> None:
        cleanup_started.set()
        try:
            remove_compact_science_stage_v1(path)
        except BaseException as error:  # pragma: no cover - diagnostic path
            cleanup_errors.append(error)
        finally:
            cleanup_finished.set()

    def observed_exclusive_json(
        path: Path,
        value: dict[str, Any],
        **kwargs: Any,
    ) -> None:
        nonlocal cleanup_thread
        if path.name == "manifest.json":
            cleanup_thread = threading.Thread(
                target=concurrent_cleanup,
                args=(path.parent,),
            )
            cleanup_thread.start()
            assert cleanup_started.wait(timeout=5)
            # The stage owner still holds its exact-path lock while the
            # manifest is being published.
            assert cleanup_finished.wait(timeout=0.05) is False
        original_exclusive_json(path, value, **kwargs)

    monkeypatch.setattr(
        compact_capture_module,
        "_write_exclusive_json_v1",
        observed_exclusive_json,
    )
    stage_root = tmp_path / ".stage-lock"
    staged = materialize_frozen_compact_science_dispatch_v1(
        stage_root,
        frozen=_frozen_fixture(),
    )
    assert staged["path"] == str(stage_root.resolve())
    assert cleanup_thread is not None
    cleanup_thread.join(timeout=5)
    assert not cleanup_thread.is_alive()
    assert cleanup_errors == []
    assert cleanup_finished.is_set()
    assert not stage_root.exists()


def test_real_stage_publication_rejects_changed_bytes_and_destination_symlink(
    tmp_path: Path,
) -> None:
    staged = materialize_frozen_compact_science_dispatch_v1(
        tmp_path / ".stage-changed", frozen=_frozen_fixture()
    )
    event_path = Path(staged["path"]) / staged["event_file"]
    payload = bytearray(event_path.read_bytes())
    payload[-1] ^= 1
    event_path.chmod(0o600)
    event_path.write_bytes(payload)
    event_path.chmod(0o400)
    with pytest.raises(CompactScienceCaptureError, match="bytes differ"):
        publish_frozen_compact_science_dispatch_v1(
            staged, destination=tmp_path / "dispatch-changed"
        )

    staged_symlink = materialize_frozen_compact_science_dispatch_v1(
        tmp_path / ".stage-symlink", frozen=_frozen_fixture()
    )
    outside = tmp_path / "outside"
    outside.mkdir()
    (tmp_path / "dispatch-symlink").symlink_to(outside, target_is_directory=True)
    with pytest.raises(CompactScienceCaptureError, match="already exists"):
        publish_frozen_compact_science_dispatch_v1(
            staged_symlink, destination=tmp_path / "dispatch-symlink"
        )


@pytest.mark.parametrize("extra_kind", ["file", "directory"])
def test_real_stage_publication_rejects_every_extra_member(
    tmp_path: Path,
    extra_kind: str,
) -> None:
    staged = materialize_frozen_compact_science_dispatch_v1(
        tmp_path / f".stage-extra-{extra_kind}",
        frozen=_frozen_fixture(),
    )
    root = Path(staged["path"])
    root.chmod(0o700)
    if extra_kind == "file":
        (root / "extra.bin").write_bytes(b"not declared")
    else:
        (root / "extra").mkdir()
    root.chmod(0o500)
    try:
        with pytest.raises(CompactScienceCaptureError, match="member set"):
            publish_frozen_compact_science_dispatch_v1(
                staged,
                destination=tmp_path / f"dispatch-extra-{extra_kind}",
            )
    finally:
        remove_compact_science_stage_v1(root)


def test_synchronous_wrapper_rejects_dangling_destination_symlink(
    tmp_path: Path,
) -> None:
    dispatch, capture, cells = _capture_fixture()
    destination = tmp_path / "dispatch-symlink"
    outside = tmp_path / "outside-target"
    destination.symlink_to(outside, target_is_directory=True)

    with pytest.raises(CompactScienceCaptureError, match="already exists"):
        write_compact_science_dispatch_v1(
            destination,
            dispatch=dispatch,
            capture=capture,
            world_cells=cells,
            native_world_control_set_sha256=CONTROL_SET,
        )

    assert not outside.exists()


def test_async_submission_rejects_destination_symlink_before_staging(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "dispatch-symlink"
    outside = tmp_path / "outside"
    outside.mkdir()
    destination.symlink_to(outside, target_is_directory=True)
    publication = CompactScienceAsyncPublicationV1(workers=1, pending_slots=1)
    try:
        with pytest.raises(
            CompactScienceAsyncPublicationError,
            match="destination already exists",
        ):
            publication.submit(
                frozen=_frozen_fixture(),
                destination=destination,
            )
        assert not tuple(tmp_path.glob(".*.async-stage-*"))
    finally:
        publication.abort()


def test_async_submission_rejects_non_host_field_before_enqueue(
    tmp_path: Path,
) -> None:
    frozen = _frozen_fixture()
    object.__setattr__(frozen, "logical_worlds_json", torch.zeros(1, device="cpu"))
    publication = CompactScienceAsyncPublicationV1(workers=1, pending_slots=1)
    try:
        with pytest.raises(
            CompactScienceAsyncPublicationError,
            match="immutable host bytes",
        ):
            publication.submit(
                frozen=frozen,
                destination=tmp_path / "dispatch-invalid-host",
            )
        assert publication.telemetry()["submitted_jobs"] == 0
        assert publication.telemetry()["queued_cuda_references"] == 0
    finally:
        publication.abort()


def test_async_submission_does_not_create_through_parent_symlink(
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(outside, target_is_directory=True)
    publication = CompactScienceAsyncPublicationV1(workers=1, pending_slots=1)
    try:
        with pytest.raises(
            CompactScienceAsyncPublicationError,
            match="destination is unsafe",
        ):
            publication.submit(
                frozen=_frozen_fixture(),
                destination=linked_parent / "missing" / "dispatch",
            )
        assert not (outside / "missing").exists()
    finally:
        publication.abort()


def test_async_publication_uses_two_slots_and_backpressures_third_submission(
    tmp_path: Path,
) -> None:
    frozen = _frozen_fixture()
    started = {index: threading.Event() for index in range(3)}
    release = {index: threading.Event() for index in range(3)}

    def delayed_materialize(
        staging_root: str | Path,
        *,
        frozen: FrozenCompactScienceDispatchHostV1,
    ) -> dict[str, Any]:
        index = frozen.start_next_absolute_tick
        started[index].set()
        assert release[index].wait(timeout=5)
        return _synthetic_receipt(staging_root, frozen)

    publication = CompactScienceAsyncPublicationV1(
        workers=2,
        pending_slots=2,
        materialize=delayed_materialize,
    )
    try:
        first = publication.submit(
            frozen=_span(frozen, 0), destination=tmp_path / "dispatch-0"
        )
        second = publication.submit(
            frozen=_span(frozen, 1), destination=tmp_path / "dispatch-1"
        )
        assert started[0].wait(timeout=5)
        assert started[1].wait(timeout=5)
        assert publication.has_capacity is False
        with pytest.raises(
            CompactScienceAsyncPublicationError,
            match="requires producer backpressure",
        ):
            publication.submit(
                frozen=_span(frozen, 2), destination=tmp_path / "dispatch-2"
            )

        timer = threading.Timer(0.05, release[0].set)
        timer.start()
        staged = dict(publication.result(first, producer_backpressure=True))
        timer.join(timeout=5)
        publication.acknowledge_published(
            first,
            published_receipt=_publish_synthetic(first, staged),
        )
        third = publication.submit(
            frozen=_span(frozen, 2), destination=tmp_path / "dispatch-2"
        )
        assert started[2].wait(timeout=5)

        release[1].set()
        release[2].set()
        for token in (second, third):
            staged = dict(publication.result(token))
            publication.acknowledge_published(
                token,
                published_receipt=_publish_synthetic(token, staged),
            )
        telemetry = publication.telemetry()
        assert telemetry["occupancy_high_water"] == 2
        assert telemetry["producer_blocked_seconds"] > 0
        assert telemetry["pending_jobs"] == 0
        assert telemetry["completed_jobs"] == 3
        assert telemetry["durable_frontier"] == 3
        publication.close()
    finally:
        for event in release.values():
            event.set()
        publication.abort()


def test_async_publication_consumes_out_of_order_workers_in_fifo_order(
    tmp_path: Path,
) -> None:
    frozen = _frozen_fixture()
    first_release = threading.Event()
    second_complete = threading.Event()

    def out_of_order_materialize(
        staging_root: str | Path,
        *,
        frozen: FrozenCompactScienceDispatchHostV1,
    ) -> dict[str, Any]:
        if frozen.start_next_absolute_tick == 0:
            assert first_release.wait(timeout=5)
        receipt = _synthetic_receipt(staging_root, frozen)
        if frozen.start_next_absolute_tick == 1:
            second_complete.set()
        return receipt

    publication = CompactScienceAsyncPublicationV1(
        workers=2,
        pending_slots=2,
        materialize=out_of_order_materialize,
    )
    try:
        first = publication.submit(
            frozen=_span(frozen, 0), destination=tmp_path / "dispatch-0"
        )
        second = publication.submit(
            frozen=_span(frozen, 1), destination=tmp_path / "dispatch-1"
        )
        assert second_complete.wait(timeout=5)
        assert publication.ready(second) is False
        with pytest.raises(
            CompactScienceAsyncPublicationError,
            match="FIFO order",
        ):
            publication.result(second)

        first_release.set()
        for token in (first, second):
            staged = dict(publication.result(token))
            publication.acknowledge_published(
                token,
                published_receipt=_publish_synthetic(token, staged),
            )
        assert publication.telemetry()["durable_frontier"] == 2
        publication.close()
    finally:
        first_release.set()
        publication.abort()


def test_async_publication_worker_failure_aborts_and_removes_all_staging(
    tmp_path: Path,
) -> None:
    frozen = _frozen_fixture()
    fail = threading.Event()
    release_second = threading.Event()
    both_started = threading.Barrier(3)

    def failing_materialize(
        staging_root: str | Path,
        *,
        frozen: FrozenCompactScienceDispatchHostV1,
    ) -> dict[str, Any]:
        root = Path(staging_root).resolve()
        root.mkdir()
        (root / "partial.bin").write_bytes(b"partial")
        both_started.wait(timeout=5)
        if frozen.start_next_absolute_tick == 0:
            assert fail.wait(timeout=5)
            raise RuntimeError("injected compact materialization failure")
        assert release_second.wait(timeout=5)
        return {
            "path": str(root),
            "start_next_absolute_tick": frozen.start_next_absolute_tick,
            "end_next_absolute_tick": frozen.end_next_absolute_tick,
            "event_content_sha256": frozen.event_content_sha256,
        }

    publication = CompactScienceAsyncPublicationV1(
        workers=2,
        pending_slots=2,
        materialize=failing_materialize,
    )
    first = publication.submit(
        frozen=_span(frozen, 0), destination=tmp_path / "dispatch-0"
    )
    second = publication.submit(
        frozen=_span(frozen, 1), destination=tmp_path / "dispatch-1"
    )
    both_started.wait(timeout=5)
    fail.set()
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            publication.poll_failure()
        except CompactScienceAsyncPublicationError:
            break
        time.sleep(0.01)
    else:  # pragma: no cover - bounded failure diagnostic
        pytest.fail("async publication did not report the injected worker failure")

    release_second.set()
    publication.abort()

    assert not Path(first.staging_root).exists()
    assert not Path(second.staging_root).exists()
    assert not Path(first.destination).exists()
    assert not Path(second.destination).exists()
    telemetry = publication.telemetry()
    assert telemetry["failed_jobs"] == 1
    assert telemetry["aborted"] is True
    assert telemetry["closed"] is True
    assert telemetry["completed_jobs"] == 0
