from __future__ import annotations

import gzip
import hashlib
import json
import os
from pathlib import Path

import pytest
import torch

from engine.campaigns import cuda_full_evolution_v2_provider as cuda_provider
from engine.campaigns.providers import CampaignProviderError
from emergence_contracts.science_compact_capture import (
    COMPACT_SCIENCE_AGENT_METRIC_FIELDS_V1,
)
from engine.native_runtime.cuda_snapshot_encoder_v1 import (
    NATIVE_SNAPSHOT_STAGE_FILE_RECEIPT_SCHEMA_V1,
)
from engine.native_runtime.cuda_snapshot_plan_v1 import (
    NATIVE_SNAPSHOT_STATE_CAPTURE_CANONICAL_PREFIX_V1,
    compile_cuda_snapshot_plan_v1,
)
from engine.native_runtime.cuda_snapshot_publication_v1 import (
    NATIVE_SNAPSHOT_IMPLEMENTATION_ENV_V1,
    NativeSnapshotPublicationError,
    NativeSnapshotPublicationServiceV1,
    NativeSnapshotVisibleRootError,
    inspect_existing_native_snapshot_root_v1,
    native_snapshot_implementation_v1,
    remove_native_snapshot_root_v1,
)


CONTROL_SHA256 = "a" * 64
PUBLICATION_CONTEXT_SHA256 = "b" * 64


def _cpu_allocator(shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    return torch.empty(shape, dtype=dtype, device="cpu")


def _cells() -> dict[str, dict[str, object]]:
    return {
        "cell-202": {
            "science_logical_world_id": 202,
            "science_ownership": {"study": "native-publication", "arm": 2},
            "science_world_sha256": {
                "algorithm": "sha256",
                "hex": "2" * 64,
            },
            "science_acquisition_sha256": {
                "algorithm": "sha256",
                "hex": "3" * 64,
            },
            "science_acquisition": {"snapshot_ticks": [7]},
        },
        "cell-101": {
            "science_logical_world_id": 101,
            "science_ownership": {"study": "native-publication", "arm": 1},
            "science_world_sha256": {
                "algorithm": "sha256",
                "hex": "1" * 64,
            },
            "science_acquisition_sha256": {
                "algorithm": "sha256",
                "hex": "2" * 64,
            },
            "science_acquisition": {"snapshot_ticks": [3, 7]},
        },
    }


def _plan():
    return compile_cuda_snapshot_plan_v1(
        world_uids=(101, 202),
        world_cells=_cells(),
        native_world_control_set_sha256=CONTROL_SHA256,
        execution_publication_context_sha256=PUBLICATION_CONTEXT_SHA256,
        catalog_absolute_tick=3,
    )


class _CaptureFence:
    def __init__(self, d2h_bytes: int) -> None:
        self._d2h_bytes = d2h_bytes

    def wait(self) -> dict[str, int]:
        return {"d2h_bytes": self._d2h_bytes, "cuda_event_time_ns": 17}


class _RuntimeOwner:
    def __init__(self) -> None:
        self.capture_ticks: list[int] = []
        self.capture_state_shapes: list[tuple[int, ...]] = []

    def capture_native_snapshot_to_pinned_v1(
        self,
        *,
        absolute_tick: int,
        state_destination: torch.Tensor,
        active_science_metrics_destination: torch.Tensor,
    ) -> _CaptureFence:
        self.capture_ticks.append(absolute_tick)
        self.capture_state_shapes.append(tuple(state_destination.shape))
        state_destination.fill_(absolute_tick)
        active_science_metrics_destination.fill_(absolute_tick * 10)
        return _CaptureFence(
            state_destination.numel() * state_destination.element_size()
            + active_science_metrics_destination.numel()
            * active_science_metrics_destination.element_size()
        )


def _native_plan_compiler(compiled_plan, *, loaded_extension=None):
    assert loaded_extension is None
    return compiled_plan


def _fake_owned_encoder(
    native_plan,
    *,
    state: torch.Tensor,
    metrics: torch.Tensor,
    absolute_tick: int,
    stage_path: str,
    encoding_policy: str,
) -> dict[str, object]:
    plan = native_plan.to_mapping()
    boundary = next(
        row
        for row in plan["snapshot_boundaries"]
        if row["absolute_tick"] == absolute_tick
    )
    raw = (
        json.dumps(
            {
                "absolute_tick": absolute_tick,
                "selected_world_uids": boundary["world_uids"],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode()
    compressed = gzip.compress(raw, compresslevel=9, mtime=0)
    descriptor = os.open(
        stage_path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
        0o600,
    )
    try:
        os.write(descriptor, compressed)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    captured = [
        {
            "world_index": world_index,
            "world_uid": world_uid,
            "active_agent_count": 0,
            "canonical_state_sha256": hashlib.sha256(
                f"{world_uid}:{absolute_tick}".encode()
            ).hexdigest(),
        }
        for world_index, world_uid in zip(
            boundary["world_indices"], boundary["world_uids"], strict=True
        )
    ]
    return {
        "schema_version": NATIVE_SNAPSHOT_STAGE_FILE_RECEIPT_SCHEMA_V1,
        "plan_sha256": native_plan.plan_sha256,
        "native_world_control_set_sha256": CONTROL_SHA256,
        "execution_publication_context_sha256": PUBLICATION_CONTEXT_SHA256,
        "absolute_tick": absolute_tick,
        "row_count": len(captured),
        "captured_world_count": len(captured),
        "active_agent_count": 0,
        "selected_world_indices": list(boundary["world_indices"]),
        "selected_world_uids": list(boundary["world_uids"]),
        "captured_worlds": captured,
        "metric_fields": list(COMPACT_SCIENCE_AGENT_METRIC_FIELDS_V1),
        "genome_catalog_included": boundary["include_genome_catalog"],
        "uncompressed_content_sha256": hashlib.sha256(raw).hexdigest(),
        "compressed_file_sha256": hashlib.sha256(compressed).hexdigest(),
        "uncompressed_bytes": len(raw),
        "compressed_bytes": len(compressed),
        "stage_path": stage_path,
        "stage_encoding_policy": encoding_policy,
        "stage_file_state": "FSYNCED_STAGE_ONLY_NOT_PUBLISHED",
        "stage_file_fsync_completed": True,
        "stage_file_close_completed": True,
        "atomic_publication_performed": False,
        "directory_fsync_performed": False,
        "stage_bytes_written": len(compressed),
        "timing": {"fake_native_encode_ns": 1},
        "state_witness": int(state[0, 0]),
        "metrics_witness": int(metrics[0, 0, 0]),
    }


def _service(
    runtime: _RuntimeOwner,
    *,
    encoder=_fake_owned_encoder,
    native_plan_compiler=_native_plan_compiler,
):
    return NativeSnapshotPublicationServiceV1(
        compiled_plan=_plan(),
        runtime_owner=runtime,
        slot_count=2,
        native_plan_compiler=native_plan_compiler,
        owned_encoder=encoder,
        slot_allocator=_cpu_allocator,
        allow_unpinned_test_slots=True,
    )


def test_implementation_selector_is_strict_and_native_default(monkeypatch) -> None:
    monkeypatch.delenv(NATIVE_SNAPSHOT_IMPLEMENTATION_ENV_V1, raising=False)
    assert native_snapshot_implementation_v1() == "native_capsule_v1"
    for value in ("python_v1", "native_v1", "native_capsule_v1"):
        monkeypatch.setenv(NATIVE_SNAPSHOT_IMPLEMENTATION_ENV_V1, value)
        assert native_snapshot_implementation_v1() == value
    for value in ("", "native", "PYTHON_V1", " native_v1"):
        monkeypatch.setenv(NATIVE_SNAPSHOT_IMPLEMENTATION_ENV_V1, value)
        with pytest.raises(NativeSnapshotPublicationError, match="must be exactly"):
            native_snapshot_implementation_v1()


def test_plan_bound_fifo_publishes_exact_legacy_root_and_reuses_slots(
    tmp_path: Path,
) -> None:
    runtime = _RuntimeOwner()
    compile_calls = 0

    def counting_compiler(compiled_plan, *, loaded_extension=None):
        nonlocal compile_calls
        compile_calls += 1
        return _native_plan_compiler(compiled_plan, loaded_extension=loaded_extension)

    service = _service(runtime, native_plan_compiler=counting_compiler)
    roots = [tmp_path / "tick-3", tmp_path / "tick-7"]
    try:
        first = service.submit(runtime, 3, roots[0])
        second = service.submit(runtime, 7, roots[1])
        assert service.has_capacity is False
        for token, root in zip((first, second), roots, strict=True):
            staged = service.result(token, producer_backpressure=True)
            published = service.publish(token, staged)
            service.acknowledge_published(token, published_receipt=published)
            assert published["path"] == str(root.absolute())
            assert set(path.name for path in root.iterdir()) == {
                "manifest.json",
                "snapshot.jsonl.gz",
            }
            manifest = json.loads((root / "manifest.json").read_bytes())
            assert manifest == {
                key: value for key, value in published.items() if key != "path"
            }
            assert (
                inspect_existing_native_snapshot_root_v1(
                    root,
                    compiled_plan=_plan(),
                    absolute_tick=token.absolute_tick,
                )
                == published
            )
            assert (
                hashlib.sha256((root / "snapshot.jsonl.gz").read_bytes()).hexdigest()
                == manifest["compressed_file_sha256"]
            )
            assert not Path(token.staging_root).exists()
        service.close()
        telemetry = service.telemetry()
        assert runtime.capture_ticks == [3, 7]
        preferred_shape = tuple(
            _plan().to_mapping()["state_capture_transport"]["accepted_modes"][0][
                "shape"
            ]
        )
        assert runtime.capture_state_shapes == [preferred_shape, preferred_shape]
        assert compile_calls == 1
        assert telemetry["publication_completed_count"] == 2
        assert telemetry["pending_jobs"] == 0
        assert telemetry["owned_partial_stage_count"] == 0
        assert telemetry["publication_artifact_mode"] == (
            "legacy_jsonl_gzip_compatibility_v1"
        )
        assert telemetry["native_capsule_published"] is False
        assert telemetry["state_capture_mode"] == (
            NATIVE_SNAPSHOT_STATE_CAPTURE_CANONICAL_PREFIX_V1
        )
        assert telemetry["horizon_dependent_buffer_bytes"] == 0
        assert telemetry["core_service"]["slot_count"] == 2
        assert telemetry["core_service"]["in_use_slots_high_water"] == 2
    finally:
        if not service.telemetry()["closed"]:
            service.abort()
        for root in roots:
            remove_native_snapshot_root_v1(root)


def test_existing_destination_is_never_replaced_or_owned_by_abort(
    tmp_path: Path,
) -> None:
    runtime = _RuntimeOwner()
    service = _service(runtime)
    root = tmp_path / "tick-3"
    token = service.submit(runtime, 3, root)
    staged = service.result(token, producer_backpressure=True)
    root.mkdir()
    marker = root / "foreign"
    marker.write_text("preserve")
    with pytest.raises(NativeSnapshotPublicationError, match="already exists"):
        service.publish(token, staged)
    service.abort()
    assert marker.read_text() == "preserve"
    assert not Path(token.staging_root).exists()


def test_abort_rejects_replaced_stage_directory_identity(tmp_path: Path) -> None:
    runtime = _RuntimeOwner()
    service = _service(runtime)
    token = service.submit(runtime, 3, tmp_path / "tick-3")
    service.result(token, producer_backpressure=True)
    stage = Path(token.staging_root)
    displaced = tmp_path / "displaced-owned-stage"
    stage.rename(displaced)
    stage.mkdir(mode=0o700)
    (stage / "manifest.json").write_text("foreign")
    (stage / "snapshot.jsonl.gz").write_bytes(b"foreign")

    with pytest.raises(
        NativeSnapshotPublicationError,
        match="owned directory identity",
    ):
        service.abort()

    assert (stage / "manifest.json").read_text() == "foreign"
    remove_native_snapshot_root_v1(stage)
    remove_native_snapshot_root_v1(displaced)


def test_submit_rejects_symlinked_publication_ancestor(tmp_path: Path) -> None:
    runtime = _RuntimeOwner()
    service = _service(runtime)
    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(real_parent, target_is_directory=True)

    with pytest.raises(
        NativeSnapshotPublicationError,
        match="symlink|canonical",
    ):
        service.submit(runtime, 3, linked_parent / "tick-3")

    service.close()
    assert list(real_parent.iterdir()) == []


@pytest.mark.parametrize(
    ("inventory_state", "root_preserved"),
    [
        ("NOT_COMMITTED_REMOVE_ROOT", False),
        ("COMMIT_UNKNOWN_PRESERVE_ROOT", True),
    ],
)
def test_visibility_failure_reaches_provider_lifecycle_reconciliation(
    tmp_path: Path,
    inventory_state: str,
    root_preserved: bool,
) -> None:
    destination = tmp_path / "tick-3"

    class Pipeline:
        def result(self, token, *, producer_backpressure):
            del token
            assert producer_backpressure is True
            return {"absolute_tick": 3}

        def publish(self, token, receipt):
            del token, receipt
            destination.mkdir()
            (destination / "manifest.json").write_text("{}")
            (destination / "snapshot.jsonl.gz").write_bytes(b"gz")
            raise NativeSnapshotVisibleRootError(
                destination,
                "injected rename-visible durability failure",
            )

    with pytest.raises(CampaignProviderError, match="publication settlement"):
        cuda_provider._settle_native_snapshot_publication_transaction_v1(
            pipeline=Pipeline(),
            token=object(),
            context=object(),
            producer_backpressure=True,
            reconcile_inventory=lambda *_args, **_kwargs: inventory_state,
        )

    assert destination.exists() is root_preserved
    if root_preserved:
        remove_native_snapshot_root_v1(destination)


def test_worker_failure_propagates_and_abort_removes_only_partial_stage(
    tmp_path: Path,
) -> None:
    def failing_encoder(*args, **kwargs):
        del args, kwargs
        raise OSError("injected native encoder failure")

    runtime = _RuntimeOwner()
    service = _service(runtime, encoder=failing_encoder)
    root = tmp_path / "tick-3"
    token = service.submit(runtime, 3, root)
    with pytest.raises(Exception, match="injected native encoder failure"):
        service.result(token, producer_backpressure=True)
    service.abort()
    assert not root.exists()
    assert not Path(token.staging_root).exists()


@pytest.mark.parametrize(
    ("inventory_state", "root_preserved"),
    [
        ("NOT_COMMITTED_REMOVE_ROOT", False),
        ("COMMIT_UNKNOWN_PRESERVE_ROOT", True),
    ],
)
def test_provider_transaction_removes_only_known_uncommitted_native_root(
    tmp_path: Path,
    inventory_state: str,
    root_preserved: bool,
) -> None:
    destination = tmp_path / "tick-3"
    staged_receipt = {
        "schema_version": "eiel_cuda_v2_compact_science_snapshot_artifact_v1",
        "artifact_sha256": "a" * 64,
        "path": str(tmp_path / ".tick-3.stage"),
        "absolute_tick": 3,
        "capture_metrics": {"d2h_bytes": 12},
        "encoder_timing": {"encode_ns": 1},
        "native_counters": {"row_count": 1},
        "publication_timing": {"stage_seal_ns": 2},
    }

    class Pipeline:
        acknowledged = False

        def result(self, token, *, producer_backpressure):
            del token
            assert producer_backpressure is True
            return dict(staged_receipt)

        def publish(self, token, receipt):
            del token
            destination.mkdir()
            (destination / "manifest.json").write_text("{}")
            return {**dict(receipt), "path": str(destination)}

        def acknowledge_published(self, token, *, published_receipt):
            del token
            assert published_receipt["path"] == str(destination)
            self.acknowledged = True

    pipeline = Pipeline()

    def remove_root(root: str | Path) -> None:
        path = Path(root)
        (path / "manifest.json").unlink()
        path.rmdir()

    with pytest.raises(CampaignProviderError, match="lifecycle record"):
        cuda_provider._settle_native_snapshot_publication_transaction_v1(
            pipeline=pipeline,
            token=object(),
            context=object(),
            producer_backpressure=True,
            record_root=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                RuntimeError("injected lifecycle record failure")
            ),
            reconcile_inventory=lambda *_args, **_kwargs: inventory_state,
            remove_root=remove_root,
        )

    assert destination.exists() is root_preserved
    assert pipeline.acknowledged is False


def test_native_timing_summary_separates_handoff_from_durable_completion() -> None:
    samples = [
        {
            "implementation": "native_v1",
            "producer_critical_path_time_ns": 5,
            "full_durable_completion_time_ns": 50,
            "publication_lag_time_ns": 45,
            "gpu_sync_host_wait_upper_bound_ns": 5,
            "evolution_producer_blocking_time_ns": 8,
            "phases": {"native_capture_and_submit": {"wall_time_ns": 5}},
            "counters": {
                "row_count": 2,
                "captured_world_count": 1,
                "d2h_bytes": 100,
                "uncompressed_bytes": 80,
                "compressed_bytes": 40,
            },
            "memory": {
                "resident_pinned_bytes": 1_024,
                "core_service": {
                    "slot_count": 2,
                    "in_use_buffer_bytes_high_water": 1_024,
                },
            },
        }
    ]
    summary = cuda_provider._compact_snapshot_timing_summary_v1(samples)
    assert summary["implementation"] == "native_v1"
    assert (
        summary["duration_distributions"]["producer_critical_path_time_ns"]["p50_ns"]
        == 5
    )
    assert (
        summary["duration_distributions"]["full_durable_completion_time_ns"]["p50_ns"]
        == 50
    )
    assert summary["memory_measurement"] == {
        **summary["memory_measurement"],
        "pinned_host_bytes": 1_024,
        "snapshot_resident_buffer_count": 2,
        "in_use_buffer_bytes_high_water": 1_024,
    }


def test_native_canonical_slot_resource_bound_scales_with_batch_not_horizon() -> None:
    short = cuda_provider.CudaFullEvolutionNativeContractV2(
        B=3,
        total_ticks=8,
        start_next_tick=0,
        temporal_k=2,
    )
    long = cuda_provider.CudaFullEvolutionNativeContractV2(
        B=3,
        total_ticks=8_000_000,
        start_next_tick=0,
        temporal_k=2,
    )
    first = cuda_provider._native_snapshot_resource_accounting_v1(short, slot_count=2)
    second = cuda_provider._native_snapshot_resource_accounting_v1(long, slot_count=2)
    assert first == second
    assert first["native_snapshot_pinned_host_bytes_upper_bound"] == (
        2 * first["native_snapshot_bytes_per_slot"]
    )
    layout = cuda_provider._cuda_v2_layout.state_layout_v2()
    assert first["native_snapshot_capture_mode"] == (
        NATIVE_SNAPSHOT_STATE_CAPTURE_CANONICAL_PREFIX_V1
    )
    assert first["native_snapshot_capture_state_bytes_per_world"] == (
        layout.canonical_checkpoint_bytes_per_world
    )
    assert first["native_snapshot_resident_state_bytes_per_world"] == (
        layout.state_bytes_per_world
    )
    assert first["native_snapshot_state_bytes_per_slot"] == (
        short.B * layout.canonical_checkpoint_bytes_per_world
    )
    assert first["native_snapshot_resident_state_bytes_per_slot"] == (
        short.B * layout.state_bytes_per_world
    )
    assert first["native_snapshot_queue_capacity"] == 2
    assert first["native_snapshot_horizon_dependent_buffer_bytes"] == 0


def test_native_snapshot_mode_must_match_plan_bound_resource_admission() -> None:
    contract = cuda_provider.CudaFullEvolutionNativeContractV2(
        B=3,
        total_ticks=8,
        start_next_tick=0,
        temporal_k=2,
    )
    native = cuda_provider._native_snapshot_resource_accounting_v1(
        contract,
        slot_count=2,
    )
    cuda_provider._validate_native_snapshot_resource_admission_v1(
        {"resource_estimate": {}},
        contract,
        implementation="python_v1",
        slot_count=2,
    )
    cuda_provider._validate_native_snapshot_resource_admission_v1(
        {"resource_estimate": native},
        contract,
        implementation="native_v1",
        slot_count=2,
    )
    capsule_native = cuda_provider._native_snapshot_resource_accounting_v1(
        contract,
        slot_count=2,
        implementation="native_capsule_v1",
    )
    cuda_provider._validate_native_snapshot_resource_admission_v1(
        {"resource_estimate": capsule_native},
        contract,
        implementation="native_capsule_v1",
        slot_count=2,
    )
    with pytest.raises(CampaignProviderError, match="plan-bound"):
        cuda_provider._validate_native_snapshot_resource_admission_v1(
            {"resource_estimate": {}},
            contract,
            implementation="native_v1",
            slot_count=2,
        )
    with pytest.raises(CampaignProviderError, match="plan-bound"):
        cuda_provider._validate_native_snapshot_resource_admission_v1(
            {"resource_estimate": native},
            contract,
            implementation="python_v1",
            slot_count=2,
        )
