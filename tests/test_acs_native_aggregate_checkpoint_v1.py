from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import engine.campaigns.acs_native_aggregate_checkpoint_v1 as aggregate
from engine.campaigns.acs_native_aggregate_checkpoint_v1 import (
    ACSNativeAggregateCheckpointError,
    load_acs_native_aggregate_checkpoint,
    write_acs_native_aggregate_checkpoint,
    write_acs_native_aggregate_checkpoint_from_cursor,
)
from engine.campaigns.providers import (
    CampaignProviderError,
    ProductionEngineLifecycleV1,
    ResidentBatchRequestV1,
    _validate_aggregate_native_production_owner_route_v1,
    _lazy_projection_bundle_sequence_v1,
)
from engine.campaigns.production_engine_identity import PRODUCTION_PROFILE_ID
from engine.perf.cuda_architecture_b_direct_resident_patch_v1 import (
    CheckpointProjectionCursorV1,
    DetachedCellProjectionV1,
    DETACHED_CELL_PROJECTION_SCHEMA_VERSION_V1,
)
from tests.perf.test_eiel_native_checkpoint_single_projection_authority import (
    _projection as _authority_projection,
)


def _projection(cell_index: int, *, family: str = "mlp") -> dict[str, object]:
    return {
        "schema_version": "cxxbackbone10_checkpoint_projection_v2",
        "metadata": {
            "generation": 4,
            "tick": 0,
            "world": "patch",
            "native_runtime_fixed_policy_family": family,
            "native_runtime_patch_profile": "P2",
            "native_runtime_patch_profile_scope": "full_profile",
            "policy_weight_mutation_enable": False,
            "event_history_reference": {
                "path": f"/tmp/emergence-test/storage/events/cell-{cell_index}.jsonl",
                "frontier": 17 + cell_index,
            },
        },
        "event_cursor": 17 + cell_index,
        "runtime_event_cursor": 17 + cell_index,
        "restore_count": 0,
        "tensors": {
            "active": torch.arange(12, dtype=torch.float32).reshape(3, 4)
            + cell_index,
            "gru_pack": torch.zeros(2, dtype=torch.float32),
            "hmem": torch.zeros(2, 2, dtype=torch.float32),
            "policy_use_gru": torch.zeros(2, dtype=torch.bool),
        },
    }


def _write(
    projections: tuple[dict[str, object], ...],
    destination: Path,
    *,
    family: str = "mlp",
) -> dict[str, object]:
    cell_ids = tuple(f"cell-{index}" for index in range(len(projections)))
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
        cell_id: {"event_frontier": 17 + index, "cell_id": cell_id}
        for index, cell_id in enumerate(cell_ids)
    }
    return write_acs_native_aggregate_checkpoint(
        projections,
        destination,
        cell_ids=cell_ids,
        generation=4,
        tick=0,
        absolute_next_tick=None,
        batch_id="batch-0000",
        segment_ordinal=1,
        source_projection_sha256s=digests,
        architecture_b_identities=identities,
        observer_continuations=observers,
        chunk_bytes=7,
    )


def test_cuda_detached_copy_has_explicit_producer_to_copy_stream_handoff() -> None:
    source = Path(aggregate.__file__).read_text(encoding="utf-8")
    copy_start = source.index("def _copy_tensor(")
    copy_end = source.index("\ndef _route_policy_family", copy_start)
    copy_body = source[copy_start:copy_end]
    assert "producer_event: Any | None = None" in copy_body
    assert "dedicated_stream.wait_event(producer_event)" in copy_body
    assert copy_body.index("wait_event") < copy_body.index(
        "with torch.cuda.stream(dedicated_stream)"
    )

    cursor_start = source.index("def write_acs_native_aggregate_checkpoint_from_cursor")
    cursor_body = source[cursor_start:]
    assert "producer_event = None" in cursor_body
    assert "producer_event.record(" in cursor_body
    assert "producer_event=producer_event" in cursor_body


def _production_aggregate_request(*, staging: bool) -> SimpleNamespace:
    return SimpleNamespace(
        profile_id=PRODUCTION_PROFILE_ID,
        cell_ids=("cell-0", "cell-1"),
        parameter_arrays=(
            (
                "cuda_architecture_b.cpu_request_staging_enable",
                (staging, staging),
            ),
        ),
    )


def _cuda_source_projections() -> tuple[dict[str, object], ...]:
    device = SimpleNamespace(type="cuda")
    return tuple(
        {"tensors": {"resident": SimpleNamespace(device=device)}}
        for _ in range(2)
    )


def test_aggregate_production_route_requires_nonstaged_request_owner() -> None:
    assert _validate_aggregate_native_production_owner_route_v1(
        _production_aggregate_request(staging=False),
        _cuda_source_projections(),
    ) == (1, 1)

    with pytest.raises(CampaignProviderError, match="cpu_request_staging_enable=false"):
        _validate_aggregate_native_production_owner_route_v1(
            _production_aggregate_request(staging=True),
            _cuda_source_projections(),
        )

    # Canonical checkpoint projections are intentionally host-materialized by
    # the bounded serializer.  Their CPU device does not imply CPU request
    # staging and must not be rejected at this later boundary.
    assert _validate_aggregate_native_production_owner_route_v1(
        _production_aggregate_request(staging=False),
        tuple({"tensors": {"resident": torch.zeros(1)}} for _ in range(2)),
    ) == (0, 0)


def test_aggregate_round_trip_is_deterministic_and_zero_fills_fixed_mlp(
    tmp_path: Path,
) -> None:
    projections = (_projection(0), _projection(1))
    first = _write(projections, tmp_path / "first.acsaggregate")
    second = _write(projections, tmp_path / "second.acsaggregate")

    first_path = Path(str(first["path"]))
    second_path = Path(str(second["path"]))
    assert first_path.read_bytes() == second_path.read_bytes()
    assert first["physical_file_bytes"] == first_path.stat().st_size
    assert first["payload_bytes"] < first["logical_bytes"]
    assert first["zero_fill_logical_bytes"] > 0
    assert first["cell_count"] == 2
    assert not list(tmp_path.glob("*.payload.tmp"))

    for index, cell_id in enumerate(("cell-0", "cell-1")):
        loaded = load_acs_native_aggregate_checkpoint(
            first_path,
            cell_id=cell_id,
            expected_source_projection_sha256="a" * 64,
        )
        restored = loaded["projection"]
        original = projections[index]
        assert restored["metadata"] == original["metadata"]
        assert torch.equal(
            restored["tensors"]["active"], original["tensors"]["active"]
        )
        assert torch.equal(
            restored["tensors"]["gru_pack"], original["tensors"]["gru_pack"]
        )
        assert torch.equal(
            restored["tensors"]["policy_use_gru"],
            original["tensors"]["policy_use_gru"],
        )
        assert loaded["architecture_b_identity"]["source_projection_sha256"] == "a" * 64


def test_zero_fill_digest_is_memoized_by_logical_byte_length(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[int] = []
    real_zero_digest = aggregate._zero_digest
    expected_digests = {
        logical_bytes: real_zero_digest(logical_bytes)
        for logical_bytes in (2, 8, 16)
    }

    def counting_zero_digest(logical_bytes: int) -> str:
        calls.append(logical_bytes)
        return real_zero_digest(logical_bytes)

    monkeypatch.setattr(aggregate, "_zero_digest", counting_zero_digest)
    receipt = _write(
        (_projection(0), _projection(1)), tmp_path / "memoized-zero.acsaggregate"
    )

    # Each zero-filled logical size is hashed once even though every physical
    # cell contains the same zero-filled fields.  The digest values themselves
    # remain the exact SHA-256 digests of the logical zero byte streams.
    assert sorted(calls) == [2, 8, 16]
    assert len(calls) == len(set(calls))
    calls.clear()
    header = load_acs_native_aggregate_checkpoint(
        Path(str(receipt["path"])), cell_id="cell-0"
    )["header"]
    assert sorted(calls) == [2, 8, 16]
    assert len(calls) == len(set(calls))
    observed_digests = {
        int(row["logical_bytes"]): row["sha256"]
        for cell in header["cells"]
        for row in cell["fields"]
        if row["storage"] == "zero_fill"
    }
    assert observed_digests == expected_digests


def test_aggregate_round_trip_allows_per_cell_dynamic_tensor_shapes(
    tmp_path: Path,
) -> None:
    first = _projection(0)
    second = _projection(1)
    second["tensors"]["active"] = torch.arange(8, dtype=torch.float32).reshape(2, 4)  # type: ignore[index]
    projections = (first, second)
    receipt = _write(projections, tmp_path / "dynamic.acsaggregate")

    loaded_header = load_acs_native_aggregate_checkpoint(
        Path(str(receipt["path"])), cell_id="cell-0"
    )["header"]
    assert loaded_header["field_schema_mode"] == "per_cell_v1"
    for index, cell_id in enumerate(("cell-0", "cell-1")):
        loaded = load_acs_native_aggregate_checkpoint(
            Path(str(receipt["path"])), cell_id=cell_id
        )
        assert torch.equal(
            loaded["projection"]["tensors"]["active"],
            projections[index]["tensors"]["active"],  # type: ignore[index]
        )


def test_aggregate_cursor_round_trip_keeps_one_cell_live(tmp_path: Path) -> None:
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
            terminal_generation=4,
            terminal_tick=0,
            live_projection_bytes=sum(
                int(tensor.numel()) * int(tensor.element_size())
                for tensor in projection["tensors"].values()  # type: ignore[union-attr]
            ),
        )

    cursor = CheckpointProjectionCursorV1.begin(
        owner=object(),
        coordinate=4,
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

    result = write_acs_native_aggregate_checkpoint_from_cursor(
        cursor,
        tmp_path / "cursor.acsaggregate",
        cell_ids=cell_ids,
        generation=4,
        tick=0,
        absolute_next_tick=None,
        batch_id="batch-0000",
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

    receipt = result["receipt"]
    assert receipt["cell_count"] == 2
    assert result["checkpoint_projection_cursor_receipt"][
        "max_live_tensor_bearing_projections"
    ] == 1
    assert result["checkpoint_projection_cursor_receipt"]["state"] == "SEALED"
    assert not list(tmp_path.glob(".*.spool.tmp"))
    for index, cell_id in enumerate(cell_ids):
        loaded = load_acs_native_aggregate_checkpoint(
            Path(str(receipt["path"])),
            cell_id=cell_id,
            expected_source_projection_sha256="a" * 64,
        )
        assert torch.equal(
            loaded["projection"]["tensors"]["active"],
            projections[index]["tensors"]["active"],  # type: ignore[index]
        )


def test_aggregate_fixed_mlp_zero_fill_is_fail_closed(tmp_path: Path) -> None:
    nonzero = _projection(0)
    nonzero["tensors"]["hmem"][0, 0] = 1.0  # type: ignore[index]
    with pytest.raises(ACSNativeAggregateCheckpointError, match="zero-fill field is nonzero"):
        _write((nonzero, _projection(1)), tmp_path / "nonzero.acsaggregate")

    missing = _projection(0)
    del missing["tensors"]["hmem"]  # type: ignore[index]
    with pytest.raises(
        ACSNativeAggregateCheckpointError,
        match="identical tensor field schema|missing candidate fields",
    ):
        _write((missing, _projection(1)), tmp_path / "missing.acsaggregate")


def test_aggregate_fixed_mlp_zero_fill_requires_route_contract(tmp_path: Path) -> None:
    projection = _projection(0)
    del projection["metadata"]["native_runtime_patch_profile_scope"]  # type: ignore[index]
    with pytest.raises(
        ACSNativeAggregateCheckpointError,
        match="zero-fill contract is missing or incompatible",
    ):
        _write((projection, _projection(1)), tmp_path / "missing-contract.acsaggregate")


def test_aggregate_non_mlp_route_keeps_candidate_fields_physical(tmp_path: Path) -> None:
    projections = (_projection(0, family="gru"), _projection(1, family="gru"))
    receipt = _write(projections, tmp_path / "gru.acsaggregate", family="gru")
    assert receipt["zero_fill_logical_bytes"] == 0
    assert receipt["payload_bytes"] == receipt["logical_bytes"]
    header = load_acs_native_aggregate_checkpoint(
        Path(str(receipt["path"])), cell_id="cell-0"
    )["header"]
    assert header["zero_fill_fields"] == []


@pytest.mark.parametrize("damage", ["truncate", "flip"])
def test_aggregate_loader_rejects_damage(tmp_path: Path, damage: str) -> None:
    receipt = _write((_projection(0), _projection(1)), tmp_path / "damage.acsaggregate")
    target = Path(str(receipt["path"]))
    data = bytearray(target.read_bytes())
    if damage == "truncate":
        data = data[:-3]
    else:
        data[len(data) // 2] ^= 1
    target.write_bytes(data)
    with pytest.raises(ACSNativeAggregateCheckpointError):
        load_acs_native_aggregate_checkpoint(target, cell_id="cell-0")


def test_aggregate_retry_is_idempotent_and_conflict_is_fail_closed(
    tmp_path: Path,
) -> None:
    target = tmp_path / "retry.acsaggregate"
    first = _write((_projection(0), _projection(1)), target)
    original = target.read_bytes()
    assert _write((_projection(0), _projection(1)), target)["artifact_sha256"] == first[
        "artifact_sha256"
    ]
    assert target.read_bytes() == original

    changed = _projection(1)
    changed["tensors"]["active"] = torch.full((3, 4), 99.0)  # type: ignore[index]
    with pytest.raises(ACSNativeAggregateCheckpointError, match="conflicting"):
        _write((_projection(0), changed), target)
    assert target.read_bytes() == original


def test_aggregate_fsync_failure_leaves_no_staging_and_preserves_prior(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "fsync.acsaggregate"
    _write((_projection(0), _projection(1)), target)
    original = target.read_bytes()

    def fail_fsync(_fd: int) -> None:
        raise OSError("injected fsync failure")

    monkeypatch.setattr(aggregate.os, "fsync", fail_fsync)
    with pytest.raises(OSError, match="injected fsync failure"):
        _write((_projection(0), _projection(1)), target)
    assert target.read_bytes() == original
    assert not list(tmp_path.glob(".*.payload.tmp"))
    assert not list(tmp_path.glob(".*.tmp"))


def test_provider_publishes_one_aggregate_object_and_cell_indexed_restore(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = ProductionEngineLifecycleV1()
    projections = (
        _authority_projection(generation=4, tick=0),
        _authority_projection(generation=4, tick=0),
    )
    cell_ids = ("cell-0", "cell-1")
    digests = {
        cell_id: provider.terminal_projection_digest(projection)
        for cell_id, projection in zip(cell_ids, projections, strict=True)
    }
    request = ResidentBatchRequestV1(
        profile_id="acs-native-test",
        batch_id="batch-0000",
        cell_ids=cell_ids,
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
    result = provider._persist_projection_checkpoint_set_v1(
        batch_root=tmp_path,
        request=request,
        source_projections=projections,
        generation=4,
        tick=0,
        absolute_next_tick=None,
        segment_ordinal=1,
        expected_source_projection_digest_sha256s=digests,
    )

    rows = result["bundle_rows"]
    assert len(rows) == 2
    assert rows[0]["path"] == rows[1]["path"]
    published = Path(rows[0]["path"])
    assert published.suffix == ".acsaggregate"
    assert published.is_file()
    assert result["checkpoint_bytes"] == published.stat().st_size
    assert result["cell_checkpoint_bytes"] == {"cell-0": 0, "cell-1": 0}
    assert result["cell_physical_bytes_attribution"]["cell-0"] > 0
    assert result["aggregate_checkpoint_receipt"]["cell_count"] == 2
    for cell_id in cell_ids:
        restored = provider._read_projection_bundle(str(published), cell_id=cell_id)
        assert provider.terminal_projection_digest(restored) == digests[cell_id]
        assert result["native_checkpoint_receipts"][cell_id]["path"] == str(published)
    payload_hash_calls: list[int] = []
    real_payload_hash = aggregate._payload_hash

    def count_payload_hash(*args: object, **kwargs: object) -> str:
        payload_hash_calls.append(1)
        return real_payload_hash(*args, **kwargs)

    monkeypatch.setattr(aggregate, "_payload_hash", count_payload_hash)
    lazy = _lazy_projection_bundle_sequence_v1(
        (str(published), str(published)),
        provider._read_projection_bundle,
        cell_ids=cell_ids,
    )
    assert [
        provider.terminal_projection_digest(lazy[index]) for index in range(2)
    ] == [digests["cell-0"], digests["cell-1"]]
    assert payload_hash_calls == [1]


def test_provider_binds_cursor_to_one_aggregate_object_without_materializing_cells(
    tmp_path: Path,
) -> None:
    provider = ProductionEngineLifecycleV1()
    projections = (
        _authority_projection(generation=4, tick=0),
        _authority_projection(generation=4, tick=0),
    )
    cell_ids = ("cell-0", "cell-1")

    def factory(index: int, cell_id: str) -> DetachedCellProjectionV1:
        return DetachedCellProjectionV1(
            schema_version=DETACHED_CELL_PROJECTION_SCHEMA_VERSION_V1,
            cell_id=cell_id,
            cell_index=index,
            projection=projections[index],
            terminal_generation=4,
            terminal_tick=0,
            live_projection_bytes=1,
        )

    cursor = CheckpointProjectionCursorV1.begin(
        owner=object(),
        coordinate=4,
        ordered_cell_ids=cell_ids,
        factory=factory,
    )
    request = ResidentBatchRequestV1(
        profile_id="acs-native-test",
        batch_id="batch-cursor-0000",
        cell_ids=cell_ids,
        generations=4,
        ticks_per_generation=160,
        checkpoint_milestones=(4,),
        parameter_arrays=(),
        seed_arrays=(),
        execution_route_sha256="b" * 64,
        acs_checkpoint_slots=0,
        acs_capture_instrumentation=True,
        native_streamed_checkpoint=True,
    )
    result = provider._persist_projection_checkpoint_set_v1(
        batch_root=tmp_path,
        request=request,
        source_projections=None,
        projection_cursor=cursor,
        generation=4,
        tick=0,
        absolute_next_tick=None,
        segment_ordinal=1,
    )

    rows = result["bundle_rows"]
    assert len(rows) == 2
    assert rows[0]["path"] == rows[1]["path"]
    assert Path(rows[0]["path"]).suffix == ".acsaggregate"
    assert result["checkpoint_projection_cursor_receipt"][
        "max_live_tensor_bearing_projections"
    ] == 1
    assert result["aggregate_checkpoint_receipt"]["cell_count"] == 2
