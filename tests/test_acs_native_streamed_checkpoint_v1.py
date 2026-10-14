from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

import engine.campaigns.acs_native_streamed_checkpoint_v1 as native_checkpoint
from engine.campaigns.acs_native_streamed_checkpoint_v1 import (
    ACSNativeCheckpointError,
    load_acs_native_streamed_checkpoint,
    streamed_projection_digest,
    write_acs_native_streamed_checkpoint,
)
from engine.campaigns.providers import ProductionEngineLifecycleV1, ResidentBatchRequestV1
from engine.native_runtime.checkpoint_digests import (
    DigestLayer,
    checkpoint_digest_manifest,
)
from engine.perf.cxxbackbone10_resident_state_owner import (
    migrate_native_checkpoint_projection_v1_to_v2,
)
from tests.perf.test_eiel_native_checkpoint_single_projection_authority import (
    _projection as _authority_projection,
)


def _projection() -> dict[str, object]:
    return {
        "schema_version": "cxxbackbone10_checkpoint_projection_v2",
        "metadata": {
            "generation": 4,
            "tick": 0,
            "world": "flat",
            "rng_state": (1, 2, 3),
            "opaque": b"checkpoint",
            "array": np.asarray([1, 2, 3], dtype=np.int16),
        },
        "restore_count": 0,
        "generation_calls": 4,
        "event_cursor": 7,
        "runtime_event_cursor": 7,
        "last_applied_actions": 0,
        "last_moved": 0,
        "last_gathered": 0,
        "last_policy_refill_calls": 0,
        "last_policy_refill_rows": 0,
        "pending_generation_free_agent_ids": [11],
        "pending_generation_free_slots": [3],
        "last_world": "flat",
        "tensors": {
            "alive_mask": torch.tensor([True, False, True]),
            "agent_grid": torch.arange(12, dtype=torch.int32).reshape(3, 4),
        },
    }


def _write(projection: dict[str, object], destination: Path) -> str:
    return write_acs_native_streamed_checkpoint(
        projection,
        destination,
        cell_id="cell-0",
        generation=4,
        tick=0,
        absolute_next_tick=None,
        batch_id="batch-0000",
        segment_ordinal=1,
        source_projection_sha256="a" * 64,
        architecture_b_identity={"snapshot": "one"},
        observer_continuation={"cursor": 1},
        chunk_bytes=7,
    )


def test_streamed_projection_digest_matches_owner_digest() -> None:
    from engine.perf.cxxbackbone10_resident_state_owner import projection_digest

    projection = _projection()
    assert streamed_projection_digest(projection) == projection_digest(projection)


def test_source_digest_tensor_payload_hashes_without_whole_host_materialization() -> None:
    from engine.native_runtime.checkpoint_digests import _tensor_payload

    tensor = torch.arange(33, dtype=torch.int32).reshape(3, 11).transpose(0, 1)
    expected = hashlib.sha256(
        tensor.detach().contiguous().numpy().tobytes()
    ).hexdigest()

    payload = _tensor_payload(tensor)

    assert payload["dtype"] == "int32"
    assert payload["shape"] == [11, 3]
    assert payload["sha256"] == expected


def test_native_streamed_round_trip_is_lossless_and_deterministic(tmp_path: Path) -> None:
    first = Path(_write(_projection(), tmp_path / "first.acsnative"))
    second = Path(_write(_projection(), tmp_path / "second.acsnative"))

    assert first.read_bytes() == second.read_bytes()
    loaded = load_acs_native_streamed_checkpoint(
        first, expected_source_projection_sha256="a" * 64
    )
    projection = loaded["projection"]
    assert projection["metadata"]["rng_state"] == (1, 2, 3)  # type: ignore[index]
    assert projection["metadata"]["opaque"] == b"checkpoint"  # type: ignore[index]
    np.testing.assert_array_equal(
        projection["metadata"]["array"],  # type: ignore[index]
        np.asarray([1, 2, 3], dtype=np.int16),
    )
    assert torch.equal(
        projection["tensors"]["agent_grid"],  # type: ignore[index]
        _projection()["tensors"]["agent_grid"],  # type: ignore[index]
    )
    assert loaded["footer"]["status"] == "complete"  # type: ignore[index]


def test_native_streamed_retry_is_idempotent_and_conflict_is_fail_closed(
    tmp_path: Path,
) -> None:
    target = Path(_write(_projection(), tmp_path / "retry.acsnative"))
    original = target.read_bytes()
    assert Path(_write(_projection(), target)).read_bytes() == original

    changed = _projection()
    changed["tensors"] = {
        "alive_mask": torch.tensor([False, False, True]),
        "agent_grid": torch.arange(12, dtype=torch.int32).reshape(3, 4),
    }
    with pytest.raises(ACSNativeCheckpointError, match="conflicting"):
        _write(changed, target)
    assert target.read_bytes() == original


def test_native_streamed_production_requirement_fails_closed_without_extension(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(native_checkpoint, "_NATIVE_CHUNK_LOOKUP_DONE", True)
    monkeypatch.setattr(native_checkpoint, "_NATIVE_CHUNK_COPY", None)
    with pytest.raises(
        ACSNativeCheckpointError, match="requires a verified prebuilt extension"
    ):
        write_acs_native_streamed_checkpoint(
            _projection(),
            tmp_path / "strict.acsnative",
            cell_id="cell-0",
            generation=4,
            tick=0,
            absolute_next_tick=None,
            batch_id="batch-0000",
            segment_ordinal=1,
            source_projection_sha256="a" * 64,
            architecture_b_identity={"snapshot": "one"},
            observer_continuation={"cursor": 1},
            require_native_extension=True,
        )


@pytest.mark.parametrize("damage", ["truncate", "flip"])
def test_native_streamed_loader_rejects_damaged_publication(
    tmp_path: Path, damage: str
) -> None:
    target = Path(_write(_projection(), tmp_path / "damaged.acsnative"))
    data = bytearray(target.read_bytes())
    if damage == "truncate":
        data = data[:-3]
    else:
        data[len(data) // 2] ^= 0x01
    target.write_bytes(data)
    with pytest.raises(ACSNativeCheckpointError):
        load_acs_native_streamed_checkpoint(target)


@pytest.mark.parametrize("damage", ["malformed_header", "malformed_footer", "torn_seal", "artifact_hash"])
def test_native_streamed_loader_rejects_seal_and_envelope_damage(
    tmp_path: Path, damage: str
) -> None:
    target = Path(_write(_projection(), tmp_path / f"{damage}.acsnative"))
    data = bytearray(target.read_bytes())
    prefix_size = native_checkpoint.PREFIX.size
    _magic, _version, _endian, header_len, payload_len, footer_len = (
        native_checkpoint.PREFIX.unpack(data[:prefix_size])
    )
    footer_offset = prefix_size + header_len + payload_len
    if damage == "malformed_header":
        data[prefix_size] = ord("!")
    elif damage == "malformed_footer":
        data[footer_offset] = ord("!")
    else:
        footer = json.loads(
            bytes(data[footer_offset : footer_offset + footer_len]).decode("utf-8")
        )
        footer["seal_sha256" if damage == "torn_seal" else "artifact_sha256"] = "0" * 64
        rewritten = native_checkpoint._canonical_json(footer)
        assert len(rewritten) == footer_len
        data[footer_offset : footer_offset + footer_len] = rewritten
    target.write_bytes(data)
    with pytest.raises(ACSNativeCheckpointError):
        load_acs_native_streamed_checkpoint(target)


def test_native_streamed_fsync_failure_cleans_staging_and_retains_prior_durable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = Path(_write(_projection(), tmp_path / "fsync.acsnative"))
    original = target.read_bytes()

    def fail_fsync(_fd: int) -> None:
        raise OSError("injected fsync failure")

    monkeypatch.setattr(native_checkpoint.os, "fsync", fail_fsync)
    with pytest.raises(OSError, match="injected fsync failure"):
        _write(_projection(), target)
    assert target.read_bytes() == original
    assert not list(tmp_path.glob(".*.payload.tmp"))
    assert not list(tmp_path.glob(".*.tmp"))


def test_native_streamed_provider_route_publishes_and_fresh_loads(tmp_path: Path) -> None:
    provider = ProductionEngineLifecycleV1()
    projection = _authority_projection(generation=4, tick=0)
    expected_digest = provider.terminal_projection_digest(projection)
    request = ResidentBatchRequestV1(
        profile_id="acs-native-test",
        batch_id="batch-0000",
        cell_ids=("cell-0",),
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
        source_projections=(projection,),
        generation=4,
        tick=0,
        absolute_next_tick=None,
        segment_ordinal=1,
        expected_source_projection_digest_sha256s={"cell-0": expected_digest},
    )

    published = Path(result["bundle_rows"][0]["path"])
    assert published.suffix == ".acsnative"
    assert published.is_file()
    assert result["terminal_cell_digests"] == {"cell-0": expected_digest}
    assert result["cell_checkpoint_bytes"] == {"cell-0": published.stat().st_size}
    receipt = result["native_checkpoint_receipts"]["cell-0"]
    assert receipt["status"] == "complete"
    assert receipt["path"] == str(published)
    assert receipt["artifact_sha256"] == result["bundle_rows"][0]["artifact_sha256"]
    assert receipt["payload_sha256"] == result["bundle_rows"][0]["payload_sha256"]
    assert receipt["seal_sha256"] == result["bundle_rows"][0]["seal_sha256"]
    assert receipt["payload_bytes"] == result["bundle_rows"][0]["payload_bytes"]
    assert receipt["chunk_count"] == result["bundle_rows"][0]["chunk_count"]
    assert result["bundle_rows"][0]["physical_format"] == (
        "acs100x_native_streamed_checkpoint_v1"
    )
    telemetry = result["cell_stage_telemetry_seconds"]["cell-0"]
    assert telemetry["native_pack_payload_bytes"] > 0
    restored = provider._read_projection_bundle(str(published))
    assert provider.terminal_projection_digest(restored) == expected_digest


def test_native_and_legacy_publications_preserve_logical_digest_layers(
    tmp_path: Path,
) -> None:
    provider = ProductionEngineLifecycleV1()
    projection = _authority_projection(generation=4, tick=0)
    expected_digest = provider.terminal_projection_digest(projection)
    legacy_request = ResidentBatchRequestV1(
        profile_id="acs-native-test",
        batch_id="batch-legacy",
        cell_ids=("cell-0",),
        generations=4,
        ticks_per_generation=160,
        checkpoint_milestones=(4,),
        parameter_arrays=(),
        seed_arrays=(),
        execution_route_sha256="c" * 64,
    )
    native_request = ResidentBatchRequestV1(
        profile_id="acs-native-test",
        batch_id="batch-native",
        cell_ids=("cell-0",),
        generations=4,
        ticks_per_generation=160,
        checkpoint_milestones=(4,),
        parameter_arrays=(),
        seed_arrays=(),
        execution_route_sha256="c" * 64,
        acs_checkpoint_slots=2,
        acs_capture_instrumentation=True,
        native_streamed_checkpoint=True,
    )
    legacy_result = provider._persist_projection_checkpoint_set_v1(
        batch_root=tmp_path / "legacy",
        request=legacy_request,
        source_projections=(projection,),
        generation=4,
        tick=0,
        absolute_next_tick=None,
        segment_ordinal=1,
        expected_source_projection_digest_sha256s={"cell-0": expected_digest},
    )
    native_result = provider._persist_projection_checkpoint_set_v1(
        batch_root=tmp_path / "native",
        request=native_request,
        source_projections=(projection,),
        generation=4,
        tick=0,
        absolute_next_tick=None,
        segment_ordinal=1,
        expected_source_projection_digest_sha256s={"cell-0": expected_digest},
    )
    legacy_path = Path(legacy_result["bundle_rows"][0]["path"])
    native_path = Path(native_result["bundle_rows"][0]["path"])
    legacy_projection = provider._read_projection_bundle(str(legacy_path))
    native_projection = provider._read_projection_bundle(str(native_path))
    assert legacy_result["terminal_cell_digests"] == {"cell-0": expected_digest}
    assert provider.terminal_projection_digest(native_projection) == expected_digest
    native_logical_projection = migrate_native_checkpoint_projection_v1_to_v2(
        native_projection
    )
    assert native_logical_projection["schema_version"] == legacy_projection[
        "schema_version"
    ]
    for layer in (DigestLayer.TRAJECTORY_STATE, DigestLayer.CANONICAL_CHECKPOINT_STORAGE):
        assert checkpoint_digest_manifest(legacy_projection, layer=layer)["digest"] == checkpoint_digest_manifest(native_logical_projection, layer=layer)["digest"]
    packed = load_acs_native_streamed_checkpoint(native_path)
    assert packed["header"]["cell_id"] == "cell-0"
    assert packed["header"]["generation"] == 4
    assert packed["architecture_b_identity"]["snapshot_digest_sha256"] == expected_digest
