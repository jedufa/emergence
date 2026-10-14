"""Focused host contracts for immutable world-sharded ACS checkpoints."""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
import threading
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import torch

from engine.campaigns import acs_world_sharded_checkpoint_v1 as sharded
from engine.campaigns import acs_physical_batch_packed_checkpoint_v1 as packed
from engine.campaigns.acs_physical_batch_packed_checkpoint_v1 import (
    RaggedFieldV1,
    ZeroFillFieldV1,
    capture_physical_batch_packed_checkpoint_v1,
    finalize_physical_batch_packed_checkpoint_v1,
)
from engine.native_runtime.checkpoint_digests import (
    checkpoint_source_projection_authority_manifest,
)
from tests.perf.test_eiel_native_checkpoint_single_projection_authority import (
    _projection as _authority_projection,
)


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")


def _recipe(
    cell_ids: tuple[str, ...], descriptors: list[dict[str, object]]
) -> dict[str, object]:
    body: dict[str, object] = {
        "schema_version": "eiel_acs_direct_resident_physical_settlement_recipe_v1",
        "batch_id": "world-shard-fixture",
        "cell_ids": list(cell_ids),
        "absolute_next_tick": 960,
        "route": {
            "schema_version": "tick_native_execution_route_v1",
            "time_authority": "absolute_tick_v1",
            "checkpoint_at_tick": 960,
        },
        "source_descriptor_recipe": {
            "schema_version": "eiel_bounded_source_descriptor_recipe_v1",
            "cell_ids": list(cell_ids),
            "descriptors": descriptors,
            "source_projection_sha256s": {
                cell_id: str(descriptors[index]["source_projection_sha256"])
                for index, cell_id in enumerate(cell_ids)
            },
        },
        "direct_result_scalars": {
            "event_rows_by_cell": [[index] for index in range(len(cell_ids))],
            "source_projection_cell_ids": list(cell_ids),
        },
        "checkpoint_projection_inputs": {
            "tw_rngv2_logical_world_uids": [
                f"logical-{index}" for index in range(len(cell_ids))
            ],
            "genome_v2_causal_physical_cell_ordinals": list(range(len(cell_ids))),
        },
        "live_bank_scalars": {
            "genome_v2_founder_profiles": [
                {"index": index} for index in range(len(cell_ids))
            ]
        },
    }
    body["recipe_sha256"] = hashlib.sha256(_canonical(body)).hexdigest()
    return body


def _packed_fixture(
    tmp_path: Path, *, cells: int
) -> tuple[Path, dict[str, object], dict[str, object]]:
    cell_ids = tuple(f"cell-{index:03d}" for index in range(cells))
    descriptors: list[dict[str, object]] = []
    source_mlp: list[torch.Tensor] = []
    final_mlp: list[torch.Tensor] = []
    source_aliases: list[torch.Tensor] = []
    indexed_sources: list[dict[str, object]] = []
    for index in range(cells):
        projection = _authority_projection(generation=4, tick=0)
        source_policy = torch.zeros((1024, 4), dtype=torch.float32)
        source_policy[index % source_policy.shape[0], 0] = float(index + 1)
        projection["tensors"]["mlp_pack"] = source_policy
        projection["tensor_manifest"]["mlp_pack"] = {
            "dtype": "torch.float32",
            "shape": [1024, 4],
            "device": "cpu",
        }
        source_alias = torch.arange(8, dtype=torch.int32) + index
        projection["tensors"]["source_alias_pack"] = source_alias
        projection["tensor_manifest"]["source_alias_pack"] = {
            "dtype": "torch.int32",
            "shape": [8],
            "device": "cpu",
        }
        source_aliases.append(source_alias)
        source_mlp.append(source_policy)
        final_policy = source_policy.clone()
        final_policy[(index + 1) % final_policy.shape[0], 1] = 2.0
        final_mlp.append(final_policy)
        component_path = tmp_path / f"source-{index}.pt"
        torch.save(projection, component_path)
        source_digest = str(
            checkpoint_source_projection_authority_manifest(projection)[
                "source_projection_sha256"
            ]
        )
        indexed_source = sharded.write_indexed_source_projection_v1(
            projection,
            tmp_path / f"source-{index}.acsindexed",
            source_projection_sha256=source_digest,
        )
        indexed_sources.append(indexed_source)
        descriptors.append(
            {
                "kind": "component",
                "path": str(component_path),
                "file_sha256": hashlib.sha256(component_path.read_bytes()).hexdigest(),
                "source_projection_sha256": source_digest,
                "indexed_source": indexed_source,
            }
        )
    common = torch.arange(cells * 6, dtype=torch.float32).reshape(cells, 2, 3)
    flags = torch.arange(cells * 2, dtype=torch.int32).reshape(cells, 2)
    ragged_values = tuple(
        torch.arange(index + 1, dtype=torch.int64) + index * 100
        for index in range(cells)
    )
    fields: dict[str, object] = {
        "common": common,
        "flags": flags,
        "alive_mask": torch.tensor(
            [[True, False, True, False]] * cells, dtype=torch.bool
        ),
        "ragged": RaggedFieldV1(ragged_values),
        "gru_pack": ZeroFillFieldV1(torch.float32, (cells, 2, 2)),
        "mlp_pack": torch.stack(final_mlp),
        "source_alias_pack": torch.stack(source_aliases),
    }
    target = tmp_path / "source.acspb"
    descriptor = capture_physical_batch_packed_checkpoint_v1(
        fields,
        target,
        cell_ids=cell_ids,
        batch_id="world-shard-fixture",
        scalar_metadata={"settlement_recipe": _recipe(cell_ids, descriptors)},
        chunk_bytes=7,
    )
    receipt = finalize_physical_batch_packed_checkpoint_v1(
        descriptor, publication_path=target
    )
    expected = {
        "cell_ids": cell_ids,
        "common": common,
        "flags": flags,
        "ragged": ragged_values,
        "source_mlp": tuple(source_mlp),
        "final_mlp": torch.stack(final_mlp),
        "source_aliases": torch.stack(source_aliases),
        "indexed_sources": indexed_sources,
    }
    return target, receipt, expected


def _write_set(
    source: Path,
    receipt: dict[str, object],
    destination: Path,
    *,
    cells: int,
    workers: int = 4,
) -> dict[str, object]:
    return sharded.write_world_sharded_checkpoint_set_from_packed_v1(
        source,
        destination,
        raw_receipt=receipt,
        worker_count=workers,
        copy_chunk_bytes=5,
        required_world_count=cells,
    )


@pytest.mark.parametrize("cells", [1, 8])
def test_world_set_round_trips_common_ragged_and_zero_fill(
    tmp_path: Path, cells: int
) -> None:
    source, receipt, expected = _packed_fixture(tmp_path, cells=cells)
    result = _write_set(source, receipt, tmp_path / "sets", cells=cells)
    assert result["source_verification_mode"] == "sealed_receipt_fingerprint_v1"
    manifest = sharded.load_world_checkpoint_set_manifest_v1(result["manifest_path"])

    assert manifest["world_count"] == cells
    assert manifest["cell_ids"] == list(expected["cell_ids"])
    assert len(tuple(Path(result["set_root"]).glob("objects/*.acsworld"))) == cells
    assert len(result["physical_compact_result_rows"]) == cells
    assert all(
        row["metrics"]["population.terminal_alive"] == 2
        and row["metrics"]["runtime.terminal_absolute_tick"] == 960
        for row in result["physical_compact_result_rows"]
    )
    for index, cell_id in enumerate(expected["cell_ids"]):
        loaded = sharded.load_selected_world_from_set_v1(
            result["manifest_path"], world_id=cell_id
        )
        assert loaded["physical_cell_index"] == index
        assert loaded["absolute_next_tick"] == 960
        torch.testing.assert_close(
            loaded["fields"]["common"], expected["common"][index]
        )
        torch.testing.assert_close(loaded["fields"]["flags"], expected["flags"][index])
        torch.testing.assert_close(
            loaded["fields"]["ragged"], expected["ragged"][index]
        )
        torch.testing.assert_close(
            loaded["fields"]["mlp_pack"], expected["final_mlp"][index]
        )
        assert any(
            row["storage"] == "payload_zstd_v1" for row in loaded["header"]["fields"]
        )
        assert loaded["header"]["source_dependency"]["kind"] == (
            "compact_source_projection_v1"
        )
        assert (
            loaded["header"]["source_dependency"]["source_projection_digest_binding"]
            == "verified_indexed_source_v1"
        )
        assert any(
            row["storage"] == "world_field_xor_zstd_v1"
            for row in loaded["header"]["source_dependency"]["tensors"]
        )
        torch.testing.assert_close(
            loaded["source_projection"]["tensors"]["mlp_pack"],
            expected["source_mlp"][index],
        )
        source_alias = loaded["source_projection"]["tensors"]["source_alias_pack"]
        torch.testing.assert_close(source_alias, expected["source_aliases"][index])
        assert (
            source_alias.data_ptr() != loaded["fields"]["source_alias_pack"].data_ptr()
        )
        torch.testing.assert_close(
            loaded["fields"]["gru_pack"], torch.zeros((2, 2), dtype=torch.float32)
        )


def test_selected_restore_reads_one_world_and_ignores_unrelated_payload_damage(
    tmp_path: Path,
) -> None:
    source, receipt, expected = _packed_fixture(tmp_path, cells=8)
    result = _write_set(source, receipt, tmp_path / "sets", cells=8)
    manifest = sharded.load_world_checkpoint_set_manifest_v1(result["manifest_path"])
    unrelated = Path(result["set_root"]) / manifest["objects"][7]["relative_path"]
    with unrelated.open("r+b") as stream:
        stream.seek(sharded.PREFIX.size)
        original = stream.read(1)
        stream.seek(sharded.PREFIX.size)
        stream.write(bytes([original[0] ^ 0xFF]))

    counter: dict[str, int] = {}
    selected = sharded.load_selected_world_from_set_v1(
        result["manifest_path"], world_id=expected["cell_ids"][3], byte_counter=counter
    )
    assert selected["cell_id"] == expected["cell_ids"][3]
    assert counter["world_objects_read"] == 1
    assert counter["set_manifests_read"] == 1
    assert counter["bytes_read"] == (
        Path(selected["path"]).stat().st_size
        + Path(result["manifest_path"]).stat().st_size
    )
    with pytest.raises(sharded.ACSWorldShardedCheckpointError, match="digest differs"):
        sharded.load_selected_world_from_set_v1(
            result["manifest_path"], world_id=expected["cell_ids"][7]
        )


def test_compact_source_delta_corruption_fails_closed(tmp_path: Path) -> None:
    source, receipt, expected = _packed_fixture(tmp_path, cells=1)
    result = _write_set(source, receipt, tmp_path / "sets", cells=1)
    manifest = sharded.load_world_checkpoint_set_manifest_v1(result["manifest_path"])
    target = Path(result["set_root"]) / manifest["objects"][0]["relative_path"]
    _prefix, _header_raw, header, _footer_raw, _payload_len = sharded._read_world_parts(
        target
    )
    dependency = header["source_dependency"]
    delta = next(
        row
        for row in dependency["tensors"]
        if row["storage"] == "world_field_xor_zstd_v1"
    )
    with target.open("r+b") as stream:
        offset = sharded.PREFIX.size + int(dependency["offset"]) + int(delta["offset"])
        stream.seek(offset)
        original = stream.read(1)
        stream.seek(offset)
        stream.write(bytes([original[0] ^ 0xFF]))

    with pytest.raises(
        sharded.ACSWorldShardedCheckpointError,
        match="embedded source dependency digest differs",
    ):
        sharded.load_selected_world_from_set_v1(
            result["manifest_path"], world_id=expected["cell_ids"][0]
        )


def test_compressed_world_field_corruption_fails_closed(tmp_path: Path) -> None:
    source, receipt, expected = _packed_fixture(tmp_path, cells=1)
    result = _write_set(source, receipt, tmp_path / "sets", cells=1)
    manifest = sharded.load_world_checkpoint_set_manifest_v1(result["manifest_path"])
    target = Path(result["set_root"]) / manifest["objects"][0]["relative_path"]
    _prefix, _header_raw, header, _footer_raw, _payload_len = sharded._read_world_parts(
        target
    )
    compressed = next(
        row for row in header["fields"] if row["storage"] == "payload_zstd_v1"
    )
    with target.open("r+b") as stream:
        offset = sharded.PREFIX.size + int(compressed["offset"])
        stream.seek(offset)
        original = stream.read(1)
        stream.seek(offset)
        stream.write(bytes([original[0] ^ 0xFF]))

    with pytest.raises(
        sharded.ACSWorldShardedCheckpointError,
        match="compressed field|field digest differs",
    ):
        sharded.load_selected_world_from_set_v1(
            result["manifest_path"], world_id=expected["cell_ids"][0]
        )


def test_unreceipted_source_projection_digest_is_verified(tmp_path: Path) -> None:
    source, _receipt, expected = _packed_fixture(tmp_path, cells=1)
    result = sharded.write_world_sharded_checkpoint_set_from_packed_v1(
        source,
        tmp_path / "sets",
        required_world_count=1,
    )
    loaded = sharded.load_selected_world_from_set_v1(
        result["manifest_path"], world_id=expected["cell_ids"][0]
    )

    assert result["source_verification_mode"] == "full_payload_reverification_v1"
    assert (
        loaded["header"]["source_dependency"]["source_projection_digest_binding"]
        == "verified_indexed_source_v1"
    )


def test_indexed_source_omits_zero_payload_and_fails_closed_on_mutation(
    tmp_path: Path,
) -> None:
    source, receipt, expected = _packed_fixture(tmp_path, cells=1)
    indexed = expected["indexed_sources"][0]
    fd, header, _payload_base = sharded._open_indexed_source_projection_v1(
        indexed,
        expected_source_projection_sha256=str(indexed["source_projection_sha256"]),
    )
    try:
        assert header["zero_fill_logical_bytes"] > 0
        assert any(row["storage"] == "zero_fill" for row in header["tensors"])
        assert (
            header["payload_bytes"] - header["skeleton"]["bytes"]
            < header["logical_tensor_bytes"]
        )
    finally:
        import os

        os.close(fd)

    indexed_path = Path(str(indexed["path"]))
    with indexed_path.open("r+b") as stream:
        stream.seek(-1, 2)
        original = stream.read(1)
        stream.seek(-1, 2)
        stream.write(bytes([original[0] ^ 0xFF]))
    with pytest.raises(
        sharded.ACSWorldShardedCheckpointError,
        match="indexed source descriptor fingerprint differs",
    ):
        _write_set(source, receipt, tmp_path / "sets", cells=1)


def test_legacy_zlib_field_decoder_remains_compatible() -> None:
    raw = (b"legacy-world-field" * 1000) + b"!"
    stored = zlib.compress(raw, level=1)
    assert (
        sharded._decompress_world_field_v1(
            stored,
            field={
                "storage": "payload_zlib_v1",
                "compression": sharded.LEGACY_FIELD_COMPRESSION,
            },
            logical_bytes=len(raw),
        )
        == raw
    )


def test_receipted_source_mutation_fails_before_sharding(tmp_path: Path) -> None:
    source, receipt, _expected = _packed_fixture(tmp_path, cells=1)
    with source.open("r+b") as stream:
        stream.seek(sharded.PREFIX.size)
        original = stream.read(1)
        stream.seek(sharded.PREFIX.size)
        stream.write(bytes([original[0] ^ 0xFF]))

    with pytest.raises(
        sharded.ACSWorldShardedCheckpointError,
        match="sealed publication fingerprint",
    ):
        _write_set(source, receipt, tmp_path / "sets", cells=1)


def test_transient_capture_shards_without_raw_finalize_and_mutation_fails_closed(
    tmp_path: Path,
) -> None:
    fixture_root = tmp_path / "valid"
    fixture_root.mkdir()
    source, _receipt, expected = _packed_fixture(fixture_root, cells=1)
    # Recreate an unsealed capture using the sealed fixture's loaded fields and
    # scalar authority so this test exercises the producer descriptor route.
    sealed = packed.load_physical_batch_packed_checkpoint_v1(source)
    transient_fields = dict(sealed["fields"])
    transient_fields["ragged"] = RaggedFieldV1(transient_fields["ragged"])
    transient_target = tmp_path / "transient.acspb"
    descriptor = capture_physical_batch_packed_checkpoint_v1(
        transient_fields,
        transient_target,
        cell_ids=tuple(expected["cell_ids"]),
        batch_id="world-shard-fixture",
        scalar_metadata=sealed["header"]["scalar_metadata"],
        chunk_bytes=7,
    )
    capture_path = Path(str(descriptor["capture_path"]))
    result = sharded.write_world_sharded_checkpoint_set_from_packed_v1(
        capture_path,
        tmp_path / "sets",
        raw_capture_descriptor=descriptor,
        required_world_count=1,
    )
    assert result["source_verification_mode"] == (
        "captured_unsealed_descriptor_fingerprint_v1"
    )
    assert result["source_raw_artifact_sha256"] is None
    assert (
        result["source_physical_authority_sha256"]
        == descriptor["capture_authority_sha256"]
    )

    damaged_target = tmp_path / "damaged.acspb"
    damaged = capture_physical_batch_packed_checkpoint_v1(
        transient_fields,
        damaged_target,
        cell_ids=tuple(expected["cell_ids"]),
        batch_id="world-shard-fixture-damaged",
        scalar_metadata=sealed["header"]["scalar_metadata"],
        chunk_bytes=7,
    )
    damaged_path = Path(str(damaged["capture_path"]))
    with damaged_path.open("r+b") as stream:
        stream.seek(packed.PREFIX.size)
        original = stream.read(1)
        stream.seek(packed.PREFIX.size)
        stream.write(bytes([original[0] ^ 0xFF]))
    with pytest.raises(
        sharded.ACSWorldShardedCheckpointError,
        match="capture fingerprint",
    ):
        sharded.write_world_sharded_checkpoint_set_from_packed_v1(
            damaged_path,
            tmp_path / "damaged-sets",
            raw_capture_descriptor=damaged,
            required_world_count=1,
        )


def test_manifest_and_world_identity_are_deterministic_across_worker_counts(
    tmp_path: Path,
) -> None:
    source, receipt, _expected = _packed_fixture(tmp_path, cells=8)
    serial = _write_set(source, receipt, tmp_path / "serial", cells=8, workers=1)
    parallel = _write_set(source, receipt, tmp_path / "parallel", cells=8, workers=4)
    serial_manifest = Path(serial["manifest_path"]).read_bytes()
    parallel_manifest = Path(parallel["manifest_path"]).read_bytes()

    assert serial_manifest == parallel_manifest
    assert serial["manifest_sha256"] == parallel["manifest_sha256"]
    assert [row["artifact_sha256"] for row in serial["objects"]] == [
        row["artifact_sha256"] for row in parallel["objects"]
    ]


def test_corruption_truncation_missing_object_and_manifest_duplication_fail_closed(
    tmp_path: Path,
) -> None:
    source, receipt, _expected = _packed_fixture(tmp_path, cells=4)
    result = _write_set(source, receipt, tmp_path / "sets", cells=4)
    manifest_path = Path(result["manifest_path"])
    manifest = sharded.load_world_checkpoint_set_manifest_v1(manifest_path)
    object_paths = [
        Path(result["set_root"]) / row["relative_path"] for row in manifest["objects"]
    ]

    truncated = tmp_path / "truncated.acsworld"
    truncated.write_bytes(object_paths[0].read_bytes()[:-1])
    with pytest.raises(sharded.ACSWorldShardedCheckpointError, match="size differs"):
        sharded.load_world_checkpoint_object_v1(truncated)

    object_paths[1].unlink()
    with pytest.raises(sharded.ACSWorldShardedCheckpointError, match="cannot be read"):
        sharded.load_selected_world_from_set_v1(
            manifest_path, world_id=manifest["cell_ids"][1]
        )

    duplicated = copy.deepcopy(manifest)
    duplicated["cell_ids"][3] = duplicated["cell_ids"][2]
    duplicated["objects"][3]["cell_id"] = duplicated["cell_ids"][2]
    body = dict(duplicated)
    body.pop("manifest_sha256")
    duplicated["manifest_sha256"] = hashlib.sha256(_canonical(body)).hexdigest()
    duplicate_root = tmp_path / str(duplicated["checkpoint_set_id"])
    duplicate_root.mkdir()
    duplicate_path = duplicate_root / "checkpoint_set_manifest.json"
    duplicate_path.write_bytes(_canonical(duplicated))
    with pytest.raises(sharded.ACSWorldShardedCheckpointError, match="duplicates"):
        sharded.load_world_checkpoint_set_manifest_v1(duplicate_path)


def test_worker_failure_quarantines_incomplete_set_without_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, receipt, _expected = _packed_fixture(tmp_path, cells=8)
    original = sharded._write_one_world

    def injected(**kwargs: object) -> dict[str, object]:
        if kwargs["cell_index"] == 2:
            raise sharded.ACSWorldShardedCheckpointError("injected shard failure")
        return original(**kwargs)

    monkeypatch.setattr(sharded, "_write_one_world", injected)
    destination = tmp_path / "sets"
    with pytest.raises(sharded.ACSWorldShardedCheckpointError, match="injected"):
        _write_set(source, receipt, destination, cells=8)

    assert not tuple(destination.glob("*/checkpoint_set_manifest.json"))
    quarantines = tuple((destination / "quarantine").glob("*/failure.json"))
    assert len(quarantines) == 1
    failure = json.loads(quarantines[0].read_text(encoding="ascii"))
    assert failure["status"] == "incomplete_quarantined"


def test_source_recipe_and_required_world_count_are_fail_closed(tmp_path: Path) -> None:
    source, receipt, expected = _packed_fixture(tmp_path, cells=4)
    with pytest.raises(sharded.ACSWorldShardedCheckpointError, match="exactly 136"):
        _write_set(source, receipt, tmp_path / "sets", cells=136)

    descriptors = [
        {
            "kind": "component",
            "path": str(tmp_path / f"source-{index}.pt"),
            "file_sha256": hashlib.sha256(
                (tmp_path / f"source-{index}.pt").read_bytes()
            ).hexdigest(),
            "source_projection_sha256": str(
                checkpoint_source_projection_authority_manifest(
                    _authority_projection(generation=4, tick=0)
                )["source_projection_sha256"]
            ),
        }
        for index in range(len(expected["cell_ids"]))
    ]
    recipe = _recipe(expected["cell_ids"], descriptors)
    recipe["absolute_next_tick"] = 961
    with pytest.raises(
        sharded.ACSWorldShardedCheckpointError, match="digest is invalid"
    ):
        sharded.project_world_scalar_metadata_v1(
            {"settlement_recipe": recipe},
            cell_index=0,
            cell_id=expected["cell_ids"][0],
            batch_cell_ids=expected["cell_ids"],
        )


def test_duplicate_publication_fails_without_replacing_complete_set(
    tmp_path: Path,
) -> None:
    source, receipt, _expected = _packed_fixture(tmp_path, cells=4)
    destination = tmp_path / "sets"
    first = _write_set(source, receipt, destination, cells=4)
    manifest_before = Path(first["manifest_path"]).read_bytes()

    with pytest.raises(sharded.ACSWorldShardedCheckpointError, match="already exists"):
        _write_set(source, receipt, destination, cells=4)

    assert Path(first["manifest_path"]).read_bytes() == manifest_before
    assert not (destination / "quarantine").exists()


def test_cancelled_set_is_quarantined_without_complete_manifest(tmp_path: Path) -> None:
    source, receipt, _expected = _packed_fixture(tmp_path, cells=4)
    cancelled = threading.Event()
    cancelled.set()
    destination = tmp_path / "sets"

    with pytest.raises(sharded.ACSWorldShardedCheckpointError, match="cancelled"):
        sharded.write_world_sharded_checkpoint_set_from_packed_v1(
            source,
            destination,
            raw_receipt=receipt,
            required_world_count=4,
            cancel_event=cancelled,
        )

    assert not tuple(destination.glob("*/checkpoint_set_manifest.json"))
    assert not tuple(
        (destination / "quarantine").glob("*/checkpoint_set_manifest.json")
    )
    assert len(tuple((destination / "quarantine").glob("*/failure.json"))) == 1


def test_directory_fsync_failure_prevents_manifest_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, receipt, _expected = _packed_fixture(tmp_path, cells=4)
    destination = tmp_path / "sets"
    real_fsync = sharded._fsync_directory
    failed = False

    def injected(path: Path) -> None:
        nonlocal failed
        if path.name == "objects" and not failed:
            failed = True
            raise OSError("injected objects fsync failure")
        real_fsync(path)

    monkeypatch.setattr(sharded, "_fsync_directory", injected)
    with pytest.raises(OSError, match="injected objects fsync failure"):
        _write_set(source, receipt, destination, cells=4)

    assert not tuple(destination.glob("*/checkpoint_set_manifest.json"))
    assert not tuple(
        (destination / "quarantine").glob("*/checkpoint_set_manifest.json")
    )


def test_deferred_object_fsync_failure_prevents_manifest_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, receipt, _expected = _packed_fixture(tmp_path, cells=4)
    destination = tmp_path / "sets"
    real_fsync = sharded.os.fsync
    failed = False

    def injected(descriptor: int) -> None:
        nonlocal failed
        if not failed:
            failed = True
            raise OSError("injected deferred object fsync failure")
        real_fsync(descriptor)

    monkeypatch.setattr(sharded.os, "fsync", injected)
    with pytest.raises(OSError, match="injected deferred object fsync failure"):
        _write_set(source, receipt, destination, cells=4)

    assert failed
    assert not tuple(destination.glob("*/checkpoint_set_manifest.json"))
    assert not tuple(
        (destination / "quarantine").glob("*/checkpoint_set_manifest.json")
    )
    assert len(tuple((destination / "quarantine").glob("*/failure.json"))) == 1


def test_final_publication_rename_failure_quarantines_without_complete_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, receipt, _expected = _packed_fixture(tmp_path, cells=4)
    destination = (tmp_path / "sets").resolve()
    real_replace = sharded.os.replace
    failed = False

    def injected(source_path: object, destination_path: object) -> None:
        nonlocal failed
        source_value = Path(source_path).resolve()
        destination_value = Path(destination_path).resolve()
        if (
            not failed
            and source_value.name.endswith(".staging")
            and destination_value.parent == destination
        ):
            failed = True
            raise OSError("injected final publication rename failure")
        real_replace(source_path, destination_path)

    monkeypatch.setattr(sharded.os, "replace", injected)
    with pytest.raises(OSError, match="injected final publication rename failure"):
        _write_set(source, receipt, destination, cells=4)

    assert not tuple(destination.glob("*/checkpoint_set_manifest.json"))
    assert not tuple(
        (destination / "quarantine").glob("*/checkpoint_set_manifest.json")
    )
    assert len(tuple((destination / "quarantine").glob("*/failure.json"))) == 1


def test_first_middle_last_selective_reads_equal_full_restore_and_one_object(
    tmp_path: Path,
) -> None:
    source, receipt, expected = _packed_fixture(tmp_path, cells=8)
    result = _write_set(source, receipt, tmp_path / "sets", cells=8)
    full_counter: dict[str, int] = {}
    full = sharded.load_full_world_checkpoint_set_v1(
        result["manifest_path"], byte_counter=full_counter
    )

    assert full_counter["set_manifests_read"] == 1
    assert full_counter["world_objects_read"] == 8
    for index in (0, 3, 7):
        counter: dict[str, int] = {}
        selected = sharded.load_selected_world_from_set_v1(
            result["manifest_path"],
            world_id=expected["cell_ids"][index],
            byte_counter=counter,
        )
        assert counter["set_manifests_read"] == 1
        assert counter["world_objects_read"] == 1
        assert selected["artifact_sha256"] == full[index]["artifact_sha256"]
        for field in selected["field_order"]:
            torch.testing.assert_close(
                selected["fields"][field], full[index]["fields"][field]
            )


def test_manifest_schema_replacement_and_premature_publication_fail_closed(
    tmp_path: Path,
) -> None:
    source, receipt, expected = _packed_fixture(tmp_path, cells=4)
    result = _write_set(source, receipt, tmp_path / "sets", cells=4)
    manifest_path = Path(result["manifest_path"])
    manifest = sharded.load_world_checkpoint_set_manifest_v1(manifest_path)

    wrong_schema = copy.deepcopy(manifest)
    wrong_schema["schema_version"] = "future_unsupported_schema"
    body = dict(wrong_schema)
    body.pop("manifest_sha256")
    wrong_schema["manifest_sha256"] = hashlib.sha256(_canonical(body)).hexdigest()
    wrong_root = tmp_path / "wrong-schema" / str(wrong_schema["checkpoint_set_id"])
    wrong_root.mkdir(parents=True)
    wrong_path = wrong_root / "checkpoint_set_manifest.json"
    wrong_path.write_bytes(_canonical(wrong_schema))
    with pytest.raises(sharded.ACSWorldShardedCheckpointError, match="authority"):
        sharded.load_world_checkpoint_set_manifest_v1(wrong_path)

    noncanonical_root = tmp_path / "noncanonical" / str(manifest["checkpoint_set_id"])
    noncanonical_root.mkdir(parents=True)
    noncanonical_path = noncanonical_root / "checkpoint_set_manifest.json"
    noncanonical_path.write_text(
        json.dumps(manifest, sort_keys=True, indent=2), encoding="ascii"
    )
    with pytest.raises(sharded.ACSWorldShardedCheckpointError, match="canonical"):
        sharded.load_world_checkpoint_set_manifest_v1(noncanonical_path)

    wrong_count = copy.deepcopy(manifest)
    wrong_count["world_count"] = 3
    body = dict(wrong_count)
    body.pop("manifest_sha256")
    wrong_count["manifest_sha256"] = hashlib.sha256(_canonical(body)).hexdigest()
    wrong_count_root = tmp_path / "wrong-count" / str(manifest["checkpoint_set_id"])
    wrong_count_root.mkdir(parents=True)
    wrong_count_path = wrong_count_root / "checkpoint_set_manifest.json"
    wrong_count_path.write_bytes(_canonical(wrong_count))
    with pytest.raises(sharded.ACSWorldShardedCheckpointError, match="authority"):
        sharded.load_world_checkpoint_set_manifest_v1(wrong_count_path)

    replaced = copy.deepcopy(manifest)
    replaced["objects"][0]["relative_path"] = replaced["objects"][1]["relative_path"]
    body = dict(replaced)
    body.pop("manifest_sha256")
    replaced["manifest_sha256"] = hashlib.sha256(_canonical(body)).hexdigest()
    manifest_path.write_bytes(_canonical(replaced))
    with pytest.raises(
        sharded.ACSWorldShardedCheckpointError,
        match="artifact identity|differs from the set manifest",
    ):
        sharded.load_selected_world_from_set_v1(
            manifest_path, world_id=expected["cell_ids"][0]
        )

    premature = copy.deepcopy(manifest)
    premature["objects"][2]["relative_path"] = "objects/not-yet-published.acsworld"
    body = dict(premature)
    body.pop("manifest_sha256")
    premature["manifest_sha256"] = hashlib.sha256(_canonical(body)).hexdigest()
    manifest_path.write_bytes(_canonical(premature))
    with pytest.raises(sharded.ACSWorldShardedCheckpointError, match="cannot be read"):
        sharded.load_selected_world_from_set_v1(
            manifest_path, world_id=expected["cell_ids"][2]
        )


def test_worker_submission_queue_is_bounded_to_worker_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, receipt, _expected = _packed_fixture(tmp_path, cells=8)
    max_unfinished = 0
    unfinished = 0
    lock = threading.Lock()

    class TrackingExecutor(ThreadPoolExecutor):
        def submit(self, fn, /, *args, **kwargs):
            nonlocal max_unfinished, unfinished
            with lock:
                unfinished += 1
                max_unfinished = max(max_unfinished, unfinished)
            future = super().submit(fn, *args, **kwargs)

            def finished(_future: object) -> None:
                nonlocal unfinished
                with lock:
                    unfinished -= 1

            future.add_done_callback(finished)
            return future

    monkeypatch.setattr(sharded, "ThreadPoolExecutor", TrackingExecutor)
    result = _write_set(source, receipt, tmp_path / "sets", cells=8, workers=2)

    assert result["bounded_in_flight_world_jobs"] == 2
    assert max_unfinished <= 2


def test_stale_staging_is_not_promoted_or_deleted(tmp_path: Path) -> None:
    source, receipt, _expected = _packed_fixture(tmp_path, cells=4)
    destination = tmp_path / "sets"
    destination.mkdir()
    stale = destination / ".unrelated-dead-process.staging"
    stale.mkdir()
    marker = stale / "partial.tmp"
    marker.write_bytes(b"incomplete")

    result = _write_set(source, receipt, destination, cells=4)

    assert Path(result["manifest_path"]).is_file()
    assert marker.read_bytes() == b"incomplete"


def test_fresh_process_selected_and_full_restore_read_exact_object_counts(
    tmp_path: Path,
) -> None:
    source, receipt, expected = _packed_fixture(tmp_path, cells=8)
    result = _write_set(source, receipt, tmp_path / "sets", cells=8)
    command = """
import json
import sys
from engine.campaigns.acs_world_sharded_checkpoint_v1 import (
    load_full_world_checkpoint_set_v1,
    load_selected_world_from_set_v1,
)
manifest, selected_id = sys.argv[1:]
selected_counter = {}
selected = load_selected_world_from_set_v1(
    manifest, world_id=selected_id, byte_counter=selected_counter
)
full_counter = {}
full = load_full_world_checkpoint_set_v1(manifest, byte_counter=full_counter)
print(json.dumps({
    "selected_artifact": selected["artifact_sha256"],
    "full_artifact": full[3]["artifact_sha256"],
    "selected_objects": selected_counter["world_objects_read"],
    "selected_manifests": selected_counter["set_manifests_read"],
    "full_objects": full_counter["world_objects_read"],
    "full_manifests": full_counter["set_manifests_read"],
}))
"""
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            command,
            result["manifest_path"],
            expected["cell_ids"][3],
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    observed = json.loads(completed.stdout)

    assert observed["selected_artifact"] == observed["full_artifact"]
    assert observed["selected_objects"] == 1
    assert observed["selected_manifests"] == 1
    assert observed["full_objects"] == 8
    assert observed["full_manifests"] == 1
