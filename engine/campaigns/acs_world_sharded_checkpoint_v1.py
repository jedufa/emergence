"""Immutable world-sharded checkpoints derived from one physical batch.

The physical producer remains the sole CUDA-facing owner.  This module runs
after that owner has emitted a tensor-free, sealed packed checkpoint.  It
copies each selected physical range directly into one independently sealed
world object; it never constructs a B1 simulation/runtime owner and never
materializes a monolithic canonical aggregate.

The complete-set manifest is the only set publication authority.  It is
written after every world object is fsynced and content-addressed.  A failure
before that point leaves only a quarantined staging directory without a
``complete`` manifest.
"""

from __future__ import annotations

import copy
import compression.zstd as zstd
import hashlib
import io
import json
import os
import struct
import threading
import time
import zlib
from collections.abc import Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Final
from uuid import uuid4

import numpy as np
import torch

from engine.campaigns import acs_native_streamed_checkpoint_v1 as streamed
from engine.campaigns import acs_physical_batch_packed_checkpoint_v1 as packed


WORLD_SCHEMA_VERSION: Final[str] = "eiel_acs_world_checkpoint_object_v1"
WORLD_FOOTER_SCHEMA_VERSION: Final[str] = "eiel_acs_world_checkpoint_object_footer_v1"
SET_MANIFEST_SCHEMA_VERSION: Final[str] = "eiel_acs_world_checkpoint_set_manifest_v1"
WORLD_SCALAR_METADATA_SCHEMA_VERSION: Final[str] = (
    "eiel_acs_world_scalar_metadata_projection_v1"
)
COMPACT_SOURCE_SCHEMA_VERSION: Final[str] = (
    "eiel_acs_world_compact_source_projection_v1"
)
INDEXED_SOURCE_SCHEMA_VERSION: Final[str] = "eiel_acs_indexed_source_projection_v1"
INDEXED_SOURCE_MAGIC: Final[bytes] = b"ACSISV1\x00"
INDEXED_SOURCE_FORMAT_VERSION: Final[int] = 1
INDEXED_SOURCE_PREFIX: Final[struct.Struct] = struct.Struct("<8sHHQ")
MAGIC: Final[bytes] = b"ACSWV1\x00\x00"
FORMAT_VERSION: Final[int] = 1
LITTLE_ENDIAN_MARKER: Final[int] = 0x0102
PREFIX: Final[struct.Struct] = struct.Struct("<8sHHQQQ")
DEFAULT_COPY_CHUNK_BYTES: Final[int] = 8 * 1024 * 1024
MAX_COPY_CHUNK_BYTES: Final[int] = 64 * 1024 * 1024
MAX_WORLD_HEADER_BYTES: Final[int] = 64 * 1024 * 1024
MAX_WORLD_FOOTER_BYTES: Final[int] = 4 * 1024 * 1024
MAX_WORKERS: Final[int] = 32
FIELD_COMPRESSION: Final[str] = "zstd_level_1"
LEGACY_FIELD_COMPRESSION: Final[str] = "zlib_level_1"
TERMINAL_METRIC_SCHEMA_VERSION: Final[str] = "eiel_genome_v2_e1_native_cell_metrics_v1"

_PER_WORLD_SEQUENCE_FIELDS: Final[tuple[str, ...]] = (
    "event_history_frontier_by_cell",
    "event_history_prefix_references_by_cell",
    "event_rows_by_cell",
    "tick_native_source_prefix_by_cell",
)
_CHECKPOINT_PER_WORLD_SEQUENCE_FIELDS: Final[tuple[str, ...]] = (
    "tw_rngv2_logical_world_uids",
    "genome_v2_causal_physical_cell_ordinals",
)


class ACSWorldShardedCheckpointError(RuntimeError):
    """The world object or complete checkpoint set is not trustworthy."""


def _fail(message: str) -> ACSWorldShardedCheckpointError:
    return ACSWorldShardedCheckpointError(message)


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise _fail("world checkpoint metadata is not deterministic JSON") from exc


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_exact_int(value: Any, *, field: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise _fail(f"{field} must be an integer >= {minimum}")
    return value


def _file_fingerprint(path: Path) -> tuple[int, int, int, int, int]:
    try:
        stat = path.stat()
    except OSError as exc:
        raise _fail(f"checkpoint source identity is unavailable: {path}") from exc
    return (
        int(stat.st_dev),
        int(stat.st_ino),
        int(stat.st_size),
        int(stat.st_mtime_ns),
        int(stat.st_ctime_ns),
    )


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_world_objects(
    objects_dir: Path, receipts: Sequence[dict[str, Any]]
) -> float:
    """Flush sealed objects serially after parallel materialization.

    Issuing fsync from every writer thread interleaves writeback with the
    remaining compression/copy work and substantially amplifies tail latency.
    A bounded ordered flush retains the same durability boundary: every object
    is individually fsynced before the directory and set manifest are fsynced.
    """

    started = time.perf_counter()
    for row in receipts:
        relative = Path(str(row["relative_path"]))
        if relative.is_absolute() or relative.parent != Path("objects"):
            raise _fail("world checkpoint receipt path is invalid before fsync")
        descriptor = os.open(objects_dir / relative.name, os.O_RDONLY)
        file_started = time.perf_counter()
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        row["file_fsync_seconds"] = time.perf_counter() - file_started
    return time.perf_counter() - started


def _slice_sequence(
    mapping: dict[str, Any],
    field: str,
    *,
    cell_index: int,
    cells: int,
    required: bool = False,
) -> None:
    if field not in mapping:
        if required:
            raise _fail(f"world scalar authority lacks {field}")
        return
    value = mapping[field]
    if (
        isinstance(value, (str, bytes, bytearray))
        or not isinstance(value, Sequence)
        or len(value) != cells
    ):
        raise _fail(f"world scalar authority {field} does not cover the batch")
    mapping[field] = [copy.deepcopy(value[cell_index])]


def project_world_scalar_metadata_v1(
    scalar_metadata: Mapping[str, Any],
    *,
    cell_index: int,
    cell_id: str,
    batch_cell_ids: Sequence[str],
) -> dict[str, Any]:
    """Project one JSON-only settlement recipe without another world's data.

    The original recipe digest and physical ordinal remain bound.  Per-world
    sequences are reduced to one element so an object can travel without the
    other 135 worlds' scalar records.  No tensor or source payload is opened.
    """

    cell_ids = tuple(batch_cell_ids)
    if (
        not cell_ids
        or not 0 <= cell_index < len(cell_ids)
        or cell_ids[cell_index] != cell_id
        or len(set(cell_ids)) != len(cell_ids)
    ):
        raise _fail("world scalar projection cell identity is invalid")
    value = copy.deepcopy(dict(scalar_metadata))
    recipe = value.get("settlement_recipe")
    if not isinstance(recipe, Mapping):
        raise _fail("packed checkpoint lacks its settlement recipe")
    recipe = copy.deepcopy(dict(recipe))
    original_recipe_sha256 = recipe.get("recipe_sha256")
    unhashed = dict(recipe)
    unhashed.pop("recipe_sha256", None)
    if (
        not _is_sha256(original_recipe_sha256)
        or _sha256(_canonical_json(unhashed)) != original_recipe_sha256
    ):
        raise _fail("physical settlement recipe digest is invalid")
    if tuple(recipe.get("cell_ids") or ()) != cell_ids:
        raise _fail("physical settlement recipe cell order differs")

    source_recipe = recipe.get("source_descriptor_recipe")
    if not isinstance(source_recipe, Mapping):
        raise _fail("physical settlement source descriptor recipe is missing")
    source_recipe = copy.deepcopy(dict(source_recipe))
    descriptors = source_recipe.get("descriptors")
    if (
        tuple(source_recipe.get("cell_ids") or ()) != cell_ids
        or not isinstance(descriptors, list)
        or len(descriptors) != len(cell_ids)
        or not isinstance(descriptors[cell_index], Mapping)
    ):
        raise _fail("physical source descriptors do not cover the batch")
    source_digests = source_recipe.get("source_projection_sha256s")
    projected_source: dict[str, Any] = {
        **source_recipe,
        "cell_ids": [cell_id],
        "descriptors": [copy.deepcopy(dict(descriptors[cell_index]))],
    }
    if source_digests is not None:
        if not isinstance(source_digests, Mapping) or cell_id not in source_digests:
            raise _fail("physical source digest authority does not cover the world")
        projected_source["source_projection_sha256s"] = {
            cell_id: copy.deepcopy(source_digests[cell_id])
        }
    recipe["source_descriptor_recipe"] = projected_source

    direct = recipe.get("direct_result_scalars")
    if not isinstance(direct, Mapping):
        raise _fail("physical direct-result scalar authority is missing")
    direct = copy.deepcopy(dict(direct))
    for field in _PER_WORLD_SEQUENCE_FIELDS:
        _slice_sequence(
            direct,
            field,
            cell_index=cell_index,
            cells=len(cell_ids),
        )
    if "source_projection_cell_ids" in direct:
        if tuple(direct["source_projection_cell_ids"]) != cell_ids:
            raise _fail("physical source projection cell order differs")
        direct["source_projection_cell_ids"] = [cell_id]
    recipe["direct_result_scalars"] = direct

    checkpoint_inputs = recipe.get("checkpoint_projection_inputs")
    if not isinstance(checkpoint_inputs, Mapping):
        raise _fail("physical checkpoint scalar authority is missing")
    checkpoint_inputs = copy.deepcopy(dict(checkpoint_inputs))
    for field in _CHECKPOINT_PER_WORLD_SEQUENCE_FIELDS:
        _slice_sequence(
            checkpoint_inputs,
            field,
            cell_index=cell_index,
            cells=len(cell_ids),
        )
    recipe["checkpoint_projection_inputs"] = checkpoint_inputs

    live_scalars = recipe.get("live_bank_scalars")
    if isinstance(live_scalars, Mapping):
        live_scalars = copy.deepcopy(dict(live_scalars))
        if "genome_v2_founder_profiles" in live_scalars:
            _slice_sequence(
                live_scalars,
                "genome_v2_founder_profiles",
                cell_index=cell_index,
                cells=len(cell_ids),
            )
        recipe["live_bank_scalars"] = live_scalars

    compatibility = recipe.get("legacy_compatibility")
    if compatibility is not None:
        if not isinstance(compatibility, Mapping):
            raise _fail("physical private compatibility authority is malformed")
        compatibility = copy.deepcopy(dict(compatibility))
        private_direct = compatibility.get("private_settlement_direct_result_scalars")
        if private_direct is not None:
            if not isinstance(private_direct, Mapping):
                raise _fail("physical private direct-result authority is malformed")
            private_direct = copy.deepcopy(dict(private_direct))
            for field in _PER_WORLD_SEQUENCE_FIELDS:
                _slice_sequence(
                    private_direct,
                    field,
                    cell_index=cell_index,
                    cells=len(cell_ids),
                )
            if "source_projection_cell_ids" in private_direct:
                if tuple(private_direct["source_projection_cell_ids"]) != cell_ids:
                    raise _fail("physical private source projection order differs")
                private_direct["source_projection_cell_ids"] = [cell_id]
            compatibility["private_settlement_direct_result_scalars"] = private_direct
        recipe["legacy_compatibility"] = compatibility

    recipe["cell_ids"] = [cell_id]
    recipe["selected_physical_cell_index"] = cell_index
    recipe["source_batch_cell_count"] = len(cell_ids)
    recipe["source_batch_recipe_sha256"] = original_recipe_sha256
    recipe.pop("recipe_sha256", None)
    recipe["recipe_sha256"] = _sha256(_canonical_json(recipe))
    projected = {
        "schema_version": WORLD_SCALAR_METADATA_SCHEMA_VERSION,
        "cell_id": cell_id,
        "physical_cell_index": cell_index,
        "source_batch_cell_ids_sha256": _sha256(_canonical_json(list(cell_ids))),
        "source_batch_recipe_sha256": original_recipe_sha256,
        "settlement_recipe": recipe,
    }
    _canonical_json(projected)
    return projected


def _selected_source_range(
    field: Mapping[str, Any], *, cell_index: int
) -> tuple[int | None, int, list[int], str]:
    kind = field.get("kind")
    dtype = field.get("dtype")
    if not isinstance(dtype, str) or dtype not in streamed._DTYPE_BY_NAME:
        raise _fail("packed world field dtype is invalid")
    if kind == "common":
        offset = int(field["offset"]) + cell_index * int(field["cell_bytes"])
        return offset, int(field["cell_bytes"]), list(field["cell_shape"]), dtype
    if kind == "ragged":
        entries = field.get("entries")
        if not isinstance(entries, list) or not 0 <= cell_index < len(entries):
            raise _fail("packed ragged field does not cover the selected world")
        entry = entries[cell_index]
        if not isinstance(entry, Mapping) or entry.get("cell_index") != cell_index:
            raise _fail("packed ragged field selected entry is invalid")
        return (
            int(entry["offset"]),
            int(entry["bytes"]),
            list(entry["shape"]),
            dtype,
        )
    if kind == "zero_fill":
        return None, int(field["cell_bytes"]), list(field["cell_shape"]), dtype
    raise _fail("packed world field storage kind is invalid")


def _content_digest(
    *, header_sha256: str, payload_sha256: str, world_state_sha256: str
) -> str:
    return _sha256(
        _canonical_json(
            {
                "header_sha256": header_sha256,
                "payload_sha256": payload_sha256,
                "world_state_sha256": world_state_sha256,
            }
        )
    )


def _artifact_digest(
    *, prefix: bytes, header: bytes, footer_without_artifact: Mapping[str, Any]
) -> str:
    return _sha256(
        _canonical_json(
            {
                "prefix_sha256": _sha256(prefix),
                "header_sha256": _sha256(header),
                "footer": dict(footer_without_artifact),
            }
        )
    )


def _checkpoint_set_id(
    *,
    source_authority_sha256: str,
    batch_id: str,
    cell_ids: Sequence[str],
    absolute_next_tick: int,
    recipe_sha256: str,
    route: Mapping[str, Any],
) -> str:
    return _sha256(
        _canonical_json(
            {
                "schema_version": SET_MANIFEST_SCHEMA_VERSION,
                "source_physical_authority_sha256": source_authority_sha256,
                "batch_id": batch_id,
                "cell_ids": list(cell_ids),
                "absolute_next_tick": absolute_next_tick,
                "recipe_sha256": recipe_sha256,
                "route": dict(route),
            }
        )
    )


def _read_exact_at(fd: int, *, offset: int, length: int) -> bytes:
    chunks: list[bytes] = []
    remaining = length
    cursor = offset
    while remaining:
        raw = os.pread(fd, min(DEFAULT_COPY_CHUNK_BYTES, remaining), cursor)
        if not raw:
            raise _fail("packed checkpoint range is truncated during sharding")
        chunks.append(raw)
        cursor += len(raw)
        remaining -= len(raw)
    return b"".join(chunks)


def write_indexed_source_projection_v1(
    projection: Mapping[str, Any],
    destination: str | os.PathLike[str],
    *,
    source_projection_sha256: str,
) -> dict[str, Any]:
    """Publish a verified sparse source index for post-release settlement.

    The ordinary component remains the compatibility/source-import authority.
    This sidecar omits exact-zero tensor payloads and exposes authenticated
    ranges for the remaining tensors, allowing the checkpoint worker to skip
    data that aliases the captured final world without loading a full B1
    projection.
    """

    if not isinstance(projection, Mapping) or not _is_sha256(source_projection_sha256):
        raise _fail("indexed source projection authority is invalid")
    source = dict(projection)
    tensors = source.pop("tensors", None)
    if not isinstance(tensors, Mapping) or any(
        not isinstance(name, str) or not isinstance(tensor, torch.Tensor)
        for name, tensor in tensors.items()
    ):
        raise _fail("indexed source tensor authority is malformed")
    from engine.native_runtime.checkpoint_digests import (
        checkpoint_source_projection_authority_manifest,
    )

    if (
        checkpoint_source_projection_authority_manifest(projection).get(
            "source_projection_sha256"
        )
        != source_projection_sha256
    ):
        raise _fail("indexed source projection digest differs")
    skeleton_stream = io.BytesIO()
    torch.save(source, skeleton_stream)
    skeleton_raw = skeleton_stream.getvalue()
    payload_bytes = len(skeleton_raw)
    tensor_rows: list[dict[str, Any]] = []
    for name in sorted(tensors):
        tensor = tensors[name].detach().to("cpu").contiguous()
        dtype = streamed._dtype_name(tensor)
        shape = list(tensor.shape)
        logical_bytes = tensor.numel() * tensor.element_size()
        is_zero = tensor.numel() == 0 or bool(torch.count_nonzero(tensor).item() == 0)
        if is_zero:
            digest = packed.aggregate._zero_digest(logical_bytes)
            tensor_rows.append(
                {
                    "name": name,
                    "dtype": dtype,
                    "shape": shape,
                    "storage": "zero_fill",
                    "offset": None,
                    "bytes": 0,
                    "logical_bytes": logical_bytes,
                    "sha256": digest,
                }
            )
            continue
        raw = streamed._tensor_bytes(tensor)
        tensor_rows.append(
            {
                "name": name,
                "dtype": dtype,
                "shape": shape,
                "storage": "payload",
                "offset": payload_bytes,
                "bytes": len(raw),
                "logical_bytes": logical_bytes,
                "sha256": _sha256(raw),
            }
        )
        payload_bytes += len(raw)
    header = {
        "schema_version": INDEXED_SOURCE_SCHEMA_VERSION,
        "format_version": INDEXED_SOURCE_FORMAT_VERSION,
        "source_projection_sha256": source_projection_sha256,
        "skeleton": {
            "offset": 0,
            "bytes": len(skeleton_raw),
            "sha256": _sha256(skeleton_raw),
        },
        "tensors": tensor_rows,
        "tensor_manifest_sha256": _sha256(_canonical_json(tensor_rows)),
        "payload_bytes": payload_bytes,
        "logical_tensor_bytes": sum(int(row["logical_bytes"]) for row in tensor_rows),
        "zero_fill_logical_bytes": sum(
            int(row["logical_bytes"])
            for row in tensor_rows
            if row["storage"] == "zero_fill"
        ),
    }
    header_raw = _canonical_json(header)
    target = Path(destination).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.{uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(
                INDEXED_SOURCE_PREFIX.pack(
                    INDEXED_SOURCE_MAGIC,
                    INDEXED_SOURCE_FORMAT_VERSION,
                    LITTLE_ENDIAN_MARKER,
                    len(header_raw),
                )
            )
            stream.write(header_raw)
            stream.write(skeleton_raw)
            for row in tensor_rows:
                if row["storage"] != "payload":
                    continue
                tensor = tensors[str(row["name"])].detach().to("cpu").contiguous()
                raw = streamed._tensor_bytes(tensor)
                if len(raw) != row["bytes"] or _sha256(raw) != row["sha256"]:
                    raise _fail("indexed source tensor changed during publication")
                stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        if target.exists():
            raise _fail("indexed source publication target already exists")
        os.replace(temporary, target)
        _fsync_directory(target.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    stat = target.stat()
    return {
        "schema_version": INDEXED_SOURCE_SCHEMA_VERSION,
        "path": str(target),
        "source_projection_sha256": source_projection_sha256,
        "header_sha256": _sha256(header_raw),
        "payload_bytes": payload_bytes,
        "logical_tensor_bytes": header["logical_tensor_bytes"],
        "zero_fill_logical_bytes": header["zero_fill_logical_bytes"],
        "file_fingerprint": [
            int(stat.st_dev),
            int(stat.st_ino),
            int(stat.st_size),
            int(stat.st_mtime_ns),
            int(stat.st_ctime_ns),
        ],
    }


def _open_indexed_source_projection_v1(
    descriptor: Mapping[str, Any],
    *,
    expected_source_projection_sha256: str,
) -> tuple[int, dict[str, Any], int]:
    if (
        descriptor.get("schema_version") != INDEXED_SOURCE_SCHEMA_VERSION
        or descriptor.get("source_projection_sha256")
        != expected_source_projection_sha256
        or not _is_sha256(descriptor.get("header_sha256"))
    ):
        raise _fail("indexed source descriptor authority is invalid")
    path = Path(str(descriptor.get("path") or "")).expanduser().resolve()
    if list(_file_fingerprint(path)) != descriptor.get("file_fingerprint"):
        raise _fail("indexed source descriptor fingerprint differs")
    fd = os.open(path, os.O_RDONLY)
    try:
        prefix = _read_exact_at(fd, offset=0, length=INDEXED_SOURCE_PREFIX.size)
        magic, version, endian, header_len = INDEXED_SOURCE_PREFIX.unpack(prefix)
        if (
            magic != INDEXED_SOURCE_MAGIC
            or version != INDEXED_SOURCE_FORMAT_VERSION
            or endian != LITTLE_ENDIAN_MARKER
            or not 0 < header_len <= MAX_WORLD_HEADER_BYTES
        ):
            raise _fail("indexed source prefix is invalid")
        header_raw = _read_exact_at(
            fd, offset=INDEXED_SOURCE_PREFIX.size, length=header_len
        )
        try:
            header = json.loads(header_raw.decode("ascii"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise _fail("indexed source header is malformed") from exc
        if (
            not isinstance(header, dict)
            or header_raw != _canonical_json(header)
            or _sha256(header_raw) != descriptor["header_sha256"]
            or header.get("schema_version") != INDEXED_SOURCE_SCHEMA_VERSION
            or header.get("source_projection_sha256")
            != expected_source_projection_sha256
            or header.get("tensor_manifest_sha256")
            != _sha256(_canonical_json(header.get("tensors")))
        ):
            raise _fail("indexed source header authority differs")
        payload_base = INDEXED_SOURCE_PREFIX.size + header_len
        if os.fstat(fd).st_size != payload_base + int(header.get("payload_bytes", -1)):
            raise _fail("indexed source payload size differs")
        return fd, header, payload_base
    except BaseException:
        os.close(fd)
        raise


def _decompress_world_field_v1(
    stored: bytes, *, field: Mapping[str, Any], logical_bytes: int
) -> bytes:
    storage = field.get("storage")
    try:
        if storage == "payload_zstd_v1":
            if field.get("compression") != FIELD_COMPRESSION:
                raise _fail("world checkpoint field compression is invalid")
            raw = zstd.decompress(stored)
        elif storage == "payload_zlib_v1":
            if field.get("compression") != LEGACY_FIELD_COMPRESSION:
                raise _fail("world checkpoint field compression is invalid")
            decompressor = zlib.decompressobj()
            raw = decompressor.decompress(stored, logical_bytes + 1)
            if (
                not decompressor.eof
                or decompressor.unconsumed_tail
                or decompressor.unused_data
            ):
                raise _fail("world checkpoint compressed field length differs")
        elif storage == "payload":
            raw = stored
        else:
            raise _fail("world checkpoint field storage is invalid")
    except (ValueError, zstd.ZstdError, zlib.error) as exc:
        raise _fail("world checkpoint compressed field is unreadable") from exc
    if len(raw) != logical_bytes:
        raise _fail("world checkpoint compressed field length differs")
    return raw


def _compact_indexed_source_projection_payload_v1(
    descriptor: Mapping[str, Any],
    *,
    expected_source_projection_sha256: str,
    field_rows: Sequence[Mapping[str, Any]],
    world_payload_fd: int,
) -> tuple[bytes, dict[str, Any], int]:
    fd, header, payload_base = _open_indexed_source_projection_v1(
        descriptor,
        expected_source_projection_sha256=expected_source_projection_sha256,
    )
    bytes_read = INDEXED_SOURCE_PREFIX.size + len(_canonical_json(header))
    world_bytes_read = 0
    try:
        skeleton = header.get("skeleton")
        source_rows = header.get("tensors")
        if (
            not isinstance(skeleton, Mapping)
            or skeleton.get("offset") != 0
            or type(skeleton.get("bytes")) is not int
            or int(skeleton["bytes"]) <= 0
            or not _is_sha256(skeleton.get("sha256"))
            or not isinstance(source_rows, list)
        ):
            raise _fail("indexed source manifest is invalid")
        skeleton_raw = _read_exact_at(
            fd,
            offset=payload_base,
            length=int(skeleton["bytes"]),
        )
        bytes_read += len(skeleton_raw)
        if _sha256(skeleton_raw) != skeleton["sha256"]:
            raise _fail("indexed source skeleton digest differs")
        payload = bytearray(skeleton_raw)
        field_by_name = {
            str(row.get("field")): row
            for row in field_rows
            if isinstance(row, Mapping) and isinstance(row.get("field"), str)
        }
        tensor_rows: list[dict[str, Any]] = []
        names: set[str] = set()
        for source_row in source_rows:
            if not isinstance(source_row, Mapping):
                raise _fail("indexed source tensor row is malformed")
            name = source_row.get("name")
            dtype = source_row.get("dtype")
            shape = source_row.get("shape")
            logical_bytes = source_row.get("logical_bytes")
            source_storage = source_row.get("storage")
            source_offset = source_row.get("offset")
            source_bytes = source_row.get("bytes")
            tensor_sha256 = source_row.get("sha256")
            if (
                not isinstance(name, str)
                or not name
                or name in names
                or not isinstance(dtype, str)
                or dtype not in streamed._DTYPE_BY_NAME
                or not isinstance(shape, list)
                or any(type(value) is not int or value < 0 for value in shape)
                or type(logical_bytes) is not int
                or logical_bytes < 0
                or not _is_sha256(tensor_sha256)
                or source_storage not in {"zero_fill", "payload"}
                or (
                    source_storage == "zero_fill"
                    and (
                        source_offset is not None
                        or source_bytes != 0
                        or tensor_sha256 != packed.aggregate._zero_digest(logical_bytes)
                    )
                )
                or (
                    source_storage == "payload"
                    and (
                        type(source_offset) is not int
                        or type(source_bytes) is not int
                        or source_bytes != logical_bytes
                        or source_offset < int(skeleton["bytes"])
                    )
                )
            ):
                raise _fail("indexed source tensor authority is invalid")
            names.add(name)
            field = field_by_name.get(name)
            same_field = bool(
                isinstance(field, Mapping)
                and field.get("dtype") == dtype
                and field.get("shape") == shape
                and field.get("logical_bytes") == logical_bytes
                and field.get("sha256") == tensor_sha256
            )
            if same_field:
                tensor_rows.append(
                    {
                        "name": name,
                        "dtype": dtype,
                        "shape": shape,
                        "storage": "world_field_alias",
                        "alias_of": name,
                        "offset": None,
                        "bytes": 0,
                        "logical_bytes": logical_bytes,
                        "sha256": tensor_sha256,
                    }
                )
                continue
            if source_storage == "zero_fill":
                tensor_rows.append(
                    {
                        "name": name,
                        "dtype": dtype,
                        "shape": shape,
                        "storage": "zero_fill",
                        "alias_of": None,
                        "offset": None,
                        "bytes": 0,
                        "logical_bytes": logical_bytes,
                        "sha256": tensor_sha256,
                    }
                )
                continue
            tensor_raw = _read_exact_at(
                fd,
                offset=payload_base + int(source_offset),
                length=int(source_bytes),
            )
            bytes_read += len(tensor_raw)
            if _sha256(tensor_raw) != tensor_sha256:
                raise _fail("indexed source tensor digest differs")
            if (
                isinstance(field, Mapping)
                and field.get("storage")
                in {"payload", "payload_zlib_v1", "payload_zstd_v1"}
                and field.get("dtype") == dtype
                and field.get("shape") == shape
                and field.get("logical_bytes") == logical_bytes
                and type(field.get("offset")) is int
                and type(field.get("bytes")) is int
            ):
                base_stored = _read_exact_at(
                    world_payload_fd,
                    offset=PREFIX.size + int(field["offset"]),
                    length=int(field["bytes"]),
                )
                world_bytes_read += len(base_stored)
                base_raw = _decompress_world_field_v1(
                    base_stored, field=field, logical_bytes=logical_bytes
                )
                if _sha256(base_raw) != field.get("sha256"):
                    raise _fail(
                        "world checkpoint field changed during source compaction"
                    )
                xor_raw = np.bitwise_xor(
                    np.frombuffer(tensor_raw, dtype=np.uint8),
                    np.frombuffer(base_raw, dtype=np.uint8),
                ).tobytes()
                compressed = zstd.compress(xor_raw, level=1)
                if len(compressed) < len(tensor_raw):
                    offset = len(payload)
                    payload.extend(compressed)
                    tensor_rows.append(
                        {
                            "name": name,
                            "dtype": dtype,
                            "shape": shape,
                            "storage": "world_field_xor_zstd_v1",
                            "alias_of": name,
                            "offset": offset,
                            "bytes": len(compressed),
                            "logical_bytes": logical_bytes,
                            "sha256": tensor_sha256,
                            "base_sha256": field["sha256"],
                            "compression": FIELD_COMPRESSION,
                        }
                    )
                    continue
            offset = len(payload)
            payload.extend(tensor_raw)
            tensor_rows.append(
                {
                    "name": name,
                    "dtype": dtype,
                    "shape": shape,
                    "storage": "payload",
                    "alias_of": None,
                    "offset": offset,
                    "bytes": len(tensor_raw),
                    "logical_bytes": logical_bytes,
                    "sha256": tensor_sha256,
                }
            )
    finally:
        os.close(fd)
    tensor_manifest_sha256 = _sha256(_canonical_json(tensor_rows))
    compact = bytes(payload)
    result = {
        "kind": "compact_source_projection_v1",
        "schema_version": COMPACT_SOURCE_SCHEMA_VERSION,
        "serialization": "torch_save_skeleton_plus_tensor_ranges_v1",
        "offset": 0,
        "bytes": len(compact),
        "sha256": _sha256(compact),
        "source_projection_sha256": expected_source_projection_sha256,
        "source_projection_digest_binding": "verified_indexed_source_v1",
        "source_component_input_bytes": bytes_read,
        "source_component_logical_bytes": int(header["payload_bytes"]),
        "source_compaction_world_bytes_read": world_bytes_read,
        "source_compaction_input_mode": "indexed_sparse_v1",
        "skeleton": {
            "offset": 0,
            "bytes": len(skeleton_raw),
            "sha256": _sha256(skeleton_raw),
        },
        "tensors": tensor_rows,
        "tensor_manifest_sha256": tensor_manifest_sha256,
    }
    return compact, result, bytes_read


def _compact_source_projection_payload_v1(
    source_component_path: Path,
    *,
    expected_file_sha256: str,
    expected_source_projection_sha256: str,
    field_rows: Sequence[Mapping[str, Any]],
    world_payload_fd: int,
    verify_source_projection_digest: bool = True,
    indexed_source: Mapping[str, Any] | None = None,
) -> tuple[bytes, dict[str, Any], int]:
    """Compact one sealed source projection without weakening its digest.

    The source component is a canonical pre-window projection.  Persist exact
    field aliases and zero-fill descriptors where possible, compact XOR deltas
    against matching final-world fields where profitable, and raw bytes only
    for residual tensors.  The exact source-projection digest remains the
    restore authority.
    """

    if indexed_source is not None:
        if not isinstance(indexed_source, Mapping):
            raise _fail("indexed source descriptor is malformed")
        return _compact_indexed_source_projection_payload_v1(
            indexed_source,
            expected_source_projection_sha256=expected_source_projection_sha256,
            field_rows=field_rows,
            world_payload_fd=world_payload_fd,
        )

    try:
        source_raw = source_component_path.read_bytes()
        source_value = torch.load(
            io.BytesIO(source_raw), map_location="cpu", weights_only=False
        )
    except Exception as exc:
        raise _fail("world checkpoint source component is unreadable") from exc
    if _sha256(source_raw) != expected_file_sha256 or not isinstance(
        source_value, Mapping
    ):
        raise _fail("world checkpoint source component digest differs")
    source_projection = dict(source_value)
    source_tensors = source_projection.pop("tensors", None)
    if not isinstance(source_tensors, Mapping) or any(
        not isinstance(name, str) or not isinstance(tensor, torch.Tensor)
        for name, tensor in source_tensors.items()
    ):
        raise _fail("world checkpoint source tensor authority is malformed")
    if verify_source_projection_digest:
        from engine.native_runtime.checkpoint_digests import (
            checkpoint_source_projection_authority_manifest,
        )

        observed_source_digest = checkpoint_source_projection_authority_manifest(
            source_value
        ).get("source_projection_sha256")
        if observed_source_digest != expected_source_projection_sha256:
            raise _fail("world checkpoint source projection digest differs")

    skeleton_stream = io.BytesIO()
    torch.save(source_projection, skeleton_stream)
    skeleton_raw = skeleton_stream.getvalue()
    payload = bytearray(skeleton_raw)
    field_by_name = {
        str(row.get("field")): row
        for row in field_rows
        if isinstance(row, Mapping) and isinstance(row.get("field"), str)
    }
    tensor_rows: list[dict[str, Any]] = []
    for name in sorted(source_tensors):
        tensor_value = source_tensors[name]
        tensor = tensor_value.detach().to("cpu").contiguous()
        dtype = streamed._dtype_name(tensor)
        shape = list(tensor.shape)
        logical_bytes = tensor.numel() * tensor.element_size()
        field = field_by_name.get(name)
        is_zero = tensor.numel() == 0 or bool(torch.count_nonzero(tensor).item() == 0)
        if is_zero:
            tensor_sha256 = packed.aggregate._zero_digest(logical_bytes)
            if (
                isinstance(field, Mapping)
                and field.get("dtype") == dtype
                and field.get("shape") == shape
                and field.get("logical_bytes") == logical_bytes
                and field.get("sha256") == tensor_sha256
            ):
                storage = "world_field_alias"
                alias_of: str | None = name
            else:
                storage = "zero_fill"
                alias_of = None
            tensor_rows.append(
                {
                    "name": name,
                    "dtype": dtype,
                    "shape": shape,
                    "storage": storage,
                    "alias_of": alias_of,
                    "offset": None,
                    "bytes": 0,
                    "logical_bytes": logical_bytes,
                    "sha256": tensor_sha256,
                }
            )
            continue
        tensor_raw = streamed._tensor_bytes(tensor)
        tensor_sha256 = _sha256(tensor_raw)
        if (
            isinstance(field, Mapping)
            and field.get("dtype") == dtype
            and field.get("shape") == shape
            and field.get("logical_bytes") == logical_bytes
            and field.get("sha256") == tensor_sha256
        ):
            tensor_rows.append(
                {
                    "name": name,
                    "dtype": dtype,
                    "shape": shape,
                    "storage": "world_field_alias",
                    "alias_of": name,
                    "offset": None,
                    "bytes": 0,
                    "logical_bytes": logical_bytes,
                    "sha256": tensor_sha256,
                }
            )
            continue
        if (
            isinstance(field, Mapping)
            and field.get("storage")
            in {"payload", "payload_zlib_v1", "payload_zstd_v1"}
            and field.get("dtype") == dtype
            and field.get("shape") == shape
            and field.get("logical_bytes") == logical_bytes
            and type(field.get("offset")) is int
            and type(field.get("bytes")) is int
        ):
            base_stored = _read_exact_at(
                world_payload_fd,
                offset=PREFIX.size + int(field["offset"]),
                length=int(field["bytes"]),
            )
            if field.get("storage") in {"payload_zlib_v1", "payload_zstd_v1"}:
                try:
                    if field.get("storage") == "payload_zstd_v1":
                        if field.get("compression") != FIELD_COMPRESSION:
                            raise _fail("world checkpoint field compression is invalid")
                        base_raw = zstd.decompress(base_stored)
                    else:
                        if field.get("compression") != LEGACY_FIELD_COMPRESSION:
                            raise _fail("world checkpoint field compression is invalid")
                        decompressor = zlib.decompressobj()
                        base_raw = decompressor.decompress(
                            base_stored, logical_bytes + 1
                        )
                        if (
                            not decompressor.eof
                            or decompressor.unconsumed_tail
                            or decompressor.unused_data
                        ):
                            raise _fail(
                                "world checkpoint compressed field length differs "
                                "during source compaction"
                            )
                except (ValueError, zstd.ZstdError, zlib.error) as exc:
                    raise _fail(
                        "world checkpoint compressed field is unreadable during source compaction"
                    ) from exc
                if len(base_raw) != logical_bytes:
                    raise _fail(
                        "world checkpoint compressed field length differs during source compaction"
                    )
            else:
                base_raw = base_stored
            if _sha256(base_raw) != field.get("sha256"):
                raise _fail("world checkpoint field changed during source compaction")
            xor_raw = np.bitwise_xor(
                np.frombuffer(tensor_raw, dtype=np.uint8),
                np.frombuffer(base_raw, dtype=np.uint8),
            ).tobytes()
            compressed = zstd.compress(xor_raw, level=1)
            if len(compressed) < len(tensor_raw):
                offset = len(payload)
                payload.extend(compressed)
                tensor_rows.append(
                    {
                        "name": name,
                        "dtype": dtype,
                        "shape": shape,
                        "storage": "world_field_xor_zstd_v1",
                        "alias_of": name,
                        "offset": offset,
                        "bytes": len(compressed),
                        "logical_bytes": logical_bytes,
                        "sha256": tensor_sha256,
                        "base_sha256": field["sha256"],
                        "compression": FIELD_COMPRESSION,
                    }
                )
                continue
        offset = len(payload)
        payload.extend(tensor_raw)
        tensor_rows.append(
            {
                "name": name,
                "dtype": dtype,
                "shape": shape,
                "storage": "payload",
                "alias_of": None,
                "offset": offset,
                "bytes": len(tensor_raw),
                "logical_bytes": logical_bytes,
                "sha256": tensor_sha256,
            }
        )
    tensor_manifest_sha256 = _sha256(_canonical_json(tensor_rows))
    descriptor = {
        "kind": "compact_source_projection_v1",
        "schema_version": COMPACT_SOURCE_SCHEMA_VERSION,
        "serialization": "torch_save_skeleton_plus_tensor_ranges_v1",
        "offset": 0,
        "bytes": len(payload),
        "sha256": _sha256(bytes(payload)),
        "source_projection_sha256": expected_source_projection_sha256,
        "source_projection_digest_binding": (
            "verified_from_projection_v1"
            if verify_source_projection_digest
            else "inherited_from_sealed_physical_recipe_v1"
        ),
        "source_component_input_bytes": len(source_raw),
        "source_component_logical_bytes": len(source_raw),
        "source_compaction_world_bytes_read": 0,
        "source_compaction_input_mode": "legacy_component_full_load_v1",
        "skeleton": {
            "offset": 0,
            "bytes": len(skeleton_raw),
            "sha256": _sha256(skeleton_raw),
        },
        "tensors": tensor_rows,
        "tensor_manifest_sha256": tensor_manifest_sha256,
    }
    return bytes(payload), descriptor, len(source_raw)


def _write_one_world(
    *,
    raw_path: Path,
    raw_fingerprint: tuple[int, int, int, int, int],
    source_authority_sha256: str,
    source_authority_kind: str,
    packed_header: Mapping[str, Any],
    scalar_metadata: Mapping[str, Any],
    checkpoint_set_id: str,
    objects_dir: Path,
    cell_index: int,
    cell_id: str,
    absolute_next_tick: int,
    chunk_bytes: int,
    cancel_event: Any | None,
    source_projection_digest_inherited: bool,
) -> dict[str, Any]:
    if cancel_event is not None and cancel_event.is_set():
        raise _fail("world checkpoint sharding was cancelled")
    fields = packed_header.get("fields")
    field_order = packed_header.get("field_order")
    batch_cell_ids = tuple(packed_header.get("cell_ids") or ())
    if (
        not isinstance(fields, list)
        or not isinstance(field_order, list)
        or len(fields) != len(field_order)
        or [row.get("name") for row in fields if isinstance(row, Mapping)]
        != field_order
    ):
        raise _fail("packed checkpoint field surface is malformed")
    projected_scalars = project_world_scalar_metadata_v1(
        scalar_metadata,
        cell_index=cell_index,
        cell_id=cell_id,
        batch_cell_ids=batch_cell_ids,
    )
    projected_recipe = projected_scalars["settlement_recipe"]
    projected_source_recipe = projected_recipe.get("source_descriptor_recipe")
    source_descriptors = (
        projected_source_recipe.get("descriptors")
        if isinstance(projected_source_recipe, Mapping)
        else None
    )
    source_descriptor = (
        dict(source_descriptors[0])
        if isinstance(source_descriptors, list)
        and len(source_descriptors) == 1
        and isinstance(source_descriptors[0], Mapping)
        else None
    )
    if (
        not isinstance(source_descriptor, dict)
        or source_descriptor.get("kind") != "component"
    ):
        raise _fail(
            "world checkpoint v1 requires one independently serialized source component"
        )
    source_component_locator = Path(str(source_descriptor.get("path") or ""))
    if source_component_locator.is_symlink():
        raise _fail("world checkpoint source component authority is invalid")
    source_component_path = source_component_locator.resolve()
    source_component_sha256 = source_descriptor.get("file_sha256")
    source_projection_sha256 = source_descriptor.get("source_projection_sha256")
    if (
        source_component_path.is_symlink()
        or not source_component_path.is_file()
        or not _is_sha256(source_component_sha256)
        or not _is_sha256(source_projection_sha256)
    ):
        raise _fail("world checkpoint source component authority is invalid")
    temporary = objects_dir / f".{cell_index:03d}-{uuid4().hex}.tmp"
    input_fd = os.open(raw_path, os.O_RDONLY)
    started = time.perf_counter()
    read_bytes = 0
    payload_bytes = 0
    logical_bytes = 0
    zero_fill_logical_bytes = 0
    payload_digest = hashlib.sha256()
    field_rows: list[dict[str, Any]] = []
    terminal_alive_count: int | None = None
    range_started = time.perf_counter()
    file_fsync_seconds = 0.0
    try:
        stat = os.fstat(input_fd)
        observed = (
            int(stat.st_dev),
            int(stat.st_ino),
            int(stat.st_size),
            int(stat.st_mtime_ns),
            int(stat.st_ctime_ns),
        )
        if observed != raw_fingerprint:
            raise _fail("packed checkpoint changed before world sharding")
        with temporary.open("x+b") as stream:
            stream.write(
                PREFIX.pack(MAGIC, FORMAT_VERSION, LITTLE_ENDIAN_MARKER, 0, 0, 0)
            )
            for field_index, field in enumerate(fields):
                if cancel_event is not None and cancel_event.is_set():
                    raise _fail("world checkpoint sharding was cancelled")
                if not isinstance(field, Mapping):
                    raise _fail("packed checkpoint field descriptor is malformed")
                source_offset, byte_count, shape, dtype = _selected_source_range(
                    field, cell_index=cell_index
                )
                logical_bytes += byte_count
                if source_offset is None:
                    digest = packed.aggregate._zero_digest(byte_count)
                    zero_fill_logical_bytes += byte_count
                    if str(field["name"]) == "alive_mask":
                        if dtype != "bool" or len(shape) != 1:
                            raise _fail(
                                "world checkpoint terminal alive mask is malformed"
                            )
                        terminal_alive_count = 0
                    field_rows.append(
                        {
                            "field": str(field["name"]),
                            "field_index": field_index,
                            "dtype": dtype,
                            "shape": shape,
                            "storage": "zero_fill",
                            "offset": None,
                            "bytes": 0,
                            "logical_bytes": byte_count,
                            "sha256": digest,
                            "zero_fill_contract": packed.ZERO_FILL_CONTRACT,
                        }
                    )
                    continue
                field_offset = payload_bytes
                remaining = byte_count
                source_cursor = packed.PREFIX.size + source_offset
                field_raw = bytearray()
                while remaining:
                    raw = os.pread(input_fd, min(chunk_bytes, remaining), source_cursor)
                    if not raw:
                        raise _fail(
                            "packed checkpoint field is truncated during sharding"
                        )
                    field_raw.extend(raw)
                    source_cursor += len(raw)
                    remaining -= len(raw)
                    read_bytes += len(raw)
                raw_bytes = bytes(field_raw)
                digest = _sha256(raw_bytes)
                if str(field["name"]) == "alive_mask":
                    if dtype != "bool" or len(shape) != 1:
                        raise _fail("world checkpoint terminal alive mask is malformed")
                    terminal_alive_count = int(
                        np.count_nonzero(np.frombuffer(raw_bytes, dtype=np.bool_))
                    )
                compressed = zstd.compress(raw_bytes, level=1)
                if len(compressed) < len(raw_bytes):
                    stored = compressed
                    storage = "payload_zstd_v1"
                    compression = FIELD_COMPRESSION
                else:
                    stored = raw_bytes
                    storage = "payload"
                    compression = None
                stream.write(stored)
                payload_digest.update(stored)
                payload_bytes += len(stored)
                field_rows.append(
                    {
                        "field": str(field["name"]),
                        "field_index": field_index,
                        "dtype": dtype,
                        "shape": shape,
                        "storage": storage,
                        "offset": field_offset,
                        "bytes": len(stored),
                        "logical_bytes": byte_count,
                        "sha256": digest,
                        "compression": compression,
                        "zero_fill_contract": None,
                    }
                )

            source_dependency_offset = payload_bytes
            stream.flush()
            (
                compact_source_raw,
                source_dependency,
                source_component_input_bytes,
            ) = _compact_source_projection_payload_v1(
                source_component_path,
                expected_file_sha256=source_component_sha256,
                expected_source_projection_sha256=source_projection_sha256,
                field_rows=field_rows,
                world_payload_fd=stream.fileno(),
                verify_source_projection_digest=(
                    not source_projection_digest_inherited
                ),
                indexed_source=source_descriptor.get("indexed_source"),
            )
            source_dependency = {
                **source_dependency,
                "offset": source_dependency_offset,
            }
            stream.write(compact_source_raw)
            payload_digest.update(compact_source_raw)
            source_dependency_bytes = len(compact_source_raw)
            payload_bytes += source_dependency_bytes
            logical_bytes += source_dependency_bytes
            read_bytes += source_component_input_bytes
            projected_source_recipe = dict(projected_source_recipe)
            projected_source_recipe["descriptors"] = [
                {
                    "kind": "embedded_compact_source_projection_v1",
                    "source_projection_sha256": source_projection_sha256,
                    "file_sha256": source_dependency["sha256"],
                    "bytes": source_dependency_bytes,
                }
            ]
            projected_recipe = dict(projected_recipe)
            projected_recipe["source_descriptor_recipe"] = projected_source_recipe
            projected_recipe.pop("recipe_sha256", None)
            projected_recipe["recipe_sha256"] = _sha256(
                _canonical_json(projected_recipe)
            )
            projected_scalars["settlement_recipe"] = projected_recipe
            projected_scalars["embedded_source_dependency"] = source_dependency
            scalar_sha256 = _sha256(_canonical_json(projected_scalars))

            terminal_metric_row = None
            if terminal_alive_count is not None:
                checkpoint_inputs = projected_recipe.get("checkpoint_projection_inputs")
                genome_sidecar = (
                    checkpoint_inputs.get("genome_upgrade_v2_checkpoint_projection")
                    if isinstance(checkpoint_inputs, Mapping)
                    else None
                )
                genome_v2 = None
                if genome_sidecar is not None:
                    from engine.evolution.genome_upgrade_v2_telemetry import (
                        project_genome_upgrade_v2_e1_metric_attribution_v1,
                    )

                    genome_v2 = project_genome_upgrade_v2_e1_metric_attribution_v1(
                        {"genome_upgrade_v2_checkpoint_projection": (genome_sidecar)},
                        expected_cell_id=cell_id,
                    )
                terminal_metric_row = {
                    "schema_version": TERMINAL_METRIC_SCHEMA_VERSION,
                    "cell_id": cell_id,
                    "metrics": {
                        "population.alive": terminal_alive_count,
                        "population.terminal_alive": terminal_alive_count,
                        "runtime.terminal_absolute_tick": absolute_next_tick,
                    },
                    "genome_v2_attribution": genome_v2,
                    "raw_tick_arrays_retained": False,
                    "reporting_influence": "forbidden",
                }

            field_manifest_sha256 = _sha256(_canonical_json(field_rows))
            world_state_sha256 = _sha256(
                _canonical_json(
                    {
                        "schema_version": WORLD_SCHEMA_VERSION,
                        "cell_id": cell_id,
                        "physical_cell_index": cell_index,
                        "absolute_next_tick": absolute_next_tick,
                        "scalar_metadata_sha256": scalar_sha256,
                        "field_manifest_sha256": field_manifest_sha256,
                        "fields": field_rows,
                        "source_dependency": source_dependency,
                    }
                )
            )
            route = projected_scalars["settlement_recipe"].get("route")
            if not isinstance(route, Mapping):
                raise _fail("world checkpoint route authority is missing")
            world_mechanics_sha256 = _sha256(
                _canonical_json(
                    {
                        "schema_version": WORLD_SCHEMA_VERSION,
                        "cell_id": cell_id,
                        "physical_cell_index": cell_index,
                        "absolute_next_tick": absolute_next_tick,
                        "route": dict(route),
                        "world_state_sha256": world_state_sha256,
                    }
                )
            )
            header = {
                "schema_version": WORLD_SCHEMA_VERSION,
                "format_version": FORMAT_VERSION,
                "endianness": "little",
                "payload_layout": "world_field_order_v1",
                "checkpoint_set_id": checkpoint_set_id,
                "batch_id": str(packed_header["batch_id"]),
                "cell_id": cell_id,
                "physical_cell_index": cell_index,
                "source_batch_cell_count": len(batch_cell_ids),
                "absolute_next_tick": absolute_next_tick,
                "source_physical_authority_sha256": source_authority_sha256,
                "source_physical_authority_kind": source_authority_kind,
                "scalar_metadata": projected_scalars,
                "scalar_metadata_sha256": scalar_sha256,
                "field_order": list(field_order),
                "fields": field_rows,
                "field_manifest_sha256": field_manifest_sha256,
                "payload_bytes": payload_bytes,
                "logical_bytes": logical_bytes,
                "zero_fill_logical_bytes": zero_fill_logical_bytes,
                "source_dependency": source_dependency,
                "world_state_sha256": world_state_sha256,
                "world_mechanics_sha256": world_mechanics_sha256,
            }
            header_bytes = _canonical_json(header)
            if not 0 < len(header_bytes) <= MAX_WORLD_HEADER_BYTES:
                raise _fail("world checkpoint header exceeds its bound")
            payload_sha256 = payload_digest.hexdigest()
            content_sha256 = _content_digest(
                header_sha256=_sha256(header_bytes),
                payload_sha256=payload_sha256,
                world_state_sha256=world_state_sha256,
            )
            provisional_footer = {
                "schema_version": WORLD_FOOTER_SCHEMA_VERSION,
                "status": "complete",
                "checkpoint_set_id": checkpoint_set_id,
                "cell_id": cell_id,
                "physical_cell_index": cell_index,
                "absolute_next_tick": absolute_next_tick,
                "payload_bytes": payload_bytes,
                "header_bytes": len(header_bytes),
                "payload_sha256": payload_sha256,
                "header_sha256": _sha256(header_bytes),
                "world_state_sha256": world_state_sha256,
                "world_mechanics_sha256": world_mechanics_sha256,
                "content_sha256": content_sha256,
                "prefix_sha256": "0" * 64,
                "artifact_sha256": "0" * 64,
            }
            footer_length = len(_canonical_json(provisional_footer))
            prefix = PREFIX.pack(
                MAGIC,
                FORMAT_VERSION,
                LITTLE_ENDIAN_MARKER,
                len(header_bytes),
                payload_bytes,
                footer_length,
            )
            footer_without_artifact = {
                **provisional_footer,
                "prefix_sha256": _sha256(prefix),
            }
            footer_without_artifact.pop("artifact_sha256")
            artifact_sha256 = _artifact_digest(
                prefix=prefix,
                header=header_bytes,
                footer_without_artifact=footer_without_artifact,
            )
            footer = {
                **footer_without_artifact,
                "artifact_sha256": artifact_sha256,
            }
            footer_bytes = _canonical_json(footer)
            if (
                len(footer_bytes) != footer_length
                or len(footer_bytes) > MAX_WORLD_FOOTER_BYTES
            ):
                raise _fail("world checkpoint footer length did not converge")
            stream.write(header_bytes)
            stream.write(footer_bytes)
            stream.seek(0)
            stream.write(prefix)
            stream.flush()
        publish_started = time.perf_counter()
        final_name = f"world-{cell_index:03d}-{artifact_sha256}.acsworld"
        final_path = objects_dir / final_name
        os.replace(temporary, final_path)
        return {
            "cell_id": cell_id,
            "physical_cell_index": cell_index,
            "relative_path": f"objects/{final_name}",
            "artifact_sha256": artifact_sha256,
            "content_sha256": content_sha256,
            "payload_sha256": payload_sha256,
            "world_state_sha256": world_state_sha256,
            "world_mechanics_sha256": world_mechanics_sha256,
            "source_projection_sha256": source_projection_sha256,
            "field_manifest_sha256": field_manifest_sha256,
            "scalar_metadata_sha256": scalar_sha256,
            "payload_bytes": payload_bytes,
            "logical_bytes": logical_bytes,
            "zero_fill_logical_bytes": zero_fill_logical_bytes,
            "source_dependency_bytes": source_dependency_bytes,
            "source_component_input_bytes": source_component_input_bytes,
            "source_component_logical_bytes": int(
                source_dependency.get(
                    "source_component_logical_bytes",
                    source_component_input_bytes,
                )
            ),
            "source_compaction_world_bytes_read": int(
                source_dependency.get("source_compaction_world_bytes_read", 0)
            ),
            "source_compaction_input_mode": str(
                source_dependency.get(
                    "source_compaction_input_mode",
                    "legacy_component_full_load_v1",
                )
            ),
            "physical_file_bytes": final_path.stat().st_size,
            "source_bytes_read": read_bytes,
            "terminal_metric_row": terminal_metric_row,
            "canonical_range_hash_write_seconds": (
                publish_started - range_started - file_fsync_seconds
            ),
            "file_fsync_seconds": file_fsync_seconds,
            "object_publish_seconds": time.perf_counter() - publish_started,
            "write_seconds": time.perf_counter() - started,
        }
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    finally:
        os.close(input_fd)


def _verify_packed_source(
    raw_path: Path,
    *,
    raw_receipt: Mapping[str, Any] | None,
    raw_capture_descriptor: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], tuple[int, int, int, int, int], str]:
    if raw_receipt is not None and raw_capture_descriptor is not None:
        raise _fail("packed source has competing sealed and transient authorities")
    if raw_capture_descriptor is not None:
        packed._assert_descriptor_scalars(raw_capture_descriptor)
        capture_path = (
            Path(str(raw_capture_descriptor.get("capture_path") or ""))
            .expanduser()
            .resolve()
        )
        authority_sha256 = packed.capture_descriptor_authority_sha256_v1(
            raw_capture_descriptor
        )
        if (
            capture_path != raw_path
            or raw_capture_descriptor.get("capture_authority_sha256")
            != authority_sha256
        ):
            raise _fail("packed transient capture authority differs")
        fingerprint_fields = (
            "device",
            "inode",
            "mode",
            "nlink",
            "size",
            "mtime_ns",
            "ctime_ns",
        )
        expected_fingerprint = raw_capture_descriptor.get("captured_file_fingerprint")
        try:
            observed_stat = raw_path.stat()
        except OSError as exc:
            raise _fail(
                f"checkpoint source identity is unavailable: {raw_path}"
            ) from exc
        observed_fingerprint = packed.PackedCheckpointFileFingerprintV1.from_stat(
            observed_stat
        )
        observed_payload = {
            field: int(getattr(observed_fingerprint, field))
            for field in fingerprint_fields
        }
        if (
            not isinstance(expected_fingerprint, Mapping)
            or dict(expected_fingerprint) != observed_payload
        ):
            raise _fail("packed checkpoint differs from its capture fingerprint")
        prefix, header, footer_raw, payload_bytes, header_bytes = packed._read_parts(
            raw_path, require_sealed=False
        )
        if footer_raw is not None:
            raise _fail("packed transient capture unexpectedly contains a footer")
        with raw_path.open("rb") as stream:
            stream.seek(packed.PREFIX.size + payload_bytes)
            header_raw = stream.read(header_bytes)
        cell_ids, fields = packed._validate_container_identity(
            header, payload_bytes=payload_bytes
        )
        packed._validate_zero_fill_summary(
            header,
            fields,
            batch_cells=len(cell_ids),
            payload_bytes=payload_bytes,
        )
        if (
            len(header_raw) != header_bytes
            or _sha256(header_raw) != raw_capture_descriptor.get("header_sha256")
            or raw_capture_descriptor.get("header_bytes") != header_bytes
            or raw_capture_descriptor.get("payload_bytes") != payload_bytes
            or raw_capture_descriptor.get("batch_id") != header.get("batch_id")
            or raw_capture_descriptor.get("cell_ids") != cell_ids
            or raw_capture_descriptor.get("field_order") != header.get("field_order")
            or raw_capture_descriptor.get("sealed") is not False
        ):
            raise _fail("packed transient descriptor does not bind its source")
        fingerprint = _file_fingerprint(raw_path)
        return (
            copy.deepcopy(dict(header)),
            {
                "capture_authority_sha256": authority_sha256,
                "source_authority_kind": (
                    "captured_unsealed_fingerprint_and_complete_range_copy_v1"
                ),
            },
            fingerprint,
            "captured_unsealed_descriptor_fingerprint_v1",
        )
    receipt_fingerprint = (
        raw_receipt.get("published_file_fingerprint")
        if isinstance(raw_receipt, Mapping)
        else None
    )
    if isinstance(receipt_fingerprint, Mapping):
        fingerprint_fields = (
            "device",
            "inode",
            "mode",
            "nlink",
            "size",
            "mtime_ns",
            "ctime_ns",
        )
        try:
            current_stat = raw_path.stat()
        except OSError as exc:
            raise _fail(
                f"checkpoint source identity is unavailable: {raw_path}"
            ) from exc
        current_fingerprint = packed.PackedCheckpointFileFingerprintV1.from_stat(
            current_stat
        )
        current_payload = {
            field: int(getattr(current_fingerprint, field))
            for field in fingerprint_fields
        }
        if (
            set(receipt_fingerprint) != set(fingerprint_fields)
            or any(
                type(receipt_fingerprint.get(field)) is not int
                for field in fingerprint_fields
            )
            or dict(receipt_fingerprint) != current_payload
        ):
            raise _fail(
                "packed checkpoint differs from its sealed publication fingerprint"
            )
        prefix, header, footer_raw, payload_bytes, header_bytes = packed._read_parts(
            raw_path, require_sealed=True
        )
        if footer_raw is None:
            raise _fail("packed checkpoint has no sealed footer")
        try:
            footer = json.loads(footer_raw.decode("ascii"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise _fail("packed checkpoint footer is malformed") from exc
        header_raw = _canonical_json(header)
        if (
            not isinstance(footer, dict)
            or footer_raw != _canonical_json(footer)
            or header.get("payload_bytes") != payload_bytes
            or footer.get("header_bytes") != header_bytes
            or footer.get("header_sha256") != _sha256(header_raw)
            or footer.get("prefix_sha256") != _sha256(prefix)
            or footer.get("payload_sha256") != raw_receipt.get("payload_sha256")
            or footer.get("artifact_sha256") != raw_receipt.get("artifact_sha256")
            or footer.get("seal_sha256") != packed._footer_seal_digest(footer)
            or footer.get("artifact_sha256")
            != packed._footer_artifact_digest(
                prefix=prefix,
                header_sha256=_sha256(header_raw),
                payload_sha256=str(footer.get("payload_sha256") or ""),
                footer=footer,
            )
            or raw_receipt.get("sealed") is not True
            or raw_receipt.get("published_path") != str(raw_path)
            or raw_receipt.get("batch_id") != header.get("batch_id")
            or raw_receipt.get("cell_ids") != header.get("cell_ids")
            or raw_receipt.get("payload_bytes") != payload_bytes
            or raw_receipt.get("header_bytes") != header_bytes
            or raw_receipt.get("footer_bytes") != len(footer_raw)
        ):
            raise _fail(
                "packed checkpoint finalization receipt does not bind its source"
            )
        cell_ids, fields = packed._validate_container_identity(
            header, payload_bytes=payload_bytes
        )
        packed._validate_zero_fill_summary(
            header,
            fields,
            batch_cells=len(cell_ids),
            payload_bytes=payload_bytes,
        )
        fingerprint = _file_fingerprint(raw_path)
        return (
            copy.deepcopy(dict(header)),
            copy.deepcopy(dict(footer)),
            fingerprint,
            "sealed_receipt_fingerprint_v1",
        )
    with packed.open_physical_batch_packed_checkpoint_v1(raw_path) as reader:
        header = copy.deepcopy(dict(reader.header))
        footer = copy.deepcopy(dict(reader.footer))
        fingerprint_value = reader.fingerprint
        fingerprint = (
            fingerprint_value.device,
            fingerprint_value.inode,
            fingerprint_value.size,
            fingerprint_value.mtime_ns,
            fingerprint_value.ctime_ns,
        )
    if raw_receipt is not None:
        if (
            raw_receipt.get("sealed") is not True
            or raw_receipt.get("published_path") != str(raw_path)
            or raw_receipt.get("artifact_sha256") != footer.get("artifact_sha256")
            or raw_receipt.get("payload_sha256") != footer.get("payload_sha256")
            or raw_receipt.get("payload_bytes") != header.get("payload_bytes")
        ):
            raise _fail(
                "packed checkpoint finalization receipt does not bind its source"
            )
    return header, footer, fingerprint, "full_payload_reverification_v1"


def write_world_sharded_checkpoint_set_from_packed_v1(
    raw_checkpoint: str | os.PathLike[str],
    destination_root: str | os.PathLike[str],
    *,
    raw_receipt: Mapping[str, Any] | None = None,
    raw_capture_descriptor: Mapping[str, Any] | None = None,
    worker_count: int = 4,
    copy_chunk_bytes: int = DEFAULT_COPY_CHUNK_BYTES,
    required_world_count: int | None = None,
    cancel_event: threading.Event | None = None,
    performance_telemetry: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Directly publish one complete immutable world-sharded checkpoint set."""

    if type(worker_count) is not int or not 1 <= worker_count <= MAX_WORKERS:
        raise _fail(f"worker_count must be in the range 1..{MAX_WORKERS}")
    if (
        type(copy_chunk_bytes) is not int
        or not 1 <= copy_chunk_bytes <= MAX_COPY_CHUNK_BYTES
    ):
        raise _fail(f"copy_chunk_bytes must be in the range 1..{MAX_COPY_CHUNK_BYTES}")
    raw_path = Path(raw_checkpoint).expanduser().resolve()
    destination = Path(destination_root).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    verify_started = time.perf_counter()
    header, footer, fingerprint, source_verification_mode = _verify_packed_source(
        raw_path,
        raw_receipt=raw_receipt,
        raw_capture_descriptor=raw_capture_descriptor,
    )
    verify_seconds = time.perf_counter() - verify_started
    cell_ids = tuple(header.get("cell_ids") or ())
    if (
        not cell_ids
        or len(set(cell_ids)) != len(cell_ids)
        or any(not isinstance(cell_id, str) or not cell_id for cell_id in cell_ids)
    ):
        raise _fail("packed checkpoint cell identity is invalid")
    if required_world_count is not None and len(cell_ids) != required_world_count:
        raise _fail(f"packed checkpoint requires exactly {required_world_count} worlds")
    scalar_metadata = header.get("scalar_metadata")
    recipe = (
        scalar_metadata.get("settlement_recipe")
        if isinstance(scalar_metadata, Mapping)
        else None
    )
    route = recipe.get("route") if isinstance(recipe, Mapping) else None
    absolute_next_tick = (
        recipe.get("absolute_next_tick") if isinstance(recipe, Mapping) else None
    )
    recipe_sha256 = recipe.get("recipe_sha256") if isinstance(recipe, Mapping) else None
    source_authority_sha256 = footer.get(
        "artifact_sha256", footer.get("capture_authority_sha256")
    )
    source_authority_kind = str(
        footer.get("source_authority_kind") or "sealed_packed_artifact_sha256_v1"
    )
    if (
        not isinstance(scalar_metadata, Mapping)
        or not isinstance(recipe, Mapping)
        or not isinstance(route, Mapping)
        or type(absolute_next_tick) is not int
        or absolute_next_tick < 0
        or not _is_sha256(recipe_sha256)
        or not _is_sha256(source_authority_sha256)
    ):
        raise _fail("packed checkpoint lacks absolute-tick settlement authority")
    set_id = _checkpoint_set_id(
        source_authority_sha256=source_authority_sha256,
        batch_id=str(header["batch_id"]),
        cell_ids=cell_ids,
        absolute_next_tick=absolute_next_tick,
        recipe_sha256=recipe_sha256,
        route=route,
    )
    final_root = destination / set_id
    if final_root.exists():
        raise _fail("checkpoint set publication target already exists")
    staging = destination / f".{set_id}.{os.getpid()}.{uuid4().hex}.staging"
    objects_dir = staging / "objects"
    objects_dir.mkdir(parents=True)
    receipts: list[dict[str, Any]] = []
    failure_event = threading.Event()

    class _CombinedCancel:
        def is_set(self) -> bool:
            return failure_event.is_set() or (
                cancel_event is not None and cancel_event.is_set()
            )

    combined_cancel = _CombinedCancel()
    try:
        bounded_workers = min(worker_count, len(cell_ids))
        with ThreadPoolExecutor(
            max_workers=bounded_workers,
            thread_name_prefix="acs-world-shard",
        ) as executor:
            pending_indices = iter(enumerate(cell_ids))
            futures: dict[Any, int] = {}

            def submit_one(index: int, cell_id: str) -> None:
                future = executor.submit(
                    _write_one_world,
                    raw_path=raw_path,
                    raw_fingerprint=fingerprint,
                    source_authority_sha256=source_authority_sha256,
                    source_authority_kind=source_authority_kind,
                    packed_header=header,
                    scalar_metadata=scalar_metadata,
                    checkpoint_set_id=set_id,
                    objects_dir=objects_dir,
                    cell_index=index,
                    cell_id=cell_id,
                    absolute_next_tick=absolute_next_tick,
                    chunk_bytes=copy_chunk_bytes,
                    cancel_event=combined_cancel,
                    # A verified packed-finalization receipt seals the raw
                    # header, including the recipe that pairs each component
                    # file hash with its source-projection digest.  Avoid a
                    # second full projection digest pass on this durability
                    # path; every restore still recomputes and checks it.
                    source_projection_digest_inherited=(raw_receipt is not None),
                )
                futures[future] = index

            for _ in range(bounded_workers):
                index, cell_id = next(pending_indices)
                submit_one(index, cell_id)
            try:
                while futures:
                    done, _ = wait(futures, return_when=FIRST_COMPLETED)
                    for future in sorted(done, key=lambda item: futures[item]):
                        futures.pop(future)
                        receipts.append(future.result())
                        next_row = next(pending_indices, None)
                        if next_row is not None:
                            submit_one(*next_row)
            except BaseException:
                failure_event.set()
                for future in futures:
                    future.cancel()
                raise
        receipts.sort(key=lambda row: int(row["physical_cell_index"]))
        if [row["cell_id"] for row in receipts] != list(cell_ids) or [
            row["physical_cell_index"] for row in receipts
        ] != list(range(len(cell_ids))):
            raise _fail("world checkpoint receipts are incomplete or unordered")
        if _file_fingerprint(raw_path) != fingerprint:
            raise _fail("packed checkpoint changed during world sharding")
        deferred_file_fsync_seconds = _fsync_world_objects(objects_dir, receipts)
        objects_fsync_started = time.perf_counter()
        _fsync_directory(objects_dir)
        objects_directory_fsync_seconds = time.perf_counter() - objects_fsync_started
        if _file_fingerprint(raw_path) != fingerprint:
            raise _fail("packed checkpoint changed during world sharding")
        manifest_build_started = time.perf_counter()
        batch_digest_sha256 = _sha256(
            _canonical_json(
                [
                    {
                        "cell_id": row["cell_id"],
                        "physical_cell_index": row["physical_cell_index"],
                        "artifact_sha256": row["artifact_sha256"],
                        "world_state_sha256": row["world_state_sha256"],
                        "world_mechanics_sha256": row["world_mechanics_sha256"],
                    }
                    for row in receipts
                ]
            )
        )
        durable_keys = (
            "cell_id",
            "physical_cell_index",
            "relative_path",
            "artifact_sha256",
            "content_sha256",
            "payload_sha256",
            "world_state_sha256",
            "world_mechanics_sha256",
            "source_projection_sha256",
            "field_manifest_sha256",
            "scalar_metadata_sha256",
            "payload_bytes",
            "logical_bytes",
            "zero_fill_logical_bytes",
            "source_dependency_bytes",
            "physical_file_bytes",
        )
        durable_objects = [{key: row[key] for key in durable_keys} for row in receipts]
        manifest_without_digest = {
            "schema_version": SET_MANIFEST_SCHEMA_VERSION,
            "status": "complete",
            "checkpoint_set_id": set_id,
            "batch_id": str(header["batch_id"]),
            "absolute_next_tick": absolute_next_tick,
            "source_physical_authority_sha256": source_authority_sha256,
            "source_physical_authority_kind": source_authority_kind,
            "source_raw_artifact_sha256": (
                source_authority_sha256
                if source_authority_kind == "sealed_packed_artifact_sha256_v1"
                else None
            ),
            "source_recipe_sha256": recipe_sha256,
            "world_count": len(cell_ids),
            "cell_ids": list(cell_ids),
            "objects": durable_objects,
            "batch_digest_sha256": batch_digest_sha256,
            "payload_bytes": sum(int(row["payload_bytes"]) for row in receipts),
            "logical_bytes": sum(int(row["logical_bytes"]) for row in receipts),
            "physical_file_bytes": sum(
                int(row["physical_file_bytes"]) for row in receipts
            ),
            "cuda_references": 0,
            "cuda_callbacks": 0,
            "publication": "objects_fsync_then_atomic_set_manifest_v1",
        }
        manifest = {
            **manifest_without_digest,
            "manifest_sha256": _sha256(_canonical_json(manifest_without_digest)),
        }
        manifest_build_seconds = time.perf_counter() - manifest_build_started
        manifest_path = staging / "checkpoint_set_manifest.json"
        manifest_tmp = staging / ".checkpoint_set_manifest.tmp"
        manifest_commit_started = time.perf_counter()
        with manifest_tmp.open("xb") as stream:
            stream.write(_canonical_json(manifest))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(manifest_tmp, manifest_path)
        _fsync_directory(staging)
        manifest_commit_seconds = time.perf_counter() - manifest_commit_started
        publication_started = time.perf_counter()
        os.replace(staging, final_root)
        _fsync_directory(destination)
        set_publication_seconds = time.perf_counter() - publication_started
    except BaseException as exc:
        if staging.exists():
            quarantine_root = destination / "quarantine"
            quarantine_root.mkdir(exist_ok=True)
            (staging / "checkpoint_set_manifest.json").unlink(missing_ok=True)
            (staging / ".checkpoint_set_manifest.tmp").unlink(missing_ok=True)
            failure = {
                "schema_version": "eiel_acs_world_checkpoint_set_failure_v1",
                "status": "incomplete_quarantined",
                "checkpoint_set_id": set_id,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            failure_path = staging / "failure.json"
            with failure_path.open("xb") as stream:
                stream.write(_canonical_json(failure))
                stream.flush()
                os.fsync(stream.fileno())
            _fsync_directory(staging)
            quarantine_path = quarantine_root / f"{set_id}-{uuid4().hex}"
            os.replace(staging, quarantine_path)
            _fsync_directory(quarantine_root)
        raise

    elapsed = time.perf_counter() - started
    manifest_file_bytes = (final_root / "checkpoint_set_manifest.json").stat().st_size
    result = {
        "schema_version": SET_MANIFEST_SCHEMA_VERSION,
        "status": "complete",
        "checkpoint_set_id": set_id,
        "set_root": str(final_root),
        "manifest_path": str(final_root / "checkpoint_set_manifest.json"),
        "manifest_sha256": manifest["manifest_sha256"],
        "batch_digest_sha256": manifest["batch_digest_sha256"],
        "source_physical_authority_sha256": manifest[
            "source_physical_authority_sha256"
        ],
        "source_physical_authority_kind": manifest["source_physical_authority_kind"],
        "source_raw_artifact_sha256": manifest.get("source_raw_artifact_sha256"),
        "source_recipe_sha256": manifest["source_recipe_sha256"],
        "world_count": len(cell_ids),
        "payload_bytes": manifest["payload_bytes"],
        "physical_file_bytes": manifest["physical_file_bytes"],
        "source_bytes_read": sum(int(row["source_bytes_read"]) for row in receipts),
        "worker_count": min(worker_count, len(cell_ids)),
        "bounded_in_flight_world_jobs": min(worker_count, len(cell_ids)),
        "source_verification_seconds": verify_seconds,
        "source_verification_mode": source_verification_mode,
        "canonical_range_hash_write_seconds": sum(
            float(row["canonical_range_hash_write_seconds"]) for row in receipts
        ),
        "file_fsync_seconds": sum(float(row["file_fsync_seconds"]) for row in receipts),
        "deferred_file_fsync_seconds": deferred_file_fsync_seconds,
        "object_publish_seconds": sum(
            float(row["object_publish_seconds"]) for row in receipts
        ),
        "worker_service_seconds_sum": sum(
            float(row["write_seconds"]) for row in receipts
        ),
        "worker_service_seconds_max": max(
            float(row["write_seconds"]) for row in receipts
        ),
        "objects_directory_fsync_seconds": objects_directory_fsync_seconds,
        "all_world_objects_durable_seconds": (
            objects_fsync_started + objects_directory_fsync_seconds - started
        ),
        "manifest_build_seconds": manifest_build_seconds,
        "manifest_commit_seconds": manifest_commit_seconds,
        "set_publication_seconds": set_publication_seconds,
        "set_durability_seconds": elapsed,
        "physical_compact_result_rows": [
            dict(row["terminal_metric_row"])
            for row in receipts
            if isinstance(row.get("terminal_metric_row"), Mapping)
        ],
        "manifest_file_bytes": manifest_file_bytes,
        "total_bytes_read": sum(int(row["source_bytes_read"]) for row in receipts),
        "total_bytes_written": int(manifest["physical_file_bytes"])
        + manifest_file_bytes,
        "temporary_peak_bytes": int(manifest["physical_file_bytes"])
        + manifest_file_bytes,
        "queued_cuda_references": 0,
        "queued_cuda_callbacks": 0,
        "objects": receipts,
    }
    if performance_telemetry is not None:
        performance_telemetry.clear()
        performance_telemetry.update(result)
    return result


def _read_world_parts(
    path: Path,
) -> tuple[bytes, bytes, dict[str, Any], bytes, int]:
    try:
        with path.open("rb") as stream:
            prefix = stream.read(PREFIX.size)
            if len(prefix) != PREFIX.size:
                raise _fail("world checkpoint prefix is truncated")
            magic, version, endian, header_len, payload_len, footer_len = PREFIX.unpack(
                prefix
            )
            if (
                magic != MAGIC
                or version != FORMAT_VERSION
                or endian != LITTLE_ENDIAN_MARKER
                or not 0 < header_len <= MAX_WORLD_HEADER_BYTES
                or payload_len < 0
                or not 0 < footer_len <= MAX_WORLD_FOOTER_BYTES
            ):
                raise _fail("world checkpoint prefix identity is invalid")
            expected = PREFIX.size + payload_len + header_len + footer_len
            if path.stat().st_size != expected:
                raise _fail("world checkpoint size differs from its prefix")
            stream.seek(PREFIX.size + payload_len)
            header_raw = stream.read(header_len)
            footer_raw = stream.read(footer_len)
        header = json.loads(header_raw.decode("ascii"))
        footer = json.loads(footer_raw.decode("ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, struct.error) as exc:
        if isinstance(exc, ACSWorldShardedCheckpointError):
            raise
        raise _fail(f"world checkpoint cannot be read: {path}") from exc
    if not isinstance(header, dict) or not isinstance(footer, dict):
        raise _fail("world checkpoint metadata is malformed")
    return prefix, header_raw, header, footer_raw, payload_len


def load_world_checkpoint_object_v1(
    path: str | os.PathLike[str],
    *,
    expected_artifact_sha256: str | None = None,
    device: str | torch.device = "cpu",
    byte_counter: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Load and fully verify exactly one independently addressable world."""

    target = Path(path).expanduser().resolve()
    prefix, stored_header_raw, header, footer_raw, payload_len = _read_world_parts(
        target
    )
    try:
        footer = json.loads(footer_raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise _fail("world checkpoint footer is malformed") from exc
    if (
        header.get("schema_version") != WORLD_SCHEMA_VERSION
        or header.get("format_version") != FORMAT_VERSION
        or header.get("payload_layout") != "world_field_order_v1"
        or footer.get("schema_version") != WORLD_FOOTER_SCHEMA_VERSION
        or footer.get("status") != "complete"
        or footer.get("checkpoint_set_id") != header.get("checkpoint_set_id")
        or footer.get("cell_id") != header.get("cell_id")
        or footer.get("physical_cell_index") != header.get("physical_cell_index")
        or footer.get("absolute_next_tick") != header.get("absolute_next_tick")
        or footer.get("payload_bytes") != payload_len
        or header.get("payload_bytes") != payload_len
    ):
        raise _fail("world checkpoint header/footer authority differs")
    header_raw = _canonical_json(header)
    if stored_header_raw != header_raw or footer_raw != _canonical_json(footer):
        raise _fail("world checkpoint metadata encoding is not canonical")
    if footer.get("header_sha256") != _sha256(header_raw):
        raise _fail("world checkpoint header digest differs")
    if footer.get("prefix_sha256") != _sha256(prefix):
        raise _fail("world checkpoint prefix digest differs")
    artifact = footer.get("artifact_sha256")
    footer_without_artifact = dict(footer)
    footer_without_artifact.pop("artifact_sha256", None)
    if not _is_sha256(artifact) or artifact != _artifact_digest(
        prefix=prefix,
        header=header_raw,
        footer_without_artifact=footer_without_artifact,
    ):
        raise _fail("world checkpoint artifact digest differs")
    if expected_artifact_sha256 is not None and artifact != expected_artifact_sha256:
        raise _fail("world checkpoint artifact identity differs from the set manifest")
    if target.name != (
        f"world-{int(header['physical_cell_index']):03d}-{artifact}.acsworld"
    ):
        raise _fail("world checkpoint filename is not its content address")
    fields = header.get("fields")
    field_order = header.get("field_order")
    if (
        not isinstance(fields, list)
        or not isinstance(field_order, list)
        or len(fields) != len(field_order)
        or [row.get("field") for row in fields if isinstance(row, Mapping)]
        != field_order
        or header.get("field_manifest_sha256") != _sha256(_canonical_json(fields))
    ):
        raise _fail("world checkpoint field manifest is invalid")
    payload_digest = hashlib.sha256()
    tensors: dict[str, torch.Tensor] = {}
    source_projection: dict[str, Any] | None = None
    expected_offset = 0
    bytes_read = PREFIX.size + len(header_raw) + len(footer_raw)
    with target.open("rb") as stream:
        for row in fields:
            if not isinstance(row, Mapping):
                raise _fail("world checkpoint field row is malformed")
            field = str(row["field"])
            dtype = row.get("dtype")
            shape = row.get("shape")
            logical = row.get("logical_bytes")
            if (
                not isinstance(dtype, str)
                or dtype not in streamed._DTYPE_BY_NAME
                or not isinstance(shape, list)
                or any(type(value) is not int or value < 0 for value in shape)
                or type(logical) is not int
                or logical < 0
            ):
                raise _fail("world checkpoint field schema is invalid")
            storage = row.get("storage")
            if storage == "zero_fill":
                if (
                    row.get("offset") is not None
                    or row.get("bytes") != 0
                    or row.get("zero_fill_contract") != packed.ZERO_FILL_CONTRACT
                    or row.get("sha256") != packed.aggregate._zero_digest(logical)
                ):
                    raise _fail("world checkpoint zero-fill authority is invalid")
                tensors[field] = torch.zeros(
                    tuple(shape), dtype=streamed._DTYPE_BY_NAME[dtype], device=device
                )
                continue
            byte_count = row.get("bytes")
            if (
                storage not in {"payload", "payload_zlib_v1", "payload_zstd_v1"}
                or row.get("offset") != expected_offset
                or type(byte_count) is not int
                or byte_count < 0
                or row.get("zero_fill_contract") is not None
                or (
                    storage == "payload"
                    and (
                        byte_count != logical
                        or row.get("compression") not in {None, ""}
                    )
                )
                or (
                    storage == "payload_zlib_v1"
                    and (
                        byte_count <= 0
                        or row.get("compression") != LEGACY_FIELD_COMPRESSION
                    )
                )
                or (
                    storage == "payload_zstd_v1"
                    and (byte_count <= 0 or row.get("compression") != FIELD_COMPRESSION)
                )
            ):
                raise _fail("world checkpoint payload range is invalid")
            stream.seek(PREFIX.size + expected_offset)
            stored = stream.read(byte_count)
            if len(stored) != byte_count:
                raise _fail(f"world checkpoint field range is truncated: {field}")
            payload_digest.update(stored)
            bytes_read += len(stored)
            expected_offset += len(stored)
            if storage == "payload_zstd_v1":
                try:
                    raw = zstd.decompress(stored)
                except (ValueError, zstd.ZstdError) as exc:
                    raise _fail(
                        f"world checkpoint compressed field is unreadable: {field}"
                    ) from exc
                if len(raw) != logical:
                    raise _fail(
                        f"world checkpoint compressed field length differs: {field}"
                    )
            elif storage == "payload_zlib_v1":
                try:
                    decompressor = zlib.decompressobj()
                    raw = decompressor.decompress(stored, logical + 1)
                except zlib.error as exc:
                    raise _fail(
                        f"world checkpoint compressed field is unreadable: {field}"
                    ) from exc
                if (
                    len(raw) != logical
                    or not decompressor.eof
                    or decompressor.unconsumed_tail
                    or decompressor.unused_data
                ):
                    raise _fail(
                        f"world checkpoint compressed field length differs: {field}"
                    )
            else:
                raw = stored
            if len(raw) != logical or _sha256(raw) != row.get("sha256"):
                raise _fail(f"world checkpoint field digest differs: {field}")
            tensor = streamed._tensor_from_bytes(
                raw, dtype_name=dtype, shape=tuple(shape)
            )
            if str(device) != "cpu":
                tensor = tensor.to(device)
            tensors[field] = tensor
        source_dependency = header.get("source_dependency")
        if (
            not isinstance(source_dependency, Mapping)
            or source_dependency.get("offset") != expected_offset
            or type(source_dependency.get("bytes")) is not int
            or int(source_dependency["bytes"]) <= 0
            or not _is_sha256(source_dependency.get("sha256"))
            or not _is_sha256(source_dependency.get("source_projection_sha256"))
        ):
            raise _fail("world checkpoint embedded source dependency is invalid")
        stream.seek(PREFIX.size + expected_offset)
        source_raw = stream.read(int(source_dependency["bytes"]))
        if (
            len(source_raw) != source_dependency["bytes"]
            or _sha256(source_raw) != source_dependency["sha256"]
        ):
            raise _fail("world checkpoint embedded source dependency digest differs")
        payload_digest.update(source_raw)
        bytes_read += len(source_raw)
        expected_offset += len(source_raw)
        source_kind = source_dependency.get("kind")
        if source_kind == "embedded_component_v1":
            if source_dependency.get("serialization") != "torch_save_component_v1":
                raise _fail("world checkpoint embedded source dependency is invalid")
            try:
                source_value = torch.load(
                    io.BytesIO(source_raw), map_location="cpu", weights_only=False
                )
            except Exception as exc:
                raise _fail(
                    "world checkpoint embedded source dependency is unreadable"
                ) from exc
            if not isinstance(source_value, Mapping):
                raise _fail("world checkpoint embedded source projection is malformed")
            source_projection = dict(source_value)
        elif source_kind == "compact_source_projection_v1":
            skeleton = source_dependency.get("skeleton")
            tensor_rows = source_dependency.get("tensors")
            if (
                source_dependency.get("schema_version") != COMPACT_SOURCE_SCHEMA_VERSION
                or source_dependency.get("serialization")
                != "torch_save_skeleton_plus_tensor_ranges_v1"
                or type(source_dependency.get("source_component_input_bytes"))
                is not int
                or int(source_dependency["source_component_input_bytes"]) <= 0
                or source_dependency.get("source_projection_digest_binding")
                not in {
                    "verified_from_projection_v1",
                    "inherited_from_sealed_physical_recipe_v1",
                    "verified_indexed_source_v1",
                }
                or not isinstance(skeleton, Mapping)
                or skeleton.get("offset") != 0
                or type(skeleton.get("bytes")) is not int
                or not 0 < int(skeleton["bytes"]) <= len(source_raw)
                or not _is_sha256(skeleton.get("sha256"))
                or not isinstance(tensor_rows, list)
                or source_dependency.get("tensor_manifest_sha256")
                != _sha256(_canonical_json(tensor_rows))
            ):
                raise _fail("world checkpoint compact source authority is invalid")
            skeleton_raw = source_raw[: int(skeleton["bytes"])]
            if _sha256(skeleton_raw) != skeleton["sha256"]:
                raise _fail("world checkpoint compact source skeleton digest differs")
            try:
                skeleton_value = torch.load(
                    io.BytesIO(skeleton_raw), map_location="cpu", weights_only=False
                )
            except Exception as exc:
                raise _fail(
                    "world checkpoint compact source skeleton is unreadable"
                ) from exc
            if not isinstance(skeleton_value, Mapping) or "tensors" in skeleton_value:
                raise _fail("world checkpoint compact source skeleton is malformed")
            source_tensors: dict[str, torch.Tensor] = {}
            compact_payload_cursor = int(skeleton["bytes"])
            for tensor_row in tensor_rows:
                if not isinstance(tensor_row, Mapping):
                    raise _fail(
                        "world checkpoint compact source tensor row is malformed"
                    )
                name = tensor_row.get("name")
                dtype = tensor_row.get("dtype")
                shape = tensor_row.get("shape")
                logical_bytes = tensor_row.get("logical_bytes")
                if (
                    not isinstance(name, str)
                    or not name
                    or name in source_tensors
                    or not isinstance(dtype, str)
                    or dtype not in streamed._DTYPE_BY_NAME
                    or not isinstance(shape, list)
                    or any(type(value) is not int or value < 0 for value in shape)
                    or type(logical_bytes) is not int
                    or logical_bytes < 0
                    or not _is_sha256(tensor_row.get("sha256"))
                ):
                    raise _fail(
                        "world checkpoint compact source tensor identity is invalid"
                    )
                storage = tensor_row.get("storage")
                if storage == "world_field_alias":
                    alias = tensor_row.get("alias_of")
                    value = tensors.get(alias) if isinstance(alias, str) else None
                    if (
                        not isinstance(value, torch.Tensor)
                        or list(value.shape) != shape
                        or streamed._dtype_name(value) != dtype
                        or tensor_row.get("offset") is not None
                        or tensor_row.get("bytes") != 0
                    ):
                        raise _fail("world checkpoint compact source alias is invalid")
                    # The legacy embedded source component produced independent
                    # tensor storage.  Preserve that restore behavior even when
                    # the durable representation aliases an identical field.
                    source_tensor = value.detach().to("cpu").contiguous().clone()
                elif storage == "zero_fill":
                    if (
                        tensor_row.get("alias_of") is not None
                        or tensor_row.get("offset") is not None
                        or tensor_row.get("bytes") != 0
                        or tensor_row.get("sha256")
                        != packed.aggregate._zero_digest(logical_bytes)
                    ):
                        raise _fail(
                            "world checkpoint compact source zero-fill is invalid"
                        )
                    source_tensor = torch.zeros(
                        tuple(shape), dtype=streamed._DTYPE_BY_NAME[dtype]
                    )
                elif storage == "payload":
                    offset = tensor_row.get("offset")
                    byte_count = tensor_row.get("bytes")
                    if (
                        tensor_row.get("alias_of") is not None
                        or type(offset) is not int
                        or type(byte_count) is not int
                        or byte_count != logical_bytes
                        or offset != compact_payload_cursor
                        or offset + byte_count > len(source_raw)
                    ):
                        raise _fail(
                            "world checkpoint compact source payload range is invalid"
                        )
                    tensor_raw = source_raw[offset : offset + byte_count]
                    if _sha256(tensor_raw) != tensor_row["sha256"]:
                        raise _fail(
                            "world checkpoint compact source tensor digest differs"
                        )
                    source_tensor = streamed._tensor_from_bytes(
                        tensor_raw, dtype_name=dtype, shape=tuple(shape)
                    )
                    compact_payload_cursor += byte_count
                elif storage in {
                    "world_field_xor_zlib_v1",
                    "world_field_xor_zstd_v1",
                }:
                    alias = tensor_row.get("alias_of")
                    offset = tensor_row.get("offset")
                    byte_count = tensor_row.get("bytes")
                    base = tensors.get(alias) if isinstance(alias, str) else None
                    if (
                        not isinstance(base, torch.Tensor)
                        or list(base.shape) != shape
                        or streamed._dtype_name(base) != dtype
                        or type(offset) is not int
                        or type(byte_count) is not int
                        or byte_count <= 0
                        or offset != compact_payload_cursor
                        or offset + byte_count > len(source_raw)
                        or tensor_row.get("compression")
                        != (
                            FIELD_COMPRESSION
                            if storage == "world_field_xor_zstd_v1"
                            else LEGACY_FIELD_COMPRESSION
                        )
                        or not _is_sha256(tensor_row.get("base_sha256"))
                    ):
                        raise _fail("world checkpoint compact source delta is invalid")
                    base_raw = streamed._tensor_bytes(base)
                    if _sha256(base_raw) != tensor_row["base_sha256"]:
                        raise _fail(
                            "world checkpoint compact source delta base differs"
                        )
                    compressed = source_raw[offset : offset + byte_count]
                    try:
                        if storage == "world_field_xor_zstd_v1":
                            xor_raw = zstd.decompress(compressed)
                        else:
                            decompressor = zlib.decompressobj()
                            xor_raw = decompressor.decompress(
                                compressed, logical_bytes + 1
                            )
                            if (
                                not decompressor.eof
                                or decompressor.unconsumed_tail
                                or decompressor.unused_data
                            ):
                                raise _fail(
                                    "world checkpoint compact source delta length differs"
                                )
                    except (ValueError, zstd.ZstdError, zlib.error) as exc:
                        raise _fail(
                            "world checkpoint compact source delta is unreadable"
                        ) from exc
                    if len(xor_raw) != logical_bytes:
                        raise _fail(
                            "world checkpoint compact source delta length differs"
                        )
                    tensor_raw = np.bitwise_xor(
                        np.frombuffer(base_raw, dtype=np.uint8),
                        np.frombuffer(xor_raw, dtype=np.uint8),
                    ).tobytes()
                    if _sha256(tensor_raw) != tensor_row["sha256"]:
                        raise _fail(
                            "world checkpoint compact source delta digest differs"
                        )
                    source_tensor = streamed._tensor_from_bytes(
                        tensor_raw, dtype_name=dtype, shape=tuple(shape)
                    )
                    compact_payload_cursor += byte_count
                else:
                    raise _fail("world checkpoint compact source storage is invalid")
                if (
                    source_tensor.numel() * source_tensor.element_size()
                    != logical_bytes
                    or _sha256(streamed._tensor_bytes(source_tensor))
                    != tensor_row["sha256"]
                ):
                    raise _fail("world checkpoint compact source tensor differs")
                source_tensors[name] = source_tensor
            if compact_payload_cursor != len(source_raw):
                raise _fail("world checkpoint compact source payload is not exact")
            source_projection = dict(skeleton_value)
            source_projection["tensors"] = source_tensors
        else:
            raise _fail("world checkpoint embedded source dependency is invalid")
        from engine.native_runtime.checkpoint_digests import (
            checkpoint_source_projection_authority_manifest,
        )

        observed_source_digest = checkpoint_source_projection_authority_manifest(
            source_projection
        ).get("source_projection_sha256")
        if observed_source_digest != source_dependency["source_projection_sha256"]:
            raise _fail("world checkpoint embedded source projection digest differs")
    if expected_offset != payload_len or payload_digest.hexdigest() != footer.get(
        "payload_sha256"
    ):
        raise _fail("world checkpoint payload digest differs")
    state_payload = {
        "schema_version": WORLD_SCHEMA_VERSION,
        "cell_id": header["cell_id"],
        "physical_cell_index": header["physical_cell_index"],
        "absolute_next_tick": header["absolute_next_tick"],
        "scalar_metadata_sha256": header["scalar_metadata_sha256"],
        "field_manifest_sha256": header["field_manifest_sha256"],
        "fields": fields,
        "source_dependency": header["source_dependency"],
    }
    if header.get("world_state_sha256") != _sha256(_canonical_json(state_payload)):
        raise _fail("world checkpoint state digest differs")
    if footer.get("world_state_sha256") != header.get("world_state_sha256"):
        raise _fail("world checkpoint state footer binding differs")
    scalar_metadata = header.get("scalar_metadata")
    recipe = (
        scalar_metadata.get("settlement_recipe")
        if isinstance(scalar_metadata, Mapping)
        else None
    )
    route = recipe.get("route") if isinstance(recipe, Mapping) else None
    expected_mechanics = (
        _sha256(
            _canonical_json(
                {
                    "schema_version": WORLD_SCHEMA_VERSION,
                    "cell_id": header["cell_id"],
                    "physical_cell_index": header["physical_cell_index"],
                    "absolute_next_tick": header["absolute_next_tick"],
                    "route": dict(route),
                    "world_state_sha256": header["world_state_sha256"],
                }
            )
        )
        if isinstance(route, Mapping)
        else None
    )
    if (
        expected_mechanics is None
        or header.get("world_mechanics_sha256") != expected_mechanics
        or footer.get("world_mechanics_sha256") != expected_mechanics
    ):
        raise _fail("world checkpoint mechanics digest differs")
    expected_content = _content_digest(
        header_sha256=footer["header_sha256"],
        payload_sha256=footer["payload_sha256"],
        world_state_sha256=footer["world_state_sha256"],
    )
    if footer.get("content_sha256") != expected_content:
        raise _fail("world checkpoint content digest differs")
    if byte_counter is not None:
        byte_counter["bytes_read"] = byte_counter.get("bytes_read", 0) + bytes_read
        byte_counter["world_objects_read"] = (
            byte_counter.get("world_objects_read", 0) + 1
        )
    return {
        "schema_version": WORLD_SCHEMA_VERSION,
        "checkpoint_set_id": header["checkpoint_set_id"],
        "cell_id": header["cell_id"],
        "physical_cell_index": header["physical_cell_index"],
        "absolute_next_tick": header["absolute_next_tick"],
        "field_order": list(field_order),
        "fields": tensors,
        "source_projection": source_projection,
        "scalar_metadata": header["scalar_metadata"],
        "header": header,
        "footer": footer,
        "artifact_sha256": artifact,
        "bytes_read": bytes_read,
        "path": str(target),
    }


def load_world_checkpoint_set_manifest_v1(
    path: str | os.PathLike[str],
    *,
    byte_counter: dict[str, int] | None = None,
) -> dict[str, Any]:
    target = Path(path).expanduser().resolve()
    try:
        raw = target.read_bytes()
        value = json.loads(raw.decode("ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise _fail(f"checkpoint-set manifest cannot be read: {target}") from exc
    if not isinstance(value, dict):
        raise _fail("checkpoint-set manifest is not an object")
    if raw != _canonical_json(value):
        raise _fail("checkpoint-set manifest encoding is not canonical")
    digest = value.get("manifest_sha256")
    body = dict(value)
    body.pop("manifest_sha256", None)
    objects = value.get("objects")
    cell_ids = value.get("cell_ids")
    if (
        value.get("schema_version") != SET_MANIFEST_SCHEMA_VERSION
        or value.get("status") != "complete"
        or not _is_sha256(digest)
        or digest != _sha256(_canonical_json(body))
        or not isinstance(objects, list)
        or not isinstance(cell_ids, list)
        or value.get("world_count") != len(cell_ids)
        or len(cell_ids) != len(objects)
        or [row.get("cell_id") for row in objects if isinstance(row, Mapping)]
        != cell_ids
        or [
            row.get("physical_cell_index")
            for row in objects
            if isinstance(row, Mapping)
        ]
        != list(range(len(cell_ids)))
    ):
        raise _fail("checkpoint-set manifest authority is invalid")
    if target.name != "checkpoint_set_manifest.json" or target.parent.name != value.get(
        "checkpoint_set_id"
    ):
        raise _fail("checkpoint-set manifest path is not its set identity")
    if len(set(cell_ids)) != len(cell_ids):
        raise _fail("checkpoint-set manifest duplicates a world identity")
    if any(
        not isinstance(row, Mapping)
        or not _is_sha256(row.get("artifact_sha256"))
        or not _is_sha256(row.get("world_state_sha256"))
        or not _is_sha256(row.get("world_mechanics_sha256"))
        or not isinstance(row.get("relative_path"), str)
        or Path(str(row["relative_path"])).is_absolute()
        or ".." in Path(str(row["relative_path"])).parts
        for row in objects
    ):
        raise _fail("checkpoint-set world object authority is invalid")
    expected_batch_digest = _sha256(
        _canonical_json(
            [
                {
                    "cell_id": row["cell_id"],
                    "physical_cell_index": row["physical_cell_index"],
                    "artifact_sha256": row["artifact_sha256"],
                    "world_state_sha256": row["world_state_sha256"],
                    "world_mechanics_sha256": row["world_mechanics_sha256"],
                }
                for row in objects
            ]
        )
    )
    if value.get("batch_digest_sha256") != expected_batch_digest:
        raise _fail("checkpoint-set batch digest differs")
    for total_name in ("payload_bytes", "logical_bytes", "physical_file_bytes"):
        if value.get(total_name) != sum(
            int(row.get(total_name, -1)) for row in objects
        ):
            raise _fail(f"checkpoint-set {total_name} total differs")
    if byte_counter is not None:
        byte_counter["bytes_read"] = byte_counter.get("bytes_read", 0) + len(raw)
        byte_counter["set_manifests_read"] = (
            byte_counter.get("set_manifests_read", 0) + 1
        )
    return value


def load_selected_world_from_set_v1(
    manifest_path: str | os.PathLike[str],
    *,
    world_id: str,
    device: str | torch.device = "cpu",
    byte_counter: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Read one manifest and exactly one selected world-object payload."""

    target = Path(manifest_path).expanduser().resolve()
    manifest = load_world_checkpoint_set_manifest_v1(target, byte_counter=byte_counter)
    rows = {
        str(row["cell_id"]): row
        for row in manifest["objects"]
        if isinstance(row, Mapping)
    }
    if world_id not in rows:
        raise _fail(f"world is absent from checkpoint set: {world_id}")
    row = rows[world_id]
    relative = row.get("relative_path")
    if (
        not isinstance(relative, str)
        or Path(relative).is_absolute()
        or ".." in Path(relative).parts
    ):
        raise _fail("checkpoint-set world object path is invalid")
    loaded = load_world_checkpoint_object_v1(
        target.parent / relative,
        expected_artifact_sha256=str(row.get("artifact_sha256") or ""),
        device=device,
        byte_counter=byte_counter,
    )
    if (
        loaded["checkpoint_set_id"] != manifest["checkpoint_set_id"]
        or loaded["cell_id"] != world_id
        or loaded["physical_cell_index"] != row["physical_cell_index"]
        or loaded["footer"]["world_state_sha256"] != row["world_state_sha256"]
        or loaded["footer"]["world_mechanics_sha256"] != row["world_mechanics_sha256"]
    ):
        raise _fail("selected world object differs from the set manifest")
    loaded["set_manifest"] = manifest
    return loaded


def load_full_world_checkpoint_set_v1(
    manifest_path: str | os.PathLike[str],
    *,
    device: str | torch.device = "cpu",
    byte_counter: dict[str, int] | None = None,
) -> tuple[dict[str, Any], ...]:
    """Load every object in manifest order through the selected-world path."""

    target = Path(manifest_path).expanduser().resolve()
    manifest = load_world_checkpoint_set_manifest_v1(target, byte_counter=byte_counter)
    loaded_rows: list[dict[str, Any]] = []
    for row in manifest["objects"]:
        relative = row["relative_path"]
        world = load_world_checkpoint_object_v1(
            target.parent / str(relative),
            expected_artifact_sha256=str(row["artifact_sha256"]),
            device=device,
            byte_counter=byte_counter,
        )
        if (
            world["checkpoint_set_id"] != manifest["checkpoint_set_id"]
            or world["cell_id"] != row["cell_id"]
            or world["physical_cell_index"] != row["physical_cell_index"]
            or world["footer"]["world_state_sha256"] != row["world_state_sha256"]
            or world["footer"]["world_mechanics_sha256"]
            != row["world_mechanics_sha256"]
        ):
            raise _fail("full checkpoint-set object differs from the set manifest")
        world["set_manifest"] = manifest
        loaded_rows.append(world)
    loaded = tuple(loaded_rows)
    if tuple(row["physical_cell_index"] for row in loaded) != tuple(range(len(loaded))):
        raise _fail("full checkpoint-set restore order differs")
    return loaded


def restore_world_checkpoint_projection_v1(
    world_object_path: str | os.PathLike[str],
    *,
    expected_artifact_sha256: str | None = None,
    byte_counter: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Restore one canonical projection through the existing private adapter.

    This adapter is intentionally off the durability critical path.  It is a
    compatibility/qualification bridge proving that the direct physical world
    object restores the authoritative legacy projection without accessing
    another world object.
    """

    loaded = load_world_checkpoint_object_v1(
        world_object_path,
        expected_artifact_sha256=expected_artifact_sha256,
        byte_counter=byte_counter,
    )
    scalar_metadata = loaded["scalar_metadata"]
    if (
        not isinstance(scalar_metadata, Mapping)
        or scalar_metadata.get("schema_version") != WORLD_SCALAR_METADATA_SCHEMA_VERSION
        or scalar_metadata.get("cell_id") != loaded["cell_id"]
    ):
        raise _fail("world checkpoint scalar projection authority is invalid")
    recipe = scalar_metadata.get("settlement_recipe")
    if not isinstance(recipe, Mapping):
        raise _fail("world checkpoint settlement recipe is missing")
    from engine.campaigns.acs_direct_resident_physical_capture_v1 import (
        SCHEMA_VERSION as PHYSICAL_CAPTURE_SCHEMA_VERSION,
    )
    from engine.campaigns.acs_direct_resident_physical_settlement_v1 import (
        build_physical_batch_settlement_cursor_v1,
    )

    capture = {
        "schema_version": PHYSICAL_CAPTURE_SCHEMA_VERSION,
        "settlement_recipe": dict(recipe),
    }
    reader = SimpleNamespace(
        load_selected_cell=lambda selected: {
            "fields": loaded["fields"],
            "selected_cell_index": 0,
            "selected_cell_id": loaded["cell_id"],
        }
    )
    source_projection = loaded.get("source_projection")
    if not isinstance(source_projection, Mapping):
        raise _fail("world checkpoint has no embedded source projection")
    cursor = build_physical_batch_settlement_cursor_v1(
        capture,
        reader=reader,
        source_projection_overrides={loaded["cell_id"]: source_projection},
    )
    detached = cursor.next_cell()
    if detached is None or detached.cell_id != loaded["cell_id"]:
        raise _fail("world checkpoint restore cursor returned no selected world")
    projection = dict(detached.require_projection())
    compact_result_rows = tuple(
        dict(row) for row in detached.compact_result_rows if isinstance(row, Mapping)
    )
    cursor.release_cell(loaded["cell_id"])
    receipt = cursor.close()
    if getattr(receipt, "state", None) != "SEALED":
        raise _fail("world checkpoint restore cursor did not seal")
    return {
        "projection": projection,
        "compact_result_rows": compact_result_rows,
        "cell_id": loaded["cell_id"],
        "physical_cell_index": loaded["physical_cell_index"],
        "checkpoint_set_id": loaded["checkpoint_set_id"],
        "world_object_artifact_sha256": loaded["artifact_sha256"],
        "bytes_read": loaded["bytes_read"],
    }


def restore_selected_world_projection_from_set_v1(
    manifest_path: str | os.PathLike[str],
    *,
    world_id: str,
    byte_counter: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Restore one manifest-bound canonical projection and no sibling object."""

    target = Path(manifest_path).expanduser().resolve()
    manifest = load_world_checkpoint_set_manifest_v1(target, byte_counter=byte_counter)
    row = next(
        (
            item
            for item in manifest["objects"]
            if isinstance(item, Mapping) and item.get("cell_id") == world_id
        ),
        None,
    )
    if not isinstance(row, Mapping):
        raise _fail(f"world is absent from checkpoint set: {world_id}")
    restored = restore_world_checkpoint_projection_v1(
        target.parent / str(row["relative_path"]),
        expected_artifact_sha256=str(row["artifact_sha256"]),
        byte_counter=byte_counter,
    )
    if (
        restored["checkpoint_set_id"] != manifest["checkpoint_set_id"]
        or restored["cell_id"] != world_id
        or restored["physical_cell_index"] != row["physical_cell_index"]
    ):
        raise _fail("restored world projection differs from the set manifest")
    restored["set_manifest"] = manifest
    return restored


def restore_full_world_checkpoint_set_v1(
    manifest_path: str | os.PathLike[str],
    *,
    byte_counter: dict[str, int] | None = None,
) -> tuple[dict[str, Any], ...]:
    """Restore all canonical world projections in checkpoint-set order."""

    target = Path(manifest_path).expanduser().resolve()
    manifest = load_world_checkpoint_set_manifest_v1(target, byte_counter=byte_counter)
    restored_rows: list[dict[str, Any]] = []
    for row in manifest["objects"]:
        restored_world = restore_world_checkpoint_projection_v1(
            target.parent / str(row["relative_path"]),
            expected_artifact_sha256=str(row["artifact_sha256"]),
            byte_counter=byte_counter,
        )
        if (
            restored_world["checkpoint_set_id"] != manifest["checkpoint_set_id"]
            or restored_world["cell_id"] != row["cell_id"]
            or restored_world["physical_cell_index"] != row["physical_cell_index"]
        ):
            raise _fail("restored world projection differs from the set manifest")
        restored_world["set_manifest"] = manifest
        restored_rows.append(restored_world)
    restored = tuple(restored_rows)
    if [row["cell_id"] for row in restored] != manifest["cell_ids"]:
        raise _fail("full canonical checkpoint-set restore order differs")
    return restored


__all__ = [
    "ACSWorldShardedCheckpointError",
    "FORMAT_VERSION",
    "MAGIC",
    "SET_MANIFEST_SCHEMA_VERSION",
    "WORLD_SCHEMA_VERSION",
    "load_selected_world_from_set_v1",
    "load_full_world_checkpoint_set_v1",
    "load_world_checkpoint_object_v1",
    "load_world_checkpoint_set_manifest_v1",
    "project_world_scalar_metadata_v1",
    "restore_world_checkpoint_projection_v1",
    "restore_selected_world_projection_from_set_v1",
    "restore_full_world_checkpoint_set_v1",
    "write_indexed_source_projection_v1",
    "write_world_sharded_checkpoint_set_from_packed_v1",
]
