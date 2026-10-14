"""Packing-neutral per-world index over one physical Workhorse checkpoint.

The physical checkpoint remains the single durable state payload and retains
its exact Attempt/B/K/device provenance.  This module adds immutable per-world
range identities so composition can gather the same worlds into a different
physical packing without duplicating the payload on disk or materializing it
in host memory.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import tempfile
from typing import Any, Final

from .authority import inspect_workhorse_checkpoint_bundle_v2
from .contract import (
    FullEvolutionCudaWorkhorseContractV2,
    FullEvolutionWorkhorseContractError,
    WORKHORSE_TIME_AUTHORITY_V2,
    canonical_sha256,
)
from .layout import workhorse_layout_v2
from .native_loader import file_sha256
from .scientific_control import (
    WorkhorseScientificControlError,
    validate_workhorse_active_scientific_control_state_v1,
)


WORKHORSE_INDEXED_CHECKPOINT_SET_SCHEMA_V1: Final[str] = (
    "eiel_workhorse_indexed_checkpoint_set_v1"
)
WORKHORSE_WORLD_STATE_SET_IDENTITY_SCHEMA_V1: Final[str] = (
    "eiel_workhorse_world_state_set_identity_v1"
)
WORKHORSE_INDEXED_CHECKPOINT_FILE_V1: Final[str] = "checkpoint-index.json"
WORKHORSE_REPACK_BINDING_SCHEMA_V1: Final[str] = (
    "eiel_workhorse_checkpoint_repack_binding_v1"
)
WORKHORSE_SELECTED_WORLD_CHECKPOINT_SCHEMA_V1: Final[str] = (
    "eiel_workhorse_selected_world_checkpoint_v1"
)
WORKHORSE_SELECTED_WORLD_CHECKPOINT_FILE_V1: Final[str] = (
    "selected-world-checkpoint.json"
)
WORKHORSE_SELECTED_WORLD_STATE_FILE_V1: Final[str] = "selected-world-state.bin"
WORKHORSE_SELECTED_WORLD_AUTHORITY_FILE_V1: Final[str] = (
    "selected-world-occurrence-projection.json"
)


class WorkhorseIndexedCheckpointError(RuntimeError):
    """Raised when an indexed checkpoint is incomplete or changes identity."""


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise WorkhorseIndexedCheckpointError(
            f"indexed checkpoint is not canonical JSON: {exc}"
        ) from exc


def _valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _read_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise WorkhorseIndexedCheckpointError(
            "indexed checkpoint manifest is not a regular file"
        )
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise WorkhorseIndexedCheckpointError(
            f"indexed checkpoint manifest cannot be read: {exc}"
        ) from exc
    if type(value) is not dict:
        raise WorkhorseIndexedCheckpointError(
            "indexed checkpoint manifest is not an object"
        )
    return value


def _world_state_set_identity_v1(
    *,
    contract: FullEvolutionCudaWorkhorseContractV2,
    checkpoint_at_tick: int,
    layout_sha256: str,
    world_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    identity_rows = [
        {
            "world_uid": int(row["world_uid"]),
            "rng_key": int(row["rng_key"]),
            "state_bytes": int(row["state_bytes"]),
            "state_sha256": str(row["state_sha256"]),
        }
        for row in sorted(world_rows, key=lambda item: int(item["world_uid"]))
    ]
    body = {
        "schema_version": WORKHORSE_WORLD_STATE_SET_IDENTITY_SCHEMA_V1,
        "time_authority": WORKHORSE_TIME_AUTHORITY_V2,
        "shape_id": contract.shape_id,
        "state_layout_sha256": layout_sha256,
        "semantic_transition_id": contract.semantic_transition_id,
        "semantic_transition_sha256": contract.semantic_transition_sha256,
        "checkpoint_at_tick": checkpoint_at_tick,
        "rng_epoch": 4,
        "world_count": len(identity_rows),
        "worlds_by_uid": identity_rows,
        "packing_excluded_fields": [
            "physical_B",
            "physical_K",
            "physical_slot",
            "physical_order",
            "backend",
            "artifact",
            "device",
            "attempt",
        ],
    }
    return {**body, "world_state_set_sha256": canonical_sha256(body)}


def _source_projection(
    *,
    checkpoint_manifest_path: Path,
    checkpoint: Mapping[str, Any],
    state_path: Path,
) -> dict[str, Any]:
    contract = FullEvolutionCudaWorkhorseContractV2.from_mapping(
        checkpoint["contract"]
    )
    return {
        "checkpoint_manifest_path": str(checkpoint_manifest_path.resolve()),
        "checkpoint_manifest_file_sha256": file_sha256(checkpoint_manifest_path),
        "checkpoint_bundle_body_sha256": checkpoint["bundle_body_sha256"],
        "checkpoint_identity_sha256": checkpoint["checkpoint_identity"][
            "checkpoint_identity_sha256"
        ],
        "canonical_state_path": str(state_path.resolve()),
        "canonical_state_sha256": checkpoint["canonical_state"]["sha256"],
        "canonical_state_bytes": checkpoint["canonical_state"]["bytes"],
        "physical_contract": contract.to_dict(),
        "physical_B": contract.B,
        "physical_K": contract.temporal_k,
        "artifact_identity_sha256": checkpoint["artifact"]["identity_sha256"],
        "sealed_artifact_identity_sha256": checkpoint["artifact"][
            "sealed_artifact_identity_sha256"
        ],
        "attempt_lineage_sha256": checkpoint["attempt_lineage_sha256"],
        "dispatch_identity_sha256": checkpoint["dispatch_identity_sha256"],
        "event_sink_receipt_path": checkpoint["event_sink_ack"]["receipt_path"],
        "event_sink_receipt_file_sha256": checkpoint["event_sink_ack"][
            "receipt_file_sha256"
        ],
    }


def write_workhorse_indexed_checkpoint_set_v1(
    checkpoint_manifest_path: str | Path,
    destination: str | Path,
    *,
    require_current_source_match: bool = True,
) -> dict[str, Any]:
    """Publish an immutable UID-sorted row index without copying state bytes."""

    source_manifest_path = Path(checkpoint_manifest_path).expanduser().resolve()
    try:
        checkpoint, state_path = inspect_workhorse_checkpoint_bundle_v2(
            source_manifest_path,
            require_current_source_match=require_current_source_match,
        )
        contract = FullEvolutionCudaWorkhorseContractV2.from_mapping(
            checkpoint["contract"]
        )
    except (OSError, RuntimeError, TypeError, ValueError, FullEvolutionWorkhorseContractError) as exc:
        raise WorkhorseIndexedCheckpointError(
            f"physical Workhorse checkpoint cannot be indexed: {exc}"
        ) from exc
    output = Path(destination).expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise WorkhorseIndexedCheckpointError(
            "indexed checkpoint destination must be fresh"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.parent.is_symlink():
        raise WorkhorseIndexedCheckpointError(
            "indexed checkpoint parent must not be a symbolic link"
        )
    layout = workhorse_layout_v2(contract.shape)
    row_bytes = layout.state_bytes_per_world
    world_rows: list[dict[str, Any]] = []
    aggregate_hasher = hashlib.sha256()
    with state_path.open("rb") as handle:
        for physical_slot in range(contract.B):
            state = handle.read(row_bytes)
            if len(state) != row_bytes:
                raise WorkhorseIndexedCheckpointError(
                    "physical checkpoint state is truncated while indexing"
                )
            aggregate_hasher.update(state)
            header = struct.unpack("<32Q", state[: 32 * 8])
            world_uid = int(header[0])
            rng_key = int(header[13])
            if (
                world_uid <= 0
                or header[1] != checkpoint["checkpoint_at_tick"]
                or header[4] != 0
                or header[14] != 4
            ):
                raise WorkhorseIndexedCheckpointError(
                    "indexed world header has another identity/frontier"
                )
            world_rows.append(
                {
                    "world_uid": world_uid,
                    "rng_key": rng_key,
                    "source_physical_slot": physical_slot,
                    "state_offset": physical_slot * row_bytes,
                    "state_bytes": row_bytes,
                    "state_sha256": hashlib.sha256(state).hexdigest(),
                }
            )
        if handle.read(1):
            raise WorkhorseIndexedCheckpointError(
                "physical checkpoint state exceeds its exact B/layout"
            )
    if (
        aggregate_hasher.hexdigest() != checkpoint["canonical_state"]["sha256"]
        or len({row["world_uid"] for row in world_rows}) != contract.B
        or [row["world_uid"] for row in world_rows] != checkpoint["world_uids"]
        or [row["rng_key"] for row in world_rows] != checkpoint["rng_keys"]
    ):
        raise WorkhorseIndexedCheckpointError(
            "physical checkpoint changed while its world index was built"
        )
    ordered_rows = sorted(world_rows, key=lambda row: row["world_uid"])
    identity = _world_state_set_identity_v1(
        contract=contract,
        checkpoint_at_tick=int(checkpoint["checkpoint_at_tick"]),
        layout_sha256=layout.layout_sha256,
        world_rows=ordered_rows,
    )
    body = {
        "schema_version": WORKHORSE_INDEXED_CHECKPOINT_SET_SCHEMA_V1,
        "time_authority": WORKHORSE_TIME_AUTHORITY_V2,
        "world_state_set_identity": identity,
        "source_physical_checkpoint": _source_projection(
            checkpoint_manifest_path=source_manifest_path,
            checkpoint=checkpoint,
            state_path=state_path,
        ),
        "world_index_by_uid": ordered_rows,
        "storage_model": {
            "canonical_state_payload_copies": 1,
            "indexed_world_payload_copies": 0,
            "world_rows_are_range_references": True,
            "source_payload_must_remain_live": True,
        },
    }
    manifest = {**body, "manifest_body_sha256": canonical_sha256(body)}
    stage = Path(
        tempfile.mkdtemp(
            prefix=f".{output.name}.", suffix=".index.tmp", dir=output.parent
        )
    )
    try:
        manifest_path = stage / WORKHORSE_INDEXED_CHECKPOINT_FILE_V1
        with manifest_path.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        manifest_path.chmod(0o444)
        _fsync_directory(stage)
        if output.exists() or output.is_symlink():
            raise WorkhorseIndexedCheckpointError(
                "indexed checkpoint destination appeared during publication"
            )
        os.rename(stage, output)
        output.chmod(0o555)
        _fsync_directory(output.parent)
    except BaseException:
        if stage.exists() and not stage.is_symlink():
            shutil.rmtree(stage)
        raise
    return {
        **manifest,
        "manifest_path": str(
            (output / WORKHORSE_INDEXED_CHECKPOINT_FILE_V1).resolve()
        ),
        "manifest_file_sha256": file_sha256(
            output / WORKHORSE_INDEXED_CHECKPOINT_FILE_V1
        ),
    }


def inspect_workhorse_indexed_checkpoint_set_v1(
    manifest_path: str | Path,
    *,
    require_current_source_match: bool = True,
) -> tuple[dict[str, Any], Path]:
    """Reopen and fully validate an indexed physical payload and every row."""

    path = Path(manifest_path).expanduser().resolve()
    value = _read_json(path)
    expected_fields = {
        "schema_version",
        "time_authority",
        "world_state_set_identity",
        "source_physical_checkpoint",
        "world_index_by_uid",
        "storage_model",
        "manifest_body_sha256",
    }
    body = {key: item for key, item in value.items() if key != "manifest_body_sha256"}
    if (
        set(value) != expected_fields
        or value.get("schema_version") != WORKHORSE_INDEXED_CHECKPOINT_SET_SCHEMA_V1
        or value.get("time_authority") != WORKHORSE_TIME_AUTHORITY_V2
        or value.get("manifest_body_sha256") != canonical_sha256(body)
        or path.name != WORKHORSE_INDEXED_CHECKPOINT_FILE_V1
        or path.parent.stat().st_mode & 0o222
        or value.get("storage_model")
        != {
            "canonical_state_payload_copies": 1,
            "indexed_world_payload_copies": 0,
            "world_rows_are_range_references": True,
            "source_payload_must_remain_live": True,
        }
    ):
        raise WorkhorseIndexedCheckpointError(
            "indexed checkpoint schema, seal, or storage model is invalid"
        )
    source = value.get("source_physical_checkpoint")
    rows = value.get("world_index_by_uid")
    identity = value.get("world_state_set_identity")
    if type(source) is not dict or type(rows) is not list or type(identity) is not dict:
        raise WorkhorseIndexedCheckpointError(
            "indexed checkpoint source/world identity is malformed"
        )
    source_fields = {
        "checkpoint_manifest_path",
        "checkpoint_manifest_file_sha256",
        "checkpoint_bundle_body_sha256",
        "checkpoint_identity_sha256",
        "canonical_state_path",
        "canonical_state_sha256",
        "canonical_state_bytes",
        "physical_contract",
        "physical_B",
        "physical_K",
        "artifact_identity_sha256",
        "sealed_artifact_identity_sha256",
        "attempt_lineage_sha256",
        "dispatch_identity_sha256",
        "event_sink_receipt_path",
        "event_sink_receipt_file_sha256",
    }
    row_fields = {
        "world_uid",
        "rng_key",
        "source_physical_slot",
        "state_offset",
        "state_bytes",
        "state_sha256",
    }
    if set(source) != source_fields or any(type(row) is not dict or set(row) != row_fields for row in rows):
        raise WorkhorseIndexedCheckpointError(
            "indexed checkpoint provenance or row fields are inexact"
        )
    source_manifest_path = Path(source["checkpoint_manifest_path"]).resolve()
    if (
        source_manifest_path.is_symlink()
        or not source_manifest_path.is_file()
        or file_sha256(source_manifest_path)
        != source["checkpoint_manifest_file_sha256"]
    ):
        raise WorkhorseIndexedCheckpointError(
            "indexed checkpoint source manifest changed"
        )
    try:
        checkpoint, state_path = inspect_workhorse_checkpoint_bundle_v2(
            source_manifest_path,
            require_current_source_match=require_current_source_match,
        )
        contract = FullEvolutionCudaWorkhorseContractV2.from_mapping(
            checkpoint["contract"]
        )
    except (OSError, RuntimeError, TypeError, ValueError, FullEvolutionWorkhorseContractError) as exc:
        raise WorkhorseIndexedCheckpointError(
            f"indexed checkpoint source cannot reopen: {exc}"
        ) from exc
    expected_source = _source_projection(
        checkpoint_manifest_path=source_manifest_path,
        checkpoint=checkpoint,
        state_path=state_path,
    )
    if source != expected_source:
        raise WorkhorseIndexedCheckpointError(
            "indexed checkpoint physical provenance changed"
        )
    layout = workhorse_layout_v2(contract.shape)
    row_bytes = layout.state_bytes_per_world
    if (
        len(rows) != contract.B
        or rows != sorted(rows, key=lambda row: row["world_uid"])
        or len({row["world_uid"] for row in rows}) != contract.B
        or {row["source_physical_slot"] for row in rows} != set(range(contract.B))
        or any(
            type(row["world_uid"]) is not int
            or row["world_uid"] <= 0
            or type(row["rng_key"]) is not int
            or row["rng_key"] < 0
            or type(row["source_physical_slot"]) is not int
            or row["state_offset"] != row["source_physical_slot"] * row_bytes
            or row["state_bytes"] != row_bytes
            or not _valid_sha256(row["state_sha256"])
            for row in rows
        )
    ):
        raise WorkhorseIndexedCheckpointError(
            "indexed checkpoint world coverage is invalid"
        )
    by_slot = {row["source_physical_slot"]: row for row in rows}
    aggregate_hasher = hashlib.sha256()
    with state_path.open("rb") as handle:
        for physical_slot in range(contract.B):
            state = handle.read(row_bytes)
            if len(state) != row_bytes:
                raise WorkhorseIndexedCheckpointError(
                    "indexed checkpoint source state is truncated"
                )
            aggregate_hasher.update(state)
            row = by_slot[physical_slot]
            header = struct.unpack("<32Q", state[: 32 * 8])
            if (
                hashlib.sha256(state).hexdigest() != row["state_sha256"]
                or header[0] != row["world_uid"]
                or header[13] != row["rng_key"]
                or header[1] != checkpoint["checkpoint_at_tick"]
                or header[4] != 0
                or header[14] != 4
            ):
                raise WorkhorseIndexedCheckpointError(
                    "indexed checkpoint world row changed"
                )
        if handle.read(1):
            raise WorkhorseIndexedCheckpointError(
                "indexed checkpoint source state exceeds its exact layout"
            )
    expected_identity = _world_state_set_identity_v1(
        contract=contract,
        checkpoint_at_tick=int(checkpoint["checkpoint_at_tick"]),
        layout_sha256=layout.layout_sha256,
        world_rows=rows,
    )
    if (
        aggregate_hasher.hexdigest() != source["canonical_state_sha256"]
        or identity != expected_identity
    ):
        raise WorkhorseIndexedCheckpointError(
            "indexed checkpoint state-set identity does not recompute"
        )
    return value, state_path


def materialize_workhorse_selected_world_checkpoint_v1(
    indexed_checkpoint_manifest_path: str | Path,
    destination: str | Path,
    *,
    world_uid: int,
    checkpoint_occurrence_projection: Mapping[str, Any],
    semantic_world_sha256: str,
    control_history_sha256: str,
    rng_state_sha256: str,
    event_prefix_sha256: str,
    active_scientific_control_state: Mapping[str, Any],
) -> dict[str, Any]:
    """Copy one exact indexed row into a bounded independently restorable object.

    The aggregate physical checkpoint remains the provenance source, but the
    selected row no longer depends on that aggregate remaining live.  This is
    the retained-parent storage boundary: the state payload is copied exactly
    once and all non-packing scientific identities are sealed beside it.
    """

    if type(world_uid) is not int or world_uid <= 0:
        raise WorkhorseIndexedCheckpointError(
            "selected Workhorse world UID must be a positive integer"
        )
    projection = dict(checkpoint_occurrence_projection)
    projection_core = {
        key: item
        for key, item in projection.items()
        if key != "projection_content_sha256"
    }
    occurrence = projection.get("occurrence_digest")
    parent_occurrence = projection.get("parent_checkpoint_digest")
    projection_fields = {
        "schema_version",
        "claim_class",
        "science_authority_granted",
        "qualification_authority_granted",
        "time_authority",
        "checkpoint_at_tick",
        "checkpoint_manifest_path",
        "checkpoint_manifest_file_sha256",
        "checkpoint_bundle_body_sha256",
        "checkpoint_identity_sha256",
        "canonical_state_sha256",
        "canonical_parent_binding_sha256",
        "operation_contract_sha256",
        "absolute_tick_bundle_path",
        "absolute_tick_bundle_digest",
        "occurrence_root",
        "occurrence_digest",
        "occurrence_descriptor_path",
        "occurrence_descriptor_file_sha256",
        "occurrence_commit_path",
        "occurrence_commit_file_sha256",
        "parent_checkpoint_digest",
        "canonical_visible",
        "restore_admitted_exact",
        "projection_content_sha256",
    }
    if (
        set(projection) != projection_fields
        or projection.get("schema_version")
        not in {
            "eiel_workhorse_row_checkpoint_occurrence_projection_v2",
            "eiel_workhorse_indexed_continuation_checkpoint_occurrence_projection_v2",
        }
        or projection.get("claim_class") != "TECHNICAL_ONLY"
        or projection.get("science_authority_granted") is not False
        or projection.get("qualification_authority_granted") is not False
        or projection.get("time_authority") != WORKHORSE_TIME_AUTHORITY_V2
        or projection.get("canonical_visible") is not True
        or projection.get("restore_admitted_exact") is not False
        or projection.get("projection_content_sha256")
        != canonical_sha256(projection_core)
        or not isinstance(occurrence, Mapping)
        or set(occurrence) != {"algorithm", "hex"}
        or occurrence.get("algorithm") != "sha256"
        or not isinstance(parent_occurrence, Mapping)
        or set(parent_occurrence) != {"algorithm", "hex"}
        or parent_occurrence.get("algorithm") != "sha256"
    ):
        raise WorkhorseIndexedCheckpointError(
            "selected Workhorse occurrence projection is not exact and committed"
        )
    digests = {
        "checkpoint_occurrence_sha256": occurrence.get("hex"),
        "parent_occurrence_sha256": parent_occurrence.get("hex"),
        "semantic_world_sha256": semantic_world_sha256,
        "control_history_sha256": control_history_sha256,
        "rng_state_sha256": rng_state_sha256,
        "event_prefix_sha256": event_prefix_sha256,
    }
    if any(not _valid_sha256(value) for value in digests.values()):
        raise WorkhorseIndexedCheckpointError(
            "selected Workhorse authority digests must be lowercase SHA-256"
        )
    try:
        active_controls = validate_workhorse_active_scientific_control_state_v1(
            active_scientific_control_state
        )
    except WorkhorseScientificControlError as exc:
        raise WorkhorseIndexedCheckpointError(
            f"selected Workhorse active controls are invalid: {exc}"
        ) from exc
    source_path = Path(indexed_checkpoint_manifest_path).expanduser().resolve()
    source, aggregate_state_path = inspect_workhorse_indexed_checkpoint_set_v1(
        source_path
    )
    matches = [
        row for row in source["world_index_by_uid"] if row["world_uid"] == world_uid
    ]
    if len(matches) != 1:
        raise WorkhorseIndexedCheckpointError(
            "selected Workhorse world is absent or ambiguous in its source index"
        )
    row = dict(matches[0])
    output = Path(destination).expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise WorkhorseIndexedCheckpointError(
            "selected Workhorse checkpoint destination must be fresh"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.parent.is_symlink():
        raise WorkhorseIndexedCheckpointError(
            "selected Workhorse checkpoint parent must not be a symbolic link"
        )
    with aggregate_state_path.open("rb") as handle:
        handle.seek(row["state_offset"])
        state = handle.read(row["state_bytes"])
    header = struct.unpack("<32Q", state[: 32 * 8]) if len(state) >= 32 * 8 else ()
    identity = source["world_state_set_identity"]
    if (
        len(state) != row["state_bytes"]
        or hashlib.sha256(state).hexdigest() != row["state_sha256"]
        or not header
        or header[0] != world_uid
        or header[1] != identity["checkpoint_at_tick"]
        or header[4] != 0
        or header[13] != row["rng_key"]
        or header[14] != identity["rng_epoch"]
        or projection.get("checkpoint_at_tick") != identity["checkpoint_at_tick"]
        or projection.get("checkpoint_identity_sha256")
        != source["source_physical_checkpoint"]["checkpoint_identity_sha256"]
    ):
        raise WorkhorseIndexedCheckpointError(
            "selected Workhorse row changed during materialization"
        )
    body = {
        "schema_version": WORKHORSE_SELECTED_WORLD_CHECKPOINT_SCHEMA_V1,
        "time_authority": WORKHORSE_TIME_AUTHORITY_V2,
        "selected_world": {
            "world_uid": world_uid,
            "rng_key": row["rng_key"],
            "checkpoint_at_tick": identity["checkpoint_at_tick"],
            "rng_epoch": identity["rng_epoch"],
            "shape_id": identity["shape_id"],
            "state_layout_sha256": identity["state_layout_sha256"],
            "semantic_transition_id": identity["semantic_transition_id"],
            "semantic_transition_sha256": identity[
                "semantic_transition_sha256"
            ],
            "state_bytes": row["state_bytes"],
            "state_sha256": row["state_sha256"],
        },
        "source_indexed_checkpoint": {
            "manifest_path": str(source_path),
            "manifest_file_sha256": file_sha256(source_path),
            "manifest_body_sha256": source["manifest_body_sha256"],
            "world_state_set_sha256": identity["world_state_set_sha256"],
            "world_count": identity["world_count"],
            "source_checkpoint_identity_sha256": source[
                "source_physical_checkpoint"
            ]["checkpoint_identity_sha256"],
            "source_physical_slot": row["source_physical_slot"],
            "source_state_offset": row["state_offset"],
        },
        "authority": {
            **dict(digests),
            "checkpoint_occurrence_projection": {
                "file_name": WORKHORSE_SELECTED_WORLD_AUTHORITY_FILE_V1,
                "bytes": len(_canonical_json_bytes(projection)),
                "sha256": hashlib.sha256(
                    _canonical_json_bytes(projection)
                ).hexdigest(),
                "projection_content_sha256": projection[
                    "projection_content_sha256"
                ],
            },
        },
        "active_scientific_control_state": active_controls,
        "state_payload": {
            "file_name": WORKHORSE_SELECTED_WORLD_STATE_FILE_V1,
            "bytes": len(state),
            "sha256": hashlib.sha256(state).hexdigest(),
        },
        "storage_model": {
            "selected_world_payload_copies": 1,
            "aggregate_payload_required_after_materialization": False,
            "logical_to_physical_repacking_is_nonsemantic": True,
        },
    }
    manifest = {**body, "manifest_body_sha256": canonical_sha256(body)}
    stage = Path(
        tempfile.mkdtemp(
            prefix=f".{output.name}.", suffix=".selected.tmp", dir=output.parent
        )
    )
    try:
        state_path = stage / WORKHORSE_SELECTED_WORLD_STATE_FILE_V1
        with state_path.open("xb") as handle:
            handle.write(state)
            handle.flush()
            os.fsync(handle.fileno())
        state_path.chmod(0o444)
        authority_path = stage / WORKHORSE_SELECTED_WORLD_AUTHORITY_FILE_V1
        with authority_path.open("xb") as handle:
            handle.write(_canonical_json_bytes(projection))
            handle.flush()
            os.fsync(handle.fileno())
        authority_path.chmod(0o444)
        manifest_path = stage / WORKHORSE_SELECTED_WORLD_CHECKPOINT_FILE_V1
        with manifest_path.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        manifest_path.chmod(0o444)
        _fsync_directory(stage)
        if output.exists() or output.is_symlink():
            raise WorkhorseIndexedCheckpointError(
                "selected Workhorse checkpoint destination appeared during publication"
            )
        os.rename(stage, output)
        output.chmod(0o555)
        _fsync_directory(output.parent)
    except BaseException:
        if stage.exists() and not stage.is_symlink():
            shutil.rmtree(stage)
        raise
    final_manifest = output / WORKHORSE_SELECTED_WORLD_CHECKPOINT_FILE_V1
    final_state = output / WORKHORSE_SELECTED_WORLD_STATE_FILE_V1
    final_authority = output / WORKHORSE_SELECTED_WORLD_AUTHORITY_FILE_V1
    return {
        **manifest,
        "manifest_path": str(final_manifest.resolve()),
        "manifest_file_sha256": file_sha256(final_manifest),
        "state_path": str(final_state.resolve()),
        "state_file_sha256": file_sha256(final_state),
        "authority_receipt_path": str(final_authority.resolve()),
        "authority_receipt_file_sha256": file_sha256(final_authority),
    }


def inspect_workhorse_selected_world_checkpoint_v1(
    manifest_path: str | Path,
) -> tuple[dict[str, Any], Path]:
    """Reopen an independently retained selected Workhorse row exactly."""

    path = Path(manifest_path).expanduser().resolve()
    value = _read_json(path)
    expected_fields = {
        "schema_version",
        "time_authority",
        "selected_world",
        "source_indexed_checkpoint",
        "authority",
        "active_scientific_control_state",
        "state_payload",
        "storage_model",
        "manifest_body_sha256",
    }
    body = {key: item for key, item in value.items() if key != "manifest_body_sha256"}
    selected = value.get("selected_world")
    source = value.get("source_indexed_checkpoint")
    authority = value.get("authority")
    active_controls = value.get("active_scientific_control_state")
    payload = value.get("state_payload")
    selected_fields = {
        "world_uid",
        "rng_key",
        "checkpoint_at_tick",
        "rng_epoch",
        "shape_id",
        "state_layout_sha256",
        "semantic_transition_id",
        "semantic_transition_sha256",
        "state_bytes",
        "state_sha256",
    }
    source_fields = {
        "manifest_path",
        "manifest_file_sha256",
        "manifest_body_sha256",
        "world_state_set_sha256",
        "world_count",
        "source_checkpoint_identity_sha256",
        "source_physical_slot",
        "source_state_offset",
    }
    authority_fields = {
        "checkpoint_occurrence_sha256",
        "parent_occurrence_sha256",
        "semantic_world_sha256",
        "control_history_sha256",
        "rng_state_sha256",
        "event_prefix_sha256",
        "checkpoint_occurrence_projection",
    }
    if (
        set(value) != expected_fields
        or value.get("schema_version")
        != WORKHORSE_SELECTED_WORLD_CHECKPOINT_SCHEMA_V1
        or value.get("time_authority") != WORKHORSE_TIME_AUTHORITY_V2
        or value.get("manifest_body_sha256") != canonical_sha256(body)
        or path.name != WORKHORSE_SELECTED_WORLD_CHECKPOINT_FILE_V1
        or path.parent.stat().st_mode & 0o222
        or type(selected) is not dict
        or set(selected) != selected_fields
        or type(source) is not dict
        or set(source) != source_fields
        or type(authority) is not dict
        or set(authority) != authority_fields
        or type(payload) is not dict
        or set(payload) != {"file_name", "bytes", "sha256"}
        or value.get("storage_model")
        != {
            "selected_world_payload_copies": 1,
            "aggregate_payload_required_after_materialization": False,
            "logical_to_physical_repacking_is_nonsemantic": True,
        }
    ):
        raise WorkhorseIndexedCheckpointError(
            "selected Workhorse checkpoint schema or seal is invalid"
        )
    try:
        validated_active_controls = (
            validate_workhorse_active_scientific_control_state_v1(active_controls)
        )
    except (TypeError, WorkhorseScientificControlError) as exc:
        raise WorkhorseIndexedCheckpointError(
            f"selected Workhorse active controls changed: {exc}"
        ) from exc
    if (
        validated_active_controls["captured_next_absolute_tick"]
        != selected["checkpoint_at_tick"]
    ):
        raise WorkhorseIndexedCheckpointError(
            "selected Workhorse active controls have another captured tick"
        )
    authority_digests = {
        key: item
        for key, item in authority.items()
        if key != "checkpoint_occurrence_projection"
    }
    if any(not _valid_sha256(item) for item in authority_digests.values()) or any(
        not _valid_sha256(selected[field])
        for field in (
            "state_layout_sha256",
            "semantic_transition_sha256",
            "state_sha256",
        )
    ) or any(
        not _valid_sha256(source[field])
        for field in (
            "manifest_file_sha256",
            "manifest_body_sha256",
            "world_state_set_sha256",
            "source_checkpoint_identity_sha256",
        )
    ):
        raise WorkhorseIndexedCheckpointError(
            "selected Workhorse checkpoint contains an invalid digest"
        )
    state_path = path.parent / str(payload["file_name"])
    authority_payload = authority["checkpoint_occurrence_projection"]
    if type(authority_payload) is not dict:
        raise WorkhorseIndexedCheckpointError(
            "selected Workhorse occurrence projection binding is invalid"
        )
    authority_path = path.parent / str(authority_payload.get("file_name", ""))
    try:
        projection = _read_json(authority_path)
    except (OSError, WorkhorseIndexedCheckpointError) as exc:
        raise WorkhorseIndexedCheckpointError(
            "selected Workhorse occurrence projection changed"
        ) from exc
    projection_core = {
        key: item
        for key, item in projection.items()
        if key != "projection_content_sha256"
    }
    if (
        payload["file_name"] != WORKHORSE_SELECTED_WORLD_STATE_FILE_V1
        or state_path.is_symlink()
        or not state_path.is_file()
        or path.stat().st_mode & 0o222
        or state_path.stat().st_mode & 0o222
        or type(payload["bytes"]) is not int
        or payload["bytes"] <= 0
        or state_path.stat().st_size != payload["bytes"]
        or file_sha256(state_path) != payload["sha256"]
        or payload["sha256"] != selected["state_sha256"]
        or payload["bytes"] != selected["state_bytes"]
        or type(authority_payload) is not dict
        or set(authority_payload)
        != {"file_name", "bytes", "sha256", "projection_content_sha256"}
        or authority_payload["file_name"]
        != WORKHORSE_SELECTED_WORLD_AUTHORITY_FILE_V1
        or authority_path.is_symlink()
        or not authority_path.is_file()
        or authority_path.stat().st_mode & 0o222
        or authority_path.stat().st_size != authority_payload["bytes"]
        or file_sha256(authority_path) != authority_payload["sha256"]
        or projection.get("projection_content_sha256")
        != authority_payload["projection_content_sha256"]
        or projection.get("projection_content_sha256")
        != canonical_sha256(projection_core)
        or projection.get("checkpoint_at_tick")
        != selected["checkpoint_at_tick"]
        or projection.get("checkpoint_identity_sha256")
        != source["source_checkpoint_identity_sha256"]
        or projection.get("occurrence_digest")
        != {
            "algorithm": "sha256",
            "hex": authority["checkpoint_occurrence_sha256"],
        }
        or projection.get("parent_checkpoint_digest")
        != {
            "algorithm": "sha256",
            "hex": authority["parent_occurrence_sha256"],
        }
    ):
        raise WorkhorseIndexedCheckpointError(
            "selected Workhorse state payload changed"
        )
    state = state_path.read_bytes()
    header = struct.unpack("<32Q", state[: 32 * 8]) if len(state) >= 32 * 8 else ()
    if (
        not header
        or type(selected["world_uid"]) is not int
        or selected["world_uid"] <= 0
        or type(selected["rng_key"]) is not int
        or selected["rng_key"] < 0
        or header[0] != selected["world_uid"]
        or header[1] != selected["checkpoint_at_tick"]
        or header[4] != 0
        or header[13] != selected["rng_key"]
        or header[14] != selected["rng_epoch"]
        or source["world_count"] < 1
        or source["source_physical_slot"] < 0
        or source["source_state_offset"] < 0
    ):
        raise WorkhorseIndexedCheckpointError(
            "selected Workhorse state carrier identity changed"
        )
    return value, state_path


def workhorse_repack_binding_v2(
    indexed_checkpoint_manifest_paths: Sequence[str | Path],
    *,
    ordered_world_uids: Sequence[int],
    target_contract: FullEvolutionCudaWorkhorseContractV2,
    require_current_source_match: bool = True,
) -> dict[str, Any]:
    """Bind an exact physical gather from one or more indexed checkpoints."""

    if (
        isinstance(indexed_checkpoint_manifest_paths, (str, bytes, bytearray))
        or not indexed_checkpoint_manifest_paths
        or isinstance(ordered_world_uids, (str, bytes, bytearray))
        or not ordered_world_uids
        or type(target_contract) is not FullEvolutionCudaWorkhorseContractV2
    ):
        raise WorkhorseIndexedCheckpointError(
            "repack requires source manifests, ordered worlds, and an exact target contract"
        )
    requested_uids = tuple(ordered_world_uids)
    if (
        any(type(uid) is not int or uid <= 0 for uid in requested_uids)
        or len(set(requested_uids)) != len(requested_uids)
        or target_contract.B != len(requested_uids)
    ):
        raise WorkhorseIndexedCheckpointError(
            "repack world order must be unique, positive, and exactly cover target B"
        )
    manifests: list[tuple[dict[str, Any], Path, Path]] = []
    seen_manifest_paths: set[Path] = set()
    all_worlds: dict[int, dict[str, Any]] = {}
    common: tuple[str, str, str, int, int] | None = None
    for raw_path in indexed_checkpoint_manifest_paths:
        manifest_path = Path(raw_path).expanduser().resolve()
        if manifest_path in seen_manifest_paths:
            raise WorkhorseIndexedCheckpointError(
                "repack repeats an indexed checkpoint manifest"
            )
        seen_manifest_paths.add(manifest_path)
        raw_manifest = _read_json(manifest_path)
        selected_source = raw_manifest.get("schema_version") == (
            WORKHORSE_SELECTED_WORLD_CHECKPOINT_SCHEMA_V1
        )
        if selected_source:
            manifest, state_path = inspect_workhorse_selected_world_checkpoint_v1(
                manifest_path
            )
            selected = manifest["selected_world"]
            source = manifest["source_indexed_checkpoint"]
            identity = {
                "shape_id": selected["shape_id"],
                "state_layout_sha256": selected["state_layout_sha256"],
                "semantic_transition_sha256": selected[
                    "semantic_transition_sha256"
                ],
                "checkpoint_at_tick": selected["checkpoint_at_tick"],
                "rng_epoch": selected["rng_epoch"],
                "world_state_set_sha256": canonical_sha256(
                    {
                        "schema_version": "eiel_workhorse_selected_world_state_identity_v1",
                        "world_uid": selected["world_uid"],
                        "rng_key": selected["rng_key"],
                        "state_bytes": selected["state_bytes"],
                        "state_sha256": selected["state_sha256"],
                        "checkpoint_at_tick": selected["checkpoint_at_tick"],
                        "shape_id": selected["shape_id"],
                        "state_layout_sha256": selected["state_layout_sha256"],
                        "semantic_transition_sha256": selected[
                            "semantic_transition_sha256"
                        ],
                        "checkpoint_occurrence_sha256": manifest["authority"][
                            "checkpoint_occurrence_sha256"
                        ],
                    }
                ),
            }
            manifest_rows = [
                {
                    "world_uid": selected["world_uid"],
                    "rng_key": selected["rng_key"],
                    "source_physical_slot": 0,
                    "state_offset": 0,
                    "state_bytes": selected["state_bytes"],
                    "state_sha256": selected["state_sha256"],
                }
            ]
            source_checkpoint_identity = source[
                "source_checkpoint_identity_sha256"
            ]
            source_world_count = source["world_count"]
            source_count_group_key = source["world_state_set_sha256"]
        else:
            manifest, state_path = inspect_workhorse_indexed_checkpoint_set_v1(
                manifest_path,
                require_current_source_match=require_current_source_match,
            )
            identity = manifest["world_state_set_identity"]
            manifest_rows = manifest["world_index_by_uid"]
            source_checkpoint_identity = manifest["source_physical_checkpoint"][
                "checkpoint_identity_sha256"
            ]
            source_world_count = identity["world_count"]
            source_count_group_key = str(manifest_path)
        source_common = (
            identity["shape_id"],
            identity["state_layout_sha256"],
            identity["semantic_transition_sha256"],
            identity["checkpoint_at_tick"],
            identity["rng_epoch"],
        )
        if common is None:
            common = source_common
        elif source_common != common:
            raise WorkhorseIndexedCheckpointError(
                "repack sources differ in shape/layout/transition/tick/RNG epoch"
            )
        set_sha256 = identity["world_state_set_sha256"]
        for row in manifest_rows:
            uid = row["world_uid"]
            if uid in all_worlds:
                raise WorkhorseIndexedCheckpointError(
                    "repack sources ambiguously duplicate a world UID"
                )
            all_worlds[uid] = {
                **row,
                "source_world_state_set_sha256": set_sha256,
                "source_checkpoint_identity_sha256": source_checkpoint_identity,
                "source_index_manifest_path": str(manifest_path),
                "source_index_manifest_file_sha256": file_sha256(manifest_path),
                "source_state_path": str(state_path.resolve()),
                "source_world_count": source_world_count,
                "source_count_group_key": source_count_group_key,
            }
        manifests.append((manifest, manifest_path, state_path))
    assert common is not None
    shape_id, layout_sha256, semantic_sha256, checkpoint_at_tick, rng_epoch = common
    if (
        target_contract.shape_id != shape_id
        or workhorse_layout_v2(target_contract.shape).layout_sha256 != layout_sha256
        or target_contract.semantic_transition_sha256 != semantic_sha256
        or target_contract.start_next_tick != checkpoint_at_tick
    ):
        raise WorkhorseIndexedCheckpointError(
            "repack target differs from source shape/layout/transition/start tick"
        )
    missing = [uid for uid in requested_uids if uid not in all_worlds]
    if missing:
        raise WorkhorseIndexedCheckpointError(
            f"repack requested absent world UIDs: {missing[:8]}"
        )
    selected_rows = [all_worlds[uid] for uid in requested_uids]
    contributing_manifests = {
        row["source_index_manifest_path"] for row in selected_rows
    }
    if contributing_manifests != {str(path) for _manifest, path, _state in manifests}:
        raise WorkhorseIndexedCheckpointError(
            "every repack source manifest must contribute a selected world"
        )
    selected_identity = _world_state_set_identity_v1(
        contract=target_contract,
        checkpoint_at_tick=checkpoint_at_tick,
        layout_sha256=layout_sha256,
        world_rows=selected_rows,
    )
    source_set_sha256s = sorted(
        {row["source_world_state_set_sha256"] for row in selected_rows}
    )
    source_world_count = sum(
        {
            row["source_count_group_key"]: row["source_world_count"]
            for row in selected_rows
        }.values()
    )
    identity_rows = [
        {
            "target_physical_slot": target_slot,
            "world_uid": row["world_uid"],
            "rng_key": row["rng_key"],
            "state_bytes": row["state_bytes"],
            "state_sha256": row["state_sha256"],
            "source_world_state_set_sha256": row[
                "source_world_state_set_sha256"
            ],
        }
        for target_slot, row in enumerate(selected_rows)
    ]
    identity_body = {
        "schema_version": WORKHORSE_REPACK_BINDING_SCHEMA_V1,
        "time_authority": WORKHORSE_TIME_AUTHORITY_V2,
        "checkpoint_at_tick": checkpoint_at_tick,
        "source_world_state_set_sha256s": source_set_sha256s,
        "source_world_count": source_world_count,
        "selected_world_count": len(selected_rows),
        "continuation_scope": (
            "FULL_SET_CONTINUATION"
            if len(selected_rows) == source_world_count
            else "EXPERIMENTAL_SUBSET_FORK"
        ),
        "selected_world_state_set_identity": selected_identity,
        "target_contract": target_contract.to_dict(),
        "selected_worlds_in_physical_order": identity_rows,
    }
    locators = [
        {
            **identity_row,
            "source_checkpoint_identity_sha256": source_row[
                "source_checkpoint_identity_sha256"
            ],
            "source_index_manifest_path": source_row[
                "source_index_manifest_path"
            ],
            "source_index_manifest_file_sha256": source_row[
                "source_index_manifest_file_sha256"
            ],
            "source_state_path": source_row["source_state_path"],
            "source_state_offset": source_row["state_offset"],
            "source_physical_slot": source_row["source_physical_slot"],
        }
        for identity_row, source_row in zip(identity_rows, selected_rows, strict=True)
    ]
    return {
        **identity_body,
        "repack_binding_sha256": canonical_sha256(identity_body),
        "source_locators": locators,
    }


def validate_workhorse_repack_binding_v2(
    value: Mapping[str, Any],
    *,
    require_current_source_match: bool = True,
) -> dict[str, Any]:
    sealed = validate_workhorse_repack_binding_seal_v2(value)
    locators = sealed["source_locators"]
    target_contract = FullEvolutionCudaWorkhorseContractV2.from_mapping(
        sealed["target_contract"]
    )
    manifest_paths: list[str] = []
    for locator in locators:
        manifest_path = locator["source_index_manifest_path"]
        if manifest_path not in manifest_paths:
            manifest_paths.append(manifest_path)
    expected = workhorse_repack_binding_v2(
        manifest_paths,
        ordered_world_uids=[
            row["world_uid"]
            for row in sealed["selected_worlds_in_physical_order"]
        ],
        target_contract=target_contract,
        require_current_source_match=require_current_source_match,
    )
    if dict(value) != expected:
        raise WorkhorseIndexedCheckpointError("repack binding does not recompute")
    return expected


def validate_workhorse_repack_binding_seal_v2(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate immutable identities/locators without requiring retained bytes."""

    if type(value) is not dict:
        raise WorkhorseIndexedCheckpointError("repack binding must be an exact object")
    expected_fields = {
        "schema_version",
        "time_authority",
        "checkpoint_at_tick",
        "source_world_state_set_sha256s",
        "source_world_count",
        "selected_world_count",
        "continuation_scope",
        "selected_world_state_set_identity",
        "target_contract",
        "selected_worlds_in_physical_order",
        "repack_binding_sha256",
        "source_locators",
    }
    if set(value) != expected_fields or type(value.get("source_locators")) is not list:
        raise WorkhorseIndexedCheckpointError("repack binding fields are inexact")
    try:
        target_contract = FullEvolutionCudaWorkhorseContractV2.from_mapping(
            value["target_contract"]
        )
    except (TypeError, ValueError, FullEvolutionWorkhorseContractError) as exc:
        raise WorkhorseIndexedCheckpointError(
            f"repack target contract is invalid: {exc}"
        ) from exc
    locators = value["source_locators"]
    world_rows = value.get("selected_worlds_in_physical_order")
    identity = value.get("selected_world_state_set_identity")
    source_set_sha256s = value.get("source_world_state_set_sha256s")
    locator_fields = {
        "target_physical_slot",
        "world_uid",
        "rng_key",
        "state_bytes",
        "state_sha256",
        "source_world_state_set_sha256",
        "source_checkpoint_identity_sha256",
        "source_index_manifest_path",
        "source_index_manifest_file_sha256",
        "source_state_path",
        "source_state_offset",
        "source_physical_slot",
    }
    world_fields = {
        "target_physical_slot",
        "world_uid",
        "rng_key",
        "state_bytes",
        "state_sha256",
        "source_world_state_set_sha256",
    }
    if (
        type(world_rows) is not list
        or type(identity) is not dict
        or type(source_set_sha256s) is not list
        or any(type(row) is not dict or set(row) != world_fields for row in world_rows)
        or any(type(row) is not dict or set(row) != locator_fields for row in locators)
    ):
        raise WorkhorseIndexedCheckpointError("repack physical world order is absent")
    row_bytes = workhorse_layout_v2(target_contract.shape).state_bytes_per_world
    selected_count = value.get("selected_world_count")
    source_count = value.get("source_world_count")
    scope = value.get("continuation_scope")
    if (
        type(selected_count) is not int
        or selected_count != target_contract.B
        or len(world_rows) != selected_count
        or len(locators) != selected_count
        or type(source_count) is not int
        or source_count < selected_count
        or scope
        != (
            "FULL_SET_CONTINUATION"
            if selected_count == source_count
            else "EXPERIMENTAL_SUBSET_FORK"
        )
        or [row["target_physical_slot"] for row in world_rows]
        != list(range(selected_count))
        or [row["target_physical_slot"] for row in locators]
        != list(range(selected_count))
        or len({row["world_uid"] for row in world_rows}) != selected_count
        or any(
            {
                key: locator[key] for key in world_fields
            }
            != row
            for row, locator in zip(world_rows, locators, strict=True)
        )
        or source_set_sha256s != sorted(set(source_set_sha256s))
        or not source_set_sha256s
        or {
            row["source_world_state_set_sha256"] for row in locators
        }
        != set(source_set_sha256s)
        or any(
            type(row["world_uid"]) is not int
            or row["world_uid"] <= 0
            or type(row["rng_key"]) is not int
            or row["rng_key"] < 0
            or row["state_bytes"] != row_bytes
            or not _valid_sha256(row["state_sha256"])
            or not _valid_sha256(row["source_world_state_set_sha256"])
            or not _valid_sha256(row["source_checkpoint_identity_sha256"])
            or not _valid_sha256(row["source_index_manifest_file_sha256"])
            or type(row["source_physical_slot"]) is not int
            or row["source_physical_slot"] < 0
            or row["source_state_offset"]
            != row["source_physical_slot"] * row_bytes
            or not Path(row["source_index_manifest_path"]).is_absolute()
            or not Path(row["source_state_path"]).is_absolute()
            for row in locators
        )
    ):
        raise WorkhorseIndexedCheckpointError(
            "repack sealed world coverage or locator identity is invalid"
        )
    expected_identity = _world_state_set_identity_v1(
        contract=target_contract,
        checkpoint_at_tick=value["checkpoint_at_tick"],
        layout_sha256=workhorse_layout_v2(target_contract.shape).layout_sha256,
        world_rows=world_rows,
    )
    identity_body = {
        key: item
        for key, item in value.items()
        if key not in {"repack_binding_sha256", "source_locators"}
    }
    if (
        value["schema_version"] != WORKHORSE_REPACK_BINDING_SCHEMA_V1
        or value["time_authority"] != WORKHORSE_TIME_AUTHORITY_V2
        or value["checkpoint_at_tick"] != target_contract.start_next_tick
        or identity != expected_identity
        or value["repack_binding_sha256"] != canonical_sha256(identity_body)
    ):
        raise WorkhorseIndexedCheckpointError(
            "repack sealed state-set or physical-plan identity changed"
        )
    return dict(value)


__all__ = [
    "WORKHORSE_INDEXED_CHECKPOINT_FILE_V1",
    "WORKHORSE_INDEXED_CHECKPOINT_SET_SCHEMA_V1",
    "WORKHORSE_REPACK_BINDING_SCHEMA_V1",
    "WORKHORSE_SELECTED_WORLD_CHECKPOINT_FILE_V1",
    "WORKHORSE_SELECTED_WORLD_CHECKPOINT_SCHEMA_V1",
    "WORKHORSE_SELECTED_WORLD_AUTHORITY_FILE_V1",
    "WORKHORSE_SELECTED_WORLD_STATE_FILE_V1",
    "WORKHORSE_WORLD_STATE_SET_IDENTITY_SCHEMA_V1",
    "WorkhorseIndexedCheckpointError",
    "inspect_workhorse_indexed_checkpoint_set_v1",
    "inspect_workhorse_selected_world_checkpoint_v1",
    "materialize_workhorse_selected_world_checkpoint_v1",
    "validate_workhorse_repack_binding_v2",
    "validate_workhorse_repack_binding_seal_v2",
    "workhorse_repack_binding_v2",
    "write_workhorse_indexed_checkpoint_set_v1",
]
