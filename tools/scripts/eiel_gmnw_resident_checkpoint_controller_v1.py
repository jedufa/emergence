#!/usr/bin/env python3
"""Persist and resume exact resident GMNW owners at native boundaries."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Final, Protocol

import torch


SCHEMA_VERSION: Final[str] = "eiel_gmnw_resident_checkpoint_v1"
POINTER_SCHEMA_VERSION: Final[str] = "eiel_gmnw_resident_checkpoint_pointer_v1"
PROVENANCE_SCHEMA_VERSION: Final[str] = "eiel_gmnw_resident_checkpoint_provenance_v1"
LATEST_POINTER_NAME: Final[str] = "latest_checkpoint.json"
PREVIOUS_POINTER_NAME: Final[str] = "previous_checkpoint.json"
POINTER_LOCK_NAME: Final[str] = ".checkpoint-pointer.lock"
SHA256_LENGTH: Final[int] = 64


class ResidentCheckpointError(RuntimeError):
    """A durable checkpoint boundary is incomplete or contradictory."""


class ResidentCheckpointRuntimeV1(Protocol):
    """Native segment boundary consumed by the durable controller."""

    def prepare_fresh(
        self, cell_ids: Sequence[str], *, run_root: Path, generations: int
    ) -> Any: ...

    def execute_segment(
        self,
        prepared: Any,
        *,
        start_generation: int,
        completed_generations: int,
        terminal_generation: int,
        run_id: str,
        run_control: Any | None = None,
    ) -> Mapping[str, Any]: ...

    def project_owner(
        self,
        prepared: Any,
        result: Mapping[str, Any],
        *,
        completed_generations: int,
        cell_ids: Sequence[str],
    ) -> Sequence[Mapping[str, Any]]: ...

    def export_checkpoint(
        self, prepared: Any, *, absolute_generation: int, cell_ids: Sequence[str]
    ) -> Mapping[str, Any]: ...

    def validate_checkpoint(
        self,
        checkpoint: Mapping[str, Any],
        *,
        absolute_generation: int,
        cell_ids: Sequence[str],
    ) -> Mapping[str, Any]: ...

    def acknowledge_completed_checkpoint(
        self,
        prepared: Any,
        result: Mapping[str, Any],
        *,
        checkpoint_path: Path,
        checkpoint_sha256: str,
    ) -> Mapping[str, Any]: ...

    def prepare_continuation(
        self,
        cell_ids: Sequence[str],
        *,
        run_root: Path,
        owner_projections: Sequence[Mapping[str, Any]],
        initial_generation: int,
        generations: int,
    ) -> Any: ...

    def restore_checkpoint(
        self,
        prepared: Any,
        checkpoint: Mapping[str, Any],
        *,
        absolute_generation: int,
        cell_ids: Sequence[str],
    ) -> int: ...

    def boundary_state_digests(
        self, prepared: Any, *, cell_ids: Sequence[str]
    ) -> Sequence[str]: ...

    def release(self, *owners: Any) -> None: ...


@dataclass(frozen=True, slots=True)
class DurableCheckpointReceiptV1:
    path: Path
    sha256: str
    bytes: int
    run_id: str
    absolute_generation: int
    schedule_sha256: str
    boundary_state_digests: tuple[str, ...]


@dataclass(slots=True)
class CheckpointedExecutionV1:
    prepared: Any
    segment_results: tuple[Mapping[str, Any], ...]
    checkpoint_receipts: tuple[DurableCheckpointReceiptV1, ...]
    latest_pointer: Mapping[str, Any]


def _fail(message: str) -> ResidentCheckpointError:
    return ResidentCheckpointError(message)


def _canonical_json_sha256(value: Any) -> str:
    try:
        payload = json.dumps(
            value,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise _fail(f"checkpoint metadata is not canonical ASCII JSON: {exc}") from exc
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_sha256(value: Any, *, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != SHA256_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise _fail(f"{label} is not a lowercase SHA-256")
    return value


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _ensure_directory_durable(path: Path) -> None:
    resolved = path.expanduser().resolve()
    missing: list[Path] = []
    cursor = resolved
    while not cursor.exists():
        missing.append(cursor)
        cursor = cursor.parent
    if not cursor.is_dir():
        raise _fail(f"checkpoint parent is not a directory: {cursor}")
    for directory in reversed(missing):
        try:
            directory.mkdir()
        except FileExistsError:
            if not directory.is_dir():
                raise
        _fsync_directory(directory.parent)


def _available_restore_run_root_v1(run_root: Path) -> Path:
    """Return a fresh runner directory after an interrupted restore attempt."""

    base = run_root.expanduser().resolve()
    if not base.exists():
        return base
    for attempt in range(2, 10_001):
        candidate = base.with_name(f"{base.name}-attempt-{attempt:04d}")
        if not candidate.exists():
            return candidate
    raise _fail(f"restore attempt directories are exhausted for {base}")


def _validate_provenance(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise _fail("checkpoint provenance must be a mapping")
    required_hashes = {
        "extension_sha256",
        "configuration_sha256",
        "checkpoint_contract_sha256",
    }
    if value.get(
        "schema_version"
    ) != PROVENANCE_SCHEMA_VERSION or not required_hashes.issubset(value):
        raise _fail("checkpoint provenance schema is incomplete")
    for name in required_hashes:
        _require_sha256(value.get(name), label=f"checkpoint provenance {name}")
    normalized = dict(value)
    _canonical_json_sha256(normalized)
    return normalized


def _write_torch_immutable(path: Path, value: Any) -> None:
    _ensure_directory_durable(path.parent)
    if path.exists():
        raise FileExistsError(path)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            torch.save(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        temporary.unlink()
        _fsync_directory(path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    _ensure_directory_durable(path.parent)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="ascii") as stream:
            json.dump(value, stream, allow_nan=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def validate_checkpoint_schedule_v1(
    values: Sequence[int], *, terminal_generation: int
) -> tuple[int, ...]:
    if type(terminal_generation) is not int or terminal_generation < 1:
        raise _fail("terminal_generation must be a positive exact integer")
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise _fail("checkpoint schedule must be a sequence of exact integers")
    schedule = tuple(values)
    if not schedule:
        raise _fail("checkpoint schedule must not be empty")
    if any(type(value) is not int for value in schedule):
        raise _fail("checkpoint schedule values must be exact integers")
    if any(value < 1 or value > terminal_generation for value in schedule):
        raise _fail("checkpoint schedule values are outside the terminal horizon")
    if any(left >= right for left, right in zip(schedule, schedule[1:], strict=False)):
        raise _fail("checkpoint schedule must be strictly increasing and unique")
    if schedule[-1] != terminal_generation:
        raise _fail("checkpoint schedule must end at terminal_generation")
    return schedule


def checkpoint_schedule_sha256_v1(
    schedule: Sequence[int], *, terminal_generation: int
) -> str:
    validated = validate_checkpoint_schedule_v1(
        schedule, terminal_generation=terminal_generation
    )
    return _canonical_json_sha256(
        {
            "schema_version": SCHEMA_VERSION,
            "terminal_generation": terminal_generation,
            "checkpoint_generations": list(validated),
        }
    )


def _validate_owner_projections(
    value: Any, *, cell_ids: Sequence[str], absolute_generation: int
) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, (list, tuple)) or len(value) != len(cell_ids):
        raise _fail("owner projection count does not match checkpoint cells")
    projections: list[Mapping[str, Any]] = []
    for index, projection in enumerate(value):
        if not isinstance(projection, Mapping):
            raise _fail(f"owner projection {index} is not a mapping")
        metadata = projection.get("metadata")
        tensors = projection.get("tensors")
        if (
            not isinstance(metadata, Mapping)
            or metadata.get("generation") != absolute_generation
            or metadata.get("tick") != 0
            or not isinstance(tensors, Mapping)
            or not tensors
            or any(not isinstance(tensor, torch.Tensor) for tensor in tensors.values())
        ):
            raise _fail(f"owner projection {index} is not an exact generation boundary")
        projections.append(projection)
    return tuple(projections)


def _validate_boundary_digests(
    value: Any, *, cell_ids: Sequence[str]
) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != len(cell_ids):
        raise _fail("boundary digest count does not match checkpoint cells")
    return tuple(
        _require_sha256(digest, label=f"boundary digest {index}")
        for index, digest in enumerate(value)
    )


def _validate_cell_ids(cell_ids: Sequence[str]) -> tuple[str, ...]:
    if isinstance(cell_ids, (str, bytes)) or not isinstance(cell_ids, Sequence):
        raise _fail("checkpoint cell IDs must be a sequence")
    validated = tuple(cell_ids)
    if (
        not validated
        or any(type(cell_id) is not str or not cell_id for cell_id in validated)
        or len(set(validated)) != len(validated)
    ):
        raise _fail("checkpoint cell IDs must be nonempty unique strings")
    return validated


def persist_owner_checkpoint_v1(
    runtime: ResidentCheckpointRuntimeV1,
    prepared: Any,
    result: Mapping[str, Any],
    *,
    checkpoint_root: Path,
    run_id: str,
    cell_ids: Sequence[str],
    completed_segment_generations: int,
    absolute_generation: int,
    schedule: Sequence[int],
    terminal_generation: int,
    provenance: Mapping[str, Any],
) -> DurableCheckpointReceiptV1:
    cell_ids = _validate_cell_ids(cell_ids)
    provenance = _validate_provenance(provenance)
    validated_schedule = validate_checkpoint_schedule_v1(
        schedule, terminal_generation=terminal_generation
    )
    if absolute_generation not in validated_schedule:
        raise _fail("checkpoint generation is not in the preregistered schedule")
    if (
        type(completed_segment_generations) is not int
        or completed_segment_generations < 1
    ):
        raise _fail("completed segment length must be a positive exact integer")
    schedule_sha256 = checkpoint_schedule_sha256_v1(
        validated_schedule, terminal_generation=terminal_generation
    )
    provenance_sha256 = _canonical_json_sha256(provenance)
    projections = tuple(
        runtime.project_owner(
            prepared,
            result,
            completed_generations=completed_segment_generations,
            cell_ids=cell_ids,
        )
    )
    _validate_owner_projections(
        projections, cell_ids=cell_ids, absolute_generation=absolute_generation
    )
    gmnw_checkpoint = runtime.export_checkpoint(
        prepared,
        absolute_generation=absolute_generation,
        cell_ids=cell_ids,
    )
    boundary_digests = _validate_boundary_digests(
        runtime.boundary_state_digests(prepared, cell_ids=cell_ids),
        cell_ids=cell_ids,
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "absolute_generation": absolute_generation,
        "terminal_generation": terminal_generation,
        "checkpoint_generations": list(validated_schedule),
        "schedule_sha256": schedule_sha256,
        "cell_ids": list(cell_ids),
        "provenance": dict(provenance),
        "provenance_sha256": provenance_sha256,
        "boundary_state_digests": list(boundary_digests),
        "owner_projection_cell_ids": list(cell_ids),
        "owner_projections": list(projections),
        "gmnw_multicell_checkpoint": dict(gmnw_checkpoint),
    }
    target = checkpoint_root.expanduser().resolve() / (
        f"g{absolute_generation:06d}-owner.pt"
    )
    if target.exists():
        # A crash may leave a fully durable next checkpoint before the latest
        # pointer advances. Reuse it only when the published pointer is older
        # and the orphan validates against the exact state being checkpointed.
        pointer_path = target.parent / LATEST_POINTER_NAME
        previous_pointer_path = target.parent / PREVIOUS_POINTER_NAME
        if pointer_path.is_file() or previous_pointer_path.is_file():
            pointer, _published_receipt = load_latest_checkpoint_v1(
                target.parent,
                run_id=run_id,
                cell_ids=cell_ids,
                schedule=validated_schedule,
                terminal_generation=terminal_generation,
            )
            if pointer["absolute_generation"] >= absolute_generation:
                raise FileExistsError(target)
        elif absolute_generation != validated_schedule[0]:
            raise FileExistsError(target)
        candidate = DurableCheckpointReceiptV1(
            path=target,
            sha256=_file_sha256(target),
            bytes=target.stat().st_size,
            run_id=run_id,
            absolute_generation=absolute_generation,
            schedule_sha256=schedule_sha256,
            boundary_state_digests=boundary_digests,
        )
        reload_validate_checkpoint_v1(
            runtime,
            candidate,
            cell_ids=cell_ids,
            schedule=validated_schedule,
            terminal_generation=terminal_generation,
            provenance=provenance,
        )
        return candidate
    _write_torch_immutable(target, payload)
    return DurableCheckpointReceiptV1(
        path=target,
        sha256=_file_sha256(target),
        bytes=target.stat().st_size,
        run_id=run_id,
        absolute_generation=absolute_generation,
        schedule_sha256=schedule_sha256,
        boundary_state_digests=boundary_digests,
    )


def reload_validate_checkpoint_v1(
    runtime: ResidentCheckpointRuntimeV1,
    receipt: DurableCheckpointReceiptV1,
    *,
    cell_ids: Sequence[str],
    schedule: Sequence[int],
    terminal_generation: int,
    provenance: Mapping[str, Any],
) -> Mapping[str, Any]:
    cell_ids = _validate_cell_ids(cell_ids)
    if not receipt.path.is_file() or _file_sha256(receipt.path) != receipt.sha256:
        raise _fail("durable checkpoint bytes do not match their receipt")
    try:
        payload = torch.load(receipt.path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise _fail(f"durable checkpoint cannot be reloaded: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise _fail("durable checkpoint payload is not a mapping")
    expected_fields = {
        "schema_version",
        "run_id",
        "absolute_generation",
        "terminal_generation",
        "checkpoint_generations",
        "schedule_sha256",
        "cell_ids",
        "provenance",
        "provenance_sha256",
        "boundary_state_digests",
        "owner_projection_cell_ids",
        "owner_projections",
        "gmnw_multicell_checkpoint",
    }
    if (
        set(payload) != expected_fields
        or payload.get("schema_version") != SCHEMA_VERSION
    ):
        raise _fail("durable checkpoint schema mismatch")
    validated_schedule = validate_checkpoint_schedule_v1(
        schedule, terminal_generation=terminal_generation
    )
    provenance = _validate_provenance(provenance)
    schedule_sha256 = checkpoint_schedule_sha256_v1(
        validated_schedule, terminal_generation=terminal_generation
    )
    if (
        payload.get("absolute_generation") != receipt.absolute_generation
        or payload.get("run_id") != receipt.run_id
        or payload.get("terminal_generation") != terminal_generation
        or payload.get("checkpoint_generations") != list(validated_schedule)
        or payload.get("schedule_sha256") != schedule_sha256
        or receipt.schedule_sha256 != schedule_sha256
        or payload.get("cell_ids") != list(cell_ids)
        or payload.get("owner_projection_cell_ids") != list(cell_ids)
        or payload.get("provenance") != dict(provenance)
        or payload.get("provenance_sha256") != _canonical_json_sha256(provenance)
    ):
        raise _fail("durable checkpoint identity or provenance mismatch")
    _validate_owner_projections(
        payload.get("owner_projections"),
        cell_ids=cell_ids,
        absolute_generation=receipt.absolute_generation,
    )
    digests = _validate_boundary_digests(
        payload.get("boundary_state_digests"), cell_ids=cell_ids
    )
    if digests != receipt.boundary_state_digests:
        raise _fail("durable checkpoint boundary digest receipt mismatch")
    checkpoint = payload.get("gmnw_multicell_checkpoint")
    if not isinstance(checkpoint, Mapping):
        raise _fail("durable checkpoint omitted the GMNW sidecar")
    runtime.validate_checkpoint(
        checkpoint,
        absolute_generation=receipt.absolute_generation,
        cell_ids=cell_ids,
    )
    return payload


def acknowledge_completed_checkpoint_v1(
    runtime: ResidentCheckpointRuntimeV1,
    prepared: Any,
    result: Mapping[str, Any],
    receipt: DurableCheckpointReceiptV1,
) -> Mapping[str, Any]:
    acknowledgement = runtime.acknowledge_completed_checkpoint(
        prepared,
        result,
        checkpoint_path=receipt.path,
        checkpoint_sha256=receipt.sha256,
    )
    if not isinstance(acknowledgement, Mapping):
        raise _fail("native checkpoint acknowledgement is not a mapping")
    return acknowledgement


def restore_fresh_owner_v1(
    runtime: ResidentCheckpointRuntimeV1,
    payload: Mapping[str, Any],
    *,
    run_root: Path,
    cell_ids: Sequence[str],
    terminal_generation: int,
) -> Any:
    cell_ids = _validate_cell_ids(cell_ids)
    absolute_generation = payload["absolute_generation"]
    owner_projections = payload["owner_projections"]
    prepared = runtime.prepare_continuation(
        cell_ids,
        run_root=_available_restore_run_root_v1(run_root),
        owner_projections=owner_projections,
        initial_generation=absolute_generation,
        generations=terminal_generation,
    )
    try:
        restored_generation = runtime.restore_checkpoint(
            prepared,
            payload["gmnw_multicell_checkpoint"],
            absolute_generation=absolute_generation,
            cell_ids=cell_ids,
        )
        if restored_generation != absolute_generation:
            raise _fail("fresh owner restored the wrong absolute generation")
        actual_digests = _validate_boundary_digests(
            runtime.boundary_state_digests(prepared, cell_ids=cell_ids),
            cell_ids=cell_ids,
        )
        expected_digests = tuple(payload["boundary_state_digests"])
        if actual_digests != expected_digests:
            raise _fail("fresh owner boundary state differs from durable checkpoint")
    except BaseException:
        runtime.release(prepared)
        raise
    return prepared


def commit_latest_checkpoint_v1(
    checkpoint_root: Path,
    receipt: DurableCheckpointReceiptV1,
    *,
    run_id: str,
    cell_ids: Sequence[str],
    schedule: Sequence[int],
    terminal_generation: int,
    expected_previous_generation: int,
    retain_all_checkpoints: bool = False,
) -> Mapping[str, Any]:
    cell_ids = _validate_cell_ids(cell_ids)
    validated_schedule = validate_checkpoint_schedule_v1(
        schedule, terminal_generation=terminal_generation
    )
    if type(retain_all_checkpoints) is not bool:
        raise _fail("retain_all_checkpoints must be an exact boolean")
    root = checkpoint_root.expanduser().resolve()
    _ensure_directory_durable(root)
    try:
        relative_path = receipt.path.relative_to(root)
    except ValueError as exc:
        raise _fail("checkpoint path is outside its publication root") from exc
    if (
        receipt.run_id != run_id
        or receipt.absolute_generation not in validated_schedule
        or receipt.schedule_sha256
        != checkpoint_schedule_sha256_v1(
            validated_schedule, terminal_generation=terminal_generation
        )
        or type(expected_previous_generation) is not int
        or expected_previous_generation < 0
        or receipt.absolute_generation <= expected_previous_generation
        or not receipt.path.is_file()
        or receipt.path.stat().st_size != receipt.bytes
        or _file_sha256(receipt.path) != receipt.sha256
    ):
        raise _fail("checkpoint receipt is not publishable at the requested frontier")
    pointer = {
        "schema_version": POINTER_SCHEMA_VERSION,
        "run_id": run_id,
        "absolute_generation": receipt.absolute_generation,
        "terminal_generation": terminal_generation,
        "cell_ids": list(cell_ids),
        "checkpoint_path": relative_path.as_posix(),
        "checkpoint_sha256": receipt.sha256,
        "checkpoint_bytes": receipt.bytes,
        "schedule_sha256": receipt.schedule_sha256,
        "boundary_state_digests": list(receipt.boundary_state_digests),
    }
    lock_path = root / POINTER_LOCK_NAME
    lock_was_absent = not lock_path.exists()
    with lock_path.open("a+b") as lock_stream:
        if lock_was_absent:
            _fsync_directory(root)
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX)
        try:
            current: Mapping[str, Any] | None = None
            has_pointer = (root / LATEST_POINTER_NAME).exists() or (
                root / PREVIOUS_POINTER_NAME
            ).exists()
            if has_pointer:
                current, _current_receipt = _load_latest_checkpoint_unlocked_v1(
                    root,
                    run_id=run_id,
                    cell_ids=cell_ids,
                    schedule=validated_schedule,
                    terminal_generation=terminal_generation,
                )
            current_generation = (
                0 if current is None else current["absolute_generation"]
            )
            if current_generation != expected_previous_generation:
                raise _fail("checkpoint pointer predecessor changed before publication")
            if current is not None:
                _write_json_atomic(root / PREVIOUS_POINTER_NAME, current)
            _write_json_atomic(root / LATEST_POINTER_NAME, pointer)

            if not retain_all_checkpoints:
                retained = {receipt.path.resolve()}
                if current is not None:
                    retained.add((root / current["checkpoint_path"]).resolve())
                removed = False
                for candidate in root.glob("g*-owner.pt"):
                    if candidate.resolve() not in retained:
                        candidate.unlink()
                        removed = True
                if removed:
                    _fsync_directory(root)
            return pointer
        finally:
            fcntl.flock(lock_stream.fileno(), fcntl.LOCK_UN)


def _load_checkpoint_pointer_v1(
    pointer_path: Path,
    *,
    root: Path,
    run_id: str,
    cell_ids: tuple[str, ...],
    schedule: tuple[int, ...],
    terminal_generation: int,
) -> tuple[Mapping[str, Any], DurableCheckpointReceiptV1]:
    label = pointer_path.name
    try:
        pointer = json.loads(pointer_path.read_text(encoding="ascii"))
    except (OSError, json.JSONDecodeError, UnicodeError) as exc:
        raise _fail(f"{label} cannot be loaded: {exc}") from exc
    if not isinstance(pointer, Mapping) or set(pointer) != {
        "schema_version",
        "run_id",
        "absolute_generation",
        "terminal_generation",
        "cell_ids",
        "checkpoint_path",
        "checkpoint_sha256",
        "checkpoint_bytes",
        "schedule_sha256",
        "boundary_state_digests",
    }:
        raise _fail(f"{label} schema mismatch")
    absolute_generation = pointer.get("absolute_generation")
    if (
        pointer.get("schema_version") != POINTER_SCHEMA_VERSION
        or pointer.get("run_id") != run_id
        or type(absolute_generation) is not int
        or absolute_generation not in schedule
        or pointer.get("terminal_generation") != terminal_generation
        or pointer.get("cell_ids") != list(cell_ids)
        or pointer.get("schedule_sha256")
        != checkpoint_schedule_sha256_v1(
            schedule, terminal_generation=terminal_generation
        )
    ):
        raise _fail(f"{label} identity mismatch")
    relative = pointer.get("checkpoint_path")
    if (
        type(relative) is not str
        or Path(relative).is_absolute()
        or ".." in Path(relative).parts
    ):
        raise _fail(f"{label} path is unsafe")
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise _fail(f"{label} escapes its root") from exc
    sha256 = _require_sha256(pointer.get("checkpoint_sha256"), label=f"{label} digest")
    size = pointer.get("checkpoint_bytes")
    if (
        type(size) is not int
        or size < 1
        or not path.is_file()
        or path.stat().st_size != size
        or _file_sha256(path) != sha256
    ):
        raise _fail(f"{label} size or path mismatch")
    digests = _validate_boundary_digests(
        pointer.get("boundary_state_digests"), cell_ids=cell_ids
    )
    return pointer, DurableCheckpointReceiptV1(
        path=path,
        sha256=sha256,
        bytes=size,
        run_id=run_id,
        absolute_generation=absolute_generation,
        schedule_sha256=pointer["schedule_sha256"],
        boundary_state_digests=digests,
    )


def _load_latest_checkpoint_unlocked_v1(
    checkpoint_root: Path,
    *,
    run_id: str,
    cell_ids: Sequence[str],
    schedule: Sequence[int],
    terminal_generation: int,
) -> tuple[Mapping[str, Any], DurableCheckpointReceiptV1]:
    cell_ids = _validate_cell_ids(cell_ids)
    root = checkpoint_root.expanduser().resolve()
    validated_schedule = validate_checkpoint_schedule_v1(
        schedule, terminal_generation=terminal_generation
    )
    failures: list[str] = []
    for name in (LATEST_POINTER_NAME, PREVIOUS_POINTER_NAME):
        pointer_path = root / name
        if not pointer_path.is_file():
            failures.append(f"{name}: missing")
            continue
        try:
            return _load_checkpoint_pointer_v1(
                pointer_path,
                root=root,
                run_id=run_id,
                cell_ids=cell_ids,
                schedule=validated_schedule,
                terminal_generation=terminal_generation,
            )
        except ResidentCheckpointError as exc:
            failures.append(f"{name}: {exc}")
    raise _fail("no valid checkpoint pointer: " + "; ".join(failures))


def load_latest_checkpoint_v1(
    checkpoint_root: Path,
    *,
    run_id: str,
    cell_ids: Sequence[str],
    schedule: Sequence[int],
    terminal_generation: int,
) -> tuple[Mapping[str, Any], DurableCheckpointReceiptV1]:
    root = checkpoint_root.expanduser().resolve()
    _ensure_directory_durable(root)
    lock_path = root / POINTER_LOCK_NAME
    lock_was_absent = not lock_path.exists()
    with lock_path.open("a+b") as lock_stream:
        if lock_was_absent:
            _fsync_directory(root)
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_SH)
        try:
            return _load_latest_checkpoint_unlocked_v1(
                root,
                run_id=run_id,
                cell_ids=cell_ids,
                schedule=schedule,
                terminal_generation=terminal_generation,
            )
        finally:
            fcntl.flock(lock_stream.fileno(), fcntl.LOCK_UN)


def execute_native_segment_v1(
    runtime: ResidentCheckpointRuntimeV1,
    prepared: Any,
    *,
    start_generation: int,
    target_generation: int,
    terminal_generation: int,
    run_id: str,
    run_control: Any | None = None,
) -> Mapping[str, Any]:
    if (
        type(start_generation) is not int
        or type(target_generation) is not int
        or start_generation < 0
        or target_generation <= start_generation
        or target_generation > terminal_generation
    ):
        raise _fail("native segment boundaries are invalid")
    completed = target_generation - start_generation
    execute_kwargs = {
        "start_generation": start_generation,
        "completed_generations": completed,
        "terminal_generation": terminal_generation,
        "run_id": run_id,
    }
    if run_control is not None:
        execute_kwargs["run_control"] = run_control
    result = runtime.execute_segment(prepared, **execute_kwargs)
    if (
        not isinstance(result, Mapping)
        or result.get("completed_generations") != completed
    ):
        raise _fail("native segment did not complete its exact requested range")
    return result


def _execute_from_prepared_v1(
    runtime: ResidentCheckpointRuntimeV1,
    prepared: Any,
    *,
    root: Path,
    checkpoint_root: Path,
    run_id: str,
    cell_ids: tuple[str, ...],
    provenance: Mapping[str, Any],
    schedule: tuple[int, ...],
    terminal_generation: int,
    start_generation: int,
    initial_pointer: Mapping[str, Any] | None = None,
    initial_receipts: Sequence[DurableCheckpointReceiptV1] = (),
    retain_all_checkpoints: bool = False,
) -> CheckpointedExecutionV1:
    remaining_schedule = tuple(
        generation for generation in schedule if generation > start_generation
    )
    segment_results: list[Mapping[str, Any]] = []
    receipts = list(initial_receipts)
    latest_pointer = initial_pointer
    try:
        for target_generation in remaining_schedule:
            result = execute_native_segment_v1(
                runtime,
                prepared,
                start_generation=start_generation,
                target_generation=target_generation,
                terminal_generation=terminal_generation,
                run_id=f"{run_id}-g{start_generation:06d}-g{target_generation:06d}",
            )
            receipt = persist_owner_checkpoint_v1(
                runtime,
                prepared,
                result,
                checkpoint_root=checkpoint_root,
                run_id=run_id,
                cell_ids=cell_ids,
                completed_segment_generations=target_generation - start_generation,
                absolute_generation=target_generation,
                schedule=schedule,
                terminal_generation=terminal_generation,
                provenance=provenance,
            )
            payload = reload_validate_checkpoint_v1(
                runtime,
                receipt,
                cell_ids=cell_ids,
                schedule=schedule,
                terminal_generation=terminal_generation,
                provenance=provenance,
            )

            # Prove the durable bytes can reconstruct the exact boundary before
            # sealing and releasing the live owner that produced them.
            restored = restore_fresh_owner_v1(
                runtime,
                payload,
                run_root=root / f"segment-g{target_generation:06d}",
                cell_ids=cell_ids,
                terminal_generation=terminal_generation,
            )
            try:
                latest_pointer = commit_latest_checkpoint_v1(
                    checkpoint_root,
                    receipt,
                    run_id=run_id,
                    cell_ids=cell_ids,
                    schedule=schedule,
                    terminal_generation=terminal_generation,
                    expected_previous_generation=start_generation,
                    retain_all_checkpoints=retain_all_checkpoints,
                )
                acknowledgement = acknowledge_completed_checkpoint_v1(
                    runtime, prepared, result, receipt
                )
                if isinstance(result, dict):
                    result["durable_checkpoint_acknowledgement"] = dict(acknowledgement)
            except BaseException:
                runtime.release(restored)
                raise
            previous = prepared
            prepared = restored
            runtime.release(previous)
            segment_results.append(result)
            receipts.append(receipt)
            start_generation = target_generation
    except BaseException:
        runtime.release(prepared)
        raise
    if latest_pointer is None:
        runtime.release(prepared)
        raise AssertionError("checkpointed execution produced no latest pointer")
    return CheckpointedExecutionV1(
        prepared=prepared,
        segment_results=tuple(segment_results),
        checkpoint_receipts=tuple(receipts),
        latest_pointer=latest_pointer,
    )


def execute_checkpointed_ub100_v1(
    runtime: ResidentCheckpointRuntimeV1,
    *,
    run_root: Path,
    run_id: str,
    cell_ids: Sequence[str],
    provenance: Mapping[str, Any],
    checkpoint_generations: Sequence[int] = (25, 50, 75, 100),
    terminal_generation: int = 100,
    retain_all_checkpoints: bool = False,
) -> CheckpointedExecutionV1:
    cell_ids = _validate_cell_ids(cell_ids)
    schedule = validate_checkpoint_schedule_v1(
        checkpoint_generations, terminal_generation=terminal_generation
    )
    root = run_root.expanduser().resolve()
    checkpoint_root = root / "checkpoints"
    prepared = runtime.prepare_fresh(
        cell_ids, run_root=root / "segment-g000000", generations=terminal_generation
    )
    return _execute_from_prepared_v1(
        runtime,
        prepared,
        root=root,
        checkpoint_root=checkpoint_root,
        run_id=run_id,
        cell_ids=cell_ids,
        provenance=provenance,
        schedule=schedule,
        terminal_generation=terminal_generation,
        start_generation=0,
        retain_all_checkpoints=retain_all_checkpoints,
    )


def resume_checkpointed_ub100_v1(
    runtime: ResidentCheckpointRuntimeV1,
    *,
    run_root: Path,
    run_id: str,
    cell_ids: Sequence[str],
    provenance: Mapping[str, Any],
    checkpoint_generations: Sequence[int] = (25, 50, 75, 100),
    terminal_generation: int = 100,
    retain_all_checkpoints: bool = False,
) -> CheckpointedExecutionV1:
    cell_ids = _validate_cell_ids(cell_ids)
    schedule = validate_checkpoint_schedule_v1(
        checkpoint_generations, terminal_generation=terminal_generation
    )
    root = run_root.expanduser().resolve()
    checkpoint_root = root / "checkpoints"
    pointer, receipt = load_latest_checkpoint_v1(
        checkpoint_root,
        run_id=run_id,
        cell_ids=cell_ids,
        schedule=schedule,
        terminal_generation=terminal_generation,
    )
    payload = reload_validate_checkpoint_v1(
        runtime,
        receipt,
        cell_ids=cell_ids,
        schedule=schedule,
        terminal_generation=terminal_generation,
        provenance=provenance,
    )
    prepared = restore_fresh_owner_v1(
        runtime,
        payload,
        run_root=root / f"resume-g{receipt.absolute_generation:06d}",
        cell_ids=cell_ids,
        terminal_generation=terminal_generation,
    )
    return _execute_from_prepared_v1(
        runtime,
        prepared,
        root=root,
        checkpoint_root=checkpoint_root,
        run_id=run_id,
        cell_ids=cell_ids,
        provenance=provenance,
        schedule=schedule,
        terminal_generation=terminal_generation,
        start_generation=receipt.absolute_generation,
        initial_pointer=pointer,
        initial_receipts=(receipt,),
        retain_all_checkpoints=retain_all_checkpoints,
    )


__all__ = [
    "CheckpointedExecutionV1",
    "DurableCheckpointReceiptV1",
    "LATEST_POINTER_NAME",
    "POINTER_SCHEMA_VERSION",
    "PREVIOUS_POINTER_NAME",
    "PROVENANCE_SCHEMA_VERSION",
    "ResidentCheckpointError",
    "ResidentCheckpointRuntimeV1",
    "SCHEMA_VERSION",
    "acknowledge_completed_checkpoint_v1",
    "checkpoint_schedule_sha256_v1",
    "commit_latest_checkpoint_v1",
    "execute_checkpointed_ub100_v1",
    "execute_native_segment_v1",
    "load_latest_checkpoint_v1",
    "persist_owner_checkpoint_v1",
    "reload_validate_checkpoint_v1",
    "resume_checkpointed_ub100_v1",
    "restore_fresh_owner_v1",
    "validate_checkpoint_schedule_v1",
]
