"""Canonical, hash-bound, failure-atomic Option D checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

from .contracts import (
    OPTION_D_CHECKPOINT_SCHEMA_V1,
    OptionDContractError,
    RNGStateBundle,
    canonical_json_bytes,
)


@dataclass(frozen=True, slots=True)
class CheckpointSnapshot:
    """Owner-thread immutable bytes detached from later persistence timing."""

    data: bytes
    sha256: str

    def validate(self) -> None:
        if not isinstance(self.data, bytes) or not self.data:
            raise OptionDContractError("checkpoint snapshot data must be nonempty bytes")
        if self.sha256 != sha256(self.data).hexdigest():
            raise OptionDContractError("checkpoint snapshot digest mismatch")


def _checkpoint_payload(
    bundle: RNGStateBundle,
    *,
    artifact_binding: str,
    schedule_sha256: str | None,
) -> dict[str, Any]:
    bundle.validate()
    if not isinstance(artifact_binding, str) or not artifact_binding.strip():
        raise OptionDContractError("artifact_binding must be a nonempty string")
    if schedule_sha256 is not None and (
        not isinstance(schedule_sha256, str)
        or len(schedule_sha256) != 64
        or any(character not in "0123456789abcdef" for character in schedule_sha256)
    ):
        raise OptionDContractError("schedule_sha256 must be a lowercase SHA-256 or null")
    return {
        "artifact_binding": artifact_binding,
        "schedule_sha256": schedule_sha256,
        "source_binding": bundle.source_binding,
        "state_bundle": bundle.to_dict(),
    }


def checkpoint_bytes(
    bundle: RNGStateBundle,
    *,
    artifact_binding: str,
    schedule_sha256: str | None = None,
) -> bytes:
    payload = _checkpoint_payload(
        bundle,
        artifact_binding=artifact_binding,
        schedule_sha256=schedule_sha256,
    )
    envelope = {
        "payload": payload,
        "payload_sha256": sha256(canonical_json_bytes(payload)).hexdigest(),
        "schema_version": OPTION_D_CHECKPOINT_SCHEMA_V1,
    }
    return canonical_json_bytes(envelope) + b"\n"


def capture_checkpoint(
    bundle: RNGStateBundle,
    *,
    artifact_binding: str,
    schedule_sha256: str | None = None,
) -> CheckpointSnapshot:
    """Capture canonical bytes synchronously at the authoritative boundary."""

    data = checkpoint_bytes(
        bundle,
        artifact_binding=artifact_binding,
        schedule_sha256=schedule_sha256,
    )
    snapshot = CheckpointSnapshot(data=data, sha256=sha256(data).hexdigest())
    snapshot.validate()
    return snapshot


def restore_checkpoint(
    data: bytes,
    *,
    expected_source_binding: str,
    expected_artifact_binding: str,
    expected_schedule_sha256: str | None = None,
) -> RNGStateBundle:
    if not isinstance(data, bytes) or not data:
        raise OptionDContractError("checkpoint must be nonempty bytes")
    try:
        envelope = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise OptionDContractError("checkpoint is not canonical UTF-8 JSON") from error
    if not isinstance(envelope, Mapping) or set(envelope) != {
        "payload",
        "payload_sha256",
        "schema_version",
    }:
        raise OptionDContractError("checkpoint envelope fields are invalid")
    if envelope["schema_version"] != OPTION_D_CHECKPOINT_SCHEMA_V1:
        raise OptionDContractError("checkpoint schema binding mismatch")
    payload = envelope["payload"]
    if not isinstance(payload, Mapping) or set(payload) != {
        "artifact_binding",
        "schedule_sha256",
        "source_binding",
        "state_bundle",
    }:
        raise OptionDContractError("checkpoint payload fields are invalid")
    actual_digest = sha256(canonical_json_bytes(payload)).hexdigest()
    if envelope["payload_sha256"] != actual_digest:
        raise OptionDContractError("checkpoint payload digest mismatch")
    if payload["source_binding"] != expected_source_binding:
        raise OptionDContractError("checkpoint source binding mismatch")
    if payload["artifact_binding"] != expected_artifact_binding:
        raise OptionDContractError("checkpoint artifact binding mismatch")
    if payload["schedule_sha256"] != expected_schedule_sha256:
        raise OptionDContractError("checkpoint schedule binding mismatch")
    if not isinstance(payload["state_bundle"], Mapping):
        raise OptionDContractError("checkpoint state bundle is invalid")
    bundle = RNGStateBundle.from_dict(payload["state_bundle"])
    if bundle.source_binding != expected_source_binding:
        raise OptionDContractError("embedded state source binding mismatch")
    return bundle


def write_snapshot_atomic(
    path: str | Path,
    snapshot: CheckpointSnapshot,
) -> str:
    """Persist already-captured owner bytes without rereading mutable state."""

    snapshot.validate()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    data = snapshot.data
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, destination)
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return snapshot.sha256


def write_checkpoint_atomic(
    path: str | Path,
    bundle: RNGStateBundle,
    *,
    artifact_binding: str,
    schedule_sha256: str | None = None,
) -> str:
    """Synchronously capture and durably publish one checkpoint."""

    snapshot = capture_checkpoint(
        bundle,
        artifact_binding=artifact_binding,
        schedule_sha256=schedule_sha256,
    )
    return write_snapshot_atomic(path, snapshot)


def load_checkpoint(
    path: str | Path,
    *,
    expected_source_binding: str,
    expected_artifact_binding: str,
    expected_schedule_sha256: str | None = None,
) -> RNGStateBundle:
    return restore_checkpoint(
        Path(path).read_bytes(),
        expected_source_binding=expected_source_binding,
        expected_artifact_binding=expected_artifact_binding,
        expected_schedule_sha256=expected_schedule_sha256,
    )
