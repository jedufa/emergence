"""CRCA checkpoint occurrence shell over unchanged CheckpointBundleV2 trees.

The BundleV2 loader remains the persistence and restore authority.  This module
only binds an already sealed bundle to Run/Attempt coordinates, publishes a
small external atomic visibility marker, and runs source-bound technical
conformance sentinels.  It never rewrites a bundle or grants qualification.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import struct
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from types import MappingProxyType

from emergence_contracts import (
    ABSOLUTE_TICK_CHECKPOINT_FORMAT_ID,
    ABSOLUTE_TICK_CHECKPOINT_TIME_AUTHORITY,
    AbsoluteTickCheckpointOccurrenceCommitV1,
    AbsoluteTickCheckpointOccurrenceInspectionV1,
    AbsoluteTickCheckpointOccurrenceV1,
    AbsoluteTickCheckpointParticipantOccurrenceV1,
    AttemptId,
    CheckpointConformanceIndexV1,
    CheckpointConformanceLayerV1,
    CheckpointConformanceResultV1,
    CheckpointConformanceRunV1,
    CheckpointConformanceSentinelV1,
    CheckpointConformanceStatusV1,
    CheckpointId,
    CheckpointOccurrenceCommitV1,
    CheckpointOccurrenceInspectionStatusV1,
    CheckpointOccurrenceInspectionV1,
    CheckpointOccurrenceV1,
    CheckpointParticipantOccurrenceV1,
    ContentDigest,
    ParticipantId,
    RunId,
    WORKHORSE_SEMANTIC_CHECKPOINT_FORMAT_ID,
    WORKHORSE_SEMANTIC_CHECKPOINT_TIME_AUTHORITY,
    WorkhorseSemanticCheckpointMaterialRoleV2,
    WorkhorseSemanticCheckpointMaterialV2,
    WorkhorseSemanticCheckpointOccurrenceCommitV2,
    WorkhorseSemanticCheckpointOccurrenceInspectionV2,
    WorkhorseSemanticCheckpointOccurrenceV2,
    canonical_json_bytes,
    canonical_sha256,
)

from .adapters import LegacyCompositeCheckpointAdapter


OCCURRENCE_DESCRIPTOR_NAME = "checkpoint_occurrence_v1.json"
OCCURRENCE_COMMIT_NAME = "checkpoint_occurrence_commit_v1.json"
BUNDLE_COMPLETION_MARKER_NAME = "bundle_complete.json"
ABSOLUTE_TICK_OCCURRENCE_DESCRIPTOR_NAME = "absolute_tick_checkpoint_occurrence_v1.json"
ABSOLUTE_TICK_OCCURRENCE_COMMIT_NAME = (
    "absolute_tick_checkpoint_occurrence_commit_v1.json"
)
ABSOLUTE_TICK_CHECKPOINT_STATE_NAME = "canonical_state.bin"
ABSOLUTE_TICK_CHECKPOINT_METADATA_NAME = "metadata.json"
ABSOLUTE_TICK_CHECKPOINT_COMPLETION_NAME = "bundle_complete.json"
ABSOLUTE_TICK_CHECKPOINT_METADATA_SCHEMA = "eiel_absolute_tick_checkpoint_metadata_v1"
ABSOLUTE_TICK_CHECKPOINT_COMPLETION_SCHEMA = (
    "eiel_absolute_tick_checkpoint_bundle_complete_v1"
)
WORKHORSE_SEMANTIC_OCCURRENCE_DESCRIPTOR_NAME = (
    "workhorse_semantic_checkpoint_occurrence_v2.json"
)
WORKHORSE_SEMANTIC_OCCURRENCE_COMMIT_NAME = (
    "workhorse_semantic_checkpoint_occurrence_commit_v2.json"
)
_WORKHORSE_FRAGMENT_SCHEMA = "eiel_workhorse_semantic_case_raw_fragment_v2"
_WORKHORSE_BASE_IDENTITY_SCHEMA = (
    "eiel_workhorse_science_compatible_base_checkpoint_identity_v2"
)
_WORKHORSE_SCIENCE_COMPATIBILITY_SCHEMA = (
    "eiel_workhorse_science_checkpoint_compatibility_binding_v2"
)


class CheckpointOccurrenceError(ValueError):
    """The external occurrence cannot be proven canonical and visible."""


def _fail(message: str) -> None:
    raise CheckpointOccurrenceError(message)


def _now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


def _digest_bytes(payload: bytes) -> ContentDigest:
    return ContentDigest(algorithm="sha256", hex=hashlib.sha256(payload).hexdigest())


def _digest_file(path: Path) -> ContentDigest:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return ContentDigest(algorithm="sha256", hex=digest.hexdigest())


def _bounded_path(root: Path, relative: str, *, field: str) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        _fail(f"{field} is not a bounded POSIX path")
    candidate = PurePosixPath(relative)
    if candidate.is_absolute() or ".." in candidate.parts or "." in candidate.parts:
        _fail(f"{field} is not a bounded POSIX path")
    resolved = root.joinpath(*candidate.parts)
    try:
        resolved.relative_to(root)
    except ValueError:
        _fail(f"{field} escapes its root")
    return resolved


def _resolve_existing_or_future(path: Path) -> Path:
    """Resolve existing ancestors so symlink parents cannot evade containment."""

    probe = path.absolute()
    missing: list[str] = []
    while not os.path.lexists(probe):
        if probe.parent == probe:
            _fail("path has no existing ancestor")
        missing.append(probe.name)
        probe = probe.parent
    resolved = probe.resolve(strict=True)
    for name in reversed(missing):
        resolved /= name
    return resolved


def _bundle_root(bundle_dir: str | Path) -> Path:
    requested = Path(bundle_dir).absolute()
    try:
        mode = requested.lstat().st_mode
    except FileNotFoundError:
        _fail("CheckpointBundleV2 root is missing")
    if stat.S_ISLNK(mode):
        _fail("CheckpointBundleV2 root must not be a symbolic link")
    if not stat.S_ISDIR(mode):
        _fail("CheckpointBundleV2 root is not a directory")
    return requested.resolve(strict=True)


def _require_regular_file(path: Path, *, field: str) -> None:
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError:
        _fail(f"{field} is missing")
    if not stat.S_ISREG(mode) or stat.S_ISLNK(mode):
        _fail(f"{field} is not a regular file")


def checkpoint_bundle_tree_digest_v1(bundle_dir: str | Path) -> ContentDigest:
    """Match BundleV2's exact published-tree digest without importing Runtime."""

    root = _bundle_root(bundle_dir)
    digest = hashlib.sha256()
    for current, directories, files in os.walk(root, topdown=True, followlinks=False):
        directories.sort()
        files.sort()
        for name in directories:
            path = Path(current, name)
            mode = path.lstat().st_mode
            if not stat.S_ISDIR(mode) or stat.S_ISLNK(mode):
                _fail("CheckpointBundleV2 contains a linked directory")
            relative = path.relative_to(root).as_posix()
            digest.update(b"directory\0" + os.fsencode(relative) + b"\0")
        for name in files:
            path = Path(current, name)
            file_stat = path.lstat()
            if not stat.S_ISREG(file_stat.st_mode) or stat.S_ISLNK(
                file_stat.st_mode
            ):
                _fail("CheckpointBundleV2 contains a non-regular file")
            relative = path.relative_to(root).as_posix()
            digest.update(b"file\0" + os.fsencode(relative) + b"\0")
            digest.update(struct.pack("<Q", int(file_stat.st_size)))
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
    return ContentDigest(algorithm="sha256", hex=digest.hexdigest())


def _read_json_object(path: Path, *, field: str) -> tuple[dict[str, object], bytes]:
    _require_regular_file(path, field=field)
    payload_bytes = path.read_bytes()
    try:
        payload = json.loads(payload_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CheckpointOccurrenceError(f"{field} is not valid JSON") from exc
    if not isinstance(payload, dict):
        _fail(f"{field} must be a JSON object")
    return payload, payload_bytes


def _load_sealed_bundle(
    bundle_dir: Path,
) -> tuple[Mapping[str, object], ContentDigest, dict[str, object], ContentDigest]:
    before = checkpoint_bundle_tree_digest_v1(bundle_dir)
    payload = LegacyCompositeCheckpointAdapter(bundle_dir).load(strict=True)
    completion, completion_bytes = _read_json_object(
        bundle_dir / BUNDLE_COMPLETION_MARKER_NAME,
        field="BundleV2 completion marker",
    )
    after = checkpoint_bundle_tree_digest_v1(bundle_dir)
    if before != after:
        _fail("CheckpointBundleV2 changed during strict validation")
    if completion.get("status") != "complete":
        _fail("BundleV2 completion marker is not complete")
    checkpoint_id = completion.get("checkpoint_id")
    if not isinstance(checkpoint_id, str) or not checkpoint_id:
        _fail("BundleV2 completion marker lacks checkpoint identity")
    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping):
        _fail("strict BundleV2 load lacks metadata")
    metadata_checkpoint_id = metadata.get("checkpoint_id")
    if metadata_checkpoint_id not in {None, "", checkpoint_id}:
        _fail("BundleV2 completion and metadata checkpoint identities differ")
    return payload, before, completion, _digest_bytes(completion_bytes)


@dataclass(frozen=True, slots=True)
class CheckpointParticipantBindingV1:
    """Bind one occurrence participant to exact files in the sealed bundle."""

    participant_id: ParticipantId
    owner_id: str
    owner_digest: ContentDigest
    bundle_relative_paths: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.participant_id, ParticipantId):
            _fail("participant binding has an invalid participant identity")
        if (
            not isinstance(self.owner_id, str)
            or not self.owner_id
            or self.owner_id.strip() != self.owner_id
        ):
            _fail("participant binding has an invalid owner identity")
        if not isinstance(self.owner_digest, ContentDigest):
            _fail("participant binding has an invalid owner digest")
        if not isinstance(self.bundle_relative_paths, tuple) or not (
            self.bundle_relative_paths
        ):
            _fail("participant binding must cover at least one bundle file")
        if len(self.bundle_relative_paths) != 1:
            _fail("current BundleV2 participant binding must cover exactly one file")
        if list(self.bundle_relative_paths) != sorted(
            set(self.bundle_relative_paths)
        ):
            _fail("participant bundle paths must be sorted and unique")
        for relative in self.bundle_relative_paths:
            _bounded_path(Path("."), relative, field="bundle_relative_paths")
            if str(self.participant_id) != f"bundle-file:{relative}":
                _fail("participant identity must bind its exact BundleV2 file")


def checkpoint_bundle_file_participant_bindings_v1(
    bundle_dir: str | Path,
    *,
    owner_id: str,
    owner_digest: ContentDigest,
) -> tuple[CheckpointParticipantBindingV1, ...]:
    """Represent the current integrated BundleV2 owner without inventing owners."""

    root = _bundle_root(bundle_dir)
    checkpoint_bundle_tree_digest_v1(root)
    if not isinstance(owner_digest, ContentDigest):
        _fail("BundleV2 owner digest is invalid")
    bindings = [
        CheckpointParticipantBindingV1(
            participant_id=ParticipantId(f"bundle-file:{path.relative_to(root).as_posix()}"),
            owner_id=owner_id,
            owner_digest=owner_digest,
            bundle_relative_paths=(path.relative_to(root).as_posix(),),
        )
        for path in sorted(root.rglob("*"))
        if path.is_file() and not path.is_symlink()
    ]
    if not bindings:
        _fail("CheckpointBundleV2 contains no participant files")
    return tuple(sorted(bindings, key=lambda item: str(item.participant_id)))


def _participant_occurrences(
    bundle_dir: Path,
    *,
    bindings: Sequence[CheckpointParticipantBindingV1],
    run_id: RunId,
    attempt_id: AttemptId,
    transaction_id: str,
    absolute_next_tick: int,
) -> tuple[CheckpointParticipantOccurrenceV1, ...]:
    if not isinstance(bindings, (tuple, list)) or not bindings:
        _fail("checkpoint occurrence needs participant bindings")
    all_files = {
        path.relative_to(bundle_dir).as_posix()
        for path in bundle_dir.rglob("*")
        if path.is_file() and not path.is_symlink()
    }
    covered: set[str] = set()
    participants: list[CheckpointParticipantOccurrenceV1] = []
    identities: set[str] = set()
    for binding in bindings:
        if not isinstance(binding, CheckpointParticipantBindingV1):
            _fail("checkpoint occurrence has an invalid participant binding")
        identity = str(binding.participant_id)
        if identity in identities:
            _fail("checkpoint occurrence repeats a participant identity")
        identities.add(identity)
        file_digests: list[dict[str, object]] = []
        for relative in binding.bundle_relative_paths:
            if relative in covered:
                _fail("checkpoint occurrence assigns one bundle file twice")
            path = _bounded_path(bundle_dir, relative, field="bundle_relative_paths")
            _require_regular_file(path, field="participant bundle file")
            covered.add(relative)
            file_digests.append(
                {"relative_path": relative, "content_digest": _digest_file(path).to_dict()}
            )
        participants.append(
            CheckpointParticipantOccurrenceV1(
                participant_id=binding.participant_id,
                run_id=run_id,
                attempt_id=attempt_id,
                transaction_id=transaction_id,
                absolute_next_tick=absolute_next_tick,
                owner_id=binding.owner_id,
                owner_digest=binding.owner_digest,
                state_digest=canonical_sha256(
                    {"bundle_file_digests": file_digests}
                ),
            )
        )
    if covered != all_files:
        _fail("checkpoint participant bindings do not exactly cover BundleV2 files")
    return tuple(sorted(participants, key=lambda item: str(item.participant_id)))


def describe_checkpoint_occurrence_v1(
    bundle_dir: str | Path,
    *,
    checkpoint_id: CheckpointId,
    run_id: RunId,
    attempt_id: AttemptId,
    transaction_id: str,
    absolute_next_tick: int,
    semantic_spec_digest: ContentDigest,
    implementation_digest: ContentDigest,
    participant_bindings: Sequence[CheckpointParticipantBindingV1],
    parent_checkpoint_digest: ContentDigest | None = None,
) -> CheckpointOccurrenceV1:
    """Describe a sealed bundle without mutating it or publishing visibility."""

    root = _bundle_root(bundle_dir)
    _payload, bundle_digest, completion, completion_digest = _load_sealed_bundle(root)
    if completion["checkpoint_id"] != str(checkpoint_id):
        _fail("requested checkpoint identity differs from the sealed bundle")
    architecture_b_identity = completion.get("architecture_b_identity")
    if isinstance(architecture_b_identity, Mapping) and (
        architecture_b_identity.get("snapshot_sequence") != absolute_next_tick
    ):
        _fail("absolute next tick differs from the native boundary identity")
    participants = _participant_occurrences(
        root,
        bindings=participant_bindings,
        run_id=run_id,
        attempt_id=attempt_id,
        transaction_id=transaction_id,
        absolute_next_tick=absolute_next_tick,
    )
    if checkpoint_bundle_tree_digest_v1(root) != bundle_digest:
        _fail("CheckpointBundleV2 changed while its occurrence was described")
    return CheckpointOccurrenceV1(
        checkpoint_id=checkpoint_id,
        run_id=run_id,
        attempt_id=attempt_id,
        transaction_id=transaction_id,
        absolute_next_tick=absolute_next_tick,
        semantic_spec_digest=semantic_spec_digest,
        implementation_digest=implementation_digest,
        participant_occurrences=participants,
        bundle_v2_digest=bundle_digest,
        completion_marker_digest=completion_digest,
        parent_checkpoint_digest=parent_checkpoint_digest,
    )


def _problem_digest(problem: str) -> ContentDigest:
    return canonical_sha256({"checkpoint_occurrence_problem": problem})


def _inspection(
    status: CheckpointOccurrenceInspectionStatusV1,
    *,
    bundle_digest: ContentDigest | None = None,
    occurrence_digest: ContentDigest | None = None,
    commit_digest: ContentDigest | None = None,
    problem: str | None = None,
) -> CheckpointOccurrenceInspectionV1:
    return CheckpointOccurrenceInspectionV1(
        status=status,
        canonical_visible=status is CheckpointOccurrenceInspectionStatusV1.VISIBLE,
        bundle_v2_digest=bundle_digest,
        occurrence_digest=occurrence_digest,
        commit_digest=commit_digest,
        problem_digest=None if problem is None else _problem_digest(problem),
    )


def _occurrence_paths(occurrence_root: Path) -> tuple[Path, Path]:
    return (
        occurrence_root / OCCURRENCE_DESCRIPTOR_NAME,
        occurrence_root / OCCURRENCE_COMMIT_NAME,
    )


def _validate_occurrence_root_entries(occurrence_root: Path) -> None:
    if not os.path.lexists(occurrence_root):
        return
    try:
        mode = occurrence_root.lstat().st_mode
    except FileNotFoundError:
        return
    if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
        _fail("occurrence root is not an unlinked directory")
    allowed = {OCCURRENCE_DESCRIPTOR_NAME, OCCURRENCE_COMMIT_NAME}
    unknown = sorted(entry.name for entry in occurrence_root.iterdir() if entry.name not in allowed)
    if unknown:
        _fail("occurrence root contains unbound entries")


def _validate_descriptor_participant_state(
    bundle: Path,
    descriptor: CheckpointOccurrenceV1,
) -> None:
    expected_files = {
        path.relative_to(bundle).as_posix()
        for path in bundle.rglob("*")
        if path.is_file() and not path.is_symlink()
    }
    observed_files: set[str] = set()
    for participant in descriptor.participant_occurrences:
        identity = str(participant.participant_id)
        if not identity.startswith("bundle-file:"):
            _fail("checkpoint participant identity is not file-bound")
        relative = identity.removeprefix("bundle-file:")
        if relative in observed_files:
            _fail("checkpoint descriptor repeats a participant file")
        path = _bounded_path(bundle, relative, field="participant_id")
        _require_regular_file(path, field="participant bundle file")
        expected_state_digest = canonical_sha256(
            {
                "bundle_file_digests": [
                    {
                        "relative_path": relative,
                        "content_digest": _digest_file(path).to_dict(),
                    }
                ]
            }
        )
        if participant.state_digest != expected_state_digest:
            _fail("checkpoint participant state digest differs from BundleV2")
        observed_files.add(relative)
    if observed_files != expected_files:
        _fail("checkpoint descriptor participants do not exactly cover BundleV2")


def _path_present(path: Path) -> bool:
    return os.path.lexists(path)


def inspect_checkpoint_occurrence_v1(
    bundle_dir: str | Path,
    occurrence_root: str | Path,
) -> CheckpointOccurrenceInspectionV1:
    """Classify crash windows without making an orphan canonically visible."""

    try:
        bundle = _bundle_root(bundle_dir)
    except Exception as exc:
        return _inspection(
            CheckpointOccurrenceInspectionStatusV1.QUARANTINED_INCOMPLETE_BUNDLE,
            problem=f"{type(exc).__name__}:{exc}",
        )
    try:
        occurrence_requested = Path(occurrence_root).absolute()
        if occurrence_requested.is_symlink():
            _fail("occurrence root is a symbolic link")
        occurrence = _resolve_existing_or_future(occurrence_requested)
    except Exception as exc:
        return _inspection(
            CheckpointOccurrenceInspectionStatusV1.QUARANTINED_CONFLICT,
            problem=f"{type(exc).__name__}:{exc}",
        )
    try:
        occurrence.relative_to(bundle)
    except ValueError:
        pass
    else:
        return _inspection(
            CheckpointOccurrenceInspectionStatusV1.QUARANTINED_CONFLICT,
            problem="occurrence root is inside CheckpointBundleV2",
        )
    try:
        _payload, bundle_digest, completion, completion_digest = _load_sealed_bundle(
            bundle
        )
    except Exception as exc:
        return _inspection(
            CheckpointOccurrenceInspectionStatusV1.QUARANTINED_INCOMPLETE_BUNDLE,
            problem=f"{type(exc).__name__}:{exc}",
        )
    descriptor_path, commit_path = _occurrence_paths(occurrence)
    try:
        _validate_occurrence_root_entries(occurrence)
    except Exception as exc:
        return _inspection(
            CheckpointOccurrenceInspectionStatusV1.QUARANTINED_CONFLICT,
            bundle_digest=bundle_digest,
            problem=f"{type(exc).__name__}:{exc}",
        )
    descriptor_present = _path_present(descriptor_path)
    commit_present = _path_present(commit_path)
    if not descriptor_present and not commit_present:
        return _inspection(
            CheckpointOccurrenceInspectionStatusV1.ORPHAN_SEALED_BUNDLE,
            bundle_digest=bundle_digest,
        )
    if descriptor_present != commit_present:
        return _inspection(
            CheckpointOccurrenceInspectionStatusV1.QUARANTINED_PARTIAL_OCCURRENCE,
            bundle_digest=bundle_digest,
            problem="occurrence descriptor and commit marker are not both present",
        )
    try:
        descriptor_value, descriptor_bytes = _read_json_object(
            descriptor_path, field="checkpoint occurrence descriptor"
        )
        descriptor = CheckpointOccurrenceV1.from_dict(descriptor_value)
        if descriptor_bytes != canonical_json_bytes(descriptor.to_dict()):
            _fail("checkpoint occurrence descriptor is not canonical JSON")
        commit_value, commit_bytes = _read_json_object(
            commit_path, field="checkpoint occurrence commit"
        )
        commit = CheckpointOccurrenceCommitV1.from_dict(commit_value)
        if commit_bytes != canonical_json_bytes(commit.to_dict()):
            _fail("checkpoint occurrence commit is not canonical JSON")
        if descriptor.bundle_v2_digest != bundle_digest:
            _fail("checkpoint occurrence binds a different BundleV2 tree")
        if descriptor.completion_marker_digest != completion_digest:
            _fail("checkpoint occurrence binds a different completion marker")
        if str(descriptor.checkpoint_id) != completion["checkpoint_id"]:
            _fail("checkpoint occurrence identity differs from the completion marker")
        architecture_b_identity = completion.get("architecture_b_identity")
        if isinstance(architecture_b_identity, Mapping) and (
            architecture_b_identity.get("snapshot_sequence")
            != descriptor.absolute_next_tick
        ):
            _fail("checkpoint occurrence differs from the native boundary tick")
        _validate_descriptor_participant_state(bundle, descriptor)
        if commit.checkpoint_id != descriptor.checkpoint_id:
            _fail("checkpoint occurrence commit identity differs from descriptor")
        if commit.occurrence_digest != descriptor.occurrence_digest():
            _fail("checkpoint occurrence commit binds a different descriptor")
        if commit.bundle_v2_digest != bundle_digest:
            _fail("checkpoint occurrence commit binds a different BundleV2 tree")
        if commit.completion_marker_digest != completion_digest:
            _fail("checkpoint occurrence commit binds a different completion marker")
        if checkpoint_bundle_tree_digest_v1(bundle) != bundle_digest:
            _fail("CheckpointBundleV2 changed during occurrence inspection")
        if (
            descriptor_path.read_bytes() != descriptor_bytes
            or commit_path.read_bytes() != commit_bytes
        ):
            _fail("checkpoint occurrence changed during inspection")
    except Exception as exc:
        return _inspection(
            CheckpointOccurrenceInspectionStatusV1.QUARANTINED_CONFLICT,
            bundle_digest=bundle_digest,
            problem=f"{type(exc).__name__}:{exc}",
        )
    return _inspection(
        CheckpointOccurrenceInspectionStatusV1.VISIBLE,
        bundle_digest=bundle_digest,
        occurrence_digest=descriptor.occurrence_digest(),
        commit_digest=commit.commit_digest(),
    )


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_create(path: Path, payload: bytes) -> bool:
    """Atomically create a regular file without replacing a racing writer."""

    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError:
            return False
        _fsync_directory(path.parent)
        return True
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def materialize_checkpoint_occurrence_v1(
    bundle_dir: str | Path,
    occurrence_root: str | Path,
    *,
    checkpoint_id: CheckpointId,
    run_id: RunId,
    attempt_id: AttemptId,
    transaction_id: str,
    absolute_next_tick: int,
    semantic_spec_digest: ContentDigest,
    implementation_digest: ContentDigest,
    participant_bindings: Sequence[CheckpointParticipantBindingV1],
    parent_checkpoint_digest: ContentDigest | None = None,
    committed_at_utc: str | None = None,
) -> tuple[CheckpointOccurrenceV1, CheckpointOccurrenceCommitV1]:
    """Publish an idempotent external occurrence descriptor and commit marker."""

    bundle = _bundle_root(bundle_dir)
    occurrence_requested = Path(occurrence_root).absolute()
    if occurrence_requested.is_symlink():
        _fail("occurrence root must not be a symbolic link")
    occurrence = _resolve_existing_or_future(occurrence_requested)
    try:
        occurrence.relative_to(bundle)
    except ValueError:
        pass
    else:
        _fail("occurrence root must be outside CheckpointBundleV2")
    occurrence_requested.mkdir(parents=True, exist_ok=True)
    if occurrence_requested.is_symlink():
        _fail("occurrence root must not be a symbolic link")
    occurrence = occurrence_requested.resolve(strict=True)
    try:
        occurrence.relative_to(bundle)
    except ValueError:
        pass
    else:
        _fail("occurrence root resolved inside CheckpointBundleV2")
    if not occurrence.is_dir():
        _fail("occurrence root must be an unlinked directory")
    _validate_occurrence_root_entries(occurrence)
    descriptor = describe_checkpoint_occurrence_v1(
        bundle,
        checkpoint_id=checkpoint_id,
        run_id=run_id,
        attempt_id=attempt_id,
        transaction_id=transaction_id,
        absolute_next_tick=absolute_next_tick,
        semantic_spec_digest=semantic_spec_digest,
        implementation_digest=implementation_digest,
        participant_bindings=participant_bindings,
        parent_checkpoint_digest=parent_checkpoint_digest,
    )
    descriptor_bytes = canonical_json_bytes(descriptor.to_dict())
    descriptor_path, commit_path = _occurrence_paths(occurrence)
    if _path_present(commit_path) and not _path_present(descriptor_path):
        _fail("partial occurrence has a commit marker without its descriptor")
    if not _atomic_create(descriptor_path, descriptor_bytes):
        _require_regular_file(
            descriptor_path, field="existing checkpoint occurrence descriptor"
        )
        if descriptor_path.read_bytes() != descriptor_bytes:
            _fail("existing checkpoint occurrence descriptor conflicts")
    if checkpoint_bundle_tree_digest_v1(bundle) != descriptor.bundle_v2_digest:
        _fail("CheckpointBundleV2 changed before occurrence commit")
    if _path_present(commit_path):
        commit_value, commit_bytes = _read_json_object(
            commit_path, field="existing checkpoint occurrence commit"
        )
        existing_commit = CheckpointOccurrenceCommitV1.from_dict(commit_value)
        if commit_bytes != canonical_json_bytes(existing_commit.to_dict()):
            _fail("existing checkpoint occurrence commit is not canonical")
        if (
            existing_commit.checkpoint_id != descriptor.checkpoint_id
            or existing_commit.occurrence_digest != descriptor.occurrence_digest()
            or existing_commit.bundle_v2_digest != descriptor.bundle_v2_digest
            or existing_commit.completion_marker_digest
            != descriptor.completion_marker_digest
        ):
            _fail("existing checkpoint occurrence commit conflicts")
        commit = existing_commit
    else:
        commit = CheckpointOccurrenceCommitV1(
            checkpoint_id=descriptor.checkpoint_id,
            occurrence_digest=descriptor.occurrence_digest(),
            bundle_v2_digest=descriptor.bundle_v2_digest,
            completion_marker_digest=descriptor.completion_marker_digest,
            committed_at_utc=committed_at_utc or _now_utc(),
        )
        commit_bytes = canonical_json_bytes(commit.to_dict())
        if not _atomic_create(commit_path, commit_bytes):
            commit_value, existing_bytes = _read_json_object(
                commit_path, field="racing checkpoint occurrence commit"
            )
            existing_commit = CheckpointOccurrenceCommitV1.from_dict(commit_value)
            if (
                existing_bytes != canonical_json_bytes(existing_commit.to_dict())
                or existing_commit.checkpoint_id != descriptor.checkpoint_id
                or existing_commit.occurrence_digest != descriptor.occurrence_digest()
                or existing_commit.bundle_v2_digest != descriptor.bundle_v2_digest
                or existing_commit.completion_marker_digest
                != descriptor.completion_marker_digest
            ):
                _fail("racing checkpoint occurrence commit conflicts")
            commit = existing_commit
    inspection = inspect_checkpoint_occurrence_v1(bundle, occurrence)
    if inspection.status is not CheckpointOccurrenceInspectionStatusV1.VISIBLE:
        _fail(f"checkpoint occurrence did not become visible: {inspection.status.value}")
    return descriptor, commit


def restore_checkpoint_occurrence_v1(
    bundle_dir: str | Path,
    occurrence_root: str | Path,
    *,
    expected_run_id: RunId,
    expected_attempt_id: AttemptId,
    expected_absolute_next_tick: int,
    expected_semantic_spec_digest: ContentDigest,
    expected_implementation_digest: ContentDigest,
    expected_occurrence_digest: ContentDigest,
) -> Mapping[str, object]:
    """Strictly load through a visible exact occurrence, returning loader payload."""

    bundle = _bundle_root(bundle_dir)
    occurrence_requested = Path(occurrence_root).absolute()
    if occurrence_requested.is_symlink():
        _fail("occurrence root must not be a symbolic link")
    occurrence = _resolve_existing_or_future(occurrence_requested)
    inspection = inspect_checkpoint_occurrence_v1(bundle, occurrence)
    if inspection.status is not CheckpointOccurrenceInspectionStatusV1.VISIBLE:
        _fail(f"checkpoint occurrence is not restorable: {inspection.status.value}")
    descriptor_value, _descriptor_bytes = _read_json_object(
        occurrence / OCCURRENCE_DESCRIPTOR_NAME,
        field="checkpoint occurrence descriptor",
    )
    descriptor = CheckpointOccurrenceV1.from_dict(descriptor_value)
    if descriptor.occurrence_digest() != expected_occurrence_digest:
        _fail("checkpoint occurrence digest differs from restore admission")
    expected = (
        ("Run", descriptor.run_id, expected_run_id),
        ("Attempt", descriptor.attempt_id, expected_attempt_id),
        (
            "absolute next tick",
            descriptor.absolute_next_tick,
            expected_absolute_next_tick,
        ),
        (
            "semantic specification",
            descriptor.semantic_spec_digest,
            expected_semantic_spec_digest,
        ),
        (
            "implementation",
            descriptor.implementation_digest,
            expected_implementation_digest,
        ),
    )
    for name, actual, wanted in expected:
        if actual != wanted:
            _fail(f"checkpoint occurrence {name} binding differs")
    payload, bundle_digest, _completion, completion_digest = _load_sealed_bundle(
        bundle
    )
    if (
        bundle_digest != descriptor.bundle_v2_digest
        or completion_digest != descriptor.completion_marker_digest
    ):
        _fail("CheckpointBundleV2 changed during occurrence restore")
    final_inspection = inspect_checkpoint_occurrence_v1(bundle, occurrence)
    if (
        final_inspection.status is not CheckpointOccurrenceInspectionStatusV1.VISIBLE
        or final_inspection.occurrence_digest != inspection.occurrence_digest
        or final_inspection.commit_digest != inspection.commit_digest
    ):
        _fail("checkpoint occurrence changed during restore")
    return payload


_ABSOLUTE_TICK_CHECKPOINT_FILES = frozenset(
    {
        ABSOLUTE_TICK_CHECKPOINT_STATE_NAME,
        ABSOLUTE_TICK_CHECKPOINT_METADATA_NAME,
        ABSOLUTE_TICK_CHECKPOINT_COMPLETION_NAME,
    }
)
_GENERATION_KEY_RE = re.compile(r"(?:^|[_-])g\d+(?:$|[_-])", re.IGNORECASE)


def _absolute_tick_bundle_root(bundle_dir: str | Path) -> Path:
    requested = Path(bundle_dir).absolute()
    try:
        mode = requested.lstat().st_mode
    except FileNotFoundError:
        _fail("absolute-tick checkpoint bundle root is missing")
    if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
        _fail("absolute-tick checkpoint bundle root is not an unlinked directory")
    root = requested.resolve(strict=True)
    if root.name != "checkpoint-tick-" + root.name.removeprefix("checkpoint-tick-"):
        _fail("absolute-tick checkpoint bundle path is not tick-shaped")
    suffix = root.name.removeprefix("checkpoint-tick-")
    if len(suffix) != 12 or not suffix.isdigit():
        _fail("absolute-tick checkpoint bundle path is not canonical")
    entries = {entry.name for entry in root.iterdir()}
    if entries != _ABSOLUTE_TICK_CHECKPOINT_FILES:
        _fail("absolute-tick checkpoint bundle must contain exactly three files")
    return root


def _require_unlinked_regular_file(path: Path, *, field: str) -> os.stat_result:
    try:
        file_stat = path.lstat()
    except FileNotFoundError:
        _fail(f"{field} is missing")
    if (
        not stat.S_ISREG(file_stat.st_mode)
        or stat.S_ISLNK(file_stat.st_mode)
        or file_stat.st_nlink != 1
    ):
        _fail(f"{field} is not an unlinked regular file")
    return file_stat


def _generation_authority_firewall(value: object, *, field: str) -> None:
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key)
            lowered = key.lower()
            if "generation" in lowered or _GENERATION_KEY_RE.search(lowered):
                _fail(f"{field} contains a generation-shaped key")
            _generation_authority_firewall(child, field=field)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _generation_authority_firewall(child, field=field)


def _absolute_tick_path_firewall(root: Path) -> None:
    for path in (root, *root.rglob("*")):
        for part in path.relative_to(root.parent).parts:
            lowered = part.lower()
            if "generation" in lowered or _GENERATION_KEY_RE.search(lowered):
                _fail("absolute-tick checkpoint contains a generation-shaped path")


def _sha256_text(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        _fail(f"{field} is not a lowercase SHA-256 digest")
    return value


def checkpoint_absolute_tick_bundle_tree_digest_v1(
    bundle_dir: str | Path,
) -> ContentDigest:
    """Digest the exact native three-file tree without invoking a legacy loader."""

    root = _absolute_tick_bundle_root(bundle_dir)
    _absolute_tick_path_firewall(root)
    digest = hashlib.sha256()
    digest.update(b"EIEL-ABSOLUTE-TICK-CHECKPOINT-BUNDLE-V1\0")
    for name in sorted(_ABSOLUTE_TICK_CHECKPOINT_FILES):
        path = root / name
        file_stat = _require_unlinked_regular_file(
            path, field=f"absolute-tick checkpoint file {name}"
        )
        digest.update(b"file\0" + os.fsencode(name) + b"\0")
        digest.update(struct.pack("<Q", int(file_stat.st_size)))
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return ContentDigest(algorithm="sha256", hex=digest.hexdigest())


def _freeze_absolute_tick_payload(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {
                str(key): _freeze_absolute_tick_payload(child)
                for key, child in value.items()
            }
        )
    if isinstance(value, list):
        return tuple(_freeze_absolute_tick_payload(child) for child in value)
    return value


def _load_sealed_absolute_tick_checkpoint_bundle_v1(
    bundle_dir: str | Path,
) -> tuple[Mapping[str, object], ContentDigest, Mapping[str, object], ContentDigest]:
    """Strictly load the native bundle while proving its three files immutable."""

    root = _absolute_tick_bundle_root(bundle_dir)
    before = checkpoint_absolute_tick_bundle_tree_digest_v1(root)
    state_path = root / ABSOLUTE_TICK_CHECKPOINT_STATE_NAME
    metadata_path = root / ABSOLUTE_TICK_CHECKPOINT_METADATA_NAME
    completion_path = root / ABSOLUTE_TICK_CHECKPOINT_COMPLETION_NAME
    _require_unlinked_regular_file(state_path, field="canonical checkpoint state")
    _require_unlinked_regular_file(metadata_path, field="checkpoint metadata")
    _require_unlinked_regular_file(completion_path, field="checkpoint completion")
    state_bytes = state_path.read_bytes()
    metadata, metadata_bytes = _read_json_object(
        metadata_path, field="absolute-tick checkpoint metadata"
    )
    completion, completion_bytes = _read_json_object(
        completion_path, field="absolute-tick checkpoint completion"
    )
    if metadata_bytes != canonical_json_bytes(metadata):
        _fail("absolute-tick checkpoint metadata is not canonical JSON")
    if completion_bytes != canonical_json_bytes(completion):
        _fail("absolute-tick checkpoint completion is not canonical JSON")
    _generation_authority_firewall(metadata, field="checkpoint metadata")
    _generation_authority_firewall(completion, field="checkpoint completion")
    expected_metadata_keys = {
        "schema_version",
        "checkpoint_format_id",
        "time_authority",
        "checkpoint_id",
        "absolute_next_tick",
        "artifact_path",
        "runtime_checkpoint",
    }
    if set(metadata) != expected_metadata_keys:
        _fail("absolute-tick checkpoint metadata has an invalid shape")
    if metadata.get("schema_version") != ABSOLUTE_TICK_CHECKPOINT_METADATA_SCHEMA:
        _fail("absolute-tick checkpoint metadata schema is invalid")
    if metadata.get("checkpoint_format_id") != ABSOLUTE_TICK_CHECKPOINT_FORMAT_ID:
        _fail("absolute-tick checkpoint format is invalid")
    if metadata.get("time_authority") != ABSOLUTE_TICK_CHECKPOINT_TIME_AUTHORITY:
        _fail("absolute-tick checkpoint time authority is invalid")
    checkpoint_id = metadata.get("checkpoint_id")
    if not isinstance(checkpoint_id, str) or not checkpoint_id:
        _fail("absolute-tick checkpoint identity is invalid")
    tick = metadata.get("absolute_next_tick")
    if type(tick) is not int or tick < 0:
        _fail("absolute-tick checkpoint boundary is invalid")
    if root.name != f"checkpoint-tick-{tick:012d}":
        _fail("absolute-tick checkpoint directory differs from its boundary")
    if (
        not isinstance(metadata.get("artifact_path"), str)
        or not metadata["artifact_path"]
    ):
        _fail("absolute-tick checkpoint artifact locator is invalid")
    runtime_checkpoint = metadata.get("runtime_checkpoint")
    if not isinstance(runtime_checkpoint, Mapping):
        _fail("absolute-tick runtime checkpoint metadata is invalid")
    expected_runtime_keys = {
        "schema_version",
        "time_authority",
        "contract",
        "canonical_state_shape",
        "canonical_state_nbytes",
        "canonical_state_sha256",
        "canonical_state_layout_sha256",
        "initial_next_absolute_tick",
        "advanced_ticks",
        "acknowledged_event_next_tick",
        "last_sink_receipt_sha256",
        "world_uids",
        "rng_keys",
        "rng_semantic_epoch",
        "artifact_sha256",
        "manifest_sha256",
        "device_identity",
        "population_fixture_id",
        "metadata_seal_sha256",
    }
    if set(runtime_checkpoint) != expected_runtime_keys:
        _fail("absolute-tick runtime checkpoint metadata has an invalid shape")
    if runtime_checkpoint.get("schema_version") != ABSOLUTE_TICK_CHECKPOINT_FORMAT_ID:
        _fail("native runtime checkpoint schema is invalid")
    if (
        runtime_checkpoint.get("time_authority")
        != ABSOLUTE_TICK_CHECKPOINT_TIME_AUTHORITY
    ):
        _fail("native runtime checkpoint time authority is invalid")
    contract = runtime_checkpoint.get("contract")
    if not isinstance(contract, Mapping):
        _fail("native runtime checkpoint contract is invalid")
    start_tick = contract.get("start_next_tick")
    advanced_ticks = runtime_checkpoint.get("advanced_ticks")
    acknowledged_tick = runtime_checkpoint.get("acknowledged_event_next_tick")
    if (
        type(start_tick) is not int
        or start_tick < 0
        or type(advanced_ticks) is not int
        or advanced_ticks < 0
        or type(acknowledged_tick) is not int
        or start_tick + advanced_ticks != tick
        or acknowledged_tick != tick
        or runtime_checkpoint.get("initial_next_absolute_tick") != start_tick
    ):
        _fail("native runtime checkpoint tick coordinates are inconsistent")
    state_nbytes = runtime_checkpoint.get("canonical_state_nbytes")
    state_shape = runtime_checkpoint.get("canonical_state_shape")
    if (
        type(state_nbytes) is not int
        or state_nbytes < 1
        or not isinstance(state_shape, list)
        or len(state_shape) != 2
        or any(type(item) is not int or item < 1 for item in state_shape)
        or state_shape[0] * state_shape[1] != state_nbytes
        or len(state_bytes) != state_nbytes
    ):
        _fail("native canonical checkpoint state layout is invalid")
    state_sha256 = hashlib.sha256(state_bytes).hexdigest()
    if (
        _sha256_text(
            runtime_checkpoint.get("canonical_state_sha256"),
            field="canonical state digest",
        )
        != state_sha256
    ):
        _fail("native canonical checkpoint state digest differs")
    expected_layout_digest = canonical_sha256(
        {
            "dtype": "uint8",
            "shape": state_shape,
            "nbytes": state_nbytes,
            "content_sha256": state_sha256,
        }
    ).hex
    if (
        _sha256_text(
            runtime_checkpoint.get("canonical_state_layout_sha256"),
            field="canonical state layout digest",
        )
        != expected_layout_digest
    ):
        _fail("native canonical checkpoint state layout seal differs")
    metadata_seal = runtime_checkpoint.get("metadata_seal_sha256")
    runtime_body = dict(runtime_checkpoint)
    runtime_body.pop("metadata_seal_sha256")
    if _sha256_text(metadata_seal, field="runtime checkpoint metadata seal") != (
        canonical_sha256(runtime_body).hex
    ):
        _fail("native runtime checkpoint metadata seal differs")
    if set(completion) != {
        "schema_version",
        "status",
        "checkpoint_format_id",
        "time_authority",
        "checkpoint_id",
        "absolute_next_tick",
        "canonical_state_file",
        "canonical_state_nbytes",
        "canonical_state_sha256",
        "metadata_sha256",
    }:
        _fail("absolute-tick checkpoint completion has an invalid shape")
    expected_completion = {
        "schema_version": ABSOLUTE_TICK_CHECKPOINT_COMPLETION_SCHEMA,
        "status": "complete",
        "checkpoint_format_id": ABSOLUTE_TICK_CHECKPOINT_FORMAT_ID,
        "time_authority": ABSOLUTE_TICK_CHECKPOINT_TIME_AUTHORITY,
        "checkpoint_id": checkpoint_id,
        "absolute_next_tick": tick,
        "canonical_state_file": ABSOLUTE_TICK_CHECKPOINT_STATE_NAME,
        "canonical_state_nbytes": state_nbytes,
        "canonical_state_sha256": state_sha256,
        "metadata_sha256": hashlib.sha256(metadata_bytes).hexdigest(),
    }
    if completion != expected_completion:
        _fail("absolute-tick checkpoint completion seal differs")
    after = checkpoint_absolute_tick_bundle_tree_digest_v1(root)
    if before != after:
        _fail("absolute-tick checkpoint bundle changed during strict validation")
    payload = MappingProxyType(
        {
            "canonical_state": state_bytes,
            "metadata": _freeze_absolute_tick_payload(metadata),
        }
    )
    return (
        payload,
        before,
        MappingProxyType(dict(completion)),
        _digest_bytes(completion_bytes),
    )


@dataclass(frozen=True, slots=True)
class AbsoluteTickCheckpointParticipantBindingV1:
    participant_id: ParticipantId
    owner_id: str
    owner_digest: ContentDigest
    bundle_relative_paths: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.participant_id, ParticipantId):
            _fail("absolute-tick participant binding has an invalid identity")
        if (
            not isinstance(self.owner_id, str)
            or not self.owner_id
            or self.owner_id.strip() != self.owner_id
        ):
            _fail("absolute-tick participant binding has an invalid owner")
        if not isinstance(self.owner_digest, ContentDigest):
            _fail("absolute-tick participant binding has an invalid owner digest")
        if (
            not isinstance(self.bundle_relative_paths, tuple)
            or len(self.bundle_relative_paths) != 1
        ):
            _fail("absolute-tick participant binding must cover exactly one file")
        relative = self.bundle_relative_paths[0]
        if str(self.participant_id) != f"absolute-tick-checkpoint-file:{relative}":
            _fail("absolute-tick participant identity must bind its exact file")
        if relative not in _ABSOLUTE_TICK_CHECKPOINT_FILES:
            _fail("absolute-tick participant binds an unknown checkpoint file")


def absolute_tick_checkpoint_bundle_file_participant_bindings_v1(
    bundle_dir: str | Path,
    *,
    owner_id: str,
    owner_digest: ContentDigest,
) -> tuple[AbsoluteTickCheckpointParticipantBindingV1, ...]:
    root = _absolute_tick_bundle_root(bundle_dir)
    _load_sealed_absolute_tick_checkpoint_bundle_v1(root)
    return tuple(
        AbsoluteTickCheckpointParticipantBindingV1(
            participant_id=ParticipantId(f"absolute-tick-checkpoint-file:{relative}"),
            owner_id=owner_id,
            owner_digest=owner_digest,
            bundle_relative_paths=(relative,),
        )
        for relative in sorted(_ABSOLUTE_TICK_CHECKPOINT_FILES)
    )


def _absolute_tick_participant_occurrences(
    bundle: Path,
    *,
    bindings: Sequence[AbsoluteTickCheckpointParticipantBindingV1],
    run_id: RunId,
    attempt_id: AttemptId,
    transaction_id: str,
    absolute_next_tick: int,
) -> tuple[AbsoluteTickCheckpointParticipantOccurrenceV1, ...]:
    if not isinstance(bindings, (tuple, list)) or not bindings:
        _fail("absolute-tick checkpoint occurrence needs participant bindings")
    covered: set[str] = set()
    identities: set[str] = set()
    participants: list[AbsoluteTickCheckpointParticipantOccurrenceV1] = []
    for binding in bindings:
        if not isinstance(binding, AbsoluteTickCheckpointParticipantBindingV1):
            _fail("absolute-tick checkpoint has an invalid participant binding")
        identity = str(binding.participant_id)
        if identity in identities:
            _fail("absolute-tick checkpoint repeats a participant identity")
        identities.add(identity)
        relative = binding.bundle_relative_paths[0]
        if relative in covered:
            _fail("absolute-tick checkpoint assigns one file twice")
        path = _bounded_path(bundle, relative, field="bundle_relative_paths")
        _require_unlinked_regular_file(path, field="native participant file")
        covered.add(relative)
        participants.append(
            AbsoluteTickCheckpointParticipantOccurrenceV1(
                participant_id=binding.participant_id,
                run_id=run_id,
                attempt_id=attempt_id,
                transaction_id=transaction_id,
                absolute_next_tick=absolute_next_tick,
                owner_id=binding.owner_id,
                owner_digest=binding.owner_digest,
                state_digest=canonical_sha256(
                    {
                        "checkpoint_file_digests": [
                            {
                                "relative_path": relative,
                                "content_digest": _digest_file(path).to_dict(),
                            }
                        ]
                    }
                ),
            )
        )
    if covered != _ABSOLUTE_TICK_CHECKPOINT_FILES:
        _fail("absolute-tick participants do not exactly cover the checkpoint")
    return tuple(sorted(participants, key=lambda item: str(item.participant_id)))


def describe_absolute_tick_checkpoint_occurrence_v1(
    bundle_dir: str | Path,
    *,
    checkpoint_id: CheckpointId,
    run_id: RunId,
    attempt_id: AttemptId,
    transaction_id: str,
    absolute_next_tick: int,
    semantic_spec_digest: ContentDigest,
    implementation_digest: ContentDigest,
    participant_bindings: Sequence[AbsoluteTickCheckpointParticipantBindingV1],
    parent_checkpoint_digest: ContentDigest | None = None,
) -> AbsoluteTickCheckpointOccurrenceV1:
    bundle = _absolute_tick_bundle_root(bundle_dir)
    _payload, bundle_digest, completion, completion_digest = (
        _load_sealed_absolute_tick_checkpoint_bundle_v1(bundle)
    )
    if completion["checkpoint_id"] != str(checkpoint_id):
        _fail("requested checkpoint identity differs from native bundle")
    if completion["absolute_next_tick"] != absolute_next_tick:
        _fail("requested absolute tick differs from native bundle")
    participants = _absolute_tick_participant_occurrences(
        bundle,
        bindings=participant_bindings,
        run_id=run_id,
        attempt_id=attempt_id,
        transaction_id=transaction_id,
        absolute_next_tick=absolute_next_tick,
    )
    if checkpoint_absolute_tick_bundle_tree_digest_v1(bundle) != bundle_digest:
        _fail("absolute-tick checkpoint changed while being described")
    return AbsoluteTickCheckpointOccurrenceV1(
        checkpoint_format_id=ABSOLUTE_TICK_CHECKPOINT_FORMAT_ID,
        time_authority=ABSOLUTE_TICK_CHECKPOINT_TIME_AUTHORITY,
        checkpoint_id=checkpoint_id,
        run_id=run_id,
        attempt_id=attempt_id,
        transaction_id=transaction_id,
        absolute_next_tick=absolute_next_tick,
        semantic_spec_digest=semantic_spec_digest,
        implementation_digest=implementation_digest,
        participant_occurrences=participants,
        checkpoint_bundle_digest=bundle_digest,
        completion_marker_digest=completion_digest,
        parent_checkpoint_digest=parent_checkpoint_digest,
    )


def _absolute_tick_inspection(
    status: CheckpointOccurrenceInspectionStatusV1,
    *,
    bundle_digest: ContentDigest | None = None,
    occurrence_digest: ContentDigest | None = None,
    commit_digest: ContentDigest | None = None,
    problem: str | None = None,
) -> AbsoluteTickCheckpointOccurrenceInspectionV1:
    return AbsoluteTickCheckpointOccurrenceInspectionV1(
        status=status,
        canonical_visible=status is CheckpointOccurrenceInspectionStatusV1.VISIBLE,
        checkpoint_bundle_digest=bundle_digest,
        occurrence_digest=occurrence_digest,
        commit_digest=commit_digest,
        problem_digest=None if problem is None else _problem_digest(problem),
    )


def _absolute_tick_occurrence_paths(occurrence: Path) -> tuple[Path, Path]:
    return (
        occurrence / ABSOLUTE_TICK_OCCURRENCE_DESCRIPTOR_NAME,
        occurrence / ABSOLUTE_TICK_OCCURRENCE_COMMIT_NAME,
    )


def _validate_absolute_tick_occurrence_root_entries(occurrence: Path) -> None:
    if not os.path.lexists(occurrence):
        return
    mode = occurrence.lstat().st_mode
    if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
        _fail("absolute-tick occurrence root is not an unlinked directory")
    allowed = {
        ABSOLUTE_TICK_OCCURRENCE_DESCRIPTOR_NAME,
        ABSOLUTE_TICK_OCCURRENCE_COMMIT_NAME,
    }
    if {entry.name for entry in occurrence.iterdir()} - allowed:
        _fail("absolute-tick occurrence root contains unbound entries")


def _validate_absolute_tick_descriptor_participants(
    bundle: Path,
    descriptor: AbsoluteTickCheckpointOccurrenceV1,
) -> None:
    observed: set[str] = set()
    for participant in descriptor.participant_occurrences:
        identity = str(participant.participant_id)
        prefix = "absolute-tick-checkpoint-file:"
        if not identity.startswith(prefix):
            _fail("absolute-tick participant identity is not file-bound")
        relative = identity.removeprefix(prefix)
        if relative in observed or relative not in _ABSOLUTE_TICK_CHECKPOINT_FILES:
            _fail("absolute-tick descriptor repeats or invents a checkpoint file")
        path = _bounded_path(bundle, relative, field="participant_id")
        _require_unlinked_regular_file(path, field="native participant file")
        expected_state_digest = canonical_sha256(
            {
                "checkpoint_file_digests": [
                    {
                        "relative_path": relative,
                        "content_digest": _digest_file(path).to_dict(),
                    }
                ]
            }
        )
        if participant.state_digest != expected_state_digest:
            _fail("absolute-tick participant state digest differs")
        observed.add(relative)
    if observed != _ABSOLUTE_TICK_CHECKPOINT_FILES:
        _fail("absolute-tick descriptor participants do not cover the bundle")


def inspect_absolute_tick_checkpoint_occurrence_v1(
    bundle_dir: str | Path,
    occurrence_root: str | Path,
) -> AbsoluteTickCheckpointOccurrenceInspectionV1:
    try:
        bundle = _absolute_tick_bundle_root(bundle_dir)
    except Exception as exc:
        return _absolute_tick_inspection(
            CheckpointOccurrenceInspectionStatusV1.QUARANTINED_INCOMPLETE_BUNDLE,
            problem=f"{type(exc).__name__}:{exc}",
        )
    try:
        requested = Path(occurrence_root).absolute()
        if requested.is_symlink():
            _fail("absolute-tick occurrence root is a symbolic link")
        occurrence = _resolve_existing_or_future(requested)
        try:
            occurrence.relative_to(bundle)
        except ValueError:
            pass
        else:
            _fail("absolute-tick occurrence root is inside its checkpoint")
    except Exception as exc:
        return _absolute_tick_inspection(
            CheckpointOccurrenceInspectionStatusV1.QUARANTINED_CONFLICT,
            problem=f"{type(exc).__name__}:{exc}",
        )
    try:
        _payload, bundle_digest, completion, completion_digest = (
            _load_sealed_absolute_tick_checkpoint_bundle_v1(bundle)
        )
    except Exception as exc:
        return _absolute_tick_inspection(
            CheckpointOccurrenceInspectionStatusV1.QUARANTINED_INCOMPLETE_BUNDLE,
            problem=f"{type(exc).__name__}:{exc}",
        )
    descriptor_path, commit_path = _absolute_tick_occurrence_paths(occurrence)
    try:
        _validate_absolute_tick_occurrence_root_entries(occurrence)
    except Exception as exc:
        return _absolute_tick_inspection(
            CheckpointOccurrenceInspectionStatusV1.QUARANTINED_CONFLICT,
            bundle_digest=bundle_digest,
            problem=f"{type(exc).__name__}:{exc}",
        )
    descriptor_present = _path_present(descriptor_path)
    commit_present = _path_present(commit_path)
    if not descriptor_present and not commit_present:
        return _absolute_tick_inspection(
            CheckpointOccurrenceInspectionStatusV1.ORPHAN_SEALED_BUNDLE,
            bundle_digest=bundle_digest,
        )
    if descriptor_present != commit_present:
        return _absolute_tick_inspection(
            CheckpointOccurrenceInspectionStatusV1.QUARANTINED_PARTIAL_OCCURRENCE,
            bundle_digest=bundle_digest,
            problem="absolute-tick descriptor and commit are not both present",
        )
    try:
        _require_unlinked_regular_file(
            descriptor_path, field="absolute-tick occurrence descriptor"
        )
        descriptor_value, descriptor_bytes = _read_json_object(
            descriptor_path, field="absolute-tick occurrence descriptor"
        )
        descriptor = AbsoluteTickCheckpointOccurrenceV1.from_dict(descriptor_value)
        if descriptor_bytes != canonical_json_bytes(descriptor.to_dict()):
            _fail("absolute-tick occurrence descriptor is not canonical JSON")
        _require_unlinked_regular_file(
            commit_path, field="absolute-tick occurrence commit"
        )
        commit_value, commit_bytes = _read_json_object(
            commit_path, field="absolute-tick occurrence commit"
        )
        commit = AbsoluteTickCheckpointOccurrenceCommitV1.from_dict(commit_value)
        if commit_bytes != canonical_json_bytes(commit.to_dict()):
            _fail("absolute-tick occurrence commit is not canonical JSON")
        if descriptor.checkpoint_bundle_digest != bundle_digest:
            _fail("absolute-tick occurrence binds a different checkpoint tree")
        if descriptor.completion_marker_digest != completion_digest:
            _fail("absolute-tick occurrence binds a different completion marker")
        if str(descriptor.checkpoint_id) != completion["checkpoint_id"]:
            _fail("absolute-tick occurrence checkpoint identity differs")
        if descriptor.absolute_next_tick != completion["absolute_next_tick"]:
            _fail("absolute-tick occurrence boundary differs")
        _validate_absolute_tick_descriptor_participants(bundle, descriptor)
        if commit.occurrence_digest != descriptor.occurrence_digest():
            _fail("absolute-tick occurrence commit binds a different descriptor")
        if commit.checkpoint_bundle_digest != bundle_digest:
            _fail("absolute-tick occurrence commit binds a different checkpoint")
        if commit.checkpoint_format_id != descriptor.checkpoint_format_id:
            _fail("absolute-tick occurrence commit format differs")
        if checkpoint_absolute_tick_bundle_tree_digest_v1(bundle) != bundle_digest:
            _fail("absolute-tick checkpoint changed during occurrence inspection")
        if (
            descriptor_path.read_bytes() != descriptor_bytes
            or commit_path.read_bytes() != commit_bytes
        ):
            _fail("absolute-tick occurrence changed during inspection")
    except Exception as exc:
        return _absolute_tick_inspection(
            CheckpointOccurrenceInspectionStatusV1.QUARANTINED_CONFLICT,
            bundle_digest=bundle_digest,
            problem=f"{type(exc).__name__}:{exc}",
        )
    return _absolute_tick_inspection(
        CheckpointOccurrenceInspectionStatusV1.VISIBLE,
        bundle_digest=bundle_digest,
        occurrence_digest=descriptor.occurrence_digest(),
        commit_digest=commit.commit_digest(),
    )


def materialize_absolute_tick_checkpoint_occurrence_v1(
    bundle_dir: str | Path,
    occurrence_root: str | Path,
    *,
    checkpoint_id: CheckpointId,
    run_id: RunId,
    attempt_id: AttemptId,
    transaction_id: str,
    absolute_next_tick: int,
    semantic_spec_digest: ContentDigest,
    implementation_digest: ContentDigest,
    participant_bindings: Sequence[AbsoluteTickCheckpointParticipantBindingV1],
    parent_checkpoint_digest: ContentDigest | None = None,
    committed_at_utc: str | None = None,
) -> tuple[
    AbsoluteTickCheckpointOccurrenceV1,
    AbsoluteTickCheckpointOccurrenceCommitV1,
]:
    bundle = _absolute_tick_bundle_root(bundle_dir)
    requested = Path(occurrence_root).absolute()
    if requested.is_symlink():
        _fail("absolute-tick occurrence root must not be a symbolic link")
    resolved_future = _resolve_existing_or_future(requested)
    try:
        resolved_future.relative_to(bundle)
    except ValueError:
        pass
    else:
        _fail("absolute-tick occurrence root must be outside its checkpoint")
    requested.mkdir(parents=True, exist_ok=True)
    if requested.is_symlink():
        _fail("absolute-tick occurrence root must not be a symbolic link")
    occurrence = requested.resolve(strict=True)
    try:
        occurrence.relative_to(bundle)
    except ValueError:
        pass
    else:
        _fail("absolute-tick occurrence root resolved inside its checkpoint")
    _validate_absolute_tick_occurrence_root_entries(occurrence)
    descriptor = describe_absolute_tick_checkpoint_occurrence_v1(
        bundle,
        checkpoint_id=checkpoint_id,
        run_id=run_id,
        attempt_id=attempt_id,
        transaction_id=transaction_id,
        absolute_next_tick=absolute_next_tick,
        semantic_spec_digest=semantic_spec_digest,
        implementation_digest=implementation_digest,
        participant_bindings=participant_bindings,
        parent_checkpoint_digest=parent_checkpoint_digest,
    )
    descriptor_bytes = canonical_json_bytes(descriptor.to_dict())
    descriptor_path, commit_path = _absolute_tick_occurrence_paths(occurrence)
    if _path_present(commit_path) and not _path_present(descriptor_path):
        _fail("partial absolute-tick occurrence has a commit without descriptor")
    if not _atomic_create(descriptor_path, descriptor_bytes):
        _require_unlinked_regular_file(
            descriptor_path, field="existing absolute-tick descriptor"
        )
        if descriptor_path.read_bytes() != descriptor_bytes:
            _fail("existing absolute-tick occurrence descriptor conflicts")
    if (
        checkpoint_absolute_tick_bundle_tree_digest_v1(bundle)
        != descriptor.checkpoint_bundle_digest
    ):
        _fail("absolute-tick checkpoint changed before occurrence commit")
    if _path_present(commit_path):
        _require_unlinked_regular_file(
            commit_path, field="existing absolute-tick occurrence commit"
        )
        commit_value, commit_bytes = _read_json_object(
            commit_path, field="existing absolute-tick occurrence commit"
        )
        commit = AbsoluteTickCheckpointOccurrenceCommitV1.from_dict(commit_value)
        if (
            commit_bytes != canonical_json_bytes(commit.to_dict())
            or commit.occurrence_digest != descriptor.occurrence_digest()
            or commit.checkpoint_bundle_digest != descriptor.checkpoint_bundle_digest
            or commit.checkpoint_format_id != descriptor.checkpoint_format_id
        ):
            _fail("existing absolute-tick occurrence commit conflicts")
    else:
        commit = AbsoluteTickCheckpointOccurrenceCommitV1(
            occurrence_digest=descriptor.occurrence_digest(),
            checkpoint_bundle_digest=descriptor.checkpoint_bundle_digest,
            checkpoint_format_id=descriptor.checkpoint_format_id,
            committed_at_utc=committed_at_utc or _now_utc(),
        )
        commit_bytes = canonical_json_bytes(commit.to_dict())
        if not _atomic_create(commit_path, commit_bytes):
            _require_unlinked_regular_file(
                commit_path, field="racing absolute-tick occurrence commit"
            )
            commit_value, existing_bytes = _read_json_object(
                commit_path, field="racing absolute-tick occurrence commit"
            )
            racing_commit = AbsoluteTickCheckpointOccurrenceCommitV1.from_dict(
                commit_value
            )
            if (
                existing_bytes != canonical_json_bytes(racing_commit.to_dict())
                or racing_commit.occurrence_digest != descriptor.occurrence_digest()
                or racing_commit.checkpoint_bundle_digest
                != descriptor.checkpoint_bundle_digest
                or racing_commit.checkpoint_format_id != descriptor.checkpoint_format_id
            ):
                _fail("racing absolute-tick occurrence commit conflicts")
            commit = racing_commit
    inspection = inspect_absolute_tick_checkpoint_occurrence_v1(bundle, occurrence)
    if inspection.status is not CheckpointOccurrenceInspectionStatusV1.VISIBLE:
        _fail(
            "absolute-tick checkpoint occurrence did not become visible: "
            + inspection.status.value
        )
    return descriptor, commit


def restore_absolute_tick_checkpoint_occurrence_v1(
    bundle_dir: str | Path,
    occurrence_root: str | Path,
    *,
    expected_run_id: RunId,
    expected_attempt_id: AttemptId,
    expected_absolute_next_tick: int,
    expected_semantic_spec_digest: ContentDigest,
    expected_implementation_digest: ContentDigest,
    expected_occurrence_digest: ContentDigest,
) -> Mapping[str, object]:
    bundle = _absolute_tick_bundle_root(bundle_dir)
    requested = Path(occurrence_root).absolute()
    if requested.is_symlink():
        _fail("absolute-tick occurrence root must not be a symbolic link")
    occurrence = _resolve_existing_or_future(requested)
    inspection = inspect_absolute_tick_checkpoint_occurrence_v1(bundle, occurrence)
    if inspection.status is not CheckpointOccurrenceInspectionStatusV1.VISIBLE:
        _fail(
            "absolute-tick checkpoint occurrence is not restorable: "
            + inspection.status.value
        )
    descriptor_value, _descriptor_bytes = _read_json_object(
        occurrence / ABSOLUTE_TICK_OCCURRENCE_DESCRIPTOR_NAME,
        field="absolute-tick occurrence descriptor",
    )
    descriptor = AbsoluteTickCheckpointOccurrenceV1.from_dict(descriptor_value)
    if descriptor.occurrence_digest() != expected_occurrence_digest:
        _fail("absolute-tick occurrence digest differs from restore admission")
    expected = (
        ("Run", descriptor.run_id, expected_run_id),
        ("Attempt", descriptor.attempt_id, expected_attempt_id),
        (
            "absolute next tick",
            descriptor.absolute_next_tick,
            expected_absolute_next_tick,
        ),
        (
            "semantic specification",
            descriptor.semantic_spec_digest,
            expected_semantic_spec_digest,
        ),
        (
            "implementation",
            descriptor.implementation_digest,
            expected_implementation_digest,
        ),
    )
    for name, actual, wanted in expected:
        if actual != wanted:
            _fail(f"absolute-tick checkpoint occurrence {name} binding differs")
    payload, bundle_digest, _completion, completion_digest = (
        _load_sealed_absolute_tick_checkpoint_bundle_v1(bundle)
    )
    if (
        bundle_digest != descriptor.checkpoint_bundle_digest
        or completion_digest != descriptor.completion_marker_digest
    ):
        _fail("absolute-tick checkpoint changed during occurrence restore")
    final_inspection = inspect_absolute_tick_checkpoint_occurrence_v1(
        bundle, occurrence
    )
    if (
        final_inspection.status is not CheckpointOccurrenceInspectionStatusV1.VISIBLE
        or final_inspection.occurrence_digest != inspection.occurrence_digest
        or final_inspection.commit_digest != inspection.commit_digest
    ):
        _fail("absolute-tick checkpoint occurrence changed during restore")
    return payload


def _stable_material_bytes_v2(
    path: str | Path,
    *,
    field: str,
    require_read_only: bool,
) -> tuple[Path, bytes]:
    """Open one source by descriptor and prove path/file identity stayed fixed."""

    requested = Path(path).absolute()
    try:
        before = requested.lstat()
    except FileNotFoundError:
        _fail(f"{field} is missing")
    if (
        stat.S_ISLNK(before.st_mode)
        or not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or (require_read_only and before.st_mode & 0o222)
    ):
        _fail(f"{field} is not an exact unlinked immutable regular file")
    descriptor = os.open(
        requested,
        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        opened = os.fstat(descriptor)
        identity = lambda value: (  # noqa: E731 - compact immutable stat projection
            value.st_dev,
            value.st_ino,
            value.st_mode,
            value.st_nlink,
            value.st_size,
            value.st_mtime_ns,
            value.st_ctime_ns,
        )
        if identity(opened) != identity(before):
            _fail(f"{field} changed while being opened")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after_open = os.fstat(descriptor)
        if identity(after_open) != identity(opened):
            _fail(f"{field} changed while being read")
    finally:
        os.close(descriptor)
    after_path = requested.lstat()
    if identity(after_path) != identity(before):
        _fail(f"{field} path identity changed while being read")
    return requested.resolve(strict=True), b"".join(chunks)


def _workhorse_material_v2(
    role: WorkhorseSemanticCheckpointMaterialRoleV2,
    path: Path,
    payload: bytes,
) -> WorkhorseSemanticCheckpointMaterialV2:
    return WorkhorseSemanticCheckpointMaterialV2(
        material_role=role,
        absolute_path=path.as_posix(),
        content_digest=_digest_bytes(payload),
        byte_count=len(payload),
    )


def _lower_sha256(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        _fail(f"{field} is not a lowercase SHA-256 digest")
    return value


def _load_workhorse_semantic_checkpoint_source_v2(
    manifest_path: str | Path,
    science_checkpoint_compatibility_path: str | Path,
) -> tuple[
    Mapping[str, object],
    tuple[WorkhorseSemanticCheckpointMaterialV2, ...],
    ContentDigest,
]:
    manifest_file, manifest_bytes = _stable_material_bytes_v2(
        manifest_path,
        field="Workhorse semantic fragment manifest",
        require_read_only=True,
    )
    try:
        manifest = json.loads(manifest_bytes.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CheckpointOccurrenceError(
            "Workhorse semantic fragment manifest is unreadable"
        ) from exc
    if not isinstance(manifest, Mapping):
        _fail("Workhorse semantic fragment manifest is not an object")
    required_manifest = {
        "schema_version",
        "claim_class",
        "qualification_authority_granted",
        "science_authority_granted",
        "observation",
        "case_id",
        "base_checkpoint",
        "fragment_body_sha256",
    }
    if not required_manifest.issubset(manifest):
        _fail("Workhorse semantic fragment lacks checkpoint authority fields")
    manifest_body = dict(manifest)
    fragment_body_digest = _lower_sha256(
        manifest_body.pop("fragment_body_sha256"),
        field="Workhorse semantic fragment body",
    )
    if (
        manifest.get("schema_version") != _WORKHORSE_FRAGMENT_SCHEMA
        or manifest.get("claim_class") != "TECHNICAL_ONLY_RAW_EVIDENCE"
        or manifest.get("qualification_authority_granted") is not False
        or manifest.get("science_authority_granted") is not False
        or manifest.get("observation")
        != {
            "mode": "OFF",
            "science_records_emitted": 0,
            "science_observation_schedule": None,
            "population_change_detection": False,
        }
        or manifest.get("case_id") != "checkpoint_same_restore_sequence"
        or canonical_sha256(manifest_body).hex != fragment_body_digest
    ):
        _fail("Workhorse semantic checkpoint source changes claim or observation law")
    base = manifest.get("base_checkpoint")
    if not isinstance(base, Mapping) or set(base) != {
        "identity",
        "state_file_name",
        "state_bytes",
        "state_sha256",
    }:
        _fail("Workhorse semantic fragment base checkpoint shape is invalid")
    identity = base.get("identity")
    identity_fields = {
        "schema_version",
        "time_authority",
        "shape_id",
        "layout_sha256",
        "semantic_transition_sha256",
        "source_artifact_identity_sha256",
        "checkpoint_at_tick",
        "B",
        "world_uids",
        "rng_keys",
        "state_bytes",
        "state_sha256",
        "last_acknowledged_event_frontier",
        "science_checkpoint_compatibility",
        "observation",
        "qualification_authority_granted",
        "science_authority_granted",
        "base_checkpoint_identity_sha256",
    }
    if not isinstance(identity, Mapping) or set(identity) != identity_fields:
        _fail("Workhorse base checkpoint identity shape is invalid")
    identity_body = dict(identity)
    declared_identity = _lower_sha256(
        identity_body.pop("base_checkpoint_identity_sha256"),
        field="base checkpoint identity",
    )
    checkpoint_tick = identity.get("checkpoint_at_tick")
    if (
        identity.get("schema_version") != _WORKHORSE_BASE_IDENTITY_SCHEMA
        or identity.get("time_authority")
        != WORKHORSE_SEMANTIC_CHECKPOINT_TIME_AUTHORITY
        or type(checkpoint_tick) is not int
        or checkpoint_tick < 0
        or identity.get("observation") != manifest.get("observation")
        or identity.get("qualification_authority_granted") is not False
        or identity.get("science_authority_granted") is not False
        or canonical_sha256(identity_body).hex != declared_identity
    ):
        _fail("Workhorse base checkpoint identity seal or authority changed")
    state_name = base.get("state_file_name")
    if (
        not isinstance(state_name, str)
        or not state_name
        or PurePosixPath(state_name).name != state_name
    ):
        _fail("Workhorse base checkpoint state locator is invalid")
    state_file, state_bytes = _stable_material_bytes_v2(
        manifest_file.parent / state_name,
        field="Workhorse base checkpoint state",
        require_read_only=True,
    )
    state_digest = _lower_sha256(base.get("state_sha256"), field="checkpoint state")
    if (
        base.get("state_bytes") != len(state_bytes)
        or identity.get("state_bytes") != len(state_bytes)
        or identity.get("state_sha256") != state_digest
        or hashlib.sha256(state_bytes).hexdigest() != state_digest
    ):
        _fail("Workhorse base checkpoint state bytes differ from identity")
    ack = identity.get("last_acknowledged_event_frontier")
    ack_fields = {
        "receipt_path",
        "receipt_file_sha256",
        "receipt_body_sha256",
        "dispatch_content_sha256",
        "event_rows_file_sha256",
        "event_counts_file_sha256",
        "world_uids",
        "start_next_absolute_tick",
        "end_next_absolute_tick",
        "artifact_identity_sha256",
        "rng_identity_sha256",
        "post_state_sha256",
        "dispatch_identity_sha256",
    }
    if not isinstance(ack, Mapping) or set(ack) != ack_fields:
        _fail("Workhorse checkpoint event acknowledgement shape is invalid")
    ack_file, ack_bytes = _stable_material_bytes_v2(
        str(ack.get("receipt_path") or ""),
        field="Workhorse event acknowledgement",
        require_read_only=True,
    )
    ack_digest = _lower_sha256(
        ack.get("receipt_file_sha256"), field="event acknowledgement"
    )
    if (
        hashlib.sha256(ack_bytes).hexdigest() != ack_digest
        or ack.get("end_next_absolute_tick") != checkpoint_tick
        or ack.get("post_state_sha256") != state_digest
        or ack.get("world_uids") != identity.get("world_uids")
        or ack.get("artifact_identity_sha256")
        != identity.get("source_artifact_identity_sha256")
    ):
        _fail("Workhorse event acknowledgement differs from checkpoint frontier")
    science = identity.get("science_checkpoint_compatibility")
    science_fields = {
        "schema_version",
        "science_consumer_extension_contract_path",
        "science_consumer_extension_contract_sha256",
        "science_extension_contract_schema",
        "science_checkpoint_extension_schema",
        "science_checkpoint_identity_schema",
        "science_restore_gate_schema",
        "science_route_binding_adapter_schema",
        "science_route_binding_wire_schema",
        "science_observation_record_schema",
        "science_observation_prefix_cursor_schema",
        "continuation_adapter_version",
        "observation_adapter_version",
        "runtime_observation_mode",
        "native_capture_runtime_integrated",
        "science_route_qualified",
        "binding_sha256",
    }
    if not isinstance(science, Mapping) or set(science) != science_fields:
        _fail("Workhorse science compatibility binding shape is invalid")
    science_body = dict(science)
    science_binding_digest = _lower_sha256(
        science_body.pop("binding_sha256"), field="science compatibility binding"
    )
    science_file, science_bytes = _stable_material_bytes_v2(
        science_checkpoint_compatibility_path,
        field="Workhorse science compatibility contract",
        require_read_only=False,
    )
    science_file_digest = _lower_sha256(
        science.get("science_consumer_extension_contract_sha256"),
        field="science compatibility contract",
    )
    declared_relative = science.get("science_consumer_extension_contract_path")
    if (
        science.get("schema_version") != _WORKHORSE_SCIENCE_COMPATIBILITY_SCHEMA
        or science.get("runtime_observation_mode") != "OFF"
        or science.get("native_capture_runtime_integrated") is not False
        or science.get("science_route_qualified") is not False
        or not isinstance(declared_relative, str)
        or not science_file.as_posix().endswith("/" + declared_relative)
        or hashlib.sha256(science_bytes).hexdigest() != science_file_digest
        or canonical_sha256(science_body).hex != science_binding_digest
    ):
        _fail("Workhorse science compatibility content or observation law changed")
    materials = tuple(
        sorted(
            (
                _workhorse_material_v2(
                    WorkhorseSemanticCheckpointMaterialRoleV2.BASE_CHECKPOINT_MANIFEST,
                    manifest_file,
                    manifest_bytes,
                ),
                _workhorse_material_v2(
                    WorkhorseSemanticCheckpointMaterialRoleV2.BASE_CHECKPOINT_STATE,
                    state_file,
                    state_bytes,
                ),
                _workhorse_material_v2(
                    WorkhorseSemanticCheckpointMaterialRoleV2.EVENT_ACKNOWLEDGEMENT,
                    ack_file,
                    ack_bytes,
                ),
                _workhorse_material_v2(
                    WorkhorseSemanticCheckpointMaterialRoleV2.SCIENCE_CHECKPOINT_COMPATIBILITY,
                    science_file,
                    science_bytes,
                ),
            ),
            key=lambda material: material.material_role.value,
        )
    )
    return (
        MappingProxyType(dict(manifest)),
        materials,
        ContentDigest(algorithm="sha256", hex=declared_identity),
    )


def describe_workhorse_semantic_checkpoint_occurrence_v2(
    manifest_path: str | Path,
    science_checkpoint_compatibility_path: str | Path,
    *,
    checkpoint_id: CheckpointId,
    run_id: RunId,
    attempt_id: AttemptId,
    transaction_id: str,
    absolute_next_tick: int,
    semantic_spec_digest: ContentDigest,
    implementation_digest: ContentDigest,
    parent_checkpoint_digest: ContentDigest | None = None,
) -> WorkhorseSemanticCheckpointOccurrenceV2:
    manifest, materials, identity_digest = (
        _load_workhorse_semantic_checkpoint_source_v2(
            manifest_path, science_checkpoint_compatibility_path
        )
    )
    checkpoint = dict(manifest["base_checkpoint"])
    identity = dict(checkpoint["identity"])
    if identity["checkpoint_at_tick"] != absolute_next_tick:
        _fail("requested absolute tick differs from Workhorse checkpoint")
    material_digest = canonical_sha256(
        [material.to_dict() for material in materials]
    )
    return WorkhorseSemanticCheckpointOccurrenceV2(
        checkpoint_format_id=WORKHORSE_SEMANTIC_CHECKPOINT_FORMAT_ID,
        time_authority=WORKHORSE_SEMANTIC_CHECKPOINT_TIME_AUTHORITY,
        checkpoint_id=checkpoint_id,
        run_id=run_id,
        attempt_id=attempt_id,
        transaction_id=transaction_id,
        absolute_next_tick=absolute_next_tick,
        semantic_spec_digest=semantic_spec_digest,
        implementation_digest=implementation_digest,
        base_checkpoint_identity_digest=identity_digest,
        source_materials=materials,
        source_materials_digest=material_digest,
        parent_checkpoint_digest=parent_checkpoint_digest,
    )


def _workhorse_inspection_v2(
    status: CheckpointOccurrenceInspectionStatusV1,
    *,
    materials_digest: ContentDigest | None = None,
    occurrence_digest: ContentDigest | None = None,
    commit_digest: ContentDigest | None = None,
    problem: str | None = None,
) -> WorkhorseSemanticCheckpointOccurrenceInspectionV2:
    return WorkhorseSemanticCheckpointOccurrenceInspectionV2(
        status=status,
        canonical_visible=status is CheckpointOccurrenceInspectionStatusV1.VISIBLE,
        source_materials_digest=materials_digest,
        occurrence_digest=occurrence_digest,
        commit_digest=commit_digest,
        problem_digest=None if problem is None else _problem_digest(problem),
    )


def _workhorse_occurrence_paths_v2(root: Path) -> tuple[Path, Path]:
    return (
        root / WORKHORSE_SEMANTIC_OCCURRENCE_DESCRIPTOR_NAME,
        root / WORKHORSE_SEMANTIC_OCCURRENCE_COMMIT_NAME,
    )


def _validate_workhorse_occurrence_root_v2(root: Path) -> None:
    if not os.path.lexists(root):
        return
    mode = root.lstat().st_mode
    if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
        _fail("Workhorse checkpoint occurrence root is not an unlinked directory")
    allowed = {
        WORKHORSE_SEMANTIC_OCCURRENCE_DESCRIPTOR_NAME,
        WORKHORSE_SEMANTIC_OCCURRENCE_COMMIT_NAME,
    }
    if {entry.name for entry in root.iterdir()} - allowed:
        _fail("Workhorse checkpoint occurrence root contains unbound entries")


def inspect_workhorse_semantic_checkpoint_occurrence_v2(
    occurrence_root: str | Path,
) -> WorkhorseSemanticCheckpointOccurrenceInspectionV2:
    try:
        requested = Path(occurrence_root).absolute()
        if requested.is_symlink():
            _fail("Workhorse checkpoint occurrence root is a symbolic link")
        root = _resolve_existing_or_future(requested)
        _validate_workhorse_occurrence_root_v2(root)
    except Exception as exc:
        return _workhorse_inspection_v2(
            CheckpointOccurrenceInspectionStatusV1.QUARANTINED_CONFLICT,
            problem=f"{type(exc).__name__}:{exc}",
        )
    descriptor_path, commit_path = _workhorse_occurrence_paths_v2(root)
    descriptor_present = _path_present(descriptor_path)
    commit_present = _path_present(commit_path)
    if not descriptor_present and not commit_present:
        return _workhorse_inspection_v2(
            CheckpointOccurrenceInspectionStatusV1.ORPHAN_SEALED_BUNDLE,
        )
    if descriptor_present != commit_present:
        return _workhorse_inspection_v2(
            CheckpointOccurrenceInspectionStatusV1.QUARANTINED_PARTIAL_OCCURRENCE,
            problem="Workhorse descriptor and commit are not both present",
        )
    try:
        if root.lstat().st_mode & 0o222:
            _fail("visible Workhorse checkpoint occurrence root is writable")
        _, descriptor_bytes = _stable_material_bytes_v2(
            descriptor_path,
            field="Workhorse checkpoint occurrence descriptor",
            require_read_only=True,
        )
        descriptor_value = json.loads(descriptor_bytes.decode("ascii"))
        descriptor = WorkhorseSemanticCheckpointOccurrenceV2.from_dict(
            descriptor_value
        )
        if descriptor_bytes != canonical_json_bytes(descriptor.to_dict()):
            _fail("Workhorse checkpoint occurrence descriptor is not canonical JSON")
        _, commit_bytes = _stable_material_bytes_v2(
            commit_path,
            field="Workhorse checkpoint occurrence commit",
            require_read_only=True,
        )
        commit_value = json.loads(commit_bytes.decode("ascii"))
        commit = WorkhorseSemanticCheckpointOccurrenceCommitV2.from_dict(commit_value)
        if commit_bytes != canonical_json_bytes(commit.to_dict()):
            _fail("Workhorse checkpoint occurrence commit is not canonical JSON")
        material_by_role = {
            material.material_role: material for material in descriptor.source_materials
        }
        manifest_material = material_by_role[
            WorkhorseSemanticCheckpointMaterialRoleV2.BASE_CHECKPOINT_MANIFEST
        ]
        science_material = material_by_role[
            WorkhorseSemanticCheckpointMaterialRoleV2.SCIENCE_CHECKPOINT_COMPATIBILITY
        ]
        _manifest, reopened_materials, identity_digest = (
            _load_workhorse_semantic_checkpoint_source_v2(
                manifest_material.absolute_path,
                science_material.absolute_path,
            )
        )
        if (
            reopened_materials != descriptor.source_materials
            or identity_digest != descriptor.base_checkpoint_identity_digest
            or commit.occurrence_digest != descriptor.occurrence_digest()
            or commit.source_materials_digest != descriptor.source_materials_digest
            or commit.checkpoint_format_id != descriptor.checkpoint_format_id
        ):
            _fail("Workhorse checkpoint occurrence digest chain differs")
        _, final_descriptor_bytes = _stable_material_bytes_v2(
            descriptor_path,
            field="Workhorse checkpoint occurrence descriptor",
            require_read_only=True,
        )
        _, final_commit_bytes = _stable_material_bytes_v2(
            commit_path,
            field="Workhorse checkpoint occurrence commit",
            require_read_only=True,
        )
        _manifest, final_materials, final_identity_digest = (
            _load_workhorse_semantic_checkpoint_source_v2(
                manifest_material.absolute_path,
                science_material.absolute_path,
            )
        )
        if (
            final_descriptor_bytes != descriptor_bytes
            or final_commit_bytes != commit_bytes
            or final_materials != reopened_materials
            or final_identity_digest != identity_digest
        ):
            _fail("Workhorse checkpoint occurrence changed during inspection")
    except Exception as exc:
        return _workhorse_inspection_v2(
            CheckpointOccurrenceInspectionStatusV1.QUARANTINED_CONFLICT,
            problem=f"{type(exc).__name__}:{exc}",
        )
    return _workhorse_inspection_v2(
        CheckpointOccurrenceInspectionStatusV1.VISIBLE,
        materials_digest=descriptor.source_materials_digest,
        occurrence_digest=descriptor.occurrence_digest(),
        commit_digest=commit.commit_digest(),
    )


def materialize_workhorse_semantic_checkpoint_occurrence_v2(
    manifest_path: str | Path,
    science_checkpoint_compatibility_path: str | Path,
    occurrence_root: str | Path,
    *,
    checkpoint_id: CheckpointId,
    run_id: RunId,
    attempt_id: AttemptId,
    transaction_id: str,
    absolute_next_tick: int,
    semantic_spec_digest: ContentDigest,
    implementation_digest: ContentDigest,
    parent_checkpoint_digest: ContentDigest | None = None,
    committed_at_utc: str | None = None,
) -> tuple[
    WorkhorseSemanticCheckpointOccurrenceV2,
    WorkhorseSemanticCheckpointOccurrenceCommitV2,
]:
    requested = Path(occurrence_root).absolute()
    if requested.is_symlink():
        _fail("Workhorse checkpoint occurrence root must not be a symbolic link")
    requested.mkdir(parents=True, exist_ok=True)
    root = requested.resolve(strict=True)
    _validate_workhorse_occurrence_root_v2(root)
    descriptor = describe_workhorse_semantic_checkpoint_occurrence_v2(
        manifest_path,
        science_checkpoint_compatibility_path,
        checkpoint_id=checkpoint_id,
        run_id=run_id,
        attempt_id=attempt_id,
        transaction_id=transaction_id,
        absolute_next_tick=absolute_next_tick,
        semantic_spec_digest=semantic_spec_digest,
        implementation_digest=implementation_digest,
        parent_checkpoint_digest=parent_checkpoint_digest,
    )
    descriptor_bytes = canonical_json_bytes(descriptor.to_dict())
    descriptor_path, commit_path = _workhorse_occurrence_paths_v2(root)
    if _path_present(commit_path) and not _path_present(descriptor_path):
        _fail("partial Workhorse checkpoint occurrence has commit without descriptor")
    if not _atomic_create(descriptor_path, descriptor_bytes):
        _, existing = _stable_material_bytes_v2(
            descriptor_path,
            field="existing Workhorse checkpoint descriptor",
            require_read_only=True,
        )
        if existing != descriptor_bytes:
            _fail("existing Workhorse checkpoint descriptor conflicts")
    else:
        descriptor_path.chmod(0o444)
        _fsync_directory(root)
    _manifest, materials_before_commit, identity_before_commit = (
        _load_workhorse_semantic_checkpoint_source_v2(
            manifest_path, science_checkpoint_compatibility_path
        )
    )
    if (
        materials_before_commit != descriptor.source_materials
        or identity_before_commit != descriptor.base_checkpoint_identity_digest
    ):
        _fail("Workhorse checkpoint source changed before occurrence commit")
    if _path_present(commit_path):
        _, commit_bytes = _stable_material_bytes_v2(
            commit_path,
            field="existing Workhorse checkpoint commit",
            require_read_only=True,
        )
        commit_value = json.loads(commit_bytes.decode("ascii"))
        commit = WorkhorseSemanticCheckpointOccurrenceCommitV2.from_dict(commit_value)
        if (
            commit_bytes != canonical_json_bytes(commit.to_dict())
            or commit.occurrence_digest != descriptor.occurrence_digest()
            or commit.source_materials_digest != descriptor.source_materials_digest
            or commit.checkpoint_format_id != descriptor.checkpoint_format_id
        ):
            _fail("existing Workhorse checkpoint commit conflicts")
    else:
        commit = WorkhorseSemanticCheckpointOccurrenceCommitV2(
            occurrence_digest=descriptor.occurrence_digest(),
            source_materials_digest=descriptor.source_materials_digest,
            checkpoint_format_id=descriptor.checkpoint_format_id,
            committed_at_utc=committed_at_utc or _now_utc(),
        )
        commit_bytes = canonical_json_bytes(commit.to_dict())
        if not _atomic_create(commit_path, commit_bytes):
            _, racing_bytes = _stable_material_bytes_v2(
                commit_path,
                field="racing Workhorse checkpoint commit",
                require_read_only=True,
            )
            racing_value = json.loads(racing_bytes.decode("ascii"))
            racing_commit = WorkhorseSemanticCheckpointOccurrenceCommitV2.from_dict(
                racing_value
            )
            if (
                racing_bytes != canonical_json_bytes(racing_commit.to_dict())
                or racing_commit.occurrence_digest != descriptor.occurrence_digest()
                or racing_commit.source_materials_digest
                != descriptor.source_materials_digest
                or racing_commit.checkpoint_format_id != descriptor.checkpoint_format_id
            ):
                _fail("racing Workhorse checkpoint commit conflicts")
            commit = racing_commit
        else:
            commit_path.chmod(0o444)
            _fsync_directory(root)
    root.chmod(0o555)
    _fsync_directory(root.parent)
    inspection = inspect_workhorse_semantic_checkpoint_occurrence_v2(root)
    if inspection.status is not CheckpointOccurrenceInspectionStatusV1.VISIBLE:
        _fail(
            "Workhorse checkpoint occurrence did not become visible: "
            + inspection.status.value
        )
    return descriptor, commit


def restore_workhorse_semantic_checkpoint_occurrence_v2(
    occurrence_root: str | Path,
    *,
    expected_run_id: RunId,
    expected_attempt_id: AttemptId,
    expected_absolute_next_tick: int,
    expected_semantic_spec_digest: ContentDigest,
    expected_implementation_digest: ContentDigest,
    expected_occurrence_digest: ContentDigest,
) -> Mapping[str, object]:
    root = _resolve_existing_or_future(Path(occurrence_root).absolute())
    inspection = inspect_workhorse_semantic_checkpoint_occurrence_v2(root)
    if inspection.status is not CheckpointOccurrenceInspectionStatusV1.VISIBLE:
        _fail(
            "Workhorse checkpoint occurrence is not restorable: "
            + inspection.status.value
        )
    descriptor_value, _ = _read_json_object(
        root / WORKHORSE_SEMANTIC_OCCURRENCE_DESCRIPTOR_NAME,
        field="Workhorse checkpoint occurrence descriptor",
    )
    descriptor = WorkhorseSemanticCheckpointOccurrenceV2.from_dict(descriptor_value)
    expected = (
        ("occurrence", descriptor.occurrence_digest(), expected_occurrence_digest),
        ("Run", descriptor.run_id, expected_run_id),
        ("Attempt", descriptor.attempt_id, expected_attempt_id),
        ("absolute tick", descriptor.absolute_next_tick, expected_absolute_next_tick),
        ("semantic specification", descriptor.semantic_spec_digest, expected_semantic_spec_digest),
        ("implementation", descriptor.implementation_digest, expected_implementation_digest),
    )
    for label, actual, wanted in expected:
        if actual != wanted:
            _fail(f"Workhorse checkpoint occurrence {label} binding differs")
    material_by_role = {
        material.material_role: material for material in descriptor.source_materials
    }
    manifest_material = material_by_role[
        WorkhorseSemanticCheckpointMaterialRoleV2.BASE_CHECKPOINT_MANIFEST
    ]
    science_material = material_by_role[
        WorkhorseSemanticCheckpointMaterialRoleV2.SCIENCE_CHECKPOINT_COMPATIBILITY
    ]
    manifest, materials, identity_digest = _load_workhorse_semantic_checkpoint_source_v2(
        manifest_material.absolute_path,
        science_material.absolute_path,
    )
    final = inspect_workhorse_semantic_checkpoint_occurrence_v2(root)
    if (
        materials != descriptor.source_materials
        or identity_digest != descriptor.base_checkpoint_identity_digest
        or final.status is not CheckpointOccurrenceInspectionStatusV1.VISIBLE
        or final.occurrence_digest != inspection.occurrence_digest
        or final.commit_digest != inspection.commit_digest
    ):
        _fail("Workhorse checkpoint occurrence changed during restore")
    state_material = material_by_role[
        WorkhorseSemanticCheckpointMaterialRoleV2.BASE_CHECKPOINT_STATE
    ]
    _, state_bytes = _stable_material_bytes_v2(
        state_material.absolute_path,
        field="Workhorse base checkpoint state",
        require_read_only=True,
    )
    return MappingProxyType(
        {
            "manifest": manifest,
            "canonical_state": state_bytes,
            "descriptor": MappingProxyType(descriptor.to_dict()),
        }
    )


_DEFAULT_CONFORMANCE_SENTINELS = (
    (
        "backend_bundle_projection_completion",
        CheckpointConformanceLayerV1.BACKEND,
        "simulation.checkpoint_bundle_v2",
        "engine/simulation/checkpoint_bundle_v2.py",
        "tests/perf/test_eiel_architecture_b_checkpoint_projection_save.py::test_projection_save_persists_detached_authority_and_completion_metadata",
    ),
    (
        "contract_execution_identity_vectors",
        CheckpointConformanceLayerV1.CONTRACT,
        "emergence_contracts.execution_identity",
        "emergence_contracts/execution_identity.py",
        "tests/modular_contracts/test_crca_execution_identity_v1.py::test_run_and_attempt_identity_reference_vectors_are_stable",
    ),
    (
        "qualification_qrr_runability_no_claim",
        CheckpointConformanceLayerV1.QUALIFICATION,
        "engine.qrr.production_front_door_v1",
        "engine/qrr/production_front_door_v1.py",
        "tests/test_eiel_qrr_v1.py::test_runability_gate_accepts_exact_runtime_proofs_without_science_claim",
    ),
    (
        "runtime_bundle_strict_state",
        CheckpointConformanceLayerV1.RUNTIME,
        "simulation.checkpoint_bundle_v2",
        "engine/simulation/checkpoint_bundle_v2.py",
        "tests/regression/test_checkpoint_bundle_contracts.py::test_checkpoint_bundle_strict_requires_metrics_state_and_e1_state",
    ),
)


def build_checkpoint_conformance_index_v1(
    repo_root: str | Path,
) -> CheckpointConformanceIndexV1:
    """Bind four existing technical sentinels to their exact owner/test sources."""

    root = Path(repo_root).resolve()
    sentinels: list[CheckpointConformanceSentinelV1] = []
    for sentinel_id, layer, owner_id, owner_source_path, test_node_id in (
        _DEFAULT_CONFORMANCE_SENTINELS
    ):
        test_source_path = test_node_id.split("::", 1)[0]
        owner_path = _bounded_path(root, owner_source_path, field="owner_source_path")
        test_path = _bounded_path(root, test_source_path, field="test_node_id")
        _require_regular_file(owner_path, field="conformance owner source")
        _require_regular_file(test_path, field="conformance test source")
        sentinels.append(
            CheckpointConformanceSentinelV1(
                sentinel_id=sentinel_id,
                layer=layer,
                owner_id=owner_id,
                owner_source_path=owner_source_path,
                owner_digest=_digest_file(owner_path),
                test_node_id=test_node_id,
                test_source_digest=_digest_file(test_path),
            )
        )
    return CheckpointConformanceIndexV1(
        sentinels=tuple(sorted(sentinels, key=lambda item: item.sentinel_id))
    )


def run_checkpoint_conformance_v1(
    repo_root: str | Path,
    index: CheckpointConformanceIndexV1,
    *,
    observed_at_utc: str | None = None,
    timeout_seconds: int = 300,
) -> CheckpointConformanceRunV1:
    """Run exact source-bound tests; the typed result has no qualification effect."""

    if not isinstance(index, CheckpointConformanceIndexV1):
        _fail("checkpoint conformance run needs a typed index")
    if not isinstance(timeout_seconds, int) or isinstance(timeout_seconds, bool) or (
        timeout_seconds < 1
    ):
        _fail("checkpoint conformance timeout must be a positive integer")
    root = Path(repo_root).resolve()
    results: list[CheckpointConformanceResultV1] = []
    for sentinel in index.sentinels:
        owner_path = _bounded_path(
            root, sentinel.owner_source_path, field="owner_source_path"
        )
        test_source_path = sentinel.test_node_id.split("::", 1)[0]
        test_path = _bounded_path(root, test_source_path, field="test_node_id")
        evidence: dict[str, object] = {
            "sentinel_id": sentinel.sentinel_id,
            "test_node_id": sentinel.test_node_id,
            "qualification_effect": False,
        }
        status = CheckpointConformanceStatusV1.FAIL
        try:
            observed_owner = _digest_file(owner_path)
            observed_test = _digest_file(test_path)
            evidence["observed_owner_digest"] = observed_owner.to_dict()
            evidence["observed_test_source_digest"] = observed_test.to_dict()
            if observed_owner != sentinel.owner_digest:
                evidence["failure"] = "owner source digest changed"
            elif observed_test != sentinel.test_source_digest:
                evidence["failure"] = "test source digest changed"
            else:
                completed = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "pytest",
                        "-q",
                        "-o",
                        "addopts=",
                        sentinel.test_node_id,
                    ],
                    cwd=root,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=timeout_seconds,
                )
                evidence.update(
                    {
                        "returncode": completed.returncode,
                        "stdout": completed.stdout,
                        "stderr": completed.stderr,
                    }
                )
                if completed.returncode == 0:
                    status = CheckpointConformanceStatusV1.PASS
        except (OSError, subprocess.TimeoutExpired) as exc:
            evidence["failure"] = f"{type(exc).__name__}:{exc}"
        results.append(
            CheckpointConformanceResultV1(
                sentinel_id=sentinel.sentinel_id,
                sentinel_digest=sentinel.sentinel_digest(),
                status=status,
                evidence_digest=canonical_sha256(evidence),
            )
        )
    return CheckpointConformanceRunV1(
        index_digest=index.index_digest(),
        results=tuple(sorted(results, key=lambda item: item.sentinel_id)),
        observed_at_utc=observed_at_utc or _now_utc(),
    )
