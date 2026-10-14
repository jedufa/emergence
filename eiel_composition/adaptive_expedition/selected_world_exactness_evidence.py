"""Internal exactness proof for one selected Workhorse world.

This is a bounded engineering evidence adapter.  It reopens already terminal
canonical execution roots and their Workhorse receipts; it never launches a
provider, grants branch authority, interprets scientific meaning, or creates a
new public contract.  The emitted receipt is deliberately internal and grants
neither science nor Qualification authority.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import tempfile
from typing import Any, Final

from emergence_contracts.canonical import canonical_sha256
from emergence_contracts.checkpoint_occurrence import (
    AbsoluteTickCheckpointOccurrenceCommitV1,
    AbsoluteTickCheckpointOccurrenceV1,
)
from emergence_contracts.ids import ContentDigest
from engine.campaigns.artifact_lifecycle import (
    validate_artifact_disposition_manifest_v1,
)
from engine.campaigns.execution_front_door_v1 import ExecutionRequestV2
from engine.perf.cuda_full_evolution_v2.workhorse.continuation import (
    validate_workhorse_indexed_continuation_contract_v2,
    validate_workhorse_indexed_continuation_result_v2,
    validate_workhorse_terminalized_indexed_continuation_receipt_v2,
)
from engine.perf.cuda_full_evolution_v2.workhorse.indexed_checkpoint import (
    inspect_workhorse_indexed_checkpoint_set_v1,
)

from .continuation_adapter import (
    _projection_v2,
    materialize_workhorse_founder_exact_state_ref_v1,
)


RECEIPT_SCHEMA: Final[str] = (
    "eiel_selected_world_exactness_engineering_receipt_internal_v2"
)
RECEIPT_CLAIM_CLASS: Final[str] = (
    "TECHNICAL_ENGINEERING_ONLY_NO_SCIENCE_OR_QUALIFICATION_AUTHORITY"
)
RECEIPT_DECISION: Final[str] = (
    "SELECTED_WORLD_PHYSICAL_REPACK_EXACTNESS_ENGINEERING_GATE_SATISFIED"
)
_TIME_AUTHORITY: Final[str] = "absolute_tick_v1"
_MAX_JSON_BYTES: Final[int] = 64 * 1024 * 1024
_SHA256_CHARS: Final[frozenset[str]] = frozenset("0123456789abcdef")
_TRAJECTORY_LABELS: Final[tuple[str, ...]] = (
    "reference_b2_same_control",
    "repacked_b1_same_control",
    "repacked_b1_changed_control",
)
_CHECK_NAMES: Final[frozenset[str]] = frozenset(
    {
        "parent_state_ref_reopened_and_authenticated",
        "physical_selected_source_matches_parent_state",
        "all_indexed_checkpoint_receipts_reopened",
        "same_control_b2_to_b1_repack_equal_at_every_compared_tick",
        "same_control_event_trajectory_equal_at_every_compared_tick",
        "changed_control_divergence_not_before_declared_branch_tick",
        "logical_to_physical_repacking_nonsemantic_for_same_control",
        "absolute_ticks_only",
        "science_and_execution_request_identities_bound",
        "low_level_carrier_scope_explicit",
    }
)
_LIMITATIONS: Final[tuple[str, ...]] = (
    "This receipt is engineering physical-repack exactness evidence for the named Workhorse roots only.",
    "The low-level G3 children bind the selected physical state, RNG key, source checkpoint, absolute tick, and source object location; they do not carry the complete StateRefV1 authority tuple.",
    "Complete StateRefV1 parent-occurrence, checkpoint-occurrence, semantic-world, control-history, RNG-state, event-prefix, and active-control binding must be proven through the Expedition child carrier before general BRANCH_LOCAL authority; this receipt additionally binds the compact per-world sparse event trajectory digest.",
    "It does not authorize a scientific conclusion, production use, Qualification, scheduling, or provider execution.",
    "The changed-control earliest divergence is bounded by retained checkpoint resolution.",
)


class SelectedWorldExactnessEvidenceError(ValueError):
    """A required exact identity is absent, mutable, or contradictory."""


@dataclass(frozen=True, slots=True)
class _LoadedJson:
    path: Path
    value: dict[str, Any]
    file_sha256: str
    size_bytes: int


@dataclass(frozen=True, slots=True)
class _ExecutionIdentity:
    run_root: Path
    request_sha256: str
    request_receipt_file_sha256: str
    campaign_spec_sha256: str
    plan_sha256: str
    execution_result_sha256: str
    execution_result_file_sha256: str
    lifecycle_manifest_sha256: str
    storage_receipt_sha256: str
    terminal_lifecycle_commit_sha256: str
    terminal_storage_receipt_sha256: str
    source_commit: str | None
    result: Mapping[str, Any]

    def projection(self) -> dict[str, Any]:
        return {
            "run_root": str(self.run_root),
            "execution_request_sha256": self.request_sha256,
            "execution_request_receipt_file_sha256": (self.request_receipt_file_sha256),
            "campaign_spec_sha256": self.campaign_spec_sha256,
            "compiled_plan_sha256": self.plan_sha256,
            "execution_result_sha256": self.execution_result_sha256,
            "execution_result_file_sha256": self.execution_result_file_sha256,
            "lifecycle_manifest_sha256": self.lifecycle_manifest_sha256,
            "terminal_lifecycle_commit_sha256": (self.terminal_lifecycle_commit_sha256),
            "terminal_storage_receipt_sha256": (self.terminal_storage_receipt_sha256),
            "source_commit": self.source_commit,
            "qualification_status": "NOT_REQUESTED",
        }


@dataclass(frozen=True, slots=True)
class _ParentEvidence:
    execution: _ExecutionIdentity
    world_uid: int
    branch_tick: int
    parent_run_id: str
    parent_occurrence_sha256: str
    checkpoint_occurrence_sha256: str
    semantic_world_sha256: str
    control_history_sha256: str
    selected_source_object_sha256: str
    exact_state_sha256: str
    rng_state_sha256: str
    event_prefix_sha256: str
    rng_epoch: int
    rng_key: int
    active_control_state_sha256: str
    source_index_manifest_path: Path
    source_index_manifest_file_sha256: str
    source_index_manifest_body_sha256: str
    source_world_state_set_sha256: str
    source_checkpoint_identity_sha256: str
    source_physical_state_path: Path
    source_physical_state_sha256: str
    source_physical_state_bytes: int
    source_physical_slot: int
    source_state_offset: int
    source_state_bytes: int

    def projection(self) -> dict[str, Any]:
        return {
            "execution": self.execution.projection(),
            "parent_run_id": self.parent_run_id,
            "world_uid": self.world_uid,
            "branch_next_absolute_tick": self.branch_tick,
            "parent_occurrence_sha256": self.parent_occurrence_sha256,
            "checkpoint_occurrence_sha256": self.checkpoint_occurrence_sha256,
            "semantic_world_sha256": self.semantic_world_sha256,
            "control_history_sha256": self.control_history_sha256,
            "selected_source_object_sha256": self.selected_source_object_sha256,
            "exact_state_sha256": self.exact_state_sha256,
            "rng_state_sha256": self.rng_state_sha256,
            "event_prefix_sha256": self.event_prefix_sha256,
            "rng_epoch": self.rng_epoch,
            "rng_key": self.rng_key,
            "active_control_state_sha256": self.active_control_state_sha256,
            "source_index_manifest": {
                "path": str(self.source_index_manifest_path),
                "file_sha256": self.source_index_manifest_file_sha256,
                "body_sha256": self.source_index_manifest_body_sha256,
                "world_state_set_sha256": self.source_world_state_set_sha256,
            },
            "source_physical_checkpoint": {
                "checkpoint_identity_sha256": (self.source_checkpoint_identity_sha256),
                "state_path": str(self.source_physical_state_path),
                "state_sha256": self.source_physical_state_sha256,
                "state_bytes": self.source_physical_state_bytes,
                "selected_physical_slot": self.source_physical_slot,
                "selected_state_offset": self.source_state_offset,
                "selected_state_bytes": self.source_state_bytes,
            },
        }


@dataclass(frozen=True, slots=True)
class _TickEvidence:
    tick: int
    occurrence_sha256: str
    occurrence_parent_sha256: str | None
    index_manifest_path: Path
    index_manifest_file_sha256: str
    index_manifest_body_sha256: str
    world_state_set_sha256: str
    world_count: int
    rng_epoch: int
    rng_key: int
    state_sha256: str
    state_bytes: int
    physical_state_path: Path
    state_offset: int
    event_trajectory_sha256: str

    def projection(self) -> dict[str, Any]:
        return {
            "next_absolute_tick": self.tick,
            "occurrence_sha256": self.occurrence_sha256,
            "occurrence_parent_sha256": self.occurrence_parent_sha256,
            "index_manifest_path": str(self.index_manifest_path),
            "index_manifest_file_sha256": self.index_manifest_file_sha256,
            "index_manifest_body_sha256": self.index_manifest_body_sha256,
            "world_state_set_sha256": self.world_state_set_sha256,
            "physical_world_count": self.world_count,
            "rng_epoch": self.rng_epoch,
            "rng_key": self.rng_key,
            "selected_state_sha256": self.state_sha256,
            "selected_state_bytes": self.state_bytes,
            "event_trajectory_sha256": self.event_trajectory_sha256,
        }


@dataclass(frozen=True, slots=True)
class _TrajectoryEvidence:
    label: str
    execution: _ExecutionIdentity
    branch_tick: int
    target_tick: int
    physical_B: int
    continuation_scope: str
    science_continuation_contract_sha256: str
    science_continuation_identity_sha256: str
    continuation_contract_sha256: str
    repack_binding_sha256: str
    attempt_id: str
    source_world_state_set_sha256: str
    source_selected_state_sha256: str
    source_rng_key: int
    source_checkpoint_identity_sha256: str
    source_index_manifest_path: Path
    source_index_manifest_file_sha256: str
    source_physical_state_path: Path
    source_physical_slot: int
    source_state_offset: int
    control_contract_sha256: str | None
    control_application_sha256: str | None
    applied_control_next_absolute_tick: int | None
    first_permitted_divergence_tick: int | None
    controls: Mapping[str, Any] | None
    ticks: tuple[_TickEvidence, ...]

    def projection(self) -> dict[str, Any]:
        return {
            "execution": self.execution.projection(),
            "branch_next_absolute_tick": self.branch_tick,
            "target_next_absolute_tick": self.target_tick,
            "physical_B": self.physical_B,
            "continuation_scope": self.continuation_scope,
            "science_continuation_contract_sha256": (
                self.science_continuation_contract_sha256
            ),
            "science_continuation_identity_sha256": (
                self.science_continuation_identity_sha256
            ),
            "continuation_contract_sha256": self.continuation_contract_sha256,
            "repack_binding_sha256": self.repack_binding_sha256,
            "attempt_id": self.attempt_id,
            "source_binding": {
                "world_state_set_sha256": self.source_world_state_set_sha256,
                "selected_state_sha256": self.source_selected_state_sha256,
                "rng_key": self.source_rng_key,
                "checkpoint_identity_sha256": (self.source_checkpoint_identity_sha256),
                "index_manifest_path": str(self.source_index_manifest_path),
                "index_manifest_file_sha256": (self.source_index_manifest_file_sha256),
                "physical_state_path": str(self.source_physical_state_path),
                "physical_slot": self.source_physical_slot,
                "state_offset": self.source_state_offset,
            },
            "control_binding": {
                "control_contract_sha256": self.control_contract_sha256,
                "control_application_sha256": self.control_application_sha256,
                "applied_next_absolute_tick": (self.applied_control_next_absolute_tick),
                "first_permitted_divergence_tick": (
                    self.first_permitted_divergence_tick
                ),
                "controls": None if self.controls is None else dict(self.controls),
            },
            "checkpoint_receipts": [row.projection() for row in self.ticks],
        }


def _fail(message: str) -> None:
    raise SelectedWorldExactnessEvidenceError(message)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _valid_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in _SHA256_CHARS for character in value)
    )


def _valid_git_commit(value: object) -> bool:
    return (
        type(value) is str
        and len(value) in {40, 64}
        and all(character in _SHA256_CHARS for character in value)
    )


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    try:
        return (
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise SelectedWorldExactnessEvidenceError(
            f"exactness evidence is not canonical JSON: {exc}"
        ) from exc


def _read_json(path: str | Path, *, field: str) -> _LoadedJson:
    requested = Path(path).expanduser()
    if requested.is_symlink():
        _fail(f"{field} must not be a symbolic link")
    try:
        resolved = requested.resolve(strict=True)
        before = os.lstat(resolved)
    except OSError as exc:
        raise SelectedWorldExactnessEvidenceError(f"{field} is unavailable") from exc
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_size <= 0
        or before.st_size > _MAX_JSON_BYTES
    ):
        _fail(f"{field} is not a bounded regular file")
    try:
        payload = resolved.read_bytes()
        after = os.lstat(resolved)
        value = json.loads(payload.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SelectedWorldExactnessEvidenceError(
            f"{field} cannot be read as JSON"
        ) from exc
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_mode,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_mode,
    )
    if identity_before != identity_after or len(payload) != before.st_size:
        _fail(f"{field} changed while being read")
    if type(value) is not dict:
        _fail(f"{field} must contain one JSON object")
    return _LoadedJson(
        path=resolved,
        value=dict(value),
        file_sha256=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
    )


def _root(path: str | Path, *, field: str) -> Path:
    requested = Path(path).expanduser()
    if requested.is_symlink():
        _fail(f"{field} must not be a symbolic link")
    try:
        resolved = requested.resolve(strict=True)
    except OSError as exc:
        raise SelectedWorldExactnessEvidenceError(f"{field} is unavailable") from exc
    if not resolved.is_dir() or resolved.is_symlink():
        _fail(f"{field} must be a regular directory")
    return resolved


def _bounded_file(path: str | Path, *, root: Path, field: str) -> Path:
    requested = Path(path).expanduser()
    if requested.is_symlink():
        _fail(f"{field} must not be a symbolic link")
    try:
        resolved = requested.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as exc:
        raise SelectedWorldExactnessEvidenceError(
            f"{field} leaves its exact run root or is unavailable"
        ) from exc
    if not resolved.is_file() or resolved.is_symlink():
        _fail(f"{field} must be a regular file")
    return resolved


def _reopen_execution_identity(run_root: str | Path) -> _ExecutionIdentity:
    root = _root(run_root, field="execution run root")
    request_file = _read_json(
        root / "execution_request.json", field="execution request receipt"
    )
    request_receipt = request_file.value
    expected_request_receipt_fields = {
        "schema_version",
        "request_sha256",
        "request",
        "run_root",
        "bound_plan_sha256",
        "source_identity",
        "created_at_utc",
    }
    request_value = request_receipt.get("request")
    if (
        set(request_receipt) != expected_request_receipt_fields
        or request_receipt.get("schema_version") != "eiel_execution_request_receipt_v2"
        or type(request_value) is not dict
        or request_receipt.get("run_root") != str(root)
    ):
        _fail("execution request receipt fields or run-root binding differ")
    try:
        request = ExecutionRequestV2.from_mapping(
            request_value, base_dir=request_file.path.parent
        )
    except Exception as exc:
        raise SelectedWorldExactnessEvidenceError(
            f"ExecutionRequestV2 cannot reopen: {exc}"
        ) from exc
    if (
        request.payload() != request_value
        or request.output_root != str(root)
        or request.qualification_profile is not None
        or request.request_sha256 != request_receipt.get("request_sha256")
    ):
        _fail("ExecutionRequestV2 identity or no-Qualification boundary differs")

    source = request_receipt.get("source_identity")
    if not isinstance(source, Mapping):
        _fail("execution request source identity is absent")
    campaign_path = Path(request.campaign_spec).resolve()
    if (
        campaign_path.is_symlink()
        or not campaign_path.is_file()
        or source.get("campaign_spec_path") != str(campaign_path)
        or source.get("campaign_spec_sha256") != _sha256_file(campaign_path)
    ):
        _fail("execution request campaign source identity changed")

    plan_file = _read_json(root / "execution_plan.json", field="execution plan")
    plan = plan_file.value
    supplied_plan = plan.get("plan_sha256")
    plan_body = {key: value for key, value in plan.items() if key != "plan_sha256"}
    if (
        supplied_plan != canonical_sha256(plan_body).hex
        or request_receipt.get("bound_plan_sha256") != supplied_plan
        or source.get("bound_plan_sha256") != supplied_plan
        or source.get("compiled_plan_sha256") != supplied_plan
    ):
        _fail("execution plan identity does not recompute")

    result_file = _read_json(root / "execution_result.json", field="execution result")
    result = result_file.value
    supplied_result = result.get("result_sha256")
    result_body = {
        key: value for key, value in result.items() if key != "result_sha256"
    }
    if (
        supplied_result != canonical_sha256(result_body).hex
        or result.get("schema_version") != "eiel_execution_result_v2"
        or result.get("request_sha256") != request.request_sha256
        or result.get("plan_sha256") != supplied_plan
        or result.get("compiled_plan_sha256") != supplied_plan
        or result.get("source_identity") != source
        or result.get("run_root") != str(root)
        or result.get("status") != "completed"
        or result.get("ok") is not True
        or result.get("result_class") != "SUCCESS"
        or result.get("qualification_evaluation") is not None
        or result.get("qualification_status") != "NOT_REQUESTED"
    ):
        _fail("execution result is not one exact terminal no-Qualification result")

    lifecycle_file = _read_json(
        root / "artifact_disposition_manifest.json",
        field="artifact disposition manifest",
    )
    try:
        lifecycle = validate_artifact_disposition_manifest_v1(lifecycle_file.value)
    except Exception as exc:
        raise SelectedWorldExactnessEvidenceError(
            f"artifact disposition manifest cannot reopen: {exc}"
        ) from exc
    if (
        lifecycle.get("runtime_root") != str(root)
        or lifecycle.get("plan_sha256") != supplied_plan
        or lifecycle.get("terminal_status") != "completed"
        or lifecycle.get("inventory", {}).get("complete") is not True
    ):
        _fail("terminal artifact lifecycle identity differs")
    storage_file = _read_json(
        root / "storage_disposition_receipt.json",
        field="storage disposition receipt",
    )
    storage = storage_file.value
    supplied_storage = storage.get("receipt_sha256")
    storage_body = {
        key: value for key, value in storage.items() if key != "receipt_sha256"
    }
    if (
        supplied_storage != canonical_sha256(storage_body).hex
        or storage.get("schema_version") != "eiel_storage_disposition_receipt_v1"
        or storage.get("manifest_sha256") != lifecycle.get("manifest_sha256")
        or storage.get("plan_sha256") != supplied_plan
    ):
        _fail("storage disposition receipt is not bound to the terminal run")
    commit_file = _read_json(
        root / "terminal_lifecycle_commit.json",
        field="terminal lifecycle commit",
    )
    commit = commit_file.value
    supplied_commit = commit.get("commit_sha256")
    commit_body = {
        key: value for key, value in commit.items() if key != "commit_sha256"
    }
    committed_manifest = commit.get("manifest")
    committed_storage = commit.get("storage_receipt")
    committed_result = commit.get("result")
    if (
        supplied_commit != canonical_sha256(commit_body).hex
        or commit.get("schema_version") != "eiel_execution_terminal_lifecycle_commit_v1"
        or commit.get("request_sha256") != request.request_sha256
        or commit.get("plan_sha256") != supplied_plan
        or commit.get("source_identity_sha256") != canonical_sha256(source).hex
        or not isinstance(committed_manifest, Mapping)
        or not isinstance(committed_storage, Mapping)
        or not isinstance(committed_result, Mapping)
        or committed_manifest.get("manifest_sha256") != lifecycle.get("manifest_sha256")
        or committed_manifest.get("file_sha256") != lifecycle_file.file_sha256
        or committed_result.get("result_sha256") != supplied_result
        or committed_result.get("file_sha256") != result_file.file_sha256
        or not _valid_sha256(committed_storage.get("receipt_sha256"))
        or not _valid_sha256(committed_storage.get("file_sha256"))
    ):
        _fail("terminal lifecycle commit does not bind the exact run")
    git = source.get("git")
    source_commit = git.get("commit") if isinstance(git, Mapping) else None
    if source_commit is not None and not _valid_git_commit(source_commit):
        _fail("execution source commit is not a supported Git identity")
    return _ExecutionIdentity(
        run_root=root,
        request_sha256=request.request_sha256,
        request_receipt_file_sha256=request_file.file_sha256,
        campaign_spec_sha256=str(source["campaign_spec_sha256"]),
        plan_sha256=str(supplied_plan),
        execution_result_sha256=str(supplied_result),
        execution_result_file_sha256=result_file.file_sha256,
        lifecycle_manifest_sha256=str(lifecycle["manifest_sha256"]),
        storage_receipt_sha256=str(supplied_storage),
        terminal_lifecycle_commit_sha256=str(supplied_commit),
        terminal_storage_receipt_sha256=str(committed_storage["receipt_sha256"]),
        source_commit=source_commit,
        result=result,
    )


def _reopen_parent(
    founder_run_root: str | Path, *, world_uid: int, branch_tick: int
) -> _ParentEvidence:
    execution = _reopen_execution_identity(founder_run_root)
    batch_manifest_path = _bounded_file(
        execution.run_root / "batches" / "batch-0000" / "manifest.json",
        root=execution.run_root,
        field="founder batch manifest",
    )
    batch = _read_json(batch_manifest_path, field="founder batch manifest").value
    checkpoint = batch.get("checkpoint")
    receipt = (
        checkpoint.get("acs_checkpoint_receipt")
        if isinstance(checkpoint, Mapping)
        else None
    )
    indexed_ref = (
        receipt.get("indexed_checkpoint") if isinstance(receipt, Mapping) else None
    )
    if (
        batch.get("status") != "completed"
        or batch.get("provider_id")
        != "eiel-full-evolution-cuda-workhorse-v2-provider-v1"
        or batch.get("plan_sha256") != execution.plan_sha256
        or not isinstance(indexed_ref, Mapping)
        or receipt.get("checkpoint_at_tick") != branch_tick
    ):
        _fail("founder terminal indexed checkpoint receipt is incomplete")

    # The selected-world package carries a path-bound occurrence projection.
    # Reauthentication must therefore use one stable destination identity;
    # random TemporaryDirectory names would make the otherwise exact manifest
    # digest change on every verification.  The directory is private scratch
    # and is removed after the bounded reopen.
    stable_key = hashlib.sha256(
        f"{execution.run_root}|{world_uid}|{branch_tick}".encode("utf-8")
    ).hexdigest()[:24]
    temp_root = Path(tempfile.gettempdir()) / (
        f"eiel-selected-exactness-parent-{stable_key}"
    )
    if temp_root.exists() or temp_root.is_symlink():
        if temp_root.is_symlink() or not temp_root.is_dir():
            _fail("selected exactness scratch destination is invalid")
        for child in sorted(
            temp_root.rglob("*"), key=lambda item: len(item.parts), reverse=True
        ):
            try:
                child.chmod(child.stat().st_mode | 0o700)
            except OSError:
                pass
        shutil.rmtree(temp_root)
    temp_root.mkdir(parents=True, mode=0o700)
    try:
        try:
            state_ref, _authority_ref, resolved = (
                materialize_workhorse_founder_exact_state_ref_v1(
                    batch_manifest_path,
                    temp_root / "selected-world",
                    world_uid=world_uid,
                    expedition_id="engineering-selected-world-exactness",
                    world_trajectory_id=f"workhorse-world-{world_uid}",
                )
            )
        except Exception as exc:
            raise SelectedWorldExactnessEvidenceError(
                f"founder exact parent cannot reopen: {exc}"
            ) from exc
        selected_source_object_sha256 = state_ref.source_object_sha256.hex
        active_controls = resolved.active_scientific_control_state
        if not isinstance(active_controls, Mapping):
            _fail("founder exact parent lacks active controls")
        active_control_state_sha256 = active_controls.get("active_control_state_sha256")
        if not _valid_sha256(active_control_state_sha256):
            _fail("founder active-control identity is invalid")
    finally:
        for child in sorted(
            temp_root.rglob("*"), key=lambda item: len(item.parts), reverse=True
        ):
            try:
                child.chmod(child.stat().st_mode | 0o700)
            except OSError:
                pass
        try:
            temp_root.chmod(temp_root.stat().st_mode | 0o700)
        except OSError:
            pass
        shutil.rmtree(temp_root, ignore_errors=True)

    index_path = _bounded_file(
        str(indexed_ref.get("manifest_path", "")),
        root=execution.run_root,
        field="founder terminal index manifest",
    )
    try:
        indexed, physical_state_path = inspect_workhorse_indexed_checkpoint_set_v1(
            index_path
        )
    except Exception as exc:
        raise SelectedWorldExactnessEvidenceError(
            f"founder terminal index cannot reopen: {exc}"
        ) from exc
    identity = indexed["world_state_set_identity"]
    rows = [
        row for row in indexed["world_index_by_uid"] if row["world_uid"] == world_uid
    ]
    source = indexed["source_physical_checkpoint"]
    if (
        len(rows) != 1
        or identity["checkpoint_at_tick"] != branch_tick
        or indexed_ref.get("manifest_file_sha256") != _sha256_file(index_path)
        or indexed_ref.get("manifest_body_sha256") != indexed["manifest_body_sha256"]
        or indexed_ref.get("world_state_set_identity") != identity
        or state_ref.logical_world_id != world_uid
        or state_ref.next_absolute_tick != branch_tick
        or state_ref.exact_state_sha256 is None
        or state_ref.rng_state_sha256 is None
        or state_ref.event_prefix_sha256 is None
        or state_ref.checkpoint_occurrence_sha256 is None
        or state_ref.parent_occurrence_sha256 is None
        or state_ref.exact_state_sha256.hex != rows[0]["state_sha256"]
        or resolved.runtime_logical_world_uid != world_uid
        or resolved.next_absolute_tick != branch_tick
    ):
        _fail("founder StateRef, occurrence, RNG, or indexed row differs")
    row = rows[0]
    return _ParentEvidence(
        execution=execution,
        world_uid=world_uid,
        branch_tick=branch_tick,
        parent_run_id=str(resolved.parent_run_id),
        parent_occurrence_sha256=state_ref.parent_occurrence_sha256.hex,
        checkpoint_occurrence_sha256=state_ref.checkpoint_occurrence_sha256.hex,
        semantic_world_sha256=state_ref.semantic_world_sha256.hex,
        control_history_sha256=state_ref.control_history_sha256.hex,
        selected_source_object_sha256=selected_source_object_sha256,
        exact_state_sha256=state_ref.exact_state_sha256.hex,
        rng_state_sha256=state_ref.rng_state_sha256.hex,
        event_prefix_sha256=state_ref.event_prefix_sha256.hex,
        rng_epoch=int(identity["rng_epoch"]),
        rng_key=int(row["rng_key"]),
        active_control_state_sha256=str(active_control_state_sha256),
        source_index_manifest_path=index_path,
        source_index_manifest_file_sha256=_sha256_file(index_path),
        source_index_manifest_body_sha256=str(indexed["manifest_body_sha256"]),
        source_world_state_set_sha256=str(identity["world_state_set_sha256"]),
        source_checkpoint_identity_sha256=str(source["checkpoint_identity_sha256"]),
        source_physical_state_path=physical_state_path.resolve(),
        source_physical_state_sha256=str(source["canonical_state_sha256"]),
        source_physical_state_bytes=int(source["canonical_state_bytes"]),
        source_physical_slot=int(row["source_physical_slot"]),
        source_state_offset=int(row["state_offset"]),
        source_state_bytes=int(row["state_bytes"]),
    )


def _reopen_occurrence_projection(
    raw: Mapping[str, Any], *, expected_tick: int, previous_occurrence: str | None
) -> tuple[dict[str, Any], str]:
    try:
        projection = _projection_v2(raw)
        descriptor_file = _read_json(
            str(projection["occurrence_descriptor_path"]),
            field="checkpoint occurrence descriptor",
        )
        commit_file = _read_json(
            str(projection["occurrence_commit_path"]),
            field="checkpoint occurrence commit",
        )
        descriptor = AbsoluteTickCheckpointOccurrenceV1.from_dict(descriptor_file.value)
        commit = AbsoluteTickCheckpointOccurrenceCommitV1.from_dict(commit_file.value)
    except Exception as exc:
        raise SelectedWorldExactnessEvidenceError(
            f"checkpoint occurrence cannot reopen: {exc}"
        ) from exc
    occurrence = descriptor.occurrence_digest().hex
    expected_parent = (
        None
        if previous_occurrence is None
        else {"algorithm": "sha256", "hex": previous_occurrence}
    )
    if (
        projection.get("checkpoint_at_tick") != expected_tick
        or descriptor.absolute_next_tick != expected_tick
        or projection.get("occurrence_digest")
        != {"algorithm": "sha256", "hex": occurrence}
        or projection.get("parent_checkpoint_digest") != expected_parent
        or descriptor.parent_checkpoint_digest
        != (
            None
            if previous_occurrence is None
            else ContentDigest("sha256", previous_occurrence)
        )
        or commit.occurrence_digest != descriptor.occurrence_digest()
        or projection.get("occurrence_descriptor_file_sha256")
        != descriptor_file.file_sha256
        or projection.get("occurrence_commit_file_sha256") != commit_file.file_sha256
    ):
        _fail("checkpoint occurrence chain, tick, or file identity differs")
    return projection, occurrence


def _reopen_trajectory(
    label: str,
    run_root: str | Path,
    *,
    parent: _ParentEvidence,
) -> _TrajectoryEvidence:
    execution = _reopen_execution_identity(run_root)
    batch_root = execution.run_root / "batches" / "batch-0000"
    result_path = _bounded_file(
        batch_root / "workhorse-indexed-continuation-result.json",
        root=execution.run_root,
        field=f"{label} continuation result",
    )
    terminal_path = _bounded_file(
        batch_root / "workhorse-terminalized-indexed-continuation-receipt.json",
        root=execution.run_root,
        field=f"{label} terminalized continuation receipt",
    )
    try:
        result = validate_workhorse_indexed_continuation_result_v2(result_path)
        continuation = validate_workhorse_indexed_continuation_contract_v2(
            result["continuation_contract"]
        )
        terminal = validate_workhorse_terminalized_indexed_continuation_receipt_v2(
            terminal_path
        )
    except Exception as exc:
        raise SelectedWorldExactnessEvidenceError(
            f"{label} Workhorse continuation cannot reopen: {exc}"
        ) from exc
    if result.get("outcome") != "CONTINUED":
        _fail(f"{label} continuation did not complete")
    logical = continuation["science_continuation_identity"]
    physical = continuation["physical_continuation_plan"]
    target_contract = physical["target_contract"]
    repack = continuation["repack_binding"]
    branch_tick = int(logical["start_next_tick"])
    target_tick = int(logical["target_next_tick"])
    if (
        branch_tick != parent.branch_tick
        or continuation.get("time_authority") != _TIME_AUTHORITY
        or logical.get("time_authority") != _TIME_AUTHORITY
        or target_contract.get("time_authority") != _TIME_AUTHORITY
        or result.get("absolute_frontiers")
        != {
            "start_next_tick": branch_tick,
            "target_next_tick": target_tick,
            "committed_next_tick": target_tick,
        }
    ):
        _fail(f"{label} absolute-tick frontier binding differs")

    dispatch_measurements = result.get("dispatch_measurements")
    if not isinstance(dispatch_measurements, list) or not dispatch_measurements:
        _fail(f"{label} event trajectory measurements are absent")
    event_by_end_tick: dict[int, Mapping[str, Any]] = {}
    for measurement in dispatch_measurements:
        if not isinstance(measurement, Mapping):
            _fail(f"{label} event trajectory measurement is malformed")
        end_tick = measurement.get("end_next_tick")
        sink = measurement.get("event_sink_receipt")
        if type(end_tick) is not int or not isinstance(sink, Mapping):
            _fail(f"{label} event trajectory measurement is incomplete")
        if end_tick in event_by_end_tick:
            _fail(f"{label} event trajectory measurement tick is duplicated")
        event_by_end_tick[end_tick] = sink

    batch_file = _read_json(
        batch_root / "manifest.json", field=f"{label} provider manifest"
    )
    batch = batch_file.value
    provider = batch.get("result")
    raw_receipts = (
        provider.get("indexed_checkpoint_receipts")
        if isinstance(provider, Mapping)
        else None
    )
    publications = terminal.get("indexed_checkpoint_publications")
    checkpoint_ticks = continuation.get("checkpoint_ticks")
    if (
        batch.get("schema_version") != "eiel_campaign_batch_manifest_v1"
        or batch.get("status") != "completed"
        or batch.get("provider_id")
        != "eiel-full-evolution-cuda-workhorse-v2-provider-v1"
        or batch.get("plan_sha256") != execution.plan_sha256
        or not isinstance(provider, Mapping)
        or Path(str(provider.get("execution_result_path", ""))).resolve() != result_path
        or provider.get("execution_result_file_sha256") != result["receipt_file_sha256"]
        or provider.get("execution_result_content_sha256")
        != result["receipt_content_sha256"]
        or not isinstance(raw_receipts, list)
        or not isinstance(publications, list)
        or not isinstance(checkpoint_ticks, list)
        or len(raw_receipts) != len(checkpoint_ticks)
        or len(publications) != len(checkpoint_ticks)
        or terminal.get("checkpoint_ticks") != checkpoint_ticks
    ):
        _fail(f"{label} provider/indexed publication chain is incomplete")
    terminal_binding = execution.result.get(
        "workhorse_terminalized_indexed_continuation"
    )
    if (
        not isinstance(terminal_binding, Mapping)
        or Path(str(terminal_binding.get("final_receipt_path", ""))).resolve()
        != terminal_path
        or terminal_binding.get("final_receipt_file_sha256")
        != terminal["receipt_file_sha256"]
        or terminal_binding.get("final_receipt_content_sha256")
        != terminal["receipt_content_sha256"]
    ):
        _fail(f"{label} ExecutionRequest result names another Workhorse receipt")

    tick_rows: list[_TickEvidence] = []
    previous_occurrence: str | None = None
    for raw, publication, tick in zip(
        raw_receipts, publications, checkpoint_ticks, strict=True
    ):
        if not isinstance(raw, Mapping) or not isinstance(publication, Mapping):
            _fail(f"{label} indexed checkpoint row is malformed")
        projection_raw = raw.get("checkpoint_occurrence_projection")
        indexed_raw = raw.get("indexed_checkpoint")
        if not isinstance(projection_raw, Mapping) or not isinstance(
            indexed_raw, Mapping
        ):
            _fail(f"{label} indexed checkpoint row lacks occurrence or index")
        projection, occurrence = _reopen_occurrence_projection(
            projection_raw,
            expected_tick=int(tick),
            previous_occurrence=previous_occurrence,
        )
        previous_occurrence = occurrence
        index_path = _bounded_file(
            str(indexed_raw.get("manifest_path", "")),
            root=execution.run_root,
            field=f"{label} indexed checkpoint manifest",
        )
        try:
            indexed, physical_state_path = inspect_workhorse_indexed_checkpoint_set_v1(
                index_path
            )
        except Exception as exc:
            raise SelectedWorldExactnessEvidenceError(
                f"{label} indexed checkpoint cannot reopen: {exc}"
            ) from exc
        identity = indexed["world_state_set_identity"]
        source = indexed["source_physical_checkpoint"]
        selected = [
            row
            for row in indexed["world_index_by_uid"]
            if row["world_uid"] == parent.world_uid
        ]
        if (
            len(selected) != 1
            or indexed_raw.get("manifest_file_sha256") != _sha256_file(index_path)
            or indexed_raw.get("manifest_body_sha256")
            != indexed["manifest_body_sha256"]
            or indexed_raw.get("world_state_set_identity") != identity
            or identity.get("checkpoint_at_tick") != tick
            or raw.get("checkpoint_at_tick") != tick
            or raw.get("canonical_state_sha256") != source.get("canonical_state_sha256")
            or raw.get("canonical_state_bytes") != source.get("canonical_state_bytes")
            or raw.get("bundle_body_sha256")
            != source.get("checkpoint_bundle_body_sha256")
            or publication.get("checkpoint_at_tick") != tick
            or publication.get("occurrence_digest") != occurrence
            or publication.get("checkpoint_occurrence_projection_content_sha256")
            != projection.get("projection_content_sha256")
            or publication.get("indexed_manifest_file_sha256")
            != _sha256_file(index_path)
            or publication.get("indexed_manifest_body_sha256")
            != indexed["manifest_body_sha256"]
            or publication.get("world_state_set_identity") != identity
            or publication.get("canonical_state_sha256")
            != source.get("canonical_state_sha256")
        ):
            _fail(f"{label} indexed checkpoint receipt/publication differs")
        row = selected[0]
        sink = event_by_end_tick.get(int(tick))
        if sink is None:
            _fail(f"{label} lacks an event trajectory receipt at tick {tick}")
        sink_dispatch = sink.get("dispatch")
        # Indexed continuation publishes the canonical event-sink receipt
        # directly.  Older fixtures wrapped the same fields under a
        # ``dispatch`` projection; accept the current receipt shape without
        # weakening any per-world digest or absolute-tick checks below.
        if sink_dispatch is None and all(
            key in sink
            for key in (
                "world_uids",
                "world_event_rows_sha256",
                "end_next_absolute_tick",
            )
        ):
            sink_dispatch = sink
        if not isinstance(sink_dispatch, Mapping):
            _fail(f"{label} event trajectory receipt lacks dispatch metadata")
        sink_uids = sink_dispatch.get("world_uids")
        sink_event_digests = sink_dispatch.get("world_event_rows_sha256")
        if (
            not isinstance(sink_uids, list)
            or parent.world_uid not in sink_uids
            or not isinstance(sink_event_digests, list)
            or len(sink_event_digests) != len(sink_uids)
        ):
            _fail(f"{label} event trajectory receipt lacks per-world digests")
        selected_index = sink_uids.index(parent.world_uid)
        event_trajectory_sha256 = sink_event_digests[selected_index]
        if not _valid_sha256(event_trajectory_sha256):
            _fail(f"{label} event trajectory digest is invalid")
        if sink_dispatch.get("end_next_absolute_tick") != tick:
            _fail(f"{label} event trajectory tick binding differs")
        tick_rows.append(
            _TickEvidence(
                tick=int(tick),
                occurrence_sha256=occurrence,
                occurrence_parent_sha256=(
                    None
                    if projection.get("parent_checkpoint_digest") is None
                    else str(projection["parent_checkpoint_digest"]["hex"])
                ),
                index_manifest_path=index_path,
                index_manifest_file_sha256=_sha256_file(index_path),
                index_manifest_body_sha256=str(indexed["manifest_body_sha256"]),
                world_state_set_sha256=str(identity["world_state_set_sha256"]),
                world_count=int(identity["world_count"]),
                rng_epoch=int(identity["rng_epoch"]),
                rng_key=int(row["rng_key"]),
                state_sha256=str(row["state_sha256"]),
                state_bytes=int(row["state_bytes"]),
                physical_state_path=physical_state_path.resolve(),
                state_offset=int(row["state_offset"]),
                event_trajectory_sha256=str(event_trajectory_sha256),
            )
        )

    selected_worlds = repack.get("selected_worlds_in_physical_order")
    source_locators = repack.get("source_locators")
    source_identity = repack.get("selected_world_state_set_identity")
    if not isinstance(selected_worlds, list) or not isinstance(source_locators, list):
        _fail(f"{label} repack source bindings are absent")
    selected_source = [
        row for row in selected_worlds if row.get("world_uid") == parent.world_uid
    ]
    locators = [
        row for row in source_locators if row.get("world_uid") == parent.world_uid
    ]
    if (
        len(selected_source) != 1
        or len(locators) != 1
        or not isinstance(source_identity, Mapping)
    ):
        _fail(f"{label} selected parent is absent or ambiguous")
    source_row = selected_source[0]
    locator = locators[0]
    if (
        repack.get("checkpoint_at_tick") != parent.branch_tick
        or source_row.get("state_sha256") != parent.exact_state_sha256
        or source_row.get("rng_key") != parent.rng_key
        or source_row.get("source_world_state_set_sha256")
        != parent.source_world_state_set_sha256
        or Path(str(locator.get("source_index_manifest_path", ""))).resolve()
        != parent.source_index_manifest_path
        or locator.get("source_index_manifest_file_sha256")
        != parent.source_index_manifest_file_sha256
        or locator.get("source_checkpoint_identity_sha256")
        != parent.source_checkpoint_identity_sha256
        or Path(str(locator.get("source_state_path", ""))).resolve()
        != parent.source_physical_state_path
        or locator.get("source_physical_slot") != parent.source_physical_slot
        or locator.get("source_state_offset") != parent.source_state_offset
        or locator.get("state_bytes") != parent.source_state_bytes
        or locator.get("state_sha256") != parent.exact_state_sha256
        or locator.get("rng_key") != parent.rng_key
        or source_identity.get("rng_epoch") != parent.rng_epoch
        or source_identity.get("checkpoint_at_tick") != parent.branch_tick
    ):
        _fail(f"{label} does not restore the exact selected parent source")

    control_contract = continuation.get("scientific_control_contract")
    control_application = result.get("control_application")
    if control_contract is None:
        if control_application is not None:
            _fail(f"{label} default-control continuation gained an application")
        control_digest = application_digest = None
        applied_tick = divergence_tick = None
        controls = None
    else:
        if not isinstance(control_contract, Mapping) or not isinstance(
            control_application, Mapping
        ):
            _fail(f"{label} changed-control binding is incomplete")
        control_digest = str(control_contract["control_contract_sha256"])
        application_digest = str(control_application["control_application_sha256"])
        applied_tick = int(control_application["applied_next_absolute_tick"])
        divergence_tick = int(control_application["first_permitted_divergence_tick"])
        controls = dict(control_contract["controls"])
    return _TrajectoryEvidence(
        label=label,
        execution=execution,
        branch_tick=branch_tick,
        target_tick=target_tick,
        physical_B=int(target_contract["B"]),
        continuation_scope=str(logical["continuation_scope"]),
        science_continuation_contract_sha256=str(
            logical["science_continuation_contract_sha256"]
        ),
        science_continuation_identity_sha256=str(
            logical["science_continuation_identity_sha256"]
        ),
        continuation_contract_sha256=str(continuation["continuation_contract_sha256"]),
        repack_binding_sha256=str(repack["repack_binding_sha256"]),
        attempt_id=str(terminal["canonical_terminal_binding"]["attempt_id"]),
        source_world_state_set_sha256=str(source_row["source_world_state_set_sha256"]),
        source_selected_state_sha256=str(source_row["state_sha256"]),
        source_rng_key=int(source_row["rng_key"]),
        source_checkpoint_identity_sha256=str(
            locator["source_checkpoint_identity_sha256"]
        ),
        source_index_manifest_path=Path(
            str(locator["source_index_manifest_path"])
        ).resolve(),
        source_index_manifest_file_sha256=str(
            locator["source_index_manifest_file_sha256"]
        ),
        source_physical_state_path=Path(str(locator["source_state_path"])).resolve(),
        source_physical_slot=int(locator["source_physical_slot"]),
        source_state_offset=int(locator["source_state_offset"]),
        control_contract_sha256=control_digest,
        control_application_sha256=application_digest,
        applied_control_next_absolute_tick=applied_tick,
        first_permitted_divergence_tick=divergence_tick,
        controls=controls,
        ticks=tuple(tick_rows),
    )


def _world_bytes(row: _TickEvidence) -> bytes:
    try:
        with row.physical_state_path.open("rb") as handle:
            handle.seek(row.state_offset)
            payload = handle.read(row.state_bytes)
    except OSError as exc:
        raise SelectedWorldExactnessEvidenceError(
            "indexed selected-world bytes cannot be reopened"
        ) from exc
    if (
        len(payload) != row.state_bytes
        or hashlib.sha256(payload).hexdigest() != row.state_sha256
    ):
        _fail("indexed selected-world bytes changed after receipt authentication")
    return payload


def _source_matches(parent: _ParentEvidence, row: _TrajectoryEvidence) -> bool:
    return (
        row.branch_tick == parent.branch_tick
        and row.source_world_state_set_sha256 == parent.source_world_state_set_sha256
        and row.source_selected_state_sha256 == parent.exact_state_sha256
        and row.source_rng_key == parent.rng_key
        and row.source_checkpoint_identity_sha256
        == parent.source_checkpoint_identity_sha256
        and row.source_index_manifest_path == parent.source_index_manifest_path
        and row.source_index_manifest_file_sha256
        == parent.source_index_manifest_file_sha256
        and row.source_physical_state_path == parent.source_physical_state_path
        and row.source_physical_slot == parent.source_physical_slot
        and row.source_state_offset == parent.source_state_offset
    )


def _assemble_receipt(
    *,
    parent: _ParentEvidence,
    reference: _TrajectoryEvidence,
    same_control: _TrajectoryEvidence,
    changed_control: _TrajectoryEvidence,
) -> dict[str, Any]:
    trajectories = (reference, same_control, changed_control)
    if any(not _source_matches(parent, row) for row in trajectories):
        _fail("one continuation does not bind the exact founder occurrence/state")
    if (
        reference.physical_B != 2
        or same_control.physical_B != 1
        or changed_control.physical_B != 1
        or reference.control_contract_sha256 is not None
        or same_control.control_contract_sha256 is not None
        or changed_control.control_contract_sha256 is None
        or changed_control.control_application_sha256 is None
        or changed_control.applied_control_next_absolute_tick != parent.branch_tick
        or changed_control.first_permitted_divergence_tick != parent.branch_tick
        or any(
            row.tick <= parent.branch_tick
            for trajectory in trajectories
            for row in trajectory.ticks
        )
    ):
        _fail("B2/B1 packing or changed-control branch-tick contract differs")
    reference_ticks = tuple(row.tick for row in reference.ticks)
    same_ticks = tuple(row.tick for row in same_control.ticks)
    if not reference_ticks or reference_ticks != same_ticks:
        _fail("same-control B2/B1 comparison tick coverage differs")
    reference_by_tick = {row.tick: row for row in reference.ticks}
    same_by_tick = {row.tick: row for row in same_control.ticks}
    equality_rows: list[dict[str, Any]] = []
    for tick in reference_ticks:
        left = reference_by_tick[tick]
        right = same_by_tick[tick]
        exact_equal = (
            left.state_sha256 == right.state_sha256
            and left.state_bytes == right.state_bytes
            and left.rng_epoch == right.rng_epoch == parent.rng_epoch
            and left.rng_key == right.rng_key == parent.rng_key
            and left.event_trajectory_sha256 == right.event_trajectory_sha256
            and _world_bytes(left) == _world_bytes(right)
        )
        if not exact_equal:
            _fail(f"same-control B2/B1 selected state differs at tick {tick}")
        equality_rows.append(
            {
                "next_absolute_tick": tick,
                "state_sha256": left.state_sha256,
                "state_bytes": left.state_bytes,
                "rng_epoch": left.rng_epoch,
                "rng_key": left.rng_key,
                "event_trajectory_sha256": left.event_trajectory_sha256,
                "byte_equal": True,
            }
        )

    changed_by_tick = {row.tick: row for row in changed_control.ticks}
    common_ticks = sorted(set(same_by_tick) & set(changed_by_tick))
    divergent_ticks = [
        tick
        for tick in common_ticks
        if same_by_tick[tick].state_sha256 != changed_by_tick[tick].state_sha256
        or same_by_tick[tick].event_trajectory_sha256
        != changed_by_tick[tick].event_trajectory_sha256
        or _world_bytes(same_by_tick[tick]) != _world_bytes(changed_by_tick[tick])
    ]
    if not common_ticks or not divergent_ticks:
        _fail("changed-control child has no observed divergence from same-control")
    first_divergence = divergent_ticks[0]
    if (
        first_divergence < parent.branch_tick
        or changed_control.first_permitted_divergence_tick is None
        or first_divergence < changed_control.first_permitted_divergence_tick
    ):
        _fail("changed-control divergence precedes the declared branch tick")
    execution_ids = {row.execution.request_sha256 for row in trajectories}
    science_ids = {row.science_continuation_identity_sha256 for row in trajectories}
    attempt_ids = {row.attempt_id for row in trajectories}
    if len(execution_ids) != 3 or len(science_ids) != 3 or len(attempt_ids) != 3:
        _fail(
            "changed branch aliases another child request, science identity, or attempt"
        )
    science_contracts = {
        row.science_continuation_contract_sha256 for row in trajectories
    }
    if len(science_contracts) != 1:
        _fail("compared trajectories use another Workhorse science contract")

    implementation_path = Path(__file__).resolve()
    body: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA,
        "claim_class": RECEIPT_CLAIM_CLASS,
        "decision": RECEIPT_DECISION,
        "carrier_identity_scope": "LOW_LEVEL_INDEXED_SELECTED_STATE_ONLY",
        "time_authority": _TIME_AUTHORITY,
        "science_authority_granted": False,
        "qualification_authority_granted": False,
        "production_authority_granted": False,
        "named_phenotype_reward_authority_granted": False,
        "validator": {
            "module": "eiel_composition.adaptive_expedition.selected_world_exactness_evidence",
            "implementation_sha256": _sha256_file(implementation_path),
        },
        "parent": parent.projection(),
        "trajectories": {
            "reference_b2_same_control": reference.projection(),
            "repacked_b1_same_control": same_control.projection(),
            "repacked_b1_changed_control": changed_control.projection(),
        },
        "same_control_repack_exactness": {
            "source_B": 2,
            "target_B": 1,
            "packing_fields_excluded_from_scientific_identity": True,
            "all_compared_post_branch_ticks_byte_equal": True,
            "compared_ticks": equality_rows,
        },
        "changed_control_divergence": {
            "shared_exact_parent_at_branch_tick": True,
            "declared_branch_next_absolute_tick": parent.branch_tick,
            "control_applied_next_absolute_tick": (
                changed_control.applied_control_next_absolute_tick
            ),
            "first_permitted_divergence_tick": (
                changed_control.first_permitted_divergence_tick
            ),
            "first_observed_divergence_tick": first_divergence,
            "compared_ticks": common_ticks,
            "divergent_ticks": divergent_ticks,
            "child_execution_request_identity_distinct": True,
            "child_science_continuation_identity_distinct": True,
            "child_attempt_identity_distinct": True,
            "changed_branch_is_new_child_not_retry": True,
        },
        "request_identity_binding": {
            "founder_execution_request_sha256": parent.execution.request_sha256,
            "reference_execution_request_sha256": reference.execution.request_sha256,
            "same_control_execution_request_sha256": (
                same_control.execution.request_sha256
            ),
            "changed_control_execution_request_sha256": (
                changed_control.execution.request_sha256
            ),
            "workhorse_science_continuation_contract_sha256": next(
                iter(science_contracts)
            ),
            "reference_science_continuation_identity_sha256": (
                reference.science_continuation_identity_sha256
            ),
            "same_control_science_continuation_identity_sha256": (
                same_control.science_continuation_identity_sha256
            ),
            "changed_control_science_continuation_identity_sha256": (
                changed_control.science_continuation_identity_sha256
            ),
        },
        "checks": {
            "parent_state_ref_reopened_and_authenticated": True,
            "physical_selected_source_matches_parent_state": True,
            "all_indexed_checkpoint_receipts_reopened": True,
            "same_control_b2_to_b1_repack_equal_at_every_compared_tick": True,
            "same_control_event_trajectory_equal_at_every_compared_tick": True,
            "changed_control_divergence_not_before_declared_branch_tick": True,
            "logical_to_physical_repacking_nonsemantic_for_same_control": True,
            "absolute_ticks_only": True,
            "science_and_execution_request_identities_bound": True,
            "low_level_carrier_scope_explicit": True,
        },
        "limitations": list(_LIMITATIONS),
    }
    return {**body, "receipt_content_sha256": canonical_sha256(body).hex}


def _exact_mapping(
    value: object, *, fields: set[str] | frozenset[str], field: str
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        _fail(f"{field} fields differ")
    return value


def _sha_field(value: Mapping[str, Any], key: str, *, field: str) -> str:
    item = value.get(key)
    if not _valid_sha256(item):
        _fail(f"{field} {key} is not an exact SHA-256 identity")
    return str(item)


def _int_field(
    value: Mapping[str, Any], key: str, *, minimum: int, field: str
) -> int:
    item = value.get(key)
    if type(item) is not int or item < minimum:
        _fail(f"{field} {key} is not a bounded integer")
    return item


def _string_field(value: Mapping[str, Any], key: str, *, field: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item:
        _fail(f"{field} {key} is not a nonempty string")
    return item


def _reject_embedded_authority(value: object, *, field: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            child = f"{field}.{key}"
            if key.endswith("_authority_granted") and item is not False:
                _fail(f"{child} grants forbidden authority")
            if key == "qualification_status" and item != "NOT_REQUESTED":
                _fail(f"{child} is not the no-Qualification status")
            if key in {
                "scientific_claim",
                "science_claim",
                "qualification_claim",
                "qualification_evaluation",
            } and not (
                item is None
                or item is False
                or item == "NONE"
                or item == "NOT_REQUESTED"
            ):
                _fail(f"{child} contains a forbidden claim")
            _reject_embedded_authority(item, field=child)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_embedded_authority(item, field=f"{field}[{index}]")


def _validate_execution_projection(
    value: object, *, field: str
) -> Mapping[str, Any]:
    execution = _exact_mapping(
        value,
        fields={
            "run_root",
            "execution_request_sha256",
            "execution_request_receipt_file_sha256",
            "campaign_spec_sha256",
            "compiled_plan_sha256",
            "execution_result_sha256",
            "execution_result_file_sha256",
            "lifecycle_manifest_sha256",
            "terminal_lifecycle_commit_sha256",
            "terminal_storage_receipt_sha256",
            "source_commit",
            "qualification_status",
        },
        field=field,
    )
    root = _string_field(execution, "run_root", field=field)
    if not Path(root).is_absolute():
        _fail(f"{field} run_root is not absolute")
    for key in (
        "execution_request_sha256",
        "execution_request_receipt_file_sha256",
        "campaign_spec_sha256",
        "compiled_plan_sha256",
        "execution_result_sha256",
        "execution_result_file_sha256",
        "lifecycle_manifest_sha256",
        "terminal_lifecycle_commit_sha256",
        "terminal_storage_receipt_sha256",
    ):
        _sha_field(execution, key, field=field)
    source_commit = execution.get("source_commit")
    if source_commit is not None and not _valid_git_commit(source_commit):
        _fail(f"{field} source_commit is invalid")
    if execution.get("qualification_status") != "NOT_REQUESTED":
        _fail(f"{field} gained Qualification status")
    return execution


def _validate_parent_projection(value: object) -> Mapping[str, Any]:
    parent = _exact_mapping(
        value,
        fields={
            "execution",
            "parent_run_id",
            "world_uid",
            "branch_next_absolute_tick",
            "parent_occurrence_sha256",
            "checkpoint_occurrence_sha256",
            "semantic_world_sha256",
            "control_history_sha256",
            "selected_source_object_sha256",
            "exact_state_sha256",
            "rng_state_sha256",
            "event_prefix_sha256",
            "rng_epoch",
            "rng_key",
            "active_control_state_sha256",
            "source_index_manifest",
            "source_physical_checkpoint",
        },
        field="parent",
    )
    _validate_execution_projection(parent.get("execution"), field="parent.execution")
    _string_field(parent, "parent_run_id", field="parent")
    _int_field(parent, "world_uid", minimum=1, field="parent")
    _int_field(parent, "branch_next_absolute_tick", minimum=0, field="parent")
    _int_field(parent, "rng_epoch", minimum=0, field="parent")
    _int_field(parent, "rng_key", minimum=0, field="parent")
    for key in (
        "parent_occurrence_sha256",
        "checkpoint_occurrence_sha256",
        "semantic_world_sha256",
        "control_history_sha256",
        "selected_source_object_sha256",
        "exact_state_sha256",
        "rng_state_sha256",
        "event_prefix_sha256",
        "active_control_state_sha256",
    ):
        _sha_field(parent, key, field="parent")
    index = _exact_mapping(
        parent.get("source_index_manifest"),
        fields={
            "path",
            "file_sha256",
            "body_sha256",
            "world_state_set_sha256",
        },
        field="parent.source_index_manifest",
    )
    _string_field(index, "path", field="parent.source_index_manifest")
    for key in ("file_sha256", "body_sha256", "world_state_set_sha256"):
        _sha_field(index, key, field="parent.source_index_manifest")
    physical = _exact_mapping(
        parent.get("source_physical_checkpoint"),
        fields={
            "checkpoint_identity_sha256",
            "state_path",
            "state_sha256",
            "state_bytes",
            "selected_physical_slot",
            "selected_state_offset",
            "selected_state_bytes",
        },
        field="parent.source_physical_checkpoint",
    )
    _string_field(physical, "state_path", field="parent.source_physical_checkpoint")
    for key in ("checkpoint_identity_sha256", "state_sha256"):
        _sha_field(physical, key, field="parent.source_physical_checkpoint")
    state_bytes = _int_field(
        physical, "state_bytes", minimum=1, field="parent.source_physical_checkpoint"
    )
    _int_field(
        physical,
        "selected_physical_slot",
        minimum=0,
        field="parent.source_physical_checkpoint",
    )
    offset = _int_field(
        physical,
        "selected_state_offset",
        minimum=0,
        field="parent.source_physical_checkpoint",
    )
    selected_bytes = _int_field(
        physical,
        "selected_state_bytes",
        minimum=1,
        field="parent.source_physical_checkpoint",
    )
    if offset + selected_bytes > state_bytes:
        _fail("parent selected-state bounds exceed the physical checkpoint")
    return parent


def _validate_tick_projection(
    value: object,
    *,
    field: str,
    branch_tick: int,
    previous_occurrence: str | None,
) -> Mapping[str, Any]:
    row = _exact_mapping(
        value,
        fields={
            "next_absolute_tick",
            "occurrence_sha256",
            "occurrence_parent_sha256",
            "index_manifest_path",
            "index_manifest_file_sha256",
            "index_manifest_body_sha256",
            "world_state_set_sha256",
            "physical_world_count",
            "rng_epoch",
            "rng_key",
            "selected_state_sha256",
            "selected_state_bytes",
            "event_trajectory_sha256",
        },
        field=field,
    )
    tick = _int_field(row, "next_absolute_tick", minimum=0, field=field)
    if tick <= branch_tick:
        _fail(f"{field} is not post-branch")
    occurrence = _sha_field(row, "occurrence_sha256", field=field)
    parent = row.get("occurrence_parent_sha256")
    if parent != previous_occurrence:
        _fail(f"{field} occurrence chain differs")
    _string_field(row, "index_manifest_path", field=field)
    for key in (
        "index_manifest_file_sha256",
        "index_manifest_body_sha256",
        "world_state_set_sha256",
        "selected_state_sha256",
        "event_trajectory_sha256",
    ):
        _sha_field(row, key, field=field)
    _int_field(row, "physical_world_count", minimum=1, field=field)
    _int_field(row, "rng_epoch", minimum=0, field=field)
    _int_field(row, "rng_key", minimum=0, field=field)
    _int_field(row, "selected_state_bytes", minimum=1, field=field)
    if occurrence == previous_occurrence:
        _fail(f"{field} occurrence does not advance")
    return row


def _validate_trajectory_projection(
    value: object,
    *,
    label: str,
    parent: Mapping[str, Any],
) -> Mapping[str, Any]:
    field = f"trajectories.{label}"
    trajectory = _exact_mapping(
        value,
        fields={
            "execution",
            "branch_next_absolute_tick",
            "target_next_absolute_tick",
            "physical_B",
            "continuation_scope",
            "science_continuation_contract_sha256",
            "science_continuation_identity_sha256",
            "continuation_contract_sha256",
            "repack_binding_sha256",
            "attempt_id",
            "source_binding",
            "control_binding",
            "checkpoint_receipts",
        },
        field=field,
    )
    _validate_execution_projection(trajectory.get("execution"), field=f"{field}.execution")
    branch_tick = _int_field(
        trajectory, "branch_next_absolute_tick", minimum=0, field=field
    )
    target_tick = _int_field(
        trajectory, "target_next_absolute_tick", minimum=0, field=field
    )
    if branch_tick != parent["branch_next_absolute_tick"] or target_tick <= branch_tick:
        _fail(f"{field} absolute frontier differs from the parent")
    _int_field(trajectory, "physical_B", minimum=1, field=field)
    _string_field(trajectory, "continuation_scope", field=field)
    _string_field(trajectory, "attempt_id", field=field)
    for key in (
        "science_continuation_contract_sha256",
        "science_continuation_identity_sha256",
        "continuation_contract_sha256",
        "repack_binding_sha256",
    ):
        _sha_field(trajectory, key, field=field)
    source = _exact_mapping(
        trajectory.get("source_binding"),
        fields={
            "world_state_set_sha256",
            "selected_state_sha256",
            "rng_key",
            "checkpoint_identity_sha256",
            "index_manifest_path",
            "index_manifest_file_sha256",
            "physical_state_path",
            "physical_slot",
            "state_offset",
        },
        field=f"{field}.source_binding",
    )
    for key in (
        "world_state_set_sha256",
        "selected_state_sha256",
        "checkpoint_identity_sha256",
        "index_manifest_file_sha256",
    ):
        _sha_field(source, key, field=f"{field}.source_binding")
    for key in ("index_manifest_path", "physical_state_path"):
        _string_field(source, key, field=f"{field}.source_binding")
    for key in ("rng_key", "physical_slot", "state_offset"):
        _int_field(source, key, minimum=0, field=f"{field}.source_binding")
    parent_index = parent["source_index_manifest"]
    parent_physical = parent["source_physical_checkpoint"]
    expected_source = {
        "world_state_set_sha256": parent_index["world_state_set_sha256"],
        "selected_state_sha256": parent["exact_state_sha256"],
        "rng_key": parent["rng_key"],
        "checkpoint_identity_sha256": parent_physical[
            "checkpoint_identity_sha256"
        ],
        "index_manifest_path": parent_index["path"],
        "index_manifest_file_sha256": parent_index["file_sha256"],
        "physical_state_path": parent_physical["state_path"],
        "physical_slot": parent_physical["selected_physical_slot"],
        "state_offset": parent_physical["selected_state_offset"],
    }
    if dict(source) != expected_source:
        _fail(f"{field} source binding differs from the exact parent")
    control = _exact_mapping(
        trajectory.get("control_binding"),
        fields={
            "control_contract_sha256",
            "control_application_sha256",
            "applied_next_absolute_tick",
            "first_permitted_divergence_tick",
            "controls",
        },
        field=f"{field}.control_binding",
    )
    changed = label == "repacked_b1_changed_control"
    if changed:
        _sha_field(control, "control_contract_sha256", field=f"{field}.control_binding")
        _sha_field(
            control, "control_application_sha256", field=f"{field}.control_binding"
        )
        if (
            control.get("applied_next_absolute_tick") != branch_tick
            or control.get("first_permitted_divergence_tick") != branch_tick
            or not isinstance(control.get("controls"), Mapping)
            or not control["controls"]
        ):
            _fail(f"{field} changed-control frontier or bounded controls differ")
    elif any(item is not None for item in control.values()):
        _fail(f"{field} same-control child gained a control application")
    receipts = trajectory.get("checkpoint_receipts")
    if not isinstance(receipts, list) or not receipts:
        _fail(f"{field} checkpoint receipts are absent")
    previous: str | None = None
    ticks: list[int] = []
    for index, raw in enumerate(receipts):
        row = _validate_tick_projection(
            raw,
            field=f"{field}.checkpoint_receipts[{index}]",
            branch_tick=branch_tick,
            previous_occurrence=previous,
        )
        ticks.append(int(row["next_absolute_tick"]))
        previous = str(row["occurrence_sha256"])
    if ticks != sorted(set(ticks)) or ticks[-1] != target_tick:
        _fail(f"{field} checkpoint ticks are not a strict terminal frontier")
    return trajectory


def _validate_receipt_relations(value: Mapping[str, Any]) -> None:
    parent = _validate_parent_projection(value.get("parent"))
    trajectories = _exact_mapping(
        value.get("trajectories"),
        fields=set(_TRAJECTORY_LABELS),
        field="trajectories",
    )
    rows = {
        label: _validate_trajectory_projection(
            trajectories[label], label=label, parent=parent
        )
        for label in _TRAJECTORY_LABELS
    }
    reference = rows["reference_b2_same_control"]
    same = rows["repacked_b1_same_control"]
    changed = rows["repacked_b1_changed_control"]
    if (
        reference["physical_B"] != 2
        or same["physical_B"] != 1
        or changed["physical_B"] != 1
    ):
        _fail("receipt B2/B1 physical packing relation differs")
    execution_ids = {
        row["execution"]["execution_request_sha256"] for row in rows.values()
    }
    science_ids = {
        row["science_continuation_identity_sha256"] for row in rows.values()
    }
    attempts = {row["attempt_id"] for row in rows.values()}
    science_contracts = {
        row["science_continuation_contract_sha256"] for row in rows.values()
    }
    if (
        len(execution_ids) != 3
        or len(science_ids) != 3
        or len(attempts) != 3
        or len(science_contracts) != 1
    ):
        _fail("receipt child, attempt, or science identities alias")

    reference_ticks = reference["checkpoint_receipts"]
    same_ticks = same["checkpoint_receipts"]
    if [row["next_absolute_tick"] for row in reference_ticks] != [
        row["next_absolute_tick"] for row in same_ticks
    ]:
        _fail("receipt same-control comparison frontier differs")
    expected_equal: list[dict[str, Any]] = []
    for left, right in zip(reference_ticks, same_ticks, strict=True):
        comparable = (
            "selected_state_sha256",
            "selected_state_bytes",
            "rng_epoch",
            "rng_key",
            "event_trajectory_sha256",
        )
        if any(left[key] != right[key] for key in comparable):
            _fail("receipt same-control B2/B1 state relation differs")
        expected_equal.append(
            {
                "next_absolute_tick": left["next_absolute_tick"],
                "state_sha256": left["selected_state_sha256"],
                "state_bytes": left["selected_state_bytes"],
                "rng_epoch": left["rng_epoch"],
                "rng_key": left["rng_key"],
                "event_trajectory_sha256": left["event_trajectory_sha256"],
                "byte_equal": True,
            }
        )
    exactness = _exact_mapping(
        value.get("same_control_repack_exactness"),
        fields={
            "source_B",
            "target_B",
            "packing_fields_excluded_from_scientific_identity",
            "all_compared_post_branch_ticks_byte_equal",
            "compared_ticks",
        },
        field="same_control_repack_exactness",
    )
    if exactness != {
        "source_B": 2,
        "target_B": 1,
        "packing_fields_excluded_from_scientific_identity": True,
        "all_compared_post_branch_ticks_byte_equal": True,
        "compared_ticks": expected_equal,
    }:
        _fail("receipt same-control exactness summary does not recompute")

    same_by_tick = {
        row["next_absolute_tick"]: row for row in same["checkpoint_receipts"]
    }
    changed_by_tick = {
        row["next_absolute_tick"]: row for row in changed["checkpoint_receipts"]
    }
    common = sorted(set(same_by_tick) & set(changed_by_tick))
    divergent = [
        tick
        for tick in common
        if same_by_tick[tick]["selected_state_sha256"]
        != changed_by_tick[tick]["selected_state_sha256"]
        or same_by_tick[tick]["event_trajectory_sha256"]
        != changed_by_tick[tick]["event_trajectory_sha256"]
    ]
    if not common or not divergent:
        _fail("receipt changed-control divergence does not recompute")
    branch_tick = int(parent["branch_next_absolute_tick"])
    first = divergent[0]
    divergence = _exact_mapping(
        value.get("changed_control_divergence"),
        fields={
            "shared_exact_parent_at_branch_tick",
            "declared_branch_next_absolute_tick",
            "control_applied_next_absolute_tick",
            "first_permitted_divergence_tick",
            "first_observed_divergence_tick",
            "compared_ticks",
            "divergent_ticks",
            "child_execution_request_identity_distinct",
            "child_science_continuation_identity_distinct",
            "child_attempt_identity_distinct",
            "changed_branch_is_new_child_not_retry",
        },
        field="changed_control_divergence",
    )
    expected_divergence = {
        "shared_exact_parent_at_branch_tick": True,
        "declared_branch_next_absolute_tick": branch_tick,
        "control_applied_next_absolute_tick": branch_tick,
        "first_permitted_divergence_tick": branch_tick,
        "first_observed_divergence_tick": first,
        "compared_ticks": common,
        "divergent_ticks": divergent,
        "child_execution_request_identity_distinct": True,
        "child_science_continuation_identity_distinct": True,
        "child_attempt_identity_distinct": True,
        "changed_branch_is_new_child_not_retry": True,
    }
    if dict(divergence) != expected_divergence or first < branch_tick:
        _fail("receipt changed-control divergence summary does not recompute")

    binding = _exact_mapping(
        value.get("request_identity_binding"),
        fields={
            "founder_execution_request_sha256",
            "reference_execution_request_sha256",
            "same_control_execution_request_sha256",
            "changed_control_execution_request_sha256",
            "workhorse_science_continuation_contract_sha256",
            "reference_science_continuation_identity_sha256",
            "same_control_science_continuation_identity_sha256",
            "changed_control_science_continuation_identity_sha256",
        },
        field="request_identity_binding",
    )
    expected_binding = {
        "founder_execution_request_sha256": parent["execution"][
            "execution_request_sha256"
        ],
        "reference_execution_request_sha256": reference["execution"][
            "execution_request_sha256"
        ],
        "same_control_execution_request_sha256": same["execution"][
            "execution_request_sha256"
        ],
        "changed_control_execution_request_sha256": changed["execution"][
            "execution_request_sha256"
        ],
        "workhorse_science_continuation_contract_sha256": next(
            iter(science_contracts)
        ),
        "reference_science_continuation_identity_sha256": reference[
            "science_continuation_identity_sha256"
        ],
        "same_control_science_continuation_identity_sha256": same[
            "science_continuation_identity_sha256"
        ],
        "changed_control_science_continuation_identity_sha256": changed[
            "science_continuation_identity_sha256"
        ],
    }
    if dict(binding) != expected_binding:
        _fail("receipt request identity summary does not recompute")
    checks = _exact_mapping(value.get("checks"), fields=_CHECK_NAMES, field="checks")
    if any(item is not True for item in checks.values()):
        _fail("receipt checks are not all satisfied")


def _reauthenticate_receipt_sources_v1(value: Mapping[str, Any]) -> None:
    parent_projection = value["parent"]
    trajectories = value["trajectories"]
    parent = _reopen_parent(
        parent_projection["execution"]["run_root"],
        world_uid=int(parent_projection["world_uid"]),
        branch_tick=int(parent_projection["branch_next_absolute_tick"]),
    )
    reference = _reopen_trajectory(
        "reference_b2_same_control",
        trajectories["reference_b2_same_control"]["execution"]["run_root"],
        parent=parent,
    )
    same = _reopen_trajectory(
        "repacked_b1_same_control",
        trajectories["repacked_b1_same_control"]["execution"]["run_root"],
        parent=parent,
    )
    changed = _reopen_trajectory(
        "repacked_b1_changed_control",
        trajectories["repacked_b1_changed_control"]["execution"]["run_root"],
        parent=parent,
    )
    reconstructed = _assemble_receipt(
        parent=parent,
        reference=reference,
        same_control=same,
        changed_control=changed,
    )
    if reconstructed != dict(value):
        _fail("sealed receipt differs from reauthenticated canonical source roots")


def validate_selected_world_exactness_receipt_v1(
    receipt_path: str | Path,
    *,
    reauthenticate_sources: bool = True,
) -> dict[str, Any]:
    """Recompute relations and, by default, reauthenticate canonical roots."""

    loaded = _read_json(receipt_path, field="selected-world exactness receipt")
    value = loaded.value
    supplied = value.get("receipt_content_sha256")
    body = {key: item for key, item in value.items() if key != "receipt_content_sha256"}
    required = {
        "schema_version",
        "claim_class",
        "decision",
        "carrier_identity_scope",
        "time_authority",
        "science_authority_granted",
        "qualification_authority_granted",
        "production_authority_granted",
        "named_phenotype_reward_authority_granted",
        "validator",
        "parent",
        "trajectories",
        "same_control_repack_exactness",
        "changed_control_divergence",
        "request_identity_binding",
        "checks",
        "limitations",
        "receipt_content_sha256",
    }
    validator = _exact_mapping(
        value.get("validator"),
        fields={"module", "implementation_sha256"},
        field="validator",
    )
    if (
        set(value) != required
        or value.get("schema_version") != RECEIPT_SCHEMA
        or value.get("claim_class") != RECEIPT_CLAIM_CLASS
        or value.get("decision") != RECEIPT_DECISION
        or value.get("carrier_identity_scope")
        != "LOW_LEVEL_INDEXED_SELECTED_STATE_ONLY"
        or value.get("time_authority") != _TIME_AUTHORITY
        or validator.get("module")
        != "eiel_composition.adaptive_expedition.selected_world_exactness_evidence"
        or not _valid_sha256(validator.get("implementation_sha256"))
        or validator.get("implementation_sha256")
        != _sha256_file(Path(__file__).resolve())
        or value.get("limitations") != list(_LIMITATIONS)
        or any(
            value.get(field) is not False
            for field in (
                "science_authority_granted",
                "qualification_authority_granted",
                "production_authority_granted",
                "named_phenotype_reward_authority_granted",
            )
        )
        or supplied != canonical_sha256(body).hex
        or not _valid_sha256(supplied)
        or loaded.path.stat().st_mode & 0o222
        or loaded.path.read_bytes() != _canonical_bytes(value)
    ):
        _fail("selected-world exactness receipt schema, seal, or authority differs")
    _reject_embedded_authority(value, field="receipt")
    _validate_receipt_relations(value)
    if reauthenticate_sources:
        _reauthenticate_receipt_sources_v1(value)
    return {
        **value,
        "receipt_path": str(loaded.path),
        "receipt_file_sha256": loaded.file_sha256,
        "receipt_file_bytes": loaded.size_bytes,
    }


def _seal_receipt(path: str | Path, value: Mapping[str, Any]) -> dict[str, Any]:
    requested = Path(path).expanduser()
    requested.parent.mkdir(parents=True, exist_ok=True)
    if requested.parent.is_symlink():
        _fail("exactness receipt parent must not be a symbolic link")
    output = requested.parent.resolve() / requested.name
    payload = _canonical_bytes(value)
    if output.exists() or output.is_symlink():
        existing = validate_selected_world_exactness_receipt_v1(
            output, reauthenticate_sources=False
        )
        comparable = {
            key: item
            for key, item in existing.items()
            if key not in {"receipt_path", "receipt_file_sha256", "receipt_file_bytes"}
        }
        if comparable != dict(value):
            _fail("existing exactness receipt has another identity")
        return existing
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(output, flags, 0o444)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)
    output.chmod(0o444)
    directory = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return validate_selected_world_exactness_receipt_v1(
        output, reauthenticate_sources=False
    )


def prove_selected_world_exactness_v1(
    *,
    founder_run_root: str | Path,
    reference_b2_run_root: str | Path,
    same_control_b1_run_root: str | Path,
    changed_control_b1_run_root: str | Path,
    world_uid: int,
    branch_tick: int,
    output_path: str | Path,
) -> dict[str, Any]:
    """Reopen four exact roots and seal one deterministic engineering receipt."""

    if type(world_uid) is not int or world_uid <= 0:
        _fail("world UID must be a positive integer")
    if type(branch_tick) is not int or branch_tick < 0:
        _fail("branch tick must be a nonnegative absolute tick")
    parent = _reopen_parent(
        founder_run_root, world_uid=world_uid, branch_tick=branch_tick
    )
    reference = _reopen_trajectory(
        "reference_b2_same_control", reference_b2_run_root, parent=parent
    )
    same = _reopen_trajectory(
        "repacked_b1_same_control", same_control_b1_run_root, parent=parent
    )
    changed = _reopen_trajectory(
        "repacked_b1_changed_control", changed_control_b1_run_root, parent=parent
    )
    receipt = _assemble_receipt(
        parent=parent,
        reference=reference,
        same_control=same,
        changed_control=changed,
    )
    _seal_receipt(output_path, receipt)
    return validate_selected_world_exactness_receipt_v1(output_path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Seal or verify internal selected-world exactness evidence."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    prove = subparsers.add_parser("prove", help="reopen sources and seal evidence")
    prove.add_argument("--founder-run-root", required=True)
    prove.add_argument("--reference-b2-run-root", required=True)
    prove.add_argument("--same-control-b1-run-root", required=True)
    prove.add_argument("--changed-control-b1-run-root", required=True)
    prove.add_argument("--world-uid", required=True, type=int)
    prove.add_argument("--branch-tick", required=True, type=int)
    prove.add_argument("--output", required=True)
    verify = subparsers.add_parser("verify", help="verify one sealed receipt")
    verify.add_argument("receipt")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "verify":
        result = validate_selected_world_exactness_receipt_v1(args.receipt)
    else:
        result = prove_selected_world_exactness_v1(
            founder_run_root=args.founder_run_root,
            reference_b2_run_root=args.reference_b2_run_root,
            same_control_b1_run_root=args.same_control_b1_run_root,
            changed_control_b1_run_root=args.changed_control_b1_run_root,
            world_uid=args.world_uid,
            branch_tick=args.branch_tick,
            output_path=args.output,
        )
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
