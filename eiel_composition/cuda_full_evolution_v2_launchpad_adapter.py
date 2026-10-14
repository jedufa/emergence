"""Canonical Scheduler/CRCA composition root for CUDA Full Evolution V2."""

from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
import stat
from typing import Any, Callable, Final, Mapping
from uuid import uuid4

from .checkpoint_occurrence import (
    ABSOLUTE_TICK_CHECKPOINT_COMPLETION_NAME,
    ABSOLUTE_TICK_CHECKPOINT_COMPLETION_SCHEMA,
    ABSOLUTE_TICK_CHECKPOINT_METADATA_NAME,
    ABSOLUTE_TICK_CHECKPOINT_METADATA_SCHEMA,
    ABSOLUTE_TICK_CHECKPOINT_STATE_NAME,
    ABSOLUTE_TICK_OCCURRENCE_COMMIT_NAME,
    ABSOLUTE_TICK_OCCURRENCE_DESCRIPTOR_NAME,
    WORKHORSE_SEMANTIC_OCCURRENCE_COMMIT_NAME,
    WORKHORSE_SEMANTIC_OCCURRENCE_DESCRIPTOR_NAME,
    absolute_tick_checkpoint_bundle_file_participant_bindings_v1,
    checkpoint_absolute_tick_bundle_tree_digest_v1,
    inspect_workhorse_semantic_checkpoint_occurrence_v2,
    materialize_absolute_tick_checkpoint_occurrence_v1,
    materialize_workhorse_semantic_checkpoint_occurrence_v2,
    restore_absolute_tick_checkpoint_occurrence_v1,
    restore_workhorse_semantic_checkpoint_occurrence_v2,
)
from .execution_identity_projection import (
    project_current_execution_v1,
    validate_shadow_projection_v1,
    write_shadow_projection_v1,
)
from .launchpad_attempt_authority import (
    AuthoritativeAttemptBundleV1,
    authoritative_attempt_bundle_from_path,
)
from emergence_contracts import (
    ABSOLUTE_TICK_CHECKPOINT_FORMAT_ID,
    ABSOLUTE_TICK_CHECKPOINT_TIME_AUTHORITY,
    AttemptFrontiersV1,
    AttemptId,
    AttemptRetryBasisCodeV1,
    AttemptRetryBasisV1,
    AttemptTerminalStatusV1,
    AttemptTerminalV1,
    CheckpointId,
    ContentDigest,
    RestoreCheckpointAuthorizationV1,
    RunId,
    WorkhorseSemanticCheckpointOccurrenceCommitV2,
    WorkhorseSemanticCheckpointOccurrenceV2,
    canonical_json_bytes,
    canonical_sha256,
)
from tools import resource_monitor
from tools.actors.common import ActorError, canonical_host_actor_state_root, utc_now
from tools.actors.resource_manager import (
    assess_dead_cuda_owner_reclaim_v1,
    assess_resources,
)
from tools.scheduler.launchpad_attempts import SchedulerLaunchpadAttemptIssuerV1
from tools.scheduler.launcher import process_start_identity
from tools.scheduler.programme_contracts import (
    ContentDigest as SchedulerContentDigest,
    LaunchpadAttemptClaimV1,
    canonical_sha256 as scheduler_sha256,
)


ATTEMPT_POINTER_SCHEMA_V1: Final[str] = (
    "eiel_cuda_full_evolution_v2_scheduler_attempt_pointer_v1"
)
TERMINAL_CUDA_MEMORY_RELEASE_NAME_V1: Final[str] = "terminal-cuda-memory-release.json"
PROCESS_OWNER_ID_V1: Final[str] = (
    "eiel_composition.cuda_full_evolution_v2_launchpad_adapter"
)
MINIMUM_CONTINUATION_TTL_SECONDS_V1: Final[int] = 24 * 60 * 60
CONTINUATION_CLOSEOUT_CUSHION_SECONDS_V1: Final[int] = 60 * 60

ResourceEvidenceFactoryV1 = Callable[..., Mapping[str, Any]]
DeadOwnerReclaimEvidenceFactoryV1 = Callable[..., Mapping[str, Any]]


class CudaFullEvolutionV2LaunchpadAdapterError(ValueError):
    """The outer composition root cannot prove exact execution authority."""


def _continuation_lineage_scope_digest_v1(
    *,
    run_id: str,
    semantic_spec_digest: Mapping[str, Any],
    run_root: str,
) -> SchedulerContentDigest:
    """Bind the Scheduler continuation fence to one physical run instance.

    ``run_id`` is deliberately deterministic for a semantic occurrence so
    repeated scientific requests retain comparable logical identity.  The
    Scheduler continuation fence is narrower: a fresh output root is a new
    physical execution instance, while a continuation/reopen of the same
    root must retain its fence.  Keeping that distinction here prevents a
    prior published terminal from poisoning an otherwise fresh run without
    changing scientific or RNG identity.
    """

    if not isinstance(run_id, str) or not run_id.strip():
        raise CudaFullEvolutionV2LaunchpadAdapterError(
            "continuation lineage run_id is invalid"
        )
    if not isinstance(semantic_spec_digest, Mapping):
        raise CudaFullEvolutionV2LaunchpadAdapterError(
            "continuation lineage semantic digest is invalid"
        )
    if not isinstance(run_root, str) or not run_root.startswith("/"):
        raise CudaFullEvolutionV2LaunchpadAdapterError(
            "continuation lineage run root must be absolute"
        )
    return scheduler_sha256(
        {
            "schema_version": "crca_launchpad_continuation_scope_v1",
            "run_id": run_id,
            "semantic_spec_digest": dict(semantic_spec_digest),
            "run_root": run_root,
        }
    )


def _continuation_ttl_seconds_v1(campaign_plan: Mapping[str, Any]) -> int:
    """Cover the sealed execution budget plus bounded publication latency."""

    execution = campaign_plan.get("execution")
    budgets = execution.get("budgets") if isinstance(execution, Mapping) else None
    wall_seconds = budgets.get("wall_seconds") if isinstance(budgets, Mapping) else None
    if type(wall_seconds) is not int or wall_seconds <= 0:
        raise CudaFullEvolutionV2LaunchpadAdapterError(
            "campaign plan lacks a positive wall_seconds continuation bound"
        )
    return max(
        MINIMUM_CONTINUATION_TTL_SECONDS_V1,
        wall_seconds + CONTINUATION_CLOSEOUT_CUSHION_SECONDS_V1,
    )


def _scheduler_digest(
    value: ContentDigest | Mapping[str, Any],
) -> SchedulerContentDigest:
    raw = value.to_dict() if isinstance(value, ContentDigest) else value
    if not isinstance(raw, Mapping):
        raise CudaFullEvolutionV2LaunchpadAdapterError("digest binding is absent")
    return SchedulerContentDigest(str(raw.get("algorithm")), str(raw.get("hex")))


def _content_digest(value: Mapping[str, Any]) -> ContentDigest:
    return ContentDigest(str(value.get("algorithm")), str(value.get("hex")))


def _workhorse_semantic_operation_v2(
    campaign_plan: Mapping[str, Any],
) -> dict[str, Any] | None:
    analysis = campaign_plan.get("analysis")
    extension = analysis.get("extension") if isinstance(analysis, Mapping) else None
    operation = extension.get("contract") if isinstance(extension, Mapping) else None
    if not isinstance(operation, Mapping) or operation.get("operation") != (
        "SEMANTIC_EVIDENCE_CASE"
    ):
        return None
    semantic_case = operation.get("semantic_case_contract")
    if not isinstance(semantic_case, Mapping):
        raise CudaFullEvolutionV2LaunchpadAdapterError(
            "Workhorse semantic operation lacks its closed case contract"
        )
    try:
        from engine.perf.cuda_full_evolution_v2.workhorse.semantic_cases import (
            workhorse_semantic_case_operation_contract_v2,
        )

        law = semantic_case.get("case_law")
        expected = workhorse_semantic_case_operation_contract_v2(
            case_id=str(law.get("case_id") if isinstance(law, Mapping) else ""),
            shape_id=str(semantic_case.get("shape_id") or ""),
            semantic_temporal_k=int(semantic_case.get("semantic_temporal_k")),
        )
    except (RuntimeError, TypeError, ValueError) as exc:
        raise CudaFullEvolutionV2LaunchpadAdapterError(
            f"Workhorse semantic operation contract is invalid: {exc}"
        ) from exc
    if dict(semantic_case) != expected:
        raise CudaFullEvolutionV2LaunchpadAdapterError(
            "Workhorse semantic operation differs from the closed case table"
        )
    return dict(operation)


def _read_exact_json_file(path: Path, *, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise CudaFullEvolutionV2LaunchpadAdapterError(f"{label} is not a regular file")
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CudaFullEvolutionV2LaunchpadAdapterError(
            f"{label} cannot be reopened: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise CudaFullEvolutionV2LaunchpadAdapterError(f"{label} is not an object")
    return value


def _canonical_utc() -> str:
    return utc_now()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _semantic_restore_parent_checkpoint_projection_v2(
    campaign_plan: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Reopen the exact parent occurrence before deriving the child RunId."""

    operation = _workhorse_semantic_operation_v2(campaign_plan)
    if operation is None:
        return None
    law = operation["semantic_case_contract"]["case_law"]
    raw_binding = operation.get("external_source_checkpoint_binding")
    if law.get("requires_source_checkpoint") is not True:
        if raw_binding is not None:
            raise CudaFullEvolutionV2LaunchpadAdapterError(
                "semantic case gained an unauthorized restore source"
            )
        return None
    try:
        from engine.perf.cuda_full_evolution_v2.workhorse.semantic_cases import (
            validate_external_checkpoint_source_binding_v2,
        )

        binding = validate_external_checkpoint_source_binding_v2(
            raw_binding,
            expected_identity_sha256=str(
                operation["source_checkpoint_identity_sha256"]
            ),
        )
        descriptor_path = Path(binding["checkpoint_occurrence_record_path"]).resolve()
        descriptor = WorkhorseSemanticCheckpointOccurrenceV2.from_dict(
            _read_exact_json_file(
                descriptor_path,
                label="Workhorse checkpoint occurrence descriptor",
            )
        )
    except Exception as exc:
        raise CudaFullEvolutionV2LaunchpadAdapterError(
            f"semantic restore parent projection is not canonical: {exc}"
        ) from exc
    if (
        descriptor_path.name != WORKHORSE_SEMANTIC_OCCURRENCE_DESCRIPTOR_NAME
        or _sha256_file(descriptor_path)
        != binding["checkpoint_occurrence_record_file_sha256"]
        or descriptor.occurrence_digest().hex
        != binding["checkpoint_occurrence_identity_sha256"]
        or descriptor.base_checkpoint_identity_digest.hex
        != binding["base_checkpoint_identity_sha256"]
        or descriptor.absolute_next_tick != law["start_next_absolute_tick"]
    ):
        raise CudaFullEvolutionV2LaunchpadAdapterError(
            "semantic restore parent projection identity changed"
        )
    return descriptor.to_dict()


def _current_process_start_ticks_v1() -> int:
    try:
        tail = (
            Path(f"/proc/{os.getpid()}/stat")
            .read_text(encoding="ascii")
            .rsplit(")", 1)[1]
        )
        value = int(tail.split()[19])
    except (OSError, UnicodeError, ValueError, IndexError) as exc:
        raise CudaFullEvolutionV2LaunchpadAdapterError(
            "current execution-owner process identity is unavailable"
        ) from exc
    if value <= 0:
        raise CudaFullEvolutionV2LaunchpadAdapterError(
            "current execution-owner process start tick is invalid"
        )
    return value


def _current_process_identity_v1() -> dict[str, Any]:
    pid = os.getpid()
    start_identity = process_start_identity(pid)
    if start_identity is None:
        raise CudaFullEvolutionV2LaunchpadAdapterError(
            "current Linux process start identity is unavailable"
        )
    return {
        "pid": pid,
        "process_start_ticks": _current_process_start_ticks_v1(),
        "process_start_identity": start_identity,
    }


def _write_immutable_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = canonical_json_bytes(dict(value))
    _write_immutable_bytes(path, payload)


def _write_immutable_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC,
        stat.S_IRUSR | stat.S_IWUSR,
    )
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise OSError("immutable authority-pointer write made no progress")
            offset += written
        os.fchmod(descriptor, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    parent = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(parent)
    finally:
        os.close(parent)


def _copy_immutable_file_v1(source: Path, destination: Path) -> None:
    """Clone or kernel-stream one immutable file without materializing it."""

    import errno
    import fcntl

    supplied = source.expanduser()
    if supplied.is_symlink() or not supplied.is_file():
        raise CudaFullEvolutionV2LaunchpadAdapterError(
            "immutable checkpoint source is not a regular file"
        )
    source_path = supplied.resolve(strict=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    source_flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        source_flags |= os.O_NOFOLLOW
    source_descriptor = os.open(source_path, source_flags)
    destination_descriptor: int | None = None
    try:
        source_before = os.fstat(source_descriptor)
        if not stat.S_ISREG(source_before.st_mode) or source_before.st_mode & (
            stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH
        ):
            raise CudaFullEvolutionV2LaunchpadAdapterError(
                "immutable checkpoint source is writable or non-regular"
            )
        destination_descriptor = os.open(
            destination,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC,
            stat.S_IRUSR | stat.S_IWUSR,
        )
        try:
            # Linux FICLONE creates a distinct, single-link inode while sharing
            # unchanged extents.  The bounded kernel-copy fallback preserves the
            # same memory property when the filesystem has no reflink support.
            fcntl.ioctl(destination_descriptor, 0x40049409, source_descriptor)
        except OSError as exc:
            if exc.errno not in {
                errno.EXDEV,
                errno.EINVAL,
                errno.ENOTTY,
                errno.EOPNOTSUPP,
                errno.ENOSYS,
            }:
                raise
            os.lseek(source_descriptor, 0, os.SEEK_SET)
            while True:
                copied = os.copy_file_range(
                    source_descriptor,
                    destination_descriptor,
                    8 * 1024 * 1024,
                )
                if copied == 0:
                    break
        destination_stat = os.fstat(destination_descriptor)
        if (
            not stat.S_ISREG(destination_stat.st_mode)
            or destination_stat.st_nlink != 1
            or destination_stat.st_size != source_before.st_size
        ):
            raise CudaFullEvolutionV2LaunchpadAdapterError(
                "immutable checkpoint projection has an invalid file identity"
            )
        os.fchmod(
            destination_descriptor,
            stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH,
        )
        os.fsync(destination_descriptor)
        source_after = os.fstat(source_descriptor)
        if (
            source_after.st_dev,
            source_after.st_ino,
            source_after.st_size,
            source_after.st_mode,
            source_after.st_mtime_ns,
            source_after.st_ctime_ns,
        ) != (
            source_before.st_dev,
            source_before.st_ino,
            source_before.st_size,
            source_before.st_mode,
            source_before.st_mtime_ns,
            source_before.st_ctime_ns,
        ):
            raise CudaFullEvolutionV2LaunchpadAdapterError(
                "immutable checkpoint source changed during projection"
            )
    except BaseException:
        if destination_descriptor is not None:
            os.close(destination_descriptor)
            destination_descriptor = None
        if destination.exists() and not destination.is_symlink():
            destination.unlink()
        raise
    finally:
        if destination_descriptor is not None:
            os.close(destination_descriptor)
        os.close(source_descriptor)
    if _sha256_file(destination) != _sha256_file(source_path):
        destination.unlink()
        raise CudaFullEvolutionV2LaunchpadAdapterError(
            "immutable checkpoint projection changed exact content"
        )
    _fsync_directory(destination.parent)


def _quarantine_checkpoint_publications_v1(
    paths: tuple[Path, ...], failure: BaseException
) -> None:
    for raw_path in paths:
        try:
            if raw_path.is_symlink():
                raise CudaFullEvolutionV2LaunchpadAdapterError(
                    "failed checkpoint publication is a symbolic link"
                )
            if not raw_path.exists():
                continue
            path = raw_path.resolve()
            failed = path.with_name(f"{path.name}.publication.failed.{uuid4().hex}")
            while failed.exists() or failed.is_symlink():
                failed = path.with_name(f"{path.name}.publication.failed.{uuid4().hex}")
            os.replace(path, failed)
            _fsync_directory(path.parent)
        except BaseException as cleanup_error:
            failure.add_note(
                "Workhorse CRCA quarantine also failed: "
                f"{type(cleanup_error).__name__}: {cleanup_error}"
            )


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _route_bindings(
    bound_execution_request: Mapping[str, Any],
) -> dict[str, Any]:
    route = bound_execution_request.get("execution_route_contract")
    components = route.get("components") if isinstance(route, Mapping) else None
    if not isinstance(components, list):
        raise CudaFullEvolutionV2LaunchpadAdapterError(
            "bound request lacks its execution-route components"
        )
    by_id = {
        str(row.get("component_id")): row
        for row in components
        if isinstance(row, Mapping) and row.get("component_id")
    }
    workhorse_operation = by_id.get("cuda_full_evolution_workhorse_v2_operation")
    workhorse_artifact = by_id.get("cuda_full_evolution_workhorse_v2_artifact_unit")
    workhorse_selection = by_id.get("cuda_full_evolution_workhorse_v2_device_selection")
    workhorse_checkpoint = by_id.get(
        "cuda_full_evolution_workhorse_v2_checkpoint_science"
    )
    if all(
        isinstance(row, Mapping)
        for row in (
            workhorse_operation,
            workhorse_artifact,
            workhorse_selection,
            workhorse_checkpoint,
        )
    ):
        operation_name = workhorse_operation.get("operation")
        resident_parameters = bound_execution_request.get("resident_request", {}).get(
            "parameter_arrays", {}
        )
        if (
            operation_name
            not in {
                "EXECUTION_ROW",
                "DEVICE_ENVELOPE_CALIBRATION",
                "SEMANTIC_EVIDENCE_CASE",
                "SCIENCE_COMPATIBILITY_DEMO",
                "INDEXED_ACS_CONTINUATION",
            }
            or workhorse_operation.get("one_operation_per_attempt") is not True
            or workhorse_checkpoint.get("observation_mode") != "OFF"
            or workhorse_checkpoint.get("science_continuation_claimed") is not False
            or workhorse_checkpoint.get("checkpoint_occurrence_authority")
            != "outer_canonical_campaign_provider"
            or resident_parameters.get("workhorse.operation", [None])[0]
            != operation_name
        ):
            raise CudaFullEvolutionV2LaunchpadAdapterError(
                "Workhorse route is not one exact technical operation"
            )
        device = workhorse_selection.get("device_identity")
        if not isinstance(device, Mapping):
            raise CudaFullEvolutionV2LaunchpadAdapterError(
                "Workhorse route lacks its exact device identity"
            )
        device_id = device.get("uuid")
        if not isinstance(device_id, str) or not device_id:
            raise CudaFullEvolutionV2LaunchpadAdapterError(
                "Workhorse device UUID is unusable"
            )
        shape_id = workhorse_operation.get("shape_id")
        backend_id = workhorse_operation.get("backend_id")
        contract = workhorse_operation.get("contract")
        if operation_name == "SEMANTIC_EVIDENCE_CASE":
            semantic_case = workhorse_operation.get("semantic_case_contract")
            law = (
                semantic_case.get("case_law")
                if isinstance(semantic_case, Mapping)
                else None
            )
            if (
                not isinstance(contract, Mapping)
                or not isinstance(semantic_case, Mapping)
                or not isinstance(law, Mapping)
                or workhorse_operation.get("offline_matrix_aggregation_only")
                is not False
                or workhorse_operation.get("operation_contract_sha256")
                != semantic_case.get("operation_contract_sha256")
                or contract.get("shape_id") != shape_id
                or contract.get("backend_id") != backend_id
                or contract.get("B") != law.get("B")
                or contract.get("temporal_k") != law.get("dispatch_temporal_k")
                or contract.get("total_ticks") != law.get("total_ticks")
                or contract.get("start_next_tick")
                != law.get("start_next_absolute_tick")
                or workhorse_operation.get("compiled_temporal_k")
                != contract.get("temporal_k")
                or shape_id != resident_parameters.get("workhorse.shape_id", [None])[0]
                or backend_id
                != resident_parameters.get("workhorse.backend_id", [None])[0]
                or contract.get("B")
                != resident_parameters.get("workhorse.B", [None])[0]
                or contract.get("temporal_k")
                != resident_parameters.get("workhorse.temporal_k", [None])[0]
                or workhorse_checkpoint.get("science_continuation_contract_sha256")
                is None
            ):
                raise CudaFullEvolutionV2LaunchpadAdapterError(
                    "Workhorse semantic route differs from its closed atomic case"
                )
            batch_size = contract.get("B")
            ticks_per_dispatch = contract.get("temporal_k")
            event_transport = "workhorse_external_exact_ack_v2"
        elif operation_name == "SCIENCE_COMPATIBILITY_DEMO":
            demo = workhorse_operation.get("science_demo_contract")
            if (
                not isinstance(contract, Mapping)
                or not isinstance(demo, Mapping)
                or workhorse_operation.get("offline_matrix_aggregation_only")
                is not False
                or demo.get("operation") != "SCIENCE_COMPATIBILITY_DEMO"
                or demo.get("shape_id") != shape_id
                or demo.get("backend_id") != backend_id
                or demo.get("B") != 10
                or demo.get("temporal_k") != 1
                or demo.get("start_next_tick") != 0
                or demo.get("total_ticks") != 40
                or demo.get("boundary_next_ticks") != [0, 8, 16, 24, 32, 40]
                or demo.get("checkpoint_ticks") != [16, 32, 40]
                or contract.get("shape_id") != shape_id
                or contract.get("backend_id") != backend_id
                or contract.get("B") != 10
                or contract.get("temporal_k") != 1
                or contract.get("start_next_tick") != 0
                or contract.get("total_ticks") != 40
                or shape_id != resident_parameters.get("workhorse.shape_id", [None])[0]
                or backend_id
                != resident_parameters.get("workhorse.backend_id", [None])[0]
                or contract.get("B")
                != resident_parameters.get("workhorse.B", [None])[0]
                or contract.get("temporal_k")
                != resident_parameters.get("workhorse.temporal_k", [None])[0]
                or workhorse_operation.get("compiled_temporal_k") != 1
                or workhorse_selection.get("B") != 10
                or workhorse_selection.get("temporal_k") != 1
                or workhorse_checkpoint.get("science_continuation_contract_sha256")
                is None
                or workhorse_checkpoint.get("source_checkpoint_identity_sha256")
                is not None
            ):
                raise CudaFullEvolutionV2LaunchpadAdapterError(
                    "Workhorse science demo route differs from its closed contract"
                )
            batch_size = 10
            ticks_per_dispatch = 1
            event_transport = "workhorse_external_exact_ack_compact_v4"
        elif operation_name in {"EXECUTION_ROW", "INDEXED_ACS_CONTINUATION"}:
            continuation = workhorse_operation.get("indexed_continuation_contract")
            logical_continuation = workhorse_operation.get(
                "science_continuation_identity"
            )
            physical_continuation = workhorse_operation.get(
                "physical_continuation_plan"
            )
            if (
                not isinstance(contract, Mapping)
                or workhorse_operation.get("offline_matrix_aggregation_only")
                != (operation_name == "EXECUTION_ROW")
                or contract.get("shape_id") != shape_id
                or contract.get("backend_id") != backend_id
                or shape_id != resident_parameters.get("workhorse.shape_id", [None])[0]
                or backend_id
                != resident_parameters.get("workhorse.backend_id", [None])[0]
                or contract.get("B") != workhorse_selection.get("B")
                or contract.get("B")
                != resident_parameters.get("workhorse.B", [None])[0]
                or contract.get("temporal_k") != workhorse_selection.get("temporal_k")
                or contract.get("temporal_k")
                != resident_parameters.get("workhorse.temporal_k", [None])[0]
                or (
                    operation_name == "EXECUTION_ROW"
                    and workhorse_operation.get("compiled_temporal_k")
                    != contract.get("temporal_k")
                )
                or not isinstance(
                    workhorse_operation.get("absolute_tick_span"), Mapping
                )
                or workhorse_operation["absolute_tick_span"].get("start_next_tick")
                != contract.get("start_next_tick")
                or workhorse_operation["absolute_tick_span"].get("total_ticks")
                != contract.get("total_ticks")
                or (
                    operation_name == "INDEXED_ACS_CONTINUATION"
                    and (
                        not isinstance(continuation, Mapping)
                        or not isinstance(logical_continuation, Mapping)
                        or not isinstance(physical_continuation, Mapping)
                        or continuation.get("science_continuation_identity")
                        != logical_continuation
                        or continuation.get("physical_continuation_plan")
                        != physical_continuation
                        or physical_continuation.get("target_contract") != contract
                        or workhorse_operation.get("compiled_temporal_k")
                        != continuation.get("dispatch_plan", {}).get(
                            "admitted_temporal_k"
                        )
                        or workhorse_selection.get("compiled_temporal_k")
                        != workhorse_operation.get("compiled_temporal_k")
                        or logical_continuation.get(
                            "science_continuation_contract_sha256"
                        )
                        != workhorse_checkpoint.get(
                            "science_continuation_contract_sha256"
                        )
                        or logical_continuation.get(
                            "source_world_state_set_sha256"
                        )
                        != workhorse_checkpoint.get(
                            "source_checkpoint_identity_sha256"
                        )
                        or workhorse_checkpoint.get("checkpoint_ticks")
                        != continuation.get("checkpoint_ticks")
                    )
                )
            ):
                raise CudaFullEvolutionV2LaunchpadAdapterError(
                    "Workhorse route components disagree about the exact row or "
                    "indexed continuation"
                )
            batch_size = contract.get("B")
            ticks_per_dispatch = contract.get("temporal_k")
            event_transport = (
                "workhorse_external_exact_ack_indexed_acs2_v2"
                if operation_name == "INDEXED_ACS_CONTINUATION"
                else "workhorse_external_exact_ack_v2"
            )
        else:
            compiled_k = workhorse_operation.get("compiled_temporal_k")
            reserve_policy = workhorse_operation.get("reserve_policy")
            span = workhorse_operation.get("absolute_tick_span")
            if (
                contract is not None
                or workhorse_operation.get("offline_matrix_aggregation_only")
                is not False
                or not isinstance(compiled_k, list)
                or not compiled_k
                or compiled_k != workhorse_selection.get("compiled_temporal_k")
                or shape_id != resident_parameters.get("workhorse.shape_id", [None])[0]
                or backend_id
                != resident_parameters.get("workhorse.backend_id", [None])[0]
                or not isinstance(reserve_policy, Mapping)
                or not isinstance(span, Mapping)
                or type(span.get("start_next_tick")) is not int
                or type(span.get("total_ticks")) is not int
                or type(span.get("target_next_tick")) is not int
                or span.get("target_next_tick")
                != span.get("start_next_tick", 0) + span.get("total_ticks", 0)
                or resident_parameters.get(
                    "workhorse.compiled_temporal_k_sha256", [None]
                )[0]
                != canonical_sha256(compiled_k).hex
                or resident_parameters.get("workhorse.reserve_bytes", [None])[0]
                != reserve_policy.get("reserve_bytes_requested")
                or workhorse_selection.get("B") != 1
                or workhorse_selection.get("temporal_k") is not None
                or workhorse_checkpoint.get("science_continuation_contract_sha256")
                is not None
                or workhorse_checkpoint.get("source_checkpoint_identity_sha256")
                is not None
            ):
                raise CudaFullEvolutionV2LaunchpadAdapterError(
                    "Workhorse calibration route binding changed"
                )
            batch_size = 1
            ticks_per_dispatch = span.get("total_ticks")
            event_transport = "workhorse_calibration_owner_measurement_v2"
        return {
            "provider_id": bound_execution_request.get("provider_id"),
            "route_id": workhorse_operation.get("route_id"),
            "device_id": device_id,
            "device_identity": dict(device),
            "batch_size": batch_size,
            "ticks_per_dispatch": ticks_per_dispatch,
            "execution_mode": backend_id,
            "event_transport": event_transport,
            "native_artifact_digest": {
                "algorithm": "sha256",
                "hex": workhorse_artifact.get("artifact_sha256"),
            },
            "workhorse_operation": dict(workhorse_operation),
            "workhorse_artifact": dict(workhorse_artifact),
            "workhorse_checkpoint": dict(workhorse_checkpoint),
        }
    native = by_id.get("cuda_full_evolution_v2_native_transition")
    selection = by_id.get("cuda_full_evolution_v2_device_selection")
    transport = by_id.get("cuda_full_evolution_v2_event_transport")
    if not all(isinstance(row, Mapping) for row in (native, selection, transport)):
        raise CudaFullEvolutionV2LaunchpadAdapterError(
            "bound request lacks exact native, device, or transport identity"
        )
    device = selection.get("device_identity")
    if not isinstance(device, Mapping):
        raise CudaFullEvolutionV2LaunchpadAdapterError(
            "bound request lacks exact device identity"
        )
    device_id = device.get("uuid") or f"cuda:{device.get('index')}"
    if not isinstance(device_id, str) or not device_id:
        raise CudaFullEvolutionV2LaunchpadAdapterError(
            "bound request device identity is unusable"
        )
    resource_policy_component = by_id.get(
        "cuda_full_evolution_v2_shared_gpu_resource_policy"
    )
    physical_batch_set = bound_execution_request.get("physical_batch_set")
    gpu_resource_policy: dict[str, Any] | None = None
    if resource_policy_component is None:
        if physical_batch_set is not None:
            raise CudaFullEvolutionV2LaunchpadAdapterError(
                "exclusive bound request unexpectedly carries a physical batch set"
            )
    else:
        from engine.campaigns.cuda_full_evolution_v2_shared_route import (
            SHARED_W32_GPU_POLICY_ID_V1,
            shared_w32_gpu_policy_v1,
        )

        expected_component = {
            "component_id": "cuda_full_evolution_v2_shared_gpu_resource_policy",
            **shared_w32_gpu_policy_v1(),
        }
        source_batch_id = bound_execution_request.get("source_batch_id")
        if (
            dict(resource_policy_component) != expected_component
            or not isinstance(physical_batch_set, Mapping)
            or physical_batch_set.get("schema_version")
            != "eiel_cuda_full_evolution_v2_physical_batch_set_v1"
            or physical_batch_set.get("gpu_resource_policy_id")
            != SHARED_W32_GPU_POLICY_ID_V1
            or source_batch_id not in physical_batch_set.get("batch_ids", [])
            or physical_batch_set.get("uniform_physical_batch_cells")
            != selection.get("B")
        ):
            raise CudaFullEvolutionV2LaunchpadAdapterError(
                "shared GPU route differs from its bound physical batch set"
            )
        gpu_resource_policy = shared_w32_gpu_policy_v1()
    return {
        "provider_id": bound_execution_request.get("provider_id"),
        "route_id": native.get("route_id"),
        "device_id": device_id,
        "device_identity": dict(device),
        "batch_size": selection.get("B"),
        "ticks_per_dispatch": selection.get("temporal_k"),
        "execution_mode": selection.get("execution_mode"),
        "event_transport": transport.get("event_transport"),
        "native_artifact_digest": None,
        "gpu_resource_policy": gpu_resource_policy,
        "physical_batch_set": (
            dict(physical_batch_set)
            if isinstance(physical_batch_set, Mapping)
            else None
        ),
        "workhorse_operation": None,
        "workhorse_artifact": None,
        "workhorse_checkpoint": None,
    }


def _scheduler_resource_topology_v1(
    route: Mapping[str, Any], *, run_id: str
) -> tuple[tuple[str, ...], tuple[tuple[str, ...], ...], dict[str, Any]]:
    """Resolve exact exclusive resources or the closed shared lane candidates."""

    policy = route.get("gpu_resource_policy")
    device_id = str(route["device_id"])
    if isinstance(policy, Mapping):
        from tools.scheduler.programme_contracts import (
            shared_gpu_lane_candidate_sets_v1,
        )

        candidates = shared_gpu_lane_candidate_sets_v1(device_id)
        return (
            (),
            candidates,
            {
                "schema_version": "eiel_shared_gpu_lane_scope_v1",
                "run_id": run_id,
                "device_id": device_id,
                "gpu_resource_policy": dict(policy),
                "candidate_resource_sets": [
                    list(candidate) for candidate in candidates
                ],
            },
        )
    resource_ids = (f"gpu:{device_id}",)
    return (
        resource_ids,
        (),
        {
            "schema_version": "eiel_exclusive_gpu_scope_v1",
            "resource_ids": list(resource_ids),
            "run_id": run_id,
        },
    )


def _default_resource_evidence_v1(
    *,
    repo_root: Path,
    route: Mapping[str, Any],
    campaign_plan: Mapping[str, Any],
    at_utc: str,
) -> Mapping[str, Any]:
    monitor_config = resource_monitor.MonitorConfig(repo_root=repo_root)
    report = resource_monitor.collect_report(monitor_config)
    observer_pid = os.getpid()
    raw_gpu = report.get("gpu")
    gpu = dict(raw_gpu) if isinstance(raw_gpu, Mapping) else {}
    raw_processes = gpu.get("processes")
    processes = (
        [dict(row) for row in raw_processes if isinstance(row, Mapping)]
        if isinstance(raw_processes, list)
        else []
    )
    observer_gpu_processes = [
        row for row in processes if row.get("pid") == observer_pid
    ]
    external_gpu_processes = [
        row for row in processes if row.get("pid") != observer_pid
    ]
    assessment_report = dict(report)
    assessment_report["gpu"] = {
        **gpu,
        "processes": external_gpu_processes,
        "compute_process_count": len(external_gpu_processes),
        "observed_vram_used_bytes": sum(
            int(row.get("vram_used_bytes") or 0) for row in external_gpu_processes
        ),
    }
    raw_workloads = report.get("running_eiel_workloads")
    assessment_report["running_eiel_workloads"] = (
        [
            dict(row)
            for row in raw_workloads
            if isinstance(row, Mapping) and row.get("pid") != observer_pid
        ]
        if isinstance(raw_workloads, list)
        else []
    )
    assessment_report["gpu"]["occupancy_state"] = resource_monitor.gpu_occupancy_state(
        assessment_report
    )
    assessment_report["blockers"] = resource_monitor.build_blockers(
        assessment_report,
        monitor_config,
    )
    assessment_report["blocker_classification"] = (
        resource_monitor.blocker_classification(assessment_report["blockers"])
    )
    assessment_report["recommendation"] = resource_monitor.recommend(
        assessment_report,
        monitor_config,
    )
    budgets = campaign_plan.get("execution", {}).get("budgets", {})
    minimum_vram = budgets.get("vram_bytes", 0) if isinstance(budgets, Mapping) else 0
    policy = route.get("gpu_resource_policy")
    shared = isinstance(policy, Mapping)
    peak_vram_gib = (
        float(policy["peak_vram_gib"])
        if shared
        else float(minimum_vram) / float(1024**3)
    )
    demand = {
        "schema_version": "eiel_resource_demand_v1",
        "device": "cuda",
        "workload_class": (
            "cuda_full_evolution_v2_shared_functional"
            if shared
            else "cuda_full_evolution_v2"
        ),
        "exclusive_gpu": not shared,
        "gpu_admission_mode": "shared_functional" if shared else "exclusive",
        "peak_vram_gib": peak_vram_gib,
        "minimum_ram_available_gib": 0.0,
        "minimum_vram_available_gib": peak_vram_gib,
        "minimum_storage_available_gib": 0.0,
        "cpu_threads": 0,
        "expected_duration_minutes": 0,
        "mission_id": None,
    }
    # The report is collected after the Attempt claim timestamp is minted.
    # Assess it at its own observation frontier; using the earlier claim time
    # makes every observation that crosses a clock tick look spuriously future
    # dated and therefore stale.
    resource_observed_at_utc = str(report.get("generated_at_utc") or at_utc)
    recommendation = assess_resources(
        demand,
        assessment_report,
        now_utc=resource_observed_at_utc,
    )
    device_id = str(route["device_id"])
    observation = {
        "schema_version": "eiel_launchpad_resource_observation_v1",
        "device_id": device_id,
        "report": report,
        "assessment_scope": {
            "schema_version": "eiel_launchpad_resource_assessment_scope_v1",
            "observer_pid": observer_pid,
            "observer_gpu_process_count": len(observer_gpu_processes),
            "external_gpu_process_count": len(external_gpu_processes),
            "observer_gpu_context_excluded_from_conflict": bool(observer_gpu_processes),
            "gpu_admission_mode": demand["gpu_admission_mode"],
            "declared_peak_vram_gib": peak_vram_gib,
            "gpu_resource_policy_id": (policy.get("policy_id") if shared else None),
        },
    }
    live_check = {
        "schema_version": "crca_launchpad_live_provider_check_v1",
        "available": recommendation.get("decision") in {"ADMIT", "ADMIT_WITH_LIMITS"},
        "device_id": device_id,
        "event_transport": route.get("event_transport"),
        "gpu_admission_mode": demand["gpu_admission_mode"],
        "exclusive_gpu": demand["exclusive_gpu"],
        "declared_peak_vram_gib": peak_vram_gib,
        "gpu_resource_policy_id": policy.get("policy_id") if shared else None,
    }
    return {
        "resource_observation_id": "ROBS-LAUNCHPAD-"
        + scheduler_sha256(observation).hex[:20],
        "resource_observation": observation,
        "resource_observed_at_utc": resource_observed_at_utc,
        "resource_recommendation_id": "RADM-LAUNCHPAD-"
        + scheduler_sha256(recommendation).hex[:20],
        "resource_recommendation": recommendation,
        "live_provider_check": live_check,
    }


def _default_dead_owner_reclaim_evidence_v1(
    *,
    repo_root: Path,
    candidate: Mapping[str, Any],
    at_utc: str,
) -> Mapping[str, Any]:
    del at_utc  # Candidate discovery time is not reclaim-evidence time.
    process_observation = resource_monitor.observe_process_identity_v1(
        int(candidate["owner_pid"]),
        str(candidate["owner_process_start_identity"]),
    )
    # The ordered proof is process death -> later CUDA process query -> later
    # assessment.  The dead process cannot create a final CUDA context between
    # the two observations; a reused PID is still visible in the later query.
    report = resource_monitor.collect_report(
        resource_monitor.MonitorConfig(repo_root=repo_root)
    )
    assessed_at_utc = utc_now()
    return assess_dead_cuda_owner_reclaim_v1(
        attempt_id=str(candidate["attempt_id"]),
        resource_lease_digest=dict(candidate["resource_lease_digest"]),
        fencing_token_digest=dict(candidate["fencing_token_digest"]),
        continuation_lease_digest=dict(candidate["continuation_lease_digest"]),
        continuation_fence_state=str(candidate["continuation_fence_state"]),
        continuation_terminal_event_digest=(
            None
            if candidate["continuation_terminal_event_digest"] is None
            else dict(candidate["continuation_terminal_event_digest"])
        ),
        resource_started_event_digest=dict(candidate["resource_started_event_digest"]),
        device_uuid=str(candidate["device_uuid"]),
        selected_resource_ids=list(candidate["selected_resource_ids"]),
        owner_pid=int(candidate["owner_pid"]),
        owner_process_start_identity=str(candidate["owner_process_start_identity"]),
        process_identity_observation=process_observation,
        report=report,
        now_utc=assessed_at_utc,
    )


class CanonicalCudaFullEvolutionV2LaunchpadAdapterV1:
    """One outer adapter for shadow, Scheduler Attempt, and occurrences."""

    def __init__(
        self,
        *,
        repo_root: str | Path,
        state_root: str | Path | None = None,
        allow_test_state_root: bool = False,
        resource_evidence_factory: ResourceEvidenceFactoryV1 = (
            _default_resource_evidence_v1
        ),
        dead_owner_reclaim_evidence_factory: (
            DeadOwnerReclaimEvidenceFactoryV1
        ) = _default_dead_owner_reclaim_evidence_v1,
        now_fn: Callable[[], str] = _canonical_utc,
    ) -> None:
        self.repo_root = Path(repo_root).expanduser().resolve()
        canonical_state_root = canonical_host_actor_state_root()
        requested_state_root = (
            canonical_state_root
            if state_root is None
            else Path(state_root).expanduser().resolve()
        )
        if requested_state_root != canonical_state_root and not allow_test_state_root:
            raise CudaFullEvolutionV2LaunchpadAdapterError(
                "non-canonical Scheduler state root requires explicit test authority"
            )
        self.state_root = requested_state_root
        self.issuer = SchedulerLaunchpadAttemptIssuerV1(
            self.state_root,
            allow_test_shared_gpu_state_root=allow_test_state_root,
        )
        self.resource_evidence_factory = resource_evidence_factory
        self.dead_owner_reclaim_evidence_factory = dead_owner_reclaim_evidence_factory
        self.now_fn = now_fn

    def reclaim_dead_gpu_owners_v1(
        self,
        *,
        at_utc: str | None = None,
        conflicting_resource_ids: tuple[str, ...] | None = None,
    ) -> Mapping[str, Any]:
        """Perform one bounded, proof-gated reclaim sweep."""

        observed_at = self.now_fn() if at_utc is None else at_utc
        scheduler_reconciliation = self.issuer.reconcile(at_utc=observed_at)
        terminal_reconciliation = dict(
            scheduler_reconciliation.get("terminal_resource_reconciliation") or {}
        )
        candidates = self.issuer.dead_gpu_reclaim_candidates_v1(
            at_utc=observed_at,
            conflicting_resource_ids=conflicting_resource_ids,
        )
        outcomes: list[dict[str, Any]] = []
        for candidate in candidates:
            if candidate.get("continuation_fence_state") == "OWNER_DEAD":
                existing_proof_digest = candidate.get("existing_reclaim_proof_digest")
                if not isinstance(existing_proof_digest, Mapping):
                    raise CudaFullEvolutionV2LaunchpadAdapterError(
                        "OWNER_DEAD continuation lacks its bound reclaim proof"
                    )
                proof = self.issuer.dead_owner_reclaim_proof_v1(
                    attempt_id=str(candidate["attempt_id"]),
                    proof_digest=existing_proof_digest,
                )
            else:
                proof = dict(
                    self.dead_owner_reclaim_evidence_factory(
                        repo_root=self.repo_root,
                        candidate=candidate,
                        at_utc=observed_at,
                    )
                )
            if proof.get("decision") != "RECLAIM":
                outcomes.append(
                    {
                        "attempt_id": candidate["attempt_id"],
                        "status": "WAIT",
                        "reason_codes": list(proof.get("reason_codes") or []),
                    }
                )
                continue
            outcome = self.issuer.reclaim_dead_gpu_occupancy_v1(
                attempt_id=str(candidate["attempt_id"]),
                proof=proof,
                at_utc=str(proof.get("assessed_at_utc") or ""),
            )
            outcomes.append(dict(outcome))
        return {
            "schema_version": "eiel_dead_gpu_reclaim_sweep_v1",
            "observed_at_utc": observed_at,
            "scheduler_reconciliation": scheduler_reconciliation,
            "terminal_reconciliation": terminal_reconciliation,
            "released_terminal_count": int(
                terminal_reconciliation.get("released_count") or 0
            ),
            "candidate_count": len(candidates),
            "reclaimed_count": sum(
                row.get("status") == "OWNER_DEAD_RECLAIMED" for row in outcomes
            ),
            "outcomes": outcomes,
        }

    def reclaim_dead_shared_gpu_owners_v1(
        self, *, at_utc: str | None = None
    ) -> Mapping[str, Any]:
        """Compatibility alias for callers of the former shared-only sweep."""

        return self.reclaim_dead_gpu_owners_v1(at_utc=at_utc)

    def persist_shadow_projection_v1(self, **values: Any) -> Mapping[str, Any]:
        target = Path(values.pop("target"))
        campaign_plan = values.pop("campaign_plan")
        bound_execution_request = values.pop("bound_execution_request")
        baseline = values.pop("baseline")
        source_identity = values.pop("source_identity")
        assurance = values.pop("assurance")
        preflight = values.pop("preflight")
        attempt_nonce = values.pop("attempt_nonce")
        proposed_reason = values.pop("proposed_reason")
        if values or preflight.get("ready_for_execution") is not True:
            raise CudaFullEvolutionV2LaunchpadAdapterError(
                "CRCA shadow projection requires exact ready preflight inputs"
            )
        parent_checkpoint = _semantic_restore_parent_checkpoint_projection_v2(
            campaign_plan
        )
        projection = project_current_execution_v1(
            campaign_plan,
            bound_execution_request,
            baseline,
            source_identity,
            assurance,
            attempt_nonce=str(attempt_nonce),
            proposed_reason=str(proposed_reason),
            parent_checkpoint=parent_checkpoint,
        )
        write_shadow_projection_v1(target, projection)
        return projection

    def _semantic_restore_authority_v2(
        self,
        *,
        campaign_plan: Mapping[str, Any],
        run_occurrence: Mapping[str, Any],
        admission: Mapping[str, Any],
        initial_frontiers: Mapping[str, Any],
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        operation = _workhorse_semantic_operation_v2(campaign_plan)
        if operation is None:
            return None, None
        semantic_case = operation["semantic_case_contract"]
        law = semantic_case["case_law"]
        raw_binding = operation.get("external_source_checkpoint_binding")
        if law.get("requires_source_checkpoint") is not True:
            if raw_binding is not None:
                raise CudaFullEvolutionV2LaunchpadAdapterError(
                    "semantic case gained an unauthorized restore source"
                )
            return None, None
        try:
            from engine.perf.cuda_full_evolution_v2.workhorse.semantic_cases import (
                validate_external_checkpoint_source_binding_v2,
            )

            binding = validate_external_checkpoint_source_binding_v2(
                raw_binding,
                expected_identity_sha256=str(
                    operation["source_checkpoint_identity_sha256"]
                ),
            )
            descriptor_path = Path(
                binding["checkpoint_occurrence_record_path"]
            ).resolve()
            descriptor_raw = _read_exact_json_file(
                descriptor_path,
                label="Workhorse checkpoint occurrence descriptor",
            )
            descriptor = WorkhorseSemanticCheckpointOccurrenceV2.from_dict(
                descriptor_raw
            )
            commit_path = (
                descriptor_path.parent / WORKHORSE_SEMANTIC_OCCURRENCE_COMMIT_NAME
            )
            commit = WorkhorseSemanticCheckpointOccurrenceCommitV2.from_dict(
                _read_exact_json_file(
                    commit_path,
                    label="Workhorse checkpoint occurrence commit",
                )
            )
        except Exception as exc:
            raise CudaFullEvolutionV2LaunchpadAdapterError(
                f"semantic restore source is not canonical: {exc}"
            ) from exc
        inspection = inspect_workhorse_semantic_checkpoint_occurrence_v2(
            descriptor_path.parent
        )
        occurrence_digest = descriptor.occurrence_digest()
        if (
            descriptor_path.name != WORKHORSE_SEMANTIC_OCCURRENCE_DESCRIPTOR_NAME
            or _sha256_file(descriptor_path)
            != binding["checkpoint_occurrence_record_file_sha256"]
            or occurrence_digest.hex != binding["checkpoint_occurrence_identity_sha256"]
            or descriptor.base_checkpoint_identity_digest.hex
            != binding["base_checkpoint_identity_sha256"]
            or commit.occurrence_digest != occurrence_digest
            or inspection.canonical_visible is not True
            or inspection.occurrence_digest != occurrence_digest
        ):
            raise CudaFullEvolutionV2LaunchpadAdapterError(
                "semantic restore occurrence identity or visibility changed"
            )
        source_record_reader = getattr(
            self.issuer, "restore_source_attempt_record", None
        )
        if not callable(source_record_reader):
            raise CudaFullEvolutionV2LaunchpadAdapterError(
                "Scheduler lacks the canonical restore-source record seam"
            )
        source_record = dict(
            source_record_reader(attempt_id=str(descriptor.attempt_id))
        )
        terminal = source_record.get("terminal")
        if not isinstance(terminal, Mapping) or terminal.get("status") != "COMPLETED":
            raise CudaFullEvolutionV2LaunchpadAdapterError(
                "semantic restore source Attempt is not canonically completed"
            )
        authorization = RestoreCheckpointAuthorizationV1(
            checkpoint_digest=occurrence_digest,
            run_id=descriptor.run_id,
            source_attempt_id=descriptor.attempt_id,
            source_attempt_terminal_digest=canonical_sha256(dict(terminal)),
            absolute_next_tick=descriptor.absolute_next_tick,
            sealed_bundle_digest=descriptor.source_materials_digest,
            completion_marker_digest=commit.commit_digest(),
        ).to_dict()
        checkpoint_digest = authorization["checkpoint_digest"]
        initial_tick = descriptor.absolute_next_tick
        projection_conditions = {
            "run_parent_checkpoint_matches_occurrence": (
                run_occurrence.get("parent_checkpoint_digest") == checkpoint_digest
            ),
            "admission_restore_checkpoint_matches_occurrence": (
                admission.get("requested_restore_checkpoint_digest")
                == checkpoint_digest
            ),
            "last_dispatched_frontier_matches_checkpoint": (
                initial_frontiers.get("last_dispatched_next_tick") == initial_tick
            ),
            "last_computed_frontier_matches_checkpoint": (
                initial_frontiers.get("last_computed_next_tick") == initial_tick
            ),
            "durable_frontier_matches_checkpoint": (
                initial_frontiers.get("durable_checkpoint_next_tick") == initial_tick
            ),
            "published_frontier_not_ahead_of_checkpoint": (
                initial_frontiers.get("published_artifact_next_tick") <= initial_tick
            ),
        }
        if not all(projection_conditions.values()):
            raise CudaFullEvolutionV2LaunchpadAdapterError(
                "child Run projection differs from its external checkpoint occurrence: "
                + json.dumps(
                    projection_conditions,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
        restore_workhorse_semantic_checkpoint_occurrence_v2(
            descriptor_path.parent,
            expected_run_id=descriptor.run_id,
            expected_attempt_id=descriptor.attempt_id,
            expected_absolute_next_tick=descriptor.absolute_next_tick,
            expected_semantic_spec_digest=descriptor.semantic_spec_digest,
            expected_implementation_digest=descriptor.implementation_digest,
            expected_occurrence_digest=occurrence_digest,
        )
        return authorization, source_record

    def bind_attempt_authority_v1(
        self,
        *,
        target: Path,
        request: Mapping[str, Any],
        campaign_plan: Mapping[str, Any],
        bound_execution_request: Mapping[str, Any],
        source_identity: Mapping[str, Any],
        preflight: Mapping[str, Any],
        shadow_projection: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        try:
            if target.exists() or target.is_symlink():
                return self._reopen_attempt_authority_v1(
                    target=target,
                    request=request,
                    campaign_plan=campaign_plan,
                    bound_execution_request=bound_execution_request,
                    preflight=preflight,
                    shadow_projection=shadow_projection,
                )
            return self._bind_attempt_authority_v1(
                target=target,
                request=request,
                campaign_plan=campaign_plan,
                bound_execution_request=bound_execution_request,
                source_identity=source_identity,
                preflight=preflight,
                shadow_projection=shadow_projection,
            )
        except (
            ActorError,
            CudaFullEvolutionV2LaunchpadAdapterError,
            OSError,
            ValueError,
        ) as exc:
            return {
                "execution_authority_granted": False,
                "reason_codes": [
                    f"CRCA_SCHEDULER_ATTEMPT_REJECTED:{type(exc).__name__}:{exc}"
                ],
            }

    def _reopen_attempt_authority_v1(
        self,
        *,
        target: Path,
        request: Mapping[str, Any],
        campaign_plan: Mapping[str, Any],
        bound_execution_request: Mapping[str, Any],
        preflight: Mapping[str, Any],
        shadow_projection: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """Reopen one exact live Attempt after a fail-closed outer wait.

        The run-local pointer and Scheduler bundle remain immutable.  A retry
        may renew their existing continuation fence only when every stable
        request, plan, projection, route, and admission input still matches.
        """

        projection = validate_shadow_projection_v1(shadow_projection)
        route = _route_bindings(bound_execution_request)
        layers = projection["layers"]
        semantic = dict(layers["semantic"]["payload"])
        implementation = dict(layers["implementation"]["payload"])
        lifecycle = dict(layers["lifecycle"]["payload"])
        admission = dict(layers["attempt_admission_request"]["payload"])
        concretized = dict(layers["concretized_execution"]["payload"])
        run = dict(projection["shadow_run_projection"]["candidate"])
        attempt_projection = projection["shadow_attempt_projection"]
        attempt_id = str(attempt_projection["candidate_attempt_id"])
        run_id = str(run["run_id"])
        start_tick = int(semantic["absolute_start_next_tick"])
        frontiers = AttemptFrontiersV1(start_tick, start_tick, start_tick, start_tick)
        retry = AttemptRetryBasisV1(
            run_id=RunId(run_id),
            candidate_attempt_id=AttemptId(attempt_id),
            prior_attempt_id=None,
            code=AttemptRetryBasisCodeV1.INITIAL_EXECUTION,
            reason="Fresh canonical ExecutionRequestV2 occurrence.",
            semantic_spec_digest=_content_digest(layers["semantic"]["digest"]),
            prior_concretized_execution_digest=None,
            candidate_concretized_execution_digest=_content_digest(
                layers["concretized_execution"]["digest"]
            ),
            basis_evidence_digests=(),
            implementation_conformance_digest=None,
        )
        plan_identity = {
            key: value for key, value in campaign_plan.items() if key != "plan_sha256"
        }
        request_identity = {
            key: value
            for key, value in bound_execution_request.items()
            if key not in {"bound_request_sha256", "execution_route_contract"}
        }
        preflight_projection = {
            "schema_version": "crca_launchpad_preflight_authority_projection_v1",
            "provider_id": route["provider_id"],
            "route_id": route["route_id"],
            "device_id": route["device_id"],
            "native_artifact_digest": (
                dict(route["native_artifact_digest"])
                if isinstance(route.get("native_artifact_digest"), Mapping)
                else {
                    "algorithm": "sha256",
                    "hex": str(campaign_plan["artifacts"]["cuda_extension"]["sha256"]),
                }
            ),
            "batch_size": route["batch_size"],
            "ticks_per_dispatch": route["ticks_per_dispatch"],
            "execution_mode": route["execution_mode"],
        }
        authority = self._authority(target)
        validation = authority.validate()
        raw = dict(authority.raw)
        pointer = json.loads(target.read_text(encoding="ascii"))
        current_process = _current_process_identity_v1()
        start_effect = dict(raw.get("start_effect") or {})
        if (
            pointer.get("owner_pid") != current_process["pid"]
            or pointer.get("owner_process_start_ticks")
            != current_process["process_start_ticks"]
            or pointer.get("owner_process_start_identity")
            != current_process["process_start_identity"]
            or start_effect.get("process_id") != current_process["pid"]
            or start_effect.get("process_start_identity")
            != current_process["process_start_identity"]
        ):
            raise CudaFullEvolutionV2LaunchpadAdapterError(
                "existing Attempt belongs to another execution-owner process; "
                "fresh-process continuation requires a successor Attempt"
            )
        expected_payloads = {
            "execution_request": request_identity,
            "campaign_plan": plan_identity,
            "preflight_receipt": preflight_projection,
            "lifecycle_binding": lifecycle,
            "semantic_run_spec": semantic,
            "run_occurrence": run,
            "execution_implementation": implementation,
            "concretized_execution": concretized,
            "attempt_admission_request": admission,
            "retry_basis": retry.to_dict(),
        }
        mismatched = [
            name
            for name, expected in expected_payloads.items()
            if canonical_sha256(raw.get(name)) != canonical_sha256(expected)
        ]
        claim = dict(raw.get("claim") or {})
        expected_claim = {
            "run_id": run_id,
            "attempt_id": attempt_id,
            "run_root": str(target.parents[2].resolve()),
            "provider_id": route["provider_id"],
            "route_id": route["route_id"],
            "device_id": route["device_id"],
            "batch_size": route["batch_size"],
            "ticks_per_dispatch": route["ticks_per_dispatch"],
            "execution_mode": route["execution_mode"],
            "programme_binding_digest": scheduler_sha256(
                {"adapter": PROCESS_OWNER_ID_V1, "target": str(target.resolve())}
            ).to_dict(),
        }
        mismatched.extend(
            f"claim.{name}"
            for name, expected in expected_claim.items()
            if claim.get(name) != expected
        )
        exact_resources, candidate_resources, resource_scope = (
            _scheduler_resource_topology_v1(route, run_id=run_id)
        )
        resource_lease = raw.get("resource_start_lease")
        selected_resources = (
            tuple(resource_lease.get("resource_ids") or ())
            if isinstance(resource_lease, Mapping)
            else ()
        )
        if (
            (candidate_resources and selected_resources not in candidate_resources)
            or (not candidate_resources and selected_resources != exact_resources)
            or not isinstance(resource_lease, Mapping)
            or resource_lease.get("resource_scope_digest")
            != scheduler_sha256(resource_scope).to_dict()
        ):
            mismatched.append("resource_start_lease.gpu_topology")
        if raw.get("attempt_start", {}).get("initial_frontiers") != frontiers.to_dict():
            mismatched.append("attempt_start.initial_frontiers")
        if mismatched:
            raise CudaFullEvolutionV2LaunchpadAdapterError(
                "existing Attempt differs from exact continuation inputs: "
                + ", ".join(sorted(mismatched))
            )
        renewed_at = self.now_fn()
        continuation_ttl_seconds = _continuation_ttl_seconds_v1(campaign_plan)
        renewal = self.issuer.heartbeat_or_renew(
            attempt_id=attempt_id,
            process_owner_id=PROCESS_OWNER_ID_V1,
            process_owner_digest=_scheduler_digest(claim["execution_owner_digest"]),
            process_id=current_process["pid"],
            process_start_identity=current_process["process_start_identity"],
            at_utc=renewed_at,
            renew_for_seconds=continuation_ttl_seconds,
        )
        if renewal.get("status") != "RENEWED":
            raise CudaFullEvolutionV2LaunchpadAdapterError(
                "Scheduler did not renew the exact continuation fence"
            )
        bundle_path = Path(str(pointer["scheduler_bundle_path"])).resolve()
        bundle_digest = canonical_sha256(raw)
        return {
            "execution_authority_granted": True,
            "attempt_authority_path": str(target.resolve()),
            "scheduler_bundle_path": str(bundle_path),
            "scheduler_bundle_digest": bundle_digest.to_dict(),
            "validation": validation,
            "continuation_reopened": True,
            "continuation_renewed_at_utc": renewed_at,
            "continuation_ttl_seconds": continuation_ttl_seconds,
        }

    def _bind_attempt_authority_v1(
        self,
        *,
        target: Path,
        request: Mapping[str, Any],
        campaign_plan: Mapping[str, Any],
        bound_execution_request: Mapping[str, Any],
        source_identity: Mapping[str, Any],
        preflight: Mapping[str, Any],
        shadow_projection: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        if target.exists() or target.is_symlink():
            raise CudaFullEvolutionV2LaunchpadAdapterError(
                "fresh Attempt target already exists"
            )
        projection = validate_shadow_projection_v1(shadow_projection)
        route = _route_bindings(bound_execution_request)
        layers = projection["layers"]
        semantic = dict(layers["semantic"]["payload"])
        implementation = dict(layers["implementation"]["payload"])
        lifecycle = dict(layers["lifecycle"]["payload"])
        admission = dict(layers["attempt_admission_request"]["payload"])
        concretized = dict(layers["concretized_execution"]["payload"])
        run = dict(projection["shadow_run_projection"]["candidate"])
        attempt_projection = projection["shadow_attempt_projection"]
        attempt_id = str(attempt_projection["candidate_attempt_id"])
        run_id = str(run["run_id"])
        start_tick = int(semantic["absolute_start_next_tick"])
        frontiers = AttemptFrontiersV1(start_tick, start_tick, start_tick, start_tick)
        retry = AttemptRetryBasisV1(
            run_id=RunId(run_id),
            candidate_attempt_id=AttemptId(attempt_id),
            prior_attempt_id=None,
            code=AttemptRetryBasisCodeV1.INITIAL_EXECUTION,
            reason="Fresh canonical ExecutionRequestV2 occurrence.",
            semantic_spec_digest=_content_digest(layers["semantic"]["digest"]),
            prior_concretized_execution_digest=None,
            candidate_concretized_execution_digest=_content_digest(
                layers["concretized_execution"]["digest"]
            ),
            basis_evidence_digests=(),
            implementation_conformance_digest=None,
        )
        plan_identity = {
            key: value for key, value in campaign_plan.items() if key != "plan_sha256"
        }
        request_identity = {
            key: value
            for key, value in bound_execution_request.items()
            if key not in {"bound_request_sha256", "execution_route_contract"}
        }
        preflight_projection = {
            "schema_version": "crca_launchpad_preflight_authority_projection_v1",
            "provider_id": route["provider_id"],
            "route_id": route["route_id"],
            "device_id": route["device_id"],
            "native_artifact_digest": (
                dict(route["native_artifact_digest"])
                if isinstance(route.get("native_artifact_digest"), Mapping)
                else {
                    "algorithm": "sha256",
                    "hex": str(campaign_plan["artifacts"]["cuda_extension"]["sha256"]),
                }
            ),
            "batch_size": route["batch_size"],
            "ticks_per_dispatch": route["ticks_per_dispatch"],
            "execution_mode": route["execution_mode"],
        }
        process_identity = _current_process_identity_v1()
        owner_digest = scheduler_sha256(
            {
                "owner_id": PROCESS_OWNER_ID_V1,
                "source_identity": dict(source_identity),
                "event_transport": route["event_transport"],
                "process_identity": process_identity,
            }
        )
        claim_created_at_utc = self.now_fn()
        run_root = str(target.parents[2].resolve())
        claim = LaunchpadAttemptClaimV1.create(
            scheduler_owner_id="tools.scheduler.launchpad_attempts",
            scheduler_owner_digest=self.issuer.owner_digest(),
            execution_owner_id=PROCESS_OWNER_ID_V1,
            execution_owner_digest=owner_digest,
            run_id=run_id,
            attempt_id=attempt_id,
            lineage_scope_digest=_continuation_lineage_scope_digest_v1(
                run_id=run_id,
                semantic_spec_digest=layers["semantic"]["digest"],
                run_root=run_root,
            ),
            execution_request_digest=scheduler_sha256(request_identity),
            campaign_plan_digest=scheduler_sha256(plan_identity),
            source_digest=_scheduler_digest(implementation["source_tree_digest"]),
            baseline_digest=_scheduler_digest(
                projection["current_lock_bindings"]["baseline_digest"]
            ),
            preflight_receipt_digest=scheduler_sha256(preflight_projection),
            lifecycle_binding_digest=scheduler_sha256(lifecycle),
            lifecycle_contract_digest=_scheduler_digest(
                lifecycle["artifact_lifecycle_contract_digest"]
            ),
            run_root=run_root,
            provider_id=str(route["provider_id"]),
            route_id=str(route["route_id"]),
            device_id=str(route["device_id"]),
            native_artifact_digest=_scheduler_digest(
                preflight_projection["native_artifact_digest"]
            ),
            batch_size=int(route["batch_size"]),
            ticks_per_dispatch=int(route["ticks_per_dispatch"]),
            execution_mode=str(route["execution_mode"]),
            semantic_run_spec_digest=scheduler_sha256(semantic),
            run_occurrence_digest=scheduler_sha256(run),
            execution_implementation_digest=scheduler_sha256(implementation),
            concretized_execution_digest=scheduler_sha256(concretized),
            attempt_admission_request_digest=scheduler_sha256(admission),
            retry_basis_digest=scheduler_sha256(retry.to_dict()),
            post_admission_attempt_plan_digest=scheduler_sha256(
                {
                    "outer_request": dict(request),
                    "preflight": dict(preflight),
                    "event_transport": route["event_transport"],
                }
            ),
            programme_binding_digest=scheduler_sha256(
                {"adapter": PROCESS_OWNER_ID_V1, "target": str(target.resolve())}
            ),
            initial_frontiers_digest=scheduler_sha256(frontiers.to_dict()),
            created_at_utc=claim_created_at_utc,
        )
        resource = dict(
            self.resource_evidence_factory(
                repo_root=self.repo_root,
                route=route,
                campaign_plan=campaign_plan,
                at_utc=claim_created_at_utc,
            )
        )
        lease_issued_at_utc = self.now_fn()
        resource_ids, resource_candidate_sets, resource_scope = (
            _scheduler_resource_topology_v1(route, run_id=run_id)
        )
        restore_authorization, restore_source_attempt_record = (
            self._semantic_restore_authority_v2(
                campaign_plan=campaign_plan,
                run_occurrence=run,
                admission=admission,
                initial_frontiers=frontiers.to_dict(),
            )
        )

        def prepare_once(
            evidence: Mapping[str, Any], *, prepared_at_utc: str
        ) -> Mapping[str, Any]:
            return self.issuer.prepare(
                claim=claim,
                execution_request=request_identity,
                campaign_plan=plan_identity,
                preflight_receipt=preflight_projection,
                lifecycle_binding=lifecycle,
                semantic_run_spec=semantic,
                run_occurrence=run,
                execution_implementation=implementation,
                concretized_execution=concretized,
                attempt_admission_request=admission,
                retry_basis=retry.to_dict(),
                initial_frontiers=frontiers.to_dict(),
                monotonic_ordinal=int(attempt_projection["monotonic_ordinal"]),
                nonce=str(attempt_projection["nonce"]),
                parent_attempt_id=None,
                resource_ids=resource_ids,
                resource_scope_digest=scheduler_sha256(resource_scope),
                resource_observation_id=str(evidence["resource_observation_id"]),
                resource_observation=dict(evidence["resource_observation"]),
                resource_observed_at_utc=str(evidence["resource_observed_at_utc"]),
                resource_recommendation_id=str(evidence["resource_recommendation_id"]),
                resource_recommendation=dict(evidence["resource_recommendation"]),
                live_provider_check=dict(evidence["live_provider_check"]),
                at_utc=prepared_at_utc,
                continuation_ttl_seconds=_continuation_ttl_seconds_v1(campaign_plan),
                restore_authorization=restore_authorization,
                restore_source_attempt_record=restore_source_attempt_record,
                resource_candidate_sets=resource_candidate_sets,
            )

        try:
            prepared = prepare_once(resource, prepared_at_utc=lease_issued_at_utc)
        except ActorError as exc:
            if exc.code != "launchpad_attempt_resource_wait":
                raise
            sweep_at_utc = self.now_fn()
            reclaim_conflict_ids = resource_ids or (f"gpu:{route['device_id']}",)
            self.reclaim_dead_gpu_owners_v1(
                at_utc=sweep_at_utc,
                conflicting_resource_ids=reclaim_conflict_ids,
            )
            # Reconcile may have completed a durable terminal intent and
            # released the conflicting lane directly, before the nested
            # terminal-head repair pass.  One fresh bounded prepare is the
            # authoritative way to decide whether the conflict actually
            # remains; no summary counter is used as admission authority.
            resource = dict(
                self.resource_evidence_factory(
                    repo_root=self.repo_root,
                    route=route,
                    campaign_plan=campaign_plan,
                    at_utc=sweep_at_utc,
                )
            )
            lease_issued_at_utc = self.now_fn()
            prepared = prepare_once(resource, prepared_at_utc=lease_issued_at_utc)
        selected_resource = dict(
            prepared.get("preparation", {}).get("resource_start_lease") or {}
        )
        selected_ids = selected_resource.get("resource_ids")
        if (
            not isinstance(selected_ids, list)
            or (
                resource_candidate_sets
                and tuple(selected_ids) not in resource_candidate_sets
            )
            or (not resource_candidate_sets and tuple(selected_ids) != resource_ids)
        ):
            raise CudaFullEvolutionV2LaunchpadAdapterError(
                "Scheduler selected resources outside the exact route topology"
            )
        started = self.issuer.start(
            attempt_id=attempt_id,
            process_owner_id=PROCESS_OWNER_ID_V1,
            process_owner_digest=owner_digest,
            process_id=process_identity["pid"],
            process_start_identity=process_identity["process_start_identity"],
            at_utc=lease_issued_at_utc,
        )
        bundle_path = Path(str(started["bundle_path"])).resolve()
        authoritative = authoritative_attempt_bundle_from_path(bundle_path)
        validation = authoritative.validate()
        bundle_digest = canonical_sha256(dict(authoritative.raw))
        pointer_core = {
            "schema_version": ATTEMPT_POINTER_SCHEMA_V1,
            "scheduler_bundle_path": str(bundle_path),
            "scheduler_bundle_digest": bundle_digest.to_dict(),
            "run_id": run_id,
            "attempt_id": attempt_id,
            "provider_id": route["provider_id"],
            "route_id": route["route_id"],
            "event_transport": route["event_transport"],
            "execution_route_sha256": bound_execution_request.get(
                "execution_route_sha256"
            ),
            "device_identity": dict(route["device_identity"]),
            "selected_resource_ids": list(selected_ids),
            "resource_scope_digest": selected_resource.get("resource_scope_digest"),
            "owner_pid": process_identity["pid"],
            "owner_process_start_ticks": process_identity["process_start_ticks"],
            "owner_process_start_identity": process_identity["process_start_identity"],
            "issued_at_utc": lease_issued_at_utc,
        }
        pointer = {
            **pointer_core,
            "pointer_sha256": canonical_sha256(pointer_core).hex,
        }
        _write_immutable_json(target, pointer)
        return {
            "execution_authority_granted": True,
            "attempt_authority_path": str(target.resolve()),
            "scheduler_bundle_path": str(bundle_path),
            "scheduler_bundle_digest": bundle_digest.to_dict(),
            "validation": validation,
        }

    @staticmethod
    def _authority(path: Path) -> AuthoritativeAttemptBundleV1:
        try:
            metadata = path.lstat()
        except OSError as exc:
            raise CudaFullEvolutionV2LaunchpadAdapterError(
                f"Attempt pointer is unavailable: {exc}"
            ) from exc
        if (
            not stat.S_ISREG(metadata.st_mode)
            or path.is_symlink()
            or metadata.st_nlink != 1
            or metadata.st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
        ):
            raise CudaFullEvolutionV2LaunchpadAdapterError(
                "Attempt pointer must be one sealed, single-link regular file"
            )
        raw = json.loads(path.read_text(encoding="ascii"))
        if not isinstance(raw, Mapping):
            raise CudaFullEvolutionV2LaunchpadAdapterError(
                "Attempt pointer is not an object"
            )
        supplied = raw.get("pointer_sha256")
        core = {key: value for key, value in raw.items() if key != "pointer_sha256"}
        if (
            raw.get("schema_version") != ATTEMPT_POINTER_SCHEMA_V1
            or supplied != canonical_sha256(core).hex
        ):
            raise CudaFullEvolutionV2LaunchpadAdapterError(
                "Attempt pointer digest is invalid"
            )
        bundle = authoritative_attempt_bundle_from_path(
            str(raw.get("scheduler_bundle_path"))
        )
        if (
            canonical_sha256(dict(bundle.raw)).to_dict()
            != raw.get("scheduler_bundle_digest")
            or str(bundle.attempt_start.attempt_id) != raw.get("attempt_id")
            or str(bundle.run_occurrence.run_id) != raw.get("run_id")
        ):
            raise CudaFullEvolutionV2LaunchpadAdapterError(
                "Scheduler bundle differs from its run-local pointer"
            )
        return bundle

    def bind_workhorse_parent_context_v2(
        self,
        *,
        attempt_authority_path: Path,
        target: Path,
        campaign_plan: Mapping[str, Any],
        bound_execution_request: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """Reopen one live Scheduler Attempt and project one exact row context.

        The returned mapping is deliberately non-authoritative.  Its only
        trusted construction path is this composition boundary, after the
        Scheduler issuer has accepted a current-owner heartbeat against the
        actual Attempt head.
        """

        source_batch_id = bound_execution_request.get("source_batch_id")
        if not isinstance(source_batch_id, str) or not source_batch_id:
            raise CudaFullEvolutionV2LaunchpadAdapterError(
                "Workhorse request lacks its one physical batch identity"
            )
        target = target.expanduser().resolve()
        if target.exists() or target.is_symlink():
            raise CudaFullEvolutionV2LaunchpadAdapterError(
                "fresh Workhorse parent-context target already exists"
            )
        pointer_path = attempt_authority_path.expanduser().resolve()
        authority = self._authority(pointer_path)
        authority.validate()
        plan_identity = {
            key: value for key, value in campaign_plan.items() if key != "plan_sha256"
        }
        request_identity = {
            key: value
            for key, value in bound_execution_request.items()
            if key not in {"bound_request_sha256", "execution_route_contract"}
        }
        if (
            scheduler_sha256(authority.raw.get("campaign_plan")).to_dict()
            != scheduler_sha256(plan_identity).to_dict()
            or scheduler_sha256(authority.raw.get("execution_request")).to_dict()
            != scheduler_sha256(request_identity).to_dict()
        ):
            raise CudaFullEvolutionV2LaunchpadAdapterError(
                "Workhorse caller differs from the Scheduler-bound plan or request"
            )
        route = _route_bindings(bound_execution_request)
        operation = route.get("workhorse_operation")
        artifact = route.get("workhorse_artifact")
        checkpoint = route.get("workhorse_checkpoint")
        if not all(
            isinstance(row, Mapping) for row in (operation, artifact, checkpoint)
        ):
            raise CudaFullEvolutionV2LaunchpadAdapterError(
                "Scheduler-bound request is not a Workhorse row"
            )
        claim = dict(authority.raw["claim"])
        expected_target = (
            Path(str(claim.get("run_root") or "")).resolve()
            / "batches"
            / source_batch_id
            / "workhorse-parent-context.json"
        ).resolve()
        if target != expected_target:
            raise CudaFullEvolutionV2LaunchpadAdapterError(
                "Workhorse parent context must use its one canonical batch target"
            )
        device = route.get("device_identity")
        if not isinstance(device, Mapping):
            raise CudaFullEvolutionV2LaunchpadAdapterError(
                "Workhorse operation identity is incomplete"
            )
        operation_name = operation.get("operation")
        contract = operation.get("contract")
        if operation_name in {
            "EXECUTION_ROW",
            "SEMANTIC_EVIDENCE_CASE",
            "SCIENCE_COMPATIBILITY_DEMO",
            "INDEXED_ACS_CONTINUATION",
        }:
            if not isinstance(contract, Mapping):
                raise CudaFullEvolutionV2LaunchpadAdapterError(
                    "Workhorse execution contract is absent"
                )
            shape_id = contract.get("shape_id")
            backend_id = contract.get("backend_id")
            compiled_temporal_k: int | list[int] = contract.get("temporal_k")
            reserve_policy = None
            absolute_tick_span = operation.get("absolute_tick_span")
            science_digest = checkpoint.get("science_continuation_contract_sha256")
            source_checkpoint = checkpoint.get("source_checkpoint_identity_sha256")
            parent_contract: dict[str, Any] | None = dict(contract)
            if operation_name == "INDEXED_ACS_CONTINUATION":
                compiled_temporal_k = list(
                    operation.get("compiled_temporal_k") or []
                )
            if operation_name == "SEMANTIC_EVIDENCE_CASE":
                semantic_operation = _workhorse_semantic_operation_v2(campaign_plan)
                if (
                    semantic_operation is None
                    or operation.get("semantic_case_contract")
                    != semantic_operation["semantic_case_contract"]
                    or operation.get("operation_contract_sha256")
                    != semantic_operation["semantic_case_contract"][
                        "operation_contract_sha256"
                    ]
                ):
                    raise CudaFullEvolutionV2LaunchpadAdapterError(
                        "Workhorse semantic parent differs from its campaign law"
                    )
        elif operation_name == "DEVICE_ENVELOPE_CALIBRATION":
            if contract is not None:
                raise CudaFullEvolutionV2LaunchpadAdapterError(
                    "Workhorse calibration gained a row contract"
                )
            shape_id = operation.get("shape_id")
            backend_id = operation.get("backend_id")
            compiled_temporal_k = list(operation.get("compiled_temporal_k") or [])
            reserve_policy = operation.get("reserve_policy")
            absolute_tick_span = operation.get("absolute_tick_span")
            science_digest = None
            source_checkpoint = None
            parent_contract = None
        else:
            raise CudaFullEvolutionV2LaunchpadAdapterError(
                "Workhorse operation discriminator changed"
            )
        expected_claim = {
            "provider_id": route["provider_id"],
            "route_id": route["route_id"],
            "device_id": route["device_id"],
            "batch_size": route["batch_size"],
            "ticks_per_dispatch": route["ticks_per_dispatch"],
            "execution_mode": route["execution_mode"],
            "execution_request_digest": scheduler_sha256(request_identity).to_dict(),
            "campaign_plan_digest": scheduler_sha256(plan_identity).to_dict(),
            "native_artifact_digest": dict(route["native_artifact_digest"]),
        }
        if any(claim.get(field) != value for field, value in expected_claim.items()):
            raise CudaFullEvolutionV2LaunchpadAdapterError(
                "Workhorse row differs from its exact Scheduler claim"
            )
        heartbeat_at = self.now_fn()
        current_process = _current_process_identity_v1()
        heartbeat = self.issuer.heartbeat_or_renew(
            attempt_id=str(authority.attempt_start.attempt_id),
            process_owner_id=PROCESS_OWNER_ID_V1,
            process_owner_digest=_scheduler_digest(claim["execution_owner_digest"]),
            process_id=current_process["pid"],
            process_start_identity=current_process["process_start_identity"],
            at_utc=heartbeat_at,
        )
        event = heartbeat.get("event")
        if (
            heartbeat.get("status") != "HEARTBEAT"
            or not isinstance(event, Mapping)
            or event.get("code") != "HEARTBEAT"
            or event.get("event_at_utc") != heartbeat_at
        ):
            raise CudaFullEvolutionV2LaunchpadAdapterError(
                "Scheduler did not prove a current live Workhorse lease"
            )
        # Reopen the exact immutable pointer again after the live-state
        # mutation, closing substitution between validation and projection.
        reopened = self._authority(pointer_path)
        if dict(reopened.raw) != dict(authority.raw):
            raise CudaFullEvolutionV2LaunchpadAdapterError(
                "Workhorse Scheduler bundle changed during heartbeat"
            )
        pointer = json.loads(pointer_path.read_text(encoding="ascii"))
        bundle_path = Path(str(pointer["scheduler_bundle_path"])).resolve()
        process_start_ticks = current_process["process_start_ticks"]
        lineage = canonical_sha256(
            {
                "lineage_scope_digest": claim["lineage_scope_digest"],
                "attempt_start_digest": authority.attempt_start.start_digest().to_dict(),
                "retry_basis_digest": claim["retry_basis_digest"],
            }
        ).hex
        core = {
            "schema_version": "eiel_workhorse_canonical_parent_attempt_binding_v2",
            "authority_source": "CANONICAL_PROVIDER_AFTER_SCHEDULER_REOPEN",
            "execution_authority_granted": False,
            "exclusive_device": True,
            "attempt_authority_path": str(pointer_path),
            "attempt_authority_file_sha256": _sha256_file(pointer_path),
            "scheduler_bundle_path": str(bundle_path),
            "scheduler_bundle_file_sha256": _sha256_file(bundle_path),
            "scheduler_bundle_digest": canonical_sha256(dict(authority.raw)).hex,
            "scheduler_live_heartbeat_sha256": scheduler_sha256(dict(event)).hex,
            "scheduler_live_heartbeat_at_utc": heartbeat_at,
            "run_id": str(authority.run_occurrence.run_id),
            "attempt_id": str(authority.attempt_start.attempt_id),
            "attempt_lineage_sha256": lineage,
            "execution_request_sha256": claim["execution_request_digest"]["hex"],
            "campaign_plan_sha256": claim["campaign_plan_digest"]["hex"],
            "semantic_run_spec_sha256": claim["semantic_run_spec_digest"]["hex"],
            "execution_implementation_sha256": claim["execution_implementation_digest"][
                "hex"
            ],
            "concretized_execution_sha256": claim["concretized_execution_digest"][
                "hex"
            ],
            "attempt_admission_request_sha256": claim[
                "attempt_admission_request_digest"
            ]["hex"],
            "profile_id": bound_execution_request.get("profile_id"),
            "source_batch_id": source_batch_id,
            "route_id": route["route_id"],
            "shape_id": shape_id,
            "backend_id": backend_id,
            "artifact_sha256": artifact.get("artifact_sha256"),
            "sealed_artifact_identity_sha256": artifact.get(
                "sealed_artifact_identity_sha256"
            ),
            "device_uuid": device.get("uuid"),
            "device_identity": dict(device),
            "owner_pid": os.getpid(),
            "owner_process_start_ticks": process_start_ticks,
            "operation": operation.get("operation"),
            "operation_contract_sha256": operation.get("operation_contract_sha256"),
            "contract": parent_contract,
            "compiled_temporal_k": compiled_temporal_k,
            "reserve_policy": (
                dict(reserve_policy) if isinstance(reserve_policy, Mapping) else None
            ),
            "absolute_tick_span": dict(absolute_tick_span),
            "science_continuation_contract_sha256": science_digest,
            "source_checkpoint_identity_sha256": source_checkpoint,
        }
        parent = {**core, "binding_sha256": canonical_sha256(core).hex}
        _write_immutable_json(target, parent)
        reopened_parent = json.loads(target.read_text(encoding="ascii"))
        if reopened_parent != parent:
            raise CudaFullEvolutionV2LaunchpadAdapterError(
                "sealed Workhorse parent context differs after publication"
            )
        return parent

    def validate_workhorse_external_checkpoint_source_v2(
        self,
        binding: Mapping[str, Any],
        *,
        expected_identity_sha256: str,
    ) -> Mapping[str, Any]:
        """Reopen a visible CRCA Workhorse occurrence for one child case."""

        try:
            from engine.perf.cuda_full_evolution_v2.workhorse.semantic_cases import (
                validate_external_checkpoint_source_binding_v2,
            )

            validated = validate_external_checkpoint_source_binding_v2(
                binding,
                expected_identity_sha256=expected_identity_sha256,
            )
            descriptor_path = Path(
                validated["checkpoint_occurrence_record_path"]
            ).resolve()
            descriptor = WorkhorseSemanticCheckpointOccurrenceV2.from_dict(
                _read_exact_json_file(
                    descriptor_path,
                    label="Workhorse checkpoint occurrence descriptor",
                )
            )
            inspection = inspect_workhorse_semantic_checkpoint_occurrence_v2(
                descriptor_path.parent
            )
            restored = restore_workhorse_semantic_checkpoint_occurrence_v2(
                descriptor_path.parent,
                expected_run_id=descriptor.run_id,
                expected_attempt_id=descriptor.attempt_id,
                expected_absolute_next_tick=descriptor.absolute_next_tick,
                expected_semantic_spec_digest=descriptor.semantic_spec_digest,
                expected_implementation_digest=descriptor.implementation_digest,
                expected_occurrence_digest=descriptor.occurrence_digest(),
            )
        except Exception as exc:
            raise CudaFullEvolutionV2LaunchpadAdapterError(
                f"external Workhorse checkpoint source cannot be reopened: {exc}"
            ) from exc
        manifest = restored.get("manifest")
        base = (
            manifest.get("base_checkpoint") if isinstance(manifest, Mapping) else None
        )
        identity = base.get("identity") if isinstance(base, Mapping) else None
        if (
            descriptor_path.name != WORKHORSE_SEMANTIC_OCCURRENCE_DESCRIPTOR_NAME
            or inspection.canonical_visible is not True
            or descriptor.occurrence_digest().hex
            != validated["checkpoint_occurrence_identity_sha256"]
            or descriptor.base_checkpoint_identity_digest.hex
            != expected_identity_sha256
            or not isinstance(identity, Mapping)
            or identity.get("base_checkpoint_identity_sha256")
            != expected_identity_sha256
            or str(Path(validated["base_checkpoint_manifest_path"]).resolve())
            != str(
                next(
                    material.absolute_path
                    for material in descriptor.source_materials
                    if material.material_role.value == "BASE_CHECKPOINT_MANIFEST"
                )
            )
        ):
            raise CudaFullEvolutionV2LaunchpadAdapterError(
                "external Workhorse checkpoint occurrence differs from its binding"
            )
        return dict(validated)

    def materialize_workhorse_semantic_checkpoint_occurrence_v2(
        self,
        *,
        canonical_parent_authority_binding: Mapping[str, Any],
        case_fragment_path: Path,
        science_checkpoint_compatibility_path: Path,
        occurrence_root: Path,
    ) -> Mapping[str, Any]:
        """Publish case 20 through the canonical CRCA occurrence boundary."""

        try:
            from engine.perf.cuda_full_evolution_v2.workhorse.canonical_rows import (
                validate_workhorse_canonical_parent_authority_binding_v2,
            )
            from engine.perf.cuda_full_evolution_v2.workhorse.semantic_cases import (
                WORKHORSE_EXTERNAL_CHECKPOINT_AUTHORIZATION_SOURCE_V2,
                WORKHORSE_EXTERNAL_CHECKPOINT_SOURCE_SCHEMA_V2,
                reopen_workhorse_semantic_case_fragment_v2,
            )

            parent = validate_workhorse_canonical_parent_authority_binding_v2(
                canonical_parent_authority_binding,
                expected_operation="SEMANTIC_EVIDENCE_CASE",
            )
            fragment, _datasets = reopen_workhorse_semantic_case_fragment_v2(
                case_fragment_path
            )
        except Exception as exc:
            raise CudaFullEvolutionV2LaunchpadAdapterError(
                f"case 20 checkpoint source is invalid: {exc}"
            ) from exc
        base = fragment.get("base_checkpoint")
        identity = base.get("identity") if isinstance(base, Mapping) else None
        if (
            fragment.get("case_id") != "checkpoint_same_restore_sequence"
            or not isinstance(identity, Mapping)
            or fragment.get("canonical_parent_binding") != parent
        ):
            raise CudaFullEvolutionV2LaunchpadAdapterError(
                "only exact semantic case 20 may publish a base occurrence"
            )
        pointer = Path(str(parent["attempt_authority_path"])).resolve()
        authority = self._authority(pointer)
        if (
            str(authority.run_occurrence.run_id) != parent["run_id"]
            or str(authority.attempt_start.attempt_id) != parent["attempt_id"]
        ):
            raise CudaFullEvolutionV2LaunchpadAdapterError(
                "case 20 parent differs from its Scheduler Attempt"
            )
        checkpoint_identity = str(identity["base_checkpoint_identity_sha256"])
        descriptor, commit = materialize_workhorse_semantic_checkpoint_occurrence_v2(
            case_fragment_path,
            science_checkpoint_compatibility_path,
            occurrence_root,
            checkpoint_id=CheckpointId(
                f"workhorse-semantic-{checkpoint_identity[:24]}"
            ),
            run_id=authority.run_occurrence.run_id,
            attempt_id=authority.attempt_start.attempt_id,
            transaction_id=f"{parent['attempt_id']}:semantic-case-20",
            absolute_next_tick=int(identity["checkpoint_at_tick"]),
            semantic_spec_digest=authority.semantic_run_spec.semantic_spec_digest(),
            implementation_digest=(
                authority.execution_implementation.implementation_digest()
            ),
            committed_at_utc=self.now_fn(),
        )
        descriptor_path = (
            occurrence_root.resolve() / WORKHORSE_SEMANTIC_OCCURRENCE_DESCRIPTOR_NAME
        )
        commit_path = (
            occurrence_root.resolve() / WORKHORSE_SEMANTIC_OCCURRENCE_COMMIT_NAME
        )
        core = {
            "schema_version": WORKHORSE_EXTERNAL_CHECKPOINT_SOURCE_SCHEMA_V2,
            "authorization_source": (
                WORKHORSE_EXTERNAL_CHECKPOINT_AUTHORIZATION_SOURCE_V2
            ),
            "checkpoint_occurrence_record_path": str(descriptor_path),
            "checkpoint_occurrence_record_file_sha256": _sha256_file(descriptor_path),
            "checkpoint_occurrence_identity_sha256": (
                descriptor.occurrence_digest().hex
            ),
            "base_checkpoint_manifest_path": str(case_fragment_path.resolve()),
            "base_checkpoint_manifest_file_sha256": _sha256_file(
                case_fragment_path.resolve()
            ),
            "base_checkpoint_identity_sha256": checkpoint_identity,
        }
        binding = {**core, "binding_sha256": canonical_sha256(core).hex}
        self.validate_workhorse_external_checkpoint_source_v2(
            binding,
            expected_identity_sha256=checkpoint_identity,
        )
        if commit.occurrence_digest != descriptor.occurrence_digest():
            raise CudaFullEvolutionV2LaunchpadAdapterError(
                "case 20 occurrence commit differs from its descriptor"
            )
        return {
            "external_source_checkpoint_binding": binding,
            "artifact_paths": [str(descriptor_path), str(commit_path)],
        }

    def _materialize_workhorse_checkpoint_occurrence_v2(
        self,
        *,
        canonical_parent_authority_binding: Mapping[str, Any],
        checkpoint_manifest_path: Path,
        absolute_bundle_root: Path,
        occurrence_root: Path,
        parent_checkpoint_digest: Mapping[str, str] | None,
        expected_operation: str,
        checkpoint_id_prefix: str,
        owner_component: str,
        projection_schema: str,
        restore_state: bool,
    ) -> Mapping[str, Any]:
        """Admit one Workhorse checkpoint through the CRCA absolute-tick seam."""

        try:
            from engine.perf.cuda_full_evolution_v2.workhorse.authority import (
                inspect_workhorse_checkpoint_bundle_v2,
            )
            from engine.perf.cuda_full_evolution_v2.workhorse.canonical_rows import (
                validate_workhorse_canonical_parent_authority_binding_v2,
            )

            parent = validate_workhorse_canonical_parent_authority_binding_v2(
                canonical_parent_authority_binding,
                expected_operation=expected_operation,
            )
            checkpoint_path = checkpoint_manifest_path.expanduser().resolve()
            checkpoint, state_path = inspect_workhorse_checkpoint_bundle_v2(
                checkpoint_path
            )
        except Exception as exc:
            raise CudaFullEvolutionV2LaunchpadAdapterError(
                f"Workhorse checkpoint cannot enter CRCA: {exc}"
            ) from exc
        contract = checkpoint["contract"]
        tick = int(checkpoint["checkpoint_at_tick"])
        checkpoint_identity = str(
            checkpoint["checkpoint_identity"]["checkpoint_identity_sha256"]
        )
        if (
            checkpoint["attempt_authority_binding"] != parent
            or contract != parent["contract"]
            or checkpoint["artifact"]["sha256"] != parent["artifact_sha256"]
            or checkpoint["artifact"]["sealed_artifact_identity_sha256"]
            != parent["sealed_artifact_identity_sha256"]
            or tick <= int(contract["start_next_tick"])
            or tick > int(contract["start_next_tick"]) + int(contract["total_ticks"])
        ):
            raise CudaFullEvolutionV2LaunchpadAdapterError(
                "Workhorse checkpoint differs from its canonical parent"
            )
        bundle = absolute_bundle_root.expanduser().resolve()
        occurrence = occurrence_root.expanduser().resolve()
        if bundle.exists() or bundle.is_symlink():
            raise CudaFullEvolutionV2LaunchpadAdapterError(
                "fresh absolute-tick checkpoint projection already exists"
            )
        bundle.mkdir(parents=True)
        state_shape = [
            int(contract["B"]),
            int(checkpoint["canonical_state"]["state_bytes_per_world"]),
        ]
        state_sha256 = str(checkpoint["canonical_state"]["sha256"])
        state_nbytes = int(checkpoint["canonical_state"]["bytes"])
        runtime_body = {
            "schema_version": ABSOLUTE_TICK_CHECKPOINT_FORMAT_ID,
            "time_authority": ABSOLUTE_TICK_CHECKPOINT_TIME_AUTHORITY,
            "contract": dict(contract),
            "canonical_state_shape": state_shape,
            "canonical_state_nbytes": state_nbytes,
            "canonical_state_sha256": state_sha256,
            "canonical_state_layout_sha256": canonical_sha256(
                {
                    "dtype": "uint8",
                    "shape": state_shape,
                    "nbytes": state_nbytes,
                    "content_sha256": state_sha256,
                }
            ).hex,
            "initial_next_absolute_tick": int(contract["start_next_tick"]),
            "advanced_ticks": tick - int(contract["start_next_tick"]),
            "acknowledged_event_next_tick": tick,
            "last_sink_receipt_sha256": checkpoint["event_sink_ack"][
                "receipt_body_sha256"
            ],
            "world_uids": list(checkpoint["world_uids"]),
            "rng_keys": list(checkpoint["rng_keys"]),
            "rng_semantic_epoch": int(checkpoint["rng_epoch"]),
            "artifact_sha256": checkpoint["artifact"]["sha256"],
            "manifest_sha256": checkpoint["artifact"]["manifest_sha256"],
            "device_identity": dict(parent["device_identity"]),
            "population_fixture_id": None,
        }
        runtime_checkpoint = {
            **runtime_body,
            "metadata_seal_sha256": canonical_sha256(runtime_body).hex,
        }
        checkpoint_id = f"{checkpoint_id_prefix}-{tick:012d}-{checkpoint_identity[:16]}"
        metadata = {
            "schema_version": ABSOLUTE_TICK_CHECKPOINT_METADATA_SCHEMA,
            "checkpoint_format_id": ABSOLUTE_TICK_CHECKPOINT_FORMAT_ID,
            "time_authority": ABSOLUTE_TICK_CHECKPOINT_TIME_AUTHORITY,
            "checkpoint_id": checkpoint_id,
            "absolute_next_tick": tick,
            "artifact_path": checkpoint["artifact"]["path"],
            "runtime_checkpoint": runtime_checkpoint,
        }
        metadata_bytes = canonical_json_bytes(metadata)
        completion = {
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
        _copy_immutable_file_v1(
            state_path,
            bundle / ABSOLUTE_TICK_CHECKPOINT_STATE_NAME,
        )
        _write_immutable_bytes(
            bundle / ABSOLUTE_TICK_CHECKPOINT_METADATA_NAME, metadata_bytes
        )
        _write_immutable_json(
            bundle / ABSOLUTE_TICK_CHECKPOINT_COMPLETION_NAME, completion
        )
        bundle.chmod(0o555)
        _fsync_directory(bundle.parent)
        owner_id = (
            "engine.campaigns.cuda_full_evolution_v2_workhorse_provider."
            + owner_component
        )
        owner_digest = str(parent["operation_contract_sha256"])
        sealed = self.seal_bundle_v1(
            bundle,
            checkpoint_id=checkpoint_id,
            absolute_next_tick=tick,
            owner_id=owner_id,
            owner_digest_sha256=owner_digest,
        )
        admitted = self.materialize_occurrence_v1(
            bundle,
            occurrence,
            attempt_authority_path=Path(str(parent["attempt_authority_path"])),
            checkpoint_id=checkpoint_id,
            absolute_next_tick=tick,
            owner_id=owner_id,
            owner_digest_sha256=owner_digest,
            source_batch_id=str(parent["source_batch_id"]),
            parent_checkpoint_digest=parent_checkpoint_digest,
        )
        restored_state: bytes | None = None
        if restore_state:
            restored = self.restore_occurrence_v1(
                bundle,
                occurrence,
                attempt_authority_path=Path(str(parent["attempt_authority_path"])),
                source_batch_id=str(parent["source_batch_id"]),
                expected_absolute_next_tick=tick,
                expected_occurrence_digest=admitted["occurrence_digest"],
            )
            restored_state = restored.get("canonical_state")
            if (
                not isinstance(restored_state, bytes)
                or len(restored_state) != state_nbytes
                or hashlib.sha256(restored_state).hexdigest() != state_sha256
            ):
                raise CudaFullEvolutionV2LaunchpadAdapterError(
                    "CRCA occurrence restored different Workhorse checkpoint bytes"
                )
        descriptor_path = occurrence / ABSOLUTE_TICK_OCCURRENCE_DESCRIPTOR_NAME
        commit_path = occurrence / ABSOLUTE_TICK_OCCURRENCE_COMMIT_NAME
        core = {
            "schema_version": projection_schema,
            "claim_class": "TECHNICAL_ONLY",
            "science_authority_granted": False,
            "qualification_authority_granted": False,
            "time_authority": ABSOLUTE_TICK_CHECKPOINT_TIME_AUTHORITY,
            "checkpoint_at_tick": tick,
            "checkpoint_manifest_path": str(checkpoint_path),
            "checkpoint_manifest_file_sha256": _sha256_file(checkpoint_path),
            "checkpoint_bundle_body_sha256": checkpoint["bundle_body_sha256"],
            "checkpoint_identity_sha256": checkpoint_identity,
            "canonical_state_sha256": state_sha256,
            "canonical_parent_binding_sha256": parent["binding_sha256"],
            "operation_contract_sha256": parent["operation_contract_sha256"],
            "absolute_tick_bundle_path": str(bundle),
            "absolute_tick_bundle_digest": sealed["checkpoint_bundle_digest"],
            "occurrence_root": str(occurrence),
            "occurrence_digest": admitted["occurrence_digest"],
            "occurrence_descriptor_path": str(descriptor_path),
            "occurrence_descriptor_file_sha256": _sha256_file(descriptor_path),
            "occurrence_commit_path": str(commit_path),
            "occurrence_commit_file_sha256": _sha256_file(commit_path),
            "parent_checkpoint_digest": (
                None
                if parent_checkpoint_digest is None
                else dict(parent_checkpoint_digest)
            ),
            "canonical_visible": True,
            "restore_admitted_exact": restore_state,
        }
        projection = {
            **core,
            "projection_content_sha256": canonical_sha256(core).hex,
        }
        return {
            "canonical_state": restored_state,
            "canonical_state_nbytes": state_nbytes,
            "canonical_state_sha256": state_sha256,
            "projection": projection,
        }

    def materialize_workhorse_science_checkpoint_occurrence_v1(
        self,
        *,
        canonical_parent_authority_binding: Mapping[str, Any],
        checkpoint_manifest_path: Path,
        absolute_bundle_root: Path,
        occurrence_root: Path,
        parent_checkpoint_digest: Mapping[str, str] | None,
    ) -> Mapping[str, Any]:
        """Admit one science checkpoint and return bytes for fresh-owner restore."""

        try:
            return self._materialize_workhorse_checkpoint_occurrence_v2(
                canonical_parent_authority_binding=canonical_parent_authority_binding,
                checkpoint_manifest_path=checkpoint_manifest_path,
                absolute_bundle_root=absolute_bundle_root,
                occurrence_root=occurrence_root,
                parent_checkpoint_digest=parent_checkpoint_digest,
                expected_operation="SCIENCE_COMPATIBILITY_DEMO",
                checkpoint_id_prefix="workhorse-science-checkpoint-tick",
                owner_component="science_compatibility_demo",
                projection_schema=(
                    "eiel_workhorse_science_checkpoint_occurrence_projection_v1"
                ),
                restore_state=True,
            )
        except BaseException as exc:
            _quarantine_checkpoint_publications_v1(
                (occurrence_root, absolute_bundle_root), exc
            )
            raise

    def materialize_workhorse_row_checkpoint_occurrence_v2(
        self,
        *,
        canonical_parent_authority_binding: Mapping[str, Any],
        checkpoint_manifest_path: Path,
        absolute_bundle_root: Path,
        occurrence_root: Path,
        parent_checkpoint_digest: Mapping[str, str] | None = None,
        expected_operation: str = "EXECUTION_ROW",
    ) -> Mapping[str, Any]:
        """Commit one row/continuation checkpoint without loading full B."""

        if expected_operation not in {
            "EXECUTION_ROW",
            "INDEXED_ACS_CONTINUATION",
        }:
            raise CudaFullEvolutionV2LaunchpadAdapterError(
                "row checkpoint occurrence operation is unsupported"
            )

        try:
            return self._materialize_workhorse_checkpoint_occurrence_v2(
                canonical_parent_authority_binding=canonical_parent_authority_binding,
                checkpoint_manifest_path=checkpoint_manifest_path,
                absolute_bundle_root=absolute_bundle_root,
                occurrence_root=occurrence_root,
                parent_checkpoint_digest=parent_checkpoint_digest,
                expected_operation=expected_operation,
                checkpoint_id_prefix=(
                    "workhorse-indexed-continuation-checkpoint-tick"
                    if expected_operation == "INDEXED_ACS_CONTINUATION"
                    else "workhorse-row-checkpoint-tick"
                ),
                owner_component=(
                    "indexed_acs_continuation"
                    if expected_operation == "INDEXED_ACS_CONTINUATION"
                    else "execution_row"
                ),
                projection_schema=(
                    "eiel_workhorse_indexed_continuation_checkpoint_occurrence_projection_v2"
                    if expected_operation == "INDEXED_ACS_CONTINUATION"
                    else "eiel_workhorse_row_checkpoint_occurrence_projection_v2"
                ),
                restore_state=False,
            )
        except BaseException as exc:
            _quarantine_checkpoint_publications_v1(
                (occurrence_root, absolute_bundle_root), exc
            )
            raise

    def restore_workhorse_science_checkpoint_occurrence_v1(
        self,
        *,
        canonical_parent_authority_binding: Mapping[str, Any],
        checkpoint_manifest_path: Path,
        projection: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """Reopen an exact science occurrence and return its admitted state digest."""

        try:
            from engine.perf.cuda_full_evolution_v2.workhorse.authority import (
                reopen_workhorse_checkpoint_bundle_v2,
            )
            from engine.perf.cuda_full_evolution_v2.workhorse.canonical_rows import (
                validate_workhorse_canonical_parent_authority_binding_v2,
            )

            parent = validate_workhorse_canonical_parent_authority_binding_v2(
                canonical_parent_authority_binding,
                expected_operation="SCIENCE_COMPATIBILITY_DEMO",
            )
            checkpoint_path = checkpoint_manifest_path.expanduser().resolve()
            checkpoint, source_state = reopen_workhorse_checkpoint_bundle_v2(
                checkpoint_path
            )
        except Exception as exc:
            raise CudaFullEvolutionV2LaunchpadAdapterError(
                f"Workhorse science checkpoint cannot be reopened: {exc}"
            ) from exc
        expected_fields = {
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
        row = dict(projection)
        core = {
            key: value
            for key, value in row.items()
            if key != "projection_content_sha256"
        }
        tick = int(checkpoint["checkpoint_at_tick"])
        bundle = Path(str(row.get("absolute_tick_bundle_path") or "")).resolve()
        occurrence = Path(str(row.get("occurrence_root") or "")).resolve()
        descriptor_path = occurrence / ABSOLUTE_TICK_OCCURRENCE_DESCRIPTOR_NAME
        commit_path = occurrence / ABSOLUTE_TICK_OCCURRENCE_COMMIT_NAME
        if (
            set(row) != expected_fields
            or row.get("schema_version")
            != "eiel_workhorse_science_checkpoint_occurrence_projection_v1"
            or row.get("claim_class") != "TECHNICAL_ONLY"
            or row.get("science_authority_granted") is not False
            or row.get("qualification_authority_granted") is not False
            or row.get("time_authority") != ABSOLUTE_TICK_CHECKPOINT_TIME_AUTHORITY
            or row.get("checkpoint_at_tick") != tick
            or row.get("checkpoint_manifest_path") != str(checkpoint_path)
            or row.get("checkpoint_manifest_file_sha256")
            != _sha256_file(checkpoint_path)
            or row.get("checkpoint_bundle_body_sha256")
            != checkpoint["bundle_body_sha256"]
            or row.get("checkpoint_identity_sha256")
            != checkpoint["checkpoint_identity"]["checkpoint_identity_sha256"]
            or row.get("canonical_state_sha256")
            != hashlib.sha256(source_state).hexdigest()
            or row.get("canonical_parent_binding_sha256") != parent["binding_sha256"]
            or row.get("operation_contract_sha256")
            != parent["operation_contract_sha256"]
            or row.get("absolute_tick_bundle_digest")
            != checkpoint_absolute_tick_bundle_tree_digest_v1(bundle).to_dict()
            or row.get("occurrence_descriptor_path") != str(descriptor_path)
            or row.get("occurrence_descriptor_file_sha256")
            != _sha256_file(descriptor_path)
            or row.get("occurrence_commit_path") != str(commit_path)
            or row.get("occurrence_commit_file_sha256") != _sha256_file(commit_path)
            or row.get("canonical_visible") is not True
            or row.get("restore_admitted_exact") is not True
            or row.get("projection_content_sha256") != canonical_sha256(core).hex
            or checkpoint["attempt_authority_binding"] != parent
        ):
            raise CudaFullEvolutionV2LaunchpadAdapterError(
                "Workhorse science occurrence projection identity changed"
            )
        restored = self.restore_occurrence_v1(
            bundle,
            occurrence,
            attempt_authority_path=Path(str(parent["attempt_authority_path"])),
            source_batch_id=str(parent["source_batch_id"]),
            expected_absolute_next_tick=tick,
            expected_occurrence_digest=row["occurrence_digest"],
        )
        restored_state = restored.get("canonical_state")
        metadata = restored.get("metadata")
        runtime = (
            metadata.get("runtime_checkpoint")
            if isinstance(metadata, Mapping)
            else None
        )
        if (
            not isinstance(restored_state, bytes)
            or restored_state != source_state
            or not isinstance(runtime, Mapping)
            or runtime.get("canonical_state_sha256")
            != checkpoint["canonical_state"]["sha256"]
            or runtime.get("world_uids") != tuple(checkpoint["world_uids"])
            or runtime.get("rng_keys") != tuple(checkpoint["rng_keys"])
            or runtime.get("artifact_sha256") != checkpoint["artifact"]["sha256"]
            or runtime.get("manifest_sha256")
            != checkpoint["artifact"]["manifest_sha256"]
            or runtime.get("device_identity") != parent["device_identity"]
        ):
            raise CudaFullEvolutionV2LaunchpadAdapterError(
                "Workhorse science occurrence restore identity changed"
            )
        return {
            "canonical_state_sha256": hashlib.sha256(restored_state).hexdigest(),
            "checkpoint_at_tick": tick,
            "occurrence_digest": dict(row["occurrence_digest"]),
        }

    def _semantic_provider_terminal_projection_v2(
        self,
        *,
        authority: AuthoritativeAttemptBundleV1,
        campaign_plan: Mapping[str, Any],
        execution_receipt: Mapping[str, Any] | None,
    ) -> dict[str, Any] | None:
        operation = _workhorse_semantic_operation_v2(campaign_plan)
        if operation is None:
            return None
        if (
            not isinstance(execution_receipt, Mapping)
            or execution_receipt.get("status") != "completed"
            or execution_receipt.get("plan_sha256") != campaign_plan.get("plan_sha256")
        ):
            raise CudaFullEvolutionV2LaunchpadAdapterError(
                "semantic Attempt lacks its completed administrative receipt"
            )
        batches = campaign_plan.get("physical_batches")
        claim = dict(authority.raw["claim"])
        if (
            not isinstance(batches, list)
            or len(batches) != 1
            or not isinstance(batches[0], Mapping)
            or not isinstance(batches[0].get("batch_id"), str)
        ):
            raise CudaFullEvolutionV2LaunchpadAdapterError(
                "semantic Attempt is not exactly one physical operation"
            )
        manifest_path = (
            Path(str(claim.get("run_root") or "")).resolve()
            / "batches"
            / str(batches[0]["batch_id"])
            / "manifest.json"
        )
        manifest = _read_exact_json_file(
            manifest_path, label="Workhorse semantic batch manifest"
        )
        provider_result = manifest.get("result")
        expected_fields = {
            "schema_version",
            "operation",
            "case_id",
            "shape_id",
            "backend_id",
            "B",
            "temporal_k",
            "semantic_temporal_k",
            "artifact_sha256",
            "sealed_artifact_identity_sha256",
            "expected_attempt_terminal_status",
            "attempt_terminal_contract",
            "provisional_result_class",
            "provider_terminalization_action",
            "case_fragment_path",
            "case_fragment_file_sha256",
            "case_fragment_body_sha256",
            "source_checkpoint_identity_sha256",
            "external_source_checkpoint_binding",
            "science_checkpoint_compatibility_binding",
            "produced_external_source_checkpoint_binding",
            "checkpoint_occurrence_artifact_paths",
            "absolute_frontiers",
        }
        if (
            not isinstance(provider_result, dict)
            or set(provider_result) != expected_fields
        ):
            raise CudaFullEvolutionV2LaunchpadAdapterError(
                "semantic provider result fields are inexact"
            )
        semantic_case = operation["semantic_case_contract"]
        law = semantic_case["case_law"]
        terminal_contract = semantic_case["attempt_terminal_contract"]
        expected_status = terminal_contract["attempt_terminal_status"]
        expected_class = (
            "SUCCESS_PENDING_COMPLETED_TERMINAL"
            if expected_status == "COMPLETED"
            else "EXPECTED_NEGATIVE_PENDING_FAILED_TERMINAL"
        )
        expected_action = (
            "RETURN_FOR_COMPLETED_TERMINALIZATION"
            if expected_status == "COMPLETED"
            else "RETURN_FOR_FAILED_ATTEMPT_TERMINALIZATION"
        )
        contract = operation["contract"]
        target = contract["start_next_tick"] + contract["total_ticks"]
        expected_administrative_frontiers = {
            "start_next_tick": contract["start_next_tick"],
            "target_next_tick": target,
            "committed_next_tick": terminal_contract["final_frontiers"][
                "published_artifact_next_tick"
            ],
        }
        fragment_path = Path(str(provider_result["case_fragment_path"])).resolve()
        try:
            from engine.perf.cuda_full_evolution_v2.workhorse.semantic_cases import (
                reopen_workhorse_semantic_case_fragment_v2,
            )

            fragment, _datasets = reopen_workhorse_semantic_case_fragment_v2(
                fragment_path
            )
        except Exception as exc:
            raise CudaFullEvolutionV2LaunchpadAdapterError(
                f"semantic raw fragment cannot be reopened: {exc}"
            ) from exc
        parent = fragment.get("canonical_parent_binding")
        if (
            manifest.get("status") != "completed"
            or provider_result["schema_version"]
            != "eiel_workhorse_semantic_case_provider_result_v2"
            or provider_result["operation"] != "SEMANTIC_EVIDENCE_CASE"
            or provider_result["case_id"] != law["case_id"]
            or provider_result["shape_id"] != contract["shape_id"]
            or provider_result["backend_id"] != contract["backend_id"]
            or provider_result["B"] != contract["B"]
            or provider_result["temporal_k"] != contract["temporal_k"]
            or provider_result["semantic_temporal_k"]
            != semantic_case["semantic_temporal_k"]
            or provider_result["artifact_sha256"] != operation["artifact"]["sha256"]
            or provider_result["sealed_artifact_identity_sha256"]
            != operation["artifact"]["sealed_artifact_identity_sha256"]
            or provider_result["expected_attempt_terminal_status"] != expected_status
            or provider_result["attempt_terminal_contract"] != terminal_contract
            or provider_result["provisional_result_class"] != expected_class
            or provider_result["provider_terminalization_action"] != expected_action
            or provider_result["case_fragment_file_sha256"]
            != _sha256_file(fragment_path)
            or provider_result["case_fragment_body_sha256"]
            != fragment.get("fragment_body_sha256")
            or provider_result["source_checkpoint_identity_sha256"]
            != operation["source_checkpoint_identity_sha256"]
            or provider_result["external_source_checkpoint_binding"]
            != operation["external_source_checkpoint_binding"]
            or provider_result["science_checkpoint_compatibility_binding"]
            != operation["science_checkpoint_compatibility_binding"]
            or provider_result["absolute_frontiers"]
            != expected_administrative_frontiers
            or not isinstance(parent, Mapping)
            or parent.get("attempt_id") != str(authority.attempt_start.attempt_id)
            or parent.get("run_id") != str(authority.run_occurrence.run_id)
            or parent.get("operation_contract_sha256")
            != semantic_case["operation_contract_sha256"]
        ):
            raise CudaFullEvolutionV2LaunchpadAdapterError(
                "semantic provisional result differs from its exact plan/Attempt"
            )
        produced = provider_result["produced_external_source_checkpoint_binding"]
        occurrence_paths = provider_result["checkpoint_occurrence_artifact_paths"]
        if law["produces_base_checkpoint"] is True:
            base = fragment.get("base_checkpoint")
            identity = base.get("identity") if isinstance(base, Mapping) else None
            if not isinstance(produced, Mapping) or not isinstance(identity, Mapping):
                raise CudaFullEvolutionV2LaunchpadAdapterError(
                    "case 20 lacks its canonical produced checkpoint binding"
                )
            validated_produced = self.validate_workhorse_external_checkpoint_source_v2(
                produced,
                expected_identity_sha256=str(
                    identity["base_checkpoint_identity_sha256"]
                ),
            )
            if dict(produced) != dict(validated_produced):
                raise CudaFullEvolutionV2LaunchpadAdapterError(
                    "case 20 produced checkpoint binding changed on reopen"
                )
        elif produced is not None:
            raise CudaFullEvolutionV2LaunchpadAdapterError(
                "non-checkpoint semantic case produced checkpoint authority"
            )
        if (
            not isinstance(occurrence_paths, list)
            or (law["produces_base_checkpoint"] is True) != bool(occurrence_paths)
            or len(occurrence_paths) != len(set(occurrence_paths))
            or any(
                not Path(path).is_absolute()
                or Path(path).is_symlink()
                or not Path(path).is_file()
                for path in occurrence_paths
            )
        ):
            raise CudaFullEvolutionV2LaunchpadAdapterError(
                "semantic checkpoint occurrence artifact inventory is invalid"
            )
        return {
            "attempt_terminal_contract": dict(terminal_contract),
            "provider_result": dict(provider_result),
            "manifest_path": str(manifest_path),
        }

    def terminalize_attempt_authority_v1(
        self,
        *,
        attempt_authority_path: Path,
        campaign_plan: Mapping[str, Any],
        execution_status: str,
        execution_receipt: Mapping[str, Any] | None,
        result_binding: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        authority = self._authority(attempt_authority_path)
        self._validate_terminal_cuda_memory_release_v1(
            authority=authority,
            attempt_authority_path=attempt_authority_path,
            campaign_plan=campaign_plan,
            execution_receipt=execution_receipt,
        )
        semantic = authority.semantic_run_spec
        start_tick = semantic.absolute_start_next_tick
        target_tick = semantic.absolute_target_next_tick
        if execution_status not in {"completed", "failed"}:
            raise CudaFullEvolutionV2LaunchpadAdapterError(
                "Attempt terminalization requires completed or failed execution"
            )
        semantic_operation = _workhorse_semantic_operation_v2(campaign_plan)
        semantic_projection = (
            self._semantic_provider_terminal_projection_v2(
                authority=authority,
                campaign_plan=campaign_plan,
                execution_receipt=execution_receipt,
            )
            if semantic_operation is not None and execution_status == "completed"
            else None
        )
        if semantic_projection is not None:
            planned_terminal = semantic_projection["attempt_terminal_contract"]
            completed = planned_terminal["attempt_terminal_status"] == "COMPLETED"
            frontiers: Mapping[str, Any] | None = dict(
                planned_terminal["final_frontiers"]
            )
        else:
            completed = execution_status == "completed"
            frontiers = None
        if semantic_projection is not None:
            pass
        elif completed:
            if (
                not isinstance(execution_receipt, Mapping)
                or execution_receipt.get("status") != "completed"
                or execution_receipt.get("plan_sha256")
                != campaign_plan.get("plan_sha256")
            ):
                raise CudaFullEvolutionV2LaunchpadAdapterError(
                    "completed Attempt lacks its exact campaign execution receipt"
                )
            claim = dict(authority.raw["claim"])
            if claim.get("provider_id") == (
                "eiel-full-evolution-cuda-workhorse-v2-provider-v1"
            ):
                batches = campaign_plan.get("physical_batches")
                run_root = Path(str(claim.get("run_root") or "")).resolve()
                if (
                    not isinstance(batches, list)
                    or len(batches) != 1
                    or not isinstance(batches[0], Mapping)
                    or not isinstance(batches[0].get("batch_id"), str)
                ):
                    raise CudaFullEvolutionV2LaunchpadAdapterError(
                        "completed Workhorse Attempt is not exactly one operation"
                    )
                manifest_path = (
                    run_root / "batches" / str(batches[0]["batch_id"]) / "manifest.json"
                )
                try:
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                    raise CudaFullEvolutionV2LaunchpadAdapterError(
                        "completed Workhorse Attempt lacks its batch manifest"
                    ) from exc
                provider_result = (
                    manifest.get("result") if isinstance(manifest, Mapping) else None
                )
                row_frontiers = (
                    provider_result.get("absolute_frontiers")
                    if isinstance(provider_result, Mapping)
                    else None
                )
                checkpoint = (
                    manifest.get("checkpoint")
                    if isinstance(manifest, Mapping)
                    else None
                )
                analysis = campaign_plan.get("analysis")
                extension = (
                    analysis.get("extension") if isinstance(analysis, Mapping) else None
                )
                planned_operation = (
                    extension.get("contract")
                    if isinstance(extension, Mapping)
                    and isinstance(extension.get("contract"), Mapping)
                    else None
                )
                operation_name = (
                    planned_operation.get("operation")
                    if isinstance(planned_operation, Mapping)
                    else None
                )
                common_invalid = (
                    manifest.get("status") != "completed"
                    or not isinstance(provider_result, Mapping)
                    or provider_result.get("operation") != operation_name
                    or not isinstance(row_frontiers, Mapping)
                    or row_frontiers.get("start_next_tick") != start_tick
                    or row_frontiers.get("target_next_tick") != target_tick
                    or row_frontiers.get("committed_next_tick") != target_tick
                    or not isinstance(checkpoint, Mapping)
                    or checkpoint.get("next_absolute_tick") != target_tick
                )
                if operation_name == "EXECUTION_ROW":
                    operation_invalid = (
                        provider_result.get("schema_version")
                        != "eiel_workhorse_single_row_provider_result_v2"
                    )
                elif operation_name == "INDEXED_ACS_CONTINUATION":
                    receipt_path = Path(
                        str(provider_result.get("execution_result_path") or "")
                    ).resolve()
                    continuation = planned_operation.get("continuation_contract", {})
                    operation_invalid = (
                        provider_result.get("schema_version")
                        != "eiel_workhorse_indexed_continuation_provider_result_v2"
                        or provider_result.get("shape_id")
                        != planned_operation.get("contract", {}).get("shape_id")
                        or provider_result.get("backend_id")
                        != planned_operation.get("contract", {}).get("backend_id")
                        or provider_result.get("B")
                        != planned_operation.get("contract", {}).get("B")
                        or provider_result.get("preferred_temporal_k")
                        != planned_operation.get("contract", {}).get("temporal_k")
                        or provider_result.get("source_world_state_set_sha256")
                        != planned_operation.get(
                            "source_checkpoint_identity_sha256"
                        )
                        or provider_result.get(
                            "science_continuation_identity_sha256"
                        )
                        != continuation.get("science_continuation_identity", {}).get(
                            "science_continuation_identity_sha256"
                        )
                        or provider_result.get("physical_continuation_plan_sha256")
                        != continuation.get("physical_continuation_plan", {}).get(
                            "physical_continuation_plan_sha256"
                        )
                        or provider_result.get("checkpoint_ticks")
                        != continuation.get("checkpoint_ticks")
                        or receipt_path
                        != manifest_path.parent
                        / "workhorse-indexed-continuation-result.json"
                        or not receipt_path.is_file()
                        or receipt_path.is_symlink()
                        or provider_result.get("execution_result_file_sha256")
                        != _sha256_file(receipt_path)
                    )
                elif operation_name == "SCIENCE_COMPATIBILITY_DEMO":
                    receipt_path = Path(
                        str(provider_result.get("science_demo_result_path") or "")
                    ).resolve()
                    operation_invalid = (
                        provider_result.get("schema_version")
                        != "eiel_workhorse_science_demo_provider_result_v1"
                        or provider_result.get("shape_id")
                        != planned_operation.get("contract", {}).get("shape_id")
                        or provider_result.get("backend_id")
                        != planned_operation.get("contract", {}).get("backend_id")
                        or provider_result.get("B") != 10
                        or provider_result.get("temporal_k") != 1
                        or provider_result.get("boundary_next_ticks")
                        != [0, 8, 16, 24, 32, 40]
                        or provider_result.get("checkpoint_ticks") != [16, 32, 40]
                        or provider_result.get("continuation_restore_ticks") != [16, 32]
                        or provider_result.get("terminal_restore_validation_tick") != 40
                        or provider_result.get("result_class")
                        != "TECHNICAL_COMPATIBILITY_DEMO_PASSED"
                        or receipt_path
                        != manifest_path.parent / "workhorse-science-demo-result.json"
                        or not receipt_path.is_file()
                        or receipt_path.is_symlink()
                        or provider_result.get("science_demo_result_file_sha256")
                        != _sha256_file(receipt_path)
                    )
                elif operation_name == "DEVICE_ENVELOPE_CALIBRATION":
                    receipt_path = Path(
                        str(provider_result.get("calibration_receipt_path") or "")
                    ).resolve()
                    operation_invalid = (
                        provider_result.get("schema_version")
                        != "eiel_workhorse_device_envelope_calibration_provider_result_v2"
                        or provider_result.get("shape_id")
                        != planned_operation.get("shape_id")
                        or provider_result.get("backend_id")
                        != planned_operation.get("backend_id")
                        or provider_result.get("artifact_sha256")
                        != planned_operation.get("artifact", {}).get("sha256")
                        or provider_result.get("sealed_artifact_identity_sha256")
                        != planned_operation.get("artifact", {}).get(
                            "sealed_artifact_identity_sha256"
                        )
                        or provider_result.get("compiled_temporal_k")
                        != planned_operation.get("compiled_temporal_k")
                        or provider_result.get("device_identity")
                        != planned_operation.get("device_identity")
                        or provider_result.get("reserve_policy")
                        != planned_operation.get("reserve_policy")
                        or receipt_path
                        != manifest_path.parent
                        / "workhorse-device-envelope-calibration-measurement.json"
                        or not receipt_path.is_file()
                        or receipt_path.is_symlink()
                        or provider_result.get("calibration_receipt_file_sha256")
                        != _sha256_file(receipt_path)
                    )
                else:
                    operation_invalid = True
                if common_invalid or operation_invalid:
                    raise CudaFullEvolutionV2LaunchpadAdapterError(
                        "completed Workhorse operation lacks exact terminal bindings"
                    )
        else:
            frontiers = (
                execution_receipt.get("crca_final_frontiers")
                if isinstance(execution_receipt, Mapping)
                else None
            )
            if (
                not isinstance(frontiers, Mapping)
                or set(frontiers)
                != {
                    "last_dispatched_next_tick",
                    "last_computed_next_tick",
                    "durable_checkpoint_next_tick",
                    "published_artifact_next_tick",
                }
                or any(type(value) is not int for value in frontiers.values())
            ):
                raise CudaFullEvolutionV2LaunchpadAdapterError(
                    "failed Attempt lacks exact absolute-tick final frontiers"
                )
            frontier_values = [int(value) for value in frontiers.values()]
            if any(
                value < start_tick or value > target_tick for value in frontier_values
            ):
                raise CudaFullEvolutionV2LaunchpadAdapterError(
                    "failed Attempt final frontiers leave its absolute tick span"
                )
        terminal = AttemptTerminalV1(
            attempt_id=authority.attempt_start.attempt_id,
            attempt_start_digest=authority.attempt_start.start_digest(),
            status=(
                AttemptTerminalStatusV1.COMPLETED
                if completed
                else AttemptTerminalStatusV1.FAILED
            ),
            final_frontiers=(
                AttemptFrontiersV1(
                    int(frontiers["last_dispatched_next_tick"]),
                    int(frontiers["last_computed_next_tick"]),
                    int(frontiers["durable_checkpoint_next_tick"]),
                    int(frontiers["published_artifact_next_tick"]),
                )
                if frontiers is not None
                else AttemptFrontiersV1(
                    target_tick, target_tick, target_tick, target_tick
                )
            ),
            result_digest=canonical_sha256(dict(execution_receipt or result_binding)),
            reason=(
                "Canonical Launchpad semantic operation completed."
                if semantic_projection is not None and completed
                else (
                    "Canonical Launchpad semantic expected-negative operation failed."
                    if semantic_projection is not None
                    else (
                        "Canonical Launchpad execution completed."
                        if completed
                        else "Canonical Launchpad execution failed before completion."
                    )
                )
            ),
        )
        claim = dict(authority.raw["claim"])
        attempt_id = str(authority.attempt_start.attempt_id)
        publication_digest = (
            scheduler_sha256(dict(execution_receipt))
            if completed and execution_receipt is not None
            else None
        )
        try:
            reopened_terminal = self.issuer.terminal_attempt_record(attempt_id)
        except ActorError as exc:
            if exc.code != "launchpad_attempt_terminal":
                raise
            reopened_terminal = None
        if reopened_terminal is None:
            current_process = _current_process_identity_v1()
            result = self.issuer.terminalize(
                attempt_id=attempt_id,
                process_owner_id=PROCESS_OWNER_ID_V1,
                process_owner_digest=_scheduler_digest(claim["execution_owner_digest"]),
                process_id=current_process["pid"],
                process_start_identity=current_process["process_start_identity"],
                terminal=terminal.to_dict(),
                at_utc=self.now_fn(),
                publication_digest=publication_digest,
            )
        else:
            terminal_bundle = reopened_terminal.get("terminal_bundle")
            continuation_event = (
                terminal_bundle.get("continuation_event")
                if isinstance(terminal_bundle, Mapping)
                else None
            )
            expected_publication = (
                publication_digest.to_dict() if publication_digest is not None else None
            )
            if (
                not isinstance(terminal_bundle, Mapping)
                or terminal_bundle.get("attempt_terminal") != terminal.to_dict()
                or not isinstance(continuation_event, Mapping)
                or continuation_event.get("publication_digest") != expected_publication
            ):
                raise CudaFullEvolutionV2LaunchpadAdapterError(
                    "committed Scheduler terminal differs from exact replay"
                )
            result = {
                "status": "TERMINAL",
                "idempotent_replay": True,
                "terminal_bundle": dict(terminal_bundle),
                "terminal_bundle_digest": reopened_terminal["terminal_bundle_digest"],
                "terminal_bundle_path": reopened_terminal["terminal_bundle_path"],
            }
        terminal_path = Path(str(result["terminal_bundle_path"])).resolve()
        try:
            terminal_bundle = json.loads(terminal_path.read_text(encoding="ascii"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise CudaFullEvolutionV2LaunchpadAdapterError(
                "Scheduler terminal bundle cannot be reopened"
            ) from exc
        terminal_digest = canonical_sha256(terminal_bundle).hex
        declared_terminal_digest = result.get("terminal_bundle_digest")
        if (
            terminal_bundle != result.get("terminal_bundle")
            or not isinstance(declared_terminal_digest, Mapping)
            or declared_terminal_digest.get("algorithm") != "sha256"
            or declared_terminal_digest.get("hex") != terminal_digest
            or terminal_bundle.get("attempt_id")
            != str(authority.attempt_start.attempt_id)
        ):
            raise CudaFullEvolutionV2LaunchpadAdapterError(
                "Scheduler terminal result differs after reopen"
            )
        response = {
            "status": result["status"],
            "terminal_bundle_path": str(terminal_path),
            "terminal_bundle_digest": declared_terminal_digest,
            "execution_status": execution_status,
            "campaign_plan_sha256": campaign_plan.get("plan_sha256"),
        }
        if claim.get("provider_id") == (
            "eiel-full-evolution-cuda-workhorse-v2-provider-v1"
        ):
            terminal_record = terminal_bundle.get("attempt_terminal")
            if not isinstance(terminal_record, Mapping):
                raise CudaFullEvolutionV2LaunchpadAdapterError(
                    "Workhorse Scheduler terminal record is absent"
                )
            terminal_core = {
                "schema_version": (
                    "eiel_workhorse_canonical_parent_terminal_binding_v2"
                ),
                "authority_source": "CANONICAL_PROVIDER_AFTER_SCHEDULER_REOPEN",
                "status": "TERMINAL",
                "attempt_terminal_status": terminal_record.get("status"),
                "run_id": str(authority.run_occurrence.run_id),
                "attempt_id": str(authority.attempt_start.attempt_id),
                "terminal_bundle_path": str(terminal_path),
                "terminal_bundle_file_sha256": _sha256_file(terminal_path),
                "terminal_bundle_digest": terminal_digest,
                "final_frontiers": dict(terminal_record.get("final_frontiers") or {}),
            }
            response["workhorse_terminal_binding"] = {
                **terminal_core,
                "binding_sha256": canonical_sha256(terminal_core).hex,
            }
        return response

    def materialize_terminal_cuda_memory_release_v1(
        self,
        *,
        attempt_authority_path: Path,
        campaign_plan: Mapping[str, Any],
        bound_execution_request: Mapping[str, Any],
        observer: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    ) -> Mapping[str, Any]:
        """Seal process-local zero-allocation evidence before releasing a lane."""

        authority = self._authority(attempt_authority_path)
        claim = dict(authority.raw["claim"])
        route = _route_bindings(bound_execution_request)
        policy = route.get("gpu_resource_policy")
        if not isinstance(policy, Mapping):
            raise CudaFullEvolutionV2LaunchpadAdapterError(
                "terminal CUDA memory release is valid only for shared GPU policy"
            )
        plan_identity = {
            key: value for key, value in campaign_plan.items() if key != "plan_sha256"
        }
        request_identity = {
            key: value
            for key, value in bound_execution_request.items()
            if key not in {"bound_request_sha256", "execution_route_contract"}
        }
        if (
            scheduler_sha256(plan_identity).to_dict()
            != claim.get("campaign_plan_digest")
            or scheduler_sha256(request_identity).to_dict()
            != claim.get("execution_request_digest")
            or route.get("device_id") != claim.get("device_id")
        ):
            raise CudaFullEvolutionV2LaunchpadAdapterError(
                "terminal CUDA release differs from its exact Attempt"
            )
        resource = authority.raw.get("resource_start_lease")
        resource_ids = (
            resource.get("resource_ids") if isinstance(resource, Mapping) else None
        )
        from tools.scheduler.programme_contracts import (
            shared_gpu_lane_candidate_sets_v1,
        )

        candidates = shared_gpu_lane_candidate_sets_v1(str(route["device_id"]))
        if not isinstance(resource_ids, list) or tuple(resource_ids) not in candidates:
            raise CudaFullEvolutionV2LaunchpadAdapterError(
                "terminal CUDA release lacks one canonical selected lane"
            )
        pointer = _read_exact_json_file(
            attempt_authority_path,
            label="Attempt authority pointer",
        )
        owner_pid = pointer.get("owner_pid")
        owner_start_ticks = pointer.get("owner_process_start_ticks")
        owner_start_identity = pointer.get("owner_process_start_identity")
        if (
            owner_pid != os.getpid()
            or owner_start_ticks != _current_process_start_ticks_v1()
            or owner_start_identity != process_start_identity(os.getpid())
            or pointer.get("execution_route_sha256")
            != bound_execution_request.get("execution_route_sha256")
            or pointer.get("device_identity") != dict(route["device_identity"])
            or pointer.get("selected_resource_ids") != list(resource_ids)
            or pointer.get("resource_scope_digest")
            != resource.get("resource_scope_digest")
        ):
            raise CudaFullEvolutionV2LaunchpadAdapterError(
                "terminal CUDA release is owned by another process"
            )
        observation = dict(observer(dict(route["device_identity"])))
        expected_observation_fields = {
            "schema_version",
            "device_index",
            "garbage_collection_completed",
            "cuda_synchronize_completed",
            "empty_cache_completed",
            "memory_allocated_bytes",
            "memory_reserved_bytes",
            "started_at_utc",
            "completed_at_utc",
        }
        if (
            set(observation) != expected_observation_fields
            or observation.get("schema_version")
            != "eiel_cuda_allocator_release_observation_v1"
            or observation.get("device_index") != route["device_identity"].get("index")
            or observation.get("garbage_collection_completed") is not True
            or observation.get("cuda_synchronize_completed") is not True
            or observation.get("empty_cache_completed") is not True
            or observation.get("memory_allocated_bytes") != 0
            or observation.get("memory_reserved_bytes") != 0
        ):
            raise CudaFullEvolutionV2LaunchpadAdapterError(
                "shared CUDA allocator was not fully released"
            )
        receipt_core = {
            "schema_version": "eiel_terminal_cuda_memory_release_receipt_v1",
            "attempt_id": str(authority.attempt_start.attempt_id),
            "plan_sha256": campaign_plan.get("plan_sha256"),
            "execution_route_sha256": bound_execution_request.get(
                "execution_route_sha256"
            ),
            "selected_resource_ids": list(resource_ids),
            "resource_scope_digest": dict(resource["resource_scope_digest"]),
            "device_id": route["device_id"],
            "device_identity": dict(route["device_identity"]),
            "owner_pid": owner_pid,
            "owner_process_start_ticks": owner_start_ticks,
            "owner_process_start_identity": owner_start_identity,
            "observation": observation,
            "released_at_utc": observation["completed_at_utc"],
        }
        receipt = {
            **receipt_core,
            "receipt_sha256": canonical_sha256(receipt_core).hex,
        }
        run_root = Path(str(claim["run_root"])).resolve()
        receipt_path = (
            run_root / "control" / "crca" / TERMINAL_CUDA_MEMORY_RELEASE_NAME_V1
        )
        if receipt_path.exists() or receipt_path.is_symlink():
            existing = _read_exact_json_file(
                receipt_path, label="terminal CUDA memory release receipt"
            )
            if (
                existing.get("attempt_id") != receipt["attempt_id"]
                or existing.get("plan_sha256") != receipt["plan_sha256"]
                or existing.get("execution_route_sha256")
                != receipt["execution_route_sha256"]
                or existing.get("selected_resource_ids")
                != receipt["selected_resource_ids"]
                or existing.get("resource_scope_digest")
                != receipt["resource_scope_digest"]
                or existing.get("device_identity") != receipt["device_identity"]
                or existing.get("owner_pid") != receipt["owner_pid"]
                or existing.get("owner_process_start_ticks")
                != receipt["owner_process_start_ticks"]
                or existing.get("owner_process_start_identity")
                != receipt["owner_process_start_identity"]
            ):
                raise CudaFullEvolutionV2LaunchpadAdapterError(
                    "terminal CUDA memory release replay differs"
                )
            receipt = existing
        else:
            _write_immutable_json(receipt_path, receipt)
        binding = {
            "schema_version": "eiel_terminal_cuda_memory_release_binding_v1",
            "path": str(receipt_path),
            "file_sha256": _sha256_file(receipt_path),
            "receipt_sha256": receipt["receipt_sha256"],
        }
        return {
            **binding,
            "binding_sha256": canonical_sha256(binding).hex,
        }

    def _validate_terminal_cuda_memory_release_v1(
        self,
        *,
        authority: AuthoritativeAttemptBundleV1,
        attempt_authority_path: Path,
        campaign_plan: Mapping[str, Any],
        execution_receipt: Mapping[str, Any] | None,
    ) -> None:
        execution = campaign_plan.get("execution")
        policy = (
            execution.get("gpu_resource_policy")
            if isinstance(execution, Mapping)
            else None
        )
        binding = (
            execution_receipt.get("cuda_memory_release")
            if isinstance(execution_receipt, Mapping)
            else None
        )
        if policy is None:
            if binding is not None:
                raise CudaFullEvolutionV2LaunchpadAdapterError(
                    "exclusive Attempt unexpectedly carries shared CUDA release"
                )
            return
        if not isinstance(binding, Mapping):
            raise CudaFullEvolutionV2LaunchpadAdapterError(
                "shared Attempt lacks terminal CUDA memory release evidence"
            )
        expected_binding_fields = {
            "schema_version",
            "path",
            "file_sha256",
            "receipt_sha256",
            "binding_sha256",
        }
        binding_core = {
            key: value for key, value in binding.items() if key != "binding_sha256"
        }
        claim = dict(authority.raw["claim"])
        expected_path = (
            Path(str(claim["run_root"])).resolve()
            / "control"
            / "crca"
            / TERMINAL_CUDA_MEMORY_RELEASE_NAME_V1
        )
        if (
            set(binding) != expected_binding_fields
            or binding.get("schema_version")
            != "eiel_terminal_cuda_memory_release_binding_v1"
            or Path(str(binding.get("path") or "")).resolve() != expected_path
            or binding.get("binding_sha256") != canonical_sha256(binding_core).hex
        ):
            raise CudaFullEvolutionV2LaunchpadAdapterError(
                "terminal CUDA memory release binding is invalid"
            )
        receipt = _read_exact_json_file(
            expected_path, label="terminal CUDA memory release receipt"
        )
        receipt_core = {
            key: value for key, value in receipt.items() if key != "receipt_sha256"
        }
        resource = authority.raw.get("resource_start_lease")
        pointer = _read_exact_json_file(
            attempt_authority_path, label="Attempt authority pointer"
        )
        observation = receipt.get("observation")
        if (
            binding.get("file_sha256") != _sha256_file(expected_path)
            or binding.get("receipt_sha256") != receipt.get("receipt_sha256")
            or receipt.get("receipt_sha256") != canonical_sha256(receipt_core).hex
            or receipt.get("attempt_id") != str(authority.attempt_start.attempt_id)
            or receipt.get("plan_sha256") != campaign_plan.get("plan_sha256")
            or receipt.get("execution_route_sha256")
            != pointer.get("execution_route_sha256")
            or not isinstance(resource, Mapping)
            or receipt.get("selected_resource_ids") != resource.get("resource_ids")
            or receipt.get("resource_scope_digest")
            != resource.get("resource_scope_digest")
            or receipt.get("device_id") != claim.get("device_id")
            or receipt.get("device_identity") != pointer.get("device_identity")
            or receipt.get("owner_pid") != pointer.get("owner_pid")
            or receipt.get("owner_process_start_ticks")
            != pointer.get("owner_process_start_ticks")
            or receipt.get("owner_process_start_identity")
            != pointer.get("owner_process_start_identity")
            or not isinstance(observation, Mapping)
            or observation.get("garbage_collection_completed") is not True
            or observation.get("cuda_synchronize_completed") is not True
            or observation.get("empty_cache_completed") is not True
            or observation.get("memory_allocated_bytes") != 0
            or observation.get("memory_reserved_bytes") != 0
        ):
            raise CudaFullEvolutionV2LaunchpadAdapterError(
                "terminal CUDA memory release receipt is not exact zero evidence"
            )

    def seal_bundle_v1(
        self,
        bundle: Path,
        *,
        checkpoint_id: str,
        absolute_next_tick: int,
        owner_id: str,
        owner_digest_sha256: str,
    ) -> Mapping[str, Any]:
        bindings = absolute_tick_checkpoint_bundle_file_participant_bindings_v1(
            bundle,
            owner_id=owner_id,
            owner_digest=ContentDigest("sha256", owner_digest_sha256),
        )
        completion = json.loads((bundle / "bundle_complete.json").read_text())
        if (
            len(bindings) != 3
            or completion.get("checkpoint_id") != checkpoint_id
            or completion.get("absolute_next_tick") != absolute_next_tick
        ):
            raise CudaFullEvolutionV2LaunchpadAdapterError(
                "checkpoint bundle differs from its absolute-tick seal"
            )
        return {
            "checkpoint_bundle_digest": checkpoint_absolute_tick_bundle_tree_digest_v1(
                bundle
            ).to_dict()
        }

    def materialize_occurrence_v1(
        self,
        bundle: Path,
        occurrence_root: Path,
        *,
        attempt_authority_path: Path,
        checkpoint_id: str,
        absolute_next_tick: int,
        owner_id: str,
        owner_digest_sha256: str,
        source_batch_id: str,
        parent_checkpoint_digest: Mapping[str, str] | None,
    ) -> Mapping[str, Any]:
        authority = self._authority(attempt_authority_path)
        bindings = absolute_tick_checkpoint_bundle_file_participant_bindings_v1(
            bundle,
            owner_id=owner_id,
            owner_digest=ContentDigest("sha256", owner_digest_sha256),
        )
        descriptor, _commit = materialize_absolute_tick_checkpoint_occurrence_v1(
            bundle,
            occurrence_root,
            checkpoint_id=CheckpointId(checkpoint_id),
            run_id=authority.run_occurrence.run_id,
            attempt_id=authority.attempt_start.attempt_id,
            transaction_id=(
                f"{authority.attempt_start.attempt_id}:"
                f"{source_batch_id}:"
                f"checkpoint-tick-{absolute_next_tick:012d}"
            ),
            absolute_next_tick=absolute_next_tick,
            semantic_spec_digest=authority.semantic_run_spec.semantic_spec_digest(),
            implementation_digest=(
                authority.execution_implementation.implementation_digest()
            ),
            participant_bindings=bindings,
            parent_checkpoint_digest=(
                None
                if parent_checkpoint_digest is None
                else _content_digest(parent_checkpoint_digest)
            ),
        )
        return {
            "checkpoint_bundle_digest": descriptor.checkpoint_bundle_digest.to_dict(),
            "occurrence_digest": descriptor.occurrence_digest().to_dict(),
        }

    def restore_occurrence_v1(
        self,
        bundle: Path,
        occurrence_root: Path,
        *,
        attempt_authority_path: Path,
        source_batch_id: str,
        expected_absolute_next_tick: int,
        expected_occurrence_digest: Mapping[str, str],
    ) -> Mapping[str, Any]:
        authority = self._authority(attempt_authority_path)
        return restore_absolute_tick_checkpoint_occurrence_v1(
            bundle,
            occurrence_root,
            expected_run_id=authority.run_occurrence.run_id,
            expected_attempt_id=authority.attempt_start.attempt_id,
            expected_absolute_next_tick=expected_absolute_next_tick,
            expected_semantic_spec_digest=(
                authority.semantic_run_spec.semantic_spec_digest()
            ),
            expected_implementation_digest=(
                authority.execution_implementation.implementation_digest()
            ),
            expected_occurrence_digest=_content_digest(expected_occurrence_digest),
        )

    def authorize_physical_batch_member_v1(
        self,
        *,
        attempt_authority_path: Path,
        source_batch_id: str,
        resident_request: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """Authorize one exact member of the Scheduler-bound physical batch set."""

        authority = self._authority(attempt_authority_path)
        authority.validate()
        execution_request = authority.raw.get("execution_request")
        if not isinstance(execution_request, Mapping):
            raise CudaFullEvolutionV2LaunchpadAdapterError(
                "Attempt authority lacks its bound execution request"
            )
        request_digest = scheduler_sha256(dict(resident_request)).hex
        physical_batch_set = execution_request.get("physical_batch_set")
        if physical_batch_set is None:
            if execution_request.get(
                "source_batch_id"
            ) != source_batch_id or execution_request.get("resident_request") != dict(
                resident_request
            ):
                raise CudaFullEvolutionV2LaunchpadAdapterError(
                    "physical batch differs from the exclusive Attempt"
                )
            authorization = {
                "schema_version": "eiel_cuda_full_evolution_v2_batch_authorization_v1",
                "authorization_scope": "exclusive_attempt_member",
                "attempt_id": str(authority.attempt_start.attempt_id),
                "source_batch_id": source_batch_id,
                "resident_request_sha256": request_digest,
            }
        else:
            if (
                not isinstance(physical_batch_set, Mapping)
                or physical_batch_set.get("schema_version")
                != "eiel_cuda_full_evolution_v2_physical_batch_set_v1"
            ):
                raise CudaFullEvolutionV2LaunchpadAdapterError(
                    "Attempt physical batch set is malformed"
                )
            bindings = physical_batch_set.get("ordered_resident_request_bindings")
            if not isinstance(bindings, list):
                raise CudaFullEvolutionV2LaunchpadAdapterError(
                    "Attempt physical batch set lacks ordered request bindings"
                )
            expected_digest = canonical_sha256(bindings).hex
            if physical_batch_set.get(
                "ordered_resident_request_bindings_sha256"
            ) != expected_digest or physical_batch_set.get("batch_ids") != [
                row.get("batch_id") for row in bindings if isinstance(row, Mapping)
            ]:
                raise CudaFullEvolutionV2LaunchpadAdapterError(
                    "Attempt physical batch set digest or order is invalid"
                )
            matches = [
                row
                for row in bindings
                if isinstance(row, Mapping) and row.get("batch_id") == source_batch_id
            ]
            if (
                len(matches) != 1
                or matches[0].get("resident_request_sha256") != request_digest
            ):
                raise CudaFullEvolutionV2LaunchpadAdapterError(
                    "physical batch is not an exact member of the Attempt"
                )
            authorization = {
                "schema_version": "eiel_cuda_full_evolution_v2_batch_authorization_v1",
                "authorization_scope": "ordered_physical_batch_set_member",
                "attempt_id": str(authority.attempt_start.attempt_id),
                "source_batch_id": source_batch_id,
                "resident_request_sha256": request_digest,
                "physical_batch_set_sha256": canonical_sha256(
                    dict(physical_batch_set)
                ).hex,
            }
        return {
            **authorization,
            "authorization_sha256": canonical_sha256(authorization).hex,
        }


__all__ = [
    "ATTEMPT_POINTER_SCHEMA_V1",
    "CanonicalCudaFullEvolutionV2LaunchpadAdapterV1",
    "CudaFullEvolutionV2LaunchpadAdapterError",
]
