"""Foreground, optional single-worker scheduler runtime for EIEL.

The runtime owns execution only.  Its local state is disposable process
bookkeeping; Work, Sessions, Assignments, Leases, and Worker Results remain in
the existing repository-facing actor stores.
"""

from __future__ import annotations

import contextlib
import json
import os
import signal
import shutil
import time
from pathlib import Path
from typing import Any, Mapping

from tools.actors.common import ActorError, advisory_lock, atomic_write_json, read_json, utc_now
from tools.scheduler.contracts import (
    add_seconds,
    assignment_store,
    lease_store,
    records,
    result_store,
    parse_utc,
    resource_store,
    scheduler_id_for,
    scheduler_root,
)
from tools.scheduler.dispatch import (
    acknowledge_worker_launch,
    bind_worker_thread_identity,
    claim_assignment,
    expire_leases,
    heartbeat_lease,
    mark_assignment_running,
    record_execution_loss,
    recover_preparing,
    request_assignment,
    submit_worker_result,
    worker_open,
)
from tools.scheduler.launcher import (
    CodexExecLauncher,
    FakeWorkerLauncher,
    LaunchResult,
    launch_contract_for_assignment,
    process_identity_is_live,
    process_group_is_live,
    terminate_process_group_id,
)
from tools.scheduler.result_settlement import (
    DEFAULT_GRACE_SECONDS as RESULT_SETTLEMENT_GRACE_SECONDS,
    DEFAULT_VALIDATION_ATTEMPTS as RESULT_SETTLEMENT_VALIDATION_ATTEMPTS,
    attempt_result_settlement as settle_worker_result,
    attempt_result_settlement_with_wait as settle_worker_result_with_wait,
)
from tools.scheduler.programme_contracts import (
    AttemptLeaseBindingsV1,
    ContentDigest,
    ProgrammeContractError,
    ResourceStartLeaseEventCodeV1,
    ResourceStartLeaseEventV1,
    ResourceStartLeaseV1,
    canonical_sha256 as contract_sha256,
)
from tools.scheduler.programme_driver import ResourceStartLeaseLedgerV1


RUNTIME_SCHEMA_VERSION = "eiel_worker_scheduler_runtime_v1"
RUNTIME_STATUSES = {"RUNNING", "PAUSED", "STOP_REQUESTED", "STOPPED"}


def _sha256_digest(value: str, *, field: str) -> ContentDigest:
    try:
        return ContentDigest(algorithm="sha256", hex=str(value))
    except Exception as error:
        raise ActorError(
            f"{field} is not exact SHA-256",
            code="scheduler_resource_admission_invalid",
        ) from error


def _grant_resource_start_lease(
    *,
    state_root: str | Path | None,
    assignment: Mapping[str, Any],
    assignment_lease: Mapping[str, Any],
    scheduler_id: str,
) -> ResourceStartLeaseV1 | None:
    """Convert an exact advisory binding into a Scheduler-owned start lease."""

    material = dict(dict(assignment.get("payload") or {}).get("material") or {})
    if not material.get("resource_admission_required", False):
        return None
    binding = dict(material.get("resource_admission") or {})
    resources = resource_store(state_root)
    recommendation = resources.get(str(binding.get("resource_recommendation_id") or ""))
    observation = resources.get(str(binding.get("resource_observation_id") or ""))
    if recommendation.get("record_sha256") != binding.get("resource_recommendation_record_sha256"):
        raise ActorError(
            "resource recommendation changed after Dispatch authorization",
            code="scheduler_resource_admission_stale",
        )
    if observation.get("record_sha256") != binding.get("resource_observation_record_sha256"):
        raise ActorError(
            "resource observation changed after Dispatch authorization",
            code="scheduler_resource_admission_stale",
        )
    recommendation_payload = dict(recommendation.get("payload") or {})
    observation_payload = dict(observation.get("payload") or {})
    if recommendation.get("status") not in {"ADMIT", "ADMIT_WITH_LIMITS"}:
        raise ActorError(
            "Resource Manager recommendation does not support a start lease",
            code="scheduler_resource_wait",
        )
    if recommendation_payload.get("observation_id") != observation.get("record_id"):
        raise ActorError(
            "resource recommendation binds another observation",
            code="scheduler_resource_admission_stale",
        )
    report = dict(observation_payload.get("report") or {})
    observed_at = str(report.get("generated_at_utc") or observation.get("updated_at_utc") or "")
    now = utc_now()
    age_seconds = (parse_utc(now) - parse_utc(observed_at)).total_seconds()
    if age_seconds < 0 or age_seconds > int(binding.get("max_observation_age_seconds") or 300):
        raise ActorError(
            "resource observation is outside the authorized freshness bound",
            code="scheduler_resource_wait",
        )
    resource_ids = tuple(binding.get("resource_ids") or ())
    if any(str(resource_id).lower().startswith(("cuda:", "gpu:")) for resource_id in resource_ids):
        try:
            import torch

            cuda_live = bool(torch.cuda.is_available() and torch.cuda.device_count() > 0)
        except Exception:
            cuda_live = False
        if not cuda_live:
            raise ActorError(
                "live CUDA provider check failed",
                code="scheduler_resource_wait",
            )
    profile = dict(material.get("worker_profile") or {})
    provider_check = {
        "schema_version": "crca_scheduler_live_provider_check_v1",
        "checked_at_utc": now,
        "resource_ids": list(resource_ids),
        "launcher_type": profile.get("launcher_type"),
        "available": True,
    }
    assignment_digest = contract_sha256(dict(assignment_lease))
    scheduler_owner_digest = contract_sha256(
        {
            "scheduler_id": scheduler_id,
            "runtime_schema_version": RUNTIME_SCHEMA_VERSION,
        }
    )
    ledger = ResourceStartLeaseLedgerV1(scheduler_root(state_root))
    generation, prior_lease_digest = ledger.next_grant_fields(
        resource_ids=resource_ids,
        at_utc=now,
    )
    seed = {
        "assignment_lease_digest": assignment_digest.to_dict(),
        "attempt_admission_request_sha256": binding.get(
            "attempt_admission_request_sha256"
        ),
        "resource_ids": list(resource_ids),
        "generation": generation,
        "prior_lease_digest": (
            None
            if prior_lease_digest is None
            else prior_lease_digest.to_dict()
        ),
        "issued_at_utc": now,
    }
    seed_digest = contract_sha256(seed)
    lease = ResourceStartLeaseV1(
        lease_id=f"RSTART-{seed_digest.hex[:24].upper()}",
        scheduler_owner_id=scheduler_id,
        scheduler_owner_digest=scheduler_owner_digest,
        assignment_lease_id=str(assignment_lease.get("record_id") or ""),
        assignment_lease_digest=assignment_digest,
        attempt_admission_request_digest=_sha256_digest(
            str(binding.get("attempt_admission_request_sha256") or ""),
            field="attempt_admission_request_sha256",
        ),
        resource_ids=resource_ids,
        resource_scope_digest=_sha256_digest(
            str(binding.get("resource_scope_sha256") or ""),
            field="resource_scope_sha256",
        ),
        resource_observation_id=str(observation.get("record_id") or ""),
        resource_observation_digest=contract_sha256(observation),
        resource_recommendation_id=str(recommendation.get("record_id") or ""),
        resource_recommendation_digest=contract_sha256(recommendation),
        live_provider_check_digest=contract_sha256(provider_check),
        generation=generation,
        fencing_token_digest=contract_sha256(
            {"resource_start_seed": seed, "scheduler_owner": scheduler_id}
        ),
        prior_lease_digest=prior_lease_digest,
        issued_at_utc=now,
        start_before_utc=add_seconds(
            now, int(binding.get("start_window_seconds") or 300)
        ),
        expires_at_utc=add_seconds(now, int(binding.get("ttl_seconds") or 300)),
    )
    ledger.grant(lease)
    return lease


def _bind_attempt_leases_before_launch(
    *,
    state_root: str | Path | None,
    assignment: Mapping[str, Any],
    assignment_lease: Mapping[str, Any],
    resource_start_lease: ResourceStartLeaseV1,
    scheduler_id: str,
) -> dict[str, Any]:
    """Persist the exact three-lease Attempt binding before provider launch."""

    material = dict(dict(assignment.get("payload") or {}).get("material") or {})
    admission = dict(material.get("resource_admission") or {})
    attempt_admission = dict(admission.get("attempt_lease_admission") or {})
    continuation = dict(attempt_admission.get("canonical_continuation_lease") or {})
    now = utc_now()
    continuation_fields = {
        "schema_version",
        "lease_id",
        "run_id",
        "attempt_id",
        "restore_checkpoint_digest",
        "lineage_scope_digest",
        "owner_id",
        "lease_epoch",
        "fencing_token_digest",
        "prior_lease_digest",
        "granted_at_utc",
        "expires_at_utc",
    }
    try:
        if (
            attempt_admission.get("schema_version")
            != "crca_scheduler_attempt_lease_admission_v1"
            or set(continuation) != continuation_fields
            or dict(continuation.get("fencing_token_digest") or {}).get(
                "algorithm"
            )
            != "sha256"
        ):
            raise ValueError("unsupported Attempt lease binding wire shape")
        continuation_digest = contract_sha256(continuation)
        continuation_fence = _sha256_digest(
            str(dict(continuation.get("fencing_token_digest") or {}).get("hex") or ""),
            field="continuation fencing token",
        )
        continuation_owner_digest = contract_sha256(
            {
                "owner_id": continuation.get("owner_id"),
                "lease_digest": continuation_digest.to_dict(),
                "lease_epoch": continuation.get("lease_epoch"),
                "fencing_token_digest": continuation_fence.to_dict(),
            }
        )
        granted_at = parse_utc(str(continuation.get("granted_at_utc") or ""))
        expires_at = parse_utc(str(continuation.get("expires_at_utc") or ""))
    except (ActorError, TypeError, ValueError, ProgrammeContractError) as error:
        raise ActorError(
            "canonical continuation lease is invalid at provider admission",
            code="scheduler_attempt_lease_binding_invalid",
        ) from error
    if (
        continuation.get("schema_version")
        != "crca_canonical_continuation_lease_v1"
        or str(continuation.get("attempt_id") or "")
        != str(attempt_admission.get("attempt_id") or "")
        or continuation_digest.hex
        != str(attempt_admission.get("canonical_continuation_lease_sha256") or "")
        or continuation_owner_digest.hex
        != str(attempt_admission.get("continuation_owner_sha256") or "")
        or granted_at > parse_utc(now)
        or parse_utc(now) >= expires_at
    ):
        raise ActorError(
            "canonical continuation lease is stale or differs from Dispatch",
            code="scheduler_attempt_lease_binding_invalid",
        )
    assignment_lease_digest = contract_sha256(dict(assignment_lease))
    scheduler_owner_digest = contract_sha256(
        {
            "scheduler_id": scheduler_id,
            "runtime_schema_version": RUNTIME_SCHEMA_VERSION,
        }
    )
    binding = AttemptLeaseBindingsV1(
        attempt_id=str(attempt_admission.get("attempt_id") or ""),
        scheduler_assignment_lease_id=str(assignment_lease.get("record_id") or ""),
        scheduler_assignment_lease_digest=assignment_lease_digest,
        scheduler_owner_id=scheduler_id,
        scheduler_owner_digest=scheduler_owner_digest,
        resource_start_lease_id=resource_start_lease.lease_id,
        resource_start_lease_digest=resource_start_lease.lease_digest(),
        resource_start_owner_id=resource_start_lease.scheduler_owner_id,
        resource_start_owner_digest=resource_start_lease.scheduler_owner_digest,
        canonical_continuation_lease_id=str(continuation.get("lease_id") or ""),
        canonical_continuation_lease_digest=continuation_digest,
        continuation_owner_id=str(continuation.get("owner_id") or ""),
        continuation_owner_digest=continuation_owner_digest,
    )
    binding.validate_runtime_records(
        assignment_lease_id=str(assignment_lease.get("record_id") or ""),
        assignment_lease_digest=assignment_lease_digest,
        scheduler_owner_id=scheduler_id,
        scheduler_owner_digest=scheduler_owner_digest,
        resource_lease=resource_start_lease,
        continuation_lease=continuation,
    )
    payload = binding.to_dict()
    binding_digest = binding.binding_digest()
    directory = scheduler_root(state_root) / "attempt-lease-bindings" / binding.attempt_id
    path = directory / f"{binding_digest.hex}.json"
    with advisory_lock(directory / ".binding.lock"):
        if path.is_file():
            existing = read_json(path, name="AttemptLeaseBindings")
            if existing != payload:
                raise ActorError(
                    "Attempt lease binding path contains conflicting bytes",
                    code="scheduler_attempt_lease_binding_invalid",
                )
        else:
            atomic_write_json(path, payload)
    return {
        "schema_version": "crca_scheduler_attempt_lease_binding_receipt_v1",
        "attempt_id": binding.attempt_id,
        "binding_digest": binding_digest.to_dict(),
        "binding_path": str(path),
        "binding": payload,
    }


def _terminal_resource_start_event(
    *,
    state_root: str | Path | None,
    lease: ResourceStartLeaseV1,
    code: ResourceStartLeaseEventCodeV1,
    reason: str,
) -> None:
    ledger = ResourceStartLeaseLedgerV1(scheduler_root(state_root))
    prior_events = ledger.events(lease.lease_id)
    event = ResourceStartLeaseEventV1(
        lease_digest=lease.lease_digest(),
        code=code,
        scheduler_owner_id=lease.scheduler_owner_id,
        fencing_token_digest=lease.fencing_token_digest,
        event_at_utc=utc_now(),
        reason=reason,
        prior_event_digest=(
            None if not prior_events else contract_sha256(prior_events[-1])
        ),
        successor_lease_digest=None,
    )
    ledger.append_event(lease.lease_id, event)


def runtime_path(state_root: str | Path | None) -> Path:
    return scheduler_root(state_root) / "runtime.json"


def runtime_lock_path(state_root: str | Path | None) -> Path:
    return scheduler_root(state_root) / "runtime.lock"


def execution_lock_path(state_root: str | Path | None) -> Path:
    return scheduler_root(state_root) / "execution.lock"


@contextlib.contextmanager
def _scheduler_execution_lock(state_root: str | Path | None):
    """Prevent multiple Scheduler processes from violating V1's one-worker gate."""

    path = execution_lock_path(state_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+", encoding="utf-8")
    try:
        try:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ActorError(
                "another Scheduler process already owns the single-worker execution lock",
                code="scheduler_already_running",
            ) from exc
        except ImportError:  # pragma: no cover - non-POSIX fallback
            with advisory_lock(path.with_suffix(".fallback.lock")):
                yield
            return
        yield
    finally:
        try:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except (ImportError, OSError):  # pragma: no cover - non-POSIX fallback
            pass
        handle.close()


def _default_runtime(*, state_root: str | Path | None, scheduler_id: str | None = None) -> dict[str, Any]:
    root = scheduler_root(state_root)
    identity = scheduler_id or scheduler_id_for({"state_root": str(root), "pid": os.getpid()})
    now = utc_now()
    return {
        "schema_version": RUNTIME_SCHEMA_VERSION,
        "scheduler_id": identity,
        "status": "STOPPED",
        "pid": None,
        "started_at_utc": None,
        "updated_at_utc": now,
        "poll_interval_seconds": 300.0,
        "max_workers": 1,
        "processes": {},
        "counters": {
            "assignments_requested": 0,
            "assignments_launched": 0,
            "results_submitted": 0,
            "worker_losses": 0,
            "idle_cycles": 0,
        },
        "last_event": None,
    }


def load_runtime(state_root: str | Path | None, *, scheduler_id: str | None = None) -> dict[str, Any]:
    path = runtime_path(state_root)
    if not path.is_file():
        return _default_runtime(state_root=state_root, scheduler_id=scheduler_id)
    value = read_json(path, name="Scheduler runtime state")
    if value.get("schema_version") != RUNTIME_SCHEMA_VERSION:
        raise ActorError("unsupported Scheduler runtime state", code="scheduler_runtime_schema_unsupported")
    if str(value.get("status")) not in RUNTIME_STATUSES:
        raise ActorError("invalid Scheduler runtime status", code="scheduler_runtime_invalid")
    value.setdefault("processes", {})
    value.setdefault("counters", {})
    return value


def save_runtime(state_root: str | Path | None, runtime: Mapping[str, Any]) -> Path:
    value = dict(runtime)
    value["schema_version"] = RUNTIME_SCHEMA_VERSION
    value["updated_at_utc"] = utc_now()
    return atomic_write_json(runtime_path(state_root), value)


def _mutate_runtime(state_root: str | Path | None, mutator) -> dict[str, Any]:
    with advisory_lock(runtime_lock_path(state_root)):
        runtime = load_runtime(state_root)
        mutator(runtime)
        save_runtime(state_root, runtime)
        return runtime


def set_runtime_status(
    status: str,
    *,
    state_root: str | Path | None,
    scheduler_id: str | None = None,
    poll_interval_seconds: float | None = None,
) -> dict[str, Any]:
    normalized = str(status).strip().upper()
    if normalized not in RUNTIME_STATUSES:
        raise ActorError("invalid Scheduler runtime status", code="scheduler_runtime_invalid")

    def mutate(runtime: dict[str, Any]) -> None:
        if scheduler_id:
            runtime["scheduler_id"] = scheduler_id
        runtime["status"] = normalized
        runtime["pid"] = os.getpid() if normalized in {"RUNNING", "PAUSED", "STOP_REQUESTED"} else None
        if normalized == "RUNNING" and not runtime.get("started_at_utc"):
            runtime["started_at_utc"] = utc_now()
        if poll_interval_seconds is not None:
            runtime["poll_interval_seconds"] = max(0.1, float(poll_interval_seconds))
        runtime["last_event"] = {"kind": "runtime_status", "status": normalized, "at_utc": utc_now()}

    return _mutate_runtime(state_root, mutate)


def _assignment_result_path(assignment: Mapping[str, Any]) -> Path:
    material = dict(dict(assignment.get("payload") or {}).get("material") or {})
    path = str(dict(material.get("result_contract") or {}).get("result_path") or "").strip()
    if not path:
        raise ActorError("Assignment has no Worker Result path", code="assignment_result_path_missing")
    return Path(path).expanduser().resolve()


def _launcher_for_assignment(assignment: Mapping[str, Any]):
    material = dict(dict(assignment.get("payload") or {}).get("material") or {})
    profile = dict(material.get("worker_profile") or {})
    launcher_type = str(profile.get("launcher_type") or "fake")
    if launcher_type == "fake":
        return FakeWorkerLauncher(
            behavior=str(profile.get("fake_behavior") or "checkpoint"),
            fixture_content=str(profile.get("fake_fixture_content") or "scheduler fake worker fixture\n"),
            timeout_seconds=int(profile.get("timeout_seconds") or 300),
            sleep_seconds=float(profile.get("fake_sleep_seconds") or 0.0),
        )
    if launcher_type == "codex_exec":
        return CodexExecLauncher()
    raise ActorError("unsupported Assignment launcher", code="scheduler_launcher_invalid")


def _record_launching(
    *,
    state_root: str | Path | None,
    assignment: Mapping[str, Any],
    lease: Mapping[str, Any],
    result_path: Path,
    identity: Mapping[str, Any],
    log_dir: Path,
    resource_start_lease: ResourceStartLeaseV1 | None = None,
    attempt_lease_binding: Mapping[str, Any] | None = None,
) -> None:
    """Persist launch intent before any process can exist."""

    assignment_id = str(assignment.get("record_id"))
    handshake_path = log_dir / "launch_handshake.json"
    ack_path = log_dir / "launch_ack.txt"

    def mutate(runtime: dict[str, Any]) -> None:
        processes = dict(runtime.get("processes") or {})
        processes[assignment_id] = {
            "assignment_id": assignment_id,
            "lease_id": lease.get("record_id"),
            "assignment_material_sha256": dict(assignment.get("payload") or {}).get("assignment_material_sha256"),
            "worker_identity": dict(identity),
            "result_path": str(result_path),
            "log_dir": str(log_dir),
            "handshake_path": str(handshake_path),
            "ack_path": str(ack_path),
            "status": "LAUNCHING",
            "resource_start_lease_id": (
                None if resource_start_lease is None else resource_start_lease.lease_id
            ),
            "resource_start_lease_digest": (
                None
                if resource_start_lease is None
                else resource_start_lease.lease_digest().to_dict()
            ),
            "attempt_lease_binding": (
                None if attempt_lease_binding is None else dict(attempt_lease_binding)
            ),
            "launch_started_at_utc": utc_now(),
            "updated_at_utc": utc_now(),
        }
        runtime["processes"] = processes
        runtime["last_event"] = {"kind": "worker_launching", "assignment_id": assignment_id, "at_utc": utc_now()}

    _mutate_runtime(state_root, mutate)


def _promote_process(
    *,
    state_root: str | Path | None,
    assignment: Mapping[str, Any],
    lease: Mapping[str, Any],
    identity: Mapping[str, Any],
) -> None:
    assignment_id = str(assignment.get("record_id"))

    def mutate(runtime: dict[str, Any]) -> None:
        processes = dict(runtime.get("processes") or {})
        row = dict(processes.get(assignment_id) or {})
        if not row:
            raise ActorError("durable LAUNCHING row is missing", code="scheduler_launch_record_missing")
        already_running = row.get("status") == "RUNNING"
        row.update({
            "lease_id": lease.get("record_id"),
            "worker_identity": dict(identity),
            "status": "RUNNING",
            "started_at_utc": row.get("started_at_utc") or utc_now(),
            "updated_at_utc": utc_now(),
        })
        processes[assignment_id] = row
        runtime["processes"] = processes
        if not already_running:
            counters = dict(runtime.get("counters") or {})
            counters["assignments_launched"] = int(counters.get("assignments_launched", 0)) + 1
            runtime["counters"] = counters
        runtime["last_event"] = {"kind": "worker_started", "assignment_id": assignment_id, "at_utc": utc_now()}

    _mutate_runtime(state_root, mutate)



def _prune_run_directories(state_root: str | Path | None, *, keep: int = 20) -> None:
    runs_root = scheduler_root(state_root) / "runs"
    if not runs_root.is_dir():
        return
    rows = sorted(
        (path for path in runs_root.iterdir() if path.is_dir()),
        key=lambda path: (path.stat().st_mtime, path.name),
        reverse=True,
    )
    for path in rows[max(1, int(keep)):]:
        shutil.rmtree(path, ignore_errors=True)

def _finish_process(
    *,
    state_root: str | Path | None,
    assignment_id: str,
    status: str,
    details: Mapping[str, Any] | None = None,
    result_submitted: bool = False,
    worker_loss: bool = False,
) -> None:
    """Forget terminal process rows while retaining bounded counters/event data."""

    def mutate(runtime: dict[str, Any]) -> None:
        processes = dict(runtime.get("processes") or {})
        row = dict(processes.pop(assignment_id, {}) or {})
        runtime["processes"] = processes
        counters = dict(runtime.get("counters") or {})
        if result_submitted:
            counters["results_submitted"] = int(counters.get("results_submitted", 0)) + 1
        if worker_loss:
            counters["worker_losses"] = int(counters.get("worker_losses", 0)) + 1
        runtime["counters"] = counters
        runtime["last_event"] = {
            "kind": "worker_finished",
            "assignment_id": assignment_id,
            "status": status,
            "at_utc": utc_now(),
            "details": {
                key: value for key, value in {**row, **dict(details or {})}.items()
                if key in {
                    "result_id",
                    "returncode",
                    "reason",
                    "error",
                    "errors",
                    "result_path",
                    "reconciled",
                }
            },
        }

    _mutate_runtime(state_root, mutate)
    _prune_run_directories(state_root)


def _bound_log_file(path: str | Path, *, maximum_bytes: int = 2 * 1024 * 1024) -> None:
    file_path = Path(path)
    if not file_path.is_file() or file_path.stat().st_size <= maximum_bytes:
        return
    with file_path.open("rb") as handle:
        handle.seek(-maximum_bytes, os.SEEK_END)
        tail = handle.read()
    marker = b"[EIEL scheduler truncated older worker output]\n"
    file_path.write_bytes(marker + tail)

def _heartbeat_callback(
    *,
    state_root: str | Path | None,
    lease_id: str,
    assignment_id: str,
    worker_identity: Mapping[str, Any],
):
    def heartbeat(_pid: int) -> None:
        current = {
            key: value for key, value in dict(worker_identity).items()
            if key in {"kind", "worker_id", "pid", "process_start_identity", "thread_id"} and value is not None
        }
        thread_id = str(current.get("thread_id") or "")
        if thread_id.startswith(("EIEL-PROVISIONAL-", "PENDING-CODEX-")):
            current.pop("thread_id", None)
        lease = heartbeat_lease(lease_id, state_root=state_root, worker=current)

        def mutate(runtime: dict[str, Any]) -> None:
            processes = dict(runtime.get("processes") or {})
            row = dict(processes.get(assignment_id) or {})
            if row:
                row["last_heartbeat_at_utc"] = dict(lease.get("payload") or {}).get("last_heartbeat_at_utc")
                row["updated_at_utc"] = utc_now()
                processes[assignment_id] = row
                runtime["processes"] = processes

        _mutate_runtime(state_root, mutate)

    return heartbeat

def _load_and_submit_result(*, result_path: Path, state_root: str | Path | None) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if not result_path.is_file():
        raise ActorError("worker exited without a structured Worker Result", code="worker_result_missing")
    result = read_json(result_path, name="Worker Result")
    return submit_worker_result(result, state_root=state_root)


def _submit_result_for_settlement(
    result_path: Path, state_root: str | Path | None
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    return _load_and_submit_result(result_path=result_path, state_root=state_root)


def _attempt_result_settlement(
    *,
    state_root: str | Path | None,
    assignment_id: str,
    lease_id: str,
    result_path: Path,
    grace_seconds: float = RESULT_SETTLEMENT_GRACE_SECONDS,
) -> dict[str, Any]:
    del grace_seconds  # Waiting is owned by the caller or the bounded helper.
    return settle_worker_result(
        mutate_runtime=_mutate_runtime,
        submit_result=_submit_result_for_settlement,
        state_root=state_root,
        assignment_id=assignment_id,
        lease_id=lease_id,
        result_path=result_path,
        validation_attempts=RESULT_SETTLEMENT_VALIDATION_ATTEMPTS,
    )


def _attempt_result_settlement_with_short_wait(
    *,
    state_root: str | Path | None,
    assignment_id: str,
    lease_id: str,
    result_path: Path,
) -> dict[str, Any]:
    return settle_worker_result_with_wait(
        mutate_runtime=_mutate_runtime,
        submit_result=_submit_result_for_settlement,
        state_root=state_root,
        assignment_id=assignment_id,
        lease_id=lease_id,
        result_path=result_path,
        grace_seconds=RESULT_SETTLEMENT_GRACE_SECONDS,
        validation_attempts=RESULT_SETTLEMENT_VALIDATION_ATTEMPTS,
    )


def _run_once_unlocked(
    *,
    state_root: str | Path | None,
    scheduler_id: str | None = None,
    dispatch_id: str | None = None,
    work_id: str | None = None,
) -> dict[str, Any]:
    """Request and execute at most one mechanically activated Assignment."""

    root = scheduler_root(state_root)
    root.mkdir(parents=True, exist_ok=True)
    runtime = load_runtime(state_root, scheduler_id=scheduler_id)
    scheduler_identity = scheduler_id or str(runtime.get("scheduler_id") or scheduler_id_for({"state_root": str(root)}))
    recover_preparing(state_root=state_root)
    expire_leases(state_root=state_root)

    requested = request_assignment(
        state_root=state_root, dispatch_id=dispatch_id, work_id=work_id,
        with_diagnostics=True,
    )
    assignment, candidate_rejections = requested
    if assignment is None:
        def idle(r: dict[str, Any]) -> None:
            counters = dict(r.get("counters") or {})
            counters["assignments_requested"] = int(counters.get("assignments_requested", 0)) + 1
            counters["idle_cycles"] = int(counters.get("idle_cycles", 0)) + 1
            r["counters"] = counters
            r["last_event"] = {
                "kind": "no_assignment", "at_utc": utc_now(),
                "candidate_rejections": candidate_rejections[-10:],
            }
        _mutate_runtime(state_root, idle)
        return {
            "status": "NO_ASSIGNMENT", "scheduler_id": scheduler_identity,
            "assignment": None, "candidate_rejections": candidate_rejections,
        }

    assignment_id = str(assignment.get("record_id"))
    claimed, lease = claim_assignment(assignment_id, scheduler_id=scheduler_identity, state_root=state_root)
    resource_start_lease: ResourceStartLeaseV1 | None = None
    try:
        resource_start_lease = _grant_resource_start_lease(
            state_root=state_root,
            assignment=claimed,
            assignment_lease=lease,
            scheduler_id=scheduler_identity,
        )
        attempt_lease_binding = (
            None
            if resource_start_lease is None
            else _bind_attempt_leases_before_launch(
                state_root=state_root,
                assignment=claimed,
                assignment_lease=lease,
                resource_start_lease=resource_start_lease,
                scheduler_id=scheduler_identity,
            )
        )
    except (ActorError, ProgrammeContractError) as exc:
        if resource_start_lease is not None:
            _terminal_resource_start_event(
                state_root=state_root,
                lease=resource_start_lease,
                code=ResourceStartLeaseEventCodeV1.RELEASED,
                reason="Attempt lease binding failed before provider launch",
            )
        reason = f"PRELAUNCH_RESOURCE_START_WAIT:{getattr(exc, 'code', type(exc).__name__)}:{exc}"
        record_execution_loss(
            assignment_id,
            str(lease.get("record_id")),
            state_root=state_root,
            reason=reason,
        )
        return {
            "status": "WAIT_RESOURCE",
            "scheduler_id": scheduler_identity,
            "assignment": assignment_store(state_root).get(assignment_id),
            "lease": lease_store(state_root).get(str(lease.get("record_id"))),
            "reason": reason,
            "attempt_created": False,
            "effect_executed": False,
        }
    result_path = _assignment_result_path(claimed)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    if result_path.exists():
        result_path.unlink()
    log_dir = root / "runs" / assignment_id / str(lease.get("record_id"))
    launcher = _launcher_for_assignment(claimed)
    is_fake = isinstance(launcher, FakeWorkerLauncher)
    worker_id = f"{'fake' if is_fake else 'codex'}-{assignment_id.lower()}"
    initial_thread_id = f"FAKE-THREAD-{assignment_id}" if is_fake else f"EIEL-PROVISIONAL-{worker_id}"
    mechanical_identity = {
        "kind": "fake" if is_fake else "codex_exec",
        "worker_id": worker_id,
        "thread_id": initial_thread_id,
        "attempt_lease_binding": attempt_lease_binding,
        **launch_contract_for_assignment(claimed, lease),
    }

    # Persist LAUNCHING before Popen.  The ACK-gated wrapper cannot execute
    # worker code until this callback durably binds PID, Assignment, Lease and
    # Session and revalidates Dispatch immediately before launch.
    _record_launching(
        state_root=state_root, assignment=claimed, lease=lease,
        result_path=result_path, identity=mechanical_identity, log_dir=log_dir,
        resource_start_lease=resource_start_lease,
        attempt_lease_binding=attempt_lease_binding,
    )
    started_identity: dict[str, Any] = dict(mechanical_identity)
    resource_start_terminal = False

    def on_started(identity: dict[str, Any]) -> None:
        nonlocal claimed, lease, started_identity, resource_start_terminal
        started_identity = {**mechanical_identity, **dict(identity)}
        if resource_start_lease is not None:
            if not resource_start_lease.is_startable_at(utc_now()):
                raise ActorError(
                    "resource-start window expired before worker start",
                    code="scheduler_resource_start_window_expired",
                )
            _terminal_resource_start_event(
                state_root=state_root,
                lease=resource_start_lease,
                code=ResourceStartLeaseEventCodeV1.STARTED,
                reason="ACK-gated Scheduler worker process started",
            )
            resource_start_terminal = True
        claimed, lease = mark_assignment_running(
            assignment_id, str(lease.get("record_id")), started_identity, state_root=state_root
        )
        worker_open(
            assignment_id,
            str(lease.get("record_id")),
            state_root=state_root,
            worker_id=worker_id,
            codex_id=initial_thread_id,
        )
        acknowledge_worker_launch(
            assignment_id, str(lease.get("record_id")), log_dir / "launch_ack.txt",
            state_root=state_root,
            launch_nonce=str(started_identity.get("launch_nonce") or ""),
        )
        _promote_process(
            state_root=state_root, assignment=claimed, lease=lease,
            identity=started_identity,
        )

    def on_thread_started(thread_id: str) -> None:
        nonlocal started_identity, lease
        bind_worker_thread_identity(
            assignment_id, str(lease.get("record_id")), state_root=state_root,
            worker_id=worker_id, thread_id=thread_id,
        )
        started_identity["thread_id"] = thread_id

    try:
        launch: LaunchResult = launcher.launch(
            assignment=claimed,
            lease=lease,
            state_root=Path(state_root).resolve() if state_root is not None else scheduler_root(None).parent,
            result_path=result_path,
            log_dir=log_dir,
            heartbeat=_heartbeat_callback(
                state_root=state_root,
                lease_id=str(lease.get("record_id")),
                assignment_id=assignment_id,
                worker_identity=started_identity,
            ),
            heartbeat_interval_seconds=float(dict(lease.get("payload") or {}).get("heartbeat_interval_seconds", 60)),
            on_started=on_started,
            on_thread_started=on_thread_started,
        )
        _bound_log_file(launch.stdout_path)
        _bound_log_file(launch.stderr_path)
        if launch.returncode != 0:
            reason = f"WORKER_PROCESS_EXIT_{launch.returncode}"
            record_execution_loss(assignment_id, str(lease.get("record_id")), state_root=state_root, reason=reason)
            _finish_process(
                state_root=state_root, assignment_id=assignment_id, status="PROCESS_FAILURE",
                details={"returncode": launch.returncode, "reason": reason}, worker_loss=True,
            )
            return {
                "status": "PROCESS_FAILURE",
                "assignment": assignment_store(state_root).get(assignment_id),
                "lease": lease_store(state_root).get(str(lease.get("record_id"))),
                "launch": launch.__dict__,
            }
        settlement_deadline = time.monotonic() + 3.0
        settlement: dict[str, Any] = {"status": "RESULT_PENDING"}
        while settlement.get("status") == "RESULT_PENDING" and time.monotonic() < settlement_deadline:
            settlement = _attempt_result_settlement(
                state_root=state_root,
                assignment_id=assignment_id,
                lease_id=str(lease.get("record_id")),
                result_path=result_path,
            )
            if settlement.get("status") == "RESULT_PENDING":
                time.sleep(0.1)
        if settlement.get("status") != "RESULT_SUBMITTED":
            raise ActorError(
                "Worker Result did not settle into a valid structured result",
                code=(
                    "worker_result_invalid"
                    if settlement.get("status") == "RESULT_INVALID"
                    else "worker_result_settlement_timeout"
                ),
                problems=list(settlement.get("errors") or [str(settlement.get("error") or settlement.get("status"))]),
            )
        result_record = dict(settlement["result_record"])
        closed_assignment = dict(settlement["assignment"])
        closed_lease = dict(settlement["lease"])
        _finish_process(
            state_root=state_root, assignment_id=assignment_id, status="RESULT_SUBMITTED",
            details={"result_id": result_record.get("record_id"), "returncode": launch.returncode},
            result_submitted=True,
        )
        return {
            "status": "RESULT_SUBMITTED", "scheduler_id": scheduler_identity,
            "assignment": closed_assignment, "lease": closed_lease,
            "worker_result": result_record, "launch": launch.__dict__,
        }
    except BaseException as exc:
        if resource_start_lease is not None and not resource_start_terminal:
            try:
                _terminal_resource_start_event(
                    state_root=state_root,
                    lease=resource_start_lease,
                    code=ResourceStartLeaseEventCodeV1.RELEASED,
                    reason=f"prelaunch failed before start: {type(exc).__name__}",
                )
                resource_start_terminal = True
            except BaseException:
                # The durable unclosed lease remains visible and blocks a
                # conflicting grant; reconciliation must close it explicitly.
                pass
        reason = f"SCHEDULER_LAUNCH_FAILURE:{type(exc).__name__}:{exc}"
        loss_error: BaseException | None = None
        try:
            record_execution_loss(
                assignment_id,
                str(lease.get("record_id")),
                state_root=state_root,
                reason=reason,
            )
        except BaseException as recovery_exc:
            loss_error = recovery_exc

        if loss_error is None:
            _finish_process(
                state_root=state_root,
                assignment_id=assignment_id,
                status="PROCESS_FAILURE",
                details={"error": str(exc), "reason": reason},
                worker_loss=True,
            )
        else:
            # Keep the durable launch row.  Removing it here would turn a
            # failed worker-loss write into an invisible execution attempt.
            # Reconciliation can retry the exact transition later.
            def mark_recovery_required(runtime: dict[str, Any]) -> None:
                processes = dict(runtime.get("processes") or {})
                row = dict(processes.get(assignment_id) or {})
                row["recovery_required"] = True
                row["recovery_reason"] = reason
                recovery_code = str(getattr(loss_error, "code", type(loss_error).__name__))
                row["recovery_error"] = f"{recovery_code}:{loss_error}"
                row["updated_at_utc"] = utc_now()
                processes[assignment_id] = row
                runtime["processes"] = processes
                runtime["last_event"] = {
                    "kind": "worker_loss_recovery_required",
                    "assignment_id": assignment_id,
                    "at_utc": utc_now(),
                    "reason": reason,
                    "error": row["recovery_error"],
                }

            _mutate_runtime(state_root, mark_recovery_required)
            raise ActorError(
                "worker launch failed and durable worker-loss recording also failed",
                code="scheduler_worker_loss_recovery_required",
                problems=[
                    f"assignment_id={assignment_id}",
                    f"launch_error={exc}",
                    f"recovery_error={loss_error}",
                ],
                next_actions=["Run Scheduler reconciliation before requesting more work."],
            ) from loss_error

        if isinstance(exc, (ActorError, KeyboardInterrupt)):
            raise
        raise ActorError(str(exc), code="scheduler_run_once_failed") from exc

def run_once(
    *,
    state_root: str | Path | None,
    scheduler_id: str | None = None,
    dispatch_id: str | None = None,
    work_id: str | None = None,
) -> dict[str, Any]:
    """Execute at most one Assignment under the global V1 single-worker gate."""

    with _scheduler_execution_lock(state_root):
        return _run_once_unlocked(
            state_root=state_root,
            scheduler_id=scheduler_id,
            dispatch_id=dispatch_id,
            work_id=work_id,
        )




def _read_launch_handshake(row: Mapping[str, Any]) -> dict[str, Any] | None:
    path = Path(str(row.get("handshake_path") or ""))
    if not path.is_file():
        return None
    try:
        value = read_json(path, name="Scheduler launch handshake")
    except ActorError:
        return None
    return value


def _thread_id_from_events(row: Mapping[str, Any]) -> str | None:
    sidecar = Path(
        str(
            dict(row.get("worker_identity") or {}).get("thread_identity_path")
            or (Path(str(row.get("log_dir") or "")) / "thread_identity.json")
        )
    )
    if sidecar.is_file():
        try:
            captured = read_json(sidecar, name="Codex thread identity")
        except ActorError:
            captured = {}
        if captured.get("status") == "CONFLICT":
            raise ActorError(
                "Codex worker emitted conflicting thread identities",
                code="codex_thread_identity_conflict",
            )
        value = str(captured.get("thread_id") or "")
        if captured.get("status") == "BOUND" and value:
            return value
    path = Path(str(row.get("log_dir") or "")) / "events.jsonl"
    if not path.is_file():
        return None
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None
    for line in lines:
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") == "thread.started":
            value = str(event.get("thread_id") or dict(event.get("thread") or {}).get("id") or "")
            if value:
                return value
    return None


def _merge_handshake_identity(row: Mapping[str, Any], handshake: Mapping[str, Any] | None) -> dict[str, Any]:
    identity = dict(row.get("worker_identity") or {})
    if handshake:
        identity.update({
            "pid": int(handshake.get("supervisor_pid") or identity.get("pid") or 0) or None,
            "process_start_identity": handshake.get("supervisor_process_start_identity") or identity.get("process_start_identity"),
            "process_group_id": int(handshake.get("supervisor_pid") or identity.get("process_group_id") or 0) or None,
        })
    return identity


def _write_launch_ack(*, state_root: str | Path | None, assignment_id: str, lease_id: str, row: Mapping[str, Any]) -> None:
    raw_path = str(row.get("ack_path") or "").strip()
    if not raw_path:
        raise ActorError("launch ACK path is missing", code="scheduler_launch_record_invalid")
    path = Path(raw_path)
    acknowledge_worker_launch(
        assignment_id,
        lease_id,
        path,
        state_root=state_root,
        launch_nonce=str(dict(row.get("worker_identity") or {}).get("launch_nonce") or ""),
    )


def _cleanup_row_process_groups(row: Mapping[str, Any]) -> None:
    handshake = _read_launch_handshake(row) or {}
    identity = dict(row.get("worker_identity") or {})
    # Runtime state is durable but not sufficient proof that an arbitrary PID
    # belongs to EIEL.  Only kill groups carrying the launch nonce written by
    # the ACK-gated launcher or a durable wrapper handshake.  This prevents a
    # corrupt/unbound row from terminating an unrelated host process.
    if not handshake and not str(identity.get("launch_nonce") or ""):
        return
    target_group = handshake.get("target_process_group_id")
    if target_group:
        terminate_process_group_id(int(target_group), grace_seconds=0.5)
    supervisor_group = identity.get("process_group_id") or identity.get("pid") or handshake.get("supervisor_pid")
    if supervisor_group:
        terminate_process_group_id(int(supervisor_group), grace_seconds=0.5)


def _activate_recovered_launch(*, state_root: str | Path | None, assignment_id: str, row: Mapping[str, Any], identity: Mapping[str, Any]) -> dict[str, Any]:
    lease_id = str(row.get("lease_id") or "")
    assignment = assignment_store(state_root).get(assignment_id)
    lease = lease_store(state_root).get(lease_id)
    worker_id = str(dict(identity).get("worker_id") or "")
    if not worker_id:
        raise ActorError("recovered launch has no worker_id", code="scheduler_launch_record_invalid")
    material = dict(dict(assignment.get("payload") or {}).get("material") or {})
    profile = dict(material.get("worker_profile") or {})
    provisional_thread = str(dict(identity).get("thread_id") or "")
    if not provisional_thread:
        provisional_thread = f"EIEL-PROVISIONAL-{worker_id}" if profile.get("launcher_type") == "codex_exec" else f"FAKE-THREAD-{assignment_id}"
    assignment, lease = mark_assignment_running(
        assignment_id, lease_id, identity, state_root=state_root,
    )
    worker_open(
        assignment_id, lease_id, state_root=state_root,
        worker_id=worker_id, codex_id=provisional_thread,
    )
    _promote_process(state_root=state_root, assignment=assignment, lease=lease, identity=identity)
    current_row = dict(load_runtime(state_root).get("processes", {}).get(assignment_id) or row)
    _write_launch_ack(
        state_root=state_root, assignment_id=assignment_id, lease_id=lease_id, row=current_row,
    )
    return current_row


def _heartbeat_recovered(*, state_root: str | Path | None, assignment_id: str, row: Mapping[str, Any]) -> dict[str, Any]:
    lease_id = str(row.get("lease_id") or "")
    identity = {
        key: value for key, value in dict(row.get("worker_identity") or {}).items()
        if key in {"kind", "worker_id", "pid", "process_start_identity", "thread_id"} and value is not None
    }
    thread_id = str(identity.get("thread_id") or "")
    if thread_id.startswith(("EIEL-PROVISIONAL-", "PENDING-CODEX-")):
        identity.pop("thread_id", None)
    lease = heartbeat_lease(lease_id, state_root=state_root, worker=identity)
    thread_id = _thread_id_from_events(row)
    if thread_id:
        bound = dict(dict(lease.get("payload") or {}).get("worker") or {})
        prior = str(bound.get("thread_id") or "")
        if prior != thread_id:
            bind_worker_thread_identity(
                assignment_id, lease_id, state_root=state_root,
                worker_id=str(identity.get("worker_id") or ""), thread_id=thread_id,
            )
    return lease

def reconcile(
    *,
    state_root: str | Path | None,
    wait_for_live: bool = False,
    poll_interval_seconds: float = 1.0,
    launch_grace_seconds: float = 10.0,
) -> dict[str, Any]:
    """Reconcile durable launch/process observations before requesting work."""

    outcomes: list[dict[str, Any]] = []
    while True:
        runtime = load_runtime(state_root)
        active_rows = {
            assignment_id: dict(raw or {})
            for assignment_id, raw in sorted(dict(runtime.get("processes") or {}).items())
            if dict(raw or {}).get("status") in {"LAUNCHING", "RUNNING", "LAUNCHED", "RESULT_PENDING"}
        }
        live_found = False
        for assignment_id, row in active_rows.items():
            result_path = Path(str(row.get("result_path") or ""))
            lease_id = str(row.get("lease_id") or "")
            handshake = _read_launch_handshake(row)
            identity = _merge_handshake_identity(row, handshake)
            if identity != dict(row.get("worker_identity") or {}):
                def update_identity(value: dict[str, Any], *, aid=assignment_id, merged=identity) -> None:
                    processes = dict(value.get("processes") or {})
                    current = dict(processes.get(aid) or {})
                    current["worker_identity"] = dict(merged)
                    current["updated_at_utc"] = utc_now()
                    processes[aid] = current
                    value["processes"] = processes
                _mutate_runtime(state_root, update_identity)
                row = dict(load_runtime(state_root).get("processes", {}).get(assignment_id) or row)
            pid = identity.get("pid")
            live = process_identity_is_live(pid, identity.get("process_start_identity"))

            waiting_for_ack = bool(
                handshake is not None
                and str(handshake.get("status") or "") in {"WAITING_ACK", "ACK_TIMEOUT"}
                and not Path(str(row.get("ack_path") or "")).is_file()
            )
            if row.get("status") in {"LAUNCHING", "RUNNING"} and handshake is not None and live and (row.get("status") == "LAUNCHING" or waiting_for_ack):
                try:
                    row = _activate_recovered_launch(
                        state_root=state_root, assignment_id=assignment_id,
                        row=row, identity=identity,
                    )
                except ActorError as exc:
                    _cleanup_row_process_groups(row)
                    try:
                        record_execution_loss(assignment_id, lease_id, state_root=state_root, reason=f"RECOVERED_LAUNCH_REJECTED:{exc.code}")
                    except ActorError:
                        pass
                    _finish_process(
                        state_root=state_root, assignment_id=assignment_id,
                        status="PROCESS_FAILURE", details={"reconciled": True, "reason": exc.code}, worker_loss=True,
                    )
                    outcomes.append({"assignment_id": assignment_id, "status": "PROCESS_FAILURE", "error": str(exc)})
                    continue

            if live:
                live_found = True
                try:
                    _heartbeat_recovered(state_root=state_root, assignment_id=assignment_id, row=row)
                except ActorError as exc:
                    _cleanup_row_process_groups(row)
                    try:
                        record_execution_loss(assignment_id, lease_id, state_root=state_root, reason=f"RECOVERY_HEARTBEAT_FAILED:{exc.code}")
                    except ActorError:
                        pass
                    _finish_process(
                        state_root=state_root, assignment_id=assignment_id,
                        status="PROCESS_FAILURE", details={"reconciled": True, "reason": exc.code}, worker_loss=True,
                    )
                    outcomes.append({"assignment_id": assignment_id, "status": "PROCESS_FAILURE", "error": str(exc)})
                    continue
                outcomes.append({"assignment_id": assignment_id, "status": "WORKER_LIVE", "pid": pid})
                continue

            if row.get("status") == "LAUNCHING" and handshake is None:
                started = str(row.get("launch_started_at_utc") or "")
                age = (parse_utc(utc_now()) - parse_utc(started)).total_seconds() if started else launch_grace_seconds + 1
                if age <= launch_grace_seconds:
                    live_found = True
                    outcomes.append({"assignment_id": assignment_id, "status": "LAUNCH_PENDING"})
                    continue

            _cleanup_row_process_groups(row)
            if result_path.is_file():
                settlement = _attempt_result_settlement_with_short_wait(
                    state_root=state_root,
                    assignment_id=assignment_id,
                    lease_id=lease_id,
                    result_path=result_path,
                )
                if settlement.get("status") == "RESULT_SUBMITTED":
                    result_record = dict(settlement["result_record"])
                    _finish_process(
                        state_root=state_root, assignment_id=assignment_id,
                        status="RESULT_SUBMITTED", details={"result_id": result_record.get("record_id"), "reconciled": True},
                        result_submitted=True,
                    )
                    outcomes.append({"assignment_id": assignment_id, "status": "RESULT_SUBMITTED", "result_id": result_record.get("record_id")})
                    continue
                if settlement.get("status") == "RESULT_PENDING":
                    if wait_for_live:
                        live_found = True
                    outcomes.append({"assignment_id": assignment_id, "status": "RESULT_PENDING"})
                    continue
                invalid = {
                    "assignment_id": assignment_id,
                    "status": "RESULT_INVALID",
                    "error": settlement.get("error"),
                    "errors": settlement.get("errors"),
                    "result_path": str(result_path),
                }
                outcomes.append(invalid)
                try:
                    record_execution_loss(
                        assignment_id, lease_id, state_root=state_root,
                        reason="SCHEDULER_RECONCILE_RESULT_INVALID",
                    )
                except ActorError as exc:
                    outcomes.append({"assignment_id": assignment_id, "status": "RECONCILE_ERROR", "error": str(exc)})
                    continue
                _finish_process(
                    state_root=state_root, assignment_id=assignment_id,
                    status="PROCESS_FAILURE", details={**invalid, "reconciled": True},
                    worker_loss=True,
                )
                outcomes.append({"assignment_id": assignment_id, "status": "PROCESS_FAILURE", "reason": "result_invalid"})
                continue
            try:
                record_execution_loss(assignment_id, lease_id, state_root=state_root, reason="SCHEDULER_RECONCILE_WORKER_MISSING")
            except ActorError as exc:
                outcomes.append({"assignment_id": assignment_id, "status": "RECONCILE_ERROR", "error": str(exc)})
                continue
            _finish_process(
                state_root=state_root, assignment_id=assignment_id,
                status="PROCESS_FAILURE",
                details={
                    "reconciled": True,
                    "reason": "worker_missing",
                    "result_path": str(result_path) if result_path.is_file() else None,
                },
                worker_loss=True,
            )
            outcomes.append({"assignment_id": assignment_id, "status": "PROCESS_FAILURE"})

        # Durable RUNNING/CLAIMED state without runtime bookkeeping cannot be
        # safely adopted.  A result may still converge; otherwise record loss.
        represented = set(active_rows)
        for assignment in records(assignment_store(state_root)):
            if assignment.get("status") not in {"CLAIMED", "RUNNING"}:
                continue
            assignment_id = str(assignment.get("record_id") or "")
            if assignment_id in represented:
                continue
            result_path = _assignment_result_path(assignment)
            payload = dict(assignment.get("payload") or {})
            lease_id = str(dict(payload.get("operational") or {}).get("current_lease_id") or "")
            if result_path.is_file():
                settlement = _attempt_result_settlement_with_short_wait(
                    state_root=state_root,
                    assignment_id=assignment_id,
                    lease_id=lease_id,
                    result_path=result_path,
                )
                if settlement.get("status") == "RESULT_SUBMITTED":
                    result_record = dict(settlement["result_record"])
                    _finish_process(
                        state_root=state_root,
                        assignment_id=assignment_id,
                        status="RESULT_SUBMITTED",
                        details={"result_id": result_record.get("record_id"), "reconciled": True},
                        result_submitted=True,
                    )
                    outcomes.append({"assignment_id": assignment_id, "status": "RESULT_SUBMITTED", "result_id": result_record.get("record_id")})
                    continue
                if settlement.get("status") == "RESULT_PENDING":
                    if wait_for_live:
                        live_found = True
                    outcomes.append({"assignment_id": assignment_id, "status": "RESULT_PENDING"})
                    continue
                outcomes.append({
                    "assignment_id": assignment_id,
                    "status": "RESULT_INVALID",
                    "error": settlement.get("error"),
                    "errors": settlement.get("errors"),
                    "result_path": str(result_path),
                })
            try:
                record_execution_loss(assignment_id, lease_id, state_root=state_root, reason="SCHEDULER_RUNTIME_STATE_MISSING")
                outcomes.append({"assignment_id": assignment_id, "status": "PROCESS_FAILURE", "reason": "runtime_state_missing"})
            except ActorError as exc:
                outcomes.append({"assignment_id": assignment_id, "status": "RECONCILE_ERROR", "error": str(exc)})

        expire_leases(state_root=state_root)
        recover_preparing(state_root=state_root)
        if not wait_for_live or not live_found:
            break
        time.sleep(max(0.1, float(poll_interval_seconds)))
    return {"status": "RECONCILED", "outcomes": outcomes, "runtime": load_runtime(state_root)}


def scheduler_status(*, state_root: str | Path | None) -> dict[str, Any]:
    runtime = load_runtime(state_root)
    assignments = records(assignment_store(state_root))
    leases = records(lease_store(state_root))
    results = records(result_store(state_root))
    claimable = sum(row.get("status") == "READY" for row in assignments)
    executing = sum(row.get("status") in {"CLAIMED", "RUNNING"} for row in assignments)
    recovery_required = sum(
        row.get("status") in {"PREPARING", "CLEANUP_REQUIRED"}
        for row in assignments
    )
    result_pending = sum(row.get("status") == "RESULT_SUBMITTED" for row in assignments) + sum(
        dict(row or {}).get("status") == "RESULT_PENDING"
        for row in dict(runtime.get("processes") or {}).values()
    )
    return {
        "status": runtime.get("status"),
        "runtime": runtime,
        "records": {
            "assignments": len(assignments),
            "active_assignments": claimable + executing + recovery_required + result_pending,
            "claimable_assignments": claimable,
            "executing_assignments": executing,
            "recovery_required_assignments": recovery_required,
            "result_pending_assignments": result_pending,
            "leases": len(leases),
            "active_leases": sum(row.get("status") == "ACTIVE" for row in leases),
            "worker_results": len(results),
        },
        "optionality_audit": "NOT_MEASURED",
    }


def worker_rows(*, state_root: str | Path | None) -> list[dict[str, Any]]:
    runtime = load_runtime(state_root)
    return [dict(row) for _, row in sorted(dict(runtime.get("processes") or {}).items())]


def start_polling(
    *,
    state_root: str | Path | None,
    poll_interval_seconds: float = 300.0,
    max_assignments: int | None = None,
    max_idle_cycles: int | None = None,
    scheduler_id: str | None = None,
) -> dict[str, Any]:
    """Run the scheduler in the foreground with one worker at a time."""

    if max_assignments is not None and max_assignments < 0:
        raise ActorError("max_assignments must be non-negative", code="scheduler_limit_invalid")
    if max_idle_cycles is not None and max_idle_cycles < 0:
        raise ActorError("max_idle_cycles must be non-negative", code="scheduler_limit_invalid")
    with _scheduler_execution_lock(state_root):
        runtime = set_runtime_status(
            "RUNNING",
            state_root=state_root,
            scheduler_id=scheduler_id,
            poll_interval_seconds=poll_interval_seconds,
        )
        completed = 0
        idle = 0
        events: list[dict[str, Any]] = []
        startup_reconcile = reconcile(
            state_root=state_root, wait_for_live=True,
            poll_interval_seconds=min(max(0.1, float(poll_interval_seconds)), 5.0),
        )
        events.append({"status": "STARTUP_RECONCILED", "outcomes": startup_reconcile.get("outcomes")})
        if max_assignments == 0:
            set_runtime_status("STOPPED", state_root=state_root)
            return {
                "status": "STOPPED", "assignments_processed": 0,
                "idle_cycles": 0, "events": events, "runtime": load_runtime(state_root),
            }
        try:
            while True:
                current = load_runtime(state_root)
                if current.get("status") in {"STOP_REQUESTED", "STOPPED"}:
                    break
                if current.get("status") == "PAUSED":
                    time.sleep(max(0.1, float(poll_interval_seconds)))
                    continue
                event = _run_once_unlocked(state_root=state_root, scheduler_id=str(current.get("scheduler_id") or scheduler_id or ""))
                events.append({"status": event.get("status"), "assignment_id": dict(event.get("assignment") or {}).get("record_id")})
                if event.get("status") == "NO_ASSIGNMENT":
                    idle += 1
                    if max_idle_cycles is not None and idle >= max_idle_cycles:
                        break
                    time.sleep(max(0.1, float(poll_interval_seconds)))
                    continue
                idle = 0
                completed += 1
                if max_assignments is not None and completed >= max_assignments:
                    break
        except KeyboardInterrupt:
            events.append({"status": "INTERRUPTED_BY_OPERATOR"})
        finally:
            set_runtime_status("STOPPED", state_root=state_root)
        return {"status": "STOPPED", "assignments_processed": completed, "idle_cycles": idle, "events": events, "runtime": load_runtime(state_root)}


__all__ = [
    "RUNTIME_SCHEMA_VERSION",
    "load_runtime",
    "reconcile",
    "run_once",
    "runtime_path",
    "save_runtime",
    "scheduler_status",
    "set_runtime_status",
    "start_polling",
    "worker_rows",
]
