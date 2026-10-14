"""Historical Workhorse Attempt-projection validation.

Canonical Workhorse execution enters only through Launchpad. Its provider
reopens Scheduler authority directly; this module cannot grant execution.
Legacy persisted projections remain readable for exact historical validation,
while the former caller-supplied Python adapter entrypoint is fail-closed.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
import os
from pathlib import Path
from typing import Any, Final, TypeAlias

from .contract import (
    FullEvolutionCudaWorkhorseContractV2,
    canonical_sha256,
)

WORKHORSE_ATTEMPT_AUTHORITY_REQUEST_SCHEMA_V2: Final[str] = (
    "eiel_workhorse_attempt_authority_request_v2"
)
WORKHORSE_INJECTED_ATTEMPT_AUTHORITY_SCHEMA_V2: Final[str] = (
    "eiel_workhorse_injected_attempt_authority_v2"
)
WORKHORSE_ATTEMPT_AUTHORITY_BINDING_SCHEMA_V2: Final[str] = (
    "eiel_workhorse_attempt_authority_binding_v2"
)
WORKHORSE_ATTEMPT_PHASES_V2: Final[frozenset[str]] = frozenset(
    {"CALIBRATION", "EXECUTION_ROW", "PLAN_REJECTION", "CLOSEOUT"}
)
WORKHORSE_STARTED_PHASES_V2: Final[frozenset[str]] = frozenset(
    {"CALIBRATION", "EXECUTION_ROW", "CLOSEOUT"}
)
WORKHORSE_ACCOUNTING_MODES_V2: Final[frozenset[str]] = frozenset(
    {"NVML_PER_PROCESS", "LEASE_BACKED_GLOBAL_DELTA"}
)
WORKHORSE_MAXIMUM_AUTHORITY_OBSERVATION_AGE_SECONDS_V2: Final[int] = 30


class WorkhorseAttemptAuthorityError(RuntimeError):
    """Raised when the injected outer authority is missing, stale, or foreign."""


# Compatibility-only annotation for the retired injected-adapter ABI.  It is
# deliberately *not* callable: canonical Workhorse execution must enter via
# Launchpad, where the provider reopens Scheduler authority directly.
AttemptAuthorityValidatorV2: TypeAlias = object


def _valid_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _parse_utc(value: object, *, label: str) -> datetime:
    if type(value) is not str or not value.endswith("Z"):
        raise WorkhorseAttemptAuthorityError(f"{label} must be canonical UTC")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise WorkhorseAttemptAuthorityError(f"{label} is invalid UTC") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise WorkhorseAttemptAuthorityError(f"{label} must be UTC")
    return parsed


def current_process_start_ticks_v2() -> int:
    """Return Linux /proc process-start ticks for PID-reuse-safe owner binding."""
    try:
        # comm may contain spaces and parentheses, so split only after its final ')'.
        stat = Path(f"/proc/{os.getpid()}/stat").read_text(encoding="ascii")
        suffix = stat.rsplit(")", 1)[1].strip().split()
        value = int(suffix[19])  # field 22 overall; suffix begins at field 3.
    except (OSError, UnicodeError, ValueError, IndexError) as exc:
        raise WorkhorseAttemptAuthorityError(
            "current process start identity is unavailable"
        ) from exc
    if value <= 0:
        raise WorkhorseAttemptAuthorityError(
            "current process start identity is invalid"
        )
    return value


def workhorse_resource_observation_sha256_v2(
    response: Mapping[str, Any],
) -> str:
    """Return the transparent digest for one exact adapter GPU observation."""
    observation = {
        "schema_version": "eiel_workhorse_resource_observation_v2",
        "phase": response.get("phase"),
        "device_uuid": response.get("device_uuid"),
        "owner_pid": response.get("owner_pid"),
        "owner_process_start_ticks": response.get("owner_process_start_ticks"),
        "visible_compute_pids": response.get("visible_compute_pids"),
        "foreign_compute_pids": response.get("foreign_compute_pids"),
        "accounting_mode": response.get("accounting_mode"),
        "owner_process_bytes": response.get("owner_process_bytes"),
        "global_device_bytes": response.get("global_device_bytes"),
        "device_free_bytes": response.get("device_free_bytes"),
        "device_total_bytes": response.get("device_total_bytes"),
        "observed_at_utc": response.get("observed_at_utc"),
    }
    return canonical_sha256(observation)


def workhorse_attempt_authority_request_v2(
    *,
    phase: str,
    contract: FullEvolutionCudaWorkhorseContractV2,
    artifact_sha256: str,
    sealed_artifact_identity_sha256: str,
    device_uuid: str,
    candidate_envelope_sha256: str | None,
    governing_contracts: Mapping[str, Any],
    prior_lineage_sha256: str | None = None,
    source_checkpoint_identity_sha256: str | None = None,
) -> dict[str, Any]:
    """Create the exact callback request without acquiring any authority."""
    if phase not in WORKHORSE_ATTEMPT_PHASES_V2:
        raise WorkhorseAttemptAuthorityError("unsupported Attempt authority phase")
    if type(contract) is not FullEvolutionCudaWorkhorseContractV2:
        raise WorkhorseAttemptAuthorityError("Attempt request contract is inexact")
    for label, value in (
        ("artifact_sha256", artifact_sha256),
        ("sealed_artifact_identity_sha256", sealed_artifact_identity_sha256),
        ("governing_contracts_sha256", canonical_sha256(dict(governing_contracts))),
    ):
        if not _valid_sha256(value):
            raise WorkhorseAttemptAuthorityError(f"{label} is invalid")
    for label, value in (
        ("candidate_envelope_sha256", candidate_envelope_sha256),
        ("prior_lineage_sha256", prior_lineage_sha256),
        ("source_checkpoint_identity_sha256", source_checkpoint_identity_sha256),
    ):
        if value is not None and not _valid_sha256(value):
            raise WorkhorseAttemptAuthorityError(f"{label} is invalid")
    if type(device_uuid) is not str or not device_uuid.strip():
        raise WorkhorseAttemptAuthorityError("device_uuid is required")
    if phase == "PLAN_REJECTION" and candidate_envelope_sha256 is None:
        raise WorkhorseAttemptAuthorityError(
            "PLAN_REJECTION must bind its sealed candidate envelope"
        )
    if phase == "PLAN_REJECTION" and prior_lineage_sha256 is not None:
        raise WorkhorseAttemptAuthorityError(
            "PLAN_REJECTION cannot refer to a started Attempt lineage"
        )
    if source_checkpoint_identity_sha256 is not None and phase != "EXECUTION_ROW":
        raise WorkhorseAttemptAuthorityError(
            "source checkpoint identity is valid only for an execution continuation"
        )
    if phase == "CLOSEOUT" and prior_lineage_sha256 is None:
        raise WorkhorseAttemptAuthorityError(
            "CLOSEOUT must reopen an exact started Attempt lineage"
        )
    body = {
        "schema_version": WORKHORSE_ATTEMPT_AUTHORITY_REQUEST_SCHEMA_V2,
        "phase": phase,
        "required_effect_state": (
            "PREPARED_NO_EFFECT"
            if phase == "PLAN_REJECTION"
            else "STARTED_LIVE_EXCLUSIVE"
        ),
        "contract": contract.to_dict(),
        "profile_id": contract.profile_id,
        "route_id": contract.route_id,
        "backend_id": contract.backend_id,
        "B": contract.B,
        "temporal_k": contract.temporal_k,
        "artifact_sha256": artifact_sha256,
        "sealed_artifact_identity_sha256": sealed_artifact_identity_sha256,
        "device_uuid": device_uuid,
        "owner_pid": os.getpid(),
        "owner_process_start_ticks": current_process_start_ticks_v2(),
        "candidate_envelope_sha256": candidate_envelope_sha256,
        "governing_contracts_sha256": canonical_sha256(dict(governing_contracts)),
        "prior_lineage_sha256": prior_lineage_sha256,
        "source_checkpoint_identity_sha256": source_checkpoint_identity_sha256,
    }
    return {**body, "authority_request_sha256": canonical_sha256(body)}


_COMMON_RESPONSE_FIELDS = {
    "schema_version",
    "phase",
    "authority_request_sha256",
    "effect_state",
    "scheduler_validation_status",
    "scheduler_authority_source",
    "scheduler_owner_id",
    "scheduler_owner_digest",
    "execution_owner_id",
    "execution_owner_digest",
    "run_id",
    "attempt_id",
    "claim_digest",
    "prepared_claim_path",
    "prepared_claim_file_sha256",
    "authoritative_bundle_path",
    "authoritative_bundle_file_sha256",
    "authoritative_bundle_digest",
    "attempt_start_present",
    "execution_effect_present",
    "execution_authority_granted",
    "resource_start_lease_digest",
    "runtime_resource_lease_digest",
    "runtime_resource_head_digest",
    "runtime_resource_lease_expires_at_utc",
    "continuation_lease_digest",
    "fencing_token_digest",
    "continuation_lease_epoch",
    "continuation_head_digest",
    "lease_expires_at_utc",
    "profile_id",
    "route_id",
    "backend_id",
    "B",
    "temporal_k",
    "artifact_sha256",
    "sealed_artifact_identity_sha256",
    "device_uuid",
    "owner_pid",
    "owner_process_start_ticks",
    "visible_compute_pids",
    "foreign_compute_pids",
    "accounting_mode",
    "owner_process_bytes",
    "global_device_bytes",
    "device_free_bytes",
    "device_total_bytes",
    "resource_observation_digest",
    "observed_at_utc",
    "outer_adapter_id",
    "outer_adapter_source_sha256",
    "response_body_sha256",
}

_REQUEST_FIELDS = {
    "schema_version",
    "phase",
    "required_effect_state",
    "contract",
    "profile_id",
    "route_id",
    "backend_id",
    "B",
    "temporal_k",
    "artifact_sha256",
    "sealed_artifact_identity_sha256",
    "device_uuid",
    "owner_pid",
    "owner_process_start_ticks",
    "candidate_envelope_sha256",
    "governing_contracts_sha256",
    "prior_lineage_sha256",
    "source_checkpoint_identity_sha256",
    "authority_request_sha256",
}


def validate_injected_workhorse_attempt_authority_v2(
    response: Mapping[str, Any],
    *,
    request: Mapping[str, Any],
    now_utc: datetime | None = None,
    enforce_current_process: bool = True,
) -> dict[str, Any]:
    """Validate one live outer-adapter response; never construct Scheduler state."""
    if type(response) is not dict or set(response) != _COMMON_RESPONSE_FIELDS:
        raise WorkhorseAttemptAuthorityError(
            "injected Attempt authority fields are missing or unknown"
        )
    if type(request) is not dict or set(request) != _REQUEST_FIELDS:
        raise WorkhorseAttemptAuthorityError("Attempt authority request is inexact")
    request_body = {
        key: value
        for key, value in request.items()
        if key != "authority_request_sha256"
    }
    try:
        request_contract = FullEvolutionCudaWorkhorseContractV2.from_mapping(
            request["contract"]
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise WorkhorseAttemptAuthorityError(
            "Attempt authority request contract is invalid"
        ) from exc
    request_phase = request.get("phase")
    request_candidate = request.get("candidate_envelope_sha256")
    request_prior = request.get("prior_lineage_sha256")
    request_checkpoint = request.get("source_checkpoint_identity_sha256")
    if (
        request.get("schema_version") != WORKHORSE_ATTEMPT_AUTHORITY_REQUEST_SCHEMA_V2
        or request.get("authority_request_sha256") != canonical_sha256(request_body)
        or request_phase not in WORKHORSE_ATTEMPT_PHASES_V2
        or request.get("required_effect_state")
        != (
            "PREPARED_NO_EFFECT"
            if request_phase == "PLAN_REJECTION"
            else "STARTED_LIVE_EXCLUSIVE"
        )
        or request.get("profile_id") != request_contract.profile_id
        or request.get("route_id") != request_contract.route_id
        or request.get("backend_id") != request_contract.backend_id
        or request.get("B") != request_contract.B
        or request.get("temporal_k") != request_contract.temporal_k
        or not _valid_sha256(request.get("artifact_sha256"))
        or not _valid_sha256(request.get("sealed_artifact_identity_sha256"))
        or not _valid_sha256(request.get("governing_contracts_sha256"))
        or (request_candidate is not None and not _valid_sha256(request_candidate))
        or request_phase == "PLAN_REJECTION"
        and (request_candidate is None or request_prior is not None)
        or request_phase == "CLOSEOUT"
        and not _valid_sha256(request_prior)
        or request_prior is not None
        and not _valid_sha256(request_prior)
        or request_checkpoint is not None
        and (request_phase != "EXECUTION_ROW" or not _valid_sha256(request_checkpoint))
        or type(request.get("device_uuid")) is not str
        or not request["device_uuid"].strip()
        or type(request.get("owner_pid")) is not int
        or request["owner_pid"] <= 0
        or type(request.get("owner_process_start_ticks")) is not int
        or request["owner_process_start_ticks"] <= 0
    ):
        raise WorkhorseAttemptAuthorityError(
            "Attempt authority request identity or seal is invalid"
        )
    body = {
        key: value for key, value in response.items() if key != "response_body_sha256"
    }
    phase = request["phase"]
    if (
        response.get("schema_version") != WORKHORSE_INJECTED_ATTEMPT_AUTHORITY_SCHEMA_V2
        or response.get("phase") != phase
        or response.get("authority_request_sha256")
        != request["authority_request_sha256"]
        or response.get("response_body_sha256") != canonical_sha256(body)
        or response.get("profile_id") != request["profile_id"]
        or response.get("route_id") != request["route_id"]
        or response.get("backend_id") != request["backend_id"]
        or response.get("B") != request["B"]
        or response.get("temporal_k") != request["temporal_k"]
        or response.get("artifact_sha256") != request["artifact_sha256"]
        or response.get("sealed_artifact_identity_sha256")
        != request["sealed_artifact_identity_sha256"]
        or response.get("device_uuid") != request["device_uuid"]
        or response.get("owner_pid") != request["owner_pid"]
        or response.get("owner_process_start_ticks")
        != request["owner_process_start_ticks"]
        or response.get("scheduler_validation_status")
        not in {"AUTHORITATIVE_BUNDLE_VALID", "PREPARED_CLAIM_VALID"}
        or type(response.get("scheduler_owner_id")) is not str
        or not response["scheduler_owner_id"]
        or not _valid_sha256(response.get("scheduler_owner_digest"))
        or type(response.get("execution_owner_id")) is not str
        or not response["execution_owner_id"]
        or not _valid_sha256(response.get("execution_owner_digest"))
        or type(response.get("claim_digest")) is not str
        or not _valid_sha256(response["claim_digest"])
        or type(response.get("outer_adapter_id")) is not str
        or not response["outer_adapter_id"]
        or not _valid_sha256(response.get("outer_adapter_source_sha256"))
        or not _valid_sha256(response.get("resource_observation_digest"))
    ):
        raise WorkhorseAttemptAuthorityError(
            "injected Attempt authority identity/request binding is invalid"
        )
    observed_at = _parse_utc(response.get("observed_at_utc"), label="observed_at_utc")
    current = now_utc or datetime.now(timezone.utc)
    if current.tzinfo is None:
        raise WorkhorseAttemptAuthorityError("now_utc must be timezone-aware")
    visible = response.get("visible_compute_pids")
    foreign = response.get("foreign_compute_pids")
    if type(visible) is not list or type(foreign) is not list:
        raise WorkhorseAttemptAuthorityError(
            "GPU process observations must be exact PID lists"
        )
    if response.get(
        "resource_observation_digest"
    ) != workhorse_resource_observation_sha256_v2(response):
        raise WorkhorseAttemptAuthorityError(
            "GPU resource observation digest does not recompute"
        )
    if enforce_current_process and (
        response.get("owner_pid") != os.getpid()
        or response.get("owner_process_start_ticks") != current_process_start_ticks_v2()
    ):
        raise WorkhorseAttemptAuthorityError(
            "Attempt authority belongs to a different or PID-reused process"
        )

    if phase in WORKHORSE_STARTED_PHASES_V2:
        expiry = _parse_utc(
            response.get("lease_expires_at_utc"), label="lease_expires_at_utc"
        )
        required_hashes = (
            "authoritative_bundle_file_sha256",
            "authoritative_bundle_digest",
            "resource_start_lease_digest",
            "runtime_resource_lease_digest",
            "runtime_resource_head_digest",
            "continuation_lease_digest",
            "fencing_token_digest",
            "continuation_head_digest",
        )
        if (
            response.get("effect_state") != "STARTED_LIVE_EXCLUSIVE"
            or response.get("scheduler_validation_status")
            != "AUTHORITATIVE_BUNDLE_VALID"
            or response.get("scheduler_authority_source") != "SCHEDULER_ISSUED_BUNDLE"
            or response.get("attempt_start_present") is not True
            or response.get("execution_effect_present") is not True
            or response.get("execution_authority_granted") is not True
            or type(response.get("run_id")) is not str
            or not response["run_id"]
            or type(response.get("attempt_id")) is not str
            or not response["attempt_id"]
            or type(response.get("authoritative_bundle_path")) is not str
            or not response["authoritative_bundle_path"]
            or any(not _valid_sha256(response.get(field)) for field in required_hashes)
            or type(response.get("continuation_lease_epoch")) is not int
            or response["continuation_lease_epoch"] < 1
            or expiry <= current
            or _parse_utc(
                response.get("runtime_resource_lease_expires_at_utc"),
                label="runtime_resource_lease_expires_at_utc",
            )
            <= current
            or observed_at > current
            or (current - observed_at).total_seconds()
            > WORKHORSE_MAXIMUM_AUTHORITY_OBSERVATION_AGE_SECONDS_V2
            or visible != [response.get("owner_pid")]
            or foreign
            or response.get("accounting_mode") not in WORKHORSE_ACCOUNTING_MODES_V2
            or type(response.get("owner_process_bytes")) is not int
            or response["owner_process_bytes"] < 0
            or type(response.get("global_device_bytes")) is not int
            or response["global_device_bytes"] < response["owner_process_bytes"]
            or type(response.get("device_free_bytes")) is not int
            or response["device_free_bytes"] < 0
            or type(response.get("device_total_bytes")) is not int
            or response["device_total_bytes"] <= 0
            or response["device_free_bytes"] > response["device_total_bytes"]
            or response["global_device_bytes"] > response["device_total_bytes"]
            or response.get("prepared_claim_path") is not None
            or response.get("prepared_claim_file_sha256") is not None
        ):
            raise WorkhorseAttemptAuthorityError(
                "executed phase lacks a live started exclusive Attempt"
            )
        if (
            response["accounting_mode"] == "NVML_PER_PROCESS"
            and response["owner_process_bytes"] > response["global_device_bytes"]
        ):
            raise WorkhorseAttemptAuthorityError("per-process accounting is invalid")
    else:
        forbidden_effect_fields = (
            "authoritative_bundle_path",
            "authoritative_bundle_file_sha256",
            "authoritative_bundle_digest",
            "resource_start_lease_digest",
            "runtime_resource_lease_digest",
            "runtime_resource_head_digest",
            "runtime_resource_lease_expires_at_utc",
            "continuation_lease_digest",
            "fencing_token_digest",
            "continuation_head_digest",
            "lease_expires_at_utc",
            "accounting_mode",
            "owner_process_bytes",
            "global_device_bytes",
            "device_free_bytes",
            "device_total_bytes",
        )
        if (
            response.get("effect_state") != "PREPARED_NO_EFFECT"
            or response.get("scheduler_validation_status") != "PREPARED_CLAIM_VALID"
            or response.get("scheduler_authority_source")
            != "SCHEDULER_PREPARED_NO_EFFECT"
            or response.get("attempt_start_present") is not False
            or response.get("execution_effect_present") is not False
            or response.get("execution_authority_granted") is not False
            or response.get("run_id") is not None
            or response.get("attempt_id") is not None
            or type(response.get("prepared_claim_path")) is not str
            or not response["prepared_claim_path"]
            or not _valid_sha256(response.get("prepared_claim_file_sha256"))
            or any(response.get(field) is not None for field in forbidden_effect_fields)
            or response.get("continuation_lease_epoch") is not None
            or observed_at > current
            or (current - observed_at).total_seconds()
            > WORKHORSE_MAXIMUM_AUTHORITY_OBSERVATION_AGE_SECONDS_V2
            or visible
            or foreign
        ):
            raise WorkhorseAttemptAuthorityError(
                "rejected plan row must bind PREPARED/no-effect authority"
            )
    return dict(response)


def resolve_injected_workhorse_attempt_authority_v2(
    _retired_adapter: object,
    *,
    request: Mapping[str, Any],
    now_utc: datetime | None = None,
    enforce_current_process: bool = True,
) -> dict[str, Any]:
    """Reject the retired caller-supplied Python authority ABI.

    Kept as a fail-closed compatibility symbol so old diagnostics fail with a
    precise error before importing CUDA, allocating VRAM, or invoking caller
    code.  A canonical provider must reopen the Scheduler Attempt pointer
    directly and pass only its non-authoritative internal execution context to
    the Workhorse row implementation.
    """
    del request, now_utc, enforce_current_process
    raise WorkhorseAttemptAuthorityError(
        "injected Workhorse Attempt adapters are retired; use one canonical "
        "ExecutionRequestV2 and Scheduler Attempt per operation"
    )


def validate_workhorse_attempt_authority_binding_v2(
    binding: Mapping[str, Any],
    *,
    now_utc: datetime | None = None,
    historical: bool = False,
) -> dict[str, Any]:
    """Revalidate a persisted request/response pair and its exact lineage."""
    if type(binding) is not dict or set(binding) != {
        "schema_version",
        "request",
        "response",
        "attempt_lineage_sha256",
        "authority_binding_sha256",
    }:
        raise WorkhorseAttemptAuthorityError(
            "Attempt authority binding fields are missing or unknown"
        )
    body = {
        key: value
        for key, value in binding.items()
        if key != "authority_binding_sha256"
    }
    effective_now = now_utc
    if historical:
        if now_utc is not None:
            raise WorkhorseAttemptAuthorityError(
                "historical validation must use the bound observation time"
            )
        effective_now = _parse_utc(
            binding["response"].get("observed_at_utc"), label="observed_at_utc"
        )
    response = validate_injected_workhorse_attempt_authority_v2(
        binding["response"],
        request=binding["request"],
        now_utc=effective_now,
        enforce_current_process=not historical,
    )
    expected_lineage = (
        None
        if binding["request"].get("phase") == "PLAN_REJECTION"
        else workhorse_attempt_lineage_sha256_v2(response)
    )
    if (
        binding.get("schema_version") != WORKHORSE_ATTEMPT_AUTHORITY_BINDING_SCHEMA_V2
        or binding.get("attempt_lineage_sha256") != expected_lineage
        or binding.get("authority_binding_sha256") != canonical_sha256(body)
    ):
        raise WorkhorseAttemptAuthorityError(
            "Attempt authority binding seal or lineage is invalid"
        )
    return dict(binding)


def workhorse_attempt_lineage_sha256_v2(binding: Mapping[str, Any]) -> str:
    """Return the stable lineage key reopened by closeout/event/checkpoint stages."""
    required = {
        "run_id": binding.get("run_id"),
        "attempt_id": binding.get("attempt_id"),
        "claim_digest": binding.get("claim_digest"),
        "authoritative_bundle_digest": binding.get("authoritative_bundle_digest"),
        "resource_start_lease_digest": binding.get("resource_start_lease_digest"),
        "runtime_resource_lease_digest": binding.get("runtime_resource_lease_digest"),
        "runtime_resource_head_digest": binding.get("runtime_resource_head_digest"),
        "continuation_lease_digest": binding.get("continuation_lease_digest"),
        "fencing_token_digest": binding.get("fencing_token_digest"),
        "continuation_lease_epoch": binding.get("continuation_lease_epoch"),
        "continuation_head_digest": binding.get("continuation_head_digest"),
        "scheduler_owner_id": binding.get("scheduler_owner_id"),
        "scheduler_owner_digest": binding.get("scheduler_owner_digest"),
        "execution_owner_id": binding.get("execution_owner_id"),
        "execution_owner_digest": binding.get("execution_owner_digest"),
        "owner_pid": binding.get("owner_pid"),
        "owner_process_start_ticks": binding.get("owner_process_start_ticks"),
        "device_uuid": binding.get("device_uuid"),
    }
    if any(value is None for value in required.values()):
        raise WorkhorseAttemptAuthorityError("Attempt lineage is incomplete")
    return canonical_sha256(required)


WORKHORSE_OUTER_ADAPTER_REQUIREMENTS_V2: Final[tuple[str, ...]] = (
    "reopen Scheduler PREPARED claim or authoritative started bundle",
    "never start an Attempt for PLAN_REJECTION",
    "bind exact canonical request and response hashes plus profile/route/B/K/artifact/device",
    "reopen current resource and continuation lease heads and expiry",
    "bind Scheduler and execution owner identities plus PID/process-start identity",
    "observe sole compute PID and reject foreign PIDs on the exact device UUID",
    "report NVML per-process bytes or lease-backed exclusive global bytes",
    "report live device free/total bytes in that same exclusive observation",
    "revalidate the exact started lineage at CLOSEOUT",
    "reopen the exact source checkpoint identity for restore continuation requests",
)


__all__ = [
    "WORKHORSE_ACCOUNTING_MODES_V2",
    "WORKHORSE_ATTEMPT_AUTHORITY_REQUEST_SCHEMA_V2",
    "WORKHORSE_ATTEMPT_AUTHORITY_BINDING_SCHEMA_V2",
    "WORKHORSE_ATTEMPT_PHASES_V2",
    "WORKHORSE_INJECTED_ATTEMPT_AUTHORITY_SCHEMA_V2",
    "WORKHORSE_MAXIMUM_AUTHORITY_OBSERVATION_AGE_SECONDS_V2",
    "WORKHORSE_OUTER_ADAPTER_REQUIREMENTS_V2",
    "WORKHORSE_STARTED_PHASES_V2",
    "WorkhorseAttemptAuthorityError",
    "current_process_start_ticks_v2",
    "validate_injected_workhorse_attempt_authority_v2",
    "validate_workhorse_attempt_authority_binding_v2",
    "workhorse_attempt_authority_request_v2",
    "workhorse_attempt_lineage_sha256_v2",
    "workhorse_resource_observation_sha256_v2",
]
