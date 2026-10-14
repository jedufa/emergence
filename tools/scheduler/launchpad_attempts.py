"""Scheduler-owned Launchpad Attempt and lease issuance.

This is the authority-bearing half of the additive Launchpad cutover.  It
does not execute a provider.  Instead, ``prepare`` reserves an exact start
window without creating an Attempt, and ``start`` creates the first durable
execution-authority effect plus the CRCA AttemptStart bundle under the same
global Scheduler lock and resource-start ledger used by Worker Assignments.

The module deliberately consumes exact CRCA wire records rather than importing
the composition layer.  The inverse, pure parser/validator lives in
``eiel_composition.launchpad_attempt_authority`` and cannot grant authority.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping

from tools.scheduler.contracts import (
    ActorError,
    atomic_write_json,
    canonical_bytes,
    canonical_host_actor_state_root,
    read_json,
    scheduler_root,
)
from tools.scheduler.programme_contracts import (
    ContentDigest,
    GpuResourceTopologyError,
    LaunchpadAttemptClaimV1,
    LaunchpadResourceStartLeaseV1,
    ProgrammeContractError,
    ResourceStartLeaseEventCodeV1,
    ResourceStartLeaseEventV1,
    ResourceStartLeaseEventV2,
    ResourceStartLeaseEventV3,
    canonical_sha256,
    gpu_uuid_equal_v1,
    resource_ids_conflict_v1,
    shared_gpu_lane_candidate_sets_v1,
    validate_shared_w32_gpu_policy_binding_v1,
)
from tools.scheduler.programme_driver import ResourceStartLeaseLedgerV1
from tools.scheduler.launcher import (
    process_start_identity as observe_process_start_identity,
)
from tools.scheduler.runtime import _scheduler_execution_lock


PREPARATION_SCHEMA_VERSION = "crca_launchpad_attempt_preparation_v1"
PREPARATION_SCHEMA_VERSION_V2 = "crca_launchpad_attempt_preparation_v2"
START_EFFECT_SCHEMA_VERSION = "crca_launchpad_attempt_start_effect_v1"
START_EFFECT_SCHEMA_VERSION_V2 = "crca_launchpad_attempt_start_effect_v2"
AUTHORITATIVE_BUNDLE_SCHEMA_VERSION = "crca_authoritative_launchpad_attempt_bundle_v1"
AUTHORITATIVE_BUNDLE_SCHEMA_VERSION_V2 = (
    "crca_authoritative_launchpad_attempt_bundle_v2"
)
CONTINUATION_HEAD_SCHEMA_VERSION = "crca_launchpad_continuation_head_v1"
TERMINAL_BUNDLE_SCHEMA_VERSION = "crca_launchpad_attempt_terminal_bundle_v1"
QUARANTINE_SCHEMA_VERSION = "crca_launchpad_attempt_quarantine_v1"
CONTINUATION_LEASE_SCHEMA_VERSION = "crca_canonical_continuation_lease_v1"
CONTINUATION_EVENT_SCHEMA_VERSION = "crca_continuation_lease_event_v1"
CONTINUATION_EVENT_SCHEMA_VERSION_V2 = "crca_continuation_lease_event_v2"
TERMINAL_INTENT_SCHEMA_VERSION = "crca_launchpad_attempt_terminal_intent_v1"
TERMINAL_INTENT_SCHEMA_VERSION_V2 = "crca_launchpad_attempt_terminal_intent_v2"
LEGACY_GPU_MIGRATION_MANIFEST_SCHEMA_VERSION = (
    "crca_pidless_gpu_started_migration_manifest_v1"
)
LEGACY_GPU_MIGRATION_AUTHORIZATION_SCHEMA_VERSION = (
    "crca_legacy_gpu_migration_authorization_v1"
)
# Replaced only by the canonical seven-row manifest generated from the sealed
# host authority records.  Non-canonical test roots may exercise other exact
# manifests through ``allow_test_shared_gpu_state_root``.
CANONICAL_LEGACY_GPU_MIGRATION_MANIFEST_SHA256 = (
    "99a852711807c063f44a382507ee72891d1f820706c9029e20253dbad91fda98"
)

_CONTINUATION_EVENT_CODES = frozenset(
    {
        "HEARTBEAT",
        "RENEWED",
        "RELEASED",
        "REASSIGNED",
        "EXPIRED",
        "RECLAIMED",
        "PUBLISH_COMMITTED",
        "OWNER_DEAD",
    }
)
_CONTINUATION_TRANSITION_CODES = frozenset({"RENEWED", "REASSIGNED", "RECLAIMED"})
_TERMINAL_STATUSES = frozenset({"COMPLETED", "INTERRUPTED", "FAILED", "QUARANTINED"})
_FRONTIER_FIELDS = (
    "last_dispatched_next_tick",
    "last_computed_next_tick",
    "durable_checkpoint_next_tick",
    "published_artifact_next_tick",
)


class _ContinuationStateV1(str, Enum):
    ACTIVE = "ACTIVE"
    RELEASED = "RELEASED"
    EXPIRED = "EXPIRED"
    PUBLISHED = "PUBLISHED"
    OWNER_DEAD = "OWNER_DEAD"


def _parse_utc(value: object, *, field: str) -> datetime:
    if not isinstance(value, str) or len(value) != 20 or not value.endswith("Z"):
        raise ActorError(
            f"{field} must use canonical second-resolution UTC",
            code="launchpad_attempt_invalid",
        )
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc
        )
    except ValueError as error:
        raise ActorError(
            f"{field} is invalid", code="launchpad_attempt_invalid"
        ) from error


def _add_seconds(value: str, seconds: int) -> str:
    if isinstance(seconds, bool) or not isinstance(seconds, int) or seconds < 1:
        raise ActorError(
            "lease duration must be positive", code="launchpad_attempt_invalid"
        )
    return (
        (_parse_utc(value, field="at_utc") + timedelta(seconds=seconds))
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _gpu_resource_kind_v1(
    resource_ids: tuple[str, ...], *, device_id: str
) -> str | None:
    """Classify only the canonical exclusive parent or one shared GPU lane."""

    if resource_ids == (f"gpu:{device_id}",):
        return "exclusive"
    if resource_ids in shared_gpu_lane_candidate_sets_v1(device_id):
        return "shared"
    return None


def _require_exact_gpu_process_owner_v2(
    *,
    bundle: Mapping[str, Any],
    resource: LaunchpadResourceStartLeaseV1,
    device_id: str,
    process_id: int | None,
    process_start_identity: str | None,
    action: str,
    allow_durable_intent_recovery: bool = False,
) -> tuple[int | None, str | None]:
    """Bind every live GPU-owner mutation to its immutable Linux process."""

    if _gpu_resource_kind_v1(resource.resource_ids, device_id=device_id) is None:
        return process_id, process_start_identity
    effect = dict(bundle.get("start_effect") or {})
    if allow_durable_intent_recovery:
        if effect.get("schema_version") == START_EFFECT_SCHEMA_VERSION:
            return None, None
        if (
            effect.get("schema_version") == START_EFFECT_SCHEMA_VERSION_V2
            and type(effect.get("process_id")) is int
            and isinstance(effect.get("process_start_identity"), str)
            and str(effect["process_start_identity"]).startswith("linux-proc:")
            and process_id == effect["process_id"]
            and process_start_identity == effect["process_start_identity"]
        ):
            return int(effect["process_id"]), str(effect["process_start_identity"])
        raise ActorError(
            f"{action} durable intent differs from the started GPU owner",
            code="launchpad_attempt_owner_mismatch",
        )
    current_pid = os.getpid()
    current_start_identity = observe_process_start_identity(current_pid)
    if (
        effect.get("schema_version") != START_EFFECT_SCHEMA_VERSION_V2
        or current_start_identity is None
        or effect.get("process_id") != current_pid
        or effect.get("process_start_identity") != current_start_identity
        or (process_id is not None and process_id != current_pid)
        or (
            process_start_identity is not None
            and process_start_identity != current_start_identity
        )
    ):
        raise ActorError(
            f"{action} requires the exact started GPU-owner process identity",
            code="launchpad_attempt_owner_mismatch",
        )
    return current_pid, current_start_identity


def _digest(value: object, *, field: str) -> ContentDigest:
    if isinstance(value, ContentDigest):
        return value
    if not isinstance(value, Mapping):
        raise ActorError(
            f"{field} must be a SHA-256 digest", code="launchpad_attempt_invalid"
        )
    try:
        return ContentDigest(
            algorithm=str(value.get("algorithm") or ""),
            hex=str(value.get("hex") or ""),
        )
    except ProgrammeContractError as error:
        raise ActorError(
            f"{field} is not an exact SHA-256 digest",
            code="launchpad_attempt_invalid",
        ) from error


def _optional_digest(value: object, *, field: str) -> ContentDigest | None:
    return None if value is None else _digest(value, field=field)


def _require_exact_fields(
    raw: Mapping[str, Any], expected: set[str], *, label: str
) -> None:
    observed = set(raw)
    if observed != expected:
        raise ActorError(
            f"{label} has missing or unknown fields",
            code="launchpad_attempt_invalid",
            problems=[
                f"missing={sorted(expected - observed)}",
                f"unknown={sorted(observed - expected)}",
            ],
        )


def _same_payload(path: Path, payload: Mapping[str, Any]) -> bool:
    if not path.is_file() or path.is_symlink():
        return False
    try:
        return canonical_bytes(read_json(path, name=path.name)) == canonical_bytes(
            dict(payload)
        )
    except ActorError:
        return False


def _write_immutable(path: Path, payload: Mapping[str, Any], *, label: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if _same_payload(path, payload):
            return path
        raise ActorError(
            f"conflicting immutable {label}",
            code="launchpad_attempt_immutable_conflict",
            problems=[str(path)],
        )
    atomic_write_json(path, dict(payload))
    return path


def _safe_key(value: str) -> str:
    return canonical_sha256({"value": value}).hex


def _record_digest(raw: Mapping[str, Any]) -> ContentDigest:
    return canonical_sha256(dict(raw))


def _continuation_lease_digest(raw: Mapping[str, Any]) -> ContentDigest:
    _require_exact_fields(
        raw,
        {
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
        },
        label="CanonicalContinuationLeaseV1",
    )
    if raw.get("schema_version") != CONTINUATION_LEASE_SCHEMA_VERSION:
        raise ActorError(
            "unsupported continuation lease schema",
            code="launchpad_attempt_invalid",
        )
    for field in ("lease_id", "run_id", "attempt_id", "owner_id"):
        if not isinstance(raw.get(field), str) or not str(raw[field]).strip():
            raise ActorError(
                f"continuation lease {field} is required",
                code="launchpad_attempt_invalid",
            )
    if (
        isinstance(raw.get("lease_epoch"), bool)
        or not isinstance(raw.get("lease_epoch"), int)
        or raw["lease_epoch"] < 1
    ):
        raise ActorError(
            "continuation lease epoch must be positive",
            code="launchpad_attempt_invalid",
        )
    _digest(raw.get("lineage_scope_digest"), field="lineage_scope_digest")
    _digest(raw.get("fencing_token_digest"), field="fencing_token_digest")
    _optional_digest(
        raw.get("restore_checkpoint_digest"), field="restore_checkpoint_digest"
    )
    _optional_digest(raw.get("prior_lease_digest"), field="prior_lease_digest")
    if _parse_utc(raw.get("expires_at_utc"), field="expires_at_utc") <= _parse_utc(
        raw.get("granted_at_utc"), field="granted_at_utc"
    ):
        raise ActorError(
            "continuation lease interval is invalid",
            code="launchpad_attempt_invalid",
        )
    return _record_digest(raw)


def _continuation_event_digest(raw: Mapping[str, Any]) -> ContentDigest:
    schema = raw.get("schema_version")
    expected_fields = {
        "schema_version",
        "lease_digest",
        "run_id",
        "attempt_id",
        "owner_id",
        "lease_epoch",
        "fencing_token_digest",
        "code",
        "event_at_utc",
        "reason",
        "prior_event_digest",
        "successor_lease_digest",
        "publication_digest",
    }
    if schema == CONTINUATION_EVENT_SCHEMA_VERSION_V2:
        expected_fields.add("owner_dead_proof_digest")
    _require_exact_fields(
        raw,
        expected_fields,
        label="ContinuationLeaseEventV1",
    )
    if schema not in {
        CONTINUATION_EVENT_SCHEMA_VERSION,
        CONTINUATION_EVENT_SCHEMA_VERSION_V2,
    }:
        raise ActorError(
            "unsupported continuation event schema",
            code="launchpad_attempt_invalid",
        )
    for field in ("run_id", "attempt_id", "owner_id", "reason"):
        if not isinstance(raw.get(field), str) or not str(raw[field]).strip():
            raise ActorError(
                f"continuation event {field} is required",
                code="launchpad_attempt_invalid",
            )
    if (
        isinstance(raw.get("lease_epoch"), bool)
        or not isinstance(raw.get("lease_epoch"), int)
        or raw["lease_epoch"] < 1
    ):
        raise ActorError(
            "continuation event epoch must be positive",
            code="launchpad_attempt_invalid",
        )
    code = raw.get("code")
    if code not in _CONTINUATION_EVENT_CODES:
        raise ActorError(
            "continuation event code is invalid",
            code="launchpad_attempt_invalid",
        )
    for field in ("lease_digest", "fencing_token_digest"):
        _digest(raw.get(field), field=field)
    for field in (
        "prior_event_digest",
        "successor_lease_digest",
        "publication_digest",
    ):
        _optional_digest(raw.get(field), field=field)
    successor_required = code in _CONTINUATION_TRANSITION_CODES
    if successor_required != (raw.get("successor_lease_digest") is not None):
        raise ActorError(
            "continuation event successor binding is invalid",
            code="launchpad_attempt_invalid",
        )
    publication_required = code == "PUBLISH_COMMITTED"
    if publication_required != (raw.get("publication_digest") is not None):
        raise ActorError(
            "continuation event publication binding is invalid",
            code="launchpad_attempt_invalid",
        )
    owner_dead_proof = raw.get("owner_dead_proof_digest")
    if schema == CONTINUATION_EVENT_SCHEMA_VERSION_V2:
        if code != "OWNER_DEAD" or owner_dead_proof is None:
            raise ActorError(
                "V2 continuation event is not an exact dead-owner fence",
                code="launchpad_attempt_invalid",
            )
        _digest(owner_dead_proof, field="owner_dead_proof_digest")
    elif code == "OWNER_DEAD":
        raise ActorError(
            "dead-owner continuation fence requires a proof-bound V2 event",
            code="launchpad_attempt_invalid",
        )
    _parse_utc(raw.get("event_at_utc"), field="event_at_utc")
    return _record_digest(raw)


def _continuation_ledger_state(
    leases: list[dict[str, Any]], events: list[dict[str, Any]]
) -> _ContinuationStateV1:
    """Validate the canonical typed continuation state without importing CRCA."""

    if not leases:
        raise ActorError(
            "continuation ledger requires an initial lease",
            code="launchpad_attempt_invalid",
        )
    lease_digests = [_continuation_lease_digest(lease) for lease in leases]
    if len(set(lease_digests)) != len(lease_digests):
        raise ActorError(
            "continuation ledger repeats a lease",
            code="launchpad_attempt_invalid",
        )
    first = leases[0]
    if first["lease_epoch"] != 1 or first.get("prior_lease_digest") is not None:
        raise ActorError(
            "initial continuation lease has an invalid predecessor",
            code="launchpad_attempt_invalid",
        )
    current = first
    current_index = 0
    state = _ContinuationStateV1.ACTIVE
    prior_event_digest: ContentDigest | None = None
    prior_event_at: datetime | None = None
    publication_count = 0
    for event in events:
        event_digest = _continuation_event_digest(event)
        code = str(event["code"])
        if (
            _optional_digest(
                event.get("prior_event_digest"), field="prior_event_digest"
            )
            != prior_event_digest
        ):
            raise ActorError(
                "continuation events are not one digest chain",
                code="launchpad_attempt_invalid",
            )
        if state is _ContinuationStateV1.PUBLISHED:
            raise ActorError(
                "continuation event follows canonical publication",
                code="launchpad_attempt_invalid",
            )
        if state is _ContinuationStateV1.RELEASED and code != "REASSIGNED":
            raise ActorError(
                "released continuation permits only reassignment",
                code="launchpad_attempt_invalid",
            )
        if state is _ContinuationStateV1.EXPIRED and code != "RECLAIMED":
            raise ActorError(
                "expired continuation permits only reclaim",
                code="launchpad_attempt_invalid",
            )
        if state is _ContinuationStateV1.OWNER_DEAD and code != "RECLAIMED":
            raise ActorError(
                "dead-owner continuation permits only reclaim",
                code="launchpad_attempt_invalid",
            )
        if code == "REASSIGNED" and state is not _ContinuationStateV1.RELEASED:
            raise ActorError(
                "continuation reassignment lacks release",
                code="launchpad_attempt_invalid",
            )
        if code == "RECLAIMED" and state not in {
            _ContinuationStateV1.EXPIRED,
            _ContinuationStateV1.OWNER_DEAD,
        }:
            raise ActorError(
                "continuation reclaim lacks expiry",
                code="launchpad_attempt_invalid",
            )
        if code == "RENEWED" and state is not _ContinuationStateV1.ACTIVE:
            raise ActorError(
                "continuation renewal requires ACTIVE state",
                code="launchpad_attempt_invalid",
            )
        if (
            _digest(event.get("lease_digest"), field="lease_digest")
            != _continuation_lease_digest(current)
            or event.get("run_id") != current.get("run_id")
            or event.get("attempt_id") != current.get("attempt_id")
            or event.get("owner_id") != current.get("owner_id")
            or event.get("lease_epoch") != current.get("lease_epoch")
            or event.get("fencing_token_digest") != current.get("fencing_token_digest")
        ):
            raise ActorError(
                "continuation event differs from its fenced lease",
                code="launchpad_attempt_invalid",
            )
        event_at = _parse_utc(event["event_at_utc"], field="event_at_utc")
        granted_at = _parse_utc(current["granted_at_utc"], field="granted_at_utc")
        expires_at = _parse_utc(current["expires_at_utc"], field="expires_at_utc")
        stale_event = code == "EXPIRED" or (
            code == "RECLAIMED" and state is _ContinuationStateV1.EXPIRED
        )
        if stale_event and event_at < expires_at:
            raise ActorError(
                "continuation expiry or reclaim predates its fence",
                code="launchpad_attempt_invalid",
            )
        if code == "REASSIGNED" and event_at < granted_at:
            raise ActorError(
                "continuation reassignment predates its grant",
                code="launchpad_attempt_invalid",
            )
        if (
            not stale_event
            and code not in {"REASSIGNED", "RECLAIMED", "OWNER_DEAD"}
            and not (granted_at <= event_at < expires_at)
        ):
            raise ActorError(
                "live continuation event lies outside its fence",
                code="launchpad_attempt_invalid",
            )
        if code == "OWNER_DEAD" and event_at < granted_at:
            raise ActorError(
                "dead-owner continuation fence predates its grant",
                code="launchpad_attempt_invalid",
            )
        if prior_event_at is not None and event_at < prior_event_at:
            raise ActorError(
                "continuation event time moved backwards",
                code="launchpad_attempt_invalid",
            )
        if code == "PUBLISH_COMMITTED":
            publication_count += 1
            if publication_count > 1:
                raise ActorError(
                    "continuation lineage already has a publisher",
                    code="launchpad_attempt_invalid",
                )
        if code in _CONTINUATION_TRANSITION_CODES:
            current_index += 1
            if current_index >= len(leases):
                raise ActorError(
                    "continuation transition has no successor",
                    code="launchpad_attempt_invalid",
                )
            successor = leases[current_index]
            successor_digest = _continuation_lease_digest(successor)
            if (
                _digest(
                    event.get("successor_lease_digest"),
                    field="successor_lease_digest",
                )
                != successor_digest
                or _optional_digest(
                    successor.get("prior_lease_digest"),
                    field="prior_lease_digest",
                )
                != _continuation_lease_digest(current)
                or successor["lease_epoch"] != current["lease_epoch"] + 1
                or successor["lease_id"] == current["lease_id"]
                or successor["fencing_token_digest"] == current["fencing_token_digest"]
                or successor["run_id"] != current["run_id"]
                or successor["lineage_scope_digest"] != current["lineage_scope_digest"]
                or successor["granted_at_utc"] != event["event_at_utc"]
            ):
                raise ActorError(
                    "continuation successor breaks its canonical transition",
                    code="launchpad_attempt_invalid",
                )
            if code == "RENEWED" and (
                successor["attempt_id"] != current["attempt_id"]
                or successor["owner_id"] != current["owner_id"]
                or successor["restore_checkpoint_digest"]
                != current["restore_checkpoint_digest"]
            ):
                raise ActorError(
                    "continuation renewal changes Attempt or owner",
                    code="launchpad_attempt_invalid",
                )
            current = successor
            state = _ContinuationStateV1.ACTIVE
        elif code == "RELEASED":
            state = _ContinuationStateV1.RELEASED
        elif code == "EXPIRED":
            state = _ContinuationStateV1.EXPIRED
        elif code == "PUBLISH_COMMITTED":
            state = _ContinuationStateV1.PUBLISHED
        elif code == "OWNER_DEAD":
            state = _ContinuationStateV1.OWNER_DEAD
        prior_event_digest = event_digest
        prior_event_at = event_at
    if current_index != len(leases) - 1:
        raise ActorError(
            "continuation lease lacks a transition event",
            code="launchpad_attempt_invalid",
        )
    return state


def _continuation_event(
    lease: Mapping[str, Any],
    *,
    code: str,
    event_at_utc: str,
    reason: str,
    prior_event_digest: ContentDigest | None,
    successor_lease_digest: ContentDigest | None = None,
    publication_digest: ContentDigest | None = None,
    owner_dead_proof_digest: ContentDigest | None = None,
) -> dict[str, Any]:
    event = {
        "schema_version": (
            CONTINUATION_EVENT_SCHEMA_VERSION_V2
            if owner_dead_proof_digest is not None
            else CONTINUATION_EVENT_SCHEMA_VERSION
        ),
        "lease_digest": _continuation_lease_digest(lease).to_dict(),
        "run_id": lease["run_id"],
        "attempt_id": lease["attempt_id"],
        "owner_id": lease["owner_id"],
        "lease_epoch": lease["lease_epoch"],
        "fencing_token_digest": lease["fencing_token_digest"],
        "code": code,
        "event_at_utc": event_at_utc,
        "reason": reason,
        "prior_event_digest": (
            None if prior_event_digest is None else prior_event_digest.to_dict()
        ),
        "successor_lease_digest": (
            None if successor_lease_digest is None else successor_lease_digest.to_dict()
        ),
        "publication_digest": (
            None if publication_digest is None else publication_digest.to_dict()
        ),
    }
    if owner_dead_proof_digest is not None:
        event["owner_dead_proof_digest"] = owner_dead_proof_digest.to_dict()
    return event


def _attempt_frontiers_v1(raw: object, *, label: str) -> dict[str, int | str]:
    if not isinstance(raw, Mapping):
        raise ActorError(
            f"{label} must be an exact AttemptFrontiersV1",
            code="launchpad_attempt_invalid",
        )
    _require_exact_fields(
        raw,
        {"schema_version", *_FRONTIER_FIELDS},
        label=label,
    )
    if raw.get("schema_version") != "crca_attempt_frontiers_v1":
        raise ActorError(
            f"{label} schema is invalid",
            code="launchpad_attempt_invalid",
        )
    values: dict[str, int | str] = {"schema_version": "crca_attempt_frontiers_v1"}
    for field in _FRONTIER_FIELDS:
        value = raw.get(field)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ActorError(
                f"{label}.{field} must be an absolute integer tick",
                code="launchpad_attempt_invalid",
            )
        values[field] = value
    if values["last_computed_next_tick"] > values["last_dispatched_next_tick"]:
        raise ActorError(
            f"{label} computed frontier exceeds dispatched frontier",
            code="launchpad_attempt_invalid",
        )
    if values["durable_checkpoint_next_tick"] > values["last_computed_next_tick"]:
        raise ActorError(
            f"{label} durable frontier exceeds computed frontier",
            code="launchpad_attempt_invalid",
        )
    if values["published_artifact_next_tick"] > values["last_computed_next_tick"]:
        raise ActorError(
            f"{label} published frontier exceeds computed frontier",
            code="launchpad_attempt_invalid",
        )
    if dict(raw) != values:
        raise ActorError(
            f"{label} is not canonical",
            code="launchpad_attempt_invalid",
        )
    return values


def _attempt_terminal_v1(
    raw: Mapping[str, Any],
    *,
    attempt_id: str,
    start: Mapping[str, Any],
    run: Mapping[str, Any],
    semantic_spec: Mapping[str, Any],
) -> dict[str, Any]:
    """Round-trip the exact CRCA terminal wire law at the Scheduler boundary."""

    _require_exact_fields(
        raw,
        {
            "schema_version",
            "attempt_id",
            "attempt_start_digest",
            "status",
            "final_frontiers",
            "result_digest",
            "reason",
        },
        label="AttemptTerminalV1",
    )
    if raw.get("schema_version") != "crca_attempt_terminal_v1":
        raise ActorError(
            "Attempt terminal schema is invalid",
            code="launchpad_attempt_invalid",
        )
    if raw.get("attempt_id") != attempt_id or start.get("attempt_id") != attempt_id:
        raise ActorError(
            "terminal Attempt differs from authoritative start",
            code="launchpad_attempt_binding_mismatch",
        )
    start_digest = _record_digest(start)
    attempt_start_digest = _digest(
        raw.get("attempt_start_digest"), field="attempt_start_digest"
    )
    if attempt_start_digest != start_digest:
        raise ActorError(
            "terminal receipt differs from authoritative start",
            code="launchpad_attempt_binding_mismatch",
        )
    status = raw.get("status")
    if status not in _TERMINAL_STATUSES:
        raise ActorError(
            "Attempt terminal status is invalid",
            code="launchpad_attempt_invalid",
        )
    result_digest = _digest(raw.get("result_digest"), field="result_digest")
    reason = raw.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        raise ActorError(
            "Attempt terminal reason is required",
            code="launchpad_attempt_invalid",
        )
    initial = _attempt_frontiers_v1(
        start.get("initial_frontiers"), label="AttemptStartV1.initial_frontiers"
    )
    final = _attempt_frontiers_v1(
        raw.get("final_frontiers"), label="AttemptTerminalV1.final_frontiers"
    )
    if start.get("run_id") != run.get("run_id"):
        raise ActorError(
            "terminal Run differs from authoritative Attempt",
            code="launchpad_attempt_binding_mismatch",
        )
    if _record_digest(semantic_spec) != _digest(
        run.get("semantic_spec_digest"), field="semantic_spec_digest"
    ):
        raise ActorError(
            "terminal semantic specification differs from Run",
            code="launchpad_attempt_binding_mismatch",
        )
    start_tick = semantic_spec.get("absolute_start_next_tick")
    target_tick = semantic_spec.get("absolute_target_next_tick")
    if (
        isinstance(start_tick, bool)
        or not isinstance(start_tick, int)
        or isinstance(target_tick, bool)
        or not isinstance(target_tick, int)
        or target_tick <= start_tick
    ):
        raise ActorError(
            "terminal semantic absolute tick span is invalid",
            code="launchpad_attempt_invalid",
        )
    for field in _FRONTIER_FIELDS:
        if final[field] < initial[field]:
            raise ActorError(
                f"terminal frontier moved backwards: {field}",
                code="launchpad_attempt_invalid",
            )
        if final[field] > target_tick:
            raise ActorError(
                f"terminal frontier exceeds absolute Run target: {field}",
                code="launchpad_attempt_invalid",
            )
    if status == "COMPLETED" and (
        final["last_dispatched_next_tick"] != target_tick
        or final["last_computed_next_tick"] != target_tick
    ):
        raise ActorError(
            "completed Attempt has not reached the absolute Run target",
            code="launchpad_attempt_invalid",
        )
    canonical = {
        "schema_version": "crca_attempt_terminal_v1",
        "attempt_id": attempt_id,
        "attempt_start_digest": attempt_start_digest.to_dict(),
        "status": status,
        "final_frontiers": final,
        "result_digest": result_digest.to_dict(),
        "reason": reason,
    }
    if dict(raw) != canonical:
        raise ActorError(
            "Attempt terminal is not canonical",
            code="launchpad_attempt_invalid",
        )
    return canonical


def _validate_external_restore_source_record_v2(
    preparation: Mapping[str, Any],
) -> None:
    """Fail before start unless a child Run carries one exact source Attempt."""

    source = preparation.get("restore_source_attempt_record")
    if not isinstance(source, Mapping):
        raise ActorError(
            "V2 child restore lacks an external source Attempt record",
            code="launchpad_attempt_binding_mismatch",
        )
    _require_exact_fields(
        source,
        {"start", "concretized_execution", "terminal"},
        label="restore_source_attempt_record",
    )
    start = source.get("start")
    concretized = source.get("concretized_execution")
    terminal = source.get("terminal")
    if not all(isinstance(value, Mapping) for value in (start, concretized, terminal)):
        raise ActorError(
            "external restore source record is incomplete",
            code="launchpad_attempt_binding_mismatch",
        )
    _require_exact_fields(
        start,
        {
            "schema_version",
            "attempt_id",
            "run_id",
            "monotonic_ordinal",
            "nonce",
            "concretized_execution_digest",
            "implementation_digest",
            "attempt_admission_request_digest",
            "scheduler_assignment_lease_digest",
            "resource_start_lease_digest",
            "post_admission_attempt_plan_digest",
            "admission_disposition_digest",
            "programme_work_session_assignment_dispatch_binding_digest",
            "parent_attempt_id",
            "restore_checkpoint_digest",
            "canonical_continuation_lease_digest",
            "owner_binding_digest",
            "execution_side_effect_digest",
            "initial_frontiers",
        },
        label="restore source AttemptStartV1",
    )
    _require_exact_fields(
        concretized,
        {
            "schema_version",
            "semantic_spec_digest",
            "implementation_digest",
            "admission_assurance_digest",
            "lifecycle_digest",
            "attempt_admission_request_digest",
            "campaign_plan_digest",
            "bound_execution_request_digest",
            "run_contract_digest",
            "baseline_digest",
            "route_contract_digest",
            "assembly_digest",
            "qualification_key_digest",
        },
        label="restore source ConcretizedExecutionV1",
    )
    _require_exact_fields(
        terminal,
        {
            "schema_version",
            "attempt_id",
            "attempt_start_digest",
            "status",
            "final_frontiers",
            "result_digest",
            "reason",
        },
        label="restore source AttemptTerminalV1",
    )
    if (
        start.get("schema_version") != "crca_attempt_start_v1"
        or concretized.get("schema_version") != "crca_concretized_execution_v1"
        or terminal.get("schema_version") != "crca_attempt_terminal_v1"
        or start.get("concretized_execution_digest")
        != _record_digest(concretized).to_dict()
        or terminal.get("attempt_id") != start.get("attempt_id")
        or terminal.get("attempt_start_digest") != _record_digest(start).to_dict()
        or terminal.get("status") not in _TERMINAL_STATUSES
    ):
        raise ActorError(
            "external restore source record digest chain differs",
            code="launchpad_attempt_binding_mismatch",
        )
    source_final = _attempt_frontiers_v1(
        terminal.get("final_frontiers"),
        label="restore source final frontiers",
    )
    restore = preparation.get("restore_authorization")
    if not isinstance(restore, Mapping):
        raise ActorError(
            "V2 child restore lacks checkpoint authorization",
            code="launchpad_attempt_binding_mismatch",
        )
    _require_exact_fields(
        restore,
        {
            "schema_version",
            "checkpoint_digest",
            "run_id",
            "source_attempt_id",
            "source_attempt_terminal_digest",
            "absolute_next_tick",
            "sealed_bundle_digest",
            "completion_marker_digest",
        },
        label="RestoreCheckpointAuthorizationV1",
    )
    run = preparation.get("run_occurrence")
    admission = preparation.get("attempt_admission_request")
    retry = preparation.get("retry_basis")
    initial = _attempt_frontiers_v1(
        preparation.get("initial_frontiers"),
        label="child restore initial frontiers",
    )
    if not all(isinstance(value, Mapping) for value in (run, admission, retry)):
        raise ActorError(
            "V2 child restore projection is incomplete",
            code="launchpad_attempt_binding_mismatch",
        )
    tick = restore.get("absolute_next_tick")
    checkpoint = restore.get("checkpoint_digest")
    if (
        restore.get("schema_version") != "crca_restore_checkpoint_authorization_v1"
        or restore.get("run_id") != start.get("run_id")
        or restore.get("source_attempt_id") != start.get("attempt_id")
        or restore.get("source_attempt_terminal_digest")
        != _record_digest(terminal).to_dict()
        or isinstance(tick, bool)
        or not isinstance(tick, int)
        or tick < 0
        or source_final["durable_checkpoint_next_tick"] < tick
        or run.get("parent_checkpoint_digest") != checkpoint
        or run.get("run_id") == restore.get("run_id")
        or admission.get("requested_restore_checkpoint_digest") != checkpoint
        or preparation.get("monotonic_ordinal") != 1
        or preparation.get("parent_attempt_id") is not None
        or preparation.get("prior_attempt_records") != []
        or retry.get("code") != "INITIAL_EXECUTION"
        or retry.get("prior_attempt_id") is not None
        or initial["last_dispatched_next_tick"] != tick
        or initial["last_computed_next_tick"] != tick
        or initial["durable_checkpoint_next_tick"] != tick
        or initial["published_artifact_next_tick"] > tick
    ):
        raise ActorError(
            "V2 child restore source, occurrence, or fresh-Run binding differs",
            code="launchpad_attempt_binding_mismatch",
        )
    for field in (
        "checkpoint_digest",
        "source_attempt_terminal_digest",
        "sealed_bundle_digest",
        "completion_marker_digest",
    ):
        _digest(restore.get(field), field=field)
    _digest(terminal.get("result_digest"), field="source terminal result_digest")


class SchedulerLaunchpadAttemptIssuerV1:
    """The sole issuer for route-specific Launchpad Attempt authority."""

    def __init__(
        self,
        state_root: str | Path,
        *,
        allow_test_shared_gpu_state_root: bool = False,
        legacy_gpu_migration_test_observer: (
            Callable[[Mapping[str, Any]], Mapping[str, Any]] | None
        ) = None,
    ) -> None:
        self.state_root = Path(state_root).expanduser().resolve()
        test_authority_requested = bool(
            allow_test_shared_gpu_state_root
            or legacy_gpu_migration_test_observer is not None
        )
        if (
            test_authority_requested
            and self.state_root == canonical_host_actor_state_root().resolve()
        ):
            raise ActorError(
                "Launchpad test authority cannot target canonical actor state",
                code="launchpad_legacy_gpu_migration_denied",
            )
        self._allow_test_shared_gpu_state_root = allow_test_shared_gpu_state_root
        if (
            legacy_gpu_migration_test_observer is not None
            and not allow_test_shared_gpu_state_root
        ):
            raise ActorError(
                "legacy migration observer injection is test-root only",
                code="launchpad_legacy_gpu_migration_denied",
            )
        self._legacy_gpu_migration_test_observer = (
            legacy_gpu_migration_test_observer
        )
        self.scheduler = scheduler_root(self.state_root)
        self.root = self.scheduler / "launchpad-attempts"
        self.claims_dir = self.root / "claims"
        self.preparations_dir = self.root / "preparations"
        self.preparation_heads_dir = self.root / "preparation-heads"
        self.start_effects_dir = self.root / "start-effects"
        self.starts_dir = self.root / "attempt-starts"
        self.bundles_dir = self.root / "authoritative-bundles"
        self.bundle_heads_dir = self.root / "bundle-heads"
        self.continuation_leases_dir = self.root / "continuation-leases"
        self.continuation_events_dir = self.root / "continuation-events"
        self.continuation_heads_dir = self.root / "continuation-heads"
        self.terminals_dir = self.root / "attempt-terminals"
        self.terminal_intents_dir = self.root / "terminal-intents"
        self.terminal_bundles_dir = self.root / "terminal-bundles"
        self.terminal_heads_dir = self.root / "terminal-heads"
        self.terminal_resource_reconciliations_dir = (
            self.root / "terminal-resource-reconciliations"
        )
        self.quarantine_dir = self.root / "quarantine"
        self.dead_owner_reclaims_dir = self.root / "dead-owner-reclaims"
        self.legacy_gpu_migrations_dir = self.root / "legacy-gpu-migrations"
        self.legacy_gpu_migration_proofs_dir = (
            self.legacy_gpu_migrations_dir / "proofs"
        )
        self.legacy_gpu_migration_authorizations_dir = (
            self.legacy_gpu_migrations_dir / "authorizations"
        )
        # Compatibility alias for read-only callers written before the
        # authorization-before-event ordering was made explicit.
        self.legacy_gpu_migration_receipts_dir = (
            self.legacy_gpu_migration_authorizations_dir
        )
        self.resource_ledger = ResourceStartLeaseLedgerV1(
            self.scheduler,
            allow_test_legacy_gpu_migration=allow_test_shared_gpu_state_root,
        )

    @staticmethod
    def owner_digest() -> ContentDigest:
        module = Path(__file__).resolve()
        return canonical_sha256(
            {
                "owner": "tools.scheduler.launchpad_attempts",
                "module_path": str(module),
                "module_sha256": hashlib.sha256(module.read_bytes()).hexdigest(),
            }
        )

    def _claim_path(self, claim: LaunchpadAttemptClaimV1) -> Path:
        return self.claims_dir / f"{claim.claim_digest().hex}.json"

    def _require_current_scheduler_owner(self, claim: LaunchpadAttemptClaimV1) -> None:
        if (
            claim.scheduler_owner_id != "tools.scheduler.launchpad_attempts"
            or claim.scheduler_owner_digest != self.owner_digest()
        ):
            raise ActorError(
                "Launchpad claim Scheduler owner is stale or foreign",
                code="launchpad_attempt_owner_mismatch",
            )

    def _preparation_head_path(self, attempt_id: str) -> Path:
        return self.preparation_heads_dir / f"{_safe_key(attempt_id)}.json"

    def _bundle_head_path(self, attempt_id: str) -> Path:
        return self.bundle_heads_dir / f"{_safe_key(attempt_id)}.json"

    def _terminal_head_path(self, attempt_id: str) -> Path:
        return self.terminal_heads_dir / f"{_safe_key(attempt_id)}.json"

    def _terminal_intent_path(self, attempt_id: str) -> Path:
        return self.terminal_intents_dir / f"{_safe_key(attempt_id)}.json"

    def _continuation_head_path(self, lineage_digest: ContentDigest) -> Path:
        return self.continuation_heads_dir / f"{lineage_digest.hex}.json"

    def _read_pointer(self, path: Path, *, label: str) -> dict[str, Any] | None:
        if not path.exists():
            return None
        if path.is_symlink() or not path.is_file():
            raise ActorError(
                f"{label} pointer is not a regular file",
                code="launchpad_attempt_invalid",
            )
        return read_json(path, name=label)

    def _preparation(self, attempt_id: str) -> dict[str, Any]:
        head = self._read_pointer(
            self._preparation_head_path(attempt_id), label="Launchpad preparation head"
        )
        if head is None:
            raise ActorError(
                "Launchpad Attempt is not prepared",
                code="launchpad_attempt_not_prepared",
            )
        path = Path(str(head.get("preparation_path") or ""))
        if (
            head.get("schema_version") != "crca_launchpad_preparation_head_v1"
            or not path.is_file()
            or path.is_symlink()
        ):
            raise ActorError(
                "Launchpad preparation head is invalid",
                code="launchpad_attempt_invalid",
            )
        preparation = read_json(path, name="LaunchpadAttemptPreparation")
        digest = _record_digest(preparation)
        if head.get("preparation_digest") != digest.to_dict():
            raise ActorError(
                "Launchpad preparation head digest differs",
                code="launchpad_attempt_invalid",
            )
        self._validate_preparation(preparation)
        return preparation

    def _authoritative_bundle(self, attempt_id: str) -> dict[str, Any]:
        head = self._read_pointer(
            self._bundle_head_path(attempt_id), label="Launchpad bundle head"
        )
        if head is None:
            raise ActorError(
                "Launchpad Attempt has not started authoritatively",
                code="launchpad_attempt_not_started",
            )
        path = Path(str(head.get("bundle_path") or ""))
        if (
            head.get("schema_version") != "crca_launchpad_attempt_bundle_head_v1"
            or head.get("attempt_id") != attempt_id
            or not path.is_file()
            or path.is_symlink()
        ):
            raise ActorError(
                "Launchpad authoritative bundle head is invalid",
                code="launchpad_attempt_invalid",
            )
        bundle = read_json(path, name="AuthoritativeAttemptBundle")
        if (
            head.get("bundle_digest") != _record_digest(bundle).to_dict()
            or bundle.get("schema_version")
            not in {
                AUTHORITATIVE_BUNDLE_SCHEMA_VERSION,
                AUTHORITATIVE_BUNDLE_SCHEMA_VERSION_V2,
            }
            or dict(bundle.get("claim") or {}).get("attempt_id") != attempt_id
            or dict(bundle.get("attempt_start") or {}).get("attempt_id") != attempt_id
        ):
            raise ActorError(
                "Launchpad authoritative bundle differs from its head",
                code="launchpad_attempt_invalid",
            )
        return bundle

    def terminal_attempt_record(self, attempt_id: str) -> dict[str, Any]:
        """Reopen one exact committed terminal Attempt without mutating it.

        This is a read-only Scheduler authority projection.  It accepts only a
        committed authoritative bundle and its matching committed terminal
        head.  The embedded restore-source projection is the exact three-record
        wire value consumed by V2 child ``prepare``.
        """

        if not isinstance(attempt_id, str) or not attempt_id.strip():
            raise ActorError(
                "terminal attempt_id is required",
                code="launchpad_attempt_invalid",
            )
        with _scheduler_execution_lock(self.state_root):
            bundle = self._authoritative_bundle(attempt_id)
            terminal_head = self._read_pointer(
                self._terminal_head_path(attempt_id),
                label="Launchpad terminal head",
            )
            if terminal_head is None:
                raise ActorError(
                    "Launchpad Attempt is not terminal",
                    code="launchpad_attempt_terminal",
                )
            _require_exact_fields(
                terminal_head,
                {
                    "schema_version",
                    "attempt_id",
                    "terminal_bundle_digest",
                    "terminal_bundle_path",
                },
                label="Launchpad terminal head",
            )
            terminal_digest = _digest(
                terminal_head.get("terminal_bundle_digest"),
                field="terminal_bundle_digest",
            )
            terminal_path = Path(str(terminal_head.get("terminal_bundle_path") or ""))
            expected_terminal_path = (
                self.terminal_bundles_dir
                / _safe_key(attempt_id)
                / f"{terminal_digest.hex}.json"
            )
            if (
                terminal_head.get("schema_version")
                != "crca_launchpad_attempt_terminal_head_v1"
                or terminal_head.get("attempt_id") != attempt_id
                or not terminal_path.is_file()
                or terminal_path.is_symlink()
                or terminal_path.resolve() != expected_terminal_path.resolve()
            ):
                raise ActorError(
                    "Launchpad terminal head is invalid",
                    code="launchpad_attempt_invalid",
                )
            terminal_bundle = read_json(terminal_path, name="LaunchpadTerminalBundle")
            _require_exact_fields(
                terminal_bundle,
                {
                    "schema_version",
                    "attempt_id",
                    "authoritative_bundle_digest",
                    "attempt_terminal",
                    "continuation_event",
                    "terminalized_at_utc",
                },
                label="LaunchpadTerminalBundle",
            )
            if (
                terminal_bundle.get("schema_version") != TERMINAL_BUNDLE_SCHEMA_VERSION
                or terminal_bundle.get("attempt_id") != attempt_id
                or _record_digest(terminal_bundle) != terminal_digest
                or _digest(
                    terminal_bundle.get("authoritative_bundle_digest"),
                    field="authoritative_bundle_digest",
                )
                != _record_digest(bundle)
            ):
                raise ActorError(
                    "Launchpad terminal differs from authority",
                    code="launchpad_attempt_binding_mismatch",
                )
            start = dict(bundle.get("attempt_start") or {})
            concretized = dict(bundle.get("concretized_execution") or {})
            terminal = _attempt_terminal_v1(
                dict(terminal_bundle.get("attempt_terminal") or {}),
                attempt_id=attempt_id,
                start=start,
                run=dict(bundle.get("run_occurrence") or {}),
                semantic_spec=dict(bundle.get("semantic_run_spec") or {}),
            )
            restore_source = {
                "start": start,
                "concretized_execution": concretized,
                "terminal": terminal,
            }
            return {
                "terminal_bundle": terminal_bundle,
                "terminal_bundle_path": str(terminal_path.resolve()),
                "terminal_bundle_digest": terminal_digest.to_dict(),
                "restore_source_attempt_record": restore_source,
            }

    def restore_source_attempt_record(self, attempt_id: str) -> dict[str, Any]:
        """Return the exact V2 child-restore source projection for an Attempt."""

        return dict(
            self.terminal_attempt_record(attempt_id)["restore_source_attempt_record"]
        )

    def _continuation_state(
        self, lineage_digest: ContentDigest
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any] | None]:
        head = self._read_pointer(
            self._continuation_head_path(lineage_digest),
            label="Launchpad continuation head",
        )
        if head is None:
            return [], [], None
        if head.get("schema_version") != CONTINUATION_HEAD_SCHEMA_VERSION:
            raise ActorError(
                "Launchpad continuation head schema is invalid",
                code="launchpad_attempt_invalid",
            )
        lease_paths = [Path(str(value)) for value in head.get("lease_paths") or []]
        event_paths = [Path(str(value)) for value in head.get("event_paths") or []]
        leases = [
            read_json(path, name="CanonicalContinuationLease") for path in lease_paths
        ]
        events = [
            read_json(path, name="ContinuationLeaseEvent") for path in event_paths
        ]
        if not leases:
            raise ActorError(
                "continuation head has no lease", code="launchpad_attempt_invalid"
            )
        for lease in leases:
            _continuation_lease_digest(lease)
        prior: ContentDigest | None = None
        for event in events:
            if (
                _optional_digest(
                    event.get("prior_event_digest"), field="prior_event_digest"
                )
                != prior
            ):
                raise ActorError(
                    "continuation event chain is invalid",
                    code="launchpad_attempt_invalid",
                )
            prior = _continuation_event_digest(event)
        if head.get("current_lease_digest") != _continuation_lease_digest(
            leases[-1]
        ).to_dict() or head.get("latest_event_digest") != (
            None if prior is None else prior.to_dict()
        ):
            raise ActorError(
                "continuation head differs from its records",
                code="launchpad_attempt_invalid",
            )
        _continuation_ledger_state(leases, events)
        return leases, events, head

    def _write_continuation_head(
        self,
        lineage_digest: ContentDigest,
        leases: list[dict[str, Any]],
        events: list[dict[str, Any]],
    ) -> Path:
        payload = {
            "schema_version": CONTINUATION_HEAD_SCHEMA_VERSION,
            "lineage_scope_digest": lineage_digest.to_dict(),
            "current_lease_digest": _continuation_lease_digest(leases[-1]).to_dict(),
            "latest_event_digest": (
                None if not events else _continuation_event_digest(events[-1]).to_dict()
            ),
            "lease_paths": [
                str(
                    self.continuation_leases_dir
                    / lineage_digest.hex
                    / f"{_continuation_lease_digest(lease).hex}.json"
                )
                for lease in leases
            ],
            "event_paths": [
                str(
                    self.continuation_events_dir
                    / lineage_digest.hex
                    / f"{index:06d}-{_continuation_event_digest(event).hex}.json"
                )
                for index, event in enumerate(events, start=1)
            ],
        }
        atomic_write_json(self._continuation_head_path(lineage_digest), payload)
        return self._continuation_head_path(lineage_digest)

    def _append_continuation_event(
        self,
        lineage_digest: ContentDigest,
        leases: list[dict[str, Any]],
        events: list[dict[str, Any]],
        event: dict[str, Any],
        successor: dict[str, Any] | None = None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        _continuation_ledger_state(leases, events)
        if events and events[-1] == event and successor is None:
            return leases, events
        if _optional_digest(
            event.get("prior_event_digest"), field="prior_event_digest"
        ) != (None if not events else _continuation_event_digest(events[-1])):
            raise ActorError(
                "continuation event prior digest differs",
                code="launchpad_attempt_invalid",
            )
        _continuation_event_digest(event)
        candidate_leases = leases
        if successor is not None:
            successor_digest = _continuation_lease_digest(successor)
            if (
                _digest(
                    event.get("successor_lease_digest"), field="successor_lease_digest"
                )
                != successor_digest
            ):
                raise ActorError(
                    "continuation successor digest differs",
                    code="launchpad_attempt_invalid",
                )
            candidate_leases = [*leases, successor]
        candidate_events = [*events, event]
        _continuation_ledger_state(candidate_leases, candidate_events)
        if successor is not None:
            _write_immutable(
                self.continuation_leases_dir
                / lineage_digest.hex
                / f"{successor_digest.hex}.json",
                successor,
                label="CanonicalContinuationLease",
            )
            leases = candidate_leases
        event_digest = _continuation_event_digest(event)
        _write_immutable(
            self.continuation_events_dir
            / lineage_digest.hex
            / f"{len(events) + 1:06d}-{event_digest.hex}.json",
            event,
            label="ContinuationLeaseEvent",
        )
        events = candidate_events
        self._write_continuation_head(lineage_digest, leases, events)
        return leases, events

    def _expire_continuation_if_stale(
        self,
        lineage_digest: ContentDigest,
        leases: list[dict[str, Any]],
        events: list[dict[str, Any]],
        *,
        at_utc: str,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], bool]:
        """Fence one ACTIVE continuation whose exact lease window elapsed."""

        if (
            _continuation_ledger_state(leases, events)
            is not _ContinuationStateV1.ACTIVE
        ):
            return leases, events, False
        current = leases[-1]
        if _parse_utc(at_utc, field="at_utc") < _parse_utc(
            current["expires_at_utc"], field="expires_at_utc"
        ):
            return leases, events, False
        event = _continuation_event(
            current,
            code="EXPIRED",
            event_at_utc=at_utc,
            reason="Scheduler observed the exact continuation lease expiry.",
            prior_event_digest=(
                None if not events else _continuation_event_digest(events[-1])
            ),
        )
        leases, events = self._append_continuation_event(
            lineage_digest,
            leases,
            events,
            event,
        )
        return leases, events, True

    @staticmethod
    def _validate_projection(
        claim: LaunchpadAttemptClaimV1,
        *,
        execution_request: Mapping[str, Any],
        campaign_plan: Mapping[str, Any],
        preflight_receipt: Mapping[str, Any],
        lifecycle_binding: Mapping[str, Any],
        semantic_run_spec: Mapping[str, Any],
        run_occurrence: Mapping[str, Any],
        execution_implementation: Mapping[str, Any],
        concretized_execution: Mapping[str, Any],
        attempt_admission_request: Mapping[str, Any],
        retry_basis: Mapping[str, Any],
        initial_frontiers: Mapping[str, Any],
    ) -> None:
        _require_exact_fields(
            preflight_receipt,
            {
                "schema_version",
                "provider_id",
                "route_id",
                "device_id",
                "native_artifact_digest",
                "batch_size",
                "ticks_per_dispatch",
                "execution_mode",
            },
            label="Launchpad preflight projection",
        )
        if preflight_receipt.get("schema_version") != (
            "crca_launchpad_preflight_authority_projection_v1"
        ):
            raise ActorError(
                "Launchpad preflight projection schema is unsupported",
                code="launchpad_attempt_binding_mismatch",
            )
        comparisons = (
            (
                claim.execution_request_digest,
                _record_digest(execution_request),
                "request",
            ),
            (claim.campaign_plan_digest, _record_digest(campaign_plan), "plan"),
            (
                claim.preflight_receipt_digest,
                _record_digest(preflight_receipt),
                "preflight",
            ),
            (
                claim.lifecycle_binding_digest,
                _record_digest(lifecycle_binding),
                "lifecycle binding",
            ),
            (
                claim.semantic_run_spec_digest,
                _record_digest(semantic_run_spec),
                "semantic Run spec",
            ),
            (
                claim.run_occurrence_digest,
                _record_digest(run_occurrence),
                "Run occurrence",
            ),
            (
                claim.execution_implementation_digest,
                _record_digest(execution_implementation),
                "implementation",
            ),
            (
                claim.concretized_execution_digest,
                _record_digest(concretized_execution),
                "concretized execution",
            ),
            (
                claim.attempt_admission_request_digest,
                _record_digest(attempt_admission_request),
                "admission request",
            ),
            (claim.retry_basis_digest, _record_digest(retry_basis), "retry basis"),
            (
                claim.initial_frontiers_digest,
                _record_digest(initial_frontiers),
                "initial frontiers",
            ),
        )
        for expected, actual, label in comparisons:
            if expected != actual:
                raise ActorError(
                    f"Launchpad claim differs from exact {label}",
                    code="launchpad_attempt_binding_mismatch",
                )
        preflight_pairs = (
            (preflight_receipt.get("provider_id"), claim.provider_id, "provider_id"),
            (preflight_receipt.get("route_id"), claim.route_id, "route_id"),
            (preflight_receipt.get("device_id"), claim.device_id, "device_id"),
            (preflight_receipt.get("batch_size"), claim.batch_size, "batch_size"),
            (
                preflight_receipt.get("ticks_per_dispatch"),
                claim.ticks_per_dispatch,
                "ticks_per_dispatch",
            ),
            (
                preflight_receipt.get("execution_mode"),
                claim.execution_mode,
                "execution_mode",
            ),
        )
        for actual, expected, field in preflight_pairs:
            if actual != expected:
                raise ActorError(
                    f"Launchpad preflight differs at {field}",
                    code="launchpad_attempt_binding_mismatch",
                )
        if (
            _digest(
                preflight_receipt.get("native_artifact_digest"),
                field="native_artifact_digest",
            )
            != claim.native_artifact_digest
        ):
            raise ActorError(
                "Launchpad preflight differs at native_artifact_digest",
                code="launchpad_attempt_binding_mismatch",
            )
        if (
            run_occurrence.get("run_id") != claim.run_id
            or retry_basis.get("run_id") != claim.run_id
            or retry_basis.get("candidate_attempt_id") != claim.attempt_id
            or execution_implementation.get("provider_id") != claim.provider_id
            or lifecycle_binding.get("output_root") != claim.run_root
        ):
            raise ActorError(
                "Launchpad claim differs from Run, Attempt, provider, or run root",
                code="launchpad_attempt_binding_mismatch",
            )
        if (
            _digest(
                lifecycle_binding.get("artifact_lifecycle_contract_digest"),
                field="artifact_lifecycle_contract_digest",
            )
            != claim.lifecycle_contract_digest
        ):
            raise ActorError(
                "Launchpad claim differs from lifecycle contract",
                code="launchpad_attempt_binding_mismatch",
            )
        concretized_pairs = (
            ("semantic_spec_digest", claim.semantic_run_spec_digest),
            ("implementation_digest", claim.execution_implementation_digest),
            (
                "attempt_admission_request_digest",
                claim.attempt_admission_request_digest,
            ),
            ("campaign_plan_digest", claim.campaign_plan_digest),
            ("bound_execution_request_digest", claim.execution_request_digest),
            ("baseline_digest", claim.baseline_digest),
            ("lifecycle_digest", claim.lifecycle_binding_digest),
        )
        for field, expected in concretized_pairs:
            if _digest(concretized_execution.get(field), field=field) != expected:
                raise ActorError(
                    f"concretized execution differs at {field}",
                    code="launchpad_attempt_binding_mismatch",
                )
        if (
            _digest(
                execution_implementation.get("source_tree_digest"),
                field="source_tree_digest",
            )
            != claim.source_digest
        ):
            raise ActorError(
                "implementation differs from source binding",
                code="launchpad_attempt_binding_mismatch",
            )
        artifact_digests = {
            _digest(item.get("digest"), field="native artifact digest")
            for item in execution_implementation.get("native_artifact_digests") or []
            if isinstance(item, Mapping)
        }
        if claim.native_artifact_digest not in artifact_digests:
            raise ActorError(
                "implementation does not bind the selected native artifact",
                code="launchpad_attempt_binding_mismatch",
            )

    def _validate_preparation(self, preparation: Mapping[str, Any]) -> None:
        schema = preparation.get("schema_version")
        if schema not in {
            PREPARATION_SCHEMA_VERSION,
            PREPARATION_SCHEMA_VERSION_V2,
        }:
            raise ActorError(
                "Launchpad preparation schema is invalid",
                code="launchpad_attempt_invalid",
            )
        if schema == PREPARATION_SCHEMA_VERSION:
            if "restore_source_attempt_record" in preparation:
                raise ActorError(
                    "V1 Launchpad preparation cannot carry a V2 restore source",
                    code="launchpad_attempt_invalid",
                )
        else:
            _validate_external_restore_source_record_v2(preparation)
        claim = LaunchpadAttemptClaimV1.from_dict(dict(preparation.get("claim") or {}))
        resource = LaunchpadResourceStartLeaseV1.from_dict(
            dict(preparation.get("resource_start_lease") or {})
        )
        continuation = dict(preparation.get("continuation_lease") or {})
        if (
            resource.execution_claim_digest != claim.claim_digest()
            or resource.run_id != claim.run_id
            or resource.attempt_id != claim.attempt_id
            or _continuation_lease_digest(continuation)
            != _digest(
                preparation.get("continuation_lease_digest"),
                field="continuation_lease_digest",
            )
            or continuation.get("run_id") != claim.run_id
            or continuation.get("attempt_id") != claim.attempt_id
            or continuation.get("owner_id") != claim.execution_owner_id
            or _digest(
                continuation.get("lineage_scope_digest"),
                field="lineage_scope_digest",
            )
            != claim.lineage_scope_digest
        ):
            raise ActorError(
                "Launchpad preparation lease binding differs",
                code="launchpad_attempt_binding_mismatch",
            )
        self._validate_projection(
            claim,
            **{
                field: dict(preparation.get(field) or {})
                for field in (
                    "execution_request",
                    "campaign_plan",
                    "preflight_receipt",
                    "lifecycle_binding",
                    "semantic_run_spec",
                    "run_occurrence",
                    "execution_implementation",
                    "concretized_execution",
                    "attempt_admission_request",
                    "retry_basis",
                    "initial_frontiers",
                )
            },
        )

    def _commit_preparation(
        self,
        preparation: Mapping[str, Any],
        *,
        preparation_path: Path,
    ) -> dict[str, Any]:
        """Finish one exact preparation intent and publish its head last."""

        self._validate_preparation(preparation)
        claim = LaunchpadAttemptClaimV1.from_dict(dict(preparation["claim"]))
        self._require_current_scheduler_owner(claim)
        resource = LaunchpadResourceStartLeaseV1.from_dict(
            dict(preparation["resource_start_lease"])
        )
        continuation = dict(preparation["continuation_lease"])
        continuation_digest = _continuation_lease_digest(continuation)
        try:
            self.resource_ledger.grant(resource)
        except ProgrammeContractError as error:
            if "overlaps an active exact resource" not in str(error):
                raise
            raise ActorError(
                "exact Scheduler GPU resource is occupied",
                code="launchpad_attempt_resource_wait",
                related_records=[resource.lease_id],
            ) from error
        leases, events, head = self._continuation_state(claim.lineage_scope_digest)
        if head is None:
            if (
                int(continuation["lease_epoch"]) != 1
                or continuation.get("prior_lease_digest") is not None
            ):
                raise ActorError(
                    "noninitial continuation lease has no durable predecessor head",
                    code="launchpad_attempt_lineage_conflict",
                )
            _write_immutable(
                self.continuation_leases_dir
                / claim.lineage_scope_digest.hex
                / f"{continuation_digest.hex}.json",
                continuation,
                label="CanonicalContinuationLease",
            )
            leases = [continuation]
            events = []
            self._write_continuation_head(claim.lineage_scope_digest, leases, events)
        elif _continuation_lease_digest(leases[-1]) != continuation_digest:
            prior_digest = _optional_digest(
                continuation.get("prior_lease_digest"), field="prior_lease_digest"
            )
            terminal_code = None if not events else str(events[-1].get("code") or "")
            if prior_digest != _continuation_lease_digest(
                leases[-1]
            ) or terminal_code not in {"RELEASED", "EXPIRED", "OWNER_DEAD"}:
                raise ActorError(
                    "continuation lineage head differs from prepared successor",
                    code="launchpad_attempt_lineage_conflict",
                )
            transition = _continuation_event(
                leases[-1],
                code=("REASSIGNED" if terminal_code == "RELEASED" else "RECLAIMED"),
                event_at_utc=str(continuation["granted_at_utc"]),
                reason="Scheduler assigned exact successor Launchpad Attempt",
                prior_event_digest=_continuation_event_digest(events[-1]),
                successor_lease_digest=continuation_digest,
            )
            self._append_continuation_event(
                claim.lineage_scope_digest,
                leases,
                events,
                transition,
                successor=continuation,
            )
        preparation_digest = _record_digest(preparation)
        atomic_write_json(
            self._preparation_head_path(claim.attempt_id),
            {
                "schema_version": "crca_launchpad_preparation_head_v1",
                "attempt_id": claim.attempt_id,
                "preparation_digest": preparation_digest.to_dict(),
                "preparation_path": str(preparation_path),
            },
        )
        return {
            "status": "PREPARED",
            "idempotent_replay": False,
            "attempt_created": False,
            "effect_executed": False,
            "preparation": dict(preparation),
            "preparation_digest": preparation_digest.to_dict(),
            "preparation_path": str(preparation_path),
        }

    def prepare(
        self,
        *,
        claim: LaunchpadAttemptClaimV1,
        execution_request: Mapping[str, Any],
        campaign_plan: Mapping[str, Any],
        preflight_receipt: Mapping[str, Any],
        lifecycle_binding: Mapping[str, Any],
        semantic_run_spec: Mapping[str, Any],
        run_occurrence: Mapping[str, Any],
        execution_implementation: Mapping[str, Any],
        concretized_execution: Mapping[str, Any],
        attempt_admission_request: Mapping[str, Any],
        retry_basis: Mapping[str, Any],
        initial_frontiers: Mapping[str, Any],
        monotonic_ordinal: int,
        nonce: str,
        parent_attempt_id: str | None,
        resource_ids: tuple[str, ...],
        resource_scope_digest: ContentDigest,
        resource_observation_id: str,
        resource_observation: Mapping[str, Any],
        resource_observed_at_utc: str,
        resource_recommendation_id: str,
        resource_recommendation: Mapping[str, Any],
        live_provider_check: Mapping[str, Any],
        at_utc: str,
        start_window_seconds: int = 300,
        resource_ttl_seconds: int = 600,
        continuation_ttl_seconds: int = 900,
        restore_authorization: Mapping[str, Any] | None = None,
        restore_source_attempt_record: Mapping[str, Any] | None = None,
        implementation_conformance: Mapping[str, Any] | None = None,
        prior_attempt_records: tuple[Mapping[str, Any], ...] = (),
        max_observation_age_seconds: int = 300,
        resource_candidate_sets: tuple[tuple[str, ...], ...] = (),
    ) -> dict[str, Any]:
        """Prepare exact authority records; create no AttemptStart or effect."""

        self._require_current_scheduler_owner(claim)
        self._validate_projection(
            claim,
            execution_request=execution_request,
            campaign_plan=campaign_plan,
            preflight_receipt=preflight_receipt,
            lifecycle_binding=lifecycle_binding,
            semantic_run_spec=semantic_run_spec,
            run_occurrence=run_occurrence,
            execution_implementation=execution_implementation,
            concretized_execution=concretized_execution,
            attempt_admission_request=attempt_admission_request,
            retry_basis=retry_basis,
            initial_frontiers=initial_frontiers,
        )
        observed_at = _parse_utc(
            resource_observed_at_utc, field="resource_observed_at_utc"
        )
        requested_at = _parse_utc(at_utc, field="at_utc")
        age = (requested_at - observed_at).total_seconds()
        if age < 0 or age > max_observation_age_seconds:
            raise ActorError(
                "resource observation is outside the Scheduler freshness bound",
                code="launchpad_attempt_resource_wait",
            )
        if resource_candidate_sets:
            if resource_ids:
                raise ActorError(
                    "exact resource_ids and candidate sets are mutually exclusive",
                    code="launchpad_attempt_invalid",
                )
            requested_resource_sets = resource_candidate_sets
        else:
            requested_resource_sets = (resource_ids,)
        execution = campaign_plan.get("execution")
        gpu_resource_policy = (
            execution.get("gpu_resource_policy")
            if isinstance(execution, Mapping)
            else None
        )
        has_lane_resource = any(
            "/lane/" in resource_id
            for candidate in requested_resource_sets
            for resource_id in candidate
        )
        if gpu_resource_policy is None:
            if has_lane_resource:
                raise ActorError(
                    "shared GPU resource requires a canonical plan policy",
                    code="launchpad_attempt_binding_mismatch",
                )
            expected_resource_ids = (f"gpu:{claim.device_id}",)
            expected_scope = {
                "schema_version": "eiel_exclusive_gpu_scope_v1",
                "resource_ids": list(expected_resource_ids),
                "run_id": claim.run_id,
            }
            if (
                resource_candidate_sets
                or resource_ids != expected_resource_ids
                or resource_scope_digest != canonical_sha256(expected_scope)
            ):
                raise ActorError(
                    "exclusive GPU resource differs from canonical topology",
                    code="launchpad_attempt_binding_mismatch",
                )
        else:
            try:
                policy = validate_shared_w32_gpu_policy_binding_v1(
                    gpu_resource_policy
                )
                expected_candidates = shared_gpu_lane_candidate_sets_v1(claim.device_id)
            except GpuResourceTopologyError as exc:
                raise ActorError(
                    str(exc), code="launchpad_attempt_binding_mismatch"
                ) from exc
            expected_scope = {
                "schema_version": "eiel_shared_gpu_lane_scope_v1",
                "run_id": claim.run_id,
                "device_id": claim.device_id,
                "gpu_resource_policy": policy,
                "candidate_resource_sets": [
                    list(candidate) for candidate in expected_candidates
                ],
            }
            if (
                not resource_candidate_sets
                or resource_candidate_sets != expected_candidates
                or resource_scope_digest != canonical_sha256(expected_scope)
            ):
                raise ActorError(
                    "shared GPU resource candidates differ from canonical topology",
                    code="launchpad_attempt_binding_mismatch",
                )
            if (
                self.state_root != canonical_host_actor_state_root()
                and not self._allow_test_shared_gpu_state_root
            ):
                raise ActorError(
                    "shared GPU admission requires the host-canonical Scheduler ledger",
                    code="launchpad_attempt_binding_mismatch",
                )
        if any(
            not candidate or tuple(sorted(set(candidate))) != candidate
            for candidate in requested_resource_sets
        ):
            raise ActorError(
                "resource candidates must be sorted, unique, and non-empty",
                code="launchpad_attempt_invalid",
            )
        if any(
            not any(
                resource_id == claim.device_id
                or resource_id.endswith(f":{claim.device_id}")
                or resource_id.startswith(f"gpu:{claim.device_id}/lane/")
                for resource_id in candidate
            )
            for candidate in requested_resource_sets
        ):
            raise ActorError(
                "resource candidates do not bind the claimed device",
                code="launchpad_attempt_binding_mismatch",
            )
        if resource_ttl_seconds < start_window_seconds:
            raise ActorError(
                "resource lease TTL must cover its start window",
                code="launchpad_attempt_invalid",
            )
        if resource_recommendation.get("decision") not in {
            "ADMIT",
            "ADMIT_WITH_LIMITS",
        }:
            raise ActorError(
                "Resource Manager recommendation does not support a start lease: "
                f"decision={resource_recommendation.get('decision')!r} "
                f"reason_codes={resource_recommendation.get('reason_codes')!r}",
                code="launchpad_attempt_resource_wait",
            )
        if live_provider_check.get("available") is not True:
            raise ActorError(
                "live provider check does not support a start lease",
                code="launchpad_attempt_resource_wait",
            )
        for label, record in (
            ("resource observation", resource_observation),
            ("live provider check", live_provider_check),
        ):
            device_id = record.get("device_id")
            if device_id is not None and device_id != claim.device_id:
                raise ActorError(
                    f"{label} binds another device",
                    code="launchpad_attempt_binding_mismatch",
                )
        with _scheduler_execution_lock(self.state_root):
            existing = self._read_pointer(
                self._preparation_head_path(claim.attempt_id),
                label="Launchpad preparation head",
            )
            if existing is not None:
                prepared = self._preparation(claim.attempt_id)
                if dict(prepared.get("claim") or {}) != claim.to_dict():
                    raise ActorError(
                        "Attempt already has another prepared claim",
                        code="launchpad_attempt_replay",
                    )
                prepared_resource = LaunchpadResourceStartLeaseV1.from_dict(
                    dict(prepared["resource_start_lease"])
                )
                replay_resource_match = (
                    prepared_resource.resource_ids in requested_resource_sets
                    if resource_candidate_sets
                    else prepared_resource.resource_ids == resource_ids
                )
                replay_bindings = (
                    replay_resource_match
                    and prepared_resource.resource_scope_digest == resource_scope_digest
                    and prepared_resource.resource_observation_id
                    == resource_observation_id
                    and prepared_resource.resource_observation_digest
                    == _record_digest(resource_observation)
                    and prepared_resource.resource_recommendation_id
                    == resource_recommendation_id
                    and prepared_resource.resource_recommendation_digest
                    == _record_digest(resource_recommendation)
                    and prepared_resource.live_provider_check_digest
                    == _record_digest(live_provider_check)
                    and prepared.get("restore_source_attempt_record")
                    == (
                        None
                        if restore_source_attempt_record is None
                        else dict(restore_source_attempt_record)
                    )
                )
                if not replay_bindings:
                    raise ActorError(
                        "prepare replay differs from the reserved resource bindings",
                        code="launchpad_attempt_replay",
                    )
                return {
                    "status": "PREPARED",
                    "idempotent_replay": True,
                    "attempt_created": False,
                    "effect_executed": False,
                    "preparation": prepared,
                }

            def resource_lease_factory(
                selected_resource_ids: tuple[str, ...],
                epoch: int,
                prior_resource_digest: ContentDigest | None,
            ) -> LaunchpadResourceStartLeaseV1:
                resource_seed = {
                    "claim_digest": claim.claim_digest().to_dict(),
                    "resource_ids": list(selected_resource_ids),
                    "lease_epoch": epoch,
                    "prior_lease_digest": (
                        None
                        if prior_resource_digest is None
                        else prior_resource_digest.to_dict()
                    ),
                    "issued_at_utc": at_utc,
                }
                resource_seed_digest = canonical_sha256(resource_seed)
                return LaunchpadResourceStartLeaseV1(
                    lease_id=f"LPRSTART-{resource_seed_digest.hex[:24].upper()}",
                    scheduler_owner_id=claim.scheduler_owner_id,
                    scheduler_owner_digest=claim.scheduler_owner_digest,
                    execution_claim_digest=claim.claim_digest(),
                    run_id=claim.run_id,
                    attempt_id=claim.attempt_id,
                    attempt_admission_request_digest=(
                        claim.attempt_admission_request_digest
                    ),
                    resource_ids=selected_resource_ids,
                    resource_scope_digest=resource_scope_digest,
                    resource_observation_id=resource_observation_id,
                    resource_observation_digest=_record_digest(resource_observation),
                    resource_recommendation_id=resource_recommendation_id,
                    resource_recommendation_digest=_record_digest(
                        resource_recommendation
                    ),
                    live_provider_check_digest=_record_digest(live_provider_check),
                    lease_epoch=epoch,
                    fencing_token_digest=canonical_sha256(
                        {
                            "resource_seed": resource_seed,
                            "scheduler_owner_digest": (
                                claim.scheduler_owner_digest.to_dict()
                            ),
                        }
                    ),
                    prior_lease_digest=prior_resource_digest,
                    issued_at_utc=at_utc,
                    start_before_utc=_add_seconds(at_utc, start_window_seconds),
                    expires_at_utc=_add_seconds(at_utc, resource_ttl_seconds),
                )

            if resource_candidate_sets:
                resource_lease, _resource_lease_path = (
                    self.resource_ledger.grant_first_available_launchpad(
                        candidate_resource_ids=resource_candidate_sets,
                        at_utc=at_utc,
                        lease_factory=resource_lease_factory,
                    )
                )
            else:
                epoch, prior_resource_digest = self.resource_ledger.next_grant_fields(
                    resource_ids=resource_ids, at_utc=at_utc
                )
                resource_lease = resource_lease_factory(
                    resource_ids, epoch, prior_resource_digest
                )
            leases, events, continuation_head = self._continuation_state(
                claim.lineage_scope_digest
            )
            if continuation_head is None:
                continuation_epoch = 1
                prior_continuation_digest = None
            else:
                leases, events, _expired = self._expire_continuation_if_stale(
                    claim.lineage_scope_digest,
                    leases,
                    events,
                    at_utc=at_utc,
                )
                terminal_code = (
                    None if not events else str(events[-1].get("code") or "")
                )
                if terminal_code not in {"RELEASED", "EXPIRED", "OWNER_DEAD"}:
                    if terminal_code == "PUBLISH_COMMITTED":
                        raise ActorError(
                            "canonical continuation lineage already has a published terminal",
                            code="launchpad_attempt_lineage_published",
                        )
                    raise ActorError(
                        "canonical continuation lineage already has a live owner",
                        code="launchpad_attempt_lineage_busy",
                    )
                continuation_epoch = int(leases[-1]["lease_epoch"]) + 1
                prior_continuation_digest = _continuation_lease_digest(leases[-1])
            continuation_seed = {
                "claim_digest": claim.claim_digest().to_dict(),
                "lease_epoch": continuation_epoch,
                "prior_lease_digest": (
                    None
                    if prior_continuation_digest is None
                    else prior_continuation_digest.to_dict()
                ),
                "granted_at_utc": at_utc,
            }
            continuation_seed_digest = canonical_sha256(continuation_seed)
            continuation_lease = {
                "schema_version": CONTINUATION_LEASE_SCHEMA_VERSION,
                "lease_id": f"LP-CONT-{continuation_seed_digest.hex[:24]}",
                "run_id": claim.run_id,
                "attempt_id": claim.attempt_id,
                "restore_checkpoint_digest": attempt_admission_request.get(
                    "requested_restore_checkpoint_digest"
                ),
                "lineage_scope_digest": claim.lineage_scope_digest.to_dict(),
                "owner_id": claim.execution_owner_id,
                "lease_epoch": continuation_epoch,
                "fencing_token_digest": canonical_sha256(
                    {
                        "continuation_seed": continuation_seed,
                        "execution_owner_digest": claim.execution_owner_digest.to_dict(),
                    }
                ).to_dict(),
                "prior_lease_digest": (
                    None
                    if prior_continuation_digest is None
                    else prior_continuation_digest.to_dict()
                ),
                "granted_at_utc": at_utc,
                "expires_at_utc": _add_seconds(at_utc, continuation_ttl_seconds),
            }
            continuation_digest = _continuation_lease_digest(continuation_lease)
            preparation_body: dict[str, Any] = {
                "schema_version": (
                    PREPARATION_SCHEMA_VERSION
                    if restore_source_attempt_record is None
                    else PREPARATION_SCHEMA_VERSION_V2
                ),
                "claim": claim.to_dict(),
                "resource_start_lease": resource_lease.to_dict(),
                "continuation_lease": continuation_lease,
                "continuation_lease_digest": continuation_digest.to_dict(),
                "execution_request": dict(execution_request),
                "campaign_plan": dict(campaign_plan),
                "preflight_receipt": dict(preflight_receipt),
                "lifecycle_binding": dict(lifecycle_binding),
                "semantic_run_spec": dict(semantic_run_spec),
                "run_occurrence": dict(run_occurrence),
                "execution_implementation": dict(execution_implementation),
                "concretized_execution": dict(concretized_execution),
                "attempt_admission_request": dict(attempt_admission_request),
                "retry_basis": dict(retry_basis),
                "initial_frontiers": dict(initial_frontiers),
                "monotonic_ordinal": monotonic_ordinal,
                "nonce": nonce,
                "parent_attempt_id": parent_attempt_id,
                "restore_authorization": (
                    None
                    if restore_authorization is None
                    else dict(restore_authorization)
                ),
                "implementation_conformance": (
                    None
                    if implementation_conformance is None
                    else dict(implementation_conformance)
                ),
                "prior_attempt_records": [dict(row) for row in prior_attempt_records],
                "resource_observed_at_utc": resource_observed_at_utc,
                "max_observation_age_seconds": max_observation_age_seconds,
                "prepared_at_utc": at_utc,
            }
            if restore_source_attempt_record is not None:
                preparation_body["restore_source_attempt_record"] = dict(
                    restore_source_attempt_record
                )
            self._validate_preparation(preparation_body)
            preparation_digest = _record_digest(preparation_body)
            preparation_path = (
                self.preparations_dir
                / _safe_key(claim.attempt_id)
                / f"{preparation_digest.hex}.json"
            )
            _write_immutable(
                self._claim_path(claim), claim.to_dict(), label="LaunchpadAttemptClaim"
            )
            # Persist a non-authoritative intent first.  The shared resource
            # grant and continuation head may then be converged after a crash;
            # only the preparation head makes the exact set startable.
            _write_immutable(
                preparation_path,
                preparation_body,
                label="LaunchpadAttemptPreparation",
            )
            return self._commit_preparation(
                preparation_body,
                preparation_path=preparation_path,
            )

    def _start_effect(
        self,
        preparation: Mapping[str, Any],
        *,
        process_owner_id: str,
        process_owner_digest: ContentDigest,
        process_id: int | None,
        process_start_identity: str | None,
        at_utc: str,
    ) -> dict[str, Any]:
        claim = LaunchpadAttemptClaimV1.from_dict(dict(preparation["claim"]))
        self._require_current_scheduler_owner(claim)
        resource = LaunchpadResourceStartLeaseV1.from_dict(
            dict(preparation["resource_start_lease"])
        )
        continuation = dict(preparation["continuation_lease"])
        if (
            process_owner_id != claim.execution_owner_id
            or process_owner_digest != claim.execution_owner_digest
        ):
            raise ActorError(
                "process owner differs from prepared execution owner",
                code="launchpad_attempt_owner_mismatch",
            )
        gpu_resource = _gpu_resource_kind_v1(
            resource.resource_ids, device_id=claim.device_id
        )
        process_identity_supplied = (
            type(process_id) is int
            and process_id > 0
            and isinstance(process_start_identity, str)
            and process_start_identity.startswith("linux-proc:")
        )
        process_identity_requested = (
            process_id is not None or process_start_identity is not None
        )
        if (process_identity_requested and not process_identity_supplied) or (
            gpu_resource is not None and not process_identity_supplied
        ):
            raise ActorError(
                "every new GPU start requires an exact Linux process identity",
                code="launchpad_attempt_owner_mismatch",
            )
        return {
            "schema_version": (
                START_EFFECT_SCHEMA_VERSION_V2
                if process_identity_supplied
                else START_EFFECT_SCHEMA_VERSION
            ),
            "effect_kind": "SCHEDULER_EXECUTION_AUTHORITY_GRANTED",
            "claim_digest": claim.claim_digest().to_dict(),
            "run_id": claim.run_id,
            "attempt_id": claim.attempt_id,
            "resource_start_lease_digest": resource.lease_digest().to_dict(),
            "continuation_lease_digest": _continuation_lease_digest(
                continuation
            ).to_dict(),
            "scheduler_owner_id": claim.scheduler_owner_id,
            "scheduler_owner_digest": claim.scheduler_owner_digest.to_dict(),
            "process_owner_id": process_owner_id,
            "process_owner_digest": process_owner_digest.to_dict(),
            **(
                {
                    "process_id": process_id,
                    "process_start_identity": process_start_identity,
                }
                if process_identity_supplied
                else {}
            ),
            "fencing_token_digest": continuation["fencing_token_digest"],
            "started_at_utc": at_utc,
        }

    def _finalize_start(
        self, preparation: Mapping[str, Any], effect: Mapping[str, Any]
    ) -> dict[str, Any]:
        claim = LaunchpadAttemptClaimV1.from_dict(dict(preparation["claim"]))
        self._require_current_scheduler_owner(claim)
        resource = LaunchpadResourceStartLeaseV1.from_dict(
            dict(preparation["resource_start_lease"])
        )
        continuation = dict(preparation["continuation_lease"])
        started_at = str(effect["started_at_utc"])
        events = self.resource_ledger.events(resource.lease_id)
        resource_event = ResourceStartLeaseEventV1(
            lease_digest=resource.lease_digest(),
            code=ResourceStartLeaseEventCodeV1.STARTED,
            scheduler_owner_id=resource.scheduler_owner_id,
            fencing_token_digest=resource.fencing_token_digest,
            event_at_utc=started_at,
            reason=f"Scheduler started Launchpad claim {claim.claim_id}",
            prior_event_digest=None,
            successor_lease_digest=None,
        )
        if events and events[-1] != resource_event.to_dict():
            raise ActorError(
                "resource-start lease was terminalized by another outcome",
                code="launchpad_attempt_quarantine_required",
            )
        self.resource_ledger.append_event(resource.lease_id, resource_event)
        disposition = {
            "schema_version": "crca_attempt_admission_disposition_v1",
            "run_id": claim.run_id,
            "attempt_admission_request_digest": claim.attempt_admission_request_digest.to_dict(),
            "code": "START_AUTHORIZED",
            "reason": "Scheduler issued exact Launchpad resource and continuation leases",
            "scheduler_assignment_lease_digest": None,
            "resource_start_lease_digest": resource.lease_digest().to_dict(),
        }
        owner_binding = canonical_sha256(
            {
                "schema_version": "crca_launchpad_attempt_owner_binding_v1",
                "claim_digest": claim.claim_digest().to_dict(),
                "scheduler_owner_id": claim.scheduler_owner_id,
                "scheduler_owner_digest": claim.scheduler_owner_digest.to_dict(),
                "execution_owner_id": claim.execution_owner_id,
                "execution_owner_digest": claim.execution_owner_digest.to_dict(),
                "resource_start_lease_digest": resource.lease_digest().to_dict(),
                "continuation_lease_digest": _continuation_lease_digest(
                    continuation
                ).to_dict(),
            }
        )
        attempt_start = {
            "schema_version": "crca_attempt_start_v1",
            "attempt_id": claim.attempt_id,
            "run_id": claim.run_id,
            "monotonic_ordinal": preparation["monotonic_ordinal"],
            "nonce": preparation["nonce"],
            "concretized_execution_digest": claim.concretized_execution_digest.to_dict(),
            "implementation_digest": claim.execution_implementation_digest.to_dict(),
            "attempt_admission_request_digest": claim.attempt_admission_request_digest.to_dict(),
            "scheduler_assignment_lease_digest": None,
            "resource_start_lease_digest": resource.lease_digest().to_dict(),
            "post_admission_attempt_plan_digest": claim.post_admission_attempt_plan_digest.to_dict(),
            "admission_disposition_digest": _record_digest(disposition).to_dict(),
            "programme_work_session_assignment_dispatch_binding_digest": claim.programme_binding_digest.to_dict(),
            "parent_attempt_id": preparation["parent_attempt_id"],
            "restore_checkpoint_digest": dict(
                preparation["attempt_admission_request"]
            ).get("requested_restore_checkpoint_digest"),
            "canonical_continuation_lease_digest": _continuation_lease_digest(
                continuation
            ).to_dict(),
            "owner_binding_digest": owner_binding.to_dict(),
            "execution_side_effect_digest": _record_digest(effect).to_dict(),
            "initial_frontiers": dict(preparation["initial_frontiers"]),
        }
        start_digest = _record_digest(attempt_start)
        _write_immutable(
            self.starts_dir / _safe_key(claim.attempt_id) / f"{start_digest.hex}.json",
            attempt_start,
            label="AttemptStartV1",
        )
        leases, continuation_events, _head = self._continuation_state(
            claim.lineage_scope_digest
        )
        v2_restore_source = preparation.get("restore_source_attempt_record")
        bundle = {
            "schema_version": (
                AUTHORITATIVE_BUNDLE_SCHEMA_VERSION
                if v2_restore_source is None
                else AUTHORITATIVE_BUNDLE_SCHEMA_VERSION_V2
            ),
            "authority_kind": "SCHEDULER_ISSUED_LAUNCHPAD_ATTEMPT",
            "claim": claim.to_dict(),
            "resource_start_lease": resource.to_dict(),
            "resource_start_event": resource_event.to_dict(),
            "continuation_lease": continuation,
            "continuation_lease_ledger": {
                "leases": leases,
                "events": continuation_events,
            },
            "execution_request": dict(preparation["execution_request"]),
            "campaign_plan": dict(preparation["campaign_plan"]),
            "preflight_receipt": dict(preparation["preflight_receipt"]),
            "lifecycle_binding": dict(preparation["lifecycle_binding"]),
            "semantic_run_spec": dict(preparation["semantic_run_spec"]),
            "run_occurrence": dict(preparation["run_occurrence"]),
            "execution_implementation": dict(preparation["execution_implementation"]),
            "concretized_execution": dict(preparation["concretized_execution"]),
            "attempt_admission_request": dict(preparation["attempt_admission_request"]),
            "admission_disposition": disposition,
            "retry_basis": dict(preparation["retry_basis"]),
            "attempt_start": attempt_start,
            "restore_authorization": preparation.get("restore_authorization"),
            "implementation_conformance": preparation.get("implementation_conformance"),
            "prior_attempt_records": list(
                preparation.get("prior_attempt_records") or []
            ),
            "start_effect": dict(effect),
            "issued_at_utc": started_at,
        }
        if v2_restore_source is not None:
            bundle["restore_source_attempt_record"] = dict(v2_restore_source)
        bundle_digest = _record_digest(bundle)
        bundle_path = (
            self.bundles_dir / _safe_key(claim.attempt_id) / f"{bundle_digest.hex}.json"
        )
        _write_immutable(
            bundle_path, bundle, label="AuthoritativeLaunchpadAttemptBundle"
        )
        # Publication of this pointer is last; partial records are never an
        # executable authority bundle and are completed only by reconciliation.
        atomic_write_json(
            self._bundle_head_path(claim.attempt_id),
            {
                "schema_version": "crca_launchpad_attempt_bundle_head_v1",
                "attempt_id": claim.attempt_id,
                "bundle_digest": bundle_digest.to_dict(),
                "bundle_path": str(bundle_path),
            },
        )
        return {
            "status": "STARTED",
            "attempt_created": True,
            "effect_executed": True,
            "bundle": bundle,
            "bundle_digest": bundle_digest.to_dict(),
            "bundle_path": str(bundle_path),
        }

    def _quarantine(
        self, *, attempt_id: str, reason: str, at_utc: str, details: Mapping[str, Any]
    ) -> Path:
        payload = {
            "schema_version": QUARANTINE_SCHEMA_VERSION,
            "attempt_id": attempt_id,
            "reason": reason,
            "details_digest": _record_digest(details).to_dict(),
            "quarantined_at_utc": at_utc,
        }
        digest = _record_digest(payload)
        return _write_immutable(
            self.quarantine_dir / _safe_key(attempt_id) / f"{digest.hex}.json",
            payload,
            label="LaunchpadAttemptQuarantine",
        )

    def start(
        self,
        *,
        attempt_id: str,
        process_owner_id: str,
        process_owner_digest: ContentDigest,
        process_id: int | None = None,
        process_start_identity: str | None = None,
        at_utc: str,
    ) -> dict[str, Any]:
        """Atomically recheck and publish one Scheduler-authoritative start."""

        with _scheduler_execution_lock(self.state_root):
            try:
                existing_head = self._read_pointer(
                    self._bundle_head_path(attempt_id),
                    label="Launchpad bundle head",
                )
                if existing_head is not None:
                    bundle_path = Path(str(existing_head.get("bundle_path") or ""))
                    bundle = read_json(bundle_path, name="AuthoritativeAttemptBundle")
                    if (
                        existing_head.get("bundle_digest")
                        != _record_digest(bundle).to_dict()
                    ):
                        raise ActorError(
                            "Launchpad bundle head digest differs",
                            code="launchpad_attempt_invalid",
                        )
                    existing_claim = LaunchpadAttemptClaimV1.from_dict(
                        dict(bundle.get("claim") or {})
                    )
                    existing_resource = LaunchpadResourceStartLeaseV1.from_dict(
                        dict(bundle.get("resource_start_lease") or {})
                    )
                    self._require_current_scheduler_owner(existing_claim)
                    if (
                        existing_claim.execution_owner_id != process_owner_id
                        or existing_claim.execution_owner_digest != process_owner_digest
                        or (
                            bundle.get("start_effect", {}).get("schema_version")
                            == START_EFFECT_SCHEMA_VERSION_V2
                            and (
                                bundle["start_effect"].get("process_id") != process_id
                                or bundle["start_effect"].get("process_start_identity")
                                != process_start_identity
                            )
                        )
                    ):
                        raise ActorError(
                            "start replay uses another process owner",
                            code="launchpad_attempt_replay",
                        )
                    _require_exact_gpu_process_owner_v2(
                        bundle=bundle,
                        resource=existing_resource,
                        device_id=existing_claim.device_id,
                        process_id=process_id,
                        process_start_identity=process_start_identity,
                        action="start replay",
                    )
                    return {
                        "status": "STARTED",
                        "idempotent_replay": True,
                        "attempt_created": False,
                        "effect_executed": False,
                        "bundle": bundle,
                        "bundle_digest": existing_head["bundle_digest"],
                        "bundle_path": str(bundle_path),
                    }
                preparation = self._preparation(attempt_id)
                claim = LaunchpadAttemptClaimV1.from_dict(dict(preparation["claim"]))
                self._require_current_scheduler_owner(claim)
                resource = LaunchpadResourceStartLeaseV1.from_dict(
                    dict(preparation["resource_start_lease"])
                )
                continuation = dict(preparation["continuation_lease"])
                effect_path = self.start_effects_dir / f"{_safe_key(attempt_id)}.json"
                if effect_path.exists():
                    effect = read_json(effect_path, name="LaunchpadAttemptStartEffect")
                    if (
                        effect.get("process_owner_id") != process_owner_id
                        or _digest(
                            effect.get("process_owner_digest"),
                            field="process_owner_digest",
                        )
                        != process_owner_digest
                        or (
                            effect.get("schema_version")
                            == START_EFFECT_SCHEMA_VERSION_V2
                            and (
                                effect.get("process_id") != process_id
                                or effect.get("process_start_identity")
                                != process_start_identity
                            )
                        )
                    ):
                        raise ActorError(
                            "start replay uses another process owner",
                            code="launchpad_attempt_replay",
                        )
                    _require_exact_gpu_process_owner_v2(
                        bundle={"start_effect": effect},
                        resource=resource,
                        device_id=claim.device_id,
                        process_id=process_id,
                        process_start_identity=process_start_identity,
                        action="partial start replay",
                    )
                    return self._finalize_start(preparation, effect)
                if not resource.is_startable_at(at_utc):
                    raise ActorError(
                        "Launchpad resource start window is stale",
                        code="launchpad_attempt_stale",
                    )
                observed_at = _parse_utc(
                    preparation["resource_observed_at_utc"],
                    field="resource_observed_at_utc",
                )
                if (
                    _parse_utc(at_utc, field="at_utc") - observed_at
                ).total_seconds() > int(preparation["max_observation_age_seconds"]):
                    raise ActorError(
                        "resource observation became stale before start",
                        code="launchpad_attempt_stale",
                    )
                if self.resource_ledger.events(resource.lease_id):
                    raise ActorError(
                        "resource-start lease is no longer startable",
                        code="launchpad_attempt_stale",
                    )
                leases, _events, head = self._continuation_state(
                    claim.lineage_scope_digest
                )
                if (
                    head is None
                    or _continuation_lease_digest(leases[-1])
                    != _continuation_lease_digest(continuation)
                    or not (
                        _parse_utc(
                            continuation["granted_at_utc"], field="granted_at_utc"
                        )
                        <= _parse_utc(at_utc, field="at_utc")
                        < _parse_utc(
                            continuation["expires_at_utc"], field="expires_at_utc"
                        )
                    )
                ):
                    raise ActorError(
                        "canonical continuation head is stale or fenced",
                        code="launchpad_attempt_stale",
                    )
                if (
                    _gpu_resource_kind_v1(
                        resource.resource_ids, device_id=claim.device_id
                    )
                    is not None
                ):
                    if (
                        process_owner_id != claim.execution_owner_id
                        or process_owner_digest != claim.execution_owner_digest
                    ):
                        raise ActorError(
                            "start owner differs from execution owner",
                            code="launchpad_attempt_owner_mismatch",
                        )
                    current_pid = os.getpid()
                    current_start_identity = observe_process_start_identity(current_pid)
                    if (
                        process_id != current_pid
                        or current_start_identity is None
                        or process_start_identity != current_start_identity
                    ):
                        raise ActorError(
                            "GPU start requires the exact Linux process identity "
                            "observed by Scheduler",
                            code="launchpad_attempt_owner_mismatch",
                        )
                effect = self._start_effect(
                    preparation,
                    process_owner_id=process_owner_id,
                    process_owner_digest=process_owner_digest,
                    process_id=process_id,
                    process_start_identity=process_start_identity,
                    at_utc=at_utc,
                )
                _write_immutable(
                    effect_path, effect, label="LaunchpadAttemptStartEffect"
                )
                return self._finalize_start(preparation, effect)
            except ActorError as error:
                self._quarantine(
                    attempt_id=attempt_id,
                    reason=f"{error.code}:{error}",
                    at_utc=at_utc,
                    details={
                        "process_owner_id": process_owner_id,
                        "process_owner_digest": process_owner_digest.to_dict(),
                    },
                )
                raise

    def heartbeat_or_renew(
        self,
        *,
        attempt_id: str,
        process_owner_id: str,
        process_owner_digest: ContentDigest,
        process_id: int | None = None,
        process_start_identity: str | None = None,
        at_utc: str,
        renew_for_seconds: int | None = None,
    ) -> dict[str, Any]:
        with _scheduler_execution_lock(self.state_root):
            preparation = self._preparation(attempt_id)
            claim = LaunchpadAttemptClaimV1.from_dict(dict(preparation["claim"]))
            self._require_current_scheduler_owner(claim)
            if (
                process_owner_id != claim.execution_owner_id
                or process_owner_digest != claim.execution_owner_digest
            ):
                raise ActorError(
                    "heartbeat owner differs from execution owner",
                    code="launchpad_attempt_owner_mismatch",
                )
            bundle = self._authoritative_bundle(attempt_id)
            if dict(bundle.get("claim") or {}) != claim.to_dict():
                raise ActorError(
                    "started bundle differs from the prepared claim",
                    code="launchpad_attempt_binding_mismatch",
                )
            resource = LaunchpadResourceStartLeaseV1.from_dict(
                dict(preparation["resource_start_lease"])
            )
            _require_exact_gpu_process_owner_v2(
                bundle=bundle,
                resource=resource,
                device_id=claim.device_id,
                process_id=process_id,
                process_start_identity=process_start_identity,
                action="heartbeat or renewal",
            )
            terminal_head = self._read_pointer(
                self._terminal_head_path(attempt_id),
                label="Launchpad terminal head",
            )
            if (
                terminal_head is not None
                or self._terminal_intent_path(attempt_id).exists()
            ):
                raise ActorError(
                    "heartbeat or renewal follows terminalization",
                    code="launchpad_attempt_terminal",
                )
            resource_events = self.resource_ledger.events(resource.lease_id)
            if (
                len(resource_events) != 1
                or resource_events[0].get("code")
                != ResourceStartLeaseEventCodeV1.STARTED.value
                or resource_events[0].get("lease_digest")
                != resource.lease_digest().to_dict()
                or resource_events[0].get("fencing_token_digest")
                != resource.fencing_token_digest.to_dict()
            ):
                raise ActorError(
                    "heartbeat lacks active running resource occupancy",
                    code="launchpad_attempt_terminal",
                )
            leases, events, head = self._continuation_state(claim.lineage_scope_digest)
            if head is None:
                raise ActorError(
                    "continuation head is missing", code="launchpad_attempt_invalid"
                )
            if (
                _continuation_ledger_state(leases, events)
                is not _ContinuationStateV1.ACTIVE
            ):
                raise ActorError(
                    "heartbeat or renewal requires ACTIVE continuation state",
                    code="launchpad_attempt_terminal",
                )
            current = leases[-1]
            if (
                current["attempt_id"] != attempt_id
                or current["owner_id"] != process_owner_id
            ):
                raise ActorError(
                    "heartbeat holds a stale continuation fence",
                    code="launchpad_attempt_stale",
                )
            if not (
                _parse_utc(current["granted_at_utc"], field="granted_at_utc")
                <= _parse_utc(at_utc, field="at_utc")
                < _parse_utc(current["expires_at_utc"], field="expires_at_utc")
            ):
                raise ActorError(
                    "continuation lease expired before heartbeat",
                    code="launchpad_attempt_stale",
                )
            if events and events[-1].get("event_at_utc") == at_utc:
                latest = events[-1]
                if renew_for_seconds is None and (
                    latest.get("code") == "HEARTBEAT"
                    and _digest(latest.get("lease_digest"), field="lease_digest")
                    == _continuation_lease_digest(current)
                ):
                    return {
                        "status": "HEARTBEAT",
                        "idempotent_replay": True,
                        "event": latest,
                    }
                if renew_for_seconds is not None and (
                    latest.get("code") == "RENEWED"
                    and _digest(
                        latest.get("successor_lease_digest"),
                        field="successor_lease_digest",
                    )
                    == _continuation_lease_digest(current)
                    and current.get("granted_at_utc") == at_utc
                    and current.get("expires_at_utc")
                    == _add_seconds(at_utc, renew_for_seconds)
                ):
                    return {
                        "status": "RENEWED",
                        "idempotent_replay": True,
                        "event": latest,
                        "lease": current,
                    }
                raise ActorError(
                    "continuation operation conflicts at the same timestamp",
                    code="launchpad_attempt_replay",
                )
            prior_event = None if not events else _continuation_event_digest(events[-1])
            if renew_for_seconds is None:
                event = _continuation_event(
                    current,
                    code="HEARTBEAT",
                    event_at_utc=at_utc,
                    reason="Scheduler observed exact Launchpad owner heartbeat",
                    prior_event_digest=prior_event,
                )
                leases, events = self._append_continuation_event(
                    claim.lineage_scope_digest, leases, events, event
                )
                return {
                    "status": "HEARTBEAT",
                    "idempotent_replay": False,
                    "event": event,
                }
            successor_seed = {
                "prior_lease_digest": _continuation_lease_digest(current).to_dict(),
                "lease_epoch": int(current["lease_epoch"]) + 1,
                "granted_at_utc": at_utc,
                "owner_id": process_owner_id,
            }
            successor_digest_seed = canonical_sha256(successor_seed)
            successor = {
                **current,
                "lease_id": f"LP-CONT-{successor_digest_seed.hex[:24]}",
                "lease_epoch": int(current["lease_epoch"]) + 1,
                "fencing_token_digest": canonical_sha256(
                    {
                        "successor_seed": successor_seed,
                        "execution_owner_digest": process_owner_digest.to_dict(),
                    }
                ).to_dict(),
                "prior_lease_digest": _continuation_lease_digest(current).to_dict(),
                "granted_at_utc": at_utc,
                "expires_at_utc": _add_seconds(at_utc, renew_for_seconds),
            }
            successor_digest = _continuation_lease_digest(successor)
            event = _continuation_event(
                current,
                code="RENEWED",
                event_at_utc=at_utc,
                reason="Scheduler renewed exact Launchpad continuation owner",
                prior_event_digest=prior_event,
                successor_lease_digest=successor_digest,
            )
            leases, events = self._append_continuation_event(
                claim.lineage_scope_digest,
                leases,
                events,
                event,
                successor=successor,
            )
            return {
                "status": "RENEWED",
                "idempotent_replay": False,
                "event": event,
                "lease": successor,
            }

    def terminalize(
        self,
        *,
        attempt_id: str,
        process_owner_id: str,
        process_owner_digest: ContentDigest,
        process_id: int | None = None,
        process_start_identity: str | None = None,
        terminal: Mapping[str, Any],
        at_utc: str,
        publication_digest: ContentDigest | None = None,
        _reconcile_durable_intent: bool = False,
    ) -> dict[str, Any]:
        with _scheduler_execution_lock(self.state_root):
            preparation = self._preparation(attempt_id)
            claim = LaunchpadAttemptClaimV1.from_dict(dict(preparation["claim"]))
            self._require_current_scheduler_owner(claim)
            if (
                claim.execution_owner_id != process_owner_id
                or claim.execution_owner_digest != process_owner_digest
            ):
                raise ActorError(
                    "terminal owner differs from execution owner",
                    code="launchpad_attempt_owner_mismatch",
                )
            bundle = self._authoritative_bundle(attempt_id)
            resource = LaunchpadResourceStartLeaseV1.from_dict(
                dict(preparation["resource_start_lease"])
            )
            verified_process_id, verified_process_start_identity = (
                _require_exact_gpu_process_owner_v2(
                    bundle=bundle,
                    resource=resource,
                    device_id=claim.device_id,
                    process_id=process_id,
                    process_start_identity=process_start_identity,
                    action="terminalization",
                    allow_durable_intent_recovery=_reconcile_durable_intent,
                )
            )
            terminal_raw = _attempt_terminal_v1(
                dict(terminal),
                attempt_id=attempt_id,
                start=dict(bundle.get("attempt_start") or {}),
                run=dict(bundle.get("run_occurrence") or {}),
                semantic_spec=dict(bundle.get("semantic_run_spec") or {}),
            )
            terminal_head = self._read_pointer(
                self._terminal_head_path(attempt_id),
                label="Launchpad terminal head",
            )
            if terminal_head is not None:
                path = Path(str(terminal_head.get("terminal_bundle_path") or ""))
                terminal_bundle = read_json(path, name="LaunchpadTerminalBundle")
                if (
                    terminal_head.get("terminal_bundle_digest")
                    != _record_digest(terminal_bundle).to_dict()
                    or terminal_bundle.get("attempt_terminal") != terminal_raw
                    or terminal_bundle.get("terminalized_at_utc") != at_utc
                    or _optional_digest(
                        dict(terminal_bundle.get("continuation_event") or {}).get(
                            "publication_digest"
                        ),
                        field="publication_digest",
                    )
                    != publication_digest
                ):
                    self._quarantine(
                        attempt_id=attempt_id,
                        reason="terminal replay differs from committed outcome",
                        at_utc=at_utc,
                        details=terminal_raw,
                    )
                    raise ActorError(
                        "terminal replay differs from the committed outcome",
                        code="launchpad_attempt_replay",
                    )
                self._release_resource_occupancy(
                    preparation=preparation,
                    at_utc=at_utc,
                )
                return {
                    "status": "TERMINAL",
                    "idempotent_replay": True,
                    "terminal_bundle": terminal_bundle,
                    "terminal_bundle_digest": terminal_head["terminal_bundle_digest"],
                    "terminal_bundle_path": str(path),
                }
            terminal_digest = _record_digest(terminal_raw)
            terminal_path = (
                self.terminals_dir
                / _safe_key(attempt_id)
                / f"{terminal_digest.hex}.json"
            )
            existing_terminals = sorted(terminal_path.parent.glob("*.json"))
            if existing_terminals and (
                len(existing_terminals) != 1
                or not _same_payload(existing_terminals[0], terminal_raw)
            ):
                self._quarantine(
                    attempt_id=attempt_id,
                    reason="conflicting Attempt terminal",
                    at_utc=at_utc,
                    details=terminal_raw,
                )
                raise ActorError(
                    "Attempt already has another terminal outcome",
                    code="launchpad_attempt_replay",
                )
            gpu_terminal = (
                _gpu_resource_kind_v1(resource.resource_ids, device_id=claim.device_id)
                is not None
            )
            terminal_intent = {
                "schema_version": (
                    TERMINAL_INTENT_SCHEMA_VERSION_V2
                    if gpu_terminal
                    else TERMINAL_INTENT_SCHEMA_VERSION
                ),
                "attempt_id": attempt_id,
                "process_owner_id": process_owner_id,
                "process_owner_digest": process_owner_digest.to_dict(),
                "attempt_terminal": terminal_raw,
                "terminalized_at_utc": at_utc,
                "publication_digest": (
                    None if publication_digest is None else publication_digest.to_dict()
                ),
            }
            if gpu_terminal:
                terminal_intent["process_id"] = verified_process_id
                terminal_intent["process_start_identity"] = (
                    verified_process_start_identity
                )
            if _reconcile_durable_intent and not _same_payload(
                self._terminal_intent_path(attempt_id), terminal_intent
            ):
                raise ActorError(
                    "terminal recovery requires the exact pre-existing durable intent",
                    code="launchpad_attempt_owner_mismatch",
                )
            leases, events, head = self._continuation_state(claim.lineage_scope_digest)
            assert head is not None
            continuation_state = _continuation_ledger_state(leases, events)
            intended_terminal_state = (
                _ContinuationStateV1.PUBLISHED
                if publication_digest is not None
                else _ContinuationStateV1.RELEASED
            )
            replaying_durable_terminal = (
                continuation_state is intended_terminal_state
                and _same_payload(
                    self._terminal_intent_path(attempt_id), terminal_intent
                )
            )
            if (
                continuation_state is not _ContinuationStateV1.ACTIVE
                and not replaying_durable_terminal
            ):
                raise ActorError(
                    "terminalization requires ACTIVE continuation state",
                    code="launchpad_attempt_terminal",
                )
            current = leases[-1]
            if (
                current["attempt_id"] != attempt_id
                or current["owner_id"] != process_owner_id
            ):
                self._quarantine(
                    attempt_id=attempt_id,
                    reason="stale publisher or terminal owner",
                    at_utc=at_utc,
                    details=terminal_raw,
                )
                raise ActorError(
                    "terminal publisher holds a stale continuation fence",
                    code="launchpad_attempt_quarantine_required",
                )
            if not (
                _parse_utc(current["granted_at_utc"], field="granted_at_utc")
                <= _parse_utc(at_utc, field="at_utc")
                < _parse_utc(current["expires_at_utc"], field="expires_at_utc")
            ):
                self._quarantine(
                    attempt_id=attempt_id,
                    reason="expired publisher continuation lease",
                    at_utc=at_utc,
                    details=terminal_raw,
                )
                raise ActorError(
                    "terminal publisher continuation lease is expired",
                    code="launchpad_attempt_quarantine_required",
                )
            _write_immutable(
                self._terminal_intent_path(attempt_id),
                terminal_intent,
                label="LaunchpadAttemptTerminalIntent",
            )
            _write_immutable(terminal_path, terminal_raw, label="AttemptTerminalV1")
            terminal_codes = {"PUBLISH_COMMITTED", "RELEASED"}
            prior_event = None if not events else _continuation_event_digest(events[-1])
            if events and events[-1].get("code") in terminal_codes:
                prior_event = (
                    None if len(events) == 1 else _continuation_event_digest(events[-2])
                )
            event = _continuation_event(
                current,
                code="PUBLISH_COMMITTED"
                if publication_digest is not None
                else "RELEASED",
                event_at_utc=at_utc,
                reason=(
                    "Scheduler committed exact canonical Launchpad publication"
                    if publication_digest is not None
                    else "Scheduler terminalized exact Launchpad Attempt"
                ),
                prior_event_digest=prior_event,
                publication_digest=publication_digest,
            )
            if events and events[-1].get("code") in terminal_codes:
                if events[-1] != event:
                    self._quarantine(
                        attempt_id=attempt_id,
                        reason="terminal continuation event replay differs",
                        at_utc=at_utc,
                        details=terminal_raw,
                    )
                    raise ActorError(
                        "terminal continuation event already records another outcome",
                        code="launchpad_attempt_replay",
                    )
            else:
                leases, events = self._append_continuation_event(
                    claim.lineage_scope_digest, leases, events, event
                )
            terminal_bundle = {
                "schema_version": TERMINAL_BUNDLE_SCHEMA_VERSION,
                "attempt_id": attempt_id,
                "authoritative_bundle_digest": _record_digest(bundle).to_dict(),
                "attempt_terminal": terminal_raw,
                "continuation_event": event,
                "terminalized_at_utc": at_utc,
            }
            terminal_bundle_digest = _record_digest(terminal_bundle)
            path = (
                self.terminal_bundles_dir
                / _safe_key(attempt_id)
                / f"{terminal_bundle_digest.hex}.json"
            )
            _write_immutable(path, terminal_bundle, label="LaunchpadTerminalBundle")
            atomic_write_json(
                self._terminal_head_path(attempt_id),
                {
                    "schema_version": "crca_launchpad_attempt_terminal_head_v1",
                    "attempt_id": attempt_id,
                    "terminal_bundle_digest": terminal_bundle_digest.to_dict(),
                    "terminal_bundle_path": str(path),
                },
            )
            # Occupancy is released only after the terminal continuation event,
            # bundle, and head are durable.  A crash before this point keeps the
            # GPU fenced; a crash after it is repaired from the authoritative
            # terminal head by reconciliation.
            self._release_resource_occupancy(
                preparation=preparation,
                at_utc=at_utc,
            )
            return {
                "status": "TERMINAL",
                "idempotent_replay": False,
                "terminal_bundle": terminal_bundle,
                "terminal_bundle_digest": terminal_bundle_digest.to_dict(),
                "terminal_bundle_path": str(path),
            }

    def _release_resource_occupancy(
        self,
        *,
        preparation: Mapping[str, Any],
        at_utc: str,
    ) -> ResourceStartLeaseEventV1:
        """Release one started Launchpad occupancy exactly once at closeout."""

        resource = LaunchpadResourceStartLeaseV1.from_dict(
            dict(preparation["resource_start_lease"])
        )
        events = self.resource_ledger.events(resource.lease_id)
        if not events or events[0].get("code") != (
            ResourceStartLeaseEventCodeV1.STARTED.value
        ):
            raise ActorError(
                "terminalization lacks a started resource occupancy",
                code="launchpad_attempt_quarantine_required",
            )
        release = ResourceStartLeaseEventV1(
            lease_digest=resource.lease_digest(),
            code=ResourceStartLeaseEventCodeV1.RELEASED,
            scheduler_owner_id=resource.scheduler_owner_id,
            fencing_token_digest=resource.fencing_token_digest,
            event_at_utc=at_utc,
            reason="Scheduler terminalized Launchpad running resource occupancy",
            prior_event_digest=canonical_sha256(events[0]),
            successor_lease_digest=None,
        )
        if len(events) == 2:
            existing = dict(events[-1])
            legacy_reconciliation_reason = (
                "Scheduler reconciled an immutable historical Launchpad "
                "terminal resource occupancy"
            )
            equivalent_legacy_release = existing.get(
                "reason"
            ) == legacy_reconciliation_reason and {
                key: value for key, value in existing.items() if key != "reason"
            } == {
                key: value
                for key, value in release.to_dict().items()
                if key != "reason"
            }
            if existing != release.to_dict() and not equivalent_legacy_release:
                raise ActorError(
                    "resource occupancy already has another terminal outcome",
                    code="launchpad_attempt_quarantine_required",
                )
            return release
        if len(events) != 1:
            raise ActorError(
                "resource occupancy event chain is invalid at terminalization",
                code="launchpad_attempt_quarantine_required",
            )
        self.resource_ledger.append_event(resource.lease_id, release)
        return release

    def reconcile_terminal_resource_occupancies_v1(
        self,
        *,
        at_utc: str,
        conflicting_resource_ids: tuple[str, ...] | None = None,
    ) -> dict[str, Any]:
        """Repair the historical terminal-head/resource-event split safely.

        Older Launchpad terminal publication did not append the resource
        ``RELEASED`` event.  The terminal head, bundle, intent, authoritative
        start bundle, and exact lease are all immutable, so their agreement is
        sufficient to append only the missing terminal resource event.  No
        historical record is rewritten or deleted.
        """

        _parse_utc(at_utc, field="at_utc")
        released: list[dict[str, Any]] = []
        receipts: list[dict[str, Any]] = []
        skipped: list[dict[str, str]] = []
        with _scheduler_execution_lock(self.state_root):
            for head_path in sorted(self.terminal_heads_dir.glob("*.json")):
                try:
                    head = read_json(head_path, name="Launchpad terminal head")
                    attempt_id = str(head.get("attempt_id") or "")
                    preparation = self._preparation(attempt_id)
                    resource = LaunchpadResourceStartLeaseV1.from_dict(
                        dict(preparation.get("resource_start_lease") or {})
                    )
                    if conflicting_resource_ids is not None and not (
                        resource_ids_conflict_v1(
                            resource.resource_ids, conflicting_resource_ids
                        )
                    ):
                        continue
                    resource_events = self.resource_ledger.events(resource.lease_id)
                    if (
                        not resource_events
                        or resource_events[0].get("code")
                        != ResourceStartLeaseEventCodeV1.STARTED.value
                        or len(resource_events) > 2
                        or (
                            len(resource_events) == 2
                            and resource_events[1].get("code")
                            != ResourceStartLeaseEventCodeV1.RELEASED.value
                        )
                    ):
                        continue
                    release_was_missing = len(resource_events) == 1
                    terminal_path = (
                        Path(str(head.get("terminal_bundle_path") or ""))
                        .expanduser()
                        .resolve()
                    )
                    try:
                        terminal_path.relative_to(
                            self.terminal_bundles_dir.expanduser().resolve()
                        )
                    except ValueError as error:
                        raise ActorError(
                            "terminal bundle lies outside Scheduler authority",
                            code="launchpad_attempt_quarantine_required",
                        ) from error
                    terminal_bundle = read_json(
                        terminal_path, name="Launchpad terminal bundle"
                    )
                    terminal_intent = read_json(
                        self._terminal_intent_path(attempt_id),
                        name="Launchpad terminal intent",
                    )
                    authoritative = self._authoritative_bundle(attempt_id)
                    claim = LaunchpadAttemptClaimV1.from_dict(
                        dict(authoritative.get("claim") or {})
                    )
                    terminalized_at = str(
                        terminal_bundle.get("terminalized_at_utc") or ""
                    )
                    terminal = dict(terminal_bundle.get("attempt_terminal") or {})
                    continuation_event = dict(
                        terminal_bundle.get("continuation_event") or {}
                    )
                    terminal_bundle_digest = _record_digest(terminal_bundle)
                    expected_terminal_path = (
                        self.terminal_bundles_dir
                        / _safe_key(attempt_id)
                        / f"{terminal_bundle_digest.hex}.json"
                    )
                    leases, continuation_events, continuation_head = (
                        self._continuation_state(claim.lineage_scope_digest)
                    )
                    continuation_event_digest = _continuation_event_digest(
                        continuation_event
                    )
                    canonical_event_matches = [
                        event
                        for event in continuation_events
                        if _continuation_event_digest(event)
                        == continuation_event_digest
                    ]
                    start_effect = dict(authoritative.get("start_effect") or {})
                    start_effect_digest = _record_digest(start_effect)
                    gpu_resource = (
                        _gpu_resource_kind_v1(
                            resource.resource_ids, device_id=claim.device_id
                        )
                        is not None
                    )
                    expected_terminal_intent_schema = (
                        TERMINAL_INTENT_SCHEMA_VERSION_V2
                        if gpu_resource
                        and start_effect.get("schema_version")
                        == START_EFFECT_SCHEMA_VERSION_V2
                        else TERMINAL_INTENT_SCHEMA_VERSION
                    )
                    expected_terminal_intent_fields = {
                        "schema_version",
                        "attempt_id",
                        "process_owner_id",
                        "process_owner_digest",
                        "attempt_terminal",
                        "terminalized_at_utc",
                        "publication_digest",
                    }
                    if (
                        expected_terminal_intent_schema
                        == TERMINAL_INTENT_SCHEMA_VERSION_V2
                    ):
                        expected_terminal_intent_fields.update(
                            {"process_id", "process_start_identity"}
                        )
                    start_event_at = str(resource_events[0].get("event_at_utc") or "")
                    if (
                        head.get("schema_version")
                        != "crca_launchpad_attempt_terminal_head_v1"
                        or head.get("attempt_id") != attempt_id
                        or head.get("terminal_bundle_digest")
                        != terminal_bundle_digest.to_dict()
                        or terminal_path != expected_terminal_path.resolve()
                        or terminal_bundle.get("schema_version")
                        != TERMINAL_BUNDLE_SCHEMA_VERSION
                        or terminal_bundle.get("attempt_id") != attempt_id
                        or terminal.get("attempt_id") != attempt_id
                        or terminal_intent.get("attempt_id") != attempt_id
                        or terminal_intent.get("schema_version")
                        != expected_terminal_intent_schema
                        or set(terminal_intent) != expected_terminal_intent_fields
                        or terminal_intent.get("attempt_terminal") != terminal
                        or terminal_intent.get("terminalized_at_utc") != terminalized_at
                        or terminal_intent.get("publication_digest")
                        != continuation_event.get("publication_digest")
                        or terminal_bundle.get("authoritative_bundle_digest")
                        != _record_digest(authoritative).to_dict()
                        or authoritative.get("resource_start_lease")
                        != resource.to_dict()
                        or start_effect.get("resource_start_lease_digest")
                        != resource.lease_digest().to_dict()
                        or start_effect.get("attempt_id") != attempt_id
                        or continuation_head is None
                        or len(canonical_event_matches) != 1
                        or canonical_event_matches[0] != continuation_event
                        or continuation_event.get("code")
                        not in {"RELEASED", "PUBLISH_COMMITTED"}
                        or continuation_event.get("event_at_utc") != terminalized_at
                        or continuation_event.get("attempt_id") != attempt_id
                        or (
                            expected_terminal_intent_schema
                            == TERMINAL_INTENT_SCHEMA_VERSION_V2
                            and (
                                terminal_intent.get("process_id")
                                != start_effect.get("process_id")
                                or terminal_intent.get("process_start_identity")
                                != start_effect.get("process_start_identity")
                            )
                        )
                    ):
                        raise ActorError(
                            "historical terminal resource binding is inconsistent",
                            code="launchpad_attempt_quarantine_required",
                        )
                    started_at_value = _parse_utc(
                        start_event_at, field="resource_started_at_utc"
                    )
                    terminalized_at_value = _parse_utc(
                        terminalized_at, field="terminalized_at_utc"
                    )
                    reconciled_at_value = _parse_utc(at_utc, field="at_utc")
                    if not (
                        started_at_value <= terminalized_at_value <= reconciled_at_value
                    ):
                        raise ActorError(
                            "historical terminal resource times are not causal",
                            code="launchpad_attempt_quarantine_required",
                        )
                    release = ResourceStartLeaseEventV1(
                        lease_digest=resource.lease_digest(),
                        code=ResourceStartLeaseEventCodeV1.RELEASED,
                        scheduler_owner_id=resource.scheduler_owner_id,
                        fencing_token_digest=resource.fencing_token_digest,
                        event_at_utc=terminalized_at,
                        reason=(
                            "Scheduler terminalized Launchpad running resource "
                            "occupancy"
                        ),
                        prior_event_digest=canonical_sha256(resource_events[0]),
                        successor_lease_digest=None,
                    )
                    release_event = release.to_dict()
                    if not release_was_missing:
                        existing_release = dict(resource_events[1])
                        allowed_reasons = {
                            release.reason,
                            (
                                "Scheduler reconciled an immutable historical "
                                "Launchpad terminal resource occupancy"
                            ),
                        }
                        if existing_release.get("reason") not in allowed_reasons or {
                            key: value
                            for key, value in existing_release.items()
                            if key != "reason"
                        } != {
                            key: value
                            for key, value in release_event.items()
                            if key != "reason"
                        }:
                            raise ActorError(
                                "historical terminal release event is inconsistent",
                                code="launchpad_attempt_quarantine_required",
                            )
                        release_event = existing_release
                    existing_receipt_paths = sorted(
                        (
                            self.terminal_resource_reconciliations_dir
                            / _safe_key(attempt_id)
                        ).glob("*.json")
                    )
                    if len(existing_receipt_paths) > 1:
                        raise ActorError(
                            "terminal resource reconciliation has multiple receipts",
                            code="launchpad_attempt_quarantine_required",
                        )
                    reconciliation_static = {
                        "schema_version": (
                            "crca_terminal_resource_occupancy_reconciliation_receipt_v1"
                        ),
                        "attempt_id": attempt_id,
                        "terminal_head_path": str(head_path.resolve()),
                        "terminal_head_digest": _record_digest(head).to_dict(),
                        "terminal_bundle_path": str(terminal_path),
                        "terminal_bundle_digest": terminal_bundle_digest.to_dict(),
                        "terminal_intent_digest": _record_digest(
                            terminal_intent
                        ).to_dict(),
                        "authoritative_bundle_digest": _record_digest(
                            authoritative
                        ).to_dict(),
                        "start_effect_digest": start_effect_digest.to_dict(),
                        "resource_lease_digest": resource.lease_digest().to_dict(),
                        "resource_started_event_digest": canonical_sha256(
                            resource_events[0]
                        ).to_dict(),
                        "continuation_event_digest": (
                            continuation_event_digest.to_dict()
                        ),
                        "release_event_digest": canonical_sha256(
                            release_event
                        ).to_dict(),
                    }
                    if existing_receipt_paths:
                        reconciliation_path = existing_receipt_paths[0]
                        reconciliation_core = read_json(
                            reconciliation_path,
                            name="TerminalResourceOccupancyReconciliationReceipt",
                        )
                        reconciliation_digest = _record_digest(reconciliation_core)
                        reconciled_at = _parse_utc(
                            reconciliation_core.get("reconciled_at_utc"),
                            field="reconciled_at_utc",
                        )
                        if (
                            set(reconciliation_core)
                            != {*reconciliation_static, "reconciled_at_utc"}
                            or {
                                key: value
                                for key, value in reconciliation_core.items()
                                if key != "reconciled_at_utc"
                            }
                            != reconciliation_static
                            or reconciliation_path.name
                            != f"{reconciliation_digest.hex}.json"
                            or reconciled_at < terminalized_at_value
                            or reconciled_at > reconciled_at_value
                        ):
                            raise ActorError(
                                "terminal resource reconciliation receipt is invalid",
                                code="launchpad_attempt_quarantine_required",
                            )
                    else:
                        reconciliation_core = {
                            **reconciliation_static,
                            "reconciled_at_utc": at_utc,
                        }
                        reconciliation_digest = _record_digest(reconciliation_core)
                        reconciliation_path = (
                            self.terminal_resource_reconciliations_dir
                            / _safe_key(attempt_id)
                            / f"{reconciliation_digest.hex}.json"
                        )
                        _write_immutable(
                            reconciliation_path,
                            reconciliation_core,
                            label="TerminalResourceOccupancyReconciliationReceipt",
                        )
                    # The receipt is the durable authority for this historical
                    # repair.  Seal and validate it before releasing occupancy;
                    # a crash in either direction is therefore replay-safe.
                    if release_was_missing:
                        self.resource_ledger.append_event(resource.lease_id, release)
                    row = {
                        "attempt_id": attempt_id,
                        "resource_lease_id": resource.lease_id,
                        "event": release_event,
                        "reconciliation_receipt_path": str(reconciliation_path),
                        "reconciliation_receipt_digest": (
                            reconciliation_digest.to_dict()
                        ),
                    }
                    receipts.append(row)
                    if release_was_missing:
                        released.append(row)
                except (ActorError, ProgrammeContractError, OSError) as error:
                    skipped.append(
                        {
                            "terminal_head": str(head_path),
                            "reason": str(error),
                        }
                    )
        return {
            "schema_version": ("crca_terminal_resource_occupancy_reconciliation_v1"),
            "observed_at_utc": at_utc,
            "released_count": len(released),
            "released": released,
            "receipt_count": len(receipts),
            "receipts": receipts,
            "skipped": skipped,
        }

    def dead_gpu_reclaim_candidates_v1(
        self,
        *,
        at_utc: str,
        conflicting_resource_ids: tuple[str, ...] | None = None,
    ) -> list[dict[str, Any]]:
        """Fence expired continuations and list proof-eligible GPU owners.

        Expiry alone never changes resource occupancy.  A separate exact
        Resource Manager proof is required by
        :meth:`reclaim_dead_gpu_occupancy_v1`.
        """

        candidates: list[dict[str, Any]] = []
        with _scheduler_execution_lock(self.state_root):
            for head_path in sorted(self.bundle_heads_dir.glob("*.json")):
                head = self._read_pointer(head_path, label="Launchpad bundle head")
                if head is None:
                    continue
                attempt_id = str(head.get("attempt_id") or "")
                bundle = self._authoritative_bundle(attempt_id)
                claim = LaunchpadAttemptClaimV1.from_dict(
                    dict(bundle.get("claim") or {})
                )
                resource = LaunchpadResourceStartLeaseV1.from_dict(
                    dict(bundle.get("resource_start_lease") or {})
                )
                resource_kind = _gpu_resource_kind_v1(
                    resource.resource_ids, device_id=claim.device_id
                )
                if resource_kind is None or (
                    conflicting_resource_ids is not None
                    and not resource_ids_conflict_v1(
                        resource.resource_ids, conflicting_resource_ids
                    )
                ):
                    continue
                resource_events = self.resource_ledger.events(resource.lease_id)
                if (
                    len(resource_events) != 1
                    or resource_events[0].get("code")
                    != ResourceStartLeaseEventCodeV1.STARTED.value
                    or self._terminal_head_path(attempt_id).exists()
                    or self._terminal_intent_path(attempt_id).exists()
                ):
                    continue
                leases, events, continuation_head = self._continuation_state(
                    claim.lineage_scope_digest
                )
                if continuation_head is None:
                    continue
                effect = dict(bundle.get("start_effect") or {})
                effect_schema = effect.get("schema_version")
                # PID-less V1 starts are historical migration inputs only.  The
                # automatic admission path may reclaim only an exact V2 process
                # owner.
                if effect_schema != START_EFFECT_SCHEMA_VERSION_V2:
                    continue
                attempt_continuation_leases = [
                    lease
                    for lease in leases
                    if lease.get("attempt_id") == attempt_id
                    and lease.get("owner_id") == claim.execution_owner_id
                ]
                if not attempt_continuation_leases:
                    continue
                selected_continuation = attempt_continuation_leases[-1]
                continuation_lease_digest = _continuation_lease_digest(
                    selected_continuation
                )
                if continuation_lease_digest != _continuation_lease_digest(leases[-1]):
                    continue
                current_state = _continuation_ledger_state(leases, events)
                if (
                    selected_continuation is leases[-1]
                    and current_state is _ContinuationStateV1.ACTIVE
                    and _parse_utc(at_utc, field="at_utc")
                    >= _parse_utc(
                        selected_continuation["expires_at_utc"],
                        field="continuation.expires_at_utc",
                    )
                ):
                    leases, events, _changed = self._expire_continuation_if_stale(
                        claim.lineage_scope_digest,
                        leases,
                        events,
                        at_utc=at_utc,
                    )
                    current_state = _continuation_ledger_state(leases, events)
                matching_terminal_events = [
                    event
                    for event in events
                    if event.get("code") in {"EXPIRED", "OWNER_DEAD"}
                    and _digest(event.get("lease_digest"), field="lease_digest")
                    == continuation_lease_digest
                ]
                if len(matching_terminal_events) == 1:
                    continuation_fence_state = str(matching_terminal_events[0]["code"])
                    continuation_terminal_event_digest = _continuation_event_digest(
                        matching_terminal_events[0]
                    ).to_dict()
                    continuation_fenced_at_utc = matching_terminal_events[0][
                        "event_at_utc"
                    ]
                    existing_reclaim_proof_digest = (
                        None
                        if continuation_fence_state != "OWNER_DEAD"
                        else _digest(
                            matching_terminal_events[0].get("owner_dead_proof_digest"),
                            field="owner_dead_proof_digest",
                        ).to_dict()
                    )
                elif (
                    not matching_terminal_events
                    and current_state is _ContinuationStateV1.ACTIVE
                ):
                    continuation_fence_state = "OWNER_DEAD_REQUIRED"
                    continuation_terminal_event_digest = None
                    continuation_fenced_at_utc = None
                    existing_reclaim_proof_digest = None
                else:
                    continue
                common = {
                    "attempt_id": attempt_id,
                    "device_uuid": claim.device_id,
                    "resource_kind": resource_kind,
                    "resource_lease_id": resource.lease_id,
                    "resource_lease_digest": resource.lease_digest().to_dict(),
                    "selected_resource_ids": list(resource.resource_ids),
                    "fencing_token_digest": (resource.fencing_token_digest.to_dict()),
                    "continuation_lease_digest": (continuation_lease_digest.to_dict()),
                    "continuation_fence_state": continuation_fence_state,
                    "continuation_terminal_event_digest": (
                        continuation_terminal_event_digest
                    ),
                    "continuation_fenced_at_utc": continuation_fenced_at_utc,
                    "existing_reclaim_proof_digest": (existing_reclaim_proof_digest),
                    "resource_started_event_digest": canonical_sha256(
                        resource_events[0]
                    ).to_dict(),
                }
                if (
                    type(effect.get("process_id")) is not int
                    or not isinstance(effect.get("process_start_identity"), str)
                    or not str(effect["process_start_identity"]).startswith(
                        "linux-proc:"
                    )
                ):
                    continue
                candidates.append(
                    {
                        "schema_version": "crca_dead_gpu_reclaim_candidate_v2",
                        "reclaim_mode": "EXACT_PROCESS_IDENTITY",
                        **common,
                        "owner_pid": effect["process_id"],
                        "owner_process_start_identity": effect[
                            "process_start_identity"
                        ],
                    }
                )
        return candidates

    def _validate_legacy_gpu_migration_manifest_v1(
        self, manifest: Mapping[str, Any]
    ) -> tuple[ContentDigest, list[dict[str, Any]]]:
        """Validate the one bounded legacy-GPU cutover manifest."""

        value = dict(manifest)
        _require_exact_fields(
            value,
            {
                "schema_version",
                "migration_id",
                "canonical_state_root",
                "device_uuid",
                "cutover_commit",
                "eligible_row_count",
                "operator_exclusive_host_attestation",
                "rows",
                "manifest_sha256",
            },
            label="LegacyGpuMigrationManifestV1",
        )
        core = {key: row for key, row in value.items() if key != "manifest_sha256"}
        manifest_digest = _record_digest(core)
        rows_value = value.get("rows")
        operator_attestation = value.get("operator_exclusive_host_attestation")
        canonical_root = canonical_host_actor_state_root().resolve()
        is_canonical_root = self.state_root == canonical_root
        if (
            value.get("schema_version")
            != LEGACY_GPU_MIGRATION_MANIFEST_SCHEMA_VERSION
            or value.get("manifest_sha256") != manifest_digest.hex
            or value.get("canonical_state_root") != str(self.state_root)
            or not isinstance(value.get("migration_id"), str)
            or not str(value["migration_id"]).strip()
            or not isinstance(value.get("device_uuid"), str)
            or not str(value["device_uuid"]).strip()
            or not isinstance(value.get("cutover_commit"), str)
            or len(str(value["cutover_commit"])) != 40
            or any(
                character not in "0123456789abcdef"
                for character in str(value["cutover_commit"])
            )
            or not isinstance(rows_value, list)
            or isinstance(value.get("eligible_row_count"), bool)
            or value.get("eligible_row_count") != len(rows_value)
            or not rows_value
            or operator_attestation
            != {
                "schema_version": "eiel_operator_exclusive_host_attestation_v1",
                "attestation_id": "OPERATOR-EXCLUSIVE-CPU-GPU-20260827",
                "operator_statement": (
                    "i have stopped all other branches, the CPU/gpu are yours alone."
                ),
                "scope": "one-time canonical legacy GPU lease cutover",
            }
        ):
            raise ActorError(
                "legacy GPU migration manifest is malformed or misbound",
                code="launchpad_legacy_gpu_migration_denied",
            )
        if is_canonical_root:
            if (
                manifest_digest.hex
                != CANONICAL_LEGACY_GPU_MIGRATION_MANIFEST_SHA256
                or len(rows_value) != 7
            ):
                raise ActorError(
                    "canonical legacy GPU migration requires the pinned seven-row manifest",
                    code="launchpad_legacy_gpu_migration_denied",
                )
        elif not self._allow_test_shared_gpu_state_root:
            raise ActorError(
                "legacy GPU migration is limited to canonical authority or an explicit test root",
                code="launchpad_legacy_gpu_migration_denied",
            )

        row_fields = {
            "classification",
            "attempt_id",
            "run_id",
            "claim_digest",
            "start_effect_digest",
            "authoritative_bundle_digest",
            "resource_lease_id",
            "resource_lease_digest",
            "resource_ids",
            "resource_started_event_digest",
            "resource_expires_at_utc",
            "lineage_scope_digest",
            "attempt_continuation_lease_digest",
            "continuation_fence_code",
            "continuation_terminal_event_digest",
            "continuation_successor_event_digest",
            "device_uuid",
            "sealed_process_context",
            "row_sha256",
        }
        context_fields = {
            "relative_path",
            "file_sha256",
            "schema_version",
            "binding_sha256",
            "scheduler_bundle_digest",
            "owner_pid",
            "owner_process_start_ticks",
        }
        rows: list[dict[str, Any]] = []
        attempt_ids: set[str] = set()
        row_digests: set[str] = set()
        for raw_row in rows_value:
            if not isinstance(raw_row, Mapping):
                raise ActorError(
                    "legacy GPU migration row is not an object",
                    code="launchpad_legacy_gpu_migration_denied",
                )
            row = dict(raw_row)
            _require_exact_fields(
                row, row_fields, label="LegacyGpuMigrationManifestRowV1"
            )
            row_core = {key: item for key, item in row.items() if key != "row_sha256"}
            row_digest = _record_digest(row_core)
            classification = row.get("classification")
            context = row.get("sealed_process_context")
            for field in (
                "claim_digest",
                "start_effect_digest",
                "authoritative_bundle_digest",
                "resource_lease_digest",
                "resource_started_event_digest",
                "lineage_scope_digest",
                "attempt_continuation_lease_digest",
            ):
                _digest(row.get(field), field=f"migration_row.{field}")
            for field in (
                "continuation_terminal_event_digest",
                "continuation_successor_event_digest",
            ):
                _optional_digest(row.get(field), field=f"migration_row.{field}")
            if (
                row.get("row_sha256") != row_digest.hex
                or classification
                not in {"SEALED_PROCESS_CONTEXT", "TRUE_PIDLESS_DEVICE_IDLE"}
                or not isinstance(row.get("attempt_id"), str)
                or not str(row["attempt_id"]).strip()
                or not isinstance(row.get("run_id"), str)
                or not str(row["run_id"]).strip()
                or not isinstance(row.get("resource_lease_id"), str)
                or not str(row["resource_lease_id"]).strip()
                or not isinstance(row.get("resource_ids"), list)
                or not row["resource_ids"]
                or row.get("device_uuid") != value["device_uuid"]
                or row.get("continuation_fence_code") not in {"ACTIVE", "EXPIRED"}
                or (
                    row.get("continuation_fence_code") == "ACTIVE"
                    and (
                        row.get("continuation_terminal_event_digest") is not None
                        or row.get("continuation_successor_event_digest") is not None
                    )
                )
                or (
                    row.get("continuation_fence_code") == "EXPIRED"
                    and row.get("continuation_terminal_event_digest") is None
                )
            ):
                raise ActorError(
                    "legacy GPU migration row is malformed or misbound",
                    code="launchpad_legacy_gpu_migration_denied",
                )
            if classification == "SEALED_PROCESS_CONTEXT":
                if not isinstance(context, Mapping):
                    raise ActorError(
                        "sealed migration row lacks process context",
                        code="launchpad_legacy_gpu_migration_denied",
                    )
                context_value = dict(context)
                _require_exact_fields(
                    context_value,
                    context_fields,
                    label="LegacyGpuMigrationProcessContextV1",
                )
                relative_path = context_value.get("relative_path")
                if (
                    not isinstance(relative_path, str)
                    or not relative_path
                    or Path(relative_path).is_absolute()
                    or ".." in Path(relative_path).parts
                    or type(context_value.get("owner_pid")) is not int
                    or context_value["owner_pid"] <= 0
                    or type(context_value.get("owner_process_start_ticks")) is not int
                    or context_value["owner_process_start_ticks"] <= 0
                    or any(
                        not isinstance(context_value.get(field), str)
                        or not context_value[field]
                        for field in (
                            "file_sha256",
                            "schema_version",
                            "binding_sha256",
                            "scheduler_bundle_digest",
                        )
                    )
                ):
                    raise ActorError(
                        "sealed migration process context is malformed",
                        code="launchpad_legacy_gpu_migration_denied",
                    )
            elif context is not None or row.get("continuation_fence_code") != "EXPIRED":
                raise ActorError(
                    "true PID-less migration row requires an expired continuation and no context",
                    code="launchpad_legacy_gpu_migration_denied",
                )
            if row["attempt_id"] in attempt_ids or row_digest.hex in row_digests:
                raise ActorError(
                    "legacy GPU migration manifest repeats a row",
                    code="launchpad_legacy_gpu_migration_denied",
                )
            attempt_ids.add(str(row["attempt_id"]))
            row_digests.add(row_digest.hex)
            rows.append(row)
        return manifest_digest, rows

    def _observe_legacy_gpu_migration_row_v1(
        self,
        *,
        manifest_digest: ContentDigest,
        row: Mapping[str, Any],
        resource: LaunchpadResourceStartLeaseV1,
        repo_root: str | Path,
    ) -> dict[str, Any]:
        """Ask the Resource Manager subprocess for a fresh live proof."""

        context = row.get("sealed_process_context")
        request = {
            "schema_version": "eiel_legacy_gpu_migration_observation_request_v1",
            "repo_root": str(Path(repo_root).expanduser().resolve()),
            "migration_manifest_sha256": manifest_digest.hex,
            "migration_row_sha256": str(row["row_sha256"]),
            "classification": str(row["classification"]),
            "attempt_id": str(row["attempt_id"]),
            "resource_lease_digest": dict(row["resource_lease_digest"]),
            "fencing_token_digest": resource.fencing_token_digest.to_dict(),
            "continuation_lease_digest": dict(
                row["attempt_continuation_lease_digest"]
            ),
            "continuation_fence_code": str(row["continuation_fence_code"]),
            "continuation_terminal_event_digest": row[
                "continuation_terminal_event_digest"
            ],
            "continuation_successor_event_digest": row[
                "continuation_successor_event_digest"
            ],
            "device_uuid": str(row["device_uuid"]),
            "selected_resource_ids": list(row["resource_ids"]),
            "start_effect_digest": dict(row["start_effect_digest"]),
            "resource_started_event_digest": dict(
                row["resource_started_event_digest"]
            ),
            "owner_pid": (
                None if not isinstance(context, Mapping) else context["owner_pid"]
            ),
            "owner_process_start_ticks": (
                None
                if not isinstance(context, Mapping)
                else context["owner_process_start_ticks"]
            ),
        }
        if self._legacy_gpu_migration_test_observer is not None:
            return dict(self._legacy_gpu_migration_test_observer(request))
        try:
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "tools.actors.legacy_gpu_migration_observer",
                ],
                input=canonical_bytes(request),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=Path(__file__).resolve().parents[2],
                check=False,
                timeout=20,
            )
        except (OSError, subprocess.SubprocessError) as error:
            raise ActorError(
                "Resource Manager migration observation was unavailable",
                code="launchpad_legacy_gpu_migration_denied",
            ) from error
        if completed.returncode != 0:
            raise ActorError(
                "Resource Manager migration observation failed closed",
                code="launchpad_legacy_gpu_migration_denied",
                problems=[completed.stderr.decode("utf-8", errors="replace")[:1000]],
            )
        try:
            value = json.loads(completed.stdout.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ActorError(
                "Resource Manager migration observation was not canonical JSON",
                code="launchpad_legacy_gpu_migration_denied",
            ) from error
        if not isinstance(value, Mapping):
            raise ActorError(
                "Resource Manager migration observation was not an object",
                code="launchpad_legacy_gpu_migration_denied",
            )
        return dict(value)

    def legacy_gpu_migration_proof_v1(
        self,
        *,
        attempt_id: str,
        proof_digest: ContentDigest | None = None,
    ) -> dict[str, Any] | None:
        """Load one immutable proof for an exact migration row."""

        directory = self.legacy_gpu_migration_proofs_dir / _safe_key(attempt_id)
        paths = sorted(directory.glob("*.json")) if directory.is_dir() else []
        if not paths:
            return None
        if proof_digest is not None:
            paths = [
                path for path in paths if path.name == f"{proof_digest.hex}.json"
            ]
            if not paths:
                return None
        elif len(paths) != 1:
            raise ActorError(
                "legacy GPU migration proof identity is ambiguous",
                code="launchpad_legacy_gpu_migration_denied",
            )
        proof = read_json(paths[0], name="LegacyGpuMigrationReclaimProof")
        core = {key: value for key, value in proof.items() if key != "proof_sha256"}
        digest = _record_digest(core)
        if (
            proof.get("attempt_id") != attempt_id
            or proof.get("proof_sha256") != digest.hex
            or paths[0].name != f"{digest.hex}.json"
        ):
            raise ActorError(
                "legacy GPU migration proof differs from its immutable identity",
                code="launchpad_legacy_gpu_migration_denied",
            )
        return proof

    def _validate_legacy_gpu_migration_proof_v1(
        self,
        *,
        manifest_digest: ContentDigest,
        row: Mapping[str, Any],
        attempt_id: str,
        proof: Mapping[str, Any],
    ) -> tuple[dict[str, Any], ContentDigest, datetime, datetime]:
        proof_value = dict(proof)
        proof_core = {
            key: value for key, value in proof_value.items() if key != "proof_sha256"
        }
        proof_digest = _record_digest(proof_core)
        expected_proof_fields = {
            "schema_version",
            "decision",
            "migration_manifest_sha256",
            "migration_row_sha256",
            "classification",
            "attempt_id",
            "resource_lease_digest",
            "fencing_token_digest",
            "continuation_lease_digest",
            "continuation_fence_code",
            "continuation_terminal_event_digest",
            "continuation_successor_event_digest",
            "device_uuid",
            "selected_resource_ids",
            "start_effect_digest",
            "resource_started_event_digest",
            "owner_pid",
            "owner_process_start_ticks",
            "process_start_ticks_observation",
            "gpu_process_query_complete",
            "matching_owner_gpu_processes",
            "owner_pid_unknown_device_processes",
            "target_device_gpu_processes",
            "unknown_device_gpu_processes",
            "resource_monitor_report_sha256",
            "resource_observed_at_utc",
            "assessed_at_utc",
            "observation_age_seconds",
            "max_observation_age_seconds",
            "reason_codes",
            "proof_sha256",
        }
        _require_exact_fields(
            proof_value,
            expected_proof_fields,
            label="LegacyGpuMigrationReclaimProofV1",
        )
        resource_observed_at = _parse_utc(
            proof_value.get("resource_observed_at_utc"),
            field="proof.resource_observed_at_utc",
        )
        assessed_at = _parse_utc(
            proof_value.get("assessed_at_utc"),
            field="proof.assessed_at_utc",
        )
        row_digest = _record_digest(
            {key: value for key, value in row.items() if key != "row_sha256"}
        )
        invalid = (
            proof_value.get("schema_version")
            != "eiel_legacy_gpu_migration_reclaim_proof_v1"
            or proof_value.get("decision") != "RECLAIM"
            or proof_value.get("proof_sha256") != proof_digest.hex
            or proof_value.get("migration_manifest_sha256") != manifest_digest.hex
            or proof_value.get("migration_row_sha256") != row_digest.hex
            or proof_value.get("classification") != row["classification"]
            or proof_value.get("attempt_id") != attempt_id
            or proof_value.get("resource_lease_digest")
            != row["resource_lease_digest"]
            or proof_value.get("continuation_lease_digest")
            != row["attempt_continuation_lease_digest"]
            or proof_value.get("continuation_fence_code")
            != row["continuation_fence_code"]
            or proof_value.get("continuation_terminal_event_digest")
            != row["continuation_terminal_event_digest"]
            or proof_value.get("continuation_successor_event_digest")
            != row["continuation_successor_event_digest"]
            or proof_value.get("device_uuid") != row["device_uuid"]
            or proof_value.get("selected_resource_ids") != row["resource_ids"]
            or proof_value.get("start_effect_digest") != row["start_effect_digest"]
            or proof_value.get("resource_started_event_digest")
            != row["resource_started_event_digest"]
            or proof_value.get("gpu_process_query_complete") is not True
            or proof_value.get("reason_codes") != []
            or not isinstance(proof_value.get("resource_monitor_report_sha256"), str)
            or len(str(proof_value.get("resource_monitor_report_sha256"))) != 64
            or not isinstance(
                proof_value.get("observation_age_seconds"), (int, float)
            )
            or not isinstance(
                proof_value.get("max_observation_age_seconds"), (int, float)
            )
            or float(proof_value.get("observation_age_seconds") or 0) < 0
            or float(proof_value.get("observation_age_seconds") or 0)
            > min(
                5.0,
                float(proof_value.get("max_observation_age_seconds") or 0),
            )
            or resource_observed_at > assessed_at
        )
        context = row.get("sealed_process_context")
        if row["classification"] == "SEALED_PROCESS_CONTEXT":
            assert isinstance(context, Mapping)
            process_observation = proof_value.get("process_start_ticks_observation")
            if isinstance(process_observation, Mapping):
                observed_process_fields = set(process_observation)
                process_observed_at = _parse_utc(
                    process_observation.get("observed_at_utc"),
                    field="proof.process_start_ticks_observation.observed_at_utc",
                )
            else:
                observed_process_fields = set()
                process_observed_at = assessed_at + timedelta(seconds=6)
            state = (
                process_observation.get("state")
                if isinstance(process_observation, Mapping)
                else None
            )
            observed_ticks = (
                process_observation.get("observed_process_start_ticks")
                if isinstance(process_observation, Mapping)
                else None
            )
            invalid = invalid or (
                proof_value.get("owner_pid") != context["owner_pid"]
                or proof_value.get("owner_process_start_ticks")
                != context["owner_process_start_ticks"]
                or observed_process_fields
                != {
                    "schema_version",
                    "pid",
                    "expected_process_start_ticks",
                    "observed_at_utc",
                    "state",
                    "observed_process_start_ticks",
                }
                or process_observation.get("schema_version")
                != "eiel_legacy_process_start_ticks_observation_v1"
                or process_observation.get("pid") != context["owner_pid"]
                or process_observation.get("expected_process_start_ticks")
                != context["owner_process_start_ticks"]
                or state not in {"ABSENT", "REUSED"}
                or (state == "ABSENT" and observed_ticks is not None)
                or (
                    state == "REUSED"
                    and (
                        type(observed_ticks) is not int
                        or observed_ticks == context["owner_process_start_ticks"]
                    )
                )
                or process_observed_at > resource_observed_at
                or (assessed_at - process_observed_at).total_seconds() > 5.0
                or proof_value.get("matching_owner_gpu_processes") != []
                or proof_value.get("owner_pid_unknown_device_processes") != []
            )
        else:
            invalid = invalid or (
                proof_value.get("owner_pid") is not None
                or proof_value.get("owner_process_start_ticks") is not None
                or proof_value.get("process_start_ticks_observation") is not None
                or proof_value.get("target_device_gpu_processes") != []
                or proof_value.get("unknown_device_gpu_processes") != []
                or row["continuation_fence_code"] != "EXPIRED"
            )
        if invalid:
            raise ActorError(
                "legacy GPU migration Resource Manager proof is invalid",
                code="launchpad_legacy_gpu_migration_denied",
            )
        return proof_value, proof_digest, resource_observed_at, assessed_at

    def apply_legacy_gpu_migration_row_v1(
        self,
        *,
        manifest: Mapping[str, Any],
        attempt_id: str,
        repo_root: str | Path | None = None,
        proof: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Apply one append-only row from the pinned historical migration.

        This is intentionally absent from normal admission and reconciliation.
        It can terminate only a row named by the canonical manifest (or an
        explicitly isolated test manifest), after an exact Resource Manager
        proof.  Historical records are never rewritten or deleted.
        """

        manifest_digest, rows = self._validate_legacy_gpu_migration_manifest_v1(
            manifest
        )
        matches = [row for row in rows if row.get("attempt_id") == attempt_id]
        if len(matches) != 1:
            raise ActorError(
                "attempt is not one exact legacy GPU migration row",
                code="launchpad_legacy_gpu_migration_denied",
            )
        row = matches[0]
        row_core = {key: value for key, value in row.items() if key != "row_sha256"}
        row_digest = _record_digest(row_core)
        context = row.get("sealed_process_context")
        if proof is not None and not self._allow_test_shared_gpu_state_root:
            raise ActorError(
                "caller-supplied migration evidence is test-root only",
                code="launchpad_legacy_gpu_migration_denied",
            )

        with _scheduler_execution_lock(self.state_root):
            bundle = self._authoritative_bundle(attempt_id)
            preparation = self._preparation(attempt_id)
            claim = LaunchpadAttemptClaimV1.from_dict(
                dict(bundle.get("claim") or {})
            )
            resource = LaunchpadResourceStartLeaseV1.from_dict(
                dict(bundle.get("resource_start_lease") or {})
            )
            effect = dict(bundle.get("start_effect") or {})
            resource_events = self.resource_ledger.events(resource.lease_id)
            if (
                self._terminal_head_path(attempt_id).exists()
                or self._terminal_intent_path(attempt_id).exists()
                or effect.get("schema_version") != START_EFFECT_SCHEMA_VERSION
                or dict(preparation.get("resource_start_lease") or {})
                != resource.to_dict()
                or claim.run_id != row["run_id"]
                or claim.claim_digest().to_dict() != row["claim_digest"]
                or _record_digest(effect).to_dict() != row["start_effect_digest"]
                or _record_digest(bundle).to_dict()
                != row["authoritative_bundle_digest"]
                or resource.lease_id != row["resource_lease_id"]
                or resource.lease_digest().to_dict() != row["resource_lease_digest"]
                or list(resource.resource_ids) != row["resource_ids"]
                or resource.expires_at_utc != row["resource_expires_at_utc"]
                or claim.lineage_scope_digest.to_dict()
                != row["lineage_scope_digest"]
                or not gpu_uuid_equal_v1(claim.device_id, row["device_uuid"])
                or not resource_events
                or resource_events[0].get("code")
                != ResourceStartLeaseEventCodeV1.STARTED.value
                or canonical_sha256(resource_events[0]).to_dict()
                != row["resource_started_event_digest"]
            ):
                raise ActorError(
                    "legacy GPU migration row differs from Scheduler authority",
                    code="launchpad_legacy_gpu_migration_denied",
                )

            if len(resource_events) == 2:
                migrated = resource_events[-1]
                if (
                    migrated.get("code")
                    != ResourceStartLeaseEventCodeV1.LEGACY_GPU_MIGRATION_RECLAIMED.value
                    or migrated.get("attempt_id") != attempt_id
                    or migrated.get("migration_manifest_digest")
                    != manifest_digest.to_dict()
                    or migrated.get("migration_row_digest") != row_digest.to_dict()
                ):
                    raise ActorError(
                        "legacy GPU migration event replay conflicts",
                        code="launchpad_attempt_replay",
                    )
                replay_proof_digest = _digest(
                    migrated.get("reclaim_proof_digest"),
                    field="reclaim_proof_digest",
                )
                replay_authorization_digest = _digest(
                    migrated.get("migration_authorization_digest"),
                    field="migration_authorization_digest",
                )
                replay_proof_path = (
                    self.legacy_gpu_migration_proofs_dir
                    / _safe_key(attempt_id)
                    / f"{replay_proof_digest.hex}.json"
                )
                replay_authorization_path = (
                    self.legacy_gpu_migration_authorizations_dir
                    / _safe_key(attempt_id)
                    / f"{replay_authorization_digest.hex}.json"
                )
                return {
                    "status": "LEGACY_GPU_MIGRATION_RECLAIMED",
                    "idempotent_replay": True,
                    "attempt_id": attempt_id,
                    "resource_lease_id": resource.lease_id,
                    "manifest_digest": manifest_digest.to_dict(),
                    "row_digest": row_digest.to_dict(),
                    "proof_path": str(replay_proof_path),
                    "proof_digest": replay_proof_digest.to_dict(),
                    "continuation_owner_dead_event": None,
                    "event": migrated,
                    "authorization_path": str(replay_authorization_path),
                    "authorization_digest": replay_authorization_digest.to_dict(),
                    "receipt_path": str(replay_authorization_path),
                    "receipt_digest": replay_authorization_digest.to_dict(),
                }
            if len(resource_events) != 1:
                raise ActorError(
                    "legacy GPU migration occupancy is not singly STARTED",
                    code="launchpad_legacy_gpu_migration_denied",
                )

            proof_value = (
                dict(proof)
                if proof is not None
                else self._observe_legacy_gpu_migration_row_v1(
                    manifest_digest=manifest_digest,
                    row=row,
                    resource=resource,
                    repo_root=(
                        Path(__file__).resolve().parents[2]
                        if repo_root is None
                        else repo_root
                    ),
                )
            )
            (
                proof_value,
                proof_digest,
                resource_observed_at,
                assessed_at,
            ) = self._validate_legacy_gpu_migration_proof_v1(
                manifest_digest=manifest_digest,
                row=row,
                attempt_id=attempt_id,
                proof=proof_value,
            )
            if (
                proof_value.get("fencing_token_digest")
                != resource.fencing_token_digest.to_dict()
            ):
                raise ActorError(
                    "legacy GPU migration proof has another resource fence",
                    code="launchpad_legacy_gpu_migration_denied",
                )
            if row["classification"] == "SEALED_PROCESS_CONTEXT":
                assert isinstance(context, Mapping)
                run_root = Path(claim.run_root).expanduser().resolve()
                context_path = (run_root / str(context["relative_path"])).resolve()
                try:
                    context_path.relative_to(run_root)
                except ValueError as error:
                    raise ActorError(
                        "sealed migration context escapes its Run root",
                        code="launchpad_legacy_gpu_migration_denied",
                    ) from error
                if not context_path.is_file() or context_path.is_symlink():
                    raise ActorError(
                        "sealed migration context is unavailable",
                        code="launchpad_legacy_gpu_migration_denied",
                    )
                context_value = read_json(
                    context_path, name="LegacyGpuMigrationProcessContext"
                )
                if (
                    hashlib.sha256(context_path.read_bytes()).hexdigest()
                    != context["file_sha256"]
                    or context_value.get("schema_version")
                    != context["schema_version"]
                    or context_value.get("binding_sha256")
                    != context["binding_sha256"]
                    or context_value.get("scheduler_bundle_digest")
                    != context["scheduler_bundle_digest"]
                    or context_value.get("owner_pid") != context["owner_pid"]
                    or context_value.get("owner_process_start_ticks")
                    != context["owner_process_start_ticks"]
                    or context_value.get("attempt_id") != attempt_id
                    or context_value.get("run_id") != claim.run_id
                ):
                    raise ActorError(
                        "sealed migration context differs from its manifest row",
                        code="launchpad_legacy_gpu_migration_denied",
                    )
                observed_identity = observe_process_start_identity(
                    int(context["owner_pid"])
                )
                observed_ticks = (
                    None
                    if observed_identity is None
                    else int(observed_identity.rsplit(":", 1)[-1])
                )
                expected_ticks = int(context["owner_process_start_ticks"])
                process_observation = dict(
                    proof_value["process_start_ticks_observation"]
                )
                if observed_ticks == expected_ticks:
                    raise ActorError(
                        "sealed legacy GPU owner is still the exact live process",
                        code="launchpad_legacy_gpu_migration_denied",
                    )
                if (
                    (observed_ticks is None and process_observation.get("state") != "ABSENT")
                    or (
                        observed_ticks is not None
                        and (
                            process_observation.get("state") != "REUSED"
                            or process_observation.get(
                                "observed_process_start_ticks"
                            )
                            != observed_ticks
                        )
                    )
                ):
                    raise ActorError(
                        "Resource Manager process evidence changed before release",
                        code="launchpad_legacy_gpu_migration_denied",
                    )

            leases, continuation_events, continuation_head = self._continuation_state(
                claim.lineage_scope_digest
            )
            continuation_digest = _digest(
                row["attempt_continuation_lease_digest"],
                field="attempt_continuation_lease_digest",
            )
            matching_leases = [
                lease
                for lease in leases
                if _continuation_lease_digest(lease) == continuation_digest
            ]
            if (
                continuation_head is None
                or len(matching_leases) != 1
                or matching_leases[0].get("attempt_id") != attempt_id
                or matching_leases[0].get("run_id") != claim.run_id
                or matching_leases[0].get("owner_id") != claim.execution_owner_id
            ):
                raise ActorError(
                    "legacy GPU migration continuation binding is unavailable",
                    code="launchpad_legacy_gpu_migration_denied",
                )
            continuation_state = _continuation_ledger_state(
                leases, continuation_events
            )
            owner_dead_event = next(
                (
                    event
                    for event in continuation_events
                    if event.get("code") == "OWNER_DEAD"
                    and _digest(event.get("lease_digest"), field="lease_digest")
                    == continuation_digest
                ),
                None,
            )
            if (
                continuation_state is _ContinuationStateV1.OWNER_DEAD
                and owner_dead_event is not None
            ):
                bound_proof_digest = _digest(
                    owner_dead_event.get("owner_dead_proof_digest"),
                    field="owner_dead_proof_digest",
                )
                if bound_proof_digest != proof_digest:
                    bound_proof = self.legacy_gpu_migration_proof_v1(
                        attempt_id=attempt_id,
                        proof_digest=bound_proof_digest,
                    )
                    if bound_proof is None:
                        raise ActorError(
                            "fenced migration proof is unavailable",
                            code="launchpad_legacy_gpu_migration_denied",
                        )
                    (
                        proof_value,
                        proof_digest,
                        resource_observed_at,
                        assessed_at,
                    ) = self._validate_legacy_gpu_migration_proof_v1(
                        manifest_digest=manifest_digest,
                        row=row,
                        attempt_id=attempt_id,
                        proof=bound_proof,
                    )
            if row["continuation_fence_code"] == "ACTIVE":
                active_binding_valid = (
                    _continuation_lease_digest(leases[-1]) == continuation_digest
                    and (
                        continuation_state is _ContinuationStateV1.ACTIVE
                        or (
                            continuation_state is _ContinuationStateV1.OWNER_DEAD
                            and owner_dead_event is not None
                            and _digest(
                                owner_dead_event.get("owner_dead_proof_digest"),
                                field="owner_dead_proof_digest",
                            )
                            == proof_digest
                        )
                    )
                )
                if not active_binding_valid:
                    raise ActorError(
                        "active legacy migration continuation changed",
                        code="launchpad_legacy_gpu_migration_denied",
                    )
            else:
                expired_event = next(
                    (
                        event
                        for event in continuation_events
                        if event.get("code") == "EXPIRED"
                        and _digest(event.get("lease_digest"), field="lease_digest")
                        == continuation_digest
                    ),
                    None,
                )
                successor_digest_value = row["continuation_successor_event_digest"]
                successor_event = (
                    None
                    if successor_digest_value is None
                    else next(
                        (
                            event
                            for event in continuation_events
                            if _continuation_event_digest(event).to_dict()
                            == successor_digest_value
                        ),
                        None,
                    )
                )
                expired_binding_valid = (
                    expired_event is not None
                    and _continuation_event_digest(expired_event).to_dict()
                    == row["continuation_terminal_event_digest"]
                    and (
                        (
                            successor_digest_value is None
                            and _continuation_lease_digest(leases[-1])
                            == continuation_digest
                            and continuation_state is _ContinuationStateV1.EXPIRED
                        )
                        or (
                            successor_digest_value is not None
                            and successor_event is not None
                            and successor_event.get("code") == "RECLAIMED"
                            and _digest(
                                successor_event.get("lease_digest"),
                                field="lease_digest",
                            )
                            == continuation_digest
                        )
                    )
                )
                if not expired_binding_valid:
                    raise ActorError(
                        "expired legacy migration continuation changed",
                        code="launchpad_legacy_gpu_migration_denied",
                    )
                temporal_fences = [
                    _parse_utc(
                        resource.expires_at_utc,
                        field="resource.expires_at_utc",
                    ),
                    _parse_utc(
                        expired_event.get("event_at_utc"),
                        field="continuation_expired.event_at_utc",
                    ),
                ]
                if successor_event is not None:
                    temporal_fences.append(
                        _parse_utc(
                            successor_event.get("event_at_utc"),
                            field="continuation_successor.event_at_utc",
                        )
                    )
                required_after = max(temporal_fences)
                if (
                    resource_observed_at < required_after
                    or assessed_at < required_after
                ):
                    raise ActorError(
                        "PID-less migration proof precedes its expiry or handoff fence",
                        code="launchpad_legacy_gpu_migration_denied",
                    )

            proof_path = (
                self.legacy_gpu_migration_proofs_dir
                / _safe_key(attempt_id)
                / f"{proof_digest.hex}.json"
            )
            _write_immutable(
                proof_path,
                proof_value,
                label="LegacyGpuMigrationReclaimProof",
            )
            if continuation_state is _ContinuationStateV1.ACTIVE:
                owner_dead_event = _continuation_event(
                    leases[-1],
                    code="OWNER_DEAD",
                    event_at_utc=str(proof_value["assessed_at_utc"]),
                    reason=(
                        "Scheduler fenced an exact sealed historical CUDA owner "
                        "before legacy migration release."
                    ),
                    prior_event_digest=(
                        None
                        if not continuation_events
                        else _continuation_event_digest(continuation_events[-1])
                    ),
                    owner_dead_proof_digest=proof_digest,
                )
                leases, continuation_events = self._append_continuation_event(
                    claim.lineage_scope_digest,
                    leases,
                    continuation_events,
                    owner_dead_event,
                )

            provisional_event = ResourceStartLeaseEventV3(
                migration_id=str(manifest["migration_id"]),
                attempt_id=attempt_id,
                resource_lease_id=resource.lease_id,
                lease_digest=resource.lease_digest(),
                code=(
                    ResourceStartLeaseEventCodeV1.LEGACY_GPU_MIGRATION_RECLAIMED
                ),
                scheduler_owner_id=resource.scheduler_owner_id,
                fencing_token_digest=resource.fencing_token_digest,
                event_at_utc=str(proof_value["assessed_at_utc"]),
                reason=(
                    "Scheduler released one exact manifest-bound historical "
                    "PID-less CUDA occupancy after migration proof"
                ),
                prior_event_digest=canonical_sha256(resource_events[0]),
                migration_manifest_digest=manifest_digest,
                migration_row_digest=row_digest,
                reclaim_proof_digest=proof_digest,
                migration_authorization_digest=ContentDigest(
                    algorithm="sha256", hex="0" * 64
                ),
            )
            owner_dead_event_digest = (
                None
                if owner_dead_event is None
                else _continuation_event_digest(owner_dead_event).to_dict()
            )
            authorization_core = {
                "schema_version": LEGACY_GPU_MIGRATION_AUTHORIZATION_SCHEMA_VERSION,
                "migration_id": manifest["migration_id"],
                "migration_manifest_digest": manifest_digest.to_dict(),
                "migration_row_digest": row_digest.to_dict(),
                "attempt_id": attempt_id,
                "resource_lease_id": resource.lease_id,
                "lease_digest": resource.lease_digest().to_dict(),
                "resource_started_event_digest": canonical_sha256(
                    resource_events[0]
                ).to_dict(),
                "reclaim_proof_digest": proof_digest.to_dict(),
                "continuation_owner_dead_event_digest": owner_dead_event_digest,
                "resource_event_intent_digest": canonical_sha256(
                    provisional_event.authorization_intent()
                ).to_dict(),
                "authorized_at_utc": proof_value["assessed_at_utc"],
            }
            authorization_digest = _record_digest(authorization_core)
            authorization = {
                **authorization_core,
                "authorization_sha256": authorization_digest.hex,
            }
            authorization_path = (
                self.legacy_gpu_migration_authorizations_dir
                / _safe_key(attempt_id)
                / f"{authorization_digest.hex}.json"
            )
            _write_immutable(
                authorization_path,
                authorization,
                label="LegacyGpuMigrationAuthorization",
            )
            event = ResourceStartLeaseEventV3(
                migration_id=provisional_event.migration_id,
                attempt_id=provisional_event.attempt_id,
                resource_lease_id=provisional_event.resource_lease_id,
                lease_digest=provisional_event.lease_digest,
                code=provisional_event.code,
                scheduler_owner_id=provisional_event.scheduler_owner_id,
                fencing_token_digest=provisional_event.fencing_token_digest,
                event_at_utc=provisional_event.event_at_utc,
                reason=provisional_event.reason,
                prior_event_digest=provisional_event.prior_event_digest,
                migration_manifest_digest=provisional_event.migration_manifest_digest,
                migration_row_digest=provisional_event.migration_row_digest,
                reclaim_proof_digest=provisional_event.reclaim_proof_digest,
                migration_authorization_digest=authorization_digest,
            )
            if len(resource_events) == 2:
                if resource_events[-1] != event.to_dict():
                    raise ActorError(
                        "legacy GPU migration event replay conflicts",
                        code="launchpad_attempt_replay",
                    )
            elif len(resource_events) == 1:
                self.resource_ledger._append_authorized_legacy_gpu_migration_event(
                    resource.lease_id, event
                )
            else:
                raise ActorError(
                    "legacy GPU migration occupancy is not singly STARTED",
                    code="launchpad_legacy_gpu_migration_denied",
                )

            return {
                "status": "LEGACY_GPU_MIGRATION_RECLAIMED",
                "idempotent_replay": False,
                "attempt_id": attempt_id,
                "resource_lease_id": resource.lease_id,
                "manifest_digest": manifest_digest.to_dict(),
                "row_digest": row_digest.to_dict(),
                "proof_path": str(proof_path),
                "proof_digest": proof_digest.to_dict(),
                "continuation_owner_dead_event": owner_dead_event,
                "event": event.to_dict(),
                "authorization_path": str(authorization_path),
                "authorization_digest": authorization_digest.to_dict(),
                "receipt_path": str(authorization_path),
                "receipt_digest": authorization_digest.to_dict(),
            }

    def dead_shared_gpu_reclaim_candidates_v1(
        self, *, at_utc: str
    ) -> list[dict[str, Any]]:
        """Compatibility projection of the generic GPU reclaim candidates."""

        return [
            row
            for row in self.dead_gpu_reclaim_candidates_v1(at_utc=at_utc)
            if row.get("resource_kind") == "shared"
        ]

    def dead_owner_reclaim_proof_v1(
        self,
        *,
        attempt_id: str,
        proof_digest: ContentDigest | Mapping[str, Any],
    ) -> dict[str, Any]:
        """Load one immutable proof already bound by an OWNER_DEAD fence."""

        digest = (
            proof_digest
            if isinstance(proof_digest, ContentDigest)
            else _digest(proof_digest, field="proof_digest")
        )
        proof_path = (
            self.dead_owner_reclaims_dir / _safe_key(attempt_id) / f"{digest.hex}.json"
        )
        proof = read_json(proof_path, name="DeadGpuOwnerReclaimProof")
        proof_core = {
            key: value for key, value in proof.items() if key != "proof_sha256"
        }
        if (
            proof.get("attempt_id") != attempt_id
            or proof.get("proof_sha256") != digest.hex
            or canonical_sha256(proof_core) != digest
        ):
            raise ActorError(
                "dead-owner reclaim proof differs from its immutable identity",
                code="launchpad_attempt_reclaim_denied",
            )
        return proof

    def reclaim_dead_gpu_occupancy_v1(
        self,
        *,
        attempt_id: str,
        proof: Mapping[str, Any],
        at_utc: str,
    ) -> dict[str, Any]:
        """Release one GPU occupancy only after exact dead-owner proof."""

        proof_value = dict(proof)
        proof_core = {
            key: value for key, value in proof_value.items() if key != "proof_sha256"
        }
        proof_digest = canonical_sha256(proof_core)
        with _scheduler_execution_lock(self.state_root):
            preparation = self._preparation(attempt_id)
            bundle = self._authoritative_bundle(attempt_id)
            claim = LaunchpadAttemptClaimV1.from_dict(dict(bundle.get("claim") or {}))
            resource = LaunchpadResourceStartLeaseV1.from_dict(
                dict(preparation.get("resource_start_lease") or {})
            )
            effect = dict(bundle.get("start_effect") or {})
            resource_kind = _gpu_resource_kind_v1(
                resource.resource_ids, device_id=claim.device_id
            )
            if (
                self._terminal_head_path(attempt_id).exists()
                or self._terminal_intent_path(attempt_id).exists()
                or resource_kind is None
                or effect.get("schema_version") != START_EFFECT_SCHEMA_VERSION_V2
            ):
                raise ActorError(
                    "Attempt is not an unterminated recoverable GPU owner",
                    code="launchpad_attempt_reclaim_denied",
                )
            leases, continuation_events, head = self._continuation_state(
                claim.lineage_scope_digest
            )
            proof_continuation_lease_digest = _digest(
                proof_value.get("continuation_lease_digest"),
                field="proof.continuation_lease_digest",
            )
            matching_continuation_leases = [
                lease
                for lease in leases
                if _continuation_lease_digest(lease) == proof_continuation_lease_digest
            ]
            matching_terminal_events = [
                event
                for event in continuation_events
                if event.get("code") in {"EXPIRED", "OWNER_DEAD"}
                and _digest(event.get("lease_digest"), field="lease_digest")
                == proof_continuation_lease_digest
            ]
            if (
                head is None
                or len(matching_continuation_leases) != 1
                or matching_continuation_leases[0].get("attempt_id") != attempt_id
                or matching_continuation_leases[0].get("owner_id")
                != claim.execution_owner_id
                or _continuation_lease_digest(leases[-1])
                != proof_continuation_lease_digest
            ):
                raise ActorError(
                    "dead-owner reclaim requires this Attempt's current continuation",
                    code="launchpad_attempt_reclaim_denied",
                )
            continuation_state = _continuation_ledger_state(leases, continuation_events)
            resource_events = self.resource_ledger.events(resource.lease_id)
            if (
                not resource_events
                or resource_events[0].get("code")
                != ResourceStartLeaseEventCodeV1.STARTED.value
            ):
                raise ActorError(
                    "dead-owner reclaim lacks one STARTED occupancy",
                    code="launchpad_attempt_reclaim_denied",
                )
            common_invalid = (
                proof_value.get("proof_sha256") != proof_digest.hex
                or proof_value.get("decision") != "RECLAIM"
                or proof_value.get("attempt_id") != attempt_id
                or proof_value.get("resource_lease_digest")
                != resource.lease_digest().to_dict()
                or proof_value.get("fencing_token_digest")
                != resource.fencing_token_digest.to_dict()
                or proof_value.get("continuation_lease_digest")
                != proof_continuation_lease_digest.to_dict()
                or proof_value.get("resource_started_event_digest")
                != canonical_sha256(resource_events[0]).to_dict()
                or not gpu_uuid_equal_v1(
                    proof_value.get("device_uuid"), claim.device_id
                )
                or proof_value.get("selected_resource_ids")
                != list(resource.resource_ids)
                or proof_value.get("gpu_process_query_complete") is not True
                or proof_value.get("assessed_at_utc") != at_utc
                or proof_value.get("reason_codes") != []
                or not isinstance(
                    proof_value.get("observation_age_seconds"), (int, float)
                )
                or not isinstance(
                    proof_value.get("max_observation_age_seconds"), (int, float)
                )
                or float(proof_value.get("observation_age_seconds") or 0) < 0
                or float(proof_value.get("observation_age_seconds") or 0)
                > min(
                    5.0,
                    float(proof_value.get("max_observation_age_seconds") or 0),
                )
            )
            expected_proof_fields = {
                "schema_version",
                "decision",
                "attempt_id",
                "resource_lease_digest",
                "fencing_token_digest",
                "continuation_lease_digest",
                "continuation_fence_state",
                "continuation_terminal_event_digest",
                "resource_started_event_digest",
                "device_uuid",
                "selected_resource_ids",
                "owner_pid",
                "owner_process_start_identity",
                "process_identity_observation",
                "gpu_process_query_complete",
                "matching_owner_gpu_processes",
                "resource_monitor_report_sha256",
                "resource_observed_at_utc",
                "assessed_at_utc",
                "observation_age_seconds",
                "max_observation_age_seconds",
                "reason_codes",
                "proof_sha256",
            }
            process_observation = proof_value.get("process_identity_observation")
            try:
                process_observed_at = _parse_utc(
                    (
                        process_observation.get("observed_at_utc")
                        if isinstance(process_observation, Mapping)
                        else None
                    ),
                    field="process_identity_observation.observed_at_utc",
                )
                resource_observed_at = _parse_utc(
                    proof_value.get("resource_observed_at_utc"),
                    field="resource_observed_at_utc",
                )
                assessed_at = _parse_utc(at_utc, field="at_utc")
            except ActorError:
                process_observed_at = None
                resource_observed_at = None
                assessed_at = None
            mode_invalid = (
                set(proof_value) != expected_proof_fields
                or proof_value.get("schema_version")
                != "eiel_dead_cuda_owner_reclaim_proof_v2"
                or proof_value.get("owner_pid") != effect.get("process_id")
                or proof_value.get("owner_process_start_identity")
                != effect.get("process_start_identity")
                or not isinstance(process_observation, Mapping)
                or process_observation.get("pid") != effect.get("process_id")
                or process_observation.get("expected_process_start_identity")
                != effect.get("process_start_identity")
                or process_observation.get("state") not in {"ABSENT", "REUSED"}
                or process_observed_at is None
                or resource_observed_at is None
                or assessed_at is None
                or process_observed_at > resource_observed_at
                or resource_observed_at > assessed_at
                or (assessed_at - process_observed_at).total_seconds() > 5.0
                or proof_value.get("matching_owner_gpu_processes") != []
            )
            proof_fence_state = proof_value.get("continuation_fence_state")
            proof_terminal_digest = proof_value.get(
                "continuation_terminal_event_digest"
            )
            expired_fence_valid = (
                proof_fence_state == "EXPIRED"
                and continuation_state is _ContinuationStateV1.EXPIRED
                and len(matching_terminal_events) == 1
                and matching_terminal_events[0].get("code") == "EXPIRED"
                and proof_terminal_digest
                == _continuation_event_digest(matching_terminal_events[0]).to_dict()
            )
            owner_dead_event = next(
                (
                    event
                    for event in matching_terminal_events
                    if event.get("code") == "OWNER_DEAD"
                ),
                None,
            )
            owner_dead_fence_valid = (
                proof_fence_state == "OWNER_DEAD_REQUIRED"
                and proof_terminal_digest is None
                and (
                    continuation_state is _ContinuationStateV1.ACTIVE
                    or (
                        continuation_state is _ContinuationStateV1.OWNER_DEAD
                        and owner_dead_event is not None
                        and _digest(
                            owner_dead_event.get("owner_dead_proof_digest"),
                            field="owner_dead_proof_digest",
                        )
                        == proof_digest
                    )
                )
            )
            if (
                common_invalid
                or mode_invalid
                or not (expired_fence_valid or owner_dead_fence_valid)
            ):
                raise ActorError(
                    "dead-owner Resource Manager proof is invalid or stale",
                    code="launchpad_attempt_reclaim_denied",
                )
            event = ResourceStartLeaseEventV2(
                lease_digest=resource.lease_digest(),
                code=ResourceStartLeaseEventCodeV1.OWNER_DEAD_RECLAIMED,
                scheduler_owner_id=resource.scheduler_owner_id,
                fencing_token_digest=resource.fencing_token_digest,
                event_at_utc=at_utc,
                reason=(
                    "Scheduler released verified crash-dead "
                    f"{resource_kind} CUDA owner via exact process identity"
                ),
                prior_event_digest=canonical_sha256(resource_events[0]),
                reclaim_proof_digest=proof_digest,
            )
            if len(resource_events) == 2:
                if resource_events[-1] != event.to_dict():
                    raise ActorError(
                        "dead-owner reclaim replay conflicts",
                        code="launchpad_attempt_replay",
                    )
            elif (
                len(resource_events) != 1
                or resource_events[0].get("code")
                != ResourceStartLeaseEventCodeV1.STARTED.value
            ):
                raise ActorError(
                    "dead-owner reclaim lacks one STARTED occupancy",
                    code="launchpad_attempt_reclaim_denied",
                )
            proof_path = (
                self.dead_owner_reclaims_dir
                / _safe_key(attempt_id)
                / f"{proof_digest.hex}.json"
            )
            _write_immutable(
                proof_path,
                proof_value,
                label="DeadGpuOwnerReclaimProof",
            )
            continuation_fence_event: dict[str, Any] | None = None
            if continuation_state is _ContinuationStateV1.ACTIVE:
                continuation_fence_event = _continuation_event(
                    leases[-1],
                    code="OWNER_DEAD",
                    event_at_utc=at_utc,
                    reason=(
                        "Scheduler fenced an exact crash-dead CUDA owner before "
                        "resource release."
                    ),
                    prior_event_digest=(
                        None
                        if not continuation_events
                        else _continuation_event_digest(continuation_events[-1])
                    ),
                    owner_dead_proof_digest=proof_digest,
                )
                leases, continuation_events = self._append_continuation_event(
                    claim.lineage_scope_digest,
                    leases,
                    continuation_events,
                    continuation_fence_event,
                )
            elif owner_dead_event is not None:
                continuation_fence_event = owner_dead_event
            if len(resource_events) == 1:
                self.resource_ledger.append_event(resource.lease_id, event)
            return {
                "status": "OWNER_DEAD_RECLAIMED",
                "attempt_id": attempt_id,
                "resource_lease_id": resource.lease_id,
                "proof_path": str(proof_path),
                "proof_digest": proof_digest.to_dict(),
                "continuation_fence_event": continuation_fence_event,
                "event": event.to_dict(),
            }

    def reclaim_dead_shared_gpu_occupancy_v1(
        self,
        *,
        attempt_id: str,
        proof: Mapping[str, Any],
        at_utc: str,
    ) -> dict[str, Any]:
        """Compatibility alias for the former shared-only reclaim API."""

        return self.reclaim_dead_gpu_occupancy_v1(
            attempt_id=attempt_id,
            proof=proof,
            at_utc=at_utc,
        )

    def reconcile(self, *, at_utc: str) -> dict[str, Any]:
        """Converge expired reservations and crash-partial exact starts."""

        outcomes: list[dict[str, Any]] = []
        expired_continuations: list[str] = []
        # First make every crash-partial start structurally reopenable.  This
        # stage does not expire a live continuation.
        with _scheduler_execution_lock(self.state_root):
            for preparation_path in sorted(self.preparations_dir.rglob("*.json")):
                preparation = read_json(
                    preparation_path, name="LaunchpadAttemptPreparation"
                )
                attempt_id = str(
                    dict(preparation.get("claim") or {}).get("attempt_id") or ""
                )
                if not attempt_id or self._preparation_head_path(attempt_id).is_file():
                    continue
                try:
                    result = self._commit_preparation(
                        preparation, preparation_path=preparation_path
                    )
                    outcomes.append(
                        {"attempt_id": attempt_id, "status": result["status"]}
                    )
                except ActorError as error:
                    self._quarantine(
                        attempt_id=attempt_id or "UNKNOWN",
                        reason=f"reconcile-prepare:{error.code}:{error}",
                        at_utc=at_utc,
                        details=preparation,
                    )
                    outcomes.append({"attempt_id": attempt_id, "status": "QUARANTINED"})
            for effect_path in sorted(self.start_effects_dir.glob("*.json")):
                effect = read_json(effect_path, name="LaunchpadAttemptStartEffect")
                attempt_id = str(effect.get("attempt_id") or "")
                if self._bundle_head_path(attempt_id).is_file():
                    continue
                try:
                    preparation = self._preparation(attempt_id)
                    result = self._finalize_start(preparation, effect)
                    outcomes.append(
                        {"attempt_id": attempt_id, "status": result["status"]}
                    )
                except ActorError as error:
                    self._quarantine(
                        attempt_id=attempt_id,
                        reason=f"reconcile:{error.code}:{error}",
                        at_utc=at_utc,
                        details=effect,
                    )
                    outcomes.append({"attempt_id": attempt_id, "status": "QUARANTINED"})
        # A terminal intent was recorded while its lease was valid and is the
        # recovery authority.  Converge it before generic TTL expiry can fence
        # the same continuation.
        for intent_path in sorted(self.terminal_intents_dir.glob("*.json")):
            intent = read_json(intent_path, name="LaunchpadAttemptTerminalIntent")
            attempt_id = str(intent.get("attempt_id") or "")
            if self._terminal_head_path(attempt_id).is_file():
                continue
            try:
                result = self.terminalize(
                    attempt_id=attempt_id,
                    process_owner_id=str(intent.get("process_owner_id") or ""),
                    process_owner_digest=_digest(
                        intent.get("process_owner_digest"),
                        field="process_owner_digest",
                    ),
                    process_id=intent.get("process_id"),
                    process_start_identity=intent.get("process_start_identity"),
                    terminal=dict(intent.get("attempt_terminal") or {}),
                    at_utc=str(intent.get("terminalized_at_utc") or ""),
                    publication_digest=_optional_digest(
                        intent.get("publication_digest"),
                        field="publication_digest",
                    ),
                    _reconcile_durable_intent=True,
                )
                outcomes.append({"attempt_id": attempt_id, "status": result["status"]})
            except ActorError as error:
                self._quarantine(
                    attempt_id=attempt_id or "UNKNOWN",
                    reason=f"reconcile-terminal:{error.code}:{error}",
                    at_utc=at_utc,
                    details=intent,
                )
                outcomes.append({"attempt_id": attempt_id, "status": "QUARANTINED"})
        terminal_resource_reconciliation = (
            self.reconcile_terminal_resource_occupancies_v1(at_utc=at_utc)
        )
        # Only after all durable terminal intents/heads converge may unused
        # reservations and genuinely unterminated continuations expire.
        with _scheduler_execution_lock(self.state_root):
            expired = self.resource_ledger.reconcile_expired(at_utc=at_utc)
            for head_path in sorted(self.continuation_heads_dir.glob("*.json")):
                try:
                    lineage_digest = ContentDigest(
                        algorithm="sha256", hex=head_path.stem
                    )
                    leases, events, head = self._continuation_state(lineage_digest)
                    if head is None:
                        continue
                    _leases, _events, changed = self._expire_continuation_if_stale(
                        lineage_digest,
                        leases,
                        events,
                        at_utc=at_utc,
                    )
                    if changed:
                        expired_continuations.append(lineage_digest.hex)
                except (ActorError, ProgrammeContractError) as error:
                    self._quarantine(
                        attempt_id=f"CONTINUATION-{head_path.stem}",
                        reason=f"reconcile-continuation:{error}",
                        at_utc=at_utc,
                        details={"continuation_head_path": str(head_path)},
                    )
                    outcomes.append(
                        {
                            "attempt_id": f"CONTINUATION-{head_path.stem}",
                            "status": "QUARANTINED",
                        }
                    )
        return {
            "status": "RECONCILED",
            "terminal_resource_reconciliation": terminal_resource_reconciliation,
            "expired_resource_leases": [str(path) for path in expired],
            "expired_continuation_lineages": expired_continuations,
            "attempts": outcomes,
        }


__all__ = [
    "AUTHORITATIVE_BUNDLE_SCHEMA_VERSION",
    "AUTHORITATIVE_BUNDLE_SCHEMA_VERSION_V2",
    "PREPARATION_SCHEMA_VERSION",
    "PREPARATION_SCHEMA_VERSION_V2",
    "SchedulerLaunchpadAttemptIssuerV1",
]
