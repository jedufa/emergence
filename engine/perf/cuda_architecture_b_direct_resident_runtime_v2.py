"""Pure V2 terminal-result contract for direct resident CUDA trajectories.

This module validates metadata only.  It deliberately exposes no execution
entrypoint or native control facade because neither V2 native surface exists
yet.  A future adapter may use these helpers after obtaining a terminal result
from an independently qualified native implementation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import Any, Final


RESULT_SCHEMA_VERSION: Final[str] = (
    "eiel_cuda_architecture_b_direct_resident_terminal_result_v2"
)
NATIVE_OWNER_KIND: Final[str] = "native_extension"

TERMINAL_COMPLETED: Final[str] = "completed"
TERMINAL_CANCELLED: Final[str] = "cancelled"
TERMINAL_FAILED: Final[str] = "failed"
TERMINAL_STATUSES: Final[frozenset[str]] = frozenset(
    {TERMINAL_COMPLETED, TERMINAL_CANCELLED, TERMINAL_FAILED}
)

OWNER_INTEGRITY_INTACT: Final[str] = "intact"
OWNER_INTEGRITY_SUSPECT_AND_SEALED: Final[str] = "suspect_and_sealed"

CONTINUATION_NOT_APPLICABLE: Final[str] = "not_applicable_terminal_complete"
CONTINUATION_NEW_NATIVE_OWNER_FROM_DURABLE_CHECKPOINT_ONLY: Final[str] = (
    "new_native_owner_from_durable_checkpoint_only"
)
CONTINUATION_FORBIDDEN: Final[str] = "forbidden_without_exact_durable_checkpoint"

OWNER_IDENTITY_FIELDS: Final[frozenset[str]] = frozenset(
    {"kind", "run_id", "controller_id", "owner_epoch"}
)
LAST_SETTLED_WINDOW_FIELDS: Final[frozenset[str]] = frozenset(
    {"start_generation", "completed_generations", "next_generation"}
)
CHECKPOINT_DIGEST_PAIRS: Final[tuple[tuple[str, str], ...]] = (
    ("authoritative_state_sha256", "checkpoint_state_sha256"),
    ("authoritative_canonical_sha256", "checkpoint_canonical_sha256"),
    ("authoritative_rng_sha256", "checkpoint_rng_sha256"),
    (
        "authoritative_event_cursor_sha256",
        "checkpoint_event_cursor_sha256",
    ),
)
CHECKPOINT_DIGEST_FIELDS: Final[frozenset[str]] = frozenset(
    field for pair in CHECKPOINT_DIGEST_PAIRS for field in pair
)
TERMINAL_RESULT_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "owner_identity",
        "terminal_status",
        "start_generation",
        "requested_generations",
        "completed_generations",
        "next_generation",
        "completed_generations_by_cell",
        "last_settled_window",
        "cancelled",
        "failed",
        "cancellation",
        "failure",
        "checkpoint_required",
        "checkpoint_produced",
        "checkpoint_boundary",
        "checkpoint_digests",
        "checkpoint_durable",
        "continuation_policy",
        "owner_integrity",
        "hidden_fallback",
        "runtime_jit",
        "python_window_ownership",
    }
)

CANCELLATION_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "request_id",
        "reason",
        "cancel_sequence",
        "accepted_generation",
        "completed_tick",
        "next_tick",
        "consumed_next_generation",
    }
)
CANCELLATION_REASONS: Final[frozenset[str]] = frozenset(
    {"operator_requested", "signal_received", "supervisor_requested"}
)
FAILURE_FIELDS: Final[frozenset[str]] = frozenset(
    {"code", "stage", "message", "generation", "tick"}
)
FAILURE_STAGES: Final[frozenset[str]] = frozenset(
    {
        "import_validation",
        "import_copy",
        "run_request_validation",
        "generation",
        "turnover",
        "snapshot",
        "cancellation",
        "finalization",
        "api_state",
    }
)

_SHA256_HEX_LENGTH: Final[int] = 64
_MAX_COUNT: Final[int] = (1 << 63) - 1


class CudaArchitectureBDirectRuntimeV2ContractError(ValueError):
    """Raised when a V2 terminal result is malformed or contradictory."""


def _fail(path: str, message: str) -> None:
    raise CudaArchitectureBDirectRuntimeV2ContractError(f"{path}: {message}")


def _require_fields(
    value: Mapping[str, object], fields: frozenset[str], *, path: str
) -> None:
    missing = sorted(fields.difference(value))
    if missing:
        _fail(path, f"missing required fields: {', '.join(missing)}")


def _exact_int(
    value: object,
    *,
    path: str,
    minimum: int = 0,
    maximum: int = _MAX_COUNT,
) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        _fail(path, f"must be an exact integer in [{minimum}, {maximum}]")
    return value


def _exact_bool(value: object, *, path: str) -> bool:
    if type(value) is not bool:
        _fail(path, "must be an exact boolean")
    return value


def _stable_ascii_identifier(value: object, *, path: str) -> str:
    if (
        type(value) is not str
        or not value
        or len(value) > 256
        or value.strip() != value
        or not value.isascii()
        or not value.isprintable()
        or "\x00" in value
    ):
        _fail(path, "must be a nonempty stable ASCII identifier")
    return value


def _sha256_hex(value: object, *, path: str) -> str:
    if (
        type(value) is not str
        or len(value) != _SHA256_HEX_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        _fail(path, "must be exactly 64 lowercase hexadecimal characters")
    return value


def _json_safe_copy(value: object, *, path: str) -> Any:
    if value is None or type(value) in {str, bool, int}:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            _fail(path, "must not contain non-finite floats")
        return value
    if isinstance(value, Mapping):
        copied: dict[str, Any] = {}
        for key, item in value.items():
            if type(key) is not str:
                _fail(path, "mapping keys must be strings")
            copied[key] = _json_safe_copy(item, path=f"{path}.{key}")
        return copied
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [
            _json_safe_copy(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    _fail(path, "must contain only JSON-safe metadata")


def validate_owner_identity_v2(value: object) -> dict[str, Any]:
    """Validate and copy one native owner identity."""

    if not isinstance(value, Mapping):
        _fail("owner_identity", "must be a mapping")
    _require_fields(value, OWNER_IDENTITY_FIELDS, path="owner_identity")
    if value["kind"] != NATIVE_OWNER_KIND:
        _fail("owner_identity.kind", f"must equal {NATIVE_OWNER_KIND!r}")
    _stable_ascii_identifier(value["run_id"], path="owner_identity.run_id")
    _stable_ascii_identifier(
        value["controller_id"], path="owner_identity.controller_id"
    )
    _exact_int(value["owner_epoch"], path="owner_identity.owner_epoch", minimum=1)
    return _json_safe_copy(value, path="owner_identity")


def validate_checkpoint_digests_v2(value: object) -> dict[str, Any]:
    """Validate and copy a pairwise authoritative/checkpoint digest receipt."""

    if not isinstance(value, Mapping):
        _fail("checkpoint_digests", "must be a mapping")
    _require_fields(value, CHECKPOINT_DIGEST_FIELDS, path="checkpoint_digests")
    for field in sorted(CHECKPOINT_DIGEST_FIELDS):
        _sha256_hex(value[field], path=f"checkpoint_digests.{field}")
    return _json_safe_copy(value, path="checkpoint_digests")


def checkpoint_digests_are_exact_v2(value: object) -> bool:
    """Return whether every validated checkpoint digest matches its owner."""

    digests = validate_checkpoint_digests_v2(value)
    return all(
        digests[owner] == digests[checkpoint]
        for owner, checkpoint in CHECKPOINT_DIGEST_PAIRS
    )


def _validate_completed_by_cell(
    value: object,
    *,
    requested_generations: int,
    expected_cell_count: int | None,
) -> list[int]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        _fail("completed_generations_by_cell", "must be a nonempty sequence")
    if not value:
        _fail("completed_generations_by_cell", "must be a nonempty sequence")
    if expected_cell_count is not None and len(value) != expected_cell_count:
        _fail(
            "completed_generations_by_cell",
            f"must contain exactly {expected_cell_count} cells",
        )
    return [
        _exact_int(
            count,
            path=f"completed_generations_by_cell[{index}]",
            maximum=requested_generations,
        )
        for index, count in enumerate(value)
    ]


def validate_last_settled_window_v2(
    value: object,
    *,
    trajectory_start: int,
    completed_generations: int,
    trajectory_next: int,
) -> dict[str, Any] | None:
    if completed_generations == 0:
        if value is not None:
            _fail(
                "last_settled_window",
                "must be null when no generation has settled",
            )
        return None
    if not isinstance(value, Mapping):
        _fail("last_settled_window", "must be a mapping")
    _require_fields(value, LAST_SETTLED_WINDOW_FIELDS, path="last_settled_window")
    window_start = _exact_int(
        value["start_generation"],
        path="last_settled_window.start_generation",
        minimum=trajectory_start,
        maximum=trajectory_next - 1,
    )
    window_completed = _exact_int(
        value["completed_generations"],
        path="last_settled_window.completed_generations",
        minimum=1,
        maximum=completed_generations,
    )
    window_next = _exact_int(
        value["next_generation"],
        path="last_settled_window.next_generation",
        minimum=trajectory_start + 1,
    )
    if window_start + window_completed != window_next:
        _fail(
            "last_settled_window",
            "next_generation must equal start_generation + completed_generations",
        )
    if window_next != trajectory_next:
        _fail(
            "last_settled_window.next_generation",
            "must equal the trajectory next_generation",
        )
    return _json_safe_copy(value, path="last_settled_window")


def _validate_expected_identity(
    actual: Mapping[str, Any], expected_value: object
) -> None:
    expected = validate_owner_identity_v2(expected_value)
    for field in sorted(OWNER_IDENTITY_FIELDS):
        if actual[field] != expected[field]:
            _fail(f"owner_identity.{field}", "does not match expected owner")


def _validate_cancellation_v2(
    value: object,
    *,
    terminal_status: str,
    next_generation: int,
) -> dict[str, Any] | None:
    if terminal_status != TERMINAL_CANCELLED:
        if value is not None:
            _fail("cancellation", "must be null unless terminal_status is cancelled")
        return None
    if not isinstance(value, Mapping):
        _fail("cancellation", "must be a mapping for a cancelled terminal")
    if set(value) != CANCELLATION_FIELDS:
        _fail("cancellation", "must contain the exact cancellation field set")
    _stable_ascii_identifier(value["request_id"], path="cancellation.request_id")
    if value["reason"] not in CANCELLATION_REASONS:
        _fail("cancellation.reason", "is not supported")
    _exact_int(
        value["cancel_sequence"],
        path="cancellation.cancel_sequence",
        minimum=1,
    )
    accepted_generation = _exact_int(
        value["accepted_generation"],
        path="cancellation.accepted_generation",
    )
    completed_tick = _exact_int(
        value["completed_tick"],
        path="cancellation.completed_tick",
    )
    next_tick = _exact_int(value["next_tick"], path="cancellation.next_tick", minimum=1)
    if next_tick != completed_tick + 1:
        _fail("cancellation.next_tick", "must follow completed_tick exactly")
    consumed_boundary = _exact_int(
        value["consumed_next_generation"],
        path="cancellation.consumed_next_generation",
        minimum=1,
    )
    if consumed_boundary != accepted_generation + 1:
        _fail(
            "cancellation.consumed_next_generation",
            "must be the complete boundary after the accepted generation",
        )
    if consumed_boundary != next_generation:
        _fail(
            "cancellation.consumed_next_generation",
            "must equal the terminal next_generation boundary",
        )
    return _json_safe_copy(value, path="cancellation")


def _validate_failure_v2(
    value: object,
    *,
    terminal_status: str,
    start_generation: int,
    requested_generations: int,
) -> dict[str, Any] | None:
    if terminal_status != TERMINAL_FAILED:
        if value is not None:
            _fail("failure", "must be null unless terminal_status is failed")
        return None
    if not isinstance(value, Mapping):
        _fail("failure", "must be a mapping for a failed terminal")
    if set(value) != FAILURE_FIELDS:
        _fail("failure", "must contain the exact first-failure field set")
    _stable_ascii_identifier(value["code"], path="failure.code")
    if value["stage"] not in FAILURE_STAGES:
        _fail("failure.stage", "is not supported")
    _stable_ascii_identifier(value["message"], path="failure.message")
    _exact_int(
        value["generation"],
        path="failure.generation",
        minimum=start_generation,
        maximum=start_generation + requested_generations - 1,
    )
    _exact_int(value["tick"], path="failure.tick")
    return _json_safe_copy(value, path="failure")


def validate_terminal_result_v2(
    value: object,
    *,
    expected_owner_identity: object | None = None,
    expected_start_generation: int | None = None,
    expected_requested_generations: int | None = None,
    expected_cell_count: int | None = None,
) -> dict[str, Any]:
    """Validate and copy one native V2 terminal trajectory result.

    ``completed_generations`` is the common settled count.  Per-cell counts may
    be one generation ahead only for a failed in-flight window.  Checkpoint
    boundaries and all window coordinates are absolute next-generation values.
    """

    if not isinstance(value, Mapping):
        _fail("terminal_result", "must be a mapping")
    _require_fields(value, TERMINAL_RESULT_FIELDS, path="terminal_result")
    if value["schema_version"] != RESULT_SCHEMA_VERSION:
        _fail("schema_version", f"must equal {RESULT_SCHEMA_VERSION!r}")

    owner_identity = validate_owner_identity_v2(value["owner_identity"])
    if expected_owner_identity is not None:
        _validate_expected_identity(owner_identity, expected_owner_identity)

    start = _exact_int(value["start_generation"], path="start_generation")
    requested = _exact_int(
        value["requested_generations"],
        path="requested_generations",
        minimum=1,
    )
    completed = _exact_int(
        value["completed_generations"],
        path="completed_generations",
        maximum=requested,
    )
    next_generation = _exact_int(value["next_generation"], path="next_generation")
    if next_generation != start + completed:
        _fail(
            "next_generation",
            "must equal start_generation + completed_generations",
        )

    if expected_start_generation is not None:
        expected_start = _exact_int(
            expected_start_generation, path="expected_start_generation"
        )
        if start != expected_start:
            _fail("start_generation", "does not match the expected start")
    if expected_requested_generations is not None:
        expected_requested = _exact_int(
            expected_requested_generations,
            path="expected_requested_generations",
            minimum=1,
        )
        if requested != expected_requested:
            _fail(
                "requested_generations",
                "does not match the expected request",
            )
    if expected_cell_count is not None:
        expected_cell_count = _exact_int(
            expected_cell_count, path="expected_cell_count", minimum=1
        )

    completed_by_cell = _validate_completed_by_cell(
        value["completed_generations_by_cell"],
        requested_generations=requested,
        expected_cell_count=expected_cell_count,
    )
    if min(completed_by_cell) != completed:
        _fail(
            "completed_generations_by_cell",
            "minimum per-cell completion must equal completed_generations",
        )

    terminal_status = value["terminal_status"]
    if type(terminal_status) is not str or terminal_status not in TERMINAL_STATUSES:
        _fail("terminal_status", f"must be one of {sorted(TERMINAL_STATUSES)}")
    cancelled = _exact_bool(value["cancelled"], path="cancelled")
    failed = _exact_bool(value["failed"], path="failed")
    expected_flags = {
        TERMINAL_COMPLETED: (False, False),
        TERMINAL_CANCELLED: (True, False),
        TERMINAL_FAILED: (False, True),
    }[terminal_status]
    if (cancelled, failed) != expected_flags:
        _fail(
            "terminal_result",
            "cancelled and failed must be exclusive and match terminal_status",
        )

    cancellation = _validate_cancellation_v2(
        value["cancellation"],
        terminal_status=terminal_status,
        next_generation=next_generation,
    )
    failure = _validate_failure_v2(
        value["failure"],
        terminal_status=terminal_status,
        start_generation=start,
        requested_generations=requested,
    )

    if terminal_status == TERMINAL_COMPLETED:
        if completed != requested or any(
            count != requested for count in completed_by_cell
        ):
            _fail(
                "completed_generations_by_cell",
                "completed terminal requires every requested generation",
            )
    elif terminal_status == TERMINAL_CANCELLED:
        if completed >= requested or any(
            count != completed for count in completed_by_cell
        ):
            _fail(
                "completed_generations_by_cell",
                "cancelled terminal requires one common pre-completion boundary",
            )
    else:
        if completed >= requested or any(
            count > min(completed + 1, requested) for count in completed_by_cell
        ):
            _fail(
                "completed_generations_by_cell",
                "failed terminal permits only the common prefix and one in-flight row",
            )

    last_window = validate_last_settled_window_v2(
        value["last_settled_window"],
        trajectory_start=start,
        completed_generations=completed,
        trajectory_next=next_generation,
    )

    checkpoint_required = _exact_bool(
        value["checkpoint_required"], path="checkpoint_required"
    )
    checkpoint_produced = _exact_bool(
        value["checkpoint_produced"], path="checkpoint_produced"
    )
    checkpoint_durable = _exact_bool(
        value["checkpoint_durable"], path="checkpoint_durable"
    )
    if checkpoint_required is not (terminal_status != TERMINAL_COMPLETED):
        _fail(
            "checkpoint_required",
            "must be true exactly for cancelled or failed terminals",
        )

    checkpoint_boundary: int | None
    checkpoint_digests: dict[str, Any] | None
    digests_exact = False
    if checkpoint_produced:
        checkpoint_boundary = _exact_int(
            value["checkpoint_boundary"], path="checkpoint_boundary"
        )
        if checkpoint_boundary != next_generation:
            _fail(
                "checkpoint_boundary",
                "must equal the terminal next_generation boundary",
            )
        checkpoint_digests = validate_checkpoint_digests_v2(value["checkpoint_digests"])
        digests_exact = checkpoint_digests_are_exact_v2(checkpoint_digests)
    else:
        checkpoint_boundary = None
        checkpoint_digests = None
        if value["checkpoint_boundary"] is not None:
            _fail(
                "checkpoint_boundary",
                "must be null when no checkpoint was produced",
            )
        if value["checkpoint_digests"] is not None:
            _fail(
                "checkpoint_digests",
                "must be null when no checkpoint was produced",
            )
        if checkpoint_durable:
            _fail(
                "checkpoint_durable",
                "cannot be true when no checkpoint was produced",
            )

    if terminal_status == TERMINAL_CANCELLED and not (
        checkpoint_produced and checkpoint_durable and digests_exact
    ):
        _fail(
            "checkpoint_durable",
            "cancelled terminal requires an exact durable checkpoint",
        )

    owner_integrity = value["owner_integrity"]
    expected_integrity = (
        OWNER_INTEGRITY_SUSPECT_AND_SEALED
        if terminal_status == TERMINAL_FAILED
        else OWNER_INTEGRITY_INTACT
    )
    if owner_integrity != expected_integrity:
        _fail("owner_integrity", f"must equal {expected_integrity!r}")

    resumable = (
        terminal_status != TERMINAL_COMPLETED
        and checkpoint_produced
        and checkpoint_durable
        and digests_exact
    )
    expected_continuation = (
        CONTINUATION_NOT_APPLICABLE
        if terminal_status == TERMINAL_COMPLETED
        else (
            CONTINUATION_NEW_NATIVE_OWNER_FROM_DURABLE_CHECKPOINT_ONLY
            if resumable
            else CONTINUATION_FORBIDDEN
        )
    )
    if value["continuation_policy"] != expected_continuation:
        _fail(
            "continuation_policy",
            f"must equal {expected_continuation!r}",
        )

    for field in ("hidden_fallback", "runtime_jit", "python_window_ownership"):
        if _exact_bool(value[field], path=field):
            _fail(field, "must be false for native resident ownership")

    result = _json_safe_copy(value, path="terminal_result")
    result["owner_identity"] = owner_identity
    result["completed_generations_by_cell"] = completed_by_cell
    result["last_settled_window"] = last_window
    result["checkpoint_boundary"] = checkpoint_boundary
    result["checkpoint_digests"] = checkpoint_digests
    result["cancellation"] = cancellation
    result["failure"] = failure
    return result


__all__ = [
    "CANCELLATION_FIELDS",
    "CANCELLATION_REASONS",
    "CHECKPOINT_DIGEST_FIELDS",
    "CHECKPOINT_DIGEST_PAIRS",
    "CONTINUATION_FORBIDDEN",
    "CONTINUATION_NEW_NATIVE_OWNER_FROM_DURABLE_CHECKPOINT_ONLY",
    "CONTINUATION_NOT_APPLICABLE",
    "CudaArchitectureBDirectRuntimeV2ContractError",
    "FAILURE_FIELDS",
    "FAILURE_STAGES",
    "LAST_SETTLED_WINDOW_FIELDS",
    "NATIVE_OWNER_KIND",
    "OWNER_IDENTITY_FIELDS",
    "OWNER_INTEGRITY_INTACT",
    "OWNER_INTEGRITY_SUSPECT_AND_SEALED",
    "RESULT_SCHEMA_VERSION",
    "TERMINAL_CANCELLED",
    "TERMINAL_COMPLETED",
    "TERMINAL_FAILED",
    "TERMINAL_RESULT_FIELDS",
    "TERMINAL_STATUSES",
    "checkpoint_digests_are_exact_v2",
    "validate_checkpoint_digests_v2",
    "validate_last_settled_window_v2",
    "validate_owner_identity_v2",
    "validate_terminal_result_v2",
]
