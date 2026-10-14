"""Qualification-only callback adapter for the cross-path mechanic gate.

This module bridges the generic durable isolation runner to one explicit
resident-segment owner.  It cannot authorize or dispatch a production batch:
the candidate loader admits only the four upstream artifacts, every request is
bound to the preregistered mechanic authority, and the direct owner must return
complete transaction-level observations from the actual resident execution.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import hashlib
import json
from pathlib import Path
from typing import Any, Final

from engine.native_runtime.inheritance_cross_path_qualified_artifact_v1 import (
    CandidateInheritanceCrossPathArtifactV1,
    InheritanceCrossPathCandidateArtifactLoaderV1,
)

from .inheritance_cross_path_qualification_controls_v1 import (
    InheritanceCrossPathQualificationControlError,
    ResidentCapacityFaultControlV1,
    ResidentCapacityFaultObservationV1,
    ResidentFixedActionControlV1,
    ResidentFixedActionObservationV1,
)
from .mixed_mode_resident_isolation_gate import (
    InheritanceAttemptV1,
    ReproductionPathReceiptV1,
    ResidentCellCaptureV1,
)
from .mixed_mode_resident_isolation_runner import (
    EXECUTION_OBSERVATION_SCHEMA_VERSION,
    ResidentIsolationExecutionRequestV1,
    bind_resident_isolation_observation_v1,
)


RUN_PAYLOAD_SCHEMA_VERSION: Final[str] = (
    "eiel_inheritance_cross_path_mechanic_gate_run_payload_v1"
)
DIRECT_RESULT_SCHEMA_VERSION: Final[str] = (
    "eiel_inheritance_cross_path_direct_mechanic_segment_result_v1"
)
DIRECT_CELL_SCHEMA_VERSION: Final[str] = (
    "eiel_inheritance_cross_path_direct_mechanic_cell_observation_v1"
)
DIRECT_PATH_SCHEMA_VERSION: Final[str] = (
    "eiel_inheritance_cross_path_direct_mechanic_path_observation_v1"
)
DIRECT_ATTEMPT_SCHEMA_VERSION: Final[str] = (
    "eiel_inheritance_cross_path_direct_mechanic_attempt_observation_v1"
)
DIRECT_COMMIT_SCHEMA_VERSION: Final[str] = (
    "eiel_inheritance_cross_path_authoritative_birth_commit_v1"
)
CALLBACK_AUTHORITY_SCHEMA_VERSION: Final[str] = (
    "eiel_inheritance_cross_path_mechanic_callback_authority_v1"
)
QUALIFICATION_EVIDENCE_SCHEMA_VERSION: Final[str] = (
    "eiel_inheritance_cross_path_mechanic_callback_evidence_v1"
)
EXACT_FULL_STORAGE_MODE: Final[str] = "exact_full"

_HEX = frozenset("0123456789abcdef")


class InheritanceCrossPathMechanicGateAdapterError(RuntimeError):
    """Candidate execution or its exact mechanic evidence is incomplete."""


DirectMechanicSegmentOwnerV1 = Callable[..., Mapping[str, Any]]


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise InheritanceCrossPathMechanicGateAdapterError(
            "mechanic-gate payload must be canonical JSON data"
        ) from exc


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _digest(name: str, value: Any) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        raise InheritanceCrossPathMechanicGateAdapterError(
            f"{name} must be a lowercase SHA-256 digest"
        )
    return value


def _text(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise InheritanceCrossPathMechanicGateAdapterError(
            f"{name} must be non-empty text"
        )
    return value


def _object(
    value: Any,
    *,
    name: str,
    keys: frozenset[str],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise InheritanceCrossPathMechanicGateAdapterError(
            f"{name} must be an object"
        )
    actual = frozenset(value)
    if actual != keys:
        raise InheritanceCrossPathMechanicGateAdapterError(
            f"{name} field set is invalid; "
            f"missing={sorted(keys - actual)}, unexpected={sorted(actual - keys)}"
        )
    return dict(value)


def _sequence(value: Any, *, name: str) -> list[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise InheritanceCrossPathMechanicGateAdapterError(
            f"{name} must be an array"
        )
    return list(value)


def _nonnegative(name: str, value: Any) -> int:
    if type(value) is not int or value < 0:
        raise InheritanceCrossPathMechanicGateAdapterError(
            f"{name} must be a non-negative integer"
        )
    return value


def _string_ids(value: Any, *, name: str, minimum: int = 0) -> tuple[str, ...]:
    rows = _sequence(value, name=name)
    if len(rows) < minimum:
        raise InheritanceCrossPathMechanicGateAdapterError(
            f"{name} must contain at least {minimum} entries"
        )
    for item in rows:
        _text(name, item)
    if len(set(rows)) != len(rows):
        raise InheritanceCrossPathMechanicGateAdapterError(
            f"{name} entries must be unique"
        )
    return tuple(rows)


def _adapter_source_sha256() -> str:
    digest = hashlib.sha256()
    with Path(__file__).resolve().open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _callback_authority(
    *,
    candidate_authority_sha256: str,
    segment_owner_id: str,
    segment_owner_sha256: str,
) -> tuple[dict[str, Any], str]:
    payload = {
        "schema_version": CALLBACK_AUTHORITY_SCHEMA_VERSION,
        "authorization_scope": "mechanic_isolation_gate_only",
        "production_authority": False,
        "candidate_authority_sha256": candidate_authority_sha256,
        "adapter_source_sha256": _adapter_source_sha256(),
        "direct_segment_owner_id": segment_owner_id,
        "direct_segment_owner_sha256": segment_owner_sha256,
    }
    return payload, _sha256(payload)


def _parse_commit(value: Any) -> dict[str, Any]:
    row = _object(
        value,
        name="authoritative_commit",
        keys=frozenset(
            {
                "schema_version",
                "generation",
                "tick",
                "path_id",
                "selected_parent_ids",
                "child_agent_id",
                "donor_draw_ordinals",
                "commit_authority_sha256",
            }
        ),
    )
    if row["schema_version"] != DIRECT_COMMIT_SCHEMA_VERSION:
        raise InheritanceCrossPathMechanicGateAdapterError(
            "authoritative commit schema mismatch"
        )
    _nonnegative("authoritative_commit.generation", row["generation"])
    _nonnegative("authoritative_commit.tick", row["tick"])
    _text("authoritative_commit.path_id", row["path_id"])
    parents = _string_ids(
        row["selected_parent_ids"],
        name="authoritative_commit.selected_parent_ids",
        minimum=2,
    )
    if len(parents) != 2:
        raise InheritanceCrossPathMechanicGateAdapterError(
            "authoritative commit must name exactly primary and mate"
        )
    _text("authoritative_commit.child_agent_id", row["child_agent_id"])
    ordinals = _sequence(
        row["donor_draw_ordinals"],
        name="authoritative_commit.donor_draw_ordinals",
    )
    for ordinal in ordinals:
        _nonnegative("authoritative_commit.donor_draw_ordinals", ordinal)
    _digest(
        "authoritative_commit.commit_authority_sha256",
        row["commit_authority_sha256"],
    )
    return row


def _parse_attempt(
    value: Any,
    *,
    path_id: str,
) -> tuple[InheritanceAttemptV1, tuple[int, ...], dict[str, Any]]:
    row = _object(
        value,
        name="direct attempt",
        keys=frozenset(
            {
                "schema_version",
                "generation",
                "tick",
                "primary_eligible_parent_ids",
                "mate_eligible_parent_ids",
                "selected_parent_ids",
                "success",
                "contributing_parent_ids",
                "failure_code",
                "authoritative_commit",
                "selection_transaction_sha256",
            }
        ),
    )
    if row["schema_version"] != DIRECT_ATTEMPT_SCHEMA_VERSION:
        raise InheritanceCrossPathMechanicGateAdapterError(
            "direct attempt schema mismatch"
        )
    _nonnegative("direct attempt generation", row["generation"])
    _nonnegative("direct attempt tick", row["tick"])
    _digest(
        "direct attempt selection_transaction_sha256",
        row["selection_transaction_sha256"],
    )
    try:
        attempt = InheritanceAttemptV1(
            primary_eligible_parent_ids=_string_ids(
                row["primary_eligible_parent_ids"],
                name="primary_eligible_parent_ids",
                minimum=1,
            ),
            mate_eligible_parent_ids=_string_ids(
                row["mate_eligible_parent_ids"],
                name="mate_eligible_parent_ids",
                minimum=1,
            ),
            selected_parent_ids=tuple(
                _string_ids(
                    row["selected_parent_ids"],
                    name="selected_parent_ids",
                    minimum=2,
                )
            ),
            success=row["success"],
            contributing_parent_ids=tuple(
                _string_ids(
                    row["contributing_parent_ids"],
                    name="contributing_parent_ids",
                )
            ),
            failure_code=row["failure_code"],
        )
    except ValueError as exc:
        raise InheritanceCrossPathMechanicGateAdapterError(
            f"direct attempt violates role-specific exposure: {exc}"
        ) from exc
    commit_value = row["authoritative_commit"]
    if attempt.success:
        commit = _parse_commit(commit_value)
        if (
            commit["generation"] != row["generation"]
            or commit["tick"] != row["tick"]
            or commit["path_id"] != path_id
            or tuple(commit["selected_parent_ids"]) != attempt.selected_parent_ids
            or attempt.contributing_parent_ids != attempt.selected_parent_ids
        ):
            raise InheritanceCrossPathMechanicGateAdapterError(
                "successful sidecar attempt disagrees with authoritative commit"
            )
        donor_ordinals = tuple(commit["donor_draw_ordinals"])
    else:
        if commit_value is not None:
            raise InheritanceCrossPathMechanicGateAdapterError(
                "failed sidecar attempt cannot carry an authoritative commit"
            )
        commit = {}
        donor_ordinals = ()
    return attempt, donor_ordinals, row


def _parse_rng_counters(value: Any) -> tuple[tuple[str, int], ...]:
    rows = _sequence(value, name="rng_counters")
    result: list[tuple[str, int]] = []
    for value in rows:
        row = _object(
            value,
            name="rng_counter",
            keys=frozenset({"counter_id", "value"}),
        )
        result.append(
            (
                _text("rng_counter.counter_id", row["counter_id"]),
                _nonnegative("rng_counter.value", row["value"]),
            )
        )
    if len({name for name, _value in result}) != len(result):
        raise InheritanceCrossPathMechanicGateAdapterError(
            "rng counter IDs must be unique"
        )
    return tuple(result)


def _parse_path(
    value: Any,
    *,
    configured_path_id: str,
) -> tuple[ReproductionPathReceiptV1, dict[str, Any]]:
    row = _object(
        value,
        name="direct path",
        keys=frozenset(
            {
                "schema_version",
                "path_id",
                "enabled",
                "rng_counters",
                "writes",
                "attempts",
                "authoritative_attempt_count",
                "authoritative_success_count",
                "authoritative_failure_count",
                "sidecar_attempt_count",
                "selection_transaction_count",
                "eligibility_snapshot_transaction_count",
                "path_authority_sha256",
            }
        ),
    )
    if row["schema_version"] != DIRECT_PATH_SCHEMA_VERSION:
        raise InheritanceCrossPathMechanicGateAdapterError(
            "direct path schema mismatch"
        )
    path_id = _text("path_id", row["path_id"])
    if type(row["enabled"]) is not bool:
        raise InheritanceCrossPathMechanicGateAdapterError(
            "direct path enabled must be boolean"
        )
    _digest("path_authority_sha256", row["path_authority_sha256"])
    attempts: list[InheritanceAttemptV1] = []
    donor_ordinals: list[int] = []
    for raw in _sequence(row["attempts"], name="attempts"):
        attempt, donors, _raw = _parse_attempt(raw, path_id=path_id)
        attempts.append(attempt)
        donor_ordinals.extend(donors)
    attempt_count = len(attempts)
    successes = sum(attempt.success for attempt in attempts)
    failures = attempt_count - successes
    observed_counts = {
        "authoritative_attempt_count": attempt_count,
        "authoritative_success_count": successes,
        "authoritative_failure_count": failures,
        "sidecar_attempt_count": attempt_count,
        "selection_transaction_count": attempt_count,
        "eligibility_snapshot_transaction_count": attempt_count,
    }
    if any(row[name] != expected for name, expected in observed_counts.items()):
        raise InheritanceCrossPathMechanicGateAdapterError(
            "path event counts do not reconcile one authoritative selection "
            "transaction to one sidecar attempt"
        )
    writes = _object(
        row["writes"],
        name="path writes",
        keys=frozenset({"counter", "sidecar", "event_row", "rng_state"}),
    )
    for name, count in writes.items():
        _nonnegative(f"path writes.{name}", count)
    enabled = row["enabled"]
    should_enable = path_id == configured_path_id
    if enabled != should_enable:
        raise InheritanceCrossPathMechanicGateAdapterError(
            "direct path enablement differs from registered cell configuration"
        )
    rng_counters = _parse_rng_counters(row["rng_counters"])
    if not enabled and (
        attempts
        or donor_ordinals
        or rng_counters
        or any(writes.values())
        or any(row[name] != 0 for name in observed_counts)
    ):
        raise InheritanceCrossPathMechanicGateAdapterError(
            "disabled path consumed RNG, selected parents, or wrote state"
        )
    receipt = ReproductionPathReceiptV1(
        path_id=path_id,
        enabled=enabled,
        rng_counters=rng_counters,
        donor_draw_ordinals=tuple(donor_ordinals),
        counter_writes=writes["counter"],
        sidecar_writes=writes["sidecar"],
        event_row_writes=writes["event_row"],
        rng_state_writes=writes["rng_state"],
        attempts=tuple(attempts),
    )
    return receipt, row


def _parse_owner_boundary(
    value: Any,
    *,
    fresh_owner: bool,
) -> dict[str, Any]:
    row = _object(
        value,
        name="owner_boundary",
        keys=frozenset(
            {
                "source_owner_id",
                "terminal_owner_id",
                "checkpoint_reloaded",
                "checkpoint_source_sha256",
                "restored_source_sha256",
            }
        ),
    )
    for name in ("source_owner_id", "terminal_owner_id"):
        _text(f"owner_boundary.{name}", row[name])
    for name in ("checkpoint_source_sha256", "restored_source_sha256"):
        _digest(f"owner_boundary.{name}", row[name])
    if type(row["checkpoint_reloaded"]) is not bool:
        raise InheritanceCrossPathMechanicGateAdapterError(
            "owner_boundary.checkpoint_reloaded must be boolean"
        )
    if fresh_owner:
        if (
            row["source_owner_id"] == row["terminal_owner_id"]
            or row["checkpoint_reloaded"] is not True
            or row["checkpoint_source_sha256"]
            != row["restored_source_sha256"]
        ):
            raise InheritanceCrossPathMechanicGateAdapterError(
                "fresh-owner observation lacks exact checkpoint reload"
            )
    elif row["checkpoint_reloaded"]:
        raise InheritanceCrossPathMechanicGateAdapterError(
            "non-fresh-owner observation claims a checkpoint reload"
        )
    return row


def _parse_fixed_action(
    control_value: Any,
    observation_value: Any,
) -> tuple[ResidentFixedActionControlV1, ResidentFixedActionObservationV1]:
    try:
        control = ResidentFixedActionControlV1.from_payload(control_value)
        observation = ResidentFixedActionObservationV1.from_payload(
            observation_value
        )
        observation.validate_against(control)
    except InheritanceCrossPathQualificationControlError as exc:
        raise InheritanceCrossPathMechanicGateAdapterError(
            f"fixed-action qualification receipt is invalid: {exc}"
        ) from exc
    return control, observation


def _parse_capacity_fault(
    control_value: Any,
    observation_value: Any,
    *,
    registered: Any,
    cell_authority_sha256: str,
    peer_count: int,
) -> tuple[
    ResidentCapacityFaultControlV1,
    ResidentCapacityFaultObservationV1,
]:
    try:
        control = ResidentCapacityFaultControlV1.from_payload(control_value)
        observation = ResidentCapacityFaultObservationV1.from_payload(
            observation_value
        )
        observation.validate_against(control)
    except InheritanceCrossPathQualificationControlError as exc:
        raise InheritanceCrossPathMechanicGateAdapterError(
            f"capacity-fault qualification receipt is invalid: {exc}"
        ) from exc
    if (
        control.target_occurrence_id != registered.occurrence_id
        or control.target_occurrence_authority_sha256
        != _sha256(registered.payload())
        or control.target_cell_index != registered.batch_position
        or control.target_cell_authority_sha256 != cell_authority_sha256
        or observation.peer_count != peer_count
    ):
        raise InheritanceCrossPathMechanicGateAdapterError(
            "capacity-fault receipt differs from preregistered target or peers"
        )
    return control, observation


def _parse_cell(
    value: Any,
    *,
    registered: Any,
    path_ids: tuple[str, ...],
    fresh_owner: bool,
    reporting_enabled: bool,
    overflow_fault: Mapping[str, Any] | None,
    batch_size: int,
) -> tuple[ResidentCellCaptureV1, dict[str, Any], dict[str, Any]]:
    row = _object(
        value,
        name="direct cell",
        keys=frozenset(
            {
                "schema_version",
                "occurrence_id",
                "config",
                "batch_position",
                "neighbor_path_ids",
                "final_state_sha256",
                "pre_fault_state_sha256",
                "checkpoint_sidecar_sha256",
                "lifecycle_state_sha256",
                "final_authority_sha256",
                "pre_fault_authority_sha256",
                "founder_source_authority_sha256",
                "ok",
                "fail_closed",
                "error_code",
                "reporting_rows_present",
                "owner_boundary",
                "fixed_action_control",
                "fixed_action_observation",
                "capacity_fault_control",
                "capacity_fault_observation",
                "paths",
                "cell_authority_sha256",
            }
        ),
    )
    if row["schema_version"] != DIRECT_CELL_SCHEMA_VERSION:
        raise InheritanceCrossPathMechanicGateAdapterError(
            "direct cell schema mismatch"
        )
    if (
        row["occurrence_id"] != registered.occurrence_id
        or row["config"] != registered.config.payload()
        or row["batch_position"] != registered.batch_position
        or tuple(row["neighbor_path_ids"])
        != registered.expected_neighbor_path_ids
    ):
        raise InheritanceCrossPathMechanicGateAdapterError(
            "direct cell differs from preregistered topology or configuration"
        )
    for name in (
        "final_state_sha256",
        "pre_fault_state_sha256",
        "checkpoint_sidecar_sha256",
        "lifecycle_state_sha256",
        "final_authority_sha256",
        "founder_source_authority_sha256",
        "cell_authority_sha256",
    ):
        _digest(name, row[name])
    if row["pre_fault_authority_sha256"] is not None:
        _digest("pre_fault_authority_sha256", row["pre_fault_authority_sha256"])
    is_fault_target = (
        overflow_fault is not None
        and overflow_fault["target_occurrence_id"] == registered.occurrence_id
    )
    expected_reporting_rows = reporting_enabled and not is_fault_target
    if row["reporting_rows_present"] is not expected_reporting_rows:
        raise InheritanceCrossPathMechanicGateAdapterError(
            "reporting row presence differs from the preregistered mode"
        )
    _parse_owner_boundary(row["owner_boundary"], fresh_owner=fresh_owner)
    fixed_control, fixed_observation = _parse_fixed_action(
        row["fixed_action_control"],
        row["fixed_action_observation"],
    )
    if is_fault_target:
        if (
            row["capacity_fault_control"] is None
            or row["capacity_fault_observation"] is None
        ):
            raise InheritanceCrossPathMechanicGateAdapterError(
                "overflow target lacks typed capacity-fault receipts"
            )
        fault_control, fault_observation = _parse_capacity_fault(
            row["capacity_fault_control"],
            row["capacity_fault_observation"],
            registered=registered,
            cell_authority_sha256=row["cell_authority_sha256"],
            peer_count=batch_size - 1,
        )
        if (
            row["pre_fault_authority_sha256"]
            != fault_control.pre_fault_authority_sha256
            or row["error_code"] != overflow_fault["error_code"]
            or row["ok"] is not False
            or row["fail_closed"] is not True
        ):
            raise InheritanceCrossPathMechanicGateAdapterError(
                "overflow target state is not bound to its typed fault receipt"
            )
    else:
        if (
            row["capacity_fault_control"] is not None
            or row["capacity_fault_observation"] is not None
        ):
            raise InheritanceCrossPathMechanicGateAdapterError(
                "non-target cell carries capacity-fault authority"
            )
        fault_control = None
        fault_observation = None
        if row["pre_fault_authority_sha256"] is not None:
            raise InheritanceCrossPathMechanicGateAdapterError(
                "pre-fault authority requires typed capacity-fault receipts"
            )
    raw_paths = _sequence(row["paths"], name="paths")
    parsed = [
        _parse_path(
            raw,
            configured_path_id=registered.config.reproduction_path,
        )
        for raw in raw_paths
    ]
    receipts = tuple(receipt for receipt, _raw in parsed)
    if tuple(receipt.path_id for receipt in receipts) != path_ids:
        raise InheritanceCrossPathMechanicGateAdapterError(
            "direct path observations differ from preregistered path order"
        )
    active = next(
        (
            receipt
            for receipt in receipts
            if receipt.path_id == registered.config.reproduction_path
        ),
        None,
    )
    if (
        active is None
        or not active.enabled
        or len(active.donor_draw_ordinals)
        != (
            active.successful_births
            * registered.config.donor_draws_per_successful_birth
        )
    ):
        raise InheritanceCrossPathMechanicGateAdapterError(
            "authoritative donor rows do not match the registered per-birth budget"
        )
    try:
        capture = ResidentCellCaptureV1(
            occurrence_id=row["occurrence_id"],
            config=registered.config,
            batch_position=row["batch_position"],
            neighbor_path_ids=tuple(row["neighbor_path_ids"]),
            final_state_sha256=row["final_state_sha256"],
            pre_fault_state_sha256=row["pre_fault_state_sha256"],
            checkpoint_sidecar_sha256=row["checkpoint_sidecar_sha256"],
            lifecycle_state_sha256=row["lifecycle_state_sha256"],
            final_authority_sha256=row["final_authority_sha256"],
            pre_fault_authority_sha256=row["pre_fault_authority_sha256"],
            ok=row["ok"],
            fail_closed=row["fail_closed"],
            error_code=row["error_code"],
            path_receipts=receipts,
        )
    except ValueError as exc:
        raise InheritanceCrossPathMechanicGateAdapterError(
            f"direct cell violates capture contract: {exc}"
        ) from exc
    authorities = {
        "occurrence_id": registered.occurrence_id,
        "fixed_action_control_authority_sha256": (
            fixed_control.control_authority_sha256
        ),
        "fixed_action_observation_authority_sha256": (
            fixed_observation.observation_authority_sha256
        ),
        "capacity_fault_control_authority_sha256": (
            None
            if fault_control is None
            else fault_control.control_authority_sha256
        ),
        "capacity_fault_observation_authority_sha256": (
            None
            if fault_observation is None
            else fault_observation.observation_authority_sha256
        ),
    }
    return capture, row, authorities


class InheritanceCrossPathMechanicGateCallbackV1:
    """Tombstone for the retired standalone mechanic-gate callback."""

    def __init__(
        self,
        *,
        candidate_loader: InheritanceCrossPathCandidateArtifactLoaderV1,
        direct_segment_owner: DirectMechanicSegmentOwnerV1,
        direct_segment_owner_id: str,
        direct_segment_owner_sha256: str,
        path_ids: Sequence[str],
    ) -> None:
        raise InheritanceCrossPathMechanicGateAdapterError(
            "P12_RETIRED_INHERITANCE_AUTHORITY "
            "cross-path mechanic-gate callback construction is disabled"
        )
        # Historical implementation retained temporarily for evidence reading.
        if not isinstance(
            candidate_loader, InheritanceCrossPathCandidateArtifactLoaderV1
        ):
            raise InheritanceCrossPathMechanicGateAdapterError(
                "candidate_loader must be the qualification-only loader"
            )
        if not callable(direct_segment_owner):
            raise InheritanceCrossPathMechanicGateAdapterError(
                "direct_segment_owner must be callable"
            )
        self._candidate_loader = candidate_loader
        self._direct_segment_owner = direct_segment_owner
        self._direct_segment_owner_id = _text(
            "direct_segment_owner_id", direct_segment_owner_id
        )
        self._direct_segment_owner_sha256 = _digest(
            "direct_segment_owner_sha256", direct_segment_owner_sha256
        )
        self._path_ids = _string_ids(path_ids, name="path_ids", minimum=2)
        authority, digest = _callback_authority(
            candidate_authority_sha256=(
                candidate_loader.candidate_authority_sha256
            ),
            segment_owner_id=self._direct_segment_owner_id,
            segment_owner_sha256=self._direct_segment_owner_sha256,
        )
        self.callback_authority = authority
        self.execution_authority_sha256 = digest
        self._candidate: CandidateInheritanceCrossPathArtifactV1 | None = None

    def _loaded_candidate(self) -> CandidateInheritanceCrossPathArtifactV1:
        candidate = self._candidate
        if candidate is None:
            candidate = self._candidate_loader()
            if (
                candidate.production_authority is not False
                or candidate.candidate_authority_sha256
                != self._candidate_loader.candidate_authority_sha256
            ):
                raise InheritanceCrossPathMechanicGateAdapterError(
                    "candidate loader returned production or substituted authority"
                )
            self._candidate = candidate
        return candidate

    def __call__(
        self, request: ResidentIsolationExecutionRequestV1
    ) -> Mapping[str, Any]:
        if not isinstance(request, ResidentIsolationExecutionRequestV1):
            raise InheritanceCrossPathMechanicGateAdapterError(
                "callback request must be ResidentIsolationExecutionRequestV1"
            )
        if request.execution_authority_sha256 != self.execution_authority_sha256:
            raise InheritanceCrossPathMechanicGateAdapterError(
                "runner plan is not bound to this qualification callback authority"
            )
        payload = _object(
            request.run.payload()["execution_payload"],
            name="registered execution payload",
            keys=frozenset(
                {
                    "schema_version",
                    "candidate_authority_sha256",
                    "exposure_storage_mode",
                    "reporting_enabled",
                    "overflow_fault",
                    "segment_request",
                }
            ),
        )
        if (
            payload["schema_version"] != RUN_PAYLOAD_SCHEMA_VERSION
            or payload["candidate_authority_sha256"]
            != self._candidate_loader.candidate_authority_sha256
            or payload["exposure_storage_mode"] != EXACT_FULL_STORAGE_MODE
            or type(payload["reporting_enabled"]) is not bool
            or not isinstance(payload["segment_request"], Mapping)
            or not payload["segment_request"]
        ):
            raise InheritanceCrossPathMechanicGateAdapterError(
                "registered exact_full mechanic payload is invalid"
            )
        overflow_fault = payload["overflow_fault"]
        if overflow_fault is not None:
            overflow_fault = _object(
                overflow_fault,
                name="overflow_fault",
                keys=frozenset({"target_occurrence_id", "error_code"}),
            )
            _text(
                "overflow_fault.target_occurrence_id",
                overflow_fault["target_occurrence_id"],
            )
            _text("overflow_fault.error_code", overflow_fault["error_code"])
            if overflow_fault["target_occurrence_id"] not in {
                cell.occurrence_id for cell in request.run.cells
            }:
                raise InheritanceCrossPathMechanicGateAdapterError(
                    "overflow target is absent from the preregistered run"
                )
        candidate = self._loaded_candidate()
        result = self._direct_segment_owner(
            candidate=candidate,
            runner_request=request,
            segment_request=json.loads(
                _canonical_bytes(payload["segment_request"]).decode("ascii")
            ),
            reporting_enabled=payload["reporting_enabled"],
            overflow_fault=overflow_fault,
            fresh_owner=request.run.fresh_owner,
        )
        row = _object(
            result,
            name="direct segment result",
            keys=frozenset(
                {
                    "schema_version",
                    "callback_authority_sha256",
                    "candidate_authority_sha256",
                    "registered_run_sha256",
                    "run_id",
                    "fresh_owner",
                    "reporting_enabled",
                    "production_authority",
                    "captured_at_utc",
                    "cells",
                    "direct_result_sha256",
                }
            ),
        )
        direct_result_sha256 = row.pop("direct_result_sha256")
        _digest("direct_result_sha256", direct_result_sha256)
        if _sha256(row) != direct_result_sha256:
            raise InheritanceCrossPathMechanicGateAdapterError(
                "direct segment result digest mismatch"
            )
        expected = {
            "schema_version": DIRECT_RESULT_SCHEMA_VERSION,
            "callback_authority_sha256": self.execution_authority_sha256,
            "candidate_authority_sha256": candidate.candidate_authority_sha256,
            "registered_run_sha256": request.run.digest_sha256,
            "run_id": request.run.run_id,
            "fresh_owner": request.run.fresh_owner,
            "reporting_enabled": payload["reporting_enabled"],
            "production_authority": False,
        }
        for name, expected_value in expected.items():
            if row[name] != expected_value:
                raise InheritanceCrossPathMechanicGateAdapterError(
                    f"direct segment result {name} is not request-bound"
                )
        raw_cells = _sequence(row["cells"], name="direct result cells")
        if len(raw_cells) != len(request.run.cells):
            raise InheritanceCrossPathMechanicGateAdapterError(
                "direct result cell count differs from preregistration"
            )
        parsed = [
            _parse_cell(
                raw,
                registered=registered,
                path_ids=self._path_ids,
                fresh_owner=request.run.fresh_owner,
                reporting_enabled=payload["reporting_enabled"],
                overflow_fault=overflow_fault,
                batch_size=len(request.run.cells),
            )
            for raw, registered in zip(
                raw_cells, request.run.cells, strict=True
            )
        ]
        captures = [capture.payload() for capture, _raw, _authority in parsed]
        evidence = {
            "schema_version": QUALIFICATION_EVIDENCE_SCHEMA_VERSION,
            "authorization_scope": "mechanic_isolation_gate_only",
            "production_authority": False,
            "callback_authority": self.callback_authority,
            "callback_authority_sha256": self.execution_authority_sha256,
            "candidate_authority_sha256": candidate.candidate_authority_sha256,
            "direct_result_sha256": direct_result_sha256,
            "exact_full_role_specific_exposure": True,
            "authoritative_commit_reconciliation_required": True,
            "direct_cells": [raw for _capture, raw, _authority in parsed],
            "qualification_control_authorities": [
                authority for _capture, _raw, authority in parsed
            ],
        }
        return bind_resident_isolation_observation_v1(
            {
                "schema_version": EXECUTION_OBSERVATION_SCHEMA_VERSION,
                "runner_plan_sha256": request.runner_plan_sha256,
                "gate_plan_sha256": request.gate_plan_sha256,
                "execution_callback_id": request.execution_callback_id,
                "execution_authority_sha256": request.execution_authority_sha256,
                "registered_run_sha256": request.run.digest_sha256,
                "run_id": request.run.run_id,
                "run_kind": request.run.run_kind,
                "captured_at_utc": row["captured_at_utc"],
                "fresh_owner": request.run.fresh_owner,
                "cells": captures,
                "execution_evidence": evidence,
            }
        )


def bind_direct_mechanic_segment_result_v1(
    result_without_digest: Mapping[str, Any],
) -> dict[str, Any]:
    """Reject publication of retired standalone direct-mechanic evidence."""

    raise InheritanceCrossPathMechanicGateAdapterError(
        "P12_RETIRED_INHERITANCE_AUTHORITY "
        "cross-path direct-mechanic evidence binding is disabled"
    )
    # Historical implementation retained temporarily for evidence reading.
    if not isinstance(result_without_digest, Mapping):
        raise InheritanceCrossPathMechanicGateAdapterError(
            "direct result must be an object"
        )
    if "direct_result_sha256" in result_without_digest:
        raise InheritanceCrossPathMechanicGateAdapterError(
            "unbound direct result cannot contain direct_result_sha256"
        )
    value = json.loads(_canonical_bytes(result_without_digest).decode("ascii"))
    value["direct_result_sha256"] = _sha256(value)
    return value


__all__ = [
    "CALLBACK_AUTHORITY_SCHEMA_VERSION",
    "DIRECT_ATTEMPT_SCHEMA_VERSION",
    "DIRECT_CELL_SCHEMA_VERSION",
    "DIRECT_COMMIT_SCHEMA_VERSION",
    "DIRECT_PATH_SCHEMA_VERSION",
    "DIRECT_RESULT_SCHEMA_VERSION",
    "DirectMechanicSegmentOwnerV1",
    "EXACT_FULL_STORAGE_MODE",
    "InheritanceCrossPathMechanicGateAdapterError",
    "InheritanceCrossPathMechanicGateCallbackV1",
    "QUALIFICATION_EVIDENCE_SCHEMA_VERSION",
    "RUN_PAYLOAD_SCHEMA_VERSION",
    "bind_direct_mechanic_segment_result_v1",
]
