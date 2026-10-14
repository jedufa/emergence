"""Pure CRCA contracts for unattended programme and start-lease operation.

The contracts in this module create no authority and perform no I/O.  They
bind existing Work, Scheduler, resource-observation, execution-occurrence,
Bridge, and notification owners into immutable content-addressed records.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Iterable, Mapping

from tools.scheduler.contracts import canonical_sha256 as _canonical_sha256_hex


# G0 retained a smaller provenance-only record under the original v1 token.
# This compatible conceptual V1 successor uses a distinct wire version rather
# than silently assigning new required fields to historical bytes.
PROGRAMME_AUTHORIZATION_SCHEMA_VERSION = "crca_programme_authorization_v1_1"
PROGRAMME_TRANSITION_SCHEMA_VERSION = "crca_programme_transition_receipt_v1"
PROGRAMME_ACTION_PLAN_SCHEMA_VERSION = "crca_programme_action_plan_v1"
PROGRAMME_ACTION_INTENT_SCHEMA_VERSION = "crca_programme_action_intent_v1"
PROGRAMME_ACTION_CLAIM_SCHEMA_VERSION = "crca_programme_action_claim_v1"
PROGRAMME_ACTION_SETTLEMENT_SCHEMA_VERSION = "crca_programme_action_settlement_v1"
PROGRAMME_DRIVER_LEASE_SCHEMA_VERSION = "crca_programme_driver_lease_v1"
RESOURCE_START_LEASE_SCHEMA_VERSION = "crca_resource_start_lease_v1"
LAUNCHPAD_ATTEMPT_CLAIM_SCHEMA_VERSION = "crca_launchpad_attempt_claim_v1"
LAUNCHPAD_RESOURCE_START_LEASE_SCHEMA_VERSION = (
    "crca_launchpad_resource_start_lease_v1"
)
RESOURCE_START_LEASE_EVENT_SCHEMA_VERSION = "crca_resource_start_lease_event_v1"
RESOURCE_START_LEASE_EVENT_SCHEMA_VERSION_V2 = (
    "crca_resource_start_lease_event_v2"
)
RESOURCE_START_LEASE_EVENT_SCHEMA_VERSION_V3 = (
    "crca_resource_start_lease_event_v3"
)
ATTEMPT_LEASE_BINDINGS_SCHEMA_VERSION = "crca_attempt_lease_bindings_v1"
TERMINAL_NOTIFICATION_INTENT_SCHEMA_VERSION = (
    "crca_terminal_notification_intent_v1"
)
TERMINAL_NOTIFICATION_DELIVERY_SCHEMA_VERSION = (
    "crca_terminal_notification_delivery_v1"
)
NOTIFICATION_DEDUPLICATION_PROOF_SCHEMA_VERSION = (
    "crca_notification_deduplication_proof_v1"
)
UNIFIED_STATUS_PROJECTION_SCHEMA_VERSION = "crca_unified_status_projection_v1"


class ProgrammeContractError(ValueError):
    """A fail-closed CRCA programme-contract violation."""


@dataclass(frozen=True, slots=True)
class ContentDigest:
    """Scheduler-local SHA-256 binding evidence with the shared wire shape."""

    algorithm: str
    hex: str

    def __post_init__(self) -> None:
        if self.algorithm != "sha256":
            raise ProgrammeContractError("digest algorithm must be sha256")
        if (
            not isinstance(self.hex, str)
            or len(self.hex) != 64
            or any(character not in "0123456789abcdef" for character in self.hex)
        ):
            raise ProgrammeContractError(
                "digest must be exactly 64 lowercase hexadecimal characters"
            )

    def to_dict(self) -> dict[str, str]:
        return {"algorithm": self.algorithm, "hex": self.hex}


def canonical_sha256(value: object) -> ContentDigest:
    return ContentDigest(algorithm="sha256", hex=_canonical_sha256_hex(value))


def _digest_dict(value: object, *, field: str) -> dict[str, str]:
    if isinstance(value, ContentDigest):
        return value.to_dict()
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        raw = to_dict()
        if isinstance(raw, Mapping):
            return ContentDigest(
                algorithm=str(raw.get("algorithm") or ""),
                hex=str(raw.get("hex") or ""),
            ).to_dict()
    raise ProgrammeContractError(f"{field} must be SHA-256 binding evidence")


def _digests_equal(left: object, right: object) -> bool:
    try:
        return _digest_dict(left, field="left") == _digest_dict(right, field="right")
    except ProgrammeContractError:
        return False


def _text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ProgrammeContractError(f"{field} must be a non-empty exact string")
    return value


def _digest(value: ContentDigest) -> dict[str, str]:
    return _digest_dict(value, field="digest")


def _optional_digest(value: ContentDigest | None) -> dict[str, str] | None:
    return None if value is None else _digest(value)


def _wire_digest(
    value: object, *, field: str, optional: bool = False
) -> ContentDigest | None:
    if value is None and optional:
        return None
    if not isinstance(value, Mapping):
        raise ProgrammeContractError(f"{field} must be a content digest")
    return ContentDigest(
        algorithm=str(value.get("algorithm") or ""),
        hex=str(value.get("hex") or ""),
    )


def _utc(value: object, *, field: str) -> datetime:
    text = _text(value, field=field)
    if len(text) != 20 or not text.endswith("Z"):
        raise ProgrammeContractError(
            f"{field} must use canonical second-resolution UTC"
        )
    try:
        return datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc
        )
    except ValueError as error:
        raise ProgrammeContractError(f"{field} is not a valid UTC timestamp") from error


def _unique_sorted(values: tuple[str, ...], *, field: str, required: bool = False) -> None:
    if not isinstance(values, tuple):
        raise ProgrammeContractError(f"{field} must be a tuple")
    for value in values:
        _text(value, field=field)
    if required and not values:
        raise ProgrammeContractError(f"{field} must not be empty")
    if tuple(sorted(set(values))) != values:
        raise ProgrammeContractError(f"{field} must be sorted and unique")


def _exact_fields(
    raw: Mapping[str, object], expected: set[str], *, label: str
) -> None:
    observed = set(raw)
    if observed != expected:
        missing = sorted(expected - observed)
        unknown = sorted(observed - expected)
        raise ProgrammeContractError(
            f"{label} fields differ; missing={missing}, unknown={unknown}"
        )


@dataclass(frozen=True, slots=True)
class ProgrammeGateV1:
    gate_id: str
    gate_class: str
    criterion: str
    allowed_successors: tuple[str, ...]

    def __post_init__(self) -> None:
        _text(self.gate_id, field="gate_id")
        _text(self.gate_class, field="gate_class")
        _text(self.criterion, field="criterion")
        _unique_sorted(self.allowed_successors, field="allowed_successors")

    def to_dict(self) -> dict[str, object]:
        return {
            "gate_id": self.gate_id,
            "gate_class": self.gate_class,
            "criterion": self.criterion,
            "allowed_successors": list(self.allowed_successors),
        }


@dataclass(frozen=True, slots=True)
class ProgrammeAuthorizationV1:
    programme_id: str
    operator_instruction_binding_digest: ContentDigest
    root_work_id: str
    root_work_digest: ContentDigest
    work_package_digest: ContentDigest
    objective: str
    scope_digest: ContentDigest
    source_digest: ContentDigest
    baseline_digest: ContentDigest
    initial_run_digest: ContentDigest | None
    gate_graph: tuple[ProgrammeGateV1, ...]
    allowed_recovery_codes: tuple[str, ...]
    allowed_fallback_codes: tuple[str, ...]
    budget_bounds_digest: ContentDigest
    provider_bounds_digest: ContentDigest
    bridge_conversation_id: str | None
    fail6_package_rule_digest: ContentDigest
    notification_recipient_digest: ContentDigest
    hard_stops: tuple[str, ...]
    human_decisions: tuple[str, ...]
    supersedes_authorization_digest: ContentDigest | None = None
    schema_version: str = PROGRAMME_AUTHORIZATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != PROGRAMME_AUTHORIZATION_SCHEMA_VERSION:
            raise ProgrammeContractError("unsupported ProgrammeAuthorization schema")
        _text(self.programme_id, field="programme_id")
        _text(self.root_work_id, field="root_work_id")
        _text(self.objective, field="objective")
        for value in (
            self.operator_instruction_binding_digest,
            self.root_work_digest,
            self.work_package_digest,
            self.scope_digest,
            self.source_digest,
            self.baseline_digest,
            self.budget_bounds_digest,
            self.provider_bounds_digest,
            self.fail6_package_rule_digest,
            self.notification_recipient_digest,
        ):
            _digest(value)
        _optional_digest(self.initial_run_digest)
        _optional_digest(self.supersedes_authorization_digest)
        if not isinstance(self.gate_graph, tuple) or not self.gate_graph:
            raise ProgrammeContractError("gate_graph must be a non-empty tuple")
        if any(not isinstance(gate, ProgrammeGateV1) for gate in self.gate_graph):
            raise ProgrammeContractError("gate_graph contains an invalid gate")
        gate_ids = tuple(gate.gate_id for gate in self.gate_graph)
        if len(set(gate_ids)) != len(gate_ids):
            raise ProgrammeContractError("gate_graph gate IDs must be unique")
        known = set(gate_ids)
        for gate in self.gate_graph:
            unknown = sorted(set(gate.allowed_successors) - known)
            if unknown:
                raise ProgrammeContractError(
                    f"gate {gate.gate_id} has unknown successors: {unknown}"
                )
        _unique_sorted(
            self.allowed_recovery_codes,
            field="allowed_recovery_codes",
            required=True,
        )
        _unique_sorted(self.allowed_fallback_codes, field="allowed_fallback_codes")
        _unique_sorted(self.hard_stops, field="hard_stops", required=True)
        _unique_sorted(self.human_decisions, field="human_decisions", required=True)
        if self.bridge_conversation_id is not None:
            _text(self.bridge_conversation_id, field="bridge_conversation_id")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "programme_id": self.programme_id,
            "operator_instruction_binding_digest": _digest(
                self.operator_instruction_binding_digest
            ),
            "root_work_id": self.root_work_id,
            "root_work_digest": _digest(self.root_work_digest),
            "work_package_digest": _digest(self.work_package_digest),
            "objective": self.objective,
            "scope_digest": _digest(self.scope_digest),
            "source_digest": _digest(self.source_digest),
            "baseline_digest": _digest(self.baseline_digest),
            "initial_run_digest": _optional_digest(self.initial_run_digest),
            "gate_graph": [gate.to_dict() for gate in self.gate_graph],
            "allowed_recovery_codes": list(self.allowed_recovery_codes),
            "allowed_fallback_codes": list(self.allowed_fallback_codes),
            "budget_bounds_digest": _digest(self.budget_bounds_digest),
            "provider_bounds_digest": _digest(self.provider_bounds_digest),
            "bridge_conversation_id": self.bridge_conversation_id,
            "fail6_package_rule_digest": _digest(self.fail6_package_rule_digest),
            "notification_recipient_digest": _digest(
                self.notification_recipient_digest
            ),
            "hard_stops": list(self.hard_stops),
            "human_decisions": list(self.human_decisions),
            "supersedes_authorization_digest": _optional_digest(
                self.supersedes_authorization_digest
            ),
        }

    def authorization_digest(self) -> ContentDigest:
        return canonical_sha256(self.to_dict())

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> "ProgrammeAuthorizationV1":
        def digest(field: str, *, optional: bool = False) -> ContentDigest | None:
            value = raw.get(field)
            if value is None and optional:
                return None
            if not isinstance(value, Mapping):
                raise ProgrammeContractError(f"{field} must be a content digest")
            return ContentDigest(
                algorithm=str(value.get("algorithm") or ""),
                hex=str(value.get("hex") or ""),
            )

        gates_raw = raw.get("gate_graph")
        if not isinstance(gates_raw, list):
            raise ProgrammeContractError("gate_graph must be an array")
        gates = tuple(
            ProgrammeGateV1(
                gate_id=str(dict(gate).get("gate_id") or ""),
                gate_class=str(dict(gate).get("gate_class") or ""),
                criterion=str(dict(gate).get("criterion") or ""),
                allowed_successors=tuple(
                    dict(gate).get("allowed_successors") or ()
                ),
            )
            for gate in gates_raw
            if isinstance(gate, Mapping)
        )
        if len(gates) != len(gates_raw):
            raise ProgrammeContractError("gate_graph contains a non-object")
        return cls(
            schema_version=str(raw.get("schema_version") or ""),
            programme_id=str(raw.get("programme_id") or ""),
            operator_instruction_binding_digest=digest(
                "operator_instruction_binding_digest"
            ),
            root_work_id=str(raw.get("root_work_id") or ""),
            root_work_digest=digest("root_work_digest"),
            work_package_digest=digest("work_package_digest"),
            objective=str(raw.get("objective") or ""),
            scope_digest=digest("scope_digest"),
            source_digest=digest("source_digest"),
            baseline_digest=digest("baseline_digest"),
            initial_run_digest=digest("initial_run_digest", optional=True),
            gate_graph=gates,
            allowed_recovery_codes=tuple(raw.get("allowed_recovery_codes") or ()),
            allowed_fallback_codes=tuple(raw.get("allowed_fallback_codes") or ()),
            budget_bounds_digest=digest("budget_bounds_digest"),
            provider_bounds_digest=digest("provider_bounds_digest"),
            bridge_conversation_id=(
                None
                if raw.get("bridge_conversation_id") is None
                else str(raw.get("bridge_conversation_id"))
            ),
            fail6_package_rule_digest=digest("fail6_package_rule_digest"),
            notification_recipient_digest=digest(
                "notification_recipient_digest"
            ),
            hard_stops=tuple(raw.get("hard_stops") or ()),
            human_decisions=tuple(raw.get("human_decisions") or ()),
            supersedes_authorization_digest=digest(
                "supersedes_authorization_digest", optional=True
            ),
        )


class ProgrammeTerminalStateV1(str, Enum):
    PROGRAM_COMPLETE = "PROGRAM_COMPLETE"
    FAIL_7_OPERATOR_REQUIRED = "FAIL-7_OPERATOR-REQUIRED"


class ProgrammeActionCodeV1(str, Enum):
    RECONCILE_OWNER = "RECONCILE_OWNER"
    EXECUTE_GATE_ACTION = "EXECUTE_GATE_ACTION"
    BRIDGE_HANDOFF = "BRIDGE_HANDOFF"
    TERMINAL_NOTIFICATION = "TERMINAL_NOTIFICATION"


@dataclass(frozen=True, slots=True)
class ProgrammeActionPlanV1:
    programme_id: str
    authorization_digest: ContentDigest
    ordinal: int
    gate_id: str
    action_code: ProgrammeActionCodeV1
    action_payload_digest: ContentDigest
    exact_owner_id: str
    exact_owner_digest: ContentDigest
    work_digest: ContentDigest
    session_digest: ContentDigest | None
    assignment_digest: ContentDigest | None
    dispatch_digest: ContentDigest | None
    scheduler_assignment_lease_digest: ContentDigest | None
    run_digest: ContentDigest | None
    attempt_digest: ContentDigest | None
    checkpoint_digest: ContentDigest | None
    success_gate_state: str
    next_safe_action: str | None
    wake_condition: str | None
    terminal_state: ProgrammeTerminalStateV1 | None
    created_at_utc: str
    schema_version: str = PROGRAMME_ACTION_PLAN_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != PROGRAMME_ACTION_PLAN_SCHEMA_VERSION:
            raise ProgrammeContractError("unsupported ProgrammeActionPlan schema")
        _text(self.programme_id, field="programme_id")
        _text(self.gate_id, field="gate_id")
        _text(self.exact_owner_id, field="exact_owner_id")
        _text(self.success_gate_state, field="success_gate_state")
        if (
            isinstance(self.ordinal, bool)
            or not isinstance(self.ordinal, int)
            or self.ordinal < 1
        ):
            raise ProgrammeContractError("action plan ordinal must be positive")
        if not isinstance(self.action_code, ProgrammeActionCodeV1):
            raise ProgrammeContractError("action plan code is invalid")
        for value in (
            self.authorization_digest,
            self.action_payload_digest,
            self.exact_owner_digest,
            self.work_digest,
        ):
            _digest(value)
        for value in (
            self.session_digest,
            self.assignment_digest,
            self.dispatch_digest,
            self.scheduler_assignment_lease_digest,
            self.run_digest,
            self.attempt_digest,
            self.checkpoint_digest,
        ):
            _optional_digest(value)
        if self.next_safe_action is not None:
            _text(self.next_safe_action, field="next_safe_action")
        if self.wake_condition is not None:
            _text(self.wake_condition, field="wake_condition")
        if self.terminal_state is not None and not isinstance(
            self.terminal_state, ProgrammeTerminalStateV1
        ):
            raise ProgrammeContractError("action plan terminal state is invalid")
        if self.terminal_state is not None and (
            self.next_safe_action is not None or self.wake_condition is not None
        ):
            raise ProgrammeContractError(
                "terminal action plan cannot retain a next action or wake condition"
            )
        _utc(self.created_at_utc, field="created_at_utc")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "programme_id": self.programme_id,
            "authorization_digest": _digest(self.authorization_digest),
            "ordinal": self.ordinal,
            "gate_id": self.gate_id,
            "action_code": self.action_code.value,
            "action_payload_digest": _digest(self.action_payload_digest),
            "exact_owner_id": self.exact_owner_id,
            "exact_owner_digest": _digest(self.exact_owner_digest),
            "work_digest": _digest(self.work_digest),
            "session_digest": _optional_digest(self.session_digest),
            "assignment_digest": _optional_digest(self.assignment_digest),
            "dispatch_digest": _optional_digest(self.dispatch_digest),
            "scheduler_assignment_lease_digest": _optional_digest(
                self.scheduler_assignment_lease_digest
            ),
            "run_digest": _optional_digest(self.run_digest),
            "attempt_digest": _optional_digest(self.attempt_digest),
            "checkpoint_digest": _optional_digest(self.checkpoint_digest),
            "success_gate_state": self.success_gate_state,
            "next_safe_action": self.next_safe_action,
            "wake_condition": self.wake_condition,
            "terminal_state": (
                None if self.terminal_state is None else self.terminal_state.value
            ),
            "created_at_utc": self.created_at_utc,
        }

    def plan_digest(self) -> ContentDigest:
        return canonical_sha256(self.to_dict())

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> "ProgrammeActionPlanV1":
        terminal = raw.get("terminal_state")
        return cls(
            schema_version=str(raw.get("schema_version") or ""),
            programme_id=str(raw.get("programme_id") or ""),
            authorization_digest=_wire_digest(
                raw.get("authorization_digest"), field="authorization_digest"
            ),
            ordinal=int(raw.get("ordinal") or 0),
            gate_id=str(raw.get("gate_id") or ""),
            action_code=ProgrammeActionCodeV1(str(raw.get("action_code") or "")),
            action_payload_digest=_wire_digest(
                raw.get("action_payload_digest"), field="action_payload_digest"
            ),
            exact_owner_id=str(raw.get("exact_owner_id") or ""),
            exact_owner_digest=_wire_digest(
                raw.get("exact_owner_digest"), field="exact_owner_digest"
            ),
            work_digest=_wire_digest(raw.get("work_digest"), field="work_digest"),
            session_digest=_wire_digest(
                raw.get("session_digest"), field="session_digest", optional=True
            ),
            assignment_digest=_wire_digest(
                raw.get("assignment_digest"),
                field="assignment_digest",
                optional=True,
            ),
            dispatch_digest=_wire_digest(
                raw.get("dispatch_digest"), field="dispatch_digest", optional=True
            ),
            scheduler_assignment_lease_digest=_wire_digest(
                raw.get("scheduler_assignment_lease_digest"),
                field="scheduler_assignment_lease_digest",
                optional=True,
            ),
            run_digest=_wire_digest(
                raw.get("run_digest"), field="run_digest", optional=True
            ),
            attempt_digest=_wire_digest(
                raw.get("attempt_digest"), field="attempt_digest", optional=True
            ),
            checkpoint_digest=_wire_digest(
                raw.get("checkpoint_digest"),
                field="checkpoint_digest",
                optional=True,
            ),
            success_gate_state=str(raw.get("success_gate_state") or ""),
            next_safe_action=(
                None
                if raw.get("next_safe_action") is None
                else str(raw.get("next_safe_action"))
            ),
            wake_condition=(
                None
                if raw.get("wake_condition") is None
                else str(raw.get("wake_condition"))
            ),
            terminal_state=(
                None
                if terminal is None
                else ProgrammeTerminalStateV1(str(terminal))
            ),
            created_at_utc=str(raw.get("created_at_utc") or ""),
        )


@dataclass(frozen=True, slots=True)
class ProgrammeActionIntentV1:
    programme_id: str
    authorization_digest: ContentDigest
    action_plan_digest: ContentDigest
    gate_id: str
    action_code: ProgrammeActionCodeV1
    action_payload_digest: ContentDigest
    exact_owner_id: str
    exact_owner_digest: ContentDigest
    fencing_token_digest: ContentDigest
    created_at_utc: str
    schema_version: str = PROGRAMME_ACTION_INTENT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != PROGRAMME_ACTION_INTENT_SCHEMA_VERSION:
            raise ProgrammeContractError("unsupported ProgrammeActionIntent schema")
        _text(self.programme_id, field="programme_id")
        _text(self.gate_id, field="gate_id")
        if not isinstance(self.action_code, ProgrammeActionCodeV1):
            raise ProgrammeContractError("action_code is invalid")
        _text(self.exact_owner_id, field="exact_owner_id")
        for value in (
            self.authorization_digest,
            self.action_plan_digest,
            self.action_payload_digest,
            self.exact_owner_digest,
            self.fencing_token_digest,
        ):
            _digest(value)
        _utc(self.created_at_utc, field="created_at_utc")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "programme_id": self.programme_id,
            "authorization_digest": _digest(self.authorization_digest),
            "action_plan_digest": _digest(self.action_plan_digest),
            "gate_id": self.gate_id,
            "action_code": self.action_code.value,
            "action_payload_digest": _digest(self.action_payload_digest),
            "exact_owner_id": self.exact_owner_id,
            "exact_owner_digest": _digest(self.exact_owner_digest),
            "fencing_token_digest": _digest(self.fencing_token_digest),
            "created_at_utc": self.created_at_utc,
        }

    def action_digest(self) -> ContentDigest:
        return canonical_sha256(self.to_dict())

    def action_intent_id(self) -> str:
        return f"CRCA-ACTION-{self.action_digest().hex[:24].upper()}"

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> "ProgrammeActionIntentV1":
        return cls(
            schema_version=str(raw.get("schema_version") or ""),
            programme_id=str(raw.get("programme_id") or ""),
            authorization_digest=_wire_digest(
                raw.get("authorization_digest"), field="authorization_digest"
            ),
            action_plan_digest=_wire_digest(
                raw.get("action_plan_digest"), field="action_plan_digest"
            ),
            gate_id=str(raw.get("gate_id") or ""),
            action_code=ProgrammeActionCodeV1(str(raw.get("action_code") or "")),
            action_payload_digest=_wire_digest(
                raw.get("action_payload_digest"), field="action_payload_digest"
            ),
            exact_owner_id=str(raw.get("exact_owner_id") or ""),
            exact_owner_digest=_wire_digest(
                raw.get("exact_owner_digest"), field="exact_owner_digest"
            ),
            fencing_token_digest=_wire_digest(
                raw.get("fencing_token_digest"), field="fencing_token_digest"
            ),
            created_at_utc=str(raw.get("created_at_utc") or ""),
        )


@dataclass(frozen=True, slots=True)
class ProgrammeActionClaimV1:
    action_intent_id: str
    action_intent_digest: ContentDigest
    programme_id: str
    authorization_digest: ContentDigest
    driver_lease_digest: ContentDigest
    driver_generation: int
    execution_fencing_token_digest: ContentDigest
    claim_generation: int
    prior_claim_digest: ContentDigest | None
    claimed_at_utc: str
    expires_at_utc: str
    schema_version: str = PROGRAMME_ACTION_CLAIM_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != PROGRAMME_ACTION_CLAIM_SCHEMA_VERSION:
            raise ProgrammeContractError("unsupported ProgrammeActionClaim schema")
        _text(self.action_intent_id, field="action_intent_id")
        _text(self.programme_id, field="programme_id")
        for value in (
            self.action_intent_digest,
            self.authorization_digest,
            self.driver_lease_digest,
            self.execution_fencing_token_digest,
        ):
            _digest(value)
        _optional_digest(self.prior_claim_digest)
        for field, value in (
            ("driver_generation", self.driver_generation),
            ("claim_generation", self.claim_generation),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ProgrammeContractError(f"{field} must be positive")
        claimed = _utc(self.claimed_at_utc, field="claimed_at_utc")
        expires = _utc(self.expires_at_utc, field="expires_at_utc")
        if claimed >= expires:
            raise ProgrammeContractError("action claim interval is invalid")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "action_intent_id": self.action_intent_id,
            "action_intent_digest": _digest(self.action_intent_digest),
            "programme_id": self.programme_id,
            "authorization_digest": _digest(self.authorization_digest),
            "driver_lease_digest": _digest(self.driver_lease_digest),
            "driver_generation": self.driver_generation,
            "execution_fencing_token_digest": _digest(
                self.execution_fencing_token_digest
            ),
            "claim_generation": self.claim_generation,
            "prior_claim_digest": _optional_digest(self.prior_claim_digest),
            "claimed_at_utc": self.claimed_at_utc,
            "expires_at_utc": self.expires_at_utc,
        }

    def claim_digest(self) -> ContentDigest:
        return canonical_sha256(self.to_dict())


@dataclass(frozen=True, slots=True)
class ProgrammeActionSettlementV1:
    action_intent_id: str
    action_intent_digest: ContentDigest
    action_claim_digest: ContentDigest
    execution_fencing_token_digest: ContentDigest
    effect_receipt_digest: ContentDigest
    result_receipt_digest: ContentDigest
    settlement_code: str
    settled_at_utc: str
    schema_version: str = PROGRAMME_ACTION_SETTLEMENT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != PROGRAMME_ACTION_SETTLEMENT_SCHEMA_VERSION:
            raise ProgrammeContractError("unsupported action settlement schema")
        _text(self.action_intent_id, field="action_intent_id")
        _text(self.settlement_code, field="settlement_code")
        _digest(self.action_intent_digest)
        _digest(self.action_claim_digest)
        _digest(self.execution_fencing_token_digest)
        _digest(self.effect_receipt_digest)
        _digest(self.result_receipt_digest)
        _utc(self.settled_at_utc, field="settled_at_utc")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "action_intent_id": self.action_intent_id,
            "action_intent_digest": _digest(self.action_intent_digest),
            "action_claim_digest": _digest(self.action_claim_digest),
            "execution_fencing_token_digest": _digest(
                self.execution_fencing_token_digest
            ),
            "effect_receipt_digest": _digest(self.effect_receipt_digest),
            "result_receipt_digest": _digest(self.result_receipt_digest),
            "settlement_code": self.settlement_code,
            "settled_at_utc": self.settled_at_utc,
        }

    def settlement_digest(self) -> ContentDigest:
        return canonical_sha256(self.to_dict())


@dataclass(frozen=True, slots=True)
class ProgrammeTransitionReceiptV1:
    programme_id: str
    authorization_digest: ContentDigest
    ordinal: int
    prior_transition_digest: ContentDigest | None
    gate_id: str
    gate_state: str
    recovery_state: str | None
    work_digest: ContentDigest
    session_digest: ContentDigest | None
    assignment_digest: ContentDigest | None
    dispatch_digest: ContentDigest | None
    scheduler_assignment_lease_digest: ContentDigest | None
    run_digest: ContentDigest | None
    attempt_digest: ContentDigest | None
    checkpoint_digest: ContentDigest | None
    action_intent_id: str | None
    action_intent_digest: ContentDigest | None
    effect_receipt_digest: ContentDigest | None
    result_receipt_digest: ContentDigest | None
    settlement_receipt_digest: ContentDigest | None
    next_safe_action: str | None
    wake_condition: str | None
    terminal_state: ProgrammeTerminalStateV1 | None
    recorded_at_utc: str
    schema_version: str = PROGRAMME_TRANSITION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != PROGRAMME_TRANSITION_SCHEMA_VERSION:
            raise ProgrammeContractError("unsupported transition schema")
        _text(self.programme_id, field="programme_id")
        _text(self.gate_id, field="gate_id")
        _text(self.gate_state, field="gate_state")
        if isinstance(self.ordinal, bool) or not isinstance(self.ordinal, int) or self.ordinal < 1:
            raise ProgrammeContractError("ordinal must be a positive integer")
        _digest(self.authorization_digest)
        _digest(self.work_digest)
        for value in (
            self.prior_transition_digest,
            self.session_digest,
            self.assignment_digest,
            self.dispatch_digest,
            self.scheduler_assignment_lease_digest,
            self.run_digest,
            self.attempt_digest,
            self.checkpoint_digest,
            self.action_intent_digest,
            self.effect_receipt_digest,
            self.result_receipt_digest,
            self.settlement_receipt_digest,
        ):
            _optional_digest(value)
        if (self.action_intent_id is None) != (self.action_intent_digest is None):
            raise ProgrammeContractError(
                "action intent ID and digest must be present together"
            )
        if self.action_intent_id is not None:
            _text(self.action_intent_id, field="action_intent_id")
        if self.recovery_state is not None:
            _text(self.recovery_state, field="recovery_state")
        if self.next_safe_action is not None:
            _text(self.next_safe_action, field="next_safe_action")
        if self.wake_condition is not None:
            _text(self.wake_condition, field="wake_condition")
        if self.terminal_state is not None and not isinstance(
            self.terminal_state, ProgrammeTerminalStateV1
        ):
            raise ProgrammeContractError("terminal_state is invalid")
        if self.terminal_state is None and not (
            self.next_safe_action or self.wake_condition
        ):
            raise ProgrammeContractError(
                "nonterminal transition needs a next action or wake condition"
            )
        if self.terminal_state is not None and (
            self.next_safe_action or self.wake_condition
        ):
            raise ProgrammeContractError(
                "terminal transition cannot claim a next action or wake condition"
            )
        _utc(self.recorded_at_utc, field="recorded_at_utc")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "programme_id": self.programme_id,
            "authorization_digest": _digest(self.authorization_digest),
            "ordinal": self.ordinal,
            "prior_transition_digest": _optional_digest(
                self.prior_transition_digest
            ),
            "gate_id": self.gate_id,
            "gate_state": self.gate_state,
            "recovery_state": self.recovery_state,
            "work_digest": _digest(self.work_digest),
            "session_digest": _optional_digest(self.session_digest),
            "assignment_digest": _optional_digest(self.assignment_digest),
            "dispatch_digest": _optional_digest(self.dispatch_digest),
            "scheduler_assignment_lease_digest": _optional_digest(
                self.scheduler_assignment_lease_digest
            ),
            "run_digest": _optional_digest(self.run_digest),
            "attempt_digest": _optional_digest(self.attempt_digest),
            "checkpoint_digest": _optional_digest(self.checkpoint_digest),
            "action_intent_id": self.action_intent_id,
            "action_intent_digest": _optional_digest(self.action_intent_digest),
            "effect_receipt_digest": _optional_digest(self.effect_receipt_digest),
            "result_receipt_digest": _optional_digest(self.result_receipt_digest),
            "settlement_receipt_digest": _optional_digest(
                self.settlement_receipt_digest
            ),
            "next_safe_action": self.next_safe_action,
            "wake_condition": self.wake_condition,
            "terminal_state": (
                None if self.terminal_state is None else self.terminal_state.value
            ),
            "recorded_at_utc": self.recorded_at_utc,
        }

    def transition_digest(self) -> ContentDigest:
        return canonical_sha256(self.to_dict())

    @classmethod
    def from_dict(
        cls, raw: Mapping[str, object]
    ) -> "ProgrammeTransitionReceiptV1":
        terminal = raw.get("terminal_state")

        def digest(field: str, *, optional: bool = False) -> ContentDigest | None:
            return _wire_digest(raw.get(field), field=field, optional=optional)

        return cls(
            schema_version=str(raw.get("schema_version") or ""),
            programme_id=str(raw.get("programme_id") or ""),
            authorization_digest=digest("authorization_digest"),
            ordinal=int(raw.get("ordinal") or 0),
            prior_transition_digest=digest(
                "prior_transition_digest", optional=True
            ),
            gate_id=str(raw.get("gate_id") or ""),
            gate_state=str(raw.get("gate_state") or ""),
            recovery_state=(
                None
                if raw.get("recovery_state") is None
                else str(raw.get("recovery_state"))
            ),
            work_digest=digest("work_digest"),
            session_digest=digest("session_digest", optional=True),
            assignment_digest=digest("assignment_digest", optional=True),
            dispatch_digest=digest("dispatch_digest", optional=True),
            scheduler_assignment_lease_digest=digest(
                "scheduler_assignment_lease_digest", optional=True
            ),
            run_digest=digest("run_digest", optional=True),
            attempt_digest=digest("attempt_digest", optional=True),
            checkpoint_digest=digest("checkpoint_digest", optional=True),
            action_intent_id=(
                None
                if raw.get("action_intent_id") is None
                else str(raw.get("action_intent_id"))
            ),
            action_intent_digest=digest("action_intent_digest", optional=True),
            effect_receipt_digest=digest("effect_receipt_digest", optional=True),
            result_receipt_digest=digest("result_receipt_digest", optional=True),
            settlement_receipt_digest=digest(
                "settlement_receipt_digest", optional=True
            ),
            next_safe_action=(
                None
                if raw.get("next_safe_action") is None
                else str(raw.get("next_safe_action"))
            ),
            wake_condition=(
                None
                if raw.get("wake_condition") is None
                else str(raw.get("wake_condition"))
            ),
            terminal_state=(
                None
                if terminal is None
                else ProgrammeTerminalStateV1(str(terminal))
            ),
            recorded_at_utc=str(raw.get("recorded_at_utc") or ""),
        )


@dataclass(frozen=True, slots=True)
class ProgrammeDriverLeaseV1:
    programme_id: str
    authorization_digest: ContentDigest
    driver_owner_id: str
    driver_owner_digest: ContentDigest
    process_id: int
    process_start_identity: str
    generation: int
    fencing_token_digest: ContentDigest
    prior_lease_digest: ContentDigest | None
    granted_at_utc: str
    heartbeat_at_utc: str
    expires_at_utc: str
    schema_version: str = PROGRAMME_DRIVER_LEASE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != PROGRAMME_DRIVER_LEASE_SCHEMA_VERSION:
            raise ProgrammeContractError("unsupported driver lease schema")
        _text(self.programme_id, field="programme_id")
        _text(self.driver_owner_id, field="driver_owner_id")
        _text(self.process_start_identity, field="process_start_identity")
        if isinstance(self.process_id, bool) or not isinstance(self.process_id, int) or self.process_id < 1:
            raise ProgrammeContractError("process_id must be positive")
        if isinstance(self.generation, bool) or not isinstance(self.generation, int) or self.generation < 1:
            raise ProgrammeContractError("generation must be positive")
        for value in (
            self.authorization_digest,
            self.driver_owner_digest,
            self.fencing_token_digest,
        ):
            _digest(value)
        _optional_digest(self.prior_lease_digest)
        granted = _utc(self.granted_at_utc, field="granted_at_utc")
        heartbeat = _utc(self.heartbeat_at_utc, field="heartbeat_at_utc")
        expires = _utc(self.expires_at_utc, field="expires_at_utc")
        if not (granted <= heartbeat < expires):
            raise ProgrammeContractError("driver lease time interval is invalid")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "programme_id": self.programme_id,
            "authorization_digest": _digest(self.authorization_digest),
            "driver_owner_id": self.driver_owner_id,
            "driver_owner_digest": _digest(self.driver_owner_digest),
            "process_id": self.process_id,
            "process_start_identity": self.process_start_identity,
            "generation": self.generation,
            "fencing_token_digest": _digest(self.fencing_token_digest),
            "prior_lease_digest": _optional_digest(self.prior_lease_digest),
            "granted_at_utc": self.granted_at_utc,
            "heartbeat_at_utc": self.heartbeat_at_utc,
            "expires_at_utc": self.expires_at_utc,
        }

    def lease_digest(self) -> ContentDigest:
        return canonical_sha256(self.to_dict())


@dataclass(frozen=True, slots=True)
class LaunchpadAttemptClaimV1:
    """One route-specific Launchpad execution claim issued only by Scheduler.

    The claim binds the complete projected execution identity while keeping the
    CRCA DTO kernel dependency pointed from composition to contracts.  Scheduler
    therefore stores exact DTO digests and wire records, but does not duplicate
    their semantic validator.
    """

    claim_id: str
    scheduler_owner_id: str
    scheduler_owner_digest: ContentDigest
    execution_owner_id: str
    execution_owner_digest: ContentDigest
    run_id: str
    attempt_id: str
    lineage_scope_digest: ContentDigest
    execution_request_digest: ContentDigest
    campaign_plan_digest: ContentDigest
    source_digest: ContentDigest
    baseline_digest: ContentDigest
    preflight_receipt_digest: ContentDigest
    lifecycle_binding_digest: ContentDigest
    lifecycle_contract_digest: ContentDigest
    run_root: str
    provider_id: str
    route_id: str
    device_id: str
    native_artifact_digest: ContentDigest
    batch_size: int
    ticks_per_dispatch: int
    execution_mode: str
    semantic_run_spec_digest: ContentDigest
    run_occurrence_digest: ContentDigest
    execution_implementation_digest: ContentDigest
    concretized_execution_digest: ContentDigest
    attempt_admission_request_digest: ContentDigest
    retry_basis_digest: ContentDigest
    post_admission_attempt_plan_digest: ContentDigest
    programme_binding_digest: ContentDigest
    initial_frontiers_digest: ContentDigest
    created_at_utc: str
    schema_version: str = LAUNCHPAD_ATTEMPT_CLAIM_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != LAUNCHPAD_ATTEMPT_CLAIM_SCHEMA_VERSION:
            raise ProgrammeContractError("unsupported Launchpad Attempt claim schema")
        for field in (
            "scheduler_owner_id",
            "execution_owner_id",
            "run_id",
            "attempt_id",
            "run_root",
            "provider_id",
            "route_id",
            "device_id",
            "execution_mode",
        ):
            _text(getattr(self, field), field=field)
        if not self.run_root.startswith("/"):
            raise ProgrammeContractError("run_root must be an exact absolute path")
        for field in ("batch_size", "ticks_per_dispatch"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ProgrammeContractError(f"{field} must be positive")
        for field in (
            "scheduler_owner_digest",
            "execution_owner_digest",
            "execution_request_digest",
            "campaign_plan_digest",
            "source_digest",
            "baseline_digest",
            "preflight_receipt_digest",
            "lifecycle_binding_digest",
            "lifecycle_contract_digest",
            "native_artifact_digest",
            "semantic_run_spec_digest",
            "run_occurrence_digest",
            "execution_implementation_digest",
            "concretized_execution_digest",
            "attempt_admission_request_digest",
            "retry_basis_digest",
            "post_admission_attempt_plan_digest",
            "programme_binding_digest",
            "initial_frontiers_digest",
            "lineage_scope_digest",
        ):
            _digest(getattr(self, field))
        _utc(self.created_at_utc, field="created_at_utc")
        expected = self.expected_claim_id()
        if self.claim_id != expected:
            raise ProgrammeContractError("Launchpad Attempt claim ID differs from identity")

    def identity_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "scheduler_owner_id": self.scheduler_owner_id,
            "scheduler_owner_digest": _digest(self.scheduler_owner_digest),
            "execution_owner_id": self.execution_owner_id,
            "execution_owner_digest": _digest(self.execution_owner_digest),
            "run_id": self.run_id,
            "attempt_id": self.attempt_id,
            "lineage_scope_digest": _digest(self.lineage_scope_digest),
            "execution_request_digest": _digest(self.execution_request_digest),
            "campaign_plan_digest": _digest(self.campaign_plan_digest),
            "source_digest": _digest(self.source_digest),
            "baseline_digest": _digest(self.baseline_digest),
            "preflight_receipt_digest": _digest(self.preflight_receipt_digest),
            "lifecycle_binding_digest": _digest(self.lifecycle_binding_digest),
            "lifecycle_contract_digest": _digest(self.lifecycle_contract_digest),
            "run_root": self.run_root,
            "provider_id": self.provider_id,
            "route_id": self.route_id,
            "device_id": self.device_id,
            "native_artifact_digest": _digest(self.native_artifact_digest),
            "batch_size": self.batch_size,
            "ticks_per_dispatch": self.ticks_per_dispatch,
            "execution_mode": self.execution_mode,
            "semantic_run_spec_digest": _digest(self.semantic_run_spec_digest),
            "run_occurrence_digest": _digest(self.run_occurrence_digest),
            "execution_implementation_digest": _digest(
                self.execution_implementation_digest
            ),
            "concretized_execution_digest": _digest(
                self.concretized_execution_digest
            ),
            "attempt_admission_request_digest": _digest(
                self.attempt_admission_request_digest
            ),
            "retry_basis_digest": _digest(self.retry_basis_digest),
            "post_admission_attempt_plan_digest": _digest(
                self.post_admission_attempt_plan_digest
            ),
            "programme_binding_digest": _digest(self.programme_binding_digest),
            "initial_frontiers_digest": _digest(self.initial_frontiers_digest),
            "created_at_utc": self.created_at_utc,
        }

    def expected_claim_id(self) -> str:
        return f"LPCLAIM-{canonical_sha256(self.identity_dict()).hex[:24].upper()}"

    def to_dict(self) -> dict[str, object]:
        return {"claim_id": self.claim_id, **self.identity_dict()}

    def claim_digest(self) -> ContentDigest:
        return canonical_sha256(self.to_dict())

    @classmethod
    def create(cls, **values: object) -> "LaunchpadAttemptClaimV1":
        provisional = cls.__new__(cls)
        for field in cls.__dataclass_fields__.values():
            if field.name == "claim_id":
                object.__setattr__(provisional, field.name, "LPCLAIM-PENDING")
            elif field.name == "schema_version":
                object.__setattr__(
                    provisional, field.name, LAUNCHPAD_ATTEMPT_CLAIM_SCHEMA_VERSION
                )
            else:
                object.__setattr__(provisional, field.name, values[field.name])
        object.__setattr__(provisional, "claim_id", provisional.expected_claim_id())
        provisional.__post_init__()
        return provisional

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> "LaunchpadAttemptClaimV1":
        expected = {field.name for field in cls.__dataclass_fields__.values()}
        _exact_fields(raw, expected, label="Launchpad Attempt claim")
        digest_fields = {
            field.name
            for field in cls.__dataclass_fields__.values()
            if field.name.endswith("_digest")
        }
        values: dict[str, object] = dict(raw)
        for field in digest_fields:
            values[field] = _wire_digest(raw[field], field=field)
        return cls(**values)


@dataclass(frozen=True, slots=True)
class LaunchpadResourceStartLeaseV1:
    """Scheduler start-window lease for Launchpad without Worker V1 aliases."""

    lease_id: str
    scheduler_owner_id: str
    scheduler_owner_digest: ContentDigest
    execution_claim_digest: ContentDigest
    run_id: str
    attempt_id: str
    attempt_admission_request_digest: ContentDigest
    resource_ids: tuple[str, ...]
    resource_scope_digest: ContentDigest
    resource_observation_id: str
    resource_observation_digest: ContentDigest
    resource_recommendation_id: str
    resource_recommendation_digest: ContentDigest
    live_provider_check_digest: ContentDigest
    lease_epoch: int
    fencing_token_digest: ContentDigest
    prior_lease_digest: ContentDigest | None
    issued_at_utc: str
    start_before_utc: str
    expires_at_utc: str
    schema_version: str = LAUNCHPAD_RESOURCE_START_LEASE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != LAUNCHPAD_RESOURCE_START_LEASE_SCHEMA_VERSION:
            raise ProgrammeContractError(
                "unsupported Launchpad resource-start lease schema"
            )
        for field in (
            "lease_id",
            "scheduler_owner_id",
            "run_id",
            "attempt_id",
            "resource_observation_id",
            "resource_recommendation_id",
        ):
            _text(getattr(self, field), field=field)
        _unique_sorted(self.resource_ids, field="resource_ids", required=True)
        for field in (
            "scheduler_owner_digest",
            "execution_claim_digest",
            "attempt_admission_request_digest",
            "resource_scope_digest",
            "resource_observation_digest",
            "resource_recommendation_digest",
            "live_provider_check_digest",
            "fencing_token_digest",
        ):
            _digest(getattr(self, field))
        _optional_digest(self.prior_lease_digest)
        if (
            isinstance(self.lease_epoch, bool)
            or not isinstance(self.lease_epoch, int)
            or self.lease_epoch < 1
        ):
            raise ProgrammeContractError("lease_epoch must be positive")
        issued = _utc(self.issued_at_utc, field="issued_at_utc")
        start_before = _utc(self.start_before_utc, field="start_before_utc")
        expires = _utc(self.expires_at_utc, field="expires_at_utc")
        if not (issued < start_before <= expires):
            raise ProgrammeContractError("resource-start lease interval is invalid")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "lease_id": self.lease_id,
            "scheduler_owner_id": self.scheduler_owner_id,
            "scheduler_owner_digest": _digest(self.scheduler_owner_digest),
            "execution_claim_digest": _digest(self.execution_claim_digest),
            "run_id": self.run_id,
            "attempt_id": self.attempt_id,
            "attempt_admission_request_digest": _digest(
                self.attempt_admission_request_digest
            ),
            "resource_ids": list(self.resource_ids),
            "resource_scope_digest": _digest(self.resource_scope_digest),
            "resource_observation_id": self.resource_observation_id,
            "resource_observation_digest": _digest(
                self.resource_observation_digest
            ),
            "resource_recommendation_id": self.resource_recommendation_id,
            "resource_recommendation_digest": _digest(
                self.resource_recommendation_digest
            ),
            "live_provider_check_digest": _digest(
                self.live_provider_check_digest
            ),
            "lease_epoch": self.lease_epoch,
            "fencing_token_digest": _digest(self.fencing_token_digest),
            "prior_lease_digest": _optional_digest(self.prior_lease_digest),
            "issued_at_utc": self.issued_at_utc,
            "start_before_utc": self.start_before_utc,
            "expires_at_utc": self.expires_at_utc,
        }

    def lease_digest(self) -> ContentDigest:
        return canonical_sha256(self.to_dict())

    def is_startable_at(self, at_utc: str) -> bool:
        observed = _utc(at_utc, field="at_utc")
        return _utc(self.issued_at_utc, field="issued_at_utc") <= observed < _utc(
            self.start_before_utc, field="start_before_utc"
        )

    @classmethod
    def from_dict(
        cls, raw: Mapping[str, object]
    ) -> "LaunchpadResourceStartLeaseV1":
        expected = {field.name for field in cls.__dataclass_fields__.values()}
        _exact_fields(raw, expected, label="Launchpad resource-start lease")
        values: dict[str, object] = dict(raw)
        values["resource_ids"] = tuple(raw["resource_ids"])
        for field in (
            "scheduler_owner_digest",
            "execution_claim_digest",
            "attempt_admission_request_digest",
            "resource_scope_digest",
            "resource_observation_digest",
            "resource_recommendation_digest",
            "live_provider_check_digest",
            "fencing_token_digest",
        ):
            values[field] = _wire_digest(raw[field], field=field)
        values["prior_lease_digest"] = _wire_digest(
            raw["prior_lease_digest"], field="prior_lease_digest", optional=True
        )
        return cls(**values)


@dataclass(frozen=True, slots=True)
class ResourceStartLeaseV1:
    lease_id: str
    scheduler_owner_id: str
    scheduler_owner_digest: ContentDigest
    assignment_lease_id: str
    assignment_lease_digest: ContentDigest
    attempt_admission_request_digest: ContentDigest
    resource_ids: tuple[str, ...]
    resource_scope_digest: ContentDigest
    resource_observation_id: str
    resource_observation_digest: ContentDigest
    resource_recommendation_id: str
    resource_recommendation_digest: ContentDigest
    live_provider_check_digest: ContentDigest
    generation: int
    fencing_token_digest: ContentDigest
    prior_lease_digest: ContentDigest | None
    issued_at_utc: str
    start_before_utc: str
    expires_at_utc: str
    schema_version: str = RESOURCE_START_LEASE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != RESOURCE_START_LEASE_SCHEMA_VERSION:
            raise ProgrammeContractError("unsupported resource-start lease schema")
        for field in (
            "lease_id",
            "scheduler_owner_id",
            "assignment_lease_id",
            "resource_observation_id",
            "resource_recommendation_id",
        ):
            _text(getattr(self, field), field=field)
        _unique_sorted(self.resource_ids, field="resource_ids", required=True)
        for value in (
            self.scheduler_owner_digest,
            self.assignment_lease_digest,
            self.attempt_admission_request_digest,
            self.resource_scope_digest,
            self.resource_observation_digest,
            self.resource_recommendation_digest,
            self.live_provider_check_digest,
            self.fencing_token_digest,
        ):
            _digest(value)
        _optional_digest(self.prior_lease_digest)
        if isinstance(self.generation, bool) or not isinstance(self.generation, int) or self.generation < 1:
            raise ProgrammeContractError("generation must be positive")
        issued = _utc(self.issued_at_utc, field="issued_at_utc")
        start_before = _utc(self.start_before_utc, field="start_before_utc")
        expires = _utc(self.expires_at_utc, field="expires_at_utc")
        if not (issued < start_before <= expires):
            raise ProgrammeContractError("resource-start lease interval is invalid")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "lease_id": self.lease_id,
            "scheduler_owner_id": self.scheduler_owner_id,
            "scheduler_owner_digest": _digest(self.scheduler_owner_digest),
            "assignment_lease_id": self.assignment_lease_id,
            "assignment_lease_digest": _digest(self.assignment_lease_digest),
            "attempt_admission_request_digest": _digest(
                self.attempt_admission_request_digest
            ),
            "resource_ids": list(self.resource_ids),
            "resource_scope_digest": _digest(self.resource_scope_digest),
            "resource_observation_id": self.resource_observation_id,
            "resource_observation_digest": _digest(
                self.resource_observation_digest
            ),
            "resource_recommendation_id": self.resource_recommendation_id,
            "resource_recommendation_digest": _digest(
                self.resource_recommendation_digest
            ),
            "live_provider_check_digest": _digest(
                self.live_provider_check_digest
            ),
            "generation": self.generation,
            "fencing_token_digest": _digest(self.fencing_token_digest),
            "prior_lease_digest": _optional_digest(self.prior_lease_digest),
            "issued_at_utc": self.issued_at_utc,
            "start_before_utc": self.start_before_utc,
            "expires_at_utc": self.expires_at_utc,
        }

    def lease_digest(self) -> ContentDigest:
        return canonical_sha256(self.to_dict())

    def is_startable_at(self, at_utc: str) -> bool:
        observed = _utc(at_utc, field="at_utc")
        return _utc(self.issued_at_utc, field="issued_at_utc") <= observed < _utc(
            self.start_before_utc, field="start_before_utc"
        )


class ResourceStartLeaseEventCodeV1(str, Enum):
    STARTED = "STARTED"
    RELEASED = "RELEASED"
    EXPIRED = "EXPIRED"
    RECLAIMED = "RECLAIMED"
    OWNER_DEAD_RECLAIMED = "OWNER_DEAD_RECLAIMED"
    LEGACY_GPU_MIGRATION_RECLAIMED = "LEGACY_GPU_MIGRATION_RECLAIMED"


@dataclass(frozen=True, slots=True)
class ResourceStartLeaseEventV1:
    lease_digest: ContentDigest
    code: ResourceStartLeaseEventCodeV1
    scheduler_owner_id: str
    fencing_token_digest: ContentDigest
    event_at_utc: str
    reason: str
    prior_event_digest: ContentDigest | None
    successor_lease_digest: ContentDigest | None
    schema_version: str = RESOURCE_START_LEASE_EVENT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != RESOURCE_START_LEASE_EVENT_SCHEMA_VERSION:
            raise ProgrammeContractError("unsupported resource-start lease event")
        _digest(self.lease_digest)
        _digest(self.fencing_token_digest)
        _optional_digest(self.prior_event_digest)
        _optional_digest(self.successor_lease_digest)
        if self.code not in {
            ResourceStartLeaseEventCodeV1.STARTED,
            ResourceStartLeaseEventCodeV1.RELEASED,
            ResourceStartLeaseEventCodeV1.EXPIRED,
            ResourceStartLeaseEventCodeV1.RECLAIMED,
        }:
            raise ProgrammeContractError("resource-start event code is invalid")
        _text(self.scheduler_owner_id, field="scheduler_owner_id")
        _text(self.reason, field="reason")
        _utc(self.event_at_utc, field="event_at_utc")
        successor_required = self.code is ResourceStartLeaseEventCodeV1.RECLAIMED
        if successor_required != (self.successor_lease_digest is not None):
            raise ProgrammeContractError("resource-start successor binding is invalid")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "lease_digest": _digest(self.lease_digest),
            "code": self.code.value,
            "scheduler_owner_id": self.scheduler_owner_id,
            "fencing_token_digest": _digest(self.fencing_token_digest),
            "event_at_utc": self.event_at_utc,
            "reason": self.reason,
            "prior_event_digest": _optional_digest(self.prior_event_digest),
            "successor_lease_digest": _optional_digest(
                self.successor_lease_digest
            ),
        }

    def event_digest(self) -> ContentDigest:
        return canonical_sha256(self.to_dict())


@dataclass(frozen=True, slots=True)
class ResourceStartLeaseEventV2:
    """Explicit proof-bound terminal event for one crash-dead GPU owner."""

    lease_digest: ContentDigest
    code: ResourceStartLeaseEventCodeV1
    scheduler_owner_id: str
    fencing_token_digest: ContentDigest
    event_at_utc: str
    reason: str
    prior_event_digest: ContentDigest
    reclaim_proof_digest: ContentDigest
    successor_lease_digest: None = None
    schema_version: str = RESOURCE_START_LEASE_EVENT_SCHEMA_VERSION_V2

    def __post_init__(self) -> None:
        if (
            self.schema_version != RESOURCE_START_LEASE_EVENT_SCHEMA_VERSION_V2
            or self.code is not ResourceStartLeaseEventCodeV1.OWNER_DEAD_RECLAIMED
            or self.successor_lease_digest is not None
        ):
            raise ProgrammeContractError(
                "unsupported crash-dead resource-start lease event"
            )
        for digest in (
            self.lease_digest,
            self.fencing_token_digest,
            self.prior_event_digest,
            self.reclaim_proof_digest,
        ):
            _digest(digest)
        _text(self.scheduler_owner_id, field="scheduler_owner_id")
        _text(self.reason, field="reason")
        _utc(self.event_at_utc, field="event_at_utc")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "lease_digest": _digest(self.lease_digest),
            "code": self.code.value,
            "scheduler_owner_id": self.scheduler_owner_id,
            "fencing_token_digest": _digest(self.fencing_token_digest),
            "event_at_utc": self.event_at_utc,
            "reason": self.reason,
            "prior_event_digest": _digest(self.prior_event_digest),
            "reclaim_proof_digest": _digest(self.reclaim_proof_digest),
            "successor_lease_digest": None,
        }

    def event_digest(self) -> ContentDigest:
        return canonical_sha256(self.to_dict())


@dataclass(frozen=True, slots=True)
class ResourceStartLeaseEventV3:
    """Manifest- and proof-bound terminal event for an exact legacy GPU row."""

    migration_id: str
    attempt_id: str
    resource_lease_id: str
    lease_digest: ContentDigest
    code: ResourceStartLeaseEventCodeV1
    scheduler_owner_id: str
    fencing_token_digest: ContentDigest
    event_at_utc: str
    reason: str
    prior_event_digest: ContentDigest
    migration_manifest_digest: ContentDigest
    migration_row_digest: ContentDigest
    reclaim_proof_digest: ContentDigest
    migration_authorization_digest: ContentDigest
    successor_lease_digest: None = None
    schema_version: str = RESOURCE_START_LEASE_EVENT_SCHEMA_VERSION_V3

    def __post_init__(self) -> None:
        if (
            self.schema_version != RESOURCE_START_LEASE_EVENT_SCHEMA_VERSION_V3
            or self.code
            is not ResourceStartLeaseEventCodeV1.LEGACY_GPU_MIGRATION_RECLAIMED
            or self.successor_lease_digest is not None
        ):
            raise ProgrammeContractError(
                "unsupported legacy GPU migration resource-start lease event"
            )
        for digest in (
            self.lease_digest,
            self.fencing_token_digest,
            self.prior_event_digest,
            self.migration_manifest_digest,
            self.migration_row_digest,
            self.reclaim_proof_digest,
            self.migration_authorization_digest,
        ):
            _digest(digest)
        _text(self.migration_id, field="migration_id")
        _text(self.attempt_id, field="attempt_id")
        _text(self.resource_lease_id, field="resource_lease_id")
        _text(self.scheduler_owner_id, field="scheduler_owner_id")
        _text(self.reason, field="reason")
        _utc(self.event_at_utc, field="event_at_utc")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "migration_id": self.migration_id,
            "attempt_id": self.attempt_id,
            "resource_lease_id": self.resource_lease_id,
            "lease_digest": _digest(self.lease_digest),
            "code": self.code.value,
            "scheduler_owner_id": self.scheduler_owner_id,
            "fencing_token_digest": _digest(self.fencing_token_digest),
            "event_at_utc": self.event_at_utc,
            "reason": self.reason,
            "prior_event_digest": _digest(self.prior_event_digest),
            "migration_manifest_digest": _digest(
                self.migration_manifest_digest
            ),
            "migration_row_digest": _digest(self.migration_row_digest),
            "reclaim_proof_digest": _digest(self.reclaim_proof_digest),
            "migration_authorization_digest": _digest(
                self.migration_authorization_digest
            ),
            "successor_lease_digest": None,
        }

    def authorization_intent(self) -> dict[str, object]:
        """Return the exact event intent sealed before this event may append."""

        return {
            key: value
            for key, value in self.to_dict().items()
            if key != "migration_authorization_digest"
        }

    def event_digest(self) -> ContentDigest:
        return canonical_sha256(self.to_dict())


SHARED_W32_GPU_POLICY_DIGEST_V1 = (
    "5eab1ee3dbc2237fc474939da48520db7a9832cd6b01f88b3f37b48e15a97b3a"
)
SHARED_W32_GPU_LANE_COUNT_V1 = 3
SHARED_W32_GPU_LANE_TOPOLOGY_V1 = "cuda-full-evolution-v2-w32-shared-v1"


class GpuResourceTopologyError(ValueError):
    """A Scheduler GPU resource identity or policy binding is not canonical."""


def gpu_uuid_equal_v1(left: object, right: object) -> bool:
    """Compare the two admitted physical-GPU UUID presentation forms."""

    if not isinstance(left, str) or not isinstance(right, str):
        return False
    normalized = []
    for value in (left, right):
        item = value.strip().lower()
        if item.startswith("gpu-"):
            item = item[4:]
        if not item:
            return False
        normalized.append(item)
    return normalized[0] == normalized[1]


def shared_gpu_lane_resource_id_v1(device_id: str, lane: int) -> str:
    """Return one Scheduler-owned child resource of a physical GPU."""

    if not isinstance(device_id, str) or not device_id or "/lane/" in device_id:
        raise GpuResourceTopologyError("shared GPU device ID is invalid")
    if type(lane) is not int or not 0 <= lane < SHARED_W32_GPU_LANE_COUNT_V1:
        raise GpuResourceTopologyError("shared GPU lane is out of range")
    return (
        f"gpu:{device_id}/lane/{SHARED_W32_GPU_LANE_TOPOLOGY_V1}/{lane}"
    )


def shared_gpu_lane_candidate_sets_v1(
    device_id: str,
) -> tuple[tuple[str, ...], ...]:
    """Return the closed three-lane Scheduler candidate topology."""

    return tuple(
        (shared_gpu_lane_resource_id_v1(device_id, lane),)
        for lane in range(SHARED_W32_GPU_LANE_COUNT_V1)
    )


def validate_shared_w32_gpu_policy_binding_v1(value: object) -> dict[str, object]:
    """Validate the Campaign policy by its frozen Scheduler admission digest."""

    if (
        not isinstance(value, Mapping)
        or canonical_sha256(dict(value)).hex != SHARED_W32_GPU_POLICY_DIGEST_V1
    ):
        raise GpuResourceTopologyError(
            "shared W32 GPU policy differs from the Scheduler admission binding"
        )
    return dict(value)


def _gpu_parent_resource_id_v1(resource_id: str) -> str | None:
    """Return the canonical parent for one scheduler-managed GPU lane."""

    if not isinstance(resource_id, str) or "/lane/" not in resource_id:
        return None
    parent, lane = resource_id.split("/lane/", 1)
    lane_parts = lane.split("/")
    if (
        not parent.startswith("gpu:")
        or len(parent) <= len("gpu:")
        or len(lane_parts) != 2
        or not lane_parts[0]
        or not lane_parts[1].isdigit()
    ):
        return None
    return parent


def resource_ids_conflict_v1(
    left: Iterable[str], right: Iterable[str]
) -> bool:
    """Apply exact-resource and parent-GPU/child-lane exclusion."""

    left_ids = set(left)
    right_ids = set(right)
    if left_ids.intersection(right_ids):
        return True
    left_parents = {
        parent
        for resource_id in left_ids
        if (parent := _gpu_parent_resource_id_v1(resource_id)) is not None
    }
    right_parents = {
        parent
        for resource_id in right_ids
        if (parent := _gpu_parent_resource_id_v1(resource_id)) is not None
    }
    return bool(left_ids.intersection(right_parents)) or bool(
        right_ids.intersection(left_parents)
    )


def require_exclusive_resource_start_lease_v1(
    candidate: ResourceStartLeaseV1 | LaunchpadResourceStartLeaseV1,
    active_leases: Iterable[
        ResourceStartLeaseV1 | LaunchpadResourceStartLeaseV1
    ],
) -> None:
    """Reject overlapping live start windows for any exact resource ID."""

    candidate_start = _utc(candidate.issued_at_utc, field="issued_at_utc")
    candidate_end = _utc(candidate.expires_at_utc, field="expires_at_utc")
    for current in active_leases:
        if current.lease_digest() == candidate.lease_digest():
            continue
        if not resource_ids_conflict_v1(
            candidate.resource_ids, current.resource_ids
        ):
            continue
        if isinstance(candidate, LaunchpadResourceStartLeaseV1) or isinstance(
            current, LaunchpadResourceStartLeaseV1
        ):
            raise ProgrammeContractError(
                "resource-start lease overlaps an active exact resource"
            )
        current_start = _utc(current.issued_at_utc, field="issued_at_utc")
        current_end = _utc(current.expires_at_utc, field="expires_at_utc")
        if max(candidate_start, current_start) < min(candidate_end, current_end):
            raise ProgrammeContractError(
                "resource-start lease overlaps an active exact resource"
            )


@dataclass(frozen=True, slots=True)
class AttemptLeaseBindingsV1:
    attempt_id: str
    scheduler_assignment_lease_id: str | None
    scheduler_assignment_lease_digest: ContentDigest | None
    scheduler_owner_id: str | None
    scheduler_owner_digest: ContentDigest | None
    resource_start_lease_id: str
    resource_start_lease_digest: ContentDigest
    resource_start_owner_id: str
    resource_start_owner_digest: ContentDigest
    canonical_continuation_lease_id: str
    canonical_continuation_lease_digest: ContentDigest
    continuation_owner_id: str
    continuation_owner_digest: ContentDigest
    schema_version: str = ATTEMPT_LEASE_BINDINGS_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ATTEMPT_LEASE_BINDINGS_SCHEMA_VERSION:
            raise ProgrammeContractError("unsupported Attempt lease bindings schema")
        _text(self.attempt_id, field="attempt_id")
        for field in (
            "resource_start_lease_id",
            "resource_start_owner_id",
            "canonical_continuation_lease_id",
            "continuation_owner_id",
        ):
            _text(getattr(self, field), field=field)
        for value in (
            self.resource_start_lease_digest,
            self.resource_start_owner_digest,
            self.canonical_continuation_lease_digest,
            self.continuation_owner_digest,
        ):
            _digest(value)
        optional_scheduler = (
            self.scheduler_assignment_lease_id,
            self.scheduler_assignment_lease_digest,
            self.scheduler_owner_id,
            self.scheduler_owner_digest,
        )
        if any(value is None for value in optional_scheduler) and any(
            value is not None for value in optional_scheduler
        ):
            raise ProgrammeContractError(
                "Scheduler lease ID, digest, and owner binding are all-or-none"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "attempt_id": self.attempt_id,
            "scheduler_assignment_lease_id": self.scheduler_assignment_lease_id,
            "scheduler_assignment_lease_digest": _optional_digest(
                self.scheduler_assignment_lease_digest
            ),
            "scheduler_owner_id": self.scheduler_owner_id,
            "scheduler_owner_digest": _optional_digest(self.scheduler_owner_digest),
            "resource_start_lease_id": self.resource_start_lease_id,
            "resource_start_lease_digest": _digest(
                self.resource_start_lease_digest
            ),
            "resource_start_owner_id": self.resource_start_owner_id,
            "resource_start_owner_digest": _digest(
                self.resource_start_owner_digest
            ),
            "canonical_continuation_lease_id": self.canonical_continuation_lease_id,
            "canonical_continuation_lease_digest": _digest(
                self.canonical_continuation_lease_digest
            ),
            "continuation_owner_id": self.continuation_owner_id,
            "continuation_owner_digest": _digest(self.continuation_owner_digest),
        }

    def binding_digest(self) -> ContentDigest:
        return canonical_sha256(self.to_dict())

    def validate_attempt_start(
        self,
        attempt: object,
        resource_lease: ResourceStartLeaseV1,
        continuation_lease: object,
    ) -> None:
        if self.attempt_id != str(getattr(attempt, "attempt_id", "")):
            raise ProgrammeContractError("Attempt lease binding has wrong Attempt ID")
        comparisons = (
            (
                self.scheduler_assignment_lease_digest,
                getattr(attempt, "scheduler_assignment_lease_digest", None),
                "scheduler assignment lease",
            ),
            (
                self.resource_start_lease_digest,
                getattr(attempt, "resource_start_lease_digest", None),
                "resource-start lease",
            ),
            (
                self.canonical_continuation_lease_digest,
                getattr(attempt, "canonical_continuation_lease_digest", None),
                "canonical continuation lease",
            ),
            (
                self.binding_digest(),
                getattr(attempt, "owner_binding_digest", None),
                "owner binding",
            ),
            (
                resource_lease.lease_digest(),
                self.resource_start_lease_digest,
                "resource-start record",
            ),
            (
                continuation_lease.lease_digest(),
                self.canonical_continuation_lease_digest,
                "continuation record",
            ),
        )
        for actual, expected, label in comparisons:
            if not _digests_equal(actual, expected):
                raise ProgrammeContractError(f"Attempt differs from {label}")
        if self.resource_start_lease_id != resource_lease.lease_id:
            raise ProgrammeContractError("resource-start lease ID differs")
        if self.resource_start_owner_id != resource_lease.scheduler_owner_id:
            raise ProgrammeContractError("resource-start owner ID differs")
        if self.resource_start_owner_digest != resource_lease.scheduler_owner_digest:
            raise ProgrammeContractError("resource-start owner digest differs")
        if self.canonical_continuation_lease_id != str(
            getattr(continuation_lease, "lease_id", "")
        ):
            raise ProgrammeContractError("continuation lease ID differs")
        if self.continuation_owner_id != getattr(
            continuation_lease, "owner_id", None
        ):
            raise ProgrammeContractError("continuation owner ID differs")
        expected_continuation_owner_digest = canonical_sha256(
            {
                "owner_id": getattr(continuation_lease, "owner_id", None),
                "lease_digest": _digest_dict(
                    continuation_lease.lease_digest(), field="continuation lease"
                ),
                "lease_epoch": getattr(continuation_lease, "lease_epoch", None),
                "fencing_token_digest": _digest_dict(
                    getattr(continuation_lease, "fencing_token_digest", None),
                    field="continuation fencing token",
                ),
            }
        )
        if not _digests_equal(
            self.continuation_owner_digest, expected_continuation_owner_digest
        ):
            raise ProgrammeContractError("continuation owner digest differs")

    def validate_runtime_records(
        self,
        *,
        assignment_lease_id: str,
        assignment_lease_digest: ContentDigest,
        scheduler_owner_id: str,
        scheduler_owner_digest: ContentDigest,
        resource_lease: ResourceStartLeaseV1,
        continuation_lease: Mapping[str, object],
    ) -> None:
        """Validate all three exact lease records before a provider can start."""

        required_scheduler = (
            self.scheduler_assignment_lease_id,
            self.scheduler_assignment_lease_digest,
            self.scheduler_owner_id,
            self.scheduler_owner_digest,
        )
        if any(value is None for value in required_scheduler):
            raise ProgrammeContractError(
                "runtime Attempt binding requires the Scheduler lease owner"
            )
        continuation_digest = canonical_sha256(dict(continuation_lease))
        fencing = _wire_digest(
            continuation_lease.get("fencing_token_digest"),
            field="continuation fencing token",
        )
        continuation_owner_digest = canonical_sha256(
            {
                "owner_id": continuation_lease.get("owner_id"),
                "lease_digest": continuation_digest.to_dict(),
                "lease_epoch": continuation_lease.get("lease_epoch"),
                "fencing_token_digest": fencing.to_dict(),
            }
        )
        comparisons = (
            (self.scheduler_assignment_lease_id, assignment_lease_id, "assignment lease ID"),
            (
                self.scheduler_assignment_lease_digest,
                assignment_lease_digest,
                "assignment lease digest",
            ),
            (self.scheduler_owner_id, scheduler_owner_id, "Scheduler owner ID"),
            (
                self.scheduler_owner_digest,
                scheduler_owner_digest,
                "Scheduler owner digest",
            ),
            (self.resource_start_lease_id, resource_lease.lease_id, "resource lease ID"),
            (
                self.resource_start_lease_digest,
                resource_lease.lease_digest(),
                "resource lease digest",
            ),
            (
                self.resource_start_owner_id,
                resource_lease.scheduler_owner_id,
                "resource owner ID",
            ),
            (
                self.resource_start_owner_digest,
                resource_lease.scheduler_owner_digest,
                "resource owner digest",
            ),
            (
                self.canonical_continuation_lease_id,
                str(continuation_lease.get("lease_id") or ""),
                "continuation lease ID",
            ),
            (
                self.canonical_continuation_lease_digest,
                continuation_digest,
                "continuation lease digest",
            ),
            (
                self.continuation_owner_id,
                str(continuation_lease.get("owner_id") or ""),
                "continuation owner ID",
            ),
            (
                self.continuation_owner_digest,
                continuation_owner_digest,
                "continuation owner digest",
            ),
            (
                self.attempt_id,
                str(continuation_lease.get("attempt_id") or ""),
                "Attempt ID",
            ),
        )
        for actual, expected, label in comparisons:
            if isinstance(actual, ContentDigest) or isinstance(expected, ContentDigest):
                if not _digests_equal(actual, expected):
                    raise ProgrammeContractError(f"runtime differs from {label}")
            elif actual != expected:
                raise ProgrammeContractError(f"runtime differs from {label}")


@dataclass(frozen=True, slots=True)
class TerminalNotificationIntentV1:
    programme_id: str
    terminal_state: ProgrammeTerminalStateV1
    terminal_transition_digest: ContentDigest
    terminal_programme_digest: ContentDigest
    recipient_digest: ContentDigest
    subject_digest: ContentDigest
    body_digest: ContentDigest
    created_at_utc: str
    schema_version: str = TERMINAL_NOTIFICATION_INTENT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != TERMINAL_NOTIFICATION_INTENT_SCHEMA_VERSION:
            raise ProgrammeContractError("unsupported terminal notification intent")
        _text(self.programme_id, field="programme_id")
        if not isinstance(self.terminal_state, ProgrammeTerminalStateV1):
            raise ProgrammeContractError("terminal notification state is invalid")
        for value in (
            self.terminal_transition_digest,
            self.terminal_programme_digest,
            self.recipient_digest,
            self.subject_digest,
            self.body_digest,
        ):
            _digest(value)
        _utc(self.created_at_utc, field="created_at_utc")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "programme_id": self.programme_id,
            "terminal_state": self.terminal_state.value,
            "terminal_transition_digest": _digest(
                self.terminal_transition_digest
            ),
            "terminal_programme_digest": _digest(
                self.terminal_programme_digest
            ),
            "recipient_digest": _digest(self.recipient_digest),
            "subject_digest": _digest(self.subject_digest),
            "body_digest": _digest(self.body_digest),
            "created_at_utc": self.created_at_utc,
        }

    def intent_digest(self) -> ContentDigest:
        return canonical_sha256(self.to_dict())

    def notification_intent_id(self) -> str:
        return f"CRCA-NOTIFY-{self.terminal_programme_digest.hex.upper()}"

    @classmethod
    def from_dict(
        cls, raw: Mapping[str, object]
    ) -> "TerminalNotificationIntentV1":
        return cls(
            schema_version=str(raw.get("schema_version") or ""),
            programme_id=str(raw.get("programme_id") or ""),
            terminal_state=ProgrammeTerminalStateV1(
                str(raw.get("terminal_state") or "")
            ),
            terminal_transition_digest=_wire_digest(
                raw.get("terminal_transition_digest"),
                field="terminal_transition_digest",
            ),
            terminal_programme_digest=_wire_digest(
                raw.get("terminal_programme_digest"),
                field="terminal_programme_digest",
            ),
            recipient_digest=_wire_digest(
                raw.get("recipient_digest"), field="recipient_digest"
            ),
            subject_digest=_wire_digest(
                raw.get("subject_digest"), field="subject_digest"
            ),
            body_digest=_wire_digest(
                raw.get("body_digest"), field="body_digest"
            ),
            created_at_utc=str(raw.get("created_at_utc") or ""),
        )


@dataclass(frozen=True, slots=True)
class NotificationDeduplicationProofV1:
    provider: str
    notification_intent_id: str
    idempotency_key: str
    capability_owner_id: str
    capability_receipt_digest: ContentDigest
    accepted_at_utc: str
    schema_version: str = NOTIFICATION_DEDUPLICATION_PROOF_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != NOTIFICATION_DEDUPLICATION_PROOF_SCHEMA_VERSION:
            raise ProgrammeContractError(
                "unsupported notification deduplication proof"
            )
        _text(self.provider, field="provider")
        _text(self.notification_intent_id, field="notification_intent_id")
        _text(self.idempotency_key, field="idempotency_key")
        _text(self.capability_owner_id, field="capability_owner_id")
        _digest(self.capability_receipt_digest)
        _utc(self.accepted_at_utc, field="accepted_at_utc")
        if self.idempotency_key != self.notification_intent_id:
            raise ProgrammeContractError(
                "notification idempotency key must be the exact intent ID"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "provider": self.provider,
            "notification_intent_id": self.notification_intent_id,
            "idempotency_key": self.idempotency_key,
            "capability_owner_id": self.capability_owner_id,
            "capability_receipt_digest": _digest(
                self.capability_receipt_digest
            ),
            "accepted_at_utc": self.accepted_at_utc,
        }

    def proof_digest(self) -> ContentDigest:
        return canonical_sha256(self.to_dict())

    @classmethod
    def from_dict(
        cls, raw: Mapping[str, object]
    ) -> "NotificationDeduplicationProofV1":
        return cls(
            schema_version=str(raw.get("schema_version") or ""),
            provider=str(raw.get("provider") or ""),
            notification_intent_id=str(
                raw.get("notification_intent_id") or ""
            ),
            idempotency_key=str(raw.get("idempotency_key") or ""),
            capability_owner_id=str(raw.get("capability_owner_id") or ""),
            capability_receipt_digest=_wire_digest(
                raw.get("capability_receipt_digest"),
                field="capability_receipt_digest",
            ),
            accepted_at_utc=str(raw.get("accepted_at_utc") or ""),
        )


@dataclass(frozen=True, slots=True)
class TerminalNotificationDeliveryV1:
    notification_intent_id: str
    notification_intent_digest: ContentDigest
    delivery_attempt: int
    provider: str
    provider_receipt_digest: ContentDigest
    delivered_at_utc: str
    deduplication_proof_digest: ContentDigest | None
    schema_version: str = TERMINAL_NOTIFICATION_DELIVERY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != TERMINAL_NOTIFICATION_DELIVERY_SCHEMA_VERSION:
            raise ProgrammeContractError("unsupported notification delivery schema")
        _text(self.notification_intent_id, field="notification_intent_id")
        _text(self.provider, field="provider")
        _digest(self.notification_intent_digest)
        _digest(self.provider_receipt_digest)
        _optional_digest(self.deduplication_proof_digest)
        if isinstance(self.delivery_attempt, bool) or not isinstance(self.delivery_attempt, int) or self.delivery_attempt < 1:
            raise ProgrammeContractError("delivery_attempt must be positive")
        _utc(self.delivered_at_utc, field="delivered_at_utc")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "notification_intent_id": self.notification_intent_id,
            "notification_intent_digest": _digest(
                self.notification_intent_digest
            ),
            "delivery_attempt": self.delivery_attempt,
            "provider": self.provider,
            "provider_receipt_digest": _digest(self.provider_receipt_digest),
            "delivered_at_utc": self.delivered_at_utc,
            "deduplication_proof_digest": _optional_digest(
                self.deduplication_proof_digest
            ),
            "effective_exactly_once": self.deduplication_proof_digest is not None,
            "delivery_semantics": (
                "EXACTLY_ONCE_DEDUPLICATED"
                if self.deduplication_proof_digest is not None
                else "AT_LEAST_ONCE_DUPLICATES_POSSIBLE"
            ),
        }

    def delivery_digest(self) -> ContentDigest:
        return canonical_sha256(self.to_dict())


@dataclass(frozen=True, slots=True)
class OwnedStatusFactV1:
    field: str
    value: object
    owner_id: str
    owner_digest: ContentDigest
    record_id: str
    record_digest: ContentDigest
    observed_at_utc: str
    fresh_until_utc: str

    def __post_init__(self) -> None:
        _text(self.field, field="field")
        _text(self.owner_id, field="owner_id")
        _text(self.record_id, field="record_id")
        _digest(self.owner_digest)
        _digest(self.record_digest)
        observed = _utc(self.observed_at_utc, field="observed_at_utc")
        fresh_until = _utc(self.fresh_until_utc, field="fresh_until_utc")
        if fresh_until < observed:
            raise ProgrammeContractError("status freshness precedes observation")
        canonical_sha256({"value": self.value})

    def to_dict(self, *, as_of_utc: str) -> dict[str, object]:
        freshness = (
            "FRESH"
            if _utc(as_of_utc, field="as_of_utc")
            <= _utc(self.fresh_until_utc, field="fresh_until_utc")
            else "STALE"
        )
        return {
            "value": self.value,
            "owner_id": self.owner_id,
            "owner_digest": _digest(self.owner_digest),
            "record_id": self.record_id,
            "record_digest": _digest(self.record_digest),
            "observed_at_utc": self.observed_at_utc,
            "fresh_until_utc": self.fresh_until_utc,
            "freshness": freshness,
        }


def project_unified_status_v1(
    facts: Iterable[OwnedStatusFactV1], *, as_of_utc: str
) -> dict[str, object]:
    """Project owner facts without creating status authority or guessing."""

    _utc(as_of_utc, field="as_of_utc")
    grouped: dict[str, list[OwnedStatusFactV1]] = {}
    for fact in facts:
        if not isinstance(fact, OwnedStatusFactV1):
            raise ProgrammeContractError("status projection contains invalid fact")
        grouped.setdefault(fact.field, []).append(fact)
    fields: dict[str, object] = {}
    contradictions: list[str] = []
    for field, rows in sorted(grouped.items()):
        ordered = sorted(
            rows,
            key=lambda row: (
                row.observed_at_utc,
                row.owner_id,
                row.record_id,
                row.record_digest.hex,
            ),
        )
        rendered = [row.to_dict(as_of_utc=as_of_utc) for row in ordered]
        fresh_values = {
            canonical_sha256({"value": row.value}).hex
            for row, observation in zip(ordered, rendered, strict=True)
            if observation["freshness"] == "FRESH"
        }
        state = "INCONSISTENT" if len(fresh_values) > 1 else (
            "FRESH" if fresh_values else "STALE"
        )
        if state == "INCONSISTENT":
            contradictions.append(field)
        fields[field] = {"state": state, "observations": rendered}
    return {
        "schema_version": UNIFIED_STATUS_PROJECTION_SCHEMA_VERSION,
        "status": "INCONSISTENT" if contradictions else "CONSISTENT",
        "as_of_utc": as_of_utc,
        "fields": fields,
        "contradictions": contradictions,
        "authority": "PURE_PROJECTION_EXISTING_OWNERS_ONLY",
    }


__all__ = [
    "AttemptLeaseBindingsV1",
    "ContentDigest",
    "GpuResourceTopologyError",
    "NotificationDeduplicationProofV1",
    "OwnedStatusFactV1",
    "ProgrammeActionClaimV1",
    "ProgrammeActionCodeV1",
    "ProgrammeActionIntentV1",
    "ProgrammeActionPlanV1",
    "ProgrammeActionSettlementV1",
    "ProgrammeAuthorizationV1",
    "ProgrammeContractError",
    "ProgrammeDriverLeaseV1",
    "ProgrammeGateV1",
    "ProgrammeTerminalStateV1",
    "ProgrammeTransitionReceiptV1",
    "LaunchpadAttemptClaimV1",
    "LaunchpadResourceStartLeaseV1",
    "ResourceStartLeaseEventCodeV1",
    "ResourceStartLeaseEventV1",
    "ResourceStartLeaseEventV2",
    "ResourceStartLeaseEventV3",
    "ResourceStartLeaseV1",
    "TerminalNotificationDeliveryV1",
    "TerminalNotificationIntentV1",
    "canonical_sha256",
    "gpu_uuid_equal_v1",
    "project_unified_status_v1",
    "resource_ids_conflict_v1",
    "require_exclusive_resource_start_lease_v1",
    "shared_gpu_lane_candidate_sets_v1",
    "shared_gpu_lane_resource_id_v1",
    "validate_shared_w32_gpu_policy_binding_v1",
]
