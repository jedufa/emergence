"""Typed qualification-only controls and observations for resident execution.

The contracts in this module bind already selected resident cells to narrowly
scoped fault and fixed-action controls.  They do not dispatch work, mutate
runtime state, or authorize production execution.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final, Self

from engine.env.gridworld_pkg.constants import ACT_REST

from .canonical import canonical_sha256


FAULT_CONTROL_SCHEMA_VERSION: Final[str] = (
    "eiel_resident_capacity_fault_control_v1"
)
FAULT_OBSERVATION_SCHEMA_VERSION: Final[str] = (
    "eiel_resident_capacity_fault_observation_v1"
)
FIXED_ACTION_CONTROL_SCHEMA_VERSION: Final[str] = (
    "eiel_resident_fixed_action_control_v1"
)
FIXED_ACTION_OBSERVATION_SCHEMA_VERSION: Final[str] = (
    "eiel_resident_fixed_action_observation_v2"
)
AUTHORITY_SCOPE: Final[str] = "qualification_only"
CAPACITY_KINDS: Final[frozenset[str]] = frozenset(
    {"attempt", "snapshot", "eligible_parent", "aggregate"}
)
STATUS_OK: Final[int] = 0

_HEX: Final[frozenset[str]] = frozenset("0123456789abcdef")


class InheritanceCrossPathQualificationControlError(ValueError):
    """A qualification control or observation is malformed or inconsistent."""


def _fail(message: str) -> InheritanceCrossPathQualificationControlError:
    return InheritanceCrossPathQualificationControlError(message)


def _exact_object(
    value: Any,
    *,
    name: str,
    fields: frozenset[str],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise _fail(f"{name} must be an object")
    actual = frozenset(value)
    if actual != fields:
        raise _fail(
            f"{name} field set is invalid; "
            f"missing={sorted(fields - actual)}, "
            f"unexpected={sorted(actual - fields)}"
        )
    return dict(value)


def _schema(value: Any, *, expected: str, name: str) -> str:
    if value != expected:
        raise _fail(f"{name} schema_version is unsupported")
    return expected


def _scope(value: Any, *, name: str) -> str:
    if value != AUTHORITY_SCOPE:
        raise _fail(f"{name} authority_scope must be qualification_only")
    return AUTHORITY_SCOPE


def _text(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise _fail(f"{field} must be non-empty text")
    return value


def _digest(value: Any, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        raise _fail(f"{field} must be a lowercase SHA-256 digest")
    return value


def _integer(value: Any, *, field: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise _fail(f"{field} must be an integer >= {minimum}")
    return value


def _boolean(value: Any, *, field: str) -> bool:
    if type(value) is not bool:
        raise _fail(f"{field} must be boolean")
    return value


def _require_authority(
    *,
    supplied: Any,
    payload: Mapping[str, Any],
    field: str,
) -> str:
    digest = _digest(supplied, field=field)
    if canonical_sha256(payload) != digest:
        raise _fail(f"{field} does not bind the canonical payload")
    return digest


@dataclass(frozen=True, slots=True)
class ResidentCapacityFaultControlV1:
    """Pre-dispatch authority for one qualification-only capacity fault."""

    target_occurrence_id: str
    target_occurrence_authority_sha256: str
    target_cell_index: int
    target_cell_authority_sha256: str
    capacity_kind: str
    nominal_bound: int
    effective_bound: int
    expected_status_code: int
    pre_fault_authority_sha256: str

    def __post_init__(self) -> None:
        _text(self.target_occurrence_id, field="target_occurrence_id")
        _digest(
            self.target_occurrence_authority_sha256,
            field="target_occurrence_authority_sha256",
        )
        _integer(self.target_cell_index, field="target_cell_index")
        _digest(
            self.target_cell_authority_sha256,
            field="target_cell_authority_sha256",
        )
        if self.capacity_kind not in CAPACITY_KINDS:
            raise _fail("capacity_kind is unsupported")
        _integer(self.nominal_bound, field="nominal_bound", minimum=1)
        _integer(self.effective_bound, field="effective_bound")
        if self.effective_bound >= self.nominal_bound:
            raise _fail("effective_bound must be smaller than nominal_bound")
        _integer(
            self.expected_status_code,
            field="expected_status_code",
            minimum=1,
        )
        _digest(
            self.pre_fault_authority_sha256,
            field="pre_fault_authority_sha256",
        )

    def authority_payload(self) -> dict[str, Any]:
        return {
            "schema_version": FAULT_CONTROL_SCHEMA_VERSION,
            "authority_scope": AUTHORITY_SCOPE,
            "target_occurrence_id": self.target_occurrence_id,
            "target_occurrence_authority_sha256": (
                self.target_occurrence_authority_sha256
            ),
            "target_cell_index": self.target_cell_index,
            "target_cell_authority_sha256": self.target_cell_authority_sha256,
            "capacity_kind": self.capacity_kind,
            "nominal_bound": self.nominal_bound,
            "effective_bound": self.effective_bound,
            "expected_status_code": self.expected_status_code,
            "pre_fault_authority_sha256": self.pre_fault_authority_sha256,
        }

    @property
    def control_authority_sha256(self) -> str:
        return canonical_sha256(self.authority_payload())

    def payload(self) -> dict[str, Any]:
        return {
            **self.authority_payload(),
            "control_authority_sha256": self.control_authority_sha256,
        }

    @classmethod
    def from_payload(cls, value: Any) -> Self:
        fields = frozenset(
            {
                "schema_version",
                "authority_scope",
                "target_occurrence_id",
                "target_occurrence_authority_sha256",
                "target_cell_index",
                "target_cell_authority_sha256",
                "capacity_kind",
                "nominal_bound",
                "effective_bound",
                "expected_status_code",
                "pre_fault_authority_sha256",
                "control_authority_sha256",
            }
        )
        row = _exact_object(value, name="fault control", fields=fields)
        _schema(
            row["schema_version"],
            expected=FAULT_CONTROL_SCHEMA_VERSION,
            name="fault control",
        )
        _scope(row["authority_scope"], name="fault control")
        result = cls(
            target_occurrence_id=row["target_occurrence_id"],
            target_occurrence_authority_sha256=(
                row["target_occurrence_authority_sha256"]
            ),
            target_cell_index=row["target_cell_index"],
            target_cell_authority_sha256=row["target_cell_authority_sha256"],
            capacity_kind=row["capacity_kind"],
            nominal_bound=row["nominal_bound"],
            effective_bound=row["effective_bound"],
            expected_status_code=row["expected_status_code"],
            pre_fault_authority_sha256=row["pre_fault_authority_sha256"],
        )
        _require_authority(
            supplied=row["control_authority_sha256"],
            payload=result.authority_payload(),
            field="control_authority_sha256",
        )
        return result


@dataclass(frozen=True, slots=True)
class ResidentCapacityFaultObservationV1:
    """Observed target-only sticky overflow and unchanged mechanics/peers."""

    control_authority_sha256: str
    target_occurrence_authority_sha256: str
    target_cell_index: int
    target_cell_authority_sha256: str
    pre_fault_authority_sha256: str
    status_before_code: int
    status_after_code: int
    status_after_drain_code: int
    mechanics_before_sha256: str
    mechanics_after_sha256: str
    peer_count: int
    peer_set_before_sha256: str
    peer_set_after_sha256: str

    def __post_init__(self) -> None:
        for field in (
            "control_authority_sha256",
            "target_occurrence_authority_sha256",
            "target_cell_authority_sha256",
            "pre_fault_authority_sha256",
            "mechanics_before_sha256",
            "mechanics_after_sha256",
            "peer_set_before_sha256",
            "peer_set_after_sha256",
        ):
            _digest(getattr(self, field), field=field)
        _integer(self.target_cell_index, field="target_cell_index")
        _integer(self.status_before_code, field="status_before_code")
        _integer(self.status_after_code, field="status_after_code")
        _integer(
            self.status_after_drain_code,
            field="status_after_drain_code",
        )
        _integer(self.peer_count, field="peer_count", minimum=1)
        if self.status_before_code != STATUS_OK:
            raise _fail("status_before_code must be the OK status")
        if self.status_after_code == STATUS_OK:
            raise _fail("status_after_code must be a fail-closed status")
        if self.status_after_drain_code != self.status_after_code:
            raise _fail("fault status must remain sticky after drain")
        if self.mechanics_before_sha256 != self.mechanics_after_sha256:
            raise _fail("target mechanics changed across the capacity fault")
        if self.peer_set_before_sha256 != self.peer_set_after_sha256:
            raise _fail("non-target peers changed across the capacity fault")

    def authority_payload(self) -> dict[str, Any]:
        return {
            "schema_version": FAULT_OBSERVATION_SCHEMA_VERSION,
            "authority_scope": AUTHORITY_SCOPE,
            "control_authority_sha256": self.control_authority_sha256,
            "target_occurrence_authority_sha256": (
                self.target_occurrence_authority_sha256
            ),
            "target_cell_index": self.target_cell_index,
            "target_cell_authority_sha256": self.target_cell_authority_sha256,
            "pre_fault_authority_sha256": self.pre_fault_authority_sha256,
            "status_before_code": self.status_before_code,
            "status_after_code": self.status_after_code,
            "status_after_drain_code": self.status_after_drain_code,
            "mechanics_before_sha256": self.mechanics_before_sha256,
            "mechanics_after_sha256": self.mechanics_after_sha256,
            "peer_count": self.peer_count,
            "peer_set_before_sha256": self.peer_set_before_sha256,
            "peer_set_after_sha256": self.peer_set_after_sha256,
        }

    @property
    def observation_authority_sha256(self) -> str:
        return canonical_sha256(self.authority_payload())

    def payload(self) -> dict[str, Any]:
        return {
            **self.authority_payload(),
            "observation_authority_sha256": (
                self.observation_authority_sha256
            ),
        }

    def validate_against(
        self,
        control: ResidentCapacityFaultControlV1,
    ) -> None:
        expected = (
            control.control_authority_sha256,
            control.target_occurrence_authority_sha256,
            control.target_cell_index,
            control.target_cell_authority_sha256,
            control.pre_fault_authority_sha256,
            control.expected_status_code,
        )
        observed = (
            self.control_authority_sha256,
            self.target_occurrence_authority_sha256,
            self.target_cell_index,
            self.target_cell_authority_sha256,
            self.pre_fault_authority_sha256,
            self.status_after_code,
        )
        if observed != expected:
            raise _fail("fault observation does not match its control")

    @classmethod
    def from_payload(cls, value: Any) -> Self:
        fields = frozenset(
            {
                "schema_version",
                "authority_scope",
                "control_authority_sha256",
                "target_occurrence_authority_sha256",
                "target_cell_index",
                "target_cell_authority_sha256",
                "pre_fault_authority_sha256",
                "status_before_code",
                "status_after_code",
                "status_after_drain_code",
                "mechanics_before_sha256",
                "mechanics_after_sha256",
                "peer_count",
                "peer_set_before_sha256",
                "peer_set_after_sha256",
                "observation_authority_sha256",
            }
        )
        row = _exact_object(value, name="fault observation", fields=fields)
        _schema(
            row["schema_version"],
            expected=FAULT_OBSERVATION_SCHEMA_VERSION,
            name="fault observation",
        )
        _scope(row["authority_scope"], name="fault observation")
        result = cls(
            control_authority_sha256=row["control_authority_sha256"],
            target_occurrence_authority_sha256=(
                row["target_occurrence_authority_sha256"]
            ),
            target_cell_index=row["target_cell_index"],
            target_cell_authority_sha256=row["target_cell_authority_sha256"],
            pre_fault_authority_sha256=row["pre_fault_authority_sha256"],
            status_before_code=row["status_before_code"],
            status_after_code=row["status_after_code"],
            status_after_drain_code=row["status_after_drain_code"],
            mechanics_before_sha256=row["mechanics_before_sha256"],
            mechanics_after_sha256=row["mechanics_after_sha256"],
            peer_count=row["peer_count"],
            peer_set_before_sha256=row["peer_set_before_sha256"],
            peer_set_after_sha256=row["peer_set_after_sha256"],
        )
        _require_authority(
            supplied=row["observation_authority_sha256"],
            payload=result.authority_payload(),
            field="observation_authority_sha256",
        )
        return result


@dataclass(frozen=True, slots=True)
class ResidentFixedActionControlV1:
    """Pre-dispatch fixed-rest and policy-bypass authority."""

    cell_mask_sha256: str
    selected_cell_count: int
    action_id: int = int(ACT_REST)
    policy_bypass_required: bool = True

    def __post_init__(self) -> None:
        _digest(self.cell_mask_sha256, field="cell_mask_sha256")
        _integer(
            self.selected_cell_count,
            field="selected_cell_count",
            minimum=1,
        )
        _integer(self.action_id, field="action_id")
        if self.action_id != int(ACT_REST):
            raise _fail("action_id must be ACT_REST")
        _boolean(
            self.policy_bypass_required,
            field="policy_bypass_required",
        )
        if not self.policy_bypass_required:
            raise _fail("policy_bypass_required must be true")

    def authority_payload(self) -> dict[str, Any]:
        return {
            "schema_version": FIXED_ACTION_CONTROL_SCHEMA_VERSION,
            "authority_scope": AUTHORITY_SCOPE,
            "cell_mask_sha256": self.cell_mask_sha256,
            "selected_cell_count": self.selected_cell_count,
            "action_id": self.action_id,
            "policy_bypass_required": self.policy_bypass_required,
        }

    @property
    def control_authority_sha256(self) -> str:
        return canonical_sha256(self.authority_payload())

    def payload(self) -> dict[str, Any]:
        return {
            **self.authority_payload(),
            "control_authority_sha256": self.control_authority_sha256,
        }

    @classmethod
    def from_payload(cls, value: Any) -> Self:
        fields = frozenset(
            {
                "schema_version",
                "authority_scope",
                "cell_mask_sha256",
                "selected_cell_count",
                "action_id",
                "policy_bypass_required",
                "control_authority_sha256",
            }
        )
        row = _exact_object(value, name="fixed action control", fields=fields)
        _schema(
            row["schema_version"],
            expected=FIXED_ACTION_CONTROL_SCHEMA_VERSION,
            name="fixed action control",
        )
        _scope(row["authority_scope"], name="fixed action control")
        result = cls(
            cell_mask_sha256=row["cell_mask_sha256"],
            selected_cell_count=row["selected_cell_count"],
            action_id=row["action_id"],
            policy_bypass_required=row["policy_bypass_required"],
        )
        _require_authority(
            supplied=row["control_authority_sha256"],
            payload=result.authority_payload(),
            field="control_authority_sha256",
        )
        return result


@dataclass(frozen=True, slots=True)
class ResidentFixedActionObservationV1:
    """Observed fixed-rest output with no policy evaluation or semantic draw."""

    control_authority_sha256: str
    cell_mask_sha256: str
    action_id: int
    observed_action_count: int
    all_rest_actions_sha256: str
    policy_evaluation_count_before: int
    policy_evaluation_count_after: int
    policy_evaluation_count_delta: int
    semantic_draw_count_before: int
    semantic_draw_count_after: int
    semantic_draw_count_delta: int

    def __post_init__(self) -> None:
        for field in (
            "control_authority_sha256",
            "cell_mask_sha256",
            "all_rest_actions_sha256",
        ):
            _digest(getattr(self, field), field=field)
        _integer(self.action_id, field="action_id")
        if self.action_id != int(ACT_REST):
            raise _fail("action_id must be ACT_REST")
        _integer(
            self.observed_action_count,
            field="observed_action_count",
            minimum=1,
        )
        expected_actions_sha256 = canonical_sha256(
            [int(ACT_REST)] * self.observed_action_count
        )
        if self.all_rest_actions_sha256 != expected_actions_sha256:
            raise _fail(
                "all_rest_actions_sha256 does not bind the declared "
                "all-REST action vector"
            )
        for field in (
            "policy_evaluation_count_before",
            "policy_evaluation_count_after",
            "policy_evaluation_count_delta",
            "semantic_draw_count_before",
            "semantic_draw_count_after",
            "semantic_draw_count_delta",
        ):
            _integer(getattr(self, field), field=field)
        if (
            self.policy_evaluation_count_after
            - self.policy_evaluation_count_before
            != self.policy_evaluation_count_delta
            or self.policy_evaluation_count_delta != 0
        ):
            raise _fail("observed policy evaluation delta must be zero")
        if (
            self.semantic_draw_count_after
            - self.semantic_draw_count_before
            != self.semantic_draw_count_delta
            or self.semantic_draw_count_delta != 0
        ):
            raise _fail("observed semantic draw delta must be zero")
    def authority_payload(self) -> dict[str, Any]:
        return {
            "schema_version": FIXED_ACTION_OBSERVATION_SCHEMA_VERSION,
            "authority_scope": AUTHORITY_SCOPE,
            "control_authority_sha256": self.control_authority_sha256,
            "cell_mask_sha256": self.cell_mask_sha256,
            "action_id": self.action_id,
            "observed_action_count": self.observed_action_count,
            "all_rest_actions_sha256": self.all_rest_actions_sha256,
            "policy_evaluation_count_before": (
                self.policy_evaluation_count_before
            ),
            "policy_evaluation_count_after": (
                self.policy_evaluation_count_after
            ),
            "policy_evaluation_count_delta": (
                self.policy_evaluation_count_delta
            ),
            "semantic_draw_count_before": self.semantic_draw_count_before,
            "semantic_draw_count_after": self.semantic_draw_count_after,
            "semantic_draw_count_delta": self.semantic_draw_count_delta,
        }

    @property
    def observation_authority_sha256(self) -> str:
        return canonical_sha256(self.authority_payload())

    def payload(self) -> dict[str, Any]:
        return {
            **self.authority_payload(),
            "observation_authority_sha256": (
                self.observation_authority_sha256
            ),
        }

    def validate_against(self, control: ResidentFixedActionControlV1) -> None:
        if (
            self.control_authority_sha256
            != control.control_authority_sha256
            or self.cell_mask_sha256 != control.cell_mask_sha256
            or self.action_id != control.action_id
        ):
            raise _fail("fixed action observation does not match its control")

    @classmethod
    def from_payload(cls, value: Any) -> Self:
        fields = frozenset(
            {
                "schema_version",
                "authority_scope",
                "control_authority_sha256",
                "cell_mask_sha256",
                "action_id",
                "observed_action_count",
                "all_rest_actions_sha256",
                "policy_evaluation_count_before",
                "policy_evaluation_count_after",
                "policy_evaluation_count_delta",
                "semantic_draw_count_before",
                "semantic_draw_count_after",
                "semantic_draw_count_delta",
                "observation_authority_sha256",
            }
        )
        row = _exact_object(
            value,
            name="fixed action observation",
            fields=fields,
        )
        _schema(
            row["schema_version"],
            expected=FIXED_ACTION_OBSERVATION_SCHEMA_VERSION,
            name="fixed action observation",
        )
        _scope(row["authority_scope"], name="fixed action observation")
        result = cls(
            control_authority_sha256=row["control_authority_sha256"],
            cell_mask_sha256=row["cell_mask_sha256"],
            action_id=row["action_id"],
            observed_action_count=row["observed_action_count"],
            all_rest_actions_sha256=row["all_rest_actions_sha256"],
            policy_evaluation_count_before=(
                row["policy_evaluation_count_before"]
            ),
            policy_evaluation_count_after=(
                row["policy_evaluation_count_after"]
            ),
            policy_evaluation_count_delta=(
                row["policy_evaluation_count_delta"]
            ),
            semantic_draw_count_before=row["semantic_draw_count_before"],
            semantic_draw_count_after=row["semantic_draw_count_after"],
            semantic_draw_count_delta=row["semantic_draw_count_delta"],
        )
        _require_authority(
            supplied=row["observation_authority_sha256"],
            payload=result.authority_payload(),
            field="observation_authority_sha256",
        )
        return result


__all__ = [
    "AUTHORITY_SCOPE",
    "CAPACITY_KINDS",
    "FAULT_CONTROL_SCHEMA_VERSION",
    "FAULT_OBSERVATION_SCHEMA_VERSION",
    "FIXED_ACTION_CONTROL_SCHEMA_VERSION",
    "FIXED_ACTION_OBSERVATION_SCHEMA_VERSION",
    "InheritanceCrossPathQualificationControlError",
    "ResidentCapacityFaultControlV1",
    "ResidentCapacityFaultObservationV1",
    "ResidentFixedActionControlV1",
    "ResidentFixedActionObservationV1",
    "STATUS_OK",
]
