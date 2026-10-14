"""Typed evidence contract for heterogeneous resident-cell isolation.

The verifier consumes observations captured by an execution owner.  It does
not execute simulations, reconstruct state, or bless a route.  In particular,
an isolation result can only be green when every required comparison is
present and exact.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from typing import Any, Mapping, Sequence


GATE_SCHEMA_VERSION = "eiel_mixed_mode_resident_isolation_gate_v2"
PLAN_SCHEMA_VERSION = "eiel_mixed_mode_resident_isolation_plan_v2"
RUN_SCHEMA_VERSION = "eiel_mixed_mode_resident_isolation_run_v2"
CELL_SCHEMA_VERSION = "eiel_mixed_mode_resident_cell_capture_v2"
PATH_RECEIPT_SCHEMA_VERSION = "eiel_resident_reproduction_path_receipt_v2"
SELECTION_SCHEMA_VERSION = "eiel_resident_parent_selection_contract_v1"
EXPOSURE_SCHEMA_VERSION = "eiel_inheritance_event_exposure_v2"
MINIMUM_EXPOSURE_SCHEMA_VERSION = "eiel_frozen_minimum_inheritance_exposure_v1"

RUN_KIND_ALONE = "cell_alone_v1"
RUN_KIND_MIXED = "mixed_reference_v1"
RUN_KIND_PERMUTED = "mixed_permuted_v1"
RUN_KIND_DUPLICATE = "mixed_duplicate_v1"
RUN_KIND_OVERFLOW = "mixed_overflow_fault_v1"
RUN_KIND_CHECKPOINT_BASELINE = "checkpoint_uninterrupted_v1"
RUN_KIND_CHECKPOINT_FRESH_OWNER = "checkpoint_fresh_owner_v1"

_RUN_KINDS = frozenset(
    {
        RUN_KIND_ALONE,
        RUN_KIND_MIXED,
        RUN_KIND_PERMUTED,
        RUN_KIND_DUPLICATE,
        RUN_KIND_OVERFLOW,
        RUN_KIND_CHECKPOINT_BASELINE,
        RUN_KIND_CHECKPOINT_FRESH_OWNER,
    }
)
_BLINDED_DIMENSIONS = frozenset({"genotype", "lineage", "policy", "strategy"})
_EXPOSURE_METRICS = frozenset({"attempts", "successful_births", "parent_contributions"})
_HEX = frozenset("0123456789abcdef")


class MixedModeIsolationContractError(ValueError):
    """A supplied plan or capture is not a valid v1 contract."""


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
        raise MixedModeIsolationContractError(
            "contract payload must be canonical JSON data"
        ) from exc


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _text(name: str, value: str) -> None:
    if not isinstance(value, str) or not value:
        raise MixedModeIsolationContractError(f"{name} must be non-empty text")


def _nonnegative(name: str, value: int) -> None:
    if type(value) is not int or value < 0:
        raise MixedModeIsolationContractError(f"{name} must be a non-negative integer")


def _digest(name: str, value: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        raise MixedModeIsolationContractError(
            f"{name} must be a lowercase SHA-256 digest"
        )


def _utc(name: str, value: str) -> datetime:
    _text(name, value)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise MixedModeIsolationContractError(
            f"{name} must be an ISO-8601 timestamp"
        ) from exc
    if parsed.tzinfo is None:
        raise MixedModeIsolationContractError(f"{name} must include a timezone")
    return parsed.astimezone(timezone.utc)


def _unique_texts(name: str, values: Sequence[str], *, minimum: int = 0) -> None:
    if isinstance(values, (str, bytes)) or len(values) < minimum:
        raise MixedModeIsolationContractError(
            f"{name} must contain at least {minimum} entries"
        )
    for value in values:
        _text(name, value)
    if len(set(values)) != len(values):
        raise MixedModeIsolationContractError(f"{name} must be unique")


def _counter_items(
    name: str, values: Sequence[tuple[str, int]]
) -> tuple[tuple[str, int], ...]:
    normalized = tuple(values)
    keys: list[str] = []
    for key, value in normalized:
        _text(f"{name}.key", key)
        _nonnegative(f"{name}.{key}", value)
        keys.append(key)
    if len(set(keys)) != len(keys):
        raise MixedModeIsolationContractError(f"{name} keys must be unique")
    return tuple(sorted(normalized))


@dataclass(frozen=True, slots=True)
class ParentSelectionContractV1:
    """Per-cell selection semantics bound before execution."""

    selection_mode: str
    eligible_parent_source: str
    draw_distribution: str
    deterministic: bool
    blinded_dimensions: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in (
            "selection_mode",
            "eligible_parent_source",
            "draw_distribution",
        ):
            _text(name, getattr(self, name))
        if type(self.deterministic) is not bool:
            raise MixedModeIsolationContractError("deterministic must be boolean")
        _unique_texts("blinded_dimensions", self.blinded_dimensions)
        object.__setattr__(
            self, "blinded_dimensions", tuple(sorted(self.blinded_dimensions))
        )

    def payload(self) -> dict[str, Any]:
        return {
            "schema_version": SELECTION_SCHEMA_VERSION,
            "selection_mode": self.selection_mode,
            "eligible_parent_source": self.eligible_parent_source,
            "draw_distribution": self.draw_distribution,
            "deterministic": self.deterministic,
            "blinded_dimensions": list(self.blinded_dimensions),
        }


@dataclass(frozen=True, slots=True)
class ResidentReproductionCellConfigV1:
    """Typed heterogeneous reproduction fields for one resident cell."""

    logical_cell_id: str
    equivalence_key: str
    reproduction_path: str
    selection: ParentSelectionContractV1
    parent_arity: int
    donor_draws_per_successful_birth: int
    authorized_rng_counter_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in ("logical_cell_id", "equivalence_key", "reproduction_path"):
            _text(name, getattr(self, name))
        if type(self.selection) is not ParentSelectionContractV1:
            raise MixedModeIsolationContractError(
                "selection must be ParentSelectionContractV1"
            )
        if type(self.parent_arity) is not int or self.parent_arity < 1:
            raise MixedModeIsolationContractError("parent_arity must be positive")
        _nonnegative(
            "donor_draws_per_successful_birth",
            self.donor_draws_per_successful_birth,
        )
        _unique_texts("authorized_rng_counter_ids", self.authorized_rng_counter_ids)
        object.__setattr__(
            self,
            "authorized_rng_counter_ids",
            tuple(sorted(self.authorized_rng_counter_ids)),
        )

    def payload(self) -> dict[str, Any]:
        return {
            "schema_version": "eiel_resident_reproduction_cell_config_v1",
            "logical_cell_id": self.logical_cell_id,
            "equivalence_key": self.equivalence_key,
            "reproduction_path": self.reproduction_path,
            "selection": self.selection.payload(),
            "parent_arity": self.parent_arity,
            "donor_draws_per_successful_birth": (self.donor_draws_per_successful_birth),
            "authorized_rng_counter_ids": list(self.authorized_rng_counter_ids),
        }


@dataclass(frozen=True, slots=True)
class InheritanceAttemptV1:
    """Complete parent exposure for one attempted birth."""

    primary_eligible_parent_ids: tuple[str, ...]
    mate_eligible_parent_ids: tuple[str, ...]
    selected_parent_ids: tuple[str, ...]
    success: bool
    contributing_parent_ids: tuple[str, ...] = ()
    failure_code: str | None = None

    def __post_init__(self) -> None:
        _unique_texts(
            "primary_eligible_parent_ids",
            self.primary_eligible_parent_ids,
            minimum=1,
        )
        _unique_texts(
            "mate_eligible_parent_ids",
            self.mate_eligible_parent_ids,
            minimum=1,
        )
        if isinstance(self.selected_parent_ids, (str, bytes)):
            raise MixedModeIsolationContractError(
                "selected_parent_ids must be a sequence"
            )
        for value in self.selected_parent_ids:
            _text("selected_parent_ids", value)
        if len(self.selected_parent_ids) != 2:
            raise MixedModeIsolationContractError(
                "selected_parent_ids must contain primary and mate"
            )
        if type(self.success) is not bool:
            raise MixedModeIsolationContractError("success must be boolean")
        if isinstance(self.contributing_parent_ids, (str, bytes)):
            raise MixedModeIsolationContractError(
                "contributing_parent_ids must be a sequence"
            )
        for value in self.contributing_parent_ids:
            _text("contributing_parent_ids", value)
        if self.failure_code is not None:
            _text("failure_code", self.failure_code)
        if self.success and self.failure_code is not None:
            raise MixedModeIsolationContractError(
                "successful attempts cannot have failure_code"
            )
        if not self.success and self.contributing_parent_ids:
            raise MixedModeIsolationContractError(
                "failed attempts cannot expose parent contributions"
            )
        if not self.success and self.failure_code is None:
            raise MixedModeIsolationContractError(
                "failed attempts require failure_code"
            )
        if self.selected_parent_ids[0] not in self.primary_eligible_parent_ids:
            raise MixedModeIsolationContractError(
                "selected primary must belong to the primary-eligible set"
            )
        if self.selected_parent_ids[1] not in self.mate_eligible_parent_ids:
            raise MixedModeIsolationContractError(
                "selected mate must belong to the mate-eligible set"
            )
        if any(
            parent not in self.selected_parent_ids
            for parent in self.contributing_parent_ids
        ):
            raise MixedModeIsolationContractError(
                "contributing parents must have been selected"
            )

    def payload(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "primary_eligible_parent_ids": list(
                self.primary_eligible_parent_ids
            ),
            "mate_eligible_parent_ids": list(self.mate_eligible_parent_ids),
            "selected_parent_ids": list(self.selected_parent_ids),
            "success": self.success,
            "contributing_parent_ids": list(self.contributing_parent_ids),
        }
        if self.failure_code is not None:
            value["failure_code"] = self.failure_code
        return value


@dataclass(frozen=True, slots=True)
class ReproductionPathReceiptV1:
    """Cell-local counters, writes, draws, and inheritance events for one path."""

    path_id: str
    enabled: bool
    rng_counters: tuple[tuple[str, int], ...]
    donor_draw_ordinals: tuple[int, ...]
    counter_writes: int
    sidecar_writes: int
    event_row_writes: int
    rng_state_writes: int
    attempts: tuple[InheritanceAttemptV1, ...]

    def __post_init__(self) -> None:
        _text("path_id", self.path_id)
        if type(self.enabled) is not bool:
            raise MixedModeIsolationContractError("enabled must be boolean")
        object.__setattr__(
            self,
            "rng_counters",
            _counter_items("rng_counters", self.rng_counters),
        )
        for ordinal in self.donor_draw_ordinals:
            _nonnegative("donor_draw_ordinals", ordinal)
        for name in (
            "counter_writes",
            "sidecar_writes",
            "event_row_writes",
            "rng_state_writes",
        ):
            _nonnegative(name, getattr(self, name))
        if isinstance(self.attempts, (str, bytes)) or any(
            type(value) is not InheritanceAttemptV1 for value in self.attempts
        ):
            raise MixedModeIsolationContractError(
                "attempts must contain InheritanceAttemptV1 values"
            )

    @property
    def successful_births(self) -> int:
        return sum(attempt.success for attempt in self.attempts)

    @property
    def failed_attempts(self) -> int:
        return len(self.attempts) - self.successful_births

    def exposure_payload(self) -> dict[str, Any]:
        contributions: dict[str, int] = {}
        for attempt in self.attempts:
            for parent_id in attempt.contributing_parent_ids:
                contributions[parent_id] = contributions.get(parent_id, 0) + 1
        return {
            "schema_version": EXPOSURE_SCHEMA_VERSION,
            "primary_eligible_parents": [
                list(attempt.primary_eligible_parent_ids)
                for attempt in self.attempts
            ],
            "mate_eligible_parents": [
                list(attempt.mate_eligible_parent_ids)
                for attempt in self.attempts
            ],
            "selected_parents": [
                list(attempt.selected_parent_ids) for attempt in self.attempts
            ],
            "attempts": len(self.attempts),
            "successful_births": self.successful_births,
            "failed_attempts": self.failed_attempts,
            "parent_contribution_counts": dict(sorted(contributions.items())),
        }

    def payload(self) -> dict[str, Any]:
        return {
            "schema_version": PATH_RECEIPT_SCHEMA_VERSION,
            "path_id": self.path_id,
            "enabled": self.enabled,
            "rng_counters": [
                {"counter_id": key, "value": value} for key, value in self.rng_counters
            ],
            "donor_draw_ordinals": list(self.donor_draw_ordinals),
            "writes": {
                "counter": self.counter_writes,
                "sidecar": self.sidecar_writes,
                "event_row": self.event_row_writes,
                "rng_state": self.rng_state_writes,
            },
            "inheritance_event_exposure": self.exposure_payload(),
            "event_rows": [attempt.payload() for attempt in self.attempts],
        }


@dataclass(frozen=True, slots=True)
class ResidentCellCaptureV1:
    """One cell observation inside one execution."""

    occurrence_id: str
    config: ResidentReproductionCellConfigV1
    batch_position: int
    neighbor_path_ids: tuple[str, ...]
    final_state_sha256: str
    pre_fault_state_sha256: str
    checkpoint_sidecar_sha256: str
    lifecycle_state_sha256: str
    final_authority_sha256: str
    pre_fault_authority_sha256: str | None
    ok: bool
    fail_closed: bool
    error_code: str | None
    path_receipts: tuple[ReproductionPathReceiptV1, ...]

    def __post_init__(self) -> None:
        _text("occurrence_id", self.occurrence_id)
        if type(self.config) is not ResidentReproductionCellConfigV1:
            raise MixedModeIsolationContractError(
                "config must be ResidentReproductionCellConfigV1"
            )
        _nonnegative("batch_position", self.batch_position)
        for value in self.neighbor_path_ids:
            _text("neighbor_path_ids", value)
        for name in (
            "final_state_sha256",
            "pre_fault_state_sha256",
            "checkpoint_sidecar_sha256",
            "lifecycle_state_sha256",
            "final_authority_sha256",
        ):
            _digest(name, getattr(self, name))
        if self.pre_fault_authority_sha256 is not None:
            _digest(
                "pre_fault_authority_sha256",
                self.pre_fault_authority_sha256,
            )
        if type(self.ok) is not bool or type(self.fail_closed) is not bool:
            raise MixedModeIsolationContractError("ok and fail_closed must be boolean")
        if self.error_code is not None:
            _text("error_code", self.error_code)
        if self.ok and (self.fail_closed or self.error_code is not None):
            raise MixedModeIsolationContractError(
                "successful cells cannot carry failure state"
            )
        if not self.ok and self.error_code is None:
            raise MixedModeIsolationContractError("failed cells require an error_code")
        path_ids = [receipt.path_id for receipt in self.path_receipts]
        if len(set(path_ids)) != len(path_ids):
            raise MixedModeIsolationContractError(
                "path_receipts must have unique path ids"
            )

    def receipt(self, path_id: str) -> ReproductionPathReceiptV1 | None:
        return next(
            (receipt for receipt in self.path_receipts if receipt.path_id == path_id),
            None,
        )

    def exact_payload(self) -> dict[str, Any]:
        """Topology-independent durable mechanics comparison surface.

        ``final_state_sha256`` is an observer snapshot that intentionally
        includes transient resident scratch and topology-dependent control
        planes.  ``final_authority_sha256`` binds that snapshot and therefore
        is also unsuitable for cell-isolation equivalence.  Durable trajectory
        and canonical checkpoint storage are compared here instead; the full
        observer identities remain published in :meth:`payload`.
        """

        return {
            "config": self.config.payload(),
            "checkpoint_sidecar_sha256": self.checkpoint_sidecar_sha256,
            "lifecycle_state_sha256": self.lifecycle_state_sha256,
            "ok": self.ok,
            "fail_closed": self.fail_closed,
            "error_code": self.error_code,
            "path_receipts": [
                receipt.payload()
                for receipt in sorted(self.path_receipts, key=lambda item: item.path_id)
            ],
        }

    def payload(self) -> dict[str, Any]:
        return {
            "schema_version": CELL_SCHEMA_VERSION,
            "occurrence_id": self.occurrence_id,
            "config": self.config.payload(),
            "batch_position": self.batch_position,
            "neighbor_path_ids": list(self.neighbor_path_ids),
            "final_state_sha256": self.final_state_sha256,
            "pre_fault_state_sha256": self.pre_fault_state_sha256,
            "checkpoint_sidecar_sha256": self.checkpoint_sidecar_sha256,
            "lifecycle_state_sha256": self.lifecycle_state_sha256,
            "final_authority_sha256": self.final_authority_sha256,
            "pre_fault_authority_sha256": self.pre_fault_authority_sha256,
            "ok": self.ok,
            "fail_closed": self.fail_closed,
            "error_code": self.error_code,
            "path_receipts": [
                receipt.payload()
                for receipt in sorted(self.path_receipts, key=lambda item: item.path_id)
            ],
        }


@dataclass(frozen=True, slots=True)
class IsolationRunCaptureV1:
    run_id: str
    run_kind: str
    plan_sha256: str
    captured_at_utc: str
    fresh_owner: bool
    cells: tuple[ResidentCellCaptureV1, ...]

    def __post_init__(self) -> None:
        _text("run_id", self.run_id)
        if self.run_kind not in _RUN_KINDS:
            raise MixedModeIsolationContractError("run_kind is not supported")
        _digest("plan_sha256", self.plan_sha256)
        _utc("captured_at_utc", self.captured_at_utc)
        if type(self.fresh_owner) is not bool:
            raise MixedModeIsolationContractError("fresh_owner must be boolean")
        if not self.cells:
            raise MixedModeIsolationContractError("runs must contain cells")
        occurrence_ids = [cell.occurrence_id for cell in self.cells]
        positions = [cell.batch_position for cell in self.cells]
        if len(set(occurrence_ids)) != len(occurrence_ids):
            raise MixedModeIsolationContractError(
                "occurrence ids must be unique within a run"
            )
        if len(set(positions)) != len(positions):
            raise MixedModeIsolationContractError(
                "batch positions must be unique within a run"
            )

    def payload(self) -> dict[str, Any]:
        return {
            "schema_version": RUN_SCHEMA_VERSION,
            "run_id": self.run_id,
            "run_kind": self.run_kind,
            "plan_sha256": self.plan_sha256,
            "captured_at_utc": self.captured_at_utc,
            "fresh_owner": self.fresh_owner,
            "cells": [
                cell.payload()
                for cell in sorted(self.cells, key=lambda item: item.batch_position)
            ],
        }


@dataclass(frozen=True, slots=True)
class FrozenMinimumInheritanceExposureV1:
    threshold_id: str
    metric: str
    minimum: int
    frozen_at_utc: str

    def __post_init__(self) -> None:
        _text("threshold_id", self.threshold_id)
        if self.metric not in _EXPOSURE_METRICS:
            raise MixedModeIsolationContractError(
                "v1 minimum exposure metric is not supported"
            )
        if type(self.minimum) is not int or self.minimum < 1:
            raise MixedModeIsolationContractError("minimum exposure must be positive")
        _utc("frozen_at_utc", self.frozen_at_utc)

    def payload(self) -> dict[str, Any]:
        return {
            "schema_version": MINIMUM_EXPOSURE_SCHEMA_VERSION,
            "threshold_id": self.threshold_id,
            "metric": self.metric,
            "minimum": self.minimum,
            "frozen_at_utc": self.frozen_at_utc,
        }


@dataclass(frozen=True, slots=True)
class OverflowCaseV1:
    reference_run_id: str
    fault_run_id: str
    target_occurrence_id: str
    overflow_error_code: str

    def __post_init__(self) -> None:
        for name in (
            "reference_run_id",
            "fault_run_id",
            "target_occurrence_id",
            "overflow_error_code",
        ):
            _text(name, getattr(self, name))

    def payload(self) -> dict[str, Any]:
        return {
            "reference_run_id": self.reference_run_id,
            "fault_run_id": self.fault_run_id,
            "target_occurrence_id": self.target_occurrence_id,
            "overflow_error_code": self.overflow_error_code,
        }


@dataclass(frozen=True, slots=True)
class CheckpointCaseV1:
    uninterrupted_run_id: str
    fresh_owner_run_id: str

    def __post_init__(self) -> None:
        _text("uninterrupted_run_id", self.uninterrupted_run_id)
        _text("fresh_owner_run_id", self.fresh_owner_run_id)

    def payload(self) -> dict[str, Any]:
        return {
            "uninterrupted_run_id": self.uninterrupted_run_id,
            "fresh_owner_run_id": self.fresh_owner_run_id,
        }


@dataclass(frozen=True, slots=True)
class MixedModeIsolationPlanV1:
    """Pre-registered comparisons and threshold for one bounded gate."""

    path_ids: tuple[str, ...]
    mixed_reference_run_id: str
    permutation_run_ids: tuple[str, ...]
    duplicate_run_ids: tuple[str, ...]
    duplicate_required_path_ids: tuple[str, ...]
    overflow_cases: tuple[OverflowCaseV1, ...]
    checkpoint_cases: tuple[CheckpointCaseV1, ...]
    minimum_live_birth_exposure: FrozenMinimumInheritanceExposureV1
    live_birth_path_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _unique_texts("path_ids", self.path_ids, minimum=2)
        _text("mixed_reference_run_id", self.mixed_reference_run_id)
        _unique_texts("permutation_run_ids", self.permutation_run_ids, minimum=1)
        _unique_texts("duplicate_run_ids", self.duplicate_run_ids, minimum=1)
        _unique_texts(
            "duplicate_required_path_ids",
            self.duplicate_required_path_ids,
            minimum=1,
        )
        _unique_texts("live_birth_path_ids", self.live_birth_path_ids, minimum=1)
        if not set(self.duplicate_required_path_ids).issubset(self.path_ids):
            raise MixedModeIsolationContractError(
                "duplicate-required paths must be declared path ids"
            )
        if not set(self.live_birth_path_ids).issubset(self.path_ids):
            raise MixedModeIsolationContractError(
                "live-birth paths must be declared path ids"
            )
        if not self.overflow_cases or not all(
            type(value) is OverflowCaseV1 for value in self.overflow_cases
        ):
            raise MixedModeIsolationContractError(
                "overflow_cases must contain OverflowCaseV1 values"
            )
        if not self.checkpoint_cases or not all(
            type(value) is CheckpointCaseV1 for value in self.checkpoint_cases
        ):
            raise MixedModeIsolationContractError(
                "checkpoint_cases must contain CheckpointCaseV1 values"
            )
        if (
            type(self.minimum_live_birth_exposure)
            is not FrozenMinimumInheritanceExposureV1
        ):
            raise MixedModeIsolationContractError(
                "minimum exposure must be a frozen v1 contract"
            )

    def payload(self) -> dict[str, Any]:
        return {
            "schema_version": PLAN_SCHEMA_VERSION,
            "path_ids": list(self.path_ids),
            "mixed_reference_run_id": self.mixed_reference_run_id,
            "permutation_run_ids": list(self.permutation_run_ids),
            "duplicate_run_ids": list(self.duplicate_run_ids),
            "duplicate_required_path_ids": list(self.duplicate_required_path_ids),
            "overflow_cases": [case.payload() for case in self.overflow_cases],
            "checkpoint_cases": [case.payload() for case in self.checkpoint_cases],
            "minimum_live_birth_exposure": (self.minimum_live_birth_exposure.payload()),
            "live_birth_path_ids": list(self.live_birth_path_ids),
        }

    @property
    def digest_sha256(self) -> str:
        return _sha256(self.payload())


@dataclass(frozen=True, slots=True)
class IsolationDiagnosticV1:
    code: str
    message: str
    context: Mapping[str, Any]

    def payload(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "context": json.loads(_canonical_bytes(self.context).decode("ascii")),
        }


@dataclass(frozen=True, slots=True)
class IsolationCheckV1:
    check_id: str
    ok: bool
    diagnostics: tuple[IsolationDiagnosticV1, ...]

    def payload(self) -> dict[str, Any]:
        return {
            "check_id": self.check_id,
            "ok": self.ok,
            "diagnostics": [diagnostic.payload() for diagnostic in self.diagnostics],
        }


@dataclass(frozen=True, slots=True)
class MixedModeIsolationGateResultV1:
    plan_sha256: str
    ok: bool
    checks: tuple[IsolationCheckV1, ...]
    live_birth_evaluability: Mapping[str, Mapping[str, Any]]

    def payload(self) -> dict[str, Any]:
        return {
            "schema_version": GATE_SCHEMA_VERSION,
            "plan_sha256": self.plan_sha256,
            "ok": self.ok,
            "checks": [check.payload() for check in self.checks],
            "live_birth_evaluability": json.loads(
                _canonical_bytes(self.live_birth_evaluability).decode("ascii")
            ),
        }


def _diagnostic(code: str, message: str, **context: Any) -> IsolationDiagnosticV1:
    return IsolationDiagnosticV1(code=code, message=message, context=context)


def _by_occurrence(
    run: IsolationRunCaptureV1,
) -> dict[str, ResidentCellCaptureV1]:
    return {cell.occurrence_id: cell for cell in run.cells}


def _by_logical(
    run: IsolationRunCaptureV1,
) -> dict[str, ResidentCellCaptureV1]:
    return {cell.config.logical_cell_id: cell for cell in run.cells}


def _derived_neighbor_paths(
    run: IsolationRunCaptureV1,
    cell: ResidentCellCaptureV1,
) -> tuple[str, ...]:
    ordered = sorted(run.cells, key=lambda item: item.batch_position)
    index = ordered.index(cell)
    return tuple(
        ordered[neighbor].config.reproduction_path
        for neighbor in (index - 1, index + 1)
        if 0 <= neighbor < len(ordered)
    )


def _compare_runs(
    expected: IsolationRunCaptureV1,
    actual: IsolationRunCaptureV1,
    *,
    key: str,
    code: str,
) -> list[IsolationDiagnosticV1]:
    getter = _by_logical if key == "logical" else _by_occurrence
    left = getter(expected)
    right = getter(actual)
    diagnostics: list[IsolationDiagnosticV1] = []
    if set(left) != set(right):
        diagnostics.append(
            _diagnostic(
                f"{code}.cell_set",
                "compared runs do not contain the same cells",
                expected_run=expected.run_id,
                actual_run=actual.run_id,
                expected_cells=sorted(left),
                actual_cells=sorted(right),
            )
        )
        return diagnostics
    for cell_id in sorted(left):
        if left[cell_id].exact_payload() != right[cell_id].exact_payload():
            diagnostics.append(
                _diagnostic(
                    f"{code}.cell_mismatch",
                    "exact cell receipt differs",
                    expected_run=expected.run_id,
                    actual_run=actual.run_id,
                    cell_id=cell_id,
                    expected_sha256=_sha256(left[cell_id].exact_payload()),
                    actual_sha256=_sha256(right[cell_id].exact_payload()),
                )
            )
    return diagnostics


def verify_mixed_mode_resident_isolation_v1(
    plan: MixedModeIsolationPlanV1,
    runs: Sequence[IsolationRunCaptureV1],
) -> MixedModeIsolationGateResultV1:
    """Verify all nine resident-isolation obligations.

    Missing or ambiguous evidence produces a failed check; it is never skipped.
    Low live-birth exposure is classified as ``not_evaluable`` and is not
    converted into negative scientific evidence.
    """

    if type(plan) is not MixedModeIsolationPlanV1:
        raise MixedModeIsolationContractError("plan must be a v1 plan")
    if isinstance(runs, (str, bytes)) or any(
        type(run) is not IsolationRunCaptureV1 for run in runs
    ):
        raise MixedModeIsolationContractError("runs must contain v1 captures")
    run_ids = [run.run_id for run in runs]
    if len(set(run_ids)) != len(run_ids):
        raise MixedModeIsolationContractError("run ids must be unique")
    indexed = {run.run_id: run for run in runs}
    checks: list[IsolationCheckV1] = []

    def finish(check_id: str, diagnostics: list[IsolationDiagnosticV1]) -> None:
        checks.append(
            IsolationCheckV1(
                check_id=check_id,
                ok=not diagnostics,
                diagnostics=tuple(diagnostics),
            )
        )

    # 1. Each mixed reference cell must have an exact one-cell counterpart.
    diagnostics = []
    reference = indexed.get(plan.mixed_reference_run_id)
    if reference is None or reference.run_kind != RUN_KIND_MIXED:
        diagnostics.append(
            _diagnostic(
                "alone.reference_missing",
                "mixed reference run is missing or mistyped",
                run_id=plan.mixed_reference_run_id,
            )
        )
    else:
        alone_by_logical: dict[str, IsolationRunCaptureV1] = {}
        for run in runs:
            if run.run_kind != RUN_KIND_ALONE:
                continue
            if len(run.cells) != 1:
                diagnostics.append(
                    _diagnostic(
                        "alone.not_single_cell",
                        "an alone run contains more than one cell",
                        run_id=run.run_id,
                    )
                )
                continue
            logical_id = run.cells[0].config.logical_cell_id
            if logical_id in alone_by_logical:
                diagnostics.append(
                    _diagnostic(
                        "alone.duplicate_evidence",
                        "multiple alone runs claim the same logical cell",
                        logical_cell_id=logical_id,
                    )
                )
            alone_by_logical[logical_id] = run
        for cell in reference.cells:
            alone = alone_by_logical.get(cell.config.logical_cell_id)
            if alone is None:
                diagnostics.append(
                    _diagnostic(
                        "alone.missing",
                        "reference cell has no alone execution",
                        logical_cell_id=cell.config.logical_cell_id,
                    )
                )
            elif cell.exact_payload() != alone.cells[0].exact_payload():
                diagnostics.append(
                    _diagnostic(
                        "alone.cell_mismatch",
                        "alone and mixed cell receipts differ",
                        logical_cell_id=cell.config.logical_cell_id,
                        mixed_sha256=_sha256(cell.exact_payload()),
                        alone_sha256=_sha256(alone.cells[0].exact_payload()),
                    )
                )
    finish("cell_alone_equivalence", diagnostics)

    # 2. Every registered permutation must reorder, but not alter, cells.
    diagnostics = []
    if reference is not None:
        reference_order = [
            cell.config.logical_cell_id
            for cell in sorted(reference.cells, key=lambda item: item.batch_position)
        ]
        for run_id in plan.permutation_run_ids:
            run = indexed.get(run_id)
            if run is None or run.run_kind != RUN_KIND_PERMUTED:
                diagnostics.append(
                    _diagnostic(
                        "permutation.missing",
                        "registered permutation run is missing or mistyped",
                        run_id=run_id,
                    )
                )
                continue
            order = [
                cell.config.logical_cell_id
                for cell in sorted(run.cells, key=lambda item: item.batch_position)
            ]
            if order == reference_order:
                diagnostics.append(
                    _diagnostic(
                        "permutation.not_permuted",
                        "permutation run retained reference cell order",
                        run_id=run_id,
                    )
                )
            diagnostics.extend(
                _compare_runs(reference, run, key="logical", code="permutation")
            )
    finish("batch_permutation_invariance", diagnostics)

    # 3. Duplicates must be exact in distinct positions and neighborhoods.
    diagnostics = []
    covered_duplicate_paths: set[str] = set()
    for run_id in plan.duplicate_run_ids:
        run = indexed.get(run_id)
        if run is None or run.run_kind != RUN_KIND_DUPLICATE:
            diagnostics.append(
                _diagnostic(
                    "duplicate.missing",
                    "registered duplicate run is missing or mistyped",
                    run_id=run_id,
                )
            )
            continue
        groups: dict[str, list[ResidentCellCaptureV1]] = {}
        for cell in run.cells:
            groups.setdefault(cell.config.equivalence_key, []).append(cell)
        duplicates = [group for group in groups.values() if len(group) >= 2]
        if not duplicates:
            diagnostics.append(
                _diagnostic(
                    "duplicate.no_pair",
                    "duplicate run contains no repeated equivalence key",
                    run_id=run_id,
                )
            )
        for group in duplicates:
            paths = {cell.config.reproduction_path for cell in group}
            if len(paths) != 1:
                diagnostics.append(
                    _diagnostic(
                        "duplicate.path_mismatch",
                        "duplicate cells do not share a reproduction path",
                        run_id=run_id,
                        equivalence_key=group[0].config.equivalence_key,
                    )
                )
                continue
            covered_duplicate_paths.update(paths)
            if len({cell.batch_position for cell in group}) != len(group):
                diagnostics.append(
                    _diagnostic(
                        "duplicate.position_collision",
                        "duplicate cells do not occupy distinct positions",
                        run_id=run_id,
                    )
                )
            derived_neighbors = {_derived_neighbor_paths(run, cell) for cell in group}
            for cell in group:
                derived = _derived_neighbor_paths(run, cell)
                if cell.neighbor_path_ids != derived:
                    diagnostics.append(
                        _diagnostic(
                            "duplicate.neighbor_receipt_mismatch",
                            "captured neighbor modes do not match batch order",
                            run_id=run_id,
                            occurrence_id=cell.occurrence_id,
                            captured=list(cell.neighbor_path_ids),
                            derived=list(derived),
                        )
                    )
            if len(derived_neighbors) < 2 and reference is not None:
                logical_ids = {
                    cell.config.logical_cell_id for cell in group
                }
                reference_matches = [
                    cell
                    for cell in reference.cells
                    if cell.config.logical_cell_id in logical_ids
                ]
                if len(logical_ids) == 1 and len(reference_matches) == 1:
                    reference_cell = reference_matches[0]
                    reference_neighbors = _derived_neighbor_paths(
                        reference,
                        reference_cell,
                    )
                    if reference_cell.neighbor_path_ids != reference_neighbors:
                        diagnostics.append(
                            _diagnostic(
                                "duplicate.reference_neighbor_receipt_mismatch",
                                (
                                    "mixed-reference neighbor modes do not "
                                    "match batch order"
                                ),
                                run_id=reference.run_id,
                                occurrence_id=reference_cell.occurrence_id,
                            )
                        )
                    derived_neighbors.add(reference_neighbors)
                    if (
                        reference_cell.exact_payload()
                        != group[0].exact_payload()
                    ):
                        diagnostics.append(
                            _diagnostic(
                                "duplicate.reference_cell_mismatch",
                                (
                                    "duplicate cell differs from its "
                                    "heterogeneous mixed reference"
                                ),
                                run_id=run_id,
                                logical_cell_id=reference_cell.config.logical_cell_id,
                            )
                        )
            if len(derived_neighbors) < 2:
                diagnostics.append(
                    _diagnostic(
                        "duplicate.neighborhood_not_varied",
                        "duplicate cells lack different neighboring modes",
                        run_id=run_id,
                        equivalence_key=group[0].config.equivalence_key,
                    )
                )
            baseline_payload = group[0].exact_payload()
            baseline_payload["config"] = {
                **baseline_payload["config"],
                "logical_cell_id": "<duplicate>",
            }
            for cell in group[1:]:
                payload = cell.exact_payload()
                payload["config"] = {
                    **payload["config"],
                    "logical_cell_id": "<duplicate>",
                }
                if payload != baseline_payload:
                    diagnostics.append(
                        _diagnostic(
                            "duplicate.cell_mismatch",
                            "duplicate cell receipts differ",
                            run_id=run_id,
                            equivalence_key=cell.config.equivalence_key,
                            occurrence_id=cell.occurrence_id,
                        )
                    )
    missing_duplicate_paths = set(plan.duplicate_required_path_ids).difference(
        covered_duplicate_paths
    )
    if missing_duplicate_paths:
        diagnostics.append(
            _diagnostic(
                "duplicate.path_coverage",
                "required reproduction paths lack duplicate isolation evidence",
                missing_paths=sorted(missing_duplicate_paths),
            )
        )
    finish("duplicate_cell_isolation", diagnostics)

    # 4. RNG namespaces and exact donor budgets are cell/path local.
    diagnostics = []
    for run in runs:
        for cell in run.cells:
            config = cell.config
            active = cell.receipt(config.reproduction_path)
            if active is None or not active.enabled:
                diagnostics.append(
                    _diagnostic(
                        "rng.active_receipt_missing",
                        "configured reproduction path lacks an enabled receipt",
                        run_id=run.run_id,
                        occurrence_id=cell.occurrence_id,
                        path_id=config.reproduction_path,
                    )
                )
                continue
            unknown = set(dict(active.rng_counters)).difference(
                config.authorized_rng_counter_ids
            )
            if unknown:
                diagnostics.append(
                    _diagnostic(
                        "rng.unauthorized_counter",
                        "path consumed an unauthorized RNG namespace",
                        run_id=run.run_id,
                        occurrence_id=cell.occurrence_id,
                        counter_ids=sorted(unknown),
                    )
                )
            expected_draws = (
                active.successful_births * config.donor_draws_per_successful_birth
            )
            if len(active.donor_draw_ordinals) != expected_draws:
                diagnostics.append(
                    _diagnostic(
                        "rng.donor_draw_budget",
                        "donor draw count does not match the frozen per-birth budget",
                        run_id=run.run_id,
                        occurrence_id=cell.occurrence_id,
                        expected=expected_draws,
                        actual=len(active.donor_draw_ordinals),
                    )
                )
            selected_count = sum(
                len(attempt.selected_parent_ids) for attempt in active.attempts
            )
            expected_selected = len(active.attempts) * config.parent_arity
            if selected_count != expected_selected:
                diagnostics.append(
                    _diagnostic(
                        "rng.selected_parent_arity",
                        "selected-parent exposure does not match parent arity",
                        run_id=run.run_id,
                        occurrence_id=cell.occurrence_id,
                        expected=expected_selected,
                        actual=selected_count,
                    )
                )
            contribution_count = sum(
                len(attempt.contributing_parent_ids) for attempt in active.attempts
            )
            expected_contributions = active.successful_births * config.parent_arity
            if contribution_count != expected_contributions:
                diagnostics.append(
                    _diagnostic(
                        "rng.parent_contribution_arity",
                        "parent contributions do not match successful-birth arity",
                        run_id=run.run_id,
                        occurrence_id=cell.occurrence_id,
                        expected=expected_contributions,
                        actual=contribution_count,
                    )
                )
            selection = config.selection
            if (
                selection.eligible_parent_source
                != "authoritative_eligible_parent_set_v1"
                or selection.draw_distribution != "uniform_v1"
                or not selection.deterministic
                or not _BLINDED_DIMENSIONS.issubset(selection.blinded_dimensions)
            ):
                diagnostics.append(
                    _diagnostic(
                        "rng.selection_contract_not_neutral",
                        "selection is not authoritative, uniform, deterministic, and fully blinded",
                        run_id=run.run_id,
                        occurrence_id=cell.occurrence_id,
                        selection=selection.payload(),
                    )
                )
    finish("rng_and_donor_draw_isolation", diagnostics)

    # 5. The fault target must stop without mutation; every peer stays exact.
    diagnostics = []
    for case in plan.overflow_cases:
        expected = indexed.get(case.reference_run_id)
        fault = indexed.get(case.fault_run_id)
        if expected is None or fault is None or fault.run_kind != RUN_KIND_OVERFLOW:
            diagnostics.append(
                _diagnostic(
                    "overflow.missing",
                    "overflow reference or fault run is missing/mistyped",
                    reference_run_id=case.reference_run_id,
                    fault_run_id=case.fault_run_id,
                )
            )
            continue
        left = _by_occurrence(expected)
        right = _by_occurrence(fault)
        if set(left) != set(right) or case.target_occurrence_id not in right:
            diagnostics.append(
                _diagnostic(
                    "overflow.cell_set",
                    "overflow comparison cell set or target is invalid",
                    fault_run_id=case.fault_run_id,
                    target_occurrence_id=case.target_occurrence_id,
                )
            )
            continue
        target = right[case.target_occurrence_id]
        if target.config.reproduction_path not in plan.live_birth_path_ids:
            diagnostics.append(
                _diagnostic(
                    "overflow.target_not_live_birth",
                    "overflow target is not a registered live-birth path",
                    fault_run_id=case.fault_run_id,
                    target_occurrence_id=case.target_occurrence_id,
                    path_id=target.config.reproduction_path,
                )
            )
        if (
            target.ok
            or not target.fail_closed
            or target.error_code != case.overflow_error_code
            or target.pre_fault_state_sha256 != target.final_state_sha256
            or target.pre_fault_authority_sha256 is None
            or target.pre_fault_authority_sha256 != target.final_authority_sha256
        ):
            diagnostics.append(
                _diagnostic(
                    "overflow.target_not_fail_closed",
                    "overflow target did not fail closed at its pre-fault state",
                    fault_run_id=case.fault_run_id,
                    target_occurrence_id=case.target_occurrence_id,
                    error_code=target.error_code,
                    pre_fault_authority_sha256=(target.pre_fault_authority_sha256),
                    final_authority_sha256=target.final_authority_sha256,
                )
            )
        for occurrence_id in sorted(set(left).difference({case.target_occurrence_id})):
            if (
                left[occurrence_id].exact_payload()
                != right[occurrence_id].exact_payload()
            ):
                diagnostics.append(
                    _diagnostic(
                        "overflow.peer_changed",
                        "non-target cell changed during peer overflow",
                        fault_run_id=case.fault_run_id,
                        occurrence_id=occurrence_id,
                    )
                )
    finish("overflow_isolation", diagnostics)

    # 6. Uninterrupted and explicitly fresh-owner continuations are exact.
    diagnostics = []
    for case in plan.checkpoint_cases:
        baseline = indexed.get(case.uninterrupted_run_id)
        resumed = indexed.get(case.fresh_owner_run_id)
        if (
            baseline is None
            or resumed is None
            or baseline.run_kind != RUN_KIND_CHECKPOINT_BASELINE
            or resumed.run_kind != RUN_KIND_CHECKPOINT_FRESH_OWNER
        ):
            diagnostics.append(
                _diagnostic(
                    "checkpoint.missing",
                    "checkpoint comparison run is missing or mistyped",
                    uninterrupted_run_id=case.uninterrupted_run_id,
                    fresh_owner_run_id=case.fresh_owner_run_id,
                )
            )
            continue
        if baseline.fresh_owner or not resumed.fresh_owner:
            diagnostics.append(
                _diagnostic(
                    "checkpoint.owner_boundary",
                    "checkpoint comparison does not prove a fresh-owner boundary",
                    uninterrupted_run_id=case.uninterrupted_run_id,
                    fresh_owner_run_id=case.fresh_owner_run_id,
                )
            )
        diagnostics.extend(
            _compare_runs(baseline, resumed, key="logical", code="checkpoint")
        )
    finish("fresh_owner_checkpoint_isolation", diagnostics)

    # 7. Every non-selected path must be entirely quiescent.
    diagnostics = []
    for run in runs:
        for cell in run.cells:
            receipts = {receipt.path_id: receipt for receipt in cell.path_receipts}
            if set(receipts) != set(plan.path_ids):
                diagnostics.append(
                    _diagnostic(
                        "disabled.path_receipt_set",
                        "cell does not expose exactly the registered path receipts",
                        run_id=run.run_id,
                        occurrence_id=cell.occurrence_id,
                        expected_paths=sorted(plan.path_ids),
                        actual_paths=sorted(receipts),
                    )
                )
                continue
            for path_id, receipt in receipts.items():
                should_enable = path_id == cell.config.reproduction_path
                if receipt.enabled != should_enable:
                    diagnostics.append(
                        _diagnostic(
                            "disabled.enablement_mismatch",
                            "path enablement does not match per-cell configuration",
                            run_id=run.run_id,
                            occurrence_id=cell.occurrence_id,
                            path_id=path_id,
                        )
                    )
                if not should_enable and (
                    receipt.rng_counters
                    or receipt.donor_draw_ordinals
                    or receipt.counter_writes
                    or receipt.sidecar_writes
                    or receipt.event_row_writes
                    or receipt.rng_state_writes
                    or receipt.attempts
                ):
                    diagnostics.append(
                        _diagnostic(
                            "disabled.path_wrote",
                            "disabled path consumed RNG or wrote state/telemetry",
                            run_id=run.run_id,
                            occurrence_id=cell.occurrence_id,
                            path_id=path_id,
                        )
                    )
    finish("disabled_path_write_isolation", diagnostics)

    # 8. Both path families must expose complete attempt-level parent data.
    diagnostics = []
    if reference is not None:
        covered_paths: set[str] = set()
        for cell in reference.cells:
            active = cell.receipt(cell.config.reproduction_path)
            if active is None:
                continue
            covered_paths.add(active.path_id)
            exposure = active.exposure_payload()
            if (
                exposure["attempts"]
                != exposure["successful_births"] + exposure["failed_attempts"]
                or len(exposure["primary_eligible_parents"])
                != exposure["attempts"]
                or len(exposure["mate_eligible_parents"])
                != exposure["attempts"]
                or len(exposure["selected_parents"]) != exposure["attempts"]
            ):
                diagnostics.append(
                    _diagnostic(
                        "exposure.inconsistent",
                        "inheritance-event exposure counters are inconsistent",
                        occurrence_id=cell.occurrence_id,
                        path_id=active.path_id,
                    )
                )
        missing_paths = set(plan.path_ids).difference(covered_paths)
        if missing_paths:
            diagnostics.append(
                _diagnostic(
                    "exposure.path_coverage",
                    "mixed reference lacks separate exposure for registered paths",
                    missing_paths=sorted(missing_paths),
                )
            )
    finish("inheritance_event_exposure", diagnostics)

    # 9. Threshold must precede and bind every capture; low cells are neutral.
    diagnostics = []
    threshold = plan.minimum_live_birth_exposure
    frozen_at = _utc("frozen_at_utc", threshold.frozen_at_utc)
    for run in runs:
        if run.plan_sha256 != plan.digest_sha256:
            diagnostics.append(
                _diagnostic(
                    "minimum.plan_binding",
                    "run is not bound to the frozen isolation plan",
                    run_id=run.run_id,
                    expected=plan.digest_sha256,
                    actual=run.plan_sha256,
                )
            )
        if _utc("captured_at_utc", run.captured_at_utc) <= frozen_at:
            diagnostics.append(
                _diagnostic(
                    "minimum.not_frozen_before_capture",
                    "minimum exposure was not frozen before capture",
                    run_id=run.run_id,
                    frozen_at_utc=threshold.frozen_at_utc,
                    captured_at_utc=run.captured_at_utc,
                )
            )
    evaluability: dict[str, dict[str, Any]] = {}
    if reference is not None:
        for cell in reference.cells:
            if cell.config.reproduction_path not in plan.live_birth_path_ids:
                continue
            active = cell.receipt(cell.config.reproduction_path)
            exposure = (
                {
                    "attempts": 0,
                    "successful_births": 0,
                    "parent_contribution_counts": {},
                }
                if active is None
                else active.exposure_payload()
            )
            if threshold.metric == "parent_contributions":
                observed = sum(exposure["parent_contribution_counts"].values())
            else:
                observed = int(exposure[threshold.metric])
            evaluability[cell.config.logical_cell_id] = {
                "path_id": cell.config.reproduction_path,
                "metric": threshold.metric,
                "observed": observed,
                "minimum": threshold.minimum,
                "classification": (
                    "evaluable"
                    if observed >= threshold.minimum
                    else "not_evaluable_low_exposure"
                ),
                "negative_scientific_evidence_allowed": (observed >= threshold.minimum),
            }
    if not evaluability:
        diagnostics.append(
            _diagnostic(
                "minimum.no_live_birth_cells",
                "mixed reference contains no registered live-birth cells",
            )
        )
    finish("frozen_minimum_live_birth_exposure", diagnostics)

    return MixedModeIsolationGateResultV1(
        plan_sha256=plan.digest_sha256,
        ok=all(check.ok for check in checks),
        checks=tuple(checks),
        live_birth_evaluability=evaluability,
    )
