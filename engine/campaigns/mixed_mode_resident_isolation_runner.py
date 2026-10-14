"""Durable execution harness for the mixed resident-cell isolation gate.

The harness deliberately does not know how to execute a simulation.  A caller
must inject a resident execution callback that returns a complete, hash-bound
observation.  The harness validates and projects that observation into the
existing isolation-gate contracts; it never manufactures missing scientific
state, counters, draws, or inheritance-event rows.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence
from uuid import uuid4

from .mixed_mode_resident_isolation_gate import (
    CELL_SCHEMA_VERSION,
    MINIMUM_EXPOSURE_SCHEMA_VERSION,
    PATH_RECEIPT_SCHEMA_VERSION,
    PLAN_SCHEMA_VERSION,
    RUN_KIND_ALONE,
    RUN_KIND_CHECKPOINT_BASELINE,
    RUN_KIND_CHECKPOINT_FRESH_OWNER,
    RUN_KIND_DUPLICATE,
    RUN_KIND_MIXED,
    RUN_KIND_OVERFLOW,
    RUN_KIND_PERMUTED,
    SELECTION_SCHEMA_VERSION,
    CheckpointCaseV1,
    FrozenMinimumInheritanceExposureV1,
    InheritanceAttemptV1,
    IsolationRunCaptureV1,
    MixedModeIsolationContractError,
    MixedModeIsolationGateResultV1,
    MixedModeIsolationPlanV1,
    OverflowCaseV1,
    ParentSelectionContractV1,
    ReproductionPathReceiptV1,
    ResidentCellCaptureV1,
    ResidentReproductionCellConfigV1,
    verify_mixed_mode_resident_isolation_v1,
)


RUNNER_PLAN_SCHEMA_VERSION = "eiel_mixed_mode_resident_isolation_runner_plan_v2"
REGISTERED_RUN_SCHEMA_VERSION = "eiel_mixed_mode_resident_registered_run_v2"
REGISTERED_CELL_SCHEMA_VERSION = "eiel_mixed_mode_resident_registered_cell_v2"
EXECUTION_REQUEST_SCHEMA_VERSION = (
    "eiel_mixed_mode_resident_isolation_execution_request_v2"
)
EXECUTION_OBSERVATION_SCHEMA_VERSION = (
    "eiel_mixed_mode_resident_isolation_execution_observation_v2"
)
CAPTURE_ARTIFACT_SCHEMA_VERSION = (
    "eiel_mixed_mode_resident_isolation_capture_artifact_v2"
)
RUNNER_RESULT_SCHEMA_VERSION = "eiel_mixed_mode_resident_isolation_runner_result_v2"

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
_HEX = frozenset("0123456789abcdef")


class MixedModeIsolationRunnerError(RuntimeError):
    """A gate execution cannot proceed without complete immutable evidence."""


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
        raise MixedModeIsolationRunnerError(
            "runner values must be canonical JSON data"
        ) from exc


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _digest(name: str, value: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        raise MixedModeIsolationRunnerError(
            f"{name} must be a lowercase SHA-256 digest"
        )


def _text(name: str, value: str) -> None:
    if not isinstance(value, str) or not value:
        raise MixedModeIsolationRunnerError(f"{name} must be non-empty text")


def _timestamp(name: str, value: str) -> datetime:
    _text(name, value)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise MixedModeIsolationRunnerError(
            f"{name} must be an ISO-8601 timestamp"
        ) from exc
    if parsed.tzinfo is None:
        raise MixedModeIsolationRunnerError(f"{name} must include a timezone")
    return parsed.astimezone(timezone.utc)


def _freeze_json(value: Any, *, name: str) -> Any:
    if value is None or type(value) in {bool, int, str}:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise MixedModeIsolationRunnerError(f"{name} must be finite")
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise MixedModeIsolationRunnerError(f"{name} keys must be text")
            frozen[key] = _freeze_json(item, name=f"{name}.{key}")
        return MappingProxyType(dict(sorted(frozen.items())))
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(
            _freeze_json(item, name=f"{name}[{index}]")
            for index, item in enumerate(value)
        )
    raise MixedModeIsolationRunnerError(f"{name} must be JSON-compatible")


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _exact_object(
    value: Any,
    *,
    name: str,
    keys: frozenset[str],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise MixedModeIsolationRunnerError(f"{name} must be an object")
    actual = frozenset(value)
    if actual != keys:
        missing = sorted(keys.difference(actual))
        unexpected = sorted(actual.difference(keys))
        raise MixedModeIsolationRunnerError(
            f"{name} field set is invalid; missing={missing}, unexpected={unexpected}"
        )
    return dict(value)


def _exact_sequence(value: Any, *, name: str) -> list[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise MixedModeIsolationRunnerError(f"{name} must be an array")
    return list(value)


@dataclass(frozen=True, slots=True)
class RegisteredResidentCellV1:
    """One cell and its physical placement frozen before execution."""

    occurrence_id: str
    config: ResidentReproductionCellConfigV1
    batch_position: int
    expected_neighbor_path_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _text("occurrence_id", self.occurrence_id)
        if type(self.config) is not ResidentReproductionCellConfigV1:
            raise MixedModeIsolationRunnerError(
                "config must be ResidentReproductionCellConfigV1"
            )
        if type(self.batch_position) is not int or self.batch_position < 0:
            raise MixedModeIsolationRunnerError(
                "batch_position must be a non-negative integer"
            )
        for path_id in self.expected_neighbor_path_ids:
            _text("expected_neighbor_path_ids", path_id)

    def payload(self) -> dict[str, Any]:
        return {
            "schema_version": REGISTERED_CELL_SCHEMA_VERSION,
            "occurrence_id": self.occurrence_id,
            "config": self.config.payload(),
            "batch_position": self.batch_position,
            "expected_neighbor_path_ids": list(self.expected_neighbor_path_ids),
        }

    @classmethod
    def from_payload(cls, value: Any) -> RegisteredResidentCellV1:
        row = _exact_object(
            value,
            name="registered cell",
            keys=frozenset(
                {
                    "schema_version",
                    "occurrence_id",
                    "config",
                    "batch_position",
                    "expected_neighbor_path_ids",
                }
            ),
        )
        if row["schema_version"] != REGISTERED_CELL_SCHEMA_VERSION:
            raise MixedModeIsolationRunnerError(
                "registered cell schema_version is invalid"
            )
        try:
            return cls(
                occurrence_id=row["occurrence_id"],
                config=_parse_config(row["config"]),
                batch_position=row["batch_position"],
                expected_neighbor_path_ids=tuple(
                    _exact_sequence(
                        row["expected_neighbor_path_ids"],
                        name="expected_neighbor_path_ids",
                    )
                ),
            )
        except MixedModeIsolationContractError as exc:
            raise MixedModeIsolationRunnerError(
                f"registered cell violates its nested contract: {exc}"
            ) from exc


@dataclass(frozen=True, slots=True)
class RegisteredIsolationRunV1:
    """Typed preregistration for one callback invocation."""

    run_id: str
    run_kind: str
    fresh_owner: bool
    cells: tuple[RegisteredResidentCellV1, ...]
    execution_payload: Mapping[str, Any]

    def __post_init__(self) -> None:
        _text("run_id", self.run_id)
        if self.run_kind not in _RUN_KINDS:
            raise MixedModeIsolationRunnerError("run_kind is not supported")
        if type(self.fresh_owner) is not bool:
            raise MixedModeIsolationRunnerError("fresh_owner must be boolean")
        if not self.cells or any(
            type(cell) is not RegisteredResidentCellV1 for cell in self.cells
        ):
            raise MixedModeIsolationRunnerError(
                "cells must contain RegisteredResidentCellV1 values"
            )
        occurrences = [cell.occurrence_id for cell in self.cells]
        positions = [cell.batch_position for cell in self.cells]
        if len(set(occurrences)) != len(occurrences):
            raise MixedModeIsolationRunnerError(
                "registered occurrence ids must be unique within a run"
            )
        if len(set(positions)) != len(positions):
            raise MixedModeIsolationRunnerError(
                "registered batch positions must be unique within a run"
            )
        frozen = _freeze_json(self.execution_payload, name="execution_payload")
        if not isinstance(frozen, Mapping):
            raise MixedModeIsolationRunnerError("execution_payload must be an object")
        object.__setattr__(self, "execution_payload", frozen)

    def payload(self) -> dict[str, Any]:
        return {
            "schema_version": REGISTERED_RUN_SCHEMA_VERSION,
            "run_id": self.run_id,
            "run_kind": self.run_kind,
            "fresh_owner": self.fresh_owner,
            "cells": [
                cell.payload()
                for cell in sorted(self.cells, key=lambda item: item.batch_position)
            ],
            "execution_payload": _thaw_json(self.execution_payload),
        }

    @classmethod
    def from_payload(cls, value: Any) -> RegisteredIsolationRunV1:
        row = _exact_object(
            value,
            name="registered run",
            keys=frozenset(
                {
                    "schema_version",
                    "run_id",
                    "run_kind",
                    "fresh_owner",
                    "cells",
                    "execution_payload",
                }
            ),
        )
        if row["schema_version"] != REGISTERED_RUN_SCHEMA_VERSION:
            raise MixedModeIsolationRunnerError(
                "registered run schema_version is invalid"
            )
        return cls(
            run_id=row["run_id"],
            run_kind=row["run_kind"],
            fresh_owner=row["fresh_owner"],
            cells=tuple(
                RegisteredResidentCellV1.from_payload(item)
                for item in _exact_sequence(row["cells"], name="registered run cells")
            ),
            execution_payload=row["execution_payload"],
        )

    @property
    def digest_sha256(self) -> str:
        return _sha256(self.payload())


def _run_by_id(
    runs: Sequence[RegisteredIsolationRunV1],
) -> dict[str, RegisteredIsolationRunV1]:
    return {run.run_id: run for run in runs}


def _require_registered_kind(
    indexed: Mapping[str, RegisteredIsolationRunV1],
    run_id: str,
    kind: str,
) -> None:
    run = indexed.get(run_id)
    if run is None or run.run_kind != kind:
        raise MixedModeIsolationRunnerError(
            f"registered run {run_id!r} must exist with kind {kind!r}"
        )


@dataclass(frozen=True, slots=True)
class MixedModeResidentIsolationRunnerPlanV1:
    """Complete immutable callback schedule around a frozen nine-check plan."""

    gate_plan: MixedModeIsolationPlanV1
    execution_callback_id: str
    execution_authority_sha256: str
    preregistered_at_utc: str
    runs: tuple[RegisteredIsolationRunV1, ...]

    def __post_init__(self) -> None:
        if type(self.gate_plan) is not MixedModeIsolationPlanV1:
            raise MixedModeIsolationRunnerError(
                "gate_plan must be MixedModeIsolationPlanV1"
            )
        _text("execution_callback_id", self.execution_callback_id)
        _digest("execution_authority_sha256", self.execution_authority_sha256)
        registered_at = _timestamp(
            "preregistered_at_utc", self.preregistered_at_utc
        )
        frozen_at = _timestamp(
            "minimum_live_birth_exposure.frozen_at_utc",
            self.gate_plan.minimum_live_birth_exposure.frozen_at_utc,
        )
        if registered_at < frozen_at:
            raise MixedModeIsolationRunnerError(
                "runner plan cannot predate its frozen minimum exposure threshold"
            )
        if not self.runs or any(
            type(run) is not RegisteredIsolationRunV1 for run in self.runs
        ):
            raise MixedModeIsolationRunnerError(
                "runs must contain RegisteredIsolationRunV1 values"
            )
        run_ids = [run.run_id for run in self.runs]
        if len(set(run_ids)) != len(run_ids):
            raise MixedModeIsolationRunnerError("registered run ids must be unique")
        indexed = _run_by_id(self.runs)
        _require_registered_kind(
            indexed, self.gate_plan.mixed_reference_run_id, RUN_KIND_MIXED
        )
        for run_id in self.gate_plan.permutation_run_ids:
            _require_registered_kind(indexed, run_id, RUN_KIND_PERMUTED)
        for run_id in self.gate_plan.duplicate_run_ids:
            _require_registered_kind(indexed, run_id, RUN_KIND_DUPLICATE)
        for case in self.gate_plan.overflow_cases:
            if case.reference_run_id not in indexed:
                raise MixedModeIsolationRunnerError(
                    f"overflow reference run {case.reference_run_id!r} is not registered"
                )
            _require_registered_kind(indexed, case.fault_run_id, RUN_KIND_OVERFLOW)
        for case in self.gate_plan.checkpoint_cases:
            _require_registered_kind(
                indexed, case.uninterrupted_run_id, RUN_KIND_CHECKPOINT_BASELINE
            )
            _require_registered_kind(
                indexed, case.fresh_owner_run_id, RUN_KIND_CHECKPOINT_FRESH_OWNER
            )
            if indexed[case.uninterrupted_run_id].fresh_owner:
                raise MixedModeIsolationRunnerError(
                    "checkpoint baseline cannot be marked fresh_owner"
                )
            if not indexed[case.fresh_owner_run_id].fresh_owner:
                raise MixedModeIsolationRunnerError(
                    "checkpoint continuation must be marked fresh_owner"
                )
        reference = indexed[self.gate_plan.mixed_reference_run_id]
        alone_ids = {
            run.cells[0].config.logical_cell_id
            for run in self.runs
            if run.run_kind == RUN_KIND_ALONE and len(run.cells) == 1
        }
        reference_ids = {cell.config.logical_cell_id for cell in reference.cells}
        if alone_ids != reference_ids:
            raise MixedModeIsolationRunnerError(
                "alone registrations must cover exactly the mixed reference cells"
            )

    def payload(self) -> dict[str, Any]:
        return {
            "schema_version": RUNNER_PLAN_SCHEMA_VERSION,
            "gate_plan": self.gate_plan.payload(),
            "gate_plan_sha256": self.gate_plan.digest_sha256,
            "execution_callback_id": self.execution_callback_id,
            "execution_authority_sha256": self.execution_authority_sha256,
            "preregistered_at_utc": self.preregistered_at_utc,
            "runs": [run.payload() for run in self.runs],
        }

    @classmethod
    def from_payload(cls, value: Any) -> MixedModeResidentIsolationRunnerPlanV1:
        row = _exact_object(
            value,
            name="runner plan",
            keys=frozenset(
                {
                    "schema_version",
                    "gate_plan",
                    "gate_plan_sha256",
                    "execution_callback_id",
                    "execution_authority_sha256",
                    "preregistered_at_utc",
                    "runs",
                }
            ),
        )
        if row["schema_version"] != RUNNER_PLAN_SCHEMA_VERSION:
            raise MixedModeIsolationRunnerError(
                "runner plan schema_version is invalid"
            )
        gate_plan = _parse_gate_plan(row["gate_plan"])
        _digest("gate_plan_sha256", row["gate_plan_sha256"])
        if row["gate_plan_sha256"] != gate_plan.digest_sha256:
            raise MixedModeIsolationRunnerError(
                "gate_plan_sha256 does not bind gate_plan"
            )
        return cls(
            gate_plan=gate_plan,
            execution_callback_id=row["execution_callback_id"],
            execution_authority_sha256=row["execution_authority_sha256"],
            preregistered_at_utc=row["preregistered_at_utc"],
            runs=tuple(
                RegisteredIsolationRunV1.from_payload(item)
                for item in _exact_sequence(row["runs"], name="runner plan runs")
            ),
        )

    @property
    def digest_sha256(self) -> str:
        return _sha256(self.payload())


@dataclass(frozen=True, slots=True)
class ResidentIsolationExecutionRequestV1:
    """Public request passed to the injected real resident execution owner."""

    runner_plan_sha256: str
    gate_plan_sha256: str
    execution_callback_id: str
    execution_authority_sha256: str
    run: RegisteredIsolationRunV1

    def payload(self) -> dict[str, Any]:
        return {
            "schema_version": EXECUTION_REQUEST_SCHEMA_VERSION,
            "runner_plan_sha256": self.runner_plan_sha256,
            "gate_plan_sha256": self.gate_plan_sha256,
            "execution_callback_id": self.execution_callback_id,
            "execution_authority_sha256": self.execution_authority_sha256,
            "registered_run_sha256": self.run.digest_sha256,
            "registered_run": self.run.payload(),
        }


ResidentIsolationExecutionCallbackV1 = Callable[
    [ResidentIsolationExecutionRequestV1], Mapping[str, Any]
]


def bind_resident_isolation_observation_v1(
    observation_without_digest: Mapping[str, Any],
) -> dict[str, Any]:
    """Attach the exact observation digest required from an execution owner.

    This helper only seals supplied observation data.  It does not fill,
    calculate, or infer any scientific field.
    """

    if not isinstance(observation_without_digest, Mapping):
        raise MixedModeIsolationRunnerError("observation must be an object")
    if "observation_sha256" in observation_without_digest:
        raise MixedModeIsolationRunnerError(
            "unbound observation must not contain observation_sha256"
        )
    payload = json.loads(_canonical_bytes(observation_without_digest).decode("ascii"))
    payload["observation_sha256"] = _sha256(payload)
    return payload


def _parse_selection(value: Any) -> ParentSelectionContractV1:
    row = _exact_object(
        value,
        name="selection",
        keys=frozenset(
            {
                "schema_version",
                "selection_mode",
                "eligible_parent_source",
                "draw_distribution",
                "deterministic",
                "blinded_dimensions",
            }
        ),
    )
    if row["schema_version"] != SELECTION_SCHEMA_VERSION:
        raise MixedModeIsolationRunnerError("selection schema_version is invalid")
    return ParentSelectionContractV1(
        selection_mode=row["selection_mode"],
        eligible_parent_source=row["eligible_parent_source"],
        draw_distribution=row["draw_distribution"],
        deterministic=row["deterministic"],
        blinded_dimensions=tuple(
            _exact_sequence(row["blinded_dimensions"], name="blinded_dimensions")
        ),
    )


def _parse_config(value: Any) -> ResidentReproductionCellConfigV1:
    row = _exact_object(
        value,
        name="config",
        keys=frozenset(
            {
                "schema_version",
                "logical_cell_id",
                "equivalence_key",
                "reproduction_path",
                "selection",
                "parent_arity",
                "donor_draws_per_successful_birth",
                "authorized_rng_counter_ids",
            }
        ),
    )
    if row["schema_version"] != "eiel_resident_reproduction_cell_config_v1":
        raise MixedModeIsolationRunnerError("config schema_version is invalid")
    return ResidentReproductionCellConfigV1(
        logical_cell_id=row["logical_cell_id"],
        equivalence_key=row["equivalence_key"],
        reproduction_path=row["reproduction_path"],
        selection=_parse_selection(row["selection"]),
        parent_arity=row["parent_arity"],
        donor_draws_per_successful_birth=row[
            "donor_draws_per_successful_birth"
        ],
        authorized_rng_counter_ids=tuple(
            _exact_sequence(
                row["authorized_rng_counter_ids"],
                name="authorized_rng_counter_ids",
            )
        ),
    )


def _parse_gate_plan(value: Any) -> MixedModeIsolationPlanV1:
    row = _exact_object(
        value,
        name="gate plan",
        keys=frozenset(
            {
                "schema_version",
                "path_ids",
                "mixed_reference_run_id",
                "permutation_run_ids",
                "duplicate_run_ids",
                "duplicate_required_path_ids",
                "overflow_cases",
                "checkpoint_cases",
                "minimum_live_birth_exposure",
                "live_birth_path_ids",
            }
        ),
    )
    if row["schema_version"] != PLAN_SCHEMA_VERSION:
        raise MixedModeIsolationRunnerError("gate plan schema_version is invalid")

    minimum = _exact_object(
        row["minimum_live_birth_exposure"],
        name="minimum_live_birth_exposure",
        keys=frozenset(
            {
                "schema_version",
                "threshold_id",
                "metric",
                "minimum",
                "frozen_at_utc",
            }
        ),
    )
    if minimum["schema_version"] != MINIMUM_EXPOSURE_SCHEMA_VERSION:
        raise MixedModeIsolationRunnerError(
            "minimum_live_birth_exposure schema_version is invalid"
        )

    overflow_cases: list[OverflowCaseV1] = []
    for item in _exact_sequence(row["overflow_cases"], name="overflow_cases"):
        case = _exact_object(
            item,
            name="overflow case",
            keys=frozenset(
                {
                    "reference_run_id",
                    "fault_run_id",
                    "target_occurrence_id",
                    "overflow_error_code",
                }
            ),
        )
        try:
            overflow_cases.append(OverflowCaseV1(**case))
        except MixedModeIsolationContractError as exc:
            raise MixedModeIsolationRunnerError(
                f"overflow case violates its nested contract: {exc}"
            ) from exc

    checkpoint_cases: list[CheckpointCaseV1] = []
    for item in _exact_sequence(
        row["checkpoint_cases"], name="checkpoint_cases"
    ):
        case = _exact_object(
            item,
            name="checkpoint case",
            keys=frozenset({"uninterrupted_run_id", "fresh_owner_run_id"}),
        )
        try:
            checkpoint_cases.append(CheckpointCaseV1(**case))
        except MixedModeIsolationContractError as exc:
            raise MixedModeIsolationRunnerError(
                f"checkpoint case violates its nested contract: {exc}"
            ) from exc

    try:
        return MixedModeIsolationPlanV1(
            path_ids=tuple(_exact_sequence(row["path_ids"], name="path_ids")),
            mixed_reference_run_id=row["mixed_reference_run_id"],
            permutation_run_ids=tuple(
                _exact_sequence(
                    row["permutation_run_ids"], name="permutation_run_ids"
                )
            ),
            duplicate_run_ids=tuple(
                _exact_sequence(row["duplicate_run_ids"], name="duplicate_run_ids")
            ),
            duplicate_required_path_ids=tuple(
                _exact_sequence(
                    row["duplicate_required_path_ids"],
                    name="duplicate_required_path_ids",
                )
            ),
            overflow_cases=tuple(overflow_cases),
            checkpoint_cases=tuple(checkpoint_cases),
            minimum_live_birth_exposure=FrozenMinimumInheritanceExposureV1(
                threshold_id=minimum["threshold_id"],
                metric=minimum["metric"],
                minimum=minimum["minimum"],
                frozen_at_utc=minimum["frozen_at_utc"],
            ),
            live_birth_path_ids=tuple(
                _exact_sequence(
                    row["live_birth_path_ids"], name="live_birth_path_ids"
                )
            ),
        )
    except MixedModeIsolationContractError as exc:
        raise MixedModeIsolationRunnerError(
            f"gate plan violates its nested contract: {exc}"
        ) from exc


def _parse_attempt(value: Any) -> InheritanceAttemptV1:
    row = _exact_object(
        value,
        name="event_row",
        keys=frozenset(
            {
                "primary_eligible_parent_ids",
                "mate_eligible_parent_ids",
                "selected_parent_ids",
                "success",
                "contributing_parent_ids",
                "failure_code",
            }
        )
        if isinstance(value, Mapping) and "failure_code" in value
        else frozenset(
            {
                "primary_eligible_parent_ids",
                "mate_eligible_parent_ids",
                "selected_parent_ids",
                "success",
                "contributing_parent_ids",
            }
        ),
    )
    return InheritanceAttemptV1(
        primary_eligible_parent_ids=tuple(
            _exact_sequence(
                row["primary_eligible_parent_ids"],
                name="primary_eligible_parent_ids",
            )
        ),
        mate_eligible_parent_ids=tuple(
            _exact_sequence(
                row["mate_eligible_parent_ids"],
                name="mate_eligible_parent_ids",
            )
        ),
        selected_parent_ids=tuple(
            _exact_sequence(row["selected_parent_ids"], name="selected_parent_ids")
        ),
        success=row["success"],
        contributing_parent_ids=tuple(
            _exact_sequence(
                row["contributing_parent_ids"], name="contributing_parent_ids"
            )
        ),
        failure_code=row.get("failure_code"),
    )


def _parse_path_receipt(value: Any) -> ReproductionPathReceiptV1:
    row = _exact_object(
        value,
        name="path_receipt",
        keys=frozenset(
            {
                "schema_version",
                "path_id",
                "enabled",
                "rng_counters",
                "donor_draw_ordinals",
                "writes",
                "inheritance_event_exposure",
                "event_rows",
            }
        ),
    )
    if row["schema_version"] != PATH_RECEIPT_SCHEMA_VERSION:
        raise MixedModeIsolationRunnerError("path receipt schema_version is invalid")
    counters = []
    for item in _exact_sequence(row["rng_counters"], name="rng_counters"):
        counter = _exact_object(
            item,
            name="rng_counter",
            keys=frozenset({"counter_id", "value"}),
        )
        counters.append((counter["counter_id"], counter["value"]))
    writes = _exact_object(
        row["writes"],
        name="writes",
        keys=frozenset({"counter", "sidecar", "event_row", "rng_state"}),
    )
    attempts = tuple(
        _parse_attempt(item)
        for item in _exact_sequence(row["event_rows"], name="event_rows")
    )
    receipt = ReproductionPathReceiptV1(
        path_id=row["path_id"],
        enabled=row["enabled"],
        rng_counters=tuple(counters),
        donor_draw_ordinals=tuple(
            _exact_sequence(
                row["donor_draw_ordinals"], name="donor_draw_ordinals"
            )
        ),
        counter_writes=writes["counter"],
        sidecar_writes=writes["sidecar"],
        event_row_writes=writes["event_row"],
        rng_state_writes=writes["rng_state"],
        attempts=attempts,
    )
    if row["inheritance_event_exposure"] != receipt.exposure_payload():
        raise MixedModeIsolationRunnerError(
            "inheritance_event_exposure does not exactly match event_rows"
        )
    return receipt


def _parse_cell(value: Any) -> ResidentCellCaptureV1:
    row = _exact_object(
        value,
        name="cell",
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
                "ok",
                "fail_closed",
                "error_code",
                "path_receipts",
            }
        ),
    )
    if row["schema_version"] != CELL_SCHEMA_VERSION:
        raise MixedModeIsolationRunnerError("cell schema_version is invalid")
    return ResidentCellCaptureV1(
        occurrence_id=row["occurrence_id"],
        config=_parse_config(row["config"]),
        batch_position=row["batch_position"],
        neighbor_path_ids=tuple(
            _exact_sequence(row["neighbor_path_ids"], name="neighbor_path_ids")
        ),
        final_state_sha256=row["final_state_sha256"],
        pre_fault_state_sha256=row["pre_fault_state_sha256"],
        checkpoint_sidecar_sha256=row["checkpoint_sidecar_sha256"],
        lifecycle_state_sha256=row["lifecycle_state_sha256"],
        final_authority_sha256=row["final_authority_sha256"],
        pre_fault_authority_sha256=row["pre_fault_authority_sha256"],
        ok=row["ok"],
        fail_closed=row["fail_closed"],
        error_code=row["error_code"],
        path_receipts=tuple(
            _parse_path_receipt(item)
            for item in _exact_sequence(row["path_receipts"], name="path_receipts")
        ),
    )


def _capture_from_observation(
    request: ResidentIsolationExecutionRequestV1,
    observation: Mapping[str, Any],
) -> tuple[IsolationRunCaptureV1, str, dict[str, Any]]:
    row = _exact_object(
        observation,
        name="execution observation",
        keys=frozenset(
            {
                "schema_version",
                "runner_plan_sha256",
                "gate_plan_sha256",
                "execution_callback_id",
                "execution_authority_sha256",
                "registered_run_sha256",
                "run_id",
                "run_kind",
                "captured_at_utc",
                "fresh_owner",
                "cells",
                "execution_evidence",
                "observation_sha256",
            }
        ),
    )
    observation_sha256 = row.pop("observation_sha256")
    _digest("observation_sha256", observation_sha256)
    if _sha256(row) != observation_sha256:
        raise MixedModeIsolationRunnerError(
            "execution observation digest does not match its exact payload"
        )
    expected = {
        "schema_version": EXECUTION_OBSERVATION_SCHEMA_VERSION,
        "runner_plan_sha256": request.runner_plan_sha256,
        "gate_plan_sha256": request.gate_plan_sha256,
        "execution_callback_id": request.execution_callback_id,
        "execution_authority_sha256": request.execution_authority_sha256,
        "registered_run_sha256": request.run.digest_sha256,
        "run_id": request.run.run_id,
        "run_kind": request.run.run_kind,
        "fresh_owner": request.run.fresh_owner,
    }
    for key, expected_value in expected.items():
        if row[key] != expected_value:
            raise MixedModeIsolationRunnerError(
                f"execution observation {key} is not bound to the request"
            )
    _timestamp("captured_at_utc", row["captured_at_utc"])
    execution_evidence = row["execution_evidence"]
    if not isinstance(execution_evidence, Mapping) or not execution_evidence:
        raise MixedModeIsolationRunnerError(
            "execution_evidence must be a non-empty exact callback observation"
        )
    execution_evidence = json.loads(
        _canonical_bytes(execution_evidence).decode("ascii")
    )
    cells = tuple(
        _parse_cell(item)
        for item in _exact_sequence(row["cells"], name="cells")
    )
    registered_cells = {
        cell.occurrence_id: cell for cell in request.run.cells
    }
    captured_cells = {cell.occurrence_id: cell for cell in cells}
    if set(registered_cells) != set(captured_cells):
        raise MixedModeIsolationRunnerError(
            "captured cell set does not match preregistered occurrences"
        )
    for occurrence_id, registered in registered_cells.items():
        captured = captured_cells[occurrence_id]
        if (
            captured.config.payload() != registered.config.payload()
            or captured.batch_position != registered.batch_position
            or captured.neighbor_path_ids != registered.expected_neighbor_path_ids
        ):
            raise MixedModeIsolationRunnerError(
                f"captured cell {occurrence_id!r} does not match preregistered topology"
            )
    try:
        capture = IsolationRunCaptureV1(
            run_id=row["run_id"],
            run_kind=row["run_kind"],
            plan_sha256=row["gate_plan_sha256"],
            captured_at_utc=row["captured_at_utc"],
            fresh_owner=row["fresh_owner"],
            cells=cells,
        )
    except MixedModeIsolationContractError as exc:
        raise MixedModeIsolationRunnerError(
            f"execution observation violates the isolation capture contract: {exc}"
        ) from exc
    return capture, observation_sha256, execution_evidence


def _atomic_create_json(path: Path, value: Mapping[str, Any]) -> str:
    """Atomically create one immutable JSON artifact without replacement."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid4().hex}.tmp")
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        except BaseException:
            try:
                os.close(descriptor)
            except OSError:
                pass
            raise
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise MixedModeIsolationRunnerError(
                f"immutable artifact already exists: {path}"
            ) from exc
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class MixedModeIsolationRunnerReceiptV1:
    artifact_root: Path
    runner_plan_sha256: str
    capture_artifact_sha256s: tuple[tuple[str, str], ...]
    result_artifact_sha256: str
    gate_result: MixedModeIsolationGateResultV1


def execute_mixed_mode_resident_isolation_gate_v1(
    plan: MixedModeResidentIsolationRunnerPlanV1,
    callback: ResidentIsolationExecutionCallbackV1,
    artifact_root: Path,
) -> MixedModeIsolationRunnerReceiptV1:
    """Preregister, execute, persist, and verify one bounded isolation gate.

    ``artifact_root`` must be a new directory.  The plan is durably published
    before the first callback invocation.  Each capture and the final result
    are atomically created and can never replace prior evidence.
    """

    if type(plan) is not MixedModeResidentIsolationRunnerPlanV1:
        raise MixedModeIsolationRunnerError("plan must be a runner-plan v1 value")
    if not callable(callback):
        raise MixedModeIsolationRunnerError("callback must be callable")
    root = Path(artifact_root)
    try:
        root.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise MixedModeIsolationRunnerError(
            f"immutable artifact root already exists: {root}"
        ) from exc
    plan_artifact = {
        **plan.payload(),
        "runner_plan_sha256": plan.digest_sha256,
    }
    _atomic_create_json(root / "plan.json", plan_artifact)

    captures: list[IsolationRunCaptureV1] = []
    capture_hashes: list[tuple[str, str]] = []
    for index, registered_run in enumerate(plan.runs):
        request = ResidentIsolationExecutionRequestV1(
            runner_plan_sha256=plan.digest_sha256,
            gate_plan_sha256=plan.gate_plan.digest_sha256,
            execution_callback_id=plan.execution_callback_id,
            execution_authority_sha256=plan.execution_authority_sha256,
            run=registered_run,
        )
        try:
            observation = callback(request)
        except Exception as exc:
            raise MixedModeIsolationRunnerError(
                f"resident execution callback failed for {registered_run.run_id!r}: {exc}"
            ) from exc
        if not isinstance(observation, Mapping):
            raise MixedModeIsolationRunnerError(
                f"resident execution callback returned no object for {registered_run.run_id!r}"
            )
        capture, observation_sha256, execution_evidence = _capture_from_observation(
            request, observation
        )
        capture_payload = capture.payload()
        capture_sha256 = _sha256(capture_payload)
        artifact = {
            "schema_version": CAPTURE_ARTIFACT_SCHEMA_VERSION,
            "runner_plan_sha256": plan.digest_sha256,
            "execution_authority_sha256": plan.execution_authority_sha256,
            "registered_run_sha256": registered_run.digest_sha256,
            "observation_sha256": observation_sha256,
            "execution_evidence_sha256": _sha256(execution_evidence),
            "execution_evidence": execution_evidence,
            "capture_sha256": capture_sha256,
            "capture": capture_payload,
        }
        artifact_name = f"{index:03d}_{registered_run.digest_sha256[:16]}.json"
        artifact_sha256 = _atomic_create_json(
            root / "captures" / artifact_name, artifact
        )
        captures.append(capture)
        capture_hashes.append((registered_run.run_id, artifact_sha256))

    gate_result = verify_mixed_mode_resident_isolation_v1(
        plan.gate_plan, tuple(captures)
    )
    result = {
        "schema_version": RUNNER_RESULT_SCHEMA_VERSION,
        "runner_plan_sha256": plan.digest_sha256,
        "gate_plan_sha256": plan.gate_plan.digest_sha256,
        "execution_callback_id": plan.execution_callback_id,
        "execution_authority_sha256": plan.execution_authority_sha256,
        "capture_artifact_sha256s": [
            {"run_id": run_id, "sha256": digest}
            for run_id, digest in capture_hashes
        ],
        "gate_result": gate_result.payload(),
    }
    result_artifact_sha256 = _atomic_create_json(root / "result.json", result)
    return MixedModeIsolationRunnerReceiptV1(
        artifact_root=root,
        runner_plan_sha256=plan.digest_sha256,
        capture_artifact_sha256s=tuple(capture_hashes),
        result_artifact_sha256=result_artifact_sha256,
        gate_result=gate_result,
    )


__all__ = [
    "CAPTURE_ARTIFACT_SCHEMA_VERSION",
    "EXECUTION_OBSERVATION_SCHEMA_VERSION",
    "EXECUTION_REQUEST_SCHEMA_VERSION",
    "MixedModeIsolationRunnerError",
    "MixedModeIsolationRunnerReceiptV1",
    "MixedModeResidentIsolationRunnerPlanV1",
    "REGISTERED_CELL_SCHEMA_VERSION",
    "REGISTERED_RUN_SCHEMA_VERSION",
    "RUNNER_PLAN_SCHEMA_VERSION",
    "RUNNER_RESULT_SCHEMA_VERSION",
    "RegisteredIsolationRunV1",
    "RegisteredResidentCellV1",
    "ResidentIsolationExecutionCallbackV1",
    "ResidentIsolationExecutionRequestV1",
    "bind_resident_isolation_observation_v1",
    "execute_mixed_mode_resident_isolation_gate_v1",
]
