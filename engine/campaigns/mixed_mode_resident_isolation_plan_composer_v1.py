"""Declarative compiler for a complete mixed-mode resident isolation gate.

The compiler materializes preregistration only.  It neither loads a candidate
artifact nor dispatches simulation work.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Final, Self

from .canonical import canonical_sha256, pretty_json
from .inheritance_cross_path_mechanic_gate_adapter_v1 import (
    EXACT_FULL_STORAGE_MODE,
    RUN_PAYLOAD_SCHEMA_VERSION,
)
from .inheritance_cross_path_exposure_projection_contract_v1 import (
    AGENT_ID_ALLOCATION_SCHEMA_VERSION,
    required_reproduction_exposure_aggregate_capacity_v1,
)
from .mixed_mode_resident_isolation_gate import (
    RUN_KIND_ALONE,
    RUN_KIND_CHECKPOINT_BASELINE,
    RUN_KIND_CHECKPOINT_FRESH_OWNER,
    RUN_KIND_DUPLICATE,
    RUN_KIND_MIXED,
    RUN_KIND_OVERFLOW,
    RUN_KIND_PERMUTED,
    CheckpointCaseV1,
    FrozenMinimumInheritanceExposureV1,
    MixedModeIsolationPlanV1,
    OverflowCaseV1,
    ParentSelectionContractV1,
    ResidentReproductionCellConfigV1,
)
from .mixed_mode_resident_isolation_runner import (
    MixedModeResidentIsolationRunnerPlanV1,
    RegisteredIsolationRunV1,
    RegisteredResidentCellV1,
)
from engine.perf.cuda_architecture_b_direct_resident_source_builder_v1 import (
    QUALIFIED_TURNOVER_EPISODES_PER_GENERATION_V1,
    required_qualified_reproduction_attempt_capacity_v1,
)
CONFIG_SCHEMA_VERSION: Final[str] = (
    "eiel_mixed_mode_resident_isolation_composer_config_v1"
)
ARTIFACT_SCHEMA_VERSION: Final[str] = (
    "eiel_mixed_mode_resident_isolation_composed_plan_v1"
)
SEGMENT_REQUEST_SCHEMA_VERSION: Final[str] = (
    "eiel_mixed_mode_resident_isolation_segment_request_v1"
)
CAPACITY_PROOF_SCHEMA_VERSION: Final[str] = (
    "eiel_exact_full_exposure_capacity_proof_v6"
)

REQUIRED_GATE_CHECK_IDS: Final[tuple[str, ...]] = (
    "cell_alone_equivalence",
    "batch_permutation_invariance",
    "duplicate_cell_isolation",
    "rng_and_donor_draw_isolation",
    "overflow_isolation",
    "fresh_owner_checkpoint_isolation",
    "disabled_path_write_isolation",
    "inheritance_event_exposure",
    "minimum_live_birth_exposure",
)
REPORTING_ENABLED_RUN_ID: Final[str] = "reporting-enabled"
REPORTING_DISABLED_RUN_ID: Final[str] = "reporting-disabled"
STORAGE_PARITY_REQUEST_SCHEMA_VERSION: Final[str] = (
    "eiel_resident_exposure_storage_parity_candidate_request_v1"
)
STORAGE_PARITY_COMPARISON_SCHEMA_VERSION: Final[str] = (
    "eiel_resident_exposure_storage_parity_comparison_contract_v1"
)
STORAGE_EXACT_FULL_REQUEST_ID: Final[str] = "storage-exact-full"
STORAGE_COMPACT_COMPLETE_REQUEST_ID: Final[str] = "storage-compact-complete"
STORAGE_PARITY_FIELDS: Final[tuple[str, ...]] = (
    "attempt_outcome_parent_commit_donor_identity_exact",
    "aggregate_eligibility_equivalent",
    "final_state_exact",
    "rng_state_exact",
    "checkpoint_exact",
)
MAX_NATIVE_INT64: Final[int] = (1 << 63) - 1

_HEX: Final[frozenset[str]] = frozenset("0123456789abcdef")
_CONFIG_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "execution_callback_id",
        "execution_authority_sha256",
        "candidate_artifact_bindings",
        "preregistered_at_utc",
        "runtime_roots",
        "execution",
        "exact_full_capacities",
        "minimum_live_birth_exposure",
        "overflow",
        "live_birth_path_ids",
        "cells",
    }
)


class MixedModeIsolationPlanComposerError(ValueError):
    """A declarative gate cannot be safely or completely preregistered."""


def _fail(message: str) -> MixedModeIsolationPlanComposerError:
    return MixedModeIsolationPlanComposerError(message)


def _object(
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


def _array(value: Any, *, name: str) -> list[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise _fail(f"{name} must be an array")
    return list(value)


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


def _integer(
    value: Any,
    *,
    field: str,
    minimum: int = 0,
    maximum: int = MAX_NATIVE_INT64,
) -> int:
    if type(value) is not int or value < minimum or value > maximum:
        raise _fail(
            f"{field} must be an integer in [{minimum}, {maximum}]"
        )
    return value


def _timestamp(value: Any, *, field: str) -> datetime:
    text = _text(value, field=field)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise _fail(f"{field} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise _fail(f"{field} must include a timezone")
    return parsed.astimezone(timezone.utc)


def _absolute_path(value: Any, *, field: str) -> Path:
    path = Path(_text(value, field=field))
    if not path.is_absolute() or path.is_symlink():
        raise _fail(f"{field} must be an absolute non-symlink path")
    return path


def _safe_component(value: str, *, field: str) -> str:
    text = _text(value, field=field)
    if Path(text).name != text or text in {".", ".."}:
        raise _fail(f"{field} must be a safe path component")
    return text


@dataclass(frozen=True, slots=True)
class RepresentativeResidentCellV1:
    """One path representative with its execution mode and seed."""

    config: ResidentReproductionCellConfigV1
    mode_id: str
    seed: int

    def __post_init__(self) -> None:
        if type(self.config) is not ResidentReproductionCellConfigV1:
            raise _fail("representative config must be typed")
        _safe_component(self.mode_id, field="mode_id")
        _safe_component(
            self.config.logical_cell_id,
            field="config.logical_cell_id",
        )
        _integer(self.seed, field="seed")

    def payload(self) -> dict[str, Any]:
        return {
            "config": self.config.payload(),
            "mode_id": self.mode_id,
            "seed": self.seed,
        }

    @classmethod
    def from_payload(cls, value: Any) -> Self:
        row = _object(
            value,
            name="representative cell",
            fields=frozenset({"config", "mode_id", "seed"}),
        )
        config = _object(
            row["config"],
            name="representative cell config",
            fields=frozenset(
                {
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
        selection = _object(
            config["selection"],
            name="selection",
            fields=frozenset(
                {
                    "selection_mode",
                    "eligible_parent_source",
                    "draw_distribution",
                    "deterministic",
                    "blinded_dimensions",
                }
            ),
        )
        if type(selection["deterministic"]) is not bool:
            raise _fail("selection.deterministic must be boolean")
        try:
            typed_selection = ParentSelectionContractV1(
                selection_mode=selection["selection_mode"],
                eligible_parent_source=selection["eligible_parent_source"],
                draw_distribution=selection["draw_distribution"],
                deterministic=selection["deterministic"],
                blinded_dimensions=tuple(
                    _array(
                        selection["blinded_dimensions"],
                        name="selection.blinded_dimensions",
                    )
                ),
            )
            typed_config = ResidentReproductionCellConfigV1(
                logical_cell_id=config["logical_cell_id"],
                equivalence_key=config["equivalence_key"],
                reproduction_path=config["reproduction_path"],
                selection=typed_selection,
                parent_arity=config["parent_arity"],
                donor_draws_per_successful_birth=(
                    config["donor_draws_per_successful_birth"]
                ),
                authorized_rng_counter_ids=tuple(
                    _array(
                        config["authorized_rng_counter_ids"],
                        name="authorized_rng_counter_ids",
                    )
                ),
            )
        except ValueError as exc:
            raise _fail(f"representative cell config is invalid: {exc}") from exc
        return cls(
            config=typed_config,
            mode_id=row["mode_id"],
            seed=row["seed"],
        )


@dataclass(frozen=True, slots=True)
class ExactFullCapacityEnvelopeV1:
    """Declared A/S/C/K capacities and their conservative gate bounds."""

    attempt_capacity_a: int
    snapshot_capacity_s: int
    eligible_parent_capacity_c: int
    aggregate_capacity_k: int
    max_resident_population: int
    current_agent_id_capacity: int
    population_next_id: int
    max_births_per_tick: int
    retained_agent_id_capacity: int

    def __post_init__(self) -> None:
        for field in self.__dataclass_fields__:
            _integer(getattr(self, field), field=field, minimum=1)
        if self.attempt_capacity_a > MAX_NATIVE_INT64 // 2:
            raise _fail("2 * attempt_capacity_a exceeds native int64")
        if self.snapshot_capacity_s < 2 * self.attempt_capacity_a:
            raise _fail(
                "exact_full snapshot capacity S must be at least 2 * "
                "attempt capacity A"
            )

    def required(
        self,
        *,
        segment_generations: int,
        allocation_generation_count: int,
        ticks: int,
    ) -> dict[str, int]:
        attempts = required_qualified_reproduction_attempt_capacity_v1(
            segment_generations=segment_generations,
            ticks_per_generation=ticks,
            resident_capacity=self.max_resident_population,
        )
        if attempts > MAX_NATIVE_INT64 // 2:
            raise _fail("declared gate capacity bound exceeds native int64")
        try:
            aggregate_capacity = (
                required_reproduction_exposure_aggregate_capacity_v1(
                    self.current_agent_id_capacity,
                    self.population_next_id,
                    self.max_births_per_tick,
                    self.max_resident_population,
                    allocation_generation_count=allocation_generation_count,
                    retained_capacity=self.retained_agent_id_capacity,
                )
            )
        except (RuntimeError, ValueError) as exc:
            raise _fail(f"agent-id capacity inputs are invalid: {exc}") from exc
        return {
            "A": attempts,
            "S": 2 * attempts,
            "C": self.max_resident_population,
            "K": aggregate_capacity,
        }

    def declared(self) -> dict[str, int]:
        return {
            "A": self.attempt_capacity_a,
            "S": self.snapshot_capacity_s,
            "C": self.eligible_parent_capacity_c,
            "K": self.aggregate_capacity_k,
        }

    def proof(
        self,
        *,
        generations: int,
        ticks: int,
        checkpoint_split_generation: int,
        maximum_physical_batch: int,
    ) -> dict[str, Any]:
        maximum_segment_generations = max(
            checkpoint_split_generation,
            generations - checkpoint_split_generation,
        )
        required = self.required(
            segment_generations=maximum_segment_generations,
            allocation_generation_count=generations,
            ticks=ticks,
        )
        declared = self.declared()
        insufficient = {
            name: {"required": required[name], "declared": declared[name]}
            for name in required
            if declared[name] < required[name]
        }
        if insufficient:
            raise _fail(
                "exact_full capacities do not bound the declared gate: "
                f"{insufficient}"
            )
        payload = {
            "schema_version": CAPACITY_PROOF_SCHEMA_VERSION,
            "storage_mode": EXACT_FULL_STORAGE_MODE,
            "bound_model": (
                "checkpoint_drained_attempt_and_compact_key_horizon_bound_v6"
            ),
            "ordinary_attempts_per_tick_bound": self.max_resident_population,
            "turnover_attempts_per_generation_bound": min(
                QUALIFIED_TURNOVER_EPISODES_PER_GENERATION_V1,
                self.max_resident_population,
            ),
            "generations": generations,
            "ticks_per_generation": ticks,
            "checkpoint_split_generation": checkpoint_split_generation,
            "maximum_segment_generations": maximum_segment_generations,
            "max_resident_population": self.max_resident_population,
            "agent_id_allocation": {
                "schema_version": AGENT_ID_ALLOCATION_SCHEMA_VERSION,
                "current_capacity": self.current_agent_id_capacity,
                "population_next_id": self.population_next_id,
                "max_births_per_tick": self.max_births_per_tick,
                "slot_capacity": self.max_resident_population,
                "allocation_generation_count": generations,
                "retained_capacity": self.retained_agent_id_capacity,
                "required_aggregate_capacity": required["K"],
            },
            "maximum_physical_batch": maximum_physical_batch,
            "required_per_cell": required,
            "declared_per_cell": declared,
            "bounded": True,
        }
        return {**payload, "capacity_proof_sha256": canonical_sha256(payload)}


@dataclass(frozen=True, slots=True)
class MixedModeIsolationComposerConfigV1:
    execution_callback_id: str
    execution_authority_sha256: str
    exact_full_candidate_authority_sha256: str
    exact_full_artifact_sha256: str
    compact_complete_candidate_authority_sha256: str
    compact_complete_artifact_sha256: str
    preregistered_at_utc: str
    runs_root: Path
    owners_root: Path
    checkpoints_root: Path
    generations: int
    ticks_per_generation: int
    checkpoint_split_generation: int
    capacities: ExactFullCapacityEnvelopeV1
    threshold_id: str
    threshold_metric: str
    threshold_minimum: int
    threshold_frozen_at_utc: str
    overflow_target_path_id: str
    overflow_error_code: str
    overflow_capacity_kind: str
    overflow_effective_bound: int
    overflow_expected_status_code: int
    live_birth_path_ids: tuple[str, ...]
    cells: tuple[RepresentativeResidentCellV1, ...]

    def __post_init__(self) -> None:
        _text(self.execution_callback_id, field="execution_callback_id")
        _digest(
            self.execution_authority_sha256,
            field="execution_authority_sha256",
        )
        for field in (
            "exact_full_candidate_authority_sha256",
            "exact_full_artifact_sha256",
            "compact_complete_candidate_authority_sha256",
            "compact_complete_artifact_sha256",
        ):
            _digest(getattr(self, field), field=field)
        if (
            self.exact_full_candidate_authority_sha256
            == self.compact_complete_candidate_authority_sha256
            or self.exact_full_artifact_sha256
            == self.compact_complete_artifact_sha256
        ):
            raise _fail(
                "exact_full and compact_complete candidate authorities and "
                "artifact hashes must be distinct"
            )
        registered_at = _timestamp(
            self.preregistered_at_utc,
            field="preregistered_at_utc",
        )
        frozen_at = _timestamp(
            self.threshold_frozen_at_utc,
            field="threshold_frozen_at_utc",
        )
        if registered_at < frozen_at:
            raise _fail(
                "preregistration must not predate the frozen exposure threshold"
            )
        roots = (self.runs_root, self.owners_root, self.checkpoints_root)
        for field, root in zip(
            ("runs_root", "owners_root", "checkpoints_root"),
            roots,
            strict=True,
        ):
            _absolute_path(str(root), field=field)
        if len(set(roots)) != len(roots):
            raise _fail("runtime roots must be distinct")
        _integer(self.generations, field="generations", minimum=2)
        _integer(
            self.ticks_per_generation,
            field="ticks_per_generation",
            minimum=1,
        )
        _integer(
            self.checkpoint_split_generation,
            field="checkpoint_split_generation",
            minimum=1,
        )
        if self.checkpoint_split_generation >= self.generations:
            raise _fail("checkpoint split must be interior to the run")
        if type(self.capacities) is not ExactFullCapacityEnvelopeV1:
            raise _fail("capacities must be ExactFullCapacityEnvelopeV1")
        _text(self.threshold_id, field="threshold_id")
        _integer(self.threshold_minimum, field="threshold_minimum", minimum=1)
        _text(self.overflow_error_code, field="overflow_error_code")
        _integer(
            self.overflow_effective_bound,
            field="overflow_effective_bound",
        )
        if self.overflow_effective_bound != 0:
            raise _fail(
                "attempt-capacity fault qualification requires the exact "
                "zero-capacity fail-closed boundary"
            )
        _integer(
            self.overflow_expected_status_code,
            field="overflow_expected_status_code",
            minimum=1,
        )
        if self.overflow_capacity_kind not in {
            "attempt",
            "snapshot",
            "eligible_parent",
            "aggregate",
        }:
            raise _fail("overflow capacity kind is unsupported")
        if not self.cells or any(
            type(cell) is not RepresentativeResidentCellV1
            for cell in self.cells
        ):
            raise _fail("cells must contain typed representatives")
        if len(self.cells) < 2:
            raise _fail("mixed-mode gate requires at least two cells")
        paths = tuple(cell.config.reproduction_path for cell in self.cells)
        logical_ids = tuple(cell.config.logical_cell_id for cell in self.cells)
        equivalence_keys = tuple(
            cell.config.equivalence_key for cell in self.cells
        )
        if (
            len(set(paths)) != len(paths)
            or len(set(logical_ids)) != len(logical_ids)
            or len(set(equivalence_keys)) != len(equivalence_keys)
        ):
            raise _fail(
                "representative paths, logical cell ids, and equivalence keys "
                "must be unique"
            )
        required_blinding = {"genotype", "lineage", "policy", "strategy"}
        for cell in self.cells:
            selection = cell.config.selection
            if (
                cell.config.parent_arity != 2
                or selection.eligible_parent_source
                != "authoritative_eligible_parent_set_v1"
                or selection.draw_distribution != "uniform_v1"
                or not selection.deterministic
                or not required_blinding.issubset(
                    selection.blinded_dimensions
                )
            ):
                raise _fail(
                    "representative selection must satisfy the frozen neutral "
                    "two-parent gate contract"
                )
        if set(self.live_birth_path_ids) - set(paths):
            raise _fail("live-birth paths must be represented")
        if self.overflow_target_path_id not in self.live_birth_path_ids:
            raise _fail("overflow target must be a represented live-birth path")
        nominal = {
            "attempt": self.capacities.attempt_capacity_a,
            "snapshot": self.capacities.snapshot_capacity_s,
            "eligible_parent": self.capacities.eligible_parent_capacity_c,
            "aggregate": self.capacities.aggregate_capacity_k,
        }[self.overflow_capacity_kind]
        if self.overflow_effective_bound >= nominal:
            raise _fail("overflow effective bound must be below nominal capacity")
        required_attempts = (
            self.capacities.max_resident_population
            * self.generations
            * self.ticks_per_generation
        )
        maximum_threshold = {
            "attempts": required_attempts,
            "successful_births": required_attempts,
            "parent_contributions": (
                required_attempts
                * max(cell.config.parent_arity for cell in self.cells)
            ),
        }.get(self.threshold_metric)
        if (
            maximum_threshold is None
            or self.threshold_minimum > maximum_threshold
        ):
            raise _fail("minimum exposure is impossible under declared bounds")
        self.capacities.proof(
            generations=self.generations,
            ticks=self.ticks_per_generation,
            checkpoint_split_generation=self.checkpoint_split_generation,
            maximum_physical_batch=max(len(self.cells), 2),
        )

    @property
    def path_ids(self) -> tuple[str, ...]:
        return tuple(cell.config.reproduction_path for cell in self.cells)

    def payload(self) -> dict[str, Any]:
        return {
            "schema_version": CONFIG_SCHEMA_VERSION,
            "execution_callback_id": self.execution_callback_id,
            "execution_authority_sha256": self.execution_authority_sha256,
            "candidate_artifact_bindings": {
                "exact_full": {
                    "candidate_authority_sha256": (
                        self.exact_full_candidate_authority_sha256
                    ),
                    "artifact_sha256": self.exact_full_artifact_sha256,
                },
                "compact_complete": {
                    "candidate_authority_sha256": (
                        self.compact_complete_candidate_authority_sha256
                    ),
                    "artifact_sha256": (
                        self.compact_complete_artifact_sha256
                    ),
                },
            },
            "preregistered_at_utc": self.preregistered_at_utc,
            "runtime_roots": {
                "runs": str(self.runs_root),
                "owners": str(self.owners_root),
                "checkpoints": str(self.checkpoints_root),
            },
            "execution": {
                "generations": self.generations,
                "ticks_per_generation": self.ticks_per_generation,
                "checkpoint_split_generation": (
                    self.checkpoint_split_generation
                ),
            },
            "exact_full_capacities": {
                "A": self.capacities.attempt_capacity_a,
                "S": self.capacities.snapshot_capacity_s,
                "C": self.capacities.eligible_parent_capacity_c,
                "K": self.capacities.aggregate_capacity_k,
                "max_resident_population": (
                    self.capacities.max_resident_population
                ),
                "current_agent_id_capacity": (
                    self.capacities.current_agent_id_capacity
                ),
                "population_next_id": self.capacities.population_next_id,
                "max_births_per_tick": self.capacities.max_births_per_tick,
                "retained_agent_id_capacity": (
                    self.capacities.retained_agent_id_capacity
                ),
            },
            "minimum_live_birth_exposure": {
                "threshold_id": self.threshold_id,
                "metric": self.threshold_metric,
                "minimum": self.threshold_minimum,
                "frozen_at_utc": self.threshold_frozen_at_utc,
            },
            "overflow": {
                "target_path_id": self.overflow_target_path_id,
                "error_code": self.overflow_error_code,
                "capacity_kind": self.overflow_capacity_kind,
                "effective_bound": self.overflow_effective_bound,
                "expected_status_code": self.overflow_expected_status_code,
            },
            "live_birth_path_ids": list(self.live_birth_path_ids),
            "cells": [cell.payload() for cell in self.cells],
        }

    @classmethod
    def from_payload(cls, value: Any) -> Self:
        row = _object(
            value,
            name="composer config",
            fields=_CONFIG_FIELDS,
        )
        if row["schema_version"] != CONFIG_SCHEMA_VERSION:
            raise _fail("composer config schema_version is unsupported")
        candidate_bindings = _object(
            row["candidate_artifact_bindings"],
            name="candidate_artifact_bindings",
            fields=frozenset({"exact_full", "compact_complete"}),
        )
        exact_binding = _object(
            candidate_bindings["exact_full"],
            name="candidate_artifact_bindings.exact_full",
            fields=frozenset(
                {"candidate_authority_sha256", "artifact_sha256"}
            ),
        )
        compact_binding = _object(
            candidate_bindings["compact_complete"],
            name="candidate_artifact_bindings.compact_complete",
            fields=frozenset(
                {"candidate_authority_sha256", "artifact_sha256"}
            ),
        )
        roots = _object(
            row["runtime_roots"],
            name="runtime_roots",
            fields=frozenset({"runs", "owners", "checkpoints"}),
        )
        execution = _object(
            row["execution"],
            name="execution",
            fields=frozenset(
                {
                    "generations",
                    "ticks_per_generation",
                    "checkpoint_split_generation",
                }
            ),
        )
        capacities = _object(
            row["exact_full_capacities"],
            name="exact_full_capacities",
            fields=frozenset(
                {
                    "A",
                    "S",
                    "C",
                    "K",
                    "max_resident_population",
                    "current_agent_id_capacity",
                    "population_next_id",
                    "max_births_per_tick",
                    "retained_agent_id_capacity",
                }
            ),
        )
        threshold = _object(
            row["minimum_live_birth_exposure"],
            name="minimum_live_birth_exposure",
            fields=frozenset(
                {"threshold_id", "metric", "minimum", "frozen_at_utc"}
            ),
        )
        overflow = _object(
            row["overflow"],
            name="overflow",
            fields=frozenset(
                {
                    "target_path_id",
                    "error_code",
                    "capacity_kind",
                    "effective_bound",
                    "expected_status_code",
                }
            ),
        )
        return cls(
            execution_callback_id=row["execution_callback_id"],
            execution_authority_sha256=row["execution_authority_sha256"],
            exact_full_candidate_authority_sha256=exact_binding[
                "candidate_authority_sha256"
            ],
            exact_full_artifact_sha256=exact_binding["artifact_sha256"],
            compact_complete_candidate_authority_sha256=compact_binding[
                "candidate_authority_sha256"
            ],
            compact_complete_artifact_sha256=compact_binding[
                "artifact_sha256"
            ],
            preregistered_at_utc=row["preregistered_at_utc"],
            runs_root=_absolute_path(roots["runs"], field="runtime_roots.runs"),
            owners_root=_absolute_path(
                roots["owners"], field="runtime_roots.owners"
            ),
            checkpoints_root=_absolute_path(
                roots["checkpoints"], field="runtime_roots.checkpoints"
            ),
            generations=execution["generations"],
            ticks_per_generation=execution["ticks_per_generation"],
            checkpoint_split_generation=execution[
                "checkpoint_split_generation"
            ],
            capacities=ExactFullCapacityEnvelopeV1(
                attempt_capacity_a=capacities["A"],
                snapshot_capacity_s=capacities["S"],
                eligible_parent_capacity_c=capacities["C"],
                aggregate_capacity_k=capacities["K"],
                max_resident_population=capacities[
                    "max_resident_population"
                ],
                current_agent_id_capacity=capacities[
                    "current_agent_id_capacity"
                ],
                population_next_id=capacities["population_next_id"],
                max_births_per_tick=capacities["max_births_per_tick"],
                retained_agent_id_capacity=capacities[
                    "retained_agent_id_capacity"
                ],
            ),
            threshold_id=threshold["threshold_id"],
            threshold_metric=threshold["metric"],
            threshold_minimum=threshold["minimum"],
            threshold_frozen_at_utc=threshold["frozen_at_utc"],
            overflow_target_path_id=overflow["target_path_id"],
            overflow_error_code=overflow["error_code"],
            overflow_capacity_kind=overflow["capacity_kind"],
            overflow_effective_bound=overflow["effective_bound"],
            overflow_expected_status_code=overflow["expected_status_code"],
            live_birth_path_ids=tuple(
                _array(
                    row["live_birth_path_ids"],
                    name="live_birth_path_ids",
                )
            ),
            cells=tuple(
                RepresentativeResidentCellV1.from_payload(cell)
                for cell in _array(row["cells"], name="cells")
            ),
        )


def _registered_cell(
    cell: RepresentativeResidentCellV1,
    *,
    occurrence_id: str,
    batch_position: int,
    neighbor_path_ids: tuple[str, ...],
) -> RegisteredResidentCellV1:
    return RegisteredResidentCellV1(
        occurrence_id=occurrence_id,
        config=cell.config,
        batch_position=batch_position,
        expected_neighbor_path_ids=neighbor_path_ids,
    )


def _base_cells(
    config: MixedModeIsolationComposerConfigV1,
    *,
    order: Sequence[int],
    alone: bool = False,
) -> tuple[RegisteredResidentCellV1, ...]:
    result = []
    for position, index in enumerate(order):
        representative = config.cells[index]
        neighbors = (
            ()
            if alone
            else tuple(
                config.cells[other].config.reproduction_path
                for other in order
                if other != index
            )
        )
        result.append(
            _registered_cell(
                representative,
                occurrence_id=representative.config.logical_cell_id,
                batch_position=position,
                neighbor_path_ids=neighbors,
            )
        )
    return tuple(result)


def _duplicate_cells(
    config: MixedModeIsolationComposerConfigV1,
    *,
    duplicate_index: int,
) -> tuple[RegisteredResidentCellV1, ...]:
    target = config.cells[duplicate_index]
    ordered = (target, target)
    result = []
    for position, representative in enumerate(ordered):
        occurrence_id = representative.config.logical_cell_id
        if representative is target:
            occurrence_id = f"{occurrence_id}-duplicate-{position + 1}"
        neighbor_path_ids = tuple(
            ordered[neighbor].config.reproduction_path
            for neighbor in (position - 1, position + 1)
            if 0 <= neighbor < len(ordered)
        )
        result.append(
            _registered_cell(
                representative,
                occurrence_id=occurrence_id,
                batch_position=position,
                neighbor_path_ids=neighbor_path_ids,
            )
        )
    return tuple(result)


def _segment_request(
    config: MixedModeIsolationComposerConfigV1,
    *,
    run_id: str,
    cells: tuple[RegisteredResidentCellV1, ...],
    reporting_enabled: bool,
    checkpoint_role: str,
    checkpoint_source_run_id: str | None,
    overflow_control: Mapping[str, Any] | None,
) -> dict[str, Any]:
    run_root = config.runs_root / run_id
    owner_root = config.owners_root / run_id
    checkpoint_root = config.checkpoints_root / run_id
    payload = {
        "schema_version": SEGMENT_REQUEST_SCHEMA_VERSION,
        "candidate_artifact_binding_sha256": (
            config.exact_full_artifact_sha256
        ),
        "run_id": run_id,
        "run_root": str(run_root),
        "owner_root": str(owner_root),
        "checkpoint_root": str(checkpoint_root),
        "checkpoint_role": checkpoint_role,
        "checkpoint_source_run_id": checkpoint_source_run_id,
        "checkpoint_source_root": (
            None
            if checkpoint_source_run_id is None
            else str(config.checkpoints_root / checkpoint_source_run_id)
        ),
        "overflow_control": (
            None if overflow_control is None else dict(overflow_control)
        ),
        "generations": config.generations,
        "ticks_per_generation": config.ticks_per_generation,
        "checkpoint_split_generation": config.checkpoint_split_generation,
        "reporting_enabled": reporting_enabled,
        "exposure": {
            "storage_mode": EXACT_FULL_STORAGE_MODE,
            "A": config.capacities.attempt_capacity_a,
            "S": config.capacities.snapshot_capacity_s,
            "C": config.capacities.eligible_parent_capacity_c,
            "K": config.capacities.aggregate_capacity_k,
        },
        "physical_order": [
            {
                "occurrence_id": cell.occurrence_id,
                "logical_cell_id": cell.config.logical_cell_id,
                "batch_position": cell.batch_position,
                "path_id": cell.config.reproduction_path,
                "mode_id": next(
                    representative.mode_id
                    for representative in config.cells
                    if representative.config.logical_cell_id
                    == cell.config.logical_cell_id
                ),
                "seed": next(
                    representative.seed
                    for representative in config.cells
                    if representative.config.logical_cell_id
                    == cell.config.logical_cell_id
                ),
                "neighbor_path_ids": list(cell.expected_neighbor_path_ids),
            }
            for cell in cells
        ],
    }
    return {**payload, "segment_request_sha256": canonical_sha256(payload)}


def _registered_run(
    config: MixedModeIsolationComposerConfigV1,
    *,
    run_id: str,
    run_kind: str,
    cells: tuple[RegisteredResidentCellV1, ...],
    reporting_enabled: bool,
    fresh_owner: bool = False,
    overflow_fault: Mapping[str, Any] | None = None,
    checkpoint_role: str = "none",
    checkpoint_source_run_id: str | None = None,
) -> RegisteredIsolationRunV1:
    return RegisteredIsolationRunV1(
        run_id=run_id,
        run_kind=run_kind,
        fresh_owner=fresh_owner,
        cells=cells,
        execution_payload={
            "schema_version": RUN_PAYLOAD_SCHEMA_VERSION,
            "candidate_authority_sha256": (
                config.exact_full_candidate_authority_sha256
            ),
            "exposure_storage_mode": EXACT_FULL_STORAGE_MODE,
            "reporting_enabled": reporting_enabled,
            "overflow_fault": (
                None
                if overflow_fault is None
                else {
                    "target_occurrence_id": overflow_fault[
                        "target_occurrence_id"
                    ],
                    "error_code": overflow_fault["error_code"],
                }
            ),
            "segment_request": _segment_request(
                config,
                run_id=run_id,
                cells=cells,
                reporting_enabled=reporting_enabled,
                checkpoint_role=checkpoint_role,
                checkpoint_source_run_id=checkpoint_source_run_id,
                overflow_control=overflow_fault,
            ),
        },
    )


def _auxiliary_storage_parity_request(
    config: MixedModeIsolationComposerConfigV1,
    *,
    request_id: str,
    storage_mode: str,
    cells: tuple[RegisteredResidentCellV1, ...],
) -> dict[str, Any]:
    """Build a non-isolation-callback request for a later parity runner."""

    if storage_mode == EXACT_FULL_STORAGE_MODE:
        candidate_authority = (
            config.exact_full_candidate_authority_sha256
        )
        artifact_sha256 = config.exact_full_artifact_sha256
        snapshot_capacity = config.capacities.snapshot_capacity_s
        eligibility_representation = (
            "role_specific_per_attempt_parent_id_sets_v1"
        )
    elif storage_mode == "compact_complete":
        candidate_authority = (
            config.compact_complete_candidate_authority_sha256
        )
        artifact_sha256 = config.compact_complete_artifact_sha256
        snapshot_capacity = 1
        eligibility_representation = (
            "role_specific_aggregate_opportunity_counts_v1"
        )
    else:
        raise _fail("auxiliary storage parity mode is unsupported")
    root_name = f"auxiliary-{request_id}"
    payload = {
        "schema_version": STORAGE_PARITY_REQUEST_SCHEMA_VERSION,
        "request_id": request_id,
        "execution_route": "auxiliary_storage_parity_only",
        "isolation_callback_authorized": False,
        "candidate_authority_sha256": candidate_authority,
        "candidate_artifact_sha256": artifact_sha256,
        "storage_mode": storage_mode,
        "run_root": str(config.runs_root / root_name),
        "owner_root": str(config.owners_root / root_name),
        "checkpoint_root": str(config.checkpoints_root / root_name),
        "generations": config.generations,
        "ticks_per_generation": config.ticks_per_generation,
        "checkpoint_split_generation": config.checkpoint_split_generation,
        "reporting_enabled": True,
        "exposure": {
            "A": config.capacities.attempt_capacity_a,
            "S": snapshot_capacity,
            "C": config.capacities.eligible_parent_capacity_c,
            "K": config.capacities.aggregate_capacity_k,
            "eligibility_representation": eligibility_representation,
            "per_attempt_eligible_parent_id_sets_available": (
                storage_mode == EXACT_FULL_STORAGE_MODE
            ),
        },
        "physical_order": [
            {
                "occurrence_id": cell.occurrence_id,
                "batch_position": cell.batch_position,
                "config": cell.config.payload(),
                "mode_id": next(
                    representative.mode_id
                    for representative in config.cells
                    if representative.config.logical_cell_id
                    == cell.config.logical_cell_id
                ),
                "seed": next(
                    representative.seed
                    for representative in config.cells
                    if representative.config.logical_cell_id
                    == cell.config.logical_cell_id
                ),
            }
            for cell in cells
        ],
    }
    return {**payload, "request_sha256": canonical_sha256(payload)}


def _storage_parity_preregistration(
    config: MixedModeIsolationComposerConfigV1,
    *,
    cells: tuple[RegisteredResidentCellV1, ...],
) -> dict[str, Any]:
    exact = _auxiliary_storage_parity_request(
        config,
        request_id=STORAGE_EXACT_FULL_REQUEST_ID,
        storage_mode=EXACT_FULL_STORAGE_MODE,
        cells=cells,
    )
    compact = _auxiliary_storage_parity_request(
        config,
        request_id=STORAGE_COMPACT_COMPLETE_REQUEST_ID,
        storage_mode="compact_complete",
        cells=cells,
    )
    comparison = {
        "schema_version": STORAGE_PARITY_COMPARISON_SCHEMA_VERSION,
        "exact_full_request_sha256": exact["request_sha256"],
        "compact_complete_request_sha256": compact["request_sha256"],
        "required_comparisons": list(STORAGE_PARITY_FIELDS),
        "exact_full_eligible_parent_id_sets_authoritative": True,
        "compact_per_attempt_eligible_parent_id_sets_forbidden": True,
        "compact_eligibility_comparison": (
            "role_specific_aggregate_opportunity_counts_v1"
        ),
        "promotion_authority": False,
    }
    return {
        "exact_full_request": exact,
        "compact_complete_request": compact,
        "comparison_contract": {
            **comparison,
            "comparison_contract_sha256": canonical_sha256(comparison),
        },
    }


def _verify_auxiliary_storage_request(
    value: Any,
    *,
    name: str,
    expected_request_id: str,
    expected_storage_mode: str,
) -> dict[str, Any]:
    fields = frozenset(
        {
            "schema_version",
            "request_id",
            "execution_route",
            "isolation_callback_authorized",
            "candidate_authority_sha256",
            "candidate_artifact_sha256",
            "storage_mode",
            "run_root",
            "owner_root",
            "checkpoint_root",
            "generations",
            "ticks_per_generation",
            "checkpoint_split_generation",
            "reporting_enabled",
            "exposure",
            "physical_order",
            "request_sha256",
        }
    )
    row = _object(value, name=name, fields=fields)
    supplied = _digest(row.pop("request_sha256"), field=f"{name}.request_sha256")
    if canonical_sha256(row) != supplied:
        raise _fail(f"{name} request_sha256 does not bind its request")
    if (
        row["schema_version"] != STORAGE_PARITY_REQUEST_SCHEMA_VERSION
        or row["request_id"] != expected_request_id
        or row["execution_route"] != "auxiliary_storage_parity_only"
        or row["isolation_callback_authorized"] is not False
        or row["storage_mode"] != expected_storage_mode
    ):
        raise _fail(f"{name} is not an auxiliary {expected_storage_mode} request")
    _digest(
        row["candidate_authority_sha256"],
        field=f"{name}.candidate_authority_sha256",
    )
    _digest(
        row["candidate_artifact_sha256"],
        field=f"{name}.candidate_artifact_sha256",
    )
    exposure = row["exposure"]
    if not isinstance(exposure, Mapping):
        raise _fail(f"{name}.exposure must be an object")
    has_ids = exposure.get("per_attempt_eligible_parent_id_sets_available")
    if (
        expected_storage_mode == "compact_complete"
        and (
            has_ids is not False
            or exposure.get("eligibility_representation")
            != "role_specific_aggregate_opportunity_counts_v1"
            or exposure.get("S") != 1
        )
    ):
        raise _fail(
            "compact auxiliary request cannot claim per-attempt eligible sets"
        )
    if (
        expected_storage_mode == EXACT_FULL_STORAGE_MODE
        and (
            has_ids is not True
            or exposure.get("eligibility_representation")
            != "role_specific_per_attempt_parent_id_sets_v1"
        )
    ):
        raise _fail("exact_full auxiliary request lacks exact eligibility")
    return {**row, "request_sha256": supplied}


def compose_mixed_mode_resident_isolation_plan_v1(
    config: MixedModeIsolationComposerConfigV1,
) -> dict[str, Any]:
    """Compile and digest a deterministic complete runner preregistration."""

    if type(config) is not MixedModeIsolationComposerConfigV1:
        raise _fail("config must be MixedModeIsolationComposerConfigV1")
    indexes = tuple(range(len(config.cells)))
    mixed_cells = _base_cells(config, order=indexes)
    permutation_cells = _base_cells(config, order=tuple(reversed(indexes)))
    duplicate_runs = tuple(
        (
            f"mixed-duplicate-{representative.config.logical_cell_id}",
            _duplicate_cells(config, duplicate_index=index),
        )
        for index, representative in enumerate(config.cells)
    )
    target_occurrence_id = next(
        cell.occurrence_id
        for cell in mixed_cells
        if cell.config.reproduction_path == config.overflow_target_path_id
    )
    mixed_id = "mixed-reference"
    permutation_id = "mixed-permutation-reverse"
    overflow_reference_id = "overflow-reference"
    overflow_fault_id = "overflow-fault"
    checkpoint_baseline_id = "checkpoint-uninterrupted"
    checkpoint_fresh_id = "checkpoint-fresh-owner"
    overflow_control = {
        "target_occurrence_id": target_occurrence_id,
        "error_code": config.overflow_error_code,
        "capacity_kind": config.overflow_capacity_kind,
        "effective_bound": config.overflow_effective_bound,
        "expected_status_code": config.overflow_expected_status_code,
    }
    gate_plan = MixedModeIsolationPlanV1(
        path_ids=config.path_ids,
        mixed_reference_run_id=mixed_id,
        permutation_run_ids=(permutation_id,),
        duplicate_run_ids=tuple(run_id for run_id, _cells in duplicate_runs),
        duplicate_required_path_ids=config.path_ids,
        overflow_cases=(
            OverflowCaseV1(
                reference_run_id=overflow_reference_id,
                fault_run_id=overflow_fault_id,
                target_occurrence_id=target_occurrence_id,
                overflow_error_code=config.overflow_error_code,
            ),
        ),
        checkpoint_cases=(
            CheckpointCaseV1(
                uninterrupted_run_id=checkpoint_baseline_id,
                fresh_owner_run_id=checkpoint_fresh_id,
            ),
        ),
        minimum_live_birth_exposure=FrozenMinimumInheritanceExposureV1(
            threshold_id=config.threshold_id,
            metric=config.threshold_metric,
            minimum=config.threshold_minimum,
            frozen_at_utc=config.threshold_frozen_at_utc,
        ),
        live_birth_path_ids=config.live_birth_path_ids,
    )
    runs: list[RegisteredIsolationRunV1] = [
        _registered_run(
            config,
            run_id=mixed_id,
            run_kind=RUN_KIND_MIXED,
            cells=mixed_cells,
            reporting_enabled=True,
        )
    ]
    for representative in config.cells:
        alone_id = f"alone-{representative.config.logical_cell_id}"
        runs.append(
            _registered_run(
                config,
                run_id=alone_id,
                run_kind=RUN_KIND_ALONE,
                cells=_base_cells(
                    config,
                    order=(config.cells.index(representative),),
                    alone=True,
                ),
                reporting_enabled=True,
            )
        )
    runs.extend(
        (
            _registered_run(
                config,
                run_id=permutation_id,
                run_kind=RUN_KIND_PERMUTED,
                cells=permutation_cells,
                reporting_enabled=True,
            ),
            *(
                _registered_run(
                    config,
                    run_id=run_id,
                    run_kind=RUN_KIND_DUPLICATE,
                    cells=cells,
                    reporting_enabled=True,
                )
                for run_id, cells in duplicate_runs
            ),
            _registered_run(
                config,
                run_id=overflow_reference_id,
                run_kind=RUN_KIND_MIXED,
                cells=mixed_cells,
                reporting_enabled=True,
            ),
            _registered_run(
                config,
                run_id=overflow_fault_id,
                run_kind=RUN_KIND_OVERFLOW,
                cells=mixed_cells,
                reporting_enabled=True,
                overflow_fault=overflow_control,
            ),
            _registered_run(
                config,
                run_id=checkpoint_baseline_id,
                run_kind=RUN_KIND_CHECKPOINT_BASELINE,
                cells=mixed_cells,
                reporting_enabled=True,
                checkpoint_role="uninterrupted_baseline",
            ),
            _registered_run(
                config,
                run_id=checkpoint_fresh_id,
                run_kind=RUN_KIND_CHECKPOINT_FRESH_OWNER,
                cells=mixed_cells,
                reporting_enabled=True,
                fresh_owner=True,
                checkpoint_role="fresh_owner_continuation",
                checkpoint_source_run_id=checkpoint_baseline_id,
            ),
            _registered_run(
                config,
                run_id=REPORTING_ENABLED_RUN_ID,
                run_kind=RUN_KIND_MIXED,
                cells=mixed_cells,
                reporting_enabled=True,
            ),
            _registered_run(
                config,
                run_id=REPORTING_DISABLED_RUN_ID,
                run_kind=RUN_KIND_MIXED,
                cells=mixed_cells,
                reporting_enabled=False,
            ),
        )
    )
    runner_plan = MixedModeResidentIsolationRunnerPlanV1(
        gate_plan=gate_plan,
        execution_callback_id=config.execution_callback_id,
        execution_authority_sha256=config.execution_authority_sha256,
        preregistered_at_utc=config.preregistered_at_utc,
        runs=tuple(runs),
    )
    capacity_proof = config.capacities.proof(
        generations=config.generations,
        ticks=config.ticks_per_generation,
        checkpoint_split_generation=config.checkpoint_split_generation,
        maximum_physical_batch=max(len(config.cells), 2),
    )
    artifact = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "config_sha256": canonical_sha256(config.payload()),
        "runner_plan": runner_plan.payload(),
        "runner_plan_sha256": runner_plan.digest_sha256,
        "capacity_proof": capacity_proof,
        "required_gate_check_ids": list(REQUIRED_GATE_CHECK_IDS),
        "reporting_parity_run_ids": [
            REPORTING_ENABLED_RUN_ID,
            REPORTING_DISABLED_RUN_ID,
        ],
        "storage_mode_parity": _storage_parity_preregistration(
            config,
            cells=mixed_cells,
        ),
    }
    return {
        **artifact,
        "artifact_sha256": canonical_sha256(artifact),
    }


def verify_composed_plan_artifact_v1(value: Any) -> dict[str, Any]:
    row = _object(
        value,
        name="composed plan artifact",
        fields=frozenset(
            {
                "schema_version",
                "config_sha256",
                "runner_plan",
                "runner_plan_sha256",
                "capacity_proof",
                "required_gate_check_ids",
                "reporting_parity_run_ids",
                "storage_mode_parity",
                "artifact_sha256",
            }
        ),
    )
    if row["schema_version"] != ARTIFACT_SCHEMA_VERSION:
        raise _fail("composed plan artifact schema_version is unsupported")
    supplied = _digest(row.pop("artifact_sha256"), field="artifact_sha256")
    if canonical_sha256(row) != supplied:
        raise _fail("artifact_sha256 does not bind the composed plan")
    _digest(row["config_sha256"], field="config_sha256")
    runner_plan = row["runner_plan"]
    if not isinstance(runner_plan, Mapping):
        raise _fail("runner_plan must be an object")
    _digest(row["runner_plan_sha256"], field="runner_plan_sha256")
    if canonical_sha256(runner_plan) != row["runner_plan_sha256"]:
        raise _fail("runner_plan_sha256 does not bind runner_plan")
    gate_plan = runner_plan.get("gate_plan")
    gate_plan_sha256 = runner_plan.get("gate_plan_sha256")
    if (
        not isinstance(gate_plan, Mapping)
        or canonical_sha256(gate_plan) != gate_plan_sha256
    ):
        raise _fail("gate_plan_sha256 does not bind gate_plan")
    proof = _object(
        row["capacity_proof"],
        name="capacity_proof",
        fields=frozenset(
            {
                "schema_version",
                "storage_mode",
                "bound_model",
                "ordinary_attempts_per_tick_bound",
                "turnover_attempts_per_generation_bound",
                "generations",
                "ticks_per_generation",
                "checkpoint_split_generation",
                "maximum_segment_generations",
                "max_resident_population",
                "agent_id_allocation",
                "maximum_physical_batch",
                "required_per_cell",
                "declared_per_cell",
                "bounded",
                "capacity_proof_sha256",
            }
        ),
    )
    supplied_proof = dict(proof)
    proof_digest = _digest(
        proof.pop("capacity_proof_sha256"),
        field="capacity_proof_sha256",
    )
    if canonical_sha256(proof) != proof_digest or proof["bounded"] is not True:
        raise _fail("capacity proof is not canonically bound and successful")
    required = _object(
        proof["required_per_cell"],
        name="capacity_proof.required_per_cell",
        fields=frozenset({"A", "S", "C", "K"}),
    )
    declared = _object(
        proof["declared_per_cell"],
        name="capacity_proof.declared_per_cell",
        fields=frozenset({"A", "S", "C", "K"}),
    )
    allocation = _object(
        proof["agent_id_allocation"],
        name="capacity_proof.agent_id_allocation",
        fields=frozenset(
            {
                "schema_version",
                "current_capacity",
                "population_next_id",
                "max_births_per_tick",
                "slot_capacity",
                "allocation_generation_count",
                "retained_capacity",
                "required_aggregate_capacity",
            }
        ),
    )
    if allocation["schema_version"] != AGENT_ID_ALLOCATION_SCHEMA_VERSION:
        raise _fail("capacity proof agent-ID allocation schema is unsupported")
    expected_proof = ExactFullCapacityEnvelopeV1(
        attempt_capacity_a=declared["A"],
        snapshot_capacity_s=declared["S"],
        eligible_parent_capacity_c=declared["C"],
        aggregate_capacity_k=declared["K"],
        max_resident_population=proof["max_resident_population"],
        current_agent_id_capacity=allocation["current_capacity"],
        population_next_id=allocation["population_next_id"],
        max_births_per_tick=allocation["max_births_per_tick"],
        retained_agent_id_capacity=allocation["retained_capacity"],
    ).proof(
        generations=proof["generations"],
        ticks=proof["ticks_per_generation"],
        checkpoint_split_generation=proof[
            "checkpoint_split_generation"
        ],
        maximum_physical_batch=proof["maximum_physical_batch"],
    )
    if (
        expected_proof != supplied_proof
        or required["K"] != allocation["required_aggregate_capacity"]
        or proof["max_resident_population"] != allocation["slot_capacity"]
        or proof["generations"] != allocation["allocation_generation_count"]
    ):
        raise _fail("capacity proof does not match its derived bounds")
    if row["required_gate_check_ids"] != list(REQUIRED_GATE_CHECK_IDS):
        raise _fail("required gate check set is incomplete")
    if row["reporting_parity_run_ids"] != [
        REPORTING_ENABLED_RUN_ID,
        REPORTING_DISABLED_RUN_ID,
    ]:
        raise _fail("reporting parity run set is incomplete")
    parity = _object(
        row["storage_mode_parity"],
        name="storage_mode_parity",
        fields=frozenset(
            {
                "exact_full_request",
                "compact_complete_request",
                "comparison_contract",
            }
        ),
    )
    exact_request = _verify_auxiliary_storage_request(
        parity["exact_full_request"],
        name="storage_mode_parity.exact_full_request",
        expected_request_id=STORAGE_EXACT_FULL_REQUEST_ID,
        expected_storage_mode=EXACT_FULL_STORAGE_MODE,
    )
    compact_request = _verify_auxiliary_storage_request(
        parity["compact_complete_request"],
        name="storage_mode_parity.compact_complete_request",
        expected_request_id=STORAGE_COMPACT_COMPLETE_REQUEST_ID,
        expected_storage_mode="compact_complete",
    )
    if (
        exact_request["candidate_authority_sha256"]
        == compact_request["candidate_authority_sha256"]
        or exact_request["candidate_artifact_sha256"]
        == compact_request["candidate_artifact_sha256"]
        or exact_request["physical_order"] != compact_request["physical_order"]
    ):
        raise _fail(
            "storage parity requests lack distinct authorities or identical cells"
        )
    comparison = _object(
        parity["comparison_contract"],
        name="storage_mode_parity.comparison_contract",
        fields=frozenset(
            {
                "schema_version",
                "exact_full_request_sha256",
                "compact_complete_request_sha256",
                "required_comparisons",
                "exact_full_eligible_parent_id_sets_authoritative",
                "compact_per_attempt_eligible_parent_id_sets_forbidden",
                "compact_eligibility_comparison",
                "promotion_authority",
                "comparison_contract_sha256",
            }
        ),
    )
    comparison_digest = _digest(
        comparison.pop("comparison_contract_sha256"),
        field="comparison_contract_sha256",
    )
    if (
        canonical_sha256(comparison) != comparison_digest
        or comparison["schema_version"]
        != STORAGE_PARITY_COMPARISON_SCHEMA_VERSION
        or comparison["exact_full_request_sha256"]
        != exact_request["request_sha256"]
        or comparison["compact_complete_request_sha256"]
        != compact_request["request_sha256"]
        or comparison["required_comparisons"] != list(STORAGE_PARITY_FIELDS)
        or comparison["exact_full_eligible_parent_id_sets_authoritative"]
        is not True
        or comparison["compact_per_attempt_eligible_parent_id_sets_forbidden"]
        is not True
        or comparison["compact_eligibility_comparison"]
        != "role_specific_aggregate_opportunity_counts_v1"
        or comparison["promotion_authority"] is not False
    ):
        raise _fail("storage-mode parity comparison contract is invalid")
    runner_runs = runner_plan.get("runs")
    invalid_standard_run = False
    if isinstance(runner_runs, list):
        for run in runner_runs:
            payload = (
                run.get("execution_payload")
                if isinstance(run, Mapping)
                else None
            )
            if (
                not isinstance(payload, Mapping)
                or payload.get("exposure_storage_mode")
                != EXACT_FULL_STORAGE_MODE
                or run.get("run_id")
                in {
                    STORAGE_EXACT_FULL_REQUEST_ID,
                    STORAGE_COMPACT_COMPLETE_REQUEST_ID,
                }
            ):
                invalid_standard_run = True
                break
    if not isinstance(runner_runs, list) or invalid_standard_run:
        raise _fail("standard isolation runner contains a non-exact storage run")
    return {**row, "artifact_sha256": supplied}


def publish_composed_plan_v1(path: Path, artifact: Mapping[str, Any]) -> None:
    """Atomically publish one immutable JSON artifact without replacement."""

    destination = Path(path)
    if not destination.is_absolute():
        raise _fail("output path must be absolute")
    verify_composed_plan_artifact_v1(artifact)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(pretty_json(artifact))
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError as exc:
            raise _fail(f"output already exists: {destination}") from exc
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compose a deterministic mixed-mode resident isolation "
            "preregistration."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        raw = json.loads(args.config.read_text(encoding="utf-8"))
        config = MixedModeIsolationComposerConfigV1.from_payload(raw)
        artifact = compose_mixed_mode_resident_isolation_plan_v1(config)
        publish_composed_plan_v1(args.output, artifact)
    except (
        OSError,
        json.JSONDecodeError,
        MixedModeIsolationPlanComposerError,
    ) as exc:
        raise SystemExit(str(exc)) from exc
    print(
        json.dumps(
            {
                "ok": True,
                "output": str(args.output),
                "artifact_sha256": artifact["artifact_sha256"],
                "runner_plan_sha256": artifact["runner_plan_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


__all__ = [
    "ARTIFACT_SCHEMA_VERSION",
    "CAPACITY_PROOF_SCHEMA_VERSION",
    "CONFIG_SCHEMA_VERSION",
    "ExactFullCapacityEnvelopeV1",
    "MixedModeIsolationComposerConfigV1",
    "MixedModeIsolationPlanComposerError",
    "MAX_NATIVE_INT64",
    "REPORTING_DISABLED_RUN_ID",
    "REPORTING_ENABLED_RUN_ID",
    "REQUIRED_GATE_CHECK_IDS",
    "RepresentativeResidentCellV1",
    "SEGMENT_REQUEST_SCHEMA_VERSION",
    "STORAGE_COMPACT_COMPLETE_REQUEST_ID",
    "STORAGE_EXACT_FULL_REQUEST_ID",
    "STORAGE_PARITY_COMPARISON_SCHEMA_VERSION",
    "STORAGE_PARITY_FIELDS",
    "STORAGE_PARITY_REQUEST_SCHEMA_VERSION",
    "compose_mixed_mode_resident_isolation_plan_v1",
    "main",
    "publish_composed_plan_v1",
    "verify_composed_plan_artifact_v1",
]


if __name__ == "__main__":
    raise SystemExit(main())
