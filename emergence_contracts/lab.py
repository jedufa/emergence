"""C-120 experiment plan and executor contract without runtime internals."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable

from ._validation import (
    require_instance,
    require_int,
    require_text,
    require_tuple_instances,
)
from .artifacts import RunArtifactRefV1
from .capabilities import RuntimeCapabilityProfileV1
from .errors import contract_invalid
from .ids import (
    AssemblyId,
    CellId,
    ContentDigest,
    ExperimentId,
    ResultSetId,
    SchemaVersion,
)
from .payloads import CanonicalPayloadV1
from .provider import RuntimeProviderV1, ValidationResultV1


CONTRACT_ID = "C-120"
CONTRACT_OWNER = "Contracts; implemented by Lab"
API_VERSION = SchemaVersion("c120_v1")
EXPERIMENT_PLAN_SCHEMA_VERSION = SchemaVersion("experiment_plan_v1")


class ExperimentResultStatusV1(str, Enum):
    PLANNED = "PLANNED"
    RUNNING = "RUNNING"
    COMPLETE = "COMPLETE"
    INCOMPLETE = "INCOMPLETE"
    FAILED = "FAILED"


@dataclass(frozen=True, slots=True)
class ExperimentSpecV1:
    experiment_id: ExperimentId
    spec_payload: CanonicalPayloadV1

    def __post_init__(self) -> None:
        require_instance(self.experiment_id, ExperimentId, field="experiment_id")
        require_instance(self.spec_payload, CanonicalPayloadV1, field="spec_payload")


@dataclass(frozen=True, slots=True)
class ExperimentCellV1:
    cell_id: CellId
    seed: int
    control_id: str
    assembly_id: AssemblyId
    config_digest: ContentDigest

    def __post_init__(self) -> None:
        require_instance(self.cell_id, CellId, field="cell_id")
        require_int(self.seed, field="seed")
        require_text(self.control_id, field="control_id")
        require_instance(self.assembly_id, AssemblyId, field="assembly_id")
        require_instance(self.config_digest, ContentDigest, field="config_digest")


@dataclass(frozen=True, slots=True)
class PairedComparisonV1:
    left_cell_id: CellId
    right_cell_id: CellId
    comparison_profile: str

    def __post_init__(self) -> None:
        require_instance(self.left_cell_id, CellId, field="left_cell_id")
        require_instance(self.right_cell_id, CellId, field="right_cell_id")
        if self.left_cell_id == self.right_cell_id:
            raise contract_invalid(
                "paired comparison requires two distinct cells", field="paired_comparison"
            )
        require_text(self.comparison_profile, field="comparison_profile")


@dataclass(frozen=True, slots=True)
class ExperimentPlanV1:
    experiment_id: ExperimentId
    logical_cells: tuple[ExperimentCellV1, ...]
    paired_comparisons: tuple[PairedComparisonV1, ...]
    runtime_requirements: CanonicalPayloadV1
    measurement_requirements: CanonicalPayloadV1
    output_contract: SchemaVersion
    repeat_parity_plan: CanonicalPayloadV1
    schema_version: SchemaVersion = EXPERIMENT_PLAN_SCHEMA_VERSION

    def __post_init__(self) -> None:
        require_instance(self.experiment_id, ExperimentId, field="experiment_id")
        require_tuple_instances(
            self.logical_cells,
            ExperimentCellV1,
            field="logical_cells",
            unique=True,
        )
        require_tuple_instances(
            self.paired_comparisons,
            PairedComparisonV1,
            field="paired_comparisons",
            unique=True,
        )
        if not self.logical_cells:
            raise contract_invalid(
                "experiment plan requires at least one logical cell",
                field="logical_cells",
            )
        cell_ids = {cell.cell_id for cell in self.logical_cells}
        if any(
            comparison.left_cell_id not in cell_ids
            or comparison.right_cell_id not in cell_ids
            for comparison in self.paired_comparisons
        ):
            raise contract_invalid(
                "paired comparison references an unknown cell",
                field="paired_comparisons",
            )
        for field_name in (
            "runtime_requirements",
            "measurement_requirements",
            "repeat_parity_plan",
        ):
            require_instance(
                getattr(self, field_name), CanonicalPayloadV1, field=field_name
            )
        require_instance(self.output_contract, SchemaVersion, field="output_contract")
        if self.schema_version != EXPERIMENT_PLAN_SCHEMA_VERSION:
            raise contract_invalid(
                "unsupported experiment plan schema", field="schema_version"
            )


@dataclass(frozen=True, slots=True)
class PhysicalBatchV1:
    batch_id: str
    cell_ids: tuple[CellId, ...]
    runtime_profile_id: str

    def __post_init__(self) -> None:
        require_text(self.batch_id, field="batch_id")
        require_tuple_instances(
            self.cell_ids,
            CellId,
            field="cell_ids",
            unique=True,
            # Physical cell order is execution-relevant on current EIEL routes.
            # Preserve the declared order; only uniqueness is a C-120 invariant.
            sorted_by_string=False,
        )
        require_text(self.runtime_profile_id, field="runtime_profile_id")


@dataclass(frozen=True, slots=True)
class PhysicalExecutionPlanV1:
    experiment_id: ExperimentId
    batches: tuple[PhysicalBatchV1, ...]

    def __post_init__(self) -> None:
        require_instance(self.experiment_id, ExperimentId, field="experiment_id")
        require_tuple_instances(self.batches, PhysicalBatchV1, field="batches", unique=True)


@dataclass(frozen=True, slots=True)
class ExperimentResultSetV1:
    result_set_id: ResultSetId
    experiment_id: ExperimentId
    status: ExperimentResultStatusV1
    run_manifest_refs: tuple[RunArtifactRefV1, ...]
    metadata: CanonicalPayloadV1

    def __post_init__(self) -> None:
        require_instance(self.result_set_id, ResultSetId, field="result_set_id")
        require_instance(self.experiment_id, ExperimentId, field="experiment_id")
        require_instance(self.status, ExperimentResultStatusV1, field="status")
        require_tuple_instances(
            self.run_manifest_refs,
            RunArtifactRefV1,
            field="run_manifest_refs",
            unique=True,
        )
        require_instance(self.metadata, CanonicalPayloadV1, field="metadata")


@runtime_checkable
class RuntimeProviderFactoryV1(Protocol):
    def create(self, target: RuntimeCapabilityProfileV1) -> RuntimeProviderV1: ...


@runtime_checkable
class ExperimentExecutorV1(Protocol):
    def validate_experiment_spec(self, spec: ExperimentSpecV1) -> ValidationResultV1: ...

    def compile(
        self, spec: ExperimentSpecV1, target: RuntimeCapabilityProfileV1
    ) -> ExperimentPlanV1: ...

    def plan_batches(self, plan: ExperimentPlanV1) -> PhysicalExecutionPlanV1: ...

    def execute(
        self, plan: ExperimentPlanV1, provider_factory: RuntimeProviderFactoryV1
    ) -> ExperimentResultSetV1: ...

    def resume(self, result_set_id: ResultSetId) -> ExperimentResultSetV1: ...

    def status(self, result_set_id: ResultSetId) -> ExperimentResultSetV1: ...
