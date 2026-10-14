"""Typed contracts shared by campaign compilation and runtime providers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Mapping

Scalar = bool | int | float | str
Predicate = Callable[[Scalar], bool]


@dataclass(frozen=True, slots=True)
class ParameterContractV1:
    """One public component parameter accepted by a runtime profile."""

    name: str
    value_type: type
    predicate: Predicate
    description: str
    per_cell: bool = True
    qualification_affecting: bool = False
    control_role: str = "independent"
    wrc_axis: bool = True
    alias_of: str | None = None
    variability_semantics: str = "legacy"
    variability_route_scopes: tuple[str, ...] = ()
    value_contract: Mapping[str, Any] = field(default_factory=dict)
    activation_contract: Mapping[str, Any] = field(default_factory=dict)

    def accepts(self, value: Any) -> bool:
        return type(value) is self.value_type and self.predicate(value)

    def variability_contract_payload_v1(self) -> dict[str, Any]:
        """Return the declarative identity used by WRC capability envelopes.

        ``predicate`` is intentionally not serialized: executable callables are
        not a stable authority surface.  A WRC-capable parameter must therefore
        supply an explicit ``value_contract`` alongside its runtime predicate.
        """

        if self.variability_semantics != "experiment_variable":
            raise ValueError(
                f"parameter {self.name!r} is not an experiment-variable capability"
            )
        if not self.value_contract:
            raise ValueError(
                f"parameter {self.name!r} lacks a declarative value contract"
            )
        return {
            "name": self.name,
            "type": self.value_type.__name__,
            "per_cell": self.per_cell,
            "qualification_affecting": self.qualification_affecting,
            "control_role": self.control_role,
            "variability_semantics": self.variability_semantics,
            "variability_route_scopes": list(self.variability_route_scopes),
            "value_contract": dict(self.value_contract),
            # This freezes the dependency/activation *contract* in the
            # capability envelope.  It does not assert that a particular
            # campaign cell has activated it successfully.
            "activation_contract": dict(self.activation_contract),
        }


@dataclass(frozen=True, slots=True)
class MetricFamilyV1:
    """A coherent family of metrics emitted by one ownership boundary."""

    family_id: str
    metrics: tuple[str, ...]
    required_state_components: tuple[str, ...]
    null_semantics: str


@dataclass(frozen=True, slots=True)
class RuntimeProfileV1:
    """Machine-readable capability description for one implementation profile."""

    profile_id: str
    profile_version: int
    backend: str
    world_base: str
    world_overlays: frozenset[str]
    policy_family: str
    selection_modes: frozenset[str]
    observation_contract: str
    state_components: tuple[str, ...]
    checkpoint_components: tuple[str, ...]
    parameter_contracts: Mapping[str, ParameterContractV1]
    uniform_batch_parameters: tuple[str, ...]
    metric_families: tuple[MetricFamilyV1, ...]
    execution_owner: str
    arbitrary_positive_generations: bool
    continuation_supported: bool
    cancellation_supported: bool
    batch_min: int
    batch_max: int
    qualified_batch_min: int
    qualified_batch_max: int
    generation_counter_bits: int
    qualification_invariants: Mapping[str, Scalar]
    qualification_status: str = "qualified"
    limitations: tuple[str, ...] = ()

    @property
    def metric_names(self) -> frozenset[str]:
        return frozenset(
            metric for family in self.metric_families for metric in family.metrics
        )

    def public_receipt(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "profile_version": self.profile_version,
            "backend": self.backend,
            "world_base": self.world_base,
            "world_overlays": sorted(self.world_overlays),
            "policy_family": self.policy_family,
            "selection_modes": sorted(self.selection_modes),
            "observation_contract": self.observation_contract,
            "state_components": list(self.state_components),
            "checkpoint_components": list(self.checkpoint_components),
            "parameters": {
                name: {
                    "type": contract.value_type.__name__,
                    "per_cell": contract.per_cell,
                    "qualification_affecting": contract.qualification_affecting,
                    "control_role": contract.control_role,
                    "wrc_axis": contract.wrc_axis,
                    "alias_of": contract.alias_of,
                    "variability_semantics": contract.variability_semantics,
                    "variability_route_scopes": list(contract.variability_route_scopes),
                    "value_contract": dict(contract.value_contract),
                    "activation_contract": dict(contract.activation_contract),
                    "description": contract.description,
                }
                for name, contract in sorted(self.parameter_contracts.items())
            },
            "uniform_batch_parameters": list(self.uniform_batch_parameters),
            "metric_families": [asdict(family) for family in self.metric_families],
            "execution_owner": self.execution_owner,
            "arbitrary_positive_generations": self.arbitrary_positive_generations,
            "continuation_supported": self.continuation_supported,
            "cancellation_supported": self.cancellation_supported,
            "batch_range": [self.batch_min, self.batch_max],
            "planning_batch_range": [self.batch_min, self.batch_max],
            "qualified_batch_range": [
                self.qualified_batch_min,
                self.qualified_batch_max,
            ],
            "generation_counter_bits": self.generation_counter_bits,
            "qualification_status": self.qualification_status,
            "limitations": list(self.limitations),
        }


@dataclass(frozen=True, slots=True)
class CampaignRequirementsV1:
    backend: str
    world_base: str
    world_overlays: frozenset[str]
    policy_family: str
    selection_mode: str
    parameter_names: frozenset[str]
    required_metrics: frozenset[str]
    require_continuation: bool
    require_cancellation: bool


@dataclass(frozen=True, slots=True)
class UnsupportedReasonV1:
    code: str
    path: str
    message: str
    profile_id: str | None = None


@dataclass(frozen=True, slots=True)
class QualificationKeyV1:
    schema_version: str
    profile_id: str
    profile_version: int
    backend: str
    artifact_hashes: tuple[tuple[str, str], ...]
    observation_contract: str
    state_components: tuple[str, ...]
    checkpoint_components: tuple[str, ...]
    fixed_invariants: tuple[tuple[str, Scalar], ...]
    admitted_batch_range: tuple[int, int]
    variability_envelope: Mapping[str, Any] = field(default_factory=dict)

    def payload(self) -> dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "profile_id": self.profile_id,
            "profile_version": self.profile_version,
            "backend": self.backend,
            "artifact_hashes": dict(self.artifact_hashes),
            "observation_contract": self.observation_contract,
            "state_components": list(self.state_components),
            "checkpoint_components": list(self.checkpoint_components),
            "fixed_invariants": dict(self.fixed_invariants),
            "admitted_batch_range": list(self.admitted_batch_range),
        }
        if self.variability_envelope:
            payload["variability_envelope"] = dict(self.variability_envelope)
        return payload


@dataclass(frozen=True, slots=True)
class ExecutionRouteContractV1:
    """Canonical identity for one provider-owned behavior route.

    Component records are deliberately embedded instead of exposed as another
    public contract type.  Ordinary campaign extent and partition values do not
    belong in this contract.
    """

    schema_version: str
    provider_id: str
    profile_id: str
    components: tuple[Mapping[str, Any], ...]
    required_obligations: tuple[str, ...]

    def __post_init__(self) -> None:
        component_ids: list[str] = []
        for component in self.components:
            component_id = component.get("component_id")
            if not isinstance(component_id, str) or not component_id:
                raise ValueError(
                    "execution route components require a non-empty component_id"
                )
            component_ids.append(component_id)
        if len(component_ids) != len(set(component_ids)):
            raise ValueError("execution route component IDs must be unique")
        if len(self.required_obligations) != len(set(self.required_obligations)) or any(
            not isinstance(obligation, str) or not obligation
            for obligation in self.required_obligations
        ):
            raise ValueError(
                "execution route required obligations must be unique non-empty strings"
            )

    def payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "provider_id": self.provider_id,
            "profile_id": self.profile_id,
            "components": [
                dict(component)
                for component in sorted(
                    self.components, key=lambda row: str(row["component_id"])
                )
            ],
            "required_obligations": sorted(self.required_obligations),
        }

    @property
    def execution_route_sha256(self) -> str:
        # Keep canonical JSON ownership in the existing campaign helper without
        # introducing a module-level import cycle.
        from .canonical import canonical_sha256

        return canonical_sha256(self.payload())


@dataclass(frozen=True, slots=True)
class ExecutionRouteQualificationReceiptV1:
    """Evidence coverage for one exact execution-route contract."""

    schema_version: str
    execution_route_sha256: str
    provider_id: str
    profile_id: str
    covered_obligations: tuple[str, ...]
    qualification_plan_ids: tuple[str, ...] = ()
    case_ids: tuple[str, ...] = ()
    artifact_sha256: str | None = None
    checkpoint_results: Mapping[str, Any] = field(default_factory=dict)
    reporting_results: Mapping[str, Any] = field(default_factory=dict)
    route_obligation_results: tuple[Mapping[str, Any], ...] = ()
    exact_comparisons: tuple[Mapping[str, Any], ...] = ()
    variability_envelope: Mapping[str, Any] = field(default_factory=dict)
    authority_state: str = "route_qualified"
    qualification_dimensions: tuple[str, ...] = ()
    long_horizon_scope: Mapping[str, Any] = field(default_factory=dict)

    def payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "authority_state": self.authority_state,
            "execution_route_sha256": self.execution_route_sha256,
            "provider_id": self.provider_id,
            "profile_id": self.profile_id,
            "covered_obligations": sorted(set(self.covered_obligations)),
            "qualification_plan_ids": list(self.qualification_plan_ids),
            "case_ids": list(self.case_ids),
            "artifact_sha256": self.artifact_sha256,
            "checkpoint_results": dict(self.checkpoint_results),
            "reporting_results": dict(self.reporting_results),
            "route_obligation_results": [
                dict(result) for result in self.route_obligation_results
            ],
            "exact_comparisons": [
                dict(comparison) for comparison in self.exact_comparisons
            ],
            "variability_envelope": dict(self.variability_envelope),
            "qualification_dimensions": sorted(set(self.qualification_dimensions)),
            "long_horizon_scope": dict(self.long_horizon_scope),
        }


@dataclass(frozen=True, slots=True)
class LogicalCellV1:
    ordinal: int
    cell_id: str
    branch_id: str
    rng_lineage_id: str
    dimensions: Mapping[str, Scalar]
    parameters: Mapping[str, Scalar]
    seeds: Mapping[str, int]


@dataclass(frozen=True, slots=True)
class PhysicalBatchV1:
    ordinal: int
    batch_id: str
    cell_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ComparisonBranchV1:
    ordinal: int
    branch_id: str
    kind: str
    source_cell_id: str
    repeat_ordinal: int | None
    split_generation: int | None
    baseline_mode: str | None = None


@dataclass(frozen=True, slots=True)
class CampaignPlanV1:
    schema_version: str
    campaign_id: str
    description: str
    no_claims: str
    profile_id: str
    qualification_key: Mapping[str, Any]
    qualification_key_sha256: str
    generations: int | None
    ticks_per_generation: int | None
    checkpoints: tuple[int, ...]
    digest_every_generation: bool
    checkpoint_storage_mode: str
    budgets: Mapping[str, int]
    cells: tuple[LogicalCellV1, ...]
    batches: tuple[PhysicalBatchV1, ...]
    comparison_branches: tuple[ComparisonBranchV1, ...]
    analysis: Mapping[str, Any]
    artifacts: Mapping[str, Mapping[str, Scalar]]
    output: Mapping[str, Scalar]
    resource_estimate: Mapping[str, int | float | str]
    technical_gate: Mapping[str, Any] = field(default_factory=dict)
    physical_window_generations: int | None = None
    plan_sha256: str = field(default="")
    tick_schedule: Mapping[str, Any] = field(default_factory=dict)
    acs_checkpoint_slots: int = 2
    acs_capture_instrumentation: bool = False
    run_contract: Mapping[str, Any] = field(default_factory=dict)
    artifact_lifecycle: Mapping[str, Any] = field(default_factory=dict)
    execution_route_identity: str | None = None
    total_ticks: int | None = None
    start_next_tick: int = 0
    checkpoint_ticks: tuple[int, ...] = ()
    deliberate_restore_tick: int | None = None
    deliberate_restore_ticks: tuple[int, ...] = ()
    internal_lifecycle_interval_ticks: int | None = None
    max_internal_lifecycle_segments_per_dispatch: int | None = None
    tick_native_control: Mapping[str, Any] = field(default_factory=dict)
    gpu_resource_policy: Mapping[str, Any] = field(default_factory=dict)
    native_streamed_checkpoint: bool = False
    world_sharded_checkpoint: bool = False

    def payload(self, *, include_hash: bool = True) -> dict[str, Any]:
        if self.execution_route_identity == "tick_native_execution_route_v1":
            if type(self.total_ticks) is not int or self.total_ticks <= 0:
                raise ValueError("tick-native plan requires a positive total_ticks")
            execution: dict[str, Any] = {
                "execution_route_identity": self.execution_route_identity,
                "total_ticks": self.total_ticks,
                "start_next_tick": self.start_next_tick,
                "checkpoint_ticks": list(self.checkpoint_ticks or (self.total_ticks,)),
                "deliberate_restore_tick": self.deliberate_restore_tick,
                "deliberate_restore_ticks": list(
                    self.deliberate_restore_ticks
                    or (
                        (self.deliberate_restore_tick,)
                        if self.deliberate_restore_tick is not None
                        else ()
                    )
                ),
                "time_authority": "absolute_tick_v1",
                "tick_native_control": dict(self.tick_native_control),
                "checkpoint_storage_mode": self.checkpoint_storage_mode,
                "budgets": dict(self.budgets),
                "technical_gate": dict(self.technical_gate),
            }
        else:
            if type(self.generations) is not int or type(self.ticks_per_generation) is not int:
                raise ValueError("legacy plan requires generation dimensions")
            execution = {
                "generations": self.generations,
                "ticks_per_generation": self.ticks_per_generation,
                "checkpoints": list(self.checkpoints),
                "digest_every_generation": self.digest_every_generation,
                "checkpoint_storage_mode": self.checkpoint_storage_mode,
                "budgets": dict(self.budgets),
                "technical_gate": dict(self.technical_gate),
            }
        value = {
            "schema_version": self.schema_version,
            "campaign_id": self.campaign_id,
            "description": self.description,
            "no_claims": self.no_claims,
            "resolved_profile": self.profile_id,
            "qualification_key": dict(self.qualification_key),
            "qualification_key_sha256": self.qualification_key_sha256,
            "execution": execution,
            "cells": [asdict(cell) for cell in self.cells],
            "physical_batches": [asdict(batch) for batch in self.batches],
            "comparisons": {
                "branches": [asdict(branch) for branch in self.comparison_branches],
                "reporting_parity": {
                    "required": True,
                    "accepted_evidence_modes": [
                        "enabled_disabled_exact_terminal_state",
                        "provider_observational_contract_with_state_mutation_sentinel",
                    ],
                },
            },
            "analysis": dict(self.analysis),
            "artifacts": {name: dict(row) for name, row in self.artifacts.items()},
            "output": dict(self.output),
            "resource_estimate": dict(self.resource_estimate),
        }
        if (
            self.physical_window_generations is not None
            and self.execution_route_identity != "tick_native_execution_route_v1"
        ):
            value["execution"][
                "physical_window_generations"
            ] = self.physical_window_generations
        if (
            self.max_internal_lifecycle_segments_per_dispatch is not None
            and self.execution_route_identity != "tick_native_execution_route_v1"
        ):
            value["execution"][
                "max_internal_lifecycle_segments_per_dispatch"
            ] = self.max_internal_lifecycle_segments_per_dispatch
        value["execution"]["acs_checkpoint_slots"] = self.acs_checkpoint_slots
        if self.gpu_resource_policy:
            value["execution"]["gpu_resource_policy"] = dict(
                self.gpu_resource_policy
            )
        if self.acs_capture_instrumentation:
            value["execution"]["acs_capture_instrumentation"] = True
        if self.native_streamed_checkpoint:
            value["execution"]["native_streamed_checkpoint"] = True
        if self.world_sharded_checkpoint:
            value["execution"]["world_sharded_checkpoint"] = True
        if self.tick_schedule:
            value["execution"]["tick_schedule"] = dict(self.tick_schedule)
        if self.run_contract:
            value["run_contract"] = dict(self.run_contract)
        if self.artifact_lifecycle:
            value["artifact_lifecycle"] = dict(self.artifact_lifecycle)
        if include_hash:
            value["plan_sha256"] = self.plan_sha256
        return value
