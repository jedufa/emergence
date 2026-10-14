"""Compatibility projection for the authoritative EIEL campaign compiler.

Compilation and batching delegate to the current campaign implementation.  The
execution half remains dormant unless an explicit compatibility port is
injected; Phase 2 does not install a second campaign or runtime lifecycle.
"""

from __future__ import annotations

import copy
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Protocol, runtime_checkable

from emergence_contracts import (
    AssemblyId,
    CanonicalPayloadV1,
    CellId,
    ContentDigest,
    ContractError,
    ContractErrorCategory,
    ContractIssueV1,
    ExperimentCellV1,
    ExperimentId,
    ExperimentPlanV1,
    ExperimentResultSetV1,
    ExperimentSpecV1,
    PhysicalBatchV1,
    PhysicalExecutionPlanV1,
    ResultSetId,
    RuntimeCapabilityProfileV1,
    RuntimeProviderFactoryV1,
    SchemaVersion,
    ValidationResultV1,
)


LEGACY_CAMPAIGN_SPEC_SCHEMA = SchemaVersion("eiel_campaign_v1")
LEGACY_RUNTIME_REQUIREMENTS_SCHEMA = SchemaVersion(
    "legacy_campaign_runtime_requirements_v1"
)
LEGACY_MEASUREMENT_REQUIREMENTS_SCHEMA = SchemaVersion(
    "legacy_campaign_measurement_requirements_v1"
)
LEGACY_REPEAT_PARITY_SCHEMA = SchemaVersion("legacy_campaign_repeat_parity_v1")
LEGACY_CAMPAIGN_OUTPUT_SCHEMA = SchemaVersion("eiel_campaign_execution_manifest_v1")


def _lab_error(
    category: ContractErrorCategory,
    message: str,
    *,
    details: Mapping[str, object] | None = None,
) -> ContractError:
    return ContractError(
        category,
        owner="eiel_composition.adapters.legacy_lab",
        message=message,
        details=details,
    )


def _legacy_campaign_api() -> tuple[object, object, object, object]:
    """Load legacy compiler symbols only when a compatibility call is made."""

    import engine

    engine_root = str(Path(engine.__file__).resolve().parent)
    if engine_root not in sys.path:
        # Current EIEL still has qualified compatibility imports such as
        # ``evolution.*``.  Localize that legacy requirement to adapter use.
        sys.path.insert(0, engine_root)
    from engine.campaigns.canonical import canonical_sha256
    from engine.campaigns.compiler import (
        CampaignError,
        compile_campaign,
        validate_campaign_spec,
    )

    return CampaignError, compile_campaign, validate_campaign_spec, canonical_sha256


@runtime_checkable
class LegacyCampaignExecutionPortV1(Protocol):
    """Injected bridge returning C-130 refs; no production bridge exists before S9."""

    def execute(
        self,
        legacy_plan: Mapping[str, object],
        provider_factory: RuntimeProviderFactoryV1,
    ) -> ExperimentResultSetV1: ...

    def resume(self, result_set_id: ResultSetId) -> ExperimentResultSetV1: ...

    def status(self, result_set_id: ResultSetId) -> ExperimentResultSetV1: ...


@dataclass(frozen=True, slots=True)
class _CompiledBindingV1:
    legacy_payload: dict[str, object]
    target: RuntimeCapabilityProfileV1


class LegacyExperimentPlannerAdapter:
    """Translate C-120 specs/plans without changing current compiler output."""

    def __init__(self, *, assembly_id: AssemblyId, profile_registry: object | None = None):
        if not isinstance(assembly_id, AssemblyId):
            raise _lab_error(
                ContractErrorCategory.CONTRACT_INVALID,
                "assembly_id must be an AssemblyId",
            )
        self._assembly_id = assembly_id
        self._profile_registry = profile_registry
        self._bindings: dict[ExperimentPlanV1, _CompiledBindingV1] = {}

    @staticmethod
    def from_legacy_spec(value: Mapping[str, object]) -> ExperimentSpecV1:
        campaign_error, _compile, validate, _digest = _legacy_campaign_api()
        try:
            canonical = validate(value)
        except campaign_error as error:
            raise _lab_error(
                ContractErrorCategory.CONFIGURATION_INVALID,
                "legacy campaign specification is invalid",
                details={"error": str(error)},
            ) from error
        return ExperimentSpecV1(
            experiment_id=ExperimentId(str(canonical["campaign_id"])),
            spec_payload=CanonicalPayloadV1.from_value(
                LEGACY_CAMPAIGN_SPEC_SCHEMA, canonical
            ),
        )

    def validate_experiment_spec(self, spec: ExperimentSpecV1) -> ValidationResultV1:
        if not isinstance(spec, ExperimentSpecV1):
            return ValidationResultV1(
                valid=False,
                issues=(
                    ContractIssueV1(
                        category=ContractErrorCategory.CONTRACT_INVALID,
                        owner="eiel_composition.adapters.legacy_lab",
                        message="spec must be an ExperimentSpecV1",
                    ),
                ),
            )
        if spec.spec_payload.schema_version != LEGACY_CAMPAIGN_SPEC_SCHEMA:
            return ValidationResultV1(
                valid=False,
                issues=(
                    ContractIssueV1(
                        category=ContractErrorCategory.CONFIGURATION_INVALID,
                        owner="eiel_composition.adapters.legacy_lab",
                        message="unsupported legacy campaign spec schema",
                    ),
                ),
            )
        campaign_error, _compile, validate, _digest = _legacy_campaign_api()
        try:
            canonical = validate(spec.spec_payload.to_value())
        except campaign_error as error:
            return ValidationResultV1(
                valid=False,
                issues=(
                    ContractIssueV1(
                        category=ContractErrorCategory.CONFIGURATION_INVALID,
                        owner="engine.campaigns.compiler",
                        message=str(error),
                    ),
                ),
            )
        if str(spec.experiment_id) != canonical.get("campaign_id"):
            return ValidationResultV1(
                valid=False,
                issues=(
                    ContractIssueV1(
                        category=ContractErrorCategory.CONFIGURATION_INVALID,
                        owner="eiel_composition.adapters.legacy_lab",
                        message="experiment_id does not match legacy campaign_id",
                    ),
                ),
            )
        return ValidationResultV1(valid=True)

    def compile(
        self, spec: ExperimentSpecV1, target: RuntimeCapabilityProfileV1
    ) -> ExperimentPlanV1:
        validation = self.validate_experiment_spec(spec)
        if not validation.valid:
            issue = validation.issues[0]
            raise _lab_error(issue.category, issue.message)
        if not isinstance(target, RuntimeCapabilityProfileV1):
            raise _lab_error(
                ContractErrorCategory.CONTRACT_INVALID,
                "target must be a RuntimeCapabilityProfileV1",
            )

        campaign_error, compile_campaign, _validate, current_digest = (
            _legacy_campaign_api()
        )
        try:
            if self._profile_registry is None:
                legacy_plan = compile_campaign(spec.spec_payload.to_value())
            else:
                legacy_plan = compile_campaign(
                    spec.spec_payload.to_value(), registry=self._profile_registry
                )
        except campaign_error as error:
            raise _lab_error(
                ContractErrorCategory.CONFIGURATION_INVALID,
                "legacy campaign compilation failed",
                details={"error": str(error)},
            ) from error

        legacy_payload = legacy_plan.payload()
        if str(target.profile_id) != legacy_plan.profile_id:
            raise _lab_error(
                ContractErrorCategory.CAPABILITY_UNSUPPORTED,
                "target profile does not match the authoritative compiled profile",
                details={
                    "target_profile_id": str(target.profile_id),
                    "compiled_profile_id": legacy_plan.profile_id,
                },
            )
        expected_backend = str(legacy_plan.qualification_key.get("backend", ""))
        if str(target.backend_id) != expected_backend:
            raise _lab_error(
                ContractErrorCategory.CAPABILITY_UNSUPPORTED,
                "target backend does not match the authoritative compiled profile",
                details={
                    "target_backend_id": str(target.backend_id),
                    "compiled_backend_id": expected_backend,
                },
            )

        legacy_cells = list(legacy_payload["cells"])
        logical_cells = tuple(
            ExperimentCellV1(
                cell_id=CellId(str(row["cell_id"])),
                # Ecology is the required primary legacy seed owner.  The full,
                # exact multi-owner seed map remains in runtime_requirements.
                seed=int(row["seeds"]["ecology"]),
                control_id=str(row["branch_id"]),
                assembly_id=self._assembly_id,
                config_digest=ContentDigest("sha256", current_digest(row)),
            )
            for row in legacy_cells
        )
        runtime_requirements = {
            "legacy_plan_schema_version": legacy_plan.schema_version,
            "legacy_plan_sha256": legacy_plan.plan_sha256,
            "runtime_profile_id": legacy_plan.profile_id,
            "qualification_key": copy.deepcopy(legacy_plan.qualification_key),
            "qualification_key_sha256": legacy_plan.qualification_key_sha256,
            "execution": copy.deepcopy(legacy_payload["execution"]),
            "assembly_id": str(self._assembly_id),
            "legacy_cell_bindings": copy.deepcopy(legacy_cells),
        }
        measurement_requirements = {
            "description": legacy_plan.description,
            "no_claims": legacy_plan.no_claims,
            "analysis": copy.deepcopy(legacy_payload["analysis"]),
            "artifacts": copy.deepcopy(legacy_payload["artifacts"]),
            "output": copy.deepcopy(legacy_payload["output"]),
        }
        repeat_parity = copy.deepcopy(legacy_payload["comparisons"])
        contract_plan = ExperimentPlanV1(
            experiment_id=spec.experiment_id,
            logical_cells=logical_cells,
            # Current fresh/resume/reporting branches are not two-cell paired
            # comparisons.  Preserve them exactly in repeat_parity_plan.
            paired_comparisons=(),
            runtime_requirements=CanonicalPayloadV1.from_value(
                LEGACY_RUNTIME_REQUIREMENTS_SCHEMA, runtime_requirements
            ),
            measurement_requirements=CanonicalPayloadV1.from_value(
                LEGACY_MEASUREMENT_REQUIREMENTS_SCHEMA, measurement_requirements
            ),
            output_contract=LEGACY_CAMPAIGN_OUTPUT_SCHEMA,
            repeat_parity_plan=CanonicalPayloadV1.from_value(
                LEGACY_REPEAT_PARITY_SCHEMA, repeat_parity
            ),
        )
        self._bindings[contract_plan] = _CompiledBindingV1(
            legacy_payload=copy.deepcopy(legacy_payload),
            target=target,
        )
        return contract_plan

    def _binding(self, plan: ExperimentPlanV1) -> _CompiledBindingV1:
        if not isinstance(plan, ExperimentPlanV1):
            raise _lab_error(
                ContractErrorCategory.CONTRACT_INVALID,
                "plan must be an ExperimentPlanV1",
            )
        binding = self._bindings.get(plan)
        if binding is None:
            raise _lab_error(
                ContractErrorCategory.CONFIGURATION_INVALID,
                "experiment plan was not compiled by this adapter instance",
            )
        return binding

    def legacy_plan_payload(self, plan: ExperimentPlanV1) -> dict[str, object]:
        return copy.deepcopy(self._binding(plan).legacy_payload)

    def target_capabilities(
        self, plan: ExperimentPlanV1
    ) -> RuntimeCapabilityProfileV1:
        return self._binding(plan).target

    def plan_batches(self, plan: ExperimentPlanV1) -> PhysicalExecutionPlanV1:
        binding = self._binding(plan)
        legacy_batches = binding.legacy_payload["physical_batches"]
        return PhysicalExecutionPlanV1(
            experiment_id=plan.experiment_id,
            batches=tuple(
                PhysicalBatchV1(
                    batch_id=str(batch["batch_id"]),
                    cell_ids=tuple(CellId(str(item)) for item in batch["cell_ids"]),
                    runtime_profile_id=str(binding.legacy_payload["resolved_profile"]),
                )
                for batch in legacy_batches
            ),
        )


class LegacyExperimentExecutorAdapter:
    """Full C-120 shape with explicitly injected, dormant execution routing."""

    def __init__(
        self,
        *,
        planner: LegacyExperimentPlannerAdapter,
        execution_port: LegacyCampaignExecutionPortV1 | None = None,
    ) -> None:
        if not isinstance(planner, LegacyExperimentPlannerAdapter):
            raise _lab_error(
                ContractErrorCategory.CONTRACT_INVALID,
                "planner must be a LegacyExperimentPlannerAdapter",
            )
        if execution_port is not None and not isinstance(
            execution_port, LegacyCampaignExecutionPortV1
        ):
            raise _lab_error(
                ContractErrorCategory.CONTRACT_INVALID,
                "execution_port does not satisfy LegacyCampaignExecutionPortV1",
            )
        self._planner = planner
        self._execution_port = execution_port

    def validate_experiment_spec(self, spec: ExperimentSpecV1) -> ValidationResultV1:
        return self._planner.validate_experiment_spec(spec)

    def compile(
        self, spec: ExperimentSpecV1, target: RuntimeCapabilityProfileV1
    ) -> ExperimentPlanV1:
        return self._planner.compile(spec, target)

    def plan_batches(self, plan: ExperimentPlanV1) -> PhysicalExecutionPlanV1:
        return self._planner.plan_batches(plan)

    def _port(self) -> LegacyCampaignExecutionPortV1:
        if self._execution_port is None:
            raise _lab_error(
                ContractErrorCategory.CAPABILITY_UNSUPPORTED,
                "Lab execution routing is dormant until an explicit compatibility port is injected",
            )
        return self._execution_port

    def execute(
        self, plan: ExperimentPlanV1, provider_factory: RuntimeProviderFactoryV1
    ) -> ExperimentResultSetV1:
        if not isinstance(provider_factory, RuntimeProviderFactoryV1):
            raise _lab_error(
                ContractErrorCategory.CONTRACT_INVALID,
                "provider_factory does not satisfy RuntimeProviderFactoryV1",
            )
        result = self._port().execute(
            self._planner.legacy_plan_payload(plan), provider_factory
        )
        return self._validate_result(result, expected_experiment=plan.experiment_id)

    def resume(self, result_set_id: ResultSetId) -> ExperimentResultSetV1:
        if not isinstance(result_set_id, ResultSetId):
            raise _lab_error(
                ContractErrorCategory.CONTRACT_INVALID,
                "result_set_id must be a ResultSetId",
            )
        return self._validate_result(self._port().resume(result_set_id))

    def status(self, result_set_id: ResultSetId) -> ExperimentResultSetV1:
        if not isinstance(result_set_id, ResultSetId):
            raise _lab_error(
                ContractErrorCategory.CONTRACT_INVALID,
                "result_set_id must be a ResultSetId",
            )
        return self._validate_result(self._port().status(result_set_id))

    @staticmethod
    def _validate_result(
        result: ExperimentResultSetV1,
        *,
        expected_experiment: ExperimentId | None = None,
    ) -> ExperimentResultSetV1:
        if not isinstance(result, ExperimentResultSetV1):
            raise _lab_error(
                ContractErrorCategory.RUNTIME_EXECUTION_FAILURE,
                "legacy execution port returned an invalid result set",
            )
        if expected_experiment is not None and result.experiment_id != expected_experiment:
            raise _lab_error(
                ContractErrorCategory.RUNTIME_EXECUTION_FAILURE,
                "legacy execution result does not match the requested experiment",
            )
        return result
