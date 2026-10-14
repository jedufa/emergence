"""Single executable capability authority frozen into a QRR."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import math
from typing import Any, Final

from engine.campaigns.canonical import canonical_sha256
from engine.campaigns.contracts import RuntimeProfileV1
from engine.campaigns.long_horizon_qualification import (
    BOUNDED_ROUTE_QUALIFIED,
    LONG_HORIZON_RUNABILITY_QUALIFIED,
    SCIENCE_ROUTE_QUALIFIED,
    LongHorizonQualificationError,
    validate_qualification_evidence_binding_v1,
)
from engine.campaigns.production_control_path_v1 import (
    PRODUCTION_HETEROGENEOUS_PARAMETER_NAMES_V1,
)
from engine.qrr.execution_request_v1 import build_execution_request_v1


QUALIFIED_EXECUTION_ENVELOPE_SCHEMA_VERSION: Final[str] = (
    "eiel_qualified_execution_envelope_v1"
)
EXPERIMENT_REPRESENTATION_CONTRACT_SCHEMA_VERSION: Final[str] = (
    "eiel_qrr_experiment_representation_contract_v1"
)
EXPERIMENT_REPRESENTABILITY_RECEIPT_SCHEMA_VERSION: Final[str] = (
    "eiel_qrr_experiment_representability_receipt_v1"
)


class QualifiedExecutionEnvelopeError(ValueError):
    """A plan differs from the release's executable capability authority."""


class ExperimentRepresentabilityError(QualifiedExecutionEnvelopeError):
    """An experiment cannot be represented by the frozen runtime artifact."""

    def __init__(self, message: str, *, receipt: Mapping[str, Any]) -> None:
        super().__init__(message)
        self.receipt = dict(receipt)


def _plan_payload(plan: Any) -> dict[str, Any]:
    payload = getattr(plan, "payload", None)
    value = payload() if callable(payload) else plan
    if not isinstance(value, Mapping):
        raise QualifiedExecutionEnvelopeError("campaign plan must be a mapping")
    return dict(value)


def _artifact_sha256s(plan: Mapping[str, Any]) -> dict[str, str]:
    artifacts = plan.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise QualifiedExecutionEnvelopeError("campaign plan has no artifact map")
    rows: dict[str, str] = {}
    for artifact_id, raw in artifacts.items():
        digest = raw.get("sha256") if isinstance(raw, Mapping) else None
        if not isinstance(digest, str) or len(digest) != 64:
            raise QualifiedExecutionEnvelopeError(
                f"artifact {artifact_id!r} has no exact SHA-256"
            )
        rows[str(artifact_id)] = digest
    return dict(sorted(rows.items()))


def _canonical_unique(values: Sequence[Any]) -> tuple[Any, ...]:
    by_digest = {canonical_sha256({"value": value}): value for value in values}
    return tuple(by_digest[key] for key in sorted(by_digest))


def _value_contract_accepts(contract: Mapping[str, Any], value: Any) -> bool:
    kind = contract.get("kind")
    type_ok = {
        "bool": type(value) is bool,
        "int": type(value) is int,
        "float": type(value) is float,
        "str": type(value) is str,
    }.get(str(kind), False)
    if not type_ok:
        return False
    if contract.get("finite_only") is True and isinstance(value, float):
        if not math.isfinite(value):
            return False
    for bound_name, comparison in (
        ("minimum", lambda left, right: left >= right),
        ("maximum", lambda left, right: left <= right),
    ):
        if bound_name not in contract:
            continue
        bound = contract[bound_name]
        inclusive = contract.get(f"{bound_name}_inclusive", True)
        if not comparison(value, bound):
            return False
        if inclusive is False and value == bound:
            return False
    return True


def _representation_family(parameter: str) -> str:
    if ".jungle_" in parameter:
        return "jungle"
    if ".eed_" in parameter:
        return "eed"
    if parameter.startswith("wdl.") or ".wdl_" in parameter:
        return "wdl"
    if ".reproduce_" in parameter:
        return "reproduction"
    if ".patch_" in parameter:
        return "patch"
    return "shared_runtime_config"


@dataclass(frozen=True, slots=True)
class QualifiedExecutionEnvelopeV1:
    """The only release-local truth for route and admitted variability."""

    envelope_id: str
    provider_id: str
    profile_id: str
    execution_route_sha256: str
    qualification_key_sha256: str
    artifact_sha256s: Mapping[str, str]
    experiment_variable_contracts: tuple[Mapping[str, Any], ...]
    fixed_parameter_values: Mapping[str, tuple[Any, ...]]
    admitted_batch_range: tuple[int, int]
    admitted_generation_range: tuple[int, int]
    admitted_ticks_per_generation: tuple[int, ...]
    checkpoint_storage_modes: tuple[str, ...]
    resource_contract: Mapping[str, Any]
    representation_contract: Mapping[str, Any]
    reporting_modes: tuple[bool, ...] = (True, False)
    time_contract: Mapping[str, Any] = field(
        default_factory=lambda: {
            "authority": "legacy_generation_route_v1",
            "arbitrary_positive_terminal_tick": False,
            "arbitrary_checkpoint_ticks": False,
        }
    )
    qualification_dimensions: tuple[str, ...] = ("BOUNDED_ROUTE_QUALIFIED",)
    qualification_evidence_binding: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name, value in (
            ("envelope_id", self.envelope_id),
            ("provider_id", self.provider_id),
            ("profile_id", self.profile_id),
        ):
            if not value:
                raise QualifiedExecutionEnvelopeError(f"{name} must be non-empty")
        if (
            len(self.execution_route_sha256) != 64
            or len(self.qualification_key_sha256) != 64
        ):
            raise QualifiedExecutionEnvelopeError(
                "route and qualification bindings must be SHA-256 values"
            )
        names = [str(row.get("name")) for row in self.experiment_variable_contracts]
        if len(names) != len(set(names)) or any(not name for name in names):
            raise QualifiedExecutionEnvelopeError(
                "experiment-variable contract names must be unique"
            )
        representation = self.representation_contract
        if representation.get("schema_version") != (
            EXPERIMENT_REPRESENTATION_CONTRACT_SCHEMA_VERSION
        ):
            raise QualifiedExecutionEnvelopeError(
                "experiment representation contract schema is unsupported"
            )
        if representation.get("policy") not in {
            "shared_scalar_configs_with_named_per_cell_exceptions_v1",
            "shared_scalar_configs_with_qualified_per_cell_exceptions_v1",
        }:
            raise QualifiedExecutionEnvelopeError(
                "experiment representation policy is unsupported"
            )
        mixed = representation.get("per_cell_mixed_parameter_names")
        uniform = representation.get("batch_uniform_parameter_names")
        if not isinstance(mixed, list) or not isinstance(uniform, list):
            raise QualifiedExecutionEnvelopeError(
                "experiment representation parameter partition is malformed"
            )
        mixed_names = {str(name) for name in mixed}
        uniform_names = {str(name) for name in uniform}
        if (
            len(mixed_names) != len(mixed)
            or len(uniform_names) != len(uniform)
            or mixed_names & uniform_names
            or mixed_names | uniform_names != set(names)
        ):
            raise QualifiedExecutionEnvelopeError(
                "experiment representation parameter partition is not exact"
            )
        if (
            representation.get("execution_route_sha256")
            != self.execution_route_sha256
            or representation.get("artifact_sha256s")
            != dict(sorted(self.artifact_sha256s.items()))
        ):
            raise QualifiedExecutionEnvelopeError(
                "experiment representation authority differs from route artifacts"
            )
        for lower, upper in (
            self.admitted_batch_range,
            self.admitted_generation_range,
        ):
            if (
                type(lower) is not int
                or type(upper) is not int
                or not 1 <= lower <= upper
            ):
                raise QualifiedExecutionEnvelopeError("admitted range is invalid")
        if not self.admitted_ticks_per_generation or any(
            type(value) is not int or value < 1
            for value in self.admitted_ticks_per_generation
        ):
            raise QualifiedExecutionEnvelopeError("admitted tick set is invalid")
        budgets = self.resource_contract.get("declared_budgets")
        if not isinstance(budgets, Mapping) or any(
            type(value) is not int or value <= 0 for value in budgets.values()
        ):
            raise QualifiedExecutionEnvelopeError(
                "release resource budgets must be positive integers"
            )
        if self.resource_contract.get("admission_policy") != (
            "release_local_preflight_current_resource_state_v1"
        ):
            raise QualifiedExecutionEnvelopeError(
                "release resource admission policy is unsupported"
            )
        if (
            not isinstance(
                self.resource_contract.get("provider_resource_estimator_id"), str
            )
            or not self.resource_contract["provider_resource_estimator_id"]
        ):
            raise QualifiedExecutionEnvelopeError(
                "release resource estimator identity is absent"
            )
        if any(type(value) is not bool for value in self.reporting_modes):
            raise QualifiedExecutionEnvelopeError(
                "release reporting modes must be exact booleans"
            )
        if self.time_contract.get("authority") not in {
            "legacy_generation_route_v1",
            "absolute_tick_v1",
        }:
            raise QualifiedExecutionEnvelopeError("release time authority is unsupported")
        allowed_dimensions = {
            "BOUNDED_ROUTE_QUALIFIED",
            "LONG_HORIZON_RUNABILITY_QUALIFIED",
            "SCIENCE_ROUTE_QUALIFIED",
        }
        if any(
            not isinstance(value, str) or value not in allowed_dimensions
            for value in self.qualification_dimensions
        ):
            raise QualifiedExecutionEnvelopeError(
                "qualification dimensions are malformed"
            )
        if self.qualification_dimensions not in {
            (),
            (BOUNDED_ROUTE_QUALIFIED,),
            (BOUNDED_ROUTE_QUALIFIED, LONG_HORIZON_RUNABILITY_QUALIFIED),
            (
                BOUNDED_ROUTE_QUALIFIED,
                LONG_HORIZON_RUNABILITY_QUALIFIED,
                SCIENCE_ROUTE_QUALIFIED,
            ),
        }:
            raise QualifiedExecutionEnvelopeError(
                "qualification dimensions must be canonical and monotonic"
            )
        if "SCIENCE_ROUTE_QUALIFIED" in self.qualification_dimensions and not {
            "BOUNDED_ROUTE_QUALIFIED",
            "LONG_HORIZON_RUNABILITY_QUALIFIED",
        }.issubset(self.qualification_dimensions):
            raise QualifiedExecutionEnvelopeError(
                "science qualification requires bounded and long-horizon dimensions"
            )
        if LONG_HORIZON_RUNABILITY_QUALIFIED in self.qualification_dimensions:
            try:
                binding = validate_qualification_evidence_binding_v1(
                    self.qualification_evidence_binding,
                    expected_identity={
                        "execution_route_sha256": self.execution_route_sha256,
                        "provider_id": self.provider_id,
                        "profile_id": self.profile_id,
                    },
                )
            except LongHorizonQualificationError as exc:
                raise QualifiedExecutionEnvelopeError(
                    f"long-horizon qualification binding is invalid: {exc}"
                ) from exc
            if not (
                set(self.qualification_dimensions) - {BOUNDED_ROUTE_QUALIFIED}
            ).issubset(
                binding["qualification_dimensions"]
            ):
                raise QualifiedExecutionEnvelopeError(
                    "envelope dimensions exceed its qualification evidence binding"
                )
        elif self.qualification_evidence_binding:
            raise QualifiedExecutionEnvelopeError(
                "bounded-only envelope must not carry long-horizon evidence authority"
            )

    def identity_payload(self) -> dict[str, Any]:
        legacy_time = self.time_contract.get("authority") == (
            "legacy_generation_route_v1"
        )
        qualification_time = (
            {"ticks_per_generation": list(self.admitted_ticks_per_generation)}
            if legacy_time
            else {
                "source_window_ticks": list(
                    self.admitted_ticks_per_generation
                )
            }
        )
        provenance_time = (
            {
                "generations": list(self.admitted_generation_range),
                "checkpoint_milestones": "inside generation horizon",
            }
            if legacy_time
            else {
                "terminal_ticks": "arbitrary_positive_int64",
                "checkpoint_ticks": "strictly_increasing_inside_terminal_tick",
            }
        )
        payload = {
            "schema_version": QUALIFIED_EXECUTION_ENVELOPE_SCHEMA_VERSION,
            "envelope_id": self.envelope_id,
            "provider_id": self.provider_id,
            "profile_id": self.profile_id,
            "execution_route_sha256": self.execution_route_sha256,
            "qualification_key_sha256": self.qualification_key_sha256,
            "artifact_sha256s": dict(sorted(self.artifact_sha256s.items())),
            "resource_contract": dict(self.resource_contract),
            "representation_contract": dict(self.representation_contract),
            "dimensions": {
                "EXPERIMENT_VARIABLE": [
                    dict(row) for row in self.experiment_variable_contracts
                ],
                "QUALIFICATION_AFFECTING": {
                    "provider_id": self.provider_id,
                    "profile_id": self.profile_id,
                    "execution_route_sha256": self.execution_route_sha256,
                    "qualification_key_sha256": self.qualification_key_sha256,
                    "artifact_sha256s": dict(sorted(self.artifact_sha256s.items())),
                    "fixed_parameter_values": {
                        name: list(values)
                        for name, values in sorted(self.fixed_parameter_values.items())
                    },
                    **qualification_time,
                },
                "PROVENANCE_ONLY": {
                    "cell_values": "admitted experiment variables",
                    "seeds": "nonnegative integers",
                    "cell_ordering": "new plan identity",
                    **provenance_time,
                    "batch_cells": list(self.admitted_batch_range),
                    "checkpoint_storage_modes": list(self.checkpoint_storage_modes),
                    "reporting_modes": list(self.reporting_modes),
                    "run_and_output_identity": "new evidence identity",
                },
            },
        }
        if not legacy_time:
            payload["time_contract"] = dict(self.time_contract)
        if (
            self.qualification_dimensions != (BOUNDED_ROUTE_QUALIFIED,)
            or not legacy_time
        ):
            payload["qualification_dimensions"] = list(self.qualification_dimensions)
        if self.qualification_evidence_binding:
            payload["qualification_evidence_binding"] = dict(
                self.qualification_evidence_binding
            )
        return payload

    @property
    def envelope_sha256(self) -> str:
        return canonical_sha256(self.identity_payload())

    def payload(self) -> dict[str, Any]:
        return {**self.identity_payload(), "envelope_sha256": self.envelope_sha256}

    def validate_experiment_representability(
        self, plan: Any
    ) -> dict[str, Any]:
        """Prove each physical batch fits the artifact's actual scalar ABI."""

        payload = _plan_payload(plan)
        raw_cells = payload.get("cells")
        batches = payload.get("physical_batches")
        if not isinstance(raw_cells, list) or not isinstance(batches, list):
            raise QualifiedExecutionEnvelopeError("compiled plan shape is malformed")
        cells = {
            str(cell.get("cell_id")): cell
            for cell in raw_cells
            if isinstance(cell, Mapping) and isinstance(cell.get("cell_id"), str)
        }
        if len(cells) != len(raw_cells):
            raise QualifiedExecutionEnvelopeError("cell identities are malformed")
        uniform_names = tuple(
            str(name)
            for name in self.representation_contract[
                "batch_uniform_parameter_names"
            ]
        )
        violations: list[dict[str, Any]] = []
        checked_batches = 0
        for batch in batches:
            batch_id = batch.get("batch_id") if isinstance(batch, Mapping) else None
            cell_ids = batch.get("cell_ids") if isinstance(batch, Mapping) else None
            if (
                not isinstance(batch_id, str)
                or not isinstance(cell_ids, Sequence)
                or isinstance(cell_ids, (str, bytes))
                or any(str(cell_id) not in cells for cell_id in cell_ids)
            ):
                raise QualifiedExecutionEnvelopeError("physical batch is malformed")
            checked_batches += 1
            for parameter in uniform_names:
                values: list[Any] = []
                missing: list[str] = []
                for cell_id in cell_ids:
                    parameters = cells[str(cell_id)].get("parameters")
                    if not isinstance(parameters, Mapping) or parameter not in parameters:
                        missing.append(str(cell_id))
                    else:
                        values.append(parameters[parameter])
                if len(missing) == len(cell_ids):
                    continue
                unique = _canonical_unique(values)
                if missing or len(unique) > 1:
                    violations.append(
                        {
                            "code": "mixed_parameter_requires_batch_uniform",
                            "parameter": parameter,
                            "physical_batch_id": batch_id,
                            "representation_family": _representation_family(parameter),
                            "observed_values": list(unique),
                            "missing_cell_ids": missing,
                        }
                    )
        contract_sha256 = canonical_sha256(dict(self.representation_contract))
        core = {
            "schema_version": (
                EXPERIMENT_REPRESENTABILITY_RECEIPT_SCHEMA_VERSION
            ),
            "status": "FAIL" if violations else "PASS",
            "failure_class": (
                "QRR_REPRESENTABILITY_BLOCKED" if violations else None
            ),
            "envelope_sha256": self.envelope_sha256,
            "execution_route_sha256": self.execution_route_sha256,
            "representation_contract_sha256": contract_sha256,
            "plan_sha256": payload.get("plan_sha256"),
            "checked_physical_batch_count": checked_batches,
            "checked_batch_uniform_parameter_count": len(uniform_names),
            "violations": violations,
        }
        receipt = {**core, "receipt_sha256": canonical_sha256(core)}
        if violations:
            unsupported = ", ".join(
                f"{row['parameter']}[{row['representation_family']}]"
                for row in violations
            )
            raise ExperimentRepresentabilityError(
                "QRR_REPRESENTABILITY_BLOCKED: experiment requires unsupported "
                f"mixed per-cell capabilities: {unsupported}",
                receipt=receipt,
            )
        return receipt

    def validate_plan(self, plan: Any, *, provider: Any) -> dict[str, Any]:
        payload = _plan_payload(plan)
        if payload.get("resolved_profile") != self.profile_id:
            raise QualifiedExecutionEnvelopeError("plan profile differs from release")
        if getattr(provider, "provider_id", None) != self.provider_id:
            raise QualifiedExecutionEnvelopeError("provider differs from release")
        if payload.get("qualification_key_sha256") != self.qualification_key_sha256:
            raise QualifiedExecutionEnvelopeError(
                "qualification key differs from release"
            )
        if _artifact_sha256s(payload) != dict(self.artifact_sha256s):
            raise QualifiedExecutionEnvelopeError(
                "artifact identity differs from release"
            )
        execution = payload.get("execution")
        cells = payload.get("cells")
        batches = payload.get("physical_batches")
        if (
            not isinstance(execution, Mapping)
            or not isinstance(cells, list)
            or not isinstance(batches, list)
        ):
            raise QualifiedExecutionEnvelopeError("compiled plan shape is malformed")
        tick_native = execution.get("execution_route_identity") == (
            "tick_native_execution_route_v1"
        )
        if tick_native:
            if self.time_contract.get("authority") != "absolute_tick_v1":
                raise QualifiedExecutionEnvelopeError(
                    "absolute-tick request differs from release time authority"
                )
            if any(
                name in execution
                for name in (
                    "generations",
                    "ticks_per_generation",
                    "physical_window_generations",
                    "internal_lifecycle_schedule",
                    "internal_lifecycle_interval_ticks",
                    "refill_chunk_ticks",
                )
            ):
                raise QualifiedExecutionEnvelopeError(
                    "tick-native plan contains forbidden compatibility aliases"
                )
            try:
                from engine.campaigns.tick_native import TickNativeExecutionRequestV1

                tick_request = TickNativeExecutionRequestV1.from_execution(
                    dict(execution)
                )
            except (ImportError, ValueError) as exc:
                raise QualifiedExecutionEnvelopeError(
                    f"absolute-tick schedule is invalid: {exc}"
                ) from exc
            control = execution.get("tick_native_control")
            expected_control = self.time_contract.get("tick_native_control")
            if isinstance(expected_control, Mapping):
                if not isinstance(control, Mapping):
                    raise QualifiedExecutionEnvelopeError(
                        "tick-native plan is missing its canonical control contract"
                    )
                if any(
                    name in control
                    for name in (
                        "refill_chunk_ticks",
                        "legacy_turnover_interval_ticks",
                    )
                ):
                    raise QualifiedExecutionEnvelopeError(
                        "tick-native control contains forbidden compatibility aliases"
                    )
                for name in (
                    "resident_owner_span_ticks",
                    "native_dispatch_window_ticks",
                    "source_window_ticks",
                    "event_window_capacity",
                ):
                    if control.get(name) != expected_control.get(name):
                        raise QualifiedExecutionEnvelopeError(
                            f"tick-native control {name} differs from the release"
                        )
            else:
                raise QualifiedExecutionEnvelopeError(
                    "tick-native release lacks its canonical control contract"
                )
            del tick_request
        else:
            generations = execution.get("generations")
            ticks = execution.get("ticks_per_generation")
            if type(generations) is not int or not (
                self.admitted_generation_range[0]
                <= generations
                <= self.admitted_generation_range[1]
            ):
                raise QualifiedExecutionEnvelopeError(
                    "generation horizon is outside the release envelope"
                )
            if ticks not in self.admitted_ticks_per_generation:
                raise QualifiedExecutionEnvelopeError("tick extent differs from release")
            checkpoints = execution.get("checkpoints")
            if not isinstance(checkpoints, list) or any(
                type(value) is not int or value < 0 or value > generations
                for value in checkpoints
            ):
                raise QualifiedExecutionEnvelopeError(
                    "checkpoint milestones are invalid"
                )
        storage_mode = str(execution.get("checkpoint_storage_mode", "disk"))
        if storage_mode not in self.checkpoint_storage_modes:
            raise QualifiedExecutionEnvelopeError(
                "checkpoint storage mode differs from release"
            )
        plan_budgets = execution.get("budgets")
        if not isinstance(plan_budgets, Mapping) or dict(plan_budgets) != dict(
            self.resource_contract["declared_budgets"]
        ):
            raise QualifiedExecutionEnvelopeError(
                "plan resource budgets differ from the release contract"
            )
        variable_contracts = {
            str(row["name"]): row for row in self.experiment_variable_contracts
        }
        fixed = dict(self.fixed_parameter_values)
        for cell in cells:
            parameters = cell.get("parameters") if isinstance(cell, Mapping) else None
            seeds = cell.get("seeds") if isinstance(cell, Mapping) else None
            if not isinstance(parameters, Mapping) or not isinstance(seeds, Mapping):
                raise QualifiedExecutionEnvelopeError("cell bindings are malformed")
            if set(parameters) - (set(variable_contracts) | set(fixed)) or not set(
                fixed
            ) <= set(parameters):
                raise QualifiedExecutionEnvelopeError(
                    "cell parameter ownership differs from release"
                )
            for name, value in parameters.items():
                contract = variable_contracts.get(str(name))
                if contract is not None:
                    value_contract = contract.get("value_contract")
                    if not isinstance(
                        value_contract, Mapping
                    ) or not _value_contract_accepts(value_contract, value):
                        raise QualifiedExecutionEnvelopeError(
                            f"experiment variable {name!r} is outside its admitted contract"
                        )
                elif value not in fixed[str(name)]:
                    raise QualifiedExecutionEnvelopeError(
                        f"qualification-affecting parameter {name!r} differs"
                    )
            if any(type(seed) is not int or seed < 0 for seed in seeds.values()):
                raise QualifiedExecutionEnvelopeError("cell seeds are malformed")
        representability = self.validate_experiment_representability(payload)
        request_rows: dict[str, str] = {}
        resident_requests: list[Any] = []
        for batch in batches:
            batch_id = batch.get("batch_id") if isinstance(batch, Mapping) else None
            batch_cells = batch.get("cell_ids") if isinstance(batch, Mapping) else None
            if (
                not isinstance(batch_id, str)
                or not isinstance(batch_cells, Sequence)
                or isinstance(batch_cells, (str, bytes))
            ):
                raise QualifiedExecutionEnvelopeError("physical batch is malformed")
            if not (
                self.admitted_batch_range[0]
                <= len(batch_cells)
                <= self.admitted_batch_range[1]
            ):
                raise QualifiedExecutionEnvelopeError(
                    "physical batch is outside the release envelope"
                )
            bound = build_execution_request_v1(provider, payload, batch_id)
            if bound.execution_route_sha256 != self.execution_route_sha256:
                raise QualifiedExecutionEnvelopeError(
                    "bound execution request differs from the qualified route"
                )
            request_rows[batch_id] = bound.bound_request_sha256
            resident_requests.append(bound.resident_request)
        estimate_resources = getattr(provider, "estimate_resources", None)
        if not callable(estimate_resources):
            raise QualifiedExecutionEnvelopeError(
                "release provider has no resource estimator"
            )
        resource_estimate = estimate_resources(payload, resident_requests)
        if not isinstance(resource_estimate, Mapping) or resource_estimate.get(
            "resource_estimator_id"
        ) != self.resource_contract.get("provider_resource_estimator_id"):
            raise QualifiedExecutionEnvelopeError(
                "provider resource estimator differs from the release contract"
            )
        return {
            "schema_version": "eiel_qrr_plan_validation_v1",
            "envelope_sha256": self.envelope_sha256,
            "plan_sha256": payload.get("plan_sha256"),
            "representability": representability,
            "representability_receipt_sha256": representability["receipt_sha256"],
            "bound_request_sha256s": request_rows,
            "valid": True,
        }


def build_qualified_execution_envelope_v1(
    plan: Any,
    *,
    provider: Any,
    profile: RuntimeProfileV1,
    qualification_authority: Mapping[str, Any] | None = None,
) -> QualifiedExecutionEnvelopeV1:
    """Derive, never independently restate, the release capability contract."""

    payload = _plan_payload(plan)
    batches = payload.get("physical_batches")
    cells = payload.get("cells")
    qualification = payload.get("qualification_key")
    if (
        not isinstance(batches, list)
        or not batches
        or not isinstance(cells, list)
        or not cells
    ):
        raise QualifiedExecutionEnvelopeError("reference plan has no physical cells")
    if not isinstance(qualification, Mapping):
        raise QualifiedExecutionEnvelopeError("reference plan has no qualification key")
    first_batch_id = batches[0].get("batch_id")
    if not isinstance(first_batch_id, str):
        raise QualifiedExecutionEnvelopeError("reference batch identity is malformed")
    bound = build_execution_request_v1(provider, payload, first_batch_id)
    if bound.execution_route_sha256 is None:
        raise QualifiedExecutionEnvelopeError("reference route has no exact identity")
    qualification_sha256 = payload.get("qualification_key_sha256")
    if not isinstance(qualification_sha256, str) or len(qualification_sha256) != 64:
        raise QualifiedExecutionEnvelopeError(
            "reference qualification key is not bound"
        )
    variability = qualification.get("variability_envelope")
    contracts = (
        variability.get("parameter_contracts")
        if isinstance(variability, Mapping)
        else None
    )
    if not isinstance(contracts, list) or not contracts:
        raise QualifiedExecutionEnvelopeError(
            "reference qualification has no variability envelope"
        )
    variable_names = {
        str(row.get("name")) for row in contracts if isinstance(row, Mapping)
    }
    if len(variable_names) != len(contracts):
        raise QualifiedExecutionEnvelopeError(
            "reference variability contracts are malformed"
        )
    parameter_names = {
        str(name)
        for cell in cells
        for name in dict(cell.get("parameters", {}))
        if isinstance(cell, Mapping)
    }
    fixed_values = {
        name: _canonical_unique(
            [
                dict(cell["parameters"])[name]
                for cell in cells
                if name in dict(cell.get("parameters", {}))
            ]
        )
        for name in sorted(parameter_names - variable_names)
    }
    artifacts = _artifact_sha256s(payload)
    mixed_parameter_names = sorted(
        variable_names & PRODUCTION_HETEROGENEOUS_PARAMETER_NAMES_V1
    )
    representation_contract = {
        "schema_version": EXPERIMENT_REPRESENTATION_CONTRACT_SCHEMA_VERSION,
        "policy": "shared_scalar_configs_with_qualified_per_cell_exceptions_v1",
        "execution_route_sha256": bound.execution_route_sha256,
        "artifact_sha256s": artifacts,
        "per_cell_mixed_parameter_names": mixed_parameter_names,
        "batch_uniform_parameter_names": sorted(
            variable_names - set(mixed_parameter_names)
        ),
        "native_mechanics_binding_id": (
            "direct_resident_wdl_jungle_eed_shared_scalar_abi_v1"
        ),
        "constraint_sources": [
            "direct_resident_source_allocator_cell_ordered_controls_v1",
            "direct_resident_wdl_jungle_cell_ordered_controls_v1",
        ],
    }
    admitted_batch = qualification.get("admitted_batch_range")
    if (
        not isinstance(admitted_batch, Sequence)
        or isinstance(admitted_batch, (str, bytes))
        or len(admitted_batch) != 2
    ):
        admitted_batch = (
            profile.qualified_batch_min,
            profile.qualified_batch_max,
        )
    if bound.execution_route_contract is not None:
        route_components = bound.execution_route_contract.get("components")
        native_component = (
            next(
                (
                    row
                    for row in route_components
                    if isinstance(row, Mapping)
                    and row.get("component_id") == "native_mechanics_binding"
                ),
                None,
            )
            if isinstance(route_components, list)
            else None
        )
        route_batch_max = (
            native_component.get("batch_max")
            if isinstance(native_component, Mapping)
            else None
        )
        if type(route_batch_max) is int and route_batch_max >= 1:
            admitted_batch = (1, route_batch_max)
    execution = payload["execution"]
    tick_native = execution.get("execution_route_identity") == (
        "tick_native_execution_route_v1"
    )
    control = execution.get("tick_native_control")
    if tick_native and (
        not isinstance(control, Mapping)
        or "refill_chunk_ticks" in control
        or "legacy_turnover_interval_ticks" in control
        or type(control.get("source_window_ticks")) is not int
        or control.get("source_window_ticks") < 1
    ):
        raise QualifiedExecutionEnvelopeError(
            "tick-native reference plan lacks a clean source-window control"
        )
    source_window_ticks = (
        int(control["source_window_ticks"])
        if tick_native and isinstance(control, Mapping)
        else None
    )
    generation_min = int(getattr(provider, "resident_segment_generation_min", 1))
    generation_max = int(
        getattr(
            provider,
            "resident_segment_generation_max",
            (
                (
                    int(execution["total_ticks"]) + source_window_ticks - 1
                )
                // source_window_ticks
                if tick_native
                else execution["generations"]
            ),
        )
    )
    ticks = (
        (source_window_ticks,)
        if tick_native and isinstance(control, Mapping)
        else (int(execution["ticks_per_generation"]),)
    )
    budgets = payload["execution"].get("budgets")
    if not isinstance(budgets, Mapping) or not budgets:
        raise QualifiedExecutionEnvelopeError(
            "reference plan has no declared resource budgets"
        )
    estimate_resources = getattr(provider, "estimate_resources", None)
    if not callable(estimate_resources):
        raise QualifiedExecutionEnvelopeError(
            "reference provider has no resource estimator"
        )
    resource_estimate = estimate_resources(payload, (bound.resident_request,))
    estimator_id = (
        resource_estimate.get("resource_estimator_id")
        if isinstance(resource_estimate, Mapping)
        else None
    )
    if not isinstance(estimator_id, str) or not estimator_id:
        raise QualifiedExecutionEnvelopeError(
            "reference provider resource estimator has no identity"
        )
    envelope_id = (
        variability.get("envelope_id") if isinstance(variability, Mapping) else None
    )
    if not isinstance(envelope_id, str) or not envelope_id:
        raise QualifiedExecutionEnvelopeError("variability envelope identity is absent")
    raw_dimensions = (
        qualification_authority.get("qualification_dimensions")
        if isinstance(qualification_authority, Mapping)
        else None
    )
    qualification_dimensions = (
        tuple(str(value) for value in raw_dimensions)
        if isinstance(raw_dimensions, Sequence)
        and not isinstance(raw_dimensions, (str, bytes))
        else (() if tick_native else (BOUNDED_ROUTE_QUALIFIED,))
    )
    evidence_binding = (
        qualification_authority.get("qualification_evidence_binding")
        if isinstance(qualification_authority, Mapping)
        else None
    )
    evidence_binding = (
        dict(evidence_binding) if isinstance(evidence_binding, Mapping) else {}
    )
    if LONG_HORIZON_RUNABILITY_QUALIFIED in qualification_dimensions:
        expected_binding = {
            "execution_route_sha256": bound.execution_route_sha256,
            "provider_id": str(provider.provider_id),
            "profile_id": profile.profile_id,
            "plan_sha256": payload.get("plan_sha256"),
        }
        try:
            validated_binding = validate_qualification_evidence_binding_v1(
                evidence_binding, expected_identity=expected_binding
            )
        except LongHorizonQualificationError as exc:
            raise QualifiedExecutionEnvelopeError(
                f"reference qualification evidence binding is invalid: {exc}"
            ) from exc
        for artifact_id, digest in validated_binding["artifact_sha256s"].items():
            if artifacts.get(artifact_id) != digest:
                raise QualifiedExecutionEnvelopeError(
                    f"qualification evidence artifact differs: {artifact_id}"
                )
        evidence_budget = validated_binding["resource_limits"]["gpu_budget_bytes"]
        plan_budget = budgets.get("vram_bytes")
        if type(plan_budget) is not int or evidence_budget != plan_budget:
            raise QualifiedExecutionEnvelopeError(
                "qualification evidence GPU budget differs from the reference plan"
            )
    return QualifiedExecutionEnvelopeV1(
        envelope_id=envelope_id,
        provider_id=str(provider.provider_id),
        profile_id=profile.profile_id,
        execution_route_sha256=bound.execution_route_sha256,
        qualification_key_sha256=qualification_sha256,
        artifact_sha256s=artifacts,
        experiment_variable_contracts=tuple(
            dict(row) for row in contracts if isinstance(row, Mapping)
        ),
        fixed_parameter_values=fixed_values,
        admitted_batch_range=(int(admitted_batch[0]), int(admitted_batch[1])),
        admitted_generation_range=(generation_min, generation_max),
        admitted_ticks_per_generation=ticks,
        checkpoint_storage_modes=(
            str(payload["execution"].get("checkpoint_storage_mode", "disk")),
        ),
        resource_contract={
            "declared_budgets": dict(sorted(budgets.items())),
            "provider_resource_estimator_id": estimator_id,
            "reference_provider_estimate": dict(resource_estimate),
            "admission_policy": ("release_local_preflight_current_resource_state_v1"),
        },
        representation_contract=representation_contract,
        time_contract=(
            {
                "authority": "absolute_tick_v1",
                "cursor_semantics": "committed_half_open_[0,next_tick)",
                "arbitrary_positive_terminal_tick": True,
                "arbitrary_checkpoint_ticks": True,
                "fresh_owner_partial_generation_restore": True,
                "source_window_ticks": ticks[0],
                **(
                    {
                        "tick_native_control": {
                            name: control.get(name)
                            for name in (
                                "resident_owner_span_ticks",
                                "native_dispatch_window_ticks",
                                "source_window_ticks",
                                "event_window_capacity",
                            )
                        }
                    }
                    if isinstance(control, Mapping)
                    else {}
                ),
            }
            if tick_native
            else {
                "authority": "legacy_generation_route_v1",
                "arbitrary_positive_terminal_tick": False,
                "arbitrary_checkpoint_ticks": False,
            }
        ),
        qualification_dimensions=qualification_dimensions,
        qualification_evidence_binding=evidence_binding,
    )


def load_qualified_execution_envelope_v1(
    value: Mapping[str, Any],
) -> QualifiedExecutionEnvelopeV1:
    """Validate the frozen manifest projection without consulting current main."""

    if value.get("schema_version") != QUALIFIED_EXECUTION_ENVELOPE_SCHEMA_VERSION:
        raise QualifiedExecutionEnvelopeError(
            "qualified execution envelope schema is unsupported"
        )
    dimensions = value.get("dimensions")
    qualification_affecting = (
        dimensions.get("QUALIFICATION_AFFECTING")
        if isinstance(dimensions, Mapping)
        else None
    )
    provenance = (
        dimensions.get("PROVENANCE_ONLY") if isinstance(dimensions, Mapping) else None
    )
    experiment = (
        dimensions.get("EXPERIMENT_VARIABLE")
        if isinstance(dimensions, Mapping)
        else None
    )
    if (
        not isinstance(qualification_affecting, Mapping)
        or not isinstance(provenance, Mapping)
        or not isinstance(experiment, list)
    ):
        raise QualifiedExecutionEnvelopeError(
            "qualified execution envelope dimensions are malformed"
        )
    batch_range = provenance.get("batch_cells")
    generation_range = provenance.get("generations")
    if not isinstance(generation_range, list):
        generation_range = [1, (1 << 63) - 1]
    fixed = qualification_affecting.get("fixed_parameter_values")
    storage = provenance.get("checkpoint_storage_modes")
    reporting = provenance.get("reporting_modes")
    resource_contract = value.get("resource_contract")
    representation_contract = value.get("representation_contract")
    time_contract = value.get("time_contract") or {
        "authority": "legacy_generation_route_v1",
        "arbitrary_positive_terminal_tick": False,
        "arbitrary_checkpoint_ticks": False,
    }
    absolute_tick = time_contract.get("authority") == "absolute_tick_v1"
    ticks = (
        qualification_affecting.get("source_window_ticks")
        if absolute_tick
        else qualification_affecting.get("ticks_per_generation")
    )
    if absolute_tick and "internal_lifecycle_interval_ticks" in qualification_affecting:
        raise QualifiedExecutionEnvelopeError(
            "absolute-tick envelope contains a forbidden lifecycle alias"
        )
    if absolute_tick and "refill_chunk_ticks" in time_contract:
        raise QualifiedExecutionEnvelopeError(
            "absolute-tick envelope contains a forbidden refill alias"
        )
    if (
        any(
            not isinstance(row, list) or len(row) != length
            for row, length in (
                (batch_range, 2),
                (generation_range, 2),
            )
        )
        or not all(isinstance(row, list) for row in (ticks, storage, reporting))
        or any(type(row) is not bool for row in reporting or ())
        or not isinstance(fixed, Mapping)
        or not isinstance(resource_contract, Mapping)
        or not isinstance(representation_contract, Mapping)
        or not isinstance(time_contract, Mapping)
    ):
        raise QualifiedExecutionEnvelopeError(
            "qualified execution envelope ranges are malformed"
        )
    envelope = QualifiedExecutionEnvelopeV1(
        envelope_id=str(value.get("envelope_id") or ""),
        provider_id=str(value.get("provider_id") or ""),
        profile_id=str(value.get("profile_id") or ""),
        execution_route_sha256=str(value.get("execution_route_sha256") or ""),
        qualification_key_sha256=str(value.get("qualification_key_sha256") or ""),
        artifact_sha256s=dict(value.get("artifact_sha256s") or {}),
        experiment_variable_contracts=tuple(
            dict(row) for row in experiment if isinstance(row, Mapping)
        ),
        fixed_parameter_values={
            str(name): tuple(values)
            for name, values in fixed.items()
            if isinstance(values, list)
        },
        admitted_batch_range=(int(batch_range[0]), int(batch_range[1])),
        admitted_generation_range=(
            int(generation_range[0]),
            int(generation_range[1]),
        ),
        admitted_ticks_per_generation=tuple(int(row) for row in ticks),
        checkpoint_storage_modes=tuple(str(row) for row in storage),
        resource_contract=dict(resource_contract),
        representation_contract=dict(representation_contract),
        reporting_modes=tuple(reporting),
        time_contract=dict(time_contract),
        qualification_dimensions=tuple(
            str(row)
            for row in value.get(
                "qualification_dimensions",
                (
                    (BOUNDED_ROUTE_QUALIFIED,)
                    if time_contract.get("authority") == "legacy_generation_route_v1"
                    else ()
                ),
            )
        ),
        qualification_evidence_binding=dict(
            value.get("qualification_evidence_binding") or {}
        ),
    )
    if len(envelope.experiment_variable_contracts) != len(experiment):
        raise QualifiedExecutionEnvelopeError(
            "experiment-variable envelope rows are malformed"
        )
    if value.get("envelope_sha256") != envelope.envelope_sha256:
        raise QualifiedExecutionEnvelopeError(
            "qualified execution envelope SHA-256 differs"
        )
    return envelope


__all__ = [
    "EXPERIMENT_REPRESENTATION_CONTRACT_SCHEMA_VERSION",
    "EXPERIMENT_REPRESENTABILITY_RECEIPT_SCHEMA_VERSION",
    "QUALIFIED_EXECUTION_ENVELOPE_SCHEMA_VERSION",
    "ExperimentRepresentabilityError",
    "QualifiedExecutionEnvelopeError",
    "QualifiedExecutionEnvelopeV1",
    "build_qualified_execution_envelope_v1",
    "load_qualified_execution_envelope_v1",
]
