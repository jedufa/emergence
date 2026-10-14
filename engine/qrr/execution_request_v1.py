"""One canonical construction and derivation boundary for resident requests."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
import inspect
from typing import Any, Final

from engine.campaigns.canonical import canonical_sha256
from engine.campaigns.contracts import CampaignPlanV1
from engine.campaigns.providers import ResidentBatchRequestV1


BOUND_EXECUTION_REQUEST_SCHEMA_VERSION: Final[str] = "eiel_bound_execution_request_v1"
EXECUTION_REQUEST_DERIVATION_SCHEMA_VERSION: Final[str] = (
    "eiel_execution_request_derivation_v1"
)


class ExecutionRequestError(ValueError):
    """The provider request cannot be represented by the canonical contract."""


def _plan_payload(plan: CampaignPlanV1 | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(plan, CampaignPlanV1):
        return plan.payload()
    if not isinstance(plan, Mapping):
        raise ExecutionRequestError("campaign plan must be a mapping or CampaignPlanV1")
    return dict(plan)


@dataclass(frozen=True, slots=True)
class ExecutionRequestDerivationV1:
    """A complete, explicit derivation from one provider-owned batch request."""

    batch_id: str
    cell_ids: tuple[str, ...]
    generations: int
    checkpoint_milestones: tuple[int, ...]
    reporting_enabled: bool | None = None
    preserve_generation_zero: bool = True

    def __post_init__(self) -> None:
        if not self.batch_id:
            raise ExecutionRequestError("derived batch_id must be non-empty")
        if not self.cell_ids or len(self.cell_ids) != len(set(self.cell_ids)):
            raise ExecutionRequestError("derived cell_ids must be non-empty and unique")
        if type(self.generations) is not int or self.generations < 1:
            raise ExecutionRequestError(
                "derived generations must be a positive integer"
            )
        if any(
            type(value) is not int or value < 0 or value > self.generations
            for value in self.checkpoint_milestones
        ):
            raise ExecutionRequestError(
                "derived checkpoints must be integers inside the requested horizon"
            )
        if len(self.checkpoint_milestones) != len(set(self.checkpoint_milestones)):
            raise ExecutionRequestError("derived checkpoints must be unique")
        if (
            self.reporting_enabled is not None
            and type(self.reporting_enabled) is not bool
        ):
            raise ExecutionRequestError("reporting_enabled must be an exact boolean")
        if type(self.preserve_generation_zero) is not bool:
            raise ExecutionRequestError(
                "preserve_generation_zero must be an exact boolean"
            )

    def payload(self) -> dict[str, Any]:
        return {
            "schema_version": EXECUTION_REQUEST_DERIVATION_SCHEMA_VERSION,
            "batch_id": self.batch_id,
            "cell_ids": list(self.cell_ids),
            "generations": self.generations,
            "checkpoint_milestones": list(self.checkpoint_milestones),
            "reporting_enabled": self.reporting_enabled,
            "preserve_generation_zero": self.preserve_generation_zero,
        }


def _positions_for_cells(
    request: ResidentBatchRequestV1, cell_ids: Sequence[str]
) -> tuple[int, ...]:
    positions: list[int] = []
    for cell_id in cell_ids:
        matches = [
            index
            for index, candidate in enumerate(request.cell_ids)
            if candidate == cell_id
        ]
        if len(matches) != 1:
            raise ExecutionRequestError(
                f"derived cell must occur exactly once in the source request: {cell_id}"
            )
        positions.append(matches[0])
    return tuple(positions)


def _validate_array_shapes(request: ResidentBatchRequestV1) -> None:
    expected = len(request.cell_ids)
    malformed = [
        name
        for name, values in (*request.parameter_arrays, *request.seed_arrays)
        if len(values) != expected
    ]
    if malformed:
        raise ExecutionRequestError(
            "provider arrays do not match source cell order: "
            + ", ".join(sorted(malformed))
        )


def derive_resident_batch_request_v1(
    request: ResidentBatchRequestV1,
    derivation: ExecutionRequestDerivationV1,
) -> ResidentBatchRequestV1:
    """Derive a request with ``replace`` so every provider field is preserved."""

    if not isinstance(request, ResidentBatchRequestV1):
        raise ExecutionRequestError("source request must be ResidentBatchRequestV1")
    _validate_array_shapes(request)
    positions = _positions_for_cells(request, derivation.cell_ids)
    milestones = set(derivation.checkpoint_milestones)
    if derivation.preserve_generation_zero and 0 in request.checkpoint_milestones:
        milestones.add(0)
    reporting_enabled = (
        request.reporting_enabled
        if derivation.reporting_enabled is None
        else derivation.reporting_enabled
    )
    return replace(
        request,
        batch_id=derivation.batch_id,
        cell_ids=derivation.cell_ids,
        generations=derivation.generations,
        checkpoint_milestones=tuple(sorted(milestones)),
        parameter_arrays=tuple(
            (name, tuple(values[index] for index in positions))
            for name, values in request.parameter_arrays
        ),
        seed_arrays=tuple(
            (name, tuple(values[index] for index in positions))
            for name, values in request.seed_arrays
        ),
        reporting_enabled=reporting_enabled,
    )


@dataclass(frozen=True, slots=True)
class BoundExecutionRequestV1:
    """One provider request bound to plan, route, qualification, and derivation."""

    resident_request: ResidentBatchRequestV1
    provider_id: str
    profile_id: str
    source_batch_id: str
    plan_sha256: str
    qualification_key_sha256: str | None
    execution_route_sha256: str | None
    physical_batch_set: Mapping[str, Any] | None = None
    execution_route_contract: Mapping[str, Any] | None = None
    derivation: ExecutionRequestDerivationV1 | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.resident_request, ResidentBatchRequestV1):
            raise ExecutionRequestError(
                "bound request must contain ResidentBatchRequestV1"
            )
        if not self.provider_id or not self.profile_id or not self.source_batch_id:
            raise ExecutionRequestError(
                "bound request provider/profile/source batch identity is incomplete"
            )
        if self.resident_request.profile_id != self.profile_id:
            raise ExecutionRequestError("resident request profile binding differs")
        if (
            self.execution_route_sha256 is not None
            and self.resident_request.execution_route_sha256
            != self.execution_route_sha256
        ):
            raise ExecutionRequestError("resident request route binding differs")
        if self.physical_batch_set is not None and not isinstance(
            self.physical_batch_set, Mapping
        ):
            raise ExecutionRequestError("physical batch-set binding must be an object")

    def identity_payload(self) -> dict[str, Any]:
        value = {
            "schema_version": BOUND_EXECUTION_REQUEST_SCHEMA_VERSION,
            "provider_id": self.provider_id,
            "profile_id": self.profile_id,
            "source_batch_id": self.source_batch_id,
            "plan_sha256": self.plan_sha256,
            "qualification_key_sha256": self.qualification_key_sha256,
            "execution_route_sha256": self.execution_route_sha256,
            "resident_request": self.resident_request.payload(),
            "derivation": None
            if self.derivation is None
            else self.derivation.payload(),
        }
        if self.physical_batch_set is not None:
            value["physical_batch_set"] = dict(self.physical_batch_set)
        return value

    @property
    def bound_request_sha256(self) -> str:
        return canonical_sha256(self.identity_payload())

    def payload(self) -> dict[str, Any]:
        value = {
            **self.identity_payload(),
            "bound_request_sha256": self.bound_request_sha256,
        }
        if self.execution_route_contract is not None:
            value["execution_route_contract"] = dict(self.execution_route_contract)
        return value

    def derive_v1(
        self, derivation: ExecutionRequestDerivationV1
    ) -> "BoundExecutionRequestV1":
        return replace(
            self,
            resident_request=derive_resident_batch_request_v1(
                self.resident_request, derivation
            ),
            derivation=derivation,
        )


def build_execution_request_v1(
    provider: Any,
    campaign_plan: CampaignPlanV1 | Mapping[str, Any],
    batch_id: str,
    *,
    derivation: ExecutionRequestDerivationV1 | None = None,
) -> BoundExecutionRequestV1:
    """Build the sole immutable provider request used by every execution path."""

    if not isinstance(batch_id, str) or not batch_id:
        raise ExecutionRequestError("batch_id must be a non-empty string")
    translate = getattr(provider, "translate_batch", None)
    if not callable(translate):
        raise ExecutionRequestError("provider has no translate_batch constructor")
    plan = _plan_payload(campaign_plan)
    request = translate(campaign_plan, batch_id)
    if not isinstance(request, ResidentBatchRequestV1):
        raise ExecutionRequestError(
            "provider translate_batch must return ResidentBatchRequestV1"
        )
    _validate_array_shapes(request)
    contract_payload: Mapping[str, Any] | None = None
    route_sha256 = request.execution_route_sha256
    batch_set_builder = getattr(provider, "bound_physical_batch_set_v1", None)
    physical_batch_set = (
        batch_set_builder(campaign_plan, batch_id)
        if callable(batch_set_builder)
        else None
    )
    if physical_batch_set is not None and not isinstance(
        physical_batch_set, Mapping
    ):
        raise ExecutionRequestError(
            "provider physical batch-set constructor must return an object or None"
        )
    route_builder = getattr(provider, "execution_route_contract", None)
    if callable(route_builder):
        parameters = inspect.signature(route_builder).parameters
        contract = (
            route_builder(campaign_plan, request=request)
            if "request" in parameters
            or any(
                parameter.kind is inspect.Parameter.VAR_KEYWORD
                for parameter in parameters.values()
            )
            else route_builder(campaign_plan)
        )
        route_hash = getattr(contract, "execution_route_sha256", None)
        payload_builder = getattr(contract, "payload", None)
        if not isinstance(route_hash, str) or len(route_hash) != 64:
            raise ExecutionRequestError(
                "provider execution route contract has no canonical SHA-256"
            )
        if route_sha256 is not None and route_sha256 != route_hash:
            raise ExecutionRequestError(
                "provider request and execution route contract disagree"
            )
        route_sha256 = route_hash
        request = replace(request, execution_route_sha256=route_hash)
        if callable(payload_builder):
            payload = payload_builder()
            if not isinstance(payload, Mapping):
                raise ExecutionRequestError(
                    "execution route contract payload must be a mapping"
                )
            contract_payload = dict(payload)
    provider_id = getattr(provider, "provider_id", None)
    profile_id = getattr(provider, "profile_id", None)
    if not isinstance(provider_id, str) or not isinstance(profile_id, str):
        raise ExecutionRequestError("provider identity is incomplete")
    raw_plan_sha256 = plan.get("plan_sha256")
    plan_sha256 = (
        raw_plan_sha256
        if isinstance(raw_plan_sha256, str) and raw_plan_sha256
        else canonical_sha256(plan)
    )
    qualification_key_sha256 = plan.get("qualification_key_sha256")
    if qualification_key_sha256 is not None and not isinstance(
        qualification_key_sha256, str
    ):
        raise ExecutionRequestError("qualification key SHA-256 is malformed")
    bound = BoundExecutionRequestV1(
        resident_request=request,
        provider_id=provider_id,
        profile_id=profile_id,
        source_batch_id=batch_id,
        plan_sha256=plan_sha256,
        qualification_key_sha256=qualification_key_sha256,
        execution_route_sha256=route_sha256,
        physical_batch_set=physical_batch_set,
        execution_route_contract=contract_payload,
    )
    return bound if derivation is None else bound.derive_v1(derivation)


__all__ = [
    "BOUND_EXECUTION_REQUEST_SCHEMA_VERSION",
    "EXECUTION_REQUEST_DERIVATION_SCHEMA_VERSION",
    "BoundExecutionRequestV1",
    "ExecutionRequestDerivationV1",
    "ExecutionRequestError",
    "build_execution_request_v1",
    "derive_resident_batch_request_v1",
]
