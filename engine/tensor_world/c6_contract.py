"""Representation-opaque C6 provider and external-contract declarations.

The Tensor World route is not C6-admitted yet.  This module makes that fact
machine-checkable: a caller asks for domain capabilities and receives either
one exact route resolution or a typed fail-closed reason.  No tensor layouts,
fallback provider, or hidden CPU path are exposed through this boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import hashlib
import json
from typing import Any, Final, Mapping

from .contracts import TensorWorldContractError


C6_CONTRACT_SCHEMA_VERSION_V1: Final[str] = "tensor_world_c6_external_contract_v1"
C6_PROVIDER_MANIFEST_SCHEMA_VERSION_V1: Final[str] = (
    "tensor_world_c6_provider_manifest_v1"
)


class TensorWorldC6CapabilityStatusV1(StrEnum):
    SUPPORTED = "SUPPORTED"
    UNSUPPORTED = "UNSUPPORTED"
    NOT_YET_QUALIFIED = "NOT_YET_QUALIFIED"


class TensorWorldC6FailureCodeV1(StrEnum):
    UNSUPPORTED_CAPABILITY = "UNSUPPORTED_CAPABILITY"
    NOT_YET_QUALIFIED = "NOT_YET_QUALIFIED"
    CONTROL_BOUNDARY = "CONTROL_BOUNDARY"
    RESOURCE_ADMISSION = "RESOURCE_ADMISSION"
    RESIDENCY_VIOLATION = "RESIDENCY_VIOLATION"
    RNG_DETERMINISM = "RNG_DETERMINISM"
    CHECKPOINT_INCOMPATIBLE = "CHECKPOINT_INCOMPATIBLE"
    SEMANTIC_MISMATCH = "SEMANTIC_MISMATCH"
    BACKEND_INTERNAL = "BACKEND_INTERNAL"


@dataclass(frozen=True, slots=True)
class TensorWorldC6CapabilityV1:
    name: str
    status: TensorWorldC6CapabilityStatusV1
    reason: str
    schema_version: str = C6_CONTRACT_SCHEMA_VERSION_V1

    def __post_init__(self) -> None:
        if self.schema_version != C6_CONTRACT_SCHEMA_VERSION_V1:
            raise TensorWorldContractError("C6 capability schema version is unsupported")
        if not self.name or not self.reason:
            raise TensorWorldContractError("C6 capability identity and reason are required")
        if not isinstance(self.status, TensorWorldC6CapabilityStatusV1):
            raise TensorWorldContractError("C6 capability status is invalid")

    def to_dict(self) -> dict[str, str]:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "status": self.status.value,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class TensorWorldC6RouteRequestV1:
    """Domain-level route request; tensor shapes never cross this boundary."""

    batch: int
    grid_size: int
    initial_population: int
    slot_capacity: int
    policy_family: str
    mechanics: tuple[str, ...]
    device: str = "cuda"
    strict: bool = True
    rng_semantics_version: int = 1
    schema_version: str = C6_CONTRACT_SCHEMA_VERSION_V1

    def __post_init__(self) -> None:
        if self.schema_version != C6_CONTRACT_SCHEMA_VERSION_V1:
            raise TensorWorldContractError("C6 route request schema version is unsupported")
        for name in ("batch", "grid_size", "initial_population", "slot_capacity"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise TensorWorldContractError(f"C6 {name} must be a positive int")
        if not self.policy_family or not self.mechanics:
            raise TensorWorldContractError("C6 route request identity is incomplete")
        if len(set(self.mechanics)) != len(self.mechanics):
            raise TensorWorldContractError("C6 route mechanics must not contain duplicates")
        if not self.device or type(self.strict) is not bool:
            raise TensorWorldContractError("C6 route device/strict fields are invalid")
        if self.rng_semantics_version not in {1, 2}:
            raise TensorWorldContractError("C6 rng_semantics_version must be 1 or 2")

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "batch": self.batch,
            "grid_size": self.grid_size,
            "initial_population": self.initial_population,
            "slot_capacity": self.slot_capacity,
            "policy_family": self.policy_family,
            "mechanics": list(self.mechanics),
            "device": self.device,
            "strict": self.strict,
        }
        if self.rng_semantics_version != 1:
            payload["rng_semantics_version"] = self.rng_semantics_version
        return payload


@dataclass(frozen=True, slots=True)
class TensorWorldC6RouteResolutionV1:
    request: TensorWorldC6RouteRequestV1
    provider_id: str
    backend_id: str
    route_id: str
    capability_statuses: Mapping[str, TensorWorldC6CapabilityStatusV1]
    accepted: bool
    failure_code: TensorWorldC6FailureCodeV1 | None
    failure_reason: str | None
    route_fingerprint: str
    schema_version: str = C6_CONTRACT_SCHEMA_VERSION_V1

    def __post_init__(self) -> None:
        if self.schema_version != C6_CONTRACT_SCHEMA_VERSION_V1:
            raise TensorWorldContractError("C6 route resolution schema version is unsupported")
        if not self.provider_id or not self.backend_id or not self.route_id:
            raise TensorWorldContractError("C6 route identity is incomplete")
        if len(self.route_fingerprint) != 64 or any(
            char not in "0123456789abcdef" for char in self.route_fingerprint
        ):
            raise TensorWorldContractError("C6 route fingerprint is not SHA256")
        if self.accepted and (self.failure_code is not None or self.failure_reason is not None):
            raise TensorWorldContractError("accepted C6 route cannot carry a failure")
        if not self.accepted and (self.failure_code is None or not self.failure_reason):
            raise TensorWorldContractError("rejected C6 route requires a failure")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "provider_id": self.provider_id,
            "backend_id": self.backend_id,
            "route_id": self.route_id,
            "request": self.request.to_dict(),
            "capability_statuses": {
                name: status.value for name, status in self.capability_statuses.items()
            },
            "accepted": self.accepted,
            "failure_code": None if self.failure_code is None else self.failure_code.value,
            "failure_reason": self.failure_reason,
            "route_fingerprint": self.route_fingerprint,
        }


@dataclass(frozen=True, slots=True)
class TensorWorldC6ProviderManifestV1:
    provider_id: str
    backend_id: str
    route_id: str
    capabilities: tuple[TensorWorldC6CapabilityV1, ...]
    provider_contract_id: str = "tensor_world_provider_contract_v1"
    schema_version: str = C6_PROVIDER_MANIFEST_SCHEMA_VERSION_V1

    def __post_init__(self) -> None:
        if self.schema_version != C6_PROVIDER_MANIFEST_SCHEMA_VERSION_V1:
            raise TensorWorldContractError("C6 provider manifest schema version is unsupported")
        if not self.provider_id or not self.backend_id or not self.route_id:
            raise TensorWorldContractError("C6 provider manifest identity is incomplete")
        if not self.provider_contract_id:
            raise TensorWorldContractError("C6 provider contract identity is required")
        names = [capability.name for capability in self.capabilities]
        if not names or len(names) != len(set(names)):
            raise TensorWorldContractError("C6 provider capabilities must be unique")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "provider_id": self.provider_id,
            "backend_id": self.backend_id,
            "route_id": self.route_id,
            "provider_contract_id": self.provider_contract_id,
            "capabilities": [capability.to_dict() for capability in self.capabilities],
        }

    def fingerprint(self) -> str:
        encoded = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def resolve(self, request: TensorWorldC6RouteRequestV1) -> TensorWorldC6RouteResolutionV1:
        by_name = {capability.name: capability for capability in self.capabilities}
        statuses: dict[str, TensorWorldC6CapabilityStatusV1] = {}
        failing: TensorWorldC6CapabilityV1 | None = None
        for name in request.mechanics:
            capability = by_name.get(name)
            if capability is None:
                statuses[name] = TensorWorldC6CapabilityStatusV1.UNSUPPORTED
                if failing is None:
                    failing = TensorWorldC6CapabilityV1(
                        name=name,
                        status=TensorWorldC6CapabilityStatusV1.UNSUPPORTED,
                        reason="the exact Tensor World provider manifest does not declare this capability",
                    )
                continue
            statuses[name] = capability.status
            if (
                failing is None
                and capability.status is not TensorWorldC6CapabilityStatusV1.SUPPORTED
            ):
                failing = capability
        accepted = failing is None
        failure_code: TensorWorldC6FailureCodeV1 | None = None
        failure_reason: str | None = None
        if failing is not None:
            failure_code = (
                TensorWorldC6FailureCodeV1.UNSUPPORTED_CAPABILITY
                if failing.status is TensorWorldC6CapabilityStatusV1.UNSUPPORTED
                else TensorWorldC6FailureCodeV1.NOT_YET_QUALIFIED
            )
            failure_reason = f"{failing.name}: {failing.reason}"
            if request.strict:
                raise TensorWorldContractError(
                    f"C6 strict route rejected ({failure_code.value}): {failure_reason}"
                )
        route_id = (
            "tensor_world_c0_c7_rngv2_v1"
            if request.rng_semantics_version == 2
            else self.route_id
        )
        payload = {
            "manifest_fingerprint": self.fingerprint(),
            "request": request.to_dict(),
            "provider_id": self.provider_id,
            "backend_id": self.backend_id,
            "route_id": route_id,
            "capability_statuses": {
                name: status.value for name, status in statuses.items()
            },
            "accepted": accepted,
            "failure_code": None if failure_code is None else failure_code.value,
            "failure_reason": failure_reason,
        }
        route_fingerprint = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return TensorWorldC6RouteResolutionV1(
            request=request,
            provider_id=self.provider_id,
            backend_id=self.backend_id,
            route_id=route_id,
            capability_statuses=statuses,
            accepted=accepted,
            failure_code=failure_code,
            failure_reason=failure_reason,
            route_fingerprint=route_fingerprint,
        )


@dataclass(frozen=True, slots=True)
class TensorWorldC6ResourceEstimateV1:
    resident_vram_bytes: int
    temporary_vram_bytes: int
    headroom_bytes: int
    device_capability: str
    schema_version: str = C6_CONTRACT_SCHEMA_VERSION_V1

    def __post_init__(self) -> None:
        if self.schema_version != C6_CONTRACT_SCHEMA_VERSION_V1:
            raise TensorWorldContractError("C6 resource estimate schema version is unsupported")
        for name in ("resident_vram_bytes", "temporary_vram_bytes", "headroom_bytes"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise TensorWorldContractError(f"C6 {name} must be a nonnegative int")
        if not self.device_capability:
            raise TensorWorldContractError("C6 device capability is required")

    @property
    def required_vram_bytes(self) -> int:
        return self.resident_vram_bytes + self.temporary_vram_bytes + self.headroom_bytes


@dataclass(frozen=True, slots=True)
class TensorWorldC6ResourceAdmissionV1:
    accepted: bool
    available_vram_bytes: int
    required_vram_bytes: int
    failure_code: TensorWorldC6FailureCodeV1 | None = None
    failure_reason: str | None = None
    schema_version: str = C6_CONTRACT_SCHEMA_VERSION_V1

    def __post_init__(self) -> None:
        if self.schema_version != C6_CONTRACT_SCHEMA_VERSION_V1:
            raise TensorWorldContractError("C6 resource admission schema version is unsupported")
        if self.available_vram_bytes < 0 or self.required_vram_bytes < 0:
            raise TensorWorldContractError("C6 VRAM values must be nonnegative")
        if self.accepted and self.failure_code is not None:
            raise TensorWorldContractError("accepted resource admission cannot carry a failure")
        if not self.accepted and self.failure_code is None:
            raise TensorWorldContractError("rejected resource admission requires a failure code")


def admit_tensor_world_resources_v1(
    estimate: TensorWorldC6ResourceEstimateV1,
    *,
    available_vram_bytes: int,
    device_capability: str,
) -> TensorWorldC6ResourceAdmissionV1:
    if type(available_vram_bytes) is not int or available_vram_bytes < 0:
        raise TensorWorldContractError("C6 available VRAM must be a nonnegative int")
    if not device_capability:
        raise TensorWorldContractError("C6 device capability is required")
    if device_capability != estimate.device_capability:
        return TensorWorldC6ResourceAdmissionV1(
            accepted=False,
            available_vram_bytes=available_vram_bytes,
            required_vram_bytes=estimate.required_vram_bytes,
            failure_code=TensorWorldC6FailureCodeV1.RESOURCE_ADMISSION,
            failure_reason="device capability does not match the frozen resource estimate",
        )
    if available_vram_bytes < estimate.required_vram_bytes:
        return TensorWorldC6ResourceAdmissionV1(
            accepted=False,
            available_vram_bytes=available_vram_bytes,
            required_vram_bytes=estimate.required_vram_bytes,
            failure_code=TensorWorldC6FailureCodeV1.RESOURCE_ADMISSION,
            failure_reason="available VRAM is below resident plus temporary headroom requirement",
        )
    return TensorWorldC6ResourceAdmissionV1(
        accepted=True,
        available_vram_bytes=available_vram_bytes,
        required_vram_bytes=estimate.required_vram_bytes,
    )


def default_tensor_world_c6_provider_manifest_v1() -> TensorWorldC6ProviderManifestV1:
    pending = (
        "C3-C5 semantics, resident integration, and fresh C7 qualification remain open"
    )
    unsupported = (
        "the current Tensor World provider has no admitted implementation for this capability"
    )
    return TensorWorldC6ProviderManifestV1(
        provider_id="tensor_world_provider_v1",
        backend_id="tensor_world_c3_v1",
        route_id="tensor_world_c0_c7_v3",
        capabilities=tuple(
            TensorWorldC6CapabilityV1(name=name, status=status, reason=reason)
            for name, status, reason in (
                ("patch", TensorWorldC6CapabilityStatusV1.NOT_YET_QUALIFIED, pending),
                ("jungle", TensorWorldC6CapabilityStatusV1.NOT_YET_QUALIFIED, pending),
                ("wdl", TensorWorldC6CapabilityStatusV1.NOT_YET_QUALIFIED, pending),
                ("eed", TensorWorldC6CapabilityStatusV1.NOT_YET_QUALIFIED, pending),
                ("observation", TensorWorldC6CapabilityStatusV1.NOT_YET_QUALIFIED, pending),
                ("fixed_mlp", TensorWorldC6CapabilityStatusV1.NOT_YET_QUALIFIED, pending),
                ("population", TensorWorldC6CapabilityStatusV1.NOT_YET_QUALIFIED, pending),
                ("checkpoint_replay", TensorWorldC6CapabilityStatusV1.NOT_YET_QUALIFIED, pending),
                ("controls", TensorWorldC6CapabilityStatusV1.NOT_YET_QUALIFIED, pending),
                ("metrics_evidence", TensorWorldC6CapabilityStatusV1.NOT_YET_QUALIFIED, pending),
                ("visualizer", TensorWorldC6CapabilityStatusV1.UNSUPPORTED, unsupported),
            )
        ),
    )


def resolve_tensor_world_c6_route_v1(
    request: TensorWorldC6RouteRequestV1,
    *,
    manifest: TensorWorldC6ProviderManifestV1 | None = None,
) -> TensorWorldC6RouteResolutionV1:
    return (manifest or default_tensor_world_c6_provider_manifest_v1()).resolve(request)


__all__ = [
    "C6_CONTRACT_SCHEMA_VERSION_V1",
    "C6_PROVIDER_MANIFEST_SCHEMA_VERSION_V1",
    "TensorWorldC6CapabilityStatusV1",
    "TensorWorldC6CapabilityV1",
    "TensorWorldC6FailureCodeV1",
    "TensorWorldC6ProviderManifestV1",
    "TensorWorldC6ResourceAdmissionV1",
    "TensorWorldC6ResourceEstimateV1",
    "TensorWorldC6RouteRequestV1",
    "TensorWorldC6RouteResolutionV1",
    "admit_tensor_world_resources_v1",
    "default_tensor_world_c6_provider_manifest_v1",
    "resolve_tensor_world_c6_route_v1",
]
