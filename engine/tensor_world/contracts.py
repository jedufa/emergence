"""Representation-opaque execution contracts for Tensor World.

These contracts are deliberately domain-oriented.  Tensor ownership stays in
the backend and no caller needs to know tensor shapes, streams, or layouts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Final, Mapping, Protocol, runtime_checkable


BACKEND_CONTRACT_VERSION_V1: Final[str] = "tensor_world_backend_contract_v1"
CAPABILITY_SCHEMA_VERSION_V1: Final[str] = "tensor_world_backend_capabilities_v1"
TICK_REQUEST_SCHEMA_VERSION_V1: Final[str] = "tensor_world_tick_request_v1"
TICK_RECEIPT_SCHEMA_VERSION_V1: Final[str] = "tensor_world_tick_receipt_v1"


class TensorWorldContractError(ValueError):
    """A backend request or declaration violates the frozen C0 contract."""


class EquivalenceClassV1(StrEnum):
    BIT_EXACT_REQUIRED = "BIT_EXACT_REQUIRED"
    TRAJECTORY_EXACT_REQUIRED = "TRAJECTORY_EXACT_REQUIRED"
    ORDER_EXACT_REQUIRED = "ORDER_EXACT_REQUIRED"
    EXTERNAL_CONTRACT_EXACT_INTERNAL_REPRESENTATION_FREE = (
        "EXTERNAL_CONTRACT_EXACT_INTERNAL_REPRESENTATION_FREE"
    )
    DIAGNOSTIC_ONLY = "DIAGNOSTIC_ONLY"


@dataclass(frozen=True, slots=True)
class BackendCapabilitiesV1:
    """Fail-closed capability declaration for one execution backend."""

    backend_id: str
    mechanics: tuple[str, ...]
    batch_sizes: tuple[int, ...]
    grid_sizes: tuple[int, ...]
    initial_populations: tuple[int, ...]
    slot_capacities: tuple[int, ...]
    policy_families: tuple[str, ...]
    observation_abi: str
    policy_abi: str
    determinism: str
    rng_mode: str
    residency_mode: str
    checkpoint_replay: bool
    status: str
    schema_version: str = CAPABILITY_SCHEMA_VERSION_V1

    def __post_init__(self) -> None:
        if self.schema_version != CAPABILITY_SCHEMA_VERSION_V1:
            raise TensorWorldContractError("capability schema version is unsupported")
        if not self.backend_id or not self.mechanics or not self.policy_families:
            raise TensorWorldContractError("capability identity is incomplete")
        for name, values in (
            ("batch_sizes", self.batch_sizes),
            ("grid_sizes", self.grid_sizes),
            ("initial_populations", self.initial_populations),
            ("slot_capacities", self.slot_capacities),
        ):
            if not values or any(
                type(value) is not int or value <= 0 for value in values
            ):
                raise TensorWorldContractError(f"{name} must contain positive integers")
        if self.status not in {"experimental", "qualified", "admitted"}:
            raise TensorWorldContractError("backend status is invalid")

    def supports(
        self,
        *,
        batch: int,
        grid_size: int,
        initial_population: int,
        slot_capacity: int,
        policy_family: str,
        mechanics: tuple[str, ...],
    ) -> bool:
        """Return exact support; there is no partial or hidden fallback."""

        return (
            batch in self.batch_sizes
            and grid_size in self.grid_sizes
            and initial_population in self.initial_populations
            and slot_capacity in self.slot_capacities
            and policy_family in self.policy_families
            and set(mechanics).issubset(self.mechanics)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "backend_id": self.backend_id,
            "mechanics": list(self.mechanics),
            "batch_sizes": list(self.batch_sizes),
            "grid_sizes": list(self.grid_sizes),
            "initial_populations": list(self.initial_populations),
            "slot_capacities": list(self.slot_capacities),
            "policy_families": list(self.policy_families),
            "observation_abi": self.observation_abi,
            "policy_abi": self.policy_abi,
            "determinism": self.determinism,
            "rng_mode": self.rng_mode,
            "residency_mode": self.residency_mode,
            "checkpoint_replay": self.checkpoint_replay,
            "status": self.status,
        }


@dataclass(frozen=True, slots=True)
class TickRequestV1:
    """One domain-level transition request.

    ``trajectory_fixture_id`` is a test-only C1 selector.  A state adapter
    resolves it and uploads its random inputs before the timed tick; callers
    never pass tensors through the domain interface.
    """

    tick_count: int = 1
    trajectory_fixture_id: str | None = None
    controls: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = TICK_REQUEST_SCHEMA_VERSION_V1

    def __post_init__(self) -> None:
        if self.schema_version != TICK_REQUEST_SCHEMA_VERSION_V1:
            raise TensorWorldContractError("tick request schema version is unsupported")
        if type(self.tick_count) is not int or self.tick_count != 1:
            raise TensorWorldContractError("C1 admits exactly one tick per request")
        if self.trajectory_fixture_id is not None and not self.trajectory_fixture_id:
            raise TensorWorldContractError(
                "trajectory fixture identity cannot be empty"
            )
        if self.controls:
            raise TensorWorldContractError("C1 admits no steady-state host controls")


@dataclass(frozen=True, slots=True)
class TickReceiptV1:
    """Bounded receipt that never contains full backend state."""

    backend_id: str
    completed_ticks: int
    host_to_device_bytes: int
    device_to_host_bytes: int
    host_callback_count: int
    synchronization_count: int
    materialization_count: int
    schema_version: str = TICK_RECEIPT_SCHEMA_VERSION_V1

    def __post_init__(self) -> None:
        if self.schema_version != TICK_RECEIPT_SCHEMA_VERSION_V1:
            raise TensorWorldContractError("tick receipt schema version is unsupported")
        if not self.backend_id:
            raise TensorWorldContractError("tick receipt backend identity is required")
        for name in (
            "completed_ticks",
            "host_to_device_bytes",
            "device_to_host_bytes",
            "host_callback_count",
            "synchronization_count",
            "materialization_count",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise TensorWorldContractError(f"{name} must be nonnegative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "backend_id": self.backend_id,
            "completed_ticks": self.completed_ticks,
            "host_to_device_bytes": self.host_to_device_bytes,
            "device_to_host_bytes": self.device_to_host_bytes,
            "host_callback_count": self.host_callback_count,
            "synchronization_count": self.synchronization_count,
            "materialization_count": self.materialization_count,
        }


@dataclass(slots=True)
class BackendRunStateV1:
    """Opaque handle held by domain callers and owned by its backend."""

    backend_id: str
    token: str
    closed: bool = False


@runtime_checkable
class ExecutionBackendV1(Protocol):
    """Small production backend surface frozen at TW-C0."""

    @property
    def capabilities(self) -> BackendCapabilitiesV1: ...

    def initialize(self, canonical_state: Mapping[str, Any]) -> BackendRunStateV1: ...

    def advance(
        self, state: BackendRunStateV1, request: TickRequestV1
    ) -> TickReceiptV1: ...

    def export_state(self, state: BackendRunStateV1) -> Mapping[str, Any]: ...

    def close(self, state: BackendRunStateV1) -> None: ...


@runtime_checkable
class StateAdapterV1(Protocol):
    """Explicit initialization/export boundary; never steady-tick plumbing."""

    def import_state(
        self, backend: ExecutionBackendV1, canonical_state: Mapping[str, Any]
    ) -> BackendRunStateV1: ...

    def export_state(
        self, backend: ExecutionBackendV1, state: BackendRunStateV1
    ) -> Mapping[str, Any]: ...


@runtime_checkable
class PerformanceProbeV1(Protocol):
    """Read-only measurement boundary kept outside normal tick semantics."""

    def begin(self, state: BackendRunStateV1) -> None: ...

    def end(self, state: BackendRunStateV1) -> Mapping[str, int | float]: ...
