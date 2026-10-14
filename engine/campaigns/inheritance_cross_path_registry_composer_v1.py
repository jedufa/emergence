"""Compose immutable cross-path registries from declarative campaign inputs.

The native semantic module owns IDs, draw budgets, and linkage.  Campaign YAML
owns runtime identity, parameter names, seed owners, and requested extent.
This module joins those authorities without encoding any campaign identifier,
seed bank, matrix size, or checkpoint schedule in Python.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import importlib
from types import ModuleType
from typing import Any, Final

from engine.perf.cuda_architecture_b_direct_resident_source_allocator_v1 import (
    CudaArchitectureBDirectResidentSourceAllocatorError,
    required_resident_agent_id_capacity_v1,
)

from .canonical import canonical_sha256
from .inheritance_cross_path_exposure_projection_contract_v1 import (
    AGENT_ID_ALLOCATION_SCHEMA_VERSION,
)
from .production_engine_provider import (
    REGISTRY_SCHEMA_VERSION,
    REPRODUCTION_EXPOSURE_CAPACITY_SCHEMA_VERSION,
)
from .qualification import EXECUTION_ROUTE_OBLIGATIONS_V1


COMPOSER_SCHEMA_VERSION: Final[str] = (
    "eiel_inheritance_cross_path_registry_composer_v1"
)
MAX_NATIVE_INT64: Final[int] = (1 << 63) - 1


class InheritanceCrossPathRegistryCompositionError(ValueError):
    """Declarative inputs cannot produce an exact native registry."""


def _mapping(value: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise InheritanceCrossPathRegistryCompositionError(
            f"{field} must be an object"
        )
    return dict(value)


def _text(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise InheritanceCrossPathRegistryCompositionError(
            f"{field} must be non-empty text"
        )
    return value


def _positive(value: Any, *, field: str) -> int:
    if type(value) is not int or not 1 <= value <= MAX_NATIVE_INT64:
        raise InheritanceCrossPathRegistryCompositionError(
            f"{field} must be a positive native int64"
        )
    return value


@dataclass(frozen=True, slots=True)
class AgentIdAllocationCapacityV1:
    """Explicit inputs and derived K for the resident identity allocation."""

    current_capacity: int
    population_next_id: int
    max_births_per_tick: int
    slot_capacity: int
    allocation_generation_count: int
    retained_capacity: int
    required_aggregate_capacity: int

    def __post_init__(self) -> None:
        for field in (
            "current_capacity",
            "population_next_id",
            "max_births_per_tick",
            "slot_capacity",
            "allocation_generation_count",
            "retained_capacity",
            "required_aggregate_capacity",
        ):
            _positive(getattr(self, field), field=field)
        if self.retained_capacity < self.current_capacity:
            raise InheritanceCrossPathRegistryCompositionError(
                "retained identity capacity must cover current capacity"
            )
        if self.required_aggregate_capacity < self.retained_capacity:
            raise InheritanceCrossPathRegistryCompositionError(
                "required aggregate capacity must cover retained identity capacity"
            )
        try:
            derived = required_resident_agent_id_capacity_v1(
                self.current_capacity,
                self.population_next_id,
                self.max_births_per_tick,
                self.slot_capacity,
                allocation_generation_count=self.allocation_generation_count,
                retained_capacity=self.retained_capacity,
            )
        except CudaArchitectureBDirectResidentSourceAllocatorError as exc:
            raise InheritanceCrossPathRegistryCompositionError(
                f"agent-ID allocation inputs are invalid: {exc}"
            ) from exc
        if self.required_aggregate_capacity != derived:
            raise InheritanceCrossPathRegistryCompositionError(
                "required aggregate capacity does not match allocation inputs: "
                f"declared={self.required_aggregate_capacity} derived={derived}"
            )

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
    ) -> AgentIdAllocationCapacityV1:
        """Parse the validated projection-contract allocation envelope."""

        value = _mapping(payload, field="agent_id_allocation")
        expected = {
            "schema_version",
            "current_capacity",
            "population_next_id",
            "max_births_per_tick",
            "slot_capacity",
            "allocation_generation_count",
            "retained_capacity",
            "required_aggregate_capacity",
        }
        if set(value) != expected:
            raise InheritanceCrossPathRegistryCompositionError(
                "agent_id_allocation field set is invalid"
            )
        if value["schema_version"] != AGENT_ID_ALLOCATION_SCHEMA_VERSION:
            raise InheritanceCrossPathRegistryCompositionError(
                "agent_id_allocation schema mismatch"
            )
        return cls(
            current_capacity=value["current_capacity"],
            population_next_id=value["population_next_id"],
            max_births_per_tick=value["max_births_per_tick"],
            slot_capacity=value["slot_capacity"],
            allocation_generation_count=value[
                "allocation_generation_count"
            ],
            retained_capacity=value["retained_capacity"],
            required_aggregate_capacity=value[
                "required_aggregate_capacity"
            ],
        )

    def payload(self) -> dict[str, Any]:
        return {
            "schema_version": AGENT_ID_ALLOCATION_SCHEMA_VERSION,
            "current_capacity": self.current_capacity,
            "population_next_id": self.population_next_id,
            "max_births_per_tick": self.max_births_per_tick,
            "slot_capacity": self.slot_capacity,
            "allocation_generation_count": self.allocation_generation_count,
            "retained_capacity": self.retained_capacity,
            "required_aggregate_capacity": self.required_aggregate_capacity,
        }


@dataclass(frozen=True, slots=True)
class ReproductionExposureCapacityV1:
    """Bounded raw resident event storage for one physical execution owner."""

    storage_mode: str
    attempt_capacity: int
    snapshot_capacity: int
    aggregate_capacity: int
    agent_id_allocation: AgentIdAllocationCapacityV1 | None = None

    def __post_init__(self) -> None:
        if self.storage_mode not in {"exact_full", "compact_complete"}:
            raise InheritanceCrossPathRegistryCompositionError(
                "reproduction exposure storage mode is unsupported"
            )
        for field in (
            "attempt_capacity",
            "snapshot_capacity",
            "aggregate_capacity",
        ):
            _positive(getattr(self, field), field=field)
        if self.storage_mode == "exact_full":
            if self.attempt_capacity > MAX_NATIVE_INT64 // 2:
                raise InheritanceCrossPathRegistryCompositionError(
                    "twice the exact-full attempt capacity exceeds native int64"
                )
            if self.snapshot_capacity < 2 * self.attempt_capacity:
                raise InheritanceCrossPathRegistryCompositionError(
                    "exact-full snapshot capacity must be at least twice "
                    "attempt capacity"
                )
        if (
            self.agent_id_allocation is not None
            and self.aggregate_capacity
            < self.agent_id_allocation.required_aggregate_capacity
        ):
            raise InheritanceCrossPathRegistryCompositionError(
                "aggregate capacity is below the required identity allocation K"
            )

    def payload(self) -> dict[str, Any]:
        if self.agent_id_allocation is None:
            raise InheritanceCrossPathRegistryCompositionError(
                "published exposure capacity requires an agent-ID allocation "
                "envelope"
            )
        payload: dict[str, Any] = {
            "schema_version": REPRODUCTION_EXPOSURE_CAPACITY_SCHEMA_VERSION,
            "storage_mode": self.storage_mode,
            "attempt_capacity": self.attempt_capacity,
            "snapshot_capacity": self.snapshot_capacity,
            "aggregate_capacity": self.aggregate_capacity,
        }
        payload["agent_id_allocation"] = self.agent_id_allocation.payload()
        return payload


@dataclass(frozen=True, slots=True)
class InheritanceCrossPathRegistryEnvelopeV1:
    """Reusable admitted extent and route obligations for one native artifact."""

    max_admitted_batch: int
    max_generations: int
    ticks_per_generation: int
    required_obligations: tuple[str, ...]
    reproduction_exposure_capacity: ReproductionExposureCapacityV1 | None = None

    def __post_init__(self) -> None:
        for field in (
            "max_admitted_batch",
            "max_generations",
            "ticks_per_generation",
        ):
            _positive(getattr(self, field), field=field)
        if (
            not self.required_obligations
            or len(self.required_obligations)
            != len(set(self.required_obligations))
            or any(
                not isinstance(value, str) or not value
                for value in self.required_obligations
            )
        ):
            raise InheritanceCrossPathRegistryCompositionError(
                "required obligations must be unique non-empty names"
            )
        unsupported = set(self.required_obligations) - set(
            EXECUTION_ROUTE_OBLIGATIONS_V1
        )
        if unsupported:
            raise InheritanceCrossPathRegistryCompositionError(
                "unsupported route obligations: " + ", ".join(sorted(unsupported))
            )


def _native_contract(native: ModuleType) -> dict[str, Any]:
    required = (
        "NATIVE_ABI",
        "NATIVE_CHECKPOINT_SCHEMA",
        "SEMANTIC_TYPES",
        "SEMANTIC_IDS",
        "INHERITANCE_SEMANTIC",
        "DRAW_BUDGETS",
        "AUDITED_LOCI",
        "LINKAGE_BLOCKS",
    )
    missing = [name for name in required if not hasattr(native, name)]
    if missing:
        raise InheritanceCrossPathRegistryCompositionError(
            "native semantic module is incomplete: " + ", ".join(missing)
        )
    semantic_types = dict(native.SEMANTIC_TYPES)
    semantic_ids = {
        str(axis): {str(token): int(identifier) for token, identifier in values.items()}
        for axis, values in dict(native.SEMANTIC_IDS).items()
    }
    if set(semantic_types) != set(semantic_ids):
        raise InheritanceCrossPathRegistryCompositionError(
            "native semantic type and ID axes differ"
        )
    return {
        "native_abi": _text(native.NATIVE_ABI, field="native.NATIVE_ABI"),
        "native_checkpoint_schema": _text(
            native.NATIVE_CHECKPOINT_SCHEMA,
            field="native.NATIVE_CHECKPOINT_SCHEMA",
        ),
        "semantic_types": semantic_types,
        "semantic_ids": semantic_ids,
        "inheritance_semantic": _text(
            native.INHERITANCE_SEMANTIC,
            field="native.INHERITANCE_SEMANTIC",
        ),
        "draw_budgets": {
            str(name): int(value) for name, value in dict(native.DRAW_BUDGETS).items()
        },
        "audited_loci": [str(value) for value in native.AUDITED_LOCI],
        "linkage_blocks": {
            str(name): [str(value) for value in values]
            for name, values in dict(native.LINKAGE_BLOCKS).items()
        },
    }


def _dimension_values(spec: Mapping[str, Any]) -> dict[str, list[Any]]:
    matrix = _mapping(spec.get("matrix"), field="matrix")
    raw = matrix.get("dimensions")
    if (
        isinstance(raw, (str, bytes))
        or not isinstance(raw, Sequence)
        or not raw
    ):
        raise InheritanceCrossPathRegistryCompositionError(
            "matrix.dimensions must be a non-empty array"
        )
    result: dict[str, list[Any]] = {}
    for index, row in enumerate(raw):
        item = _mapping(row, field=f"matrix.dimensions[{index}]")
        identifier = _text(item.get("id"), field=f"matrix.dimensions[{index}].id")
        values = item.get("values")
        if (
            isinstance(values, (str, bytes))
            or not isinstance(values, Sequence)
            or not values
            or identifier in result
        ):
            raise InheritanceCrossPathRegistryCompositionError(
                f"matrix dimension is invalid: {identifier}"
            )
        result[identifier] = list(values)
    return result


def compose_inheritance_cross_path_registry_v2(
    spec: Mapping[str, Any],
    envelope: InheritanceCrossPathRegistryEnvelopeV1,
    *,
    native_module: str = "engine.perf.inheritance_cross_path_gpu_native",
    native_api: ModuleType | None = None,
) -> dict[str, Any]:
    """Reject composition of a new standalone cross-path registry."""

    raise InheritanceCrossPathRegistryCompositionError(
        "P12_RETIRED_INHERITANCE_AUTHORITY "
        "cross-path registry composition is disabled"
    )
    # Historical implementation retained temporarily for evidence reading.
    if not isinstance(spec, Mapping):
        raise InheritanceCrossPathRegistryCompositionError(
            "campaign spec must be an object"
        )
    module_name = _text(native_module, field="native_module")
    native = native_api or importlib.import_module(module_name)
    native_contract = _native_contract(native)

    matrix = _mapping(spec.get("matrix"), field="matrix")
    bindings = _mapping(
        matrix.get("parameter_bindings"), field="matrix.parameter_bindings"
    )
    semantic_axes = set(native_contract["semantic_types"])
    if set(bindings) != semantic_axes or any(
        not isinstance(value, str) or not value for value in bindings.values()
    ):
        raise InheritanceCrossPathRegistryCompositionError(
            "matrix parameter bindings differ from native semantic axes"
        )
    dimensions = _dimension_values(spec)
    for axis, ids in native_contract["semantic_ids"].items():
        if axis not in dimensions or set(dimensions[axis]) != set(ids):
            raise InheritanceCrossPathRegistryCompositionError(
                f"matrix values differ from native semantic IDs: {axis}"
            )

    parameters = _mapping(spec.get("parameters"), field="parameters")
    overlap = set(parameters) & set(bindings.values())
    if overlap:
        raise InheritanceCrossPathRegistryCompositionError(
            "semantic matrix parameters cannot also be fixed: "
            + ", ".join(sorted(overlap))
        )
    seed_bindings = _mapping(
        matrix.get("seed_bindings"), field="matrix.seed_bindings"
    )
    seed_owners = sorted(set(seed_bindings.values()))
    if not seed_owners or any(
        not isinstance(value, str) or not value for value in seed_owners
    ):
        raise InheritanceCrossPathRegistryCompositionError(
            "matrix seed bindings must declare non-empty owners"
        )

    world = _mapping(spec.get("world"), field="world")
    overlays = world.get("overlays")
    if (
        not isinstance(overlays, list)
        or any(not isinstance(value, str) or not value for value in overlays)
        or len(overlays) != len(set(overlays))
    ):
        raise InheritanceCrossPathRegistryCompositionError(
            "world.overlays must contain unique text values"
        )
    policy = _mapping(spec.get("policy"), field="policy")
    selection = _mapping(spec.get("selection"), field="selection")
    execution = _mapping(spec.get("execution"), field="execution")
    requested_batch = _positive(
        _mapping(execution.get("batch"), field="execution.batch").get("cells"),
        field="execution.batch.cells",
    )
    requested_generations = _positive(
        execution.get("generations"), field="execution.generations"
    )
    requested_ticks = _positive(
        execution.get("ticks_per_generation"),
        field="execution.ticks_per_generation",
    )
    if (
        envelope.max_admitted_batch < requested_batch
        or envelope.max_generations < requested_generations
        or envelope.ticks_per_generation != requested_ticks
    ):
        raise InheritanceCrossPathRegistryCompositionError(
            "registry envelope does not cover the requested campaign extent"
        )

    payload: dict[str, Any] = {
        "schema_version": REGISTRY_SCHEMA_VERSION,
        "native_module": module_name,
        "native_abi": native_contract["native_abi"],
        "native_checkpoint_schema": native_contract[
            "native_checkpoint_schema"
        ],
        "runtime_identity": {
            "world_base": _text(world.get("base"), field="world.base"),
            "world_overlays": sorted(overlays),
            "policy_family": _text(policy.get("family"), field="policy.family"),
            "selection_mode": _text(selection.get("mode"), field="selection.mode"),
        },
        "parameter_bindings": dict(sorted(bindings.items())),
        "fixed_parameters": dict(sorted(parameters.items())),
        "seed_owners": seed_owners,
        "semantic_types": native_contract["semantic_types"],
        "semantic_ids": native_contract["semantic_ids"],
        "inheritance_semantic": native_contract["inheritance_semantic"],
        "draw_budgets": native_contract["draw_budgets"],
        "audited_loci": native_contract["audited_loci"],
        "linkage_blocks": native_contract["linkage_blocks"],
        "max_admitted_batch": envelope.max_admitted_batch,
        "max_generations": envelope.max_generations,
        "ticks_per_generation": envelope.ticks_per_generation,
        "required_obligations": sorted(envelope.required_obligations),
    }
    if envelope.reproduction_exposure_capacity is not None:
        payload["reproduction_exposure_capacity"] = (
            envelope.reproduction_exposure_capacity.payload()
        )
    return payload


def composed_registry_identity_v1(payload: Mapping[str, Any]) -> dict[str, str]:
    """Expose a versioned canonical identity for staging and receipts."""

    return {
        "schema_version": COMPOSER_SCHEMA_VERSION,
        "registry_sha256": canonical_sha256(payload),
    }


__all__ = [
    "AgentIdAllocationCapacityV1",
    "COMPOSER_SCHEMA_VERSION",
    "InheritanceCrossPathRegistryCompositionError",
    "InheritanceCrossPathRegistryEnvelopeV1",
    "ReproductionExposureCapacityV1",
    "compose_inheritance_cross_path_registry_v2",
    "composed_registry_identity_v1",
]
