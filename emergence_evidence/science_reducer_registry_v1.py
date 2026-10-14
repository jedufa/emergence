"""Canonical registry for pure science reducers.

The registry is deliberately small and immutable.  It maps a reducer identity
to the descriptor and protocol implementation that share that identity; it
does not select routes, inspect a provider, or execute a simulation.
"""

from __future__ import annotations

from dataclasses import dataclass

from emergence_contracts import (
    CanonicalPayloadV1,
    ContentDigest,
    ScienceEvidencePartitionV1,
    ScienceReducerDescriptorV1,
)

from .science_reducer_protocol_v1 import (
    ScienceReducerCheckpointV1,
    ScienceReducerError,
    ScienceReducerImplementationV1,
    ScienceReducerProgressV1,
    execute_science_reducer_v1,
)


@dataclass(frozen=True, slots=True)
class ScienceReducerRegistrationV1:
    """One descriptor/implementation pair in the reducer registry."""

    descriptor: ScienceReducerDescriptorV1
    implementation: ScienceReducerImplementationV1

    def __post_init__(self) -> None:
        if self.implementation.reducer_id != self.descriptor.reducer_id:
            raise ScienceReducerError("registered implementation ID differs")
        if (
            self.implementation.implementation_sha256
            != self.descriptor.implementation_sha256
        ):
            raise ScienceReducerError("registered implementation identity differs")

    @property
    def reducer_id(self) -> str:
        return self.descriptor.reducer_id


@dataclass(frozen=True, slots=True)
class ScienceReducerRegistryV1:
    """Deterministic, duplicate-free registry of reducer registrations."""

    registrations: tuple[ScienceReducerRegistrationV1, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.registrations, tuple):
            raise ScienceReducerError("reducer registrations must be a tuple")
        if any(
            not isinstance(item, ScienceReducerRegistrationV1)
            for item in self.registrations
        ):
            raise ScienceReducerError("reducer registration has an invalid type")
        ids = tuple(item.reducer_id for item in self.registrations)
        if ids != tuple(sorted(ids)) or len(ids) != len(set(ids)):
            raise ScienceReducerError(
                "reducer registrations must be sorted and have unique IDs"
            )

    def get(self, reducer_id: str) -> ScienceReducerRegistrationV1:
        for registration in self.registrations:
            if registration.reducer_id == reducer_id:
                return registration
        raise ScienceReducerError(f"reducer is not registered: {reducer_id!r}")

    def descriptor(self, reducer_id: str) -> ScienceReducerDescriptorV1:
        return self.get(reducer_id).descriptor

    def implementation(self, reducer_id: str) -> ScienceReducerImplementationV1:
        return self.get(reducer_id).implementation

    def reducer_ids(self) -> tuple[str, ...]:
        return tuple(item.reducer_id for item in self.registrations)


def default_science_reducer_registry_v1() -> ScienceReducerRegistryV1:
    """Return the immutable canonical registry for this source distribution."""

    from .science_mutsup_direct_retention_v1 import (
        mutsup_direct_retention_reducer_descriptor_v1,
        mutsup_direct_retention_reducer_v1,
    )

    registration = ScienceReducerRegistrationV1(
        descriptor=mutsup_direct_retention_reducer_descriptor_v1(),
        implementation=mutsup_direct_retention_reducer_v1(),
    )
    return ScienceReducerRegistryV1(registrations=(registration,))


def science_reducer_registry_v1() -> ScienceReducerRegistryV1:
    """Alias for callers using the registry name as a constructor."""

    return default_science_reducer_registry_v1()


def execute_registered_science_reducer_v1(
    reducer_id: str,
    configuration: CanonicalPayloadV1,
    partitions: tuple[ScienceEvidencePartitionV1, ...],
    *,
    evidence_capability_ids: tuple[str, ...],
    campaign_spec_sha256: ContentDigest,
    instrument_manifest_sha256: ContentDigest,
    route_binding_sha256: ContentDigest,
    science_semantic_acquisition_sha256: ContentDigest,
    physical_execution_plan_sha256: ContentDigest,
    science_execution_plan_sha256: ContentDigest,
    reduction_plan_sha256: ContentDigest,
    registry: ScienceReducerRegistryV1 | None = None,
    resume: ScienceReducerCheckpointV1 | None = None,
    stop_after_partitions: int | None = None,
) -> ScienceReducerProgressV1:
    """Resolve a registered implementation and invoke the canonical protocol."""

    selected = (registry or default_science_reducer_registry_v1()).get(reducer_id)
    return execute_science_reducer_v1(
        selected.descriptor,
        selected.implementation,
        configuration,
        partitions,
        evidence_capability_ids=evidence_capability_ids,
        campaign_spec_sha256=campaign_spec_sha256,
        instrument_manifest_sha256=instrument_manifest_sha256,
        route_binding_sha256=route_binding_sha256,
        science_semantic_acquisition_sha256=science_semantic_acquisition_sha256,
        physical_execution_plan_sha256=physical_execution_plan_sha256,
        science_execution_plan_sha256=science_execution_plan_sha256,
        reduction_plan_sha256=reduction_plan_sha256,
        resume=resume,
        stop_after_partitions=stop_after_partitions,
    )


__all__ = (
    "ScienceReducerRegistrationV1",
    "ScienceReducerRegistryV1",
    "default_science_reducer_registry_v1",
    "execute_registered_science_reducer_v1",
    "science_reducer_registry_v1",
)
