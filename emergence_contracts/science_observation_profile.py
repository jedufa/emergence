"""Frozen route-neutral observation profile descriptors and catalog identity."""

from __future__ import annotations

from dataclasses import dataclass

from ._validation import require_instance, require_int
from .canonical import canonical_sha256
from .errors import contract_invalid
from .ids import ContentDigest, SchemaVersion
from .science_instrument import ScienceObservationProfileV1, ScienceRecordKindV1

SCIENCE_OBSERVATION_PROFILE_DESCRIPTOR_SCHEMA_VERSION = SchemaVersion(
    "eiel_science_observation_profile_descriptor_v1"
)

_KIND_ORDINAL = {kind: ordinal for ordinal, kind in enumerate(ScienceRecordKindV1)}
_PROFILE_ORDINAL = {
    profile: ordinal for ordinal, profile in enumerate(ScienceObservationProfileV1)
}


def _token_tuple(value: object, *, field: str) -> tuple[str, ...]:
    if (
        not isinstance(value, tuple)
        or not value
        or any(not isinstance(item, str) or not item for item in value)
        or value != tuple(sorted(set(value)))
    ):
        raise contract_invalid(
            "value must be a nonempty sorted unique token tuple", field=field
        )
    return value


@dataclass(frozen=True, slots=True)
class ScienceObservationProfileDescriptorV1:
    profile: ScienceObservationProfileV1
    observation_abi_version: SchemaVersion
    record_kinds: tuple[ScienceRecordKindV1, ...]
    required_nonempty_record_kinds: tuple[ScienceRecordKindV1, ...]
    capability_ids: tuple[str, ...]
    producer_phase_registry_ordinals: tuple[tuple[str, int], ...]
    default_partition_span_ticks: int
    schema_version: SchemaVersion = (
        SCIENCE_OBSERVATION_PROFILE_DESCRIPTOR_SCHEMA_VERSION
    )

    def __post_init__(self) -> None:
        require_instance(self.profile, ScienceObservationProfileV1, field="profile")
        require_instance(
            self.observation_abi_version,
            SchemaVersion,
            field="observation_abi_version",
        )
        for field_name in ("record_kinds", "required_nonempty_record_kinds"):
            kinds = getattr(self, field_name)
            if (
                not isinstance(kinds, tuple)
                or not kinds
                or any(not isinstance(item, ScienceRecordKindV1) for item in kinds)
                or kinds
                != tuple(sorted(set(kinds), key=lambda item: _KIND_ORDINAL[item]))
            ):
                raise contract_invalid(
                    "record kinds must be nonempty, unique, and in registry order",
                    field=field_name,
                )
        if not set(self.required_nonempty_record_kinds).issubset(self.record_kinds):
            raise contract_invalid(
                "required nonempty record kinds must belong to the profile",
                field="required_nonempty_record_kinds",
            )
        _token_tuple(self.capability_ids, field="capability_ids")
        registry = self.producer_phase_registry_ordinals
        if (
            not isinstance(registry, tuple)
            or not registry
            or any(not isinstance(row, tuple) or len(row) != 2 for row in registry)
            or registry != tuple(sorted(registry))
            or len({row[0] for row in registry}) != len(registry)
        ):
            raise contract_invalid(
                "producer phase registry must be nonempty, sorted, and unique",
                field="producer_phase_registry_ordinals",
            )
        if any(
            not isinstance(component_id, str)
            or not component_id
            or type(ordinal) is not int
            or ordinal < 0
            for component_id, ordinal in registry
        ):
            raise contract_invalid(
                "producer phase registry row is invalid",
                field="producer_phase_registry_ordinals",
            )
        require_int(
            self.default_partition_span_ticks,
            field="default_partition_span_ticks",
            minimum=1,
        )
        if self.schema_version != (
            SCIENCE_OBSERVATION_PROFILE_DESCRIPTOR_SCHEMA_VERSION
        ):
            raise contract_invalid(
                "unsupported observation profile descriptor schema",
                field="schema_version",
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": str(self.schema_version),
            "profile_id": self.profile.value,
            "observation_abi_version": str(self.observation_abi_version),
            "record_kinds": [kind.value for kind in self.record_kinds],
            "required_nonempty_record_kinds": [
                kind.value for kind in self.required_nonempty_record_kinds
            ],
            "capability_ids": list(self.capability_ids),
            "producer_phase_registry_ordinals": {
                component_id: ordinal
                for component_id, ordinal in self.producer_phase_registry_ordinals
            },
            "default_partition_span_ticks": self.default_partition_span_ticks,
        }

    def descriptor_sha256(self) -> ContentDigest:
        return canonical_sha256(self.to_dict())

    @classmethod
    def from_dict(cls, value: object) -> "ScienceObservationProfileDescriptorV1":
        required = {
            "schema_version",
            "profile_id",
            "observation_abi_version",
            "record_kinds",
            "required_nonempty_record_kinds",
            "capability_ids",
            "producer_phase_registry_ordinals",
            "default_partition_span_ticks",
        }
        if not isinstance(value, dict) or set(value) != required:
            raise contract_invalid(
                "observation profile descriptor has unknown or missing fields",
                field="science_observation_profile_descriptor",
            )
        registry = value["producer_phase_registry_ordinals"]
        if not isinstance(registry, dict):
            raise contract_invalid(
                "producer phase registry must be an object",
                field="producer_phase_registry_ordinals",
            )
        for field_name in (
            "record_kinds",
            "required_nonempty_record_kinds",
            "capability_ids",
        ):
            if not isinstance(value[field_name], list):
                raise contract_invalid("value must be an array", field=field_name)
        return cls(
            profile=ScienceObservationProfileV1(value["profile_id"]),
            observation_abi_version=SchemaVersion(value["observation_abi_version"]),
            record_kinds=tuple(
                ScienceRecordKindV1(item) for item in value["record_kinds"]
            ),
            required_nonempty_record_kinds=tuple(
                ScienceRecordKindV1(item)
                for item in value["required_nonempty_record_kinds"]
            ),
            capability_ids=tuple(value["capability_ids"]),
            producer_phase_registry_ordinals=tuple(sorted(registry.items())),
            default_partition_span_ticks=value["default_partition_span_ticks"],
            schema_version=SchemaVersion(value["schema_version"]),
        )


def science_observation_profile_catalog_sha256_v1(
    descriptors: tuple[ScienceObservationProfileDescriptorV1, ...],
) -> ContentDigest:
    if (
        not isinstance(descriptors, tuple)
        or not descriptors
        or any(
            not isinstance(item, ScienceObservationProfileDescriptorV1)
            for item in descriptors
        )
        or descriptors
        != tuple(
            sorted(
                descriptors,
                key=lambda item: _PROFILE_ORDINAL[item.profile],
            )
        )
        or len({item.profile for item in descriptors}) != len(descriptors)
    ):
        raise contract_invalid(
            "observation profile catalog must be nonempty, unique, and ordered",
            field="observation_profile_descriptors",
        )
    return canonical_sha256([item.to_dict() for item in descriptors])


__all__ = (
    "SCIENCE_OBSERVATION_PROFILE_DESCRIPTOR_SCHEMA_VERSION",
    "ScienceObservationProfileDescriptorV1",
    "science_observation_profile_catalog_sha256_v1",
)
