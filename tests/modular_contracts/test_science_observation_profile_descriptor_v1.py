from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import jsonschema
import pytest

from emergence_contracts import (
    ContractError,
    SchemaVersion,
    ScienceObservationProfileDescriptorV1,
    ScienceObservationProfileV1,
    ScienceRecordKindV1,
    science_observation_profile_catalog_sha256_v1,
)


def _descriptor() -> ScienceObservationProfileDescriptorV1:
    return ScienceObservationProfileDescriptorV1(
        profile=ScienceObservationProfileV1.STANDARD,
        observation_abi_version=SchemaVersion("eiel_science_observation_abi_v1"),
        record_kinds=(
            ScienceRecordKindV1.CONTROL_BINDING,
            ScienceRecordKindV1.WORLD_STATE_SNAPSHOT,
            ScienceRecordKindV1.WORLD_COMPLETION,
        ),
        required_nonempty_record_kinds=(
            ScienceRecordKindV1.CONTROL_BINDING,
            ScienceRecordKindV1.WORLD_COMPLETION,
        ),
        capability_ids=("causal_lifecycle", "terminal_state"),
        producer_phase_registry_ordinals=(
            ("science-control-producer-v1", 0),
            ("science-world-producer-v1", 1),
        ),
        default_partition_span_ticks=1000,
    )


def test_profile_descriptor_round_trips_and_matches_wire_schema() -> None:
    descriptor = _descriptor()
    payload = descriptor.to_dict()
    schema = json.loads(
        (
            Path(__file__).parents[2]
            / "_spec/science_instrument/v1/science_observation_profile_descriptor.schema.json"
        ).read_text(encoding="utf-8")
    )

    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.validate(payload, schema)
    assert ScienceObservationProfileDescriptorV1.from_dict(payload) == descriptor


def test_profile_catalog_hash_changes_only_with_profile_contract_content() -> None:
    descriptor = _descriptor()
    original = science_observation_profile_catalog_sha256_v1((descriptor,))
    changed = science_observation_profile_catalog_sha256_v1(
        (replace(descriptor, default_partition_span_ticks=500),)
    )

    assert changed != original


def test_profile_descriptor_rejects_noncanonical_or_incomplete_rows() -> None:
    descriptor = _descriptor()
    with pytest.raises(ContractError, match="registry order"):
        replace(descriptor, record_kinds=tuple(reversed(descriptor.record_kinds)))
    with pytest.raises(ContractError, match="belong to the profile"):
        replace(
            descriptor,
            required_nonempty_record_kinds=(ScienceRecordKindV1.DEATH,),
        )
    with pytest.raises(ContractError, match="sorted unique token"):
        replace(descriptor, capability_ids=("terminal_state", "causal_lifecycle"))
