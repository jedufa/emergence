from __future__ import annotations

import hashlib
import json
from pathlib import Path

import jsonschema
import pytest

from eiel_composition.science_request_v2 import (
    SCIENCE_CAMPAIGN_REF_SCHEMA_VERSION,
    ScienceRequestV2Error,
    normalize_rich_production_science_request_v2,
)
from engine.campaigns.baseline_v1 import BaselineDeclarationV1
from engine.qrr.production_front_door_v1 import (
    PRODUCTION_SCIENCE_REQUEST_V2_SCHEMA_VERSION,
    validate_production_science_request_v2,
)
from emergence_contracts import ContentDigest, PRODUCT_ID


class _Resolver:
    def __init__(self, content: bytes) -> None:
        self.content = content

    def resolve(self, immutable_object_id: str) -> bytes:
        assert immutable_object_id == "campaign-object-001"
        return self.content


def _baseline() -> dict[str, object]:
    return BaselineDeclarationV1(
        profile_id="EIEL_PRODUCTION_BASELINE_V1",
        intent="development",
        deviations=(),
    ).payload()


def _campaign() -> dict[str, object]:
    return {
        "schema_version": "eiel_science_campaign_v1",
        "product_id": PRODUCT_ID,
        "campaign_id": "generic-rich-request-001",
        "instrument": {
            "product_id": PRODUCT_ID,
            "instrument_manifest_sha256": ContentDigest("sha256", "1" * 64).to_dict(),
        },
        "question": {"question_id": "Q-GENERIC", "no_claims": True},
        "design": {"dimensions": [], "expected_cell_count": 1},
        "world": {
            "world_profile_id": "world-v1",
            "geometry": {"width": 32, "height": 32},
            "resident_capacity": 512,
        },
        "initialization": {
            "initial_population": 256,
            "initial_live_count": 256,
            "founder_profiles": ["founder-v1"],
            "exact_founder_composition": [
                {"founder_profile_id": "founder-v1", "count": 256}
            ],
            "spatial_placement_profile": "placement-v1",
            "standing_variation_profile": "variation-v1",
        },
        "controls": [],
        "seeds": [
            {"owner": "task_environment", "binding_kind": "literal", "literal": 7}
        ],
        "time": {
            "authority": "absolute_tick_v1",
            "start_next_tick": 0,
            "total_ticks": 5000,
            "observation_ticks": [2500, 5000],
            "checkpoint_ticks": [2500, 5000],
            "branches": [],
        },
        "observations": {
            "profile_id": "SCIENCE_STANDARD_V1",
            "required_capability_ids": ["causal_lifecycle"],
        },
        "execution": {
            "machine_profile_id": "EIEL_PRODUCTION_BASELINE_V1",
            "logical_world_count": 1,
            "batching_policy": "authority_selected",
        },
        "reducers": [],
    }


def _inline_request() -> dict[str, object]:
    return {
        "schema_version": PRODUCTION_SCIENCE_REQUEST_V2_SCHEMA_VERSION,
        "request_id": "rich-request-001",
        "baseline": _baseline(),
        "science_campaign": _campaign(),
    }


def test_inline_rich_request_normalizes_science_compact_and_is_stable() -> None:
    first = normalize_rich_production_science_request_v2(
        _inline_request()
    ).normalized_request
    second = normalize_rich_production_science_request_v2(
        _inline_request()
    ).normalized_request

    assert first == second
    assert first["science_campaign"]["artifact_lifecycle"]["profile_id"] == (
        "SCIENCE_COMPACT"
    )
    assert set(first) == {
        "schema_version",
        "request_id",
        "baseline",
        "science_campaign",
        "request_sha256",
    }


def test_reference_and_inline_normalize_to_same_request_identity() -> None:
    inline = _inline_request()
    content = json.dumps(_campaign(), sort_keys=True).encode("utf-8")
    referenced = {
        "schema_version": PRODUCTION_SCIENCE_REQUEST_V2_SCHEMA_VERSION,
        "request_id": inline["request_id"],
        "baseline": inline["baseline"],
        "science_campaign_ref": {
            "schema_version": SCIENCE_CAMPAIGN_REF_SCHEMA_VERSION,
            "resolver_id": "test-registry",
            "immutable_object_id": "campaign-object-001",
            "content_sha256": ContentDigest(
                "sha256", hashlib.sha256(content).hexdigest()
            ).to_dict(),
        },
    }
    resolved = normalize_rich_production_science_request_v2(
        referenced, resolvers={"test-registry": _Resolver(content)}
    ).normalized_request
    assert (
        resolved
        == normalize_rich_production_science_request_v2(inline).normalized_request
    )


@pytest.mark.parametrize(
    "mutation, message",
    (
        (lambda value: value.update(world_width=32), "flat-only"),
        (
            lambda value: value.update(science_campaign_ref={}),
            "exactly one campaign discriminant",
        ),
        (lambda value: value.update(executable_hook="module:function"), "flat-only"),
    ),
)
def test_rich_request_rejects_mixed_unknown_or_dual_forms(
    mutation, message: str
) -> None:
    request = _inline_request()
    mutation(request)
    with pytest.raises(ScienceRequestV2Error, match=message):
        normalize_rich_production_science_request_v2(request)


def test_reference_rejects_unregistered_pathlike_or_tampered_content() -> None:
    content = json.dumps(_campaign(), sort_keys=True).encode("utf-8")
    reference = {
        "schema_version": PRODUCTION_SCIENCE_REQUEST_V2_SCHEMA_VERSION,
        "request_id": "rich-request-001",
        "baseline": _baseline(),
        "science_campaign_ref": {
            "schema_version": SCIENCE_CAMPAIGN_REF_SCHEMA_VERSION,
            "resolver_id": "https://example.invalid/campaign",
            "immutable_object_id": "/tmp/campaign.json",
            "content_sha256": ContentDigest("sha256", "0" * 64).to_dict(),
        },
    }
    with pytest.raises(ScienceRequestV2Error, match="path-free"):
        normalize_rich_production_science_request_v2(
            reference, resolvers={"test-registry": _Resolver(content)}
        )

    reference["science_campaign_ref"]["resolver_id"] = "test-registry"
    reference["science_campaign_ref"]["immutable_object_id"] = "campaign-object-001"
    with pytest.raises(ScienceRequestV2Error, match="SHA-256 mismatch"):
        normalize_rich_production_science_request_v2(
            reference, resolvers={"test-registry": _Resolver(content)}
        )


def test_flat_v2_normalization_remains_on_legacy_path() -> None:
    flat = {
        "schema_version": PRODUCTION_SCIENCE_REQUEST_V2_SCHEMA_VERSION,
        "request_id": "flat-request-001",
        "world_width": 4,
        "batch_cells": 1,
        "total_ticks": 2,
        "checkpoint_ticks": [2],
        "seeds": [7],
        "configurations": [{"id": "config-0", "controls": {"patch.grid_size": 4}}],
        "deliberate_restore_ticks": [],
        "observability": {"living_lineage_canopy": False},
        "baseline": _baseline(),
    }
    normalized = validate_production_science_request_v2(flat)
    assert "science_campaign" not in normalized
    assert normalized["world_width"] == 4
    assert normalized["batch_cells"] == 1


def test_generic_factorial_example_is_normalizable_and_schema_valid() -> None:
    root = Path(__file__).parents[1]
    request = json.loads(
        (
            root / "_spec/science_instrument/v1/examples/"
            "generic_factorial_request_v2.json"
        ).read_text(encoding="utf-8")
    )
    campaign_schema = json.loads(
        (root / "_spec/science_instrument/v1/science_campaign.schema.json").read_text(
            encoding="utf-8"
        )
    )

    jsonschema.Draft202012Validator.check_schema(campaign_schema)
    jsonschema.Draft202012Validator(campaign_schema).validate(
        request["science_campaign"]
    )
    normalized = normalize_rich_production_science_request_v2(request)

    assert normalized.campaign.design.to_value()["expected_cell_count"] == 12
    assert normalized.campaign.execution.to_value()["logical_world_count"] == 12
    assert (
        normalized.normalized_request["science_campaign"]["artifact_lifecycle"][
            "profile_id"
        ]
        == "SCIENCE_COMPACT"
    )
