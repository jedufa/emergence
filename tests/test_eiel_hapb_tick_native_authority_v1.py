from __future__ import annotations

import inspect
from pathlib import Path

import pytest
import yaml

from engine.campaigns.compiler import compile_campaign
from engine.campaigns.providers import (
    CampaignProviderError,
    ProductionEngineLifecycleV1,
    _checkpoint_identity_contract_v1,
    _native_public_authoritative_state_receipts_v1,
    _native_public_execution_receipt_v1,
    default_execution_provider_registry,
)
from engine.campaigns.tick_native import TickNativeExecutionRequestV1, TickNativeError
from engine.campaigns.tick_native_control import assert_tick_native_plan_payload_v1


ROOT = Path(__file__).resolve().parents[1]
SPEC_NAMES = (
    "b48_total960_ticks_rngv2_force_v1.yaml",
    "b48_total960_ticks_rngv2_reuse_v1.yaml",
)
LEGACY_FIELDS = {
    "generations",
    "ticks_per_generation",
    "physical_window_generations",
    "checkpoint_milestones",
}
PUBLIC_GENERATION_FIELDS = {
    "generation",
    "generations",
    "generation_min",
    "generation_max",
    "generation_count",
    "logical_generation_count",
    "initial_generation",
    "terminal_generation",
    "start_generation",
    "end_generation",
    "absolute_generation",
    "ticks_per_generation",
    "admitted_ticks_per_generation",
    "request_ticks_per_generation",
    "physical_window_generations",
    "resident_segment_generations",
    "physical_storage_generations",
}


def _plan(name: str):
    path = ROOT / "_spec" / "campaigns" / "v1" / "hapb_rngv2" / name
    return compile_campaign(yaml.safe_load(path.read_text(encoding="utf-8")))


def _assert_no_public_generation_fields(value: object) -> None:
    if isinstance(value, dict):
        assert not PUBLIC_GENERATION_FIELDS.intersection(value)
        for item in value.values():
            _assert_no_public_generation_fields(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _assert_no_public_generation_fields(item)


@pytest.mark.parametrize("name", SPEC_NAMES)
def test_hapb_tick_native_plan_has_no_legacy_execution_authority(name: str) -> None:
    payload = _plan(name).payload()
    assert_tick_native_plan_payload_v1(payload)
    execution = payload["execution"]
    assert execution["execution_route_identity"] == "tick_native_execution_route_v1"
    assert execution["total_ticks"] == 960
    assert execution["checkpoint_ticks"] == [480, 960]
    assert not LEGACY_FIELDS.intersection(execution)


def test_hapb_tick_native_provider_request_and_resources_are_tick_authoritative() -> None:
    payload = _plan(SPEC_NAMES[0]).payload()
    provider = default_execution_provider_registry().get(payload["resolved_profile"])
    assert provider is not None
    request = provider.translate_batch(
        payload, payload["physical_batches"][0]["batch_id"]
    )
    request_payload = request.payload()
    assert request_payload["time_authority"] == "absolute_tick_v1"
    assert request_payload["total_ticks"] == 960
    assert request_payload["checkpoint_ticks"] == [480, 960]
    assert request_payload["tick_native_control"]["time_authority"] == (
        "absolute_tick_v1"
    )
    assert not LEGACY_FIELDS.intersection(request_payload)

    route = provider.execution_route_contract(payload, request)
    assert request.execution_route_sha256 == route.execution_route_sha256
    route_payload = route.payload()
    _assert_no_public_generation_fields(route_payload)
    absolute_tick = next(
        component
        for component in route_payload["components"]
        if component["component_id"] == "absolute_tick_time_contract"
    )
    assert absolute_tick["fresh_owner_partial_tick_restore"] is True
    assert absolute_tick["legacy_horizon_authority"] is False
    assert "fresh_owner_partial_generation_restore" not in absolute_tick
    assert "legacy_generation_authority" not in absolute_tick
    estimate = provider.estimate_resources(payload, (request,))
    assert estimate["time_authority"] == "absolute_tick_v1"
    assert estimate["total_ticks"] == 960
    assert not LEGACY_FIELDS.intersection(estimate)
    assert "legacy_compatibility" in estimate["tick_native_resource_projection"]


def test_tick_native_request_rejects_legacy_horizon_fields() -> None:
    with pytest.raises(TickNativeError):
        TickNativeExecutionRequestV1.from_execution(
            {
                "execution_route_identity": "tick_native_execution_route_v1",
                "total_ticks": 960,
                "generations": 6,
                "ticks_per_generation": 160,
            }
        )


def test_native_receipt_and_checkpoint_identity_survive_compatibility_removal() -> None:
    sha = "a" * 64
    digest_map = {"cell-0": sha}
    identity = _checkpoint_identity_contract_v1(
        execution_route_sha256=sha,
        generation=None,
        absolute_tick=480,
        next_absolute_tick=480,
        source_import_projection_digest_sha256s=digest_map,
        parent_checkpoint_projection_digest_sha256s=None,
        terminal_projection_digest_sha256s=digest_map,
        canonical_terminal_state_digest_sha256s=digest_map,
        canonical_checkpoint_storage_digest_sha256s=digest_map,
    )
    assert identity["time_authority"] == "absolute_tick_v1"
    assert identity["absolute_tick"] == 480
    assert "generation" not in identity

    receipt = _native_public_execution_receipt_v1(
        {
            "segment_ordinal": 1,
            "start_generation": 3,
            "end_generation": 6,
            "ticks_per_generation": 160,
            "start_next_tick": 480,
            "end_next_tick": 960,
        }
    )
    compatibility = receipt.pop("legacy_compatibility")
    assert receipt["time_authority"] == "absolute_tick_v1"
    assert receipt["start_next_tick"] == 480
    assert receipt["end_next_tick"] == 960
    assert compatibility["start_generation"] == 3
    assert compatibility["end_generation"] == 6
    assert compatibility["ticks_per_generation"] == 160


def test_authoritative_state_receipt_is_tick_authoritative_after_projection() -> None:
    receipts = _native_public_authoritative_state_receipts_v1(
        [
            {
                "schema_version": "eiel_authoritative_state_pre_dispatch_receipt_v1",
                "segment_ordinal": 1,
                "state_schema_sha256": "a" * 64,
                "start_generation": 20,
                "end_generation": 21,
                "start_next_tick": 3359,
                "end_next_tick": 3360,
                "decision": "admitted",
            }
        ]
    )
    assert len(receipts) == 1
    receipt = receipts[0]

    compatibility = receipt["legacy_compatibility"]
    assert receipt["time_authority"] == "absolute_tick_v1"
    assert receipt["start_next_tick"] == 3359
    assert receipt["end_next_tick"] == 3360
    assert compatibility == {
        "start_generation": 20,
        "end_generation": 21,
    }

    without_compatibility = dict(receipt)
    without_compatibility.pop("legacy_compatibility")
    assert not PUBLIC_GENERATION_FIELDS.intersection(without_compatibility)
    assert without_compatibility["time_authority"] == receipt["time_authority"]
    assert without_compatibility["start_next_tick"] == 3359
    assert without_compatibility["end_next_tick"] == 3360
    assert without_compatibility["decision"] == receipt["decision"] == "admitted"


def test_authoritative_state_emission_is_tick_native_and_fail_closed() -> None:
    provider_source = inspect.getsource(ProductionEngineLifecycleV1._execute_segments)
    assert '"start_next_tick": segment_start_next_tick' in provider_source
    assert '"end_next_tick": segment_end_next_tick' in provider_source
    assert "_native_public_authoritative_state_receipts_v1(" in provider_source

    with pytest.raises(CampaignProviderError, match="exact absolute-tick span"):
        _native_public_authoritative_state_receipts_v1(
            [
                {
                    "schema_version": "eiel_authoritative_state_pre_dispatch_receipt_v1",
                    "start_generation": 20,
                    "end_generation": 21,
                }
            ]
        )
