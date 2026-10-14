from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import jsonschema
import pytest

from eiel_composition import ScienceCompilationBundleV1
from engine.campaigns.baseline_v1 import BaselineDeclarationV1
from eiel_composition.science_qrr_projection_v1 import (
    ScienceQRRProjectionError,
    ScienceQRRRouteAdapterV1,
    bind_science_qrr_resolution_v1,
    project_rich_science_request_to_flat_v2,
)
from eiel_composition.science_request_v2 import (
    normalize_rich_production_science_request_v2,
)
from emergence_contracts import (
    PHYSICAL_EXECUTION_CORE_SCHEMA_VERSION,
    SEMANTIC_ACQUISITION_CORE_SCHEMA_VERSION,
    CanonicalPayloadV1,
    ContentDigest,
    PRODUCT_ID,
    SCIENCE_ROUTE_BINDING_SCHEMA_VERSION,
    SchemaVersion,
    ScienceCampaignV1,
    ScienceExecutionPlanV1,
    ScienceReductionPlanV1,
    ScienceRouteBindingV1,
    canonical_sha256,
)


def _digest(character: str) -> ContentDigest:
    return ContentDigest("sha256", character * 64)


def _campaign_document() -> dict[str, object]:
    return {
        "schema_version": "eiel_science_campaign_v1",
        "product_id": PRODUCT_ID,
        "campaign_id": "qrr-factorization-v1",
        "instrument": {
            "product_id": PRODUCT_ID,
            "instrument_manifest_sha256": _digest("1").to_dict(),
        },
        "question": {"question_id": "Q-QRR", "no_claims": True},
        "design": {
            "dimensions": [
                {
                    "dimension_id": "task_seed",
                    "binding_path": "seeds.task_environment",
                    "canonical_values": [7, 11],
                }
            ],
            "expected_cell_count": 2,
        },
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
            {
                "owner": "task_environment",
                "binding_kind": "dimension_bound",
                "dimension_id": "task_seed",
            }
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
            "logical_world_count": 2,
            "batching_policy": "authority_selected",
        },
        "reducers": [],
    }


def _normalized_request():
    request = {
        "schema_version": "eiel_production_science_request_v2",
        "request_id": "qrr-rich-request-v1",
        "baseline": BaselineDeclarationV1(
            profile_id="EIEL_PRODUCTION_BASELINE_V1",
            intent="development",
            deviations=(),
        ).payload(),
        "science_campaign": _campaign_document(),
    }
    return normalize_rich_production_science_request_v2(request)


def _adapter(route_id: str = "route-a") -> ScienceQRRRouteAdapterV1:
    return ScienceQRRRouteAdapterV1(
        route_id=route_id,
        machine_profile_id="EIEL_PRODUCTION_BASELINE_V1",
        world_profile_id="world-v1",
        world_width=32,
        maximum_batch_cells=136,
        task_seed_owner="task_environment",
        control_path_by_id=(),
        seed_path_by_owner=(),
        fixed_flat_controls=CanonicalPayloadV1.from_value(
            SchemaVersion("eiel_science_qrr_fixed_controls_v1"),
            {"patch.grid_size": 32},
        ),
        acs_capture_instrumentation=True,
        acs_checkpoint_slots=2,
        living_lineage_canopy=False,
        lifecycle_total_max_bytes=12_000_000_000,
    )


def _route(adapter: ScienceQRRRouteAdapterV1) -> ScienceRouteBindingV1:
    return ScienceRouteBindingV1(
        instrument_manifest_sha256=_digest("1"),
        route_id=adapter.route_id,
        route_profile_sha256=_digest("2"),
        machine_profile_id=adapter.machine_profile_id,
        capability_projection_sha256=_digest("3"),
        native_build_closure_sha256=_digest("4"),
        execution_adapter_sha256=adapter.execution_adapter_sha256(),
        control_adapter_sha256=adapter.control_adapter_sha256(),
        observation_adapter_sha256=_digest("5"),
        checkpoint_adapter_sha256=_digest("6"),
        semantic_conformance_sha256=_digest("7"),
        resource_envelope_sha256=_digest("8"),
        qualified_envelope_sha256=_digest("9"),
        qualification_authority_sha256=_digest("a"),
        schema_version=SCIENCE_ROUTE_BINDING_SCHEMA_VERSION,
    )


def _bundle(campaign: ScienceCampaignV1, route: ScienceRouteBindingV1):
    cells = [
        {
            "semantic_cell_id": f"cell-{seed}",
            "logical_world_id": index,
            "dimension_bindings": {"task_seed": seed},
            "seed_bindings": {"task_environment": seed},
            "control_bindings": [],
            "world": campaign.world.to_value(),
            "initialization": campaign.initialization.to_value(),
        }
        for index, seed in enumerate((7, 11))
    ]
    execution_plan = ScienceExecutionPlanV1(
        instrument_manifest_sha256=_digest("1"),
        route_binding_sha256=route.route_binding_sha256(),
        crca_semantic_run_digest=_digest("b"),
        semantic_acquisition_core=CanonicalPayloadV1.from_value(
            SEMANTIC_ACQUISITION_CORE_SCHEMA_VERSION,
            {
                "cells": cells,
                "time": campaign.time.to_value(),
                "observations": campaign.observations.to_value(),
            },
        ),
        physical_execution_core=CanonicalPayloadV1.from_value(
            PHYSICAL_EXECUTION_CORE_SCHEMA_VERSION,
            {"partition": [[0, 1]], "route_id": route.route_id},
        ),
    )
    reduction_plan = ScienceReductionPlanV1(
        reducer_descriptors=(),
        normalized_configurations=CanonicalPayloadV1.from_value(
            SchemaVersion("eiel_science_reducer_configurations_v1"), []
        ),
        evidence_requirements=CanonicalPayloadV1.from_value(
            SchemaVersion("eiel_science_reduction_evidence_requirements_v1"),
            {"required_observation_capability_ids": []},
        ),
        offline_decisions=CanonicalPayloadV1.from_value(
            SchemaVersion("eiel_science_offline_decisions_v1"), {}
        ),
        output_contract_ids=(),
    )
    return ScienceCompilationBundleV1(
        campaign_spec_sha256=campaign.campaign_spec_sha256(),
        compiler_component_sha256=_digest("c"),
        crca_projection_sha256=_digest("d"),
        execution_plan=execution_plan,
        reduction_plan=reduction_plan,
        semantic_cells=CanonicalPayloadV1.from_value(
            SchemaVersion("eiel_science_semantic_cells_v1"), cells
        ),
    )


def _project(route_id: str = "route-a"):
    normalized = _normalized_request()
    adapter = _adapter(route_id)
    route = _route(adapter)
    bundle = _bundle(normalized.campaign, route)
    projection = project_rich_science_request_to_flat_v2(
        normalized.normalized_request,
        normalized.campaign,
        bundle,
        route_binding=route,
        route_adapter=adapter,
    )
    return normalized, adapter, route, bundle, projection


def test_rich_projection_is_exact_factorized_and_science_compact() -> None:
    _, adapter, _, _, projection = _project()
    flat = projection.flat_request

    assert flat["batch_cells"] == 2
    assert flat["seeds"] == [7, 11]
    assert len(flat["configurations"]) == 1
    assert flat["configurations"][0]["controls"] == {"patch.grid_size": 32}
    assert flat["artifact_lifecycle"]["lifecycle_profile"] == "SCIENCE_COMPACT"
    assert flat["artifact_lifecycle"]["checkpoints"]["checkpoint_ticks"] == [
        2500,
        5000,
    ]
    assert projection.projection_receipt["effects_performed"] is False
    schema = json.loads(
        (
            Path(__file__).parents[1]
            / "_spec/science_instrument/v1/science_qrr_route_adapter.schema.json"
        ).read_text(encoding="utf-8")
    )
    jsonschema.validate(adapter.to_dict(), schema)


def test_route_swap_preserves_flat_semantics_but_changes_binding_receipt() -> None:
    _, _, _, _, first = _project("route-a")
    _, _, _, _, second = _project("route-b")

    assert first.flat_request == second.flat_request
    assert (
        first.projection_receipt["route_binding_sha256"]
        != second.projection_receipt["route_binding_sha256"]
    )
    assert (
        first.projection_receipt["receipt_sha256"]
        != second.projection_receipt["receipt_sha256"]
    )


def test_adapter_identity_and_nonrectangular_semantics_fail_closed() -> None:
    normalized, adapter, route, bundle, _ = _project()
    with pytest.raises(ScienceQRRProjectionError, match="execution adapter identity"):
        project_rich_science_request_to_flat_v2(
            normalized.normalized_request,
            normalized.campaign,
            bundle,
            route_binding=route,
            route_adapter=replace(
                adapter,
                lifecycle_total_max_bytes=adapter.lifecycle_total_max_bytes + 1,
            ),
        )

    cells = list(bundle.semantic_cells.to_value())
    malformed_cells = CanonicalPayloadV1.from_value(
        SchemaVersion("eiel_science_semantic_cells_v1"),
        [
            cells[0],
            {
                **cells[1],
                "dimension_bindings": {
                    "task_seed": 11,
                    "unfactorable_condition": "only-one-seed",
                },
            },
        ],
    )
    malformed_bundle = replace(
        bundle,
        semantic_cells=malformed_cells,
        execution_plan=replace(
            bundle.execution_plan,
            semantic_acquisition_core=CanonicalPayloadV1.from_value(
                SEMANTIC_ACQUISITION_CORE_SCHEMA_VERSION,
                {
                    **bundle.execution_plan.semantic_acquisition_core.to_value(),
                    "cells": malformed_cells.to_value(),
                },
            ),
        ),
    )
    with pytest.raises(ScienceQRRProjectionError, match="cannot be factored"):
        project_rich_science_request_to_flat_v2(
            normalized.normalized_request,
            normalized.campaign,
            malformed_bundle,
            route_binding=route,
            route_adapter=adapter,
        )


def test_additive_qrr_binding_does_not_modify_resolution_receipt() -> None:
    _, _, _, _, projection = _project()
    resolution_core = {
        "schema_version": "eiel_active_qrr_resolution_receipt_v1",
        "status": "PASS",
        "release_id": "qrr-release-001",
        "manifest_sha256": "1" * 64,
        "registry_sha256": "2" * 64,
        "science_request_sha256": projection.flat_request["request_sha256"],
        "qualification_evidence_binding_sha256": "3" * 64,
        "manual_internal_decisions_required": 0,
        "reasons": [],
    }
    resolution = {
        **resolution_core,
        "receipt_sha256": canonical_sha256(resolution_core).hex,
    }
    original = dict(resolution)
    binding = bind_science_qrr_resolution_v1(projection, resolution)

    assert resolution == original
    assert (
        binding["active_qrr_resolution_receipt_sha256"] == resolution["receipt_sha256"]
    )
    assert binding["effects_performed"] is False
