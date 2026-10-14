from __future__ import annotations

import json
from pathlib import Path

from engine.campaigns.qualification_framework import (
    AUTHORITY_CURRENT,
    AUTHORITY_PLANNED_NON_AUTHORITATIVE,
    DELTA_ACCEPTED,
    DELTA_PLANNED,
    DELTA_REQUIRED,
    DELTA_REQUIRED_BELOW_ANCHOR,
    FULL_QUALIFICATION_REQUIRED,
    HISTORICAL_EVIDENCE_UNAVAILABLE,
    AxisRangeV1,
    EvidenceBindingV1,
    ExclusionRegionV1,
    HistoricalAuthorityRecordV1,
    OperatingEnvelopeV1,
    QualificationCatalogV1,
    QUALIFIED,
    REVALIDATION_REQUIRED,
    RECIPE_BATCH_CAPACITY_DELTA,
    RECIPE_COMBINED_INTERACTION_DELTA,
    ResourceCapacityEnvelopeV1,
    SUPERSEDED,
    candidate_matrix_status,
    continuation_compatibility_class,
    recipe_obligations,
    load_catalog,
    plan_delta_campaign,
    resolve_qualification,
    resolve_resource_capacity,
    validate_delta_decision_acceptance,
    validate_delta_plan,
)
from tools.scripts import eiel_qualification


CATALOG = Path("_spec/qualification/qf1_synthetic_catalog_v1.json")
BASE_REQUEST = {
    "provider_id": "eiel-production-provider-v1",
    "runtime_id": "eiel-production-runtime-v1",
    "world_family": "patch",
    "execution_route_sha256": "route-synthetic-patch-v1",
}


def _catalog():
    return load_catalog(CATALOG)


def test_exact_anchor_lookup_is_scoped_and_source_bound() -> None:
    request = {
        **BASE_REQUEST,
        "grid_width": 16,
        "grid_height": 16,
        "ticks_per_generation": 80,
    }

    result = resolve_qualification(request, _catalog())

    assert result["decision"] == QUALIFIED
    assert result["scope"] == "CONFIGURATION_OPERATING_ENVELOPE"
    assert result["record_id"] == "synthetic-w16-t80"


def test_source_drift_requires_revalidation_not_silent_reuse() -> None:
    request = {
        **BASE_REQUEST,
        "grid_width": 16,
        "grid_height": 16,
        "ticks_per_generation": 80,
        "source_commit": "new-source",
    }

    result = resolve_qualification(request, _catalog())

    assert result["decision"] == REVALIDATION_REQUIRED
    assert result["reasons"][0]["code"] == "source_commit_mismatch"


def test_missing_corner_prevents_naive_rectangle_qualification() -> None:
    request = {
        **BASE_REQUEST,
        "grid_width": 24,
        "grid_height": 24,
        "ticks_per_generation": 120,
    }

    result = resolve_qualification(request, _catalog())

    assert result["decision"] == DELTA_REQUIRED
    assert result["required_campaign_class"] == "missing_corner_or_interaction_delta"
    assert result["reasons"][0]["code"] == "missing_required_corners"
    assert {
        "grid_height": 16,
        "grid_width": 16,
        "ticks_per_generation": 160,
    } in result["reasons"][0]["missing_corners"]


def test_below_w16_returns_available_below_anchor_delta_not_prequalified() -> None:
    request = {
        **BASE_REQUEST,
        "grid_width": 8,
        "grid_height": 8,
        "ticks_per_generation": 80,
    }

    result = resolve_qualification(request, _catalog())

    assert result["decision"] == DELTA_REQUIRED_BELOW_ANCHOR
    assert result["required_campaign_class"] == "below_anchor_delta"


def test_categorical_boundary_requires_full_qualification() -> None:
    request = {
        **BASE_REQUEST,
        "provider_id": "other-provider",
        "grid_width": 24,
        "grid_height": 24,
        "ticks_per_generation": 120,
    }

    result = resolve_qualification(request, _catalog())

    assert result["decision"] == FULL_QUALIFICATION_REQUIRED
    assert result["reasons"][0]["code"] == "categorical_boundary_crossed"


def test_missing_route_receipt_fails_closed() -> None:
    request = {
        "provider_id": "eiel-production-provider-v1",
        "runtime_id": "eiel-production-runtime-v1",
        "world_family": "patch",
        "grid_width": 24,
        "grid_height": 24,
        "ticks_per_generation": 120,
    }

    result = resolve_qualification(request, _catalog())

    assert result["decision"] == FULL_QUALIFICATION_REQUIRED
    assert result["reasons"][0]["code"] == "missing_route_receipt"


def test_mismatched_route_receipt_fails_closed() -> None:
    request = {
        **BASE_REQUEST,
        "execution_route_sha256": "other-route",
        "grid_width": 24,
        "grid_height": 24,
        "ticks_per_generation": 120,
    }

    result = resolve_qualification(request, _catalog())

    assert result["decision"] == FULL_QUALIFICATION_REQUIRED
    assert result["reasons"][0]["code"] == "execution_route_mismatch"


def test_superseded_envelope_fails_closed() -> None:
    catalog = _catalog()
    envelope = catalog.envelopes[0]
    superseded = type(catalog)(
        anchors=catalog.anchors,
        envelopes=(
            OperatingEnvelopeV1.from_mapping(
                {**envelope.payload(), "superseded_by": "new-record"}
            ),
        ),
    )
    request = {
        **BASE_REQUEST,
        "grid_width": 24,
        "grid_height": 24,
        "ticks_per_generation": 120,
    }

    result = resolve_qualification(request, superseded)

    assert result["decision"] == SUPERSEDED


def test_delta_planner_handles_width_height_and_below_anchor_points() -> None:
    catalog = _catalog()
    envelope = catalog.envelopes[0]
    request = {
        **BASE_REQUEST,
        "grid_width": 8,
        "grid_height": 8,
        "ticks_per_generation": 80,
    }

    plan = plan_delta_campaign(request, envelope)

    assert plan["plan_class"] == "below_anchor_delta"
    assert "grid_width" in plan["changed_axes"]
    assert "grid_height" in plan["changed_axes"]
    assert any(
        row["purpose"] == "explicit downscale extrapolation; smaller is not assumed safer"
        for row in plan["candidate_points"]
    )


def test_cli_validate_plan_rejects_tampered_plan(tmp_path: Path) -> None:
    catalog = _catalog()
    plan = plan_delta_campaign(
        {
            **BASE_REQUEST,
            "grid_width": 24,
            "grid_height": 24,
            "ticks_per_generation": 120,
        },
        catalog.envelopes[0],
    )
    plan["changed_axes"].append("tampered")
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True), encoding="utf-8")

    result = eiel_qualification.main(["validate-plan", "--plan", str(plan_path)])

    assert result == 1


def _resource_capacity(batch_max: int = 48) -> ResourceCapacityEnvelopeV1:
    return ResourceCapacityEnvelopeV1(
        resource_id="synthetic-b48-resource",
        backend="cuda_resident",
        device="cuda",
        provider_id="eiel-production-provider-v1",
        runtime_id="eiel-production-runtime-v1",
        source_commit="source-a",
        execution_route_sha256="route-gpu",
        batch_size=AxisRangeV1("batch_size", 1, batch_max),
        footprint_dimensions={"grid_width": 32, "grid_height": 32, "resident_capacity": 256},
        resource_authority={
            "peak_allocated_bytes": 100,
            "peak_reserved_bytes": 120,
            "headroom_bytes": 1000,
        },
        safety_policy={"minimum_headroom_bytes": 1},
        fallback_policy={"silent_fallback_allowed": False},
        evidence=(
            EvidenceBindingV1(
                evidence_id="resource",
                kind="synthetic",
                uri="synthetic://resource",
                sha256="0" * 64,
            ),
        ),
    )


def test_delta_status_separates_resolver_lifecycle_and_authority() -> None:
    result = resolve_qualification(
        {**BASE_REQUEST, "grid_width": 24, "grid_height": 24, "ticks_per_generation": 120},
        _catalog(),
    )
    assert result["decision"] == DELTA_REQUIRED
    assert result["lifecycle_stage"] == "DELTA_CLASSIFIED"
    assert result["authority_state"] == AUTHORITY_PLANNED_NON_AUTHORITATIVE

    plan = plan_delta_campaign(result["config"], _catalog().envelopes[0])
    assert plan["lifecycle_stage"] == DELTA_PLANNED
    assert plan["authority_state"] == AUTHORITY_PLANNED_NON_AUTHORITATIVE
    assert validate_delta_plan(plan)["ok"] is True


def test_validated_plan_is_not_accepted_authority() -> None:
    catalog = _catalog()
    request = {**BASE_REQUEST, "grid_width": 24, "grid_height": 24, "ticks_per_generation": 120}
    plan = plan_delta_campaign(request, catalog.envelopes[0])

    assert validate_delta_plan(plan)["ok"] is True
    assert resolve_qualification(request, catalog)["decision"] == DELTA_REQUIRED
    assert plan["authority_state"] != AUTHORITY_CURRENT


def test_resource_capacity_layer_does_not_prove_configuration_authority() -> None:
    resource = _resource_capacity(batch_max=72)
    request = {
        "backend": "cuda_resident",
        "device": "cuda",
        "provider_id": "eiel-production-provider-v1",
        "runtime_id": "eiel-production-runtime-v1",
        "execution_route_sha256": "route-gpu",
        "batch_size": 72,
        "grid_width": 16,
        "grid_height": 16,
        "ticks_per_generation": 160,
    }

    resource_result = resolve_resource_capacity(request, (resource,))
    assert resource_result["decision"] == QUALIFIED
    assert resource_result["scope"] == "RESOURCE_CAPACITY_ENVELOPE"

    config_result = resolve_qualification(request, QualificationCatalogV1())
    assert config_result["decision"] == FULL_QUALIFICATION_REQUIRED


def test_b72_capacity_alone_does_not_qualify_w16_or_t160() -> None:
    resource = _resource_capacity(batch_max=72)
    envelope = OperatingEnvelopeV1(
        envelope_id="gpu-config-needs-resource",
        scope="CONFIGURATION_OPERATING_ENVELOPE",
        axes=(
            AxisRangeV1("grid_width", 32, 32),
            AxisRangeV1("grid_height", 32, 32),
            AxisRangeV1("ticks_per_generation", 80, 80),
            AxisRangeV1("batch_size", 48, 72),
        ),
        categorical_invariants={
            "backend": "cuda_resident",
            "device": "cuda",
            "provider_id": "eiel-production-provider-v1",
            "runtime_id": "eiel-production-runtime-v1",
            "execution_route_sha256": "route-gpu",
        },
        anchor_ids=(),
        required_corners=(
            {"grid_width": 32, "grid_height": 32, "ticks_per_generation": 80, "batch_size": 48},
        ),
        accepted_points=(
            {"grid_width": 32, "grid_height": 32, "ticks_per_generation": 80, "batch_size": 48},
        ),
        evidence=(),
        required_resource_capacity_id=resource.resource_id,
    )
    catalog = QualificationCatalogV1(envelopes=(envelope,), resource_capacity_envelopes=(resource,))

    result = resolve_qualification(
        {
            "backend": "cuda_resident",
            "device": "cuda",
            "provider_id": "eiel-production-provider-v1",
            "runtime_id": "eiel-production-runtime-v1",
            "execution_route_sha256": "route-gpu",
            "batch_size": 72,
            "grid_width": 16,
            "grid_height": 16,
            "ticks_per_generation": 160,
        },
        catalog,
    )
    assert result["decision"] != QUALIFIED
    assert result["scope"] == "CONFIGURATION_OPERATING_ENVELOPE"


def test_broad_b48_b72_t80_t160_rectangle_requires_interaction_corner() -> None:
    envelope = OperatingEnvelopeV1(
        envelope_id="broad-gpu-rectangle",
        scope="CONFIGURATION_OPERATING_ENVELOPE",
        axes=(
            AxisRangeV1("batch_size", 48, 72),
            AxisRangeV1("ticks_per_generation", 80, 160),
            AxisRangeV1("grid_width", 32, 32),
            AxisRangeV1("grid_height", 32, 32),
        ),
        categorical_invariants={"backend": "cuda_resident", "device": "cuda"},
        anchor_ids=(),
        required_corners=(
            {"batch_size": 48, "ticks_per_generation": 80, "grid_width": 32, "grid_height": 32},
            {"batch_size": 48, "ticks_per_generation": 160, "grid_width": 32, "grid_height": 32},
            {"batch_size": 72, "ticks_per_generation": 80, "grid_width": 32, "grid_height": 32},
            {"batch_size": 72, "ticks_per_generation": 160, "grid_width": 32, "grid_height": 32},
        ),
        accepted_points=(
            {"batch_size": 48, "ticks_per_generation": 80, "grid_width": 32, "grid_height": 32},
            {"batch_size": 48, "ticks_per_generation": 160, "grid_width": 32, "grid_height": 32},
            {"batch_size": 72, "ticks_per_generation": 80, "grid_width": 32, "grid_height": 32},
        ),
        evidence=(),
    )
    result = resolve_qualification(
        {
            "backend": "cuda_resident",
            "device": "cuda",
            "batch_size": 72,
            "ticks_per_generation": 160,
            "grid_width": 32,
            "grid_height": 32,
        },
        QualificationCatalogV1(envelopes=(envelope,)),
    )
    assert result["decision"] == DELTA_REQUIRED
    assert result["reasons"][0]["code"] == "missing_required_corners"


def test_narrow_b72w16t160_point_can_exclude_unresolved_w32t160_corner() -> None:
    envelope = OperatingEnvelopeV1(
        envelope_id="narrow-b72-w16-t160",
        scope="CONFIGURATION_OPERATING_ENVELOPE",
        axes=(
            AxisRangeV1("batch_size", 72, 72),
            AxisRangeV1("ticks_per_generation", 160, 160),
            AxisRangeV1("grid_width", 16, 16),
            AxisRangeV1("grid_height", 16, 16),
        ),
        categorical_invariants={"backend": "cuda_resident", "device": "cuda"},
        anchor_ids=(),
        required_corners=(
            {"batch_size": 72, "ticks_per_generation": 160, "grid_width": 16, "grid_height": 16},
        ),
        accepted_points=(
            {"batch_size": 72, "ticks_per_generation": 160, "grid_width": 16, "grid_height": 16},
        ),
        evidence=(),
        accepted_region="exact_point",
        exclusions=(
            ExclusionRegionV1(
                exclusion_id="exclude-b72w32t160",
                reason="unresolved broad rectangle interaction corner",
                axes=(
                    AxisRangeV1("batch_size", 72, 72),
                    AxisRangeV1("ticks_per_generation", 160, 160),
                    AxisRangeV1("grid_width", 32, 32),
                    AxisRangeV1("grid_height", 32, 32),
                ),
            ),
        ),
    )
    result = resolve_qualification(
        {
            "backend": "cuda_resident",
            "device": "cuda",
            "batch_size": 72,
            "ticks_per_generation": 160,
            "grid_width": 16,
            "grid_height": 16,
        },
        QualificationCatalogV1(envelopes=(envelope,)),
    )
    assert result["decision"] == QUALIFIED
    assert result["lifecycle_stage"] == DELTA_ACCEPTED


def test_combined_interaction_recipe_expands_all_obligations() -> None:
    plan = plan_delta_campaign(
        {
            "backend": "cuda_resident",
            "device": "cuda",
            "batch_size": 72,
            "grid_width": 16,
            "grid_height": 16,
            "initial_population": 64,
            "ticks_per_generation": 160,
        },
        OperatingEnvelopeV1(
            envelope_id="combined",
            scope="CONFIGURATION_OPERATING_ENVELOPE",
            axes=(
                AxisRangeV1("batch_size", 48, 72),
                AxisRangeV1("grid_width", 16, 32),
                AxisRangeV1("grid_height", 16, 32),
                AxisRangeV1("initial_population", 64, 256),
                AxisRangeV1("ticks_per_generation", 80, 160),
            ),
            categorical_invariants={"backend": "cuda_resident", "device": "cuda"},
            anchor_ids=(),
            required_corners=(),
            accepted_points=(
                {
                    "batch_size": 48,
                    "grid_width": 32,
                    "grid_height": 32,
                    "initial_population": 256,
                    "ticks_per_generation": 80,
                },
            ),
            evidence=(),
        ),
    )
    assert RECIPE_COMBINED_INTERACTION_DELTA in plan["selected_recipes"]
    assert RECIPE_BATCH_CAPACITY_DELTA in plan["selected_recipes"]
    for obligation in recipe_obligations(plan["selected_recipes"]):
        assert obligation in plan["required_obligations"]


def test_missing_recipe_obligations_fail_plan_and_publication() -> None:
    catalog = _catalog()
    plan = plan_delta_campaign(
        {**BASE_REQUEST, "grid_width": 24, "grid_height": 24, "ticks_per_generation": 120},
        catalog.envelopes[0],
    )
    broken = dict(plan)
    broken["required_obligations"] = []
    unsigned = dict(broken)
    unsigned.pop("plan_sha256", None)
    from engine.campaigns.canonical import canonical_sha256

    broken["plan_sha256"] = canonical_sha256(unsigned)
    assert validate_delta_plan(broken)["ok"] is False
    decision = {
        "decision": "A",
        "accepted_for_envelope_expansion": True,
        "no_post_hoc_widening": True,
        "plan_sha256": plan["plan_sha256"],
        "required_obligations_sha256": plan["required_obligations_sha256"],
        "satisfied_recipe_obligations": [],
    }
    assert validate_delta_decision_acceptance(plan, decision)["ok"] is False


def test_cpu_continuation_classes_are_shape_sensitive() -> None:
    base = {
        "source_commit": "source",
        "provider_id": "eiel-production-provider-v1",
        "runtime_id": "eiel-production-runtime-v1-cpu-native",
        "backend": "cpu_native",
        "device": "cpu",
        "execution_route_sha256": "route-cpu",
        "checkpoint_schema": "checkpoint-v1",
        "serialization_schema": "serialization-v1",
        "resident_capacity": 256,
        "ticks_per_generation": 160,
    }
    w16 = continuation_compatibility_class({**base, "grid_width": 16, "grid_height": 16, "initial_population": 64})
    w32 = continuation_compatibility_class({**base, "grid_width": 32, "grid_height": 32, "initial_population": 256})
    assert w16["compatibility_class_sha256"] != w32["compatibility_class_sha256"]


def test_historical_stale_authority_is_preserved_but_not_resolver_eligible() -> None:
    record = HistoricalAuthorityRecordV1(
        historical_id="old-w32-t160-cpu",
        state=HISTORICAL_EVIDENCE_UNAVAILABLE,
        original_decision="W32-C256-CPU-B1-T160-A",
        source_commit="old-source",
        execution_route_sha256="old-route",
        provenance={"date": "2026-07-31"},
        missing_evidence=("artifact.so", "execution_route_qualification.json"),
    )
    catalog = QualificationCatalogV1(historical_authority=(record,))
    assert catalog.payload()["historical_authority"][0]["resolver_eligible"] is False
    assert resolve_qualification(
        {
            "source_commit": "old-source",
            "execution_route_sha256": "old-route",
            "grid_width": 32,
            "grid_height": 32,
            "ticks_per_generation": 160,
        },
        catalog,
    )["decision"] == FULL_QUALIFICATION_REQUIRED


def test_candidate_runtime_matrix_is_non_authoritative() -> None:
    status = candidate_matrix_status({"tracks": {"w32t160cpu": {"decision": "DELTA_REQUIRED"}}})
    assert status["status"] == "NON_AUTHORITATIVE_CANDIDATE"
    assert status["resolver_eligible"] is False
    assert status["current_authority"] is False
