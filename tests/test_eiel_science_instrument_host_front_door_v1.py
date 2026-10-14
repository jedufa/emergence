from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import jsonschema
import pytest
from referencing import Registry, Resource

from eiel_composition import (
    ScienceCRCASemanticBindingV1,
    science_control_catalog_sha256,
)
from engine.campaigns.baseline_v1 import BaselineDeclarationV1
from eiel_composition.science_instrument_host_front_door_v1 import (
    ScienceInstrumentHostContextV1,
    ScienceInstrumentHostPreparationError,
    prepare_rich_science_request_v2,
)
from eiel_composition.science_instrument_artifacts_v1 import (
    ScienceInstrumentArtifactError,
    parse_science_host_context_v1,
    serialize_science_host_context_v1,
)
from eiel_composition.science_qrr_projection_v1 import ScienceQRRRouteAdapterV1
from eiel_composition.science_qualification_envelope_projection_v1 import (
    ScienceQualifiedDomainRowV1,
    ScienceQualifiedEnvelopeProjectionV1,
)
from emergence_contracts import (
    PHYSICAL_EXECUTION_CORE_SCHEMA_VERSION,
    ProfileResourceEnvelopeV1,
    CanonicalPayloadV1,
    ContentDigest,
    EEDReleaseStatusV1,
    PRODUCT_ID,
    SCIENCE_ROUTE_BINDING_SCHEMA_VERSION,
    SchemaVersion,
    ScienceAdmissionOutcomeV1,
    ScienceInstrumentManifestV1,
    ScienceObservationProfileV1,
    ScienceObservationProfileDescriptorV1,
    ScienceRecordKindV1,
    ScienceRouteBindingV1,
    canonical_sha256,
    science_observation_profile_catalog_sha256_v1,
)
from tools.eiel import main as eiel_main


def _digest(character: str) -> ContentDigest:
    return ContentDigest("sha256", character * 64)


def _manifest() -> ScienceInstrumentManifestV1:
    profiles = _observation_profiles()
    return ScienceInstrumentManifestV1(
        component_graph_sha256=_digest("0"),
        control_catalog_sha256=science_control_catalog_sha256(()),
        world_profile_catalog_sha256=_digest("1"),
        founder_profile_catalog_sha256=_digest("2"),
        policy_profile_catalog_sha256=_digest("3"),
        mutation_operator_catalog_sha256=_digest("4"),
        observation_abi_sha256=_digest("5"),
        observation_profile_catalog_sha256=(
            science_observation_profile_catalog_sha256_v1(profiles)
        ),
        reducer_protocol_version=SchemaVersion("eiel_science_reducer_protocol_v1"),
        route_binding_protocol_version=SCIENCE_ROUTE_BINDING_SCHEMA_VERSION,
        eed_release_status=EEDReleaseStatusV1.EXCLUDED,
    )


def _observation_profiles() -> tuple[ScienceObservationProfileDescriptorV1, ...]:
    return (
        ScienceObservationProfileDescriptorV1(
            profile=ScienceObservationProfileV1.STANDARD,
            observation_abi_version=SchemaVersion("eiel_science_observation_abi_v1"),
            record_kinds=tuple(ScienceRecordKindV1),
            required_nonempty_record_kinds=(
                ScienceRecordKindV1.CONTROL_BINDING,
                ScienceRecordKindV1.WORLD_COMPLETION,
            ),
            capability_ids=("causal_lifecycle",),
            producer_phase_registry_ordinals=(
                ("science-control-producer-v1", 0),
                ("science-world-producer-v1", 1),
            ),
            default_partition_span_ticks=1000,
        ),
    )


def _campaign(manifest: ScienceInstrumentManifestV1) -> dict[str, object]:
    return {
        "schema_version": "eiel_science_campaign_v1",
        "product_id": PRODUCT_ID,
        "campaign_id": "host-front-door-v1",
        "instrument": {
            "product_id": PRODUCT_ID,
            "instrument_manifest_sha256": manifest.manifest_sha256().to_dict(),
        },
        "question": {"question_id": "Q-HOST", "no_claims": True},
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
            "overlay_profile_ids": [],
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


def _request(campaign: dict[str, object]) -> dict[str, object]:
    return {
        "schema_version": "eiel_production_science_request_v2",
        "request_id": "host-front-door-request-v1",
        "baseline": BaselineDeclarationV1(
            profile_id="EIEL_PRODUCTION_BASELINE_V1",
            intent="development",
            deviations=(),
        ).payload(),
        "science_campaign": campaign,
    }


def _adapter(route_id: str) -> ScienceQRRRouteAdapterV1:
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


def _envelope(
    manifest: ScienceInstrumentManifestV1,
    *,
    route_id: str,
    authority: ContentDigest,
) -> ScienceQualifiedEnvelopeProjectionV1:
    values = {
        "artifact_lifecycle.profile_id": "SCIENCE_COMPACT",
        "execution.batching_policy": "authority_selected",
        "execution.logical_world_count": 2,
        "execution.machine_profile_id": "EIEL_PRODUCTION_BASELINE_V1",
        "initialization.initial_live_count": 256,
        "initialization.initial_population": 256,
        "interaction.control_ids": [],
        "observations.profile_id": "SCIENCE_STANDARD_V1",
        "time.checkpoint_ticks": [2500, 5000],
        "time.observation_ticks": [2500, 5000],
        "time.start_next_tick": 0,
        "time.total_ticks": 5000,
        "world.geometry.height": 32,
        "world.geometry.width": 32,
        "world.resident_capacity": 512,
        "world.world_profile_id": "world-v1",
    }
    rows = tuple(
        ScienceQualifiedDomainRowV1.exact(
            subject_path=path,
            values=(value,),
            envelope_component_id=f"host-fixture:{path}",
            evidence_refs=("TECHNICAL_ONLY:host-fixture",),
        )
        for path, value in sorted(values.items())
    )
    return ScienceQualifiedEnvelopeProjectionV1(
        instrument_manifest_sha256=manifest.manifest_sha256(),
        route_id=route_id,
        machine_profile_id="EIEL_PRODUCTION_BASELINE_V1",
        qualification_authority_sha256=authority,
        rows=rows,
    )


def _context(route_id: str = "host-route-a") -> ScienceInstrumentHostContextV1:
    manifest = _manifest()
    adapter = _adapter(route_id)
    authority = _digest("a")
    envelope = _envelope(manifest, route_id=route_id, authority=authority)
    resource_envelope = ProfileResourceEnvelopeV1(
        profile=ScienceObservationProfileV1.STANDARD,
        max_records_per_tick_per_world_by_kind=tuple(
            (kind, 2)
            for kind in sorted(ScienceRecordKindV1, key=lambda item: item.value)
        ),
        max_record_bytes_by_kind=tuple(
            (kind, 4_096)
            for kind in sorted(ScienceRecordKindV1, key=lambda item: item.value)
        ),
        chunk_max_records=16,
        chunk_max_bytes=20_000,
        max_in_flight_chunks=2,
        fixed_buffer_bytes=100,
        per_world_cursor_bytes=16,
        per_resident_agent_observation_state_bytes=1,
        logical_world_count=136,
        resident_capacity=512,
        max_resident_observation_bytes=111_908,
    )
    route = ScienceRouteBindingV1(
        instrument_manifest_sha256=manifest.manifest_sha256(),
        route_id=route_id,
        route_profile_sha256=_digest("7"),
        machine_profile_id="EIEL_PRODUCTION_BASELINE_V1",
        capability_projection_sha256=_digest("8"),
        native_build_closure_sha256=_digest("9"),
        execution_adapter_sha256=adapter.execution_adapter_sha256(),
        control_adapter_sha256=adapter.control_adapter_sha256(),
        observation_adapter_sha256=_digest("b"),
        checkpoint_adapter_sha256=_digest("c"),
        semantic_conformance_sha256=_digest("d"),
        resource_envelope_sha256=canonical_sha256(resource_envelope.to_dict()),
        qualified_envelope_sha256=envelope.projection_sha256(),
        qualification_authority_sha256=authority,
    )
    return ScienceInstrumentHostContextV1(
        instrument_manifest=manifest,
        route_binding=route,
        route_adapter=adapter,
        control_descriptors=(),
        reducer_descriptors=(),
        observation_profile_descriptors=_observation_profiles(),
        crca_semantic_binding=ScienceCRCASemanticBindingV1(
            semantic_run_digest=_digest("f"),
            projection_sha256=_digest("0"),
            absolute_start_next_tick=0,
            absolute_target_next_tick=5000,
            trajectory_checkpoint_ticks=(0, 2500, 5000),
        ),
        physical_execution_core=CanonicalPayloadV1.from_value(
            PHYSICAL_EXECUTION_CORE_SCHEMA_VERSION,
            {"partition": [[0, 1]], "route_id": route_id},
        ),
        resource_envelope=resource_envelope,
        compiler_component_sha256=_digest("1"),
        qualified_envelope=envelope,
        available_scientific_capability_ids=tuple(
            sorted(
                {
                    "founder_profile:founder-v1",
                    "placement_profile:placement-v1",
                    "variation_profile:variation-v1",
                    "world_profile:world-v1",
                }
            )
        ),
        available_observation_capability_ids=("causal_lifecycle",),
        machine_profile_sha256=_digest("2"),
        admission_kernel_receipt_sha256=_digest("3"),
    )


class _QRRResolver:
    def __init__(self) -> None:
        self.calls = 0

    def resolve(self, flat_request: dict[str, object]) -> dict[str, object]:
        self.calls += 1
        core = {
            "schema_version": "eiel_active_qrr_resolution_receipt_v1",
            "status": "PASS",
            "release_id": "host-qrr-release-v1",
            "manifest_sha256": "4" * 64,
            "registry_sha256": "5" * 64,
            "science_request_sha256": flat_request["request_sha256"],
            "qualification_evidence_binding_sha256": "6" * 64,
            "manual_internal_decisions_required": 0,
            "reasons": [],
        }
        return {**core, "receipt_sha256": canonical_sha256(core).hex}


def test_host_front_door_prepares_full_admission_without_runtime_effects() -> None:
    context = _context()
    resolver = _QRRResolver()
    prepared = prepare_rich_science_request_v2(
        _request(_campaign(context.instrument_manifest)),
        context,
        active_qrr_resolver=resolver,
    )

    assert resolver.calls == 1
    assert prepared.classification.primary_outcome is ScienceAdmissionOutcomeV1.ADMITTED
    assert prepared.admission_receipt is not None
    assert (
        prepared.admission_receipt.primary_outcome is ScienceAdmissionOutcomeV1.ADMITTED
    )
    assert prepared.rich_to_flat_projection is not None
    assert prepared.rich_to_flat_projection.flat_request["batch_cells"] == 2
    assert prepared.effects_performed is False
    assert prepared.to_dict()["effects_performed"] is False


def test_unsupported_fork_stops_before_qrr_resolution_or_lowering() -> None:
    context = _context()
    campaign = deepcopy(_campaign(context.instrument_manifest))
    campaign["time"]["branches"] = [
        {
            "branch_id": "future-fork",
            "parent_checkpoint_tick": 2500,
            "intervention_control_ids": [],
        }
    ]
    resolver = _QRRResolver()
    prepared = prepare_rich_science_request_v2(
        _request(campaign), context, active_qrr_resolver=resolver
    )

    assert resolver.calls == 0
    assert prepared.classification.primary_outcome is (
        ScienceAdmissionOutcomeV1.MISSING_SCIENTIFIC_CAPABILITY
    )
    assert prepared.rich_to_flat_projection is None
    assert prepared.admission_receipt is None
    assert any(
        finding.extension_target == "B136_SCIENCE_INSTRUMENT_V2_EXTENSION"
        for finding in prepared.classification.findings
    )


def test_parallel_engine_context_changes_only_route_bound_host_receipts() -> None:
    first_context = _context("host-route-a")
    second_context = _context("parallel-gpu-engine-v1")
    first = prepare_rich_science_request_v2(
        _request(_campaign(first_context.instrument_manifest)),
        first_context,
        active_qrr_resolver=_QRRResolver(),
    )
    second = prepare_rich_science_request_v2(
        _request(_campaign(second_context.instrument_manifest)),
        second_context,
        active_qrr_resolver=_QRRResolver(),
    )

    assert first.normalized.normalized_request == second.normalized.normalized_request
    assert (
        first.compilation_bundle.execution_plan.science_semantic_acquisition_sha256()
        == second.compilation_bundle.execution_plan.science_semantic_acquisition_sha256()
    )
    assert first.rich_to_flat_projection.flat_request == (
        second.rich_to_flat_projection.flat_request
    )
    assert first.admission_receipt.route_binding_sha256 != (
        second.admission_receipt.route_binding_sha256
    )


def test_host_context_artifact_round_trips_and_rejects_tampering() -> None:
    context = _context()
    artifact = serialize_science_host_context_v1(context)

    schema_root = Path(__file__).parents[1] / "_spec/science_instrument/v1"
    schema_names = (
        "instrument_manifest.schema.json",
        "science_route_binding.schema.json",
        "science_qrr_route_adapter.schema.json",
        "science_profile_resource_envelope.schema.json",
        "science_observation_profile_descriptor.schema.json",
        "control_descriptor.schema.json",
        "reducer_descriptor.schema.json",
        "science_host_context_artifact.schema.json",
    )
    schemas = {
        name: json.loads((schema_root / name).read_text(encoding="utf-8"))
        for name in schema_names
    }
    registry = Registry()
    for schema in schemas.values():
        registry = registry.with_resource(schema["$id"], Resource.from_contents(schema))
    jsonschema.Draft202012Validator.check_schema(
        schemas["science_host_context_artifact.schema.json"]
    )
    jsonschema.Draft202012Validator(
        schemas["science_host_context_artifact.schema.json"], registry=registry
    ).validate(artifact)

    assert parse_science_host_context_v1(artifact) == context

    tampered = deepcopy(artifact)
    tampered["route_adapter"]["execution_adapter_core"]["route_id"] = "other-route"
    with pytest.raises(ScienceInstrumentArtifactError, match="SHA-256 mismatch"):
        parse_science_host_context_v1(tampered)

    mismatched_resource = replace(
        context.resource_envelope,
        fixed_buffer_bytes=context.resource_envelope.fixed_buffer_bytes + 1,
        max_resident_observation_bytes=(
            context.resource_envelope.max_resident_observation_bytes + 1
        ),
    )
    with pytest.raises(
        ScienceInstrumentHostPreparationError, match="resource envelope identity"
    ):
        replace(context, resource_envelope=mismatched_resource)


def test_science_cli_normalizes_and_prepares_without_effects(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    context = _context()
    request = _request(_campaign(context.instrument_manifest))
    request_path = tmp_path / "request.json"
    context_path = tmp_path / "context.json"
    qrr_receipt_path = tmp_path / "active-qrr-receipt.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    context_path.write_text(
        json.dumps(serialize_science_host_context_v1(context)), encoding="utf-8"
    )

    assert eiel_main(["science", "normalize", str(request_path)]) == 0
    normalized_output = json.loads(capsys.readouterr().out)
    assert normalized_output["status"] == "NORMALIZED"
    assert normalized_output["effects_performed"] is False

    prepared = prepare_rich_science_request_v2(
        request, context, active_qrr_resolver=_QRRResolver()
    )
    qrr_receipt_path.write_text(
        json.dumps(prepared.active_qrr_resolution_receipt), encoding="utf-8"
    )
    assert (
        eiel_main(
            [
                "science",
                "prepare",
                str(request_path),
                "--context",
                str(context_path),
                "--active-qrr-receipt",
                str(qrr_receipt_path),
            ]
        )
        == 0
    )
    preparation_output = json.loads(capsys.readouterr().out)
    assert preparation_output["status"] == "ADMITTED_HOST_READY"
    assert preparation_output["primary_outcome"] == "ADMITTED"
    assert preparation_output["effects_performed"] is False

    assert (
        eiel_main(
            [
                "validate",
                str(request_path),
                "--science-context",
                str(context_path),
                "--active-qrr-receipt",
                str(qrr_receipt_path),
            ]
        )
        == 0
    )
    validation_output = json.loads(capsys.readouterr().out)
    assert validation_output["status"] == "ADMITTED_HOST_READY"
    assert validation_output["result_class"] == "SCIENCE_HOST_PREPARATION"
    assert validation_output["validation_only"] is True
    assert validation_output["effects_performed"] is False

    assert (
        eiel_main(
            [
                "execute",
                str(request_path),
                "--science-context",
                str(context_path),
                "--active-qrr-receipt",
                str(qrr_receipt_path),
            ]
        )
        == 2
    )
    execution_output = json.loads(capsys.readouterr().out)
    assert execution_output["ok"] is False
    assert "SCIENCE_INSTRUMENT_EXECUTION_NOT_ENABLED" in execution_output["error"]
