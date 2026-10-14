"""Deterministic, technical-only functional demonstration of the science core.

The demonstration deliberately stops before QRR lowering, provider launch, or
Qualification.  It proves that one generic factorial campaign can be compiled,
observed, checkpointed, published, and reduced independently of equivalent
physical route partitioning while admission remains fail-closed without an
accepted qualification envelope.
"""

from __future__ import annotations

from typing import Any

from emergence_contracts import (
    PHYSICAL_EXECUTION_CORE_SCHEMA_VERSION,
    SCIENCE_ROUTE_BINDING_SCHEMA_VERSION,
    CanonicalPayloadV1,
    EEDReleaseStatusV1,
    ProfileResourceEnvelopeV1,
    SchemaVersion,
    ScienceAdmissionOutcomeV1,
    ScienceControlDescriptorV1,
    ScienceInstrumentManifestV1,
    ScienceObservationDraftV1,
    ScienceObservationPrefixCursorV1,
    ScienceObservationProfileDescriptorV1,
    ScienceObservationProfileV1,
    ScienceRecordKindV1,
    ScienceReducerDescriptorV1,
    ScienceRouteBindingV1,
    canonical_sha256,
    finalize_science_observation_drafts_from_cursor_v1,
    finalize_science_observation_drafts_v1,
    science_observation_profile_catalog_sha256_v1,
    seal_science_observation_prefix_cursor_v1,
)
from emergence_evidence import (
    CountingScienceReducerV1,
    build_science_observation_publication_plan_v1,
    execute_science_reducer_v1,
    publish_science_world_evidence_v1,
    science_evidence_manifest_sha256_v1,
)

from .science_instrument import (
    ScienceCRCASemanticBindingV1,
    compile_science_campaign_v1,
    science_control_catalog_sha256,
)
from .science_instrument_admission_v1 import classify_science_campaign_v1
from .science_qualification_envelope_projection_v1 import (
    ScienceQualifiedEnvelopeProjectionV1,
)
from .science_request_v2 import normalize_rich_production_science_request_v2


SCIENCE_FUNCTIONAL_DEMONSTRATION_SCHEMA_VERSION = (
    "eiel_science_functional_demonstration_v1"
)
_PRODUCER_ID = "portable-science-demonstration-producer-v1"
_PRODUCER_VERSION = SchemaVersion("portable_science_demonstration_producer_v1")
_PRODUCER_PHASE = 2
_COUNTING_IMPLEMENTATION_SHA256 = canonical_sha256(
    {
        "component": "emergence_evidence.CountingScienceReducerV1",
        "contract_version": "science_counting_reducer_v1",
        "demonstration_binding": "technical_only_v1",
    }
)


def _payload(schema: str, value: object) -> CanonicalPayloadV1:
    return CanonicalPayloadV1.from_value(SchemaVersion(schema), value)


def _control_descriptors() -> tuple[ScienceControlDescriptorV1, ...]:
    return (
        ScienceControlDescriptorV1(
            control_id="environment.resource_multiplier",
            value_contract=_payload(
                "eiel_science_demo_numeric_control_v1",
                {"kind": "number", "minimum": 0.0, "maximum": 2.0},
            ),
            unit="ratio",
            activation_phases=("environment",),
            heterogeneity_modes=("global",),
            schedule_kinds=("pulse_absolute_ticks",),
            readback_capability_ids=("control_readback",),
            interaction_ids=("resource_selection",),
            component_id="demo.environment",
            qualified_envelope_subjects=("controls.environment.resource_multiplier",),
        ),
        ScienceControlDescriptorV1(
            control_id="selection.pressure",
            value_contract=_payload(
                "eiel_science_demo_numeric_control_v1",
                {"kind": "number", "minimum": 0.0, "maximum": 1.0},
            ),
            unit="ratio",
            activation_phases=("selection",),
            heterogeneity_modes=("global",),
            schedule_kinds=("static",),
            readback_capability_ids=("control_readback",),
            interaction_ids=("resource_selection",),
            component_id="demo.selection",
            qualified_envelope_subjects=("controls.selection.pressure",),
        ),
    )


def _reducer_descriptor() -> ScienceReducerDescriptorV1:
    return ScienceReducerDescriptorV1(
        reducer_id="science.counting",
        contract_version=SchemaVersion("science_counting_reducer_v1"),
        implementation_sha256=_COUNTING_IMPLEMENTATION_SHA256,
        configuration_schema_sha256=canonical_sha256(
            {
                "schema_version": "science_counting_config_v1",
                "fields": ["group_by"],
            }
        ),
        required_observation_capability_ids=("causal_lifecycle",),
        output_contract_id="eiel_science_count_result_v1",
        purity_declarations=(
            "DETERMINISTIC_CANONICAL_OUTPUT",
            "NO_SIMULATOR_IMPORT",
            "NO_TRAJECTORY_MUTATION",
            "OFFLINE_ONLY",
        ),
    )


def _profile_descriptor() -> ScienceObservationProfileDescriptorV1:
    return ScienceObservationProfileDescriptorV1(
        profile=ScienceObservationProfileV1.STANDARD,
        observation_abi_version=SchemaVersion("eiel_science_observation_abi_v1"),
        record_kinds=(ScienceRecordKindV1.CONTROL_BINDING,),
        required_nonempty_record_kinds=(ScienceRecordKindV1.CONTROL_BINDING,),
        capability_ids=("causal_lifecycle", "control_readback"),
        producer_phase_registry_ordinals=((_PRODUCER_ID, _PRODUCER_PHASE),),
        default_partition_span_ticks=10,
    )


def _resource_envelope() -> ProfileResourceEnvelopeV1:
    fixed = 1_024
    chunks = 2
    chunk_bytes = 16_384
    worlds = 12
    cursor_bytes = 256
    resident_capacity = 16
    per_agent_bytes = 0
    return ProfileResourceEnvelopeV1(
        profile=ScienceObservationProfileV1.STANDARD,
        max_records_per_tick_per_world_by_kind=(
            (ScienceRecordKindV1.CONTROL_BINDING, 2),
        ),
        max_record_bytes_by_kind=((ScienceRecordKindV1.CONTROL_BINDING, 4_096),),
        chunk_max_records=24,
        chunk_max_bytes=chunk_bytes,
        max_in_flight_chunks=chunks,
        fixed_buffer_bytes=fixed,
        per_world_cursor_bytes=cursor_bytes,
        per_resident_agent_observation_state_bytes=per_agent_bytes,
        logical_world_count=worlds,
        resident_capacity=resident_capacity,
        max_resident_observation_bytes=(
            fixed
            + chunks * chunk_bytes
            + worlds * cursor_bytes
            + worlds * resident_capacity * per_agent_bytes
        ),
    )


def _instrument_manifest(
    controls: tuple[ScienceControlDescriptorV1, ...],
    reducer: ScienceReducerDescriptorV1,
    profile: ScienceObservationProfileDescriptorV1,
) -> ScienceInstrumentManifestV1:
    component_graph = {
        "controls": [item.descriptor_sha256().to_dict() for item in controls],
        "observation_profile": profile.descriptor_sha256().to_dict(),
        "reducer": reducer.descriptor_sha256().to_dict(),
        "purpose": "technical_only_portable_demonstration",
    }
    return ScienceInstrumentManifestV1(
        component_graph_sha256=canonical_sha256(component_graph),
        control_catalog_sha256=science_control_catalog_sha256(controls),
        world_profile_catalog_sha256=canonical_sha256(
            ["portable-demo-world-v1", "portable-demo-resource-overlay-v1"]
        ),
        founder_profile_catalog_sha256=canonical_sha256(
            ["portable-demo-founder-a-v1", "portable-demo-founder-b-v1"]
        ),
        policy_profile_catalog_sha256=canonical_sha256(["portable-demo-policy-v1"]),
        mutation_operator_catalog_sha256=canonical_sha256(
            ["portable-demo-mutation-v1"]
        ),
        observation_abi_sha256=canonical_sha256(
            {"schema_version": "eiel_science_observation_abi_v1"}
        ),
        observation_profile_catalog_sha256=(
            science_observation_profile_catalog_sha256_v1((profile,))
        ),
        reducer_protocol_version=SchemaVersion("eiel_science_reducer_protocol_v1"),
        route_binding_protocol_version=SCIENCE_ROUTE_BINDING_SCHEMA_VERSION,
        eed_release_status=EEDReleaseStatusV1.EXCLUDED,
    )


def _request(
    manifest: ScienceInstrumentManifestV1,
    reducer: ScienceReducerDescriptorV1,
) -> dict[str, Any]:
    return {
        "schema_version": "eiel_production_science_request_v2",
        "request_id": "portable-science-functional-demo-v1",
        "baseline": {
            "schema_version": "eiel_baseline_declaration_v1",
            "profile_id": "EIEL_PRODUCTION_BASELINE_V1",
            "intent": "development",
            "deviations": [],
        },
        "science_campaign": {
            "schema_version": "eiel_science_campaign_v1",
            "product_id": "EIEL_SCIENCE_INSTRUMENT_V1",
            "campaign_id": "portable-science-functional-demo-v1",
            "instrument": {
                "product_id": "EIEL_SCIENCE_INSTRUMENT_V1",
                "instrument_manifest_sha256": manifest.manifest_sha256().to_dict(),
            },
            "question": {
                "question_id": "Q-PORTABLE-DEMO",
                "hypothesis_ids": ["H-ROUTE-NEUTRAL-PIPELINE"],
                "labels": ["factorial", "route-neutral", "technical-only"],
                "description": (
                    "Demonstrate generic campaign mechanics without a science claim."
                ),
                "preregistration_ref": None,
                "science_ownership": {
                    "test_id": "HOSTCORE",
                    "test_version": "1",
                    "study_id": "HOSTCORE-ROUTE-NEUTRAL-DEMO",
                    "arm_dimension_ids": ["selection_pressure"],
                    "replicate_dimension_id": "task_seed",
                },
                "no_claims": True,
            },
            "design": {
                "dimensions": [
                    {
                        "dimension_id": "mutation_seed",
                        "binding_path": "seeds.mutation",
                        "canonical_values": [501, 502],
                        "ordering": "declared",
                    },
                    {
                        "dimension_id": "selection_pressure",
                        "binding_path": "controls.selection.pressure",
                        "canonical_values": [0.1, 0.5],
                        "ordering": "canonical_json_bytes",
                    },
                    {
                        "dimension_id": "task_seed",
                        "binding_path": "seeds.task_environment",
                        "canonical_values": [101, 102, 103],
                        "ordering": "declared",
                    },
                ],
                "pairing_groups": ["task_seed"],
                "repeat_groups": ["mutation_seed"],
                "expected_cell_count": 12,
            },
            "world": {
                "world_profile_id": "portable-demo-world-v1",
                "overlay_profile_ids": ["portable-demo-resource-overlay-v1"],
                "geometry": {"width": 8, "height": 8},
                "resident_capacity": 16,
            },
            "initialization": {
                "initial_population": 8,
                "initial_live_count": 8,
                "founder_profiles": [
                    "portable-demo-founder-a-v1",
                    "portable-demo-founder-b-v1",
                ],
                "exact_founder_composition": [
                    {
                        "founder_profile_id": "portable-demo-founder-a-v1",
                        "count": 4,
                    },
                    {
                        "founder_profile_id": "portable-demo-founder-b-v1",
                        "count": 4,
                    },
                ],
                "spatial_placement_profile": "portable-demo-placement-v1",
                "standing_variation_profile": "portable-demo-variation-v1",
            },
            "controls": [
                {
                    "control_id": "selection.pressure",
                    "value": 0.0,
                    "dimension_binding": "selection_pressure",
                    "schedule": {"kind": "static"},
                },
                {
                    "control_id": "environment.resource_multiplier",
                    "value": 0.75,
                    "schedule": {
                        "kind": "pulse_absolute_ticks",
                        "start_absolute_tick": 10,
                        "duration_ticks": 2,
                    },
                },
            ],
            "seeds": [
                {"owner": "ecology", "binding_kind": "literal", "literal": 7001},
                {
                    "owner": "mutation",
                    "binding_kind": "dimension_bound",
                    "dimension_id": "mutation_seed",
                    "held_equal_in_pairing_groups": ["task_seed"],
                },
                {
                    "owner": "task_environment",
                    "binding_kind": "dimension_bound",
                    "dimension_id": "task_seed",
                },
            ],
            "time": {
                "authority": "absolute_tick_v1",
                "start_next_tick": 0,
                "total_ticks": 20,
                "observation_ticks": [2, 12],
                "checkpoint_ticks": [10, 20],
                "branches": [],
            },
            "observations": {
                "profile_id": "SCIENCE_STANDARD_V1",
                "required_capability_ids": [
                    "causal_lifecycle",
                    "control_readback",
                ],
                "event_observations": ["birth", "death"],
                "state_observations": ["genome_identity", "lineage_ancestry"],
                "runtime_counters": ["birth_count", "population_count"],
                "milestone_ticks": [2, 12, 20],
            },
            "execution": {
                "machine_profile_id": "EIEL_PRODUCTION_BASELINE_V1",
                "logical_world_count": 12,
                "batching_policy": "authority_selected",
                "maximum_physical_batch_worlds": 12,
            },
            "reducers": [
                {
                    "reducer_id": reducer.reducer_id,
                    "contract_version": str(reducer.contract_version),
                    "descriptor_sha256": reducer.descriptor_sha256().to_dict(),
                    "configuration": {"group_by": "kind"},
                    "required_observation_capability_ids": list(
                        reducer.required_observation_capability_ids
                    ),
                    "output_contract_id": reducer.output_contract_id,
                }
            ],
            "artifact_lifecycle": {
                "profile_id": "SCIENCE_COMPACT",
                "contract_ref": "eiel_run_artifact_lifecycle_contract_v1",
                "explicit_intent_allowed_override": None,
            },
            "decision": {
                "offline_decision_rule_ids": ["technical-demo-no-claim-v1"],
                "expected_outputs": [reducer.output_contract_id],
                "presentation_metadata": {"technical_only": True},
            },
        },
    }


def _route_and_empty_envelope(
    manifest: ScienceInstrumentManifestV1,
    resource: ProfileResourceEnvelopeV1,
    *,
    route_id: str,
) -> tuple[ScienceRouteBindingV1, ScienceQualifiedEnvelopeProjectionV1]:
    no_authority = canonical_sha256(
        {
            "status": "TECHNICAL_DEMO_NO_QUALIFICATION_AUTHORITY",
            "route_id": route_id,
        }
    )
    envelope = ScienceQualifiedEnvelopeProjectionV1(
        instrument_manifest_sha256=manifest.manifest_sha256(),
        route_id=route_id,
        machine_profile_id="EIEL_PRODUCTION_BASELINE_V1",
        qualification_authority_sha256=no_authority,
        rows=(),
    )
    route = ScienceRouteBindingV1(
        instrument_manifest_sha256=manifest.manifest_sha256(),
        route_id=route_id,
        route_profile_sha256=canonical_sha256(
            {"route_id": route_id, "profile": "equivalent-host-demo-v1"}
        ),
        machine_profile_id="EIEL_PRODUCTION_BASELINE_V1",
        capability_projection_sha256=canonical_sha256(
            {"capabilities": ["causal_lifecycle", "control_readback"]}
        ),
        native_build_closure_sha256=canonical_sha256(
            {"route_id": route_id, "native": "NOT_BUILT_DEMONSTRATION_ONLY"}
        ),
        execution_adapter_sha256=canonical_sha256(
            {"route_id": route_id, "adapter": "execution-demo-v1"}
        ),
        control_adapter_sha256=canonical_sha256(
            {"route_id": route_id, "adapter": "control-demo-v1"}
        ),
        observation_adapter_sha256=canonical_sha256(
            {"route_id": route_id, "adapter": "observation-demo-v1"}
        ),
        checkpoint_adapter_sha256=canonical_sha256(
            {"route_id": route_id, "adapter": "checkpoint-demo-v1"}
        ),
        semantic_conformance_sha256=canonical_sha256(
            {
                "semantic_profile": "equivalent-host-demo-v1",
                "status": "TECHNICAL_DEMONSTRATION_ONLY",
            }
        ),
        resource_envelope_sha256=canonical_sha256(resource.to_dict()),
        qualified_envelope_sha256=envelope.projection_sha256(),
        qualification_authority_sha256=no_authority,
    )
    return route, envelope


def _scientific_capabilities() -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                "control:environment.resource_multiplier",
                "control:selection.pressure",
                "founder_profile:portable-demo-founder-a-v1",
                "founder_profile:portable-demo-founder-b-v1",
                "placement_profile:portable-demo-placement-v1",
                "schedule:pulse_absolute_ticks",
                "schedule:static",
                "variation_profile:portable-demo-variation-v1",
                "world_overlay:portable-demo-resource-overlay-v1",
                "world_profile:portable-demo-world-v1",
            }
        )
    )


def _drafts_for_cell(cell: dict[str, Any]) -> tuple[ScienceObservationDraftV1, ...]:
    control_values = {
        row["control_id"]: row["resolved_value"] for row in cell["control_bindings"]
    }
    schedules = {row["control_id"]: row["schedule"] for row in cell["control_bindings"]}
    binding_sha256 = canonical_sha256(
        {
            "semantic_cell_id": cell["semantic_cell_id"],
            "seed_bindings": cell["seed_bindings"],
            "control_values": control_values,
            "control_schedules": schedules,
        }
    )
    result: list[ScienceObservationDraftV1] = []
    for tick, stage in ((2, "initial"), (12, "post-pulse")):
        result.append(
            ScienceObservationDraftV1(
                producer_component_id=_PRODUCER_ID,
                producer_contract_version=_PRODUCER_VERSION,
                producer_phase_registry_ordinal=_PRODUCER_PHASE,
                absolute_tick=tick,
                semantic_source_address=(
                    f"tick:{tick}/phase:{_PRODUCER_PHASE}/control-binding:{stage}"
                ),
                record_kind=ScienceRecordKindV1.CONTROL_BINDING,
                record_kind_version=1,
                control_binding_sha256=binding_sha256,
                payload=_payload(
                    "science_record_payload_v1",
                    {
                        "binding_id": f"{cell['semantic_cell_id']}:{stage}",
                        "semantic_cell_id": cell["semantic_cell_id"],
                        "seed_bindings": cell["seed_bindings"],
                        "control_values": control_values,
                        "control_schedules": schedules,
                        "registered_profile_ids": [
                            "portable-demo-world-v1",
                            "SCIENCE_STANDARD_V1",
                        ],
                        "semantic_envelope_resolution_sha256": canonical_sha256(
                            {"status": "NO_ACCEPTED_ENVELOPE", "stage": stage}
                        ).hex,
                        "semantic_applied_readback_sha256": canonical_sha256(
                            {"binding": binding_sha256.to_dict(), "stage": stage}
                        ).hex,
                        "continuation_parent_execution_identity": None,
                    },
                ),
            )
        )
    return tuple(result)


def _world_key(cell: dict[str, Any]) -> CanonicalPayloadV1:
    return _payload(
        "science_semantic_world_key_v1",
        {"semantic_cell_id": cell["semantic_cell_id"], "repeat_ordinal": 0},
    )


def build_science_functional_demonstration_v1() -> dict[str, object]:
    """Return one deterministic receipt with no runtime or authority effects."""

    controls = _control_descriptors()
    reducer = _reducer_descriptor()
    profile = _profile_descriptor()
    resource = _resource_envelope()
    manifest = _instrument_manifest(controls, reducer, profile)
    normalized = normalize_rich_production_science_request_v2(
        _request(manifest, reducer)
    )
    route_a, envelope_a = _route_and_empty_envelope(
        manifest, resource, route_id="portable-demo-route-a"
    )
    route_b, _ = _route_and_empty_envelope(
        manifest, resource, route_id="portable-demo-route-b"
    )
    crca = ScienceCRCASemanticBindingV1(
        semantic_run_digest=canonical_sha256(
            {"semantic_run": "portable-demo-v1", "absolute_tick_span": [0, 20]}
        ),
        projection_sha256=canonical_sha256(
            {"projection": "portable-demo-crca-shadow-v1"}
        ),
        absolute_start_next_tick=0,
        absolute_target_next_tick=20,
        trajectory_checkpoint_ticks=(0, 10, 20),
    )
    compiler_sha256 = canonical_sha256(
        {"component": "eiel_composition.compile_science_campaign_v1", "version": 1}
    )

    def compile_for(route: ScienceRouteBindingV1, partitions: list[list[int]]):
        return compile_science_campaign_v1(
            normalized.campaign,
            instrument_manifest=manifest,
            route_binding=route,
            control_descriptors=controls,
            reducer_descriptors=(reducer,),
            crca_semantic_binding=crca,
            physical_execution_core=_payload(
                str(PHYSICAL_EXECUTION_CORE_SCHEMA_VERSION),
                {"route_id": route.route_id, "logical_world_partitions": partitions},
            ),
            compiler_component_sha256=compiler_sha256,
        )

    bundle_a = compile_for(route_a, [list(range(12))])
    bundle_b = compile_for(route_b, [list(range(0, 6)), list(range(6, 12))])
    admission = classify_science_campaign_v1(
        normalized.campaign,
        bundle_a,
        instrument_manifest=manifest,
        route_binding=route_a,
        qualified_envelope=envelope_a,
        available_scientific_capability_ids=_scientific_capabilities(),
        available_observation_capability_ids=(
            "causal_lifecycle",
            "control_readback",
        ),
    )

    semantic_acquisition_sha256 = (
        bundle_a.execution_plan.science_semantic_acquisition_sha256()
    )
    cells = bundle_a.semantic_cells.to_value()
    publications = []
    all_partitions = []
    cursor_hashes: list[dict[str, str]] = []
    world_uids: list[str] = []
    chunk_invariant = True
    cursor_resume_invariant = True
    for cell in cells:
        drafts = _drafts_for_cell(cell)
        world_key = _world_key(cell)
        uninterrupted = finalize_science_observation_drafts_v1(
            drafts,
            instrument_manifest_sha256=manifest.manifest_sha256(),
            crca_semantic_run_digest=crca.semantic_run_digest,
            science_semantic_acquisition_sha256=semantic_acquisition_sha256,
            cell_id=cell["semantic_cell_id"],
            logical_world_id=cell["logical_world_id"],
            semantic_world_key=world_key,
        )
        prefix = finalize_science_observation_drafts_v1(
            (drafts[0],),
            instrument_manifest_sha256=manifest.manifest_sha256(),
            crca_semantic_run_digest=crca.semantic_run_digest,
            science_semantic_acquisition_sha256=semantic_acquisition_sha256,
            cell_id=cell["semantic_cell_id"],
            logical_world_id=cell["logical_world_id"],
            semantic_world_key=world_key,
        )
        cursor = seal_science_observation_prefix_cursor_v1(
            prefix,
            instrument_manifest_sha256=manifest.manifest_sha256(),
            crca_semantic_run_digest=crca.semantic_run_digest,
            science_semantic_acquisition_sha256=semantic_acquisition_sha256,
            cell_id=cell["semantic_cell_id"],
            logical_world_id=cell["logical_world_id"],
            semantic_world_key=world_key,
            checkpoint_at_tick=10,
            producer_phase_registry_ordinals={_PRODUCER_ID: _PRODUCER_PHASE},
        )
        resumed = finalize_science_observation_drafts_from_cursor_v1(
            (drafts[1],),
            cursor=cursor,
            instrument_manifest_sha256=manifest.manifest_sha256(),
            crca_semantic_run_digest=crca.semantic_run_digest,
            science_semantic_acquisition_sha256=semantic_acquisition_sha256,
            cell_id=cell["semantic_cell_id"],
            logical_world_id=cell["logical_world_id"],
            semantic_world_key=world_key,
        )
        terminal_cursor = seal_science_observation_prefix_cursor_v1(
            resumed,
            instrument_manifest_sha256=manifest.manifest_sha256(),
            crca_semantic_run_digest=crca.semantic_run_digest,
            science_semantic_acquisition_sha256=semantic_acquisition_sha256,
            cell_id=cell["semantic_cell_id"],
            logical_world_id=cell["logical_world_id"],
            semantic_world_key=world_key,
            checkpoint_at_tick=20,
            producer_phase_registry_ordinals={_PRODUCER_ID: _PRODUCER_PHASE},
            prior_cursor=cursor,
        )
        cursor_resume_invariant = cursor_resume_invariant and (
            prefix + resumed == uninterrupted
            and ScienceObservationPrefixCursorV1.from_dict(terminal_cursor.to_dict())
            == terminal_cursor
        )
        plan = build_science_observation_publication_plan_v1(
            instrument_manifest_sha256=manifest.manifest_sha256(),
            science_semantic_acquisition_sha256=semantic_acquisition_sha256,
            profile_descriptor=profile,
            resource_envelope=resource,
            absolute_tick_start=0,
            absolute_tick_end=20,
        )
        contiguous = publish_science_world_evidence_v1((uninterrupted,), plan)
        permuted = publish_science_world_evidence_v1(
            ((uninterrupted[1],), (uninterrupted[0],)), plan
        )
        chunk_invariant = chunk_invariant and contiguous.to_dict() == permuted.to_dict()
        publications.append(contiguous)
        all_partitions.extend(contiguous.partitions)
        world_uids.append(contiguous.world_uid.hex)
        cursor_hashes.append(terminal_cursor.cursor_sha256().to_dict())

    partitions = tuple(all_partitions)
    config = _payload("science_counting_config_v1", {"group_by": "kind"})

    def reduce_for(bundle, route):
        return execute_science_reducer_v1(
            reducer,
            CountingScienceReducerV1(
                implementation_sha256=_COUNTING_IMPLEMENTATION_SHA256
            ),
            config,
            partitions,
            evidence_capability_ids=("causal_lifecycle", "control_readback"),
            campaign_spec_sha256=bundle.campaign_spec_sha256,
            instrument_manifest_sha256=manifest.manifest_sha256(),
            route_binding_sha256=route.route_binding_sha256(),
            science_semantic_acquisition_sha256=(
                bundle.execution_plan.science_semantic_acquisition_sha256()
            ),
            physical_execution_plan_sha256=(
                bundle.execution_plan.physical_execution_plan_sha256()
            ),
            science_execution_plan_sha256=(
                bundle.execution_plan.science_execution_plan_sha256()
            ),
            reduction_plan_sha256=bundle.reduction_plan.reduction_plan_sha256(),
        )

    result_a = reduce_for(bundle_a, route_a)
    result_b = reduce_for(bundle_b, route_b)
    interrupted = execute_science_reducer_v1(
        reducer,
        CountingScienceReducerV1(implementation_sha256=_COUNTING_IMPLEMENTATION_SHA256),
        config,
        partitions,
        evidence_capability_ids=("causal_lifecycle", "control_readback"),
        campaign_spec_sha256=bundle_a.campaign_spec_sha256,
        instrument_manifest_sha256=manifest.manifest_sha256(),
        route_binding_sha256=route_a.route_binding_sha256(),
        science_semantic_acquisition_sha256=semantic_acquisition_sha256,
        physical_execution_plan_sha256=(
            bundle_a.execution_plan.physical_execution_plan_sha256()
        ),
        science_execution_plan_sha256=(
            bundle_a.execution_plan.science_execution_plan_sha256()
        ),
        reduction_plan_sha256=bundle_a.reduction_plan.reduction_plan_sha256(),
        stop_after_partitions=len(partitions) // 2,
    )
    assert interrupted.checkpoint is not None
    resumed_result = execute_science_reducer_v1(
        reducer,
        CountingScienceReducerV1(implementation_sha256=_COUNTING_IMPLEMENTATION_SHA256),
        config,
        partitions,
        evidence_capability_ids=("causal_lifecycle", "control_readback"),
        campaign_spec_sha256=bundle_a.campaign_spec_sha256,
        instrument_manifest_sha256=manifest.manifest_sha256(),
        route_binding_sha256=route_a.route_binding_sha256(),
        science_semantic_acquisition_sha256=semantic_acquisition_sha256,
        physical_execution_plan_sha256=(
            bundle_a.execution_plan.physical_execution_plan_sha256()
        ),
        science_execution_plan_sha256=(
            bundle_a.execution_plan.science_execution_plan_sha256()
        ),
        reduction_plan_sha256=bundle_a.reduction_plan.reduction_plan_sha256(),
        resume=interrupted.checkpoint,
    )
    assert result_a.result_payload is not None
    assert result_a.result_receipt is not None
    assert result_b.result_payload is not None
    assert result_b.result_receipt is not None
    assert resumed_result.result_payload is not None
    assert resumed_result.result_receipt is not None

    invariants = {
        "semantic_cell_count_is_12": len(cells) == 12,
        "semantic_acquisition_is_route_neutral": (
            semantic_acquisition_sha256
            == bundle_b.execution_plan.science_semantic_acquisition_sha256()
        ),
        "physical_execution_identity_changes_by_route": (
            bundle_a.execution_plan.physical_execution_plan_sha256()
            != bundle_b.execution_plan.physical_execution_plan_sha256()
        ),
        "world_uids_are_unique": len(world_uids) == len(set(world_uids)) == 12,
        "observation_chunking_is_neutral": chunk_invariant,
        "checkpoint_cursor_resume_matches_uninterrupted": cursor_resume_invariant,
        "reducer_resume_matches_uninterrupted": (
            resumed_result.result_payload == result_a.result_payload
            and resumed_result.result_receipt.result_sha256
            == result_a.result_receipt.result_sha256
        ),
        "equivalent_route_result_payload_is_neutral": (
            result_a.result_payload == result_b.result_payload
            and result_a.result_receipt.result_sha256
            == result_b.result_receipt.result_sha256
        ),
        "route_bound_result_receipts_remain_distinct": (
            result_a.result_receipt.receipt_sha256()
            != result_b.result_receipt.receipt_sha256()
        ),
        "admission_fails_closed_without_qualification": (
            admission.primary_outcome
            is ScienceAdmissionOutcomeV1.OUTSIDE_QUALIFIED_ENVELOPE
        ),
    }
    core: dict[str, object] = {
        "schema_version": SCIENCE_FUNCTIONAL_DEMONSTRATION_SCHEMA_VERSION,
        "product_id": "EIEL_SCIENCE_INSTRUMENT_V1",
        "status": (
            "TECHNICAL_DEMONSTRATION_PASS"
            if all(invariants.values())
            else "TECHNICAL_DEMONSTRATION_FAIL"
        ),
        "ok": all(invariants.values()),
        "campaign_id": normalized.campaign.campaign_id,
        "request_sha256": normalized.normalized_request["request_sha256"],
        "instrument_manifest_sha256": manifest.manifest_sha256().to_dict(),
        "primary_outcome": admission.primary_outcome.value,
        "admission_decision_sha256": admission.decision_sha256().to_dict(),
        "semantic_cell_count": len(cells),
        "world_count": len(world_uids),
        "record_count": sum(len(item.records) for item in publications),
        "partition_count": len(partitions),
        "observation_cursor_count": len(cursor_hashes),
        "world_uid_set_sha256": canonical_sha256(sorted(world_uids)).to_dict(),
        "observation_cursor_set_sha256": canonical_sha256(cursor_hashes).to_dict(),
        "evidence_manifest_sha256": science_evidence_manifest_sha256_v1(
            partitions
        ).to_dict(),
        "result_sha256": result_a.result_receipt.result_sha256.to_dict(),
        "result_payload": result_a.result_payload.to_dict(),
        "route_demonstration": {
            "relationship": "SEMANTICALLY_EQUIVALENT_HOST_DEMO_REBINDINGS",
            "route_a": {
                "route_id": route_a.route_id,
                "route_binding_sha256": route_a.route_binding_sha256().to_dict(),
                "physical_execution_plan_sha256": (
                    bundle_a.execution_plan.physical_execution_plan_sha256().to_dict()
                ),
                "result_receipt_sha256": (
                    result_a.result_receipt.receipt_sha256().to_dict()
                ),
            },
            "route_b": {
                "route_id": route_b.route_id,
                "route_binding_sha256": route_b.route_binding_sha256().to_dict(),
                "physical_execution_plan_sha256": (
                    bundle_b.execution_plan.physical_execution_plan_sha256().to_dict()
                ),
                "result_receipt_sha256": (
                    result_b.result_receipt.receipt_sha256().to_dict()
                ),
            },
            "science_semantic_acquisition_sha256": (
                semantic_acquisition_sha256.to_dict()
            ),
        },
        "reducer_resume": {
            "checkpoint_sha256": interrupted.checkpoint.checkpoint_sha256().to_dict(),
            "resumed_result_receipt_sha256": (
                resumed_result.result_receipt.receipt_sha256().to_dict()
            ),
        },
        "invariants": invariants,
        "runtime_effects_performed": False,
        "trajectory_effects_performed": False,
        "provider_launch_performed": False,
        "native_build_performed": False,
        "gpu_used": False,
        "qualification_started": False,
        "qualification_authority_effect": False,
        "scientific_claim": False,
        "limitations": [
            "Uses synthetic equivalent host route bindings, not B136 or CUDA V2 authority.",
            "Stops before QRR lowering, Launchpad, provider execution, and resource measurement.",
            "Does not establish observation neutrality against a native trajectory.",
            "Does not start QualPrep or Qualification and grants no envelope row.",
        ],
    }
    return {**core, "receipt_sha256": canonical_sha256(core).to_dict()}


__all__ = (
    "SCIENCE_FUNCTIONAL_DEMONSTRATION_SCHEMA_VERSION",
    "build_science_functional_demonstration_v1",
)
