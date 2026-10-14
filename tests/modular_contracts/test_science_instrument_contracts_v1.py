from __future__ import annotations

import json
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import jsonschema
import pytest

from emergence_contracts import (
    EEDReleaseStatusV1,
    PHYSICAL_EXECUTION_CORE_SCHEMA_VERSION,
    PRODUCT_ID,
    ProfileResourceEnvelopeV1,
    SCIENCE_ADMISSION_OUTCOME_PRECEDENCE,
    SCIENCE_OBSERVATION_RECORD_SCHEMA_VERSION,
    SCIENCE_ROUTE_BINDING_SCHEMA_VERSION,
    SEMANTIC_ACQUISITION_CORE_SCHEMA_VERSION,
    CanonicalPayloadV1,
    ContentDigest,
    ContractError,
    SchemaVersion,
    ScienceAdmissionOutcomeV1,
    ScienceAdmissionReceiptV1,
    ScienceCampaignV1,
    ScienceControlDescriptorV1,
    ScienceEvidencePartitionV1,
    ScienceExecutionPlanV1,
    ScienceFindingV1,
    ScienceInstrumentManifestV1,
    ScienceObservationProfileV1,
    ScienceObservationRecordV1,
    ScienceRecordKindV1,
    ScienceReducerDescriptorV1,
    ScienceReductionPlanV1,
    ScienceRouteBindingV1,
    V2_RESERVED_RECORD_KINDS,
    canonical_json_bytes,
    canonical_sha256,
)

SPEC_ROOT = Path(__file__).parents[2] / "_spec" / "science_instrument" / "v1"


def _digest(character: str = "a") -> ContentDigest:
    return ContentDigest("sha256", character * 64)


def _payload(schema: str, value: object) -> CanonicalPayloadV1:
    return CanonicalPayloadV1.from_value(SchemaVersion(schema), value)


def _campaign(**overrides: object) -> ScienceCampaignV1:
    values: dict[str, object] = {
        "campaign_id": "generic-campaign-001",
        "instrument": _payload(
            "eiel_science_instrument_binding_v1",
            {
                "product_id": PRODUCT_ID,
                "instrument_manifest_sha256": _digest("1").to_dict(),
            },
        ),
        "question": _payload(
            "eiel_science_question_v1",
            {"question_id": "Q-001", "labels": ["unseen-use-case"]},
        ),
        "design": _payload(
            "eiel_science_design_v1",
            {"dimensions": [], "expected_cell_count": 1},
        ),
        "world": _payload(
            "eiel_science_world_v1",
            {"world_profile_id": "world-v1", "resident_capacity": 8},
        ),
        "initialization": _payload(
            "eiel_science_initialization_v1", {"initial_live_count": 1}
        ),
        "controls": _payload("eiel_science_controls_v1", {"bindings": []}),
        "seeds": _payload("eiel_science_seeds_v1", {"ecology": 1, "mutation": 2}),
        "time": _payload(
            "eiel_science_time_v1",
            {
                "authority": "absolute_tick_v1",
                "start_next_tick": 0,
                "total_ticks": 10,
                "observation_ticks": [5, 10],
                "checkpoint_ticks": [5, 10],
                "branches": [
                    {
                        "branch_id": "future-fork-1",
                        "parent_checkpoint_tick": 5,
                        "intervention_control_ids": ["pressure.switch"],
                    }
                ],
            },
        ),
        "observations": _payload(
            "eiel_science_observations_v1",
            {
                "profile_id": "SCIENCE_STANDARD_V1",
                "required_capability_ids": ["causal_lifecycle"],
            },
        ),
        "execution": _payload(
            "eiel_science_execution_request_v1",
            {
                "machine_profile_id": "EIEL_PRODUCTION_BASELINE_V1",
                "logical_world_count": 1,
                "batching_policy": "authority_selected",
            },
        ),
        "reducers": _payload("eiel_science_reducers_v1", {"bindings": []}),
        "artifact_lifecycle": _payload(
            "eiel_run_artifact_lifecycle_contract_v1",
            {
                "profile_id": "SCIENCE_COMPACT",
                "contract_ref": "eiel_run_artifact_lifecycle_contract_v1",
                "explicit_intent_allowed_override": None,
            },
        ),
        "decision": _payload("eiel_science_decision_v1", {}),
    }
    values.update(overrides)
    return ScienceCampaignV1(**values)


def _reducer(character: str = "2") -> ScienceReducerDescriptorV1:
    return ScienceReducerDescriptorV1(
        reducer_id="generic.reducer",
        contract_version=SchemaVersion("generic_reducer_v1"),
        implementation_sha256=_digest(character),
        configuration_schema_sha256=_digest("3"),
        required_observation_capability_ids=("causal_lifecycle",),
        output_contract_id="generic_result_v1",
        purity_declarations=(
            "DETERMINISTIC_CANONICAL_OUTPUT",
            "NO_SIMULATOR_IMPORT",
            "NO_TRAJECTORY_MUTATION",
            "OFFLINE_ONLY",
        ),
    )


def _route_binding(
    *,
    instrument_manifest_sha256: ContentDigest | None = None,
    route_id: str = "tw-b136-w32p256c512-tick5000-acs2-ca344",
) -> ScienceRouteBindingV1:
    return ScienceRouteBindingV1(
        instrument_manifest_sha256=instrument_manifest_sha256 or _digest("0"),
        route_id=route_id,
        route_profile_sha256=_digest("1"),
        machine_profile_id="EIEL_PRODUCTION_BASELINE_V1",
        capability_projection_sha256=_digest("2"),
        native_build_closure_sha256=_digest("3"),
        execution_adapter_sha256=_digest("4"),
        control_adapter_sha256=_digest("5"),
        observation_adapter_sha256=_digest("6"),
        checkpoint_adapter_sha256=_digest("7"),
        semantic_conformance_sha256=_digest("8"),
        resource_envelope_sha256=_digest("9"),
        qualified_envelope_sha256=_digest("a"),
        qualification_authority_sha256=_digest("b"),
    )


def _record(
    *,
    kind: ScienceRecordKindV1 = ScienceRecordKindV1.CONTROL_BINDING,
    payload: object | None = None,
    legacy: object | None = None,
) -> ScienceObservationRecordV1:
    return ScienceObservationRecordV1(
        instrument_manifest_sha256=_digest("1"),
        producer_component_id="science-producer-v1",
        producer_contract_version=SchemaVersion("science_producer_v1"),
        semantic_execution_id=_digest("2"),
        crca_semantic_run_digest=_digest("3"),
        science_semantic_acquisition_sha256=_digest("4"),
        cell_id="cell-001",
        logical_world_id=0,
        world_uid=_digest("5"),
        absolute_tick=7,
        semantic_source_address="tick:7/phase:4/entity:1/op:0",
        pre_sequence_key_sha256=_digest("6"),
        world_local_sequence=0,
        deterministic_event_id=_digest("7"),
        record_kind=kind,
        record_kind_version=1,
        control_binding_sha256=_digest("8"),
        payload=_payload(
            "science_record_payload_v1",
            payload
            if payload is not None
            else {
                "binding_id": "b1",
                "semantic_cell_id": "cell-001",
                "seed_bindings": {},
                "control_values": {},
                "control_schedules": {},
                "registered_profile_ids": [],
                "semantic_envelope_resolution_sha256": "9" * 64,
                "semantic_applied_readback_sha256": "a" * 64,
                "continuation_parent_execution_identity": None,
            },
        ),
        legacy_compatibility=(
            None
            if legacy is None
            else _payload("science_legacy_display_metadata_v1", legacy)
        ),
    )


def _raw_campaign() -> dict[str, object]:
    digest = _digest("1").to_dict()
    return {
        "schema_version": "eiel_science_campaign_v1",
        "product_id": PRODUCT_ID,
        "campaign_id": "novel-capability-composition-001",
        "instrument": {
            "product_id": PRODUCT_ID,
            "instrument_manifest_sha256": digest,
        },
        "question": {"question_id": "Q-NEW", "no_claims": True},
        "design": {
            "dimensions": [
                {
                    "dimension_id": "pressure",
                    "binding_path": "controls.pressure",
                    "canonical_values": [0.1, 0.5],
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
        "controls": [
            {
                "control_id": "selection.pressure",
                "value": 0.1,
                "schedule": {"kind": "static"},
            }
        ],
        "seeds": [{"owner": "ecology", "binding_kind": "literal", "literal": 11}],
        "time": {
            "authority": "absolute_tick_v1",
            "start_next_tick": 0,
            "total_ticks": 5000,
            "observation_ticks": [2500, 5000],
            "checkpoint_ticks": [2500, 5000],
            "branches": [
                {
                    "branch_id": "typed-but-release-gated",
                    "parent_checkpoint_tick": 2500,
                    "intervention_control_ids": ["selection.pressure"],
                }
            ],
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


def _load_schema(name: str) -> dict[str, object]:
    return json.loads((SPEC_ROOT / name).read_text(encoding="utf-8"))


def test_exact_admission_outcomes_and_precedence_are_closed() -> None:
    assert {item.value for item in ScienceAdmissionOutcomeV1} == {
        "ADMITTED",
        "OUTSIDE_QUALIFIED_ENVELOPE",
        "MISSING_SCIENTIFIC_CAPABILITY",
        "MISSING_OBSERVATION_CAPABILITY",
        "INVALID_CAMPAIGN",
        "INSTRUMENT_INTEGRITY_FAILURE",
    }
    assert SCIENCE_ADMISSION_OUTCOME_PRECEDENCE == (
        ScienceAdmissionOutcomeV1.INSTRUMENT_INTEGRITY_FAILURE,
        ScienceAdmissionOutcomeV1.INVALID_CAMPAIGN,
        ScienceAdmissionOutcomeV1.MISSING_SCIENTIFIC_CAPABILITY,
        ScienceAdmissionOutcomeV1.MISSING_OBSERVATION_CAPABILITY,
        ScienceAdmissionOutcomeV1.OUTSIDE_QUALIFIED_ENVELOPE,
        ScienceAdmissionOutcomeV1.ADMITTED,
    )
    assert all("WAIT_" not in item.value for item in ScienceAdmissionOutcomeV1)


def test_campaign_is_immutable_strict_canonical_and_absolute_tick_only() -> None:
    campaign = _campaign()
    assert ScienceCampaignV1.from_dict(campaign.to_dict()) == campaign
    assert campaign.canonical_bytes() == canonical_json_bytes(campaign.to_dict())
    assert campaign.campaign_spec_sha256() == canonical_sha256(campaign.to_dict())
    with pytest.raises(FrozenInstanceError):
        campaign.campaign_id = "changed"  # type: ignore[misc]

    unknown = campaign.to_dict()
    unknown["experiment_name"] = "MUTSUP"
    with pytest.raises(ContractError):
        ScienceCampaignV1.from_dict(unknown)

    bad_time = _payload(
        "eiel_science_time_v1",
        {
            "authority": "absolute_tick_v1",
            "start_next_tick": 0,
            "total_ticks": 10,
            "generation": 2,
        },
    )
    with pytest.raises(ContractError, match="generation-shaped"):
        _campaign(time=bad_time)


def test_wire_campaign_normalizes_defaults_and_round_trips_through_schema() -> None:
    schema = _load_schema("science_campaign.schema.json")
    raw = _raw_campaign()
    jsonschema.validate(raw, schema)
    normalized = ScienceCampaignV1.from_campaign_document(raw)
    wire = normalized.to_campaign_document()
    jsonschema.validate(wire, schema)
    assert wire["artifact_lifecycle"] == {
        "profile_id": "SCIENCE_COMPACT",
        "contract_ref": "eiel_run_artifact_lifecycle_contract_v1",
        "explicit_intent_allowed_override": None,
    }
    assert wire["decision"] == {}
    assert ScienceCampaignV1.from_dict(normalized.to_dict()) == normalized


def test_branch_is_structurally_valid_but_fork_identity_is_not_a_v1_record() -> None:
    assert _campaign().time.to_value()["branches"]
    with pytest.raises(ContractError, match="frozen record-kind"):
        _record(
            kind=ScienceRecordKindV1.CHECKPOINT_MARKER,
            payload={"fork_identity": "future-fork"},
        )


def test_execution_identity_scopes_are_independent_and_legacy_is_sidecar() -> None:
    semantic = _payload(
        str(SEMANTIC_ACQUISITION_CORE_SCHEMA_VERSION),
        {"cells": ["c1"], "total_ticks": 10, "profile": "SCIENCE_STANDARD_V1"},
    )
    first = ScienceExecutionPlanV1(
        instrument_manifest_sha256=_digest("1"),
        route_binding_sha256=_digest("0"),
        crca_semantic_run_digest=_digest("2"),
        semantic_acquisition_core=semantic,
        physical_execution_core=_payload(
            str(PHYSICAL_EXECUTION_CORE_SCHEMA_VERSION), {"batches": [["c1"]]}
        ),
        legacy_campaign_plan_sha256=_digest("3"),
    )
    rebatch = replace(
        first,
        physical_execution_core=_payload(
            str(PHYSICAL_EXECUTION_CORE_SCHEMA_VERSION),
            {"batches": [["c1"], []]},
        ),
    )
    different_legacy = replace(first, legacy_campaign_plan_sha256=_digest("4"))
    different_route = replace(first, route_binding_sha256=_digest("5"))
    assert (
        first.science_semantic_acquisition_sha256()
        == rebatch.science_semantic_acquisition_sha256()
    )
    assert (
        first.physical_execution_plan_sha256()
        != rebatch.physical_execution_plan_sha256()
    )
    assert (
        first.science_execution_plan_sha256() != rebatch.science_execution_plan_sha256()
    )
    assert (
        first.science_execution_plan_sha256()
        == different_legacy.science_execution_plan_sha256()
    )
    assert (
        first.science_semantic_acquisition_sha256()
        == different_route.science_semantic_acquisition_sha256()
    )
    assert (
        first.physical_execution_plan_sha256()
        != different_route.physical_execution_plan_sha256()
    )
    assert (
        first.science_execution_plan_sha256()
        != different_route.science_execution_plan_sha256()
    )
    assert ScienceExecutionPlanV1.from_dict(first.to_dict()) == first

    tampered = first.to_dict()
    tampered["identities"]["science_execution_plan_sha256"] = _digest("f").to_dict()
    with pytest.raises(ContractError, match="identity digest mismatch"):
        ScienceExecutionPlanV1.from_dict(tampered)

    with pytest.raises(ContractError, match="route-specific authority"):
        replace(
            first,
            semantic_acquisition_core=_payload(
                str(SEMANTIC_ACQUISITION_CORE_SCHEMA_VERSION),
                {"cells": ["c1"], "route_id": "engine-specific-route"},
            ),
        )


def test_reduction_plan_is_external_to_execution_identity() -> None:
    first = ScienceReductionPlanV1(
        reducer_descriptors=(_reducer("2"),),
        normalized_configurations=_payload("reducer_configs_v1", {"threshold": 1}),
        evidence_requirements=_payload(
            "science_evidence_requirements_v1", {"record_kinds": ["world_completion"]}
        ),
        offline_decisions=_payload("offline_decisions_v1", {}),
        output_contract_ids=("generic_result_v1",),
    )
    second = replace(first, reducer_descriptors=(_reducer("9"),))
    assert first.reduction_plan_sha256() != second.reduction_plan_sha256()


def test_legacy_metadata_does_not_change_observation_semantic_identity() -> None:
    without_legacy = _record()
    with_legacy = _record(legacy={"generation_label": "G7"})
    assert without_legacy.semantic_dict() == with_legacy.semantic_dict()
    assert (
        without_legacy.semantic_record_sha256() == with_legacy.semantic_record_sha256()
    )
    assert without_legacy.to_dict() != with_legacy.to_dict()
    assert ScienceObservationRecordV1.from_dict(with_legacy.to_dict()) == with_legacy


def test_every_v1_record_kind_has_a_strict_required_payload_shape() -> None:
    abi = json.loads(
        (SPEC_ROOT / "science_observation_abi_v1.json").read_text(encoding="utf-8")
    )
    partition_schema = _load_schema("science_evidence_partition.schema.json")
    for row in abi["record_kinds"]:
        payload = {field: None for field in row["required_payload_fields"]}
        if row["record_kind"] == "world_completion":
            payload["completion_class"] = "EXTINCT"
        kind = ScienceRecordKindV1(row["record_kind"])
        record = _record(kind=kind, payload=payload)
        assert record.record_kind is kind
        partition = ScienceEvidencePartitionV1(
            instrument_manifest_sha256=record.instrument_manifest_sha256,
            science_semantic_acquisition_sha256=(
                record.science_semantic_acquisition_sha256
            ),
            observation_abi_version=SchemaVersion("eiel_science_observation_abi_v1"),
            profile=ScienceObservationProfileV1.STANDARD,
            world_uid=record.world_uid,
            record_kind=kind,
            absolute_tick_start=record.absolute_tick,
            absolute_tick_end=record.absolute_tick + 1,
            rows=(record,),
        ).to_dict()
        jsonschema.validate(partition, partition_schema)
        invalid_partition = json.loads(json.dumps(partition))
        invalid_partition["rows"][0]["payload"]["payload"].pop(
            row["required_payload_fields"][0]
        )
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(invalid_partition, partition_schema)
        payload.pop(row["required_payload_fields"][0])
        with pytest.raises(ContractError, match="frozen record-kind"):
            _record(kind=kind, payload=payload)


def test_high_risk_record_kinds_reject_empty_and_unknown_payloads() -> None:
    schema = _load_schema("science_evidence_partition.schema.json")
    required_fields = {
        "birth_committed": {
            "birth_attempt_id",
            "opportunity_id",
            "child_agent_id",
            "parent_agent_ids",
            "parent_genome_state_ids",
            "birth_mode_id",
            "inheritance_operator_id",
            "recombination_operator_id",
            "parent_hereditary_package_state_ids",
            "child_pre_mutation_genome_state_id",
            "child_post_birth_genome_state_id",
            "child_hereditary_package_state_ids",
            "child_linkage_state_id",
            "founder_id",
            "lineage_id",
            "lineage_depth",
        },
        "decision_action": {
            "decision_id",
            "agent_id",
            "agent_state_before_id",
            "genome_state_id",
            "policy_id",
            "decision_input_state_id",
            "qualified_policy_output_summary",
            "selected_action_id",
            "executed_action_id",
            "execution_status",
            "execution_reason_code",
        },
        "mutation_outcome": {
            "mutation_opportunity_id",
            "mutation_attempt_id",
            "subject_agent_id",
            "origin_agent_id",
            "origin_absolute_tick",
            "attempted",
            "result_class",
            "operator_id",
            "mutation_class_id",
            "affected_coordinates",
            "pre_value_or_state_digest",
            "post_value_or_state_digest",
            "bounded_delta",
            "pre_genome_state_id",
            "post_genome_state_id",
            "reason_code",
        },
    }
    for kind_name, fields in required_fields.items():
        kind = ScienceRecordKindV1(kind_name)
        with pytest.raises(ContractError, match="frozen record-kind"):
            _record(kind=kind, payload={})

        valid = _record(kind=kind, payload={field: None for field in fields})
        partition = ScienceEvidencePartitionV1(
            instrument_manifest_sha256=valid.instrument_manifest_sha256,
            science_semantic_acquisition_sha256=(
                valid.science_semantic_acquisition_sha256
            ),
            observation_abi_version=SchemaVersion("eiel_science_observation_abi_v1"),
            profile=ScienceObservationProfileV1.STANDARD,
            world_uid=valid.world_uid,
            record_kind=kind,
            absolute_tick_start=valid.absolute_tick,
            absolute_tick_end=valid.absolute_tick + 1,
            rows=(valid,),
        ).to_dict()
        unknown = json.loads(json.dumps(partition))
        unknown["rows"][0]["payload"]["payload"]["unknown_field"] = None
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(unknown, schema)


def test_world_completion_and_typed_zero_row_partition() -> None:
    completion = _record(
        kind=ScienceRecordKindV1.WORLD_COMPLETION,
        payload={
            "completion_id": "done-1",
            "completion_class": "EMPTY_INITIAL",
            "terminal_state_sha256": "0" * 64,
            "terminal_population_count": 0,
            "terminal_absolute_tick": 0,
            "next_absolute_tick": 1,
            "terminal_record_count": 0,
            "canonical_record_stream_sha256": "1" * 64,
            "checkpoint_occurrence_ids": [],
            "reason_code": "EMPTY_INITIAL",
        },
    )
    assert completion.record_kind is ScienceRecordKindV1.WORLD_COMPLETION

    partition = ScienceEvidencePartitionV1(
        instrument_manifest_sha256=_digest("1"),
        science_semantic_acquisition_sha256=_digest("4"),
        observation_abi_version=SchemaVersion("eiel_science_observation_abi_v1"),
        profile=ScienceObservationProfileV1.STANDARD,
        world_uid=_digest("5"),
        record_kind=ScienceRecordKindV1.MUTATION_OUTCOME,
        absolute_tick_start=0,
        absolute_tick_end=1,
        rows=(),
    )
    value = partition.to_dict()
    assert value["row_count"] == 0
    assert value["rows"] == []
    assert value["first_sequence"] is None
    assert value["last_sequence"] is None
    assert partition.rows_sha256() == canonical_sha256([])
    assert ScienceEvidencePartitionV1.from_dict(value) == partition


def test_profile_memory_equation_is_integer_and_horizon_independent() -> None:
    expected = 100 + 2 * 1000 + 4 * 10 + 4 * 8 * 3
    envelope = ProfileResourceEnvelopeV1(
        profile=ScienceObservationProfileV1.STANDARD,
        max_records_per_tick_per_world_by_kind=(
            (ScienceRecordKindV1.WORLD_COMPLETION, 1),
        ),
        max_record_bytes_by_kind=((ScienceRecordKindV1.WORLD_COMPLETION, 512),),
        chunk_max_records=64,
        chunk_max_bytes=1000,
        max_in_flight_chunks=2,
        fixed_buffer_bytes=100,
        per_world_cursor_bytes=10,
        per_resident_agent_observation_state_bytes=3,
        logical_world_count=4,
        resident_capacity=8,
        max_resident_observation_bytes=expected,
    )
    assert envelope.computed_resident_bytes() == expected
    assert "total_ticks" not in envelope.to_dict()
    with pytest.raises(ContractError, match="equation mismatch"):
        replace(envelope, max_resident_observation_bytes=expected + 1)


def test_admission_receipt_is_effect_free_and_findings_are_canonical() -> None:
    finding = ScienceFindingV1(
        code="CAPABILITY_ABSENT",
        subject_path="time.branches[0]",
        detail="Scientific forks are a V2 extension.",
        extension_target="B136_SCIENCE_INSTRUMENT_V2_EXTENSION",
    )
    receipt = ScienceAdmissionReceiptV1(
        campaign_spec_sha256=_digest("1"),
        instrument_manifest_sha256=_digest("2"),
        route_binding_sha256=_digest("0"),
        qualified_envelope_projection_sha256=_digest("3"),
        machine_profile_id="EIEL_PRODUCTION_BASELINE_V1",
        machine_profile_sha256=_digest("4"),
        lifecycle_contract_sha256=_digest("5"),
        campaign_compiler_component_sha256=_digest("6"),
        science_semantic_acquisition_sha256=_digest("7"),
        physical_execution_plan_sha256=_digest("8"),
        science_execution_plan_sha256=_digest("9"),
        reduction_plan_sha256=_digest("a"),
        compilation_bundle_sha256=_digest("b"),
        active_qrr_resolution_receipt_sha256=_digest("c"),
        canonical_crca_semantic_identity=_digest("d"),
        admission_kernel_receipt_sha256=_digest("e"),
        findings=(finding,),
        primary_outcome=ScienceAdmissionOutcomeV1.MISSING_SCIENTIFIC_CAPABILITY,
    )
    assert receipt.to_dict()["effects_performed"] is False
    assert receipt.receipt_sha256() == canonical_sha256(receipt.to_dict())
    assert ScienceAdmissionReceiptV1.from_dict(receipt.to_dict()) == receipt
    with pytest.raises(ContractError, match="effect-free"):
        replace(receipt, effects_performed=True)
    wire_with_effect = receipt.to_dict()
    wire_with_effect["effects_performed"] = True
    with pytest.raises(ContractError, match="effect-free"):
        ScienceAdmissionReceiptV1.from_dict(wire_with_effect)
    wire_with_unknown = receipt.to_dict()
    wire_with_unknown["unexpected"] = True
    with pytest.raises(ContractError, match="invalid shape"):
        ScienceAdmissionReceiptV1.from_dict(wire_with_unknown)
    later = replace(finding, subject_path="z.path")
    earlier = replace(finding, subject_path="a.path")
    with pytest.raises(ContractError, match="canonical subject order"):
        replace(receipt, findings=(later, earlier))


def test_descriptors_and_manifest_are_content_addressed_and_reducer_external() -> None:
    control = ScienceControlDescriptorV1(
        control_id="selection.pressure",
        value_contract=_payload("scalar_interval_v1", {"minimum": 0.0, "maximum": 1.0}),
        unit="dimensionless",
        activation_phases=("turnover",),
        heterogeneity_modes=("global",),
        schedule_kinds=("static",),
        readback_capability_ids=("selection_pressure_readback",),
        interaction_ids=(),
        component_id="selection-component-v1",
        qualified_envelope_subjects=("selection.pressure",),
    )
    reducer = _reducer()
    manifest = ScienceInstrumentManifestV1(
        component_graph_sha256=_digest("0"),
        control_catalog_sha256=control.descriptor_sha256(),
        world_profile_catalog_sha256=_digest("1"),
        founder_profile_catalog_sha256=_digest("2"),
        policy_profile_catalog_sha256=_digest("3"),
        mutation_operator_catalog_sha256=_digest("4"),
        observation_abi_sha256=_digest("5"),
        observation_profile_catalog_sha256=_digest("6"),
        reducer_protocol_version=SchemaVersion("eiel_science_reducer_protocol_v1"),
        route_binding_protocol_version=SCIENCE_ROUTE_BINDING_SCHEMA_VERSION,
        eed_release_status=EEDReleaseStatusV1.EXCLUDED,
    )
    assert "reducer" not in manifest.to_dict()
    assert reducer.descriptor_sha256() != manifest.manifest_sha256()
    assert manifest.to_dict()["eed_release_status"] in {"INCLUDED", "EXCLUDED"}

    jsonschema.validate(
        control.to_dict(), _load_schema("control_descriptor.schema.json")
    )
    jsonschema.validate(
        reducer.to_dict(), _load_schema("reducer_descriptor.schema.json")
    )
    jsonschema.validate(
        manifest.to_dict(), _load_schema("instrument_manifest.schema.json")
    )
    with pytest.raises(ContractError, match="schedule kind"):
        replace(control, schedule_kinds=("arbitrary_hook",))
    with pytest.raises(ContractError, match="required reducer purity"):
        replace(reducer, purity_declarations=("OFFLINE_ONLY",))


def test_route_binding_is_additive_and_core_identity_is_route_neutral() -> None:
    manifest = ScienceInstrumentManifestV1(
        component_graph_sha256=_digest("0"),
        control_catalog_sha256=_digest("1"),
        world_profile_catalog_sha256=_digest("2"),
        founder_profile_catalog_sha256=_digest("3"),
        policy_profile_catalog_sha256=_digest("4"),
        mutation_operator_catalog_sha256=_digest("5"),
        observation_abi_sha256=_digest("6"),
        observation_profile_catalog_sha256=_digest("7"),
        reducer_protocol_version=SchemaVersion("eiel_science_reducer_protocol_v1"),
        route_binding_protocol_version=SCIENCE_ROUTE_BINDING_SCHEMA_VERSION,
        eed_release_status=EEDReleaseStatusV1.EXCLUDED,
    )
    b136 = _route_binding(instrument_manifest_sha256=manifest.manifest_sha256())
    next_engine = replace(
        b136,
        route_id="next-gpu-engine-v1",
        native_build_closure_sha256=_digest("c"),
        execution_adapter_sha256=_digest("d"),
        qualified_envelope_sha256=_digest("e"),
        qualification_authority_sha256=_digest("f"),
    )

    assert b136.instrument_manifest_sha256 == next_engine.instrument_manifest_sha256
    assert b136.route_binding_sha256() != next_engine.route_binding_sha256()
    assert ScienceRouteBindingV1.from_dict(b136.to_dict()) == b136
    jsonschema.validate(
        b136.to_dict(), _load_schema("science_route_binding.schema.json")
    )
    assert "native_build_closure_sha256" not in manifest.to_dict()
    assert "qualified_envelope_sha256" not in manifest.to_dict()

    with pytest.raises(ContractError, match="path-free"):
        replace(b136, route_id="branches/engine-next")
    unknown = b136.to_dict()
    unknown["source_path"] = "/tmp/engine"
    with pytest.raises(ContractError, match="invalid shape"):
        ScienceRouteBindingV1.from_dict(unknown)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(unknown, _load_schema("science_route_binding.schema.json"))


def test_all_json_schemas_are_valid_strict_and_campaign_has_no_name_dispatch() -> None:
    for path in sorted(SPEC_ROOT.glob("*.schema.json")):
        schema = json.loads(path.read_text(encoding="utf-8"))
        jsonschema.Draft202012Validator.check_schema(schema)
        assert schema["additionalProperties"] is False

    campaign_schema = _load_schema("science_campaign.schema.json")
    raw = _raw_campaign()
    jsonschema.validate(raw, campaign_schema)
    assert "experiment_name" not in campaign_schema["properties"]

    unknown = json.loads(json.dumps(raw))
    unknown["time"]["generation"] = 31
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(unknown, campaign_schema)

    invalid_discriminant = json.loads(json.dumps(raw))
    invalid_discriminant["controls"][0]["profile_ref"] = "also-set"
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(invalid_discriminant, campaign_schema)


def test_observation_abi_profiles_and_extensions_are_exact() -> None:
    abi = json.loads(
        (SPEC_ROOT / "science_observation_abi_v1.json").read_text(encoding="utf-8")
    )
    assert abi["time_authority"] == "absolute_tick_v1"
    assert {profile["profile_id"] for profile in abi["profiles"]} == {
        "SCIENCE_MINIMAL_V1",
        "SCIENCE_STANDARD_V1",
        "SCIENCE_FORENSIC_V1",
    }
    kinds = {item["record_kind"] for item in abi["record_kinds"]}
    assert kinds == {item.value for item in ScienceRecordKindV1}
    assert "world_completion" in kinds
    assert {item["record_kind"] for item in abi["reserved_extensions"]} == set(
        V2_RESERVED_RECORD_KINDS
    )
    assert abi["resident_byte_equation_forbidden_inputs"] == ["total_ticks"]


def test_empty_partition_is_schema_valid_and_unknown_fields_fail() -> None:
    partition = ScienceEvidencePartitionV1(
        instrument_manifest_sha256=_digest("1"),
        science_semantic_acquisition_sha256=_digest("4"),
        observation_abi_version=SchemaVersion("eiel_science_observation_abi_v1"),
        profile=ScienceObservationProfileV1.MINIMAL,
        world_uid=_digest("5"),
        record_kind=ScienceRecordKindV1.DEATH,
        absolute_tick_start=0,
        absolute_tick_end=1,
        rows=(),
    ).to_dict()
    schema = _load_schema("science_evidence_partition.schema.json")
    jsonschema.validate(partition, schema)
    partition["unknown"] = True
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(partition, schema)


def test_observation_schema_version_constant_is_public() -> None:
    assert str(SCIENCE_OBSERVATION_RECORD_SCHEMA_VERSION) == (
        "eiel_science_observation_record_v1"
    )
