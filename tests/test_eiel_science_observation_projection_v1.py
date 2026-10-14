from __future__ import annotations

import pytest

from emergence_contracts import CanonicalPayloadV1, ContentDigest, SchemaVersion, ScienceRecordKindV1
from emergence_contracts.errors import ContractError
from emergence_contracts.science_observations import canonical_science_record_stream_sha256_v1

from engine.native_runtime.science_observation_projection_v1 import (
    NativeCudaV2ObservationContextV1,
    NativeCudaV2ObservationOutputV1,
    SCIENCE_NATIVE_OBSERVATION_OUTPUT_SCHEMA_VERSION,
    finalize_native_cuda_v2_observations_v1,
    project_native_cuda_v2_event_dispatch_v1,
    project_native_cuda_v2_state_checkpoint_v1,
)


def _digest(character: str) -> ContentDigest:
    return ContentDigest("sha256", character * 64)


def _payload(kind: ScienceRecordKindV1, **overrides: object) -> dict[str, object]:
    from emergence_contracts.science_instrument import _RECORD_REQUIRED_PAYLOAD_FIELDS

    value = {field: f"{kind.value}:{field}" for field in _RECORD_REQUIRED_PAYLOAD_FIELDS[kind]}
    value.update(overrides)
    return value


def _row(
    kind: ScienceRecordKindV1,
    *,
    address: str,
    tick: int,
    phase: int,
    payload: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "producer_component_id": "cuda-v2-science",
        "producer_contract_version": "eiel_cuda_v2_provider_v1",
        "producer_phase_registry_ordinal": phase,
        "absolute_tick": tick,
        "semantic_source_address": address,
        "record_kind": kind.value,
        "record_kind_version": 1,
        "control_binding_sha256": _digest("a"),
        "payload": {
            "schema_version": "eiel_science_test_payload_v1",
            "payload": payload if payload is not None else _payload(kind),
        },
    }


def _context() -> NativeCudaV2ObservationContextV1:
    return NativeCudaV2ObservationContextV1(
        instrument_manifest_sha256=_digest("b"),
        crca_semantic_run_digest=_digest("c"),
        science_semantic_acquisition_sha256=_digest("d"),
        cell_id="cell-a",
        logical_world_id=7,
        semantic_world_key=CanonicalPayloadV1.from_value(
            SchemaVersion("eiel_science_semantic_world_key_v1"),
            {"world": "world-a"},
        ),
    )


def test_native_output_finalizes_through_canonical_identity_authority() -> None:
    event = _row(
        ScienceRecordKindV1.BIRTH_ATTEMPT,
        address="world/7/event/1",
        tick=3,
        phase=20,
    )
    output = NativeCudaV2ObservationOutputV1(event_dispatch=(event,))

    records = finalize_native_cuda_v2_observations_v1(output, context=_context())

    assert len(records) == 1
    assert records[0].record_kind is ScienceRecordKindV1.BIRTH_ATTEMPT
    assert records[0].absolute_tick == 3
    assert records[0].lifecycle_epoch is None
    assert records[0].semantic_dict()["semantic_source_address"] == "world/7/event/1"


def test_state_checkpoint_and_event_order_is_batch_neutral() -> None:
    state = _row(
        ScienceRecordKindV1.WORLD_STATE_SNAPSHOT,
        address="world/7/state/1",
        tick=1,
        phase=10,
        payload=_payload(
            ScienceRecordKindV1.WORLD_STATE_SNAPSHOT,
            world_state_id="state-1",
            field_set_version="fields-v1",
            qualified_field_values={},
            snapshot_reason="initial",
        ),
    )
    event = _row(
        ScienceRecordKindV1.BIRTH_ATTEMPT,
        address="world/7/event/1",
        tick=2,
        phase=20,
    )
    event_late = _row(
        ScienceRecordKindV1.BIRTH_COMMITTED,
        address="world/7/event/2",
        tick=3,
        phase=20,
    )
    checkpoint = _row(
        ScienceRecordKindV1.CHECKPOINT_MARKER,
        address="world/7/checkpoint/1",
        tick=4,
        phase=30,
        payload=_payload(
            ScienceRecordKindV1.CHECKPOINT_MARKER,
            authoritative_state_sha256=_digest("e").to_dict(),
            observation_prefix_sha256=_digest("f").to_dict(),
        ),
    )
    context = _context()
    forward = NativeCudaV2ObservationOutputV1(
        event_dispatch=(event, event_late),
        state_projection=(state,),
        checkpoint_projection=(checkpoint,),
    )
    reverse = NativeCudaV2ObservationOutputV1(
        event_dispatch=(event_late, event),
        state_projection=(state,),
        checkpoint_projection=(checkpoint,),
    )

    first = finalize_native_cuda_v2_observations_v1(forward, context=context)
    second = finalize_native_cuda_v2_observations_v1(reverse, context=context)

    assert tuple(row.absolute_tick for row in first) == (1, 2, 3, 4)
    assert canonical_science_record_stream_sha256_v1(first) == canonical_science_record_stream_sha256_v1(
        second
    )
    assert all(
        key not in row.semantic_dict()
        for row in first
        for key in ("route_id", "batch_index", "physical_slot", "cuda_device")
    )


def test_decision_action_without_explicit_causal_fields_fails_closed() -> None:
    action = _row(
        ScienceRecordKindV1.DECISION_ACTION,
        address="world/7/agent/2/decision/1",
        tick=5,
        phase=20,
        payload=_payload(
            ScienceRecordKindV1.DECISION_ACTION,
            decision_id="decision-1",
            agent_id="agent-2",
            agent_state_before_id="state-before",
            genome_state_id="genome-1",
            policy_id="policy-1",
            decision_input_state_id="input-1",
            qualified_policy_output_summary={},
            # selected_action_id, executed_action_id, status, and reason are
            # deliberately absent: logits/compact rows cannot substitute for
            # these causal fields.
        ),
    )
    for field in (
        "selected_action_id",
        "executed_action_id",
        "execution_status",
        "execution_reason_code",
    ):
        action["payload"]["payload"].pop(field)

    with pytest.raises(ContractError, match="MISSING_OBSERVATION_CAPABILITY"):
        project_native_cuda_v2_event_dispatch_v1((action,))


def test_route_and_raw_checkpoint_fields_are_rejected() -> None:
    route_contaminated = _row(
        ScienceRecordKindV1.BIRTH_ATTEMPT,
        address="world/7/event/route",
        tick=1,
        phase=20,
    )
    route_contaminated["physical_batch_index"] = 0
    with pytest.raises(ContractError, match="route/physical"):
        project_native_cuda_v2_event_dispatch_v1((route_contaminated,))

    nested_route_contaminated = _row(
        ScienceRecordKindV1.WORLD_STATE_SNAPSHOT,
        address="world/7/state/route",
        tick=1,
        phase=10,
        payload=_payload(
            ScienceRecordKindV1.WORLD_STATE_SNAPSHOT,
            qualified_field_values={"native_route_id": "B136"},
        ),
    )
    with pytest.raises(ContractError, match="route/physical"):
        project_native_cuda_v2_state_checkpoint_v1((nested_route_contaminated,), ())

    raw_checkpoint = {
        "schema_version": "cuda_checkpoint_projection_v2",
        "tensors": {"population": "opaque"},
    }
    with pytest.raises(ContractError):
        project_native_cuda_v2_state_checkpoint_v1((), (raw_checkpoint,))


def test_output_wire_requires_version_and_reports_no_effects() -> None:
    base = {
        "schema_version": str(SCIENCE_NATIVE_OBSERVATION_OUTPUT_SCHEMA_VERSION),
        "event_dispatch": [],
        "state_projection": [],
        "checkpoint_projection": [],
        "effects_performed": False,
    }
    assert NativeCudaV2ObservationOutputV1.from_mapping(base).event_dispatch == ()

    with pytest.raises(ContractError, match="effects_performed"):
        NativeCudaV2ObservationOutputV1.from_mapping({**base, "effects_performed": True})

    with pytest.raises(ContractError, match="missing fields"):
        NativeCudaV2ObservationOutputV1.from_mapping({key: value for key, value in base.items() if key != "schema_version"})
