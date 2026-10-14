from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONTRACT_ROOT = PROJECT_ROOT / "docs" / "40_eiel_design"

API_CONTRACT = "eiel_architecture_b_api_contract_v1.json"
PHASE4_CONTRACT = "eiel_architecture_b_phase4_boundary_contract.json"
SCHEDULE_CONTRACT = (
    "eiel_architecture_b_reporting_checkpoint_schedule_contract.json"
)

CANCEL_RECEIPT_FIELDS = [
    "request_id",
    "controller_id",
    "owner_epoch",
    "disposition",
    "first_cancel_request_id",
    "cancel_sequence",
    "state_observed",
    "terminal_status_changed",
]

PERSISTENCE_REQUEST_FIELDS = [
    "api_version",
    "request_id",
    "expected_controller_id",
    "expected_owner_epoch",
    "snapshot_id",
    "snapshot_digest_sha256",
    "outcome",
    "artifact_manifest_digest_sha256",
    "acknowledged_channel_cursor_ranges",
    "failure_code",
]

PERSISTENCE_RECEIPT_FIELDS = [
    "schema_version",
    "request_id",
    "controller_id",
    "owner_epoch",
    "snapshot_id",
    "disposition",
    "persistence_barrier_released",
    "terminal_status_changed_to_failed",
]

INTER_TICK_CONTROL_MARKER_FIELDS = [
    "status",
    "generation_transactions_started",
    "snapshot_frontier",
    "cancel_requested",
    "cancel_consumed",
    "first_cancel_request_id",
    "inter_tick_control_window_open",
    "inter_tick_control_generation",
    "inter_tick_completed_tick",
    "inter_tick_next_tick",
    "inter_tick_control_window_count",
    "inter_tick_control_wait_timeout_ms",
    "controller_protocol_capabilities",
]

REQUIRED_CANCELLATION_CAPABILITIES = {
    "inter_tick_cancellation_control_v1",
    "checkpoint_on_cancel_v1",
}


def _load_contract(filename: str) -> dict[str, Any]:
    with (CONTRACT_ROOT / filename).open(encoding="utf-8") as handle:
        value = json.load(handle)
    assert isinstance(value, dict)
    return value


def _field_names(schema: dict[str, Any]) -> list[str]:
    return [field["name"] for field in schema["fields"]]


def test_phase4_contracts_remain_inside_the_flat_cpu_fixed_mlp_scope() -> None:
    api = _load_contract(API_CONTRACT)
    phase4 = _load_contract(PHASE4_CONTRACT)
    schedule = _load_contract(SCHEDULE_CONTRACT)

    assert phase4["scope"] == {
        "world": "flat",
        "simulation_device": "cpu",
        "policy_family": "fixed_mlp",
        "shock": False,
        "tier4": False,
        "juvenile_steps": 0,
        "runtime_jit": False,
        "certified_flat_mechanics_may_change": False,
    }
    assert schedule["scope"]["world"] == "flat"
    assert schedule["scope"]["simulation_device"] == "cpu"
    assert schedule["scope"]["policy_family"] == "mlp"
    assert schedule["scope"]["policy_selection"] == "fixed"
    assert schedule["scope"]["shock"] is False
    assert api["initial_supported_envelope"]["required_values"]["world"] == "flat"
    assert api["initial_supported_envelope"]["required_values"]["simulation_device"] == "cpu"


def test_cancellation_and_persistence_receipts_are_closed_and_exact() -> None:
    api = _load_contract(API_CONTRACT)
    phase4 = _load_contract(PHASE4_CONTRACT)
    schedule = _load_contract(SCHEDULE_CONTRACT)

    cancel_receipt = api["result_types"]["CancelReceiptV1"]
    assert cancel_receipt["closed_schema"] is True
    assert cancel_receipt["additional_fields_allowed"] is False
    assert cancel_receipt["exact_required_fields"] == CANCEL_RECEIPT_FIELDS
    assert list(cancel_receipt["fields"]) == CANCEL_RECEIPT_FIELDS
    assert "initialized" in cancel_receipt["fields"]["state_observed"]
    assert (
        phase4["cancellation"]["cancel_receipt"]["exact_required_fields"]
        == CANCEL_RECEIPT_FIELDS
    )

    acknowledgement = api["request_types"]["PersistenceAcknowledgementV1"]
    assert acknowledgement["closed_schema"] is True
    assert acknowledgement["additional_fields_allowed"] is False
    assert acknowledgement["exact_required_fields"] == PERSISTENCE_REQUEST_FIELDS
    assert _field_names(acknowledgement) == PERSISTENCE_REQUEST_FIELDS
    assert (
        schedule["persistence_acknowledgement_contract"][
            "request_exact_required_fields"
        ]
        == PERSISTENCE_REQUEST_FIELDS
    )

    persistence_receipt = api["result_types"][
        "PersistenceAcknowledgementReceiptV1"
    ]
    assert persistence_receipt["closed_schema"] is True
    assert persistence_receipt["additional_fields_allowed"] is False
    assert (
        persistence_receipt["exact_required_fields"]
        == PERSISTENCE_RECEIPT_FIELDS
    )
    assert list(persistence_receipt["fields"]) == PERSISTENCE_RECEIPT_FIELDS
    assert (
        schedule["persistence_acknowledgement_contract"]["receipt_contract"][
            "exact_required_fields"
        ]
        == PERSISTENCE_RECEIPT_FIELDS
    )


def test_inter_tick_request_uses_exact_generation_2_control_coordinates() -> None:
    api = _load_contract(API_CONTRACT)
    phase4 = _load_contract(PHASE4_CONTRACT)
    schedule = _load_contract(SCHEDULE_CONTRACT)

    assert set(api["required_capabilities"]) == REQUIRED_CANCELLATION_CAPABILITIES
    assert set(phase4["required_capabilities"]) == REQUIRED_CANCELLATION_CAPABILITIES
    assert set(schedule["required_capabilities"]) == REQUIRED_CANCELLATION_CAPABILITIES

    marker = api["result_types"]["InterTickCancellationControlMarkerV1"]
    assert marker["closed_schema"] is True
    assert marker["additional_fields_allowed"] is False
    assert marker["exact_required_fields"] == INTER_TICK_CONTROL_MARKER_FIELDS
    assert list(marker["fields"]) == INTER_TICK_CONTROL_MARKER_FIELDS

    cancellation = phase4["cancellation"]
    assert cancellation["required_capability"] == "inter_tick_cancellation_control_v1"
    assert (
        cancellation["focused_case_request_phase"]
        == "after_completed_tick_before_next_tick"
    )
    assert cancellation["request_must_be_accepted_while_generation_transaction_open"] is True
    assert cancellation["focused_live_acceptance_coordinate"] == {
        "inter_tick_control_generation": 2,
        "inter_tick_completed_tick": 0,
        "inter_tick_next_tick": 1,
        "inter_tick_control_window_count": 3,
        "steps_per_generation": 80,
        "transaction_state": "open",
        "status": "inter_tick_control",
        "inter_tick_control_window_open": True,
    }
    assert (
        cancellation["focused_live_consumption_coordinate"]
        == "generation:3:tick:0:post_turnover_pre_snapshot_cutoff"
    )
    assert cancellation["observation_cardinality"] == 1
    assert cancellation["next_generation_may_start_after_observation"] is False
    assert "after generation G finalization and turnover commit" in cancellation[
        "required_consumption_coordinate"
    ]
    assert "arrival after a boundary snapshot cutoff" in cancellation[
        "cutoff_race_reservation"
    ]

    race = api["cancellation_race_rule"]
    assert race["required_capability"] == "inter_tick_cancellation_control_v1"
    assert race["tick_interruption_or_mutation_allowed"] is False
    assert "inter_tick_control_generation:2" in race[
        "phase4_live_control_coordinate"
    ]
    assert "exactly once" in race["next_boundary_observation"]
    assert "after G finalization and turnover commit" in race[
        "next_boundary_observation"
    ]
    assert "cannot alter or reserve that boundary" in race[
        "late_arrival_after_snapshot_cutoff"
    ]

    schedule_cancellation = schedule["cancellation_contract"]
    assert schedule_cancellation["terminal_outcome_cardinality"] == 1
    assert (
        schedule_cancellation["focused_phase4_request_phase"]
        == "after_completed_tick_before_next_tick"
    )
    assert "exactly once" in schedule_cancellation["required_observation_point"]
    assert "generation 2 after completed tick 0 before tick 1" in schedule_cancellation[
        "phase4_exact_coordinates"
    ]
    assert "arriving after the snapshot cutoff" in schedule["schedule_resolution"][
        "post_cutoff_cancellation_rule"
    ]
    case = next(
        case for case in schedule["conformance_cases"] if case["case_id"] == "E"
    )
    assert (
        case["name"]
        == "generation_2_inter_tick_safe_boundary_cancellation_with_resume"
    )
    assert "generation 2 after completed tick 0 and before tick 1" in case["setup"]

    serialized = " ".join(json.dumps(value) for value in (api, phase4, schedule))
    assert "mid_generation_tick_execution" not in serialized
    assert "simultaneously satisfied" not in serialized


def test_cancelled_terminal_requires_a_durable_resumable_checkpoint() -> None:
    api = _load_contract(API_CONTRACT)
    phase4 = _load_contract(PHASE4_CONTRACT)
    schedule = _load_contract(SCHEDULE_CONTRACT)

    schedule_fields = {
        field["name"]: field
        for field in api["request_types"]["SnapshotScheduleV1"]["fields"]
    }
    assert schedule_fields["checkpoint_on_cancel"]["required"] is True
    assert schedule_fields["checkpoint_on_cancel"]["type"] == "boolean"

    checkpoint = phase4["checkpoint"]
    cancellation = phase4["cancellation"]
    assert checkpoint["checkpoint_on_cancel_required_for_phase4_case"] is True
    assert checkpoint["cancelled_terminal_requires_durable_acknowledgement"] is True
    assert cancellation["checkpoint_on_cancel_required"] is True
    assert cancellation["checkpoint_must_be_durable_before_cancelled_terminal"] is True
    assert cancellation[
        "checkpoint_resume_must_match_uninterrupted_boundary_authority"
    ] is True

    configured_checkpoint = schedule["schedule_configuration"]["checkpoint"]
    assert configured_checkpoint["checkpoint_on_cancel"] is True
    assert configured_checkpoint["partial_generation_checkpoint_allowed"] is False
    assert "strict-load" in configured_checkpoint[
        "cancellation_checkpoint_resume_requirement"
    ]
    assert "fresh owner" in schedule["cancellation_contract"][
        "checkpoint_resumability"
    ]


def test_failure_publishes_one_failed_outcome_and_no_false_success() -> None:
    api = _load_contract(API_CONTRACT)
    phase4 = _load_contract(PHASE4_CONTRACT)
    schedule = _load_contract(SCHEDULE_CONTRACT)

    failure = phase4["failure_atomicity"]
    assert failure["terminal_outcome_cardinality"] == 1
    assert failure["terminal_status_after_failure"] == "failed"
    assert failure["success_after_failure_allowed"] is False
    assert failure["cancelled_after_failure_allowed"] is False
    assert failure["cancellation_checkpoint_failure_effect"] == "failed_not_cancelled"
    assert {
        "run_complete terminal status",
        "cancelled terminal status",
        "success true",
        "valid checkpoint marker for an unacknowledged checkpoint",
    }.issubset(failure["forbidden_false_success_publications"])

    terminal_status = api["result_types"]["FinalResultV1"]["fields"][
        "terminal_status"
    ]
    assert terminal_status == "enum run_complete, cancelled, failed"
    assert schedule["cancellation_contract"]["terminal_outcome_cardinality"] == 1
    failure_case = next(
        case for case in schedule["conformance_cases"] if case["case_id"] == "F"
    )
    required_result = " ".join(failure_case["required_result"])
    assert "exactly one failed terminal outcome" in required_result
    assert "Publish no cancelled result, run_complete result, success marker" in required_result
