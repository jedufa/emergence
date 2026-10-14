from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from engine.authoritative_state.chronicle import (
    ChronicleSchemaBindingError,
    chronicle_event_id,
    validate_chronicle_contract,
    validate_chronicle_event,
)


SPEC = Path(__file__).parents[1] / "_spec" / "authoritative_state" / "v1"


def _contract() -> dict:
    return json.loads((SPEC / "chronicle_schema.json").read_text(encoding="utf-8"))


def _event(contract: dict, bindings: dict[str, str]) -> dict:
    manifest = deepcopy(contract["fields"])
    values = {
        "chronicle.run_id.v1": "run-1",
        "chronicle.generation.v1": 7,
        "chronicle.step.v1": 12,
        "chronicle.phase_order.v1": 720,
        "chronicle.intra_step_sequence.v1": 3,
        "chronicle.event_type.v1": "provider_native_reproduction_attempt",
        "chronicle.subject_id.v1": "0:3",
        "chronicle.source_component.v1": "segment_artifact_observer",
    }
    event = {
        "chronicle_schema_version": contract["schema_version"],
        "chronicle_schema_hash": contract["contract_hash"],
        "bindings": dict(bindings),
        "field_manifest": manifest,
        "values": values,
        "open_values": {"measurements": {"success": True}},
        "open_value_provenance": {
            "measurements": {
                "success": {
                    "source_field_id": "reproduction.success.v1",
                    "source_record_hash": "a" * 64,
                    "transform": "copied",
                }
            }
        },
    }
    event["event_id"] = chronicle_event_id(contract=contract, values=values)
    return event


def test_chronicle_contract_and_event_bind_exact_identity() -> None:
    contract = _contract()
    bindings = {
        "authoritative_schema_hash": "1" * 64,
        "binary_schema_hash": "2" * 64,
        "checkpoint_schema_hash": "3" * 64,
        "provider_projection_hash": "4" * 64,
    }
    assert validate_chronicle_contract(contract) == contract["contract_hash"]
    event = _event(contract, bindings)
    assert validate_chronicle_event(
        event, contract=contract, expected_bindings=bindings
    ) == event["event_id"]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda event: event.__setitem__("chronicle_schema_hash", "f" * 64),
            "schema hash",
        ),
        (
            lambda event: event["bindings"].__setitem__(
                "provider_projection_hash", "f" * 64
            ),
            "provider_projection_hash",
        ),
        (
            lambda event: event["field_manifest"][0].__setitem__(
                "shape", ["rows"]
            ),
            "shape",
        ),
        (
            lambda event: event["open_values"].__setitem__("undeclared", {}),
            "undeclared",
        ),
        (
            lambda event: event["open_value_provenance"]["measurements"].pop(
                "success"
            ),
            "provenance keys",
        ),
        (
            lambda event: event.__setitem__("event_id", "evt_wrong"),
            "event ID",
        ),
    ],
)
def test_chronicle_fails_closed_for_schema_and_provenance_drift(
    mutation, message: str
) -> None:
    contract = _contract()
    bindings = {
        "authoritative_schema_hash": "1" * 64,
        "binary_schema_hash": "2" * 64,
        "checkpoint_schema_hash": "3" * 64,
        "provider_projection_hash": "4" * 64,
    }
    event = _event(contract, bindings)
    mutation(event)
    with pytest.raises(ChronicleSchemaBindingError, match=message):
        validate_chronicle_event(
            event, contract=contract, expected_bindings=bindings
        )


def test_chronicle_identity_changes_when_identity_value_changes() -> None:
    contract = _contract()
    values = _event(contract, {})["values"]
    first = chronicle_event_id(contract=contract, values=values)
    changed = dict(values)
    changed["chronicle.intra_step_sequence.v1"] += 1
    assert chronicle_event_id(contract=contract, values=changed) != first
