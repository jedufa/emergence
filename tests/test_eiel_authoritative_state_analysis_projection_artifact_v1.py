from __future__ import annotations

import json
from pathlib import Path

import pytest

from engine.authoritative_state.schema import load_schema_v1
from engine.campaigns import authoritative_state_analysis_projection_v1 as artifact
from engine.campaigns.provider_native_chronicle_v1 import (
    authoritative_chronicle_event_v1,
    authoritative_provider_projection_v1,
)


def _identity() -> dict[str, str]:
    schema = load_schema_v1()
    return {
        "authoritative_schema_hash": schema.schema_hash,
        "binary_schema_hash": "b" * 64,
        "checkpoint_schema_hash": schema.components[
            "checkpoint_schema.json"
        ]["contract_hash"],
    }


def _attempt() -> dict:
    return {
        "attempt_index": 0,
        "eligible_parent_counts": [2, 1],
        "selected_parent_ids": [3, 5],
        "success": True,
    }


def _source_values(*, offspring: object = 2) -> dict:
    return {
        "reporting.injury_start.v1": [0.25],
        "reporting.injury_end.v1": [0.75],
        "identity.lineage_founder_id.v1": [7],
        "reporting.offspring_count.v1": [offspring],
    }


def _coordinates() -> dict[str, object]:
    return {
        "cell_id": "cell-a",
        "start_generation": 0,
        "terminal_generation": 1,
        "population_semantics": "surviving_source_cohort_after_turnover_v1",
    }


def _formal_event() -> tuple[dict, str]:
    identity = _identity()
    provider = authoritative_provider_projection_v1(
        attempt=_attempt(),
        source_record_sha256="a" * 64,
        identity_bindings=identity,
    )
    chronicle = authoritative_chronicle_event_v1(
        run_id="cell-a",
        generation=0,
        step=4,
        intra_step_sequence=0,
        subject_id="0:0",
        provider_projection=provider,
        identity_bindings=identity,
    )
    values = chronicle["values"]
    event = {
        "run_id": values["chronicle.run_id.v1"],
        "generation": values["chronicle.generation.v1"],
        "step": values["chronicle.step.v1"],
        "phase_order": values["chronicle.phase_order.v1"],
        "intra_step_sequence": values[
            "chronicle.intra_step_sequence.v1"
        ],
        "event_type": values["chronicle.event_type.v1"],
        "subject_id": values["chronicle.subject_id.v1"],
        "source_component": values["chronicle.source_component.v1"],
        "authoritative_state_schema_v1": {
            "provider_projection": provider,
            "chronicle_event": chronicle,
        },
    }
    return event, chronicle["event_id"]


def _persist_inputs(tmp_path: Path) -> tuple[Path, Path, str]:
    source = artifact.authoritative_analysis_source_payload_v1(
        source_values=_source_values(),
        identity_bindings=_identity(),
        source_coordinates=_coordinates(),
        row_agent_ids=[3],
    )
    source_path = tmp_path / "analysis-source.json"
    source_path.write_text(
        json.dumps(source, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    event, event_id = _formal_event()
    events_path = tmp_path / "formal-events.jsonl"
    events_path.write_text(
        json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return source_path, events_path, event_id


@pytest.mark.parametrize("value", [True, 2**31, -1])
def test_persisted_analysis_source_rejects_non_int32_offspring(
    value: object,
) -> None:
    with pytest.raises(ValueError, match="offspring_count"):
        artifact.authoritative_analysis_source_payload_v1(
            source_values=_source_values(offspring=value),
            identity_bindings=_identity(),
            source_coordinates=_coordinates(),
            row_agent_ids=[3],
        )


def test_persisted_analysis_source_accepts_int32_maximum(
    tmp_path: Path,
) -> None:
    source = artifact.authoritative_analysis_source_payload_v1(
        source_values=_source_values(offspring=2**31 - 1),
        identity_bindings=_identity(),
        source_coordinates=_coordinates(),
        row_agent_ids=[3],
    )
    assert source["source_values"]["reporting.offspring_count.v1"] == [
        2**31 - 1
    ]
    path = tmp_path / "source.json"
    path.write_text(
        json.dumps(source, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    assert artifact.load_analysis_source_v1(path) == source


def test_persisted_analysis_source_writer_is_non_overwritable(
    tmp_path: Path,
) -> None:
    source = artifact.authoritative_analysis_source_payload_v1(
        source_values=_source_values(),
        identity_bindings=_identity(),
        source_coordinates=_coordinates(),
        row_agent_ids=[3],
    )
    path = tmp_path / "immutable-source.json"

    assert artifact.write_immutable_analysis_source_v1(path, source) == (
        source["source_payload_sha256"]
    )
    assert artifact.load_analysis_source_v1(path) == source
    with pytest.raises(ValueError, match="already exists"):
        artifact.write_immutable_analysis_source_v1(path, source)


def test_streamed_projection_receipt_is_deterministic_and_non_overwritable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_path, events_path, event_id = _persist_inputs(tmp_path)
    # The integration owner regenerates the schema once all source edits land.
    monkeypatch.setattr(
        artifact, "_validate_registry_projection_dtypes", lambda: None
    )
    receipt = artifact.build_analysis_projection_receipt_v1(
        analysis_source_path=source_path,
        chronicle_events_path=events_path,
        chronicle_event_id=event_id,
    )
    repeated = artifact.build_analysis_projection_receipt_v1(
        analysis_source_path=source_path,
        chronicle_events_path=events_path,
        chronicle_event_id=event_id,
    )
    assert repeated == receipt
    assert receipt["analysis_projection"]["values"] == {
        "analysis.injury_delta.v1": [0.5],
        "analysis.lineage_id.v1": ["7"],
        "analysis.offspring_count.v1": [2],
    }
    output = tmp_path / "projection-receipt.json"
    assert (
        artifact.write_immutable_analysis_projection_receipt_v1(
            output, receipt
        )
        == receipt["receipt_sha256"]
    )
    with pytest.raises(ValueError, match="already exists"):
        artifact.write_immutable_analysis_projection_receipt_v1(
            output, receipt
        )


def test_streamed_projection_rejects_formal_schema_identity_drift(
    tmp_path: Path,
) -> None:
    source_path, events_path, event_id = _persist_inputs(tmp_path)
    del source_path
    event = json.loads(events_path.read_text(encoding="utf-8"))
    event["step"] += 1
    events_path.write_text(json.dumps(event) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="event identities differ"):
        artifact.stream_exact_chronicle_authority_v1(
            events_path,
            expected_event_id=event_id,
            identity_bindings=_identity(),
        )


def test_streamed_projection_rejects_duplicate_exact_event(
    tmp_path: Path,
) -> None:
    _, events_path, event_id = _persist_inputs(tmp_path)
    line = events_path.read_text(encoding="utf-8")
    events_path.write_text(line + line, encoding="utf-8")
    with pytest.raises(ValueError, match="duplicated"):
        artifact.stream_exact_chronicle_authority_v1(
            events_path,
            expected_event_id=event_id,
            identity_bindings=_identity(),
        )
