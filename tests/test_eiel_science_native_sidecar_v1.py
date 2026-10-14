from __future__ import annotations

from copy import deepcopy
import hashlib
import json

import pytest

from emergence_contracts import (
    CanonicalPayloadV1,
    ContentDigest,
    SchemaVersion,
    ScienceObservationDraftV1,
    ScienceRecordKindV1,
    canonical_json_bytes,
    canonical_science_record_stream_sha256_v1,
    finalize_science_observation_drafts_v1,
)
from emergence_evidence.science_native_sidecar_v1 import (
    SCIENCE_NATIVE_SIDECAR_FRONTIER_SCHEMA_V1,
    SCIENCE_NATIVE_SIDECAR_MODE_V1,
    SCIENCE_NATIVE_SIDECAR_OBSERVATION_ABI_V1,
    SCIENCE_NATIVE_SIDECAR_SCHEMA_V1,
    ScienceNativeSidecarError,
    load_native_science_sidecar_v1,
)


_PRODUCER = "native-sidecar-test-producer-v1"
_PHASE = 7


def _digest(label: str) -> ContentDigest:
    return ContentDigest("sha256", hashlib.sha256(label.encode()).hexdigest())


def _payload(value: dict[str, object]) -> CanonicalPayloadV1:
    return CanonicalPayloadV1.from_value(
        SchemaVersion("native_sidecar_test_payload_v1"), value
    )


def _records():
    drafts = (
        ScienceObservationDraftV1(
            producer_component_id=_PRODUCER,
            producer_contract_version=SchemaVersion("native_sidecar_test_producer_v1"),
            producer_phase_registry_ordinal=_PHASE,
            absolute_tick=0,
            semantic_source_address="world/test/control",
            record_kind=ScienceRecordKindV1.CONTROL_BINDING,
            record_kind_version=1,
            control_binding_sha256=_digest("control"),
            payload=_payload(
                {
                    "binding_id": "binding-a",
                    "semantic_cell_id": "cell-a",
                    "seed_bindings": {"task": 11},
                    "control_values": [],
                    "control_schedules": {},
                    "registered_profile_ids": ["SCIENCE_STANDARD_V1"],
                    "semantic_envelope_resolution_sha256": _digest("envelope").hex,
                    "semantic_applied_readback_sha256": _digest("readback").hex,
                    "continuation_parent_execution_identity": None,
                }
            ),
        ),
        ScienceObservationDraftV1(
            producer_component_id=_PRODUCER,
            producer_contract_version=SchemaVersion("native_sidecar_test_producer_v1"),
            producer_phase_registry_ordinal=_PHASE,
            absolute_tick=20,
            semantic_source_address="world/test/completion",
            record_kind=ScienceRecordKindV1.WORLD_COMPLETION,
            record_kind_version=1,
            control_binding_sha256=_digest("control"),
            payload=_payload(
                {
                    "completion_id": "completion-a",
                    "completion_class": "TERMINAL_HORIZON",
                    "terminal_state_sha256": _digest("terminal").hex,
                    "terminal_population_count": 1,
                    "terminal_absolute_tick": 20,
                    "next_absolute_tick": 20,
                    "terminal_record_count": 2,
                    "canonical_record_stream_sha256": _digest("prefix").hex,
                    "checkpoint_occurrence_ids": [],
                    "reason_code": "TERMINAL_HORIZON",
                }
            ),
        ),
    )
    return finalize_science_observation_drafts_v1(
        drafts,
        instrument_manifest_sha256=_digest("instrument"),
        crca_semantic_run_digest=_digest("crca"),
        science_semantic_acquisition_sha256=_digest("acquisition"),
        cell_id="cell-a",
        logical_world_id=11,
        semantic_world_key=CanonicalPayloadV1.from_value(
            SchemaVersion("eiel_science_semantic_world_key_v1"),
            {"semantic_cell_id": "cell-a"},
        ),
    )


def _sidecar(records):
    prefix = canonical_science_record_stream_sha256_v1(records).to_dict()
    frontier = {
        "schema_version": SCIENCE_NATIVE_SIDECAR_FRONTIER_SCHEMA_V1,
        "observation_mode": SCIENCE_NATIVE_SIDECAR_MODE_V1,
        "observation_abi": SCIENCE_NATIVE_SIDECAR_OBSERVATION_ABI_V1,
        "next_absolute_tick": 20,
        "per_world": {
            "cell-a": {
                "cell_id": "cell-a",
                "logical_world_id": 11,
                "world_uid": records[0].world_uid.to_dict(),
                "record_count": 2,
                "next_world_local_sequence": 2,
                "last_absolute_tick": 20,
                "prefix_chain_sha256": prefix,
            }
        },
    }
    return {
        "schema_version": SCIENCE_NATIVE_SIDECAR_SCHEMA_V1,
        "observation_abi": SCIENCE_NATIVE_SIDECAR_OBSERVATION_ABI_V1,
        "observation_mode": SCIENCE_NATIVE_SIDECAR_MODE_V1,
        "batch_id": "science-batch-0000",
        "cell_ids": ["cell-a"],
        "records_by_cell": {"cell-a": [row.to_dict() for row in records]},
        "frontier": frontier,
    }


def test_load_native_science_sidecar_validates_bytes_stream_and_frontier(
    tmp_path,
) -> None:
    body = _sidecar(_records())
    payload = canonical_json_bytes(body)
    path = tmp_path / "records-terminal.json"
    path.write_bytes(payload)

    loaded = load_native_science_sidecar_v1(
        path,
        expected_sha256=hashlib.sha256(payload).hexdigest(),
        expected_batch_id="science-batch-0000",
        expected_cell_ids=("cell-a",),
        expected_next_absolute_tick=20,
        producer_phase_registry_ordinals={_PRODUCER: _PHASE},
    )

    assert loaded.cell_ids == ("cell-a",)
    assert len(loaded.records_for_cell("cell-a")) == 2
    assert loaded.frontier == body["frontier"]


def test_load_native_science_sidecar_rejects_frontier_or_noncanonical_bytes(
    tmp_path,
) -> None:
    body = _sidecar(_records())
    corrupt = deepcopy(body)
    corrupt["frontier"]["per_world"]["cell-a"]["record_count"] = 1
    path = tmp_path / "corrupt.json"
    path.write_bytes(canonical_json_bytes(corrupt))
    with pytest.raises(ScienceNativeSidecarError, match="not self-consistent"):
        load_native_science_sidecar_v1(
            path,
            producer_phase_registry_ordinals={_PRODUCER: _PHASE},
        )

    pretty = tmp_path / "pretty.json"
    pretty.write_text(json.dumps(body, indent=2), encoding="ascii")
    with pytest.raises(ScienceNativeSidecarError, match="bytes are not canonical"):
        load_native_science_sidecar_v1(
            pretty,
            producer_phase_registry_ordinals={_PRODUCER: _PHASE},
        )


def test_load_native_science_sidecar_rejects_symbolic_link(tmp_path) -> None:
    body = _sidecar(_records())
    target = tmp_path / "records-terminal.json"
    target.write_bytes(canonical_json_bytes(body))
    link = tmp_path / "records-link.json"
    link.symlink_to(target)

    with pytest.raises(ScienceNativeSidecarError, match="symlink"):
        load_native_science_sidecar_v1(
            link,
            producer_phase_registry_ordinals={_PRODUCER: _PHASE},
        )
