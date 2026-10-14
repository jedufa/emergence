from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pytest

import eiel_composition.checkpoint_occurrence as checkpoint_occurrence
from eiel_composition.checkpoint_occurrence import (
    CheckpointOccurrenceError,
    inspect_workhorse_semantic_checkpoint_occurrence_v2,
    materialize_workhorse_semantic_checkpoint_occurrence_v2,
    restore_workhorse_semantic_checkpoint_occurrence_v2,
)
from emergence_contracts import (
    AttemptId,
    CheckpointId,
    CheckpointOccurrenceInspectionStatusV1,
    RunId,
    WorkhorseSemanticCheckpointOccurrenceCommitV2,
    WorkhorseSemanticCheckpointOccurrenceV2,
    canonical_sha256,
)


SCIENCE_RELATIVE = (
    "_artifacts/tw-cuda-full-evolution-reform-20260825/"
    "science_consumer_extension_contract.json"
)


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _source(tmp_path: Path) -> dict[str, Any]:
    science_path = tmp_path / "repo" / SCIENCE_RELATIVE
    science_path.parent.mkdir(parents=True)
    science_bytes = b'{"schema_version":"science-consumer-fixture-v1"}\n'
    science_path.write_bytes(science_bytes)

    source_root = tmp_path / "semantic-fragment"
    source_root.mkdir()
    state_path = source_root / "base_checkpoint_state.bin"
    state_bytes = bytes(range(64))
    state_path.write_bytes(state_bytes)
    state_path.chmod(0o444)

    ack_path = tmp_path / "event-ack.json"
    ack_bytes = b'{"schema_version":"workhorse-event-ack-fixture-v2"}\n'
    ack_path.write_bytes(ack_bytes)
    ack_path.chmod(0o444)
    state_sha = _sha(state_bytes)
    artifact_sha = "a" * 64
    ack = {
        "receipt_path": str(ack_path.resolve()),
        "receipt_file_sha256": _sha(ack_bytes),
        "receipt_body_sha256": "1" * 64,
        "dispatch_content_sha256": "2" * 64,
        "event_rows_file_sha256": "3" * 64,
        "event_counts_file_sha256": "4" * 64,
        "world_uids": [97000],
        "start_next_absolute_tick": 21,
        "end_next_absolute_tick": 22,
        "artifact_identity_sha256": artifact_sha,
        "rng_identity_sha256": "5" * 64,
        "post_state_sha256": state_sha,
        "dispatch_identity_sha256": "6" * 64,
    }
    science_body = {
        "schema_version": "eiel_workhorse_science_checkpoint_compatibility_binding_v2",
        "science_consumer_extension_contract_path": SCIENCE_RELATIVE,
        "science_consumer_extension_contract_sha256": _sha(science_bytes),
        "science_extension_contract_schema": (
            "eiel_cuda_full_evolution_science_extension_contract_v1"
        ),
        "science_checkpoint_extension_schema": (
            "eiel_cuda_full_evolution_science_checkpoint_extension_v1"
        ),
        "science_checkpoint_identity_schema": (
            "eiel_cuda_full_evolution_science_checkpoint_identity_v1"
        ),
        "science_restore_gate_schema": (
            "eiel_cuda_full_evolution_science_restore_gate_v1"
        ),
        "science_route_binding_adapter_schema": (
            "eiel_cuda_full_evolution_science_route_binding_adapter_v1"
        ),
        "science_route_binding_wire_schema": "eiel_science_route_binding_v1",
        "science_observation_record_schema": "eiel_science_observation_record_v1",
        "science_observation_prefix_cursor_schema": (
            "eiel_science_observation_prefix_cursor_v1"
        ),
        "continuation_adapter_version": (
            "eiel_cuda_full_evolution_science_checkpoint_continuation_adapter_v1"
        ),
        "observation_adapter_version": (
            "eiel_cuda_full_evolution_science_observation_adapter_v1"
        ),
        "runtime_observation_mode": "OFF",
        "native_capture_runtime_integrated": False,
        "science_route_qualified": False,
    }
    science = {**science_body, "binding_sha256": canonical_sha256(science_body).hex}
    observation = {
        "mode": "OFF",
        "science_records_emitted": 0,
        "science_observation_schedule": None,
        "population_change_detection": False,
    }
    identity_body = {
        "schema_version": "eiel_workhorse_science_compatible_base_checkpoint_identity_v2",
        "time_authority": "absolute_tick_v1",
        "shape_id": "workhorse-B1-fixture",
        "layout_sha256": "7" * 64,
        "semantic_transition_sha256": "8" * 64,
        "source_artifact_identity_sha256": artifact_sha,
        "checkpoint_at_tick": 22,
        "B": 1,
        "world_uids": [97000],
        "rng_keys": [53],
        "state_bytes": len(state_bytes),
        "state_sha256": state_sha,
        "last_acknowledged_event_frontier": ack,
        "science_checkpoint_compatibility": science,
        "observation": observation,
        "qualification_authority_granted": False,
        "science_authority_granted": False,
    }
    identity = {
        **identity_body,
        "base_checkpoint_identity_sha256": canonical_sha256(identity_body).hex,
    }
    manifest_body = {
        "schema_version": "eiel_workhorse_semantic_case_raw_fragment_v2",
        "claim_class": "TECHNICAL_ONLY_RAW_EVIDENCE",
        "qualification_authority_granted": False,
        "science_authority_granted": False,
        "observation": observation,
        "case_id": "checkpoint_same_restore_sequence",
        "base_checkpoint": {
            "identity": identity,
            "state_file_name": state_path.name,
            "state_bytes": len(state_bytes),
            "state_sha256": state_sha,
        },
    }
    manifest = {
        **manifest_body,
        "fragment_body_sha256": canonical_sha256(manifest_body).hex,
    }
    manifest_path = source_root / "semantic_case_fragment.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    manifest_path.chmod(0o444)
    source_root.chmod(0o555)
    return {
        "manifest_path": manifest_path,
        "state_path": state_path,
        "state_bytes": state_bytes,
        "ack_path": ack_path,
        "science_path": science_path,
    }


def _materialize(source: dict[str, Any], root: Path) -> tuple[Any, Any]:
    return materialize_workhorse_semantic_checkpoint_occurrence_v2(
        source["manifest_path"],
        source["science_path"],
        root,
        checkpoint_id=CheckpointId("workhorse-semantic-checkpoint-22"),
        run_id=RunId("RUN-WORKHORSE-SOURCE"),
        attempt_id=AttemptId("ATT-WORKHORSE-SOURCE-1"),
        transaction_id="workhorse-semantic-case-20",
        absolute_next_tick=22,
        semantic_spec_digest=canonical_sha256({"semantic": "fixture"}),
        implementation_digest=canonical_sha256({"implementation": "fixture"}),
        committed_at_utc="2026-08-26T12:00:00Z",
    )


def test_workhorse_occurrence_round_trip_and_restore_are_exact(tmp_path: Path) -> None:
    source = _source(tmp_path)
    root = tmp_path / "occurrence"

    descriptor, commit = _materialize(source, root)
    assert WorkhorseSemanticCheckpointOccurrenceV2.from_dict(
        descriptor.to_dict()
    ) == descriptor
    assert WorkhorseSemanticCheckpointOccurrenceCommitV2.from_dict(
        commit.to_dict()
    ) == commit
    inspection = inspect_workhorse_semantic_checkpoint_occurrence_v2(root)
    assert inspection.status is CheckpointOccurrenceInspectionStatusV1.VISIBLE
    assert inspection.occurrence_digest == descriptor.occurrence_digest()
    restored = restore_workhorse_semantic_checkpoint_occurrence_v2(
        root,
        expected_run_id=descriptor.run_id,
        expected_attempt_id=descriptor.attempt_id,
        expected_absolute_next_tick=22,
        expected_semantic_spec_digest=descriptor.semantic_spec_digest,
        expected_implementation_digest=descriptor.implementation_digest,
        expected_occurrence_digest=descriptor.occurrence_digest(),
    )
    assert restored["canonical_state"] == source["state_bytes"]
    assert root.stat().st_mode & 0o222 == 0
    assert all(path.stat().st_mode & 0o222 == 0 for path in root.iterdir())


@pytest.mark.parametrize("target", ["state_path", "ack_path", "science_path"])
def test_bound_source_mutation_quarantines_occurrence(
    tmp_path: Path, target: str
) -> None:
    source = _source(tmp_path)
    root = tmp_path / "occurrence"
    _materialize(source, root)
    path = source[target]
    path.chmod(0o644)
    path.write_bytes(path.read_bytes() + b"mutated")
    if target != "science_path":
        path.chmod(0o444)

    assert inspect_workhorse_semantic_checkpoint_occurrence_v2(root).status is (
        CheckpointOccurrenceInspectionStatusV1.QUARANTINED_CONFLICT
    )


def test_linked_or_writable_checkpoint_source_fails_before_commit(
    tmp_path: Path,
) -> None:
    linked = _source(tmp_path / "linked")
    os.link(linked["state_path"], tmp_path / "state-hardlink.bin")
    with pytest.raises(CheckpointOccurrenceError, match="unlinked immutable"):
        _materialize(linked, tmp_path / "linked-occurrence")
    assert not (tmp_path / "linked-occurrence").exists() or not list(
        (tmp_path / "linked-occurrence").glob("*commit*.json")
    )

    writable = _source(tmp_path / "writable")
    writable["manifest_path"].chmod(0o644)
    with pytest.raises(CheckpointOccurrenceError, match="unlinked immutable"):
        _materialize(writable, tmp_path / "writable-occurrence")


def test_source_mutation_during_commit_cannot_become_visible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _source(tmp_path)
    root = tmp_path / "occurrence"
    original = checkpoint_occurrence._atomic_create

    def mutate_before_commit(path: Path, payload: bytes) -> bool:
        if path.name == checkpoint_occurrence.WORKHORSE_SEMANTIC_OCCURRENCE_COMMIT_NAME:
            source["science_path"].write_bytes(b"changed-during-commit")
        return original(path, payload)

    monkeypatch.setattr(checkpoint_occurrence, "_atomic_create", mutate_before_commit)
    with pytest.raises(CheckpointOccurrenceError, match="did not become visible"):
        _materialize(source, root)
    assert inspect_workhorse_semantic_checkpoint_occurrence_v2(root).status is (
        CheckpointOccurrenceInspectionStatusV1.QUARANTINED_CONFLICT
    )
