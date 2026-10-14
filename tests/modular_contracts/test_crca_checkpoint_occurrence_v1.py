from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from engine.campaigns.providers import ProductionEngineLifecycleV1
from eiel_composition import (
    ABSOLUTE_TICK_OCCURRENCE_COMMIT_NAME,
    ABSOLUTE_TICK_OCCURRENCE_DESCRIPTOR_NAME,
    OCCURRENCE_COMMIT_NAME,
    OCCURRENCE_DESCRIPTOR_NAME,
    CheckpointOccurrenceError,
    absolute_tick_checkpoint_bundle_file_participant_bindings_v1,
    build_checkpoint_conformance_index_v1,
    checkpoint_bundle_file_participant_bindings_v1,
    checkpoint_bundle_tree_digest_v1,
    checkpoint_absolute_tick_bundle_tree_digest_v1,
    describe_absolute_tick_checkpoint_occurrence_v1,
    describe_checkpoint_occurrence_v1,
    inspect_checkpoint_occurrence_v1,
    inspect_absolute_tick_checkpoint_occurrence_v1,
    materialize_checkpoint_occurrence_v1,
    materialize_absolute_tick_checkpoint_occurrence_v1,
    restore_checkpoint_occurrence_v1,
    restore_absolute_tick_checkpoint_occurrence_v1,
    run_checkpoint_conformance_v1,
)
from emergence_contracts import (
    ABSOLUTE_TICK_CHECKPOINT_FORMAT_ID,
    ABSOLUTE_TICK_CHECKPOINT_TIME_AUTHORITY,
    AttemptId,
    CheckpointConformanceIndexV1,
    CheckpointConformanceLayerV1,
    CheckpointConformanceStatusV1,
    CheckpointId,
    CheckpointOccurrenceInspectionStatusV1,
    CheckpointOccurrenceV1,
    ContentDigest,
    RunId,
    canonical_json_bytes,
    canonical_sha256,
)
from simulation.checkpoint_bundle_v2 import (
    _checkpoint_bundle_tree_sha256,
    checkpoint_bundle_v2_load,
)
from tests.perf.test_eiel_architecture_b_checkpoint_projection_save import (
    _projection,
    _source_digest,
    checkpoint_bundle_v2_save_projection,
)


def _digest(label: str) -> ContentDigest:
    import hashlib

    return ContentDigest(
        algorithm="sha256",
        hex=hashlib.sha256(label.encode("utf-8")).hexdigest(),
    )


@pytest.fixture()
def sealed_bundle(tmp_path: Path) -> Path:
    generation = 3
    absolute_next_tick = 480
    projection = _projection(generation=generation, tick=0)
    source_digest = _source_digest(projection)
    controller_id = "crca-g6-checkpoint-owner"
    identity = {
        "run_id": "run-crca-g6-parity",
        "controller_id": controller_id,
        "owner_epoch": absolute_next_tick,
        "snapshot_id": (
            f"{controller_id}:owner:{absolute_next_tick}:boundary:"
            f"{absolute_next_tick}"
        ),
        "snapshot_sequence": absolute_next_tick,
        "snapshot_digest_sha256": source_digest,
        "next_generation": generation,
        "tick": 0,
        "phase": "generation_boundary",
        "source_projection_sha256": source_digest,
    }
    continuation = ProductionEngineLifecycleV1._observer_continuation(
        generation,
        tick=0,
        event_cursor=0,
        snapshot_sequence=absolute_next_tick,
        absolute_tick_authority=True,
    )
    bundle = tmp_path / "source" / "sealed-at-tick-480"
    checkpoint_bundle_v2_save_projection(
        projection,
        str(bundle),
        generation_index=generation,
        expected_source_projection_sha256=source_digest,
        architecture_b_identity=identity,
        observer_continuation=continuation,
    )
    checkpoint_bundle_v2_load(str(bundle), strict=True)
    return bundle


def _arguments(bundle: Path) -> dict[str, object]:
    metadata = json.loads((bundle / "metadata.json").read_text(encoding="utf-8"))
    owner_digest = _digest("simulation.checkpoint_bundle_v2@frozen-owner")
    return {
        "checkpoint_id": CheckpointId(metadata["checkpoint_id"]),
        "run_id": RunId("run-crca-g6-parity"),
        "attempt_id": AttemptId("attempt-crca-g6-parity-1"),
        "transaction_id": "transaction-at-absolute-tick-480",
        "absolute_next_tick": 480,
        "semantic_spec_digest": _digest("semantic-spec"),
        "implementation_digest": _digest("implementation"),
        "participant_bindings": checkpoint_bundle_file_participant_bindings_v1(
            bundle,
            owner_id="simulation.checkpoint_bundle_v2.integrated-owner",
            owner_digest=owner_digest,
        ),
        "parent_checkpoint_digest": None,
    }


def _tree_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _rewrite_descriptor_and_commit(
    occurrence_root: Path,
    descriptor_value: dict[str, object],
) -> None:
    descriptor = CheckpointOccurrenceV1.from_dict(descriptor_value)
    descriptor_path = occurrence_root / OCCURRENCE_DESCRIPTOR_NAME
    commit_path = occurrence_root / OCCURRENCE_COMMIT_NAME
    commit_value = json.loads(commit_path.read_text(encoding="utf-8"))
    commit_value["occurrence_digest"] = descriptor.occurrence_digest().to_dict()
    descriptor_path.write_bytes(canonical_json_bytes(descriptor.to_dict()))
    commit_path.write_bytes(canonical_json_bytes(commit_value))


def _assert_semantically_equal(left: object, right: object) -> None:
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0, equal_nan=True)
        return
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        np.testing.assert_array_equal(left, right, strict=True)
        return
    if isinstance(left, dict) and isinstance(right, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_semantically_equal(left[key], right[key])
        return
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        assert type(left) is type(right)
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right, strict=True):
            _assert_semantically_equal(left_item, right_item)
        return
    assert left == right


def test_occurrence_wrapper_preserves_exact_bytes_and_loader_semantics(
    sealed_bundle: Path,
    tmp_path: Path,
) -> None:
    occurrence_root = tmp_path / "occurrences" / "checkpoint-at-tick-480"
    arguments = _arguments(sealed_bundle)
    bytes_before = _tree_bytes(sealed_bundle)
    direct = checkpoint_bundle_v2_load(str(sealed_bundle), strict=True)

    descriptor, commit = materialize_checkpoint_occurrence_v1(
        sealed_bundle,
        occurrence_root,
        committed_at_utc="2026-08-23T07:00:00Z",
        **arguments,
    )
    wrapped = restore_checkpoint_occurrence_v1(
        sealed_bundle,
        occurrence_root,
        expected_run_id=arguments["run_id"],
        expected_attempt_id=arguments["attempt_id"],
        expected_absolute_next_tick=480,
        expected_semantic_spec_digest=arguments["semantic_spec_digest"],
        expected_implementation_digest=arguments["implementation_digest"],
        expected_occurrence_digest=descriptor.occurrence_digest(),
    )

    assert _tree_bytes(sealed_bundle) == bytes_before
    _assert_semantically_equal(wrapped, direct)
    assert descriptor.bundle_v2_digest.hex == _checkpoint_bundle_tree_sha256(
        str(sealed_bundle)
    )
    assert checkpoint_bundle_tree_digest_v1(sealed_bundle) == (
        descriptor.bundle_v2_digest
    )
    assert commit.occurrence_digest == descriptor.occurrence_digest()
    # Legacy BundleV2 identity is a frozen CRCA seam.  Native absolute-tick
    # support must remain strictly additive and cannot perturb these digests.
    assert descriptor.bundle_v2_digest.hex == (
        "8583fab6a9068810b394d567ff6ff071bb1d4a20edb117b922212d12ced8dd53"
    )
    assert descriptor.occurrence_digest().hex == (
        "814842ae1a3b8880a8a3166fc2add2f9217624c49c3f4317358786ba9e23a734"
    )
    assert commit.commit_digest().hex == (
        "802c28befe48648e929711bd5c0666efa78e12606fd17a88a4c41ffd5b22a560"
    )
    assert all(
        participant.absolute_next_tick == 480
        and participant.run_id == arguments["run_id"]
        and participant.attempt_id == arguments["attempt_id"]
        and participant.transaction_id == arguments["transaction_id"]
        for participant in descriptor.participant_occurrences
    )
    assert "generation" not in json.dumps(descriptor.to_dict()).lower()
    assert (occurrence_root / OCCURRENCE_DESCRIPTOR_NAME).is_file()
    assert (occurrence_root / OCCURRENCE_COMMIT_NAME).is_file()


def test_occurrence_publication_is_concurrent_and_idempotent(
    sealed_bundle: Path,
    tmp_path: Path,
) -> None:
    occurrence_root = tmp_path / "racing-occurrence"
    arguments = _arguments(sealed_bundle)

    def publish() -> tuple[str, str]:
        descriptor, commit = materialize_checkpoint_occurrence_v1(
            sealed_bundle,
            occurrence_root,
            committed_at_utc="2026-08-23T07:01:00Z",
            **arguments,
        )
        return descriptor.occurrence_digest().hex, commit.commit_digest().hex

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(lambda _index: publish(), range(8)))
    assert len(set(results)) == 1
    assert inspect_checkpoint_occurrence_v1(
        sealed_bundle, occurrence_root
    ).status is CheckpointOccurrenceInspectionStatusV1.VISIBLE


def test_occurrence_crash_windows_fail_closed_and_recover_exact_descriptor(
    sealed_bundle: Path,
    tmp_path: Path,
) -> None:
    arguments = _arguments(sealed_bundle)
    orphan_root = tmp_path / "orphan"
    assert inspect_checkpoint_occurrence_v1(
        sealed_bundle, orphan_root
    ).status is CheckpointOccurrenceInspectionStatusV1.ORPHAN_SEALED_BUNDLE
    with pytest.raises(CheckpointOccurrenceError, match="not restorable"):
        restore_checkpoint_occurrence_v1(
            sealed_bundle,
            orphan_root,
            expected_run_id=arguments["run_id"],
            expected_attempt_id=arguments["attempt_id"],
            expected_absolute_next_tick=480,
            expected_semantic_spec_digest=arguments["semantic_spec_digest"],
            expected_implementation_digest=arguments["implementation_digest"],
            expected_occurrence_digest=_digest("orphan-has-no-admitted-occurrence"),
        )

    partial_root = tmp_path / "descriptor-only"
    partial_root.mkdir()
    descriptor = describe_checkpoint_occurrence_v1(sealed_bundle, **arguments)
    (partial_root / OCCURRENCE_DESCRIPTOR_NAME).write_bytes(
        canonical_json_bytes(descriptor.to_dict())
    )
    assert inspect_checkpoint_occurrence_v1(
        sealed_bundle, partial_root
    ).status is CheckpointOccurrenceInspectionStatusV1.QUARANTINED_PARTIAL_OCCURRENCE
    recovered, _commit = materialize_checkpoint_occurrence_v1(
        sealed_bundle,
        partial_root,
        committed_at_utc="2026-08-23T07:02:00Z",
        **arguments,
    )
    assert recovered == descriptor
    assert inspect_checkpoint_occurrence_v1(
        sealed_bundle, partial_root
    ).status is CheckpointOccurrenceInspectionStatusV1.VISIBLE
    with pytest.raises(CheckpointOccurrenceError, match="restore admission"):
        restore_checkpoint_occurrence_v1(
            sealed_bundle,
            partial_root,
            expected_run_id=arguments["run_id"],
            expected_attempt_id=arguments["attempt_id"],
            expected_absolute_next_tick=480,
            expected_semantic_spec_digest=arguments["semantic_spec_digest"],
            expected_implementation_digest=arguments["implementation_digest"],
            expected_occurrence_digest=_digest("unadmitted-occurrence"),
        )

    incomplete_bundle = tmp_path / "incomplete-bundle"
    shutil.copytree(sealed_bundle, incomplete_bundle)
    (incomplete_bundle / "metrics_state.json").unlink()
    assert inspect_checkpoint_occurrence_v1(
        incomplete_bundle, tmp_path / "incomplete-occurrence"
    ).status is (
        CheckpointOccurrenceInspectionStatusV1.QUARANTINED_INCOMPLETE_BUNDLE
    )

    conflict_root = tmp_path / "conflict"
    materialize_checkpoint_occurrence_v1(
        sealed_bundle,
        conflict_root,
        committed_at_utc="2026-08-23T07:03:00Z",
        **arguments,
    )
    descriptor_path = conflict_root / OCCURRENCE_DESCRIPTOR_NAME
    tampered = json.loads(descriptor_path.read_text(encoding="utf-8"))
    tampered["run_id"] = "different-run"
    descriptor_path.write_bytes(canonical_json_bytes(tampered))
    assert inspect_checkpoint_occurrence_v1(
        sealed_bundle, conflict_root
    ).status is CheckpointOccurrenceInspectionStatusV1.QUARANTINED_CONFLICT


def test_occurrence_root_cannot_resolve_inside_bundle(
    sealed_bundle: Path,
    tmp_path: Path,
) -> None:
    bundle_link = tmp_path / "bundle-link"
    bundle_link.symlink_to(sealed_bundle, target_is_directory=True)
    with pytest.raises(CheckpointOccurrenceError, match="outside CheckpointBundleV2"):
        materialize_checkpoint_occurrence_v1(
            sealed_bundle,
            bundle_link / "occurrence",
            committed_at_utc="2026-08-23T07:03:30Z",
            **_arguments(sealed_bundle),
        )


def test_bundle_root_symlink_and_unknown_occurrence_sidecars_fail_closed(
    sealed_bundle: Path,
    tmp_path: Path,
) -> None:
    bundle_link = tmp_path / "symlinked-bundle-root"
    bundle_link.symlink_to(sealed_bundle, target_is_directory=True)
    with pytest.raises(CheckpointOccurrenceError, match="symbolic link"):
        checkpoint_bundle_tree_digest_v1(bundle_link)
    assert inspect_checkpoint_occurrence_v1(
        bundle_link, tmp_path / "symlink-occurrence"
    ).status is (
        CheckpointOccurrenceInspectionStatusV1.QUARANTINED_INCOMPLETE_BUNDLE
    )
    with pytest.raises(CheckpointOccurrenceError, match="symbolic link"):
        materialize_checkpoint_occurrence_v1(
            bundle_link,
            tmp_path / "symlink-occurrence",
            committed_at_utc="2026-08-23T07:03:45Z",
            **_arguments(sealed_bundle),
        )

    occurrence_root = tmp_path / "unknown-sidecar"
    materialize_checkpoint_occurrence_v1(
        sealed_bundle,
        occurrence_root,
        committed_at_utc="2026-08-23T07:03:50Z",
        **_arguments(sealed_bundle),
    )
    (occurrence_root / ".checkpoint_occurrence_v1.json.crash.tmp").write_bytes(
        b"partial"
    )
    assert inspect_checkpoint_occurrence_v1(
        sealed_bundle, occurrence_root
    ).status is CheckpointOccurrenceInspectionStatusV1.QUARANTINED_CONFLICT
    with pytest.raises(CheckpointOccurrenceError, match="unbound entries"):
        materialize_checkpoint_occurrence_v1(
            sealed_bundle,
            occurrence_root,
            committed_at_utc="2026-08-23T07:03:50Z",
            **_arguments(sealed_bundle),
        )


@pytest.mark.parametrize("tamper", ["absolute_tick", "participant_state"])
def test_fully_rehashed_occurrence_tamper_fails_internal_validation(
    sealed_bundle: Path,
    tmp_path: Path,
    tamper: str,
) -> None:
    occurrence_root = tmp_path / f"rehashed-{tamper}"
    materialize_checkpoint_occurrence_v1(
        sealed_bundle,
        occurrence_root,
        committed_at_utc="2026-08-23T07:03:55Z",
        **_arguments(sealed_bundle),
    )
    descriptor_path = occurrence_root / OCCURRENCE_DESCRIPTOR_NAME
    descriptor_value = json.loads(descriptor_path.read_text(encoding="utf-8"))
    participants = descriptor_value["participant_occurrences"]
    assert isinstance(participants, list)
    if tamper == "absolute_tick":
        descriptor_value["absolute_next_tick"] = 481
        for participant in participants:
            participant["absolute_next_tick"] = 481
    else:
        participants[0]["state_digest"] = _digest("fabricated-state").to_dict()
    _rewrite_descriptor_and_commit(occurrence_root, descriptor_value)
    assert inspect_checkpoint_occurrence_v1(
        sealed_bundle, occurrence_root
    ).status is CheckpointOccurrenceInspectionStatusV1.QUARANTINED_CONFLICT


def test_occurrence_rejects_incomplete_participant_coverage(
    sealed_bundle: Path,
) -> None:
    arguments = _arguments(sealed_bundle)
    arguments["participant_bindings"] = arguments["participant_bindings"][:-1]
    with pytest.raises(CheckpointOccurrenceError, match="exactly cover"):
        describe_checkpoint_occurrence_v1(sealed_bundle, **arguments)


def test_typed_conformance_index_runs_exact_existing_sentinels() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    index = build_checkpoint_conformance_index_v1(repo_root)
    assert {sentinel.layer for sentinel in index.sentinels} == set(
        CheckpointConformanceLayerV1
    )
    assert all(not sentinel.qualification_effect for sentinel in index.sentinels)

    run = run_checkpoint_conformance_v1(
        repo_root,
        index,
        observed_at_utc="2026-08-23T07:04:00Z",
    )
    assert run.status is CheckpointConformanceStatusV1.PASS
    assert not run.qualification_authority
    assert all(
        result.status is CheckpointConformanceStatusV1.PASS
        for result in run.results
    )


def test_typed_conformance_runner_fails_closed_on_source_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    index = build_checkpoint_conformance_index_v1(repo_root)
    first = replace(index.sentinels[0], owner_digest=_digest("drifted-owner"))
    drifted = CheckpointConformanceIndexV1(
        sentinels=(first, *index.sentinels[1:])
    )

    class Completed:
        returncode = 0
        stdout = "mocked technical sentinel pass"
        stderr = ""

    calls: list[list[str]] = []

    def run(command: list[str], **_kwargs: object) -> Completed:
        calls.append(copy.deepcopy(command))
        return Completed()

    monkeypatch.setattr("eiel_composition.checkpoint_occurrence.subprocess.run", run)
    conformance = run_checkpoint_conformance_v1(
        repo_root,
        drifted,
        observed_at_utc="2026-08-23T07:05:00Z",
    )
    assert conformance.status is CheckpointConformanceStatusV1.FAIL
    assert conformance.results[0].status is CheckpointConformanceStatusV1.FAIL
    assert len(calls) == 3
    assert not conformance.qualification_authority


def _native_bundle(tmp_path: Path, *, tick: int = 8) -> Path:
    bundle = tmp_path / "native" / f"checkpoint-tick-{tick:012d}"
    bundle.mkdir(parents=True)
    state = bytes(range(16))
    state_sha256 = hashlib.sha256(state).hexdigest()
    runtime_body = {
        "schema_version": ABSOLUTE_TICK_CHECKPOINT_FORMAT_ID,
        "time_authority": ABSOLUTE_TICK_CHECKPOINT_TIME_AUTHORITY,
        "contract": {
            "profile_id": "eiel_full_evolution_cuda_native_v2",
            "B": 1,
            "total_ticks": tick,
            "start_next_tick": 0,
            "temporal_k": 4,
        },
        "canonical_state_shape": [1, 16],
        "canonical_state_nbytes": len(state),
        "canonical_state_sha256": state_sha256,
        "canonical_state_layout_sha256": canonical_sha256(
            {
                "dtype": "uint8",
                "shape": [1, 16],
                "nbytes": len(state),
                "content_sha256": state_sha256,
            }
        ).hex,
        "initial_next_absolute_tick": 0,
        "advanced_ticks": tick,
        "acknowledged_event_next_tick": tick,
        "last_sink_receipt_sha256": _digest("sink").hex,
        "world_uids": [7],
        "rng_keys": [11],
        "rng_semantic_epoch": 3,
        "artifact_sha256": _digest("native-artifact").hex,
        "manifest_sha256": _digest("native-manifest").hex,
        "device_identity": {"uuid": "GPU-test"},
        "population_fixture_id": None,
    }
    runtime_checkpoint = {
        **runtime_body,
        "metadata_seal_sha256": canonical_sha256(runtime_body).hex,
    }
    checkpoint_id = f"native-checkpoint-tick-{tick:012d}"
    metadata = {
        "schema_version": "eiel_absolute_tick_checkpoint_metadata_v1",
        "checkpoint_format_id": ABSOLUTE_TICK_CHECKPOINT_FORMAT_ID,
        "time_authority": ABSOLUTE_TICK_CHECKPOINT_TIME_AUTHORITY,
        "checkpoint_id": checkpoint_id,
        "absolute_next_tick": tick,
        "artifact_path": "/immutable/native/artifact.so",
        "runtime_checkpoint": runtime_checkpoint,
    }
    metadata_bytes = canonical_json_bytes(metadata)
    completion = {
        "schema_version": "eiel_absolute_tick_checkpoint_bundle_complete_v1",
        "status": "complete",
        "checkpoint_format_id": ABSOLUTE_TICK_CHECKPOINT_FORMAT_ID,
        "time_authority": ABSOLUTE_TICK_CHECKPOINT_TIME_AUTHORITY,
        "checkpoint_id": checkpoint_id,
        "absolute_next_tick": tick,
        "canonical_state_file": "canonical_state.bin",
        "canonical_state_nbytes": len(state),
        "canonical_state_sha256": state_sha256,
        "metadata_sha256": hashlib.sha256(metadata_bytes).hexdigest(),
    }
    (bundle / "canonical_state.bin").write_bytes(state)
    (bundle / "metadata.json").write_bytes(metadata_bytes)
    (bundle / "bundle_complete.json").write_bytes(canonical_json_bytes(completion))
    return bundle


def _native_arguments(bundle: Path, *, tick: int = 8) -> dict[str, object]:
    return {
        "checkpoint_id": CheckpointId(f"native-checkpoint-tick-{tick:012d}"),
        "run_id": RunId("run-native-absolute-tick"),
        "attempt_id": AttemptId("attempt-native-absolute-tick-1"),
        "transaction_id": f"transaction-tick-{tick:012d}",
        "absolute_next_tick": tick,
        "semantic_spec_digest": _digest("native-semantic-spec"),
        "implementation_digest": _digest("native-implementation"),
        "participant_bindings": (
            absolute_tick_checkpoint_bundle_file_participant_bindings_v1(
                bundle,
                owner_id="engine.cuda_full_evolution_v2_provider",
                owner_digest=_digest("native-provider-owner"),
            )
        ),
        "parent_checkpoint_digest": None,
    }


def test_absolute_tick_occurrence_is_visible_restorable_and_bundle_immutable(
    tmp_path: Path,
) -> None:
    bundle = _native_bundle(tmp_path)
    occurrence = tmp_path / "control" / "checkpoint-occurrences" / bundle.name
    arguments = _native_arguments(bundle)
    before = _tree_bytes(bundle)
    descriptor, commit = materialize_absolute_tick_checkpoint_occurrence_v1(
        bundle,
        occurrence,
        committed_at_utc="2026-08-26T07:00:00Z",
        **arguments,
    )
    restored = restore_absolute_tick_checkpoint_occurrence_v1(
        bundle,
        occurrence,
        expected_run_id=arguments["run_id"],
        expected_attempt_id=arguments["attempt_id"],
        expected_absolute_next_tick=8,
        expected_semantic_spec_digest=arguments["semantic_spec_digest"],
        expected_implementation_digest=arguments["implementation_digest"],
        expected_occurrence_digest=descriptor.occurrence_digest(),
    )
    assert restored["canonical_state"] == bytes(range(16))
    with pytest.raises(TypeError):
        restored["metadata"]["absolute_next_tick"] = 9
    assert _tree_bytes(bundle) == before
    assert descriptor.checkpoint_format_id == ABSOLUTE_TICK_CHECKPOINT_FORMAT_ID
    assert descriptor.time_authority == ABSOLUTE_TICK_CHECKPOINT_TIME_AUTHORITY
    assert commit.occurrence_digest == descriptor.occurrence_digest()
    assert checkpoint_absolute_tick_bundle_tree_digest_v1(bundle) == (
        descriptor.checkpoint_bundle_digest
    )
    assert (
        inspect_absolute_tick_checkpoint_occurrence_v1(bundle, occurrence).status
        is CheckpointOccurrenceInspectionStatusV1.VISIBLE
    )
    assert (occurrence / ABSOLUTE_TICK_OCCURRENCE_DESCRIPTOR_NAME).is_file()
    assert (occurrence / ABSOLUTE_TICK_OCCURRENCE_COMMIT_NAME).is_file()
    assert "generation" not in json.dumps(descriptor.to_dict()).lower()


def test_absolute_tick_occurrence_crash_windows_and_exact_coordinate_binding(
    tmp_path: Path,
) -> None:
    bundle = _native_bundle(tmp_path)
    arguments = _native_arguments(bundle)
    orphan = tmp_path / "orphan-native-occurrence"
    assert (
        inspect_absolute_tick_checkpoint_occurrence_v1(bundle, orphan).status
        is CheckpointOccurrenceInspectionStatusV1.ORPHAN_SEALED_BUNDLE
    )
    partial = tmp_path / "partial-native-occurrence"
    partial.mkdir()
    descriptor = describe_absolute_tick_checkpoint_occurrence_v1(bundle, **arguments)
    (partial / ABSOLUTE_TICK_OCCURRENCE_DESCRIPTOR_NAME).write_bytes(
        canonical_json_bytes(descriptor.to_dict())
    )
    assert (
        inspect_absolute_tick_checkpoint_occurrence_v1(bundle, partial).status
        is CheckpointOccurrenceInspectionStatusV1.QUARANTINED_PARTIAL_OCCURRENCE
    )
    occurrence = tmp_path / "bound-native-occurrence"
    materialize_absolute_tick_checkpoint_occurrence_v1(
        bundle,
        occurrence,
        committed_at_utc="2026-08-26T07:01:00Z",
        **arguments,
    )
    for field, value, match in (
        ("expected_run_id", RunId("wrong-run"), "Run binding"),
        ("expected_attempt_id", AttemptId("wrong-attempt"), "Attempt binding"),
        ("expected_absolute_next_tick", 9, "absolute next tick binding"),
    ):
        restore_arguments = {
            "expected_run_id": arguments["run_id"],
            "expected_attempt_id": arguments["attempt_id"],
            "expected_absolute_next_tick": 8,
            "expected_semantic_spec_digest": arguments["semantic_spec_digest"],
            "expected_implementation_digest": arguments["implementation_digest"],
            "expected_occurrence_digest": descriptor.occurrence_digest(),
        }
        restore_arguments[field] = value
        with pytest.raises(CheckpointOccurrenceError, match=match):
            restore_absolute_tick_checkpoint_occurrence_v1(
                bundle, occurrence, **restore_arguments
            )


@pytest.mark.parametrize(
    "fault",
    (
        "missing",
        "state-tamper",
        "metadata-tamper",
        "completion-tamper",
        "sidecar",
        "state-symlink",
        "state-hardlink",
        "generation-key",
    ),
)
def test_absolute_tick_bundle_fault_corpus_fails_closed(
    tmp_path: Path,
    fault: str,
) -> None:
    bundle = _native_bundle(tmp_path)
    if fault == "missing":
        (bundle / "canonical_state.bin").unlink()
    elif fault == "state-tamper":
        (bundle / "canonical_state.bin").write_bytes(b"tampered")
    elif fault == "metadata-tamper":
        (bundle / "metadata.json").write_bytes(b"{}")
    elif fault == "completion-tamper":
        (bundle / "bundle_complete.json").write_bytes(b"{}")
    elif fault == "sidecar":
        (bundle / "unbound.txt").write_text("unbound", encoding="utf-8")
    elif fault == "state-symlink":
        state = bundle / "canonical_state.bin"
        payload = state.read_bytes()
        state.unlink()
        target = tmp_path / "linked-state.bin"
        target.write_bytes(payload)
        state.symlink_to(target)
    elif fault == "state-hardlink":
        os.link(bundle / "canonical_state.bin", tmp_path / "second-state-link.bin")
    elif fault == "generation-key":
        metadata_path = bundle / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["runtime_checkpoint"]["generation"] = 1
        metadata_path.write_bytes(canonical_json_bytes(metadata))
    inspection = inspect_absolute_tick_checkpoint_occurrence_v1(
        bundle, tmp_path / "native-occurrence"
    )
    assert inspection.status is (
        CheckpointOccurrenceInspectionStatusV1.QUARANTINED_INCOMPLETE_BUNDLE
    )


@pytest.mark.parametrize(
    "fault", ("descriptor-symlink", "descriptor-hardlink", "sidecar")
)
def test_absolute_tick_occurrence_fault_corpus_fails_closed(
    tmp_path: Path,
    fault: str,
) -> None:
    bundle = _native_bundle(tmp_path)
    occurrence = tmp_path / "native-occurrence"
    materialize_absolute_tick_checkpoint_occurrence_v1(
        bundle,
        occurrence,
        committed_at_utc="2026-08-26T07:02:00Z",
        **_native_arguments(bundle),
    )
    descriptor = occurrence / ABSOLUTE_TICK_OCCURRENCE_DESCRIPTOR_NAME
    if fault == "descriptor-symlink":
        payload = descriptor.read_bytes()
        descriptor.unlink()
        target = tmp_path / "descriptor-target.json"
        target.write_bytes(payload)
        descriptor.symlink_to(target)
    elif fault == "descriptor-hardlink":
        os.link(descriptor, tmp_path / "descriptor-second-link.json")
    else:
        (occurrence / "unbound-sidecar.json").write_text("{}", encoding="utf-8")
    assert (
        inspect_absolute_tick_checkpoint_occurrence_v1(bundle, occurrence).status
        is CheckpointOccurrenceInspectionStatusV1.QUARANTINED_CONFLICT
    )
