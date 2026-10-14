from __future__ import annotations

import hashlib
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

import eiel_composition.cuda_full_evolution_v2_launchpad_adapter as launchpad_adapter
import engine.perf.cuda_full_evolution_v2.workhorse.authority as workhorse_authority
import engine.perf.cuda_full_evolution_v2.workhorse.canonical_rows as canonical_rows
from emergence_contracts import AttemptId, RunId, canonical_sha256
from engine.perf.cuda_full_evolution_v2.workhorse import science_demo
from engine.perf.cuda_full_evolution_v2.workhorse.contract import (
    FULL_EVOLUTION_CAPABILITIES_V2,
    workhorse_artifact_identity_v2,
    workhorse_shape_v2,
)


def _contract_fixture(monkeypatch: pytest.MonkeyPatch) -> tuple[dict, dict, dict, dict]:
    shape = workhorse_shape_v2(science_demo.SCIENCE_DEMO_SHAPE_ID_V1)
    artifact = {
        "path": "/tmp/workhorse-science-demo-test.so",
        "sha256": "1" * 64,
        "manifest_path": "/tmp/workhorse-science-demo-test.so.manifest.json",
        "manifest_file_sha256": "2" * 64,
        "manifest_sha256": "3" * 64,
        "sealed_artifact_identity_sha256": "4" * 64,
        "source_inventory_sha256": "5" * 64,
        "identity": workhorse_artifact_identity_v2(
            shape, science_demo.SCIENCE_DEMO_BACKEND_ID_V1
        ),
    }
    envelope_binding = {
        "path": "/tmp/workhorse-science-demo-envelope.json",
        "file_sha256": "6" * 64,
        "content_sha256": "7" * 64,
    }
    envelope = {
        "receipt_path": envelope_binding["path"],
        "receipt_file_sha256": envelope_binding["file_sha256"],
        "receipt_content_sha256": envelope_binding["content_sha256"],
        "artifact": artifact,
    }
    semantic = {
        "schema_version": science_demo.WORKHORSE_SCIENCE_SEMANTIC_INHERITANCE_SCHEMA_V1,
        "binding_sha256": "8" * 64,
    }
    science = {
        "schema_version": "eiel_workhorse_science_checkpoint_compatibility_binding_v2",
        "binding_sha256": "9" * 64,
    }
    monkeypatch.setattr(
        science_demo,
        "workhorse_artifact_binding_v2",
        lambda _path, **_kwargs: deepcopy(artifact),
    )
    monkeypatch.setattr(
        science_demo,
        "validate_workhorse_device_envelope_calibration_receipt_v2",
        lambda _path: deepcopy(envelope),
    )
    monkeypatch.setattr(
        science_demo,
        "validate_workhorse_science_semantic_inheritance_binding_v1",
        lambda value, **_kwargs: deepcopy(dict(value)),
    )
    monkeypatch.setattr(
        science_demo,
        "validate_workhorse_science_checkpoint_compatibility_binding_v2",
        lambda value: deepcopy(dict(value)),
    )
    return artifact, envelope_binding, semantic, science


def test_science_demo_hashes_json_projected_paths_as_paths(tmp_path: Path) -> None:
    checkpoint_manifest = tmp_path / "checkpoint.json"
    payload = b'{"checkpoint_at_tick":16}\n'
    checkpoint_manifest.write_bytes(payload)

    assert science_demo._projected_file_sha256(str(checkpoint_manifest)) == (
        hashlib.sha256(payload).hexdigest()
    )


def test_science_demo_contract_is_exact_absolute_tick_technical_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact, envelope, semantic, science = _contract_fixture(monkeypatch)
    contract = science_demo.workhorse_science_demo_contract_v1(
        artifact_binding=artifact,
        envelope_binding=envelope,
        semantic_inheritance_binding=semantic,
        science_checkpoint_compatibility_binding=science,
    )

    for field, value in (
        ("boundary_next_ticks", [0, 10, 20, 30, 40]),
        ("checkpoint_ticks", [20, 40]),
        ("science_authority_granted", True),
        ("runtime_observation_mode", "TYPED_SIDECAR_V1"),
    ):
        tampered = deepcopy(contract)
        tampered[field] = value
        with pytest.raises(
            science_demo.WorkhorseScienceDemoError,
            match="not canonical",
        ):
            science_demo.validate_workhorse_science_demo_contract_v1(
                tampered,
                artifact_binding=artifact,
                envelope_binding=envelope,
                semantic_inheritance_binding=semantic,
                science_checkpoint_compatibility_binding=science,
            )

    assert contract["operation"] == "SCIENCE_COMPATIBILITY_DEMO"
    assert contract["time_authority"] == "absolute_tick_v1"
    assert contract["B"] == 10
    assert contract["temporal_k"] == 1
    assert contract["boundary_next_ticks"] == [0, 8, 16, 24, 32, 40]
    assert contract["checkpoint_ticks"] == [16, 32, 40]
    assert contract["continuation_restore_ticks"] == [16, 32]
    assert contract["terminal_restore_validation_tick"] == 40
    assert contract["full_evolution_components"] == sorted(
        FULL_EVOLUTION_CAPABILITIES_V2
    )
    assert len(contract["world_mapping"]["rows"]) == 10
    assert len(contract["static_control_table"]["rows"]) == 10
    assert contract["runtime_observation_mode"] == "OFF"
    assert contract["runtime_control_adapter_integrated"] is False
    assert contract["native_capture_runtime_integrated"] is False
    assert contract["science_continuable"] is False
    assert contract["science_authority_granted"] is False
    assert contract["qualification_authority_granted"] is False
    assert contract["artifact_lifecycle_profile"] == "SCIENCE_COMPACT"
    assert contract["raw_tick_or_event_payloads_retained"] is False

    assert (
        science_demo.validate_workhorse_science_demo_contract_v1(
            contract,
            artifact_binding=artifact,
            envelope_binding=envelope,
            semantic_inheritance_binding=semantic,
            science_checkpoint_compatibility_binding=science,
        )
        == contract
    )


def test_science_checkpoint_uses_canonical_absolute_tick_occurrence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = bytes(range(32))
    state_sha256 = hashlib.sha256(state).hexdigest()
    contract = {
        "shape_id": science_demo.SCIENCE_DEMO_SHAPE_ID_V1,
        "backend_id": science_demo.SCIENCE_DEMO_BACKEND_ID_V1,
        "B": 2,
        "temporal_k": 1,
        "total_ticks": 40,
        "start_next_tick": 0,
    }
    parent = {
        "operation": "SCIENCE_COMPATIBILITY_DEMO",
        "source_batch_id": "batch-0000",
        "contract": contract,
        "artifact_sha256": "1" * 64,
        "sealed_artifact_identity_sha256": "2" * 64,
        "device_identity": {"uuid": "GPU-fixture"},
        "attempt_authority_path": str(tmp_path / "attempt-authority.json"),
        "operation_contract_sha256": "3" * 64,
        "binding_sha256": "4" * 64,
    }
    checkpoint_path = tmp_path / "workhorse-checkpoint" / "checkpoint.json"
    checkpoint_path.parent.mkdir()
    checkpoint_path.write_bytes(b"workhorse-checkpoint-fixture")
    checkpoint_path.chmod(0o444)
    state_path = checkpoint_path.parent / "canonical_state.bin"
    state_path.write_bytes(state)
    state_path.chmod(0o444)
    checkpoint = {
        "contract": contract,
        "checkpoint_at_tick": 16,
        "checkpoint_identity": {"checkpoint_identity_sha256": "5" * 64},
        "canonical_state": {
            "sha256": state_sha256,
            "bytes": len(state),
            "state_bytes_per_world": 16,
        },
        "world_uids": [11, 12],
        "rng_keys": [21, 22],
        "rng_epoch": 4,
        "attempt_authority_binding": parent,
        "artifact": {
            "path": "/immutable/workhorse.so",
            "sha256": parent["artifact_sha256"],
            "manifest_sha256": "6" * 64,
            "sealed_artifact_identity_sha256": parent[
                "sealed_artifact_identity_sha256"
            ],
        },
        "event_sink_ack": {"receipt_body_sha256": "7" * 64},
        "bundle_body_sha256": "8" * 64,
    }
    monkeypatch.setattr(
        canonical_rows,
        "validate_workhorse_canonical_parent_authority_binding_v2",
        lambda value, **_kwargs: deepcopy(dict(value)),
    )
    monkeypatch.setattr(
        workhorse_authority,
        "inspect_workhorse_checkpoint_bundle_v2",
        lambda _path, **_kwargs: (deepcopy(checkpoint), state_path),
    )
    authority = SimpleNamespace(
        run_occurrence=SimpleNamespace(run_id=RunId("RUN-SCIENCE-FIXTURE")),
        attempt_start=SimpleNamespace(attempt_id=AttemptId("ATT-SCIENCE-FIXTURE-1")),
        semantic_run_spec=SimpleNamespace(
            semantic_spec_digest=lambda: canonical_sha256({"semantic": "fixture"})
        ),
        execution_implementation=SimpleNamespace(
            implementation_digest=lambda: canonical_sha256(
                {"implementation": "fixture"}
            )
        ),
    )
    monkeypatch.setattr(
        launchpad_adapter.CanonicalCudaFullEvolutionV2LaunchpadAdapterV1,
        "_authority",
        staticmethod(lambda _path: authority),
    )
    adapter = object.__new__(
        launchpad_adapter.CanonicalCudaFullEvolutionV2LaunchpadAdapterV1
    )
    result = adapter.materialize_workhorse_science_checkpoint_occurrence_v1(
        canonical_parent_authority_binding=parent,
        checkpoint_manifest_path=checkpoint_path,
        absolute_bundle_root=(tmp_path / "absolute" / "checkpoint-tick-000000000016"),
        occurrence_root=(tmp_path / "occurrences" / "checkpoint-tick-000000000016"),
        parent_checkpoint_digest=None,
    )

    assert result["canonical_state"] == state
    projection = result["projection"]
    assert projection["checkpoint_at_tick"] == 16
    assert projection["canonical_visible"] is True
    assert projection["restore_admitted_exact"] is True
    assert projection["parent_checkpoint_digest"] is None
    restored = adapter.restore_workhorse_science_checkpoint_occurrence_v1(
        canonical_parent_authority_binding=parent,
        checkpoint_manifest_path=checkpoint_path,
        projection=projection,
    )
    assert restored["canonical_state_sha256"] == state_sha256

    tampered = deepcopy(projection)
    tampered["science_authority_granted"] = True
    with pytest.raises(
        launchpad_adapter.CudaFullEvolutionV2LaunchpadAdapterError,
        match="projection identity changed",
    ):
        adapter.restore_workhorse_science_checkpoint_occurrence_v1(
            canonical_parent_authority_binding=parent,
            checkpoint_manifest_path=checkpoint_path,
            projection=tampered,
        )


def test_row_checkpoint_occurrence_streams_without_full_state_materialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = bytes(range(64))
    state_sha256 = hashlib.sha256(state).hexdigest()
    contract = {
        "shape_id": science_demo.SCIENCE_DEMO_SHAPE_ID_V1,
        "backend_id": science_demo.SCIENCE_DEMO_BACKEND_ID_V1,
        "B": 2,
        "temporal_k": 1,
        "total_ticks": 40,
        "start_next_tick": 0,
    }
    parent = {
        "operation": "EXECUTION_ROW",
        "source_batch_id": "batch-0000",
        "contract": contract,
        "artifact_sha256": "1" * 64,
        "sealed_artifact_identity_sha256": "2" * 64,
        "device_identity": {"uuid": "GPU-fixture"},
        "attempt_authority_path": str(tmp_path / "attempt-authority.json"),
        "operation_contract_sha256": "3" * 64,
        "binding_sha256": "4" * 64,
    }
    checkpoint_path = tmp_path / "workhorse-checkpoint" / "checkpoint.json"
    checkpoint_path.parent.mkdir()
    checkpoint_path.write_bytes(b"workhorse-checkpoint-fixture")
    checkpoint_path.chmod(0o444)
    state_path = checkpoint_path.parent / "canonical_state.bin"
    state_path.write_bytes(state)
    state_path.chmod(0o444)
    checkpoint = {
        "contract": contract,
        "checkpoint_at_tick": 40,
        "checkpoint_identity": {"checkpoint_identity_sha256": "5" * 64},
        "canonical_state": {
            "sha256": state_sha256,
            "bytes": len(state),
            "state_bytes_per_world": 32,
        },
        "world_uids": [11, 12],
        "rng_keys": [21, 22],
        "rng_epoch": 4,
        "attempt_authority_binding": parent,
        "artifact": {
            "path": "/immutable/workhorse.so",
            "sha256": parent["artifact_sha256"],
            "manifest_sha256": "6" * 64,
            "sealed_artifact_identity_sha256": parent[
                "sealed_artifact_identity_sha256"
            ],
        },
        "event_sink_ack": {"receipt_body_sha256": "7" * 64},
        "bundle_body_sha256": "8" * 64,
    }
    monkeypatch.setattr(
        canonical_rows,
        "validate_workhorse_canonical_parent_authority_binding_v2",
        lambda value, **_kwargs: deepcopy(dict(value)),
    )
    monkeypatch.setattr(
        workhorse_authority,
        "inspect_workhorse_checkpoint_bundle_v2",
        lambda _path, **_kwargs: (deepcopy(checkpoint), state_path),
    )
    monkeypatch.setattr(
        workhorse_authority,
        "reopen_workhorse_checkpoint_bundle_v2",
        lambda *_args, **_kwargs: pytest.fail(
            "row occurrence materialized the complete Workhorse state"
        ),
    )
    authority = SimpleNamespace(
        run_occurrence=SimpleNamespace(run_id=RunId("RUN-ROW-FIXTURE")),
        attempt_start=SimpleNamespace(attempt_id=AttemptId("ATT-ROW-FIXTURE-1")),
        semantic_run_spec=SimpleNamespace(
            semantic_spec_digest=lambda: canonical_sha256({"semantic": "fixture"})
        ),
        execution_implementation=SimpleNamespace(
            implementation_digest=lambda: canonical_sha256(
                {"implementation": "fixture"}
            )
        ),
    )
    monkeypatch.setattr(
        launchpad_adapter.CanonicalCudaFullEvolutionV2LaunchpadAdapterV1,
        "_authority",
        staticmethod(lambda _path: authority),
    )
    adapter = object.__new__(
        launchpad_adapter.CanonicalCudaFullEvolutionV2LaunchpadAdapterV1
    )
    absolute_root = tmp_path / "absolute" / "checkpoint-tick-000000000040"
    occurrence_root = tmp_path / "occurrences" / "checkpoint-tick-000000000040"
    result = adapter.materialize_workhorse_row_checkpoint_occurrence_v2(
        canonical_parent_authority_binding=parent,
        checkpoint_manifest_path=checkpoint_path,
        absolute_bundle_root=absolute_root,
        occurrence_root=occurrence_root,
    )

    assert result["canonical_state"] is None
    assert result["canonical_state_nbytes"] == len(state)
    assert result["canonical_state_sha256"] == state_sha256
    projection = result["projection"]
    assert projection["schema_version"] == (
        "eiel_workhorse_row_checkpoint_occurrence_projection_v2"
    )
    assert projection["canonical_visible"] is True
    assert projection["restore_admitted_exact"] is False
    projected_state = absolute_root / "canonical_state.bin"
    assert projected_state.read_bytes() == state
    assert projected_state.stat().st_ino != state_path.stat().st_ino
    assert projected_state.stat().st_nlink == state_path.stat().st_nlink == 1
