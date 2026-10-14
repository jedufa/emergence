from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from tools.scripts import (
    eiel_cuda_architecture_b_patch_p2_sticky_failure_analyzer as analyzer,
)


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64


def _contract() -> dict[str, Any]:
    return analyzer.load_json(analyzer.DEFAULT_CONTRACT_PATH)


def _summary() -> dict[str, Any]:
    return {
        "requested_generations": 2,
        "cells": 1,
        "completed_row_count": 0,
        "validity": [[0], [0]],
        "status_codes": [[[0, 0]], [[0, 0]]],
        "integer_values": [[[0, 0]], [[0, 0]]],
        "float_values": [[[0.0, 0.0]], [[0.0, 0.0]]],
        "digest_values": [[[[0] * 32]], [[[0] * 32]]],
        "patch_optionality_mask": [[0], [0]],
    }


def _snapshot() -> dict[str, Any]:
    return {
        "failure_status": {
            "status": "failed",
            "failure_code": 1,
            "status_name": "event_capacity_exceeded",
            "stage": "birth_event_append",
            "cell": 0,
            "generation": 0,
            "tick": 0,
            "attempted_index": 8,
            "required_capacity": 9,
            "available_capacity": 8,
            "sequence": 1,
        },
        "progress": {
            "completed_generations": 0,
            "ticks_started": 1,
            "ticks_completed": 0,
            "last_started_generation": 0,
            "last_started_tick": 0,
            "event_capacity": 8,
            "event_cursor": 8,
            "event_rows_committed": 8,
            "rng_values_consumed": 17,
            "agent_id_allocations": 0,
            "mutation_commits": 0,
            "turnover_commits": 0,
        },
        "state_digests": {
            field: hashlib.sha256(field.encode("ascii")).hexdigest()
            for field in analyzer.STATE_DIGEST_FIELDS
        },
        "compact_summary": _summary(),
    }


def _control(*, terminal_digest: str, process: str) -> dict[str, Any]:
    return {
        "process_identity": process,
        "artifact_sha256": SHA_C,
        "runs": [
            {
                "dispatch_ordinal": ordinal,
                "status": "completed",
                "completed_generations": 2,
                "terminal_digest": terminal_digest,
            }
            for ordinal in (1, 2)
        ],
    }


def _replay(*, evidence_kind: str = "synthetic") -> dict[str, Any]:
    snapshot = _snapshot()
    return {
        "schema_version": analyzer.REPLAY_SCHEMA_VERSION,
        "evidence_kind": evidence_kind,
        "fixture_id": analyzer.FIXTURE_ID,
        "source_artifacts": [],
        "snapshots": {
            "at_latch": snapshot,
            "final": copy.deepcopy(snapshot),
        },
        "negative_controls": {
            "flat_g2_same_process_repeat": _control(
                terminal_digest=SHA_A, process="flat-process-123"
            ),
            "patch_p2_g2_same_process_repeat": _control(
                terminal_digest=SHA_B, process="patch-process-456"
            ),
        },
    }


def test_contract_only_validates_analyzer_without_claiming_live_fixture() -> None:
    result = analyzer.validate_analyzer_contract(_contract())

    assert result["analyzer_valid"] is True
    assert result["artifact_replay_performed"] is False
    assert result["replay_contract_satisfied"] is None
    assert result["live_fixture_status"] == "pending"
    assert result["qualification"] == "analyzer_valid_live_fixture_pending"
    assert result["cuda_dispatches_started"] == 0


def test_synthetic_replay_proves_analyzer_path_but_not_live_fixture() -> None:
    result = analyzer.validate_replay(_replay(), _contract())

    assert result["ok"] is True
    assert result["analyzer_valid"] is True
    assert result["replay_contract_satisfied"] is True
    assert result["evidence_kind"] == "synthetic"
    assert result["live_fixture_status"] == "pending"
    assert result["qualification"] == "analyzer_valid_live_fixture_pending"
    assert result["completed_generations"] == 0
    assert all(result["checks"].values())
    assert result["simulation_runs_started"] == 0
    assert result["cuda_builds_started"] == 0
    assert result["cuda_dispatches_started"] == 0


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("failure_code", 2, "failure_code"),
        ("stage", "turnover_commit", "stage"),
        ("generation", 1, "generation"),
        ("tick", 1, "tick"),
        ("sequence", 2, "sequence"),
        ("attempted_index", 7, "attempted_index"),
        ("required_capacity", 10, "required_capacity"),
    ),
)
def test_first_failure_metadata_must_match_exact_fixture(
    field: str, value: Any, message: str
) -> None:
    replay = _replay()
    replay["snapshots"]["at_latch"]["failure_status"][field] = value

    with pytest.raises(analyzer.ReplayValidationError, match=message):
        analyzer.validate_replay(replay, _contract())


@pytest.mark.parametrize(
    ("field", "increment", "message"),
    (
        ("ticks_started", 1, "ticks_started"),
        ("ticks_completed", 1, "ticks_completed"),
        ("completed_generations", 1, "completed_generations"),
        ("event_cursor", 1, "event_cursor"),
        ("event_rows_committed", 1, "event_rows_committed"),
        ("rng_values_consumed", 1, "changed after"),
        ("agent_id_allocations", 1, "changed after"),
        ("mutation_commits", 1, "mutation_commits"),
        ("turnover_commits", 1, "turnover_commits"),
    ),
)
def test_final_receipt_rejects_any_post_latch_progress(
    field: str, increment: int, message: str
) -> None:
    replay = _replay()
    replay["snapshots"]["final"]["progress"][field] += increment

    with pytest.raises(analyzer.ReplayValidationError, match=message):
        analyzer.validate_replay(replay, _contract())


def test_final_receipt_rejects_patch_or_agent_mutation() -> None:
    replay = _replay()
    replay["snapshots"]["final"]["state_digests"]["patch_state"] = SHA_A

    with pytest.raises(analyzer.ReplayValidationError, match="changed after"):
        analyzer.validate_replay(replay, _contract())


@pytest.mark.parametrize(
    ("plane", "value", "message"),
    (
        ("validity", 1, "no-op value"),
        ("status_codes", 4, "no-op value"),
        ("integer_values", 3, "no-op value"),
        ("float_values", 0, "float64 no-op"),
        ("digest_values", 255, "no-op value"),
        ("patch_optionality_mask", 1, "no-op value"),
    ),
)
def test_compact_summary_tail_must_remain_raw_invalid_noop(
    plane: str, value: Any, message: str
) -> None:
    replay = _replay()
    target = replay["snapshots"]["final"]["compact_summary"][plane]
    if plane in {"validity", "patch_optionality_mask"}:
        target[1][0] = value
    elif plane == "digest_values":
        target[1][0][0][0] = value
    else:
        target[1][0][0] = value

    with pytest.raises(analyzer.ReplayValidationError, match=message):
        analyzer.validate_replay(replay, _contract())


def test_required_same_process_controls_fail_closed_on_repeat_drift() -> None:
    replay = _replay()
    control = replay["negative_controls"]["patch_p2_g2_same_process_repeat"]
    control["runs"][1]["terminal_digest"] = SHA_A

    with pytest.raises(analyzer.ReplayValidationError, match="not exact"):
        analyzer.validate_replay(replay, _contract())


def test_live_replay_requires_and_verifies_bound_source_artifacts(
    tmp_path: Path,
) -> None:
    replay = _replay(evidence_kind="live")
    with pytest.raises(analyzer.ReplayValidationError, match="source artifact"):
        analyzer.validate_replay(replay, _contract(), base_directory=tmp_path)

    source = tmp_path / "live-receipt.bin"
    source.write_bytes(b"stored-live-fixture-receipt")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    replay["source_artifacts"] = [{"path": source.name, "sha256": digest}]
    for control in replay["negative_controls"].values():
        control["artifact_sha256"] = digest

    result = analyzer.validate_replay(
        replay, _contract(), base_directory=tmp_path
    )
    assert result["live_fixture_status"] == "validated"
    assert result["qualification"] == "qualified_by_supplied_live_artifacts"
    assert result["source_artifacts"] == [
        {"path": str(source.resolve()), "sha256": digest}
    ]

    replay["source_artifacts"][0]["sha256"] = SHA_A
    with pytest.raises(analyzer.ReplayValidationError, match="digest mismatch"):
        analyzer.validate_replay(replay, _contract(), base_directory=tmp_path)


def test_malformed_receipt_fails_closed_with_typed_validation_error() -> None:
    replay = _replay()
    replay["evidence_kind"] = {"unexpected": "object"}

    with pytest.raises(analyzer.ReplayValidationError, match="evidence_kind"):
        analyzer.validate_replay(replay, _contract())


def test_cli_replays_stored_synthetic_artifact_without_cuda(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    artifact = tmp_path / "synthetic-replay.json"
    artifact.write_text(json.dumps(_replay()), encoding="utf-8")

    assert analyzer.main(["--artifact", str(artifact)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["replay_contract_satisfied"] is True
    assert result["live_fixture_status"] == "pending"
    assert result["cuda_dispatches_started"] == 0
