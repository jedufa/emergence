from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pytest

from tools.qualification_review.engine import (
    RECEIPT_SCHEMA_PATH,
    SPEC_SCHEMA_PATH,
    review_spec,
)


EXAMPLE = (
    Path(__file__).resolve().parents[1]
    / "tools"
    / "qualification_review"
    / "examples"
)


def _utc(hour: int, minute: int = 0) -> datetime:
    return datetime(2026, 8, 14, hour, minute, tzinfo=timezone.utc)


def _copy_example(tmp_path: Path) -> Path:
    root = tmp_path / "review"
    shutil.copytree(EXAMPLE, root)
    return root / "review_spec.v1.json"


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def test_versioned_schemas_and_complete_example_pass() -> None:
    from jsonschema.validators import validator_for

    for path in (SPEC_SCHEMA_PATH, RECEIPT_SCHEMA_PATH):
        schema = _read(path)
        validator_for(schema).check_schema(schema)

    receipt = review_spec(EXAMPLE / "review_spec.v1.json", now=_utc(10))

    assert receipt["MECHANICAL_REVIEW"]["status"] == "PASS"
    assert receipt["MECHANICAL_REVIEW"]["counts"] == {
        "checks": 10,
        "passed": 10,
        "failed": 0,
        "warned": 0,
        "errors": 0,
        "warnings": 0,
    }
    semantic = receipt["SEMANTIC_REVIEW_REQUIRED"]
    assert semantic["status"] == "REQUIRED"
    assert [row["trigger_type"] for row in semantic["questions"]] == ["authority_change"]
    assert semantic["questions"][0]["evidence_pointers"] == ["evidence/packet.json"]
    assert receipt["freshness"] == {
        "reusable": True,
        "checked_at_utc": "2026-08-14T10:00:00Z",
        "valid_until_utc": "2026-08-14T11:00:00Z",
    }
    assert {row["path"] for row in receipt["input_hashes"]} == {
        "evidence/artifact.json",
        "evidence/expected_negative.json",
        "evidence/packet.json",
        "evidence/packet.schema.json",
        "evidence/process_snapshots.json",
    }


def test_semantic_questions_are_only_created_by_declared_risk_triggers(tmp_path: Path) -> None:
    spec_path = _copy_example(tmp_path)
    spec = _read(spec_path)
    transition = spec["authority_transitions"][0]
    transition.update(
        {
            "changed_runtime_dependencies": ["evidence/runtime-delta.json"],
            "inherited_evidence_boundaries": [
                {"evidence_pointer": "evidence/inheritance.json", "dependency_changed": True}
            ],
            "evidence_conditions": [
                {"evidence_pointer": "evidence/anomaly.json", "status": "ANOMALOUS"}
            ],
            "limitations": ["evidence/limitations.md"],
            "no_claims": ["evidence/no-claims.md"],
            "decision": {
                "kind": "acceptance",
                "irreversible": True,
                "evidence_pointers": ["evidence/decision.json"],
            },
        }
    )
    _write(spec_path, spec)

    receipt = review_spec(spec_path, now=_utc(10))

    assert receipt["MECHANICAL_REVIEW"]["status"] == "PASS"
    assert {row["trigger_type"] for row in receipt["SEMANTIC_REVIEW_REQUIRED"]["questions"]} == {
        "authority_change",
        "changed_runtime_dependency",
        "inheritance_boundary_change",
        "evidence_condition",
        "limitations_or_no_claims",
        "irreversible_decision",
    }
    for question in receipt["SEMANTIC_REVIEW_REQUIRED"]["questions"]:
        assert question["question"]
        assert question["evidence_pointers"]


def test_no_stage_only_semantic_review_without_a_risk_trigger(tmp_path: Path) -> None:
    spec_path = _copy_example(tmp_path)
    spec = _read(spec_path)
    transition = spec["authority_transitions"][0]
    transition["proposed_authority"] = transition["current_authority"]
    _write(spec_path, spec)

    receipt = review_spec(spec_path, now=_utc(10))

    assert receipt["MECHANICAL_REVIEW"]["status"] == "PASS"
    assert receipt["SEMANTIC_REVIEW_REQUIRED"] == {
        "status": "NOT_REQUIRED",
        "question_count": 0,
        "questions": [],
    }


@pytest.mark.parametrize(
    ("relative_path", "mutate", "expected_fragment"),
    [
        (
            "evidence/expected_negative.json",
            lambda row: row["reason_codes"].append("UNEXPECTED"),
            "expected exact reasons",
        ),
        (
            "evidence/process_snapshots.json",
            lambda row: row["snapshots"][1]["processes"].append(
                {"owner_id": "example-helper", "pid": 9999}
            ),
            "unexpected owned PIDs",
        ),
    ],
)
def test_expected_negative_and_pid_convergence_fail_closed(
    tmp_path: Path,
    relative_path: str,
    mutate,
    expected_fragment: str,
) -> None:
    spec_path = _copy_example(tmp_path)
    target = spec_path.parent / relative_path
    document = _read(target)
    mutate(document)
    _write(target, document)

    receipt = review_spec(spec_path, now=_utc(10))

    assert receipt["MECHANICAL_REVIEW"]["status"] == "FAIL"
    assert any(expected_fragment in error for error in receipt["MECHANICAL_REVIEW"]["errors"])


def test_green_receipt_reuse_requires_self_hash_rules_inputs_and_freshness(tmp_path: Path) -> None:
    spec_path = _copy_example(tmp_path)
    original = review_spec(spec_path, now=_utc(10))
    receipt_path = tmp_path / "green.json"
    _write(receipt_path, original)

    reused = review_spec(spec_path, now=_utc(10, 30), reuse_receipt=receipt_path)
    assert reused["MECHANICAL_REVIEW"]["status"] == "PASS"
    assert reused["reuse"] == {
        "reused": True,
        "source_receipt_sha256": original["receipt_sha256"],
    }

    packet_path = spec_path.parent / "evidence" / "packet.json"
    packet = _read(packet_path)
    packet["packet_id"] = "TAMPERED"
    _write(packet_path, packet)
    refreshed = review_spec(spec_path, now=_utc(10, 30), reuse_receipt=receipt_path)
    assert refreshed["reuse"]["reused"] is False
    assert refreshed["MECHANICAL_REVIEW"]["status"] == "FAIL"

    stale = review_spec(spec_path, now=_utc(12), reuse_receipt=receipt_path)
    assert stale["reuse"]["reused"] is False
    assert stale["MECHANICAL_REVIEW"]["status"] == "FAIL"


def test_reuse_rejects_a_self_consistent_receipt_with_an_omitted_input(tmp_path: Path) -> None:
    from tools.qualification_review.io import receipt_sha256

    spec_path = _copy_example(tmp_path)
    original = review_spec(spec_path, now=_utc(10))
    original["input_hashes"] = original["input_hashes"][1:]
    original["receipt_sha256"] = receipt_sha256(original)
    receipt_path = tmp_path / "incomplete-green.json"
    _write(receipt_path, original)

    refreshed = review_spec(spec_path, now=_utc(10, 30), reuse_receipt=receipt_path)

    assert refreshed["MECHANICAL_REVIEW"]["status"] == "PASS"
    assert refreshed["reuse"]["reused"] is False
    assert len(refreshed["input_hashes"]) == 5


def test_closed_spec_rejects_commands_before_any_evidence_review(tmp_path: Path) -> None:
    spec_path = _copy_example(tmp_path)
    spec = _read(spec_path)
    spec["command"] = "touch should-never-exist"
    _write(spec_path, spec)

    receipt = review_spec(spec_path, now=_utc(10))

    assert receipt["MECHANICAL_REVIEW"]["status"] == "FAIL"
    assert receipt["MECHANICAL_REVIEW"]["counts"]["checks"] == 1
    assert "Additional properties are not allowed" in receipt["MECHANICAL_REVIEW"]["errors"][0]
    assert not (spec_path.parent / "should-never-exist").exists()


def test_paths_must_remain_beneath_spec_root(tmp_path: Path) -> None:
    spec_path = _copy_example(tmp_path)
    outside = tmp_path / "outside.json"
    _write(outside, {"ready": True})
    spec = _read(spec_path)
    spec["json_assertions"][0]["path"] = "../outside.json"
    _write(spec_path, spec)

    receipt = review_spec(spec_path, now=_utc(10))

    assert receipt["MECHANICAL_REVIEW"]["status"] == "FAIL"
    assert any("escapes review root" in error for error in receipt["MECHANICAL_REVIEW"]["errors"])
