from __future__ import annotations

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from tools.actors.common import ActorError
from tools.scheduler.contracts import (
    assignment_material_sha256,
    assignment_store,
    dispatch_store,
    lease_store,
    result_store,
    normalize_assignment_payload,
    normalize_dispatch_payload,
    normalize_lease_payload,
    normalize_worker_result_payload,
    verify_assignment_payload,
)

ROOT = Path(__file__).resolve().parents[1]


def test_all_scheduler_schemas_are_draft_2020_12_valid() -> None:
    paths = sorted((ROOT / "tools/scheduler/schemas").glob("*.json"))
    assert len(paths) == 8
    for path in paths:
        Draft202012Validator.check_schema(json.loads(path.read_text(encoding="utf-8")))




def test_scheduler_examples_validate_against_their_schemas() -> None:
    mapping = {
        "dispatch_request.example.json": "eiel_dispatch_request_v1.schema.json",
        "assignment.example.json": "eiel_worker_assignment_v1.schema.json",
        "lease.example.json": "eiel_assignment_lease_v1.schema.json",
        "worker_result.example.json": "eiel_worker_result_v1.schema.json",
    }
    for example_name, schema_name in mapping.items():
        example = json.loads((ROOT / "tools/scheduler/examples" / example_name).read_text(encoding="utf-8"))
        schema = json.loads((ROOT / "tools/scheduler/schemas" / schema_name).read_text(encoding="utf-8"))
        errors = sorted(Draft202012Validator(schema).iter_errors(example), key=lambda row: list(row.path))
        assert not errors, [error.message for error in errors]


def test_scheduler_records_are_isolated_below_optional_scheduler_root(tmp_path: Path) -> None:
    assert dispatch_store(tmp_path).base == (tmp_path / "scheduler" / "dispatch-requests").resolve()
    assert assignment_store(tmp_path).base == (tmp_path / "scheduler" / "assignments").resolve()
    assert lease_store(tmp_path).base == (tmp_path / "scheduler" / "leases").resolve()
    assert result_store(tmp_path).base == (tmp_path / "scheduler" / "worker-results").resolve()


def test_dispatch_request_is_explicit_and_never_authorizes_work_completion(tmp_path: Path) -> None:
    payload = normalize_dispatch_payload({
        "work_id": "WORK-EXPLICIT",
        "repository": {"repo_root": str(tmp_path)},
        "allowed_authority_phases": ["implementation", "review"],
        "automation_policy": {"allow_fresh_session": True, "allow_resume": True, "allow_independent_review": True, "allow_repair": True, "allow_work_completion": True},
    })
    assert payload["automation_policy"]["allow_work_completion"] is False
    assert payload["enabled"] is True


def test_assignment_material_digest_is_canonical_and_tamper_evident() -> None:
    material = {"work": {"record_id": "WORK-X"}, "session": {"record_id": "SES-X"}}
    first = normalize_assignment_payload({"material": material, "preparation": {}, "operational": {}})
    second = normalize_assignment_payload({"material": {"session": material["session"], "work": material["work"]}, "preparation": {}, "operational": {}})
    assert first["assignment_material_sha256"] == second["assignment_material_sha256"]
    tampered = dict(first)
    tampered["material"] = {"work": {"record_id": "WORK-Y"}, "session": {"record_id": "SES-X"}}
    assert verify_assignment_payload(tampered, require_ready=True) == ["Assignment material digest mismatch"]


def test_assignment_rejects_supplied_wrong_digest() -> None:
    with pytest.raises(ActorError, match="digest mismatch"):
        normalize_assignment_payload({"material": {"x": 1}, "assignment_material_sha256": "0" * 64})


def test_lease_binds_assignment_digest_and_worker_identity() -> None:
    payload = normalize_lease_payload({
        "assignment_id": "ASSIGN-ABC",
        "assignment_material_sha256": "a" * 64,
        "scheduler_id": "SCH-ABC",
        "claimed_at_utc": "2026-08-05T00:00:00Z",
        "expires_at_utc": "2026-08-05T01:00:00Z",
        "last_heartbeat_at_utc": "2026-08-05T00:00:00Z",
        "duration_seconds": 3600,
        "heartbeat_interval_seconds": 60,
        "worker": {"kind": "fake", "worker_id": "fake-1", "pid": 123, "process_start_identity": "proc:1"},
    })
    assert payload["assignment_material_sha256"] == "a" * 64
    assert payload["worker"]["pid"] == 123


def test_worker_result_requires_structured_known_outcome() -> None:
    with pytest.raises(ActorError, match="outcome"):
        normalize_worker_result_payload({"outcome": "PROSE_SAYS_DONE"})


def test_heartbeat_preserves_the_claimed_lease_duration(tmp_path: Path) -> None:
    from tests.eiel_scheduler_v110_helpers import make_scheduler_fixture
    from tools.scheduler.dispatch import claim_assignment, heartbeat_lease, request_assignment
    from tools.scheduler.contracts import parse_utc

    fixture = make_scheduler_fixture(tmp_path)
    assignment = request_assignment(state_root=fixture.state, generated_at="2026-08-05T00:00:00Z")
    assert assignment is not None
    claimed, lease = claim_assignment(
        assignment["record_id"],
        scheduler_id="SCH-HEARTBEAT-TEST",
        state_root=fixture.state,
        generated_at="2026-08-05T00:00:00Z",
    )
    from tools.scheduler.dispatch import mark_assignment_running
    _, running_lease = mark_assignment_running(
        claimed["record_id"],
        lease["record_id"],
        {"kind": "fake", "worker_id": "worker-heartbeat"},
        state_root=fixture.state,
        generated_at="2026-08-05T00:00:10Z",
    )
    assert int((parse_utc(running_lease["payload"]["expires_at_utc"]) - parse_utc(running_lease["payload"]["last_heartbeat_at_utc"])).total_seconds()) == 3600
    renewed = heartbeat_lease(
        lease["record_id"],
        state_root=fixture.state,
        generated_at="2026-08-05T00:00:30Z",
    )
    payload = renewed["payload"]
    assert int((parse_utc(payload["expires_at_utc"]) - parse_utc(payload["last_heartbeat_at_utc"])).total_seconds()) == 3600


def test_dispatch_boolean_fields_fail_closed_instead_of_coercing_strings() -> None:
    from tools.actors.common import ActorError
    from tools.scheduler.contracts import normalize_dispatch_payload

    raw = {
        "work_id": "WORK-BOOL-TEST",
        "enabled": True,
        "repository": {"repo_root": "."},
        "automation_policy": {"allow_resume": "false"},
    }
    try:
        normalize_dispatch_payload(raw)
    except ActorError as exc:
        assert exc.code == "scheduler_contract_invalid"
    else:
        raise AssertionError("string boolean was silently coerced")
