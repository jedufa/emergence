from __future__ import annotations

import json
from pathlib import Path

from jsonschema import Draft202012Validator

from tools.actors import mission_planner

ROOT = Path(__file__).resolve().parents[1]
ACTORS = ROOT / "tools/actors"
GUIDE = ACTORS / "CODEX_OPERATOR_GUIDE.md"


def _schema(name: str) -> dict:
    return json.loads((ACTORS / "schemas" / name).read_text(encoding="utf-8"))


def test_realistic_mission_examples_are_structurally_valid_preflight_blocked_v2_drafts() -> None:
    # The authority context is an external durable binding, so examples remain
    # deliberately blocked drafts.  They must still be structurally complete.
    paths = sorted((ACTORS / "examples").glob("mission*.example.json"))
    assert {path.name for path in paths} >= {
        "mission_technical_qualification.example.json",
        "mission_scientific_probe.example.json",
        "mission_confirmation.example.json",
        "mission_implementation_validation.example.json",
        "mission_closure.example.json",
    }
    for path in paths:
        raw = json.loads(path.read_text(encoding="utf-8"))
        normalized = mission_planner.normalize_mission_spec(raw)
        result = mission_planner.validate_mission_spec(normalized)
        assert result["valid"] is False, path
        non_authority = [
            row for row in result["errors"]
            if "authority_context" not in row and "Knowledge Librarian preflight" not in row
        ]
        assert non_authority == [], (path, non_authority)
        assert result["next_actions"][0]["action"] == "run_evidence_preflight"
        assert normalized["schema_version"] == "eiel_mission_specification_v2"
        assert "campaign_spec_paths" in normalized["references"]
        assert "librarian_document_ids" in normalized["required_closeout_refs"]


def test_resource_and_qualification_examples_match_schemas() -> None:
    resource = json.loads((ACTORS / "examples/resource_demand.example.json").read_text(encoding="utf-8"))
    qualification = json.loads((ACTORS / "examples/qualification_request.example.json").read_text(encoding="utf-8"))
    Draft202012Validator(_schema("eiel_resource_demand_v1.schema.json")).validate(resource)
    Draft202012Validator(_schema("eiel_qualification_request_v1.schema.json")).validate(qualification)


def test_machine_readable_workflow_is_complete_authority_first_and_non_executing() -> None:
    workflow = json.loads((ACTORS / "examples/codex_workflow_v1_2.json").read_text(encoding="utf-8"))
    steps = {row["id"]: row for row in workflow["steps"]}
    assert {
        "create", "preflight", "qualification", "attach_qualification", "resources", "attach_resources",
        "external_execution", "attach_outputs", "evidence", "attach_evidence", "closeout", "verify",
    } <= set(steps)
    assert "tools.scripts.eiel_campaign" in steps["external_execution"]["command"]
    text = json.dumps(workflow).lower()
    assert "resource manager does not reserve or kill processes" in text
    assert "actors do not launch campaigns" in text
    assert "knowledge librarian" in text


def test_codex_guide_contains_exact_commands_authority_binding_and_recovery_path() -> None:
    text = GUIDE.read_text(encoding="utf-8")
    for actor in ("evidenceManager", "qualificationManager", "resourceManager", "missionPlanner"):
        assert f"python -m tools.eiel {actor}" in text
    for command in (
        "python -m tools.eiel missionPlanner preflight MIS-...",
        "python -m tools.eiel missionPlanner status MIS-...",
        "python -m tools.eiel missionPlanner history MIS-...",
        "python -m tools.eiel missionPlanner next MIS-...",
        "python -m tools.eiel missionPlanner handoff MIS-...",
    ):
        assert command in text
    assert "None of these four actor CLIs launches the campaign" in text
    assert "Never treat exit code `0` alone as permission to run" in text
    assert "ACTIVE_GOAL.md" in text
    assert "## Evidence Preflight" in text
