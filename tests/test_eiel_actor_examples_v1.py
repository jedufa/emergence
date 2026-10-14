from __future__ import annotations

import json
from pathlib import Path

from jsonschema import Draft202012Validator, RefResolver

from tools.actors import mission_planner

ROOT = Path(__file__).resolve().parents[1]
ACTORS = ROOT / "tools" / "actors"


def test_actor_json_schemas_are_valid_draft_2020_12() -> None:
    for path in sorted((ACTORS / "schemas").glob("*.schema.json")):
        schema = json.loads(path.read_text(encoding="utf-8"))
        Draft202012Validator.check_schema(schema)


def test_all_actor_examples_are_valid_json_objects() -> None:
    paths = sorted((ACTORS / "examples").glob("*.json"))
    assert paths
    for path in paths:
        value = json.loads(path.read_text(encoding="utf-8"))
        assert isinstance(value, dict), path


def test_example_mission_is_a_deliberate_preflight_blocked_draft() -> None:
    path = ACTORS / "examples" / "mission_specification.example.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    mission = mission_planner.normalize_mission_spec(raw)
    validation = mission_planner.validate_mission_spec(mission)
    assert validation["valid"] is False
    assert validation["authority"]["ready_for_work"] is False
    assert validation["next_actions"][0]["action"] == "run_evidence_preflight"
    assert any("retained Knowledge Librarian preflight" in row for row in validation["errors"])


def test_all_mission_examples_use_v2_and_include_authority_and_campaign_handoff_fields() -> None:
    for path in sorted((ACTORS / "examples").glob("mission_*.example.json")):
        raw = json.loads(path.read_text(encoding="utf-8"))
        assert raw["schema_version"] == "eiel_mission_specification_v2", path
        assert raw["authority_context"]["schema_version"] == "eiel_mission_authority_context_v1", path
        assert "campaign_spec_paths" in raw["references"], path
        assert "librarian_document_ids" in raw["required_closeout_refs"], path
