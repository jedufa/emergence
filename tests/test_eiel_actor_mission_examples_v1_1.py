from __future__ import annotations

import json
from pathlib import Path

from tools.actors import mission_planner

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "tools" / "actors" / "examples"

EXPECTED = {
    "mission_implementation_validation.example.json": "IMPLEMENTATION_VALIDATION",
    "mission_technical_qualification.example.json": "TECHNICAL_QUALIFICATION",
    "mission_scientific_probe.example.json": "SCIENTIFIC_PROBE",
    "mission_confirmation.example.json": "CONFIRMATION",
    "mission_closure.example.json": "CLOSURE",
}


def test_realistic_mission_class_examples_are_distinct_authority_blocked_drafts() -> None:
    loaded = {}
    for filename, mission_type in EXPECTED.items():
        raw = json.loads((EXAMPLES / filename).read_text(encoding="utf-8"))
        normalized = mission_planner.normalize_mission_spec(raw)
        validation = mission_planner.validate_mission_spec(normalized)
        assert validation["valid"] is False
        assert validation["authority"]["ready_for_work"] is False
        assert validation["next_actions"][0]["action"] == "run_evidence_preflight"
        assert normalized["mission_type"] == mission_type
        loaded[mission_type] = normalized

    assert "campaign_output_refs" in loaded["SCIENTIFIC_PROBE"]["required_closeout_refs"]
    assert "campaign_output_refs" in loaded["CONFIRMATION"]["required_closeout_refs"]
    assert loaded["SCIENTIFIC_PROBE"]["controls"]
    assert loaded["CONFIRMATION"]["controls"]
    assert loaded["TECHNICAL_QUALIFICATION"]["qualification_request"]
    assert loaded["IMPLEMENTATION_VALIDATION"]["claim_ceiling"].startswith("Implementation")
    assert loaded["CLOSURE"]["resource_demand"]["device"] == "none"
    assert all("librarian_document_ids" in row["required_closeout_refs"] for row in loaded.values())
