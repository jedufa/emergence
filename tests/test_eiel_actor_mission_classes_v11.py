from __future__ import annotations

from tools.actors import mission_planner


def materialize(template: dict) -> dict:
    payload = dict(template)
    payload.update({
        "title": "Concrete bounded mission",
        "question": "What bounded result is supported?",
        "decision_to_enable": "Choose the next finite programme step.",
        "scope": ["declared scope"],
        "out_of_scope": ["everything else"],
        "success_criteria": ["predeclared criterion passes"],
        "failure_criteria": ["predeclared criterion fails"],
        "stopping_rules": ["stop after fixed scope"],
        "decision_rule": "Apply decision rule V1.",
        "tasks": ["one bounded task"],
        "technical_qualification_requirements": ["current route authority"],
    })
    payload["required_evidence"] = [str(x).replace("<required evidence artifact>", "durable report") for x in payload["required_evidence"]]
    return mission_planner.normalize_mission_spec(payload)


def _assert_only_authority_blocks(payload: dict) -> None:
    result = mission_planner.validate_mission_spec(payload)
    assert result["valid"] is False
    non_authority = [
        row for row in result["errors"]
        if "authority_context" not in row and "Knowledge Librarian preflight" not in row
    ]
    assert non_authority == [], non_authority


def test_mission_classes_have_distinct_required_surfaces_and_never_self_authorize() -> None:
    scientific = mission_planner._default_template("SCIENTIFIC_PROBE")
    technical = mission_planner._default_template("TECHNICAL_QUALIFICATION")
    performance = mission_planner._default_template("PERFORMANCE")
    closure = mission_planner._default_template("CLOSURE")

    assert scientific["controls"]
    assert any("reward" in item for item in scientific["required_evidence"])
    assert "campaign_output_refs" in scientific["required_closeout_refs"]
    assert technical["claim_ceiling"].startswith("Technical route trust")
    assert "qualification_refs" in technical["required_closeout_refs"]
    assert performance["resource_demand"]["device"] == "either"
    assert closure["resource_demand"]["device"] == "none"
    assert all("librarian_document_ids" in row["required_closeout_refs"] for row in (scientific, technical, performance, closure))

    for row in (scientific, technical, performance, closure):
        _assert_only_authority_blocks(materialize(row))
