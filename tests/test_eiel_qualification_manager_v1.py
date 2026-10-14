from __future__ import annotations

import json
from pathlib import Path

from engine.campaigns.qualification_framework import QF_SCHEMA_VERSION
from tools.actors import qualification_manager

FIXED = "2026-08-03T12:00:00Z"
QUALIFIED_COMMIT = "a" * 40


def _empty_catalog(path: Path) -> Path:
    path.write_text(json.dumps({
        "schema_version": QF_SCHEMA_VERSION,
        "anchors": [],
        "envelopes": [],
        "resource_capacity_envelopes": [],
        "historical_authority": [],
    }), encoding="utf-8")
    return path


def test_qualification_manager_register_check_and_impact(tmp_path: Path, capsys) -> None:
    catalog = _empty_catalog(tmp_path / "catalog.json")
    request = tmp_path / "request.json"
    request.write_text(json.dumps({
        "grid_width": 16,
        "grid_height": 16,
        "ticks_per_generation": 80,
        "generations": 12,
        "population": 64,
        "initial_population": 64,
        "resident_capacity": 64,
        "batch_size": 1,
        "backend": "python",
        "device": "cpu",
        "provider_id": "provider",
        "runtime_id": "runtime",
        "execution_route_sha256": "route",
        "source_commit": "commit",
    }), encoding="utf-8")

    assert qualification_manager.main(["register", "--catalog", str(catalog)]) == 0
    register = json.loads(capsys.readouterr().out)
    assert register["summary"]["anchors"]["count"] == 0

    assert qualification_manager.main([
        "check", "--catalog", str(catalog), "--request", str(request),
    ]) == 1
    checked = json.loads(capsys.readouterr().out)
    assert checked["decision"] == "FULL_QUALIFICATION_REQUIRED"

    assert qualification_manager.main([
        "impact", "--axis", "batch_size", "--changed-path", "engine/simulation/checkpoint.py",
    ]) == 0
    impact = json.loads(capsys.readouterr().out)["impact"]
    assert "physical_batch_execution" in impact["impact_families"]
    assert "checkpoint_resume" in impact["impact_families"]
    assert impact["automatic_invalidation"] is False


def test_qualification_manager_validates_declarative_plan(tmp_path: Path, capsys) -> None:
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({
        "schema_version": "eiel_qualification_plan_v1",
        "plan_id": "small-plan",
        "description": "One data-only qualification case.",
        "dimensions": [
            {"id": "device", "description": "device", "values": ["cpu"]},
        ],
        "capability_requirements": [],
        "cases": [
            {
                "id": "case-one",
                "description": "one",
                "assignments": {"device": "cpu"},
                "prerequisites": [],
                "capability_requirements": [],
            }
        ],
    }), encoding="utf-8")
    assert qualification_manager.main(["validate", "--plan", str(plan)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["validation"]["plan_kind"] == "declarative_qualification_plan"
    assert result["validation"]["ok"] is True


def _catalog_with_envelope(path: Path) -> Path:
    from engine.campaigns.qualification_framework import AxisRangeV1, OperatingEnvelopeV1, QualificationCatalogV1

    envelope = OperatingEnvelopeV1(
        envelope_id="env-one",
        scope="CONFIGURATION_OPERATING_ENVELOPE",
        axes=(
            AxisRangeV1("grid_width", 16, 32),
            AxisRangeV1("grid_height", 16, 32),
            AxisRangeV1("ticks_per_generation", 80, 80),
        ),
        categorical_invariants={
            "provider_id": "provider",
            "runtime_id": "runtime",
            "world_family": "patch",
            "execution_route_sha256": "route",
        },
        anchor_ids=(),
        required_corners=(
            {"grid_width": 16, "grid_height": 16, "ticks_per_generation": 80},
            {"grid_width": 16, "grid_height": 32, "ticks_per_generation": 80},
            {"grid_width": 32, "grid_height": 16, "ticks_per_generation": 80},
            {"grid_width": 32, "grid_height": 32, "ticks_per_generation": 80},
        ),
        accepted_points=({"grid_width": 16, "grid_height": 16, "ticks_per_generation": 80},),
        evidence=(),
    )
    path.write_text(json.dumps(QualificationCatalogV1(envelopes=(envelope,)).payload()), encoding="utf-8")
    return path


def test_qualification_manager_explain_plan_audit_and_decide(tmp_path: Path, capsys) -> None:
    catalog = _catalog_with_envelope(tmp_path / "catalog.json")
    request = tmp_path / "request.json"
    request.write_text(json.dumps({
        "provider_id": "provider",
        "runtime_id": "runtime",
        "world_family": "patch",
        "execution_route_sha256": "route",
        "grid_width": 24,
        "grid_height": 24,
        "ticks_per_generation": 80,
    }), encoding="utf-8")
    state = tmp_path / "state"

    assert qualification_manager.main([
        "explain", "--catalog", str(catalog), "--request", str(request),
    ]) == 1
    explained = json.loads(capsys.readouterr().out)
    assert explained["explanation"]["decision"] == "DELTA_REQUIRED"

    assert qualification_manager.main([
        "plan", "--catalog", str(catalog), "--request", str(request),
        "--envelope-id", "env-one", "--state-root", str(state), "--generated-at", FIXED,
    ]) == 0
    planned = json.loads(capsys.readouterr().out)
    assert planned["validation"]["ok"] is True
    assert planned["plan"]["plan_class"] == "delta"

    assert qualification_manager.main(["audit", "--catalog", str(catalog)]) == 0
    audited = json.loads(capsys.readouterr().out)
    assert audited["ok"] is True

    evidence = tmp_path / "decision.json"
    evidence.write_text(json.dumps({
        "decision": "A",
        "required_obligations": ["repeat_parity"],
        "satisfied_recipe_obligations": ["repeat_parity"],
        "plan_sha256": "plan-sha",
        "required_obligations_sha256": "obligations-sha",
    }), encoding="utf-8")
    assert qualification_manager.main([
        "decide", "--evidence", str(evidence), "--state-root", str(state),
        "--source-commit", QUALIFIED_COMMIT,
        "--generated-at", "2026-08-03T12:01:00Z",
    ]) == 0
    decided = json.loads(capsys.readouterr().out)
    assert decided["qualification_decision"]["accepted_for_envelope_expansion"] is True
    assert decided["qualification_decision"]["source_commit"] == QUALIFIED_COMMIT

    assert qualification_manager.main([
        "register", "--catalog", str(catalog), "--include-manager-records", "--state-root", str(state),
    ]) == 0
    registered = json.loads(capsys.readouterr().out)
    assert len(registered["manager_records"]) == 2
