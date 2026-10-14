from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.actors import qualification_manager

FIXED = "2026-08-03T12:00:00Z"

pytestmark = pytest.mark.skipif(not qualification_manager.DEFAULT_AUTHORITY_REGISTER.is_file(), reason="published qualification register is not included in this core transfer")


def read_output(capsys) -> dict:
    return json.loads(capsys.readouterr().out)


def test_real_register_summary_and_known_authority_states(capsys) -> None:
    assert qualification_manager.DEFAULT_AUTHORITY_REGISTER.is_file()
    assert qualification_manager.main(["status", "--generated-at", FIXED]) == 0
    summary = read_output(capsys)["summary"]
    assert summary["track_count"] == 23
    assert summary["status_counts"] == {"blocked": 8, "qualified": 14, "superseded": 1}

    assert qualification_manager.main(["check", "--track-label", "b48w16p64c128gNt160gpu"]) == 0
    qualified = read_output(capsys)
    assert qualified["decision"] == "QUALIFIED"
    assert qualified["full_result"]["qualified"] is True
    assert qualified["full_result"]["limitations"]

    assert qualification_manager.main(["check", "--track-label", "b1w32p256c512g32t80cpu-wdl_delayed_hazard"]) == 1
    blocked = read_output(capsys)
    assert blocked["decision"] == "BLOCKED"
    assert blocked["ok"] is False
    assert "authority_blocked" in blocked["reason_codes"]
    assert blocked["next_actions"]

    assert qualification_manager.main(["check", "--track-label", "b72w16p64c128gNt160gpu"]) == 1
    superseded = read_output(capsys)
    assert superseded["decision"] == "SUPERSEDED"
    assert superseded["ok"] is False
    assert "authority_superseded" in superseded["reason_codes"]


def test_named_track_cannot_hide_mismatched_route_axes(tmp_path: Path, capsys) -> None:
    request = tmp_path / "request.json"
    request.write_text(json.dumps({
        "track_label": "b48w16p64c128gNt160gpu",
        "world": "patch",
        "grid_width": 16,
        "grid_height": 16,
        "initial_population": 64,
        "resident_capacity": 128,
        "ticks_per_generation": 160,
        "device": "cpu",
        "batch_cells": 48,
    }), encoding="utf-8")
    assert qualification_manager.main(["check", "--request", str(request)]) == 1
    result = read_output(capsys)
    assert result["decision"] == "REQUEST_OUTSIDE_AUTHORITY"
    assert result["ok"] is False
    assert result["full_result"]["mismatches"]["device"] == {"requested": "cpu", "track": "gpu"}


def test_real_register_check_plan_show_history_next_and_handoff(tmp_path: Path, capsys) -> None:
    state = tmp_path / "state"
    assert qualification_manager.main([
        "check", "--track-label", "b48w16p64c128gNt160gpu", "--store",
        "--state-root", str(state), "--generated-at", FIXED,
    ]) == 0
    checked = read_output(capsys)
    check_id = checked["check_id"]

    assert qualification_manager.main(["show", check_id, "--state-root", str(state)]) == 0
    shown = read_output(capsys)
    assert shown["record"]["record_id"] == check_id

    assert qualification_manager.main(["history", check_id, "--state-root", str(state)]) == 0
    history = read_output(capsys)
    assert history["revision_count"] == 1

    assert qualification_manager.main(["next", check_id, "--state-root", str(state)]) == 0
    next_result = read_output(capsys)
    assert next_result["status"] == "QUALIFIED"

    assert qualification_manager.main(["handoff", check_id, "--state-root", str(state)]) == 0
    handoff = read_output(capsys)["handoff"]
    assert handoff["record_id"] == check_id
    assert handoff["technical_only"] is True

    unknown = tmp_path / "unknown.json"
    unknown.write_text(json.dumps({
        "world": "patch", "grid_width": 17, "grid_height": 17,
        "initial_population": 65, "resident_capacity": 129,
        "ticks_per_generation": 81, "device": "gpu", "batch_cells": 49,
    }), encoding="utf-8")
    assert qualification_manager.main([
        "plan", "--request", str(unknown), "--state-root", str(state),
        "--generated-at", "2026-08-03T12:01:00Z",
    ]) == 0
    plan = read_output(capsys)
    assert plan["plan"]["plan_class"] == "bounded_missing_authority"
    assert plan["plan"]["execution_authority"] == "none_plan_only"
