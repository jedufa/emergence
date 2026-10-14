from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[5]
ENGINE_ROOT = REPO_ROOT / "experiments" / "projects" / "emergence" / "engine"
for required in (str(REPO_ROOT), str(ENGINE_ROOT)):
    while required in sys.path:
        sys.path.remove(required)
sys.path.insert(0, str(ENGINE_ROOT))
sys.path.insert(1, str(REPO_ROOT))

from simulation.chronicle.storage import read_json, read_jsonl, write_json, write_jsonl  # noqa: E402
from simulation.chronicle.trace import (  # noqa: E402
    TRACE_ARTIFACTS,
    TRACE_FIELD_GROUPS,
    TRACE_REQUEST_SCHEMA,
    TraceBudgets,
    TraceError,
    estimate_trace_storage,
    make_trace_request,
    plan_trace_windows,
    produce_fixture_trace,
    query_trace,
    storage_projection_table,
    trace_from_chronicle_fixture,
    write_trace_outputs,
)


def _event(
    event_id: str,
    *,
    generation: int = 2,
    step: int = 10,
    event_type: str = "notable_death",
    priority: str = "critical",
    subject_id: str = "7",
) -> dict[str, object]:
    return {
        "schema_version": "eiel_formal_event_v1",
        "event_id": event_id,
        "run_id": "trace-run",
        "campaign_id": "campaign",
        "branch_id": "branch",
        "generation": generation,
        "step": step,
        "phase_order": 300,
        "intra_step_sequence": 0,
        "event_time_scope": "generation",
        "event_type": event_type,
        "event_category": "lifecycle",
        "subject_type": "agent",
        "subject_id": subject_id,
        "agent_ids": [int(subject_id)],
        "lineage_ids": [f"L{int(subject_id):08d}"],
        "patch_coordinates": [1, 2],
        "region_id": "patch:1,2",
        "severity": "critical",
        "live_priority": priority,
        "scientific_importance": 5,
        "historical_importance": 5,
        "confidence_class": "measured",
        "causality_class": "not_assessed",
        "source_component": "test",
        "source_record_refs": ["test"],
    }


def test_trace_request_ids_are_deterministic_and_canonical() -> None:
    a = make_trace_request(run_id="r", profile="trace_subjects", selected_lineages=["L2", "L1"], selected_agents=[3, 2])
    b = make_trace_request(run_id="r", profile="trace_subjects", selected_lineages=["L1", "L2"], selected_agents=[2, 3])
    assert a.request_id == b.request_id
    assert a.schema_version == TRACE_REQUEST_SCHEMA
    assert a.selected_agents == (2, 3)
    assert a.selected_lineages == ("L1", "L2")


def test_trace_profiles_validate_heavy_mode_and_subject_budget() -> None:
    with pytest.raises(TraceError, match="requires allow_large_output"):
        make_trace_request(run_id="r", profile="trace_diagnostic_full")
    with pytest.raises(TraceError, match="selected subjects"):
        make_trace_request(
            run_id="r",
            profile="trace_subjects",
            selected_agents=range(10),
            selected_lineages=range(10),
            budgets=TraceBudgets(max_subjects=4),
        )
    request = make_trace_request(
        run_id="r",
        profile="trace_diagnostic_full",
        generation_ranges=[(0, 0)],
        step_ranges=[(0, 10)],
        allow_large_output=True,
    )
    assert set(request.captured_field_groups) == set(TRACE_FIELD_GROUPS)


def test_trace_planner_merges_overlap_and_accounts_drops() -> None:
    request = make_trace_request(
        run_id="trace-run",
        profile="trace_triggered",
        steps_before=3,
        steps_after=3,
        budgets=TraceBudgets(max_windows=1),
    )
    plan = plan_trace_windows(
        request,
        [
            _event("evt_a", step=10, subject_id="7"),
            _event("evt_b", step=12, subject_id="7"),
            _event("evt_c", step=30, subject_id="8"),
        ],
    )
    assert len(plan["windows"]) == 1
    assert len(plan["dropped_windows"]) == 1
    window = plan["windows"][0]
    assert window["start_step"] == 7
    assert window["end_step"] == 15
    assert window["trigger_source_ids"] == ["evt_a", "evt_b"]
    assert plan["estimated_records"] == 9


def test_trace_fixture_outputs_are_deterministic_queryable_and_read_only(tmp_path: Path) -> None:
    request = make_trace_request(run_id="trace-run", profile="trace_triggered", steps_before=1, steps_after=1)
    plan = plan_trace_windows(request, [_event("evt_a", step=10)])
    fixture_a = produce_fixture_trace(request, plan)
    fixture_b = produce_fixture_trace(request, plan)
    assert fixture_a == fixture_b
    output = tmp_path / "trace"
    result = write_trace_outputs(output, request, plan, fixture_a)
    assert result["request_id"] == request.request_id
    manifest = read_json(output / TRACE_ARTIFACTS["manifest"])
    status = read_json(output / TRACE_ARTIFACTS["status"])
    records = read_jsonl(output / TRACE_ARTIFACTS["steps"])
    assert manifest["source_writes_attempted"] is False
    assert manifest["rng_consumed"] is False
    assert status["runtime_hooks_used"] is False
    assert len(records) == 3
    assert len(query_trace(records, action="gather")) >= 1
    assert query_trace(records, field_group="state_core")


def test_trace_storage_projection_rejects_unrestricted_full_trace_as_default() -> None:
    rows = storage_projection_table(steps_per_generation=150)
    by_shape = {row["shape"]: row for row in rows}
    assert by_shape["full_trace_pop64_g10000"]["preflight_risk"] == "rejected_default"
    assert by_shape["triggered_8_windows_g36"]["preflight_risk"] == "bounded"
    estimate = estimate_trace_storage(field_groups=("identity", "action"), step_records=150, windows=1, agents_per_step=64)
    assert estimate["estimated_bytes"] > 1_000_000


def test_trace_cli_fixture_and_query_from_chronicle(tmp_path: Path) -> None:
    run = tmp_path / "run"
    chronicle = run / "chronicle"
    chronicle.mkdir(parents=True)
    write_jsonl(chronicle / "formal_events.jsonl", [_event("evt_a", step=10)])
    write_json(chronicle / "chronicle_manifest.json", {"run_id": "trace-run", "source_commit": "abc"})
    output = tmp_path / "trace"
    completed = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "experiments" / "projects" / "emergence" / "tools" / "scientific_chronicle.py"),
            "trace-fixture",
            str(run),
            str(output),
            "--steps-before",
            "1",
            "--steps-after",
            "1",
        ],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    assert json.loads(completed.stdout)["files"]
    queried = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "experiments" / "projects" / "emergence" / "tools" / "scientific_chronicle.py"),
            "trace-query",
            str(output),
            "--field-group",
            "state_core",
        ],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    assert json.loads(queried.stdout)["items"]


def test_trace_from_chronicle_fixture_helper(tmp_path: Path) -> None:
    run = tmp_path / "run"
    chronicle = run / "chronicle"
    chronicle.mkdir(parents=True)
    write_jsonl(chronicle / "formal_events.jsonl", [_event("evt_a", step=10)])
    write_json(chronicle / "chronicle_manifest.json", {"run_id": "trace-run", "source_commit": "abc"})
    result = trace_from_chronicle_fixture(run, tmp_path / "trace", steps_before=1, steps_after=1)
    assert result["request_id"].startswith("trq_")
