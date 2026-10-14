from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


REPO_ROOT = Path(__file__).resolve().parents[5]
ENGINE_ROOT = REPO_ROOT / "experiments" / "projects" / "emergence" / "engine"
for required in (str(REPO_ROOT), str(ENGINE_ROOT)):
    while required in sys.path:
        sys.path.remove(required)
sys.path.insert(0, str(ENGINE_ROOT))
sys.path.insert(1, str(REPO_ROOT))

from infra.webui.server_chronicle import (
    comparison_detail,
    context_detail,
    discover_comparisons,
    discover_runs,
    episode_detail,
    event_detail,
    handle_get_request,
    run_overview,
    subject_detail,
    timeline,
)
from simulation.chronicle.analytics import (
    ChronicleAnalyticsError,
    aggregate_selected_runs,
    analyze_run,
    search_run,
    write_group_exports,
    write_run_exports,
)
from simulation.chronicle.compare import compare_chronicles
from simulation.chronicle.recorder import ScientificChronicleRecorder


@dataclass
class _Stats:
    lineage_id: str
    founder_id: int


@dataclass
class _Agent:
    agent_id: int
    stats: _Stats


def _agent(agent_id: int, lineage: str) -> _Agent:
    return _Agent(agent_id=agent_id, stats=_Stats(lineage_id=lineage, founder_id=agent_id))


def _chronicle(run_root: Path, *, run_id: str, depleted: float = 0.40) -> Path:
    cfg = SimpleNamespace(
        chronicle_enable=True,
        chronicle_profile="notable",
        chronicle_run_id=run_id,
        chronicle_max_stored_events_per_generation=64,
        chronicle_minimum_persistence_generations=1,
        campaign_id="campaign-a",
        branch_id=f"branch-{run_id}",
        generations=3,
        allow_large_output_profile=False,
    )
    recorder = ScientificChronicleRecorder(SimpleNamespace(sim_cfg=cfg, run_dir=str(run_root)))
    recorder.observe_generation(
        generation=0,
        step=9,
        alive_agents=[_agent(1, "L1"), _agent(2, "L2")],
        deaths=0,
        births=0,
        failed_births=0,
        cohort_size=2,
        patch_metrics={"patch_food_depleted_frac": 0.0},
    )
    recorder.observe_generation(
        generation=1,
        step=9,
        alive_agents=[_agent(1, "L1")],
        deaths=1,
        births=0,
        failed_births=1,
        cohort_size=2,
        patch_metrics={
            "patch_food_depleted_frac": depleted,
            "patch_density_hotspot_frac": 0.4,
            "occupied_patch_waste_mean": 0.5,
            "jungle_predator_events": 2,
            "jungle_predator_damage_total": 0.5,
            "jungle_predator_deaths": 1,
            "wdl_delayed_hazard_event_count": 1,
            "wdl_delayed_hazard_damage_total": 0.2,
        },
        death_contexts=[{"agent_id": 2, "biological_lineage_id": "L2", "y": 2, "x": 3}],
        predator_contexts=[{"agent_id": 2, "biological_lineage_id": "L2", "y": 2, "x": 3}],
        wdl_contexts=[{"phase": "active"}],
    )
    recorder.finalize()
    return recorder.root


def _config(root: Path) -> SimpleNamespace:
    return SimpleNamespace(runs_root=root, chronicle_roots=(root,), chronicle_scan_depth=5)


def test_explorer_discovery_overview_timeline_and_read_only(tmp_path: Path) -> None:
    chronicle_root = _chronicle(tmp_path / "runs" / "run-a", run_id="run-a")
    cfg = _config(tmp_path / "runs")
    before = {path.name: path.read_bytes() for path in chronicle_root.iterdir() if path.is_file()}

    rows = discover_runs(cfg)
    assert [(row["run_id"], row["status"]) for row in rows] == [("run-a", "ready")]
    run_key = rows[0]["run_key"]
    overview = run_overview(cfg, run_key)
    assert overview["source_completeness"] == "live_generation_boundary_complete"
    assert overview["event_counts_by_category"]["summary"] == 2
    assert overview["episode_counts_by_family"]["regional_food_depletion"] == 1
    assert overview["generation_range"]["first"] == {"generation": 0, "step": 9}
    assert overview["generation_range"]["last"] == {"generation": 1, "step": 9}

    page = timeline(
        cfg,
        run_key,
        {
            "generation_start": ["1"],
            "generation_end": ["1"],
            "category": ["ecology"],
            "priority": ["high"],
            "scientific_min": ["4"],
            "limit": ["2"],
            "offset": ["0"],
        },
    )
    assert page["total"] >= 1
    assert len(page["items"]) <= 2
    assert all(item["formal_event"]["event_category"] == "ecology" for item in page["items"])
    assert all(item["formal_event"]["event_id"] in item["entry"]["source_event_ids"] for item in page["items"])

    after = {path.name: path.read_bytes() for path in chronicle_root.iterdir() if path.is_file()}
    assert before == after


def test_explorer_episode_subject_context_comparison_and_isolated_corruption(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    _chronicle(root / "run-a", run_id="run-a", depleted=0.30)
    _chronicle(root / "run-b", run_id="run-b", depleted=0.45)
    compare_chronicles(root / "run-a", root / "run-b", root / "comparisons" / "a-v-b")
    corrupt = root / "corrupt" / "chronicle"
    corrupt.mkdir(parents=True)
    (corrupt / "formal_events.jsonl").write_text("{}\n", encoding="utf-8")
    (corrupt / "chronicle_manifest.json").write_text("not-json", encoding="utf-8")
    cfg = _config(root)
    rows = discover_runs(cfg)
    assert sorted(row["status"] for row in rows) == ["invalid", "ready", "ready"]
    run_key = next(row["run_key"] for row in rows if row.get("run_id") == "run-a")
    episode_page = timeline(cfg, run_key, {"item_kind": ["episode"], "limit": ["200"]})
    episode = episode_page["items"][0]["episode"]
    detail = episode_detail(cfg, run_key, episode["episode_id"])
    assert detail["source_events"]
    assert set(detail["supporting_generation_summary_ids"]).issubset(episode["source_event_ids"])
    assert detail["supporting_summary_semantics"] == "references_to_formal_events_not_duplicated_evidence"

    subject = subject_detail(cfg, run_key, "lineage", "L2")
    assert subject["events"]
    assert subject["partial_coverage"] is True
    event_id = subject["events"][0]["event_id"]
    formal_event = event_detail(cfg, run_key, event_id)
    assert formal_event["formal_event"]["event_id"] == event_id
    context = context_detail(cfg, run_key, event_id=event_id)
    assert context["label"] == "formal-event context window"
    assert {row["position"] for row in context["events"]}.intersection({"before", "trigger", "after"})
    assert "not a full step/action Simulation Trace" in context["limitations"][-1]

    comparisons = discover_comparisons(cfg)
    assert len(comparisons) == 1
    comparison = comparison_detail(cfg, comparisons[0]["comparison_key"], include_raw=True)
    assert comparison["default_view"] == "meaningful_divergence"
    assert comparison["first_raw_difference"] is not None
    assert comparison["run_provenance"]["run_a"][0]["source_commit"] != ""
    assert "immaterial" in comparison["raw_difference_disclaimer"]


def test_explorer_rejects_path_traversal_and_bounds(tmp_path: Path) -> None:
    _chronicle(tmp_path / "runs" / "run-a", run_id="run-a")
    cfg = _config(tmp_path / "runs")
    status, payload = handle_get_request(
        cfg=cfg,
        path="/api/chronicle/run",
        qs={"run_key": ["root0/../outside"]},
    )
    assert status == 400
    assert payload["error_code"] in {"invalid_run_key", "path_traversal"}
    run_key = discover_runs(cfg)[0]["run_key"]
    status, payload = handle_get_request(
        cfg=cfg,
        path="/api/chronicle/timeline",
        qs={"run_key": [run_key], "limit": ["201"]},
    )
    assert status == 400
    assert payload["error_code"] == "invalid_paging"


def test_analytics_are_deterministic_searchable_and_exportable(tmp_path: Path) -> None:
    root = tmp_path / "run-a"
    chronicle_root = _chronicle(root, run_id="run-a")
    source_before = {path.name: path.read_bytes() for path in chronicle_root.iterdir() if path.is_file()}
    first = analyze_run(root)
    second = analyze_run(root)
    assert first == second
    assert first["events"]["by_category"]["summary"] == 2
    assert first["episodes"]["overlap"]["maximum_concurrent_episodes"] >= 1
    assert first["density_by_generation"]
    assert any(row["lineage_id"] == "L2" for row in first["lineage_histories"])

    search = search_run(root, {"lineage_id": "L2", "generation_start": 1, "importance_min": 3})
    assert search["events"]
    assert all("L2" in event["lineage_ids"] for event in search["events"])
    exact = search_run(root, {"id": search["events"][0]["event_id"]})
    assert [value["event_id"] for value in exact["events"]] == [search["events"][0]["event_id"]]
    entry_exact = search_run(root, {"id": exact["entries"][0]["chronicle_entry_id"]})
    assert entry_exact["events"][0]["event_id"] == exact["events"][0]["event_id"]

    output = tmp_path / "analytics"
    result = write_run_exports(root, output)
    assert set(result["files"]) == {
        "chronicle_analytics.json",
        "chronicle_analytics_report.md",
        "density_by_generation.csv",
        "episode_counts.csv",
        "event_counts.csv",
        "lineage_histories.csv",
        "region_histories.csv",
    }
    source_after = {path.name: path.read_bytes() for path in chronicle_root.iterdir() if path.is_file()}
    assert source_before == source_after


def test_cross_run_analytics_require_explicit_selection_and_report_pooling(tmp_path: Path) -> None:
    run_a = tmp_path / "run-a"
    run_b = tmp_path / "run-b"
    _chronicle(run_a, run_id="run-a", depleted=0.30)
    _chronicle(run_b, run_id="run-b", depleted=0.45)
    with pytest.raises(ChronicleAnalyticsError, match="silent pooling"):
        aggregate_selected_runs([run_a, run_b], {})

    grouping = {
        "selection_mode": "explicit_operator_selection",
        "group_id": "bounded-review",
        "selection_reason": "operator selected bounded validation pair",
    }
    result = aggregate_selected_runs([run_a, run_b], grouping)
    assert result["selected_run_count"] == 2
    assert result["grouping"]["pooling_eligible"] is False
    assert result["episode_family_occurrence"]
    assert {row["run_id"] for row in result["source_runs"]} == {"run-a", "run-b"}
    assert all("source_completeness" in row for row in result["source_runs"])

    output = tmp_path / "group"
    exported = write_group_exports([run_a, run_b], grouping, output)
    assert set(exported["files"]) == {
        "episode_family_occurrence.csv",
        "selected_run_summary.json",
        "selected_run_summary.md",
    }
