from __future__ import annotations

from pathlib import Path

from experiments.projects.emergence.tools import pressure_search


def test_pressure_search_dry_step_waits_for_manual_promotion(tmp_path: Path) -> None:
    campaigns_root = tmp_path / "campaigns"
    runs_root = tmp_path / "runs"

    created = pressure_search.ensure_campaign(
        "demo_pressure",
        campaigns_root=campaigns_root,
        duration_days=28,
        max_active_children=1,
    )
    assert created["status"] == "idle"

    started = pressure_search.request_action("demo_pressure", "start", campaigns_root=campaigns_root)
    assert started["status"] == "running"

    stepped = pressure_search.supervisor_step(
        "demo_pressure",
        campaigns_root=campaigns_root,
        runs_root=runs_root,
        dry_run=True,
    )
    assert stepped["status"] == "waiting_for_promotion"
    assert len(stepped["pending_candidates"]) == 1
    candidate_key = stepped["pending_candidates"][0]["run_key"]
    assert stepped["pending_candidates"][0]["mutation"]["path"] in pressure_search.MUTATION_ALLOWLIST

    promoted = pressure_search.request_action(
        "demo_pressure",
        "promote",
        target_run_key=candidate_key,
        campaigns_root=campaigns_root,
    )
    assert promoted["status"] == "running"
    assert promoted["current_parent_run_key"] == candidate_key
    assert promoted["pending_candidates"] == []


def test_pressure_search_rejects_pending_candidate(tmp_path: Path) -> None:
    campaigns_root = tmp_path / "campaigns"
    payload = pressure_search.supervisor_step(
        "reject_pressure",
        campaigns_root=campaigns_root,
        runs_root=tmp_path / "runs",
        dry_run=True,
    )
    candidate_key = payload["pending_candidates"][0]["run_key"]

    rejected = pressure_search.request_action(
        "reject_pressure",
        "reject",
        target_run_key=candidate_key,
        campaigns_root=campaigns_root,
    )

    assert rejected["status"] == "running"
    assert rejected["pending_candidates"] == []
    assert rejected["rejected_candidates"][-1]["run_key"] == candidate_key
    assert rejected["consecutive_rejections"] == 1


def test_pressure_search_scores_completed_run(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "run_a"
    run_dir.mkdir(parents=True)
    (run_dir / "metrics.csv").write_text(
        "\n".join(
            [
                "generation,alive,cohort_survival_rate,genome_diversity_norm_std_mean,successful_births,diversity_collapse_streak",
                "1,10,0.4,0.2,8,0",
                "2,15,0.7,0.5,20,0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "epi_av_timeseries.csv").write_text(
        "\n".join(
            [
                "generation,epi_value,av_value",
                "1,1,1",
                "2,2,2",
                "3,5,4",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    score = pressure_search.score_run(run_dir)

    assert score["telemetry_ok"] is True
    assert score["survival"] == 0.7
    assert score["epi_latest"] == 5.0
    assert score["av_latest"] == 4.0
    assert score["score"] > 0
