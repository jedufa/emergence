from __future__ import annotations

import importlib.util
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = PROJECT_ROOT / "tools/bin/lit1_phase2_cross_world_localization_report.py"
SPEC = importlib.util.spec_from_file_location("lit1_phase2_cross_world_localization_report", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
lit1_report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(lit1_report)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_phase1_outputs(
    root: Path,
    *,
    run_dir: Path,
    final_top1: float = 0.2,
    final_effective: float = 7.0,
    event_file_present: bool = False,
) -> None:
    output_dir = root / f"child_{run_dir.name}"
    _write_json(
        output_dir / "lit1_lineage_funnel_phase1_summary.json",
        {
            "generation_count": 2,
            "lineage_generation_row_count": 4,
            "attempt_sources": ["generation_csv_birth_attempts"],
        },
    )
    _write_json(
        output_dir / "lit1_lineage_funnel_phase1_metric_coverage.json",
        {"event_file_present": event_file_present, "missing_required_metrics": []},
    )
    _write_json(
        output_dir / "lit1_lineage_funnel_phase1_collapse_stage_summary.json",
        {
            "top1_dominance_generation": -1,
            "effective_lineage_collapse_generation": -1,
            "by_generation": [
                {
                    "generation": 0,
                    "top1_post_refill_alive_share": 0.1,
                    "effective_post_refill_alive_lineages": 12.0,
                },
                {
                    "generation": 1,
                    "top1_post_refill_alive_share": final_top1,
                    "effective_post_refill_alive_lineages": final_effective,
                },
            ],
        },
    )


def test_phase2_report_aggregates_cells_and_marks_csv_pass_valid(tmp_path: Path) -> None:
    phase1_root = tmp_path / "phase1"
    run1 = tmp_path / "runs/r0001_aaaa"
    run2 = tmp_path / "runs/r0002_bbbb"
    _write_phase1_outputs(phase1_root, run_dir=run1, final_top1=0.25, final_effective=8.0)
    _write_phase1_outputs(phase1_root, run_dir=run2, final_top1=0.35, final_effective=6.0)
    campaign = {
        "registry": "v1/example.yaml",
        "batch_tag": "lit1_phase2_cross_world_localization_csv",
        "campaign_output_profile": "repeat_parity",
        "planned_runs": 2,
        "workers": 2,
        "failures": [],
        "total_wall_s": 10.0,
        "mean_child_wall_s": 5.0,
        "results": [
            {
                "index": 1,
                "run_id": "lit1__context_flat_control__init_legacy__seed000101",
                "status": "ok",
                "wall_s": 4.0,
                "run_dir": str(run1),
            },
            {
                "index": 2,
                "run_id": "lit1__context_flat_control__init_legacy__seed000202",
                "status": "ok",
                "wall_s": 6.0,
                "run_dir": str(run2),
            },
        ],
    }
    campaign_path = tmp_path / "parallel_campaign_summary.json"
    _write_json(campaign_path, campaign)

    report = lit1_report.build_report(
        campaign_summary_path=campaign_path,
        phase1_output_root=phase1_root,
        expected_runs=2,
    )

    assert report["summary"]["decision"]["decision"] == "LIT-1-PHASE2-CSV-A"
    assert report["summary"]["decision"]["event_files_present"] == 0
    assert len(report["cells"]) == 2
    assert report["groups"][0]["runs"] == 2
    assert report["groups"][0]["final_top1_mean"] == 0.3
    assert report["groups"][0]["final_effective_mean"] == 7.0


def test_phase2_report_fails_if_expected_outputs_are_incomplete(tmp_path: Path) -> None:
    phase1_root = tmp_path / "phase1"
    run1 = tmp_path / "runs/r0001_aaaa"
    _write_phase1_outputs(phase1_root, run_dir=run1)
    campaign = {
        "planned_runs": 2,
        "failures": [],
        "results": [
            {
                "index": 1,
                "run_id": "lit1__context_flat_control__init_legacy__seed000101",
                "status": "ok",
                "wall_s": 4.0,
                "run_dir": str(run1),
            },
        ],
    }
    campaign_path = tmp_path / "parallel_campaign_summary.json"
    _write_json(campaign_path, campaign)

    report = lit1_report.build_report(
        campaign_summary_path=campaign_path,
        phase1_output_root=phase1_root,
        expected_runs=2,
    )

    assert report["summary"]["decision"]["decision"] == "LIT-1-PHASE2-CSV-C"
    assert "phase1_output_count" in report["summary"]["decision"]["failed_criteria"]
