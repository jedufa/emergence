from __future__ import annotations

import importlib.util
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = PROJECT_ROOT / "tools/bin/lit1_phase3_scan_v4_sample_report.py"
SPEC = importlib.util.spec_from_file_location("lit1_phase3_scan_v4_sample_report", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
lit1_report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(lit1_report)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_phase1_outputs(root: Path, *, run_dir: Path, schema_versions: list[str], scan_rows: int) -> None:
    output_dir = root / f"child_{run_dir.name}"
    _write_json(
        output_dir / "lit1_lineage_funnel_phase1_summary.json",
        {
            "attempt_sources": ["schema_v4_scan_event"] if scan_rows else ["generation_csv_birth_attempts"],
            "event_summary": {
                "event_file_present": True,
                "event_rows_read": 20,
                "event_schema_versions_seen": schema_versions,
                "event_type_counts": {
                    "reproduction_scan_exposure": scan_rows,
                    "reproduction_scan_eligible": 2 if scan_rows else 0,
                    "reproduction_scan_ineligible": 3 if scan_rows else 0,
                    "reproduction_scan_no_attempt": 4 if scan_rows else 0,
                    "reproduction_scan_attempt_recorded": 5 if scan_rows else 0,
                },
            },
        },
    )
    _write_json(
        output_dir / "lit1_lineage_funnel_phase1_metric_coverage.json",
        {
            "missing_required_metrics": [],
            "required_metric_status": {
                "scan_exposure_count": "available_from_schema_v4_events"
                if scan_rows
                else "present_zero_or_events_missing"
            },
        },
    )
    _write_json(
        output_dir / "lit1_lineage_funnel_phase1_collapse_stage_summary.json",
        {"top1_dominance_generation": -1, "effective_lineage_collapse_generation": -1},
    )


def test_scan_v4_report_marks_corrected_sample_valid(tmp_path: Path) -> None:
    phase1_root = tmp_path / "phase1"
    run_dir = tmp_path / "runs/r0001_aaaa"
    _write_phase1_outputs(phase1_root, run_dir=run_dir, schema_versions=["4"], scan_rows=7)
    campaign = {
        "planned_runs": 1,
        "failures": [],
        "results": [
            {
                "index": 1,
                "run_id": "lit1__context_flat_control__init_legacy__seed000101",
                "run_dir": str(run_dir),
                "status": "ok",
            }
        ],
    }
    campaign_path = tmp_path / "parallel_campaign_summary.json"
    _write_json(campaign_path, campaign)

    report = lit1_report.build_report(
        campaign_summary_path=campaign_path,
        phase1_output_root=phase1_root,
        expected_runs=1,
    )

    assert report["summary"]["decision"]["decision"] == "LIT-1-PHASE3-SCANV4-A"
    assert report["summary"]["decision"]["totals"]["event_schema_versions_seen"] == ["4"]
    assert report["summary"]["decision"]["totals"]["reproduction_scan_exposure_rows"] == 7
    assert report["cells"][0]["attempt_sources"] == "schema_v4_scan_event"


def test_scan_v4_report_downgrades_schema3_event_attempt(tmp_path: Path) -> None:
    phase1_root = tmp_path / "phase1"
    run_dir = tmp_path / "runs/r0001_aaaa"
    _write_phase1_outputs(phase1_root, run_dir=run_dir, schema_versions=["3"], scan_rows=0)
    campaign = {
        "planned_runs": 1,
        "failures": [],
        "results": [
            {
                "index": 1,
                "run_id": "lit1__context_flat_control__init_legacy__seed000101",
                "run_dir": str(run_dir),
                "status": "ok",
            }
        ],
    }
    campaign_path = tmp_path / "parallel_campaign_summary.json"
    _write_json(campaign_path, campaign)

    report = lit1_report.build_report(
        campaign_summary_path=campaign_path,
        phase1_output_root=phase1_root,
        expected_runs=1,
    )

    assert report["summary"]["decision"]["decision"] == "LIT-1-PHASE3-SCANV4-C"
    assert "schema_v4_only" in report["summary"]["decision"]["failed_criteria"]
    assert "scan_exposure_rows" in report["summary"]["decision"]["failed_criteria"]
