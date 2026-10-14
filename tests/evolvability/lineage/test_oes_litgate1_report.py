from __future__ import annotations

import importlib.util
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = PROJECT_ROOT / "tools/bin/oes_litgate1_report.py"
SPEC = importlib.util.spec_from_file_location("oes_litgate1_report", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
oes_report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(oes_report)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _campaign(path: Path, *, run_dirs: list[Path]) -> None:
    _write_json(
        path,
        {
            "planned_runs": len(run_dirs),
            "failures": [],
            "results": [
                {
                    "index": index,
                    "run_id": (
                        "oes_litgate1_open_ended_search_20260709__context_flat_control"
                        f"__init_legacy__seed{100 + index:06d}"
                    ),
                    "run_dir": str(run_dir),
                    "status": "ok",
                }
                for index, run_dir in enumerate(run_dirs, start=1)
            ],
        },
    )


def _phase1_compact(root: Path, *, run_dir: Path, collapse: bool = False) -> None:
    out = root / f"child_{run_dir.name}"
    _write_json(
        out / "lit1_lineage_funnel_phase1_summary.json",
        {
            "generation_count": 2,
            "lineage_generation_row_count": 4,
            "attempt_sources": ["generation_csv_birth_attempts"],
            "event_summary": {"event_file_present": False},
        },
    )
    _write_json(
        out / "lit1_lineage_funnel_phase1_metric_coverage.json",
        {"missing_required_metrics": [], "required_metric_status": {"scan_exposure_count": "present_zero_or_events_missing"}},
    )
    _write_json(
        out / "lit1_lineage_funnel_phase1_collapse_stage_summary.json",
        {
            "top1_dominance_generation": 1 if collapse else -1,
            "effective_lineage_collapse_generation": -1,
            "by_generation": [
                {
                    "generation": 0,
                    "top1_post_refill_alive_share": 0.1,
                    "effective_post_refill_alive_lineages": 12.0,
                },
                {
                    "generation": 1,
                    "top1_post_refill_alive_share": 0.2,
                    "effective_post_refill_alive_lineages": 10.0,
                },
            ],
        },
    )


def _phase1_scan(root: Path, *, run_dir: Path, schema: str = "4", scan_rows: int = 10) -> None:
    out = root / f"child_{run_dir.name}"
    _write_json(
        out / "lit1_lineage_funnel_phase1_summary.json",
        {
            "attempt_sources": ["schema_v4_scan_event"] if schema == "4" and scan_rows else ["generation_csv_birth_attempts"],
            "event_summary": {
                "event_file_present": True,
                "event_rows_read": 100,
                "event_schema_versions_seen": [schema],
                "event_type_counts": {
                    "reproduction_scan_exposure": scan_rows,
                    "reproduction_scan_eligible": 2 if scan_rows else 0,
                    "reproduction_scan_ineligible": 4 if scan_rows else 0,
                    "reproduction_scan_no_attempt": 3 if scan_rows else 0,
                    "reproduction_scan_attempt_recorded": 5 if scan_rows else 0,
                },
            },
        },
    )
    _write_json(
        out / "lit1_lineage_funnel_phase1_metric_coverage.json",
        {
            "missing_required_metrics": [],
            "required_metric_status": {
                "scan_exposure_count": "available_from_schema_v4_events"
                if schema == "4" and scan_rows
                else "present_zero_or_events_missing"
            },
        },
    )


def test_oes_litgate1_report_decides_a_when_gate_passes(tmp_path: Path) -> None:
    compact_root = tmp_path / "compact_phase1"
    scan_root = tmp_path / "scan_phase1"
    compact_runs = [tmp_path / "runs/r0001_a", tmp_path / "runs/r0002_b"]
    scan_runs = [tmp_path / "scan/r0001_a"]
    for run_dir in compact_runs:
        _phase1_compact(compact_root, run_dir=run_dir)
    for run_dir in scan_runs:
        _phase1_scan(scan_root, run_dir=run_dir)
    compact_summary = tmp_path / "compact_summary.json"
    scan_summary = tmp_path / "scan_summary.json"
    _campaign(compact_summary, run_dirs=compact_runs)
    _campaign(scan_summary, run_dirs=scan_runs)

    report = oes_report.build_report(
        compact_campaign_summary_path=compact_summary,
        compact_phase1_output_root=compact_root,
        scan_campaign_summary_path=scan_summary,
        scan_phase1_output_root=scan_root,
        expected_compact_runs=2,
        expected_scan_runs=1,
    )

    assert report["decision"]["decision"] == "OES-LITGATE1-A"
    assert report["decision"]["lit1_gate_status"] == "pass"
    assert report["decision"]["open_ended_outcomes_interpretable"] is True


def test_oes_litgate1_report_decides_b_on_lineage_collapse_warning(tmp_path: Path) -> None:
    compact_root = tmp_path / "compact_phase1"
    scan_root = tmp_path / "scan_phase1"
    compact_runs = [tmp_path / "runs/r0001_a"]
    scan_runs = [tmp_path / "scan/r0001_a"]
    _phase1_compact(compact_root, run_dir=compact_runs[0], collapse=True)
    _phase1_scan(scan_root, run_dir=scan_runs[0])
    compact_summary = tmp_path / "compact_summary.json"
    scan_summary = tmp_path / "scan_summary.json"
    _campaign(compact_summary, run_dirs=compact_runs)
    _campaign(scan_summary, run_dirs=scan_runs)

    report = oes_report.build_report(
        compact_campaign_summary_path=compact_summary,
        compact_phase1_output_root=compact_root,
        scan_campaign_summary_path=scan_summary,
        scan_phase1_output_root=scan_root,
        expected_compact_runs=1,
        expected_scan_runs=1,
    )

    assert report["decision"]["decision"] == "OES-LITGATE1-B"
    assert report["decision"]["lit1_gate_status"] == "warn"
    assert report["decision"]["open_ended_outcomes_interpretable"] is False


def test_oes_litgate1_report_decides_c_when_scan_v4_is_missing(tmp_path: Path) -> None:
    compact_root = tmp_path / "compact_phase1"
    scan_root = tmp_path / "scan_phase1"
    compact_runs = [tmp_path / "runs/r0001_a"]
    scan_runs = [tmp_path / "scan/r0001_a"]
    _phase1_compact(compact_root, run_dir=compact_runs[0])
    _phase1_scan(scan_root, run_dir=scan_runs[0], schema="3", scan_rows=0)
    compact_summary = tmp_path / "compact_summary.json"
    scan_summary = tmp_path / "scan_summary.json"
    _campaign(compact_summary, run_dirs=compact_runs)
    _campaign(scan_summary, run_dirs=scan_runs)

    report = oes_report.build_report(
        compact_campaign_summary_path=compact_summary,
        compact_phase1_output_root=compact_root,
        scan_campaign_summary_path=scan_summary,
        scan_phase1_output_root=scan_root,
        expected_compact_runs=1,
        expected_scan_runs=1,
    )

    assert report["decision"]["decision"] == "OES-LITGATE1-C"
    assert "scan_schema_v4_only" in report["decision"]["failed_criteria"]
    assert "scan_exposure_rows" in report["decision"]["failed_criteria"]
