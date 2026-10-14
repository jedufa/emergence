from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = PROJECT_ROOT / "tools/bin/oes_litgate3_report.py"
SPEC = importlib.util.spec_from_file_location("oes_litgate3_report", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
oes2 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(oes2)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _campaign(path: Path, run_id: str, run_dir: Path, failures: int = 0) -> None:
    _write_json(
        path,
        {
            "planned_runs": 1,
            "failures": failures,
            "results": [
                {
                    "index": 1,
                    "run_id": run_id,
                    "run_dir": str(run_dir),
                    "status": "ok" if failures == 0 else "failed",
                }
            ],
        },
    )


def _phase1(
    root: Path,
    run_dir: Path,
    *,
    event: bool = False,
    top1: float = 0.2,
    effective_lineages: float = 5.0,
    generation: int = 27,
) -> None:
    out = root / f"child_{run_dir.name}"
    out.mkdir(parents=True, exist_ok=True)
    _write_json(
        out / "lit1_lineage_funnel_phase1_summary.json",
        {
            "generation_count": 1,
            "lineage_generation_row_count": 2,
            "event_summary": {
                "event_file_present": event,
                "event_rows_read": 10 if event else 0,
                "event_schema_versions_seen": ["4"] if event else [],
                "event_type_counts": {
                    "reproduction_scan_exposure": 5 if event else 0,
                    "reproduction_scan_attempt_recorded": 3 if event else 0,
                },
                "unsupported_event_schema_versions": [],
            },
        },
    )
    _write_json(
        out / "lit1_lineage_funnel_phase1_metric_coverage.json",
        {"missing_required_metrics": []},
    )
    with (out / "lit1_lineage_funnel_phase1_by_generation.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        fields = [
            "generation",
            "lineage_id",
            "post_refill_alive_count",
            "post_refill_alive_share",
            "selected_parent_effective_lineage_count",
            "selected_parent_share",
            "offspring_share",
            "scan_exposure_share",
            "attempt_share",
            "refill_amplification_ratio",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        other = max(0.0, 1.0 - top1)
        writer.writerow(
            {
                "generation": generation,
                "lineage_id": "L1",
                "post_refill_alive_count": 8,
                "post_refill_alive_share": top1,
                "selected_parent_effective_lineage_count": effective_lineages,
                "selected_parent_share": 0.4,
                "offspring_share": 0.5,
                "scan_exposure_share": 0.4,
                "attempt_share": 0.4,
                "refill_amplification_ratio": 1.0,
            }
        )
        writer.writerow(
            {
                "generation": generation,
                "lineage_id": "L2",
                "post_refill_alive_count": 8,
                "post_refill_alive_share": other,
                "selected_parent_effective_lineage_count": effective_lineages,
                "selected_parent_share": 0.6,
                "offspring_share": 0.5,
                "scan_exposure_share": 0.6,
                "attempt_share": 0.6,
                "refill_amplification_ratio": 1.0,
            }
        )


def _repeat(root: Path) -> None:
    metrics = root / "children/0001/e/hash/bbatch/r/r0001/metrics"
    metrics.mkdir(parents=True, exist_ok=True)
    for name in [
        "lineage_reproduction_funnel_generation_summary.csv",
        "lineage_reproduction_funnel_run_summary.json",
    ]:
        (metrics / name).write_text("same\n", encoding="utf-8")
    (metrics.parent / "metrics.csv").write_text("same\n", encoding="utf-8")


def _build(
    tmp_path: Path,
    *,
    top1: float = 0.4,
    event: bool = True,
    repeat: bool = True,
    run_id: str | None = None,
    generation: int = 27,
) -> dict:
    run_id = run_id or (
        "oes_litgate3_gate_attached_search_20260710"
        "__context_wdl_delayed_hazard_full_stack__init_founder_balanced__seed000202"
    )
    compact_run = tmp_path / "runs/r0001"
    scan_run = tmp_path / "scan/r0001"
    compact_summary = tmp_path / "compact.json"
    scan_summary = tmp_path / "scan.json"
    _campaign(compact_summary, run_id, compact_run)
    _campaign(scan_summary, run_id, scan_run)
    _phase1(tmp_path / "compact_phase1", compact_run, top1=top1, generation=generation)
    _phase1(tmp_path / "scan_phase1", scan_run, event=event, top1=top1, generation=generation)
    repeat_a = repeat_b = None
    if repeat:
        repeat_a = tmp_path / "repeat_a"
        repeat_b = tmp_path / "repeat_b"
        _repeat(repeat_a)
        _repeat(repeat_b)
    return oes2.build_report(
        compact_campaign_summary_path=compact_summary,
        compact_phase1_output_root=tmp_path / "compact_phase1",
        scan_campaign_summary_path=scan_summary,
        scan_phase1_output_root=tmp_path / "scan_phase1",
        output_dir=tmp_path / "out",
        expected_compact_runs=1,
        expected_scan_runs=1,
        repeat_a=repeat_a,
        repeat_b=repeat_b,
    )


def test_oes_litgate3_decides_a_when_gate_passes(tmp_path: Path) -> None:
    report = _build(tmp_path)
    assert report["decision"]["decision"] == "OES-LITGATE3-A"
    assert report["decision"]["gate_result"] == "pass"


def test_oes_litgate3_accepts_exact_phase1_run_dir_names(tmp_path: Path) -> None:
    run_id = (
        "oes_litgate3_gate_attached_search_20260710"
        "__context_wdl_delayed_hazard_full_stack__init_founder_balanced__seed000202"
    )
    run_dir = tmp_path / "runs/r0001"
    compact_summary = tmp_path / "compact.json"
    scan_summary = tmp_path / "scan.json"
    _campaign(compact_summary, run_id, run_dir)
    _campaign(scan_summary, run_id, run_dir)
    _phase1(tmp_path / "compact_phase1", run_dir)
    exact_scan_dir = tmp_path / "scan_phase1" / run_dir.name
    exact_scan_dir.parent.mkdir(parents=True, exist_ok=True)
    _phase1(tmp_path, run_dir, event=True)
    (tmp_path / f"child_{run_dir.name}").rename(exact_scan_dir)
    repeat_a = tmp_path / "repeat_a"
    repeat_b = tmp_path / "repeat_b"
    _repeat(repeat_a)
    _repeat(repeat_b)

    report = oes2.build_report(
        compact_campaign_summary_path=compact_summary,
        compact_phase1_output_root=tmp_path / "compact_phase1",
        scan_campaign_summary_path=scan_summary,
        scan_phase1_output_root=tmp_path / "scan_phase1",
        output_dir=tmp_path / "out",
        expected_compact_runs=1,
        expected_scan_runs=1,
        repeat_a=repeat_a,
        repeat_b=repeat_b,
    )

    assert report["decision"]["decision"] in {"OES-LITGATE3-A", "OES-LITGATE3-B"}
    assert report["decision"]["failed_criteria"] == []


def test_oes_litgate3_decides_b_for_bounded_top1_warning(tmp_path: Path) -> None:
    report = _build(tmp_path, top1=0.85)
    assert report["decision"]["decision"] == "OES-LITGATE3-B"
    assert report["warning_rows"][0]["warning_stage"] == "mixed late dominance"
    assert report["caveat_assessment"]["stable_bounded_caveat"] is True
    assert (tmp_path / "out/oes_litgate3_caveat_assessment.json").exists()
    assert (tmp_path / "out/oes_litgate3_warning_stage_classification.json").exists()


def test_oes_litgate3_decides_c_when_seed202_scan_missing(tmp_path: Path) -> None:
    report = _build(tmp_path, top1=0.85, event=False)
    assert report["decision"]["decision"] == "OES-LITGATE3-C"
    assert "warning_scan_v4_missing" in report["decision"]["failed_criteria"]


def test_oes_litgate3_decides_c_when_warning_broadens(tmp_path: Path) -> None:
    run_id = (
        "oes_litgate3_gate_attached_search_20260710"
        "__context_wdl_delayed_hazard_full_stack__init_founder_balanced__seed000303"
    )
    report = _build(tmp_path, top1=0.85, run_id=run_id)
    assert report["decision"]["decision"] == "OES-LITGATE3-C"
    assert "warning_broadened" in report["decision"]["failed_criteria"]


def test_oes_litgate3_decides_c_when_warning_is_early(tmp_path: Path) -> None:
    report = _build(tmp_path, top1=0.85, generation=3)
    assert report["decision"]["decision"] == "OES-LITGATE3-C"
    assert "early_top1_warning" in report["decision"]["failed_criteria"]


def test_oes_litgate3_decides_d_when_repeat_parity_missing(tmp_path: Path) -> None:
    report = _build(tmp_path, repeat=False)
    assert report["decision"]["decision"] == "OES-LITGATE3-D"
    assert "repeat_parity" in report["decision"]["failed_criteria"]
