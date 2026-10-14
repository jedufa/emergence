from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = PROJECT_ROOT / "tools/bin/lit1_lineage_funnel_phase1_audit.py"
SPEC = importlib.util.spec_from_file_location("lit1_lineage_funnel_phase1_audit", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
lit1_audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(lit1_audit)


CSV_FIELDS = [
    "schema_version",
    "generation",
    "lineage_id",
    "alive_count_before_selection",
    "alive_share_before_selection",
    "eligible_parent_count",
    "eligible_parent_share",
    "selected_parent_count",
    "selected_parent_share",
    "birth_attempts",
    "successful_births",
    "failed_births",
    "failed_births_by_reason_json",
    "offspring_produced_count",
    "offspring_share",
    "post_refill_alive_count",
    "post_refill_alive_share",
    "pre_cull_alive_share",
    "parent_child_distance_mean",
    "child_patch_id_or_coordinates",
    "child_basin_id",
    "food_child_site_mean",
    "waste_child_site_mean",
    "scent_child_site_mean",
    "shelter_cover_child_site_mean",
    "predator_hazard_exposure_child_site_mean",
    "first_generation_zero_selected_parents",
    "first_generation_zero_successful_births",
    "eligible_parent_share_over_alive_share",
    "selected_parent_share_over_eligible_parent_share",
    "offspring_share_over_selected_parent_share",
    "post_refill_share_over_pre_refill_alive_share",
]


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})


def _base_csv_rows() -> list[dict[str, object]]:
    return [
        {
            "schema_version": 1,
            "generation": 0,
            "lineage_id": "L00000001",
            "alive_count_before_selection": 2,
            "alive_share_before_selection": 0.5,
            "eligible_parent_count": 2,
            "eligible_parent_share": 0.5,
            "selected_parent_count": 3,
            "selected_parent_share": 0.75,
            "birth_attempts": 2,
            "successful_births": 1,
            "failed_births": 1,
            "failed_births_by_reason_json": '{"cooldown_age":1}',
            "offspring_produced_count": 1,
            "offspring_share": 0.25,
            "post_refill_alive_count": 9,
            "post_refill_alive_share": 0.9,
            "pre_cull_alive_share": 0.6,
            "parent_child_distance_mean": 1.5,
            "child_patch_id_or_coordinates": "2,3",
            "child_basin_id": "b0",
            "food_child_site_mean": 0.7,
            "waste_child_site_mean": 0.2,
            "scent_child_site_mean": 0.1,
            "shelter_cover_child_site_mean": 0.4,
            "predator_hazard_exposure_child_site_mean": 0.05,
            "first_generation_zero_selected_parents": -1,
            "first_generation_zero_successful_births": -1,
            "eligible_parent_share_over_alive_share": 1.0,
            "selected_parent_share_over_eligible_parent_share": 1.5,
            "offspring_share_over_selected_parent_share": 0.3333333333,
            "post_refill_share_over_pre_refill_alive_share": 1.5,
        },
        {
            "schema_version": 1,
            "generation": 0,
            "lineage_id": "L00000002",
            "alive_count_before_selection": 2,
            "alive_share_before_selection": 0.5,
            "eligible_parent_count": 2,
            "eligible_parent_share": 0.5,
            "selected_parent_count": 1,
            "selected_parent_share": 0.25,
            "birth_attempts": 1,
            "successful_births": 3,
            "failed_births": 0,
            "failed_births_by_reason_json": "{}",
            "offspring_produced_count": 3,
            "offspring_share": 0.75,
            "post_refill_alive_count": 1,
            "post_refill_alive_share": 0.1,
            "pre_cull_alive_share": 0.4,
            "parent_child_distance_mean": 2.5,
            "child_patch_id_or_coordinates": "4,5",
            "child_basin_id": "b1",
            "first_generation_zero_selected_parents": -1,
            "first_generation_zero_successful_births": -1,
            "eligible_parent_share_over_alive_share": 1.0,
            "selected_parent_share_over_eligible_parent_share": 0.5,
            "offspring_share_over_selected_parent_share": 3.0,
            "post_refill_share_over_pre_refill_alive_share": 0.25,
        },
    ]


def _write_events(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def test_phase1_audit_aggregates_schema_v4_scan_events(tmp_path: Path) -> None:
    generation_csv = tmp_path / "metrics/lineage_reproduction_funnel_generation_summary.csv"
    events_jsonl = tmp_path / "metrics/lineage_reproduction_funnel_events.jsonl"
    _write_csv(generation_csv, _base_csv_rows())
    _write_events(
        events_jsonl,
        [
            {
                "event_schema_version": 4,
                "generation": 0,
                "event_type": "reproduction_scan_exposure",
                "parent_lineage_id": "L00000001",
            },
            {
                "event_schema_version": 4,
                "generation": 0,
                "event_type": "reproduction_scan_exposure",
                "parent_lineage_id": "L00000001",
            },
            {
                "event_schema_version": 4,
                "generation": 0,
                "event_type": "reproduction_scan_eligible",
                "parent_lineage_id": "L00000001",
            },
            {
                "event_schema_version": 4,
                "generation": 0,
                "event_type": "reproduction_scan_attempt_recorded",
                "parent_lineage_id": "L00000001",
            },
            {
                "event_schema_version": 4,
                "generation": 0,
                "event_type": "reproduction_scan_exposure",
                "parent_lineage_id": "L00000002",
                "path2_roz_reporting_enable": 1,
            },
        ],
    )

    outputs = lit1_audit.build_phase1_outputs(generation_csv=generation_csv, events_jsonl=events_jsonl)
    rows = {row["lineage_id"]: row for row in outputs["by_generation"]}

    assert rows["L00000001"]["scan_exposure_count"] == 2
    assert rows["L00000001"]["scan_eligible_count"] == 1
    assert rows["L00000001"]["attempt_count"] == 1
    assert rows["L00000001"]["attempt_source"] == "schema_v4_scan_event"
    assert rows["L00000001"]["scan_exposure_share"] == pytest.approx(2 / 3)
    assert rows["L00000001"]["attempt_share"] == pytest.approx(1.0)
    assert rows["L00000001"]["scan_share_over_selected_parent_share"] == pytest.approx((2 / 3) / 0.75)
    assert rows["L00000001"]["attempt_share_over_scan_share"] == pytest.approx(1.0 / (2 / 3))
    assert json.loads(rows["L00000001"]["child_patch_context_summary"])["child_basin_id"] == "b0"
    assert outputs["collapse_stage_summary"]["top1_dominance_generation"] == 0
    assert outputs["collapse_stage_summary"]["effective_lineage_collapse_generation"] == 0
    assert outputs["summary"]["event_summary"]["path2_roz_event_rows"] == 1
    assert outputs["metric_coverage"]["missing_required_metrics"] == []


def test_phase1_audit_includes_event_only_lineages(tmp_path: Path) -> None:
    generation_csv = tmp_path / "metrics/lineage_reproduction_funnel_generation_summary.csv"
    events_jsonl = tmp_path / "metrics/lineage_reproduction_funnel_events.jsonl"
    _write_csv(generation_csv, _base_csv_rows())
    _write_events(
        events_jsonl,
        [
            {
                "event_schema_version": 4,
                "generation": 0,
                "event_type": "reproduction_scan_exposure",
                "parent_lineage_id": "L00000003",
            },
        ],
    )

    outputs = lit1_audit.build_phase1_outputs(generation_csv=generation_csv, events_jsonl=events_jsonl)
    rows = {row["lineage_id"]: row for row in outputs["by_generation"]}

    assert rows["L00000003"]["alive_count"] == 0
    assert rows["L00000003"]["scan_exposure_count"] == 1
    assert rows["L00000003"]["scan_exposure_share"] == 1.0


def test_phase1_audit_falls_back_to_csv_attempts_without_events(tmp_path: Path) -> None:
    generation_csv = tmp_path / "lineage_reproduction_funnel_generation_summary.csv"
    _write_csv(generation_csv, _base_csv_rows())

    outputs = lit1_audit.build_phase1_outputs(generation_csv=generation_csv)
    rows = {row["lineage_id"]: row for row in outputs["by_generation"]}

    assert rows["L00000001"]["attempt_count"] == 2
    assert rows["L00000001"]["attempt_share"] == pytest.approx(2 / 3)
    assert rows["L00000001"]["attempt_source"] == "generation_csv_birth_attempts"
    assert outputs["metric_coverage"]["required_metric_status"]["scan_exposure_count"] == "present_zero_or_events_missing"


def test_phase1_audit_fails_closed_for_unknown_event_schema(tmp_path: Path) -> None:
    generation_csv = tmp_path / "lineage_reproduction_funnel_generation_summary.csv"
    events_jsonl = tmp_path / "lineage_reproduction_funnel_events.jsonl"
    _write_csv(generation_csv, _base_csv_rows())
    _write_events(
        events_jsonl,
        [
            {
                "event_schema_version": 99,
                "generation": 0,
                "event_type": "reproduction_scan_exposure",
                "parent_lineage_id": "L00000001",
            },
        ],
    )

    with pytest.raises(ValueError, match="unsupported event_schema_version 99"):
        lit1_audit.build_phase1_outputs(generation_csv=generation_csv, events_jsonl=events_jsonl)

    outputs = lit1_audit.build_phase1_outputs(
        generation_csv=generation_csv,
        events_jsonl=events_jsonl,
        ignore_unsupported_event_schema=True,
    )
    assert outputs["summary"]["event_summary"]["unsupported_event_schema_versions"] == ["99"]


def test_phase1_audit_cli_writes_stable_outputs(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    run_dir = tmp_path / "run"
    generation_csv = run_dir / "metrics/lineage_reproduction_funnel_generation_summary.csv"
    _write_csv(generation_csv, _base_csv_rows())
    output_dir = tmp_path / "out"

    rc = lit1_audit.main(["--run-dir", str(run_dir), "--output-dir", str(output_dir)])

    assert rc == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["ok"] is True
    assert sorted(path.name for path in output_dir.iterdir()) == [
        "lit1_lineage_funnel_phase1_by_generation.csv",
        "lit1_lineage_funnel_phase1_collapse_stage_summary.json",
        "lit1_lineage_funnel_phase1_metric_coverage.json",
        "lit1_lineage_funnel_phase1_summary.json",
    ]
    summary = json.loads((output_dir / "lit1_lineage_funnel_phase1_summary.json").read_text(encoding="utf-8"))
    assert summary["mechanics_changes"] == "none_offline_reporting_helper_only"
    rows = list(csv.DictReader((output_dir / "lit1_lineage_funnel_phase1_by_generation.csv").open(encoding="utf-8")))
    assert list(rows[0]) == lit1_audit.BY_GENERATION_FIELDS
