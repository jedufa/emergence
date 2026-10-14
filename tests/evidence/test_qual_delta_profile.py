from __future__ import annotations

import csv
import json
from pathlib import Path

from tools.evidence.contracts import check_campaign_schema
from tools.evidence.decisions import analyze_profile


def _write_run(root: Path, *, region_supported: int, no_integrity_failure: int = 1) -> Path:
    run = root / "run"
    run.mkdir(parents=True)
    metrics = run / "metrics.csv"
    fields = [
        "generation",
        "alive",
        "qf_route_receipt_present_flag",
        "qf_route_receipt_sha256_match_flag",
        "qf_route_binding_exact_flag",
        "qf_source_binding_exact_flag",
        "qf_coverage_complete_flag",
        "qf_repeat_parity_pass_flag",
        "qf_checkpoint_resume_pass_flag",
        "qf_finite_outputs_flag",
        "qf_no_integrity_failure_flag",
        "qf_no_posthoc_widening_flag",
        "qf_region_supported_flag",
    ]
    with metrics.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "generation": 1,
                "alive": 1,
                "qf_route_receipt_present_flag": 1,
                "qf_route_receipt_sha256_match_flag": 1,
                "qf_route_binding_exact_flag": 1,
                "qf_source_binding_exact_flag": 1,
                "qf_coverage_complete_flag": 1,
                "qf_repeat_parity_pass_flag": 1,
                "qf_checkpoint_resume_pass_flag": 1,
                "qf_finite_outputs_flag": 1,
                "qf_no_integrity_failure_flag": no_integrity_failure,
                "qf_no_posthoc_widening_flag": 1,
                "qf_region_supported_flag": region_supported,
            }
        )
    (run / "config.json").write_text(
        json.dumps({"sim_cfg": {}, "env_cfg": {}}, indent=2),
        encoding="utf-8",
    )
    return run


def _manifest(tmp_path: Path, run: Path) -> Path:
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "runs": [
                    {
                        "run_id": "qual-delta-synthetic",
                        "run_dir": str(run),
                        "status": "ok",
                        "returncode": 0,
                        "axes": {
                            "ecology_context": "qualification_delta",
                            "init_arm": "candidate",
                            "perception_condition": "normal",
                        },
                        "seed": 1,
                    }
                ]
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    manifest = tmp_path / "campaign_evidence_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "branch": "integration/eiel-qualification-framework-v1",
                "campaign_id": "qual-delta-synthetic",
                "completed_run_count": 1,
                "config_files": [],
                "config_glob": "config.json",
                "device": "synthetic",
                "failed_run_count": 0,
                "generations": 1,
                "git_commit": "e8ddd045791313f5cffbc4f61330274b248b88ac",
                "grid_size": 16,
                "harness_manifest_paths": [],
                "harness_summary_paths": [str(summary)],
                "metrics_glob": "metrics.csv",
                "output_paths": [],
                "planned_run_count": 1,
                "population_size": 1,
                "profile": "QUAL-DELTA-1",
                "registry_path": str(manifest),
                "run_root": str(tmp_path),
                "scenario_matrix": {
                    "dimensions": [
                        {"name": "ecology_context", "values": ["qualification_delta"]},
                        {"name": "init_arm", "values": ["candidate"]},
                        {"name": "perception_condition", "values": ["normal"]},
                    ]
                },
                "seed_list": [1],
                "steps_per_generation": 1,
                "timestamp_utc": "1970-01-01T00:00:00Z",
                "worktree": str(tmp_path),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return manifest


def test_qual_delta_a_accepts_only_complete_supported_region(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, _write_run(tmp_path, region_supported=1))
    schema = check_campaign_schema(manifest)

    decision = analyze_profile(schema, "QUAL-DELTA-1")

    assert schema["ok"]
    assert decision["decision"] == "A"
    assert decision["region_supported"] is True


def test_qual_delta_b_does_not_expand_when_region_not_supported(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, _write_run(tmp_path, region_supported=0))
    schema = check_campaign_schema(manifest)

    decision = analyze_profile(schema, "QUAL-DELTA-1")

    assert schema["ok"]
    assert decision["decision"] == "B"
    assert "do not expand" in " ".join(decision["claims_allowed"]).lower()


def test_qual_delta_d_for_integrity_failure(tmp_path: Path) -> None:
    manifest = _manifest(
        tmp_path,
        _write_run(tmp_path, region_supported=1, no_integrity_failure=0),
    )
    schema = check_campaign_schema(manifest)

    decision = analyze_profile(schema, "QUAL-DELTA-1")

    assert decision["decision"] == "D"
    assert decision["integrity_failure"] is True
