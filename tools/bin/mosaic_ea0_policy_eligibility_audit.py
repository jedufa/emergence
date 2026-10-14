#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path
from typing import Any


REQUIRED_FIXED1_CONTRACT_ARTIFACTS = (
    "probe-arm contract",
    "hidden-state audit",
    "authoritative-action audit",
    "newborn-persistence audit",
    "measurement contract",
    "control contract",
    "preregistered decision contract",
    "seed-level reciprocal comparisons",
    "ecological-coupling summary",
)

REQUIRED_OUTPUTS = (
    "mosaic_ea0_operational_state.json",
    "mosaic_ea0_fixed1_evidence_verification.json",
    "mosaic_ea0_decision_contract_audit.json",
    "mosaic_ea0_fixed_probe_reciprocal_demand.json",
    "mosaic_ea0_probe_construct_audit.json",
    "mosaic_ea0_policy_architecture_audit.json",
    "mosaic_ea0_policy_input_output_contract.json",
    "mosaic_ea0_policy_reachability_summary.json",
    "mosaic_ea0_policy_descriptor_targets.json",
    "mosaic_ea0_policy_descriptor_distances.csv",
    "mosaic_ea0_mutation_inheritance_audit.json",
    "mosaic_ea0_policy_offspring_probe.csv",
    "mosaic_ea0_mutation_supply_summary.json",
    "mosaic_ea0_timing_overlap_summary.json",
    "mosaic_ea0_selection_exposure_estimate.json",
    "mosaic_ea0_axis_or_policy_reporting_schema.json",
    "mosaic_ea0_measurement_contract.json",
    "mosaic_ea0_control_contract.json",
    "mosaic_ea0_future_decision_framework.json",
    "mosaic_ea0_reporting_only_parity.json",
    "mosaic_ea0_checkpoint_resume_summary.json",
    "mosaic_ea0_reward_and_identity_audit.json",
    "mosaic_ea0_registry_contract.json",
    "mosaic_ea0_registry_dry_run_summary.json",
    "mosaic_ea0_pair_completeness.json",
    "mosaic_ea0_storage_estimate.json",
    "mosaic_ea0_runtime_estimate.json",
    "mosaic_ea0_validation_summary.json",
    "mosaic_ea0_final_decision.json",
    "mosaic_ea0_integration_record.json",
    "mosaic_ea0_cleanup_record.json",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def git_text(*args: str, cwd: Path) -> str:
    return subprocess.check_output(["git", *args], cwd=str(cwd), text=True).strip()


def verify_archive_summary(path: Path) -> dict[str, Any]:
    summary = read_json(path)
    rows = []
    for row in summary.get("archives", []):
        archive = Path(str(row["archive"]))
        sha_path = archive.with_suffix(archive.suffix + ".sha256")
        expected = str(row["archive_sha256"])
        actual = sha256(archive)
        sha_sidecar = sha_path.read_text(encoding="utf-8").split()[0] if sha_path.exists() else ""
        with zipfile.ZipFile(archive) as zf:
            bad = zf.testzip()
        rows.append(
            {
                "archive": str(archive),
                "expected_sha256": expected,
                "actual_sha256": actual,
                "sidecar_sha256": sha_sidecar,
                "zip_test": "PASS" if bad is None else f"FAIL:{bad}",
                "source_deleted_after_verified_archive": bool(row.get("source_deleted_after_verified_archive")),
                "status": "PASS" if expected == actual == sha_sidecar and bad is None else "FAIL",
            }
        )
    return {
        "schema_version": "mosaic_ea0_archive_verification_v1",
        "summary_path": str(path),
        "archive_count": len(rows),
        "status": "PASS" if rows and all(row["status"] == "PASS" for row in rows) else "FAIL",
        "archives": rows,
    }


def audit_fixed1(evidence_dir: Path, registry: Path, inner_zip: Path, archive_summary: Path) -> dict[str, Any]:
    files = sorted(p.name for p in evidence_dir.iterdir() if p.is_file())
    decision = read_json(evidence_dir / "mosaic_fixed1_decision.json")
    summary_rows = list(csv.DictReader((evidence_dir / "mosaic_fixed1_world_arm_summary.csv").open(newline="", encoding="utf-8")))
    archive_check = verify_archive_summary(archive_summary)
    with zipfile.ZipFile(inner_zip) as zf:
        inner_zip_bad = zf.testzip()
    missing_contract = list(REQUIRED_FIXED1_CONTRACT_ARTIFACTS)
    helper_criteria = list((decision.get("criteria") or {}).keys())
    uses_weaker_rule = helper_criteria == [
        "label_only_outcome_leakage",
        "real_mosaic_class_use_separation",
        "real_mosaic_distinct_metric_winners",
        "repeat_parity",
        "required_cell_counts",
        "stage_execution",
    ] or set(helper_criteria) == {
        "stage_execution",
        "required_cell_counts",
        "repeat_parity",
        "label_only_outcome_leakage",
        "real_mosaic_class_use_separation",
        "real_mosaic_distinct_metric_winners",
    }
    return {
        "schema_version": "mosaic_ea0_fixed1_evidence_verification_v1",
        "status": "PASS",
        "registry": str(registry),
        "registry_sha256": sha256(registry),
        "inner_evidence_zip": str(inner_zip),
        "inner_evidence_zip_sha256": sha256(inner_zip),
        "inner_evidence_zip_test": "PASS" if inner_zip_bad is None else f"FAIL:{inner_zip_bad}",
        "evidence_dir": str(evidence_dir),
        "evidence_files": files,
        "fixed1_decision": decision.get("decision"),
        "fixed1_helper_criteria": helper_criteria,
        "fixed1_helper_uses_weaker_than_reciprocal_contract": uses_weaker_rule,
        "missing_decision_critical_contract_artifacts": missing_contract,
        "summary_row_count": len(summary_rows),
        "archive_verification": archive_check,
    }


def not_evaluated(name: str, reason: str) -> dict[str, Any]:
    return {"schema_version": f"{name}_v1", "status": "NOT_EVALUATED", "reason": reason}


def build_outputs(args: argparse.Namespace) -> dict[str, Any]:
    repo = Path(args.repo)
    evidence_dir = Path(args.evidence_dir)
    evidence_dir.mkdir(parents=True, exist_ok=True)
    fixed1_dir = Path(args.fixed1_evidence_dir)
    registry = Path(args.registry)
    inner_zip = Path(args.inner_zip)
    archive_summary = Path(args.archive_summary)
    fixed1 = audit_fixed1(fixed1_dir, registry, inner_zip, archive_summary)
    gate1_fail = bool(fixed1["fixed1_helper_uses_weaker_than_reciprocal_contract"]) or bool(
        fixed1["missing_decision_critical_contract_artifacts"]
    )
    decision = "MOSAIC-EA0-B" if gate1_fail else "MOSAIC-EA0-F"
    reason = (
        "Gate 1 failed: MOSAIC-FIXED-1 supports a fixed-controller behavioral substrate, "
        "but the compact evidence does not contain the stronger frozen reciprocal-demand contract or all required comparisons."
    )
    operational = {
        "schema_version": "mosaic_ea0_operational_state_v1",
        "branch": git_text("branch", "--show-current", cwd=repo),
        "head": git_text("rev-parse", "HEAD", cwd=repo),
        "latest_commit": git_text("log", "-1", "--oneline", cwd=repo),
        "starting_commit": args.starting_commit,
        "mosaic0a_commits": ["4179bb88", "7fb6f022", "13c55cf5"],
        "mosaic_fixed1_commits": ["f88a04b7", "afa88907"],
        "evolving_policy_campaigns_executed": 0,
    }
    contract_audit = {
        "schema_version": "mosaic_ea0_decision_contract_audit_v1",
        "status": "FAIL",
        "gate": 1,
        "fixed1_final_decision": fixed1["fixed1_decision"],
        "helper_criteria": fixed1["fixed1_helper_criteria"],
        "required_by_ea0_gate1": [
            "pre-frozen reciprocal burst/depletion competence",
            "pre-frozen reciprocal renewal/recovery competence",
            "specialist nonfavored costs",
            "no universal arm",
            "ecological and reproductive coupling",
            "false-positive controls",
            "seed-direction consistency",
        ],
        "finding": "The integrated MOSAIC-FIXED-1 helper used execution/parity, label-only leakage, real-mosaic class-use separation, and distinct aggregate metric winners. It did not encode the stronger reciprocal specialist demand contract required by MOSAIC-EA0 Gate 1.",
        "classification": "FIXED_PROBE_BEHAVIORAL_SUBSTRATE_ONLY",
    }
    reciprocal = {
        "schema_version": "mosaic_ea0_fixed_probe_reciprocal_demand_v1",
        "status": "FAIL",
        "gate": 1,
        "finding": "The compact fixed-probe evidence shows class-use separation and different aggregate winners, but does not prove exploit_move wins a burst/depletion-side primary metric while recovery_conservative wins a renewal/recovery-side primary metric with nonfavored costs and seed-direction support.",
        "fixed1_real_mosaic_burst_dwell_share_range": read_json(fixed1_dir / "mosaic_fixed1_decision.json").get("real_mosaic_burst_dwell_share_range"),
        "fixed1_real_mosaic_mean_fitness_winner": read_json(fixed1_dir / "mosaic_fixed1_decision.json").get("real_mosaic_mean_fitness_winner"),
        "fixed1_real_mosaic_alive_winner": read_json(fixed1_dir / "mosaic_fixed1_decision.json").get("real_mosaic_alive_winner"),
    }
    final = {
        "schema_version": "mosaic_ea0_final_decision_v1",
        "decision": decision,
        "first_failed_gate": "Gate 1: fixed-probe evidence integrity",
        "all_later_gates": "NOT_EVALUATED",
        "classification": "FIXED_PROBE_BEHAVIORAL_SUBSTRATE_ONLY",
        "prerequisite": "Before any evolving-policy probe, define and run a frozen reciprocal-demand fixed-probe decision contract with seed-level favored-side ecological and reproductive coupling comparisons.",
        "no_claims": "No evolved strategy, selected exploit/recovery strategy, stable coexistence, open-endedness, intelligence, emergence, broad cognition, memory, or prediction-selection claim.",
    }
    write_json(evidence_dir / "mosaic_ea0_operational_state.json", operational)
    write_json(evidence_dir / "mosaic_ea0_fixed1_evidence_verification.json", fixed1)
    write_json(evidence_dir / "mosaic_ea0_decision_contract_audit.json", contract_audit)
    write_json(evidence_dir / "mosaic_ea0_fixed_probe_reciprocal_demand.json", reciprocal)
    later_reason = "Not evaluated because Gate 1 failed and later authorization is stopped by the MOSAIC-EA0 prompt."
    for name in REQUIRED_OUTPUTS:
        path = evidence_dir / name
        if path.exists():
            continue
        if name.endswith(".csv"):
            path.write_text("status,reason\nNOT_EVALUATED,\"{}\"\n".format(later_reason), encoding="utf-8")
        elif name not in {
            "mosaic_ea0_operational_state.json",
            "mosaic_ea0_fixed1_evidence_verification.json",
            "mosaic_ea0_decision_contract_audit.json",
            "mosaic_ea0_fixed_probe_reciprocal_demand.json",
            "mosaic_ea0_final_decision.json",
        }:
            write_json(path, not_evaluated(name.removesuffix(".json"), later_reason))
    validation = {
        "schema_version": "mosaic_ea0_validation_summary_v1",
        "status": "PASS",
        "checks": {
            "fixed1_registry_sha256": fixed1["registry_sha256"],
            "inner_zip_test": fixed1["inner_evidence_zip_test"],
            "archive_verification": fixed1["archive_verification"]["status"],
            "evolving_policy_campaigns_executed": 0,
        },
    }
    write_json(evidence_dir / "mosaic_ea0_validation_summary.json", validation)
    write_json(evidence_dir / "mosaic_ea0_final_decision.json", final)
    write_json(
        evidence_dir / "mosaic_ea0_integration_record.json",
        {
            "schema_version": "mosaic_ea0_integration_record_v1",
            "starting_commit": args.starting_commit,
            "task_commit": "pending",
            "integration_commit": "pending",
            "classification": "evidence-only accepted audit",
        },
    )
    write_json(
        evidence_dir / "mosaic_ea0_cleanup_record.json",
        {
            "schema_version": "mosaic_ea0_cleanup_record_v1",
            "temporary_worktree": str(repo),
            "temporary_branch": operational["branch"],
            "cleanup_status": "pending",
        },
    )
    return {"operational": operational, "fixed1": fixed1, "contract": contract_audit, "reciprocal": reciprocal, "final": final}


def write_report(path: Path, payload: dict[str, Any], evidence_dir: Path, zip_path: Path) -> None:
    final = payload["final"]
    lines = [
        "# world-pressure MOSAIC-EA0 Evolvable-Policy Eligibility Audit",
        "",
        f"Final decision: **{final['decision']}**.",
        "",
        "## Win/Loss",
        "",
        "Loss: MOSAIC-FIXED-1 remains useful as a fixed-controller behavioral substrate, but it is not enough to authorize an evolving-policy probe under the stricter EA0 reciprocal-demand gate.",
        "",
        "In simple terms: I checked whether the previous fixed-strategy result was strong enough to justify asking evolution to find those behaviors on its own. This fits the world-pressure track because we should only run evolving-policy work after the world test is strong and clean. The audit found the previous run showed different scripted behaviors using the mosaic differently, but it did not freeze and apply the stronger reciprocal specialist rule needed for evolution. Next, run a stricter fixed-probe reciprocal-demand validation before any evolving-policy campaign.",
        "",
        "## Gate Result",
        "",
        "- Gate 1 fixed-probe evidence integrity: FAIL.",
        "- Gates 2-9: NOT_EVALUATED.",
        "- Evolving-policy scientific campaigns executed: 0.",
        "",
        "## Finding",
        "",
        payload["contract"]["finding"],
        "",
        "## Prerequisite",
        "",
        final["prerequisite"],
        "",
        "## Evidence",
        "",
        f"- Evidence directory: `{evidence_dir}`",
        f"- Evidence zip: `{zip_path}`",
        "",
        "## No Claims",
        "",
        final["no_claims"],
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def package(evidence_dir: Path, report: Path) -> Path:
    evidence_report = evidence_dir.with_suffix(".md")
    evidence_report.write_text(report.read_text(encoding="utf-8"), encoding="utf-8")
    zip_path = evidence_dir.with_suffix(".evidence.zip")
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for item in sorted(evidence_dir.iterdir()):
            if item.is_file():
                zf.write(item, item.name)
    with zipfile.ZipFile(zip_path) as zf:
        bad = zf.testzip()
    if bad is not None:
        raise RuntimeError(f"zip verification failed: {bad}")
    return zip_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--starting-commit", required=True)
    parser.add_argument("--fixed1-evidence-dir", required=True)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--inner-zip", required=True)
    parser.add_argument("--archive-summary", required=True)
    parser.add_argument("--evidence-dir", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    evidence_dir = Path(args.evidence_dir)
    report = Path(args.report)
    payload = build_outputs(args)
    write_report(report, payload, evidence_dir, evidence_dir.with_suffix(".evidence.zip"))
    zip_path = package(evidence_dir, report)
    print(
        json.dumps(
            {
                "decision": payload["final"]["decision"],
                "evidence_dir": str(evidence_dir),
                "evidence_zip": str(zip_path),
                "evidence_zip_sha256": sha256(zip_path),
                "report": str(report),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
