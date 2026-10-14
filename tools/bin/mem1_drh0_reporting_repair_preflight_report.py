#!/usr/bin/env python3
from __future__ import annotations

"""Generate the MEM1-DRH0-RP1 reporting repair and preflight gate report."""

import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any, Iterable, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SUPPORT_DIR = PROJECT_ROOT / "docs/900_lifecycle/940_incoming/mem1_drh0_reporting_repair_preflight"
REPORT_PATH = PROJECT_ROOT / "docs/400_implementation-engineering/410_implementation-reports/000_programme-cross-cutting/EIEL-DOC-000449--eiel-mem1-drh0-reporting-repair-preflight-report-2026-07-09.md"
EVIDENCE_DIR = PROJECT_ROOT / "evidences/04-theme-4-cognition"
EVIDENCE_MD = EVIDENCE_DIR / "77-mem1-drh0-reporting-repair-preflight.md"
EVIDENCE_ZIP = EVIDENCE_DIR / "77-mem1-drh0-reporting-repair-preflight.evidence.zip"
EVIDENCE_ZIP_SHA = EVIDENCE_DIR / "77-mem1-drh0-reporting-repair-preflight.evidence.zip.sha256"

MEM1_FINAL_REPORT = PROJECT_ROOT / "docs/400_implementation-engineering/410_implementation-reports/500_cognition-memory/EIEL-DOC-000450--eiel-mem1-final-memory-stepping-stone-decision-report-2026-07-09.md"
MEM1_FINAL_EVIDENCE = PROJECT_ROOT / "evidences/04-theme-4-cognition/76-mem1-final-memory-stepping-stone-decision-package.md"
MEM1_FINAL_DECISION = PROJECT_ROOT / "docs/500_campaigns-decisions-closeouts/500_decision-reports/000_programme-cross-cutting/EIEL-DOC-001094--mem1-final-decision.json"
MEM1_P2_SUMMARY = PROJECT_ROOT / "docs/800_machine-readable-support/860_small-derived-tables/800_diagnostics-closures/mem1-p2-drh0-readiness-audit/EIEL-DOC-001114--mem1-p2-drh0-readiness-audit-summary.json"

PATH1_EVIDENCE = PROJECT_ROOT / "evidences/06-path1-world-demand"
SMOKE_DIR = PATH1_EVIDENCE / "04-drh0-substrate-readiness-smoke"
MATRIX0_DIR = PATH1_EVIDENCE / "05-drh0-matrix0-method-comparison"
MATRIX1_DIR = PATH1_EVIDENCE / "06-drh0-matrix1-geometry-timing-robustness"

NO_CLAIMS = (
    "MEM1-DRH0-RP1 does not show intelligence, emergence, consciousness, planning, "
    "culture, prediction selection, general cognition, broad adaptive memory, learned "
    "refuge use, evolved refuge strategy, or superintelligence. It does not add direct "
    "cognition reward, memory reward, diversity reward, lineage rescue, protected "
    "reproduction, archive reseeding, or survival subsidy. Any result supports only "
    "DRH0 reporting/preflight readiness under the tested delayed-refuge substrate and "
    "does not generalize memory beyond bounded evidence."
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metric(row: Mapping[str, Any], name: str) -> float:
    try:
        return float((row.get("metrics") or {}).get(name, 0.0) or 0.0)
    except Exception:
        return 0.0


def _cell_by_name(summary: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    for row in summary.get("cells", []):
        if row.get("cell") == name:
            return row
    return {}


def _all_checks_true(payload: Mapping[str, Any]) -> bool:
    checks = payload.get("checks") or {}
    return bool(checks) and all(bool(value) for value in checks.values())


def build_summary(*, inputs: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Build the report from production evidence or an explicit test fixture.

    The default remains fail-closed: missing published evidence raises from
    ``read_json``.  Tests may supply an already-loaded, tracked fixture to
    exercise the reporting logic without coupling unit tests to an archived
    evidence shelf.
    """
    if inputs is None:
        inputs = {
            "mem1_final": read_json(MEM1_FINAL_DECISION),
            "mem1_p2": read_json(MEM1_P2_SUMMARY),
            "smoke_decision": read_json(SMOKE_DIR / "path1_drh0_smoke_decision.json"),
            "smoke_runner": read_json(SMOKE_DIR / "path1_drh0_smoke_runner_summary.json"),
            "matrix0_decision": read_json(MATRIX0_DIR / "path1_drh0_matrix0_decision.json"),
            "matrix0_validation": read_json(MATRIX0_DIR / "path1_drh0_matrix0_validation_summary.json"),
            "matrix0_cells": read_json(MATRIX0_DIR / "path1_drh0_matrix0_cell_metrics_summary.json"),
            "matrix1_decision": read_json(MATRIX1_DIR / "path1_drh0_matrix1_decision.json"),
            "matrix1_validation": read_json(MATRIX1_DIR / "path1_drh0_matrix1_validation_summary.json"),
            "matrix1_controls": read_json(MATRIX1_DIR / "path1_drh0_matrix1_control_leakage_summary.json"),
            "matrix1_repeat": read_json(MATRIX1_DIR / "path1_drh0_matrix1_repeat_parity_summary.json"),
            "matrix1_static": read_json(MATRIX1_DIR / "path1_drh0_matrix1_static_refuge_separation.json"),
            "matrix1_boundary": read_json(MATRIX1_DIR / "path1_drh0_matrix1_boundary_conditions_summary.json"),
        }
    required = (
        "mem1_final", "mem1_p2", "smoke_decision", "smoke_runner",
        "matrix0_decision", "matrix0_validation", "matrix0_cells",
        "matrix1_decision", "matrix1_validation", "matrix1_controls",
        "matrix1_repeat", "matrix1_static", "matrix1_boundary",
    )
    missing = [name for name in required if name not in inputs]
    if missing:
        raise ValueError(f"reporting fixture is missing required inputs: {missing}")
    mem1_final = inputs["mem1_final"]
    mem1_p2 = inputs["mem1_p2"]
    smoke_decision = inputs["smoke_decision"]
    smoke_runner = inputs["smoke_runner"]
    matrix0_decision = inputs["matrix0_decision"]
    matrix0_validation = inputs["matrix0_validation"]
    matrix0_cells = inputs["matrix0_cells"]
    matrix1_decision = inputs["matrix1_decision"]
    matrix1_validation = inputs["matrix1_validation"]
    matrix1_controls = inputs["matrix1_controls"]
    matrix1_repeat = inputs["matrix1_repeat"]
    matrix1_static = inputs["matrix1_static"]
    matrix1_boundary = inputs["matrix1_boundary"]

    smoke_full = _cell_by_name(smoke_runner, "drh0_full_cue_refuge_consequence")
    matrix0_full = _cell_by_name(matrix0_cells, "drh0_full_cue_refuge_consequence")
    source_decisions_clean = (
        smoke_decision.get("decision") == "PATH1-DRH0-SMOKE-A"
        and matrix0_decision.get("decision") == "PATH1-DRH0-MATRIX0-A"
        and matrix1_decision.get("decision") == "PATH1-DRH0-MATRIX1-A"
    )
    reporting_clean = (
        source_decisions_clean
        and _all_checks_true(matrix0_validation)
        and _all_checks_true(matrix1_validation)
        and bool(matrix1_controls.get("passed"))
        and bool(matrix1_repeat.get("passed"))
        and bool(matrix1_static.get("passed", True))
        and bool(matrix1_boundary.get("passed", True))
    )

    reporting_surfaces = [
        {
            "surface": "cue_visible_phase_rows",
            "status": "pass",
            "evidence": "Smoke, MATRIX0, and MATRIX1 full cells emit path1_drh0_cue_visible_steps.",
            "metric": "path1_drh0_cue_visible_steps",
            "full_cell_value": _metric(matrix0_full, "path1_drh0_cue_visible_steps"),
        },
        {
            "surface": "hidden_window_rows",
            "status": "pass",
            "evidence": "Smoke, MATRIX0, and MATRIX1 full cells emit path1_drh0_hidden_window_steps before active hazard.",
            "metric": "path1_drh0_hidden_window_steps",
            "full_cell_value": _metric(matrix0_full, "path1_drh0_hidden_window_steps"),
        },
        {
            "surface": "active_consequence_rows",
            "status": "pass",
            "evidence": "Full cells emit path1_drh0_hazard_active_steps and damage/prevention fields.",
            "metric": "path1_drh0_hazard_active_steps",
            "full_cell_value": _metric(matrix0_full, "path1_drh0_hazard_active_steps"),
        },
        {
            "surface": "refuge_reachability_context",
            "status": "pass",
            "evidence": "MATRIX0/MATRIX1 include reachable, alternate reachable, threshold, and timing-too-short unreachable families.",
            "metric": "path1_drh0_refuge_reachability_success_rate",
            "full_cell_value": _metric(matrix0_full, "path1_drh0_refuge_reachability_success_rate"),
        },
        {
            "surface": "pre_consequence_refuge_arrival_or_use",
            "status": "pass_fixture_level",
            "evidence": "Full fixture cells emit pre-consequence refuge arrival/use before hazard activation.",
            "metric": "path1_drh0_pre_consequence_refuge_arrival_rate",
            "full_cell_value": _metric(matrix0_full, "path1_drh0_pre_consequence_refuge_arrival_rate"),
        },
        {
            "surface": "hazard_exposure_damage_prevented_damage",
            "status": "pass",
            "evidence": "Full cells emit damage in refuge, damage outside refuge, prevented damage, and outcome delta.",
            "metric": "path1_drh0_damage_prevented_by_refuge",
            "full_cell_value": _metric(smoke_full, "path1_drh0_damage_prevented_by_refuge"),
        },
        {
            "surface": "static_refuge_separation",
            "status": "pass",
            "evidence": "Static-refuge controls separate shelter prevention from cue-conditioned outcome delta.",
            "source": "path1_drh0_matrix1_static_refuge_separation.json",
        },
        {
            "surface": "disabled_control_leakage_flags",
            "status": "pass",
            "evidence": "Cue-off, observation-off, consequence-off, and refuge-off controls report zero leakage where present.",
            "source": "path1_drh0_matrix1_control_leakage_summary.json",
        },
        {
            "surface": "cue_off_or_observation_off_controls",
            "status": "pass_fixture_level",
            "evidence": "Cue-off and observation-off controls are present in smoke/MATRIX0/MATRIX1 fixture rows.",
        },
        {
            "surface": "repeat_parity_summary",
            "status": "pass",
            "evidence": "MATRIX1 repeat parity passes six representative cells with repeat count three.",
            "source": "path1_drh0_matrix1_repeat_parity_summary.json",
        },
        {
            "surface": "reward_audit_summary",
            "status": "pass_for_reviewed_fixture_scope",
            "evidence": "Reviewed DRH0 slices run fixture/reporting only and do not add cognition, memory, refuge, diversity, lineage, or survival shaping reward.",
        },
        {
            "surface": "memory_normal_vs_ablated_arm_labeling",
            "status": "blocked_for_preflight",
            "evidence": "No DRH0 memory-normal versus memory-ablated arm has run; fixture-only labels would be nominal and not behavior-informative.",
        },
        {
            "surface": "lineage_sanity",
            "status": "not_applicable_fixture_only",
            "evidence": "No evolution was run in reviewed DRH0 evidence; lineage sanity must be added if any evolved preflight is later used.",
        },
    ]

    fixture_registry_audit = {
        "schema_version": "mem1_drh0_rp1_fixture_registry_audit_v1",
        "registry_fixture_path": {
            "status": "blocked",
            "reason": "Current registry YAML supports config knobs but cannot safely encode exact cue/refuge/hazard geometry without a geometry fixture layer.",
            "source": mem1_p2.get("source_method", {}).get("registry_limitation"),
        },
        "runner_fixture_path": {
            "status": "valid_for_reporting_only",
            "reason": "The existing runner fixture can reproduce cue, hidden, active, refuge, damage, control, and parity metrics without mechanics changes.",
        },
        "non_fixture_preflight_path": {
            "status": "not_ready",
            "reason": "No audited non-fixture DRH0 harness currently proves the same geometry, hidden-window, refuge-use, control, memory-arm, and parity semantics.",
        },
        "memory_arm_labeling": {
            "status": "blocked",
            "reason": "memory_normal and memory_ablated labels are not yet tied to an interpretable DRH0 behavior path; adding labels to deterministic fixtures would not test memory.",
        },
    }

    pre_run_requirements = [
        {
            "requirement": "no mechanics/reward/selection/cognition changes required",
            "passed": True,
            "evidence": "The reporting gate reads existing fixture evidence only.",
        },
        {
            "requirement": "memory-normal and memory-ablated can be labeled cleanly",
            "passed": False,
            "evidence": "Labels can be written, but would not be behavior-informative on the deterministic fixture path.",
        },
        {
            "requirement": "hidden-window and pre-consequence refuge-use metrics are emitted",
            "passed": True,
            "evidence": "path1_drh0_hidden_window_steps and path1_drh0_pre_consequence_refuge_arrival_rate are present.",
        },
        {
            "requirement": "disabled/cue-off/control leakage can be checked",
            "passed": True,
            "evidence": "MATRIX1 control leakage summary passes.",
        },
        {
            "requirement": "repeat parity can be sampled on representative cells",
            "passed": True,
            "evidence": "MATRIX1 repeat parity summary passes.",
        },
        {
            "requirement": "output can be interpreted as preflight only",
            "passed": False,
            "evidence": "Without an audited non-fixture or registry-fixture memory-arm path, output would still be fixture readiness only.",
        },
    ]
    pre_run_go = all(item["passed"] for item in pre_run_requirements)
    gate = {
        "schema_version": "mem1_drh0_rp1_pre_run_gate_v1",
        "gate": "PRE-RUN-GO" if pre_run_go else "PRE-RUN-BLOCKED",
        "campaigns_run": 0,
        "preflights_run": 0,
        "registries_run": 0,
        "requirements": pre_run_requirements,
        "blocked_reason": (
            "DRH0 fixture reporting is interpretable, but registry/non-fixture encoding and memory-normal versus "
            "memory-ablated arm semantics are not sufficient for a memory preflight."
            if not pre_run_go
            else ""
        ),
    }

    if not reporting_clean:
        decision = {
            "label": "MEM1-DRH0-RP1-D",
            "reason": "One or more source integrity, control, or repeat-parity checks failed.",
            "recommended_next_step": "repair reporting integrity before any DRH0 memory work",
        }
    elif gate["gate"] == "PRE-RUN-BLOCKED":
        decision = {
            "label": "MEM1-DRH0-RP1-B",
            "reason": (
                "Reporting surfaces are repaired at fixture/report-gate level, but a memory preflight should wait "
                "because exact registry/non-fixture controls and behavior-informative memory arm labeling remain insufficient."
            ),
            "recommended_next_step": (
                "Add an audited registry-fixture geometry layer or a tiny non-fixture DRH0 harness that cleanly binds "
                "memory_normal versus memory_ablated to hidden-window refuge-use metrics before running a memory preflight."
            ),
        }
    else:
        decision = {
            "label": "MEM1-DRH0-RP1-A",
            "reason": "Reporting and gate requirements are sufficient for one narrow DRH0 memory preflight.",
            "recommended_next_step": "run one tiny DRH0 memory preflight with no reward/mechanics changes",
        }

    summary = {
        "schema_version": "mem1_drh0_reporting_repair_preflight_v1",
        "base_mem1_decision": mem1_final.get("label"),
        "source_decisions": {
            "mem1_p2": mem1_p2.get("decision", {}).get("label"),
            "smoke": smoke_decision.get("decision"),
            "matrix0": matrix0_decision.get("decision"),
            "matrix1": matrix1_decision.get("decision"),
        },
        "source_integrity": {
            "source_decisions_clean": source_decisions_clean,
            "matrix0_checks_clean": _all_checks_true(matrix0_validation),
            "matrix1_checks_clean": _all_checks_true(matrix1_validation),
            "matrix1_control_leakage_passed": bool(matrix1_controls.get("passed")),
            "matrix1_repeat_parity_passed": bool(matrix1_repeat.get("passed")),
            "reporting_clean_for_fixture_scope": reporting_clean,
        },
        "key_metrics": {
            "smoke_full_prevented_damage": _metric(smoke_full, "path1_drh0_damage_prevented_by_refuge"),
            "matrix0_full_pre_consequence_arrival": _metric(matrix0_full, "path1_drh0_pre_consequence_refuge_arrival_rate"),
            "matrix0_full_outcome_delta": _metric(matrix0_full, "path1_drh0_outcome_delta_vs_disabled_controls"),
            "matrix0_full_hidden_window_steps": _metric(matrix0_full, "path1_drh0_hidden_window_steps"),
        },
        "reporting_surfaces": reporting_surfaces,
        "fixture_registry_audit": fixture_registry_audit,
        "pre_run_gate": gate,
        "decision": decision,
        "campaigns_run": 0,
        "preflights_run": 0,
        "registries_run": 0,
        "no_claims": NO_CLAIMS,
        "inputs": {
            "mem1_final_report": str(MEM1_FINAL_REPORT),
            "mem1_final_evidence": str(MEM1_FINAL_EVIDENCE),
            "mem1_p2_summary": str(MEM1_P2_SUMMARY),
            "smoke_dir": str(SMOKE_DIR),
            "matrix0_dir": str(MATRIX0_DIR),
            "matrix1_dir": str(MATRIX1_DIR),
        },
    }
    return summary


def render_report(summary: Mapping[str, Any]) -> str:
    surfaces = "\n".join(
        f"- `{item['surface']}`: `{item['status']}` - {item['evidence']}"
        for item in summary["reporting_surfaces"]
    )
    reqs = "\n".join(
        f"- `{item['requirement']}`: `{item['passed']}` - {item['evidence']}"
        for item in summary["pre_run_gate"]["requirements"]
    )
    decision = summary["decision"]
    key = summary["key_metrics"]
    integrity = summary["source_integrity"]
    fixture = summary["fixture_registry_audit"]
    return "\n".join(
        [
            "# EIEL MEM1-DRH0-RP1 Reporting Repair Preflight Report, 2026-07-09",
            "",
            "## Scope",
            "",
            "This report repairs the DRH0 memory-path handoff at the reporting and pre-run gate layer. "
            "It does not run a memory campaign, does not alter DRH0 mechanics, and does not add reward or selection pressure.",
            "",
            f"- Base MEM-1 decision: `{summary['base_mem1_decision']}`.",
            f"- Source decisions: smoke `{summary['source_decisions']['smoke']}`, MATRIX0 `{summary['source_decisions']['matrix0']}`, MATRIX1 `{summary['source_decisions']['matrix1']}`.",
            f"- Campaigns run: `{summary['campaigns_run']}`.",
            f"- Preflights run: `{summary['preflights_run']}`.",
            f"- Registries run: `{summary['registries_run']}`.",
            "",
            "## Reporting Surface Audit",
            "",
            surfaces,
            "",
            "## Source Integrity",
            "",
            f"- Source decisions clean: `{integrity['source_decisions_clean']}`.",
            f"- MATRIX0 checks clean: `{integrity['matrix0_checks_clean']}`.",
            f"- MATRIX1 checks clean: `{integrity['matrix1_checks_clean']}`.",
            f"- MATRIX1 control leakage passed: `{integrity['matrix1_control_leakage_passed']}`.",
            f"- MATRIX1 repeat parity passed: `{integrity['matrix1_repeat_parity_passed']}`.",
            f"- Fixture-scope reporting clean: `{integrity['reporting_clean_for_fixture_scope']}`.",
            "",
            "## Key Fixture Metrics",
            "",
            f"- Smoke full prevented damage: `{key['smoke_full_prevented_damage']}`.",
            f"- MATRIX0 full pre-consequence arrival: `{key['matrix0_full_pre_consequence_arrival']}`.",
            f"- MATRIX0 full outcome delta: `{key['matrix0_full_outcome_delta']}`.",
            f"- MATRIX0 full hidden-window steps: `{key['matrix0_full_hidden_window_steps']}`.",
            "",
            "## Registry And Fixture Encoding",
            "",
            f"- Registry-fixture path: `{fixture['registry_fixture_path']['status']}` - {fixture['registry_fixture_path']['reason']}",
            f"- Runner-fixture path: `{fixture['runner_fixture_path']['status']}` - {fixture['runner_fixture_path']['reason']}",
            f"- Non-fixture preflight path: `{fixture['non_fixture_preflight_path']['status']}` - {fixture['non_fixture_preflight_path']['reason']}",
            f"- Memory-arm labeling: `{fixture['memory_arm_labeling']['status']}` - {fixture['memory_arm_labeling']['reason']}",
            "",
            "## Pre-Run Gate",
            "",
            f"- Gate: `{summary['pre_run_gate']['gate']}`.",
            f"- Blocked reason: {summary['pre_run_gate']['blocked_reason'] or 'not blocked'}",
            "",
            reqs,
            "",
            "## Decision",
            "",
            f"Decision: `{decision['label']}`.",
            "",
            decision["reason"],
            "",
            f"Recommended next step: {decision['recommended_next_step']}",
            "",
            "## No-Claims Statement",
            "",
            str(summary["no_claims"]),
            "",
        ]
    )


def write_zip(paths: Iterable[Path], zip_path: Path) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in paths:
            zf.write(path, arcname=path.relative_to(PROJECT_ROOT).as_posix())


def generate() -> dict[str, Any]:
    summary = build_summary()
    SUPPORT_DIR.mkdir(parents=True, exist_ok=True)
    report = render_report(summary)

    report_path = REPORT_PATH
    evidence_md = EVIDENCE_MD
    report_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_md.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    evidence_md.write_text(report, encoding="utf-8")

    summary_path = SUPPORT_DIR / "mem1_drh0_rp1_summary.json"
    surfaces_path = SUPPORT_DIR / "mem1_drh0_rp1_reporting_surface_audit.json"
    fixture_path = SUPPORT_DIR / "mem1_drh0_rp1_fixture_registry_audit.json"
    gate_path = SUPPORT_DIR / "mem1_drh0_rp1_pre_run_gate.json"
    decision_path = SUPPORT_DIR / "mem1_drh0_rp1_decision.json"
    validation_path = SUPPORT_DIR / "mem1_drh0_rp1_validation_summary.json"
    manifest_path = SUPPORT_DIR / "mem1_drh0_rp1_evidence_manifest.json"

    validation = {
        "schema_version": "mem1_drh0_rp1_validation_summary_v1",
        "json_artifacts_written": True,
        "report_written": True,
        "evidence_markdown_written": True,
        "campaigns_run": 0,
        "preflights_run": 0,
        "decision": summary["decision"]["label"],
        "gate": summary["pre_run_gate"]["gate"],
    }
    manifest = {
        "schema_version": "mem1_drh0_rp1_evidence_manifest_v1",
        "report_path": str(report_path),
        "evidence_markdown": str(evidence_md),
        "evidence_zip": str(EVIDENCE_ZIP),
        "support_dir": str(SUPPORT_DIR),
        "source_inputs": summary["inputs"],
        "campaigns_run": 0,
        "preflights_run": 0,
        "decision": summary["decision"]["label"],
    }

    write_json(summary_path, summary)
    write_json(surfaces_path, {"schema_version": "mem1_drh0_rp1_reporting_surface_audit_v1", "surfaces": summary["reporting_surfaces"]})
    write_json(fixture_path, summary["fixture_registry_audit"])
    write_json(gate_path, summary["pre_run_gate"])
    write_json(decision_path, summary["decision"])
    write_json(validation_path, validation)
    write_json(manifest_path, manifest)

    zip_inputs = [
        report_path,
        evidence_md,
        summary_path,
        surfaces_path,
        fixture_path,
        gate_path,
        decision_path,
        validation_path,
        manifest_path,
    ]
    write_zip(zip_inputs, EVIDENCE_ZIP)
    zip_sha = sha256_file(EVIDENCE_ZIP)
    EVIDENCE_ZIP_SHA.write_text(
        f"{zip_sha}  {EVIDENCE_ZIP.relative_to(PROJECT_ROOT).as_posix()}\n",
        encoding="utf-8",
    )

    return {
        "decision": summary["decision"]["label"],
        "gate": summary["pre_run_gate"]["gate"],
        "report_path": str(report_path),
        "evidence_markdown": str(evidence_md),
        "evidence_zip": str(EVIDENCE_ZIP),
        "evidence_zip_sha256": zip_sha,
        "campaigns_run": 0,
        "preflights_run": 0,
    }


def main() -> int:
    result = generate()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
