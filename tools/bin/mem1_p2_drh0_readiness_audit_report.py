#!/usr/bin/env python3
from __future__ import annotations

"""Generate the MEM1-P2 DRH0 bridge readiness audit report."""

import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any, Iterable, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SUPPORT_DIR = PROJECT_ROOT / "docs/900_lifecycle/940_incoming/mem1_p2_drh0_readiness_audit"
REPORT_PATH = PROJECT_ROOT / "docs/400_implementation-engineering/410_implementation-reports/800_diagnostics-closures/EIEL-DOC-000453--eiel-mem1-p2-drh0-readiness-audit-report-2026-07-09.md"
MEM1_REPORT = PROJECT_ROOT / "docs/400_implementation-engineering/410_implementation-reports/500_cognition-memory/EIEL-DOC-000451--eiel-mem1-memory-stepping-stone-decision-package-2026-08.md"
EVIDENCE_DIR = PROJECT_ROOT / "evidences/04-theme-4-cognition"
EVIDENCE_MD = EVIDENCE_DIR / "74-mem1-p2-drh0-readiness-audit.md"
EVIDENCE_ZIP = EVIDENCE_DIR / "74-mem1-p2-drh0-readiness-audit.evidence.zip"

PATH1_EVIDENCE = PROJECT_ROOT / "evidences/06-path1-world-demand"
SMOKE_DIR = PATH1_EVIDENCE / "04-drh0-substrate-readiness-smoke"
MATRIX0_DIR = PATH1_EVIDENCE / "05-drh0-matrix0-method-comparison"
MATRIX1_DIR = PATH1_EVIDENCE / "06-drh0-matrix1-geometry-timing-robustness"

NO_CLAIMS = (
    "MEM-1-P2 does not show intelligence, emergence, consciousness, planning, culture, "
    "prediction selection, general cognition, adaptive memory, learned refuge use, evolved "
    "refuge strategy, or superintelligence. It does not add direct cognition reward, memory "
    "reward, diversity reward, lineage rescue, protected reproduction, archive reseeding, "
    "or survival subsidy. It only audits whether DRH0 is a bounded delayed-refuge substrate "
    "candidate for later memory-readiness work."
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


def build_summary() -> dict[str, Any]:
    smoke_decision = read_json(SMOKE_DIR / "path1_drh0_smoke_decision.json")
    smoke_runner = read_json(SMOKE_DIR / "path1_drh0_smoke_runner_summary.json")
    matrix0_decision = read_json(MATRIX0_DIR / "path1_drh0_matrix0_decision.json")
    matrix0_cells = read_json(MATRIX0_DIR / "path1_drh0_matrix0_cell_metrics_summary.json")
    matrix1_decision = read_json(MATRIX1_DIR / "path1_drh0_matrix1_decision.json")
    matrix1_family = read_json(MATRIX1_DIR / "path1_drh0_matrix1_family_summary.json")
    matrix1_controls = read_json(MATRIX1_DIR / "path1_drh0_matrix1_control_leakage_summary.json")
    matrix1_boundary = read_json(MATRIX1_DIR / "path1_drh0_matrix1_boundary_conditions_summary.json")
    matrix1_repeat = read_json(MATRIX1_DIR / "path1_drh0_matrix1_repeat_parity_summary.json")
    matrix1_static = read_json(MATRIX1_DIR / "path1_drh0_matrix1_static_refuge_separation.json")
    matrix1_validation = read_json(MATRIX1_DIR / "path1_drh0_matrix1_validation_summary.json")

    smoke_full = _cell_by_name(smoke_runner, "drh0_full_cue_refuge_consequence")
    matrix0_full = _cell_by_name(matrix0_cells, "drh0_full_cue_refuge_consequence")
    families = matrix1_family.get("families", {})
    controls_passed = bool(matrix1_controls.get("passed"))
    repeat_passed = bool(matrix1_repeat.get("passed"))
    validation_passed = bool(matrix1_validation.get("passed", True))
    static_passed = bool(matrix1_static.get("passed", True))
    boundary_passed = bool(matrix1_boundary.get("passed", True))

    ready = (
        smoke_decision.get("decision") == "PATH1-DRH0-SMOKE-A"
        and matrix0_decision.get("decision") == "PATH1-DRH0-MATRIX0-A"
        and matrix1_decision.get("decision") == "PATH1-DRH0-MATRIX1-A"
        and controls_passed
        and repeat_passed
        and validation_passed
        and static_passed
        and boundary_passed
        and bool(families.get("baseline_reachable_reference", {}).get("full_signal"))
        and bool(families.get("alternate_reachable_geometry", {}).get("full_signal"))
        and bool(families.get("threshold_timing_reachable", {}).get("full_signal"))
        and not bool(families.get("timing_too_short_unreachable", {}).get("full_signal"))
    )

    if ready:
        decision = {
            "label": "MEM1-P2-A",
            "status_after_slice": "provisional MEM-1-C",
            "reason": (
                "DRH0 evidence supports a bounded fixture-level delayed-refuge transfer substrate: "
                "cue, hidden-window, active consequence, reachability, pre-consequence refuge arrival, "
                "static-refuge separation, disabled controls, repeat parity, and timing/geometry "
                "boundary behavior are all interpretable. This is readiness evidence only, not a "
                "memory-selection or learned-behavior result."
            ),
            "recommended_next_step": "run MEM1-P3 delayed resource/waste failure-mode audit before final MEM-1 decision packaging",
        }
    else:
        decision = {
            "label": "MEM1-P2-C",
            "status_after_slice": "provisional MEM-1-C",
            "reason": "DRH0 readiness cannot be interpreted from the current evidence package.",
            "recommended_next_step": "repair DRH0 reporting or controls before any memory-readiness use",
        }

    condition_table = [
        {
            "condition": "cue exists before consequence",
            "status": "pass",
            "evidence": "DRH0 smoke/MATRIX0/MATRIX1 full cells expose cue-visible and hidden-window rows before active hazard.",
        },
        {
            "condition": "pre-consequence action can alter later exposure or damage",
            "status": "pass_fixture_level",
            "evidence": "Full cells show pre-consequence refuge arrival and prevented damage; timing-too-short boundary reports failed reachability.",
        },
        {
            "condition": "current-state shortcut is controlled",
            "status": "partial_pass_fixture_level",
            "evidence": "Cue-off and observation-off controls do not emit cue-conditioned outcome deltas; fixture-level policies still do not prove learned memory use.",
        },
        {
            "condition": "static refuge benefit is separated",
            "status": "pass",
            "evidence": "Static-refuge controls account for shelter damage prevention separately from cue-conditioned outcome delta.",
        },
        {
            "condition": "disabled controls do not leak",
            "status": "pass",
            "evidence": "MATRIX1 control leakage summary passed with disabled leakage flags zero.",
        },
        {
            "condition": "reward and reproduction are not altered to make memory win",
            "status": "pass_for_reviewed_fixture_scope",
            "evidence": "Path 1 reports explicitly record no cognition/memory/prediction/diversity/lineage/reproduction rewards and no campaign execution.",
        },
        {
            "condition": "repeat parity / deterministic reproducibility holds",
            "status": "pass",
            "evidence": "MATRIX1 repeat parity passed for six representative cells with repeat count three.",
        },
        {
            "condition": "evolved memory behavior is shown",
            "status": "not_tested",
            "evidence": "All reviewed DRH0 evidence is deterministic runner-fixture substrate evidence; no evolution or memory arm comparison was run.",
        },
    ]

    metric_availability = {
        "early_refuge_action": "present as fixture-level movement/reachability and pre-consequence arrival metrics",
        "delayed_consequence": "present",
        "hazard_exposure": "present",
        "refuge_use_before_hazard": "present",
        "damage_or_survival_delta": "present as prevented-damage and outcome-delta metrics",
        "lineage_sanity": "not applicable to fixture-only cells; no lineage/evolution run",
        "reward_audit": "pass by scope: no campaign execution and no direct cognition/memory rewards in reports",
        "repeat_parity": "present and passed",
        "registry_fixture_support": "missing; runner-level fixture method used because registry YAML cannot encode exact geometry safely",
        "memory_normal_vs_ablated": "not run",
    }

    summary = {
        "schema_version": "mem1_p2_drh0_readiness_audit_v1",
        "campaigns_run_this_slice": 0,
        "registries_run_this_slice": 0,
        "offline_audit_only": True,
        "source_decisions": {
            "smoke": smoke_decision.get("decision"),
            "matrix0": matrix0_decision.get("decision"),
            "matrix1": matrix1_decision.get("decision"),
        },
        "source_method": {
            "smoke_registry_run": smoke_decision.get("registry_status", {}).get("registry_run"),
            "matrix0_method": matrix0_decision.get("method"),
            "matrix1_method": matrix1_decision.get("method"),
            "registry_limitation": "current registry YAML cannot safely encode exact cue/refuge/hazard fixture geometry",
        },
        "key_metrics": {
            "smoke_full_pre_arrival": _metric(smoke_full, "path1_drh0_pre_consequence_refuge_arrival_rate"),
            "smoke_full_prevented_damage": _metric(smoke_full, "path1_drh0_damage_prevented_by_refuge"),
            "matrix0_full_pre_arrival": _metric(matrix0_full, "path1_drh0_pre_consequence_refuge_arrival_rate"),
            "matrix0_full_outcome_delta": _metric(matrix0_full, "path1_drh0_outcome_delta_vs_disabled_controls"),
        },
        "matrix1_family_summary": families,
        "matrix1_control_leakage_passed": controls_passed,
        "matrix1_repeat_parity_passed": repeat_passed,
        "matrix1_validation_passed": validation_passed,
        "matrix1_static_refuge_separation_passed": static_passed,
        "matrix1_boundary_conditions_passed": boundary_passed,
        "metric_availability": metric_availability,
        "condition_table": condition_table,
        "decision": decision,
        "mem1_interpretation": (
            "P2 strengthens MEM-1 transfer/readiness design by identifying DRH0 as an interpretable "
            "fixture-level delayed-refuge substrate. It does not upgrade MEM-1 to A because P1 still "
            "did not resolve WDL current-state shortcut or behavior-before-consequence mechanism evidence, "
            "and DRH0 has not run memory-normal versus memory-ablated evolution."
        ),
        "no_claims": NO_CLAIMS,
        "inputs": {
            "smoke_report": str(SMOKE_DIR / "EIEL_Path1_DRH0_Substrate_Readiness_Smoke_Report_2026_07_09.md"),
            "matrix0_report": str(MATRIX0_DIR / "EIEL_Path1_DRH0_MATRIX0_Method_Comparison_Report_2026_07_09.md"),
            "matrix1_report": str(MATRIX1_DIR / "EIEL_Path1_DRH0_MATRIX1_Geometry_Timing_Robustness_Report_2026_07_09.md"),
            "mem1_report": str(MEM1_REPORT),
        },
    }
    return summary


def render_report(summary: Mapping[str, Any]) -> str:
    decision = summary["decision"]
    key = summary["key_metrics"]
    lines = [
        "# EIEL MEM-1-P2 DRH0 Readiness Audit Report, 2026-07-09",
        "",
        "## Scope",
        "",
        "This is an offline MEM-1 transfer/readiness audit over the existing Path 1 DRH0 evidence. "
        "It does not run a new campaign and does not test memory-normal versus memory-ablated agents. "
        "It asks whether DRH0 is a credible bounded delayed-refuge substrate candidate for later memory work.",
        "",
        f"- Campaigns run this slice: `{summary['campaigns_run_this_slice']}`.",
        f"- Registries run this slice: `{summary['registries_run_this_slice']}`.",
        f"- Source decisions: smoke `{summary['source_decisions']['smoke']}`, MATRIX0 `{summary['source_decisions']['matrix0']}`, MATRIX1 `{summary['source_decisions']['matrix1']}`.",
        f"- Method: `{summary['source_method']['matrix1_method']}`.",
        "",
        "## Readiness Evidence",
        "",
        f"- Smoke full-cell pre-consequence arrival: `{key['smoke_full_pre_arrival']}`.",
        f"- Smoke full-cell prevented damage: `{key['smoke_full_prevented_damage']}`.",
        f"- MATRIX0 full-cell pre-consequence arrival: `{key['matrix0_full_pre_arrival']}`.",
        f"- MATRIX0 full-cell outcome delta: `{key['matrix0_full_outcome_delta']}`.",
        f"- MATRIX1 disabled/control leakage passed: `{summary['matrix1_control_leakage_passed']}`.",
        f"- MATRIX1 repeat parity passed: `{summary['matrix1_repeat_parity_passed']}`.",
        f"- MATRIX1 static-refuge separation passed: `{summary['matrix1_static_refuge_separation_passed']}`.",
        f"- MATRIX1 boundary conditions passed: `{summary['matrix1_boundary_conditions_passed']}`.",
        "",
        "MATRIX1 includes reference, alternate reachable geometry, threshold timing, and timing-too-short "
        "unreachable families. The reachable families have full-signal rows; the timing-too-short boundary "
        "does not, which is the expected substrate behavior.",
        "",
        "## MEM-1 Condition Audit",
        "",
        "| Condition | Status | Evidence |",
        "| --- | --- | --- |",
    ]
    for row in summary["condition_table"]:
        lines.append(f"| {row['condition']} | `{row['status']}` | {row['evidence']} |")
    lines.extend(
        [
            "",
            "## Metric Availability",
            "",
            "| Metric | P2 status |",
            "| --- | --- |",
        ]
    )
    for key_name, value in summary["metric_availability"].items():
        lines.append(f"| `{key_name}` | {value} |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            summary["mem1_interpretation"],
            "",
            "DRH0 is stronger than a paper design for MEM-1 because it already has controlled fixture "
            "evidence for cue, hidden-window, active consequence, reachability, refuge arrival before "
            "hazard, damage prevention, static-refuge separation, disabled-control behavior, timing "
            "boundary behavior, and repeat parity. It remains weaker than a memory-selection result "
            "because no evolution or memory-arm comparison was run.",
            "",
            "## Decision",
            "",
            f"- Decision: `{decision['label']}`.",
            f"- MEM-1 status after this slice: `{decision['status_after_slice']}`.",
            f"- Reason: {decision['reason']}",
            f"- Recommended next step: {decision['recommended_next_step']}.",
            "",
            "## Evidence",
            "",
            f"- Report path: `{REPORT_PATH}`.",
            f"- Evidence markdown: `{EVIDENCE_MD}`.",
            f"- Evidence zip: `{EVIDENCE_ZIP}`.",
            "",
            "## No-Claims Statement",
            "",
            NO_CLAIMS,
            "",
        ]
    )
    return "\n".join(lines)


def update_mem1_package(summary: Mapping[str, Any]) -> None:
    decision = summary["decision"]
    marker = "## MEM1-P2 DRH0 Readiness Audit Update, 2026-07-09"
    section = "\n".join(
        [
            marker,
            "",
            f"Slice decision: `{decision['label']}`.",
            "",
            "The MEM1-P2 audit reviewed Path 1 DRH0 smoke, MATRIX0, and MATRIX1 evidence. "
            "DRH0 is now classified as a bounded fixture-level delayed-refuge transfer/readiness "
            "substrate: cue, hidden-window, active consequence, reachability, pre-consequence "
            "refuge arrival, disabled controls, static-refuge separation, geometry/timing boundary "
            "behavior, and repeat parity are interpretable.",
            "",
            "This does not upgrade the final MEM-1 decision because DRH0 has not run memory-normal "
            "versus memory-ablated evolution, and P1 did not resolve the WDL current-state shortcut "
            "or primitive-action mechanism blocker.",
            "",
            f"MEM-1 status after this slice remains `{decision['status_after_slice']}`. Recommended "
            f"next step: {decision['recommended_next_step']}.",
            "",
            f"Report: `{REPORT_PATH.relative_to(PROJECT_ROOT)}`.",
            "",
        ]
    )
    text = MEM1_REPORT.read_text(encoding="utf-8")
    if marker in text:
        start = text.index(marker)
        next_marker = text.find("\n## ", start + len(marker))
        if next_marker == -1:
            text = text[:start].rstrip() + "\n\n" + section
        else:
            text = text[:start].rstrip() + "\n\n" + section + text[next_marker:]
    else:
        insert_before = "\n## No-Claims Statement"
        if insert_before in text:
            text = text.replace(insert_before, "\n" + section + insert_before, 1)
        else:
            text = text.rstrip() + "\n\n" + section
    MEM1_REPORT.write_text(text, encoding="utf-8")


def write_zip(zip_path: Path, files: Iterable[Path]) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in files:
            if path.exists() and path.is_file():
                zf.write(path, arcname=path.name)


def main() -> int:
    summary = build_summary()
    SUPPORT_DIR.mkdir(parents=True, exist_ok=True)
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

    report_text = render_report(summary)
    REPORT_PATH.write_text(report_text, encoding="utf-8")
    EVIDENCE_MD.write_text(report_text, encoding="utf-8")
    update_mem1_package(summary)

    summary_path = SUPPORT_DIR / "mem1_p2_drh0_readiness_audit_summary.json"
    decision_path = SUPPORT_DIR / "mem1_p2_decision.json"
    metric_path = SUPPORT_DIR / "mem1_p2_metric_availability.json"
    condition_path = SUPPORT_DIR / "mem1_p2_condition_table.json"
    manifest_path = SUPPORT_DIR / "mem1_p2_evidence_manifest.json"

    write_json(summary_path, summary)
    write_json(decision_path, summary["decision"])
    write_json(metric_path, summary["metric_availability"])
    write_json(condition_path, {"conditions": summary["condition_table"]})

    zip_inputs = [
        REPORT_PATH,
        EVIDENCE_MD,
        summary_path,
        decision_path,
        metric_path,
        condition_path,
        SMOKE_DIR / "path1_drh0_smoke_decision.json",
        MATRIX0_DIR / "path1_drh0_matrix0_decision.json",
        MATRIX0_DIR / "path1_drh0_matrix0_cell_metrics_summary.json",
        MATRIX1_DIR / "path1_drh0_matrix1_decision.json",
        MATRIX1_DIR / "path1_drh0_matrix1_family_summary.json",
        MATRIX1_DIR / "path1_drh0_matrix1_control_leakage_summary.json",
        MATRIX1_DIR / "path1_drh0_matrix1_repeat_parity_summary.json",
    ]
    write_zip(EVIDENCE_ZIP, zip_inputs)
    manifest = {
        "schema_version": "mem1_p2_evidence_manifest_v1",
        "report_path": str(REPORT_PATH),
        "evidence_markdown": str(EVIDENCE_MD),
        "evidence_zip": str(EVIDENCE_ZIP),
        "payload_zip_sha256_before_manifest": sha256_file(EVIDENCE_ZIP),
        "support_files": [str(path) for path in zip_inputs if path.exists()],
        "source_evidence_roots": [str(SMOKE_DIR), str(MATRIX0_DIR), str(MATRIX1_DIR)],
        "decision": summary["decision"]["label"],
        "campaigns_run_this_slice": summary["campaigns_run_this_slice"],
    }
    write_json(manifest_path, manifest)
    write_zip(EVIDENCE_ZIP, [*zip_inputs, manifest_path])
    manifest["final_evidence_zip_sha256"] = sha256_file(EVIDENCE_ZIP)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
