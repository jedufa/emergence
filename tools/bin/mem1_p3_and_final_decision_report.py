#!/usr/bin/env python3
from __future__ import annotations

"""Generate MEM1-P3 failure-mode audit and final MEM-1 decision package."""

import csv
import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any, Iterable, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BRIDGE_DIR = PROJECT_ROOT / "docs/900_lifecycle/940_incoming/memory_generalization_bridge_delayed_ecological_consequences"
MEM1_DIR = PROJECT_ROOT / "docs/900_lifecycle/940_incoming/mem1_memory_stepping_stone_consolidation"
P1_DIR = PROJECT_ROOT / "docs/900_lifecycle/940_incoming/mem1_p1_wdl_timing_cue_shortcut_preflight"
P2_DIR = PROJECT_ROOT / "docs/900_lifecycle/940_incoming/mem1_p2_drh0_readiness_audit"
P3_DIR = PROJECT_ROOT / "docs/900_lifecycle/940_incoming/mem1_p3_delayed_resource_waste_failure_mode_audit"
FINAL_DIR = PROJECT_ROOT / "docs/900_lifecycle/940_incoming/mem1_final_decision_package"

MEM1_REPORT = PROJECT_ROOT / "docs/400_implementation-engineering/410_implementation-reports/500_cognition-memory/EIEL-DOC-000451--eiel-mem1-memory-stepping-stone-decision-package-2026-08.md"
P3_REPORT = PROJECT_ROOT / "docs/500_campaigns-decisions-closeouts/520_campaign-reports/400_worlds-ecology/EIEL-DOC-000454--eiel-mem1-p3-delayed-resource-waste-failure-mode-audit-2026-07-09.md"
FINAL_REPORT = PROJECT_ROOT / "docs/400_implementation-engineering/410_implementation-reports/500_cognition-memory/EIEL-DOC-000450--eiel-mem1-final-memory-stepping-stone-decision-report-2026-07-09.md"

EVIDENCE_DIR = PROJECT_ROOT / "evidences/04-theme-4-cognition"
P3_EVIDENCE_MD = EVIDENCE_DIR / "75-mem1-p3-delayed-resource-waste-failure-mode-audit.md"
P3_EVIDENCE_ZIP = EVIDENCE_DIR / "75-mem1-p3-delayed-resource-waste-failure-mode-audit.evidence.zip"
FINAL_EVIDENCE_MD = EVIDENCE_DIR / "76-mem1-final-memory-stepping-stone-decision-package.md"
FINAL_EVIDENCE_ZIP = EVIDENCE_DIR / "76-mem1-final-memory-stepping-stone-decision-package.evidence.zip"

NO_CLAIMS = (
    "MEM-1 does not show intelligence, emergence, consciousness, planning, culture, "
    "prediction selection, general cognition, broad adaptive memory, learned refuge use, "
    "evolved refuge strategy, or superintelligence. It does not add direct cognition reward, "
    "memory reward, diversity reward, lineage rescue, protected reproduction, archive reseeding, "
    "or survival subsidy. The final decision only protects a bounded WDL delayed-hazard "
    "memory stepping stone and defines constrained next work."
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _by_world(rows: Iterable[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    return {str(row.get("world_context", "")): row for row in rows}


def build_p3_summary() -> dict[str, Any]:
    decision = read_json(BRIDGE_DIR / "decision.json")
    main_summary = read_json(BRIDGE_DIR / "main_summary.json")
    preflight = read_json(BRIDGE_DIR / "preflight_summary.json")
    parity = read_json(BRIDGE_DIR / "parity_summary.json")
    reward = read_json(BRIDGE_DIR / "reward_audit.json")
    integrity = read_json(BRIDGE_DIR / "demand_loop_integrity_summary.json")
    world_rows = read_csv(BRIDGE_DIR / "memory_effect_by_world.csv")
    interaction_rows = read_csv(BRIDGE_DIR / "generalization_interaction_summary.csv")
    behavior_rows = read_csv(BRIDGE_DIR / "behavior_readout_summary.csv")
    cost_rows = read_csv(BRIDGE_DIR / "cost_sanity_summary.csv")
    lineage_rows = read_csv(BRIDGE_DIR / "lineage_sanity_summary.csv")

    worlds = _by_world(world_rows)
    interactions = _by_world(interaction_rows)
    behavior = _by_world(behavior_rows)
    cost = _by_world(cost_rows)
    lineage = _by_world(lineage_rows)

    failure_modes = []
    resource = worlds.get("delayed_resource_depletion", {})
    waste = worlds.get("delayed_crowding_waste", {})
    flat = worlds.get("flat_control", {})
    resource_i = interactions.get("delayed_resource_depletion", {})
    waste_i = interactions.get("delayed_crowding_waste", {})
    resource_b = behavior.get("delayed_resource_depletion", {})
    waste_b = behavior.get("delayed_crowding_waste", {})

    if _float(resource_i.get("alive_interaction_vs_flat")) <= 0.25:
        failure_modes.append(
            {
                "world": "delayed_resource_depletion",
                "classification": "flat-control-like outcome with weak incremental alive interaction",
                "evidence": (
                    f"alive delta={resource.get('alive_delta_mean')}, flat alive delta={flat.get('alive_delta_mean')}, "
                    f"alive interaction={resource_i.get('alive_interaction_vs_flat')}"
                ),
            }
        )
    if _float(resource_b.get("delayed_resource_depletion_avoidance_proxy_delta_mean")) < 0.0:
        failure_modes.append(
            {
                "world": "delayed_resource_depletion",
                "classification": "behavior readout not directionally aligned with avoidance",
                "evidence": (
                    "avoidance proxy delta="
                    f"{resource_b.get('delayed_resource_depletion_avoidance_proxy_delta_mean')}"
                ),
            }
        )
    if _float(waste_i.get("alive_interaction_vs_flat")) < 0.0:
        failure_modes.append(
            {
                "world": "delayed_crowding_waste",
                "classification": "negative alive/birth interaction despite behavior and fitness readouts",
                "evidence": (
                    f"alive interaction={waste_i.get('alive_interaction_vs_flat')}, "
                    f"birth interaction={waste_i.get('successful_births_interaction_vs_flat')}, "
                    f"fitness interaction={waste_i.get('mean_fitness_interaction_vs_flat')}"
                ),
            }
        )
    if _float(waste_b.get("delayed_crowding_waste_agent_hidden_gap_exposure_count_delta_mean")) > 0.0:
        failure_modes.append(
            {
                "world": "delayed_crowding_waste",
                "classification": "behavior changed but not in an outcome-aligned survival direction",
                "evidence": (
                    f"hidden-gap exposure delta={waste_b.get('delayed_crowding_waste_agent_hidden_gap_exposure_count_delta_mean')}, "
                    f"active exposure delta={waste_b.get('delayed_crowding_waste_agent_active_exposure_count_delta_mean')}"
                ),
            }
        )

    p3_decision = {
        "label": "MEM1-P3-B",
        "status_after_slice": "ready_for_final_MEM-1-B",
        "reason": (
            "The delayed resource/waste bridge is integrity-clean and behavior-informative, but it does not "
            "provide outcome support for broad memory generalization. Resource depletion behaves near flat "
            "control on alive interaction, while crowding/waste has negative alive and birth interactions "
            "despite behavior/fitness readouts. This classifies the failure mode well enough to stop reruns "
            "and preserve WDL-specific claims."
        ),
        "recommended_next_step": "final MEM-1 decision packaging; do not rerun resource/waste before substrate redesign",
    }

    return {
        "schema_version": "mem1_p3_delayed_resource_waste_failure_mode_v1",
        "campaigns_run_this_slice": 0,
        "source_campaign": {
            "planned_runs": preflight.get("planned_runs"),
            "completed_runs": main_summary.get("completed_runs"),
            "failed_runs": main_summary.get("failed_runs"),
            "repeat_parity_completed_runs": parity.get("completed_runs"),
            "decision": decision.get("decision"),
            "decision_label": decision.get("label"),
            "integrity_ok": integrity.get("ok"),
            "reward_leaks": len(reward.get("reward_leaks", [])),
        },
        "world_effects": world_rows,
        "generalization_interactions": interaction_rows,
        "behavior_readouts": behavior_rows,
        "cost_sanity": cost_rows,
        "lineage_sanity": lineage_rows,
        "failure_modes": failure_modes,
        "decision": p3_decision,
        "mem1_interpretation": (
            "P3 blocks broad delayed ecological memory generalization. It does not invalidate WDL, "
            "because the bridge is a transfer failure rather than reward leakage, cost inversion, "
            "determinism failure, or WDL reproduction/survival contamination."
        ),
        "no_claims": NO_CLAIMS,
    }


def build_final_summary(p3_summary: Mapping[str, Any]) -> dict[str, Any]:
    p1 = read_json(P1_DIR / "mem1_p1_decision.json")
    p2 = read_json(P2_DIR / "mem1_p2_decision.json")
    p3 = p3_summary["decision"]
    confounds = read_json(MEM1_DIR / "confound_audit.json")
    minimum = read_json(MEM1_DIR / "minimum_conditions_checklist.json")
    mechanisms = read_json(MEM1_DIR / "mechanism_reconstruction_table.json")
    metrics = read_json(MEM1_DIR / "metric_availability_table.json")

    final_decision = {
        "label": "MEM-1-B",
        "plain_language": (
            "MEM-1 protects a bounded WDL delayed-hazard memory stepping stone with clean integrity and "
            "positive WDL-family support, but the mechanism and transfer evidence are not sufficient to "
            "justify broad memory generalization or a larger cognition expansion."
        ),
        "reason": (
            "P1 left current-state shortcut and hidden-gap primitive-action mechanism uncertainty unresolved; "
            "P2 found DRH0 ready only as a fixture-level future substrate; P3 classified delayed resource/waste "
            "as a behavior-informative but outcome-negative transfer failure. No fatal integrity, reward, cost, "
            "or parity failure was found."
        ),
        "recommendation": (
            "Proceed only to a narrow DRH0 non-fixture or registry-fixture memory preflight after reporting repair; "
            "do not launch broad memory generalization, prediction-memory, culture, or superintelligence campaigns."
        ),
    }

    requirements = [
        {
            "requirement": "reconstruct WDL memory-positive mechanism",
            "status": "partial",
            "evidence": "WDL cue/hidden/active substrate and selection/outcome support are reconstructed; primitive-action pathway remains unsupported.",
        },
        {
            "requirement": "identify minimum world-demand conditions",
            "status": "complete_bounded",
            "evidence": "Minimum conditions checklist plus P2 DRH0 condition table distinguish cue, hidden gap, consequence strength, action opportunity, controls, and repeat parity.",
        },
        {
            "requirement": "audit confounds",
            "status": "complete_for_current_evidence",
            "evidence": "Reward, cost, parity, run completeness, lineage, current-state shortcut, primitive-action, and transfer confounds are audited; shortcut remains a blocker for MEM-1-A.",
        },
        {
            "requirement": "test or design at least two bounded transfer/readiness probes",
            "status": "complete",
            "evidence": "P1 WDL shortcut preflight, P2 DRH0 readiness audit, and P3 resource/waste failure-mode audit are all recorded.",
        },
        {
            "requirement": "produce tracked docs and paired evidence",
            "status": "complete",
            "evidence": "Reports and evidence packages 72, 73, 74, 75, and 76 are under docs and evidences/04-theme-4-cognition.",
        },
        {
            "requirement": "reach one final MEM-1-A/B/C/D decision",
            "status": "complete",
            "evidence": "Final decision is MEM-1-B.",
        },
    ]

    return {
        "schema_version": "mem1_final_decision_package_v1",
        "final_decision": final_decision,
        "slice_decisions": {
            "MEM1-P1": p1.get("label"),
            "MEM1-P2": p2.get("label"),
            "MEM1-P3": p3.get("label"),
        },
        "requirements_audit": requirements,
        "supporting_tables": {
            "confound_audit_path": str(MEM1_DIR / "confound_audit.json"),
            "minimum_conditions_path": str(MEM1_DIR / "minimum_conditions_checklist.json"),
            "mechanism_reconstruction_path": str(MEM1_DIR / "mechanism_reconstruction_table.json"),
            "metric_availability_path": str(MEM1_DIR / "metric_availability_table.json"),
            "confound_entries": len(confounds.get("confounds", [])) if isinstance(confounds, dict) else 0,
            "minimum_condition_entries": len(minimum.get("conditions", [])) if isinstance(minimum, dict) else 0,
            "mechanism_entries": len(mechanisms.get("mechanism_steps", [])) if isinstance(mechanisms, dict) else 0,
            "metric_entries": len(metrics.get("metrics", [])) if isinstance(metrics, dict) else 0,
        },
        "no_claims": NO_CLAIMS,
    }


def render_p3_report(summary: Mapping[str, Any]) -> str:
    source = summary["source_campaign"]
    decision = summary["decision"]
    lines = [
        "# EIEL MEM-1-P3 Delayed Resource/Waste Failure-Mode Audit, 2026-07-09",
        "",
        "## Scope",
        "",
        "This is an offline failure-mode audit over the existing memory generalization bridge across "
        "delayed resource depletion and delayed crowding/waste. No new campaign was run.",
        "",
        f"- Source bridge planned/completed/failed: `{source['planned_runs']}` / `{source['completed_runs']}` / `{source['failed_runs']}`.",
        f"- Source decision: `{source['decision']}` ({source['decision_label']}).",
        f"- Integrity ok: `{source['integrity_ok']}`.",
        f"- Reward leaks: `{source['reward_leaks']}`.",
        f"- Repeat parity completed runs: `{source['repeat_parity_completed_runs']}`.",
        "",
        "## Failure Modes",
        "",
        "| World | Classification | Evidence |",
        "| --- | --- | --- |",
    ]
    for row in summary["failure_modes"]:
        lines.append(f"| `{row['world']}` | {row['classification']} | {row['evidence']} |")
    lines.extend(
        [
            "",
            "## World Effects",
            "",
            "| World | Alive delta | Birth delta | Offspring delta | Fitness delta |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in summary["world_effects"]:
        lines.append(
            f"| `{row['world_context']}` | {row['alive_delta_mean']} | {row['successful_births_delta_mean']} | "
            f"{row['mean_offspring_total_delta_mean']} | {row['mean_fitness_delta_mean']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            summary["mem1_interpretation"],
            "",
            "The failure is not an integrity failure. It is a transfer/readiness boundary: the non-hazard delayed "
            "ecological loops changed behavior readouts but did not produce aligned alive/birth outcome support "
            "beyond flat control. Rerunning before substrate redesign would be weak evidence.",
            "",
            "## Decision",
            "",
            f"- Decision: `{decision['label']}`.",
            f"- Status after slice: `{decision['status_after_slice']}`.",
            f"- Reason: {decision['reason']}",
            f"- Recommended next step: {decision['recommended_next_step']}.",
            "",
            "## Evidence",
            "",
            f"- Report path: `{P3_REPORT}`.",
            f"- Evidence markdown: `{P3_EVIDENCE_MD}`.",
            f"- Evidence zip: `{P3_EVIDENCE_ZIP}`.",
            "",
            "## No-Claims Statement",
            "",
            NO_CLAIMS,
            "",
        ]
    )
    return "\n".join(lines)


def render_final_report(summary: Mapping[str, Any]) -> str:
    decision = summary["final_decision"]
    lines = [
        "# EIEL MEM-1 Final Memory Stepping-Stone Decision Report, 2026-07-09",
        "",
        "## Final Decision",
        "",
        f"Final decision: `{decision['label']}`.",
        "",
        decision["plain_language"],
        "",
        f"Reason: {decision['reason']}",
        "",
        f"Recommendation: {decision['recommendation']}",
        "",
        "## Slice Decisions",
        "",
        f"- MEM1-P1: `{summary['slice_decisions']['MEM1-P1']}`.",
        f"- MEM1-P2: `{summary['slice_decisions']['MEM1-P2']}`.",
        f"- MEM1-P3: `{summary['slice_decisions']['MEM1-P3']}`.",
        "",
        "## Completion Audit",
        "",
        "| Requirement | Status | Evidence |",
        "| --- | --- | --- |",
    ]
    for row in summary["requirements_audit"]:
        lines.append(f"| {row['requirement']} | `{row['status']}` | {row['evidence']} |")
    lines.extend(
        [
            "",
            "## Decision Boundary",
            "",
            "`MEM-1-A` is not earned because behavior-before-consequence mechanism evidence and broad transfer "
            "support are insufficient. `MEM-1-C` is too weak because the WDL selection/outcome result is "
            "interpretable and integrity-clean at bounded scope. `MEM-1-D` is not supported because reward, "
            "cost, repeat parity, and run integrity checks did not invalidate the result.",
            "",
            "## Evidence",
            "",
            f"- Report path: `{FINAL_REPORT}`.",
            f"- Evidence markdown: `{FINAL_EVIDENCE_MD}`.",
            f"- Evidence zip: `{FINAL_EVIDENCE_ZIP}`.",
            "",
            "## No-Claims Statement",
            "",
            NO_CLAIMS,
            "",
        ]
    )
    return "\n".join(lines)


def update_mem1_report(final_summary: Mapping[str, Any], p3_summary: Mapping[str, Any]) -> None:
    final = final_summary["final_decision"]
    p3_decision = p3_summary["decision"]
    text = MEM1_REPORT.read_text(encoding="utf-8")
    text = text.replace("Working decision so far: provisional `MEM-1-C`.", f"Final decision: `{final['label']}`.")
    text = text.replace(
        "Status: Phase 0 complete; Phase 1 mechanism reconstruction advanced; Phase 2\nconfound audit drafted; Phase 4 transfer/readiness probes designed.",
        "Status: final MEM-1 decision package complete.",
    )
    marker = "## MEM-1 Final Decision, 2026-07-09"
    section = "\n".join(
        [
            "## MEM1-P3 Delayed Resource/Waste Failure-Mode Audit Update, 2026-07-09",
            "",
            f"Slice decision: `{p3_decision['label']}`.",
            "",
            "The delayed resource/waste bridge is classified as a behavior-informative but outcome-negative "
            "transfer failure. It does not invalidate WDL, but it blocks broad delayed ecological memory "
            "generalization without substrate redesign.",
            "",
            f"Report: `{P3_REPORT.relative_to(PROJECT_ROOT)}`.",
            "",
            marker,
            "",
            f"Final decision: `{final['label']}`.",
            "",
            final["plain_language"],
            "",
            f"Recommendation: {final['recommendation']}",
            "",
            f"Report: `{FINAL_REPORT.relative_to(PROJECT_ROOT)}`.",
            "",
        ]
    )
    if marker in text:
        start = text.rfind("## MEM1-P3 Delayed Resource/Waste Failure-Mode Audit Update, 2026-07-09")
        if start == -1:
            start = text.index(marker)
        next_marker = text.find("\n## No-Claims Statement", start)
        if next_marker == -1:
            text = text[:start].rstrip() + "\n\n" + section
        else:
            text = text[:start].rstrip() + "\n\n" + section + text[next_marker:]
    else:
        insert_before = "\n## No-Claims Statement"
        text = text.replace(insert_before, "\n" + section + insert_before, 1)
    MEM1_REPORT.write_text(text, encoding="utf-8")


def write_zip(zip_path: Path, files: Iterable[Path]) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in files:
            if path.exists() and path.is_file():
                zf.write(path, arcname=path.name)


def main() -> int:
    P3_DIR.mkdir(parents=True, exist_ok=True)
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

    p3_summary = build_p3_summary()
    final_summary = build_final_summary(p3_summary)

    P3_REPORT.write_text(render_p3_report(p3_summary), encoding="utf-8")
    FINAL_REPORT.write_text(render_final_report(final_summary), encoding="utf-8")
    P3_EVIDENCE_MD.write_text(P3_REPORT.read_text(encoding="utf-8"), encoding="utf-8")
    FINAL_EVIDENCE_MD.write_text(FINAL_REPORT.read_text(encoding="utf-8"), encoding="utf-8")
    update_mem1_report(final_summary, p3_summary)

    p3_summary_path = P3_DIR / "mem1_p3_failure_mode_summary.json"
    p3_decision_path = P3_DIR / "mem1_p3_decision.json"
    p3_modes_path = P3_DIR / "mem1_p3_failure_modes.json"
    final_summary_path = FINAL_DIR / "mem1_final_decision_summary.json"
    final_decision_path = FINAL_DIR / "mem1_final_decision.json"
    final_audit_path = FINAL_DIR / "mem1_final_completion_audit.json"
    p3_manifest_path = P3_DIR / "mem1_p3_evidence_manifest.json"
    final_manifest_path = FINAL_DIR / "mem1_final_evidence_manifest.json"

    write_json(p3_summary_path, p3_summary)
    write_json(p3_decision_path, p3_summary["decision"])
    write_json(p3_modes_path, {"failure_modes": p3_summary["failure_modes"]})
    write_json(final_summary_path, final_summary)
    write_json(final_decision_path, final_summary["final_decision"])
    write_json(final_audit_path, {"requirements": final_summary["requirements_audit"]})

    p3_zip_inputs = [
        P3_REPORT,
        P3_EVIDENCE_MD,
        p3_summary_path,
        p3_decision_path,
        p3_modes_path,
        BRIDGE_DIR / "decision.json",
        BRIDGE_DIR / "memory_effect_by_world.csv",
        BRIDGE_DIR / "generalization_interaction_summary.csv",
        BRIDGE_DIR / "behavior_readout_summary.csv",
        BRIDGE_DIR / "cost_sanity_summary.csv",
        BRIDGE_DIR / "lineage_sanity_summary.csv",
        BRIDGE_DIR / "reward_audit.json",
        BRIDGE_DIR / "parity_summary.json",
    ]
    write_zip(P3_EVIDENCE_ZIP, p3_zip_inputs)
    p3_manifest = {
        "schema_version": "mem1_p3_evidence_manifest_v1",
        "report_path": str(P3_REPORT),
        "evidence_markdown": str(P3_EVIDENCE_MD),
        "evidence_zip": str(P3_EVIDENCE_ZIP),
        "payload_zip_sha256_before_manifest": sha256_file(P3_EVIDENCE_ZIP),
        "support_files": [str(path) for path in p3_zip_inputs if path.exists()],
        "decision": p3_summary["decision"]["label"],
    }
    write_json(p3_manifest_path, p3_manifest)
    write_zip(P3_EVIDENCE_ZIP, [*p3_zip_inputs, p3_manifest_path])
    p3_manifest["final_evidence_zip_sha256"] = sha256_file(P3_EVIDENCE_ZIP)

    final_zip_inputs = [
        FINAL_REPORT,
        FINAL_EVIDENCE_MD,
        final_summary_path,
        final_decision_path,
        final_audit_path,
        P3_DIR / "mem1_p3_decision.json",
        P2_DIR / "mem1_p2_decision.json",
        P1_DIR / "mem1_p1_decision.json",
        MEM1_DIR / "confound_audit.json",
        MEM1_DIR / "minimum_conditions_checklist.json",
        MEM1_DIR / "mechanism_reconstruction_table.json",
        MEM1_DIR / "metric_availability_table.json",
    ]
    write_zip(FINAL_EVIDENCE_ZIP, final_zip_inputs)
    final_manifest = {
        "schema_version": "mem1_final_evidence_manifest_v1",
        "report_path": str(FINAL_REPORT),
        "evidence_markdown": str(FINAL_EVIDENCE_MD),
        "evidence_zip": str(FINAL_EVIDENCE_ZIP),
        "payload_zip_sha256_before_manifest": sha256_file(FINAL_EVIDENCE_ZIP),
        "support_files": [str(path) for path in final_zip_inputs if path.exists()],
        "decision": final_summary["final_decision"]["label"],
    }
    write_json(final_manifest_path, final_manifest)
    write_zip(FINAL_EVIDENCE_ZIP, [*final_zip_inputs, final_manifest_path])
    final_manifest["final_evidence_zip_sha256"] = sha256_file(FINAL_EVIDENCE_ZIP)

    print(
        json.dumps(
            {
                "p3": p3_manifest,
                "final": final_manifest,
                "final_decision": final_summary["final_decision"]["label"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
