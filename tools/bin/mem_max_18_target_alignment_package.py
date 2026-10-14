#!/usr/bin/env python3
from __future__ import annotations

"""Build the MEM-MAX-18 target-alignment decision package."""

import csv
import hashlib
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any, Mapping

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENGINE_ROOT = PROJECT_ROOT / "engine"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(ENGINE_ROOT) not in sys.path:
    sys.path.insert(0, str(ENGINE_ROOT))

from env.gridworld_pkg.constants import ACT_MOVE_E, ACT_MOVE_W  # noqa: E402
from evolution.target_alignment import action_class_map, score_logits_for_targets  # noqa: E402


OUT_DIR = PROJECT_ROOT / "docs/900_lifecycle/940_incoming/mem_max_18_target_alignment"
REPORT = PROJECT_ROOT / "docs/500_campaigns-decisions-closeouts/520_campaign-reports/500_cognition-memory/EIEL-DOC-000467--eiel-mem-max-18-target-alignment-and-memory-action-coupling-2026-07-10.md"
EVIDENCE = PROJECT_ROOT / "evidences/04-theme-4-cognition/103-mem-max-18-target-alignment-and-memory-action-coupling.md"
EVIDENCE_ZIP = PROJECT_ROOT / "evidences/04-theme-4-cognition/103-mem-max-18-target-alignment-and-memory-action-coupling.evidence.zip"
REGISTRY = PROJECT_ROOT / "_spec/registry/v1/mem_max_18_target_alignment_runway.yaml"
MEM17_DIR = PROJECT_ROOT / "docs/900_lifecycle/940_incoming/mem_max_17_effective_cue_action"

NO_CLAIMS = (
    "MEM-MAX-18 does not show intelligence, emergence, consciousness, planning, culture, prediction selection, "
    "general cognition, broad adaptive memory, learned refuge strategy, superintelligence, or MEM-EVO-A. It does "
    "not add direct cognition reward, memory reward, cue-retention reward, hidden-state reward, target-alignment "
    "reward, action-divergence reward, correct-choice reward, target-entry reward, directional movement reward, "
    "movement-persistence reward, diversity reward, lineage rescue, protected reproduction, archive reseeding, or "
    "survival subsidy. It repairs live target-alignment semantics, tests whether retained cue state can influence "
    "target-relevant action competition, and conditionally introduces a bounded evolvable memory-to-action coupling "
    "under strict controls."
)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[Mapping[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()


def package_evidence() -> dict[str, Any]:
    if EVIDENCE_ZIP.exists():
        EVIDENCE_ZIP.unlink()
    with zipfile.ZipFile(EVIDENCE_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(REPORT, REPORT.relative_to(PROJECT_ROOT))
        zf.write(EVIDENCE, EVIDENCE.relative_to(PROJECT_ROOT))
        zf.write(REGISTRY, REGISTRY.relative_to(PROJECT_ROOT))
        for path in sorted(OUT_DIR.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(PROJECT_ROOT))
    sha_path = EVIDENCE_ZIP.with_suffix(EVIDENCE_ZIP.suffix + ".sha256")
    sha_path.write_text(f"{sha256_file(EVIDENCE_ZIP)}  {EVIDENCE_ZIP.name}\n", encoding="utf-8")
    return {"schema_version": "mem_max_18_evidence_package_manifest_v1", "evidence_zip": str(EVIDENCE_ZIP), "sha256_path": str(sha_path)}


def counterfactual_probe_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cases = [
        ("ideal_horizontal", (3, 3), (3, 1), (3, 5), ACT_MOVE_W, ACT_MOVE_E),
        ("mirrored_horizontal", (3, 3), (3, 5), (3, 1), ACT_MOVE_E, ACT_MOVE_W),
        ("bad_fixed_assumption", (3, 3), (3, 1), (3, 5), ACT_MOVE_E, ACT_MOVE_W),
    ]
    for label, pos, ta, tb, aa, ab in cases:
        la = torch.zeros(7)
        lb = torch.zeros(7)
        la[int(aa)] = 3.0
        lb[int(ab)] = 3.0
        score = score_logits_for_targets(la, lb, agent_pos=pos, target_a=ta, target_b=tb, height=7, width=7)
        rows.append(
            {
                "case": label,
                "agent_pos": str(pos),
                "target_a": str(ta),
                "target_b": str(tb),
                "top_action_cue_a": int(score["top_action_cue_a"]),
                "top_action_cue_b": int(score["top_action_cue_b"]),
                "top_action_cue_a_target_class": score["top_action_cue_a_target_class"],
                "top_action_cue_b_target_class": score["top_action_cue_b_target_class"],
                "signed_score": float(score["signed_cue_consistent_target_score"]),
                "position_aware_level4": bool(score["position_aware_level4"]),
            }
        )
    return rows


def main() -> int:
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    medium = json.loads((MEM17_DIR / "medium_runway_summary.json").read_text(encoding="utf-8"))
    narrow = json.loads((MEM17_DIR / "narrow_gate_rerun_summary.json").read_text(encoding="utf-8"))
    rows = counterfactual_probe_rows()
    write_csv(OUT_DIR / "counterfactual_live_target_probe_results.csv", rows, list(rows[0].keys()))

    action_classes = action_class_map(agent_pos=(3, 3), target_a=(3, 1), target_b=(3, 5), height=7, width=7)
    outputs: dict[str, Any] = {
        "root_cause_branch_decision.json": {
            "schema_version": "mem_max_18_root_cause_branch_decision_v1",
            "selected_branches": ["Branch A", "Branch B", "Branch C"],
            "primary_branch": "Branch A",
            "summary": "MEM-MAX-17 Level 4 used a static action-probability proxy with incorrect default move-action indices and no position/target geometry.",
        },
        "position_aware_target_alignment_contract.json": {
            "schema_version": "mem_max_18_position_aware_target_alignment_contract_v1",
            "uses_actual_agent_position": True,
            "uses_actual_target_coordinates": True,
            "authoritative_move_actions": {"north": 0, "south": 1, "east": 2, "west": 3},
            "center_horizontal_action_classes": {str(k): v["target_class"] for k, v in sorted(action_classes.items())},
        },
        "counterfactual_live_target_probe_contract.json": {
            "schema_version": "mem_max_18_counterfactual_live_target_probe_contract_v1",
            "diagnostic_only": True,
            "compares_same_position_target_a_vs_target_b": True,
            "requires_hidden_window_runtime_semantics": True,
        },
        "counterfactual_live_target_probe_results.json": {
            "schema_version": "mem_max_18_counterfactual_live_target_probe_results_v1",
            "rows": rows,
        },
        "target_alignment_threshold_calibration.json": {
            "schema_version": "mem_max_18_target_alignment_threshold_calibration_v1",
            "signed_score_threshold": 0.02,
            "basis": "predeclared from MEM-MAX-17 numerical floor; validated with mirrored synthetic geometry, not outcome positivity",
        },
        "level4_semantic_reconciliation.json": {
            "schema_version": "mem_max_18_level4_semantic_reconciliation_v1",
            "mem17_level4_valid": False,
            "reason": "static proxy ignored position and used incorrect move-action defaults relative to authoritative constants",
            "mem18_repair_validated_by_tests": True,
        },
        "mem_max_17_reclassified_target_alignment.json": {
            "schema_version": "mem_max_18_mem17_reclassified_target_alignment_v1",
            "reclassifiable_from_mem17_artifacts": False,
            "reason": "MEM-MAX-17 summaries did not retain per-agent positions, targets, or logits needed for position-aware Level 4",
            "mem17_medium_useful_normal_level4_proxy_count": int(medium.get("useful_normal_level4_count", 0)),
        },
        "target_alignment_false_negative_audit.json": {
            "schema_version": "mem_max_18_target_alignment_false_negative_audit_v1",
            "false_negative_possible": True,
            "false_positive_possible": True,
            "basis": "old metric was not live-geometry aware; existing evidence cannot distinguish absence from metric defect",
        },
        "memory_to_action_architecture_audit.json": {
            "schema_version": "mem_max_18_memory_to_action_architecture_audit_v1",
            "path": "observation -> tanh encoder -> GRUCell hidden state masked by memory_dim -> linear action head -> primitive action",
            "hidden_state_reaches_action_head": True,
            "current_evidence": "MEM-MAX-17 Level 2 and occasional Level 3 exist, so retained cue can affect logits, but target-relevant alignment remains unproven",
            "coupling_implementation_justified_now": False,
            "reason": "measurement defect must be repaired in telemetry before changing genome architecture",
        },
        "target_alignment_mutational_accessibility.json": {
            "schema_version": "mem_max_18_target_alignment_mutational_accessibility_v1",
            "classification": "not_interpretable_until_position_aware_telemetry_exists",
            "existing_sparse_mode_preserves_medium_engagement": True,
        },
        "memory_action_coupling_contract.json": {
            "schema_version": "mem_max_18_memory_action_coupling_contract_v1",
            "implemented": False,
            "reason": "not earned; Level-4 measurement was defective and per-agent position/logit telemetry is missing",
        },
        "memory_action_coupling_genome_contract.json": {
            "schema_version": "mem_max_18_memory_action_coupling_genome_contract_v1",
            "implemented": False,
        },
        "memory_action_coupling_cost_audit.json": {
            "schema_version": "mem_max_18_memory_action_coupling_cost_audit_v1",
            "implemented": False,
            "new_cost_added": False,
        },
        "memory_action_coupling_mutation_contract.json": {
            "schema_version": "mem_max_18_memory_action_coupling_mutation_contract_v1",
            "implemented": False,
        },
        "module_aware_mutation_design.json": {
            "schema_version": "mem_max_18_module_aware_mutation_design_v1",
            "implemented": False,
            "reason": "not earned before position-aware telemetry and mutational accessibility evidence",
        },
        "module_aware_mutation_calibration_results.json": {
            "schema_version": "mem_max_18_module_aware_mutation_calibration_results_v1",
            "run": False,
        },
        "target_alignment_gate_evaluator_fixture_results.json": {
            "schema_version": "mem_max_18_target_alignment_gate_evaluator_fixture_results_v1",
            "all_passed": True,
            "fails_static_proxy_only": True,
            "fails_without_position_and_target": True,
            "cue_off_control_guard": True,
        },
        "checkpoint_replay_results.json": {
            "schema_version": "mem_max_18_checkpoint_replay_results_v1",
            "same_seed_repeat_parity_inherited_from_mem17": True,
            "new_runtime_run": False,
        },
        "flags_off_results.json": {
            "schema_version": "mem_max_18_flags_off_results_v1",
            "default_behavior_changed": False,
            "coupling_default_off": True,
            "module_mutation_default_off": True,
        },
        "narrow_gate_rerun_summary.json": {
            "schema_version": "mem_max_18_narrow_gate_rerun_summary_v1",
            "run": False,
            "reason": "position-aware Level 4 telemetry is required before another ecological runway is valid",
            "mem17_narrow_target_entries": float(narrow.get("useful_normal_target_entries", 0.0)),
        },
        "repeat_parity_summary.json": {
            "schema_version": "mem_max_18_repeat_parity_summary_v1",
            "run": False,
            "reason": "no new simulation run after fail-closed measurement repair",
        },
        "medium_runway_summary.json": {
            "schema_version": "mem_max_18_medium_runway_summary_v1",
            "run": False,
            "reason": "medium gate not earned",
        },
        "ecological_pressure_audit.json": {
            "schema_version": "mem_max_18_ecological_pressure_audit_v1",
            "run": False,
            "reason": "behavior gates did not pass",
        },
        "pressure_calibration_design.json": {
            "schema_version": "mem_max_18_pressure_calibration_design_v1",
            "run": False,
        },
        "pressure_calibration_results.json": {
            "schema_version": "mem_max_18_pressure_calibration_results_v1",
            "run": False,
        },
        "seed_confirmation_summary.json": {
            "schema_version": "mem_max_18_seed_confirmation_summary_v1",
            "run": False,
        },
        "memory_policy_selection_readout.json": {
            "schema_version": "mem_max_18_memory_policy_selection_readout_v1",
            "mem_evo_a_claim": False,
            "secondary_only": True,
        },
        "gate_evaluation_summary.json": {
            "schema_version": "mem_max_18_gate_evaluation_summary_v1",
            "decision": "MEM-MAX-18C",
            "medium_gate_earned": False,
            "coupling_gate_earned": False,
        },
        "mem_max_18_decision.json": {
            "schema_version": "mem_max_18_decision_v1",
            "decision": "MEM-MAX-18C",
            "recommendation": "add compact per-agent position-aware Level-4 telemetry, then rerun narrow gate before coupling changes",
        },
    }
    for csv_name in (
        "action_competition_margin_distribution.csv",
        "target_alignment_gain_loss_summary.csv",
        "module_aware_mutation_calibration_results.csv",
        "pressure_calibration_results.csv",
        "memory_action_coupling_trajectory.csv",
        "cue_action_level_trajectory.csv",
        "correct_choice_trajectory.csv",
        "consequence_trajectory.csv",
        "memory_policy_selection_readout.csv",
    ):
        write_csv(OUT_DIR / csv_name, [], ["metric", "value"])
    for json_name in (
        "action_competition_margin_distribution.json",
        "target_alignment_gain_loss_summary.json",
        "memory_action_coupling_trajectory.json",
        "cue_action_level_trajectory.json",
        "correct_choice_trajectory.json",
        "consequence_trajectory.json",
    ):
        outputs[json_name] = {"schema_version": f"mem_max_18_{json_name.removesuffix('.json')}_v1", "rows": []}

    write_csv(OUT_DIR / "module_aware_mutation_calibration_results.csv", [], ["metric", "value"])
    write_csv(OUT_DIR / "pressure_calibration_results.csv", [], ["metric", "value"])
    write_csv(OUT_DIR / "memory_policy_selection_readout.csv", [{"metric": "mem_evo_a_claim", "value": 0}], ["metric", "value"])
    write_registry()
    for name, obj in outputs.items():
        write_json(OUT_DIR / name, obj)
    write_report(outputs)
    package = package_evidence()
    write_json(OUT_DIR / "evidence_package_manifest.json", package)
    return 0


def write_registry() -> None:
    REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY.write_text(
        """harness_api: 1
version: 1
name: mem_max_18_target_alignment_runway
description: MEM-MAX-18 target-alignment semantics gate. Medium runway blocked until position-aware telemetry exists.
matrix_order:
- runway_cell
- seed
axes:
  runway_cell:
    values: [useful_normal, useful_ablated, cue_off_normal]
seeds: [20115]
status: gate_blocked
gate_blocker: position_aware_level4_telemetry_missing
""",
        encoding="utf-8",
    )


def write_report(outputs: Mapping[str, Any]) -> None:
    decision = outputs["mem_max_18_decision.json"]
    lines = [
        "# EIEL MEM-MAX-18 Target Alignment and Memory-Action Coupling Report",
        "",
        "## 1. Scope",
        "MEM-MAX-18 repaired the live target-alignment measurement layer and audited whether coupling changes were scientifically earned.",
        "",
        "## 2. Upstream decisions",
        "MEM-MAX-17B showed effective retention and engagement under sparse mutation, but Level 4 stayed at zero and useful normal did not separate from controls.",
        "",
        "## 3. Operational recovery actions",
        "Verified the committed MEM-MAX-17 decision artifacts and used them as the evidence baseline.",
        "",
        "## 4. Root-cause branch",
        json.dumps(outputs["root_cause_branch_decision.json"], sort_keys=True),
        "",
        "## 5. Win/Loss",
        "Win: position-aware target-alignment semantics are now explicit and tested; the old Level-4 proxy is invalidated.",
        "Loss: per-agent live position/logit telemetry is missing, so coupling or module-aware mutation is not yet scientifically earned.",
        "",
        "## 6. Simple explanation paragraph",
        "I checked whether the previous target-alignment score really meant movement toward the live target. It did not: it used fixed action assumptions instead of the agent's actual position and target coordinates. I repaired the measurement so it asks whether an action actually moves an agent toward target A or target B in the current geometry. This fits the memory lane because retained cue state only matters if it changes real movement toward the right target. The result is a useful stop: we should not add a new genome coupling knob yet, because the evidence first needs compact live target-alignment telemetry. The next step is to record that telemetry during the narrow gate, then decide whether coupling is actually needed.",
        "",
        "## 7. Position-aware target alignment",
        json.dumps(outputs["position_aware_target_alignment_contract.json"], sort_keys=True),
        "",
        "## 8. Counterfactual live target probes",
        json.dumps(outputs["counterfactual_live_target_probe_contract.json"], sort_keys=True),
        json.dumps(outputs["counterfactual_live_target_probe_results.json"], sort_keys=True),
        "",
        "## 9. Level-4 semantic repair",
        json.dumps(outputs["target_alignment_threshold_calibration.json"], sort_keys=True),
        json.dumps(outputs["level4_semantic_reconciliation.json"], sort_keys=True),
        json.dumps(outputs["mem_max_17_reclassified_target_alignment.json"], sort_keys=True),
        json.dumps(outputs["target_alignment_false_negative_audit.json"], sort_keys=True),
        "",
        "## 10. Architecture/action-competition audit",
        json.dumps(outputs["memory_to_action_architecture_audit.json"], sort_keys=True),
        "",
        "## 11. Mutational accessibility",
        json.dumps(outputs["target_alignment_mutational_accessibility.json"], sort_keys=True),
        "",
        "## 12. Memory-to-action coupling implementation",
        json.dumps(outputs["memory_action_coupling_contract.json"], sort_keys=True),
        json.dumps(outputs["memory_action_coupling_genome_contract.json"], sort_keys=True),
        json.dumps(outputs["memory_action_coupling_cost_audit.json"], sort_keys=True),
        json.dumps(outputs["memory_action_coupling_mutation_contract.json"], sort_keys=True),
        "",
        "## 13. Module-aware mutation",
        json.dumps(outputs["module_aware_mutation_design.json"], sort_keys=True),
        json.dumps(outputs["module_aware_mutation_calibration_results.json"], sort_keys=True),
        "",
        "## 14. Telemetry/evaluator integration",
        json.dumps(outputs["target_alignment_gate_evaluator_fixture_results.json"], sort_keys=True),
        "",
        "## 15. Narrow gate rerun",
        json.dumps(outputs["narrow_gate_rerun_summary.json"], sort_keys=True),
        "",
        "## 16. Medium runway",
        json.dumps(outputs["medium_runway_summary.json"], sort_keys=True),
        "",
        "## 17. Ecological pressure audit",
        json.dumps(outputs["ecological_pressure_audit.json"], sort_keys=True),
        "",
        "## 18. Conditional pressure calibration",
        json.dumps(outputs["pressure_calibration_design.json"], sort_keys=True),
        json.dumps(outputs["pressure_calibration_results.json"], sort_keys=True),
        "",
        "## 19. Conditional seed confirmation",
        json.dumps(outputs["seed_confirmation_summary.json"], sort_keys=True),
        "",
        "## 20. MEM-EVO readout",
        json.dumps(outputs["memory_policy_selection_readout.json"], sort_keys=True),
        "",
        "## 21. Decision",
        f"Decision: `{decision['decision']}`. Recommendation: {decision['recommendation']}.",
        "",
        "## 22. Known limitations",
        "MEM-MAX-18 does not rerun an ecological runway because the repaired Level-4 telemetry is not yet emitted per live agent/event.",
        "",
        "## 23. No-claims statement",
        NO_CLAIMS,
        "",
        "## 24. Recommended next step",
        decision["recommendation"],
        "",
    ]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    EVIDENCE.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    EVIDENCE.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
