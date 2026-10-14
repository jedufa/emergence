from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.theme4_campaigns import (
    affordance_outcome_correlation_report,
    behavior_expression_diagnostic,
    canonical_json,
    campaign_metadata_summary,
    cost_response_report,
    diagnostic_source_manifest,
    execute_campaign,
    execute_fixture_campaign,
    execute_live_campaign,
    fake_evolution_audit,
    behavior_linked_variant_selection,
    behavior_proxy_audit,
    behavior_proxy_report,
    behavior_proxy_validation_report,
    evidence_metadata_cleanup_report,
    lineage_retention_report,
    load_campaign_manifest,
    paired_cost_summary,
    paired_transfer_summary,
    perception_ablation_validity_audit,
    perception_fake_evolution_audit,
    perception_channel_visibility_audit,
    perception_lineage_retention_report,
    perception_pressure_isolation_report,
    plan_campaign,
    scenario_pressure_audit,
    summarize_paired_runs,
    stage10g_decision_report,
    behavior_outcome_calibration_report,
    perception_testbed_selection_report,
    scenario_classification_report,
    stage10h_decision_report,
    stage10i_decision_report,
    stage10i_gate_report,
    stage10j_behavior_outcome_coupling_report,
    stage10j_evidence_audit,
    stage10j_offline_artifact_sanity,
    stage10j_outcome_saturation_report,
    stage10j_theory_discrimination_update,
    stage10k_decision_report,
    calibrated_perception_paired_summary,
    clean_pressure_vs_flat_report,
    stage10l_behavior_outcome_coupling_report,
    stage10l_decision_report,
    stage10l_fake_evolution_audit,
    stage10l_flat_artifact_check,
    stage10l_lineage_retention_report,
    summarize_calibration_runs,
    visibility_calibration_report,
    theory_discrimination_report,
    transfer_robustness_report,
    validate_campaign_metrics,
    write_campaign_artifacts,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Theme 4 campaign planning and fixture evidence tools")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in (
        "plan",
        "fixture",
        "live",
        "execute",
        "validate",
        "summarize",
        "cost-summary",
        "cost-response",
        "transfer-summary",
        "transfer-report",
        "lineage-report",
        "perception-pressure-report",
        "behavior-proxy-report",
        "behavior-proxy-audit",
        "behavior-proxy-validation-report",
        "behavior-linked-variant-selection",
        "perception-lineage-report",
        "perception-audit",
        "metadata-summary",
        "perception-visibility-audit",
        "affordance-outcome-report",
        "behavior-expression-diagnostic",
        "scenario-pressure-audit",
        "ablation-validity-audit",
        "evidence-metadata-cleanup-report",
        "stage10g-decision-report",
        "calibration-summary",
        "visibility-calibration-report",
        "behavior-outcome-calibration-report",
        "scenario-classification-report",
        "testbed-selection-report",
        "stage10h-decision-report",
        "stage10i-gate",
        "stage10i-decision-report",
        "theory-discrimination-report",
        "stage10j-evidence-audit",
        "stage10j-offline-artifact-sanity",
        "stage10j-outcome-saturation-report",
        "stage10j-behavior-outcome-coupling-report",
        "stage10j-theory-discrimination-update",
        "stage10k-decision-report",
        "stage10l-calibrated-paired-summary",
        "stage10l-clean-pressure-vs-flat-report",
        "stage10l-behavior-outcome-coupling-report",
        "stage10l-lineage-retention-report",
        "stage10l-flat-artifact-check",
        "stage10l-audit",
        "stage10l-decision-report",
        "audit",
        "artifacts",
    ):
        p = sub.add_parser(name)
        p.add_argument("--manifest", required=True)
        p.add_argument("--output-dir", default="")
    args = parser.parse_args()
    manifest = load_campaign_manifest(args.manifest)
    output_dir = Path(args.output_dir or manifest["output_directory"])
    source_manifest = diagnostic_source_manifest(manifest)
    if args.command == "plan":
        print(canonical_json(plan_campaign(manifest)), end="")
    elif args.command == "fixture":
        print(canonical_json(execute_fixture_campaign(manifest, output_root=output_dir)), end="")
    elif args.command == "live":
        print(canonical_json(execute_live_campaign(manifest, output_root=output_dir)), end="")
    elif args.command == "execute":
        print(canonical_json(execute_campaign(manifest, output_root=output_dir)), end="")
    elif args.command == "validate":
        print(canonical_json(validate_campaign_metrics(manifest, plan_campaign(manifest))), end="")
    elif args.command == "summarize":
        print(canonical_json(summarize_paired_runs(manifest, plan_campaign(manifest))), end="")
    elif args.command == "cost-summary":
        print(canonical_json(paired_cost_summary(summarize_paired_runs(manifest, plan_campaign(manifest)))), end="")
    elif args.command == "cost-response":
        summary = summarize_paired_runs(manifest, plan_campaign(manifest))
        print(canonical_json(cost_response_report(paired_cost_summary(summary))), end="")
    elif args.command == "transfer-summary":
        print(canonical_json(paired_transfer_summary(summarize_paired_runs(manifest, plan_campaign(manifest)))), end="")
    elif args.command == "transfer-report":
        summary = summarize_paired_runs(manifest, plan_campaign(manifest))
        print(canonical_json(transfer_robustness_report(paired_transfer_summary(summary))), end="")
    elif args.command == "lineage-report":
        print(canonical_json(lineage_retention_report(summarize_paired_runs(manifest, plan_campaign(manifest)))), end="")
    elif args.command == "perception-pressure-report":
        print(canonical_json(perception_pressure_isolation_report(summarize_paired_runs(manifest, plan_campaign(manifest)))), end="")
    elif args.command == "behavior-proxy-report":
        print(canonical_json(behavior_proxy_report(summarize_paired_runs(manifest, plan_campaign(manifest)))), end="")
    elif args.command == "behavior-proxy-audit":
        print(canonical_json(behavior_proxy_audit()), end="")
    elif args.command == "behavior-proxy-validation-report":
        print(canonical_json(behavior_proxy_validation_report(manifest, plan_campaign(manifest))), end="")
    elif args.command == "behavior-linked-variant-selection":
        print(canonical_json(behavior_linked_variant_selection(summarize_paired_runs(manifest, plan_campaign(manifest)))), end="")
    elif args.command == "perception-lineage-report":
        print(canonical_json(perception_lineage_retention_report(summarize_paired_runs(manifest, plan_campaign(manifest)))), end="")
    elif args.command == "perception-audit":
        plan = plan_campaign(manifest)
        validation = validate_campaign_metrics(manifest, plan)
        summary = summarize_paired_runs(manifest, plan)
        pressure = perception_pressure_isolation_report(summary)
        print(canonical_json(perception_fake_evolution_audit(manifest, validation, summary, pressure)), end="")
    elif args.command == "metadata-summary":
        plan = plan_campaign(source_manifest)
        execution_path = Path(str(source_manifest["output_directory"])) / "execution_summary.json"
        execution = json.loads(execution_path.read_text(encoding="utf-8")) if execution_path.exists() else {}
        validation = validate_campaign_metrics(source_manifest, plan)
        print(canonical_json(campaign_metadata_summary(source_manifest, plan, execution, validation)), end="")
    elif args.command == "perception-visibility-audit":
        print(canonical_json(perception_channel_visibility_audit(summarize_paired_runs(source_manifest, plan_campaign(source_manifest)))), end="")
    elif args.command == "affordance-outcome-report":
        print(canonical_json(affordance_outcome_correlation_report(summarize_paired_runs(source_manifest, plan_campaign(source_manifest)))), end="")
    elif args.command == "behavior-expression-diagnostic":
        print(canonical_json(behavior_expression_diagnostic(summarize_paired_runs(source_manifest, plan_campaign(source_manifest)))), end="")
    elif args.command == "scenario-pressure-audit":
        print(canonical_json(scenario_pressure_audit(summarize_paired_runs(source_manifest, plan_campaign(source_manifest)))), end="")
    elif args.command == "ablation-validity-audit":
        plan = plan_campaign(source_manifest)
        validation = validate_campaign_metrics(source_manifest, plan)
        summary = summarize_paired_runs(source_manifest, plan)
        print(canonical_json(perception_ablation_validity_audit(source_manifest, summary, validation)), end="")
    elif args.command == "evidence-metadata-cleanup-report":
        plan = plan_campaign(source_manifest)
        execution_path = Path(str(source_manifest["output_directory"])) / "execution_summary.json"
        execution = json.loads(execution_path.read_text(encoding="utf-8")) if execution_path.exists() else {}
        validation = validate_campaign_metrics(source_manifest, plan)
        meta = campaign_metadata_summary(source_manifest, plan, execution, validation)
        print(canonical_json(evidence_metadata_cleanup_report(meta)), end="")
    elif args.command == "stage10g-decision-report":
        plan = plan_campaign(source_manifest)
        validation = validate_campaign_metrics(source_manifest, plan)
        summary = summarize_paired_runs(source_manifest, plan)
        visibility = perception_channel_visibility_audit(summary)
        affordance = affordance_outcome_correlation_report(summary)
        behavior = behavior_expression_diagnostic(summary)
        pressure = scenario_pressure_audit(summary)
        ablation = perception_ablation_validity_audit(source_manifest, summary, validation)
        print(canonical_json(stage10g_decision_report(visibility, affordance, behavior, pressure, ablation)), end="")
    elif args.command == "calibration-summary":
        print(canonical_json(summarize_calibration_runs(manifest, plan_campaign(manifest))), end="")
    elif args.command == "visibility-calibration-report":
        print(canonical_json(visibility_calibration_report(summarize_calibration_runs(manifest, plan_campaign(manifest)))), end="")
    elif args.command == "behavior-outcome-calibration-report":
        print(canonical_json(behavior_outcome_calibration_report(summarize_calibration_runs(manifest, plan_campaign(manifest)))), end="")
    elif args.command == "scenario-classification-report":
        print(canonical_json(scenario_classification_report(summarize_calibration_runs(manifest, plan_campaign(manifest)))), end="")
    elif args.command == "testbed-selection-report":
        calibration = summarize_calibration_runs(manifest, plan_campaign(manifest))
        classifications = scenario_classification_report(calibration)
        print(canonical_json(perception_testbed_selection_report(classifications)), end="")
    elif args.command == "stage10h-decision-report":
        calibration = summarize_calibration_runs(manifest, plan_campaign(manifest))
        behavior = behavior_outcome_calibration_report(calibration)
        classifications = scenario_classification_report(calibration)
        selection = perception_testbed_selection_report(classifications)
        print(canonical_json(stage10h_decision_report(selection, classifications, behavior)), end="")
    elif args.command == "stage10i-gate":
        plan = plan_campaign(manifest)
        validation = validate_campaign_metrics(manifest, plan)
        proxy_validation = behavior_proxy_validation_report(manifest, plan)
        summary = summarize_paired_runs(manifest, plan)
        selection = behavior_linked_variant_selection(summary)
        failures = " ".join(str(f) for f in validation.get("failures", []))
        direct_reward_present = "reward" in failures and "direct" in failures
        print(canonical_json(stage10i_gate_report(proxy_validation, selection, direct_reward_present=direct_reward_present)), end="")
    elif args.command == "stage10i-decision-report":
        print(canonical_json(stage10i_decision_report(summarize_paired_runs(manifest, plan_campaign(manifest)))), end="")
    elif args.command == "theory-discrimination-report":
        plan = plan_campaign(manifest)
        proxy_validation = behavior_proxy_validation_report(manifest, plan)
        summary = summarize_paired_runs(manifest, plan)
        selection = behavior_linked_variant_selection(summary)
        gate = stage10i_gate_report(proxy_validation, selection)
        print(canonical_json(theory_discrimination_report(proxy_validation, selection, gate)), end="")
    elif args.command == "stage10j-evidence-audit":
        print(canonical_json(stage10j_evidence_audit(summarize_paired_runs(manifest, plan_campaign(manifest)))), end="")
    elif args.command == "stage10j-offline-artifact-sanity":
        print(canonical_json(stage10j_offline_artifact_sanity(summarize_paired_runs(manifest, plan_campaign(manifest)))), end="")
    elif args.command == "stage10j-outcome-saturation-report":
        print(canonical_json(stage10j_outcome_saturation_report(summarize_paired_runs(manifest, plan_campaign(manifest)))), end="")
    elif args.command == "stage10j-behavior-outcome-coupling-report":
        print(canonical_json(stage10j_behavior_outcome_coupling_report(summarize_paired_runs(manifest, plan_campaign(manifest)))), end="")
    elif args.command == "stage10j-theory-discrimination-update":
        summary = summarize_paired_runs(manifest, plan_campaign(manifest))
        audit = stage10j_evidence_audit(summary)
        artifact = stage10j_offline_artifact_sanity(summary)
        saturation = stage10j_outcome_saturation_report(summary)
        coupling = stage10j_behavior_outcome_coupling_report(summary)
        print(canonical_json(stage10j_theory_discrimination_update(audit, artifact, saturation, coupling)), end="")
    elif args.command == "stage10k-decision-report":
        summary = summarize_paired_runs(manifest, plan_campaign(manifest))
        audit = stage10j_evidence_audit(summary)
        artifact = stage10j_offline_artifact_sanity(summary)
        saturation = stage10j_outcome_saturation_report(summary)
        coupling = stage10j_behavior_outcome_coupling_report(summary)
        theory = stage10j_theory_discrimination_update(audit, artifact, saturation, coupling)
        print(canonical_json(stage10k_decision_report(audit, saturation, coupling, theory)), end="")
    elif args.command == "stage10l-calibrated-paired-summary":
        print(canonical_json(calibrated_perception_paired_summary(summarize_paired_runs(manifest, plan_campaign(manifest)))), end="")
    elif args.command == "stage10l-clean-pressure-vs-flat-report":
        print(canonical_json(clean_pressure_vs_flat_report(summarize_paired_runs(manifest, plan_campaign(manifest)))), end="")
    elif args.command == "stage10l-behavior-outcome-coupling-report":
        print(canonical_json(stage10l_behavior_outcome_coupling_report(summarize_paired_runs(manifest, plan_campaign(manifest)))), end="")
    elif args.command == "stage10l-lineage-retention-report":
        print(canonical_json(stage10l_lineage_retention_report(summarize_paired_runs(manifest, plan_campaign(manifest)))), end="")
    elif args.command == "stage10l-flat-artifact-check":
        summary = summarize_paired_runs(manifest, plan_campaign(manifest))
        pressure = clean_pressure_vs_flat_report(summary)
        print(canonical_json(stage10l_flat_artifact_check(summary, pressure)), end="")
    elif args.command == "stage10l-audit":
        plan = plan_campaign(manifest)
        validation = validate_campaign_metrics(manifest, plan)
        summary = summarize_paired_runs(manifest, plan)
        pressure = clean_pressure_vs_flat_report(summary)
        coupling = stage10l_behavior_outcome_coupling_report(summary)
        print(canonical_json(stage10l_fake_evolution_audit(manifest, validation, summary, pressure, coupling)), end="")
    elif args.command == "stage10l-decision-report":
        plan = plan_campaign(manifest)
        validation = validate_campaign_metrics(manifest, plan)
        summary = summarize_paired_runs(manifest, plan)
        pressure = clean_pressure_vs_flat_report(summary)
        coupling = stage10l_behavior_outcome_coupling_report(summary)
        lineage = stage10l_lineage_retention_report(summary)
        flat = stage10l_flat_artifact_check(summary, pressure)
        print(canonical_json(stage10l_decision_report(pressure, coupling, lineage, flat, validation)), end="")
    elif args.command == "audit":
        plan = plan_campaign(manifest)
        validation = validate_campaign_metrics(manifest, plan)
        summary = summarize_paired_runs(manifest, plan)
        print(canonical_json(fake_evolution_audit(manifest, validation, summary)), end="")
    elif args.command == "artifacts":
        print(canonical_json(write_campaign_artifacts(manifest, output_dir)), end="")
    else:
        raise AssertionError(args.command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
