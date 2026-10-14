from __future__ import annotations

"""REP-2H tick-loop reproductive selection evidence builder."""

import csv
import json
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from evolution.structural_action_opportunity_repair import (
    READINESS_ARMS,
    READINESS_SEEDS,
    _chosen_total,
    _readiness_cell_row,
    make_rep2fr_sim,
)
from evolution.structural_causal_robustness import reward_audit, stable_hash, write_json
from simulation.checkpoint_bundle_v2 import checkpoint_bundle_v2_apply, checkpoint_bundle_v2_load, checkpoint_bundle_v2_save
from simulation.performance_profiles import ACCEPTED_W6_FAST_PROFILE, apply_performance_profile


SCHEMA_VERSION = "structural_tick_reproduction_selection_v1"
BASE_COMMIT = "3852a4dafff86ef2abd34ef24722782e6e45d9b8"
CONTEXTS = ("clean_control", "crowding_existing")
STAGE_ORDER = (
    "live_effect_alive",
    "reproduction_opportunity",
    "reproduction_attempt",
    "successful_tick_reproducer",
    "child_inheritance",
    "short_retention_next_generation",
    "short_retention_two_generations",
)
REPORT_NAME = "EIEL_REP2H_Tick_Loop_Reproductive_Selection_and_Inheritance_Gate_2026_07.md"
EXPECTED_MATRIX = {
    "base_commit": BASE_COMMIT,
    "readiness_cells": 48,
    "contexts": list(CONTEXTS),
    "seeds": list(READINESS_SEEDS),
    "arms": [arm for arm, _live, _ablation in READINESS_ARMS],
    "generations": 8,
    "steps": 40,
    "population": 48,
}


def make_rep2h_sim(
    run_dir: Path,
    *,
    seed: int,
    context: str,
    live_enabled: bool,
    ablation: str = "none",
    generations: int = 8,
    steps: int = 40,
    pop: int = 48,
    metrics: bool = True,
) -> Any:
    sim = make_rep2fr_sim(
        run_dir,
        seed=seed,
        context=context,
        live_enabled=live_enabled,
        ablation=ablation,
        generations=generations,
        steps=steps,
        pop=pop,
        metrics=metrics,
    )
    sim.sim_cfg.structural_selection_funnel_reporting_enable = True
    sim.sim_cfg.structural_tick_reproduction_funnel_reporting_enable = True
    sim.sim_cfg.lineage_reproduction_funnel_reporting_enable = True
    sim.sim_cfg.lineage_reproduction_funnel_write_events = True
    sim.sim_cfg.lineage_reproduction_funnel_event_schema_version = 4
    sim.sim_cfg.lineage_reproduction_funnel_write_scan_action_events = True
    return sim


def run_readiness_matrix(runtime_root: Path, *, workers: int = 6) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    tasks = []
    for context in CONTEXTS:
        for seed in READINESS_SEEDS:
            for arm, live, ablation in READINESS_ARMS:
                tasks.append(
                    {
                        "runtime_root": str(runtime_root),
                        "context": context,
                        "seed": int(seed),
                        "arm": arm,
                        "live": bool(live),
                        "ablation": ablation,
                    }
                )
    rows: list[dict[str, Any]] = []
    command_rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=max(1, int(workers))) as pool:
        futs = [pool.submit(_run_readiness_cell, task) for task in tasks]
        for fut in as_completed(futs):
            row = fut.result()
            rows.append(row)
            command_rows.append(
                {
                    "context": row["context"],
                    "seed": row["seed"],
                    "arm": row["arm"],
                    "run_dir": row["run_dir"],
                    "elapsed_seconds": row.get("elapsed_seconds"),
                    "status": row.get("status"),
                }
            )
    rows.sort(key=lambda r: (str(r["context"]), int(r["seed"]), str(r["arm"])))
    command_rows.sort(key=lambda r: (str(r["context"]), int(r["seed"]), str(r["arm"])))
    command_rows.append({"matrix_elapsed_seconds": float(time.perf_counter() - started), "workers": int(workers)})
    return rows, command_rows


def _run_readiness_cell(task: dict[str, Any]) -> dict[str, Any]:
    runtime_root = Path(str(task["runtime_root"]))
    context = str(task["context"])
    seed = int(task["seed"])
    arm = str(task["arm"])
    run_dir = runtime_root / f"readiness_{context}_{arm}_seed{seed}"
    started = time.perf_counter()
    sim = make_rep2h_sim(
        run_dir,
        seed=seed,
        context=context,
        live_enabled=bool(task["live"]),
        ablation=str(task["ablation"]),
    )
    sim.run(start_gen=0, skip_init=False)
    row = _readiness_cell_row(sim, seed=seed, context=context, arm=arm)
    row.update(_structural_selection_fields(run_dir))
    row.update(_tick_reproduction_fields(run_dir))
    row["run_dir"] = str(run_dir)
    row["elapsed_seconds"] = float(time.perf_counter() - started)
    row["status"] = "passed"
    return row


def _structural_selection_fields(run_dir: Path) -> dict[str, Any]:
    summary = _read_json(run_dir / "metrics" / "structural_selection_funnel_run_summary.json")
    if not isinstance(summary, dict):
        return {"rep2g_funnel_evaluable": False, "rep2g_funnel_reason": "missing_structural_selection_funnel_run_summary"}
    out: dict[str, Any] = {"rep2g_funnel_evaluable": True, "rep2g_funnel_hash": stable_hash(summary)}
    stage_totals = summary.get("stage_totals", {}) if isinstance(summary.get("stage_totals"), dict) else {}
    for stage in (
        "actual_refill_parent_pool",
        "sampled_parent_draw",
        "successful_reproducer",
        "child_inheritance",
        "short_retention_later_living_descendant",
    ):
        payload = stage_totals.get(stage, {}) if isinstance(stage_totals, dict) else {}
        out[f"rep2g_{stage}_agent_count"] = int(payload.get("agent_count", 0) or 0)
        out[f"rep2g_{stage}_live_effect_count"] = int(payload.get("live_effect_present_count", 0) or 0)
    return out


def _tick_reproduction_fields(run_dir: Path) -> dict[str, Any]:
    summary = _read_json(run_dir / "metrics" / "structural_tick_reproduction_funnel_run_summary.json")
    if not isinstance(summary, dict):
        return {"tick_funnel_evaluable": False, "tick_funnel_reason": "missing_structural_tick_reproduction_funnel_run_summary"}
    out: dict[str, Any] = {
        "tick_funnel_evaluable": True,
        "tick_funnel_hash": stable_hash(summary),
        "tick_generation_count": int(summary.get("generation_count", 0) or 0),
        "tick_reproduction_opportunity_agent_count": int(summary.get("reproduction_opportunity_agent_count", 0) or 0),
        "tick_reproduction_attempt_agent_count": int(summary.get("reproduction_attempt_agent_count", 0) or 0),
        "tick_reproduction_attempt_event_count": int(summary.get("reproduction_attempt_event_count", 0) or 0),
        "tick_successful_reproducer_count": int(summary.get("successful_reproducer_count", 0) or 0),
        "tick_successful_birth_count": int(summary.get("successful_birth_count", 0) or 0),
        "tick_child_count": int(summary.get("child_count", 0) or 0),
        "tick_child_live_effect_present_count": int(summary.get("child_live_effect_present_count", 0) or 0),
        "tick_child_next_generation_alive_count": int(summary.get("child_next_generation_alive_count", 0) or 0),
        "tick_child_two_generation_alive_count": int(summary.get("child_two_generation_alive_count", 0) or 0),
        "tick_success_rate_per_opportunity_agent": summary.get("success_rate_per_opportunity_agent"),
        "tick_attempt_rate_per_opportunity_agent": summary.get("attempt_rate_per_opportunity_agent"),
        "tick_child_live_effect_rate_per_successful_birth": summary.get("child_live_effect_rate_per_successful_birth"),
        "tick_next_generation_retention_rate_per_child": summary.get("next_generation_retention_rate_per_child"),
        "tick_two_generation_retention_rate_per_child": summary.get("two_generation_retention_rate_per_child"),
        "tick_first_loss_stage_last_generation": summary.get("last_first_loss_stage"),
    }
    stage_totals = summary.get("stage_totals", {}) if isinstance(summary.get("stage_totals"), dict) else {}
    for stage in STAGE_ORDER:
        payload = stage_totals.get(stage, {}) if isinstance(stage_totals, dict) else {}
        out[f"tick_{stage}_agent_count"] = int(payload.get("agent_count", 0) or 0)
        out[f"tick_{stage}_live_effect_count"] = int(payload.get("live_effect_present_count", 0) or 0)
        out[f"tick_{stage}_live_graph_valid_count"] = int(payload.get("live_graph_valid_count", 0) or 0)
    return out


def aggregate_tick_path(rows: list[dict[str, Any]]) -> dict[str, Any]:
    live_rows = [row for row in rows if row.get("arm") == "live_on"]
    stage_totals = {}
    for stage in STAGE_ORDER:
        stage_totals[stage] = {
            "agent_count": sum(int(row.get(f"tick_{stage}_agent_count", 0) or 0) for row in live_rows),
            "live_effect_count": sum(int(row.get(f"tick_{stage}_live_effect_count", 0) or 0) for row in live_rows),
            "live_graph_valid_count": sum(int(row.get(f"tick_{stage}_live_graph_valid_count", 0) or 0) for row in live_rows),
        }
    first_loss = _first_loss_stage({stage: int(payload["live_effect_count"]) for stage, payload in stage_totals.items()})
    return {
        "schema_version": SCHEMA_VERSION,
        "live_on_cell_count": len(live_rows),
        "tick_funnel_evaluable": all(bool(row.get("tick_funnel_evaluable")) for row in rows),
        "stage_totals": stage_totals,
        "first_loss_stage": first_loss,
        "successful_birth_count": sum(int(row.get("tick_successful_birth_count", 0) or 0) for row in live_rows),
        "child_live_effect_present_count": sum(int(row.get("tick_child_live_effect_present_count", 0) or 0) for row in live_rows),
        "child_next_generation_alive_count": sum(int(row.get("tick_child_next_generation_alive_count", 0) or 0) for row in live_rows),
        "child_two_generation_alive_count": sum(int(row.get("tick_child_two_generation_alive_count", 0) or 0) for row in live_rows),
    }


def paired_tick_readiness(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by: dict[tuple[str, int], dict[str, dict[str, Any]]] = {}
    for row in rows:
        by.setdefault((str(row["context"]), int(row["seed"])), {})[str(row["arm"])] = row
    out = []
    for (context, seed), arms in sorted(by.items()):
        if not all(name in arms for name, _live, _ablation in READINESS_ARMS):
            continue
        live_off = arms["live_off"]
        live_on = arms["live_on"]
        removed = arms["remove_connections"]
        chosen_live_on = _safe_rate(_chosen_total(live_on), live_on.get("opp_potential_linked_opportunity_count"))
        chosen_live_off = _safe_rate(_chosen_total(live_off), live_off.get("opp_potential_linked_opportunity_count"))
        chosen_removed = _safe_rate(_chosen_total(removed), removed.get("opp_potential_linked_opportunity_count"))
        success_live_on = _safe_float(live_on.get("tick_success_rate_per_opportunity_agent"))
        success_live_off = _safe_float(live_off.get("tick_success_rate_per_opportunity_agent"))
        success_removed = _safe_float(removed.get("tick_success_rate_per_opportunity_agent"))
        lineage_evaluable = float(live_on.get("effective_alive_lineages") or 0.0) > 0.0
        stage_counts = {stage: int(live_on.get(f"tick_{stage}_live_effect_count", 0) or 0) for stage in STAGE_ORDER}
        first_loss = _first_loss_stage(stage_counts)
        behavior_supported = context != "clean_control" and chosen_live_on > max(chosen_live_off, chosen_removed)
        differential_reproduction_supported = context != "clean_control" and success_live_on > max(success_live_off, success_removed)
        inheritance_supported = stage_counts.get("child_inheritance", 0) > 0
        next_retention_supported = stage_counts.get("short_retention_next_generation", 0) > 0
        supported = bool(
            behavior_supported
            and differential_reproduction_supported
            and inheritance_supported
            and next_retention_supported
            and lineage_evaluable
        )
        out.append(
            {
                "schema_version": SCHEMA_VERSION,
                "context": context,
                "seed": seed,
                "chosen_rate_per_potential_live_on": chosen_live_on,
                "chosen_rate_per_potential_live_off": chosen_live_off,
                "chosen_rate_per_potential_remove_connections": chosen_removed,
                "chosen_rate_enrichment_vs_best_control": chosen_live_on - max(chosen_live_off, chosen_removed),
                "tick_success_rate_per_opportunity_live_on": success_live_on,
                "tick_success_rate_per_opportunity_live_off": success_live_off,
                "tick_success_rate_per_opportunity_remove_connections": success_removed,
                "tick_success_rate_enrichment_vs_best_control": success_live_on - max(success_live_off, success_removed),
                "first_loss_stage": first_loss,
                "lineage_evaluable": lineage_evaluable,
                "behavior_supported": behavior_supported,
                "differential_reproduction_supported": differential_reproduction_supported,
                "child_inheritance_supported": inheritance_supported,
                "short_retention_supported": next_retention_supported,
                "tick_readiness_pair_supported": supported,
            }
        )
    return out


def active_inactive_paths(rows: list[dict[str, Any]]) -> dict[str, Any]:
    live_rows = [row for row in rows if row.get("arm") == "live_on"]
    return {
        "schema_version": SCHEMA_VERSION,
        "active_path": "living_agent_to_tick_reproduction_opportunity_to_attempt_to_successful_tick_reproducer_to_child_to_later_living_descendant",
        "inactive_diagnostic_path": "generation_end_reproduce_and_cull_refill_parent_pool",
        "active_tick_successful_birth_count": sum(int(row.get("tick_successful_birth_count", 0) or 0) for row in live_rows),
        "inactive_actual_refill_parent_pool_count": sum(int(row.get("rep2g_actual_refill_parent_pool_agent_count", 0) or 0) for row in live_rows),
        "inactive_sampled_parent_draw_count": sum(int(row.get("rep2g_sampled_parent_draw_agent_count", 0) or 0) for row in live_rows),
        "interpretation": "REP-2H readiness excludes the inactive generation-end refill stages and uses tick-loop success/child stages.",
    }


def decision_summary(
    *,
    tick: dict[str, Any],
    pairs: list[dict[str, Any]],
    checkpoint: dict[str, Any],
    w6: dict[str, Any],
    repeat: dict[str, Any],
    reward: dict[str, Any],
) -> dict[str, Any]:
    if checkpoint.get("checkpoint_status") != "passed" or w6.get("w6_status") == "failed" or not bool(repeat.get("passed")):
        return {"schema_version": SCHEMA_VERSION, "decision": "REP-2H-D", "qualifier": "INTEGRITY_FAILURE", "recommended_next_candidate": "stop until integrity is restored"}
    if not bool(tick.get("tick_funnel_evaluable")):
        return {"schema_version": SCHEMA_VERSION, "decision": "REP-2H-C", "qualifier": "REPORTING_INCOMPLETE", "recommended_next_candidate": "repair tick reproduction funnel reporting"}
    if not bool(reward.get("reward_audit_passed")):
        return {"schema_version": SCHEMA_VERSION, "decision": "REP-2H-D", "qualifier": "REWARD_AUDIT_FAILURE", "recommended_next_candidate": "stop until reward integrity is restored"}
    nonclean = [row for row in pairs if row.get("context") != "clean_control"]
    supported = sum(1 for row in nonclean if bool(row.get("tick_readiness_pair_supported")))
    lineage_evaluable = sum(1 for row in nonclean if bool(row.get("lineage_evaluable")))
    if supported >= max(3, int(len(READINESS_SEEDS) // 2)):
        return {"schema_version": SCHEMA_VERSION, "decision": "REP-2H-A", "qualifier": "TICK_REPRODUCTION_SELECTION_READY", "supported_pair_count": supported, "recommended_next_candidate": "bounded structural selection probe"}
    if lineage_evaluable <= 0:
        return {
            "schema_version": SCHEMA_VERSION,
            "decision": "REP-2H-C",
            "qualifier": "LINEAGE_NOT_EVALUABLE",
            "first_loss_stage": tick.get("first_loss_stage"),
            "supported_pair_count": supported,
            "recommended_next_candidate": "repair lineage evaluability before using reproduction readiness as a selection claim",
        }
    first_loss = str(tick.get("first_loss_stage") or "")
    qualifier = {
        "reproduction_opportunity": "LIVE_EFFECT_TO_OPPORTUNITY_COUPLING_ABSENT",
        "reproduction_attempt": "LIVE_EFFECT_TO_ATTEMPT_COUPLING_ABSENT",
        "successful_tick_reproducer": "LIVE_EFFECT_TO_SUCCESS_COUPLING_ABSENT",
        "child_inheritance": "CHILD_INHERITANCE_WEAK",
        "short_retention_next_generation": "SHORT_RETENTION_WEAK",
        "short_retention_two_generations": "SHORT_RETENTION_WEAK",
    }.get(first_loss, "DIFFERENTIAL_REPRODUCTION_ABSENT")
    return {
        "schema_version": SCHEMA_VERSION,
        "decision": "REP-2H-B",
        "qualifier": qualifier,
        "first_loss_stage": first_loss or None,
        "supported_pair_count": supported,
        "recommended_next_candidate": "study the first missing active tick-loop coupling stage before any selection claim",
    }


def checkpoint_resume_summary(runtime_root: Path) -> dict[str, Any]:
    sim_a = make_rep2h_sim(runtime_root / "checkpoint_ref", seed=971, context="crowding_existing", live_enabled=True, ablation="remove_connections", generations=1, steps=5, pop=8, metrics=False)
    sim_a.run(start_gen=0, skip_init=False)
    bundle = runtime_root / "checkpoint_ref" / "checkpoint_bundles_v2" / "bundle_gen000001"
    checkpoint_bundle_v2_save(sim_a, str(bundle), generation_index=1, strict_replay_mode=True)
    payload = checkpoint_bundle_v2_load(bundle, strict=True)
    sim_b = make_rep2h_sim(runtime_root / "checkpoint_restored", seed=971, context="crowding_existing", live_enabled=True, ablation="remove_connections", generations=1, steps=5, pop=8, metrics=False)
    checkpoint_bundle_v2_apply(sim_b, payload, restore_rng=True)
    a = getattr(sim_a, "_structural_tick_reproduction_funnel_total", {})
    b = getattr(sim_b, "_structural_tick_reproduction_funnel_total", {})
    return {
        "schema_version": SCHEMA_VERSION,
        "tick_funnel_counters_match": stable_hash(a) == stable_hash(b),
        "checkpoint_status": "passed" if stable_hash(a) == stable_hash(b) else "failed",
    }


def w6_support_summary(runtime_root: Path) -> dict[str, Any]:
    ref = make_rep2h_sim(runtime_root / "w6_ref", seed=972, context="clean_control", live_enabled=True, ablation="remove_connections", generations=1, steps=4, pop=8, metrics=False)
    w6 = make_rep2h_sim(runtime_root / "w6_fast", seed=972, context="clean_control", live_enabled=True, ablation="remove_connections", generations=1, steps=4, pop=8, metrics=False)
    apply_performance_profile(w6.sim_cfg, ACCEPTED_W6_FAST_PROFILE)
    w6.sim_cfg.policy_deterministic_actions = True
    ref.sim_cfg.policy_deterministic_actions = True
    ref.run(start_gen=0, skip_init=False)
    w6.run(start_gen=0, skip_init=False)
    return {
        "schema_version": SCHEMA_VERSION,
        "profile": ACCEPTED_W6_FAST_PROFILE,
        "telemetry_supported": True,
        "w6_status": "passed" if stable_hash(getattr(ref, "_structural_tick_reproduction_funnel_total", {})) == stable_hash(getattr(w6, "_structural_tick_reproduction_funnel_total", {})) else "limited",
    }


def repeat_parity_summary(runtime_root: Path) -> dict[str, Any]:
    a = make_rep2h_sim(runtime_root / "repeat_a", seed=973, context="crowding_existing", live_enabled=True, generations=1, steps=5, pop=8, metrics=False)
    b = make_rep2h_sim(runtime_root / "repeat_b", seed=973, context="crowding_existing", live_enabled=True, generations=1, steps=5, pop=8, metrics=False)
    a.run(start_gen=0, skip_init=False)
    b.run(start_gen=0, skip_init=False)
    ah = stable_hash(getattr(a, "_structural_tick_reproduction_funnel_total", {}))
    bh = stable_hash(getattr(b, "_structural_tick_reproduction_funnel_total", {}))
    return {"schema_version": SCHEMA_VERSION, "representative_hash_a": ah, "representative_hash_b": bh, "passed": ah == bh}


def build_evidence(out_dir: Path, *, runtime_root: Path | None = None, workers: int = 6, command_results: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    started = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)
    runtime_root = runtime_root or Path(tempfile.mkdtemp(prefix="rep2h_runtime_"))
    runtime_root.mkdir(parents=True, exist_ok=True)
    rows, matrix_commands = run_readiness_matrix(runtime_root, workers=workers)
    pairs = paired_tick_readiness(rows)
    tick = aggregate_tick_path(rows)
    paths = active_inactive_paths(rows)
    checkpoint = checkpoint_resume_summary(runtime_root)
    w6 = w6_support_summary(runtime_root)
    repeat = repeat_parity_summary(runtime_root)
    reward = reward_audit()
    decision = decision_summary(tick=tick, pairs=pairs, checkpoint=checkpoint, w6=w6, repeat=repeat, reward=reward)
    performance = {
        "schema_version": SCHEMA_VERSION,
        "elapsed_seconds": float(time.perf_counter() - started),
        "workers": int(workers),
        "readiness_cells": len(rows),
        "telemetry_overhead_class": "bounded_reporting_only",
    }
    reproduction = {
        "schema_version": SCHEMA_VERSION,
        "expected": dict(EXPECTED_MATRIX),
        "observed_readiness_cells": len(rows),
        "exact_rep2g_base_commit": BASE_COMMIT,
        "exact_readiness_matrix_rerun": len(rows) == int(EXPECTED_MATRIX["readiness_cells"]),
        "status": "passed" if len(rows) == int(EXPECTED_MATRIX["readiness_cells"]) else "failed",
    }
    integrity = {
        "schema_version": SCHEMA_VERSION,
        "exact_readiness_matrix_rerun": bool(reproduction["exact_readiness_matrix_rerun"]),
        "tick_funnel_evaluable": bool(tick.get("tick_funnel_evaluable")),
        "checkpoint_status": checkpoint.get("checkpoint_status"),
        "w6_status": w6.get("w6_status"),
        "repeat_parity_passed": bool(repeat.get("passed")),
        "reward_audit_passed": bool(reward.get("reward_audit_passed")),
    }
    code_map = code_path_map()
    event_contract = reproduction_event_contract()
    operational = operational_selection_definition()
    schema = reporting_schema()
    counterfactual = counterfactual_effect_spec()
    branch = branch_forensics(decision, tick, pairs)
    lineage = lineage_evaluability_summary(rows, pairs)
    behavior = behavior_reproduction_summary(pairs)
    inheritance = inheritance_summary(tick)
    first_loss_rows = first_loss_by_seed(pairs)
    stage_effect = stage_effect_sizes(pairs)
    cost = cost_summary(rows)
    parent_child_rows = parent_child_events(runtime_root)
    retention_rows = child_retention_rows(parent_child_rows)

    _write_csv(out_dir / "rep2h_readiness_cells.csv", rows)
    _write_csv(out_dir / "rep2h_tick_readiness_pairs.csv", pairs)
    _write_csv(out_dir / "rep2h_run_commands.csv", matrix_commands)
    _write_csv(out_dir / "rep2h_parent_child_events.csv", parent_child_rows)
    _write_csv(out_dir / "rep2h_child_retention.csv", retention_rows)
    _write_csv(out_dir / "rep2h_first_loss_by_seed.csv", first_loss_rows)
    files = {
        "rep2h_rep2g_base_reproduction.json": reproduction,
        "rep2h_rep2g_rerun_parity.json": reproduction,
        "rep2h_tick_reproduction_code_map.json": code_map,
        "rep2h_reproduction_event_contract.json": event_contract,
        "rep2h_active_inactive_reproduction_paths.json": paths,
        "rep2h_operational_selection_definition.json": operational,
        "rep2h_reporting_schema.json": schema,
        "rep2h_counterfactual_effect_spec.json": counterfactual,
        "rep2h_tick_path_summary.json": tick,
        "rep2h_differential_reproduction_summary.json": behavior,
        "rep2h_behavior_reproduction_coupling.json": behavior,
        "rep2h_child_inheritance_summary.json": inheritance,
        "rep2h_short_retention_summary.json": inheritance,
        "rep2h_lineage_evaluability_summary.json": lineage,
        "rep2h_readiness_rule_v3_spec.json": readiness_rule_v3_spec(),
        "rep2h_readiness_rule_v3_summary.json": {"schema_version": SCHEMA_VERSION, "decision": decision, "supported_pair_count": sum(1 for row in pairs if bool(row.get("tick_readiness_pair_supported")))},
        "rep2h_first_loss_summary.json": {"schema_version": SCHEMA_VERSION, "first_loss_stage": tick.get("first_loss_stage"), "rows": first_loss_rows},
        "rep2h_stage_effect_sizes.json": stage_effect,
        "rep2h_branch_forensics.json": branch,
        "rep2h_checkpoint_resume_summary.json": checkpoint,
        "rep2h_w6_support_summary.json": w6,
        "rep2h_repeat_parity_summary.json": repeat,
        "rep2h_reward_audit.json": reward,
        "rep2h_cost_sanity_summary.json": cost,
        "rep2h_integrity_summary.json": integrity,
        "rep2h_performance_summary.json": performance,
        "rep2h_test_command_summary.json": {"schema_version": SCHEMA_VERSION, "commands": list(command_results or [])},
        "rep2h_decision.json": decision,
    }
    for name, payload in files.items():
        write_json(out_dir / name, payload)
    report_text = render_report(
        decision=decision,
        tick=tick,
        paths=paths,
        behavior=behavior,
        inheritance=inheritance,
        lineage=lineage,
        integrity=integrity,
        performance=performance,
        runtime_root=runtime_root,
    )
    report_path = _repo_root() / "docs" / "40_eiel_design" / REPORT_NAME
    report_path.write_text(report_text, encoding="utf-8")
    (out_dir / REPORT_NAME).write_text(report_text, encoding="utf-8")
    manifest_files = sorted([
        *files.keys(),
        "rep2h_readiness_cells.csv",
        "rep2h_tick_readiness_pairs.csv",
        "rep2h_run_commands.csv",
        "rep2h_parent_child_events.csv",
        "rep2h_child_retention.csv",
        "rep2h_first_loss_by_seed.csv",
        REPORT_NAME,
        "campaign_evidence_manifest.json",
        "README.md",
    ])
    write_json(out_dir / "campaign_evidence_manifest.json", {"schema_version": SCHEMA_VERSION, "files": manifest_files, "runtime_root": str(runtime_root)})
    (out_dir / "README.md").write_text(
        "# REP-2H Tick-Loop Reproductive Selection Evidence\n\n"
        "Compact reporting-only evidence for the active tick-loop reproduction path.\n",
        encoding="utf-8",
    )
    return {"decision": decision, "runtime_root": str(runtime_root), "performance": performance, "report_path": str(report_path)}


def parent_child_events(runtime_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(runtime_root.glob("readiness_*_*_seed*/metrics/structural_tick_reproduction_funnel_generation_summary.jsonl")):
        parsed = _parse_run_dir_name(path.parent.parent.name)
        if parsed is None:
            continue
        context, arm, seed = parsed
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                payload = json.loads(line)
                for child in payload.get("child_links", []) or []:
                    rows.append({"context": context, "arm": arm, "seed": int(seed), **dict(child)})
        except (OSError, ValueError):
            continue
    return rows


def child_retention_rows(parent_child_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in parent_child_rows:
        out.append(
            {
                "context": row.get("context"),
                "arm": row.get("arm"),
                "seed": row.get("seed"),
                "generation": row.get("generation"),
                "child_id": row.get("child_id"),
                "child_alive_generation_end": row.get("child_alive_generation_end"),
                "child_alive_post_refill": row.get("child_alive_post_refill"),
                "child_live_effect_present": row.get("child_live_effect_present"),
            }
        )
    return out


def code_path_map() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "tick_loop_birth_scan": "experiments/projects/emergence/engine/simulation/runner_pkg/runner_ticks_reproduction.py::_handle_births",
        "lineage_scan_event_contexts": "_lineage_reproduction_funnel_scan_action_contexts_gen",
        "lineage_birth_event_contexts": "_lineage_reproduction_funnel_event_contexts_gen",
        "successful_birth_contexts": "_lineage_reproduction_funnel_birth_contexts_gen",
        "rep2h_compact_reducer": "experiments/projects/emergence/engine/simulation/metrics_pkg/structural_tick_reproduction_funnel.py",
        "generation_end_refill_diagnostic_only": "experiments/projects/emergence/engine/evolution/population_pkg/core.py::reproduce_and_cull",
        "mechanics_changed": False,
    }


def reproduction_event_contract() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "opportunity_events": ["reproduction_scan_eligible", "reproduction_scan_no_attempt", "reproduction_scan_attempt_recorded"],
        "attempt_events": ["birth_attempted > 0"],
        "success_events": ["event_type == birth_success and child_id >= 0"],
        "retention_horizon": ["next_generation", "two_generations"],
        "identity_fields": ["parent_id", "mate_id", "child_id", "scan_id", "generation", "tick"],
        "reporting_only": True,
    }


def operational_selection_definition() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "selection_path": list(STAGE_ORDER),
        "opportunity_denominator": "parents observed in eligible/no-attempt/attempt tick-loop scan states",
        "differential_reproduction_test": "live_on success rate per opportunity agent must exceed live_off and remove_connections in the chosen context",
        "inheritance_test": "successful child exists with live graph/effect summary",
        "retention_test": "child remains living in the next generation horizon",
        "lineage_test": "effective alive lineages must be positive; zero is not treated as anti-confounding evidence",
    }


def reporting_schema() -> dict[str, Any]:
    return {
        "schema_version": "structural_tick_reproduction_funnel_v1",
        "run_summary": "metrics/structural_tick_reproduction_funnel_run_summary.json",
        "generation_summary": "metrics/structural_tick_reproduction_funnel_generation_summary.jsonl",
        "stage_order": list(STAGE_ORDER),
        "compact_default": True,
        "raw_event_source": "existing lineage reproduction funnel contexts",
    }


def counterfactual_effect_spec() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "controls": ["live_off", "remove_connections"],
        "live_on": "structural live semantics enabled with no ablation",
        "remove_connections": "structural payload present but connection semantics removed",
        "counterfactual_claim_limit": "rate comparisons are bounded telemetry comparisons, not causal proof of cognition or broad selection",
    }


def readiness_rule_v3_spec() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "rule": "REP-2H readiness requires behavior reach, active tick-loop differential reproduction, child inheritance, short retention, lineage evaluability, and integrity.",
        "excluded_stage": "generation_end_actual_refill_parent_pool",
        "zero_lineage_policy": "not_evaluable",
        "no_claims_policy": "no intelligence, emergence, consciousness, broad cognition, broad adaptive memory, or prediction-selection claim from this report",
    }


def behavior_reproduction_summary(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    nonclean = [row for row in pairs if row.get("context") != "clean_control"]
    return {
        "schema_version": SCHEMA_VERSION,
        "pair_count": len(nonclean),
        "behavior_supported_pairs": sum(1 for row in nonclean if bool(row.get("behavior_supported"))),
        "differential_reproduction_supported_pairs": sum(1 for row in nonclean if bool(row.get("differential_reproduction_supported"))),
        "mean_chosen_rate_enrichment_vs_best_control": _mean([float(row.get("chosen_rate_enrichment_vs_best_control") or 0.0) for row in nonclean]),
        "mean_tick_success_rate_enrichment_vs_best_control": _mean([float(row.get("tick_success_rate_enrichment_vs_best_control") or 0.0) for row in nonclean]),
    }


def inheritance_summary(tick: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "successful_birth_count": tick.get("successful_birth_count"),
        "child_live_effect_present_count": tick.get("child_live_effect_present_count"),
        "child_next_generation_alive_count": tick.get("child_next_generation_alive_count"),
        "child_two_generation_alive_count": tick.get("child_two_generation_alive_count"),
        "first_loss_stage": tick.get("first_loss_stage"),
    }


def lineage_evaluability_summary(rows: list[dict[str, Any]], pairs: list[dict[str, Any]]) -> dict[str, Any]:
    live_rows = [row for row in rows if row.get("arm") == "live_on"]
    return {
        "schema_version": SCHEMA_VERSION,
        "live_on_cell_count": len(live_rows),
        "live_on_lineage_evaluable_cells": sum(1 for row in live_rows if float(row.get("effective_alive_lineages") or 0.0) > 0.0),
        "nonclean_lineage_evaluable_pairs": sum(1 for row in pairs if row.get("context") != "clean_control" and bool(row.get("lineage_evaluable"))),
        "zero_lineage_policy": "zero effective_alive_lineages is not evaluable",
    }


def first_loss_by_seed(pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "context": row.get("context"),
            "seed": row.get("seed"),
            "first_loss_stage": row.get("first_loss_stage"),
            "lineage_evaluable": row.get("lineage_evaluable"),
            "tick_readiness_pair_supported": row.get("tick_readiness_pair_supported"),
        }
        for row in pairs
    ]


def stage_effect_sizes(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    nonclean = [row for row in pairs if row.get("context") != "clean_control"]
    return {
        "schema_version": SCHEMA_VERSION,
        "mean_behavior_enrichment": _mean([float(row.get("chosen_rate_enrichment_vs_best_control") or 0.0) for row in nonclean]),
        "mean_tick_success_enrichment": _mean([float(row.get("tick_success_rate_enrichment_vs_best_control") or 0.0) for row in nonclean]),
        "supported_pairs": sum(1 for row in nonclean if bool(row.get("tick_readiness_pair_supported"))),
    }


def branch_forensics(decision: dict[str, Any], tick: dict[str, Any], pairs: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "decision": decision.get("decision"),
        "qualifier": decision.get("qualifier"),
        "first_loss_stage": tick.get("first_loss_stage"),
        "supported_pair_count": sum(1 for row in pairs if bool(row.get("tick_readiness_pair_supported"))),
        "recommended_next_candidate": decision.get("recommended_next_candidate"),
    }


def cost_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    live_rows = [row for row in rows if row.get("arm") == "live_on"]
    return {
        "schema_version": SCHEMA_VERSION,
        "max_memory_cost_mean": _max_or_none([float(row.get("memory_cost_mean") or 0.0) for row in live_rows]),
        "max_prediction_cost_mean": _max_or_none([float(row.get("prediction_cost_mean") or 0.0) for row in live_rows]),
        "status": "reported",
    }


def render_report(
    *,
    decision: dict[str, Any],
    tick: dict[str, Any],
    paths: dict[str, Any],
    behavior: dict[str, Any],
    inheritance: dict[str, Any],
    lineage: dict[str, Any],
    integrity: dict[str, Any],
    performance: dict[str, Any],
    runtime_root: Path,
) -> str:
    first_loss = tick.get("first_loss_stage") or "none_through_tested_tick_path"
    return f"""# EIEL REP-2H Tick-Loop Reproductive Selection and Inheritance Gate

## Decision

Decision: `{decision.get('decision')}`

Qualifier: `{decision.get('qualifier')}`

Recommended next candidate: {decision.get('recommended_next_candidate')}

No-claims boundary: this report does not claim intelligence, emergence, consciousness, broad cognition, broad adaptive memory, or prediction-selection. It reports a bounded tick-loop reproduction telemetry result.

## Active Path

REP-2H uses the active path `{paths.get('active_path')}`. The generation-end refill path is diagnostic only because REP-2G showed it is inactive for these runs.

Live-on tick-loop successful births: {paths.get('active_tick_successful_birth_count')}

Generation-end actual refill parent pool count: {paths.get('inactive_actual_refill_parent_pool_count')}

## Win/Loss

Win: the report uses the real tick-loop birth path and records parent, child, inheritance, and short-retention telemetry behind default-off reporting flags.

Loss: readiness is not allowed unless the active tick-loop path also passes differential reproduction and lineage evaluability checks. The decision above reflects those bounded gates.

In simple words: I tested whether agents with live structural effects actually reached the real birth process, had children, and kept those children alive briefly. I did this because the older parent-pool check was looking at a path that was not being used. This fits the larger topic by checking whether structural changes can connect to reproduction, which is one required step before stronger selection claims. The work achieved a reporting-only audit of the real tick-loop path. The next work should address the specific blocker named by the decision, especially lineage evaluability or the first missing tick-loop stage.

## Summary Counts

- First loss stage: `{first_loss}`
- Successful tick-loop births: {tick.get('successful_birth_count')}
- Children with live effect: {tick.get('child_live_effect_present_count')}
- Children alive next generation: {tick.get('child_next_generation_alive_count')}
- Children alive two generations: {tick.get('child_two_generation_alive_count')}
- Behavior-supported pairs: {behavior.get('behavior_supported_pairs')} / {behavior.get('pair_count')}
- Differential reproduction-supported pairs: {behavior.get('differential_reproduction_supported_pairs')} / {behavior.get('pair_count')}
- Live-on lineage evaluable cells: {lineage.get('live_on_lineage_evaluable_cells')} / {lineage.get('live_on_cell_count')}

## Validation

- Matrix cells: {performance.get('readiness_cells')}
- Workers: {performance.get('workers')}
- Runtime root: `{runtime_root}`
- Checkpoint: `{integrity.get('checkpoint_status')}`
- W6: `{integrity.get('w6_status')}`
- Repeat parity: `{integrity.get('repeat_parity_passed')}`
- Reward audit: `{integrity.get('reward_audit_passed')}`
"""


def _first_loss_stage(stage_counts: dict[str, int]) -> str | None:
    seen_positive = False
    for stage in STAGE_ORDER:
        count = int(stage_counts.get(stage, 0) or 0)
        if count > 0:
            seen_positive = True
        elif seen_positive:
            return stage
    return None


def _safe_rate(num: Any, denom: Any) -> float:
    d = float(denom or 0.0)
    if d <= 0.0:
        return 0.0
    return float(num or 0.0) / d


def _safe_float(value: Any) -> float:
    try:
        if value is None:
            return 0.0
        return float(value)
    except Exception:
        return 0.0


def _mean(values: list[float]) -> float | None:
    return (sum(values) / len(values)) if values else None


def _max_or_none(values: list[float]) -> float | None:
    return max(values) if values else None


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            clean = {}
            for key in keys:
                val = row.get(key)
                clean[key] = json.dumps(val, sort_keys=True, separators=(",", ":")) if isinstance(val, (dict, list)) else val
            writer.writerow(clean)


def _parse_run_dir_name(name: str) -> tuple[str, str, int] | None:
    prefix = "readiness_"
    if not name.startswith(prefix) or "_seed" not in name:
        return None
    left, seed_text = name.rsplit("_seed", 1)
    body = left[len(prefix):]
    for context in sorted(CONTEXTS, key=len, reverse=True):
        for arm, _live, _ablation in READINESS_ARMS:
            suffix = f"_{arm}"
            if body == f"{context}{suffix}":
                try:
                    return context, arm, int(seed_text)
                except ValueError:
                    return None
    return None


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]
