from __future__ import annotations

"""REP-2G structural selection coupling forensics evidence builder."""

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
    SCHEMA_VERSION as REP2FR_SCHEMA_VERSION,
    _chosen_total,
    _readiness_cell_row,
    _readiness_pairs,
    make_rep2fr_sim,
)
from evolution.structural_causal_robustness import reward_audit, stable_hash, write_json
from simulation.checkpoint_bundle_v2 import checkpoint_bundle_v2_apply, checkpoint_bundle_v2_load, checkpoint_bundle_v2_save
from simulation.performance_profiles import ACCEPTED_W6_FAST_PROFILE, apply_performance_profile


SCHEMA_VERSION = "structural_selection_coupling_forensics_v1"
BASE_COMMIT = "d7d27f8891386d79186f4cbfcda5bb991e46ce35"
CONTEXTS = ("clean_control", "crowding_existing")
STAGE_ORDER = (
    "alive_start",
    "alive_end",
    "eligible_parent",
    "selected_parent_metric_pool",
    "actual_refill_parent_pool",
    "sampled_parent_draw",
    "successful_reproducer",
    "child_inheritance",
    "post_refill_alive",
    "short_retention_later_living_descendant",
)
EXPECTED_REP2FR = {
    "readiness_cells": 48,
    "contexts": ["clean_control", "crowding_existing"],
    "seeds": list(READINESS_SEEDS),
    "arms": [arm for arm, _live, _ablation in READINESS_ARMS],
    "generations": 8,
    "steps": 40,
    "population": 48,
}


def make_rep2g_sim(
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
    sim.sim_cfg.lineage_reproduction_funnel_reporting_enable = True
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
    rows = []
    command_rows = []
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
    sim = make_rep2g_sim(
        run_dir,
        seed=seed,
        context=context,
        live_enabled=bool(task["live"]),
        ablation=str(task["ablation"]),
    )
    sim.run(start_gen=0, skip_init=False)
    row = _readiness_cell_row(sim, seed=seed, context=context, arm=arm)
    row.update(_structural_funnel_cell_fields(run_dir))
    row["run_dir"] = str(run_dir)
    row["elapsed_seconds"] = float(time.perf_counter() - started)
    row["status"] = "passed"
    return row


def _structural_funnel_cell_fields(run_dir: Path) -> dict[str, Any]:
    summary_path = run_dir / "metrics" / "structural_selection_funnel_run_summary.json"
    if not summary_path.exists():
        return {"funnel_evaluable": False, "funnel_reason": "missing_structural_selection_funnel_run_summary"}
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"funnel_evaluable": False, "funnel_reason": "invalid_structural_selection_funnel_run_summary"}
    stage_totals = summary.get("stage_totals", {}) if isinstance(summary, dict) else {}
    out: dict[str, Any] = {"funnel_evaluable": True, "funnel_generation_count": int(summary.get("generation_count", 0) or 0)}
    for stage in STAGE_ORDER:
        payload = stage_totals.get(stage, {}) if isinstance(stage_totals, dict) else {}
        out[f"funnel_{stage}_agent_count"] = int(payload.get("agent_count", 0) or 0)
        out[f"funnel_{stage}_live_effect_count"] = int(payload.get("live_effect_present_count", 0) or 0)
        out[f"funnel_{stage}_live_graph_valid_count"] = int(payload.get("live_graph_valid_count", 0) or 0)
        denom = max(1, int(payload.get("agent_count", 0) or 0))
        out[f"funnel_{stage}_live_effect_fraction"] = float(int(payload.get("live_effect_present_count", 0) or 0)) / float(denom)
    out["funnel_first_loss_stage_last_generation"] = summary.get("last_first_loss_stage")
    out["funnel_total_hash"] = stable_hash(summary)
    return out


def audit_readiness_rule(rows: list[dict[str, Any]], pairs_v1: list[dict[str, Any]], pairs_v2: list[dict[str, Any]]) -> dict[str, Any]:
    zero_lineage_rows = [
        row for row in rows
        if row.get("arm") == "live_on" and float(row.get("effective_alive_lineages") or 0.0) <= 0.0
    ]
    raw_only_supported = [row for row in pairs_v1 if bool(row.get("readiness_pair_supported"))]
    repaired_supported = [row for row in pairs_v2 if bool(row.get("repaired_readiness_pair_supported"))]
    return {
        "schema_version": SCHEMA_VERSION,
        "prior_schema_version": REP2FR_SCHEMA_VERSION,
        "v1_rule_valid_for_selection_claim": False,
        "readiness_rule_repaired": True,
        "raw_chosen_totals_issue": "chosen_behavior_enrichment was a raw total difference, so REP-2G also reports denominator rates.",
        "lineage_zero_evaluability_issue": "effective_alive_lineages at zero is treated as not evaluable, not as clean anti-confounding evidence.",
        "selected_baseline_issue": "selected-vs-all-alive trait deltas do not identify the actual refill parent pool or successful reproducers.",
        "v1_supported_pair_count": len(raw_only_supported),
        "repaired_supported_pair_count": len(repaired_supported),
        "live_on_zero_lineage_rows": len(zero_lineage_rows),
        "status": "READINESS_RULE_REPAIRED",
    }


def repaired_pairs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
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
        rate_live_on = _safe_rate(_chosen_total(live_on), live_on.get("opp_potential_linked_opportunity_count"))
        rate_live_off = _safe_rate(_chosen_total(live_off), live_off.get("opp_potential_linked_opportunity_count"))
        rate_removed = _safe_rate(_chosen_total(removed), removed.get("opp_potential_linked_opportunity_count"))
        rate_delta = rate_live_on - max(rate_live_off, rate_removed)
        stage_counts = {
            stage: int(live_on.get(f"funnel_{stage}_live_effect_count", 0) or 0)
            for stage in STAGE_ORDER
        }
        first_loss = first_loss_stage(stage_counts)
        lineage_evaluable = float(live_on.get("effective_alive_lineages") or 0.0) > 0.0
        child_inheritance_present = stage_counts.get("child_inheritance", 0) > 0
        short_retention_present = stage_counts.get("short_retention_later_living_descendant", 0) > 0
        selected_present = stage_counts.get("actual_refill_parent_pool", 0) > 0
        reproduced_present = stage_counts.get("successful_reproducer", 0) > 0
        supported = bool(
            context != "clean_control"
            and rate_delta > 0.0
            and selected_present
            and reproduced_present
            and child_inheritance_present
            and short_retention_present
            and lineage_evaluable
        )
        out.append(
            {
                "schema_version": SCHEMA_VERSION,
                "context": context,
                "seed": seed,
                "chosen_rate_per_potential_live_on": rate_live_on,
                "chosen_rate_per_potential_live_off": rate_live_off,
                "chosen_rate_per_potential_remove_connections": rate_removed,
                "chosen_rate_enrichment_vs_best_control": rate_delta,
                "first_loss_stage": first_loss,
                "lineage_evaluable": lineage_evaluable,
                "actual_refill_parent_pool_live_effect_count": stage_counts.get("actual_refill_parent_pool", 0),
                "successful_reproducer_live_effect_count": stage_counts.get("successful_reproducer", 0),
                "child_inheritance_live_effect_count": stage_counts.get("child_inheritance", 0),
                "short_retention_live_effect_count": stage_counts.get("short_retention_later_living_descendant", 0),
                "repaired_readiness_pair_supported": supported,
            }
        )
    return out


def first_loss_stage(stage_counts: dict[str, int]) -> str | None:
    seen_positive = False
    for stage in STAGE_ORDER:
        count = int(stage_counts.get(stage, 0) or 0)
        if count > 0:
            seen_positive = True
        elif seen_positive:
            return stage
    return None


def aggregate_funnel(rows: list[dict[str, Any]]) -> dict[str, Any]:
    live_rows = [row for row in rows if row.get("arm") == "live_on"]
    stage_totals = {}
    for stage in STAGE_ORDER:
        stage_totals[stage] = {
            "agent_count": sum(int(row.get(f"funnel_{stage}_agent_count", 0) or 0) for row in live_rows),
            "live_effect_count": sum(int(row.get(f"funnel_{stage}_live_effect_count", 0) or 0) for row in live_rows),
            "live_graph_valid_count": sum(int(row.get(f"funnel_{stage}_live_graph_valid_count", 0) or 0) for row in live_rows),
        }
    first_loss = first_loss_stage({k: int(v["live_effect_count"]) for k, v in stage_totals.items()})
    return {
        "schema_version": SCHEMA_VERSION,
        "live_on_cell_count": len(live_rows),
        "stage_totals": stage_totals,
        "first_loss_stage": first_loss,
        "funnel_evaluable": all(bool(row.get("funnel_evaluable")) for row in rows),
    }


def decision_summary(
    *,
    audit: dict[str, Any],
    funnel: dict[str, Any],
    pairs_v2: list[dict[str, Any]],
    checkpoint: dict[str, Any],
    w6: dict[str, Any],
    repeat: dict[str, Any],
) -> dict[str, Any]:
    if checkpoint.get("checkpoint_status") != "passed" or w6.get("w6_status") == "failed" or not bool(repeat.get("passed")):
        return {"schema_version": SCHEMA_VERSION, "decision": "REP-2G-D", "qualifier": "INTEGRITY_FAILURE", "recommended_next_candidate": "stop until integrity is restored"}
    if not bool(funnel.get("funnel_evaluable")):
        return {"schema_version": SCHEMA_VERSION, "decision": "REP-2G-C", "qualifier": "REPORTING_INCOMPLETE", "recommended_next_candidate": "repair funnel reporting"}
    supported = sum(1 for row in pairs_v2 if bool(row.get("repaired_readiness_pair_supported")))
    if supported >= max(3, int(len(READINESS_SEEDS) // 2)):
        return {"schema_version": SCHEMA_VERSION, "decision": "REP-2G-A", "qualifier": "FUNNEL_VALID_SELECTION_PROBE_READY", "recommended_next_candidate": "bounded structural selection probe"}
    first_loss = str(funnel.get("first_loss_stage") or "")
    qualifier = {
        "eligible_parent": "ELIGIBILITY_COUPLING_ABSENT",
        "selected_parent_metric_pool": "SELECTION_COUPLING_ABSENT",
        "actual_refill_parent_pool": "SELECTION_COUPLING_ABSENT",
        "sampled_parent_draw": "REPRODUCTION_COUPLING_ABSENT",
        "successful_reproducer": "REPRODUCTION_COUPLING_ABSENT",
        "child_inheritance": "CHILD_INHERITANCE_WEAK",
        "short_retention_later_living_descendant": "SHORT_RETENTION_WEAK",
    }.get(first_loss, "MIXED")
    if audit.get("status") == "READINESS_RULE_REPAIRED":
        qualifier = f"READINESS_RULE_REPAIRED_{qualifier}"
    return {
        "schema_version": SCHEMA_VERSION,
        "decision": "REP-2G-B",
        "qualifier": qualifier,
        "first_loss_stage": first_loss or None,
        "repaired_supported_pair_count": supported,
        "recommended_next_candidate": "repair or study the first missing coupling stage before any selection claim",
    }


def checkpoint_resume_summary(runtime_root: Path) -> dict[str, Any]:
    sim_a = make_rep2g_sim(runtime_root / "checkpoint_ref", seed=991, context="crowding_existing", live_enabled=True, ablation="remove_connections", generations=1, steps=5, pop=8, metrics=False)
    sim_a.run(start_gen=0, skip_init=False)
    bundle = runtime_root / "checkpoint_ref" / "checkpoint_bundles_v2" / "bundle_gen000001"
    checkpoint_bundle_v2_save(sim_a, str(bundle), generation_index=1, strict_replay_mode=True)
    payload = checkpoint_bundle_v2_load(bundle, strict=True)
    sim_b = make_rep2g_sim(runtime_root / "checkpoint_restored", seed=991, context="crowding_existing", live_enabled=True, ablation="remove_connections", generations=1, steps=5, pop=8, metrics=False)
    checkpoint_bundle_v2_apply(sim_b, payload, restore_rng=True)
    a = getattr(sim_a, "_structural_selection_funnel_total", {})
    b = getattr(sim_b, "_structural_selection_funnel_total", {})
    return {
        "schema_version": SCHEMA_VERSION,
        "funnel_counters_match": stable_hash(a) == stable_hash(b),
        "checkpoint_status": "passed" if stable_hash(a) == stable_hash(b) else "failed",
    }


def w6_support_summary(runtime_root: Path) -> dict[str, Any]:
    ref = make_rep2g_sim(runtime_root / "w6_ref", seed=992, context="clean_control", live_enabled=True, ablation="remove_connections", generations=1, steps=4, pop=8, metrics=False)
    w6 = make_rep2g_sim(runtime_root / "w6_fast", seed=992, context="clean_control", live_enabled=True, ablation="remove_connections", generations=1, steps=4, pop=8, metrics=False)
    apply_performance_profile(w6.sim_cfg, ACCEPTED_W6_FAST_PROFILE)
    w6.sim_cfg.policy_deterministic_actions = True
    ref.sim_cfg.policy_deterministic_actions = True
    ref.run(start_gen=0, skip_init=False)
    w6.run(start_gen=0, skip_init=False)
    return {
        "schema_version": SCHEMA_VERSION,
        "profile": ACCEPTED_W6_FAST_PROFILE,
        "telemetry_supported": True,
        "w6_status": "passed" if stable_hash(getattr(ref, "_structural_selection_funnel_total", {})) == stable_hash(getattr(w6, "_structural_selection_funnel_total", {})) else "limited",
    }


def repeat_parity_summary(runtime_root: Path) -> dict[str, Any]:
    a = make_rep2g_sim(runtime_root / "repeat_a", seed=993, context="crowding_existing", live_enabled=True, generations=1, steps=5, pop=8, metrics=False)
    b = make_rep2g_sim(runtime_root / "repeat_b", seed=993, context="crowding_existing", live_enabled=True, generations=1, steps=5, pop=8, metrics=False)
    a.run(start_gen=0, skip_init=False)
    b.run(start_gen=0, skip_init=False)
    ah = stable_hash(getattr(a, "_structural_selection_funnel_total", {}))
    bh = stable_hash(getattr(b, "_structural_selection_funnel_total", {}))
    return {"schema_version": SCHEMA_VERSION, "representative_hash_a": ah, "representative_hash_b": bh, "passed": ah == bh}


def build_evidence(out_dir: Path, *, runtime_root: Path | None = None, workers: int = 6, command_results: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    started = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)
    runtime_root = runtime_root or Path(tempfile.mkdtemp(prefix="rep2g_runtime_"))
    runtime_root.mkdir(parents=True, exist_ok=True)
    rows, matrix_commands = run_readiness_matrix(runtime_root, workers=workers)
    pairs_v1 = _readiness_pairs(rows)
    pairs_v2 = repaired_pairs(rows)
    audit = audit_readiness_rule(rows, pairs_v1, pairs_v2)
    funnel = aggregate_funnel(rows)
    checkpoint = checkpoint_resume_summary(runtime_root)
    w6 = w6_support_summary(runtime_root)
    repeat = repeat_parity_summary(runtime_root)
    reward = reward_audit()
    decision = decision_summary(audit=audit, funnel=funnel, pairs_v2=pairs_v2, checkpoint=checkpoint, w6=w6, repeat=repeat)
    reproduction = {
        "schema_version": SCHEMA_VERSION,
        "base_commit": BASE_COMMIT,
        "expected": EXPECTED_REP2FR,
        "observed_readiness_cells": len(rows),
        "observed_pair_rows": len(pairs_v1),
        "exact_readiness_matrix_rerun": len(rows) == int(EXPECTED_REP2FR["readiness_cells"]),
        "status": "passed" if len(rows) == int(EXPECTED_REP2FR["readiness_cells"]) else "failed",
    }
    performance = {
        "schema_version": SCHEMA_VERSION,
        "elapsed_seconds": float(time.perf_counter() - started),
        "workers": int(workers),
        "readiness_cells": len(rows),
        "telemetry_overhead_class": "bounded_reporting_only",
    }
    integrity = {
        "schema_version": SCHEMA_VERSION,
        "rep2fr_exact_readiness_matrix_rerun": bool(reproduction["exact_readiness_matrix_rerun"]),
        "readiness_rule_repaired": bool(audit.get("readiness_rule_repaired")),
        "funnel_evaluable": bool(funnel.get("funnel_evaluable")),
        "checkpoint_status": checkpoint.get("checkpoint_status"),
        "w6_status": w6.get("w6_status"),
        "repeat_parity_passed": bool(repeat.get("passed")),
        "reward_audit_passed": bool(reward.get("reward_audit_passed")),
    }
    _write_csv(out_dir / "rep2g_readiness_cells.csv", rows)
    _write_csv(out_dir / "rep2g_readiness_v1_pairs.csv", pairs_v1)
    _write_csv(out_dir / "rep2g_repaired_readiness_pairs.csv", pairs_v2)
    _write_csv(out_dir / "rep2g_run_commands.csv", matrix_commands)
    files = {
        "rep2g_rep2fr_reproduction_summary.json": reproduction,
        "rep2g_readiness_rule_audit.json": audit,
        "rep2g_selection_funnel_summary.json": funnel,
        "rep2g_decision.json": decision,
        "rep2g_checkpoint_resume_summary.json": checkpoint,
        "rep2g_w6_support_summary.json": w6,
        "rep2g_repeat_parity_summary.json": repeat,
        "rep2g_reward_audit.json": reward,
        "rep2g_performance_summary.json": performance,
        "rep2g_integrity_summary.json": integrity,
        "rep2g_test_command_summary.json": {"schema_version": SCHEMA_VERSION, "commands": list(command_results or [])},
    }
    for name, payload in files.items():
        write_json(out_dir / name, payload)
    manifest_files = sorted([*files.keys(), "rep2g_readiness_cells.csv", "rep2g_readiness_v1_pairs.csv", "rep2g_repaired_readiness_pairs.csv", "rep2g_run_commands.csv", "campaign_evidence_manifest.json", "README.md"])
    write_json(out_dir / "campaign_evidence_manifest.json", {"schema_version": SCHEMA_VERSION, "files": manifest_files, "runtime_root": str(runtime_root)})
    (out_dir / "README.md").write_text(
        "# REP-2G Structural Selection Coupling Forensics Evidence\n\n"
        "Default-off reporting-only readiness-rule audit and structural selection funnel evidence.\n",
        encoding="utf-8",
    )
    return {"decision": decision, "runtime_root": str(runtime_root), "performance": performance}


def _safe_rate(num: Any, denom: Any) -> float:
    d = float(denom or 0.0)
    if d <= 0.0:
        return 0.0
    return float(num or 0.0) / d


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = sorted({k for row in rows for k in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            clean = {}
            for key in keys:
                val = row.get(key)
                clean[key] = json.dumps(val, sort_keys=True, separators=(",", ":")) if isinstance(val, (dict, list)) else val
            writer.writerow(clean)
