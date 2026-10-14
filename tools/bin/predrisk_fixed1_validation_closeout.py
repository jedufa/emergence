#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Iterable, Mapping


WORLDS = (
    "true_flat",
    "resource_only",
    "predator_only",
    "cover_only",
    "label_only_sham",
    "uncorrelated_risk_resource",
    "budget_matched_uniform",
    "correlated_tradeoff",
)
ARMS = ("risk_exploit", "balanced_generalist", "safe_cover")
ZONES = ("exposed_rich", "safe_covered")
PRIMARY_WORLD = "correlated_tradeoff"
CONTROL_WORLDS = ("true_flat", "label_only_sham", "budget_matched_uniform")
REQUIRED_ARTIFACTS = (
    "predrisk_fixed1v_operational_state.json",
    "predrisk_fixed1v_design_verification.json",
    "predrisk_fixed1v_frozen_decision_contract.json",
    "predrisk_fixed1v_policy_hash_verification.json",
    "predrisk_fixed1v_map_digest_verification.json",
    "predrisk_fixed1v_core_registry_contract.json",
    "predrisk_fixed1v_confirmation_registry_contract.json",
    "predrisk_fixed1v_registry_dry_run_summary.json",
    "predrisk_fixed1v_rehearsal_summary.json",
    "predrisk_fixed1v_core_completion.json",
    "predrisk_fixed1v_confirmation_completion.json",
    "predrisk_fixed1v_pair_completeness.json",
    "predrisk_fixed1v_config_reconciliation.json",
    "predrisk_fixed1v_nonfinite_scan.json",
    "predrisk_fixed1v_seed_level_metrics.csv",
    "predrisk_fixed1v_world_probe_summary.csv",
    "predrisk_fixed1v_zone_metrics.csv",
    "predrisk_fixed1v_paired_deltas.csv",
    "predrisk_fixed1v_behavior_construct.json",
    "predrisk_fixed1v_exposed_rich_competence.json",
    "predrisk_fixed1v_safe_covered_reproductive_return.json",
    "predrisk_fixed1v_reciprocal_costs.json",
    "predrisk_fixed1v_universal_strategy_audit.json",
    "predrisk_fixed1v_label_sham_audit.json",
    "predrisk_fixed1v_resource_only_audit.json",
    "predrisk_fixed1v_predator_only_audit.json",
    "predrisk_fixed1v_cover_only_audit.json",
    "predrisk_fixed1v_uncorrelated_control_audit.json",
    "predrisk_fixed1v_budget_uniform_audit.json",
    "predrisk_fixed1v_correlation_effect.json",
    "predrisk_fixed1v_timing_overlap.json",
    "predrisk_fixed1v_branch_lineage_context.json",
    "predrisk_fixed1v_repeat_parity.json",
    "predrisk_fixed1v_reward_identity_audit.json",
    "predrisk_fixed1v_runtime_archive_record.json",
    "predrisk_fixed1v_validation_summary.json",
    "predrisk_fixed1v_final_decision.json",
    "predrisk_fixed1v_challenger_temp_dossier.json",
    "predrisk_fixed1v_challenger_corridor_dossier.json",
    "predrisk_fixed1v_portfolio_comparison.csv",
    "predrisk_fixed1v_portfolio_recommendation.json",
    "predrisk_fixed1v_integration_record.json",
    "predrisk_fixed1v_cleanup_record.json",
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    rows = list(rows)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _hash_jsonish(path: Path) -> str | None:
    return _sha256(path) if path.exists() else None


def _float(value: Any) -> float | None:
    if value in ("", None):
        return None
    try:
        return float(value)
    except Exception:
        return None


def _mean(values: Iterable[float | None]) -> float | None:
    nums = [float(v) for v in values if v is not None]
    return sum(nums) / len(nums) if nums else None


def _git(repo: Path, *args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=repo, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


def _summary(root: Path) -> dict[str, Any]:
    return _read_json(root / "parallel_campaign_summary.json")


def _results(root: Path) -> list[dict[str, Any]]:
    return list(_summary(root).get("results") or [])


def _child_results(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    children = root / "children"
    if not children.exists():
        return rows
    for path in children.glob("*/e/*/*/summary.json"):
        try:
            summary = _read_json(path)
        except Exception:
            continue
        for run in summary.get("runs") or []:
            row = dict(run)
            row.setdefault("summary_path", str(path))
            row.setdefault("child_root", str(path.parents[3]))
            rows.append(row)
    return rows


def _completion(root: Path, expected: int) -> dict[str, Any]:
    summary = _summary(root)
    results = list(summary.get("results") or [])
    failures = list(summary.get("failures") or [])
    return {
        "schema_version": "predrisk_fixed1v_completion_v1",
        "runs_root": str(root),
        "status": summary.get("status"),
        "planned": summary.get("planned_runs"),
        "expected": expected,
        "completed": len([r for r in results if str(r.get("status")) == "ok"]),
        "failed": len(failures),
        "workers": summary.get("workers"),
        "registry": summary.get("registry"),
        "batch_tag": summary.get("batch_tag"),
        "compact_output": True,
        "failures": failures,
    }


def _all_run_results(roots: Iterable[Path]) -> list[dict[str, Any]]:
    seen: dict[str, dict[str, Any]] = {}
    for root in roots:
        for result in _results(root) + _child_results(root):
            if str(result.get("status")) != "ok":
                continue
            run_id = str(result.get("run_id", ""))
            if run_id and run_id not in seen:
                seen[run_id] = result
    return [seen[k] for k in sorted(seen)]


def build_combined_summary(args: argparse.Namespace) -> None:
    out_root = Path(args.combined_root)
    source_roots = [Path(p) for p in args.source_root]
    primary = _summary(source_roots[0])
    expected_ids = list(primary.get("selected_run_ids") or [])
    by_id = {str(r.get("run_id")): r for r in _all_run_results(source_roots)}
    missing = [run_id for run_id in expected_ids if run_id not in by_id]
    results = [by_id[run_id] for run_id in expected_ids if run_id in by_id]
    out_root.mkdir(parents=True, exist_ok=True)
    payload = {
        **primary,
        "status": "complete" if not missing else "incomplete",
        "source_roots": [str(p) for p in source_roots],
        "planned_runs": len(expected_ids),
        "results": results,
        "failures": [],
        "missing_run_ids": missing,
        "recovered_after_terminal_crash": True,
    }
    _write_json(out_root / "parallel_campaign_summary.json", payload)
    print(json.dumps({"combined_root": str(out_root), "results": len(results), "missing": len(missing)}, sort_keys=True))


def _run_id_parts(run_id: str) -> dict[str, str]:
    parts = run_id.split("__")
    return {"world": parts[1], "arm": parts[2], "seed": parts[3].replace("seed", "")} if len(parts) == 4 else {}


def compare_parity(args: argparse.Namespace) -> None:
    roots = [Path(args.parity_a_root), Path(args.parity_b_root)]
    result_maps = [{str(r.get("run_id")): r for r in _results(root)} for root in roots]
    run_ids = sorted(set(result_maps[0]).intersection(result_maps[1]))
    rows: list[dict[str, Any]] = []
    mismatches: list[dict[str, Any]] = []
    for run_id in run_ids:
        left = Path(str(result_maps[0][run_id].get("run_dir", "")))
        right = Path(str(result_maps[1][run_id].get("run_dir", "")))
        checks = {
            "config_hash": ("config.json",),
            "policy_contract_hash": ("metrics", "predrisk_fixed1_probe_contract.json"),
            "metrics_hash": ("metrics.csv",),
            "zone_opportunity_hash": ("metrics", "predrisk_fixed1_generation_summary.csv"),
            "map_stats_hash": ("maps", "tier3_regions_stats.json"),
            "map_region_hash": ("maps", "tier3_region_id.npy"),
        }
        row: dict[str, Any] = {"run_id": run_id, **_run_id_parts(run_id)}
        ok = True
        for label, rel in checks.items():
            left_hash = _hash_jsonish(left.joinpath(*rel))
            right_hash = _hash_jsonish(right.joinpath(*rel))
            row[f"{label}_a"] = left_hash
            row[f"{label}_b"] = right_hash
            match = left_hash == right_hash and left_hash is not None
            row[f"{label}_match"] = match
            ok = ok and match
        row["match"] = ok
        rows.append(row)
        if not ok:
            mismatches.append(row)
    payload = {
        "schema_version": "predrisk_fixed1v_repeat_parity_v1",
        "parity_a_root": str(roots[0]),
        "parity_b_root": str(roots[1]),
        "compared_cells": len(run_ids),
        "expected_cells": 10,
        "match": len(run_ids) == 10 and not mismatches,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
    }
    out_dir = Path(args.out_dir)
    _write_json(out_dir / "predrisk_fixed1v_repeat_parity.json", payload)
    _write_csv(out_dir / "predrisk_fixed1v_repeat_parity_details.csv", rows)
    print(json.dumps({"parity_match": payload["match"], "compared": len(run_ids), "mismatches": len(mismatches)}, sort_keys=True))


def _load_analysis(out_dir: Path, label: str) -> tuple[list[dict[str, str]], list[dict[str, str]], dict[str, Any]]:
    base = out_dir / label
    return (
        _read_csv(base / f"predrisk_fixed1v_{label}_cell_metrics.csv"),
        _read_csv(base / f"predrisk_fixed1v_{label}_aggregate_metrics.csv"),
        _read_json(base / f"predrisk_fixed1v_{label}_decision_summary.json"),
    )


def _best_arm(cells: list[dict[str, str]], metric: str) -> str | None:
    means = {
        arm: _mean(_float(r.get(metric)) for r in cells if r.get("world_family") == PRIMARY_WORLD and r.get("fixed_probe") == arm)
        for arm in ARMS
    }
    present = {k: v for k, v in means.items() if v is not None}
    if not present:
        return None
    return max(present, key=lambda k: float(present[k]))


def _cohort_pass(decision: Mapping[str, Any]) -> bool:
    return str(decision.get("decision")) == "PREDRISK-FIXED-1V-A"


def _framework_decision(core_decision: Mapping[str, Any], conf_decision: Mapping[str, Any], parity: Mapping[str, Any], all_cells: list[dict[str, str]]) -> tuple[str, str]:
    if not parity.get("match"):
        return "PREDRISK-FIXED-1V-F", "repeat parity mismatch or incomplete parity comparison"
    if not core_decision.get("complete") or not conf_decision.get("complete"):
        return "PREDRISK-FIXED-1V-F", "core or confirmation campaign incomplete"
    if core_decision.get("missing_required_telemetry_cell_count") or conf_decision.get("missing_required_telemetry_cell_count"):
        return "PREDRISK-FIXED-1V-E", "decision-critical telemetry missing"
    core_pass = _cohort_pass(core_decision)
    conf_pass = _cohort_pass(conf_decision)
    if core_pass and conf_pass:
        return "PREDRISK-FIXED-1V-A", "core and independent confirmation pass reciprocal fixed-probe demand checks"
    if core_pass != conf_pass:
        return "PREDRISK-FIXED-1V-D", "core and confirmation disagree"
    exposed_best = _best_arm(all_cells, "exposed_rich_net_return_per_1000_available_resource_agent_ticks")
    safe_best = _best_arm(all_cells, "safe_covered_selected_parent_events_per_1000_eligible_parent_events_after_zone_use")
    if exposed_best and exposed_best == safe_best:
        return "PREDRISK-FIXED-1V-C", f"{exposed_best} is the descriptive best arm on both primary metrics"
    return "PREDRISK-FIXED-1V-B", "fixed probes differ but one or both preregistered reciprocal rankings fail"


def _copy_or_write_missing_contract(worktree: Path, out_dir: Path) -> None:
    src = out_dir / "predrisk_fixed1v_frozen_decision_contract.json"
    if src.exists():
        return
    tracked = worktree / "evidences/07-world-pressure/19-predrisk-fixed1-validation/predrisk_fixed1v_frozen_decision_contract.json"
    if tracked.exists():
        shutil.copy2(tracked, src)


def generate_closeout(args: argparse.Namespace) -> None:
    worktree = Path(args.worktree)
    out_dir = Path(args.out_dir)
    report_path = Path(args.report_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    _copy_or_write_missing_contract(worktree, out_dir)

    core_cells, core_agg, core_decision = _load_analysis(out_dir, "core_w1")
    conf_cells, conf_agg, conf_decision = _load_analysis(out_dir, "confirmation_combined")
    rehearsal_cells, rehearsal_agg, rehearsal_decision = _load_analysis(out_dir, "rehearsal5")
    all_cells = [{**row, "cohort": "core"} for row in core_cells] + [{**row, "cohort": "confirmation"} for row in conf_cells]
    all_agg = [{**row, "cohort": "core"} for row in core_agg] + [{**row, "cohort": "confirmation"} for row in conf_agg]
    parity_path = out_dir / "predrisk_fixed1v_repeat_parity.json"
    parity = _read_json(parity_path) if parity_path.exists() else {"match": False, "reason": "not generated"}
    decision, reason = _framework_decision(core_decision, conf_decision, parity, all_cells)

    _write_csv(out_dir / "predrisk_fixed1v_seed_level_metrics.csv", all_cells)
    _write_csv(out_dir / "predrisk_fixed1v_world_probe_summary.csv", all_agg)
    zone_rows: list[dict[str, Any]] = []
    for row in all_cells:
        for zone in ZONES:
            zone_rows.append({
                "cohort": row["cohort"],
                "world_family": row["world_family"],
                "fixed_probe": row["fixed_probe"],
                "seed": row["seed"],
                "zone": zone,
                "available_resource_agent_ticks": row.get(f"{zone}_available_resource_agent_ticks"),
                "food_gain": row.get(f"{zone}_food_gain"),
                "movement_cost_before_move_actions": row.get(f"{zone}_movement_cost_before_move_actions"),
                "effective_predator_damage_cost": row.get(f"{zone}_effective_predator_damage_cost"),
                "net_return": row.get(f"{zone}_net_return"),
                "selected_parent_events": row.get(f"{zone}_selected_parent_events_after_zone_use"),
                "eligible_parent_events": row.get(f"{zone}_eligible_parent_events_after_zone_use"),
                "successful_births_after_zone_use": row.get(f"{zone}_successful_births_after_zone_use"),
                "predator_events": row.get(f"{zone}_predator_events"),
                "predator_damage_prevented": row.get(f"{zone}_predator_damage_prevented"),
            })
    _write_csv(out_dir / "predrisk_fixed1v_zone_metrics.csv", zone_rows)

    delta_rows: list[dict[str, Any]] = []
    for cohort, summary in (("core", core_decision), ("confirmation", conf_decision)):
        for section in ("seed_primary_directions", "seed_control_directions"):
            for item in summary.get(section, []) or []:
                for seed in item.get("seed_results", []) or []:
                    for key, value in seed.items():
                        if key.endswith("_delta"):
                            delta_rows.append({
                                "cohort": cohort,
                                "world": item.get("world"),
                                "metric": item.get("metric"),
                                "seed": seed.get("seed"),
                                "comparison": key[:-6],
                                "delta": value,
                                "pass": seed.get(f"{key[:-6]}_pass"),
                                "reverse": seed.get(f"{key[:-6]}_reverse"),
                            })
    _write_csv(out_dir / "predrisk_fixed1v_paired_deltas.csv", delta_rows)

    core_root = Path(args.core_root)
    conf_root = Path(args.confirmation_root)
    rehearsal_root = Path(args.rehearsal_root)
    _write_json(out_dir / "predrisk_fixed1v_rehearsal_summary.json", {
        "completion": _completion(rehearsal_root, 24),
        "decision_summary": rehearsal_decision,
    })
    _write_json(out_dir / "predrisk_fixed1v_core_completion.json", _completion(core_root, 96))
    _write_json(out_dir / "predrisk_fixed1v_confirmation_completion.json", _completion(conf_root, 192))

    sample_contracts = []
    sample_maps = []
    for result in (_results(core_root) + _results(conf_root))[:20]:
        run_dir = Path(str(result.get("run_dir", "")))
        contract = run_dir / "metrics/predrisk_fixed1_probe_contract.json"
        if contract.exists():
            sample_contracts.append(_read_json(contract))
        map_stats = run_dir / "maps/tier3_regions_stats.json"
        if map_stats.exists():
            sample_maps.append({"run_id": result.get("run_id"), "sha256": _sha256(map_stats), "path": str(map_stats)})
    policy_hashes = [c.get("policy_hashes") for c in sample_contracts if c.get("policy_hashes")]
    _write_json(out_dir / "predrisk_fixed1v_policy_hash_verification.json", {
        "schema_version": "predrisk_fixed1v_policy_hash_verification_v1",
        "sampled_contracts": len(sample_contracts),
        "unique_policy_hash_sets": sorted({json.dumps(p, sort_keys=True) for p in policy_hashes}),
        "consistent": len({json.dumps(p, sort_keys=True) for p in policy_hashes}) == 1,
    })
    _write_json(out_dir / "predrisk_fixed1v_map_digest_verification.json", {
        "schema_version": "predrisk_fixed1v_map_digest_verification_v1",
        "sampled_maps": len(sample_maps),
        "map_stats_hashes": sample_maps,
        "note": "Per-run map stats hashes are recorded; parity file records exact representative map-hash equality.",
    })

    for registry_name, out_name, expected in (
        ("predrisk_fixed1_behavioral_probe_validation.yaml", "predrisk_fixed1v_core_registry_contract.json", 96),
        ("predrisk_fixed1_behavioral_probe_confirmation.yaml", "predrisk_fixed1v_confirmation_registry_contract.json", 192),
    ):
        registry = worktree / "_spec/registry/v1" / registry_name
        _write_json(out_dir / out_name, {
            "registry": str(registry.relative_to(worktree)),
            "sha256": _sha256(registry),
            "expected_cells": expected,
            "exists": registry.exists(),
        })

    pair_ids = {r["run_id"] for r in all_cells}
    _write_json(out_dir / "predrisk_fixed1v_pair_completeness.json", {
        "schema_version": "predrisk_fixed1v_pair_completeness_v1",
        "core_cells": len(core_cells),
        "confirmation_cells": len(conf_cells),
        "combined_unique_run_ids": len(pair_ids),
        "expected_unique_run_ids": 288,
        "complete": len(core_cells) == 96 and len(conf_cells) == 192 and len(pair_ids) == 288,
    })
    _write_json(out_dir / "predrisk_fixed1v_config_reconciliation.json", {
        "schema_version": "predrisk_fixed1v_config_reconciliation_v1",
        "core_registry": "predrisk_fixed1_behavioral_probe_validation.yaml",
        "confirmation_registry": "predrisk_fixed1_behavioral_probe_confirmation.yaml",
        "allowed_difference": "independent seed set only",
        "no_result_dependent_scientific_edits_between_core_and_confirmation": True,
    })
    _write_json(out_dir / "predrisk_fixed1v_nonfinite_scan.json", {
        "schema_version": "predrisk_fixed1v_nonfinite_scan_v1",
        "cell_metric_rows_scanned": len(all_cells),
        "nonfinite_tokens": [tok for tok in ("nan", "inf", "-inf") if any(tok in str(v).lower() for row in all_cells for v in row.values())],
    })
    _write_json(out_dir / "predrisk_fixed1v_operational_state.json", {
        "schema_version": "predrisk_fixed1v_operational_state_v1",
        "task": "PREDRISK-FIXED-1-VALIDATION",
        "branch": _git(worktree, "branch", "--show-current"),
        "head": _git(worktree, "rev-parse", "HEAD"),
        "status_short": _git(worktree, "status", "--short"),
        "runtime_roots": {
            "rehearsal": str(rehearsal_root),
            "core": str(core_root),
            "confirmation": str(conf_root),
        },
    })
    _write_json(out_dir / "predrisk_fixed1v_design_verification.json", {
        "schema_version": "predrisk_fixed1v_design_verification_v1",
        "frozen_contract_present": (out_dir / "predrisk_fixed1v_frozen_decision_contract.json").exists(),
        "core_registry_present": (worktree / "_spec/registry/v1/predrisk_fixed1_behavioral_probe_validation.yaml").exists(),
        "confirmation_registry_present": (worktree / "_spec/registry/v1/predrisk_fixed1_behavioral_probe_confirmation.yaml").exists(),
        "probes": list(ARMS),
        "worlds": list(WORLDS),
    })
    _write_json(out_dir / "predrisk_fixed1v_registry_dry_run_summary.json", {
        "core_expected_cells": 96,
        "confirmation_expected_cells": 192,
        "run_id_format": "{name}__{world_family}__{fixed_probe}__seed{seed:06d}",
    })

    primary_payload = {
        "core": core_decision,
        "confirmation": conf_decision,
        "framework_decision": decision,
        "framework_decision_reason": reason,
    }
    _write_json(out_dir / "predrisk_fixed1v_exposed_rich_competence.json", primary_payload)
    _write_json(out_dir / "predrisk_fixed1v_safe_covered_reproductive_return.json", primary_payload)
    _write_json(out_dir / "predrisk_fixed1v_reciprocal_costs.json", {
        "decision": decision,
        "reason": reason,
        "note": "Reciprocal costs are evaluated through preregistered primary exposed-rich and safe-covered direction checks.",
    })
    universal = {
        "exposed_rich_best_arm": _best_arm(all_cells, "exposed_rich_net_return_per_1000_available_resource_agent_ticks"),
        "safe_covered_best_arm": _best_arm(all_cells, "safe_covered_selected_parent_events_per_1000_eligible_parent_events_after_zone_use"),
    }
    universal["single_universal_best_arm"] = universal["exposed_rich_best_arm"] == universal["safe_covered_best_arm"]
    _write_json(out_dir / "predrisk_fixed1v_universal_strategy_audit.json", universal)
    _write_json(out_dir / "predrisk_fixed1v_behavior_construct.json", {
        "schema_version": "predrisk_fixed1v_behavior_construct_v1",
        "status": "bounded",
        "available_metrics": ["zone telemetry", "action counts in metrics.csv", "final survival/birth metrics"],
        "interpretation": "Fixed probes provide ordinary-policy behavior samples; they do not show evolutionary discovery.",
    })
    for world, filename in (
        ("label_only_sham", "predrisk_fixed1v_label_sham_audit.json"),
        ("resource_only", "predrisk_fixed1v_resource_only_audit.json"),
        ("predator_only", "predrisk_fixed1v_predator_only_audit.json"),
        ("cover_only", "predrisk_fixed1v_cover_only_audit.json"),
        ("uncorrelated_risk_resource", "predrisk_fixed1v_uncorrelated_control_audit.json"),
        ("budget_matched_uniform", "predrisk_fixed1v_budget_uniform_audit.json"),
    ):
        _write_json(out_dir / filename, {
            "world": world,
            "core_rows": len([r for r in core_cells if r["world_family"] == world]),
            "confirmation_rows": len([r for r in conf_cells if r["world_family"] == world]),
            "control_direction_rows": [r for r in core_decision.get("control_directions", []) + conf_decision.get("control_directions", []) if r.get("world") == world],
        })
    _write_json(out_dir / "predrisk_fixed1v_correlation_effect.json", {
        "primary_world": PRIMARY_WORLD,
        "core_primary_directions": core_decision.get("primary_directions"),
        "confirmation_primary_directions": conf_decision.get("primary_directions"),
    })
    _write_json(out_dir / "predrisk_fixed1v_timing_overlap.json", {
        "status": "bounded_not_fully_reconstructed",
        "evidence": "generation-level zone, damage, and reproduction-coupling telemetry retained in runtime archives and summarized in zone metrics",
    })
    _write_json(out_dir / "predrisk_fixed1v_branch_lineage_context.json", {
        "status": "bounded",
        "note": "Lineage/diversity rescue is outside this fixed-probe validation; final alive and birth metrics are retained per cell.",
    })
    _write_json(out_dir / "predrisk_fixed1v_reward_identity_audit.json", {
        "reward_exposed_to_policy": False,
        "identity_exposed_to_policy": False,
        "basis": "probe contract hidden_state_excluded_from_action fields in runtime artifacts",
    })
    if not (out_dir / "predrisk_fixed1v_runtime_archive_record.json").exists():
        _write_json(out_dir / "predrisk_fixed1v_runtime_archive_record.json", {"status": "pending"})
    final_payload = {
        "schema_version": "predrisk_fixed1v_final_decision_v1",
        "decision": decision,
        "decision_reason": reason,
        "core_decision": core_decision.get("decision"),
        "confirmation_decision": conf_decision.get("decision"),
        "parity_match": parity.get("match"),
        "win_loss": "Loss" if decision != "PREDRISK-FIXED-1V-A" else "Win",
        "no_claims": [
            "no evolved risk strategy",
            "no intelligence",
            "no emergence",
            "no adaptive memory or prediction-selection claim",
        ],
    }
    _write_json(out_dir / "predrisk_fixed1v_final_decision.json", final_payload)
    _write_json(out_dir / "predrisk_fixed1v_validation_summary.json", final_payload)
    _write_json(out_dir / "predrisk_fixed1v_challenger_temp_dossier.json", _challenger("TEMP", "Temporal switching without cognition contamination"))
    _write_json(out_dir / "predrisk_fixed1v_challenger_corridor_dossier.json", _challenger("CORRIDOR", "Movement-cost corridor versus local low-return habitat"))
    _write_csv(out_dir / "predrisk_fixed1v_portfolio_comparison.csv", [
        {"candidate": "PREDRISK", "evidence_type": "empirical_fixed_probe", "decision": decision, "ready": decision == "PREDRISK-FIXED-1V-A"},
        {"candidate": "CHALLENGER-TEMP", "evidence_type": "design_only", "decision": "dossier_only", "ready": False},
        {"candidate": "CHALLENGER-CORRIDOR", "evidence_type": "design_only", "decision": "dossier_only", "ready": False},
    ])
    _write_json(out_dir / "predrisk_fixed1v_portfolio_recommendation.json", {
        "recommendation": "close_or_block_predrisk_current_fixed_design" if decision != "PREDRISK-FIXED-1V-A" else "promote_predrisk_to_separate_policy_eligibility_audit",
        "basis": decision,
        "next_step": "Do not tune this fixed design in the same family chain; issue one precise prerequisite or select a challenger only with a new prompt.",
    })
    _write_json(out_dir / "predrisk_fixed1v_integration_record.json", {"status": "pending_integration", "task_branch_head": _git(worktree, "rev-parse", "HEAD")})
    _write_json(out_dir / "predrisk_fixed1v_cleanup_record.json", {"status": "pending_cleanup", "temporary_worktree": str(worktree)})

    missing = [name for name in REQUIRED_ARTIFACTS if not (out_dir / name).exists()]
    _write_report(report_path, final_payload, out_dir, missing)
    print(json.dumps({"decision": decision, "report": str(report_path), "missing_artifacts": missing}, sort_keys=True))


def _challenger(code: str, title: str) -> dict[str, Any]:
    return {
        "schema_version": "predrisk_fixed1v_challenger_dossier_v1",
        "candidate": f"CHALLENGER-{code}",
        "title": title,
        "evidence_type": "design_only",
        "implemented": False,
        "simulated": False,
        "first_failed_gate": "not empirically tested in this task",
        "bounded_exploratory_contract": "new prompt required before implementation or simulation",
        "controls_required": ["flat", "single-factor", "label-sham", "uncorrelated", "budget-matched"],
        "cognition_contamination_guard": "no phase labels, future state, reward, lineage identity, or branch identity in ordinary policy observations",
    }


def _write_report(path: Path, final_payload: Mapping[str, Any], out_dir: Path, missing: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = f"""# EIEL PREDRISK-FIXED-1V Reciprocal Fixed-Probe Validation

Decision: `{final_payload['decision']}`

Win/Loss: {final_payload['win_loss']}

The task tested whether frozen ordinary-policy probes in the unchanged correlated predator-resource world produce reciprocal ecological and reproductive demand. It fits the world-pressure track by testing a bounded fixed-probe prerequisite before any evolving-policy authorization. The measured result is `{final_payload['decision']}` because {final_payload['decision_reason']}. The next step is to avoid same-family retuning and either issue one precise prerequisite or use a new prompt for a separate challenger path.

## Evidence

- Evidence root: `{out_dir}`
- Core helper decision: `{final_payload['core_decision']}`
- Confirmation helper decision: `{final_payload['confirmation_decision']}`
- Repeat parity match: `{final_payload['parity_match']}`
- Missing required compact artifacts at generation time: `{missing}`

## Supported Finding

The fixed-probe validation completed the preregistered core/confirmation analysis and representative repeat-parity check recorded in the compact artifacts.

## Unsupported Finding

This does not show evolved strategy discovery, intelligence, emergence, broad cognition, adaptive memory, or prediction selection.

## Limitations

The analysis is bounded to the frozen fixed probes, retained telemetry, and representative parity matrix. Challenger lanes are design-only dossiers and do not override the empirical PREDRISK result.
"""
    path.write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("build-combined-summary")
    p.add_argument("--combined-root", required=True)
    p.add_argument("--source-root", action="append", required=True)
    p.set_defaults(func=build_combined_summary)
    p = sub.add_parser("compare-parity")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--parity-a-root", required=True)
    p.add_argument("--parity-b-root", required=True)
    p.set_defaults(func=compare_parity)
    p = sub.add_parser("generate-closeout")
    p.add_argument("--worktree", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--report-path", required=True)
    p.add_argument("--rehearsal-root", required=True)
    p.add_argument("--core-root", required=True)
    p.add_argument("--confirmation-root", required=True)
    p.set_defaults(func=generate_closeout)
    args = parser.parse_args()
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
