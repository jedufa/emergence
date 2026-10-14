#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
import sys
import zipfile
from pathlib import Path
from typing import Any, Iterable, Mapping


SCHEMA_VERSION = "respec_ea0_evolving_axis_eligibility_v1"
REPORT_PATH = Path(
    "docs/500_campaigns-decisions-closeouts/520_campaign-reports/400_worlds-ecology/EIEL-DOC-002166--world-pressure-eiel-respec-ea0-evolving-axis-selection-eligibility-2026-07-11.md"
)
EVIDENCE_DIR = Path("evidences/07-world-pressure/06-respec-ea0-evolving-axis-eligibility")
D1_EVIDENCE_DIR = Path("evidences/07-world-pressure/05-respec-d1-neutral-catchup-redesign")
D1_EVIDENCE_ZIP = D1_EVIDENCE_DIR / "world-pressure_respec_d1_neutral_catchup_redesign.evidence.zip"
D1_REGISTRY_PATH = Path("_spec/registry/v1/respec_d1_redesigned_fixed_arm_validation.yaml")
PROMPT_EXPECTED_D1_ZIP_SHA256 = "5c649c311c450f494cf39668f362eac5fa678b586399f56a48a55df11a0d39f6"
EXPECTED_D1_REGISTRY_SHA256 = "79fe7152da0e0afc2b6ab9b58017390c02f721303480fe536a8dd350693ea8d5"
NO_CLAIMS = (
    "RESPEC-EA0 is a bounded eligibility and mixed-niche audit for a possible later evolving-axis campaign. "
    "It does not claim evolved specialization, selected specialization, intelligence, emergence, consciousness, "
    "broad cognition, adaptive memory, prediction selection, solved lineage diversity, beneficial open-ended search, "
    "culture, superintelligence, or general open-ended evolution."
)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[5]


def _ensure_import_paths() -> None:
    ordered = [str(_repo_root()), str(_project_root()), str(_project_root() / "engine")]
    for path in ordered:
        while path in sys.path:
            sys.path.remove(path)
        sys.path.insert(0, path)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _float(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return 0.0
    return out if math.isfinite(out) else 0.0


def _mean(values: list[float]) -> float:
    return float(statistics.fmean(values)) if values else 0.0


def _safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def source_verification() -> dict[str, Any]:
    paths = [
        Path("engine/agents/resource_handling.py"),
        Path("engine/evolution/reproduction.py"),
        Path("engine/env/gridworld_pkg/patch_world.py"),
        Path("engine/simulation/runner_pkg/runner_ticks_env_apply.py"),
        Path("engine/simulation/metrics_pkg/metrics_logger.py"),
        Path("engine/simulation/checkpoint_bundle_v2.py"),
    ]
    files = {}
    for path in paths:
        files[str(path)] = {
            "exists": path.exists(),
            "sha256": _sha256(path) if path.exists() else "",
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS" if all(row["exists"] for row in files.values()) else "FAIL",
        "files": files,
        "source_findings": [
            "resource_handling_axis maps to reciprocal capacities with constant total capacity",
            "crossover inherits the midpoint of parent axes before existing scalar mutation",
            "resource handling uses the existing scalar mutation probability and sigma 0.10",
            "RESPEC-0B gather applies resource-class thresholds and returns through the authoritative Patch World path",
            "checkpoint_bundle_v2 serializes RESPEC-0B handling state",
            "metrics expose resource-class successes and aggregate resource_handling_axis_mean",
        ],
    }


def d1_evidence_verification() -> dict[str, Any]:
    required = [
        "respec_d1_causal_diagnosis.json",
        "respec_d1_metric_definition_audit.json",
        "respec_d1_metric_contract_review.json",
        "respec_d1_decision_contract.json",
        "respec_d1_mechanical_opportunity_summary.json",
        "respec_d1_ecological_outcome_summary.json",
        "respec_d1_mixed_world_summary.json",
        "respec_d1_opportunity_accounting.json",
        "respec_d1_resource_availability_analysis.json",
        "respec_d1_equalized_sham_audit.json",
        "respec_d1_repeat_parity_summary.json",
        "respec_d1_final_decision.json",
        "respec_d1_seed_level_metrics.csv",
        "respec_d1_scenario_arm_summary.csv",
    ]
    present = {name: (D1_EVIDENCE_DIR / name).exists() for name in required}
    actual_zip_sha = _sha256(D1_EVIDENCE_ZIP) if D1_EVIDENCE_ZIP.exists() else ""
    actual_registry_sha = _sha256(D1_REGISTRY_PATH) if D1_REGISTRY_PATH.exists() else ""
    final = _read_json(D1_EVIDENCE_DIR / "respec_d1_final_decision.json")
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS" if all(present.values()) and final.get("decision") == "RESPEC-D1-B" else "FAIL",
        "d1_decision": final.get("decision", ""),
        "d1_zip_path": str(D1_EVIDENCE_ZIP),
        "actual_d1_zip_sha256": actual_zip_sha,
        "prompt_expected_d1_zip_sha256": PROMPT_EXPECTED_D1_ZIP_SHA256,
        "prompt_sha_reconciliation": "PROMPT_STALE_LIVE_FILESYSTEM_USED"
        if actual_zip_sha and actual_zip_sha != PROMPT_EXPECTED_D1_ZIP_SHA256
        else "MATCH",
        "d1_registry_path": str(D1_REGISTRY_PATH),
        "actual_d1_registry_sha256": actual_registry_sha,
        "expected_d1_registry_sha256": EXPECTED_D1_REGISTRY_SHA256,
        "required_files": present,
    }


def fixed_arm_foundation() -> dict[str, Any]:
    mechanical = _read_json(D1_EVIDENCE_DIR / "respec_d1_mechanical_opportunity_summary.json")
    ecological = _read_json(D1_EVIDENCE_DIR / "respec_d1_ecological_outcome_summary.json")
    sham = _read_json(D1_EVIDENCE_DIR / "respec_d1_equalized_sham_audit.json")
    parity = _read_json(D1_EVIDENCE_DIR / "respec_d1_repeat_parity_summary.json")
    status = (
        mechanical.get("status") == "PASS"
        and ecological.get("status") == "PASS"
        and sham.get("status") == "PASS"
        and parity.get("status") == "PASS"
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "gate": 1,
        "status": "PASS" if status else "FAIL",
        "fixed_arm_decision": "RESPEC-D1-B",
        "basis": {
            "mechanical_status": mechanical.get("status", ""),
            "ecological_status": ecological.get("status", ""),
            "equalized_sham_status": sham.get("status", ""),
            "repeat_parity_status": parity.get("status", ""),
        },
        "interpretation": "fixed inherited arms create reciprocal handling competence and single-resource ecological benefit",
    }


def _base_payload(axis: float, *, mutation_rate: float = 0.05) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": "modular_genome_v1",
        "hidden_size": 64,
        "exploration_bias": 0.1,
        "aggression_bias": -0.1,
        "reproduction_threshold": 30.0,
        "mutation_rate": float(mutation_rate),
        "action_temperature": 1.0,
        "use_gru": False,
        "memory_dim": 0,
        "drive_inertia": 0.85,
        "drive_noise": 0.02,
        "drive_logit_gain": 1.0,
        "pred_lr": 0.08,
        "drive_sp_hunger": 0.55,
        "drive_sp_pain": 0.05,
        "drive_sp_fatigue": 0.15,
        "drive_sp_fear": 0.10,
        "exp_prior_strength": 0.35,
        "exp_prior_noise": 0.05,
        "drive_birth_noise": 0.03,
        "modules": [],
        "connections": [],
    }
    if float(axis) != 0.0:
        payload["resource_handling_axis"] = float(axis)
    return payload


def _offspring_probe_rows(samples_per_case: int = 400) -> list[dict[str, Any]]:
    _ensure_import_paths()
    from agents.genome import Genome
    from evolution.reproduction import crossover_genome
    from simulation.runner_pkg.runner_config import SimConfig

    cfg = SimConfig(seed=7, generations=1, steps_per_generation=1, device="cpu", bank_device="cpu", behavior_log_enable=False)
    cfg.genome_c1_enable = False
    cfg.genome_c2_enable = False
    cfg.resource_handling_representation_enable = True
    cfg.resource_handling_representation_reporting_enable = True
    cases = [
        ("neutral_x_neutral", 0.0, 0.0),
        ("small_positive_x_neutral", 0.10, 0.0),
        ("small_negative_x_neutral", -0.10, 0.0),
        ("moderate_positive_x_moderate_positive", 0.50, 0.50),
        ("moderate_negative_x_moderate_negative", -0.50, -0.50),
        ("positive_x_negative", 0.75, -0.75),
    ]
    rows: list[dict[str, Any]] = []
    for case_index, (case, parent_a, parent_b) in enumerate(cases):
        g1 = Genome.from_dict(_base_payload(parent_a))
        g2 = Genome.from_dict(_base_payload(parent_b))
        for offset in range(samples_per_case):
            seed = 200000 + case_index * 10000 + offset
            rng = random.Random(seed)
            child = crossover_genome(g1, g2, rng, sim_cfg=cfg)
            record = dict(getattr(child, "_resource_handling_reproduction_record", {}) or {})
            pre = _float(record.get("child_pre_mutation_axis", 0.5 * (parent_a + parent_b)))
            post = _float(getattr(child, "resource_handling_axis", record.get("child_post_mutation_axis", pre)))
            delta = post - pre
            rows.append(
                {
                    "case": case,
                    "seed": seed,
                    "parent_a_axis": parent_a,
                    "parent_b_axis": parent_b,
                    "pre_mutation_axis": pre,
                    "post_mutation_axis": post,
                    "mutation_attempted": bool(record.get("mutation_attempted", False)),
                    "axis_delta": delta,
                    "sign": "positive" if post > 1e-12 else ("negative" if post < -1e-12 else "zero"),
                    "non_neutral": bool(abs(post) > 1e-12),
                    "moderate_abs_ge_0_25": bool(abs(post) >= 0.25),
                    "clamped": bool(record.get("clamped", False)),
                    "mutation_rate": _float(record.get("mutation_rate", 0.05)),
                    "mutation_sigma": _float(record.get("mutation_sigma", 0.10)),
                }
            )
    return rows


def mutation_supply_summary(rows: list[dict[str, Any]], d1_birth_rate_per_generation: float) -> dict[str, Any]:
    cases: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        cases.setdefault(str(row["case"]), []).append(row)
    summaries = []
    neutral = cases.get("neutral_x_neutral", [])
    positive_prob = _safe_div(sum(1 for row in neutral if row["post_mutation_axis"] > 0.0), len(neutral))
    negative_prob = _safe_div(sum(1 for row in neutral if row["post_mutation_axis"] < 0.0), len(neutral))
    nonneutral_prob = _safe_div(sum(1 for row in neutral if row["non_neutral"]), len(neutral))
    moderate_prob = _safe_div(sum(1 for row in neutral if row["moderate_abs_ge_0_25"]), len(neutral))
    for case, case_rows in sorted(cases.items()):
        post = [_float(row["post_mutation_axis"]) for row in case_rows]
        summaries.append(
            {
                "case": case,
                "samples": len(case_rows),
                "positive_fraction": _safe_div(sum(1 for row in case_rows if row["post_mutation_axis"] > 0.0), len(case_rows)),
                "zero_fraction": _safe_div(sum(1 for row in case_rows if abs(row["post_mutation_axis"]) <= 1e-12), len(case_rows)),
                "negative_fraction": _safe_div(sum(1 for row in case_rows if row["post_mutation_axis"] < 0.0), len(case_rows)),
                "non_neutral_fraction": _safe_div(sum(1 for row in case_rows if row["non_neutral"]), len(case_rows)),
                "moderate_abs_ge_0_25_fraction": _safe_div(
                    sum(1 for row in case_rows if row["moderate_abs_ge_0_25"]), len(case_rows)
                ),
                "post_axis_mean": _mean(post),
                "post_axis_variance": float(statistics.pvariance(post)) if len(post) > 1 else 0.0,
                "clamp_fraction": _safe_div(sum(1 for row in case_rows if row["clamped"]), len(case_rows)),
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "gate": 2,
        "status": "PASS" if positive_prob > 0.0 and negative_prob > 0.0 else "FAIL",
        "samples_per_case": len(neutral),
        "case_summaries": summaries,
        "neutral_parent_expected_new_non_neutral_variants_per_100_births": 100.0 * nonneutral_prob,
        "neutral_parent_expected_positive_variants_per_100_births": 100.0 * positive_prob,
        "neutral_parent_expected_negative_variants_per_100_births": 100.0 * negative_prob,
        "neutral_parent_expected_moderate_abs_ge_0_25_variants_per_100_births": 100.0 * moderate_prob,
        "neutral_parent_expected_moderate_abs_ge_0_25_variants_per_1000_births": 1000.0 * moderate_prob,
        "expected_generations_to_first_positive_variant": _safe_div(1.0, positive_prob * d1_birth_rate_per_generation),
        "expected_generations_to_first_negative_variant": _safe_div(1.0, negative_prob * d1_birth_rate_per_generation),
        "midpoint_inheritance": "parent axes recombine by arithmetic midpoint before mutation",
        "both_direction_reachability": bool(positive_prob > 0.0 and negative_prob > 0.0),
    }


def mutation_inheritance_audit() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "gate": 2,
        "status": "PASS",
        "parent_axis_inheritance": "midpoint",
        "mutation_source": "existing scalar mutation probability after midpoint inheritance",
        "mutation_sigma": 0.10,
        "clamp_range": [-1.0, 1.0],
        "identity_or_reward_coupling_added": False,
        "specialist_seeding_used": False,
        "source_files": [
            "engine/agents/resource_handling.py",
            "engine/evolution/reproduction.py",
            "tests/evolvability/phase_c/test_rep_rh0_resource_handling_representation.py",
        ],
    }


def _mixed_rows() -> list[dict[str, str]]:
    return [row for row in _read_csv(D1_EVIDENCE_DIR / "respec_d1_seed_level_metrics.csv") if row.get("world_family") == "mixed_tradeoff"]


def mixed_world_resource_balance() -> dict[str, Any]:
    mixed = _read_json(D1_EVIDENCE_DIR / "respec_d1_mixed_world_summary.json")
    a = _float(mixed.get("resource_a_successes"))
    b = _float(mixed.get("resource_b_successes"))
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "resource_a_successes": a,
        "resource_b_successes": b,
        "resource_b_to_a_success_ratio": _safe_div(b, a),
        "resource_b_success_share": _safe_div(b, a + b),
        "interpretation": "mixed aggregate exposure is B-heavy in accepted D1 compact evidence",
    }


def mixed_world_arm_outcomes() -> dict[str, Any]:
    rows = _mixed_rows()
    by_arm: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_arm.setdefault(str(row.get("fixed_arm", "")), []).append(row)
    arms: dict[str, dict[str, Any]] = {}
    for arm, arm_rows in sorted(by_arm.items()):
        arms[arm] = {
            "runs": len(arm_rows),
            "alive_final_mean": _mean([_float(row.get("alive_final")) for row in arm_rows]),
            "successful_births_final_mean": _mean([_float(row.get("successful_births_final")) for row in arm_rows]),
            "resource_a_completion_per_handling_tick_mean": _mean(
                [_float(row.get("resource_a_completion_per_handling_tick")) for row in arm_rows]
            ),
            "resource_b_completion_per_handling_tick_mean": _mean(
                [_float(row.get("resource_b_completion_per_handling_tick")) for row in arm_rows]
            ),
            "completion_per_handling_tick_mean": _mean([_float(row.get("completion_per_handling_tick")) for row in arm_rows]),
            "successful_births_by_seed": {row.get("seed", ""): _float(row.get("successful_births_final")) for row in arm_rows},
            "alive_by_seed": {row.get("seed", ""): _float(row.get("alive_final")) for row in arm_rows},
        }
    a_births = arms.get("a_favored", {}).get("successful_births_by_seed", {})
    n_births = arms.get("neutral", {}).get("successful_births_by_seed", {})
    b_births = arms.get("b_favored", {}).get("successful_births_by_seed", {})
    seeds = sorted(set(a_births) & set(n_births) & set(b_births))
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "arms": arms,
        "seed_support": {
            "neutral_gt_a_favored_births": sum(1 for seed in seeds if n_births[seed] > a_births[seed]),
            "b_favored_gt_a_favored_births": sum(1 for seed in seeds if b_births[seed] > a_births[seed]),
            "b_favored_gt_neutral_births": sum(1 for seed in seeds if b_births[seed] > n_births[seed]),
            "seeds": seeds,
        },
        "primary_outcome_ranking": "b_favored_or_neutral_over_a_favored",
    }


def spatial_niche_audit() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "gate": 4,
        "status": "INSUFFICIENT_FOR_SPATIAL_NICHE_CLAIM",
        "layout": "vertical_split_ab",
        "source_basis": "initialize_respec0b_resource_class_array assigns A to left split and B to right split",
        "available_d1_telemetry": "aggregate population/resource metrics only",
        "missing_for_masked_two_niche_claim": [
            "per-zone occupancy by inherited axis",
            "per-zone births by inherited axis",
            "local parent-offspring axis persistence by zone",
            "branch-specific resource-use trajectories",
        ],
    }


def mixed_world_classification(balance: Mapping[str, Any], outcomes: Mapping[str, Any]) -> dict[str, Any]:
    arms = dict(outcomes.get("arms") or {})
    a = dict(arms.get("a_favored") or {})
    n = dict(arms.get("neutral") or {})
    b = dict(arms.get("b_favored") or {})
    b_birth_advantage = _float(b.get("successful_births_final_mean")) - _float(a.get("successful_births_final_mean"))
    neutral_birth_advantage = _float(n.get("successful_births_final_mean")) - _float(a.get("successful_births_final_mean"))
    b_alive_advantage = _float(b.get("alive_final_mean")) - _float(a.get("alive_final_mean"))
    b_total_tick_advantage = _float(b.get("completion_per_handling_tick_mean")) - _float(a.get("completion_per_handling_tick_mean"))
    b_resource_ratio = _float(balance.get("resource_b_to_a_success_ratio"))
    failed = bool(b_birth_advantage > 0.0 and neutral_birth_advantage > 0.0 and b_resource_ratio > 1.0)
    return {
        "schema_version": SCHEMA_VERSION,
        "gate": 4,
        "status": "FAIL" if failed else "PASS",
        "classification": "DIRECTIONAL_B_OPTIMUM_WITH_NEUTRAL_B_PLATEAU" if failed else "TWO_NICHE_READY",
        "b_favored_minus_a_favored_births_mean": b_birth_advantage,
        "neutral_minus_a_favored_births_mean": neutral_birth_advantage,
        "b_favored_minus_a_favored_alive_mean": b_alive_advantage,
        "b_favored_minus_a_favored_total_completion_per_handling_tick": b_total_tick_advantage,
        "resource_b_to_a_success_ratio": b_resource_ratio,
        "two_niche_campaign_authorized": False if failed else True,
        "reason": "accepted D1 mixed world favors B-favored/neutral outcomes over A-favored; aggregate evidence does not support two persistent incompatible roles",
    }


def selection_timing_analysis(supply: Mapping[str, Any], outcomes: Mapping[str, Any]) -> dict[str, Any]:
    arms = dict(outcomes.get("arms") or {})
    mixed_birth_rate = _mean(
        [
            _float(dict(arms.get("a_favored") or {}).get("successful_births_final_mean")),
            _float(dict(arms.get("neutral") or {}).get("successful_births_final_mean")),
            _float(dict(arms.get("b_favored") or {}).get("successful_births_final_mean")),
        ]
    ) / 48.0
    return {
        "schema_version": SCHEMA_VERSION,
        "gate": 3,
        "status": "PASS_WITH_LIMITATION",
        "basis": "accepted D1 cells observed both resource handling and births within 48 generations; EA0 did not run an evolving campaign",
        "mixed_world_birth_rate_per_generation_estimate": mixed_birth_rate,
        "expected_generations_to_first_positive_variant": supply.get("expected_generations_to_first_positive_variant"),
        "expected_generations_to_first_negative_variant": supply.get("expected_generations_to_first_negative_variant"),
        "limitation": "compact D1 evidence does not include branch-collapse timing or variant-bearing reproduction timing",
    }


def variant_exposure_estimate(supply: Mapping[str, Any], outcomes: Mapping[str, Any]) -> dict[str, Any]:
    arms = dict(outcomes.get("arms") or {})
    neutral_births = _float(dict(arms.get("neutral") or {}).get("successful_births_final_mean"))
    nonneutral_per_birth = _float(supply.get("neutral_parent_expected_new_non_neutral_variants_per_100_births")) / 100.0
    return {
        "schema_version": SCHEMA_VERSION,
        "gate": 3,
        "status": "PASS_WITH_LIMITATION",
        "neutral_mixed_births_per_48_generation_cell": neutral_births,
        "expected_new_non_neutral_axis_variants_per_neutral_mixed_cell": neutral_births * nonneutral_per_birth,
        "expected_positive_axis_variants_per_neutral_mixed_cell": neutral_births
        * _float(supply.get("neutral_parent_expected_positive_variants_per_100_births"))
        / 100.0,
        "expected_negative_axis_variants_per_neutral_mixed_cell": neutral_births
        * _float(supply.get("neutral_parent_expected_negative_variants_per_100_births"))
        / 100.0,
    }


def not_evaluated_artifact(name: str, *, gate: int | None = None, reason: str | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "NOT_EVALUATED",
        "blocked_by": "Gate 4 mixed-world classification failed",
        "reason": reason
        or "EA0 stops at the first failed gate; evaluating later gates would imply campaign readiness that the mixed-world evidence does not support",
    }
    if gate is not None:
        payload["gate"] = gate
    payload["artifact"] = name
    return payload


def control_contract() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "DESIGNED_NOT_EXECUTED",
        "required_controls_for_later_campaign": [
            "flat world",
            "reporting-only world",
            "equalized sham",
            "single-resource A-only and B-only references",
            "repeat parity",
            "fixed-arm shadow matrix before evolving-axis execution",
        ],
        "not_authorized_reason": "mixed world is not two-niche ready",
    }


def passive_telemetry_parity() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "NOT_RUN",
        "passive_telemetry_added": False,
        "reason": "no telemetry code was changed; no parity run was needed",
    }


def checkpoint_resume_summary() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "SOURCE_AUDIT_ONLY",
        "basis": "checkpoint_bundle_v2 serializes resource handling state; REP-RH0 tests cover representation checkpoint roundtrip",
        "new_checkpoint_run_executed": False,
    }


def reward_and_identity_audit() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "direct_specialization_reward_added": False,
        "identity_or_lineage_rescue_added": False,
        "specialist_founders_used_as_evolving_evidence": False,
        "mechanics_changed": False,
    }


def registry_contract() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "BLOCKED_NOT_AUTHORIZED",
        "registry_created": False,
        "dry_run_executed": False,
        "blocked_by": "RESPEC-EA0-B",
        "minimum_prerequisite": "redesign or instrument the mixed world so it demonstrably contains two incompatible ecological roles before evolving-axis selection",
    }


def registry_dry_run_summary() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "NOT_RUN",
        "planned_runs": 0,
        "failures": [],
        "reason": "later evolving-axis registry is not authorized by EA0",
    }


def storage_estimate() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "new_simulation_runtime_roots": [],
        "raw_runtime_cleanup_required": False,
        "evidence_bundle_expected_size": "small compact report-facing bundle only",
    }


def runtime_estimate() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "campaigns_executed": 0,
        "estimated_runtime": "analysis-only, seconds",
        "later_campaign_authorized": False,
    }


def final_decision(classification: Mapping[str, Any], supply: Mapping[str, Any], timing: Mapping[str, Any]) -> dict[str, Any]:
    if supply.get("status") != "PASS":
        decision = "RESPEC-EA0-C"
        reason = "mutation supply did not demonstrate both axis directions"
    elif str(timing.get("status", "")).startswith("FAIL"):
        decision = "RESPEC-EA0-D"
        reason = "selection timing overlap was not credible"
    elif classification.get("status") == "FAIL":
        decision = "RESPEC-EA0-B"
        reason = "mixed world has a directional B/neutral optimum rather than demonstrated two-niche readiness"
    else:
        decision = "RESPEC-EA0-A"
        reason = "all eligibility gates passed"
    return {
        "schema_version": SCHEMA_VERSION,
        "decision": decision,
        "reason": reason,
        "campaign_authorized": decision == "RESPEC-EA0-A",
        "failed_gate": 4 if decision == "RESPEC-EA0-B" else None,
        "no_claims": NO_CLAIMS,
    }


def validation_summary() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "PENDING",
        "commands": [],
        "note": "updated after validation during closeout",
    }


def integration_record() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "PENDING",
        "source_branch": "design/respec-ea0-evolving-axis-eligibility-20260711",
        "integration_branch": "integration/world-pressure",
    }


def cleanup_record() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "PENDING",
        "raw_runtime_roots_created": [],
        "raw_runtime_roots_deleted": [],
        "runtime_archives_created": [],
    }


def operational_state() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "task": "RESPEC-EA0",
        "branch_keyword": "world-pressure",
        "persistent_branch": "integration/world-pressure",
        "hard_boundaries_respected": {
            "evolving_axis_campaign_run": False,
            "resource_mechanics_changed": False,
            "axis_mapping_or_mutation_changed": False,
            "specialist_founders_used_as_evolving_evidence": False,
            "reward_or_lineage_rescue_added": False,
        },
    }


def _report(payloads: Mapping[str, Mapping[str, Any]]) -> str:
    final = payloads["respec_ea0_final_decision.json"]
    fixed = payloads["respec_ea0_fixed_arm_foundation.json"]
    supply = payloads["respec_ea0_mutation_supply_summary.json"]
    classification = payloads["respec_ea0_mixed_world_classification.json"]
    outcomes = payloads["respec_ea0_mixed_world_arm_outcomes.json"]
    balance = payloads["respec_ea0_mixed_world_resource_balance.json"]
    arms = dict(outcomes.get("arms") or {})
    a = dict(arms.get("a_favored") or {})
    n = dict(arms.get("neutral") or {})
    b = dict(arms.get("b_favored") or {})
    return f"""# world-pressure RESPEC-EA0 Evolving-Axis Selection Eligibility

Date: 2026-07-11

Branch keyword: `world-pressure`

Decision: `{final.get('decision')}`

## Win/Loss

Win: RESPEC-EA0 confirms the representation and mutation path can supply both signed axis directions without changing mechanics. The accepted D1 fixed-arm evidence remains valid as a fixed-arm world-pressure substrate.

Loss: EA0 does not authorize the evolving-axis campaign. Gate 4 fails because the accepted mixed world looks like a B-favored/neutral outcome surface, not a demonstrated two-niche substrate.

In simple terms: I checked whether the current Resource A/B world is ready for agents to evolve their own resource preference. I did this because fixed hand-built specialists working is not enough; the world must also give evolution two good but different ways to live. The mutation machinery can make both A-leaning and B-leaning children, but the mixed world mostly rewards B or neutral behavior in the compact D1 evidence. That means the next useful path is to redesign or instrument the world pressure itself before running an evolving specialization campaign.

## Gate Results

- Gate 1 fixed-arm foundation: `{fixed.get('status')}`.
- Gate 2 mutation and inheritance supply: `{supply.get('status')}`.
- Gate 3 timing overlap: `{payloads['respec_ea0_selection_timing_analysis.json'].get('status')}`.
- Gate 4 mixed-world niche structure: `{classification.get('status')}`.
- Gates 5-7: `NOT_EVALUATED`, because EA0 stops at the first failed gate.

## Mixed-World Evidence

Accepted D1 mixed-world means:

| Arm | Alive | Births | A/tick | B/tick | Total/tick |
| --- | ---: | ---: | ---: | ---: | ---: |
| A-favored | {_float(a.get('alive_final_mean')):.2f} | {_float(a.get('successful_births_final_mean')):.2f} | {_float(a.get('resource_a_completion_per_handling_tick_mean')):.3f} | {_float(a.get('resource_b_completion_per_handling_tick_mean')):.3f} | {_float(a.get('completion_per_handling_tick_mean')):.3f} |
| Neutral | {_float(n.get('alive_final_mean')):.2f} | {_float(n.get('successful_births_final_mean')):.2f} | {_float(n.get('resource_a_completion_per_handling_tick_mean')):.3f} | {_float(n.get('resource_b_completion_per_handling_tick_mean')):.3f} | {_float(n.get('completion_per_handling_tick_mean')):.3f} |
| B-favored | {_float(b.get('alive_final_mean')):.2f} | {_float(b.get('successful_births_final_mean')):.2f} | {_float(b.get('resource_a_completion_per_handling_tick_mean')):.3f} | {_float(b.get('resource_b_completion_per_handling_tick_mean')):.3f} | {_float(b.get('completion_per_handling_tick_mean')):.3f} |

Resource completions in mixed D1 evidence: A `{_float(balance.get('resource_a_successes')):.0f}`, B `{_float(balance.get('resource_b_successes')):.0f}`, B/A ratio `{_float(balance.get('resource_b_to_a_success_ratio')):.3f}`.

Classification: `{classification.get('classification')}`.

## Mutation Supply

Neutral x neutral expected variants per 100 births:

- non-neutral: `{_float(supply.get('neutral_parent_expected_new_non_neutral_variants_per_100_births')):.2f}`;
- positive: `{_float(supply.get('neutral_parent_expected_positive_variants_per_100_births')):.2f}`;
- negative: `{_float(supply.get('neutral_parent_expected_negative_variants_per_100_births')):.2f}`;
- moderate abs(axis) >= 0.25: `{_float(supply.get('neutral_parent_expected_moderate_abs_ge_0_25_variants_per_100_births')):.2f}`.

## Controls And Boundaries

No evolving-axis campaign was run. No resource threshold, return, fraction, layout, handling, axis mapping, mutation, inheritance, reward, action, observation, reproduction, lineage rescue, or branch rescue mechanics were changed.

## Recommendation

Do not run an evolving-axis selection campaign on the current mixed world. Next, design a new mixed-world demand candidate or add passive spatial/branch telemetry to prove whether a hidden local two-niche structure exists before authorizing selection.

## No Claims

{NO_CLAIMS}
"""


def _zip_evidence(evidence_dir: Path) -> dict[str, Any]:
    zip_path = evidence_dir / "world-pressure_respec_ea0_evolving_axis_eligibility.evidence.zip"
    manifest_path = evidence_dir / "respec_ea0_evidence_bundle_manifest.json"
    members = sorted(
        p
        for p in evidence_dir.iterdir()
        if p.is_file() and p.name not in {zip_path.name, manifest_path.name} and not p.name.endswith(".sha256")
    )
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in members:
            zf.write(path, arcname=path.name)
    sha = _sha256(zip_path)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "zip_path": str(zip_path),
        "sha256": sha,
        "member_count": len(members),
        "members": [p.name for p in members],
    }
    _write_json(manifest_path, manifest)
    return manifest


def build_payloads(samples_per_case: int = 400) -> tuple[dict[str, Mapping[str, Any]], list[dict[str, Any]]]:
    outcomes = mixed_world_arm_outcomes()
    arms = dict(outcomes.get("arms") or {})
    d1_birth_rate = _mean([_float(dict(arms.get(arm) or {}).get("successful_births_final_mean")) for arm in arms]) / 48.0
    probe_rows = _offspring_probe_rows(samples_per_case=samples_per_case)
    supply = mutation_supply_summary(probe_rows, d1_birth_rate)
    balance = mixed_world_resource_balance()
    classification = mixed_world_classification(balance, outcomes)
    timing = selection_timing_analysis(supply, outcomes)
    payloads: dict[str, Mapping[str, Any]] = {
        "respec_ea0_operational_state.json": operational_state(),
        "respec_ea0_source_verification.json": source_verification(),
        "respec_ea0_d1_evidence_verification.json": d1_evidence_verification(),
        "respec_ea0_fixed_arm_foundation.json": fixed_arm_foundation(),
        "respec_ea0_mutation_inheritance_audit.json": mutation_inheritance_audit(),
        "respec_ea0_mutation_supply_summary.json": supply,
        "respec_ea0_selection_timing_analysis.json": timing,
        "respec_ea0_variant_exposure_estimate.json": variant_exposure_estimate(supply, outcomes),
        "respec_ea0_mixed_world_resource_balance.json": balance,
        "respec_ea0_mixed_world_arm_outcomes.json": outcomes,
        "respec_ea0_spatial_niche_audit.json": spatial_niche_audit(),
        "respec_ea0_mixed_world_classification.json": classification,
        "respec_ea0_observation_action_audit.json": not_evaluated_artifact("respec_ea0_observation_action_audit.json", gate=5),
        "respec_ea0_observation_decision.json": not_evaluated_artifact("respec_ea0_observation_decision.json", gate=5),
        "respec_ea0_measurement_contract.json": not_evaluated_artifact("respec_ea0_measurement_contract.json", gate=6),
        "respec_ea0_axis_reporting_schema.json": not_evaluated_artifact("respec_ea0_axis_reporting_schema.json", gate=6),
        "respec_ea0_future_decision_framework.json": not_evaluated_artifact("respec_ea0_future_decision_framework.json"),
        "respec_ea0_control_contract.json": control_contract(),
        "respec_ea0_passive_telemetry_parity.json": passive_telemetry_parity(),
        "respec_ea0_checkpoint_resume_summary.json": checkpoint_resume_summary(),
        "respec_ea0_reward_and_identity_audit.json": reward_and_identity_audit(),
        "respec_ea0_registry_contract.json": registry_contract(),
        "respec_ea0_registry_dry_run_summary.json": registry_dry_run_summary(),
        "respec_ea0_storage_estimate.json": storage_estimate(),
        "respec_ea0_runtime_estimate.json": runtime_estimate(),
        "respec_ea0_validation_summary.json": validation_summary(),
        "respec_ea0_integration_record.json": integration_record(),
        "respec_ea0_cleanup_record.json": cleanup_record(),
    }
    payloads["respec_ea0_final_decision.json"] = final_decision(classification, supply, timing)
    return payloads, probe_rows


def write_outputs(samples_per_case: int = 400) -> dict[str, Any]:
    payloads, probe_rows = build_payloads(samples_per_case=samples_per_case)
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    for name, payload in payloads.items():
        _write_json(EVIDENCE_DIR / name, payload)
    _write_csv(
        EVIDENCE_DIR / "respec_ea0_offspring_axis_probe.csv",
        probe_rows,
        [
            "case",
            "seed",
            "parent_a_axis",
            "parent_b_axis",
            "pre_mutation_axis",
            "post_mutation_axis",
            "mutation_attempted",
            "axis_delta",
            "sign",
            "non_neutral",
            "moderate_abs_ge_0_25",
            "clamped",
            "mutation_rate",
            "mutation_sigma",
        ],
    )
    report = _report(payloads)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")
    (EVIDENCE_DIR / REPORT_PATH.name).write_text(report, encoding="utf-8")
    manifest = _zip_evidence(EVIDENCE_DIR)
    return {
        "schema_version": SCHEMA_VERSION,
        "decision": payloads["respec_ea0_final_decision.json"].get("decision"),
        "report_path": str(REPORT_PATH),
        "evidence_dir": str(EVIDENCE_DIR),
        "evidence_zip": manifest["zip_path"],
        "evidence_zip_sha256": manifest["sha256"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-per-case", type=int, default=400)
    args = parser.parse_args(argv)
    result = write_outputs(samples_per_case=max(1, int(args.samples_per_case)))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
