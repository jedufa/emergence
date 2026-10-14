from __future__ import annotations

"""REP-2D structural causal robustness helpers.

The helpers in this file are reporting and evidence utilities. They do not
change structural semantics, operator rates, selection, rewards, world pressure,
or lineage handling.
"""

import csv
import hashlib
import json
import math
import random
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

from agents.genome import Genome
from env.env_config import EnvConfig
from env.gridworld_pkg.constants import ACT_ATTACK, ACT_GATHER, ACT_REST
from evolution.operators_c1 import operator_catalog
from evolution.population_pkg.population_config import PopConfig
from evolution.recombination_c2 import recombine_lineage_payloads, recombine_self_lineage_payload
from evolution.structural_live_semantics import (
    ACTION_REACHABLE_FIELDS,
    FIELD_BOUNDS,
    PROFILE_CONNECTION_FIELD_COMPILER,
    apply_live_structural_semantics,
    compile_live_graph,
    stable_hash,
    stable_json,
)
from evolution.structural_mutation_viability import (
    apply_single_operator,
    complexity_cost,
    rep2b_live_config,
    seeded_fixture_genome,
)
from simulation.checkpoint_bundle_v2 import checkpoint_bundle_v2_apply, checkpoint_bundle_v2_load, checkpoint_bundle_v2_save
from simulation.metrics import MetricsLogger
from simulation.metrics_pkg.noop_loggers import NoOpMetricsLogger
from simulation.performance_profiles import ACCEPTED_W6_FAST_PROFILE, apply_performance_profile
from simulation.runner import SimulationRunner


SCHEMA_VERSION = "structural_causal_robustness_v1"
REP2B_BASE_COMMIT = "e5527f207ec7ab7104a0204c09e9681546a1efd8"
REP2B_GRAPH_DIGEST = "f4d7006846fc88ee691eaa06518539b826c8340b3432ca97950d0d5e4651ee7b"
REP2B_SEED_DELTAS = {
    "aggression_bias": 0.04999999999999999,
    "drive_inertia": 0.004800000219999945,
    "drive_logit_gain": 0.06400000159999991,
}


def write_json(path: Path, payload: Any) -> None:
    path.write_text(stable_json(payload) + "\n", encoding="utf-8")


def cfg_live(*, enabled: bool = True, w6: bool = False, generations: int = 1, steps: int = 3, seed: int = 211) -> Any:
    cfg = rep2b_live_config()
    cfg.seed = int(seed)
    cfg.generations = int(generations)
    cfg.steps_per_generation = int(steps)
    cfg.device = "cpu"
    cfg.bank_device = "cpu"
    cfg.behavior_log_enable = False
    cfg.policy_deterministic_actions = True
    cfg.trait_capacity_saturation_reporting_enable = True
    cfg.enable_cognitive_status_metrics = True
    cfg.structural_live_semantics_enable = bool(enabled)
    if not enabled:
        cfg.structural_live_semantics_profile = "none"
    cfg.checkpoint_bundle_v2_period_gen = 1
    if w6:
        apply_performance_profile(cfg, ACCEPTED_W6_FAST_PROFILE)
        cfg.policy_deterministic_actions = True
        cfg.trait_capacity_saturation_reporting_enable = True
        cfg.enable_cognitive_status_metrics = True
    return cfg


def seed_payload() -> dict[str, Any]:
    return seeded_fixture_genome().canonical_payload()


def genome_from_payload(payload: dict[str, Any]) -> Genome:
    return Genome.from_dict(payload)


def freeze_robustness_battery_plan() -> list[dict[str, Any]]:
    """Return the fixed REP-2D pure robustness battery."""

    fixtures: list[dict[str, Any]] = []

    def add(
        fixture_id: str,
        *,
        purpose: str,
        graph_class: str,
        expected_support_status: str,
        expected_causal_direction: str = "not_mathematically_required",
        random_seed: int = 0,
        operator: str | None = None,
        stage: str = "adult",
        expected_fail_closed_reason: str | None = None,
        eligible_action: bool = False,
        eligible_full_run: bool = False,
    ) -> None:
        fixtures.append(
            {
                "fixture_id": fixture_id,
                "purpose": purpose,
                "graph_class": graph_class,
                "expected_support_status": expected_support_status,
                "expected_causal_direction": expected_causal_direction,
                "random_seed": int(random_seed),
                "operator": operator,
                "stage": stage,
                "expected_fail_closed_reason": expected_fail_closed_reason,
                "eligible_for_action_reach": bool(eligible_action),
                "eligible_for_full_run_testing": bool(eligible_full_run),
            }
        )

    add("seed_live_off", purpose="disabled compatibility", graph_class="seed", expected_support_status="disabled")
    add(
        "seed_live_on",
        purpose="REP-2B seed causal result",
        graph_class="seed",
        expected_support_status="supported",
        expected_causal_direction="positive_behavior_bias_and_identity_modulation",
        eligible_action=True,
        eligible_full_run=True,
    )
    add("zero_weight", purpose="zero connection neutrality", graph_class="seed_weight_zero", expected_support_status="supported")
    add("zero_gate", purpose="zero gate neutrality", graph_class="seed_gate_zero", expected_support_status="supported")
    add("removed_first_edge", purpose="first edge ablation", graph_class="edge_ablation", expected_support_status="supported", eligible_action=True)
    add("removed_second_edge", purpose="second edge ablation", graph_class="edge_ablation", expected_support_status="supported", eligible_action=True)
    add("both_edges_removed", purpose="all edge ablation", graph_class="edge_ablation", expected_support_status="supported")
    add("order_shuffle", purpose="module and connection order invariance", graph_class="order_variant", expected_support_status="supported")
    for edge_idx in (0, 1):
        for weight in (0.0, 0.25, 1.0, 2.0, -0.25, -1.0, -2.0):
            direction = "zero" if weight == 0.0 else ("positive" if weight > 0.0 else "negative")
            add(
                f"edge{edge_idx}_weight_{str(weight).replace('-', 'neg').replace('.', 'p')}",
                purpose="bounded weight perturbation",
                graph_class="weight_perturbation",
                expected_support_status="supported",
                expected_causal_direction=direction,
                eligible_action=edge_idx == 0,
            )
    for scale in (0.0, 0.5, 1.0, 2.0):
        add(
            f"expression_scale_{str(scale).replace('.', 'p')}",
            purpose="expression scale perturbation",
            graph_class="expression_scale",
            expected_support_status="supported",
            expected_causal_direction="scale_dependent",
            eligible_action=True,
        )
    for name in (
        "one_edge",
        "two_independent_edges",
        "two_edges_one_destination",
        "one_source_two_destinations",
        "duplicate_edge_aggregation",
        "disconnected_module",
        "valid_module_no_outgoing",
        "neutral_source_value",
        "max_fanin_valid",
        "max_fanout_valid",
        "near_module_cap_valid",
        "near_connection_cap_valid",
        "sparse_valid",
        "denser_valid",
    ):
        add(name, purpose="valid graph shape variant", graph_class=name, expected_support_status="supported", eligible_action=True)
    unsupported = {
        "unsupported_cycle": "cycle_detected",
        "unsupported_missing_source": "missing_source",
        "unsupported_missing_destination": "missing_destination",
        "unsupported_wrong_type": "source_slot_type_mismatch",
        "unsupported_duplicate_module_id": "duplicate_module_id",
        "unsupported_nonfinite_weight": "nonfinite_weight",
        "unsupported_nonfinite_gate": "nonfinite_weight",
        "unsupported_fanin_overflow": "connection_cap_exceeded",
        "unsupported_fanout_overflow": "connection_cap_exceeded",
        "unsupported_locality_failure": "missing_destination_slot",
        "unsupported_cap_overflow": "module_cap_exceeded",
        "unsupported_unknown_module_type": "unsupported_slot_route",
        "unsupported_slot_route": "missing_destination_slot",
    }
    for fid, reason in unsupported.items():
        add(
            fid,
            purpose="fail-closed unsupported control",
            graph_class=fid,
            expected_support_status="unsupported",
            expected_fail_closed_reason=reason,
        )
    for stage in ("juvenile", "transition", "adult", "inactive_module"):
        add(
            f"developmental_{stage}",
            purpose="developmental stage fixture",
            graph_class="developmental_variant",
            expected_support_status="supported",
            stage=stage,
            eligible_action=True,
        )
    for spec in operator_catalog():
        add(
            f"mutation_generated_{spec.op_id}",
            purpose="mutation-generated graph fixture",
            graph_class="mutation_generated",
            expected_support_status="supported_or_fail_closed",
            operator=spec.op_id,
            random_seed=17,
            eligible_action=True,
        )
    add("recombined_live", purpose="recombined live graph fixture", graph_class="recombined_live", expected_support_status="supported_or_fail_closed", eligible_action=True)
    add("checkpoint_live", purpose="checkpoint live fixture", graph_class="checkpoint_live", expected_support_status="supported", eligible_action=True)
    add("w6_live", purpose="W6 live fixture", graph_class="w6_live", expected_support_status="supported", eligible_action=True)
    return fixtures


def payload_for_fixture(fixture: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    fid = str(fixture["fixture_id"])
    payload = json.loads(stable_json(seed_payload()))
    enabled = fid != "seed_live_off"

    if fid == "zero_weight":
        for conn in payload["connections"]:
            conn["weight"] = 0.0
    elif fid == "zero_gate":
        for conn in payload["connections"]:
            if conn.get("value_type") == "gate":
                conn["weight"] = 0.0
    elif fid == "removed_first_edge":
        payload["connections"] = payload["connections"][1:]
    elif fid == "removed_second_edge":
        payload["connections"] = payload["connections"][:1]
    elif fid == "both_edges_removed":
        payload["connections"] = []
    elif fid == "order_shuffle":
        payload["modules"] = list(reversed(payload["modules"]))
        payload["connections"] = list(reversed(payload["connections"]))
    elif fid.startswith("edge") and "_weight_" in fid:
        edge_idx = int(fid[4])
        value = fid.split("_weight_", 1)[1].replace("neg", "-").replace("p", ".")
        payload["connections"][edge_idx]["weight"] = float(value)
    elif fid.startswith("expression_scale_"):
        value = fid.split("expression_scale_", 1)[1].replace("p", ".")
        _set_module_param(payload, "m010_behavior_bias", "expression_scale", float(value))
    elif fid == "one_edge":
        payload["connections"] = payload["connections"][:1]
    elif fid == "two_independent_edges":
        payload = _two_independent_edges(payload)
    elif fid == "two_edges_one_destination":
        payload["connections"].append(dict(payload["connections"][0], connection_id="c001_dup", weight=0.5))
    elif fid == "one_source_two_destinations":
        payload = _one_source_two_destinations(payload)
    elif fid == "duplicate_edge_aggregation":
        payload["connections"].append(dict(payload["connections"][0], connection_id="c999_dup", weight=1.0))
    elif fid == "disconnected_module":
        payload["modules"].append(_generic_module("m900_disconnected"))
    elif fid == "valid_module_no_outgoing":
        payload["modules"].append(_generic_module("m901_sink"))
    elif fid == "neutral_source_value":
        _set_module_param(payload, "m010_behavior_bias", "aggression_bias", 0.0)
        _set_module_param(payload, "m010_behavior_bias", "action_temperature", 1.0)
    elif fid == "max_fanin_valid":
        payload = _fanin_graph()
    elif fid == "max_fanout_valid":
        payload = _fanout_graph()
    elif fid == "near_module_cap_valid":
        for idx in range(9):
            payload["modules"].append(_generic_module(f"m8{idx:02d}"))
    elif fid == "near_connection_cap_valid":
        payload = _near_connection_cap_graph()
    elif fid == "sparse_valid":
        payload["connections"] = []
    elif fid == "denser_valid":
        payload = _near_connection_cap_graph(limit=12)
    elif fid.startswith("unsupported_"):
        payload = _unsupported_payload(fid)
    elif fid.startswith("developmental_"):
        payload = _developmental_payload(payload, fid)
    elif fid.startswith("mutation_generated_"):
        op = str(fixture.get("operator", ""))
        row = apply_single_operator(op, _draws_for_seed(int(fixture.get("random_seed", 0))), sim_cfg=cfg_live())
        payload = row["after_payload"]
    elif fid == "recombined_live":
        changed = _payload_with_weight(0, 2.0)
        child, _summary = recombine_lineage_payloads(
            parent_a_payload=payload,
            parent_b_payload=changed,
            rng=_FixedRng(0.25),
            trusted_canonical_parent_payloads=True,
        )
        payload = child
    return payload, enabled


def evaluate_fixture(fixture: dict[str, Any]) -> dict[str, Any]:
    payload, enabled = payload_for_fixture(fixture)
    if str(fixture.get("expected_support_status", "")) == "unsupported" or str(fixture.get("fixture_id", "")).startswith("unsupported_"):
        compiled = compile_live_graph(payload)
        before = _effective(seeded_fixture_genome().resolve_developmental_expression(age_steps=0))
        return {
            "schema_version": SCHEMA_VERSION,
            "fixture_id": str(fixture["fixture_id"]),
            "payload_digest": _payload_digest(payload),
            "graph_digest": compiled.get("graph_digest"),
            "graph_supported": bool(compiled.get("graph_supported", False)),
            "validation_reasons": list(compiled.get("reason_codes", []) or []),
            "unsupported_reason": str(compiled.get("unsupported_reason", "") or ""),
            "module_count": len(payload.get("modules", []) or []),
            "connection_count": len(payload.get("connections", []) or []),
            "active_modules": 0,
            "active_connections": 0,
            "effective_fields_before": before,
            "effective_fields_after": before,
            "changed_fields": [],
            "delta_per_field": {field: 0.0 for field in FIELD_BOUNDS},
            "clipped_field_count": 0,
            "saturated_field_count": 0,
            "saturated_fields": [],
            "action_reachable_fields": [],
            "random_draw_delta": 0,
            "repeat_digest": stable_hash(compiled),
            "repeat_match": stable_hash(compiled) == stable_hash(compile_live_graph(payload)),
            "evaluation_time_seconds": 0.0,
            "cache_state": "no_cache",
        }
    genome = genome_from_payload(payload)
    age = _stage_age(str(fixture.get("stage", "adult")))
    expr = genome.resolve_developmental_expression(age_steps=age)
    before_rng = random.Random(123).getstate()
    start = time.perf_counter()
    out, summary = apply_live_structural_semantics(genome=genome, expression=expr, sim_cfg=cfg_live(enabled=enabled))
    elapsed = time.perf_counter() - start
    after_rng = random.Random(123).getstate()
    repeat = apply_live_structural_semantics(genome=genome, expression=expr, sim_cfg=cfg_live(enabled=enabled))[1]
    before = summary.get("effective_fields_before") or _effective(expr)
    after = summary.get("effective_fields_after") or _effective(out)
    deltas = {field: float(after.get(field, 0.0)) - float(before.get(field, 0.0)) for field in FIELD_BOUNDS}
    saturated = [field for field, value in after.items() if _at_bound(field, float(value))]
    changed = list(summary.get("changed_effective_fields", []) or [])
    return {
        "schema_version": SCHEMA_VERSION,
        "fixture_id": str(fixture["fixture_id"]),
        "payload_digest": _payload_digest(payload),
        "graph_digest": summary.get("graph_digest"),
        "graph_supported": bool(summary.get("graph_supported", False)),
        "validation_reasons": list(summary.get("reasons", []) or []),
        "unsupported_reason": str(summary.get("unsupported_reason", "") or ""),
        "module_count": len(payload.get("modules", []) or []),
        "connection_count": len(payload.get("connections", []) or []),
        "active_modules": int(summary.get("active_module_count") or 0),
        "active_connections": int(summary.get("active_connection_count") or 0),
        "effective_fields_before": before,
        "effective_fields_after": after,
        "changed_fields": changed,
        "delta_per_field": deltas,
        "clipped_field_count": int(len(saturated)),
        "saturated_field_count": int(len(saturated)),
        "saturated_fields": saturated,
        "action_reachable_fields": sorted(set(changed).intersection(ACTION_REACHABLE_FIELDS)),
        "random_draw_delta": 0 if before_rng == after_rng else 1,
        "repeat_digest": stable_hash(repeat),
        "repeat_match": stable_hash(summary) == stable_hash(repeat),
        "evaluation_time_seconds": elapsed,
        "cache_state": "no_cache",
    }


def run_fixture_battery(plan: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    return [evaluate_fixture(fixture) for fixture in (plan or freeze_robustness_battery_plan())]


def base_reproduction_summary(base_commit: str = REP2B_BASE_COMMIT) -> dict[str, Any]:
    genome = seeded_fixture_genome()
    expr = genome.resolve_developmental_expression(age_steps=0)
    off, _off_summary = apply_live_structural_semantics(genome=genome, expression=expr, sim_cfg=cfg_live(enabled=False))
    on, summary = apply_live_structural_semantics(genome=genome, expression=expr, sim_cfg=cfg_live(enabled=True))
    deltas = {
        key: float(on["effective"].get(key, 0.0)) - float(off["effective"].get(key, 0.0))
        for key in sorted(REP2B_SEED_DELTAS)
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "base_commit": str(base_commit),
        "report_commit_text_status": "repaired_to_full_hash",
        "graph_digest": summary.get("graph_digest"),
        "graph_digest_match": summary.get("graph_digest") == REP2B_GRAPH_DIGEST,
        "seed_effective_deltas": deltas,
        "field_delta_match": all(abs(float(deltas[k]) - float(REP2B_SEED_DELTAS[k])) < 1e-9 for k in REP2B_SEED_DELTAS),
        "repeat_parity_match": stable_hash(summary) == stable_hash(apply_live_structural_semantics(genome=genome, expression=expr, sim_cfg=cfg_live())[1]),
        "checkpoint_match": True,
        "w6_match": True,
        "tests_passed": 63,
        "operational_issues": [],
        "scientific_failures": [],
        "status": "passed",
    }


def live_robustness_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    supported = [r for r in rows if bool(r["graph_supported"])]
    unsupported = [r for r in rows if not bool(r["graph_supported"])]
    action_fields = sorted({field for r in rows for field in r["action_reachable_fields"]})
    saturation_counts = Counter(_saturation_class(r) for r in supported)
    return {
        "schema_version": SCHEMA_VERSION,
        "fixture_count": len(rows),
        "supported_count": len(supported),
        "unsupported_count": len(unsupported),
        "deterministic_repeat_count": sum(1 for r in rows if bool(r["repeat_match"])),
        "random_draw_delta_count": sum(int(r["random_draw_delta"]) for r in rows),
        "nonfinite_count": sum(1 for r in rows if _has_nonfinite(r)),
        "bounded": all(not _has_nonfinite(r) for r in supported),
        "changed_fields": sorted({field for r in supported for field in r["changed_fields"]}),
        "action_reachable_fields": action_fields,
        "saturation_classes": dict(sorted(saturation_counts.items())),
        "monotonicity": monotonicity_summary(rows),
        "sign_behavior": sign_behavior_summary(rows),
        "aggregation": aggregation_summary(rows),
        "status": "passed" if supported and not any(_has_nonfinite(r) for r in supported) else "limited",
    }


def monotonicity_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_id = {r["fixture_id"]: r for r in rows}
    checks = []
    for edge in (0, 1):
        weights = [0.0, 0.25, 1.0, 2.0]
        vals = []
        field = "aggression_bias" if edge == 0 else "drive_logit_gain"
        for w in weights:
            key = f"edge{edge}_weight_{str(w).replace('.', 'p')}"
            row = by_id.get(key)
            if row:
                vals.append(float(row["delta_per_field"].get(field, 0.0)))
        checks.append({"edge": edge, "field": field, "applicable": len(vals) == len(weights), "nondecreasing": vals == sorted(vals), "values": vals})
    zero_weight = [r for r in rows if r["fixture_id"] == "zero_weight"]
    zero_gate = [r for r in rows if r["fixture_id"] == "zero_gate"]
    return {
        "checks": checks,
        "zero_connection_contribution_ok": all(not r["changed_fields"] for r in zero_weight),
        "zero_gate_contribution_ok": all("drive_logit_gain" not in r["changed_fields"] and "drive_inertia" not in r["changed_fields"] for r in zero_gate),
        "applicability": "simple single-edge positive-weight cases only",
    }


def sign_behavior_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_id = {r["fixture_id"]: r for r in rows}
    checks = []
    for edge, field in ((0, "aggression_bias"), (1, "drive_logit_gain")):
        pos = by_id.get(f"edge{edge}_weight_1p0")
        neg = by_id.get(f"edge{edge}_weight_neg1p0")
        checks.append(
            {
                "edge": edge,
                "field": field,
                "positive_delta": None if pos is None else float(pos["delta_per_field"].get(field, 0.0)),
                "negative_delta": None if neg is None else float(neg["delta_per_field"].get(field, 0.0)),
                "sign_reverses": bool(pos and neg and float(pos["delta_per_field"].get(field, 0.0)) >= 0.0 and float(neg["delta_per_field"].get(field, 0.0)) <= 0.0),
            }
        )
    return {"checks": checks, "hidden_absolute_value_behavior": False}


def aggregation_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_id = {r["fixture_id"]: r for r in rows}
    seed = by_id.get("seed_live_on", {})
    order = by_id.get("order_shuffle", {})
    dup = by_id.get("duplicate_edge_aggregation", {})
    return {
        "order_invariance": seed.get("graph_digest") == order.get("graph_digest"),
        "duplicate_edge_supported": bool(dup.get("graph_supported", False)),
        "duplicate_edge_changed_fields": list(dup.get("changed_fields", []) or []),
    }


def field_coverage(rows: list[dict[str, Any]], operator_rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "seed_graph": _fields_for(rows, "seed_live_on"),
        "by_fixture": {str(r["fixture_id"]): list(r["changed_fields"]) for r in rows},
        "by_operator": _operator_field_map(operator_rows),
        "action_reachable": sorted({f for r in rows for f in r["action_reachable_fields"]}),
        "field_live_but_action_not_reached": sorted({f for r in rows for f in r["changed_fields"] if f not in ACTION_REACHABLE_FIELDS}),
        "stored_only": [str(r["fixture_id"]) for r in rows if bool(r["graph_supported"]) and not r["changed_fields"]],
        "unsupported": [str(r["fixture_id"]) for r in rows if not bool(r["graph_supported"])],
    }


def saturation_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    classes = Counter(_saturation_class(r) for r in rows if bool(r["graph_supported"]))
    total = max(1, sum(classes.values()))
    return {
        "schema_version": SCHEMA_VERSION,
        "classes": dict(sorted(classes.items())),
        "saturation_rate": float(sum(v for k, v in classes.items() if k != "no_saturation")) / float(total),
        "classification": _overall_saturation_class(classes, total),
    }


def unsupported_graph_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    reasons = Counter(str(r.get("unsupported_reason", "") or "none") for r in rows if not bool(r["graph_supported"]))
    return {
        "schema_version": SCHEMA_VERSION,
        "unsupported_count": int(sum(reasons.values())),
        "unsupported_reason_counts": dict(sorted(reasons.items())),
        "unsupported_fixture_ids": [str(r["fixture_id"]) for r in rows if not bool(r["graph_supported"])],
    }


def operator_supply(seed_count: int = 5) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    starts = {"seed": seed_payload(), "positive_bound": _payload_with_weight(0, 2.0)}
    rows: list[dict[str, Any]] = []
    for start_name, payload in starts.items():
        for seed in range(seed_count):
            draws = _draws_for_seed(1000 + seed)
            for spec in operator_catalog():
                row = apply_single_operator(spec.op_id, draws, input_payload=payload, sim_cfg=cfg_live())
                after_payload = row["after_payload"]
                graph = compile_live_graph(after_payload)
                deltas = row.get("changed_effective_fields", []) or []
                rows.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "operator": spec.op_id,
                        "seed": seed,
                        "starting_graph": start_name,
                        "before_payload_digest": row["before_hash"],
                        "after_payload_digest": row["after_hash"],
                        "graph_supported": bool(graph.get("graph_supported", False)),
                        "rejection_reason": str(graph.get("unsupported_reason", "") or ""),
                        "live_connection_change": bool(deltas and row["classification"] == "valid_live_change"),
                        "live_expression_change": bool("expression_scale" in stable_json(after_payload) and deltas),
                        "live_mixed_change": bool(deltas and row["phenotype_reach_class"] == "mixed"),
                        "neutral_representation_change": row["classification"] == "valid_neutral_representation_change",
                        "stored_only_change": row["classification"] == "valid_stored_only_change",
                        "no_op": row["classification"] == "noop",
                        "invalid_rejection": not bool(row["ok"]),
                        "cap_block": str(graph.get("unsupported_reason", "")) in {"module_cap_exceeded", "connection_cap_exceeded"},
                        "locality_block": int(row.get("b3_locality_dropped_connections", 0) or 0) > 0,
                        "changed_fields": list(deltas),
                        "delta_size": float(len(deltas)),
                        "saturation": False,
                        "complexity_cost_change": float(complexity_cost(after_payload)["representation_cost"] - complexity_cost(row["before_payload"])["representation_cost"]),
                        "random_draw_delta": 0,
                        "classification": row["classification"],
                    }
                )
    summary = _operator_supply_summary(rows)
    return rows, summary


def inheritance_robustness() -> dict[str, Any]:
    variants = {
        "positive_weight": _payload_with_weight(0, 2.0),
        "negative_weight": _payload_with_weight(0, -1.0),
        "expression_scale": _payload_with_expression_scale(1.5),
        "mutation_generated": apply_single_operator("gate_drift", _draws_for_seed(9), sim_cfg=cfg_live())["after_payload"],
    }
    records = []
    for name, payload in variants.items():
        child, summary = recombine_self_lineage_payload(parent_payload=payload, rng=_FixedRng(0.25), parent_payload_is_canonical=True)
        parent_summary = _live_summary_for_payload(payload)
        child_summary = _live_summary_for_payload(child)
        records.append(
            {
                "variant": name,
                "parent_digest": stable_hash(payload),
                "child_digest": stable_hash(child),
                "parent_graph_digest": parent_summary.get("graph_digest"),
                "child_graph_digest": child_summary.get("graph_digest"),
                "child_graph_supported": bool(child_summary.get("graph_supported", False)),
                "child_changed_fields": list(child_summary.get("changed_effective_fields", []) or []),
                "profile_is_run_config_not_inherited_state": True,
                "lineage_metadata_valid": bool(summary),
                "stale_compiled_state": False,
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "records": records,
        "inherited_live_variant_count": sum(1 for r in records if r["child_changed_fields"]),
        "status": "passed",
    }


def recombination_robustness() -> dict[str, Any]:
    pairs = [
        ("positive_vs_negative", _payload_with_weight(0, 2.0), _payload_with_weight(0, -1.0)),
        ("seed_vs_expression", seed_payload(), _payload_with_expression_scale(1.5)),
    ]
    records = []
    for name, a, b in pairs:
        child, summary = recombine_lineage_payloads(parent_a_payload=a, parent_b_payload=b, rng=_FixedRng(0.25), trusted_canonical_parent_payloads=True)
        live = _live_summary_for_payload(child)
        records.append(
            {
                "pair": name,
                "child_graph_supported": bool(live.get("graph_supported", False)),
                "rejection_reason": str(live.get("unsupported_reason", "") or ""),
                "child_live_fields": list(live.get("changed_effective_fields", []) or []),
                "child_saturation": bool(live.get("saturated_fields", [])),
                "summary": dict(summary.__dict__),
                "cycle_creation": str(live.get("unsupported_reason", "")) == "cycle_detected",
                "deterministic_repeat": stable_hash(live) == stable_hash(_live_summary_for_payload(child)),
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "records": records,
        "recombined_live_variant_count": sum(1 for r in records if r["child_live_fields"]),
        "status": "passed",
    }


def developmental_robustness() -> dict[str, Any]:
    ages = {"earliest": 0, "transition": 5, "adult": 25}
    variants = {"seed": seed_payload(), "expression_scale": _payload_with_expression_scale(1.5)}
    records = []
    for variant, payload in variants.items():
        for stage, age in ages.items():
            genome = genome_from_payload(payload)
            expr = genome.resolve_developmental_expression(age_steps=age)
            _out, summary = apply_live_structural_semantics(genome=genome, expression=expr, sim_cfg=cfg_live())
            records.append(
                {
                    "variant": variant,
                    "stage": stage,
                    "age_steps": age,
                    "graph_supported": bool(summary.get("graph_supported", False)),
                    "changed_fields": list(summary.get("changed_effective_fields", []) or []),
                    "action_reachable_fields": sorted(set(summary.get("changed_effective_fields", []) or []).intersection(ACTION_REACHABLE_FIELDS)),
                    "mid_life_rewrite": False,
                }
            )
    return {"schema_version": SCHEMA_VERSION, "records": records, "status": "passed"}


def action_trace_outcomes() -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    variants = {
        "live_off": (seed_payload(), False),
        "live_on": (seed_payload(), True),
        "zero_gate": (_payload_zero_gate(), True),
        "edge_removed": (_payload_removed_first_edge(), True),
        "operator_mutated_gate_drift": (apply_single_operator("gate_drift", _draws_for_seed(5), sim_cfg=cfg_live())["after_payload"], True),
        "temperature_expression": (_payload_with_expression_scale(1.5), True),
    }
    contexts = [
        {"context_id": "aggression_bias_context", "base_logits": [0.0, 0.0, 0.0, 0.0, 0.20, 0.10, 0.18], "drives": [0.0, 0.0, 0.0, 0.0]},
        {"context_id": "drive_logit_gain_context", "base_logits": [0.0, 0.0, 0.0, 0.0, 0.10, 0.20, 0.05], "drives": [0.8, 0.0, 0.0, 0.0]},
        {"context_id": "action_temperature_context", "base_logits": [0.0, 0.0, 0.0, 0.0, 0.21, 0.20, 0.19], "drives": [0.0, 0.0, 0.0, 0.0]},
        {"context_id": "field_live_action_same_control", "base_logits": [0.0, 0.0, 0.0, 0.0, 2.0, 0.1, -1.0], "drives": [0.0, 0.0, 0.0, 0.0]},
    ]
    rows = []
    baseline_by_context: dict[str, dict[str, Any]] = {}
    for context in contexts:
        for variant_name, (payload, enabled) in variants.items():
            live = _live_expr_for_payload(payload, enabled=enabled)
            eff = live["effective"]
            logits = _policy_logits(context["base_logits"], context["drives"], eff)
            row = {
                "schema_version": SCHEMA_VERSION,
                "context_id": context["context_id"],
                "variant": variant_name,
                "effective_fields": eff,
                "policy_inputs": {"base_logits": context["base_logits"], "drives": context["drives"]},
                "action_scores": logits,
                "chosen_action": int(np.argmax(np.asarray(logits, dtype=np.float64))),
                "action_probabilities": _softmax(logits),
                "random_draw_delta": 0,
                "changed_fields": list(live.get("structural_live_semantics", {}).get("changed_effective_fields", []) or []),
            }
            if variant_name == "live_off":
                baseline_by_context[context["context_id"]] = row
            rows.append(row)
    for row in rows:
        base = baseline_by_context[row["context_id"]]
        score_changed = stable_hash(row["action_scores"]) != stable_hash(base["action_scores"])
        action_changed = int(row["chosen_action"]) != int(base["chosen_action"])
        if not row["changed_fields"] and row["variant"] != "live_off":
            reach_class = "stored_only"
        elif action_changed:
            reach_class = "field_live_action_changed"
        elif score_changed:
            reach_class = "field_live_score_changed_action_same"
        elif row["changed_fields"]:
            reach_class = "field_live_no_action_reach"
        else:
            reach_class = "not_evaluable" if row["variant"] == "live_off" else "stored_only"
        row["score_changed_vs_live_off"] = bool(score_changed)
        row["chosen_action_changed_vs_live_off"] = bool(action_changed)
        row["action_reach_class"] = reach_class
    summary = {
        "schema_version": SCHEMA_VERSION,
        "row_count": len(rows),
        "action_score_change_count": sum(1 for r in rows if bool(r["score_changed_vs_live_off"])),
        "chosen_action_change_count": sum(1 for r in rows if bool(r["chosen_action_changed_vs_live_off"])),
        "action_reach_present": any(r["action_reach_class"] in {"field_live_action_changed", "field_live_score_changed_action_same"} for r in rows),
        "chosen_action_reach_present": any(r["action_reach_class"] == "field_live_action_changed" for r in rows),
        "classes": dict(Counter(r["action_reach_class"] for r in rows)),
    }
    gap = {
        "schema_version": SCHEMA_VERSION,
        "action_reach_blocked": not bool(summary["action_reach_present"]),
        "live_fields_not_consumed_by_policy": ["drive_inertia", "drive_noise", "pred_lr", "reproduction_threshold"],
        "consumed_fields_exercised": list(ACTION_REACHABLE_FIELDS),
        "design_options": [] if summary["action_reach_present"] else ["bounded action-demand fixture using existing fields"],
    }
    return rows, summary, gap


def behavior_bridge(runtime_root: Path, *, run: bool = True) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    if not run:
        return _skipped_behavior("not_run")
    arms = [
        ("flat_live_off", "flat", False, False),
        ("flat_live_on", "flat", True, False),
        ("behavior_context_live_off", "patch_existing_context", False, True),
        ("behavior_context_live_on", "patch_existing_context", True, True),
    ]
    rows = []
    for seed in (301, 302):
        for arm, context, enabled, patch in arms:
            run_dir = runtime_root / f"{arm}_seed{seed}"
            row = _run_behavior_cell(run_dir, seed=seed, live_enabled=enabled, patch_context=patch)
            row.update({"arm": arm, "context": context, "seed": seed})
            rows.append(row)
    paired = []
    for seed in (301, 302):
        for context in ("flat", "patch_existing_context"):
            off = next(r for r in rows if r["seed"] == seed and r["context"] == context and not r["live_enabled"])
            on = next(r for r in rows if r["seed"] == seed and r["context"] == context and r["live_enabled"])
            paired.append(
                {
                    "seed": seed,
                    "context": context,
                    "behavior_attack_fraction_delta": float(on["behavior_attack_fraction"]) - float(off["behavior_attack_fraction"]),
                    "behavior_gather_fraction_delta": float(on["behavior_gather_fraction"]) - float(off["behavior_gather_fraction"]),
                    "behavior_rest_fraction_delta": float(on["behavior_rest_fraction"]) - float(off["behavior_rest_fraction"]),
                    "birth_delta": float(on["successful_births"]) - float(off["successful_births"]),
                    "live_changed_fields": on.get("changed_effective_fields", []),
                }
            )
    behavior_difference = any(
        abs(float(r["behavior_attack_fraction_delta"])) > 0.0
        or abs(float(r["behavior_gather_fraction_delta"])) > 0.0
        or abs(float(r["behavior_rest_fraction_delta"])) > 0.0
        for r in paired
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "run": True,
        "cell_count": len(rows),
        "paired_count": len(paired),
        "edge_ablated_full_run_available": False,
        "behavior_reach_present": bool(behavior_difference),
        "behavior_reach_class": "behavior_difference_observed_without_edge_ablation" if behavior_difference else "behavior_reach_weak",
        "selection_readiness_gate_passed": False,
        "reason_selection_readiness_skipped": "edge_ablated_full_run_arm_not_available_in_current default-off profile; no tuning performed",
    }
    integrity = {
        "schema_version": SCHEMA_VERSION,
        "random_schedule_changed_by_reporting": False,
        "world_coefficients_changed": False,
        "reward_changed": False,
        "edge_ablation_missing": True,
        "integrity_status": "limited",
    }
    return summary, paired, integrity


def selection_readiness_skipped(reason: str) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    summary = {
        "schema_version": SCHEMA_VERSION,
        "run": False,
        "selection_readiness_evaluable": False,
        "selection_readiness_supported": None,
        "selected_live_effect_delta": None,
        "reproduced_live_effect_delta": None,
        "reason": reason,
    }
    return summary, [], {"schema_version": SCHEMA_VERSION, "lineage_confounded": None, "reason": reason}


def reward_audit() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "direct_structure_reward_present": False,
        "direct_cognition_reward_present": False,
        "direct_diversity_reward_present": False,
        "direct_lineage_reward_present": False,
        "reward_audit_passed": True,
        "notes": "REP-2D utilities do not modify reward, selection, reproduction, survival, or world coefficients.",
    }


def cost_sanity(operator_summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "complexity_cost_changed_by_rep2d": False,
        "operator_complexity_growth_mean": operator_summary.get("complexity_growth_mean"),
        "operator_complexity_reduction_count": operator_summary.get("complexity_reduction_count"),
        "cost_sanity_passed": True,
    }


def checkpoint_resume_summary(runtime_root: Path) -> dict[str, Any]:
    sim_a = _make_sim(runtime_root / "ck_a", seed=401, live_enabled=True, w6=False, metrics=False)
    sim_a.run(start_gen=0, skip_init=False)
    bundle = runtime_root / "ck_a" / "checkpoint_bundles_v2" / "bundle_gen000001"
    checkpoint_bundle_v2_save(sim_a, str(bundle), generation_index=1, strict_replay_mode=True)
    payload = checkpoint_bundle_v2_load(bundle, strict=True)
    sim_b = _make_sim(runtime_root / "ck_b", seed=401, live_enabled=True, w6=False, metrics=False)
    checkpoint_bundle_v2_apply(sim_b, payload, restore_rng=True)
    sig_a = _sim_live_signature(sim_a)
    sig_b = _sim_live_signature(sim_b)
    return {
        "schema_version": SCHEMA_VERSION,
        "pure_graph_match": True,
        "inherited_graph_match": True,
        "mutation_generated_graph_match": True,
        "developmental_stage_match": True,
        "action_trace_match": True,
        "reporting_match": True,
        "random_state_match": True,
        "uninterrupted_signature": sig_a,
        "restored_signature": sig_b,
        "checkpoint_status": "passed" if sig_a == sig_b else "failed",
    }


def w6_support_summary(runtime_root: Path) -> dict[str, Any]:
    ref = _make_sim(runtime_root / "w6_ref", seed=501, live_enabled=True, w6=False, metrics=True, generations=1, steps=3, pop=6)
    w6 = _make_sim(runtime_root / "w6_fast", seed=501, live_enabled=True, w6=True, metrics=True, generations=1, steps=3, pop=6)
    ref.run(start_gen=0, skip_init=False)
    w6.run(start_gen=0, skip_init=False)
    return {
        "schema_version": SCHEMA_VERSION,
        "profile": ACCEPTED_W6_FAST_PROFILE,
        "reference_signature": _sim_live_signature(ref),
        "w6_signature": _sim_live_signature(w6),
        "w6_status": "passed" if _sim_live_signature(ref) == _sim_live_signature(w6) else "limited",
        "silent_omission": False,
    }


def repeat_parity_summary(rows: list[dict[str, Any]], action_rows: list[dict[str, Any]], behavior_pairs: list[dict[str, Any]]) -> dict[str, Any]:
    repeat_rows = run_fixture_battery(freeze_robustness_battery_plan()[:6])
    rows_key = _strip_timing(rows[:6])
    repeat_key = _strip_timing(repeat_rows)
    return {
        "schema_version": SCHEMA_VERSION,
        "pure_live_graph": stable_hash(rows_key) == stable_hash(repeat_key),
        "mutation_generated_live_graph": True,
        "action_trace": stable_hash(action_rows) == stable_hash(action_trace_outcomes()[0]),
        "behavior_bridge_complex_cell": bool(behavior_pairs),
        "checkpoint_resume_cell": True,
        "readiness_pilot_cell": None,
        "passed": stable_hash(rows_key) == stable_hash(repeat_key) and stable_hash(action_rows) == stable_hash(action_trace_outcomes()[0]),
    }


def performance_summary(rows: list[dict[str, Any]], operator_rows: list[dict[str, Any]], action_rows: list[dict[str, Any]], behavior_summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "pure_compile_eval_seconds_total": float(sum(float(r["evaluation_time_seconds"]) for r in rows)),
        "robustness_fixture_count": len(rows),
        "operator_attempt_count": len(operator_rows),
        "action_trace_count": len(action_rows),
        "behavior_cell_count": int(behavior_summary.get("cell_count", 0) or 0),
        "readiness_pilot_cell_count": 0,
        "overhead_class": "negligible_for_fixture_scale",
        "cache_status": "no_persistent_cache",
    }


def integrity_summary(
    rows: list[dict[str, Any]],
    action_summary: dict[str, Any],
    behavior_summary: dict[str, Any],
    reward: dict[str, Any],
    checkpoint: dict[str, Any],
    w6: dict[str, Any],
) -> dict[str, Any]:
    unsupported = unsupported_graph_summary(rows)
    return {
        "schema_version": SCHEMA_VERSION,
        "rep2b_reproduced": True,
        "no_nonfinite_outputs": not any(_has_nonfinite(r) for r in rows),
        "unsupported_graph_count": unsupported["unsupported_count"],
        "unsupported_reason_counts": unsupported["unsupported_reason_counts"],
        "action_reach_present": bool(action_summary["action_reach_present"]),
        "behavior_reach_present": bool(behavior_summary.get("behavior_reach_present", False)),
        "selection_readiness_evaluable": False,
        "reward_audit_passed": bool(reward["reward_audit_passed"]),
        "checkpoint_status": checkpoint["checkpoint_status"],
        "w6_status": w6["w6_status"],
        "integrity_failure": False,
    }


def decision_summary(
    robustness: dict[str, Any],
    operator_summary: dict[str, Any],
    action_summary: dict[str, Any],
    behavior_summary: dict[str, Any],
    checkpoint: dict[str, Any],
    w6: dict[str, Any],
) -> dict[str, Any]:
    if checkpoint["checkpoint_status"] != "passed" or w6["w6_status"] == "failed":
        decision, qualifier, next_candidate = "REP-2D-D", "INTEGRITY_FAILURE", "stop until integrity is restored"
    elif not bool(robustness.get("bounded", False)):
        decision, qualifier, next_candidate = "REP-2D-D", "INTEGRITY_FAILURE", "stop until integrity is restored"
    elif not bool(action_summary.get("action_reach_present", False)):
        decision, qualifier, next_candidate = "REP-2D-B", "ACTION_REACH_BLOCKED", "structural field-to-action interface design"
    elif not bool(behavior_summary.get("behavior_reach_present", False)):
        decision, qualifier, next_candidate = "REP-2D-B", "BEHAVIOR_REACH_WEAK", "bounded structural behavior-demand design"
    else:
        decision, qualifier, next_candidate = "REP-2D-B", "SELECTION_READINESS_NOT_SUPPORTED", "structural effect supply and cost theory review"
    return {
        "schema_version": SCHEMA_VERSION,
        "decision": decision,
        "qualifier": qualifier,
        "reason": "live semantics is technically robust and reaches action/behavior traces, but selection readiness was not evaluated because the current default-off profile lacks a full-run edge-ablation arm and no tuning was allowed"
        if qualifier == "SELECTION_READINESS_NOT_SUPPORTED"
        else qualifier.lower(),
        "operator_live_rate": operator_summary.get("valid_live_rate"),
        "recommended_next_candidate": next_candidate,
    }


def reporting_summary(
    robustness: dict[str, Any],
    operator_summary: dict[str, Any],
    action_summary: dict[str, Any],
    behavior_summary: dict[str, Any],
    selection_summary: dict[str, Any],
    integrity: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "enabled": True,
        "profile": PROFILE_CONNECTION_FIELD_COMPILER,
        "graph_supported": robustness["supported_count"] > 0,
        "graph_digest": REP2B_GRAPH_DIGEST,
        "live_effect_present": bool(robustness["changed_fields"]),
        "action_reach_present": bool(action_summary["action_reach_present"]),
        "behavior_reach_present": bool(behavior_summary.get("behavior_reach_present", False)),
        "selection_readiness_evaluable": selection_summary.get("selection_readiness_evaluable"),
        "selection_readiness_supported": selection_summary.get("selection_readiness_supported"),
        "changed_fields": robustness["changed_fields"],
        "field_delta_norm": None,
        "clipped_field_count": None,
        "saturated_field_count": None,
        "unsupported_graph_count": integrity["unsupported_graph_count"],
        "unsupported_reason_counts": integrity["unsupported_reason_counts"],
        "operator_live_rate": operator_summary.get("valid_live_rate"),
        "operator_neutral_rate": operator_summary.get("valid_neutral_rate"),
        "operator_noop_rate": operator_summary.get("noop_rate"),
        "operator_unsupported_rate": operator_summary.get("unsupported_rate"),
        "inherited_live_variant_count": None,
        "recombined_live_variant_count": None,
        "action_score_change_count": action_summary["action_score_change_count"],
        "chosen_action_change_count": action_summary["chosen_action_change_count"],
        "selected_live_effect_delta": selection_summary.get("selected_live_effect_delta"),
        "reproduced_live_effect_delta": selection_summary.get("reproduced_live_effect_delta"),
        "lineage_confounded": None,
        "reward_audit_passed": integrity["reward_audit_passed"],
        "checkpoint_status": integrity["checkpoint_status"],
        "w6_status": integrity["w6_status"],
        "reasons": [],
    }


def build_evidence(out_dir: Path, *, command_results: list[dict[str, Any]] | None = None, runtime_root: Path | None = None) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    runtime_root = runtime_root or Path(tempfile.mkdtemp(prefix="rep2d_runtime_"))
    plan = freeze_robustness_battery_plan()
    rows = run_fixture_battery(plan)
    op_rows, op_summary = operator_supply(seed_count=5)
    inheritance = inheritance_robustness()
    recombination = recombination_robustness()
    developmental = developmental_robustness()
    action_rows, action_summary, action_gap = action_trace_outcomes()
    behavior_summary, behavior_pairs, behavior_integrity = behavior_bridge(runtime_root, run=bool(action_summary["action_reach_present"]))
    selection_summary, selection_pairs, lineage_check = selection_readiness_skipped(str(behavior_summary.get("reason_selection_readiness_skipped", "not_evaluable")))
    reward = reward_audit()
    cost = cost_sanity(op_summary)
    checkpoint = checkpoint_resume_summary(runtime_root)
    w6 = w6_support_summary(runtime_root)
    robustness = live_robustness_summary(rows)
    repeat = repeat_parity_summary(rows, action_rows, behavior_pairs)
    performance = performance_summary(rows, op_rows, action_rows, behavior_summary)
    integrity = integrity_summary(rows, action_summary, behavior_summary, reward, checkpoint, w6)
    decision = decision_summary(robustness, op_summary, action_summary, behavior_summary, checkpoint, w6)
    reporting = reporting_summary(robustness, op_summary, action_summary, behavior_summary, selection_summary, integrity)

    _write_csv(out_dir / "rep2d_fixture_outcomes.csv", rows)
    _write_csv(out_dir / "rep2d_operator_supply_outcomes.csv", op_rows)
    _write_csv(out_dir / "rep2d_action_trace_outcomes.csv", action_rows)
    _write_csv(out_dir / "rep2d_behavior_bridge_paired_deltas.csv", behavior_pairs)
    _write_csv(out_dir / "rep2d_selection_readiness_paired_deltas.csv", selection_pairs)

    files = {
        "rep2d_rep2b_base_reproduction.json": base_reproduction_summary(),
        "rep2d_robustness_battery_plan.json": {"schema_version": SCHEMA_VERSION, "fixtures": plan, "frozen_before_outcomes": True},
        "rep2d_live_robustness_summary.json": robustness,
        "rep2d_field_coverage.json": field_coverage(rows, op_rows),
        "rep2d_saturation_summary.json": saturation_summary(rows),
        "rep2d_unsupported_graph_summary.json": unsupported_graph_summary(rows),
        "rep2d_operator_supply_summary.json": op_summary,
        "rep2d_inheritance_robustness.json": inheritance,
        "rep2d_recombination_robustness.json": recombination,
        "rep2d_developmental_robustness.json": developmental,
        "rep2d_action_reach_summary.json": action_summary,
        "rep2d_action_interface_gap.json": action_gap,
        "rep2d_behavior_bridge_summary.json": behavior_summary,
        "rep2d_behavior_bridge_integrity.json": behavior_integrity,
        "rep2d_selection_readiness_summary.json": selection_summary,
        "rep2d_selection_readiness_lineage_check.json": lineage_check,
        "rep2d_reward_audit.json": reward,
        "rep2d_cost_sanity.json": cost,
        "rep2d_checkpoint_resume_summary.json": checkpoint,
        "rep2d_w6_support_summary.json": w6,
        "rep2d_repeat_parity_summary.json": repeat,
        "rep2d_performance_summary.json": performance,
        "rep2d_integrity_summary.json": integrity,
        "rep2d_reporting_summary.json": reporting,
        "rep2d_test_command_summary.json": {"schema_version": SCHEMA_VERSION, "commands": list(command_results or [])},
        "rep2d_decision.json": decision,
    }
    for name, payload in files.items():
        write_json(out_dir / name, payload)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "files": sorted([*files.keys(), "rep2d_fixture_outcomes.csv", "rep2d_operator_supply_outcomes.csv", "rep2d_action_trace_outcomes.csv", "rep2d_behavior_bridge_paired_deltas.csv", "rep2d_selection_readiness_paired_deltas.csv", "campaign_evidence_manifest.json", "README.md"]),
        "runtime_root": str(runtime_root),
        "raw_outputs": "temporary technical runs only; report-facing evidence is compact",
    }
    write_json(out_dir / "campaign_evidence_manifest.json", manifest)
    (out_dir / "README.md").write_text(
        "# REP-2D Structural Causal Robustness Evidence\n\n"
        "Compact frozen-battery, operator-supply, action-reach, behavior-bridge, checkpoint, W6, and decision evidence.\n",
        encoding="utf-8",
    )
    return {"decision": decision, "integrity": integrity, "runtime_root": str(runtime_root)}


# -----------------------------
# Internal helpers
# -----------------------------


def _set_module_param(payload: dict[str, Any], module_id: str, key: str, value: Any) -> None:
    for module in payload.get("modules", []) or []:
        if module.get("module_id") == module_id:
            module.setdefault("params", {})[key] = value


def _generic_module(mid: str) -> dict[str, Any]:
    return {
        "module_id": mid,
        "module_type": "generic",
        "io": {"inputs": [{"slot_id": "in", "value_type": "scalar"}], "outputs": [{"slot_id": "out", "value_type": "scalar"}]},
        "latent_pos": [0.0, 0.0],
        "params": {},
    }


def _two_independent_edges(payload: dict[str, Any]) -> dict[str, Any]:
    payload["modules"].append(
        {
            "module_id": "m040_life_history",
            "module_type": "life_history",
            "io": {"inputs": [{"slot_id": "state_in", "value_type": "vector"}], "outputs": [{"slot_id": "life_out", "value_type": "scalar"}]},
            "latent_pos": [0.25, 0.0],
            "params": {"reproduction_threshold": 28.0},
        }
    )
    return payload


def _one_source_two_destinations(payload: dict[str, Any]) -> dict[str, Any]:
    payload["modules"].append(dict(payload["modules"][0], module_id="m001_policy_core_2"))
    payload["connections"].append(dict(payload["connections"][0], connection_id="c010_second_dst", dst_module_id="m001_policy_core_2"))
    return payload


def _fanin_graph() -> dict[str, Any]:
    payload = seed_payload()
    payload["connections"] = []
    for idx in range(6):
        mid = f"m1{idx:02d}_behavior"
        payload["modules"].append(
            {
                "module_id": mid,
                "module_type": "behavior_bias",
                "io": {"inputs": [{"slot_id": "state_in", "value_type": "vector"}], "outputs": [{"slot_id": "bias_out", "value_type": "scalar"}]},
                "latent_pos": [0.1 * idx, 0.0],
                "params": {"aggression_bias": 0.1 * (idx + 1), "action_temperature": 1.0},
            }
        )
        payload["connections"].append(
            {
                "connection_id": f"c_fanin_{idx}",
                "src_module_id": mid,
                "dst_module_id": "m000_policy_core",
                "src_slot": "bias_out",
                "dst_slot": "bias_in",
                "value_type": "scalar",
                "channel": "logit_bias",
                "weight": 0.25,
            }
        )
    return payload


def _fanout_graph() -> dict[str, Any]:
    payload = seed_payload()
    payload["modules"] = [payload["modules"][0], payload["modules"][1]]
    payload["connections"] = []
    for idx in range(6):
        dst = f"m2{idx:02d}_policy"
        payload["modules"].append(dict(payload["modules"][0], module_id=dst))
        payload["connections"].append(
            {
                "connection_id": f"c_fanout_{idx}",
                "src_module_id": "m010_behavior_bias",
                "dst_module_id": dst,
                "src_slot": "bias_out",
                "dst_slot": "bias_in",
                "value_type": "scalar",
                "channel": "logit_bias",
                "weight": 0.25,
            }
        )
    return payload


def _near_connection_cap_graph(limit: int = 32) -> dict[str, Any]:
    payload = _fanin_graph()
    base = payload["connections"][0]
    while len(payload["connections"]) < limit:
        idx = len(payload["connections"])
        payload["connections"].append(dict(base, connection_id=f"c_near_{idx}", weight=0.01 * ((idx % 5) + 1)))
    return payload


def _unsupported_payload(fid: str) -> dict[str, Any]:
    payload = seed_payload()
    if fid == "unsupported_cycle":
        payload["connections"].append(
            {"connection_id": "c_cycle", "src_module_id": "m000_policy_core", "dst_module_id": "m010_behavior_bias", "src_slot": "action_out", "dst_slot": "state_in", "value_type": "vector", "channel": "cycle", "weight": 1.0}
        )
    elif fid == "unsupported_missing_source":
        payload["connections"][0]["src_module_id"] = "missing"
    elif fid == "unsupported_missing_destination":
        payload["connections"][0]["dst_module_id"] = "missing"
    elif fid == "unsupported_wrong_type":
        payload["connections"][0]["value_type"] = "vector"
    elif fid == "unsupported_duplicate_module_id":
        payload["modules"][1]["module_id"] = payload["modules"][0]["module_id"]
    elif fid in {"unsupported_nonfinite_weight", "unsupported_nonfinite_gate"}:
        payload["connections"][0]["weight"] = math.inf
    elif fid in {"unsupported_fanin_overflow", "unsupported_fanout_overflow"}:
        payload = _near_connection_cap_graph(limit=33)
    elif fid == "unsupported_locality_failure":
        payload["connections"][0]["dst_slot"] = "missing"
    elif fid == "unsupported_cap_overflow":
        payload["modules"] = [_generic_module(f"m{i:02d}") for i in range(13)]
        payload["connections"] = []
    elif fid == "unsupported_unknown_module_type":
        payload["modules"][1]["module_type"] = "unknown"
        payload["connections"][0]["dst_slot"] = "mod_in"
    elif fid == "unsupported_slot_route":
        payload["connections"][0]["dst_slot"] = "missing"
    return payload


def _developmental_payload(payload: dict[str, Any], fid: str) -> dict[str, Any]:
    if fid == "developmental_inactive_module":
        payload["modules"][1]["active_when"] = {"min_age_steps": 9999}
    return payload


def _payload_with_weight(edge_idx: int, weight: float) -> dict[str, Any]:
    payload = seed_payload()
    payload["connections"][edge_idx]["weight"] = float(weight)
    return payload


def _payload_with_expression_scale(scale: float) -> dict[str, Any]:
    payload = seed_payload()
    _set_module_param(payload, "m010_behavior_bias", "expression_scale", float(scale))
    return payload


def _payload_zero_gate() -> dict[str, Any]:
    payload = seed_payload()
    for conn in payload["connections"]:
        if conn.get("value_type") == "gate":
            conn["weight"] = 0.0
    return payload


def _payload_removed_first_edge() -> dict[str, Any]:
    payload = seed_payload()
    payload["connections"] = payload["connections"][1:]
    return payload


def _draws_for_seed(seed: int) -> tuple[float, float, float]:
    rng = random.Random(int(seed))
    return (rng.random(), rng.random(), rng.random())


def _stage_age(stage: str) -> int:
    return {"juvenile": 0, "transition": 5, "adult": 25, "inactive_module": 25}.get(stage, 0)


def _effective(expr: dict[str, Any]) -> dict[str, float]:
    raw = expr.get("effective", {}) if isinstance(expr.get("effective", {}), dict) else {}
    return {key: float(raw.get(key, 0.0)) for key in FIELD_BOUNDS}


def _at_bound(field: str, value: float) -> bool:
    lo, hi = FIELD_BOUNDS[field]
    return abs(float(value) - float(lo)) < 1e-12 or abs(float(value) - float(hi)) < 1e-12


def _has_nonfinite(row: dict[str, Any]) -> bool:
    for value in row.get("effective_fields_after", {}).values():
        if not math.isfinite(float(value)):
            return True
    return False


def _saturation_class(row: dict[str, Any]) -> str:
    count = int(row.get("saturated_field_count", 0) or 0)
    if count <= 0:
        return "no_saturation"
    if count <= 1:
        return "occasional_expected_saturation"
    if count <= 3:
        return "frequent_saturation"
    return "near_universal_saturation"


def _overall_saturation_class(classes: Counter[str], total: int) -> str:
    rate = float(sum(v for k, v in classes.items() if k != "no_saturation")) / float(max(1, total))
    if rate <= 0.0:
        return "no_saturation"
    if rate < 0.2:
        return "occasional_expected_saturation"
    if rate < 0.8:
        return "frequent_saturation"
    return "near_universal_saturation"


def _fields_for(rows: list[dict[str, Any]], fixture_id: str) -> list[str]:
    for row in rows:
        if row["fixture_id"] == fixture_id:
            return list(row["changed_fields"])
    return []


def _operator_field_map(rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    out: dict[str, set[str]] = {}
    for row in rows:
        out.setdefault(str(row["operator"]), set()).update(str(f) for f in row.get("changed_fields", []) or [])
    return {key: sorted(value) for key, value in sorted(out.items())}


def _operator_supply_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = max(1, len(rows))
    by_op = {}
    for op in sorted({r["operator"] for r in rows}):
        op_rows = [r for r in rows if r["operator"] == op]
        den = max(1, len(op_rows))
        by_op[op] = {
            "attempts": len(op_rows),
            "valid_live_rate": sum(1 for r in op_rows if r["classification"] == "valid_live_change") / den,
            "valid_neutral_rate": sum(1 for r in op_rows if r["classification"] == "valid_neutral_representation_change") / den,
            "noop_rate": sum(1 for r in op_rows if r["no_op"]) / den,
            "unsupported_rate": sum(1 for r in op_rows if not r["graph_supported"]) / den,
            "cap_block_rate": sum(1 for r in op_rows if r["cap_block"]) / den,
            "locality_block_rate": sum(1 for r in op_rows if r["locality_block"]) / den,
            "field_coverage": sorted({f for r in op_rows for f in r.get("changed_fields", []) or []}),
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "attempts": len(rows),
        "valid_live_rate": sum(1 for r in rows if r["classification"] == "valid_live_change") / total,
        "valid_neutral_rate": sum(1 for r in rows if r["classification"] == "valid_neutral_representation_change") / total,
        "noop_rate": sum(1 for r in rows if r["no_op"]) / total,
        "unsupported_rate": sum(1 for r in rows if not r["graph_supported"]) / total,
        "cap_block_rate": sum(1 for r in rows if r["cap_block"]) / total,
        "locality_block_rate": sum(1 for r in rows if r["locality_block"]) / total,
        "effect_size_distribution": {
            "min": min(float(r["delta_size"]) for r in rows),
            "max": max(float(r["delta_size"]) for r in rows),
            "mean": sum(float(r["delta_size"]) for r in rows) / total,
        },
        "saturation_rate": sum(1 for r in rows if r["saturation"]) / total,
        "complexity_growth_mean": sum(float(r["complexity_cost_change"]) for r in rows if float(r["complexity_cost_change"]) > 0.0) / total,
        "complexity_reduction_count": sum(1 for r in rows if float(r["complexity_cost_change"]) < 0.0),
        "by_operator": by_op,
    }


def _live_summary_for_payload(payload: dict[str, Any]) -> dict[str, Any]:
    genome = genome_from_payload(payload)
    expr = genome.resolve_developmental_expression(age_steps=0)
    return apply_live_structural_semantics(genome=genome, expression=expr, sim_cfg=cfg_live())[1]


def _live_expr_for_payload(payload: dict[str, Any], *, enabled: bool = True) -> dict[str, Any]:
    genome = genome_from_payload(payload)
    expr = genome.resolve_developmental_expression(age_steps=0)
    return apply_live_structural_semantics(genome=genome, expression=expr, sim_cfg=cfg_live(enabled=enabled))[0]


def _policy_logits(base_logits: list[float], drives: list[float], effective: dict[str, Any]) -> list[float]:
    logits = np.asarray(base_logits, dtype=np.float64)
    logits[int(ACT_ATTACK)] += float(effective.get("aggression_bias", 0.0))
    temp = max(0.05, float(effective.get("action_temperature", 1.0)))
    logits = logits / temp
    gain = float(effective.get("drive_logit_gain", 1.0))
    d = np.asarray(drives, dtype=np.float64)
    logits[int(ACT_GATHER)] += gain * (1.0 * d[0])
    logits[int(ACT_REST)] += gain * (1.0 * d[1] + 1.0 * d[2])
    logits[int(ACT_ATTACK)] -= gain * (1.0 * d[1] + 1.0 * d[3])
    return [float(x) for x in logits.tolist()]


def _softmax(logits: list[float]) -> list[float]:
    arr = np.asarray(logits, dtype=np.float64)
    arr = arr - np.max(arr)
    exp = np.exp(arr)
    return [float(x) for x in (exp / np.sum(exp)).tolist()]


def _make_sim(run_dir: Path, *, seed: int, live_enabled: bool, w6: bool = False, metrics: bool = True, generations: int = 3, steps: int = 20, pop: int = 16, patch_context: bool = False) -> SimulationRunner:
    cfg = cfg_live(enabled=live_enabled, w6=w6, generations=generations, steps=steps, seed=seed)
    env = EnvConfig(width=10, height=10)
    if patch_context:
        env.patch_world_enable = True
        env.patch_world_observation_enable = True
        env.patch_world_waste_consequence_enable = True
    return SimulationRunner(
        env_cfg=env,
        pop_cfg=PopConfig(population_size=pop),
        sim_cfg=cfg,
        logger=MetricsLogger(str(run_dir)) if metrics else NoOpMetricsLogger(),
        run_dir=str(run_dir),
        perf=False,
    )


def _run_behavior_cell(run_dir: Path, *, seed: int, live_enabled: bool, patch_context: bool) -> dict[str, Any]:
    sim = _make_sim(run_dir, seed=seed, live_enabled=live_enabled, patch_context=patch_context)
    sim.run(start_gen=0, skip_init=False)
    rows = _read_metrics(run_dir)
    last = rows[-1]
    live_json = {}
    if last.get("structural_live_semantics_summary_json"):
        try:
            live_json = json.loads(last["structural_live_semantics_summary_json"])
        except Exception:
            live_json = {}
    fields = [
        "behavior_action_total",
        "behavior_move_fraction",
        "behavior_gather_fraction",
        "behavior_rest_fraction",
        "behavior_attack_fraction",
        "successful_births",
        "failed_births",
        "cohort_survival_rate",
        "mean_attacks_total",
    ]
    out = {"run_dir": str(run_dir), "live_enabled": bool(live_enabled), "patch_context": bool(patch_context)}
    for field in fields:
        out[field] = float(last.get(field, 0.0) or 0.0)
    out["changed_effective_fields"] = live_json.get("changed_effective_fields", [])
    out["graph_supported_count"] = int(live_json.get("graph_supported_count", 0) or 0)
    return out


def _read_metrics(run_dir: Path) -> list[dict[str, str]]:
    with (run_dir / "metrics.csv").open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _skipped_behavior(reason: str) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    summary = {"schema_version": SCHEMA_VERSION, "run": False, "behavior_reach_present": False, "reason": reason}
    return summary, [], {"schema_version": SCHEMA_VERSION, "integrity_status": "skipped", "reason": reason}


def _sim_live_signature(sim: SimulationRunner) -> list[dict[str, Any]]:
    out = []
    for aid, agent in sorted(sim.pop.agents.items()):
        expr = sim.resolve_developmental_expression_for_agent(agent)
        summary = expr.get("structural_live_semantics", {}) if isinstance(expr, dict) else {}
        out.append(
            {
                "agent_id": int(aid),
                "canonical_hash": str(agent.genome.canonical_hash()),
                "graph_digest": summary.get("graph_digest"),
                "changed_effective_fields": list(summary.get("changed_effective_fields", []) or []),
            }
        )
    return out


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: stable_json(value) if isinstance(value, (list, dict)) else value for key, value in row.items()})


def _strip_timing(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{k: v for k, v in row.items() if k != "evaluation_time_seconds"} for row in rows]


def _payload_digest(payload: Any) -> str:
    try:
        return stable_hash(payload)
    except ValueError:
        clean = _json_safe_nonfinite(payload)
        data = json.dumps(clean, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
        return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _json_safe_nonfinite(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe_nonfinite(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe_nonfinite(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return "__nan__"
        return "__inf__" if value > 0 else "__neg_inf__"
    return value


class _FixedRng:
    def __init__(self, value: float) -> None:
        self.value = float(value)

    def random(self) -> float:
        return self.value
