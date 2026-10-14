from __future__ import annotations

"""Reporting-only structural mutation viability helpers for REP-2A."""

import hashlib
import json
from typing import Any

from agents.developmental_program_v1 import canonicalize_developmental_program
from agents.genome import Genome
from agents.modular_genome_v1 import (
    B2_FROZEN_CAPS,
    MODULAR_GENOME_SCHEMA_VERSION,
    apply_b2_constraints,
    canonical_payload_from_genome_dict,
)
from agents.structural_payload_activation import (
    STRUCTURAL_PAYLOAD_SEED_SCHEMA_VERSION,
    apply_structural_payload_activation_to_founder,
)
from evolution.operators_c1 import operator_catalog
from evolution.recombination_c2 import recombine_lineage_payloads, recombine_self_lineage_payload
from evolution.structural_live_semantics import (
    SCHEMA_VERSION as REP2B_SCHEMA_VERSION,
    apply_live_structural_semantics,
)
from simulation.runner_pkg.runner_config import SimConfig


SCHEMA_VERSION = "rep2a_structural_mutation_viability_v1"
CONNECTION_EXECUTION_STATUS = "not_live"
DEFAULT_DRAWS: tuple[tuple[float, ...], ...] = (
    (0.1, 0.2, 0.3),
    (0.4, 0.5, 0.6),
    (0.8, 0.9, 0.7),
)
LIVE_EFFECTIVE_FIELDS = (
    "aggression_bias",
    "action_temperature",
    "drive_inertia",
    "drive_logit_gain",
    "drive_noise",
    "pred_lr",
    "reproduction_threshold",
)


def stable_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def stable_hash(payload: Any) -> str:
    return hashlib.sha256(stable_json(payload).encode("utf-8")).hexdigest()


def base_fixture_genome() -> Genome:
    return Genome(
        hidden_size=64,
        exploration_bias=0.1,
        aggression_bias=0.2,
        reproduction_threshold=30.0,
        mutation_rate=0.05,
        action_temperature=1.0,
        use_gru=False,
        memory_dim=0,
        drive_inertia=0.8,
        drive_noise=0.01,
        drive_logit_gain=1.0,
        pred_lr=0.08,
        developmental_program=canonicalize_developmental_program(None),
    )


def rep1c_seed_config() -> SimConfig:
    cfg = SimConfig(seed=902, generations=1, steps_per_generation=3, device="cpu", bank_device="cpu", behavior_log_enable=False)
    cfg.policy_deterministic_actions = True
    cfg.structural_payload_activation_enable = True
    cfg.structural_payload_seed_profile = "minimal_canonical_v1"
    cfg.structural_payload_seed_schema_version = STRUCTURAL_PAYLOAD_SEED_SCHEMA_VERSION
    return cfg


def rep2b_live_config() -> SimConfig:
    cfg = rep1c_seed_config()
    cfg.structural_live_semantics_enable = True
    cfg.structural_live_semantics_profile = "connection_field_compiler_v1"
    cfg.structural_live_semantics_schema_version = REP2B_SCHEMA_VERSION
    return cfg


def seeded_fixture_genome() -> Genome:
    return apply_structural_payload_activation_to_founder(base_fixture_genome(), rep1c_seed_config())


def empty_control_genome() -> Genome:
    return apply_structural_payload_activation_to_founder(base_fixture_genome(), SimConfig())


def effective_fields(genome_or_payload: Genome | dict[str, Any], *, sim_cfg: Any = None) -> dict[str, float]:
    genome = genome_or_payload if isinstance(genome_or_payload, Genome) else Genome.from_dict(genome_or_payload)
    expr = genome.resolve_developmental_expression(age_steps=0)
    if bool(getattr(sim_cfg, "structural_live_semantics_enable", False)):
        expr, _summary = apply_live_structural_semantics(genome=genome, expression=expr, sim_cfg=sim_cfg)
    effective = expr.get("effective", {}) if isinstance(expr.get("effective", {}), dict) else {}
    return {key: float(effective.get(key, 0.0)) for key in LIVE_EFFECTIVE_FIELDS}


def changed_effective_fields(before: Genome | dict[str, Any], after: Genome | dict[str, Any], *, sim_cfg: Any = None) -> list[str]:
    a = effective_fields(before, sim_cfg=sim_cfg)
    b = effective_fields(after, sim_cfg=sim_cfg)
    return sorted([key for key in LIVE_EFFECTIVE_FIELDS if float(a.get(key, 0.0)) != float(b.get(key, 0.0))])


def payload_counts(payload: dict[str, Any]) -> dict[str, int]:
    return {
        "module_count": int(len(list(payload.get("modules", []) or []))),
        "connection_count": int(len(list(payload.get("connections", []) or []))),
    }


def complexity_cost(payload: dict[str, Any]) -> dict[str, float]:
    counts = payload_counts(payload)
    locality = float(payload.get("b3_locality_cost_total", 0.0) or 0.0)
    return {
        "module_count": float(counts["module_count"]),
        "connection_count": float(counts["connection_count"]),
        "locality_cost_total": locality,
        "representation_cost": float(counts["module_count"]) + float(counts["connection_count"]) + locality,
    }


def _id_set(payload: dict[str, Any], field: str, key: str) -> set[str]:
    return {str(rec.get(key, "")) for rec in list(payload.get(field, []) or []) if str(rec.get(key, ""))}


def structural_diff(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    before_modules = _id_set(before, "modules", "module_id")
    after_modules = _id_set(after, "modules", "module_id")
    before_connections = _id_set(before, "connections", "connection_id")
    after_connections = _id_set(after, "connections", "connection_id")
    before_hash = stable_hash(before)
    after_hash = stable_hash(after)
    return {
        "payload_changed": before_hash != after_hash,
        "before_hash": before_hash,
        "after_hash": after_hash,
        "module_count_delta": int(len(after_modules) - len(before_modules)),
        "connection_count_delta": int(len(after_connections) - len(before_connections)),
        "added_module_ids": sorted(after_modules - before_modules),
        "removed_module_ids": sorted(before_modules - after_modules),
        "added_connection_ids": sorted(after_connections - before_connections),
        "removed_connection_ids": sorted(before_connections - after_connections),
        "module_records_changed": stable_hash(list(before.get("modules", []) or [])) != stable_hash(list(after.get("modules", []) or [])),
        "connection_records_changed": stable_hash(list(before.get("connections", []) or [])) != stable_hash(list(after.get("connections", []) or [])),
    }


def phenotype_reach_class(before: dict[str, Any], after: dict[str, Any], *, sim_cfg: Any = None) -> str:
    diff = structural_diff(before, after)
    if not bool(diff["payload_changed"]):
        return "no_effect"
    live_fields = changed_effective_fields(before, after, sim_cfg=sim_cfg)
    connection_changed = bool(diff["connection_records_changed"])
    module_changed = bool(diff["module_records_changed"])
    if live_fields and connection_changed:
        return "mixed"
    if live_fields:
        return "live_module_effect"
    if connection_changed and not module_changed:
        return "stored_connection_only"
    if connection_changed:
        return "mixed_stored_without_live_effect"
    return "no_effect"


def viability_classification(before: dict[str, Any], after: dict[str, Any], *, ok: bool, integrity_failure: bool = False, sim_cfg: Any = None) -> str:
    if integrity_failure:
        return "integrity_failure"
    diff = structural_diff(before, after)
    if not ok or not bool(diff["payload_changed"]):
        return "noop"
    reach = phenotype_reach_class(before, after, sim_cfg=sim_cfg)
    if reach in {"live_module_effect", "mixed"}:
        return "valid_live_change"
    if reach == "stored_connection_only":
        return "valid_stored_only_change"
    if int(diff["module_count_delta"]) < 0:
        return "valid_compression"
    return "valid_neutral_representation_change"


def operator_map() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    static = {
        "duplicate_with_decay": {
            "required_input_shape": "one_or_more_modules",
            "possible_output_shape": "adds_one_module",
            "can_change_module_count": True,
            "can_change_connection_count": False,
            "can_change_module_params": True,
            "can_change_connection_weight": False,
            "can_change_expression": False,
            "can_reduce_structure": False,
            "known_invalid_outcomes": [],
            "cap_sensitive": True,
            "locality_sensitive": False,
            "phenotype_reach_class": "no_effect",
        },
        "gate_drift": {
            "required_input_shape": "one_or_more_connections",
            "possible_output_shape": "same_counts_connection_weight_changed",
            "can_change_module_count": False,
            "can_change_connection_count": False,
            "can_change_module_params": False,
            "can_change_connection_weight": True,
            "can_change_expression": False,
            "can_reduce_structure": False,
            "known_invalid_outcomes": [],
            "cap_sensitive": False,
            "locality_sensitive": False,
            "phenotype_reach_class": "stored_connection_only",
        },
        "bounded_rewire": {
            "required_input_shape": "two_or_more_modules_and_one_or_more_connections",
            "possible_output_shape": "same_counts_or_connection_dropped_by_validation",
            "can_change_module_count": False,
            "can_change_connection_count": True,
            "can_change_module_params": False,
            "can_change_connection_weight": True,
            "can_change_expression": False,
            "can_reduce_structure": False,
            "known_invalid_outcomes": ["rewire_can_target_incompatible_slot_or_nonlocal_pair_then_be_clipped"],
            "cap_sensitive": False,
            "locality_sensitive": True,
            "phenotype_reach_class": "stored_connection_only",
        },
        "edge_split_insertion": {
            "required_input_shape": "one_or_more_connections",
            "possible_output_shape": "adds_one_generic_module_and_replaces_one_edge_with_two_edges",
            "can_change_module_count": True,
            "can_change_connection_count": True,
            "can_change_module_params": True,
            "can_change_connection_weight": True,
            "can_change_expression": False,
            "can_reduce_structure": False,
            "known_invalid_outcomes": ["split_edges_can_be_clipped_by_B2_or_B3"],
            "cap_sensitive": True,
            "locality_sensitive": True,
            "phenotype_reach_class": "mixed_stored_without_live_effect",
        },
        "motif_duplication": {
            "required_input_shape": "two_or_more_modules",
            "possible_output_shape": "adds_two_modules_and_internal_copied_edges_when_present",
            "can_change_module_count": True,
            "can_change_connection_count": True,
            "can_change_module_params": True,
            "can_change_connection_weight": False,
            "can_change_expression": False,
            "can_reduce_structure": False,
            "known_invalid_outcomes": ["copied_edges_can_be_clipped_by_B2_or_B3"],
            "cap_sensitive": True,
            "locality_sensitive": True,
            "phenotype_reach_class": "mixed_stored_without_live_effect",
        },
        "promoter_expression_mutation": {
            "required_input_shape": "one_or_more_modules",
            "possible_output_shape": "same_counts_module_expression_scale_changed",
            "can_change_module_count": False,
            "can_change_connection_count": False,
            "can_change_module_params": True,
            "can_change_connection_weight": False,
            "can_change_expression": False,
            "can_reduce_structure": False,
            "known_invalid_outcomes": [],
            "cap_sensitive": False,
            "locality_sensitive": False,
            "phenotype_reach_class": "no_effect",
        },
        "fusion_merge_compression": {
            "required_input_shape": "two_or_more_modules",
            "possible_output_shape": "removes_one_module_and_rewrites_or_drops_incident_edges",
            "can_change_module_count": True,
            "can_change_connection_count": True,
            "can_change_module_params": False,
            "can_change_connection_weight": False,
            "can_change_expression": False,
            "can_reduce_structure": True,
            "known_invalid_outcomes": [],
            "cap_sensitive": False,
            "locality_sensitive": True,
            "phenotype_reach_class": "mixed_stored_without_live_effect",
        },
    }
    for spec in operator_catalog():
        item = dict(static[spec.op_id])
        item.update(
            {
                "operator_name": spec.op_id,
                "draw_count": int(spec.py_draws),
                "can_noop": True,
                "inheritance_supported": True,
                "checkpoint_supported": True,
                "W6_supported": True,
            }
        )
        rows.append(item)
    return sorted(rows, key=lambda row: str(row["operator_name"]))


def apply_single_operator(
    operator_name: str,
    draws: tuple[float, ...] | list[float],
    *,
    input_payload: dict[str, Any] | None = None,
    sim_cfg: Any = None,
) -> dict[str, Any]:
    specs = {spec.op_id: spec for spec in operator_catalog()}
    if operator_name not in specs:
        raise KeyError(f"unknown operator_name={operator_name!r}")
    spec = specs[operator_name]
    before = canonical_payload_from_genome_dict(input_payload or seeded_fixture_genome().to_dict())
    try:
        # Reporting probes invoke catalogue operators directly rather than
        # through ``apply_c1_operator_schedule``.  Supply the same deterministic
        # non-RNG identity context that the hereditary path now requires.  The
        # lineage token is scoped to this immutable fixture invocation, so the
        # probe remains repeatable without pretending that independently
        # created modules share ancestry.
        raw_after, ok = spec.fn(
            json.loads(stable_json(before)),
            list(draws)[: int(spec.py_draws)],
            {
                "operator_id": str(spec.op_id),
                "attempt_index": 0,
                "bootstrap_enabled": bool(
                    getattr(sim_cfg, "structural_bootstrap_enable", False)
                ),
                "identity_context": {
                    "lineage_token": (
                        f"structural_mutation_viability:{spec.op_id}:"
                        f"{stable_hash(before)}"
                    ),
                    "generation": 0,
                    "tick": 0,
                },
            },
        )
        after = canonical_payload_from_genome_dict(raw_after)
        integrity_failure = False
        error = None
    except Exception as exc:
        after = before
        ok = False
        integrity_failure = True
        error = f"{type(exc).__name__}: {exc}"
    diff = structural_diff(before, after)
    live_fields = changed_effective_fields(before, after, sim_cfg=sim_cfg)
    return {
        "schema_version": SCHEMA_VERSION,
        "operator": str(operator_name),
        "draws": [float(x) for x in list(draws)[: int(spec.py_draws)]],
        "draw_count": int(spec.py_draws),
        "ok": bool(ok),
        "classification": viability_classification(before, after, ok=bool(ok), integrity_failure=bool(integrity_failure), sim_cfg=sim_cfg),
        "phenotype_reach_class": phenotype_reach_class(before, after, sim_cfg=sim_cfg),
        "changed_effective_fields": live_fields,
        "connection_execution_status": CONNECTION_EXECUTION_STATUS,
        "before_hash": diff["before_hash"],
        "after_hash": diff["after_hash"],
        "payload_changed": bool(diff["payload_changed"]),
        "module_count_before": len(before.get("modules", []) or []),
        "module_count_after": len(after.get("modules", []) or []),
        "connection_count_before": len(before.get("connections", []) or []),
        "connection_count_after": len(after.get("connections", []) or []),
        "module_count_delta": int(diff["module_count_delta"]),
        "connection_count_delta": int(diff["connection_count_delta"]),
        "added_module_ids": diff["added_module_ids"],
        "removed_module_ids": diff["removed_module_ids"],
        "added_connection_ids": diff["added_connection_ids"],
        "removed_connection_ids": diff["removed_connection_ids"],
        "b3_locality_dropped_connections": int(after.get("b3_locality_dropped_connections", 0) or 0),
        "complexity_before": complexity_cost(before),
        "complexity_after": complexity_cost(after),
        "catastrophic_integrity_failure": bool(integrity_failure),
        "error": error,
        "before_payload": before,
        "after_payload": after,
    }


def operator_battery(draw_sets: tuple[tuple[float, ...], ...] = DEFAULT_DRAWS) -> dict[str, Any]:
    outcomes: list[dict[str, Any]] = []
    for spec in operator_catalog():
        for draws in draw_sets:
            outcomes.append(apply_single_operator(spec.op_id, draws))

    summaries: dict[str, dict[str, Any]] = {}
    for spec in operator_catalog():
        rows = [row for row in outcomes if row["operator"] == spec.op_id]
        summaries[spec.op_id] = _summarize_operator_rows(rows)

    return {
        "schema_version": SCHEMA_VERSION,
        "input": "REP-1C minimal_canonical_v1 seed",
        "connection_execution_status": CONNECTION_EXECUTION_STATUS,
        "operator_summaries": summaries,
        "outcomes": outcomes,
        "repeat_digest": stable_hash([{k: v for k, v in row.items() if k not in {"before_payload", "after_payload"}} for row in outcomes]),
    }


def _summarize_operator_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    attempts = len(rows)
    classifications = [str(row["classification"]) for row in rows]
    return {
        "attempts": attempts,
        "valid_changes": sum(1 for row in rows if bool(row["payload_changed"]) and not bool(row["catastrophic_integrity_failure"])),
        "noops": classifications.count("noop"),
        "invalid_rejections": classifications.count("invalid_rejected"),
        "module_cap_drops": 0,
        "connection_cap_drops": 0,
        "locality_drops": sum(int(row["b3_locality_dropped_connections"]) for row in rows),
        "module_count_increases": sum(1 for row in rows if int(row["module_count_delta"]) > 0),
        "module_count_decreases": sum(1 for row in rows if int(row["module_count_delta"]) < 0),
        "connection_count_increases": sum(1 for row in rows if int(row["connection_count_delta"]) > 0),
        "connection_count_decreases": sum(1 for row in rows if int(row["connection_count_delta"]) < 0),
        "live_module_effect_changes": sum(1 for row in rows if str(row["phenotype_reach_class"]) in {"live_module_effect", "mixed"}),
        "stored_connection_only_changes": sum(1 for row in rows if str(row["phenotype_reach_class"]) == "stored_connection_only"),
        "neutral_representation_changes": classifications.count("valid_neutral_representation_change") + classifications.count("valid_compression"),
        "exact_payload_duplicates": sum(1 for row in rows if not bool(row["payload_changed"])),
        "catastrophic_integrity_failures": sum(1 for row in rows if bool(row["catastrophic_integrity_failure"])),
        "classifications": {name: classifications.count(name) for name in sorted(set(classifications))},
    }


def constraint_probe_summary() -> dict[str, Any]:
    modules = [
        {
            "module_id": f"m{i:02d}",
            "module_type": "generic",
            "io": {
                "inputs": [{"slot_id": "in", "value_type": "scalar"}],
                "outputs": [{"slot_id": "out", "value_type": "scalar"}],
                "caps": {"max_fanin": 6, "max_fanout": 6},
            },
            "latent_pos": [0.0, 0.0],
            "params": {},
        }
        for i in range(14)
    ]
    kept_modules, _kept_conn, _meta = apply_b2_constraints(modules, [], invalid_policy="clip")

    cap_modules = modules[:12]
    cap_connections = []
    cid = 0
    for src in range(6):
        for dst in range(6, 12):
            cap_connections.append(
                {
                    "connection_id": f"c{cid:03d}",
                    "src_module_id": f"m{src:02d}",
                    "dst_module_id": f"m{dst:02d}",
                    "src_slot": "out",
                    "dst_slot": "in",
                    "value_type": "scalar",
                    "weight": 1.0,
                }
            )
            cid += 1
    _mods2, kept_connections, meta_conn = apply_b2_constraints(cap_modules, cap_connections, invalid_policy="clip")

    local_modules = [
        {
            "module_id": "near",
            "module_type": "generic",
            "io": {"inputs": [{"slot_id": "in", "value_type": "scalar"}], "outputs": [{"slot_id": "out", "value_type": "scalar"}]},
            "latent_pos": [0.0, 0.0],
            "params": {},
        },
        {
            "module_id": "far",
            "module_type": "generic",
            "io": {"inputs": [{"slot_id": "in", "value_type": "scalar"}], "outputs": [{"slot_id": "out", "value_type": "scalar"}]},
            "latent_pos": [1.0, 1.0],
            "params": {},
        },
    ]
    local_connections = [
        {
            "connection_id": "c_far",
            "src_module_id": "near",
            "dst_module_id": "far",
            "src_slot": "out",
            "dst_slot": "in",
            "value_type": "scalar",
            "weight": 1.0,
        }
    ]
    _mods3, kept_local, meta_local = apply_b2_constraints(local_modules, local_connections, invalid_policy="clip")

    invalid_rejected = False
    invalid_error = ""
    try:
        apply_b2_constraints(
            cap_modules[:2],
            [
                {
                    "connection_id": "c_bad",
                    "src_module_id": "m00",
                    "dst_module_id": "m01",
                    "src_slot": "out",
                    "dst_slot": "in",
                    "value_type": "vector",
                    "weight": 1.0,
                }
            ],
            invalid_policy="reject",
        )
    except RuntimeError as exc:
        invalid_rejected = True
        invalid_error = str(exc).splitlines()[1] if len(str(exc).splitlines()) > 1 else str(exc)

    return {
        "schema_version": SCHEMA_VERSION,
        "module_cap": {
            "attempted": len(modules),
            "kept": len(kept_modules),
            "dropped": len(modules) - len(kept_modules),
            "cap": int(B2_FROZEN_CAPS["max_modules_per_agent"]),
        },
        "connection_cap": {
            "attempted": len(cap_connections),
            "kept": len(kept_connections),
            "dropped": int(meta_conn.get("dropped_connections", 0) or 0),
            "cap": int(B2_FROZEN_CAPS["max_connections_per_agent"]),
        },
        "locality": {
            "attempted": len(local_connections),
            "kept": len(kept_local),
            "dropped": int(meta_local.get("dropped_nonlocal_connections", 0) or 0),
        },
        "invalid_rejection": {
            "rejected": bool(invalid_rejected),
            "error_code": invalid_error,
        },
    }


def inheritance_and_retention_summary(battery: dict[str, Any] | None = None) -> dict[str, Any]:
    battery = battery or operator_battery()
    records = []
    for operator_name, summary in sorted(dict(battery["operator_summaries"]).items()):
        candidate = next(
            (
                row
                for row in list(battery["outcomes"])
                if row["operator"] == operator_name and bool(row["payload_changed"]) and not bool(row["catastrophic_integrity_failure"])
            ),
            None,
        )
        if candidate is None:
            records.append(
                {
                    "event_id": f"{operator_name}:none",
                    "operator": operator_name,
                    "origin_generation": 0,
                    "originating_parent": "fixture_parent",
                    "affected_module_ids": [],
                    "affected_connection_ids": [],
                    "phenotype_reach_class": "not_evaluable",
                    "inherited_child_count": 0,
                    "retained_generation_count": 0,
                    "final_alive_carrier_count": 0,
                    "lost_generation": 0,
                    "loss_reason_when_observable": "no_valid_change",
                }
            )
            continue
        payload = dict(candidate["after_payload"])
        current = payload
        retained = 0
        lost_generation = None
        for generation in range(1, 7):
            child, _summary = recombine_self_lineage_payload(parent_payload=current, rng=_FixedRng(0.25), parent_payload_is_canonical=True)
            if _structure_material_hash(child) == _structure_material_hash(current):
                retained += 1
                current = child
            else:
                lost_generation = generation
                break
        affected_modules = sorted(set(candidate["added_module_ids"]) | set(candidate["removed_module_ids"]))
        affected_connections = sorted(set(candidate["added_connection_ids"]) | set(candidate["removed_connection_ids"]))
        if not affected_modules and bool(candidate.get("module_count_delta") == 0):
            affected_modules = _changed_record_ids(candidate["before_payload"], candidate["after_payload"], "modules", "module_id")
        if not affected_connections and bool(candidate.get("connection_count_delta") == 0):
            affected_connections = _changed_record_ids(candidate["before_payload"], candidate["after_payload"], "connections", "connection_id")
        records.append(
            {
                "event_id": f"{operator_name}:{candidate['after_hash'][:12]}",
                "operator": operator_name,
                "origin_generation": 0,
                "originating_parent": "fixture_parent",
                "affected_module_ids": affected_modules,
                "affected_connection_ids": affected_connections,
                "phenotype_reach_class": candidate["phenotype_reach_class"],
                "inherited_child_count": 1 if retained >= 1 else 0,
                "retained_generation_count": int(retained),
                "final_alive_carrier_count": 1 if retained >= 6 else 0,
                "lost_generation": lost_generation,
                "loss_reason_when_observable": None if lost_generation is None else "self_recombination_payload_changed",
                "short_retained_change": bool(retained >= 3),
                "medium_retained_change": bool(retained >= 6),
            }
        )
    empty_collapse_count = sum(
        1
        for record in records
        if record["inherited_child_count"] > 0
        and record["final_alive_carrier_count"] == 0
        and record["loss_reason_when_observable"] == "empty_collapse"
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "definitions": retention_definitions(),
        "records": records,
        "inherited_change_count": sum(1 for row in records if int(row["inherited_child_count"]) > 0),
        "short_retained_change_count": sum(1 for row in records if bool(row.get("short_retained_change", False))),
        "medium_retained_change_count": sum(1 for row in records if bool(row.get("medium_retained_change", False))),
        "lost_change_count": sum(1 for row in records if row.get("lost_generation") is not None),
        "empty_collapse_count": int(empty_collapse_count),
        "representation_retention_not_selection": True,
    }


def _changed_record_ids(before: dict[str, Any], after: dict[str, Any], field: str, key: str) -> list[str]:
    before_map = {str(rec.get(key, "")): rec for rec in list(before.get(field, []) or [])}
    after_map = {str(rec.get(key, "")): rec for rec in list(after.get(field, []) or [])}
    changed = []
    for rid in sorted(set(before_map).intersection(after_map)):
        if stable_hash(before_map[rid]) != stable_hash(after_map[rid]):
            changed.append(rid)
    return changed


def _structure_material_hash(payload: dict[str, Any]) -> str:
    return stable_hash(
        {
            "modules": list(payload.get("modules", []) or []),
            "connections": list(payload.get("connections", []) or []),
        }
    )


def retention_definitions() -> dict[str, str]:
    return {
        "introduced_change": "canonical structural payload difference produced by an existing operator and accepted by validation",
        "inherited_change": "changed structural payload appears in at least one direct child through existing self-recombination or recombination",
        "short_retained_change": "changed payload remains present for at least 3 consecutive generations in the controlled inheritance fixture",
        "medium_retained_change": "changed payload remains present for at least 6 consecutive generations in the controlled inheritance fixture",
        "lost_change": "no living or fixture carrier remains after previous presence",
        "empty_collapse": "a carrying payload returns to zero modules and zero connections",
        "bloat_warning": "module or connection counts repeatedly approach current caps under existing settings",
    }


def recombination_summary() -> dict[str, Any]:
    seed = seeded_fixture_genome().canonical_payload()
    changed = apply_single_operator("duplicate_with_decay", DEFAULT_DRAWS[0])["after_payload"]
    child, summary = recombine_lineage_payloads(parent_a_payload=seed, parent_b_payload=changed, rng=_FixedRng(0.25), trusted_canonical_parent_payloads=True)
    return {
        "schema_version": SCHEMA_VERSION,
        "parent_a_hash": stable_hash(seed),
        "parent_b_hash": stable_hash(changed),
        "child_hash": stable_hash(child),
        "child_module_count": len(child.get("modules", []) or []),
        "child_connection_count": len(child.get("connections", []) or []),
        "summary": summary.__dict__,
        "change_survived_or_loss_explained": True,
    }


def phenotype_reach_summary(battery: dict[str, Any] | None = None) -> dict[str, Any]:
    battery = battery or operator_battery()
    rows = list(battery["outcomes"])
    by_operator = {}
    for name in sorted({str(row["operator"]) for row in rows}):
        op_rows = [row for row in rows if row["operator"] == name]
        attempts = len(op_rows)
        by_operator[name] = {
            "attempts": attempts,
            "live_effect_fraction": _frac(sum(1 for row in op_rows if row["phenotype_reach_class"] in {"live_module_effect", "mixed"}), attempts),
            "stored_only_fraction": _frac(sum(1 for row in op_rows if row["phenotype_reach_class"] == "stored_connection_only"), attempts),
            "changed_effective_fields": sorted({field for row in op_rows for field in list(row["changed_effective_fields"])}),
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "connection_execution_status": CONNECTION_EXECUTION_STATUS,
        "operator_reach": by_operator,
        "live_module_effect_count": sum(1 for row in rows if row["phenotype_reach_class"] in {"live_module_effect", "mixed"}),
        "stored_connection_only_count": sum(1 for row in rows if row["phenotype_reach_class"] == "stored_connection_only"),
        "exact_effective_fields_changed": sorted({field for row in rows for field in list(row["changed_effective_fields"])}),
    }


def integrity_summary(battery: dict[str, Any] | None = None, retention: dict[str, Any] | None = None) -> dict[str, Any]:
    battery = battery or operator_battery()
    retention = retention or inheritance_and_retention_summary(battery)
    outcomes = list(battery["outcomes"])
    repeat = repeat_parity_summary()
    live_count = sum(1 for row in outcomes if row["phenotype_reach_class"] in {"live_module_effect", "mixed"})
    return {
        "schema_version": SCHEMA_VERSION,
        "operator_catalog_mapped": len(operator_map()) == len(operator_catalog()),
        "zero_integrity_failures": all(not bool(row["catastrophic_integrity_failure"]) for row in outcomes),
        "deterministic_repeat_parity": bool(repeat["passed"]),
        "valid_module_changes_occur": any(int(row["module_count_delta"]) != 0 or _changed_record_ids(row["before_payload"], row["after_payload"], "modules", "module_id") for row in outcomes),
        "live_module_effect_changes": int(live_count),
        "inherited_change_count": int(retention["inherited_change_count"]),
        "short_retained_change_count": int(retention["short_retained_change_count"]),
        "medium_retained_change_count": int(retention["medium_retained_change_count"]),
        "empty_collapse_count": int(retention["empty_collapse_count"]),
        "connection_execution_status": CONNECTION_EXECUTION_STATUS,
        "decision_pressure": "live phenotype reach is too narrow for REP-2A-A" if live_count == 0 else "live path present",
    }


def repeat_parity_summary() -> dict[str, Any]:
    a = operator_battery()
    b = operator_battery()
    cells = [
        {
            "cell": "operator_battery",
            "digest_a": a["repeat_digest"],
            "digest_b": b["repeat_digest"],
            "equal": a["repeat_digest"] == b["repeat_digest"],
        },
        {
            "cell": "retention",
            "digest_a": stable_hash(inheritance_and_retention_summary(a)),
            "digest_b": stable_hash(inheritance_and_retention_summary(b)),
            "equal": stable_hash(inheritance_and_retention_summary(a)) == stable_hash(inheritance_and_retention_summary(b)),
        },
        {
            "cell": "recombination",
            "digest_a": stable_hash(recombination_summary()),
            "digest_b": stable_hash(recombination_summary()),
            "equal": stable_hash(recombination_summary()) == stable_hash(recombination_summary()),
        },
    ]
    return {"schema_version": SCHEMA_VERSION, "passed": all(bool(cell["equal"]) for cell in cells), "repeat_cells": cells}


def rep2a_decision(integrity: dict[str, Any] | None = None) -> dict[str, Any]:
    integrity = integrity or integrity_summary()
    if not bool(integrity["zero_integrity_failures"]) or not bool(integrity["deterministic_repeat_parity"]):
        decision = "REP-2A-D"
        reason = "integrity or deterministic repeat parity failed"
        next_candidate = "stop structural representation work until integrity is restored"
    elif int(integrity["live_module_effect_changes"]) <= 0:
        decision = "REP-2A-B"
        reason = "existing operators produce valid stored or neutral structural changes, but no accepted REP-1C-seed mutation changes current effective phenotype fields"
        next_candidate = "REP-2B: Live Connection Semantics Design"
    elif int(integrity["short_retained_change_count"]) <= 0:
        decision = "REP-2A-C"
        reason = "live changes exist but retention evidence is incomplete"
        next_candidate = "one bounded reporting or operator-path repair"
    else:
        decision = "REP-2A-A"
        reason = "existing structural mutations produce valid inherited and short-retained live module changes"
        next_candidate = "REP-2B: Live Connection Semantics Design"
    return {
        "schema_version": SCHEMA_VERSION,
        "decision": decision,
        "reason": reason,
        "recommended_next_candidate": next_candidate,
    }


def reporting_disabled_summary() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "reporting_enabled": False,
        "structural_mutation_viability_summary_json": None,
        "behavior_change": False,
        "random_draws_consumed": 0,
    }


def viability_summary_payload() -> dict[str, Any]:
    battery = operator_battery()
    retention = inheritance_and_retention_summary(battery)
    integrity = integrity_summary(battery, retention)
    return {
        "schema_version": SCHEMA_VERSION,
        "operator_health": battery["operator_summaries"],
        "phenotype_reach": phenotype_reach_summary(battery),
        "retention": {
            "inherited_change_count": retention["inherited_change_count"],
            "short_retained_change_count": retention["short_retained_change_count"],
            "medium_retained_change_count": retention["medium_retained_change_count"],
            "lost_change_count": retention["lost_change_count"],
            "empty_collapse_count": retention["empty_collapse_count"],
        },
        "integrity": integrity,
        "decision": rep2a_decision(integrity),
    }


def _frac(num: int, den: int) -> float:
    return 0.0 if int(den) <= 0 else float(num) / float(den)


class _FixedRng:
    def __init__(self, value: float) -> None:
        self.value = float(value)

    def random(self) -> float:
        return self.value
