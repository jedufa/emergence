from __future__ import annotations

"""Reporting-only structural variant identity helpers for REP-2L."""

import hashlib
import json
import math
from typing import Any

from agents.modular_genome_v1 import canonical_payload_from_genome_dict
from evolution.structural_live_semantics import (
    PROFILE_CONNECTION_FIELD_COMPILER,
    compile_live_graph,
)


SCHEMA_VERSION = "structural_variant_identity_v1"
SEMANTIC_SCHEMA_VERSION = "structural_graph_semantic_identity_v1"
FULL_SCHEMA_VERSION = "structural_graph_identity_v1"
HISTORY_SCHEMA_VERSION = "structural_variant_history_v1"
EVENT_ID_SCHEMA_VERSION = "structural_variant_event_identity_v1"
PARENT_CHILD_EVENT_SCHEMA_VERSION = "structural_parent_child_variant_event_v2"
MUTATION_DELTA_SCHEMA_VERSION = "structural_mutation_delta_event_v1"

C1_OPERATOR_FIELDS = (
    "bounded_rewire",
    "duplicate_with_decay",
    "edge_split_insertion",
    "fusion_merge_compression",
    "gate_drift",
    "motif_duplication",
    "promoter_expression_mutation",
)
C2_FIELDS = (
    "c2_recombination_version",
    "c2_alignment_key",
    "c2_unmatched_policy",
    "c2_matched_modules",
    "c2_unmatched_modules_parent_a",
    "c2_unmatched_modules_parent_b",
    "c2_matched_connections",
    "c2_unmatched_connections_parent_a",
    "c2_unmatched_connections_parent_b",
    "c2_budget_draws",
)


def enabled(sim_cfg: Any) -> bool:
    return bool(getattr(sim_cfg, "structural_variant_identity_reporting_enable", False)) if sim_cfg is not None else False


def stable_json(payload: Any) -> str:
    return json.dumps(_canon(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def stable_hash(payload: Any) -> str:
    return hashlib.sha256(stable_json(payload).encode("utf-8")).hexdigest()


def global_event_id(
    *,
    run_id: str,
    local_event_id: str,
    event_kind: str,
    generation: int | None = None,
    tick: int | None = None,
    event_sequence: int | None = None,
) -> str:
    return stable_hash(
        {
            "schema_version": EVENT_ID_SCHEMA_VERSION,
            "run_id": str(run_id or ""),
            "local_event_id": str(local_event_id or ""),
            "event_kind": str(event_kind or ""),
            "generation": None if generation is None else int(generation),
            "tick": None if tick is None else int(tick),
            "event_sequence": None if event_sequence is None else int(event_sequence),
        }
    )


def payload_from_genome(genome: Any) -> dict[str, Any]:
    if genome is None:
        return canonical_payload_from_genome_dict({})
    if hasattr(genome, "canonical_payload"):
        return canonical_payload_from_genome_dict(dict(genome.canonical_payload()))
    if hasattr(genome, "to_dict"):
        return canonical_payload_from_genome_dict(dict(genome.to_dict()))
    if isinstance(genome, dict):
        return canonical_payload_from_genome_dict(dict(genome))
    return canonical_payload_from_genome_dict({})


def semantic_identity_payload(payload_or_genome: Any, *, profile: str = PROFILE_CONNECTION_FIELD_COMPILER) -> dict[str, Any]:
    payload = payload_from_genome(payload_or_genome)
    compiled = compile_live_graph(payload, profile=profile)
    return {
        "schema_version": SEMANTIC_SCHEMA_VERSION,
        "structural_schema_version": str(payload.get("schema_version", "")),
        "interface_schema_version": str(payload.get("interface_schema_version", "")),
        "locality_schema_version": str(payload.get("b3_locality_schema_version", "")),
        "b2_invalid_policy": str(payload.get("b2_invalid_policy", "")),
        "b2_caps": payload.get("b2_caps", {}),
        "b3_locality": {
            "max_connection_distance": payload.get("b3_locality_max_connection_distance"),
            "distance_cost_scale": payload.get("b3_distance_cost_scale"),
            "distance_cost_max": payload.get("b3_distance_cost_max"),
        },
        "live_semantics_profile": str(profile),
        "graph_supported": bool(compiled.get("graph_supported", False)),
        "unsupported_reason": str(compiled.get("unsupported_reason", "")),
        "cycle_status": str(compiled.get("cycle_status", "")),
        "modules": sorted(
            [_semantic_module_record(row) for row in list(payload.get("modules", []) or [])],
            key=lambda row: str(row.get("module_id", "")),
        ),
        "connections": sorted(
            [_semantic_connection_record(row) for row in list(payload.get("connections", []) or [])],
            key=lambda row: str(row.get("connection_id", "")),
        ),
        "developmental_program": payload.get("developmental_program", None),
    }


def full_identity_payload(payload_or_genome: Any, *, profile: str = PROFILE_CONNECTION_FIELD_COMPILER) -> dict[str, Any]:
    payload = payload_from_genome(payload_or_genome)
    compiled = compile_live_graph(payload, profile=profile)
    genes = dict(payload.get("genes", {}) or {}) if isinstance(payload.get("genes", {}), dict) else {}
    return {
        "schema_version": FULL_SCHEMA_VERSION,
        "semantic_identity": semantic_identity_payload(payload, profile=profile),
        "history_metadata": {
            "genes_c1": {key: genes.get(key, "") for key in sorted(_c1_gene_keys())},
            "genes_c2": {key: genes.get(key, "") for key in C2_FIELDS},
        },
        "compiled_graph_digest": str(compiled.get("graph_digest", "") or ""),
    }


def graph_semantic_digest(payload_or_genome: Any, *, profile: str = PROFILE_CONNECTION_FIELD_COMPILER) -> str:
    return stable_hash(semantic_identity_payload(payload_or_genome, profile=profile))


def graph_identity_digest(payload_or_genome: Any, *, profile: str = PROFILE_CONNECTION_FIELD_COMPILER) -> str:
    return stable_hash(full_identity_payload(payload_or_genome, profile=profile))


def graph_identity_summary(payload_or_genome: Any, *, profile: str = PROFILE_CONNECTION_FIELD_COMPILER) -> dict[str, Any]:
    payload = payload_from_genome(payload_or_genome)
    compiled = compile_live_graph(payload, profile=profile)
    return {
        "schema_version": SCHEMA_VERSION,
        "graph_identity_digest": graph_identity_digest(payload, profile=profile),
        "graph_semantic_digest": graph_semantic_digest(payload, profile=profile),
        "compiled_graph_digest": str(compiled.get("graph_digest", "") or ""),
        "graph_supported": bool(compiled.get("graph_supported", False)),
        "unsupported_reason": str(compiled.get("unsupported_reason", "") or ""),
        "operator_origin_set": operator_origin_set(payload),
    }


def mutation_trace_identity(payload: dict[str, Any]) -> dict[str, Any]:
    return graph_identity_summary(payload)


def operator_origin_set(payload_or_genome: Any) -> list[str]:
    payload = payload_from_genome(payload_or_genome)
    genes = dict(payload.get("genes", {}) or {}) if isinstance(payload.get("genes", {}), dict) else {}
    origins: set[str] = set()
    applied = str(genes.get("c1_operator_applied", "") or "")
    for item in applied.split(","):
        op_id = item.strip()
        if op_id:
            origins.add(f"mutation:{op_id}")
    for op_id in C1_OPERATOR_FIELDS:
        if _int(genes.get(f"c1_op_count__{op_id}", 0)) > 0:
            origins.add(f"mutation:{op_id}")
    if _int(genes.get("c2_matched_modules", 0)) > 0 or _int(genes.get("c2_matched_connections", 0)) > 0:
        origins.add("recombination:c2_lineage")
    return sorted(origins)


def c1_attempt_count(payload_or_genome: Any) -> int:
    payload = payload_from_genome(payload_or_genome)
    genes = dict(payload.get("genes", {}) or {}) if isinstance(payload.get("genes", {}), dict) else {}
    return sum(_int(genes.get(f"c1_op_count__{op_id}", 0)) for op_id in C1_OPERATOR_FIELDS)


def c1_noop_count(payload_or_genome: Any) -> int:
    payload = payload_from_genome(payload_or_genome)
    genes = dict(payload.get("genes", {}) or {}) if isinstance(payload.get("genes", {}), dict) else {}
    return _int(genes.get("c1_operator_noop_count", 0))


def agent_identity(
    agent: Any,
    *,
    canonical_semantic_digest: str = "",
    profile: str = PROFILE_CONNECTION_FIELD_COMPILER,
    live_effect_present: bool | None = None,
) -> dict[str, Any]:
    genome = getattr(agent, "genome", agent)
    payload = payload_from_genome(genome)
    compiled = compile_live_graph(payload, profile=profile)
    semantic_digest = graph_semantic_digest(payload, profile=profile)
    full_digest = graph_identity_digest(payload, profile=profile)
    origins = operator_origin_set(payload)
    mutation_origins = [item for item in origins if item.startswith("mutation:")]
    recombination_origins = [item for item in origins if item.startswith("recombination:")]
    canonical = bool(canonical_semantic_digest and semantic_digest == canonical_semantic_digest)
    graph_supported = bool(compiled.get("graph_supported", False))
    live = bool(live_effect_present) if live_effect_present is not None else False
    return {
        "schema_version": SCHEMA_VERSION,
        "agent_id": _int(getattr(agent, "agent_id", -1), -1),
        "graph_identity_digest": full_digest,
        "graph_semantic_digest": semantic_digest,
        "compiled_graph_digest": str(compiled.get("graph_digest", "") or ""),
        "graph_supported": graph_supported,
        "unsupported_reason": str(compiled.get("unsupported_reason", "") or ""),
        "canonical_seed_exact": canonical and not origins,
        "canonical_seed_semantic_match": canonical,
        "operator_origin_available": bool(origins),
        "operator_origin_set": origins,
        "mutation_operator_origin_set": mutation_origins,
        "recombination_origin_set": recombination_origins,
        "c1_attempt_count": c1_attempt_count(payload),
        "c1_noop_count": c1_noop_count(payload),
        "variant_class": classify_variant(
            payload,
            canonical_semantic_digest=canonical_semantic_digest,
            profile=profile,
            live_effect_present=live,
        ),
    }


def birth_identity_context(
    parent: Any,
    mate: Any | None,
    child: Any,
    *,
    canonical_semantic_digest: str = "",
    profile: str = PROFILE_CONNECTION_FIELD_COMPILER,
    run_id: str = "",
    local_birth_event_id: str = "",
    generation: int | None = None,
    tick: int | None = None,
    event_sequence: int | None = None,
) -> dict[str, Any]:
    parent_identity = agent_identity(parent, canonical_semantic_digest=canonical_semantic_digest, profile=profile)
    mate_identity = agent_identity(mate, canonical_semantic_digest=canonical_semantic_digest, profile=profile) if mate is not None else {}
    child_identity = agent_identity(child, canonical_semantic_digest=canonical_semantic_digest, profile=profile)
    child_genome = getattr(child, "genome", child)
    child_post_c2_identity = dict(getattr(child_genome, "_rep2m_child_post_c2_identity", {}) or {})
    child_final_identity = dict(getattr(child_genome, "_rep2m_final_child_identity", {}) or {})
    mutation_delta_events = list(getattr(child_genome, "_rep2m_mutation_delta_events", []) or [])
    parent_semantics = [str(parent_identity.get("graph_semantic_digest", ""))]
    if mate_identity:
        parent_semantics.append(str(mate_identity.get("graph_semantic_digest", "")))
    child_semantic = str(child_identity.get("graph_semantic_digest", ""))
    child_changed_vs_parent = child_semantic not in set(parent_semantics)
    child_origins = list(child_identity.get("operator_origin_set", []) or [])
    mutation_origins = [item for item in child_origins if str(item).startswith("mutation:")]
    recombination_origins = [item for item in child_origins if str(item).startswith("recombination:")]
    if mutation_origins and recombination_origins:
        event_kind = "mutation_after_recombination"
    elif mutation_origins:
        event_kind = "mutation"
    elif recombination_origins:
        event_kind = "recombination"
    else:
        event_kind = "inherited_unchanged"
    event_id = stable_hash(
        {
            "schema_version": HISTORY_SCHEMA_VERSION,
            "parent": parent_identity.get("graph_identity_digest", ""),
            "mate": mate_identity.get("graph_identity_digest", ""),
            "child": child_identity.get("graph_identity_digest", ""),
            "kind": event_kind,
        }
    )
    return {
        "structural_variant_history_schema_version": HISTORY_SCHEMA_VERSION,
        "structural_parent_child_variant_event_schema_version": PARENT_CHILD_EVENT_SCHEMA_VERSION,
        "structural_variant_event_id": event_id,
        "structural_variant_event_kind": event_kind,
        "local_birth_event_id": str(local_birth_event_id or ""),
        "global_birth_event_id": global_event_id(
            run_id=str(run_id or ""),
            local_event_id=str(local_birth_event_id or ""),
            event_kind="birth_success",
            generation=generation,
            tick=tick,
            event_sequence=event_sequence,
        )
        if str(run_id or "") and str(local_birth_event_id or "")
        else "",
        "parent_graph_identity_digest": parent_identity.get("graph_identity_digest", ""),
        "parent_graph_semantic_digest": parent_identity.get("graph_semantic_digest", ""),
        "parent_graph_supported": int(bool(parent_identity.get("graph_supported", False))),
        "parent_operator_origin_set": ",".join(str(item) for item in list(parent_identity.get("operator_origin_set", []) or [])),
        "mate_graph_identity_digest": mate_identity.get("graph_identity_digest", ""),
        "mate_graph_semantic_digest": mate_identity.get("graph_semantic_digest", ""),
        "mate_graph_supported": int(bool(mate_identity.get("graph_supported", False))) if mate_identity else "",
        "mate_operator_origin_set": ",".join(str(item) for item in list(mate_identity.get("operator_origin_set", []) or [])),
        "child_post_recombination_graph_identity_digest": child_post_c2_identity.get("graph_identity_digest", ""),
        "child_post_recombination_graph_semantic_digest": child_post_c2_identity.get("graph_semantic_digest", ""),
        "child_post_recombination_graph_supported": int(bool(child_post_c2_identity.get("graph_supported", False)))
        if child_post_c2_identity
        else "",
        "child_graph_identity_digest": child_identity.get("graph_identity_digest", ""),
        "child_graph_semantic_digest": child_identity.get("graph_semantic_digest", ""),
        "child_final_graph_identity_digest": child_final_identity.get("graph_identity_digest", child_identity.get("graph_identity_digest", "")),
        "child_final_graph_semantic_digest": child_final_identity.get("graph_semantic_digest", child_identity.get("graph_semantic_digest", "")),
        "attached_newborn_graph_identity_digest": child_identity.get("graph_identity_digest", ""),
        "attached_newborn_graph_semantic_digest": child_identity.get("graph_semantic_digest", ""),
        "attached_newborn_matches_final_child": int(
            bool(
                (not child_final_identity)
                or (
                    child_final_identity.get("graph_identity_digest", "") == child_identity.get("graph_identity_digest", "")
                    and child_final_identity.get("graph_semantic_digest", "") == child_identity.get("graph_semantic_digest", "")
                )
            )
        ),
        "child_variant_class": child_identity.get("variant_class", "not_evaluable"),
        "child_canonical_seed_exact": bool(child_identity.get("canonical_seed_exact", False)),
        "child_canonical_seed_semantic_match": bool(child_identity.get("canonical_seed_semantic_match", False)),
        "child_operator_origin_available": bool(child_identity.get("operator_origin_available", False)),
        "child_operator_origin_set": ",".join(str(item) for item in child_origins),
        "child_mutation_operator_origin_set": ",".join(str(item) for item in mutation_origins),
        "child_recombination_origin_set": ",".join(str(item) for item in recombination_origins),
        "child_c1_attempt_count": int(child_identity.get("c1_attempt_count", 0) or 0),
        "child_c1_noop_count": int(child_identity.get("c1_noop_count", 0) or 0),
        "mutation_delta_event_count": int(len(mutation_delta_events)),
        "mutation_delta_event_ids": ",".join(str(row.get("mutation_attempt_event_id", "")) for row in mutation_delta_events),
        "mutation_delta_statuses": ",".join(str(row.get("operator_status", "")) for row in mutation_delta_events),
        "mutation_delta_events_json": stable_json(mutation_delta_events),
        "mutation_delta_semantic_change_count": int(sum(1 for row in mutation_delta_events if bool(row.get("semantic_changed", False)))),
        "mutation_delta_historical_change_count": int(sum(1 for row in mutation_delta_events if bool(row.get("historical_changed", False)))),
        "mutation_delta_valid_noop_count": int(sum(1 for row in mutation_delta_events if str(row.get("operator_status", "")) == "valid_noop")),
        "mutation_delta_rejected_count": int(sum(1 for row in mutation_delta_events if str(row.get("operator_status", "")) == "rejected")),
        "child_graph_changed_vs_parent": int(bool(child_changed_vs_parent)),
        "child_semantic_graph_changed_vs_parent": int(bool(child_changed_vs_parent)),
        "child_graph_supported": int(bool(child_identity.get("graph_supported", False))),
    }


def classify_variant(
    payload_or_genome: Any,
    *,
    canonical_semantic_digest: str = "",
    profile: str = PROFILE_CONNECTION_FIELD_COMPILER,
    live_effect_present: bool = False,
) -> str:
    payload = payload_from_genome(payload_or_genome)
    compiled = compile_live_graph(payload, profile=profile)
    if not bool(compiled.get("graph_supported", False)):
        return "unsupported_graph"
    semantic_digest = graph_semantic_digest(payload, profile=profile)
    origins = operator_origin_set(payload)
    canonical = bool(canonical_semantic_digest and semantic_digest == canonical_semantic_digest)
    live_suffix = "live" if bool(live_effect_present) else "neutral"
    if canonical and not origins:
        return "canonical_seed_exact"
    if canonical:
        return "canonical_seed_history_only_variant"
    has_mutation = any(str(item).startswith("mutation:") for item in origins)
    has_recombination = any(str(item).startswith("recombination:") for item in origins)
    modules = list(payload.get("modules", []) or [])
    connections = list(payload.get("connections", []) or [])
    expression_changed = any(
        isinstance(row, dict)
        and isinstance(row.get("params", {}), dict)
        and "expression_scale" in dict(row.get("params", {}) or {})
        for row in modules
    )
    if has_mutation and has_recombination:
        return f"mixed_variant_{live_suffix}"
    if has_recombination:
        return f"recombination_variant_{live_suffix}"
    if expression_changed:
        return f"expression_variant_{live_suffix}"
    if len(connections) != 2 or len(modules) != 3:
        return f"topology_variant_{live_suffix}"
    return f"parameter_variant_{live_suffix}"


def _semantic_module_record(row: Any) -> dict[str, Any]:
    rec = dict(row) if isinstance(row, dict) else {}
    return {
        "module_id": str(rec.get("module_id", "")),
        "parent_module_id": str(rec.get("parent_module_id", "")),
        "module_type": str(rec.get("module_type", "")),
        "params": rec.get("params", {}) if isinstance(rec.get("params", {}), dict) else {},
        "io": rec.get("io", {}) if isinstance(rec.get("io", {}), dict) else {},
        "latent_pos": rec.get("latent_pos", []),
    }


def _semantic_connection_record(row: Any) -> dict[str, Any]:
    rec = dict(row) if isinstance(row, dict) else {}
    return {
        "connection_id": str(rec.get("connection_id", "")),
        "src_module_id": str(rec.get("src_module_id", "")),
        "dst_module_id": str(rec.get("dst_module_id", "")),
        "src_slot": str(rec.get("src_slot", "")),
        "dst_slot": str(rec.get("dst_slot", "")),
        "value_type": str(rec.get("value_type", "")),
        "channel": str(rec.get("channel", "")),
        "weight": rec.get("weight", 0.0),
        "gate": rec.get("gate", None),
        "slot_data": rec.get("slot_data", None),
    }


def _c1_gene_keys() -> tuple[str, ...]:
    return (
        "c1_catalog_version",
        "c1_operator_schedule",
        "c1_operator_applied",
        "c1_operator_noop_count",
        "c1_operator_budget_status",
        *(f"c1_op_count__{op_id}" for op_id in C1_OPERATOR_FIELDS),
    )


def _canon(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _canon(value[key]) for key in sorted(value.keys(), key=lambda item: str(item))}
    if isinstance(value, (list, tuple)):
        return [_canon(item) for item in value]
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return 0.0
        x = 0.0 if value == 0.0 else float(value)
        return round(x, 8)
    return str(value)


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)
