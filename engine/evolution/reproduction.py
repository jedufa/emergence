# evolution/reproduction.py
from __future__ import annotations

"""evolution/reproduction.py

Genome + policy reproduction utilities.

This module performs:
- Genome crossover (numeric blend + mutation)
- Policy crossover (MLP weights) for non-GRU agents
- Memory dimension inheritance (discrete ladder)

Identity Pressure genes
----------------------
We include the following genes in crossover + mutation:
- drive_inertia
- drive_noise
- drive_logit_gain
- pred_lr

Cross-episode internal consistency genes
---------------------------------------
We include these genes in crossover (inheritance) to avoid defaulting:
- drive_sp_hunger, drive_sp_pain, drive_sp_fatigue, drive_sp_fear
- exp_prior_strength, exp_prior_noise
- drive_birth_noise

NOTE: We do not add new mutation behavior here; we only ensure proper inheritance.
"""

import random
import copy
import hashlib
import json
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from agents.developmental_program_v1 import canonicalize_developmental_program
from agents.genome import Genome
from agents.modular_genome_v1 import (
    B2_FROZEN_CAPS,
    B2_INTERFACE_SCHEMA_VERSION,
    B3_FROZEN_LOCALITY,
    B3_LOCALITY_SCHEMA_VERSION,
    MODULAR_GENOME_SCHEMA_VERSION,
    canon_float,
    canonical_payload_from_genome_dict,
    resolve_b2_caps,
)
from agents.introspection import (
    implemented_introspection_channel_mask,
    mutate_introspection_channel_mask,
    recombine_introspection_channel_mask,
    sanitize_introspection_channel_mask,
)
from agents.self_regulation import (
    implemented_self_regulation_control_mask,
    mutate_self_regulation_control_mask,
    recombine_self_regulation_control_mask,
    sanitize_self_regulation_control_mask,
)
from agents.self_modification import (
    implemented_self_modification_target_mask,
    mutate_self_modification_target_mask,
    recombine_self_modification_target_mask,
    sanitize_self_modification_target_mask,
)
from agents.directed_inheritance import (
    implemented_directed_inheritance_channel_mask,
    mutate_directed_inheritance_channel_mask,
    recombine_directed_inheritance_channel_mask,
    sanitize_directed_inheritance_channel_mask,
)
from agents.family_head_residuals import (
    FAMILY_HEAD_RESIDUAL_FIELD,
    FAMILY_HEAD_RESIDUAL_MUTATION_OPERATOR_ID,
    canonical_family_head_residual_block,
    family_head_residual_canonical_digest,
    family_head_residual_semantic_digest,
    mutate_family_head_residual_block,
    recombine_family_head_residual_blocks,
)
from agents.resource_handling import (
    RESOURCE_HANDLING_AXIS_FIELD,
    RESOURCE_HANDLING_AXIS_NEUTRAL,
    RESOURCE_HANDLING_MUTATION_SIGMA,
    clamp_resource_handling_axis,
    mutate_resource_handling_axis,
)
from agents.perception import (
    DEFAULT_SENSOR_RADIUS,
    implemented_sensor_channel_mask,
    mutate_sensor_channel_mask,
    mutate_sensor_radius,
    recombine_sensor_channel_mask,
    recombine_sensor_radius,
    sanitize_sensor_channel_mask,
    snap_sensor_radius,
)
from agents.agent_policy import MLPPolicy
from agents.gru_policy import GRUPolicy
from evolution.operators_c1 import (
    C1_FROZEN_INVALID_OPERATOR_POLICY,
    C1_FROZEN_MAX_OPS_PER_OFFSPRING,
    apply_operator_pipeline,
)
from evolution.structural_variant_identity import graph_identity_summary, mutation_trace_identity
from evolution.mutation_weather import (
    attach_mutation_weather_metadata,
    mutation_weather_state,
)
from evolution.inheritance_registry_v1 import (
    active_inheritance_profile,
    canonical_parent_order,
    draw_linkage_parent_context,
)
from evolution.diploid_inheritance_v1 import (
    DIPLOID_INHERITANCE_PROFILE,
    DIPLOID_RNG_BUDGET_BOUNDARY,
    MEIOSIS_DRAW_COUNT,
    MUTATION_SEMANTIC_DRAW_COUNT,
    assemble_diploid_child,
    attach_diploid_genome,
)
from evolution.genome_upgrade_v2 import (
    GENOME_UPGRADE_V2_RNG_BUDGET_BOUNDARY,
    MEIOSIS_DRAW_COUNT as GENOME_UPGRADE_V2_MEIOSIS_DRAW_COUNT,
    MUTATION_SEMANTIC_DRAW_COUNT as GENOME_UPGRADE_V2_MUTATION_DRAW_COUNT,
    GenomeUpgradeV2V1,
    assemble_child_from_rng_v1 as assemble_genome_upgrade_child_from_rng_v1,
    attach_genome_upgrade_v2,
    genome_upgrade_v2_from_genome,
)
from evolution.recombination_c2 import (
    C2_FROZEN_ALIGNMENT_KEY,
    C2_FROZEN_UNMATCHED_POLICY,
    recombine_lineage_payloads,
    recombine_self_lineage_payload,
)


_MEMORY_LADDER = (0, 8, 16, 32, 64)
_MEMORY_INDEX = {v: i for i, v in enumerate(_MEMORY_LADDER)}
RUNNERCORE11_C1_TYPED_GENOME_ROW_SCHEMA_VERSION = "runnercore11_c1_typed_genome_row_v1"


def _stable_json_hash(payload: Any) -> str:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _rep2m_reporting_enabled(sim_cfg: Optional[Any]) -> bool:
    return bool(sim_cfg is not None and getattr(sim_cfg, "structural_variant_event_join_reporting_enable", False))


def _gen1_f1_capture_enabled(sim_cfg: Optional[Any]) -> bool:
    return bool(sim_cfg is not None and getattr(sim_cfg, "gen1_f1_capture_enable", False))


def _attach_gen1_f1_canonical_stages(
    genome: Any,
    sim_cfg: Optional[Any],
    *,
    pre_c1_payload: dict[str, Any],
    post_c1_payload: dict[str, Any],
) -> Any:
    """Attach passive canonical stage snapshots without touching RNG or mechanics."""

    if genome is None or not _gen1_f1_capture_enabled(sim_cfg):
        return genome
    try:
        setattr(
            genome,
            "_gen1_f1_canonical_stages",
            {
                "schema_version": "eiel_gen1_f1_canonical_stages_v1",
                "stage_boundary": "post_c2_pre_c1_to_post_c1",
                "mutation_operator": "c1_operator_pipeline_v1",
                "child_pre_mutation_genome": canonical_payload_from_genome_dict(pre_c1_payload),
                "child_post_mutation_genome": canonical_payload_from_genome_dict(post_c1_payload),
            },
        )
    except Exception:
        # GEN1-F1 is observer-only; a telemetry attachment failure cannot alter birth.
        pass
    return genome


def _rep2m_attach_variant_event_join_metadata(
    genome: Any,
    *,
    post_c2_payload: dict[str, Any] | None,
    final_payload: dict[str, Any] | None,
    trace_records: list[dict[str, Any]] | None,
) -> Any:
    if genome is None:
        return genome
    try:
        post_c2_identity = graph_identity_summary(post_c2_payload or {}) if post_c2_payload is not None else {}
        final_identity = graph_identity_summary(final_payload or {}) if final_payload is not None else {}
    except Exception as exc:
        post_c2_identity = {"error": str(exc)}
        final_identity = {"error": str(exc)}
    events: list[dict[str, Any]] = []
    for idx, row in enumerate(list(trace_records or [])):
        pre_sem = str(row.get("pre_graph_semantic_digest", "") or "")
        post_sem = str(row.get("normalized_graph_semantic_digest", row.get("post_graph_semantic_digest", "")) or "")
        pre_hist = str(row.get("pre_graph_identity_digest", "") or "")
        post_hist = str(row.get("normalized_graph_identity_digest", row.get("post_graph_identity_digest", "")) or "")
        status = str(row.get("operator_status", "") or ("applied" if row.get("ok") else "valid_noop"))
        semantic_changed = bool(pre_sem and post_sem and pre_sem != post_sem)
        historical_changed = bool(pre_hist and post_hist and pre_hist != post_hist)
        if status == "valid_noop" and (semantic_changed or historical_changed):
            status = "applied"
        events.append(
            {
                "schema_version": "structural_mutation_delta_event_v1",
                "attempt_index": int(idx),
                "selected_operator": str(row.get("operator_id", "") or ""),
                "operator_origin_source": "direct_c1_operator_pipeline",
                "pre_graph_identity_digest": pre_hist,
                "post_graph_identity_digest": post_hist,
                "pre_graph_semantic_digest": pre_sem,
                "post_graph_semantic_digest": post_sem,
                "graph_valid_before": int(bool(row.get("pre_graph_supported", False))),
                "graph_valid_after": int(bool(row.get("normalized_graph_supported", row.get("post_graph_supported", False)))),
                "operator_status": status,
                "rejection_or_noop_reason": "operator_returned_no_change" if status == "valid_noop" else "",
                "historical_changed": int(historical_changed),
                "semantic_changed": int(semantic_changed),
                "topology_changed": int(semantic_changed),
                "parameter_changed": int(semantic_changed),
                "expression_changed": int(semantic_changed),
                "origin_metadata_changed": int(historical_changed and not semantic_changed),
                "live_effect_before": "",
                "live_effect_after": "",
                "live_effect_changed": "",
                "rng_draw_delta": int(row.get("draw_count", 0) or 0),
            }
        )
    try:
        setattr(genome, "_rep2m_child_post_c2_identity", post_c2_identity)
        setattr(genome, "_rep2m_final_child_identity", final_identity)
        setattr(genome, "_rep2m_mutation_delta_events", events)
    except Exception:
        pass
    return genome


def _runnercore11_active(sim_cfg: Optional[Any]) -> bool:
    return bool(
        sim_cfg is not None
        and (
            bool(getattr(sim_cfg, "runnercore11_profile_enable", False))
            or bool(getattr(sim_cfg, "runnercore11_candidate_enable", False))
            or bool(getattr(sim_cfg, "runnercore11_native_enable", False))
            or bool(getattr(sim_cfg, "runnercore11_c1_typed_rows_enable", False))
        )
    )


def _runnercore11_stats(sim_cfg: Optional[Any]) -> dict[str, Any] | None:
    if not _runnercore11_active(sim_cfg):
        return None
    stats = getattr(sim_cfg, "_runnercore11_c1_typed_genome_row_stats", None)
    if not isinstance(stats, dict):
        stats = {
            "schema_version": RUNNERCORE11_C1_TYPED_GENOME_ROW_SCHEMA_VERSION,
            "typed_c1_contract": "canonical_c1_payload_projected_to_resident_genome_row",
            "native_cpp_status": "not_attached_python_projection",
            "c1_operator_pipeline_calls": 0,
            "c1_typed_rows_created": 0,
            "c1_typed_rows_attached": 0,
            "policy_rows_attached": 0,
            "hmem_rows_attached": 0,
            "c1_typed_rows_missing_field_fail_closed": 0,
            "c1_typed_rows_unsupported_kind": 0,
            "c1_payload_dict_roundtrips_avoided": 0,
            "genome_from_dict_avoided": 0,
            "genome_from_dict_forced": 0,
            "genome_materialization_avoided": 0,
            "genome_materialization_forced": 0,
            "generation_end_proxy_agents_created": 0,
            "generation_end_proxy_genome_materialization_avoided": 0,
            "policy_object_creation_avoided": 0,
            "policy_object_creation_forced": 0,
            "forced_materialization_policy_source": 0,
            "forced_materialization_policy_bank_sync": 0,
            "forced_materialization_behavior_log": 0,
            "compatibility_export_forced": 0,
            "compatibility_export_avoided": 0,
            "birth_success_count": 0,
            "birth_failure_count": 0,
            "hidden_agent_sync_rows": 0,
            "per_tick_agent_sync_rows": 0,
            "unsupported_fallback_count": 0,
            "runnercore12_c1_payload_digest_reuse_enabled": 0,
            "runnercore12_c1_payload_digest_reuse_rows": 0,
            "runnercore12_c1_payload_digest_reuse_fallbacks": 0,
            "runnercore13_c1_projection_digest_sample_only_enabled": 0,
            "runnercore13_c1_projection_digest_full_rows": 0,
            "runnercore13_c1_projection_digest_skipped_rows": 0,
            "runnercore15_c1_noop_normalize_skip_enabled": 0,
            "runnercore15_c1_noop_normalize_skipped": 0,
            "runnercore16_c1_empty_structure_fastpath_enabled": 0,
            "runnercore16_c1_empty_structure_fastpath_hits": 0,
            "runnercore16_c1_empty_structure_fastpath_misses": 0,
            "runnercore16_c1_empty_structure_fastpath_noop_draws": 0,
            "runnercore17_resident_birth_parent_candidates": 0,
            "runnercore17_resident_birth_parent_specs_used": 0,
            "runnercore17_resident_birth_parent_state_refreshes": 0,
            "runnercore17_resident_birth_parent_materializations_avoided": 0,
            "runnercore17_resident_birth_parent_parent_import_syncs": 0,
            "runnercore17_resident_birth_parent_missing_specs": 0,
            "runnercore17_resident_birth_parent_unsupported_fallback_count": 0,
            "runnercore18_resident_parent_minimal_refresh_enabled": 0,
            "runnercore18_resident_parent_minimal_refreshes": 0,
            "runnercore18_resident_parent_full_refreshes": 0,
            "runnercore18_resident_parent_full_refresh_skipped_no_slot": 0,
            "runnercore18_resident_parent_minimal_import_syncs": 0,
            "runnercore18_resident_parent_unsupported_fallback_count": 0,
            "forced_materialization_reasons": {},
            "c1_projection_trace_rows": 0,
            "c1_projection_trace_sample": [],
            "required_field_failures": {},
        }
        try:
            setattr(sim_cfg, "_runnercore11_c1_typed_genome_row_stats", stats)
        except Exception:
            return None
    return stats


def _runnercore11_stat_add(sim_cfg: Optional[Any], key: str, value: int | float = 1) -> None:
    stats = _runnercore11_stats(sim_cfg)
    if stats is None:
        return
    old = stats.get(str(key), 0)
    if isinstance(value, float):
        stats[str(key)] = float(old or 0.0) + float(value)
    else:
        stats[str(key)] = int(old or 0) + int(value)


def _runnercore11_required_failure(sim_cfg: Optional[Any], field: str) -> None:
    stats = _runnercore11_stats(sim_cfg)
    if stats is None:
        return
    failures = stats.get("required_field_failures", {})
    if not isinstance(failures, dict):
        failures = {}
    failures[str(field)] = int(failures.get(str(field), 0) or 0) + 1
    stats["required_field_failures"] = failures
    _runnercore11_stat_add(sim_cfg, "c1_typed_rows_missing_field_fail_closed", 1)


def _runnercore11_trace_projection(sim_cfg: Optional[Any], row: "RunnerCoreC1TypedGenomeRow") -> None:
    stats = _runnercore11_stats(sim_cfg)
    if stats is None:
        return
    stats["c1_projection_trace_rows"] = int(stats.get("c1_projection_trace_rows", 0) or 0) + 1
    sample = stats.get("c1_projection_trace_sample", [])
    if not isinstance(sample, list):
        sample = []
    if len(sample) < 32:
        sample.append(
            {
                "schema_version": row.schema_version,
                "payload_hash": row.c1_payload_hash,
                "c1_payload_hash": row.c1_payload_hash,
                "projection_digest": row.projection_digest,
                "policy_kind": row.policy_kind,
                "hidden_size": int(row.hidden_size),
                "use_gru": bool(row.use_gru),
                "memory_dim": int(row.memory_dim),
                "modules_count": int(row.modules_count),
                "connections_count": int(row.connections_count),
                "latent_pos_digest": row.latent_pos_digest,
            }
        )
    stats["c1_projection_trace_sample"] = sample


def _runnercore11_trace_sample_pending(sim_cfg: Optional[Any]) -> bool:
    stats = _runnercore11_stats(sim_cfg)
    if stats is None:
        return True
    sample = stats.get("c1_projection_trace_sample", [])
    return not isinstance(sample, list) or len(sample) < 32


@dataclass
class RunnerCoreC1TypedGenomeRow:
    """Typed projection of canonical C1 output before compatibility Genome materialization."""

    c1_payload: dict[str, Any]
    schema_version: str = RUNNERCORE11_C1_TYPED_GENOME_ROW_SCHEMA_VERSION
    hidden_size: int = 64
    exploration_bias: float = 0.0
    aggression_bias: float = 0.0
    reproduction_threshold: float = 30.0
    mutation_rate: float = 0.05
    action_temperature: float = 1.0
    use_gru: bool = False
    memory_dim: int = 0
    sensor_channel_mask: int = 0
    sensor_radius: int = DEFAULT_SENSOR_RADIUS
    introspection_channel_mask: int = 0
    self_regulation_control_mask: int = 0
    self_modification_target_mask: int = 0
    directed_inheritance_channel_mask: int = 0
    drive_inertia: float = 0.85
    drive_noise: float = 0.02
    drive_logit_gain: float = 1.0
    pred_lr: float = 0.08
    drive_sp_hunger: float = 0.55
    drive_sp_pain: float = 0.05
    drive_sp_fatigue: float = 0.15
    drive_sp_fear: float = 0.10
    exp_prior_strength: float = 0.35
    exp_prior_noise: float = 0.05
    drive_birth_noise: float = 0.03
    resource_handling_axis: float = RESOURCE_HANDLING_AXIS_NEUTRAL
    b3_locality_cost_total: float = 0.0
    b3_locality_mean_distance: float = 0.0
    b3_locality_max_distance: float = 0.0
    c1_catalog_version: str = ""
    c1_operator_schedule: str = ""
    c1_operator_applied: str = ""
    c1_operator_budget_status: str = ""
    c1_operator_noop_count: int = 0
    c2_recombination_version: str = ""
    c2_alignment_key: str = ""
    c2_unmatched_policy: str = ""
    c2_matched_modules: int = 0
    c2_matched_connections: int = 0
    policy_kind: str = "pending_spawn_policy"
    policy_source_kind: str = "pending_spawn_policy_source"
    modules_count: int = 0
    connections_count: int = 0
    latent_pos_digest: str = ""
    c1_payload_hash: str = ""
    projection_digest: str = ""
    compatibility_export_state: str = "typed_projection_no_genome"
    unsupported_reason: str = ""
    rng_replay_marker: str = "c1_typed_projection"
    _materialized_genome: Genome | None = field(default=None, repr=False, compare=False)

    @classmethod
    def from_canonical_payload(
        cls,
        payload: dict[str, Any],
        *,
        sim_cfg: Optional[Any] = None,
        c1_payload_hash: str | None = None,
        emit_projection_digest: bool = True,
    ) -> "RunnerCoreC1TypedGenomeRow":
        if not isinstance(payload, dict) or not isinstance(payload.get("genes"), dict):
            _runnercore11_required_failure(sim_cfg, "payload.genes")
            raise RuntimeError("RUNNERCORE11-UNSUPPORTED: canonical C1 payload missing genes")
        genes = dict(payload.get("genes", {}) or {})
        required = (
            "hidden_size",
            "exploration_bias",
            "aggression_bias",
            "reproduction_threshold",
            "mutation_rate",
            "action_temperature",
            "use_gru",
            "memory_dim",
            "drive_inertia",
            "drive_noise",
            "drive_logit_gain",
            "pred_lr",
            "drive_sp_hunger",
            "drive_sp_pain",
            "drive_sp_fatigue",
            "drive_sp_fear",
            "exp_prior_strength",
            "exp_prior_noise",
            "drive_birth_noise",
        )
        for field_name in required:
            if field_name not in genes:
                _runnercore11_required_failure(sim_cfg, f"genes.{field_name}")
                raise RuntimeError(f"RUNNERCORE11-UNSUPPORTED: canonical C1 payload missing {field_name}")
        modules = list(payload.get("modules", []) or [])
        connections = list(payload.get("connections", []) or [])
        latent_positions = [
            list(module.get("latent_pos", [0.0, 0.0]))
            for module in modules
            if isinstance(module, dict)
        ]
        digest = str(c1_payload_hash or "")
        if len(digest) == 64:
            _runnercore11_stat_add(sim_cfg, "runnercore12_c1_payload_digest_reuse_rows", 1)
        else:
            digest = _stable_json_hash(payload)
            if sim_cfg is not None and bool(getattr(sim_cfg, "runnercore12_c1_payload_digest_reuse_enable", False)):
                _runnercore11_stat_add(sim_cfg, "runnercore12_c1_payload_digest_reuse_fallbacks", 1)
        if bool(emit_projection_digest):
            latent_pos_digest = _stable_json_hash(latent_positions)
            _runnercore11_stat_add(sim_cfg, "runnercore13_c1_projection_digest_full_rows", 1)
        else:
            latent_pos_digest = ""
            _runnercore11_stat_add(sim_cfg, "runnercore13_c1_projection_digest_skipped_rows", 1)
        row = cls(
            c1_payload=dict(payload),
            hidden_size=int(genes.get("hidden_size", 64)),
            exploration_bias=float(genes.get("exploration_bias", 0.0)),
            aggression_bias=float(genes.get("aggression_bias", 0.0)),
            reproduction_threshold=float(genes.get("reproduction_threshold", 30.0)),
            mutation_rate=float(genes.get("mutation_rate", 0.05)),
            action_temperature=float(genes.get("action_temperature", 1.0)),
            use_gru=bool(genes.get("use_gru", False)),
            memory_dim=int(genes.get("memory_dim", 0) or 0),
            sensor_channel_mask=int(genes.get("sensor_channel_mask", implemented_sensor_channel_mask())),
            sensor_radius=int(genes.get("sensor_radius", DEFAULT_SENSOR_RADIUS)),
            introspection_channel_mask=int(genes.get("introspection_channel_mask", implemented_introspection_channel_mask())),
            self_regulation_control_mask=int(
                genes.get("self_regulation_control_mask", implemented_self_regulation_control_mask())
            ),
            self_modification_target_mask=int(
                genes.get("self_modification_target_mask", implemented_self_modification_target_mask())
            ),
            directed_inheritance_channel_mask=int(
                genes.get("directed_inheritance_channel_mask", implemented_directed_inheritance_channel_mask())
            ),
            drive_inertia=float(genes.get("drive_inertia", 0.85)),
            drive_noise=float(genes.get("drive_noise", 0.02)),
            drive_logit_gain=float(genes.get("drive_logit_gain", 1.0)),
            pred_lr=float(genes.get("pred_lr", 0.08)),
            drive_sp_hunger=float(genes.get("drive_sp_hunger", 0.55)),
            drive_sp_pain=float(genes.get("drive_sp_pain", 0.05)),
            drive_sp_fatigue=float(genes.get("drive_sp_fatigue", 0.15)),
            drive_sp_fear=float(genes.get("drive_sp_fear", 0.10)),
            exp_prior_strength=float(genes.get("exp_prior_strength", 0.35)),
            exp_prior_noise=float(genes.get("exp_prior_noise", 0.05)),
            drive_birth_noise=float(genes.get("drive_birth_noise", 0.03)),
            resource_handling_axis=clamp_resource_handling_axis(
                genes.get(RESOURCE_HANDLING_AXIS_FIELD, RESOURCE_HANDLING_AXIS_NEUTRAL)
            ),
            b3_locality_cost_total=float(payload.get("b3_locality_cost_total", 0.0) or 0.0),
            b3_locality_mean_distance=float(payload.get("b3_locality_mean_distance", 0.0) or 0.0),
            b3_locality_max_distance=float(payload.get("b3_locality_max_distance", 0.0) or 0.0),
            c1_catalog_version=str(genes.get("c1_catalog_version", "") or ""),
            c1_operator_schedule=str(genes.get("c1_operator_schedule", "") or ""),
            c1_operator_applied=str(genes.get("c1_operator_applied", "") or ""),
            c1_operator_budget_status=str(genes.get("c1_operator_budget_status", "") or ""),
            c1_operator_noop_count=int(genes.get("c1_operator_noop_count", 0) or 0),
            c2_recombination_version=str(genes.get("c2_recombination_version", "") or ""),
            c2_alignment_key=str(genes.get("c2_alignment_key", "") or ""),
            c2_unmatched_policy=str(genes.get("c2_unmatched_policy", "") or ""),
            c2_matched_modules=int(genes.get("c2_matched_modules", 0) or 0),
            c2_matched_connections=int(genes.get("c2_matched_connections", 0) or 0),
            modules_count=int(len(modules)),
            connections_count=int(len(connections)),
            latent_pos_digest=latent_pos_digest,
            c1_payload_hash=digest,
        )
        if bool(emit_projection_digest):
            projection_payload = {
                    "schema_version": row.schema_version,
                    "payload_hash": row.c1_payload_hash,
                    "hidden_size": row.hidden_size,
                    "use_gru": row.use_gru,
                    "memory_dim": row.memory_dim,
                    "aggression_bias": row.aggression_bias,
                    "action_temperature": row.action_temperature,
                    "drive_logit_gain": row.drive_logit_gain,
                    "reproduction_threshold": row.reproduction_threshold,
                    "modules_count": row.modules_count,
                    "connections_count": row.connections_count,
                    "latent_pos_digest": row.latent_pos_digest,
            }
            if float(row.resource_handling_axis) != RESOURCE_HANDLING_AXIS_NEUTRAL:
                projection_payload[RESOURCE_HANDLING_AXIS_FIELD] = float(row.resource_handling_axis)
            row.projection_digest = _stable_json_hash(projection_payload)
        return row

    def set_gene(self, name: str, value: Any) -> None:
        setattr(self, str(name), value)
        genes = self.c1_payload.setdefault("genes", {})
        if isinstance(genes, dict):
            genes[str(name)] = value
        self._materialized_genome = None
        self.c1_payload_hash = _stable_json_hash(self.c1_payload)
        projection_payload = {
                "schema_version": self.schema_version,
                "payload_hash": self.c1_payload_hash,
                "hidden_size": int(self.hidden_size),
                "use_gru": bool(self.use_gru),
                "memory_dim": int(self.memory_dim),
                "aggression_bias": float(self.aggression_bias),
                "action_temperature": float(self.action_temperature),
                "drive_logit_gain": float(self.drive_logit_gain),
                "reproduction_threshold": float(self.reproduction_threshold),
                "modules_count": int(self.modules_count),
                "connections_count": int(self.connections_count),
                "latent_pos_digest": self.latent_pos_digest,
        }
        if float(getattr(self, RESOURCE_HANDLING_AXIS_FIELD, RESOURCE_HANDLING_AXIS_NEUTRAL)) != RESOURCE_HANDLING_AXIS_NEUTRAL:
            projection_payload[RESOURCE_HANDLING_AXIS_FIELD] = float(getattr(self, RESOURCE_HANDLING_AXIS_FIELD))
        self.projection_digest = _stable_json_hash(projection_payload)

    def to_dict(self) -> dict[str, Any]:
        return dict(self.c1_payload)

    def to_raw_dict(self) -> dict[str, Any]:
        return dict(self.c1_payload)

    def canonical_payload(self) -> dict[str, Any]:
        return dict(self.c1_payload)

    def canonical_hash(self) -> str:
        return str(self.c1_payload_hash)

    def materialize_genome(self) -> Genome:
        if self._materialized_genome is None:
            self._materialized_genome = Genome.from_trusted_canonical_payload(self.c1_payload)
            self.compatibility_export_state = "materialized_genome"
        return self._materialized_genome


def _runnercore4_structural_enabled(sim_cfg: Optional[Any], g1: Genome, g2: Genome) -> bool:
    return bool(
        sim_cfg is not None
        and g1 is g2
        and bool(getattr(sim_cfg, "runnercore4_candidate_enable", False))
        and bool(getattr(sim_cfg, "runnercore4_structural_payload_cache_enable", False))
    )


def _runnercore4_stats(sim_cfg: Optional[Any]) -> dict[str, Any] | None:
    if sim_cfg is None:
        return None
    stats = getattr(sim_cfg, "_runnercore4_structural_payload_stats", None)
    if not isinstance(stats, dict):
        stats = {
            "schema_version": "runnercore4_structural_payload_fastpath_v1",
            "cache_version_guard": "parent_genome_identity_modules_connections_identity",
        }
        try:
            setattr(sim_cfg, "_runnercore4_structural_payload_stats", stats)
        except Exception:
            return None
    return stats


def _runnercore4_stat_add(sim_cfg: Optional[Any], key: str, value: int | float = 1) -> None:
    stats = _runnercore4_stats(sim_cfg)
    if stats is None:
        return
    old = stats.get(str(key), 0)
    if isinstance(value, float):
        stats[str(key)] = float(old or 0.0) + float(value)
    else:
        stats[str(key)] = int(old or 0) + int(value)


def _runnercore4_bucket_add(buckets: Optional[dict[str, float]], key: str, seconds: float) -> None:
    if not isinstance(buckets, dict) or seconds < 0.0:
        return
    buckets[str(key)] = float(buckets.get(str(key), 0.0) or 0.0) + float(seconds)


def _runnercore5_stats(sim_cfg: Optional[Any]) -> dict[str, Any] | None:
    if sim_cfg is None:
        return None
    if not (
        bool(getattr(sim_cfg, "runnercore5_profile_enable", False))
        or bool(getattr(sim_cfg, "runnercore5_candidate_enable", False))
        or bool(getattr(sim_cfg, "runnercore5_flat_self_policy_rng_enable", False))
        or bool(getattr(sim_cfg, "runnercore5_direct_newborn_mlp_bank_attach_enable", False))
        or bool(getattr(sim_cfg, "runnercore5_c1_canonical_boundary_fastpath_enable", False))
        or bool(getattr(sim_cfg, "runnercore6_trusted_canonical_genome_materialization_enable", False))
        or bool(getattr(sim_cfg, "runnercore7_self_mlp_policy_source_alias_enable", False))
        or bool(getattr(sim_cfg, "runnercore7_self_gru_policy_source_alias_enable", False))
    ):
        return None
    stats = getattr(sim_cfg, "_runnercore5_child_construction_stats", None)
    if not isinstance(stats, dict):
        stats = {
            "schema_version": "runnercore5_child_construction_ownership_v1",
            "policy_contract": "independent_child_policy_objects_flat_self_rng",
        }
        try:
            setattr(sim_cfg, "_runnercore5_child_construction_stats", stats)
        except Exception:
            return None
    return stats


def _runnercore5_stat_add(sim_cfg: Optional[Any], key: str, value: int | float = 1) -> None:
    stats = _runnercore5_stats(sim_cfg)
    if stats is None:
        return
    old = stats.get(str(key), 0)
    if isinstance(value, float):
        stats[str(key)] = float(old or 0.0) + float(value)
    else:
        stats[str(key)] = int(old or 0) + int(value)


def _runnercore5_bucket_add(buckets: Optional[dict[str, float]], key: str, seconds: float) -> None:
    if not isinstance(buckets, dict) or seconds < 0.0:
        return
    buckets[str(key)] = float(buckets.get(str(key), 0.0) or 0.0) + float(seconds)


def _runnercore9_stats(sim_cfg: Optional[Any]) -> dict[str, Any] | None:
    if sim_cfg is None:
        return None
    if not (
        bool(getattr(sim_cfg, "runnercore9_profile_enable", False))
        or bool(getattr(sim_cfg, "runnercore9_candidate_enable", False))
        or bool(getattr(sim_cfg, "runnercore9_c1_trace_enable", False))
        or bool(getattr(sim_cfg, "runnercore9_c1_scalar_op_normalize_skip_enable", False))
        or bool(getattr(sim_cfg, "runnercore9_lazy_newborn_materialization_enable", False))
    ):
        return None
    stats = getattr(sim_cfg, "_runnercore9_c1_payload_or_lazy_newborn_stats", None)
    if not isinstance(stats, dict):
        stats = {
            "schema_version": "runnercore9_c1_payload_or_lazy_newborn_v1",
            "c1_payload_contract": "trace_only_scalar_normalize_skip_parity_blocked",
            "lazy_newborn_contract": (
                "resident_rows_defer_compat_agent_until_export"
                if bool(getattr(sim_cfg, "runnercore9_lazy_newborn_materialization_enable", False))
                else "not_enabled"
            ),
            "native_cpp_status": "not_attached_python_c1_and_materialization_boundary",
            "c1_operator_pipeline_calls": 0,
            "c1_scalar_normalize_skipped": 0,
            "c1_normalize_calls": 0,
            "c1_trace_rows": 0,
            "c1_trace_sample": [],
            "newborn_agent_materializations_deferred": 0,
            "newborn_agent_materializations_on_demand": 0,
            "policy_source_forced_materializations": 0,
            "policy_bank_sync_forced_materializations": 0,
            "generation_end_materializations": 0,
            "checkpoint_forced_materializations": 0,
            "unsupported_reader_materializations": 0,
            "resident_rows_without_agent_objects": 0,
            "event_boundary_export_rows_avoided": 0,
            "hidden_agent_sync_rows": 0,
            "unsupported_fallback_count": 0,
        }
        try:
            setattr(sim_cfg, "_runnercore9_c1_payload_or_lazy_newborn_stats", stats)
        except Exception:
            return None
    return stats


def _runnercore9_stat_add(sim_cfg: Optional[Any], key: str, value: int | float = 1) -> None:
    stats = _runnercore9_stats(sim_cfg)
    if stats is None:
        return
    old = stats.get(str(key), 0)
    if isinstance(value, float):
        stats[str(key)] = float(old or 0.0) + float(value)
    else:
        stats[str(key)] = int(old or 0) + int(value)


def _runnercore9_trace_extend(sim_cfg: Optional[Any], records: list[dict[str, Any]]) -> None:
    if not records:
        return
    stats = _runnercore9_stats(sim_cfg)
    if stats is None:
        return
    stats["c1_trace_rows"] = int(stats.get("c1_trace_rows", 0) or 0) + int(len(records))
    sample = stats.get("c1_trace_sample", [])
    if not isinstance(sample, list):
        sample = []
    remaining = max(0, 32 - int(len(sample)))
    if remaining:
        sample.extend(dict(row) for row in records[:remaining])
    stats["c1_trace_sample"] = sample


def _runnercore4_parent_version_key(g: Genome) -> tuple[Any, ...]:
    return (
        id(g),
        id(getattr(g, "modules", None)),
        len(list(getattr(g, "modules", []) or [])),
        id(getattr(g, "connections", None)),
        len(list(getattr(g, "connections", []) or [])),
        str(getattr(g, "c1_operator_schedule", "") or ""),
        str(getattr(g, "c1_operator_applied", "") or ""),
        str(getattr(g, "c2_recombination_version", "") or ""),
        int(getattr(g, "c2_matched_modules", 0) or 0),
        int(getattr(g, "c2_matched_connections", 0) or 0),
        tuple(sorted(resolve_b2_caps(getattr(g, "b2_caps", None)).items())),
        float(clamp_resource_handling_axis(getattr(g, RESOURCE_HANDLING_AXIS_FIELD, RESOURCE_HANDLING_AXIS_NEUTRAL))),
    )


def _runnercore4_cached_parent_payload(g: Genome, sim_cfg: Optional[Any]) -> dict[str, Any]:
    version_key = _runnercore4_parent_version_key(g)
    cached = getattr(g, "_runnercore4_self_canonical_payload_cache", None)
    cached_version = getattr(g, "_runnercore4_self_canonical_payload_version", None)
    if isinstance(cached, dict) and cached_version == version_key:
        _runnercore4_stat_add(sim_cfg, "structural_payload_cache_hits", 1)
        return cached
    payload = canonical_payload_from_genome_dict(g.to_raw_dict())
    try:
        setattr(g, "_runnercore4_self_canonical_payload_cache", payload)
        setattr(g, "_runnercore4_self_canonical_payload_version", version_key)
    except Exception:
        pass
    _runnercore4_stat_add(sim_cfg, "structural_payload_cache_misses", 1)
    _runnercore4_stat_add(sim_cfg, "parent_payload_normalization_calls", 1)
    return payload


def _runnercore4_child_payload_from_parts(
    *,
    child_kwargs: dict[str, Any],
    rec_payload: dict[str, Any],
    c2_genes: dict[str, Any],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "hidden_size": int(child_kwargs["hidden_size"]),
        "exploration_bias": float(child_kwargs["exploration_bias"]),
        "aggression_bias": float(child_kwargs["aggression_bias"]),
        "reproduction_threshold": float(child_kwargs["reproduction_threshold"]),
        "mutation_rate": float(child_kwargs["mutation_rate"]),
        "action_temperature": float(child_kwargs["action_temperature"]),
        "use_gru": bool(child_kwargs["use_gru"]),
        "memory_dim": int(child_kwargs["memory_dim"]),
        "sensor_channel_mask": int(child_kwargs["sensor_channel_mask"]),
        "sensor_radius": int(child_kwargs["sensor_radius"]),
        "introspection_channel_mask": int(child_kwargs["introspection_channel_mask"]),
        "self_regulation_control_mask": int(child_kwargs["self_regulation_control_mask"]),
        "self_modification_target_mask": int(child_kwargs["self_modification_target_mask"]),
        "directed_inheritance_channel_mask": int(child_kwargs["directed_inheritance_channel_mask"]),
        "drive_inertia": float(child_kwargs["drive_inertia"]),
        "drive_noise": float(child_kwargs["drive_noise"]),
        "drive_logit_gain": float(child_kwargs["drive_logit_gain"]),
        "pred_lr": float(child_kwargs["pred_lr"]),
        "drive_sp_hunger": float(child_kwargs["drive_sp_hunger"]),
        "drive_sp_pain": float(child_kwargs["drive_sp_pain"]),
        "drive_sp_fatigue": float(child_kwargs["drive_sp_fatigue"]),
        "drive_sp_fear": float(child_kwargs["drive_sp_fear"]),
        "exp_prior_strength": float(child_kwargs["exp_prior_strength"]),
        "exp_prior_noise": float(child_kwargs["exp_prior_noise"]),
        "drive_birth_noise": float(child_kwargs["drive_birth_noise"]),
        "schema_version": MODULAR_GENOME_SCHEMA_VERSION,
        "interface_schema_version": B2_INTERFACE_SCHEMA_VERSION,
        "b2_invalid_policy": str(rec_payload.get("b2_invalid_policy", "clip") or "clip"),
        "b2_caps": dict(rec_payload.get("b2_caps", B2_FROZEN_CAPS) or B2_FROZEN_CAPS),
        "b3_locality_schema_version": B3_LOCALITY_SCHEMA_VERSION,
        "b3_locality_max_connection_distance": float(
            rec_payload.get("b3_locality_max_connection_distance", B3_FROZEN_LOCALITY["max_connection_distance"])
        ),
        "b3_distance_cost_scale": float(rec_payload.get("b3_distance_cost_scale", B3_FROZEN_LOCALITY["distance_cost_scale"])),
        "b3_distance_cost_max": float(rec_payload.get("b3_distance_cost_max", B3_FROZEN_LOCALITY["distance_cost_max"])),
        "b3_locality_cost_total": float(rec_payload.get("b3_locality_cost_total", 0.0) or 0.0),
        "b3_locality_mean_distance": float(rec_payload.get("b3_locality_mean_distance", 0.0) or 0.0),
        "b3_locality_max_distance": float(rec_payload.get("b3_locality_max_distance", 0.0) or 0.0),
        "b3_locality_dropped_connections": int(rec_payload.get("b3_locality_dropped_connections", 0) or 0),
        "c1_catalog_version": "",
        "c1_operator_schedule": "",
        "c1_operator_applied": "",
        "c1_operator_noop_count": 0,
        "c1_operator_budget_status": "",
        "c1_op_count__bounded_rewire": 0,
        "c1_op_count__duplicate_with_decay": 0,
        "c1_op_count__edge_split_insertion": 0,
        "c1_op_count__fusion_merge_compression": 0,
        "c1_op_count__gate_drift": 0,
        "c1_op_count__motif_duplication": 0,
        "c1_op_count__promoter_expression_mutation": 0,
        "developmental_program": canonicalize_developmental_program(child_kwargs.get("developmental_program", None)),
        "modules": list(rec_payload.get("modules", [])),
        "connections": list(rec_payload.get("connections", [])),
    }
    resource_axis = clamp_resource_handling_axis(
        child_kwargs.get(RESOURCE_HANDLING_AXIS_FIELD, RESOURCE_HANDLING_AXIS_NEUTRAL)
    )
    if resource_axis != RESOURCE_HANDLING_AXIS_NEUTRAL:
        payload[RESOURCE_HANDLING_AXIS_FIELD] = float(resource_axis)
    if child_kwargs.get(FAMILY_HEAD_RESIDUAL_FIELD) is not None:
        payload[FAMILY_HEAD_RESIDUAL_FIELD] = canonical_family_head_residual_block(
            child_kwargs.get(FAMILY_HEAD_RESIDUAL_FIELD)
        )
    for key in (
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
        "c2_parent_order_rule",
        "c2_legacy_unresolved_modules",
    ):
        default = "" if key in {
            "c2_recombination_version",
            "c2_alignment_key",
            "c2_unmatched_policy",
            "c2_parent_order_rule",
        } else 0
        payload[key] = c2_genes.get(key, default)
    if str(c2_genes.get("c2_inheritance_profile", "") or ""):
        payload["c2_inheritance_profile"] = str(c2_genes["c2_inheritance_profile"])
        payload["c2_structural_linkage_rule"] = str(
            c2_genes.get("c2_structural_linkage_rule", "") or ""
        )
        payload["c2_linked_component_count"] = int(
            c2_genes.get("c2_linked_component_count", 0) or 0
        )
    return payload

# mutation specs: field -> (sigma, lo, hi)
_MUT_SPECS: Dict[str, Tuple[float, float, float]] = {
    "exploration_bias": (0.10, -1.0, 1.0),
    "aggression_bias": (0.15, -2.0, 2.0),
    "reproduction_threshold": (2.0, 2.0, 90.0),
    "action_temperature": (0.10, 0.25, 2.5),

    # Identity Pressure genes
    "drive_inertia": (0.05, 0.0, 0.999),
    "drive_noise": (0.01, 0.0, 0.10),
    "drive_logit_gain": (0.20, 0.0, 3.0),
    "pred_lr": (0.03, 0.0, 1.0),
}

_MUT_RATE_BOUNDS = (0.001, 0.20)


def _genome_payload_for_canonical_boundary(genome: Genome) -> dict[str, Any]:
    """Return raw genome fields for APIs that canonicalize at their boundary."""
    to_raw = getattr(genome, "to_raw_dict", None)
    if callable(to_raw):
        return dict(to_raw())
    return dict(genome.to_dict())


def _cached_canonical_payload_for_trusted_boundary(genome: Genome) -> dict[str, Any]:
    cached = getattr(genome, "cached_canonical_payload", None)
    if callable(cached):
        return dict(cached())
    return dict(genome.canonical_payload())


def _direct_child_payload_for_canonical_boundary(
    *,
    hidden_size: int,
    exploration: float,
    aggression: float,
    repro_th: float,
    mut_rate: float,
    temp: float,
    use_gru: bool,
    memory_dim: int,
    sensor_channel_mask: int,
    sensor_radius: int,
    introspection_channel_mask: int,
    self_regulation_control_mask: int,
    self_modification_target_mask: int,
    directed_inheritance_channel_mask: int,
    inertia: float,
    noise: float,
    gain: float,
    pred_lr: float,
    sp_hunger: float,
    sp_pain: float,
    sp_fatigue: float,
    sp_fear: float,
    exp_strength: float,
    exp_noise: float,
    birth_noise: float,
    developmental_program: dict[str, Any],
    b2_caps: dict[str, int],
    resource_handling_axis: float = RESOURCE_HANDLING_AXIS_NEUTRAL,
    family_head_residuals: dict[str, Any] | None = None,
) -> dict[str, Any]:
    md = _snap_memory_dim(int(memory_dim))
    use_gru_final = bool(use_gru) or bool(md > 0)
    payload = {
        "hidden_size": int(hidden_size),
        "exploration_bias": canon_float(float(exploration), field="exploration_bias"),
        "aggression_bias": canon_float(float(aggression), field="aggression_bias"),
        "reproduction_threshold": canon_float(float(repro_th), field="reproduction_threshold"),
        "mutation_rate": canon_float(float(mut_rate), field="mutation_rate"),
        "action_temperature": canon_float(float(temp), field="action_temperature"),
        "use_gru": bool(use_gru_final),
        "memory_dim": int(md),
        "sensor_channel_mask": int(sanitize_sensor_channel_mask(sensor_channel_mask)),
        "sensor_radius": int(snap_sensor_radius(sensor_radius)),
        "introspection_channel_mask": int(sanitize_introspection_channel_mask(introspection_channel_mask)),
        "self_regulation_control_mask": int(sanitize_self_regulation_control_mask(self_regulation_control_mask)),
        "self_modification_target_mask": int(sanitize_self_modification_target_mask(self_modification_target_mask)),
        "directed_inheritance_channel_mask": int(sanitize_directed_inheritance_channel_mask(directed_inheritance_channel_mask)),
        "drive_inertia": canon_float(min(0.999, max(0.0, float(inertia))), field="drive_inertia"),
        "drive_noise": canon_float(min(0.10, max(0.0, float(noise))), field="drive_noise"),
        "drive_logit_gain": canon_float(min(3.0, max(0.0, float(gain))), field="drive_logit_gain"),
        "pred_lr": canon_float(min(1.0, max(0.0, float(pred_lr))), field="pred_lr"),
        "drive_sp_hunger": canon_float(min(1.0, max(0.0, float(sp_hunger))), field="drive_sp_hunger"),
        "drive_sp_pain": canon_float(min(1.0, max(0.0, float(sp_pain))), field="drive_sp_pain"),
        "drive_sp_fatigue": canon_float(min(1.0, max(0.0, float(sp_fatigue))), field="drive_sp_fatigue"),
        "drive_sp_fear": canon_float(min(1.0, max(0.0, float(sp_fear))), field="drive_sp_fear"),
        "exp_prior_strength": canon_float(min(1.0, max(0.0, float(exp_strength))), field="exp_prior_strength"),
        "exp_prior_noise": canon_float(min(0.5, max(0.0, float(exp_noise))), field="exp_prior_noise"),
        "drive_birth_noise": canon_float(min(0.2, max(0.0, float(birth_noise))), field="drive_birth_noise"),
        "schema_version": MODULAR_GENOME_SCHEMA_VERSION,
        "interface_schema_version": B2_INTERFACE_SCHEMA_VERSION,
        "b2_invalid_policy": "clip",
        "b2_caps": resolve_b2_caps(b2_caps),
        "b3_locality_schema_version": B3_LOCALITY_SCHEMA_VERSION,
        "b3_locality_max_connection_distance": canon_float(
            B3_FROZEN_LOCALITY["max_connection_distance"],
            field="b3_locality_max_connection_distance",
        ),
        "b3_distance_cost_scale": canon_float(
            B3_FROZEN_LOCALITY["distance_cost_scale"],
            field="b3_distance_cost_scale",
        ),
        "b3_distance_cost_max": canon_float(
            B3_FROZEN_LOCALITY["distance_cost_max"],
            field="b3_distance_cost_max",
        ),
        "b3_locality_cost_total": canon_float(0.0, field="b3_locality_cost_total"),
        "b3_locality_mean_distance": canon_float(0.0, field="b3_locality_mean_distance"),
        "b3_locality_max_distance": canon_float(0.0, field="b3_locality_max_distance"),
        "b3_locality_dropped_connections": 0,
        "c1_catalog_version": "",
        "c1_operator_schedule": "",
        "c1_operator_applied": "",
        "c1_operator_noop_count": 0,
        "c1_operator_budget_status": "",
        "c1_op_count__bounded_rewire": 0,
        "c1_op_count__duplicate_with_decay": 0,
        "c1_op_count__edge_split_insertion": 0,
        "c1_op_count__fusion_merge_compression": 0,
        "c1_op_count__gate_drift": 0,
        "c1_op_count__motif_duplication": 0,
        "c1_op_count__promoter_expression_mutation": 0,
        "c2_recombination_version": "",
        "c2_alignment_key": "",
        "c2_unmatched_policy": "",
        "c2_matched_modules": 0,
        "c2_unmatched_modules_parent_a": 0,
        "c2_unmatched_modules_parent_b": 0,
        "c2_matched_connections": 0,
        "c2_unmatched_connections_parent_a": 0,
        "c2_unmatched_connections_parent_b": 0,
        "c2_budget_draws": 0,
        "modules": [],
        "connections": [],
        "developmental_program": canonicalize_developmental_program(developmental_program),
    }
    resource_axis = clamp_resource_handling_axis(resource_handling_axis)
    if resource_axis != RESOURCE_HANDLING_AXIS_NEUTRAL:
        payload[RESOURCE_HANDLING_AXIS_FIELD] = float(resource_axis)
    if family_head_residuals is not None:
        payload[FAMILY_HEAD_RESIDUAL_FIELD] = canonical_family_head_residual_block(family_head_residuals)
    return payload


def _add_timing_bucket(buckets: Optional[Dict[str, float]], name: str, started: float) -> None:
    if isinstance(buckets, dict):
        buckets[name] = buckets.get(name, 0.0) + (time.perf_counter() - started)


def _mutate_float(x: float, rng: random.Random, rate: float, sigma: float, lo: float, hi: float) -> float:
    if rng.random() < rate:
        x = x + rng.gauss(0.0, sigma)
    return float(np.clip(x, lo, hi))


def _mutate_mutation_rate(mut_rate: float, rng: random.Random) -> float:
    if rng.random() < 0.2:
        mut_rate = mut_rate + rng.gauss(0.0, 0.005)
    return float(np.clip(mut_rate, _MUT_RATE_BOUNDS[0], _MUT_RATE_BOUNDS[1]))


def _attach_resource_handling_record(
    genome_or_row: Any,
    sim_cfg: Optional[Any],
    record: dict[str, Any] | None,
) -> Any:
    if (
        genome_or_row is not None
        and isinstance(record, dict)
        and sim_cfg is not None
        and bool(getattr(sim_cfg, "resource_handling_representation_reporting_enable", False))
    ):
        try:
            setattr(genome_or_row, "_resource_handling_reproduction_record", dict(record))
        except Exception:
            pass
    return genome_or_row


def _attach_family_head_residual_record(
    genome_or_row: Any,
    sim_cfg: Optional[Any],
    record: dict[str, Any] | None,
) -> Any:
    if (
        genome_or_row is not None
        and isinstance(record, dict)
        and sim_cfg is not None
        and bool(getattr(sim_cfg, "policy_family_head_residual_reporting_enable", False))
    ):
        try:
            setattr(genome_or_row, "_family_head_residual_reproduction_record", dict(record))
        except Exception:
            pass
    return genome_or_row


def _snap_memory_dim(d: int) -> int:
    try:
        dd = int(d)
    except Exception:
        dd = 0
    return int(min(_MEMORY_LADDER, key=lambda x: abs(x - dd)))


def _mutate_memory_dim_step(d: int, rng: random.Random, mut_rate: float) -> int:
    """Slow discrete step mutation on the ladder."""
    d = _snap_memory_dim(d)
    p = 0.25 * float(mut_rate)
    p = float(np.clip(p, 0.002, 0.05))  # slow, bounded
    if rng.random() >= p:
        return d

    idx = _MEMORY_INDEX[d]
    step = rng.choice([-1, 1])
    idx = max(0, min(len(_MEMORY_LADDER) - 1, idx + step))
    return int(_MEMORY_LADDER[idx])


def _family_head_residual_block_from_genome_like(genome_like: Any) -> dict[str, Any] | None:
    block = getattr(genome_like, FAMILY_HEAD_RESIDUAL_FIELD, None)
    if block is not None:
        return canonical_family_head_residual_block(block)
    try:
        payload = genome_like.to_dict() if callable(getattr(genome_like, "to_dict", None)) else {}
        if isinstance(payload, dict):
            genes = payload.get("genes", payload)
            if isinstance(genes, dict) and genes.get(FAMILY_HEAD_RESIDUAL_FIELD) is not None:
                return canonical_family_head_residual_block(genes.get(FAMILY_HEAD_RESIDUAL_FIELD))
    except Exception:
        return None
    return None


def _inherit_memory_dim(g1: Genome, g2: Genome, rng: random.Random, mut_rate: float) -> int:
    # midpoint inheritance, snapped to ladder
    base = int(round(0.5 * (int(getattr(g1, "memory_dim", 0)) + int(getattr(g2, "memory_dim", 0)))))
    base = _snap_memory_dim(base)
    # then slow step mutation
    return _mutate_memory_dim_step(base, rng, mut_rate)


def _attach_transient_linkage_context(genome_or_row: Any, context: dict[str, Any] | None) -> Any:
    """Attach non-serialized donor provenance for policy construction in spawn."""

    if genome_or_row is not None and isinstance(context, dict):
        transient = {key: value for key, value in context.items() if key != "donors"}
        try:
            setattr(genome_or_row, "_recomb1_linkage_context", transient)
        except Exception:
            pass
    return genome_or_row


def _attach_diploid_context(
    genome_or_row: Any,
    genotype: Any,
    receipt: dict[str, Any] | None,
) -> Any:
    if genome_or_row is None or genotype is None:
        return genome_or_row
    attach_diploid_genome(genome_or_row, genotype)
    if isinstance(receipt, dict):
        setattr(
            genome_or_row,
            "_diploid_inheritance_receipt_v1",
            dict(receipt),
        )
    return genome_or_row


def _attach_genome_upgrade_v2_context(
    genome_or_row: Any,
    genotype: GenomeUpgradeV2V1 | None,
    receipt: dict[str, Any] | None,
) -> Any:
    if genome_or_row is None or genotype is None:
        return genome_or_row
    attach_genome_upgrade_v2(genome_or_row, genotype)
    if isinstance(receipt, dict):
        setattr(
            genome_or_row,
            "_genome_upgrade_v2_inheritance_receipt_v1",
            dict(receipt),
        )
    return genome_or_row


def _recomb72_audit_crossover_genome(
    g1: Genome,
    g2: Genome,
    rng: random.Random,
    *,
    sim_cfg: Any,
    return_typed_projection: bool,
) -> Genome:
    """Execute the frozen mutation-off RECOMB-72 scalar boundary.

    Six draws are consumed for every mode. This keeps the downstream RNG
    schedule identical across comparison arms while leaving the normal
    production path completely untouched when the audit flag is disabled.
    """

    from evolution.recomb72_inheritance import (
        AUDITED_LOCI,
        BOUNDED_INTERPOLATION,
        CURRENT_AUTHORITATIVE,
        INHERITANCE_MODES,
        LOCUS_BOUNDS,
        MUTATION_OFF_CONTRACT,
        PARENT_COPY_PER_LOCUS,
        WHOLE_PACKAGE_SINGLE_PARENT,
        mode_sha256,
        registry_sha256,
    )

    if not isinstance(g1, Genome) or not isinstance(g2, Genome):
        raise RuntimeError("RECOMB72_AUDIT_CONTRACT: canonical Genome parents required")
    if return_typed_projection:
        raise RuntimeError(
            "RECOMB72_AUDIT_CONTRACT: typed structural projection is outside the audit"
        )
    if not bool(getattr(sim_cfg, "recomb72_mutation_off", False)):
        raise RuntimeError("RECOMB72_AUDIT_CONTRACT: mutation-off must be explicit")
    mode = str(getattr(sim_cfg, "recomb72_inheritance_mode", "") or "")
    if mode not in INHERITANCE_MODES:
        raise RuntimeError(f"RECOMB72_AUDIT_CONTRACT: unsupported mode {mode!r}")

    parent_a = {name: float(getattr(g1, name)) for name in AUDITED_LOCI}
    parent_b = {name: float(getattr(g2, name)) for name in AUDITED_LOCI}
    draws = tuple(float(rng.random()) for _ in AUDITED_LOCI)
    donor_by_locus: dict[str, str] = {}
    package_donor: str | None = None
    child_values: dict[str, float] = {}
    if mode == CURRENT_AUTHORITATIVE:
        for name in AUDITED_LOCI:
            child_values[name] = 0.5 * (parent_a[name] + parent_b[name])
    elif mode == PARENT_COPY_PER_LOCUS:
        for index, name in enumerate(AUDITED_LOCI):
            donor = "a" if draws[index] < 0.5 else "b"
            donor_by_locus[name] = donor
            child_values[name] = (
                parent_a[name] if donor == "a" else parent_b[name]
            )
    elif mode == BOUNDED_INTERPOLATION:
        for index, name in enumerate(AUDITED_LOCI):
            child_values[name] = parent_a[name] + draws[index] * (
                parent_b[name] - parent_a[name]
            )
    elif mode == WHOLE_PACKAGE_SINGLE_PARENT:
        package_donor = "a" if draws[0] < 0.5 else "b"
        donor_by_locus = {name: package_donor for name in AUDITED_LOCI}
        donor = parent_a if package_donor == "a" else parent_b
        child_values = dict(donor)

    for name, value in tuple(child_values.items()):
        lower, upper = LOCUS_BOUNDS[name]
        child_values[name] = min(float(upper), max(float(lower), float(value)))

    # Parent A is the deterministic authority for every non-audited genome
    # surface. No structural, memory, mutation-rate, or categorical operator is
    # entered in this audit branch.
    child = copy.deepcopy(g1)
    for name, value in child_values.items():
        setattr(child, name, canon_float(float(value), field=name))
    child.mutation_rate = 0.0
    refresh = getattr(child, "refresh_canonical_payload_cache", None)
    if callable(refresh):
        refresh()
    record = {
        "schema_version": "recomb72_crossover_event_v1",
        "mode": mode,
        "mode_sha256": mode_sha256(mode),
        "registry_sha256": registry_sha256(),
        "parent_a": parent_a,
        "parent_b": parent_b,
        "child": dict(child_values),
        "donor_by_locus": donor_by_locus,
        "package_donor": package_donor,
        "draw_count": len(draws),
        "fixed_draw_budget": len(AUDITED_LOCI),
        "mutation": dict(MUTATION_OFF_CONTRACT),
        "non_audited_surface": "exact_parent_a_copy",
    }
    setattr(child, "_recomb72_inheritance_event", record)
    return child


def _rng_semantics_version(sim_cfg: Optional[Any]) -> int:
    if sim_cfg is None:
        return 1
    try:
        version = int(getattr(sim_cfg, "rng_semantics_version", 1) or 1)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("UNSUPPORTED_RNG_SEMANTICS_VERSION") from exc
    if version not in (1, 2):
        raise RuntimeError(f"UNSUPPORTED_RNG_SEMANTICS_VERSION {version}")
    return version


def _v2_birth_identity(structural_identity_context: Optional[Dict[str, Any]]) -> Dict[str, int]:
    identity = dict(structural_identity_context or {})
    if "child_agent_id" not in identity:
        raise RuntimeError("GENOME_UPGRADE_V2_RNGV2_REQUIRES_CHILD_IDENTITY")
    return {
        "tick": int(identity.get("tick", 0) or 0),
        "child_agent_id": int(identity["child_agent_id"]),
    }


def crossover_genome(
    g1: Genome,
    g2: Genome,
    rng: random.Random,
    *,
    rng_manager: Optional[Any] = None,
    sim_cfg: Optional[Any] = None,
    buckets: Optional[Dict[str, float]] = None,
    return_typed_projection: bool = False,
    structural_identity_context: Optional[Dict[str, Any]] = None,
) -> Genome | RunnerCoreC1TypedGenomeRow:
    recomb72_audit = bool(getattr(sim_cfg, "recomb72_audit_enable", False))
    recomb72_mode = str(
        getattr(sim_cfg, "recomb72_inheritance_mode", "") or ""
    )
    if recomb72_audit and return_typed_projection:
        raise RuntimeError(
            "RECOMB72_AUDIT_CONTRACT: typed structural projection is outside the audit"
        )
    if recomb72_audit and not bool(
        getattr(sim_cfg, "recomb72_mutation_off", False)
    ):
        raise RuntimeError("RECOMB72_AUDIT_CONTRACT: mutation-off must be explicit")
    if recomb72_audit and recomb72_mode != "current_authoritative":
        return _recomb72_audit_crossover_genome(
            g1,
            g2,
            rng,
            sim_cfg=sim_cfg,
            return_typed_projection=return_typed_projection,
        )
    b = buckets if isinstance(buckets, dict) else None
    _t_base0 = time.perf_counter() if b is not None else 0.0
    parent_caps = resolve_b2_caps(getattr(g1, "b2_caps", None))
    if parent_caps != resolve_b2_caps(getattr(g2, "b2_caps", None)):
        raise RuntimeError("CAPACITY_EPOCH_PARENT_PROFILE_MISMATCH")
    weather_generation = int(getattr(sim_cfg, "_mutation_weather_generation", 0) or 0) if sim_cfg is not None else 0
    weather = mutation_weather_state(sim_cfg, weather_generation)
    weather_scalar_multiplier = (
        float(weather.scalar_multiplier)
        if bool(getattr(sim_cfg, "mutation_weather_apply_to_scalar_sigma", True)) and bool(weather.active)
        else 1.0
    )
    weather_count_multiplier = (
        float(weather.scalar_multiplier)
        if bool(getattr(sim_cfg, "mutation_weather_apply_to_mutation_probability", False)) and bool(weather.active)
        else 1.0
    )
    inheritance_profile = active_inheritance_profile(sim_cfg)
    linkage_context: dict[str, Any] | None = None
    diploid_inheritance = inheritance_profile == DIPLOID_INHERITANCE_PROFILE
    flat_linkage_component = diploid_inheritance
    diploid_genotype = None
    diploid_receipt: dict[str, Any] | None = None
    genome_upgrade_enabled = bool(
        getattr(sim_cfg, "genome_upgrade_v2_enable", False)
    )
    genome_upgrade_genotype: GenomeUpgradeV2V1 | None = None
    genome_upgrade_receipt: dict[str, Any] | None = None
    if genome_upgrade_enabled and not diploid_inheritance:
        raise RuntimeError("GENOME_UPGRADE_V2_REQUIRES_DIPLOID_INHERITANCE")
    if diploid_inheritance and return_typed_projection:
        raise RuntimeError(
            "DIPLOID_UNSUPPORTED_TYPED_PROJECTION "
            "diploid genotype materialization is required"
        )
    if flat_linkage_component:
        # This validation/draw boundary precedes every inheritance or mutation
        # draw. The diploid/Genome V2 compatibility component retains the
        # frozen seven-draw prelude without admitting its retired source
        # profile as a production authority.
        if not isinstance(g1, Genome) or not isinstance(g2, Genome):
            raise RuntimeError(
                "RECOMB1_INVALID_PARENT_PAYLOAD "
                "linked/diploid inheritance requires two canonical Genome parents"
            )
        if rng_manager is None or rng is not rng_manager.py_rng("mutation"):
            raise RuntimeError(
                "[DETERMINISM_VIOLATION]\n"
                "code=DET_FORBIDDEN_RNG_API\n"
                "context=linked/diploid inheritance requires the authoritative RNGManager mutation.py stream\n"
            )
        # Canonicalize once at the top of the repaired lane.  The seven repaired
        # linkage blocks already use this order internally; applying it here also
        # freezes the still-valid categorical/mask inheritance surfaces that run
        # later in this function.
        (g1, g2), _canonical_parent_digests = canonical_parent_order(g1, g2)
        linkage_context = draw_linkage_parent_context(g1, g2, rng_manager)
        donors = linkage_context["donors"]
        behavioral_parent = donors["behavioral_strategy_v1"]
        regulation_parent = donors["regulation_prediction_v1"]
        life_history_parent = donors["life_history_v1"]
        mutation_rate_parent = donors["mutation_rate_v1"]
        memory_parent = donors["memory_controller_architecture_v1"]

        hidden_size = int(getattr(memory_parent, "hidden_size"))
        exploration = float(getattr(behavioral_parent, "exploration_bias"))
        aggression = float(getattr(behavioral_parent, "aggression_bias"))
        temp = float(getattr(behavioral_parent, "action_temperature", 1.0))
        repro_th = float(getattr(life_history_parent, "reproduction_threshold"))
        mut_rate = float(getattr(mutation_rate_parent, "mutation_rate"))
        inertia = float(getattr(regulation_parent, "drive_inertia", 0.85))
        noise = float(getattr(regulation_parent, "drive_noise", 0.02))
        gain = float(getattr(regulation_parent, "drive_logit_gain", 1.0))
        pred_lr = float(getattr(regulation_parent, "pred_lr", 0.08))
        sp_hunger = float(getattr(regulation_parent, "drive_sp_hunger", 0.55))
        sp_pain = float(getattr(regulation_parent, "drive_sp_pain", 0.05))
        sp_fatigue = float(getattr(regulation_parent, "drive_sp_fatigue", 0.15))
        sp_fear = float(getattr(regulation_parent, "drive_sp_fear", 0.10))
        exp_strength = float(getattr(regulation_parent, "exp_prior_strength", 0.35))
        exp_noise = float(getattr(regulation_parent, "exp_prior_noise", 0.05))
        birth_noise = float(getattr(regulation_parent, "drive_birth_noise", 0.03))
    else:
        hidden_size = g1.hidden_size if rng.random() < 0.5 else g2.hidden_size
        exploration = 0.5 * (g1.exploration_bias + g2.exploration_bias)
        aggression = 0.5 * (g1.aggression_bias + g2.aggression_bias)
        repro_th = 0.5 * (g1.reproduction_threshold + g2.reproduction_threshold)
        mut_rate = 0.5 * (g1.mutation_rate + g2.mutation_rate)
        temp = 0.5 * (float(getattr(g1, "action_temperature", 1.0)) + float(getattr(g2, "action_temperature", 1.0)))

        # Identity Pressure genes (safe defaults if absent)
        inertia = 0.5 * (float(getattr(g1, "drive_inertia", 0.85)) + float(getattr(g2, "drive_inertia", 0.85)))
        noise = 0.5 * (float(getattr(g1, "drive_noise", 0.02)) + float(getattr(g2, "drive_noise", 0.02)))
        gain = 0.5 * (float(getattr(g1, "drive_logit_gain", 1.0)) + float(getattr(g2, "drive_logit_gain", 1.0)))
        pred_lr = 0.5 * (float(getattr(g1, "pred_lr", 0.08)) + float(getattr(g2, "pred_lr", 0.08)))

        # Cross-episode internal consistency genes (inherit via midpoint; safe defaults if absent)
        sp_hunger = 0.5 * (float(getattr(g1, "drive_sp_hunger", 0.55)) + float(getattr(g2, "drive_sp_hunger", 0.55)))
        sp_pain = 0.5 * (float(getattr(g1, "drive_sp_pain", 0.05)) + float(getattr(g2, "drive_sp_pain", 0.05)))
        sp_fatigue = 0.5 * (float(getattr(g1, "drive_sp_fatigue", 0.15)) + float(getattr(g2, "drive_sp_fatigue", 0.15)))
        sp_fear = 0.5 * (float(getattr(g1, "drive_sp_fear", 0.10)) + float(getattr(g2, "drive_sp_fear", 0.10)))
        exp_strength = 0.5 * (float(getattr(g1, "exp_prior_strength", 0.35)) + float(getattr(g2, "exp_prior_strength", 0.35)))
        exp_noise = 0.5 * (float(getattr(g1, "exp_prior_noise", 0.05)) + float(getattr(g2, "exp_prior_noise", 0.05)))
        birth_noise = 0.5 * (float(getattr(g1, "drive_birth_noise", 0.03)) + float(getattr(g2, "drive_birth_noise", 0.03)))
    resource_handling_enabled = (
        bool(getattr(sim_cfg, "resource_handling_representation_enable", False))
        if sim_cfg is not None
        else False
    )
    if diploid_inheritance:
        mutation_enabled = not bool(
            getattr(sim_cfg, "recomb72_mutation_off", False)
        ) and bool(getattr(sim_cfg, "diploid_mutation_enable", True))
        if _rng_semantics_version(sim_cfg) == 2:
            from engine.tensor_world.c5_rngv2_fullroute import (
                assemble_diploid_child_from_v2_v1,
            )

            identity = _v2_birth_identity(structural_identity_context)
            diploid_genotype, diploid_phenotype, diploid_receipt = (
                assemble_diploid_child_from_v2_v1(
                    g1,
                    g2,
                    world_uid=int(getattr(sim_cfg, "world_uid", 0) or 0),
                    absolute_tick=identity["tick"],
                    subject_uid=identity["child_agent_id"],
                    mutation_enabled=mutation_enabled,
                    resource_handling_enabled=resource_handling_enabled,
                    mutation_sigma_multiplier=weather_scalar_multiplier,
                )
            )
        else:
            diploid_draws = MEIOSIS_DRAW_COUNT + (
                MUTATION_SEMANTIC_DRAW_COUNT if mutation_enabled else 0
            )
            expected = rng_manager.expected_delta_for_boundary(
                DIPLOID_RNG_BUDGET_BOUNDARY,
                ctx={"diploid_draws": diploid_draws},
            )
            with rng_manager.budget_scope(
                DIPLOID_RNG_BUDGET_BOUNDARY,
                expected_delta=expected,
            ):
                diploid_genotype, diploid_phenotype, diploid_receipt = (
                    assemble_diploid_child(
                        g1,
                        g2,
                        rng,
                        mutation_enabled=mutation_enabled,
                        resource_handling_enabled=resource_handling_enabled,
                        mutation_sigma_multiplier=weather_scalar_multiplier,
                    )
                )
        exploration = diploid_phenotype["exploration_bias"]
        aggression = diploid_phenotype["aggression_bias"]
        temp = diploid_phenotype["action_temperature"]
        inertia = diploid_phenotype["drive_inertia"]
        noise = diploid_phenotype["drive_noise"]
        gain = diploid_phenotype["drive_logit_gain"]
        pred_lr = diploid_phenotype["pred_lr"]
        sp_hunger = diploid_phenotype["drive_sp_hunger"]
        sp_pain = diploid_phenotype["drive_sp_pain"]
        sp_fatigue = diploid_phenotype["drive_sp_fatigue"]
        sp_fear = diploid_phenotype["drive_sp_fear"]
        exp_strength = diploid_phenotype["exp_prior_strength"]
        exp_noise = diploid_phenotype["exp_prior_noise"]
        birth_noise = diploid_phenotype["drive_birth_noise"]
        repro_th = diploid_phenotype["reproduction_threshold"]
        mut_rate = diploid_phenotype["mutation_rate"]
        if genome_upgrade_enabled:
            parent_a_upgrade = (
                genome_upgrade_v2_from_genome(g1)
                or GenomeUpgradeV2V1.neutral()
            )
            parent_b_upgrade = (
                genome_upgrade_v2_from_genome(g2)
                or GenomeUpgradeV2V1.neutral()
            )
            mutation_rate_q10 = int(
                getattr(sim_cfg, "genome_upgrade_v2_mutation_rate_q10", 20)
            )
            mutation_step_q10 = int(
                getattr(sim_cfg, "genome_upgrade_v2_mutation_step_q10", 64)
            )
            if _rng_semantics_version(sim_cfg) == 2:
                from engine.tensor_world.c5_rngv2_fullroute import (
                    assemble_child_from_v2_v1,
                )

                identity = _v2_birth_identity(structural_identity_context)
                (
                    genome_upgrade_genotype,
                    genome_upgrade_receipt,
                ) = assemble_child_from_v2_v1(
                    parent_a_upgrade,
                    parent_b_upgrade,
                    world_uid=int(getattr(sim_cfg, "world_uid", 0) or 0),
                    absolute_tick=identity["tick"],
                    subject_uid=identity["child_agent_id"],
                    mutation_rate_q10=mutation_rate_q10,
                    mutation_step_q10=mutation_step_q10,
                )
            else:
                genome_upgrade_draws = (
                    GENOME_UPGRADE_V2_MEIOSIS_DRAW_COUNT
                    + GENOME_UPGRADE_V2_MUTATION_DRAW_COUNT
                )
                expected = rng_manager.expected_delta_for_boundary(
                    GENOME_UPGRADE_V2_RNG_BUDGET_BOUNDARY,
                    ctx={"genome_upgrade_draws": genome_upgrade_draws},
                )
                with rng_manager.budget_scope(
                    GENOME_UPGRADE_V2_RNG_BUDGET_BOUNDARY,
                    expected_delta=expected,
                ):
                    (
                        genome_upgrade_genotype,
                        genome_upgrade_receipt,
                    ) = assemble_genome_upgrade_child_from_rng_v1(
                        parent_a_upgrade,
                        parent_b_upgrade,
                        rng,
                        mutation_rate_q10=mutation_rate_q10,
                        mutation_step_q10=mutation_step_q10,
                    )
    resource_handling_axis_pre = RESOURCE_HANDLING_AXIS_NEUTRAL
    resource_handling_axis = RESOURCE_HANDLING_AXIS_NEUTRAL
    resource_handling_record: dict[str, Any] | None = None

    # mutate continuous genes. MUTRAD-1 scalar weather only multiplies sigma and
    # is default-off; mutation probability stays unchanged unless explicitly
    # enabled for diagnostics.
    scalar_mut_rate = float(np.clip(float(mut_rate) * float(weather_count_multiplier), 0.0, 1.0))
    if not diploid_inheritance:
        sigma, lo, hi = _MUT_SPECS["exploration_bias"]
        exploration = _mutate_float(exploration, rng, scalar_mut_rate, sigma=sigma * weather_scalar_multiplier, lo=lo, hi=hi)

        sigma, lo, hi = _MUT_SPECS["aggression_bias"]
        aggression = _mutate_float(aggression, rng, scalar_mut_rate, sigma=sigma * weather_scalar_multiplier, lo=lo, hi=hi)

        sigma, lo, hi = _MUT_SPECS["reproduction_threshold"]
        repro_th = _mutate_float(repro_th, rng, scalar_mut_rate, sigma=sigma * weather_scalar_multiplier, lo=lo, hi=hi)

        sigma, lo, hi = _MUT_SPECS["action_temperature"]
        temp = _mutate_float(temp, rng, scalar_mut_rate, sigma=sigma * weather_scalar_multiplier, lo=lo, hi=hi)

        sigma, lo, hi = _MUT_SPECS["drive_inertia"]
        inertia = _mutate_float(inertia, rng, scalar_mut_rate, sigma=sigma * weather_scalar_multiplier, lo=lo, hi=hi)

        sigma, lo, hi = _MUT_SPECS["drive_noise"]
        noise = _mutate_float(noise, rng, scalar_mut_rate, sigma=sigma * weather_scalar_multiplier, lo=lo, hi=hi)

        sigma, lo, hi = _MUT_SPECS["drive_logit_gain"]
        gain = _mutate_float(gain, rng, scalar_mut_rate, sigma=sigma * weather_scalar_multiplier, lo=lo, hi=hi)

        sigma, lo, hi = _MUT_SPECS["pred_lr"]
        pred_lr = _mutate_float(pred_lr, rng, scalar_mut_rate, sigma=sigma * weather_scalar_multiplier, lo=lo, hi=hi)

    if diploid_inheritance:
        pass
    elif recomb72_audit and bool(
        getattr(sim_cfg, "recomb72_mutation_off", False)
    ):
        # The current arm must traverse the normal production inheritance
        # implementation while the campaign's explicit mutation-off boundary
        # prevents mutation-rate drift.
        mut_rate = 0.0
    else:
        mut_rate = _mutate_mutation_rate(mut_rate, rng)

    if resource_handling_enabled:
        parent_a_axis = clamp_resource_handling_axis(
            getattr(g1, RESOURCE_HANDLING_AXIS_FIELD, RESOURCE_HANDLING_AXIS_NEUTRAL)
        )
        parent_b_axis = clamp_resource_handling_axis(
            getattr(g2, RESOURCE_HANDLING_AXIS_FIELD, RESOURCE_HANDLING_AXIS_NEUTRAL)
        )
        if diploid_inheritance:
            resource_handling_axis_pre = diploid_phenotype[
                "resource_handling_axis"
            ]
        else:
            resource_handling_axis_pre = clamp_resource_handling_axis(
                0.5 * (float(parent_a_axis) + float(parent_b_axis))
            )
        if diploid_inheritance:
            resource_handling_axis = resource_handling_axis_pre
            resource_handling_record = {
                "schema_version": "diploid_resource_handling_projection_v1",
                "mutated": any(
                    event["locus_id"] == "resource_handling_axis"
                    and event["semantic_change"]
                    for event in diploid_receipt["mutation"]["events"]
                ),
            }
        else:
            resource_handling_axis, resource_handling_record = mutate_resource_handling_axis(
                resource_handling_axis_pre,
                rng,
                mutation_rate=scalar_mut_rate,
                sigma=RESOURCE_HANDLING_MUTATION_SIGMA * weather_scalar_multiplier,
            )
        resource_handling_record.update(
            {
                "parent_a_axis": float(parent_a_axis),
                "parent_b_axis": float(parent_b_axis),
                "child_pre_mutation_axis": float(resource_handling_axis_pre),
                "child_post_mutation_axis": float(resource_handling_axis),
            }
        )

    if flat_linkage_component:
        memory_dim = _mutate_memory_dim_step(
            int(getattr(linkage_context["donors"]["memory_controller_architecture_v1"], "memory_dim", 0)),
            rng,
            mut_rate,
        )
    else:
        memory_dim = _inherit_memory_dim(g1, g2, rng, mut_rate)
    use_gru = bool(memory_dim > 0)

    preserve_basic_sensor = bool(getattr(sim_cfg, "sensor_channel_preserve_basic_channel", True)) if sim_cfg is not None else True
    if bool(getattr(sim_cfg, "enable_evolvable_sensor_channel_mask", False)) if sim_cfg is not None else False:
        sensor_channel_mask = recombine_sensor_channel_mask(
            getattr(g1, "sensor_channel_mask", implemented_sensor_channel_mask()),
            getattr(g2, "sensor_channel_mask", implemented_sensor_channel_mask()),
            rng,
            preserve_basic=preserve_basic_sensor,
        )
        sensor_channel_mask = mutate_sensor_channel_mask(
            sensor_channel_mask,
            rng,
            float(getattr(sim_cfg, "sensor_channel_mask_mutation_rate", 0.02)),
            preserve_basic=preserve_basic_sensor,
        )
    else:
        sensor_channel_mask = sanitize_sensor_channel_mask(
            getattr(g1, "sensor_channel_mask", implemented_sensor_channel_mask()),
            preserve_basic=preserve_basic_sensor,
        )

    try:
        max_sensor_radius = int(getattr(getattr(sim_cfg, "env_cfg", None), "view_radius", 0) or 0)
    except Exception:
        max_sensor_radius = 0
    max_sensor_radius = max_sensor_radius if max_sensor_radius > 0 else None
    if bool(getattr(sim_cfg, "enable_evolvable_sensor_radius", False)) if sim_cfg is not None else False:
        sensor_radius = recombine_sensor_radius(
            getattr(g1, "sensor_radius", DEFAULT_SENSOR_RADIUS),
            getattr(g2, "sensor_radius", DEFAULT_SENSOR_RADIUS),
            rng,
            max_radius=max_sensor_radius,
        )
        sensor_radius = mutate_sensor_radius(
            sensor_radius,
            rng,
            float(getattr(sim_cfg, "sensor_radius_mutation_rate", 0.05)),
            max_radius=max_sensor_radius,
        )
    else:
        sensor_radius = snap_sensor_radius(getattr(g1, "sensor_radius", DEFAULT_SENSOR_RADIUS), max_radius=max_sensor_radius)

    if bool(getattr(sim_cfg, "enable_evolvable_introspection_channels", False)) if sim_cfg is not None else False:
        if bool(getattr(sim_cfg, "enable_introspection_channel_recombination", False)):
            introspection_channel_mask = recombine_introspection_channel_mask(
                getattr(g1, "introspection_channel_mask", implemented_introspection_channel_mask()),
                getattr(g2, "introspection_channel_mask", implemented_introspection_channel_mask()),
                rng,
            )
        else:
            introspection_channel_mask = sanitize_introspection_channel_mask(
                getattr(g1, "introspection_channel_mask", implemented_introspection_channel_mask())
            )
        if bool(getattr(sim_cfg, "enable_introspection_channel_mutation", False)):
            introspection_channel_mask = mutate_introspection_channel_mask(
                introspection_channel_mask,
                rng,
                float(getattr(sim_cfg, "introspection_channel_mask_mutation_rate", 0.02)),
            )
    else:
        introspection_channel_mask = sanitize_introspection_channel_mask(
            getattr(g1, "introspection_channel_mask", implemented_introspection_channel_mask())
        )

    if bool(getattr(sim_cfg, "enable_evolvable_self_regulation_controls", False)) if sim_cfg is not None else False:
        if bool(getattr(sim_cfg, "enable_self_regulation_control_recombination", False)):
            self_regulation_control_mask = recombine_self_regulation_control_mask(
                getattr(g1, "self_regulation_control_mask", implemented_self_regulation_control_mask()),
                getattr(g2, "self_regulation_control_mask", implemented_self_regulation_control_mask()),
                rng,
            )
        else:
            self_regulation_control_mask = sanitize_self_regulation_control_mask(
                getattr(g1, "self_regulation_control_mask", implemented_self_regulation_control_mask())
            )
        if bool(getattr(sim_cfg, "enable_self_regulation_control_mutation", False)):
            self_regulation_control_mask = mutate_self_regulation_control_mask(
                self_regulation_control_mask,
                rng,
                float(getattr(sim_cfg, "self_regulation_control_mask_mutation_rate", 0.02)),
            )
    else:
        self_regulation_control_mask = sanitize_self_regulation_control_mask(
            getattr(g1, "self_regulation_control_mask", implemented_self_regulation_control_mask())
        )

    if bool(getattr(sim_cfg, "enable_evolvable_self_modification_targets", False)) if sim_cfg is not None else False:
        if bool(getattr(sim_cfg, "enable_self_modification_target_recombination", False)):
            self_modification_target_mask = recombine_self_modification_target_mask(
                getattr(g1, "self_modification_target_mask", implemented_self_modification_target_mask()),
                getattr(g2, "self_modification_target_mask", implemented_self_modification_target_mask()),
                rng,
            )
        else:
            self_modification_target_mask = sanitize_self_modification_target_mask(
                getattr(g1, "self_modification_target_mask", implemented_self_modification_target_mask())
            )
        if bool(getattr(sim_cfg, "enable_self_modification_target_mutation", False)):
            self_modification_target_mask = mutate_self_modification_target_mask(
                self_modification_target_mask,
                rng,
                float(getattr(sim_cfg, "self_modification_target_mask_mutation_rate", 0.02)),
            )
    else:
        self_modification_target_mask = sanitize_self_modification_target_mask(
            getattr(g1, "self_modification_target_mask", implemented_self_modification_target_mask())
        )

    if bool(getattr(sim_cfg, "enable_evolvable_directed_inheritance_channels", False)) if sim_cfg is not None else False:
        if bool(getattr(sim_cfg, "enable_directed_inheritance_channel_recombination", False)):
            directed_inheritance_channel_mask = recombine_directed_inheritance_channel_mask(
                getattr(g1, "directed_inheritance_channel_mask", implemented_directed_inheritance_channel_mask()),
                getattr(g2, "directed_inheritance_channel_mask", implemented_directed_inheritance_channel_mask()),
                rng,
            )
        else:
            directed_inheritance_channel_mask = sanitize_directed_inheritance_channel_mask(
                getattr(g1, "directed_inheritance_channel_mask", implemented_directed_inheritance_channel_mask())
            )
        if bool(getattr(sim_cfg, "enable_directed_inheritance_channel_mutation", False)):
            directed_inheritance_channel_mask = mutate_directed_inheritance_channel_mask(
                directed_inheritance_channel_mask,
                rng,
                float(getattr(sim_cfg, "directed_inheritance_channel_mask_mutation_rate", 0.02)),
            )
    else:
        directed_inheritance_channel_mask = sanitize_directed_inheritance_channel_mask(
            getattr(g1, "directed_inheritance_channel_mask", implemented_directed_inheritance_channel_mask())
        )

    dp1 = canonicalize_developmental_program(getattr(g1, "developmental_program", None))
    dp2 = canonicalize_developmental_program(getattr(g2, "developmental_program", None))
    s1 = json.dumps(dp1, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    s2 = json.dumps(dp2, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    child_dev_program = dp1 if s1 <= s2 else dp2

    family_head_residuals: dict[str, Any] | None = None
    family_head_residual_record: dict[str, Any] | None = None
    family_head_residual_enabled = (
        bool(getattr(sim_cfg, "policy_family_head_residual_genome_enable", False))
        if sim_cfg is not None
        else False
    )
    if family_head_residual_enabled:
        parent_a_block = _family_head_residual_block_from_genome_like(g1)
        parent_b_block = _family_head_residual_block_from_genome_like(g2)
        if flat_linkage_component:
            family_parent = linkage_context["donors"]["family_head_strategy_v1"]
            donor_block = _family_head_residual_block_from_genome_like(family_parent)
            recombination_record = {
                "child": donor_block or canonical_family_head_residual_block(None),
            }
        else:
            recombination_record = recombine_family_head_residual_blocks(parent_a_block, parent_b_block)
        family_head_residuals = dict(recombination_record["child"])
        mutation_event = None
        if bool(getattr(sim_cfg, "policy_family_head_residual_mutation_enable", False)):
            step = float(getattr(sim_cfg, "policy_family_head_residual_mutation_step", 0.1) or 0.1)
            family_head_residuals, mutation_event = mutate_family_head_residual_block(
                family_head_residuals,
                rng,
                step=step,
            )
        family_head_residual_record = {
            "schema_version": "family_head_residual_reproduction_record_v1",
            "representation_enabled": True,
            "mutation_enabled": bool(getattr(sim_cfg, "policy_family_head_residual_mutation_enable", False)),
            "operator_id": FAMILY_HEAD_RESIDUAL_MUTATION_OPERATOR_ID if mutation_event is not None else "",
            "parent_a_block": parent_a_block or canonical_family_head_residual_block(None),
            "parent_b_block": parent_b_block or canonical_family_head_residual_block(None),
            "post_recombination_block": dict(recombination_record["child"]),
            "mutation_event": mutation_event,
            "final_block": dict(family_head_residuals),
            "canonical_digest": family_head_residual_canonical_digest(family_head_residuals),
            "semantic_digest": family_head_residual_semantic_digest(family_head_residuals),
            "draw_budget": {"mutation.py": 2 if mutation_event is not None else 0},
        }

    c2_enable = bool(getattr(sim_cfg, "genome_c2_enable", True)) if sim_cfg is not None else True
    c1_enable = bool(getattr(sim_cfg, "genome_c1_enable", True)) if sim_cfg is not None else True
    c2_payload_passthrough = (
        bool(getattr(sim_cfg, "cxxrewrite_reproduction_c2_payload_passthrough_enable", False))
        if sim_cfg is not None
        else False
    )
    c1_trusted_c2_payload = (
        bool(getattr(sim_cfg, "cxxrewrite_reproduction_c1_trusted_c2_payload_enable", False))
        if sim_cfg is not None
        else False
    )
    c2_trusted_parent_payloads = (
        bool(getattr(sim_cfg, "cxxrewrite_reproduction_c2_trusted_parent_payloads_enable", False))
        if sim_cfg is not None
        else False
    )
    runnercore4_structural = bool(_runnercore4_structural_enabled(sim_cfg, g1, g2) and c2_enable and c1_enable)
    direct_child_payload = (
        bool(getattr(sim_cfg, "cxxrewrite_reproduction_direct_child_payload_enable", False))
        if sim_cfg is not None
        else False
    )
    direct_child_payload = bool(
        direct_child_payload
        and c2_enable
        and c1_enable
        and c2_payload_passthrough
        and c1_trusted_c2_payload
        and not runnercore4_structural
    )
    runnercore11_typed_projection = bool(
        return_typed_projection
        and runnercore4_structural
        and sim_cfg is not None
        and bool(getattr(sim_cfg, "runnercore11_c1_typed_rows_enable", False))
    )
    runnercore12_digest_reuse = bool(
        runnercore11_typed_projection
        and sim_cfg is not None
        and bool(getattr(sim_cfg, "runnercore12_c1_payload_digest_reuse_enable", False))
    )
    runnercore13_projection_sample_only = bool(
        runnercore11_typed_projection
        and sim_cfg is not None
        and bool(getattr(sim_cfg, "runnercore13_c1_projection_digest_sample_only_enable", False))
    )
    runnercore15_noop_normalize_skip = bool(
        runnercore11_typed_projection
        and sim_cfg is not None
        and bool(getattr(sim_cfg, "runnercore15_c1_noop_normalize_skip_enable", False))
    )
    runnercore16_empty_structure_fastpath = bool(
        runnercore15_noop_normalize_skip
        and sim_cfg is not None
        and bool(getattr(sim_cfg, "runnercore16_c1_empty_structure_fastpath_enable", False))
    )
    if return_typed_projection and not runnercore11_typed_projection:
        _runnercore11_required_failure(sim_cfg, "runnercore4_structural_c1_payload")
        if bool(getattr(sim_cfg, "runnercore11_fail_closed_on_unsupported", False)):
            raise RuntimeError("RUNNERCORE11-UNSUPPORTED: typed C1 projection requires structural C1 payload")
    if runnercore4_structural:
        _runnercore4_stat_add(sim_cfg, "child_construction_calls", 1)
        _runnercore4_stat_add(sim_cfg, "intermediate_child_genome_construction_avoided", 1)
        _runnercore4_stat_add(sim_cfg, "child_genome_to_dict_avoided", 1)

    child_kwargs = {
        "hidden_size": int(hidden_size),
        "exploration_bias": float(exploration),
        "aggression_bias": float(aggression),
        "reproduction_threshold": float(repro_th),
        "mutation_rate": float(mut_rate),
        "action_temperature": float(temp),
        "use_gru": bool(use_gru),
        "memory_dim": int(memory_dim),
        "sensor_channel_mask": int(sensor_channel_mask),
        "sensor_radius": int(sensor_radius),
        "introspection_channel_mask": int(introspection_channel_mask),
        "self_regulation_control_mask": int(self_regulation_control_mask),
        "self_modification_target_mask": int(self_modification_target_mask),
        "directed_inheritance_channel_mask": int(directed_inheritance_channel_mask),
        "drive_inertia": float(inertia),
        "drive_noise": float(noise),
        "drive_logit_gain": float(gain),
        "pred_lr": float(pred_lr),
        "drive_sp_hunger": float(sp_hunger),
        "drive_sp_pain": float(sp_pain),
        "drive_sp_fatigue": float(sp_fatigue),
        "drive_sp_fear": float(sp_fear),
        "exp_prior_strength": float(exp_strength),
        "exp_prior_noise": float(exp_noise),
        "drive_birth_noise": float(birth_noise),
        "resource_handling_axis": float(resource_handling_axis),
        "family_head_residuals": family_head_residuals,
        "developmental_program": child_dev_program,
        "b2_caps": dict(parent_caps),
    }
    child: Optional[Genome]
    child_base_payload: Optional[dict[str, Any]]
    if direct_child_payload:
        child = None
        child_base_payload = _direct_child_payload_for_canonical_boundary(
            hidden_size=int(hidden_size),
            exploration=float(exploration),
            aggression=float(aggression),
            repro_th=float(repro_th),
            mut_rate=float(mut_rate),
            temp=float(temp),
            use_gru=bool(use_gru),
            memory_dim=int(memory_dim),
            sensor_channel_mask=int(sensor_channel_mask),
            sensor_radius=int(sensor_radius),
            introspection_channel_mask=int(introspection_channel_mask),
            self_regulation_control_mask=int(self_regulation_control_mask),
            self_modification_target_mask=int(self_modification_target_mask),
            directed_inheritance_channel_mask=int(directed_inheritance_channel_mask),
            inertia=float(inertia),
            noise=float(noise),
            gain=float(gain),
            pred_lr=float(pred_lr),
            sp_hunger=float(sp_hunger),
            sp_pain=float(sp_pain),
            sp_fatigue=float(sp_fatigue),
            sp_fear=float(sp_fear),
            exp_strength=float(exp_strength),
            exp_noise=float(exp_noise),
            birth_noise=float(birth_noise),
            resource_handling_axis=float(resource_handling_axis),
            family_head_residuals=family_head_residuals,
            developmental_program=child_dev_program,
            b2_caps=parent_caps,
        )
    else:
        child_base_payload = None
        child = None if runnercore4_structural else Genome(**child_kwargs)
    child_payload: Optional[dict[str, Any]] = None
    if b is not None:
        _add_timing_bucket(b, "make_child_genome_base", _t_base0)

    if c2_enable:
        align_key = str(getattr(sim_cfg, "genome_c2_alignment_key", C2_FROZEN_ALIGNMENT_KEY) or C2_FROZEN_ALIGNMENT_KEY) if sim_cfg is not None else C2_FROZEN_ALIGNMENT_KEY
        unmatched_policy = str(getattr(sim_cfg, "genome_c2_unmatched_policy", C2_FROZEN_UNMATCHED_POLICY) or C2_FROZEN_UNMATCHED_POLICY) if sim_cfg is not None else C2_FROZEN_UNMATCHED_POLICY
        self_fast = g1 is g2 and (
            bool(getattr(sim_cfg, "runnercore3_self_recombination_fastpath_enable", False))
            or runnercore4_structural
        )
        if self_fast:
            if runnercore4_structural:
                parent_payload = _runnercore4_cached_parent_payload(g1, sim_cfg)
                parent_payload_is_canonical = True
                skip_output_canonicalize = True
                _runnercore4_stat_add(sim_cfg, "c2_parent_payload_canonicalization_avoided", 1)
                _runnercore4_stat_add(sim_cfg, "c2_output_payload_canonicalization_avoided", 1)
            else:
                parent_payload = getattr(g1, "_runnercore3_self_payload_dict_cache", None)
                if not isinstance(parent_payload, dict):
                    parent_payload = g1.to_dict()
                    try:
                        setattr(g1, "_runnercore3_self_payload_dict_cache", parent_payload)
                    except Exception:
                        pass
                parent_payload_is_canonical = False
                skip_output_canonicalize = False
            t_c2 = time.perf_counter() if runnercore4_structural else 0.0
            rec_payload, _summary = recombine_self_lineage_payload(
                parent_payload=parent_payload,
                rng=rng,
                rng_manager=rng_manager,
                alignment_key=align_key,
                unmatched_policy=unmatched_policy,
                parent_payload_is_canonical=parent_payload_is_canonical,
                skip_output_canonicalize=skip_output_canonicalize,
                inheritance_profile=inheritance_profile,
                sustainable_growth_enabled=bool(
                    getattr(sim_cfg, "structural_sustainable_growth_enable", False)
                ),
            )
            if runnercore4_structural:
                _runnercore4_bucket_add(buckets, "runnercore4_c2_self_payload", time.perf_counter() - t_c2)
                _runnercore4_stat_add(sim_cfg, "c2_rng_budget_checks", 1)
        else:
            _t_c2_recombine0 = time.perf_counter() if b is not None else 0.0
            if c2_trusted_parent_payloads:
                parent_a_payload = _cached_canonical_payload_for_trusted_boundary(g1)
                parent_b_payload = _cached_canonical_payload_for_trusted_boundary(g2)
            else:
                parent_a_payload = _genome_payload_for_canonical_boundary(g1)
                parent_b_payload = _genome_payload_for_canonical_boundary(g2)
            rec_payload, _summary = recombine_lineage_payloads(
                parent_a_payload=parent_a_payload,
                parent_b_payload=parent_b_payload,
                rng=rng,
                rng_manager=rng_manager,
                alignment_key=align_key,
                unmatched_policy=unmatched_policy,
                trusted_canonical_parent_payloads=bool(c2_trusted_parent_payloads),
                inheritance_profile=inheritance_profile,
                sustainable_growth_enabled=bool(
                    getattr(sim_cfg, "structural_sustainable_growth_enable", False)
                ),
            )
            if b is not None:
                _add_timing_bucket(b, "make_child_genome_c2_recombine", _t_c2_recombine0)
        # Keep scalar genes evolved above while replacing structural substrate with lineage-aware recombination.
        c2g = dict(rec_payload.get("genes", {})) if isinstance(rec_payload.get("genes", {}), dict) else {}
        c2_keys = (
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
            "c2_parent_order_rule",
            "c2_legacy_unresolved_modules",
        )
        if flat_linkage_component:
            c2_keys += (
                "c2_inheritance_profile",
                "c2_structural_linkage_rule",
                "c2_linked_component_count",
            )
        if self_fast:
            if runnercore4_structural:
                t_payload = time.perf_counter()
                child_payload = _runnercore4_child_payload_from_parts(
                    child_kwargs=child_kwargs,
                    rec_payload=rec_payload,
                    c2_genes=c2g,
                )
                _runnercore4_bucket_add(buckets, "runnercore4_child_payload_build", time.perf_counter() - t_payload)
            else:
                assert child is not None
                child.modules = list(rec_payload.get("modules", []))
                child.connections = list(rec_payload.get("connections", []))
                for k in c2_keys:
                    setattr(child, k, c2g.get(k, getattr(child, k, 0)))
        else:
            _t_c2_payload0 = time.perf_counter() if b is not None else 0.0
            cp = (
                dict(child_base_payload)
                if child_base_payload is not None
                else _genome_payload_for_canonical_boundary(child)
            )
            cp["modules"] = list(rec_payload.get("modules", []))
            cp["connections"] = list(rec_payload.get("connections", []))
            cp["b2_caps"] = dict(rec_payload.get("b2_caps", parent_caps) or parent_caps)
            for k in c2_keys:
                cp[k] = c2g.get(k, cp.get(k, 0))
            if b is not None:
                _add_timing_bucket(b, "make_child_genome_c2_payload_merge", _t_c2_payload0)
            if c1_enable and c2_payload_passthrough:
                child_payload = cp
            else:
                _t_c2_from_dict0 = time.perf_counter() if b is not None else 0.0
                child = Genome.from_dict(cp)
                if b is not None:
                    _add_timing_bucket(b, "make_child_genome_c2_from_dict", _t_c2_from_dict0)

    if not c1_enable:
        if child is None:
            out = Genome.from_dict(child_payload or child_base_payload or {})
            _attach_gen1_f1_canonical_stages(
                out,
                sim_cfg,
                pre_c1_payload=child_payload or child_base_payload or {},
                post_c1_payload=child_payload or child_base_payload or {},
            )
            _attach_resource_handling_record(out, sim_cfg, resource_handling_record)
            _attach_family_head_residual_record(out, sim_cfg, family_head_residual_record)
            attach_mutation_weather_metadata(out, weather)
            if _rep2m_reporting_enabled(sim_cfg):
                _rep2m_attach_variant_event_join_metadata(
                    out,
                    post_c2_payload=child_payload or child_base_payload or {},
                    final_payload=child_payload or child_base_payload or {},
                    trace_records=[],
                )
            _attach_transient_linkage_context(out, linkage_context)
            _attach_diploid_context(
                out,
                diploid_genotype,
                diploid_receipt,
            )
            _attach_genome_upgrade_v2_context(
                out,
                genome_upgrade_genotype,
                genome_upgrade_receipt,
            )
            return out
        no_c1_payload = _genome_payload_for_canonical_boundary(child)
        _attach_gen1_f1_canonical_stages(
            child,
            sim_cfg,
            pre_c1_payload=no_c1_payload,
            post_c1_payload=no_c1_payload,
        )
        _attach_resource_handling_record(child, sim_cfg, resource_handling_record)
        _attach_family_head_residual_record(child, sim_cfg, family_head_residual_record)
        attach_mutation_weather_metadata(child, weather)
        if _rep2m_reporting_enabled(sim_cfg):
            _rep2m_attach_variant_event_join_metadata(
                child,
                post_c2_payload=_genome_payload_for_canonical_boundary(child),
                final_payload=_genome_payload_for_canonical_boundary(child),
                trace_records=[],
            )
        _attach_transient_linkage_context(child, linkage_context)
        _attach_diploid_context(
            child,
            diploid_genotype,
            diploid_receipt,
        )
        _attach_genome_upgrade_v2_context(
            child,
            genome_upgrade_genotype,
            genome_upgrade_receipt,
        )
        return child
    max_ops = int(getattr(sim_cfg, "genome_c1_max_ops_per_offspring", C1_FROZEN_MAX_OPS_PER_OFFSPRING)) if sim_cfg is not None else C1_FROZEN_MAX_OPS_PER_OFFSPRING
    invalid_policy = str(getattr(sim_cfg, "genome_c1_invalid_operator_policy", C1_FROZEN_INVALID_OPERATOR_POLICY) or C1_FROZEN_INVALID_OPERATOR_POLICY) if sim_cfg is not None else C1_FROZEN_INVALID_OPERATOR_POLICY
    _t_c1_pipeline0 = time.perf_counter() if b is not None else 0.0
    pipeline_payload = child_payload if child_payload is not None else _genome_payload_for_canonical_boundary(child)
    t_c1 = time.perf_counter() if runnercore4_structural else 0.0
    runnercore5_c1_canonical_fastpath = bool(
        runnercore4_structural
        and child_payload is not None
        and sim_cfg is not None
        and getattr(sim_cfg, "runnercore5_c1_canonical_boundary_fastpath_enable", False)
    )
    if runnercore5_c1_canonical_fastpath:
        _runnercore5_stat_add(sim_cfg, "c1_canonical_boundary_fastpath_calls", 1)
    cxx_c1_canonical_fastpath = bool(
        child_payload is not None
        and c1_trusted_c2_payload
        and c2_payload_passthrough
        and c1_enable
        and c2_enable
        and max_ops > 0
        and sim_cfg is not None
        and getattr(sim_cfg, "cxxrewrite_reproduction_c1_final_normalize_skip_enable", False)
    )
    cxx_c1_noop_normalize_skip = bool(
        cxx_c1_canonical_fastpath
        and sim_cfg is not None
        and getattr(sim_cfg, "cxxrewrite_reproduction_c1_noop_normalize_skip_enable", False)
    )
    cxx_c1_empty_structure_fastpath = bool(
        cxx_c1_noop_normalize_skip
        and sim_cfg is not None
        and getattr(sim_cfg, "cxxrewrite_reproduction_c1_empty_structure_fastpath_enable", False)
    )
    runnercore9_c1_scalar_skip = bool(
        runnercore4_structural
        and sim_cfg is not None
        and getattr(sim_cfg, "runnercore9_c1_scalar_op_normalize_skip_enable", False)
    )
    runnercore9_trace_records: list[dict[str, Any]] | None = (
        []
        if (
            runnercore4_structural
            and sim_cfg is not None
            and getattr(sim_cfg, "runnercore9_c1_trace_enable", False)
        )
        else None
    )
    rep2m_event_join_trace_records: list[dict[str, Any]] | None = [] if _rep2m_reporting_enabled(sim_cfg) else None
    c1_trace_records = runnercore9_trace_records if runnercore9_trace_records is not None else rep2m_event_join_trace_records
    c1_payload, _telemetry = apply_operator_pipeline(
        payload=pipeline_payload,
        rng=rng,
        rng_manager=rng_manager,
        max_ops=max_ops,
        invalid_operator_policy=invalid_policy,
        payload_is_canonical=bool(
            runnercore5_c1_canonical_fastpath
            or (c1_trusted_c2_payload and child_payload is not None)
        ),
        skip_final_normalize=bool(runnercore5_c1_canonical_fastpath or cxx_c1_canonical_fastpath),
        skip_scalar_op_normalize=runnercore9_c1_scalar_skip,
        skip_noop_normalize=bool(runnercore15_noop_normalize_skip or cxx_c1_noop_normalize_skip),
        empty_structure_fastpath=bool(runnercore16_empty_structure_fastpath or cxx_c1_empty_structure_fastpath),
        empty_structure_fastpath_fail_closed=bool(
            sim_cfg is not None
            and (
                getattr(sim_cfg, "runnercore16_c1_empty_structure_fastpath_fail_closed", False)
                or getattr(sim_cfg, "cxxrewrite_reproduction_c1_empty_structure_fastpath_fail_closed", False)
            )
        ),
        trace_records=c1_trace_records,
        trace_payload_identity_fn=mutation_trace_identity if c1_trace_records is not None else None,
        emit_final_payload_digest=runnercore12_digest_reuse,
        structural_bootstrap_enable=bool(
            getattr(sim_cfg, "structural_bootstrap_enable", False)
        ) if sim_cfg is not None else False,
        structural_identity_context=dict(structural_identity_context or {}),
        observation_routing_mutation_enable=bool(
            getattr(sim_cfg, "structural_observation_routing_enable", False)
        ) if sim_cfg is not None else False,
        observation_routing_operator_attempts=int(
            getattr(sim_cfg, "structural_observation_routing_operator_attempts", 0) or 0
        ) if sim_cfg is not None else 0,
        output_routing_mutation_enable=bool(
            getattr(sim_cfg, "structural_output_routing_enable", False)
        ) if sim_cfg is not None else False,
        output_routing_operator_attempts=int(
            getattr(sim_cfg, "structural_output_routing_operator_attempts", 0) or 0
        ) if sim_cfg is not None else 0,
        structural_development_mutation_enable=bool(
            getattr(sim_cfg, "structural_development_enable", False)
        ) if sim_cfg is not None else False,
        structural_development_operator_attempts=int(
            getattr(sim_cfg, "structural_development_operator_attempts", 0) or 0
        ) if sim_cfg is not None else 0,
        structural_gated_recurrent_mutation_enable=bool(
            getattr(sim_cfg, "structural_gated_recurrent_enable", False)
        ) if sim_cfg is not None else False,
        structural_gated_recurrent_operator_attempts=int(
            getattr(sim_cfg, "structural_gated_recurrent_operator_attempts", 0) or 0
        ) if sim_cfg is not None else 0,
        structural_sustainable_growth_enable=bool(
            getattr(sim_cfg, "structural_sustainable_growth_enable", False)
        ) if sim_cfg is not None else False,
        structural_sustainable_growth_operator_attempts=int(
            getattr(sim_cfg, "structural_sustainable_growth_operator_attempts", 0) or 0
        ) if sim_cfg is not None else 0,
    )
    if runnercore9_c1_scalar_skip or runnercore9_trace_records is not None:
        _runnercore9_stat_add(sim_cfg, "c1_operator_pipeline_calls", 1)
        _runnercore9_stat_add(sim_cfg, "c1_scalar_normalize_skipped", int(_telemetry.get("scalar_normalize_skipped", 0) or 0))
        _runnercore9_stat_add(sim_cfg, "c1_normalize_calls", int(_telemetry.get("normalize_calls", 0) or 0))
        for op_id, count in dict(_telemetry.get("usage_counts", {}) or {}).items():
            _runnercore9_stat_add(sim_cfg, f"c1_op_count__{op_id}", int(count or 0))
        if runnercore9_trace_records is not None:
            _runnercore9_trace_extend(sim_cfg, runnercore9_trace_records)
    if runnercore15_noop_normalize_skip:
        _runnercore11_stat_add(sim_cfg, "runnercore15_c1_noop_normalize_skip_enabled", 1)
        _runnercore11_stat_add(
            sim_cfg,
            "runnercore15_c1_noop_normalize_skipped",
            int(_telemetry.get("noop_normalize_skipped", 0) or 0),
        )
    if runnercore16_empty_structure_fastpath:
        _runnercore11_stat_add(sim_cfg, "runnercore16_c1_empty_structure_fastpath_enabled", 1)
        _runnercore11_stat_add(
            sim_cfg,
            "runnercore16_c1_empty_structure_fastpath_hits",
            int(_telemetry.get("empty_structure_fastpath_hits", 0) or 0),
        )
        misses = int(_telemetry.get("empty_structure_fastpath_misses", 0) or 0)
        _runnercore11_stat_add(sim_cfg, "runnercore16_c1_empty_structure_fastpath_misses", misses)
        _runnercore11_stat_add(
            sim_cfg,
            "runnercore16_c1_empty_structure_fastpath_noop_draws",
            sum(int(v or 0) for v in dict(_telemetry.get("usage_counts", {}) or {}).values()),
        )
        if misses:
            _runnercore11_stat_add(sim_cfg, "unsupported_fallback_count", misses)
    if b is not None:
        _add_timing_bucket(b, "make_child_genome_c1_pipeline", _t_c1_pipeline0)
    if runnercore4_structural:
        _runnercore4_bucket_add(buckets, "runnercore4_c1_operator_pipeline", time.perf_counter() - t_c1)
        _runnercore4_stat_add(sim_cfg, "c1_rng_budget_checks", 1)
    if runnercore11_typed_projection:
        _runnercore11_stat_add(sim_cfg, "c1_operator_pipeline_calls", 1)
        if runnercore12_digest_reuse:
            _runnercore11_stat_add(sim_cfg, "runnercore12_c1_payload_digest_reuse_enabled", 1)
        emit_projection_digest = True
        if runnercore13_projection_sample_only:
            _runnercore11_stat_add(sim_cfg, "runnercore13_c1_projection_digest_sample_only_enabled", 1)
            emit_projection_digest = _runnercore11_trace_sample_pending(sim_cfg)
        row = RunnerCoreC1TypedGenomeRow.from_canonical_payload(
            c1_payload,
            sim_cfg=sim_cfg,
            c1_payload_hash=str(_telemetry.get("final_payload_digest", "") or "") if runnercore12_digest_reuse else None,
            emit_projection_digest=emit_projection_digest,
        )
        _attach_gen1_f1_canonical_stages(
            row,
            sim_cfg,
            pre_c1_payload=pipeline_payload,
            post_c1_payload=c1_payload,
        )
        if _rep2m_reporting_enabled(sim_cfg):
            _rep2m_attach_variant_event_join_metadata(
                row,
                post_c2_payload=pipeline_payload,
                final_payload=c1_payload,
                trace_records=c1_trace_records,
            )
        _runnercore11_stat_add(sim_cfg, "c1_typed_rows_created", 1)
        _runnercore11_stat_add(sim_cfg, "genome_from_dict_avoided", 1)
        _runnercore11_stat_add(sim_cfg, "genome_materialization_avoided", 1)
        _runnercore11_trace_projection(sim_cfg, row)
        _attach_resource_handling_record(row, sim_cfg, resource_handling_record)
        _attach_family_head_residual_record(row, sim_cfg, family_head_residual_record)
        attach_mutation_weather_metadata(row, weather)
        _attach_transient_linkage_context(row, linkage_context)
        _attach_diploid_context(
            row,
            diploid_genotype,
            diploid_receipt,
        )
        _attach_genome_upgrade_v2_context(
            row,
            genome_upgrade_genotype,
            genome_upgrade_receipt,
        )
        return row
    t_from = time.perf_counter() if runnercore4_structural else 0.0
    _t_c1_from_dict0 = time.perf_counter() if b is not None else 0.0
    runnercore6_trusted_genome_materialization = bool(
        runnercore4_structural
        and sim_cfg is not None
        and getattr(sim_cfg, "runnercore6_trusted_canonical_genome_materialization_enable", False)
    )
    if runnercore6_trusted_genome_materialization:
        out = Genome.from_trusted_canonical_payload(c1_payload)
        _runnercore5_stat_add(sim_cfg, "runnercore6_trusted_genome_materialization_calls", 1)
    else:
        out = Genome.from_dict(c1_payload)
    _attach_gen1_f1_canonical_stages(
        out,
        sim_cfg,
        pre_c1_payload=pipeline_payload,
        post_c1_payload=c1_payload,
    )
    if b is not None:
        _add_timing_bucket(b, "make_child_genome_c1_from_dict", _t_c1_from_dict0)
    if runnercore4_structural:
        _runnercore4_bucket_add(buckets, "runnercore4_genome_from_c1_payload", time.perf_counter() - t_from)
    _attach_resource_handling_record(out, sim_cfg, resource_handling_record)
    _attach_family_head_residual_record(out, sim_cfg, family_head_residual_record)
    attach_mutation_weather_metadata(out, weather)
    if _rep2m_reporting_enabled(sim_cfg):
        _rep2m_attach_variant_event_join_metadata(
            out,
            post_c2_payload=pipeline_payload,
            final_payload=c1_payload,
            trace_records=c1_trace_records,
        )
    _attach_transient_linkage_context(out, linkage_context)
    _attach_diploid_context(
        out,
        diploid_genotype,
        diploid_receipt,
    )
    _attach_genome_upgrade_v2_context(
        out,
        genome_upgrade_genotype,
        genome_upgrade_receipt,
    )
    if recomb72_audit:
        from evolution.recomb72_inheritance import (
            AUDITED_LOCI,
            MUTATION_OFF_CONTRACT,
            mode_sha256,
            registry_sha256,
        )

        parent_a = {name: float(getattr(g1, name)) for name in AUDITED_LOCI}
        parent_b = {name: float(getattr(g2, name)) for name in AUDITED_LOCI}
        child_values = {name: float(getattr(out, name)) for name in AUDITED_LOCI}
        setattr(
            out,
            "_recomb72_inheritance_event",
            {
                "schema_version": "recomb72_crossover_event_v1",
                "mode": "current_authoritative",
                "mode_sha256": mode_sha256("current_authoritative"),
                "registry_sha256": registry_sha256(),
                "parent_a": parent_a,
                "parent_b": parent_b,
                "child": child_values,
                "donor_by_locus": {},
                "package_donor": None,
                "draw_count": None,
                "draw_budget_contract": "production_current_path_v1",
                "mutation": dict(MUTATION_OFF_CONTRACT),
                "non_audited_surface": "production_current_inheritance",
                "production_current_path": True,
            },
        )
    return out


def clone_parameters_to_new(src: MLPPolicy, obs_size: int, hidden_size: int, n_actions: int) -> MLPPolicy:
    """Create a new MLPPolicy and copy parameters from `src`.

    Performance note:
    `state_dict()` + `load_state_dict()` is convenient but allocates Python dicts and
    does extra bookkeeping. For our tiny MLP, a direct parameter copy is faster and
    preserves semantics (no RNG, same weights).
    """
    child = MLPPolicy(obs_size=obs_size, hidden_size=hidden_size, n_actions=n_actions)
    if src is None:
        return child

    with torch.no_grad():
        for p_src, p_dst in zip(src.parameters(), child.parameters()):
            p_dst.copy_(p_src)
    return child


def consume_self_policy_crossover_rng(
    policy: MLPPolicy,
    *,
    rng_manager: Optional[Any],
    sim_cfg: Optional[Any] = None,
    buckets: Optional[dict[str, float]] = None,
) -> None:
    """Consume the deterministic self-policy crossover RNG budget without cloning.

    Supported resident child-row paths can attach the policy bank row directly
    from the parent. When the child policy is byte-identical to the parent, this
    helper preserves the RNG contract while allowing compatibility policy object
    creation to be deferred or avoided.
    """
    if policy is None:
        return
    params = tuple(policy.parameters())
    if not params:
        return
    if rng_manager is None:
        raise RuntimeError(
            "[DETERMINISM_VIOLATION]\n"
            "code=DET_FORBIDDEN_RNG_API\n"
            "context=missing RNGManager torch stream for mutation consume_self_policy_crossover_rng\n"
        )
    torch_numel = 0
    for param in params:
        torch_numel += int(param.numel())
    expected = rng_manager.expected_delta_for_boundary(
        "mutation.reproduction.crossover_policy",
        ctx={"torch_numel": int(torch_numel)},
    )
    devices = {str(param.device) for param in params}
    if len(devices) != 1:
        _runnercore5_stat_add(sim_cfg, "self_policy_source_alias_unsupported_device_mismatch", 1)
        if bool(getattr(sim_cfg, "runnercore5_fail_closed_on_unsupported", False)):
            raise RuntimeError("RUNNERCORE7-UNSUPPORTED: self policy source alias requires one tensor device")
        return
    device = next(iter(devices))
    tr = rng_manager.torch_rng("mutation", device)
    if tr is None:
        raise RuntimeError(
            "[DETERMINISM_VIOLATION]\n"
            "code=DET_RNG_UNKNOWN_STREAM\n"
            f"context=missing mutation torch stream for device={device}\n"
        )
    t0 = time.perf_counter()
    with rng_manager.budget_scope("mutation.reproduction.crossover_policy", expected_delta=expected):
        _ = tr.rand((int(torch_numel),), device=params[0].device, dtype=torch.float32)
    _runnercore5_stat_add(sim_cfg, "runnercore7_self_policy_source_alias_calls", 1)
    _runnercore5_stat_add(sim_cfg, "runnercore7_self_policy_source_alias_rng_elements", int(torch_numel))
    _runnercore5_bucket_add(buckets, "runnercore7_self_policy_source_alias", time.perf_counter() - t0)


def consume_mlp_policy_init_rng(
    *,
    obs_size: int,
    hidden_size: int,
    n_actions: int,
    rng_manager: Optional[Any] = None,
    sim_cfg: Optional[Any] = None,
    buckets: Optional[dict[str, float]] = None,
) -> None:
    """Replay MLPPolicy constructor RNG side effects without retaining a module.

    `MLPPolicy(...)` initializes three `nn.Linear` layers before crossover
    overwrites every parameter. The values are dead, but the default torch RNG
    advance is observable by later GRU constructors. The R7 source-alias path
    must preserve that side effect for parity.
    """
    t0 = time.perf_counter()
    if rng_manager is None:
        raise RuntimeError(
            "[DETERMINISM_VIOLATION]\n"
            "code=DET_FORBIDDEN_RNG_API\n"
            "context=missing RNGManager torch stream for MLP policy init replay\n"
        )
    total_numel = (
        int(hidden_size) * int(obs_size)
        + int(hidden_size)
        + int(hidden_size) * int(hidden_size)
        + int(hidden_size)
        + int(n_actions) * int(hidden_size)
        + int(n_actions)
    )
    with torch.no_grad():
        _ = MLPPolicy(obs_size=int(obs_size), hidden_size=int(hidden_size), n_actions=int(n_actions))
    _runnercore5_stat_add(sim_cfg, "runnercore7_mlp_policy_init_rng_replay_calls", 1)
    _runnercore5_stat_add(sim_cfg, "runnercore7_mlp_policy_init_rng_replay_elements", int(total_numel))
    _runnercore5_bucket_add(buckets, "runnercore7_mlp_policy_init_rng_replay", time.perf_counter() - t0)


def consume_gru_policy_init_rng(
    *,
    obs_size: int,
    hidden_size: int,
    n_actions: int,
    rng_manager: Optional[Any] = None,
    sim_cfg: Optional[Any] = None,
    buckets: Optional[dict[str, float]] = None,
) -> None:
    """Replay GRUPolicy constructor RNG side effects without retaining a module."""
    t0 = time.perf_counter()
    if rng_manager is None:
        raise RuntimeError(
            "[DETERMINISM_VIOLATION]\n"
            "code=DET_FORBIDDEN_RNG_API\n"
            "context=missing RNGManager torch stream for GRU policy init replay\n"
        )
    hidden = int(hidden_size)
    total_numel = (
        hidden * int(obs_size)
        + hidden
        + (3 * hidden * hidden)
        + (3 * hidden * hidden)
        + (3 * hidden)
        + (3 * hidden)
        + int(n_actions) * hidden
        + int(n_actions)
    )
    with torch.no_grad():
        _ = GRUPolicy(int(obs_size), hidden, int(n_actions))
    _runnercore5_stat_add(sim_cfg, "runnercore7_gru_policy_source_alias_calls", 1)
    _runnercore5_stat_add(sim_cfg, "runnercore7_gru_policy_init_rng_replay_elements", int(total_numel))
    _runnercore5_bucket_add(buckets, "runnercore7_gru_policy_init_rng_replay", time.perf_counter() - t0)


def mutate_policy_weights(
    policy: object,
    *,
    rng_manager: Optional[Any],
    sim_cfg: Optional[Any] = None,
    buckets: Optional[dict[str, float]] = None,
) -> dict[str, Any]:
    """Apply optional zero-mean offspring policy-weight mutation.

    Default-off. This is a general policy variation hook and deliberately has no
    access to cue identity, target correctness, memory state, fitness, or task
    outcome.
    """
    enabled = bool(getattr(sim_cfg, "policy_weight_mutation_enable", False)) if sim_cfg is not None else False
    mode = str(getattr(sim_cfg, "policy_weight_mutation_mode", "dense") or "dense").strip().lower() if sim_cfg is not None else "dense"
    rate = float(getattr(sim_cfg, "policy_weight_mutation_rate", 1.0) or 0.0) if sim_cfg is not None else 1.0
    std = float(getattr(sim_cfg, "policy_weight_mutation_std", 0.0) or 0.0) if sim_cfg is not None else 0.0
    if not enabled or std <= 0.0 or policy is None or not hasattr(policy, "parameters"):
        return {
            "enabled": bool(enabled),
            "mutated": False,
            "mode": str(mode),
            "rate": float(max(0.0, min(1.0, rate))),
            "std": float(max(0.0, std)),
            "parameter_count": 0,
            "mutated_parameter_count": 0,
        }
    if rng_manager is None:
        raise RuntimeError(
            "[DETERMINISM_VIOLATION]\n"
            "code=DET_FORBIDDEN_RNG_API\n"
            "context=missing RNGManager torch stream for mutation mutate_policy_weights\n"
        )
    params = tuple(policy.parameters())
    torch_numel = sum(int(param.numel()) for param in params)
    if torch_numel <= 0:
        return {
            "enabled": True,
            "mutated": False,
            "mode": str(mode),
            "rate": float(max(0.0, min(1.0, rate))),
            "std": float(std),
            "parameter_count": 0,
            "mutated_parameter_count": 0,
        }
    mode = "dense" if mode in ("", "gaussian") else mode
    rate = float(max(0.0, min(1.0, rate)))
    budget_label = "mutation.reproduction.policy_weight_mutation"
    expected_ctx = {"torch_numel": int(torch_numel)}
    if mode == "sparse":
        budget_label = "mutation.reproduction.policy_weight_mutation_sparse"
        expected_ctx = {"torch_draws": int(torch_numel) * 2}
    elif mode != "dense":
        raise ValueError(f"unsupported policy_weight_mutation_mode={mode!r}")
    expected = rng_manager.expected_delta_for_boundary(budget_label, ctx=expected_ctx)
    tr_by_device: Dict[str, Any] = {}
    mutated_elements = 0
    t0 = time.perf_counter()
    with rng_manager.budget_scope(budget_label, expected_delta=expected):
        with torch.no_grad():
            for param in params:
                dev = str(param.device)
                tr = tr_by_device.get(dev)
                if tr is None:
                    tr = rng_manager.torch_rng("mutation", dev)
                    tr_by_device[dev] = tr
                if tr is None:
                    raise RuntimeError(
                        "[DETERMINISM_VIOLATION]\n"
                        "code=DET_RNG_UNKNOWN_STREAM\n"
                        f"context=missing mutation torch stream for device={dev}\n"
                    )
                shape = tuple(int(x) for x in param.shape)
                if mode == "sparse":
                    mask = tr.rand(shape, device=param.device, dtype=torch.float32) < float(rate)
                    noise = tr.randn(shape, device=param.device, dtype=param.dtype) * float(std)
                    param.add_(noise * mask.to(dtype=param.dtype))
                    mutated_elements += int(mask.sum().item())
                else:
                    param.add_(tr.randn(shape, device=param.device, dtype=param.dtype) * float(std))
                    mutated_elements += int(param.numel())
    _runnercore5_stat_add(sim_cfg, "policy_weight_mutation_calls", 1)
    _runnercore5_stat_add(sim_cfg, "policy_weight_mutation_elements", int(mutated_elements))
    _runnercore5_bucket_add(buckets, "policy_weight_mutation", time.perf_counter() - t0)
    return {
        "enabled": True,
        "mutated": bool(mutated_elements > 0),
        "mode": str(mode),
        "rate": float(rate),
        "std": float(std),
        "parameter_count": int(torch_numel),
        "mutated_parameter_count": int(mutated_elements),
    }


def crossover_policy(
    p1: MLPPolicy,
    p2: MLPPolicy,
    obs_size: int,
    hidden_size: int,
    n_actions: int,
    rng: random.Random,
    torch_rng: Optional[Any] = None,
    rng_manager: Optional[Any] = None,
    sim_cfg: Optional[Any] = None,
    self_fastpath: bool = False,
    buckets: Optional[dict[str, float]] = None,
) -> MLPPolicy:
    """
    Parameter-wise crossover for MLPPolicy (fast path).

    This replaces the older state_dict-based implementation to reduce Python overhead.
    Determinism contract: the default path consumes one random tensor draw per
    parameter tensor in module-parameter order. The guarded flat path consumes
    the same total elements in the same flattened order when all tensors share a
    device.

    The Python `rng` is ONLY used in rare fallback paths (shape mismatches), matching
    the previous behavior.
    """
    if p1 is None or p2 is None:
        chosen = p1 if p2 is None else p2
        return clone_parameters_to_new(chosen, obs_size, hidden_size, n_actions)

    p1_params = tuple(p1.parameters())
    p2_params = tuple(p2.parameters())

    # Safety: if parameter layouts differ, fall back to cloning one parent.
    if len(p1_params) != len(p2_params):
        chosen = p1 if rng.random() < 0.5 else p2
        return clone_parameters_to_new(chosen, obs_size, hidden_size, n_actions)

    for a, b in zip(p1_params, p2_params):
        if tuple(a.shape) != tuple(b.shape):
            chosen = p1 if rng.random() < 0.5 else p2
            return clone_parameters_to_new(chosen, obs_size, hidden_size, n_actions)

    child = MLPPolicy(obs_size, hidden_size, n_actions)
    child_params = tuple(child.parameters())

    # Extra safety: if child layout differs (should not happen), fall back deterministically.
    if len(child_params) != len(p1_params):
        chosen = p1 if rng.random() < 0.5 else p2
        return clone_parameters_to_new(chosen, obs_size, hidden_size, n_actions)

    for a, c in zip(p1_params, child_params):
        if tuple(a.shape) != tuple(c.shape):
            chosen = p1 if rng.random() < 0.5 else p2
            return clone_parameters_to_new(chosen, obs_size, hidden_size, n_actions)

    with torch.no_grad():
        if torch_rng is None and rng_manager is None:
            raise RuntimeError(
                "[DETERMINISM_VIOLATION]\n"
                "code=DET_FORBIDDEN_RNG_API\n"
                "context=missing RNGManager torch stream for mutation crossover_policy\n"
            )
        torch_numel = 0
        for a in p1_params:
            torch_numel += int(a.numel())
        _budget = nullcontext()
        if rng_manager is not None:
            expected = rng_manager.expected_delta_for_boundary(
                "mutation.reproduction.crossover_policy",
                ctx={"torch_numel": int(torch_numel)},
            )
            _budget = rng_manager.budget_scope(
                "mutation.reproduction.crossover_policy",
                expected_delta=expected,
            )
        flat_crossover = bool(
            getattr(sim_cfg, "cxxrewrite_reproduction_flat_policy_crossover_enable", False)
        ) if sim_cfg is not None else False
        runnercore5_flat_self = bool(
            self_fastpath
            and p1 is p2
            and sim_cfg is not None
            and getattr(sim_cfg, "runnercore5_flat_self_policy_rng_enable", False)
        )
        same_device = False
        flat_device = None
        if flat_crossover or runnercore5_flat_self:
            try:
                devices = {
                    str(p.device)
                    for group in (p1_params, p2_params, child_params)
                    for p in group
                }
                same_device = len(devices) == 1
                flat_device = next(iter(devices)) if same_device else None
            except Exception:
                same_device = False
                flat_device = None
        if runnercore5_flat_self and not same_device:
            _runnercore5_stat_add(sim_cfg, "self_policy_flat_rng_unsupported_device_mismatch", 1)
            if bool(getattr(sim_cfg, "runnercore5_fail_closed_on_unsupported", False)):
                raise RuntimeError("RUNNERCORE5-UNSUPPORTED: self policy flat RNG requires one tensor device")
        tr_by_device: Dict[str, Any] = {}
        with _budget:
            if runnercore5_flat_self and same_device and flat_device is not None:
                t_runnercore5 = time.perf_counter()
                tr = torch_rng
                if tr is None:
                    dev = str(flat_device)
                    tr = tr_by_device.get(dev)
                    if tr is None:
                        tr = rng_manager.torch_rng("mutation", dev)
                        tr_by_device[dev] = tr
                    if tr is None:
                        raise RuntimeError(
                            "[DETERMINISM_VIOLATION]\n"
                            "code=DET_RNG_UNKNOWN_STREAM\n"
                            f"context=missing mutation torch stream for device={dev}\n"
                        )
                _ = tr.rand((int(torch_numel),), device=p1_params[0].device, dtype=torch.float32)
                for a, c in zip(p1_params, child_params):
                    c.copy_(a)
                _runnercore5_stat_add(sim_cfg, "self_policy_flat_rng_calls", 1)
                _runnercore5_stat_add(sim_cfg, "self_policy_flat_rng_elements", int(torch_numel))
                _runnercore5_bucket_add(buckets, "runnercore5_self_policy_flat_rng", time.perf_counter() - t_runnercore5)
            elif flat_crossover and same_device and flat_device is not None:
                tr = torch_rng
                if tr is None:
                    dev = str(flat_device)
                    tr = tr_by_device.get(dev)
                    if tr is None:
                        tr = rng_manager.torch_rng("mutation", dev)
                        tr_by_device[dev] = tr
                    if tr is None:
                        raise RuntimeError(
                            "[DETERMINISM_VIOLATION]\n"
                            "code=DET_RNG_UNKNOWN_STREAM\n"
                            f"context=missing mutation torch stream for device={dev}\n"
                        )
                mask_flat = tr.rand((int(torch_numel),), device=p1_params[0].device, dtype=torch.float32) < 0.5
                off = 0
                for a, b, c in zip(p1_params, p2_params, child_params):
                    n = int(a.numel())
                    mask = mask_flat[off:off + n].view_as(a)
                    c.copy_(torch.where(mask, a, b))
                    off += n
            elif self_fastpath and p1 is p2:
                for a, c in zip(p1_params, child_params):
                    tr = torch_rng
                    if tr is None:
                        dev = str(a.device)
                        tr = tr_by_device.get(dev)
                        if tr is None:
                            tr = rng_manager.torch_rng("mutation", dev)
                            tr_by_device[dev] = tr
                        if tr is None:
                            raise RuntimeError(
                                "[DETERMINISM_VIOLATION]\n"
                                "code=DET_RNG_UNKNOWN_STREAM\n"
                                f"context=missing mutation torch stream for device={dev}\n"
                            )
                    _ = tr.rand(tuple(int(x) for x in a.shape), device=a.device, dtype=torch.float32)
                    c.copy_(a)
            else:
                for a, b, c in zip(p1_params, p2_params, child_params):
                    tr = torch_rng
                    if tr is None:
                        dev = str(a.device)
                        tr = tr_by_device.get(dev)
                        if tr is None:
                            tr = rng_manager.torch_rng("mutation", dev)
                            tr_by_device[dev] = tr
                        if tr is None:
                            raise RuntimeError(
                                "[DETERMINISM_VIOLATION]\n"
                                "code=DET_RNG_UNKNOWN_STREAM\n"
                                f"context=missing mutation torch stream for device={dev}\n"
                            )
                    mask = tr.rand(tuple(int(x) for x in a.shape), device=a.device, dtype=torch.float32) < 0.5
                    c.copy_(torch.where(mask, a, b))

    return child
