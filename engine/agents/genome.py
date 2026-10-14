# agents/genome.py
from __future__ import annotations

"""agents/genome.py

Genome definition for an agent.
"""

from dataclasses import dataclass, field
import random
import time
from typing import Any

from agents.developmental_program_v1 import (
    canonicalize_developmental_program,
    module_structure_signature,
    resolve_module_active,
    resolve_stage_id,
)
from agents.modular_genome_v1 import (
    B2_FROZEN_CAPS,
    B2_INTERFACE_SCHEMA_VERSION,
    B3_FROZEN_LOCALITY,
    B3_LOCALITY_SCHEMA_VERSION,
    MODULAR_GENOME_SCHEMA_VERSION,
    canon_float,
    canonical_bytes_from_genome_dict,
    canonical_hash_from_genome_dict,
    canonical_payload_from_genome_dict,
    resolve_b2_caps,
    strict_mode_enabled,
)
from agents.perception import (
    DEFAULT_SENSOR_RADIUS,
    implemented_sensor_channel_mask,
    sanitize_sensor_channel_mask,
    snap_sensor_radius,
)
from agents.introspection import (
    implemented_introspection_channel_mask,
    sanitize_introspection_channel_mask,
)
from agents.self_regulation import (
    implemented_self_regulation_control_mask,
    sanitize_self_regulation_control_mask,
)
from agents.self_modification import (
    implemented_self_modification_target_mask,
    sanitize_self_modification_target_mask,
)
from agents.directed_inheritance import (
    implemented_directed_inheritance_channel_mask,
    sanitize_directed_inheritance_channel_mask,
)
from agents.family_head_residuals import (
    FAMILY_HEAD_RESIDUAL_FIELD,
    canonicalize_optional_family_head_residual_block,
)
from agents.resource_handling import (
    RESOURCE_HANDLING_AXIS_FIELD,
    RESOURCE_HANDLING_AXIS_NEUTRAL,
    clamp_resource_handling_axis,
    resource_handling_axis_is_neutral,
)


_MEMORY_LADDER = (0, 8, 16, 32, 64)


def _envhot7_developmental_diag_payload(raw: Any) -> tuple[Any, str, float]:
    try:
        from simulation.env_hotpath_expression_diagnostics import current_diagnostics, stable_payload_key  # noqa: WPS433

        diag = current_diagnostics()
        if diag is None or not bool(getattr(diag, "developmental_residual_enabled", False)):
            return None, "", 0.0
        payload_key, _payload_bytes = stable_payload_key(raw)
        return diag, str(payload_key), time.perf_counter()
    except Exception:
        return None, "", 0.0


@dataclass
class Genome:
    # --- Policy architecture ---
    hidden_size: int

    # --- Behavioral biases ---
    exploration_bias: float
    aggression_bias: float

    # --- Life history ---
    reproduction_threshold: float
    mutation_rate: float

    # Sprint C: evolvable decision stochasticity
    action_temperature: float

    # --- Memory / cognition ---
    use_gru: bool
    memory_dim: int

    # --- Identity Pressure: internal regulation traits ---
    drive_inertia: float
    drive_noise: float
    drive_logit_gain: float
    pred_lr: float

    # --- Theme 4 perception genes (fixed-shape masks; gated at expression time) ---
    sensor_channel_mask: int = field(default_factory=implemented_sensor_channel_mask)
    sensor_radius: int = field(default=DEFAULT_SENSOR_RADIUS)

    # --- Theme 4 introspection genes (fixed-shape masks; gated at expression time) ---
    introspection_channel_mask: int = field(default_factory=implemented_introspection_channel_mask)

    # --- Theme 4 self-regulation genes (fixed-shape masks; gated at expression time) ---
    self_regulation_control_mask: int = field(default_factory=implemented_self_regulation_control_mask)

    # --- Theme 4 self-modification genes (fixed-shape masks; gated at expression time) ---
    self_modification_target_mask: int = field(default_factory=implemented_self_modification_target_mask)

    # --- Theme 4 directed-inheritance genes (fixed-shape masks; gated at expression time) ---
    directed_inheritance_channel_mask: int = field(default_factory=implemented_directed_inheritance_channel_mask)

    # --- Cross-episode internal consistency (temperament priors) ---
    drive_sp_hunger: float = field(default=0.55)
    drive_sp_pain: float = field(default=0.05)
    drive_sp_fatigue: float = field(default=0.15)
    drive_sp_fear: float = field(default=0.10)

    exp_prior_strength: float = field(default=0.35)
    exp_prior_noise: float = field(default=0.05)
    drive_birth_noise: float = field(default=0.03)

    # Phase B1 modular schema v1 substrate
    schema_version: str = field(default=MODULAR_GENOME_SCHEMA_VERSION)
    interface_schema_version: str = field(default=B2_INTERFACE_SCHEMA_VERSION)
    b2_invalid_policy: str = field(default="clip")
    b2_caps: dict[str, int] = field(default_factory=lambda: dict(B2_FROZEN_CAPS))
    b3_locality_schema_version: str = field(default=B3_LOCALITY_SCHEMA_VERSION)
    b3_locality_max_connection_distance: float = field(default=float(B3_FROZEN_LOCALITY["max_connection_distance"]))
    b3_distance_cost_scale: float = field(default=float(B3_FROZEN_LOCALITY["distance_cost_scale"]))
    b3_distance_cost_max: float = field(default=float(B3_FROZEN_LOCALITY["distance_cost_max"]))
    b3_locality_cost_total: float = field(default=0.0)
    b3_locality_mean_distance: float = field(default=0.0)
    b3_locality_max_distance: float = field(default=0.0)
    b3_locality_dropped_connections: int = field(default=0)
    c1_catalog_version: str = field(default="")
    c1_operator_schedule: str = field(default="")
    c1_operator_applied: str = field(default="")
    c1_operator_noop_count: int = field(default=0)
    c1_capacity_blocked_count: int = field(default=0)
    c1_capacity_blocked_reason_counts_json: str = field(default="{}")
    c1_noop_reason_counts_json: str = field(default="{}")
    escap8_growth_attempt_count: int = field(default=0)
    escap8_growth_last_operation: str = field(default="")
    escap8_growth_last_status: str = field(default="")
    escap8_growth_last_reason: str = field(default="")
    escap8_blocked_duplication_attempt_count: int = field(default=0)
    escap8_blocked_prune_attempt_count: int = field(default=0)
    escap8_cap_contact_count: int = field(default=0)
    escap8_cap_pressure_streak: int = field(default=0)
    escap8_specialization_metrics_json: str = field(default="{}")
    structural_bootstrap_enabled: int = field(default=0)
    structural_bootstrap_accepted_count: int = field(default=0)
    c1_operator_budget_status: str = field(default="")
    c1_op_count__bounded_rewire: int = field(default=0)
    c1_op_count__duplicate_with_decay: int = field(default=0)
    c1_op_count__edge_split_insertion: int = field(default=0)
    c1_op_count__fusion_merge_compression: int = field(default=0)
    c1_op_count__gate_drift: int = field(default=0)
    c1_op_count__motif_duplication: int = field(default=0)
    c1_op_count__promoter_expression_mutation: int = field(default=0)
    c2_recombination_version: str = field(default="")
    c2_alignment_key: str = field(default="")
    c2_unmatched_policy: str = field(default="")
    c2_matched_modules: int = field(default=0)
    c2_unmatched_modules_parent_a: int = field(default=0)
    c2_unmatched_modules_parent_b: int = field(default=0)
    c2_matched_connections: int = field(default=0)
    c2_unmatched_connections_parent_a: int = field(default=0)
    c2_unmatched_connections_parent_b: int = field(default=0)
    c2_budget_draws: int = field(default=0)
    c2_parent_order_rule: str = field(default="")
    c2_legacy_unresolved_modules: int = field(default=0)
    c2_inheritance_profile: str = field(default="")
    c2_structural_linkage_rule: str = field(default="")
    c2_linked_component_count: int = field(default=0)
    modules: list[dict[str, Any]] = field(default_factory=list)
    connections: list[dict[str, Any]] = field(default_factory=list)
    developmental_program: dict[str, Any] = field(default_factory=canonicalize_developmental_program)
    resource_handling_axis: float = field(default=RESOURCE_HANDLING_AXIS_NEUTRAL)
    family_head_residuals: dict[str, Any] | None = field(default=None)

    def __post_init__(self) -> None:
        try:
            md = int(self.memory_dim)
        except Exception:
            md = 0
        md = min(_MEMORY_LADDER, key=lambda x: abs(x - md))
        self.memory_dim = int(md)

        try:
            use_gru_in = bool(getattr(self, "use_gru", False))
        except Exception:
            use_gru_in = False
        if int(self.memory_dim) > 0:
            self.use_gru = True
        else:
            self.use_gru = bool(use_gru_in)

        self.sensor_channel_mask = int(sanitize_sensor_channel_mask(getattr(self, "sensor_channel_mask", implemented_sensor_channel_mask())))
        self.sensor_radius = int(snap_sensor_radius(getattr(self, "sensor_radius", DEFAULT_SENSOR_RADIUS)))
        self.introspection_channel_mask = int(
            sanitize_introspection_channel_mask(
                getattr(self, "introspection_channel_mask", implemented_introspection_channel_mask())
            )
        )
        self.self_regulation_control_mask = int(
            sanitize_self_regulation_control_mask(
                getattr(self, "self_regulation_control_mask", implemented_self_regulation_control_mask())
            )
        )
        self.self_modification_target_mask = int(
            sanitize_self_modification_target_mask(
                getattr(self, "self_modification_target_mask", implemented_self_modification_target_mask())
            )
        )
        self.directed_inheritance_channel_mask = int(
            sanitize_directed_inheritance_channel_mask(
                getattr(self, "directed_inheritance_channel_mask", implemented_directed_inheritance_channel_mask())
            )
        )

        self.drive_inertia = float(min(0.999, max(0.0, float(getattr(self, "drive_inertia", 0.85)))))
        self.drive_noise = float(min(0.10, max(0.0, float(getattr(self, "drive_noise", 0.02)))))
        self.drive_logit_gain = float(min(3.0, max(0.0, float(getattr(self, "drive_logit_gain", 1.0)))))
        self.pred_lr = float(min(1.0, max(0.0, float(getattr(self, "pred_lr", 0.08)))))

        self.drive_sp_hunger = float(min(1.0, max(0.0, float(getattr(self, "drive_sp_hunger", 0.55)))))
        self.drive_sp_pain = float(min(1.0, max(0.0, float(getattr(self, "drive_sp_pain", 0.05)))))
        self.drive_sp_fatigue = float(min(1.0, max(0.0, float(getattr(self, "drive_sp_fatigue", 0.15)))))
        self.drive_sp_fear = float(min(1.0, max(0.0, float(getattr(self, "drive_sp_fear", 0.10)))))
        self.exp_prior_strength = float(min(1.0, max(0.0, float(getattr(self, "exp_prior_strength", 0.35)))))
        self.exp_prior_noise = float(min(0.5, max(0.0, float(getattr(self, "exp_prior_noise", 0.05)))))
        self.drive_birth_noise = float(min(0.2, max(0.0, float(getattr(self, "drive_birth_noise", 0.03)))))

        # Float canonicalization policy
        self.exploration_bias = canon_float(self.exploration_bias, field="exploration_bias")
        self.aggression_bias = canon_float(self.aggression_bias, field="aggression_bias")
        self.reproduction_threshold = canon_float(self.reproduction_threshold, field="reproduction_threshold")
        self.mutation_rate = canon_float(self.mutation_rate, field="mutation_rate")
        self.action_temperature = canon_float(getattr(self, "action_temperature", 1.0), field="action_temperature")

        self.drive_inertia = canon_float(self.drive_inertia, field="drive_inertia")
        self.drive_noise = canon_float(self.drive_noise, field="drive_noise")
        self.drive_logit_gain = canon_float(self.drive_logit_gain, field="drive_logit_gain")
        self.pred_lr = canon_float(self.pred_lr, field="pred_lr")

        self.drive_sp_hunger = canon_float(self.drive_sp_hunger, field="drive_sp_hunger")
        self.drive_sp_pain = canon_float(self.drive_sp_pain, field="drive_sp_pain")
        self.drive_sp_fatigue = canon_float(self.drive_sp_fatigue, field="drive_sp_fatigue")
        self.drive_sp_fear = canon_float(self.drive_sp_fear, field="drive_sp_fear")
        self.exp_prior_strength = canon_float(self.exp_prior_strength, field="exp_prior_strength")
        self.exp_prior_noise = canon_float(self.exp_prior_noise, field="exp_prior_noise")
        self.drive_birth_noise = canon_float(self.drive_birth_noise, field="drive_birth_noise")
        self.resource_handling_axis = clamp_resource_handling_axis(
            getattr(self, RESOURCE_HANDLING_AXIS_FIELD, RESOURCE_HANDLING_AXIS_NEUTRAL)
        )
        raw_family_head_residuals = getattr(self, FAMILY_HEAD_RESIDUAL_FIELD, None)
        self.family_head_residuals = canonicalize_optional_family_head_residual_block(raw_family_head_residuals)

        self.schema_version = str(getattr(self, "schema_version", MODULAR_GENOME_SCHEMA_VERSION) or MODULAR_GENOME_SCHEMA_VERSION)
        self.interface_schema_version = str(getattr(self, "interface_schema_version", B2_INTERFACE_SCHEMA_VERSION) or B2_INTERFACE_SCHEMA_VERSION)
        self.b2_invalid_policy = str(getattr(self, "b2_invalid_policy", "clip") or "clip")
        self.b2_caps = dict(getattr(self, "b2_caps", B2_FROZEN_CAPS) or B2_FROZEN_CAPS)
        self.b3_locality_schema_version = str(getattr(self, "b3_locality_schema_version", B3_LOCALITY_SCHEMA_VERSION) or B3_LOCALITY_SCHEMA_VERSION)
        self.b3_locality_max_connection_distance = canon_float(
            getattr(self, "b3_locality_max_connection_distance", B3_FROZEN_LOCALITY["max_connection_distance"]),
            field="b3_locality_max_connection_distance",
        )
        self.b3_distance_cost_scale = canon_float(
            getattr(self, "b3_distance_cost_scale", B3_FROZEN_LOCALITY["distance_cost_scale"]),
            field="b3_distance_cost_scale",
        )
        self.b3_distance_cost_max = canon_float(
            getattr(self, "b3_distance_cost_max", B3_FROZEN_LOCALITY["distance_cost_max"]),
            field="b3_distance_cost_max",
        )
        self.b3_locality_cost_total = canon_float(getattr(self, "b3_locality_cost_total", 0.0), field="b3_locality_cost_total")
        self.b3_locality_mean_distance = canon_float(getattr(self, "b3_locality_mean_distance", 0.0), field="b3_locality_mean_distance")
        self.b3_locality_max_distance = canon_float(getattr(self, "b3_locality_max_distance", 0.0), field="b3_locality_max_distance")
        try:
            self.b3_locality_dropped_connections = int(getattr(self, "b3_locality_dropped_connections", 0) or 0)
        except Exception:
            self.b3_locality_dropped_connections = 0
        # Canonicalize modular records deterministically.
        can = canonical_payload_from_genome_dict(self.to_raw_dict())
        self.schema_version = str(can.get("schema_version", MODULAR_GENOME_SCHEMA_VERSION))
        self.interface_schema_version = str(can.get("interface_schema_version", B2_INTERFACE_SCHEMA_VERSION))
        self.b2_invalid_policy = str(can.get("b2_invalid_policy", "clip") or "clip")
        self.b2_caps = dict(can.get("b2_caps", B2_FROZEN_CAPS) or B2_FROZEN_CAPS)
        self.b3_locality_schema_version = str(can.get("b3_locality_schema_version", B3_LOCALITY_SCHEMA_VERSION))
        self.b3_locality_max_connection_distance = canon_float(
            can.get("b3_locality_max_connection_distance", B3_FROZEN_LOCALITY["max_connection_distance"]),
            field="b3_locality_max_connection_distance",
        )
        self.b3_distance_cost_scale = canon_float(
            can.get("b3_distance_cost_scale", B3_FROZEN_LOCALITY["distance_cost_scale"]),
            field="b3_distance_cost_scale",
        )
        self.b3_distance_cost_max = canon_float(
            can.get("b3_distance_cost_max", B3_FROZEN_LOCALITY["distance_cost_max"]),
            field="b3_distance_cost_max",
        )
        self.b3_locality_cost_total = canon_float(can.get("b3_locality_cost_total", 0.0), field="b3_locality_cost_total")
        self.b3_locality_mean_distance = canon_float(can.get("b3_locality_mean_distance", 0.0), field="b3_locality_mean_distance")
        self.b3_locality_max_distance = canon_float(can.get("b3_locality_max_distance", 0.0), field="b3_locality_max_distance")
        self.b3_locality_dropped_connections = int(can.get("b3_locality_dropped_connections", 0) or 0)
        self.c1_catalog_version = str(can.get("genes", {}).get("c1_catalog_version", "") or "")
        self.c1_operator_schedule = str(can.get("genes", {}).get("c1_operator_schedule", "") or "")
        self.c1_operator_applied = str(can.get("genes", {}).get("c1_operator_applied", "") or "")
        self.c1_operator_budget_status = str(can.get("genes", {}).get("c1_operator_budget_status", "") or "")
        try:
            self.c1_operator_noop_count = int(can.get("genes", {}).get("c1_operator_noop_count", 0) or 0)
        except Exception:
            self.c1_operator_noop_count = 0
        try:
            self.c1_capacity_blocked_count = int(can.get("genes", {}).get("c1_capacity_blocked_count", 0) or 0)
        except Exception:
            self.c1_capacity_blocked_count = 0
        self.c1_capacity_blocked_reason_counts_json = str(
            can.get("genes", {}).get("c1_capacity_blocked_reason_counts_json", "{}") or "{}"
        )
        self.c1_noop_reason_counts_json = str(
            can.get("genes", {}).get("c1_noop_reason_counts_json", "{}") or "{}"
        )
        for key in (
            "escap8_growth_attempt_count",
            "escap8_blocked_duplication_attempt_count",
            "escap8_blocked_prune_attempt_count",
            "escap8_cap_contact_count",
            "escap8_cap_pressure_streak",
        ):
            try:
                setattr(self, key, int(can.get("genes", {}).get(key, 0) or 0))
            except Exception:
                setattr(self, key, 0)
        for key in (
            "escap8_growth_last_operation",
            "escap8_growth_last_status",
            "escap8_growth_last_reason",
        ):
            setattr(self, key, str(can.get("genes", {}).get(key, "") or ""))
        self.escap8_specialization_metrics_json = str(
            can.get("genes", {}).get("escap8_specialization_metrics_json", "{}") or "{}"
        )
        self.structural_bootstrap_enabled = int(
            can.get("genes", {}).get("structural_bootstrap_enabled", 0) or 0
        )
        self.structural_bootstrap_accepted_count = int(
            can.get("genes", {}).get("structural_bootstrap_accepted_count", 0) or 0
        )
        for key in (
            "c1_op_count__bounded_rewire",
            "c1_op_count__duplicate_with_decay",
            "c1_op_count__edge_split_insertion",
            "c1_op_count__fusion_merge_compression",
            "c1_op_count__gate_drift",
            "c1_op_count__motif_duplication",
            "c1_op_count__promoter_expression_mutation",
        ):
            try:
                setattr(self, key, int(can.get("genes", {}).get(key, 0) or 0))
            except Exception:
                setattr(self, key, 0)
        self.c2_recombination_version = str(can.get("genes", {}).get("c2_recombination_version", "") or "")
        self.c2_alignment_key = str(can.get("genes", {}).get("c2_alignment_key", "") or "")
        self.c2_unmatched_policy = str(can.get("genes", {}).get("c2_unmatched_policy", "") or "")
        self.c2_parent_order_rule = str(can.get("genes", {}).get("c2_parent_order_rule", "") or "")
        self.c2_legacy_unresolved_modules = int(
            can.get("genes", {}).get("c2_legacy_unresolved_modules", 0) or 0
        )
        self.c2_inheritance_profile = str(
            can.get("genes", {}).get("c2_inheritance_profile", "") or ""
        )
        self.c2_structural_linkage_rule = str(
            can.get("genes", {}).get("c2_structural_linkage_rule", "") or ""
        )
        self.c2_linked_component_count = int(
            can.get("genes", {}).get("c2_linked_component_count", 0) or 0
        )
        self.resource_handling_axis = clamp_resource_handling_axis(
            can.get("genes", {}).get(RESOURCE_HANDLING_AXIS_FIELD, RESOURCE_HANDLING_AXIS_NEUTRAL)
        )
        self.family_head_residuals = canonicalize_optional_family_head_residual_block(
            can.get("genes", {}).get(FAMILY_HEAD_RESIDUAL_FIELD, None)
        )
        for key in (
            "c2_matched_modules",
            "c2_unmatched_modules_parent_a",
            "c2_unmatched_modules_parent_b",
            "c2_matched_connections",
            "c2_unmatched_connections_parent_a",
            "c2_unmatched_connections_parent_b",
            "c2_budget_draws",
        ):
            try:
                setattr(self, key, int(can.get("genes", {}).get(key, 0) or 0))
            except Exception:
                setattr(self, key, 0)
        self.modules = list(can.get("modules", []))
        self.connections = list(can.get("connections", []))
        self.developmental_program = canonicalize_developmental_program(
            can.get("developmental_program", getattr(self, "developmental_program", None))
        )
        self._canonical_payload_cache = can

    def refresh_canonical_payload_cache(self) -> dict[str, Any]:
        can = canonical_payload_from_genome_dict(self.to_raw_dict())
        self._canonical_payload_cache = can
        return dict(can)

    def cached_canonical_payload(self) -> dict[str, Any]:
        cache = getattr(self, "_canonical_payload_cache", None)
        if isinstance(cache, dict):
            return dict(cache)
        return self.refresh_canonical_payload_cache()

    def to_raw_dict(self) -> dict[str, Any]:
        out = {
            "hidden_size": int(self.hidden_size),
            "exploration_bias": float(self.exploration_bias),
            "aggression_bias": float(self.aggression_bias),
            "reproduction_threshold": float(self.reproduction_threshold),
            "mutation_rate": float(self.mutation_rate),
            "action_temperature": float(self.action_temperature),
            "use_gru": bool(self.use_gru),
            "memory_dim": int(self.memory_dim),
            "sensor_channel_mask": int(self.sensor_channel_mask),
            "sensor_radius": int(self.sensor_radius),
            "introspection_channel_mask": int(self.introspection_channel_mask),
            "self_regulation_control_mask": int(self.self_regulation_control_mask),
            "self_modification_target_mask": int(self.self_modification_target_mask),
            "directed_inheritance_channel_mask": int(self.directed_inheritance_channel_mask),
            "drive_inertia": float(self.drive_inertia),
            "drive_noise": float(self.drive_noise),
            "drive_logit_gain": float(self.drive_logit_gain),
            "pred_lr": float(self.pred_lr),
            "drive_sp_hunger": float(self.drive_sp_hunger),
            "drive_sp_pain": float(self.drive_sp_pain),
            "drive_sp_fatigue": float(self.drive_sp_fatigue),
            "drive_sp_fear": float(self.drive_sp_fear),
            "exp_prior_strength": float(self.exp_prior_strength),
            "exp_prior_noise": float(self.exp_prior_noise),
            "drive_birth_noise": float(self.drive_birth_noise),
            "schema_version": str(self.schema_version or MODULAR_GENOME_SCHEMA_VERSION),
            "interface_schema_version": str(self.interface_schema_version or B2_INTERFACE_SCHEMA_VERSION),
            "b2_invalid_policy": str(self.b2_invalid_policy or "clip"),
            "b2_caps": dict(self.b2_caps or B2_FROZEN_CAPS),
            "b3_locality_schema_version": str(self.b3_locality_schema_version or B3_LOCALITY_SCHEMA_VERSION),
            "b3_locality_max_connection_distance": float(self.b3_locality_max_connection_distance),
            "b3_distance_cost_scale": float(self.b3_distance_cost_scale),
            "b3_distance_cost_max": float(self.b3_distance_cost_max),
            "b3_locality_cost_total": float(self.b3_locality_cost_total),
            "b3_locality_mean_distance": float(self.b3_locality_mean_distance),
            "b3_locality_max_distance": float(self.b3_locality_max_distance),
            "b3_locality_dropped_connections": int(self.b3_locality_dropped_connections),
            "c1_catalog_version": str(self.c1_catalog_version or ""),
            "c1_operator_schedule": str(self.c1_operator_schedule or ""),
            "c1_operator_applied": str(self.c1_operator_applied or ""),
            "c1_operator_noop_count": int(self.c1_operator_noop_count or 0),
            "c1_capacity_blocked_count": int(self.c1_capacity_blocked_count or 0),
            "c1_capacity_blocked_reason_counts_json": str(self.c1_capacity_blocked_reason_counts_json or "{}"),
            "c1_noop_reason_counts_json": str(self.c1_noop_reason_counts_json or "{}"),
            "structural_bootstrap_enabled": int(self.structural_bootstrap_enabled or 0),
            "structural_bootstrap_accepted_count": int(self.structural_bootstrap_accepted_count or 0),
            "c1_operator_budget_status": str(self.c1_operator_budget_status or ""),
            "c1_op_count__bounded_rewire": int(self.c1_op_count__bounded_rewire or 0),
            "c1_op_count__duplicate_with_decay": int(self.c1_op_count__duplicate_with_decay or 0),
            "c1_op_count__edge_split_insertion": int(self.c1_op_count__edge_split_insertion or 0),
            "c1_op_count__fusion_merge_compression": int(self.c1_op_count__fusion_merge_compression or 0),
            "c1_op_count__gate_drift": int(self.c1_op_count__gate_drift or 0),
            "c1_op_count__motif_duplication": int(self.c1_op_count__motif_duplication or 0),
            "c1_op_count__promoter_expression_mutation": int(self.c1_op_count__promoter_expression_mutation or 0),
            "c2_recombination_version": str(self.c2_recombination_version or ""),
            "c2_alignment_key": str(self.c2_alignment_key or ""),
            "c2_unmatched_policy": str(self.c2_unmatched_policy or ""),
            "c2_matched_modules": int(self.c2_matched_modules or 0),
            "c2_unmatched_modules_parent_a": int(self.c2_unmatched_modules_parent_a or 0),
            "c2_unmatched_modules_parent_b": int(self.c2_unmatched_modules_parent_b or 0),
            "c2_matched_connections": int(self.c2_matched_connections or 0),
            "c2_unmatched_connections_parent_a": int(self.c2_unmatched_connections_parent_a or 0),
            "c2_unmatched_connections_parent_b": int(self.c2_unmatched_connections_parent_b or 0),
            "c2_budget_draws": int(self.c2_budget_draws or 0),
            "c2_parent_order_rule": str(self.c2_parent_order_rule or ""),
            "c2_legacy_unresolved_modules": int(self.c2_legacy_unresolved_modules or 0),
            "modules": list(self.modules or []),
            "connections": list(self.connections or []),
            "developmental_program": canonicalize_developmental_program(getattr(self, "developmental_program", None)),
        }
        resource_axis = clamp_resource_handling_axis(
            getattr(self, RESOURCE_HANDLING_AXIS_FIELD, RESOURCE_HANDLING_AXIS_NEUTRAL)
        )
        if not resource_handling_axis_is_neutral(resource_axis):
            out[RESOURCE_HANDLING_AXIS_FIELD] = float(resource_axis)
        if getattr(self, FAMILY_HEAD_RESIDUAL_FIELD, None) is not None:
            out[FAMILY_HEAD_RESIDUAL_FIELD] = canonicalize_optional_family_head_residual_block(
                getattr(self, FAMILY_HEAD_RESIDUAL_FIELD, None)
            )
        # ESCAP-8 telemetry is hereditary only after the operator has produced
        # a record.  Omit the all-default block so pre-ESCAP-8 genome bytes,
        # hashes, parent ordering, and checkpoint payloads remain unchanged.
        escap8_growth_attempt_count = int(
            getattr(self, "escap8_growth_attempt_count", 0) or 0
        )
        escap8_growth_last_operation = str(
            getattr(self, "escap8_growth_last_operation", "") or ""
        )
        escap8_growth_last_status = str(
            getattr(self, "escap8_growth_last_status", "") or ""
        )
        escap8_growth_last_reason = str(
            getattr(self, "escap8_growth_last_reason", "") or ""
        )
        escap8_blocked_duplication_attempt_count = int(
            getattr(self, "escap8_blocked_duplication_attempt_count", 0) or 0
        )
        escap8_blocked_prune_attempt_count = int(
            getattr(self, "escap8_blocked_prune_attempt_count", 0) or 0
        )
        escap8_cap_contact_count = int(
            getattr(self, "escap8_cap_contact_count", 0) or 0
        )
        escap8_cap_pressure_streak = int(
            getattr(self, "escap8_cap_pressure_streak", 0) or 0
        )
        escap8_specialization_metrics_json = str(
            getattr(self, "escap8_specialization_metrics_json", "{}") or "{}"
        )
        if any(
            (
                escap8_growth_attempt_count,
                escap8_growth_last_operation,
                escap8_growth_last_status,
                escap8_growth_last_reason,
                escap8_blocked_duplication_attempt_count,
                escap8_blocked_prune_attempt_count,
                escap8_cap_contact_count,
                escap8_cap_pressure_streak,
                escap8_specialization_metrics_json != "{}",
            )
        ):
            out.update(
                {
                    "escap8_growth_attempt_count": escap8_growth_attempt_count,
                    "escap8_growth_last_operation": escap8_growth_last_operation,
                    "escap8_growth_last_status": escap8_growth_last_status,
                    "escap8_growth_last_reason": escap8_growth_last_reason,
                    "escap8_blocked_duplication_attempt_count": (
                        escap8_blocked_duplication_attempt_count
                    ),
                    "escap8_blocked_prune_attempt_count": (
                        escap8_blocked_prune_attempt_count
                    ),
                    "escap8_cap_contact_count": escap8_cap_contact_count,
                    "escap8_cap_pressure_streak": escap8_cap_pressure_streak,
                    "escap8_specialization_metrics_json": (
                        escap8_specialization_metrics_json
                    ),
                }
            )
        # RECOMB-1 linkage annotations are repaired-profile metadata.  Omit all
        # default values so historical/legacy genome bytes and hashes remain
        # unchanged; a non-empty profile freezes the complete linked record.
        if str(getattr(self, "c2_inheritance_profile", "") or ""):
            out["c2_inheritance_profile"] = str(self.c2_inheritance_profile)
            out["c2_structural_linkage_rule"] = str(
                getattr(self, "c2_structural_linkage_rule", "") or ""
            )
            out["c2_linked_component_count"] = int(
                getattr(self, "c2_linked_component_count", 0) or 0
            )
        diploid_bundle = getattr(self, "_diploid_scalar_genome_v1", None)
        if diploid_bundle is not None:
            from evolution.diploid_inheritance_v1 import (  # noqa: WPS433
                DiploidScalarGenomeV1,
            )

            to_payload = getattr(diploid_bundle, "to_payload", None)
            if callable(to_payload):
                diploid_payload = to_payload()
            else:
                diploid_bundle = DiploidScalarGenomeV1.from_payload(
                    diploid_bundle
                )
                diploid_payload = diploid_bundle.to_payload()
            out["diploid_scalar_genome_v1"] = diploid_payload
        genome_upgrade = getattr(self, "_genome_upgrade_v2_v1", None)
        if genome_upgrade is not None:
            from evolution.genome_upgrade_v2 import (  # noqa: WPS433
                GenomeUpgradeV2V1,
            )

            if not isinstance(genome_upgrade, GenomeUpgradeV2V1):
                genome_upgrade = GenomeUpgradeV2V1.from_payload(genome_upgrade)
            out["genome_upgrade_v2_v1"] = genome_upgrade.to_payload()
        return out

    def to_dict(self) -> dict[str, Any]:
        can = canonical_payload_from_genome_dict(self.to_raw_dict())
        out = dict(can.get("genes", {}))
        out["schema_version"] = str(can.get("schema_version", MODULAR_GENOME_SCHEMA_VERSION))
        out["interface_schema_version"] = str(can.get("interface_schema_version", B2_INTERFACE_SCHEMA_VERSION))
        out["b2_invalid_policy"] = str(can.get("b2_invalid_policy", "clip") or "clip")
        out["b2_caps"] = dict(can.get("b2_caps", B2_FROZEN_CAPS) or B2_FROZEN_CAPS)
        out["b3_locality_schema_version"] = str(can.get("b3_locality_schema_version", B3_LOCALITY_SCHEMA_VERSION))
        out["b3_locality_max_connection_distance"] = float(can.get("b3_locality_max_connection_distance", B3_FROZEN_LOCALITY["max_connection_distance"]))
        out["b3_distance_cost_scale"] = float(can.get("b3_distance_cost_scale", B3_FROZEN_LOCALITY["distance_cost_scale"]))
        out["b3_distance_cost_max"] = float(can.get("b3_distance_cost_max", B3_FROZEN_LOCALITY["distance_cost_max"]))
        out["b3_locality_cost_total"] = float(can.get("b3_locality_cost_total", 0.0) or 0.0)
        out["b3_locality_mean_distance"] = float(can.get("b3_locality_mean_distance", 0.0) or 0.0)
        out["b3_locality_max_distance"] = float(can.get("b3_locality_max_distance", 0.0) or 0.0)
        out["b3_locality_dropped_connections"] = int(can.get("b3_locality_dropped_connections", 0) or 0)
        out["modules"] = list(can.get("modules", []))
        out["connections"] = list(can.get("connections", []))
        out["developmental_program"] = canonicalize_developmental_program(getattr(self, "developmental_program", None))
        return out

    def canonical_payload(self) -> dict[str, Any]:
        return canonical_payload_from_genome_dict(self.to_dict())

    def canonical_bytes(self) -> bytes:
        return canonical_bytes_from_genome_dict(self.to_dict())

    def canonical_hash(self) -> str:
        return canonical_hash_from_genome_dict(self.to_dict())

    @staticmethod
    def from_dict(d: dict) -> "Genome":
        """Reconstruct a Genome from plain dict without consuming RNG."""
        if not isinstance(d, dict):
            d = {}
        schema = str(d.get("schema_version", "") or "")
        if schema and schema != MODULAR_GENOME_SCHEMA_VERSION:
            raise RuntimeError(
                "[DETERMINISM_VIOLATION]\n"
                "code=DET_GENOME_SCHEMA_UNSUPPORTED\n"
                f"context=schema_version={schema}\n"
            )

        def _as_int(x, default: int) -> int:
            try:
                return int(x)
            except Exception:
                return int(default)

        def _as_float(x, default: float) -> float:
            try:
                return float(x)
            except Exception:
                return float(default)

        def _as_bool(x, default: bool, *, field: str = "") -> bool:
            if isinstance(x, bool):
                return bool(x)
            try:
                if isinstance(x, int):
                    if int(x) in (0, 1):
                        return bool(int(x))
                    if strict_mode_enabled():
                        raise RuntimeError(
                            "[DETERMINISM_VIOLATION]\n"
                            "code=DET_GENOME_SCHEMA_MALFORMED\n"
                            f"context=field={field};reason=invalid_bool_int;value={x}\n"
                        )
                    return bool(default)
                if isinstance(x, str):
                    s = str(x).strip().lower()
                    if s in {"1", "true", "t", "yes", "y", "on"}:
                        return True
                    if s in {"0", "false", "f", "no", "n", "off"}:
                        return False
                    if strict_mode_enabled():
                        raise RuntimeError(
                            "[DETERMINISM_VIOLATION]\n"
                            "code=DET_GENOME_SCHEMA_MALFORMED\n"
                            f"context=field={field};reason=invalid_bool_string;value={x}\n"
                        )
                    return bool(default)
                return bool(x)
            except Exception:
                if strict_mode_enabled():
                    raise
                return bool(default)

        can = canonical_payload_from_genome_dict(d)
        genes = dict(can.get("genes", {}))
        modules = list(can.get("modules", []))
        connections = list(can.get("connections", []))
        interface_schema_version = str(can.get("interface_schema_version", B2_INTERFACE_SCHEMA_VERSION))
        b2_invalid_policy = str(can.get("b2_invalid_policy", "clip") or "clip")
        b2_caps = dict(can.get("b2_caps", B2_FROZEN_CAPS) or B2_FROZEN_CAPS)
        b3_locality_schema_version = str(can.get("b3_locality_schema_version", B3_LOCALITY_SCHEMA_VERSION))
        b3_locality_max_connection_distance = _as_float(can.get("b3_locality_max_connection_distance", B3_FROZEN_LOCALITY["max_connection_distance"]), float(B3_FROZEN_LOCALITY["max_connection_distance"]))
        b3_distance_cost_scale = _as_float(can.get("b3_distance_cost_scale", B3_FROZEN_LOCALITY["distance_cost_scale"]), float(B3_FROZEN_LOCALITY["distance_cost_scale"]))
        b3_distance_cost_max = _as_float(can.get("b3_distance_cost_max", B3_FROZEN_LOCALITY["distance_cost_max"]), float(B3_FROZEN_LOCALITY["distance_cost_max"]))
        b3_locality_cost_total = _as_float(can.get("b3_locality_cost_total", 0.0), 0.0)
        b3_locality_mean_distance = _as_float(can.get("b3_locality_mean_distance", 0.0), 0.0)
        b3_locality_max_distance = _as_float(can.get("b3_locality_max_distance", 0.0), 0.0)
        b3_locality_dropped_connections = _as_int(can.get("b3_locality_dropped_connections", 0), 0)
        c1_catalog_version = str(genes.get("c1_catalog_version", "") or "")
        c1_operator_schedule = str(genes.get("c1_operator_schedule", "") or "")
        c1_operator_applied = str(genes.get("c1_operator_applied", "") or "")
        c1_operator_budget_status = str(genes.get("c1_operator_budget_status", "") or "")
        c1_operator_noop_count = _as_int(genes.get("c1_operator_noop_count", 0), 0)
        c1_capacity_blocked_count = _as_int(genes.get("c1_capacity_blocked_count", 0), 0)
        c1_capacity_blocked_reason_counts_json = str(
            genes.get("c1_capacity_blocked_reason_counts_json", "{}") or "{}"
        )
        c1_noop_reason_counts_json = str(genes.get("c1_noop_reason_counts_json", "{}") or "{}")
        escap8_growth_attempt_count = _as_int(genes.get("escap8_growth_attempt_count", 0), 0)
        escap8_growth_last_operation = str(genes.get("escap8_growth_last_operation", "") or "")
        escap8_growth_last_status = str(genes.get("escap8_growth_last_status", "") or "")
        escap8_growth_last_reason = str(genes.get("escap8_growth_last_reason", "") or "")
        escap8_blocked_duplication_attempt_count = _as_int(
            genes.get("escap8_blocked_duplication_attempt_count", 0), 0
        )
        escap8_blocked_prune_attempt_count = _as_int(
            genes.get("escap8_blocked_prune_attempt_count", 0), 0
        )
        escap8_cap_contact_count = _as_int(genes.get("escap8_cap_contact_count", 0), 0)
        escap8_cap_pressure_streak = _as_int(genes.get("escap8_cap_pressure_streak", 0), 0)
        escap8_specialization_metrics_json = str(
            genes.get("escap8_specialization_metrics_json", "{}") or "{}"
        )
        structural_bootstrap_enabled = _as_int(genes.get("structural_bootstrap_enabled", 0), 0)
        structural_bootstrap_accepted_count = _as_int(
            genes.get("structural_bootstrap_accepted_count", 0), 0
        )
        c1_op_count__bounded_rewire = _as_int(genes.get("c1_op_count__bounded_rewire", 0), 0)
        c1_op_count__duplicate_with_decay = _as_int(genes.get("c1_op_count__duplicate_with_decay", 0), 0)
        c1_op_count__edge_split_insertion = _as_int(genes.get("c1_op_count__edge_split_insertion", 0), 0)
        c1_op_count__fusion_merge_compression = _as_int(genes.get("c1_op_count__fusion_merge_compression", 0), 0)
        c1_op_count__gate_drift = _as_int(genes.get("c1_op_count__gate_drift", 0), 0)
        c1_op_count__motif_duplication = _as_int(genes.get("c1_op_count__motif_duplication", 0), 0)
        c1_op_count__promoter_expression_mutation = _as_int(genes.get("c1_op_count__promoter_expression_mutation", 0), 0)
        c2_recombination_version = str(genes.get("c2_recombination_version", "") or "")
        c2_alignment_key = str(genes.get("c2_alignment_key", "") or "")
        c2_unmatched_policy = str(genes.get("c2_unmatched_policy", "") or "")
        c2_matched_modules = _as_int(genes.get("c2_matched_modules", 0), 0)
        c2_unmatched_modules_parent_a = _as_int(genes.get("c2_unmatched_modules_parent_a", 0), 0)
        c2_unmatched_modules_parent_b = _as_int(genes.get("c2_unmatched_modules_parent_b", 0), 0)
        c2_matched_connections = _as_int(genes.get("c2_matched_connections", 0), 0)
        c2_unmatched_connections_parent_a = _as_int(genes.get("c2_unmatched_connections_parent_a", 0), 0)
        c2_unmatched_connections_parent_b = _as_int(genes.get("c2_unmatched_connections_parent_b", 0), 0)
        c2_budget_draws = _as_int(genes.get("c2_budget_draws", 0), 0)
        c2_parent_order_rule = str(genes.get("c2_parent_order_rule", "") or "")
        c2_legacy_unresolved_modules = _as_int(genes.get("c2_legacy_unresolved_modules", 0), 0)
        c2_inheritance_profile = str(genes.get("c2_inheritance_profile", "") or "")
        c2_structural_linkage_rule = str(genes.get("c2_structural_linkage_rule", "") or "")
        c2_linked_component_count = _as_int(genes.get("c2_linked_component_count", 0), 0)
        resource_handling_axis = clamp_resource_handling_axis(
            genes.get(RESOURCE_HANDLING_AXIS_FIELD, RESOURCE_HANDLING_AXIS_NEUTRAL)
        )
        family_head_residuals = canonicalize_optional_family_head_residual_block(
            genes.get(FAMILY_HEAD_RESIDUAL_FIELD, None)
        )
        developmental_program = canonicalize_developmental_program(d.get("developmental_program", None))

        hidden_size = _as_int(genes.get("hidden_size", 64), 64)
        exploration_bias = _as_float(genes.get("exploration_bias", 0.0), 0.0)
        aggression_bias = _as_float(genes.get("aggression_bias", 0.0), 0.0)
        reproduction_threshold = _as_float(genes.get("reproduction_threshold", 30.0), 30.0)
        mutation_rate = _as_float(genes.get("mutation_rate", 0.05), 0.05)
        action_temperature = _as_float(genes.get("action_temperature", 1.0), 1.0)

        use_gru = _as_bool(genes.get("use_gru", False), False, field="use_gru")
        memory_dim = _as_int(genes.get("memory_dim", 0), 0)
        sensor_channel_mask = sanitize_sensor_channel_mask(
            genes.get("sensor_channel_mask", implemented_sensor_channel_mask())
        )
        sensor_radius = snap_sensor_radius(genes.get("sensor_radius", DEFAULT_SENSOR_RADIUS))
        introspection_channel_mask = sanitize_introspection_channel_mask(
            genes.get("introspection_channel_mask", implemented_introspection_channel_mask())
        )
        self_regulation_control_mask = sanitize_self_regulation_control_mask(
            genes.get("self_regulation_control_mask", implemented_self_regulation_control_mask())
        )
        self_modification_target_mask = sanitize_self_modification_target_mask(
            genes.get("self_modification_target_mask", implemented_self_modification_target_mask())
        )
        directed_inheritance_channel_mask = sanitize_directed_inheritance_channel_mask(
            genes.get("directed_inheritance_channel_mask", implemented_directed_inheritance_channel_mask())
        )

        drive_inertia = _as_float(genes.get("drive_inertia", 0.85), 0.85)
        drive_noise = _as_float(genes.get("drive_noise", 0.02), 0.02)
        drive_logit_gain = _as_float(genes.get("drive_logit_gain", 1.0), 1.0)
        pred_lr = _as_float(genes.get("pred_lr", 0.08), 0.08)

        drive_sp_hunger = _as_float(genes.get("drive_sp_hunger", 0.55), 0.55)
        drive_sp_pain = _as_float(genes.get("drive_sp_pain", 0.05), 0.05)
        drive_sp_fatigue = _as_float(genes.get("drive_sp_fatigue", 0.15), 0.15)
        drive_sp_fear = _as_float(genes.get("drive_sp_fear", 0.10), 0.10)
        exp_prior_strength = _as_float(genes.get("exp_prior_strength", 0.35), 0.35)
        exp_prior_noise = _as_float(genes.get("exp_prior_noise", 0.05), 0.05)
        drive_birth_noise = _as_float(genes.get("drive_birth_noise", 0.03), 0.03)

        out = Genome(
            hidden_size=hidden_size,
            exploration_bias=exploration_bias,
            aggression_bias=aggression_bias,
            reproduction_threshold=reproduction_threshold,
            mutation_rate=mutation_rate,
            action_temperature=action_temperature,
            use_gru=use_gru,
            memory_dim=memory_dim,
            sensor_channel_mask=sensor_channel_mask,
            sensor_radius=sensor_radius,
            introspection_channel_mask=introspection_channel_mask,
            self_regulation_control_mask=self_regulation_control_mask,
            self_modification_target_mask=self_modification_target_mask,
            directed_inheritance_channel_mask=directed_inheritance_channel_mask,
            drive_inertia=drive_inertia,
            drive_noise=drive_noise,
            drive_logit_gain=drive_logit_gain,
            pred_lr=pred_lr,
            drive_sp_hunger=drive_sp_hunger,
            drive_sp_pain=drive_sp_pain,
            drive_sp_fatigue=drive_sp_fatigue,
            drive_sp_fear=drive_sp_fear,
            exp_prior_strength=exp_prior_strength,
            exp_prior_noise=exp_prior_noise,
            drive_birth_noise=drive_birth_noise,
            schema_version=MODULAR_GENOME_SCHEMA_VERSION,
            interface_schema_version=interface_schema_version,
            b2_invalid_policy=b2_invalid_policy,
            b2_caps=b2_caps,
            b3_locality_schema_version=b3_locality_schema_version,
            b3_locality_max_connection_distance=b3_locality_max_connection_distance,
            b3_distance_cost_scale=b3_distance_cost_scale,
            b3_distance_cost_max=b3_distance_cost_max,
            b3_locality_cost_total=b3_locality_cost_total,
            b3_locality_mean_distance=b3_locality_mean_distance,
            b3_locality_max_distance=b3_locality_max_distance,
            b3_locality_dropped_connections=b3_locality_dropped_connections,
            c1_catalog_version=c1_catalog_version,
            c1_operator_schedule=c1_operator_schedule,
            c1_operator_applied=c1_operator_applied,
            c1_operator_noop_count=c1_operator_noop_count,
            c1_capacity_blocked_count=c1_capacity_blocked_count,
            c1_capacity_blocked_reason_counts_json=c1_capacity_blocked_reason_counts_json,
            c1_noop_reason_counts_json=c1_noop_reason_counts_json,
            escap8_growth_attempt_count=escap8_growth_attempt_count,
            escap8_growth_last_operation=escap8_growth_last_operation,
            escap8_growth_last_status=escap8_growth_last_status,
            escap8_growth_last_reason=escap8_growth_last_reason,
            escap8_blocked_duplication_attempt_count=escap8_blocked_duplication_attempt_count,
            escap8_blocked_prune_attempt_count=escap8_blocked_prune_attempt_count,
            escap8_cap_contact_count=escap8_cap_contact_count,
            escap8_cap_pressure_streak=escap8_cap_pressure_streak,
            escap8_specialization_metrics_json=escap8_specialization_metrics_json,
            structural_bootstrap_enabled=structural_bootstrap_enabled,
            structural_bootstrap_accepted_count=structural_bootstrap_accepted_count,
            c1_operator_budget_status=c1_operator_budget_status,
            c1_op_count__bounded_rewire=c1_op_count__bounded_rewire,
            c1_op_count__duplicate_with_decay=c1_op_count__duplicate_with_decay,
            c1_op_count__edge_split_insertion=c1_op_count__edge_split_insertion,
            c1_op_count__fusion_merge_compression=c1_op_count__fusion_merge_compression,
            c1_op_count__gate_drift=c1_op_count__gate_drift,
            c1_op_count__motif_duplication=c1_op_count__motif_duplication,
            c1_op_count__promoter_expression_mutation=c1_op_count__promoter_expression_mutation,
            c2_recombination_version=c2_recombination_version,
            c2_alignment_key=c2_alignment_key,
            c2_unmatched_policy=c2_unmatched_policy,
            c2_matched_modules=c2_matched_modules,
            c2_unmatched_modules_parent_a=c2_unmatched_modules_parent_a,
            c2_unmatched_modules_parent_b=c2_unmatched_modules_parent_b,
            c2_matched_connections=c2_matched_connections,
            c2_unmatched_connections_parent_a=c2_unmatched_connections_parent_a,
            c2_unmatched_connections_parent_b=c2_unmatched_connections_parent_b,
            c2_budget_draws=c2_budget_draws,
            c2_parent_order_rule=c2_parent_order_rule,
            c2_legacy_unresolved_modules=c2_legacy_unresolved_modules,
            c2_inheritance_profile=c2_inheritance_profile,
            c2_structural_linkage_rule=c2_structural_linkage_rule,
            c2_linked_component_count=c2_linked_component_count,
            modules=modules,
            connections=connections,
            developmental_program=developmental_program,
            resource_handling_axis=resource_handling_axis,
            family_head_residuals=family_head_residuals,
        )
        diploid_payload = genes.get("diploid_scalar_genome_v1")
        if diploid_payload is not None:
            from evolution.diploid_inheritance_v1 import (  # noqa: WPS433
                DiploidScalarGenomeV1,
                attach_diploid_genome,
            )

            attach_diploid_genome(
                out,
                DiploidScalarGenomeV1.from_payload(diploid_payload),
            )
        genome_upgrade_payload = genes.get("genome_upgrade_v2_v1")
        if genome_upgrade_payload is not None:
            from evolution.genome_upgrade_v2 import (  # noqa: WPS433
                GenomeUpgradeV2V1,
                attach_genome_upgrade_v2,
            )

            attach_genome_upgrade_v2(
                out,
                GenomeUpgradeV2V1.from_payload(genome_upgrade_payload),
            )
        return out

    @staticmethod
    def from_trusted_canonical_payload(d: dict) -> "Genome":
        """Fast constructor for payloads already canonicalized by the C1/C2 boundary."""
        if not isinstance(d, dict) or not isinstance(d.get("genes", None), dict):
            return Genome.from_dict(d if isinstance(d, dict) else {})
        schema = str(d.get("schema_version", "") or "")
        if schema and schema != MODULAR_GENOME_SCHEMA_VERSION:
            raise RuntimeError(
                "[DETERMINISM_VIOLATION]\n"
                "code=DET_GENOME_SCHEMA_UNSUPPORTED\n"
                f"context=schema_version={schema}\n"
            )

        def _as_int(x: Any, default: int) -> int:
            try:
                return int(x)
            except Exception:
                return int(default)

        def _as_float(x: Any, default: float) -> float:
            try:
                return float(x)
            except Exception:
                return float(default)

        def _as_bool(x: Any, default: bool) -> bool:
            if isinstance(x, bool):
                return bool(x)
            if isinstance(x, int):
                return bool(int(x)) if int(x) in (0, 1) else bool(default)
            if isinstance(x, str):
                s = str(x).strip().lower()
                if s in {"1", "true", "t", "yes", "y", "on"}:
                    return True
                if s in {"0", "false", "f", "no", "n", "off"}:
                    return False
                return bool(default)
            return bool(x)

        genes = dict(d.get("genes", {}) or {})
        obj = object.__new__(Genome)

        memory_dim = _as_int(genes.get("memory_dim", 0), 0)
        memory_dim = min(_MEMORY_LADDER, key=lambda x: abs(x - memory_dim))
        use_gru = True if int(memory_dim) > 0 else _as_bool(genes.get("use_gru", False), False)

        values: dict[str, Any] = {
            "hidden_size": _as_int(genes.get("hidden_size", 64), 64),
            "exploration_bias": canon_float(_as_float(genes.get("exploration_bias", 0.0), 0.0), field="exploration_bias"),
            "aggression_bias": canon_float(_as_float(genes.get("aggression_bias", 0.0), 0.0), field="aggression_bias"),
            "reproduction_threshold": canon_float(
                _as_float(genes.get("reproduction_threshold", 30.0), 30.0),
                field="reproduction_threshold",
            ),
            "mutation_rate": canon_float(_as_float(genes.get("mutation_rate", 0.05), 0.05), field="mutation_rate"),
            "action_temperature": canon_float(
                _as_float(genes.get("action_temperature", 1.0), 1.0),
                field="action_temperature",
            ),
            "use_gru": bool(use_gru),
            "memory_dim": int(memory_dim),
            "sensor_channel_mask": int(
                sanitize_sensor_channel_mask(genes.get("sensor_channel_mask", implemented_sensor_channel_mask()))
            ),
            "sensor_radius": int(snap_sensor_radius(genes.get("sensor_radius", DEFAULT_SENSOR_RADIUS))),
            "introspection_channel_mask": int(
                sanitize_introspection_channel_mask(
                    genes.get("introspection_channel_mask", implemented_introspection_channel_mask())
                )
            ),
            "self_regulation_control_mask": int(
                sanitize_self_regulation_control_mask(
                    genes.get("self_regulation_control_mask", implemented_self_regulation_control_mask())
                )
            ),
            "self_modification_target_mask": int(
                sanitize_self_modification_target_mask(
                    genes.get("self_modification_target_mask", implemented_self_modification_target_mask())
                )
            ),
            "directed_inheritance_channel_mask": int(
                sanitize_directed_inheritance_channel_mask(
                    genes.get("directed_inheritance_channel_mask", implemented_directed_inheritance_channel_mask())
                )
            ),
            "drive_inertia": canon_float(
                min(0.999, max(0.0, _as_float(genes.get("drive_inertia", 0.85), 0.85))),
                field="drive_inertia",
            ),
            "drive_noise": canon_float(
                min(0.10, max(0.0, _as_float(genes.get("drive_noise", 0.02), 0.02))),
                field="drive_noise",
            ),
            "drive_logit_gain": canon_float(
                min(3.0, max(0.0, _as_float(genes.get("drive_logit_gain", 1.0), 1.0))),
                field="drive_logit_gain",
            ),
            "pred_lr": canon_float(min(1.0, max(0.0, _as_float(genes.get("pred_lr", 0.08), 0.08))), field="pred_lr"),
            "drive_sp_hunger": canon_float(
                min(1.0, max(0.0, _as_float(genes.get("drive_sp_hunger", 0.55), 0.55))),
                field="drive_sp_hunger",
            ),
            "drive_sp_pain": canon_float(
                min(1.0, max(0.0, _as_float(genes.get("drive_sp_pain", 0.05), 0.05))),
                field="drive_sp_pain",
            ),
            "drive_sp_fatigue": canon_float(
                min(1.0, max(0.0, _as_float(genes.get("drive_sp_fatigue", 0.15), 0.15))),
                field="drive_sp_fatigue",
            ),
            "drive_sp_fear": canon_float(
                min(1.0, max(0.0, _as_float(genes.get("drive_sp_fear", 0.10), 0.10))),
                field="drive_sp_fear",
            ),
            "exp_prior_strength": canon_float(
                min(1.0, max(0.0, _as_float(genes.get("exp_prior_strength", 0.35), 0.35))),
                field="exp_prior_strength",
            ),
            "exp_prior_noise": canon_float(
                min(0.5, max(0.0, _as_float(genes.get("exp_prior_noise", 0.05), 0.05))),
                field="exp_prior_noise",
            ),
            "drive_birth_noise": canon_float(
                min(0.2, max(0.0, _as_float(genes.get("drive_birth_noise", 0.03), 0.03))),
                field="drive_birth_noise",
            ),
            "resource_handling_axis": clamp_resource_handling_axis(
                genes.get(RESOURCE_HANDLING_AXIS_FIELD, RESOURCE_HANDLING_AXIS_NEUTRAL)
            ),
            "family_head_residuals": canonicalize_optional_family_head_residual_block(
                genes.get(FAMILY_HEAD_RESIDUAL_FIELD, None)
            ),
            "schema_version": str(d.get("schema_version", MODULAR_GENOME_SCHEMA_VERSION) or MODULAR_GENOME_SCHEMA_VERSION),
            "interface_schema_version": str(d.get("interface_schema_version", B2_INTERFACE_SCHEMA_VERSION) or B2_INTERFACE_SCHEMA_VERSION),
            "b2_invalid_policy": str(d.get("b2_invalid_policy", "clip") or "clip"),
            "b2_caps": resolve_b2_caps(d.get("b2_caps", None)),
            "b3_locality_schema_version": str(
                d.get("b3_locality_schema_version", B3_LOCALITY_SCHEMA_VERSION) or B3_LOCALITY_SCHEMA_VERSION
            ),
            "b3_locality_max_connection_distance": canon_float(
                _as_float(
                    d.get("b3_locality_max_connection_distance", B3_FROZEN_LOCALITY["max_connection_distance"]),
                    float(B3_FROZEN_LOCALITY["max_connection_distance"]),
                ),
                field="b3_locality_max_connection_distance",
            ),
            "b3_distance_cost_scale": canon_float(
                _as_float(d.get("b3_distance_cost_scale", B3_FROZEN_LOCALITY["distance_cost_scale"]), float(B3_FROZEN_LOCALITY["distance_cost_scale"])),
                field="b3_distance_cost_scale",
            ),
            "b3_distance_cost_max": canon_float(
                _as_float(d.get("b3_distance_cost_max", B3_FROZEN_LOCALITY["distance_cost_max"]), float(B3_FROZEN_LOCALITY["distance_cost_max"])),
                field="b3_distance_cost_max",
            ),
            "b3_locality_cost_total": canon_float(
                _as_float(d.get("b3_locality_cost_total", 0.0), 0.0),
                field="b3_locality_cost_total",
            ),
            "b3_locality_mean_distance": canon_float(
                _as_float(d.get("b3_locality_mean_distance", 0.0), 0.0),
                field="b3_locality_mean_distance",
            ),
            "b3_locality_max_distance": canon_float(
                _as_float(d.get("b3_locality_max_distance", 0.0), 0.0),
                field="b3_locality_max_distance",
            ),
            "b3_locality_dropped_connections": _as_int(d.get("b3_locality_dropped_connections", 0), 0),
            "c1_catalog_version": str(genes.get("c1_catalog_version", "") or ""),
            "c1_operator_schedule": str(genes.get("c1_operator_schedule", "") or ""),
            "c1_operator_applied": str(genes.get("c1_operator_applied", "") or ""),
            "c1_operator_noop_count": _as_int(genes.get("c1_operator_noop_count", 0), 0),
            "c1_capacity_blocked_count": _as_int(genes.get("c1_capacity_blocked_count", 0), 0),
            "c1_capacity_blocked_reason_counts_json": str(
                genes.get("c1_capacity_blocked_reason_counts_json", "{}") or "{}"
            ),
            "c1_noop_reason_counts_json": str(genes.get("c1_noop_reason_counts_json", "{}") or "{}"),
            "escap8_growth_attempt_count": _as_int(genes.get("escap8_growth_attempt_count", 0), 0),
            "escap8_growth_last_operation": str(genes.get("escap8_growth_last_operation", "") or ""),
            "escap8_growth_last_status": str(genes.get("escap8_growth_last_status", "") or ""),
            "escap8_growth_last_reason": str(genes.get("escap8_growth_last_reason", "") or ""),
            "escap8_blocked_duplication_attempt_count": _as_int(
                genes.get("escap8_blocked_duplication_attempt_count", 0), 0
            ),
            "escap8_blocked_prune_attempt_count": _as_int(
                genes.get("escap8_blocked_prune_attempt_count", 0), 0
            ),
            "escap8_cap_contact_count": _as_int(genes.get("escap8_cap_contact_count", 0), 0),
            "escap8_cap_pressure_streak": _as_int(genes.get("escap8_cap_pressure_streak", 0), 0),
            "escap8_specialization_metrics_json": str(
                genes.get("escap8_specialization_metrics_json", "{}") or "{}"
            ),
            "structural_bootstrap_enabled": _as_int(genes.get("structural_bootstrap_enabled", 0), 0),
            "structural_bootstrap_accepted_count": _as_int(
                genes.get("structural_bootstrap_accepted_count", 0), 0
            ),
            "c1_operator_budget_status": str(genes.get("c1_operator_budget_status", "") or ""),
            "c1_op_count__bounded_rewire": _as_int(genes.get("c1_op_count__bounded_rewire", 0), 0),
            "c1_op_count__duplicate_with_decay": _as_int(genes.get("c1_op_count__duplicate_with_decay", 0), 0),
            "c1_op_count__edge_split_insertion": _as_int(genes.get("c1_op_count__edge_split_insertion", 0), 0),
            "c1_op_count__fusion_merge_compression": _as_int(genes.get("c1_op_count__fusion_merge_compression", 0), 0),
            "c1_op_count__gate_drift": _as_int(genes.get("c1_op_count__gate_drift", 0), 0),
            "c1_op_count__motif_duplication": _as_int(genes.get("c1_op_count__motif_duplication", 0), 0),
            "c1_op_count__promoter_expression_mutation": _as_int(genes.get("c1_op_count__promoter_expression_mutation", 0), 0),
            "c2_recombination_version": str(genes.get("c2_recombination_version", "") or ""),
            "c2_alignment_key": str(genes.get("c2_alignment_key", "") or ""),
            "c2_unmatched_policy": str(genes.get("c2_unmatched_policy", "") or ""),
            "c2_matched_modules": _as_int(genes.get("c2_matched_modules", 0), 0),
            "c2_unmatched_modules_parent_a": _as_int(genes.get("c2_unmatched_modules_parent_a", 0), 0),
            "c2_unmatched_modules_parent_b": _as_int(genes.get("c2_unmatched_modules_parent_b", 0), 0),
            "c2_matched_connections": _as_int(genes.get("c2_matched_connections", 0), 0),
            "c2_unmatched_connections_parent_a": _as_int(genes.get("c2_unmatched_connections_parent_a", 0), 0),
            "c2_unmatched_connections_parent_b": _as_int(genes.get("c2_unmatched_connections_parent_b", 0), 0),
            "c2_budget_draws": _as_int(genes.get("c2_budget_draws", 0), 0),
            "c2_parent_order_rule": str(genes.get("c2_parent_order_rule", "") or ""),
            "c2_legacy_unresolved_modules": _as_int(genes.get("c2_legacy_unresolved_modules", 0), 0),
            "c2_inheritance_profile": str(genes.get("c2_inheritance_profile", "") or ""),
            "c2_structural_linkage_rule": str(genes.get("c2_structural_linkage_rule", "") or ""),
            "c2_linked_component_count": _as_int(genes.get("c2_linked_component_count", 0), 0),
            "modules": list(d.get("modules", []) or []),
            "connections": list(d.get("connections", []) or []),
            "developmental_program": canonicalize_developmental_program(d.get("developmental_program", None)),
        }
        for key, value in values.items():
            setattr(obj, key, value)
        diploid_payload = genes.get("diploid_scalar_genome_v1")
        if diploid_payload is not None:
            from evolution.diploid_inheritance_v1 import (  # noqa: WPS433
                DiploidScalarGenomeV1,
                attach_diploid_genome,
            )

            attach_diploid_genome(
                obj,
                DiploidScalarGenomeV1.from_payload(diploid_payload),
            )
        genome_upgrade_payload = genes.get("genome_upgrade_v2_v1")
        if genome_upgrade_payload is not None:
            from evolution.genome_upgrade_v2 import (  # noqa: WPS433
                GenomeUpgradeV2V1,
                attach_genome_upgrade_v2,
            )

            attach_genome_upgrade_v2(
                obj,
                GenomeUpgradeV2V1.from_payload(genome_upgrade_payload),
            )
        return obj

    @staticmethod
    def random(
        rng: random.Random,
        *,
        b2_caps: dict[str, int] | None = None,
    ) -> "Genome":
        hidden_size = 64
        memory_dim = rng.choices(
            population=_MEMORY_LADDER,
            weights=[0.55, 0.20, 0.15, 0.07, 0.03],
            k=1,
        )[0]
        use_gru = bool(memory_dim > 0)

        return Genome(
            hidden_size=hidden_size,
            exploration_bias=rng.uniform(-0.5, 0.5),
            aggression_bias=rng.uniform(-0.5, 0.5),
            reproduction_threshold=rng.uniform(20.0, 50.0),
            mutation_rate=rng.uniform(0.01, 0.08),
            action_temperature=rng.uniform(0.8, 1.2),
            use_gru=use_gru,
            memory_dim=int(memory_dim),
            sensor_channel_mask=implemented_sensor_channel_mask(),
            sensor_radius=DEFAULT_SENSOR_RADIUS,
            introspection_channel_mask=implemented_introspection_channel_mask(),
            self_regulation_control_mask=implemented_self_regulation_control_mask(),
            self_modification_target_mask=implemented_self_modification_target_mask(),
            drive_inertia=rng.uniform(0.75, 0.95),
            drive_noise=rng.uniform(0.00, 0.05),
            drive_logit_gain=rng.uniform(0.6, 1.6),
            pred_lr=rng.uniform(0.03, 0.18),
            drive_sp_hunger=rng.uniform(0.35, 0.85),
            drive_sp_pain=rng.uniform(0.0, 0.25),
            drive_sp_fatigue=rng.uniform(0.0, 0.45),
            drive_sp_fear=rng.uniform(0.0, 0.45),
            exp_prior_strength=rng.uniform(0.15, 0.65),
            exp_prior_noise=rng.uniform(0.00, 0.12),
            drive_birth_noise=rng.uniform(0.00, 0.08),
            schema_version=MODULAR_GENOME_SCHEMA_VERSION,
            interface_schema_version=B2_INTERFACE_SCHEMA_VERSION,
            b2_invalid_policy="clip",
            b2_caps=resolve_b2_caps(b2_caps),
            b3_locality_schema_version=B3_LOCALITY_SCHEMA_VERSION,
            b3_locality_max_connection_distance=float(B3_FROZEN_LOCALITY["max_connection_distance"]),
            b3_distance_cost_scale=float(B3_FROZEN_LOCALITY["distance_cost_scale"]),
            b3_distance_cost_max=float(B3_FROZEN_LOCALITY["distance_cost_max"]),
            b3_locality_cost_total=0.0,
            b3_locality_mean_distance=0.0,
            b3_locality_max_distance=0.0,
            b3_locality_dropped_connections=0,
            c1_catalog_version="",
            c1_operator_schedule="",
            c1_operator_applied="",
            c1_operator_noop_count=0,
            c1_capacity_blocked_count=0,
            c1_capacity_blocked_reason_counts_json="{}",
            c1_noop_reason_counts_json="{}",
            structural_bootstrap_enabled=0,
            structural_bootstrap_accepted_count=0,
            c1_operator_budget_status="",
            c1_op_count__bounded_rewire=0,
            c1_op_count__duplicate_with_decay=0,
            c1_op_count__edge_split_insertion=0,
            c1_op_count__fusion_merge_compression=0,
            c1_op_count__gate_drift=0,
            c1_op_count__motif_duplication=0,
            c1_op_count__promoter_expression_mutation=0,
            c2_recombination_version="",
            c2_alignment_key="",
            c2_unmatched_policy="",
            c2_matched_modules=0,
            c2_unmatched_modules_parent_a=0,
            c2_unmatched_modules_parent_b=0,
            c2_matched_connections=0,
            c2_unmatched_connections_parent_a=0,
            c2_unmatched_connections_parent_b=0,
            c2_budget_draws=0,
            c2_parent_order_rule="",
            c2_legacy_unresolved_modules=0,
            developmental_program=canonicalize_developmental_program(None),
            resource_handling_axis=RESOURCE_HANDLING_AXIS_NEUTRAL,
        )

    def developmental_stage_id(self, *, age_steps: int) -> str:
        return resolve_stage_id(canonicalize_developmental_program(self.developmental_program), age_steps=int(age_steps))

    def resolve_developmental_expression(self, *, age_steps: int) -> dict[str, Any]:
        _diag, _payload_key, _t0 = _envhot7_developmental_diag_payload(self.developmental_program)
        stage_id = self.developmental_stage_id(age_steps=int(age_steps))
        modules = sorted(list(self.modules or []), key=lambda m: str(m.get("module_id", "")))
        active_by_id: dict[str, bool] = {}
        reasons: dict[str, str] = {}
        for m in modules:
            mid = str(m.get("module_id", ""))
            active, reason, _ = resolve_module_active(
                canonicalize_developmental_program(self.developmental_program),
                module_id=mid,
                age_steps=int(age_steps),
            )
            active_by_id[mid] = bool(active)
            reasons[mid] = str(reason)

        out = {
            "aggression_bias": float(self.aggression_bias),
            "action_temperature": float(self.action_temperature),
            "drive_logit_gain": float(self.drive_logit_gain),
            "drive_inertia": float(self.drive_inertia),
            "drive_noise": float(self.drive_noise),
            "pred_lr": float(self.pred_lr),
            "reproduction_threshold": float(self.reproduction_threshold),
        }
        for m in modules:
            mid = str(m.get("module_id", ""))
            if not bool(active_by_id.get(mid, True)):
                continue
            mtype = str(m.get("module_type", ""))
            params = m.get("params", {}) if isinstance(m.get("params", {}), dict) else {}
            if mtype == "behavior_bias":
                if "aggression_bias" in params:
                    out["aggression_bias"] = float(params.get("aggression_bias"))
                if "action_temperature" in params:
                    out["action_temperature"] = float(params.get("action_temperature"))
            elif mtype == "identity_regulation":
                if "drive_logit_gain" in params:
                    out["drive_logit_gain"] = float(params.get("drive_logit_gain"))
                if "drive_inertia" in params:
                    out["drive_inertia"] = float(params.get("drive_inertia"))
                if "drive_noise" in params:
                    out["drive_noise"] = float(params.get("drive_noise"))
                if "pred_lr" in params:
                    out["pred_lr"] = float(params.get("pred_lr"))
            elif mtype == "life_history":
                if "reproduction_threshold" in params:
                    out["reproduction_threshold"] = float(params.get("reproduction_threshold"))

        result = {
            "stage_id": stage_id,
            "active_by_module_id": active_by_id,
            "inactive_reason_by_module_id": reasons,
            "active_module_count": int(sum(1 for v in active_by_id.values() if bool(v))),
            "module_count": int(len(active_by_id)),
            "module_structure_signature": module_structure_signature(modules),
            "effective": out,
        }
        if _diag is not None:
            try:
                _diag.record_developmental_genome_resolve(
                    genome_key=f"genome_obj:{id(self)}",
                    program_key=str(_payload_key),
                    age_steps=int(age_steps),
                    stage_id=str(stage_id),
                    module_count=int(len(active_by_id)),
                    seconds=time.perf_counter() - _t0,
                )
            except Exception:
                pass
        return result
