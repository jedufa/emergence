from __future__ import annotations

"""Bounded directed-inheritance helpers for Theme 4 Stage 8.

This module defines a non-genomic parent-to-offspring influence surface. It
never edits genomes, mutation rates, recombination rules, host state, source
code, simulator internals, RNG state, metrics, checkpoint machinery, files,
network state, or tools.
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

import numpy as np


DIRECTED_INHERITANCE_REGISTRY_SCHEMA_VERSION = "directed_inheritance_registry_v1"
DIRECTED_INHERITANCE_EXPRESSION_SCHEMA_VERSION = "directed_inheritance_expression_v1"
DIRECTED_INHERITANCE_PAYLOAD_SCHEMA_VERSION = "directed_inheritance_payload_v1"


FORBIDDEN_DIRECTED_INHERITANCE_TARGETS: tuple[str, ...] = (
    "genome_direct_edit",
    "genome_fields",
    "mutation_rate_edit",
    "mutation_rates",
    "recombination_rule_edit",
    "recombination_rules",
    "source_code",
    "simulator_code",
    "filesystem",
    "network",
    "external_tools",
    "rng_state",
    "checkpoint_internals",
    "replay_internals",
    "metrics_internals",
    "direct_reward_set",
    "direct_fitness_set",
    "direct_reproduction_success_set",
    "direct_death_state_set",
    "arbitrary_child_energy_set",
    "arbitrary_child_health_set",
    "arbitrary_child_position_set",
    "arbitrary_child_age_set",
    "host_state",
    "self_modification_state_inheritance",
    "self_regulation_state_inheritance",
)


@dataclass(frozen=True)
class DirectedInheritanceChannelMetadata:
    influence_id: str
    display_name: str
    category: str
    active_now: bool
    future_reserved: bool
    default_enabled: bool
    maskable: bool
    cost_weight: float
    target_offspring_state: str
    update_rule: str
    min_value: float
    max_value: float
    max_delta_per_reproduction: float
    neutral_influence_value: float
    persistence_scope: str
    reset_condition: str
    checkpoint_required: bool
    metric_name_prefix: str
    safe_for_directed_inheritance: bool
    can_affect_genome: bool
    can_affect_mutation_rate: bool
    can_affect_recombination: bool
    can_affect_host_or_simulator: bool
    can_affect_source_code: bool
    direct_reward_risk: bool
    description: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _channel(
    influence_id: str,
    display_name: str,
    *,
    category: str,
    active_now: bool,
    target_offspring_state: str,
    min_value: float,
    max_value: float,
    max_delta_per_reproduction: float,
    future_reserved: bool = False,
    default_enabled: bool = True,
    maskable: bool = True,
    cost_weight: float = 1.0,
    update_rule: str = "bounded_non_genomic_offset_add_with_clamp",
    neutral: float = 0.0,
    persistence_scope: str = "offspring_lifecycle_non_genomic_payload",
    reset_condition: str = "birth_or_maturity_if_configured",
    checkpoint_required: bool = True,
    safe: bool = True,
    direct_reward_risk: bool = False,
    description: str = "",
) -> DirectedInheritanceChannelMetadata:
    return DirectedInheritanceChannelMetadata(
        influence_id=str(influence_id),
        display_name=str(display_name),
        category=str(category),
        active_now=bool(active_now),
        future_reserved=bool(future_reserved),
        default_enabled=bool(default_enabled),
        maskable=bool(maskable),
        cost_weight=float(cost_weight),
        target_offspring_state=str(target_offspring_state),
        update_rule=str(update_rule),
        min_value=float(min_value),
        max_value=float(max_value),
        max_delta_per_reproduction=float(max_delta_per_reproduction),
        neutral_influence_value=float(neutral),
        persistence_scope=str(persistence_scope),
        reset_condition=str(reset_condition),
        checkpoint_required=bool(checkpoint_required),
        metric_name_prefix=f"directed_inheritance_channel_{influence_id}",
        safe_for_directed_inheritance=bool(safe),
        can_affect_genome=False,
        can_affect_mutation_rate=False,
        can_affect_recombination=False,
        can_affect_host_or_simulator=False,
        can_affect_source_code=False,
        direct_reward_risk=bool(direct_reward_risk),
        description=str(description),
    )


_DIRECTED_INHERITANCE_CHANNELS: tuple[DirectedInheritanceChannelMetadata, ...] = (
    _channel(
        "offspring_developmental_expression_bias",
        "Offspring Developmental Expression Bias",
        category="developmental_bias",
        active_now=True,
        target_offspring_state="non_genomic_developmental_expression_bias",
        min_value=-0.25,
        max_value=0.25,
        max_delta_per_reproduction=0.05,
        description="Bounded non-genomic bias surfaced in offspring developmental phenotype expression.",
    ),
    _channel(
        "offspring_juvenile_cognition_scale_bias",
        "Offspring Juvenile Cognition Scale Bias",
        category="juvenile_cognition_bias",
        active_now=True,
        target_offspring_state="non_genomic_juvenile_cognition_scale_bias",
        min_value=-0.25,
        max_value=0.25,
        max_delta_per_reproduction=0.05,
        description="Bounded non-genomic juvenile-cognition scale bias; genome remains unchanged.",
    ),
    _channel(
        "offspring_memory_expression_bias",
        "Offspring Memory Expression Bias",
        category="offspring_phenotype_bias",
        active_now=True,
        target_offspring_state="non_genomic_memory_expression_bias",
        min_value=-0.25,
        max_value=0.25,
        max_delta_per_reproduction=0.05,
        description="Telemetry phenotype bias marker for memory expression; no tensor-shape change.",
    ),
    _channel(
        "offspring_prediction_lr_bias",
        "Offspring Prediction Learning Rate Bias",
        category="learning_bias",
        active_now=True,
        target_offspring_state="non_genomic_prediction_lr_bias",
        min_value=-0.20,
        max_value=0.20,
        max_delta_per_reproduction=0.025,
        description="Bounded non-genomic prediction learning-rate phenotype bias.",
    ),
    _channel(
        "offspring_exploration_bias_marker",
        "Offspring Exploration Bias Marker",
        category="regulatory_bias",
        active_now=True,
        target_offspring_state="non_genomic_exploration_bias_marker",
        min_value=-0.25,
        max_value=0.25,
        max_delta_per_reproduction=0.05,
        description="Bounded exploration-bias marker for offspring phenotype telemetry.",
    ),
    _channel(
        "offspring_reproduction_threshold_bias",
        "Offspring Reproduction Threshold Bias",
        category="regulatory_bias",
        active_now=True,
        target_offspring_state="non_genomic_reproduction_threshold_bias",
        min_value=-5.0,
        max_value=5.0,
        max_delta_per_reproduction=1.0,
        description="Bounded reproduction-threshold phenotype bias; does not set reproduction success.",
    ),
    _channel(
        "parental_energy_investment",
        "Parental Energy Investment",
        category="future_reserved",
        active_now=False,
        future_reserved=True,
        default_enabled=False,
        maskable=False,
        target_offspring_state="future_parental_energy_investment",
        min_value=0.0,
        max_value=0.0,
        max_delta_per_reproduction=0.0,
        checkpoint_required=False,
        description="Reserved because arbitrary child energy setting is forbidden in Stage 8.",
    ),
    _channel(
        "stress_shaped_development_marker",
        "Stress-Shaped Development Marker",
        category="future_reserved",
        active_now=False,
        future_reserved=True,
        default_enabled=False,
        maskable=False,
        target_offspring_state="future_stress_shaped_development_marker",
        min_value=0.0,
        max_value=0.0,
        max_delta_per_reproduction=0.0,
        checkpoint_required=False,
        description="Reserved until stress semantics are stable and safely bounded.",
    ),
    _channel(
        "offspring_introspection_bias",
        "Offspring Introspection Bias",
        category="future_reserved",
        active_now=False,
        future_reserved=True,
        default_enabled=False,
        maskable=False,
        target_offspring_state="future_offspring_introspection_bias",
        min_value=0.0,
        max_value=0.0,
        max_delta_per_reproduction=0.0,
        checkpoint_required=False,
        description="Reserved; Stage 8 does not inherit introspection access or state.",
    ),
    _channel(
        "offspring_self_regulation_bias",
        "Offspring Self-Regulation Bias",
        category="future_reserved",
        active_now=False,
        future_reserved=True,
        default_enabled=False,
        maskable=False,
        target_offspring_state="future_offspring_self_regulation_bias",
        min_value=0.0,
        max_value=0.0,
        max_delta_per_reproduction=0.0,
        checkpoint_required=False,
        description="Reserved; self-regulation runtime state is not inherited.",
    ),
)

_CHANNEL_INDEX = {c.influence_id: i for i, c in enumerate(_DIRECTED_INHERITANCE_CHANNELS)}
_CHANNEL_BY_ID = {c.influence_id: c for c in _DIRECTED_INHERITANCE_CHANNELS}


def directed_inheritance_channel_registry() -> tuple[DirectedInheritanceChannelMetadata, ...]:
    return _DIRECTED_INHERITANCE_CHANNELS


def directed_inheritance_registry_dict() -> dict[str, object]:
    return {
        "schema_version": DIRECTED_INHERITANCE_REGISTRY_SCHEMA_VERSION,
        "channels": [c.to_dict() for c in _DIRECTED_INHERITANCE_CHANNELS],
        "forbidden_targets": list(FORBIDDEN_DIRECTED_INHERITANCE_TARGETS),
    }


def forbidden_directed_inheritance_targets() -> tuple[str, ...]:
    return FORBIDDEN_DIRECTED_INHERITANCE_TARGETS


def implemented_directed_inheritance_channel_ids() -> tuple[str, ...]:
    return tuple(
        c.influence_id
        for c in _DIRECTED_INHERITANCE_CHANNELS
        if c.active_now
        and not c.future_reserved
        and c.maskable
        and c.safe_for_directed_inheritance
        and not c.can_affect_genome
        and not c.can_affect_mutation_rate
        and not c.can_affect_recombination
        and not c.can_affect_host_or_simulator
        and not c.can_affect_source_code
    )


def implemented_directed_inheritance_channel_mask() -> int:
    mask = 0
    for influence_id in implemented_directed_inheritance_channel_ids():
        mask |= 1 << int(_CHANNEL_INDEX[influence_id])
    return int(mask)


def sanitize_directed_inheritance_channel_mask(mask: Any) -> int:
    try:
        out = int(mask)
    except Exception:
        out = implemented_directed_inheritance_channel_mask()
    return int(out & implemented_directed_inheritance_channel_mask())


def directed_inheritance_channel_mask_contains(mask: int, influence_id: str) -> bool:
    return bool(int(mask) & (1 << int(_CHANNEL_INDEX[str(influence_id)])))


def directed_inheritance_channel_ids_from_mask(mask: int) -> tuple[str, ...]:
    mask = sanitize_directed_inheritance_channel_mask(mask)
    return tuple(
        cid for cid in implemented_directed_inheritance_channel_ids() if directed_inheritance_channel_mask_contains(mask, cid)
    )


def recombine_directed_inheritance_channel_mask(a: Any, b: Any, rng: Any) -> int:
    ma = sanitize_directed_inheritance_channel_mask(a)
    mb = sanitize_directed_inheritance_channel_mask(b)
    out = 0
    for cid in implemented_directed_inheritance_channel_ids():
        bit = 1 << int(_CHANNEL_INDEX[cid])
        out |= (ma if bool(rng.random() < 0.5) else mb) & bit
    return sanitize_directed_inheritance_channel_mask(out)


def mutate_directed_inheritance_channel_mask(mask: Any, rng: Any, rate: float) -> int:
    out = sanitize_directed_inheritance_channel_mask(mask)
    p = max(0.0, min(1.0, float(rate)))
    if rng.random() >= p:
        return out
    ids = implemented_directed_inheritance_channel_ids()
    if not ids:
        return out
    cid = ids[int(rng.randrange(len(ids)))]
    out ^= 1 << int(_CHANNEL_INDEX[cid])
    return sanitize_directed_inheritance_channel_mask(out)


def _cfg_bool(obj: Any, name: str, default: bool = False) -> bool:
    try:
        return bool(getattr(obj, name))
    except Exception:
        return bool(default)


def _cfg_float(obj: Any, name: str, default: float = 0.0) -> float:
    try:
        return float(getattr(obj, name))
    except Exception:
        return float(default)


def _gget(g: Any, name: str, default: Any) -> Any:
    if g is None:
        return default
    if hasattr(g, name):
        try:
            return getattr(g, name)
        except Exception:
            return default
    if isinstance(g, Mapping):
        return g.get(name, default)
    return default


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        out = float(default)
    if not np.isfinite(out):
        out = float(default)
    return float(out)


def _clip(value: Any, lo: float, hi: float, default: float = 0.0) -> float:
    v = _finite(value, default)
    return float(max(float(lo), min(float(hi), v)))


def directed_inheritance_ablation_flags(sim_cfg: Any = None) -> dict[str, bool]:
    ablate = _cfg_bool(sim_cfg, "ablate_directed_inheritance", False)
    force_no = _cfg_bool(sim_cfg, "force_no_directed_inheritance", False)
    force_all = _cfg_bool(sim_cfg, "force_all_directed_inheritance_channels", False)
    force_default = _cfg_bool(sim_cfg, "force_default_directed_inheritance_channels", False)
    mask_neutral = _cfg_bool(sim_cfg, "mask_directed_inheritance_commands_to_neutral", False)
    freeze = _cfg_bool(sim_cfg, "freeze_directed_inheritance_payload", False)
    reset_birth = _cfg_bool(sim_cfg, "reset_directed_inheritance_payload_on_birth", False)
    reset_maturity = _cfg_bool(sim_cfg, "reset_directed_inheritance_payload_on_maturity", False)
    disable_costs = _cfg_bool(sim_cfg, "disable_directed_inheritance_costs", False)
    return {
        "ablate": bool(ablate),
        "force_no": bool(force_no),
        "force_all": bool(force_all),
        "force_default": bool(force_default),
        "mask_to_neutral": bool(mask_neutral),
        "freeze": bool(freeze),
        "reset_on_birth": bool(reset_birth),
        "reset_on_maturity": bool(reset_maturity),
        "disable_costs": bool(disable_costs),
        "active": bool(ablate or force_no or force_all or force_default or mask_neutral or freeze or reset_birth or reset_maturity or disable_costs),
    }


@dataclass(frozen=True)
class DirectedInheritanceExpression:
    schema_version: str
    directed_inheritance_enabled: bool
    directed_inheritance_surface_enabled: bool
    directed_inheritance_policy_output_enabled: bool
    active_directed_inheritance_channel_count: int
    potential_directed_inheritance_channel_count: int
    active_directed_inheritance_channel_mask: int
    active_directed_inheritance_channel_ids: tuple[str, ...]
    directed_inheritance_capacity_fraction: float
    directed_inheritance_cost_hypothetical: float
    directed_inheritance_cost_charged: float
    directed_inheritance_ablation_active: bool
    force_no_directed_inheritance_active: bool
    force_all_directed_inheritance_active: bool
    force_default_directed_inheritance_active: bool
    mask_directed_inheritance_commands_to_neutral_active: bool
    freeze_directed_inheritance_payload_active: bool
    reset_directed_inheritance_payload_on_birth_active: bool
    reset_directed_inheritance_payload_on_maturity_active: bool
    directed_inheritance_runtime_state_available: bool
    non_genomic_payload_available: bool
    non_genomic_payload_applied: bool
    genome_editing_active: bool
    mutation_rate_control_active: bool
    recombination_control_active: bool
    inherited_self_modification_state_active: bool

    def to_dict(self) -> dict[str, object]:
        out = asdict(self)
        out["active_directed_inheritance_channel_ids"] = list(self.active_directed_inheritance_channel_ids)
        return out


def express_directed_inheritance(genome: Any, *, payload: Any = None, sim_cfg: Any = None) -> DirectedInheritanceExpression:
    potential = len(implemented_directed_inheritance_channel_ids())
    feature_enabled = _cfg_bool(sim_cfg, "enable_directed_inheritance", False) or _cfg_bool(sim_cfg, "enable_directed_inheritance_surface", False)
    evolvable = _cfg_bool(sim_cfg, "enable_evolvable_directed_inheritance_channels", False)
    flags = directed_inheritance_ablation_flags(sim_cfg)
    default_mask = implemented_directed_inheritance_channel_mask() if _cfg_bool(sim_cfg, "default_directed_inheritance_channels_enabled", False) else 0
    raw_mask = sanitize_directed_inheritance_channel_mask(_gget(genome, "directed_inheritance_channel_mask", default_mask))
    if flags["force_all"]:
        active_mask = implemented_directed_inheritance_channel_mask()
    elif flags["force_no"] or flags["ablate"]:
        active_mask = 0
    elif flags["force_default"] or not evolvable:
        active_mask = default_mask
    else:
        active_mask = raw_mask
    if not feature_enabled:
        active_mask = 0
    active_mask = sanitize_directed_inheritance_channel_mask(active_mask)
    active_ids = directed_inheritance_channel_ids_from_mask(active_mask)
    active_weight = sum(float(_CHANNEL_BY_ID[cid].cost_weight) for cid in active_ids)
    base = max(0.0, _cfg_float(sim_cfg, "directed_inheritance_command_cost_base", 0.0))
    per_channel = max(0.0, _cfg_float(sim_cfg, "directed_inheritance_channel_cost", 0.001))
    disabled_costs = bool(flags["disable_costs"])
    hypothetical = 0.0 if disabled_costs or not feature_enabled else float(base + per_channel * active_weight)
    charge = _cfg_bool(sim_cfg, "enable_experimental_directed_inheritance_cost_charging", False) and not disabled_costs
    payload_obj = payload if isinstance(payload, DirectedInheritancePayload) else DirectedInheritancePayload.from_dict(payload)
    return DirectedInheritanceExpression(
        schema_version=DIRECTED_INHERITANCE_EXPRESSION_SCHEMA_VERSION,
        directed_inheritance_enabled=bool(feature_enabled and len(active_ids) > 0 and not flags["mask_to_neutral"]),
        directed_inheritance_surface_enabled=bool(feature_enabled),
        directed_inheritance_policy_output_enabled=False,
        active_directed_inheritance_channel_count=int(len(active_ids)),
        potential_directed_inheritance_channel_count=int(potential),
        active_directed_inheritance_channel_mask=int(active_mask),
        active_directed_inheritance_channel_ids=tuple(active_ids),
        directed_inheritance_capacity_fraction=float(len(active_ids)) / float(potential) if potential > 0 else 0.0,
        directed_inheritance_cost_hypothetical=float(hypothetical),
        directed_inheritance_cost_charged=float(hypothetical if charge else 0.0),
        directed_inheritance_ablation_active=bool(flags["active"]),
        force_no_directed_inheritance_active=bool(flags["force_no"] or flags["ablate"]),
        force_all_directed_inheritance_active=bool(flags["force_all"]),
        force_default_directed_inheritance_active=bool(flags["force_default"]),
        mask_directed_inheritance_commands_to_neutral_active=bool(flags["mask_to_neutral"]),
        freeze_directed_inheritance_payload_active=bool(flags["freeze"]),
        reset_directed_inheritance_payload_on_birth_active=bool(flags["reset_on_birth"]),
        reset_directed_inheritance_payload_on_maturity_active=bool(flags["reset_on_maturity"]),
        directed_inheritance_runtime_state_available=bool(feature_enabled),
        non_genomic_payload_available=True,
        non_genomic_payload_applied=bool(payload_obj.payload_applied and feature_enabled),
        genome_editing_active=False,
        mutation_rate_control_active=False,
        recombination_control_active=False,
        inherited_self_modification_state_active=False,
    )


def _neutral_vector() -> tuple[float, ...]:
    return tuple(0.0 for _ in implemented_directed_inheritance_channel_ids())


@dataclass
class DirectedInheritancePayload:
    schema_version: str = DIRECTED_INHERITANCE_PAYLOAD_SCHEMA_VERSION
    active_channel_mask: int = 0
    applied_influence_values: dict[str, float] = field(default_factory=dict)
    clipped_influence_values: dict[str, float] = field(default_factory=dict)
    parental_influence_cost: float = 0.0
    source_parent_summary: dict[str, float] = field(default_factory=dict)
    offspring_developmental_bias_offsets: dict[str, float] = field(default_factory=dict)
    offspring_phenotype_bias_offsets: dict[str, float] = field(default_factory=dict)
    influence_event_count: int = 0
    influence_saturation_count: int = 0
    influence_reset_count: int = 0
    payload_applied: bool = False
    payload_frozen: bool = False
    checkpoint_required: bool = True

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "DirectedInheritancePayload":
        if isinstance(data, DirectedInheritancePayload):
            return data
        if not isinstance(data, Mapping):
            return cls()
        payload = cls()
        payload.active_channel_mask = sanitize_directed_inheritance_channel_mask(data.get("active_channel_mask", 0))
        payload.applied_influence_values = {str(k): _finite(v, 0.0) for k, v in dict(data.get("applied_influence_values", {}) or {}).items()}
        payload.clipped_influence_values = {str(k): _finite(v, 0.0) for k, v in dict(data.get("clipped_influence_values", {}) or {}).items()}
        payload.parental_influence_cost = max(0.0, _finite(data.get("parental_influence_cost", 0.0), 0.0))
        payload.source_parent_summary = {str(k): _finite(v, 0.0) for k, v in dict(data.get("source_parent_summary", {}) or {}).items()}
        payload.offspring_developmental_bias_offsets = {
            str(k): _finite(v, 0.0) for k, v in dict(data.get("offspring_developmental_bias_offsets", {}) or {}).items()
        }
        payload.offspring_phenotype_bias_offsets = {
            str(k): _finite(v, 0.0) for k, v in dict(data.get("offspring_phenotype_bias_offsets", {}) or {}).items()
        }
        for name in ("influence_event_count", "influence_saturation_count", "influence_reset_count"):
            try:
                setattr(payload, name, int(data.get(name, 0) or 0))
            except Exception:
                setattr(payload, name, 0)
        payload.payload_applied = bool(data.get("payload_applied", False))
        payload.payload_frozen = bool(data.get("payload_frozen", False))
        payload.checkpoint_required = bool(data.get("checkpoint_required", True))
        return payload


def _vector(command_vector: Any) -> tuple[float, ...]:
    ids = implemented_directed_inheritance_channel_ids()
    if command_vector is None:
        raw: list[Any] = []
    else:
        try:
            raw = list(command_vector)
        except Exception:
            raw = []
    return tuple(_clip(raw[i], -1.0, 1.0, 0.0) if i < len(raw) else 0.0 for i in range(len(ids)))


def _target_dict_for_channel(influence_id: str) -> str:
    if influence_id in {
        "offspring_developmental_expression_bias",
        "offspring_juvenile_cognition_scale_bias",
        "offspring_memory_expression_bias",
    }:
        return "developmental"
    return "phenotype"


def apply_directed_inheritance_commands(
    payload: DirectedInheritancePayload | Mapping[str, Any] | None,
    command_vector: Any,
    *,
    parent_genome: Any = None,
    offspring_genome: Any = None,
    sim_cfg: Any = None,
) -> DirectedInheritancePayload:
    current = payload if isinstance(payload, DirectedInheritancePayload) else DirectedInheritancePayload.from_dict(payload)
    expression = express_directed_inheritance(parent_genome, payload=current, sim_cfg=sim_cfg)
    flags = directed_inheritance_ablation_flags(sim_cfg)
    ids = implemented_directed_inheritance_channel_ids()
    values = _neutral_vector() if flags["mask_to_neutral"] else _vector(command_vector)

    if flags["reset_on_birth"]:
        current.applied_influence_values = {}
        current.clipped_influence_values = {}
        current.offspring_developmental_bias_offsets = {}
        current.offspring_phenotype_bias_offsets = {}
        current.influence_reset_count += 1

    current.active_channel_mask = int(expression.active_directed_inheritance_channel_mask)
    current.payload_frozen = bool(flags["freeze"])
    current.checkpoint_required = True
    current.source_parent_summary = {
        "parent_mutation_rate": _finite(_gget(parent_genome, "mutation_rate", 0.0), 0.0),
        "offspring_mutation_rate": _finite(_gget(offspring_genome, "mutation_rate", 0.0), 0.0),
    }

    if (
        not expression.directed_inheritance_enabled
        or flags["force_no"]
        or flags["ablate"]
        or flags["freeze"]
    ):
        current.payload_applied = False
        return current

    active = set(expression.active_directed_inheritance_channel_ids)
    event_cost = 0.0
    current.applied_influence_values = {}
    current.clipped_influence_values = {}
    for idx, cid in enumerate(ids):
        meta = _CHANNEL_BY_ID[cid]
        raw = float(values[idx])
        if cid not in active:
            raw = float(meta.neutral_influence_value)
        bounded_raw = _clip(raw, -1.0, 1.0, 0.0)
        proposed_delta = bounded_raw * float(meta.max_delta_per_reproduction)
        clipped_delta = _clip(proposed_delta, float(meta.min_value), float(meta.max_value), 0.0)
        current.applied_influence_values[cid] = float(proposed_delta)
        current.clipped_influence_values[cid] = float(clipped_delta)
        target = current.offspring_developmental_bias_offsets if _target_dict_for_channel(cid) == "developmental" else current.offspring_phenotype_bias_offsets
        old_offset = _finite(target.get(cid, 0.0), 0.0)
        new_offset = _clip(old_offset + clipped_delta, float(meta.min_value), float(meta.max_value), 0.0)
        target[cid] = float(new_offset)
        if abs(new_offset - old_offset - proposed_delta) > 1e-12:
            current.influence_saturation_count += 1
        event_cost += abs(new_offset - old_offset) * float(meta.cost_weight)

    current.influence_event_count += 1
    current.parental_influence_cost = 0.0 if flags["disable_costs"] else float(max(0.0, _cfg_float(sim_cfg, "directed_inheritance_update_cost_scale", 0.001)) * event_cost)
    current.payload_applied = bool(any(abs(v) > 1e-12 for v in list(current.offspring_developmental_bias_offsets.values()) + list(current.offspring_phenotype_bias_offsets.values())))
    return current


def reset_directed_inheritance_payload(payload: DirectedInheritancePayload | Mapping[str, Any] | None) -> DirectedInheritancePayload:
    current = payload if isinstance(payload, DirectedInheritancePayload) else DirectedInheritancePayload.from_dict(payload)
    current.applied_influence_values = {}
    current.clipped_influence_values = {}
    current.offspring_developmental_bias_offsets = {}
    current.offspring_phenotype_bias_offsets = {}
    current.influence_reset_count += 1
    current.payload_applied = False
    return current


def directed_inheritance_payload_summary(payload: Any) -> dict[str, float]:
    current = payload if isinstance(payload, DirectedInheritancePayload) else DirectedInheritancePayload.from_dict(payload)
    deltas = [abs(float(v)) for v in current.clipped_influence_values.values()]
    offsets = [abs(float(v)) for v in list(current.offspring_developmental_bias_offsets.values()) + list(current.offspring_phenotype_bias_offsets.values())]
    active = max(1, len(implemented_directed_inheritance_channel_ids()))
    neutral = active - sum(1 for v in current.clipped_influence_values.values() if abs(float(v)) > 1e-12)
    return {
        "delta_abs_mean": float(sum(deltas) / len(deltas)) if deltas else 0.0,
        "delta_abs_p90": float(np.percentile(np.asarray(deltas, dtype=np.float64), 90)) if deltas else 0.0,
        "payload_abs_mean": float(sum(offsets) / len(offsets)) if offsets else 0.0,
        "payload_abs_p90": float(np.percentile(np.asarray(offsets, dtype=np.float64), 90)) if offsets else 0.0,
        "clipped_fraction": float(current.influence_saturation_count) / float(max(1, current.influence_event_count * active)),
        "saturation_fraction": float(current.influence_saturation_count) / float(max(1, current.influence_event_count * active)),
        "event_count": float(current.influence_event_count),
        "neutral_command_fraction": float(neutral) / float(active),
        "payload_reset_count": float(current.influence_reset_count),
        "payload_applied_fraction": 1.0 if current.payload_applied else 0.0,
        "payload_frozen_fraction": 1.0 if current.payload_frozen else 0.0,
        "cost": float(current.parental_influence_cost),
        "offspring_developmental_expression_bias": float(current.offspring_developmental_bias_offsets.get("offspring_developmental_expression_bias", 0.0)),
        "offspring_juvenile_cognition_scale_bias": float(current.offspring_developmental_bias_offsets.get("offspring_juvenile_cognition_scale_bias", 0.0)),
        "offspring_memory_expression_bias": float(current.offspring_developmental_bias_offsets.get("offspring_memory_expression_bias", 0.0)),
        "offspring_prediction_lr_bias": float(current.offspring_phenotype_bias_offsets.get("offspring_prediction_lr_bias", 0.0)),
        "offspring_exploration_bias_marker": float(current.offspring_phenotype_bias_offsets.get("offspring_exploration_bias_marker", 0.0)),
        "offspring_reproduction_threshold_bias": float(current.offspring_phenotype_bias_offsets.get("offspring_reproduction_threshold_bias", 0.0)),
    }
