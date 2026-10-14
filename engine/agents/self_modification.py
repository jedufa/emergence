from __future__ import annotations

"""Bounded self-modification helpers for Theme 4 Stage 7.

This module defines lifecycle-scoped simulated phenotype offsets. It never
mutates genomes, mutation rates, host state, source code, simulator internals,
RNG state, metrics, checkpoint machinery, files, network state, or tools.
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

import numpy as np


SELF_MODIFICATION_TARGET_REGISTRY_SCHEMA_VERSION = "self_modification_target_registry_v1"
SELF_MODIFICATION_EXPRESSION_SCHEMA_VERSION = "self_modification_expression_v1"
SELF_MODIFICATION_RUNTIME_SCHEMA_VERSION = "self_modification_runtime_state_v1"


FORBIDDEN_SELF_MODIFICATION_TARGETS: tuple[str, ...] = (
    "genome",
    "genome_fields",
    "mutation_rate",
    "mutation_rates",
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
    "energy_direct_set",
    "health_direct_set",
    "position_direct_set",
    "age_direct_set",
    "reproduction_success_direct_set",
    "reward_direct_set",
    "fitness_direct_set",
    "death_direct_set",
    "host_state",
    "directed_inheritance",
)


@dataclass(frozen=True)
class SelfModificationTargetMetadata:
    target_id: str
    display_name: str
    category: str
    active_now: bool
    future_reserved: bool
    default_enabled: bool
    modifiable: bool
    cost_weight: float
    target_variable: str
    update_rule: str
    min_value: float
    max_value: float
    max_delta_per_event: float
    max_total_offset: float
    neutral_modification_value: float
    decay_or_recovery_rule: str
    persistence_scope: str
    reset_condition: str
    checkpoint_required: bool
    metric_name_prefix: str
    safe_for_agent_modification: bool
    can_affect_genome: bool
    can_affect_host_or_simulator: bool
    can_affect_source_code: bool
    direct_reward_risk: bool
    description: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _target(
    target_id: str,
    display_name: str,
    *,
    category: str,
    active_now: bool,
    target_variable: str,
    min_value: float,
    max_value: float,
    max_delta_per_event: float,
    max_total_offset: float,
    future_reserved: bool = False,
    default_enabled: bool = True,
    modifiable: bool = True,
    cost_weight: float = 1.0,
    update_rule: str = "bounded_lifetime_offset_add_with_clamp",
    neutral: float = 0.0,
    decay_rule: str = "linear_decay_toward_zero",
    persistence_scope: str = "lifetime_runtime_offset",
    reset_condition: str = "birth_death_reproduction_if_configured",
    checkpoint_required: bool = True,
    safe: bool = True,
    direct_reward_risk: bool = False,
    description: str = "",
) -> SelfModificationTargetMetadata:
    return SelfModificationTargetMetadata(
        target_id=str(target_id),
        display_name=str(display_name),
        category=str(category),
        active_now=bool(active_now),
        future_reserved=bool(future_reserved),
        default_enabled=bool(default_enabled),
        modifiable=bool(modifiable),
        cost_weight=float(cost_weight),
        target_variable=str(target_variable),
        update_rule=str(update_rule),
        min_value=float(min_value),
        max_value=float(max_value),
        max_delta_per_event=float(max_delta_per_event),
        max_total_offset=float(max_total_offset),
        neutral_modification_value=float(neutral),
        decay_or_recovery_rule=str(decay_rule),
        persistence_scope=str(persistence_scope),
        reset_condition=str(reset_condition),
        checkpoint_required=bool(checkpoint_required),
        metric_name_prefix=f"self_modification_target_{target_id}",
        safe_for_agent_modification=bool(safe),
        can_affect_genome=False,
        can_affect_host_or_simulator=False,
        can_affect_source_code=False,
        direct_reward_risk=bool(direct_reward_risk),
        description=str(description),
    )


_SELF_MODIFICATION_TARGETS: tuple[SelfModificationTargetMetadata, ...] = (
    _target(
        "exploration_bias_lifetime_offset",
        "Exploration Bias Lifetime Offset",
        category="action_policy",
        active_now=True,
        target_variable="lifetime_exploration_bias_offset",
        min_value=-0.5,
        max_value=0.5,
        max_delta_per_event=0.025,
        max_total_offset=0.5,
        description="Lifecycle-scoped offset on exploration bias; not inherited.",
    ),
    _target(
        "action_temperature_lifetime_offset",
        "Action Temperature Lifetime Offset",
        category="action_policy",
        active_now=True,
        target_variable="lifetime_action_temperature_offset",
        min_value=-0.5,
        max_value=0.5,
        max_delta_per_event=0.025,
        max_total_offset=0.5,
        description="Lifecycle-scoped action-temperature offset; fixed shape only.",
    ),
    _target(
        "drive_gain_lifetime_offset",
        "Drive Gain Lifetime Offset",
        category="drive_regulation",
        active_now=True,
        target_variable="lifetime_drive_logit_gain_offset",
        min_value=-0.75,
        max_value=0.75,
        max_delta_per_event=0.025,
        max_total_offset=0.75,
        description="Lifecycle-scoped drive-gain offset; does not set drives directly.",
    ),
    _target(
        "reproduction_threshold_lifetime_offset",
        "Reproduction Threshold Lifetime Offset",
        category="reproduction_timing",
        active_now=True,
        target_variable="lifetime_reproduction_threshold_offset",
        min_value=-10.0,
        max_value=10.0,
        max_delta_per_event=0.5,
        max_total_offset=10.0,
        description="Lifecycle-scoped threshold offset; does not set reproduction success.",
    ),
    _target(
        "prediction_lr_lifetime_offset",
        "Prediction Learning Rate Lifetime Offset",
        category="prediction_update",
        active_now=True,
        target_variable="lifetime_prediction_lr_offset",
        min_value=-0.25,
        max_value=0.25,
        max_delta_per_event=0.0125,
        max_total_offset=0.25,
        description="Lifecycle-scoped prediction-learning-rate offset.",
    ),
    _target(
        "fear_weight_lifetime_offset",
        "Fear Weight Lifetime Offset",
        category="future_reserved",
        active_now=False,
        future_reserved=True,
        default_enabled=False,
        modifiable=False,
        target_variable="future_fear_weight_lifetime_offset",
        min_value=0.0,
        max_value=0.0,
        max_delta_per_event=0.0,
        max_total_offset=0.0,
        checkpoint_required=False,
        description="Reserved until drive weighting has a safe phenotype-offset hook.",
    ),
    _target(
        "hunger_weight_lifetime_offset",
        "Hunger Weight Lifetime Offset",
        category="future_reserved",
        active_now=False,
        future_reserved=True,
        default_enabled=False,
        modifiable=False,
        target_variable="future_hunger_weight_lifetime_offset",
        min_value=0.0,
        max_value=0.0,
        max_delta_per_event=0.0,
        max_total_offset=0.0,
        checkpoint_required=False,
        description="Reserved until drive weighting has a safe phenotype-offset hook.",
    ),
    _target(
        "memory_allocation_lifetime_offset",
        "Memory Allocation Lifetime Offset",
        category="future_reserved",
        active_now=False,
        future_reserved=True,
        default_enabled=False,
        modifiable=False,
        target_variable="future_memory_allocation_lifetime_offset",
        min_value=0.0,
        max_value=0.0,
        max_delta_per_event=0.0,
        max_total_offset=0.0,
        checkpoint_required=False,
        description="Reserved; Stage 7 does not change memory architecture or tensor shape.",
    ),
    _target(
        "cognitive_budget_allocation_offset",
        "Cognitive Budget Allocation Offset",
        category="future_reserved",
        active_now=False,
        future_reserved=True,
        default_enabled=False,
        modifiable=False,
        target_variable="future_cognitive_budget_allocation_offset",
        min_value=0.0,
        max_value=0.0,
        max_delta_per_event=0.0,
        max_total_offset=0.0,
        checkpoint_required=False,
        description="Reserved for future bounded allocation; not active in Stage 7.",
    ),
)

_TARGET_INDEX = {t.target_id: i for i, t in enumerate(_SELF_MODIFICATION_TARGETS)}
_TARGET_BY_ID = {t.target_id: t for t in _SELF_MODIFICATION_TARGETS}


def self_modification_target_registry() -> tuple[SelfModificationTargetMetadata, ...]:
    return _SELF_MODIFICATION_TARGETS


def self_modification_target_registry_dict() -> dict[str, object]:
    return {
        "schema_version": SELF_MODIFICATION_TARGET_REGISTRY_SCHEMA_VERSION,
        "targets": [t.to_dict() for t in _SELF_MODIFICATION_TARGETS],
        "forbidden_targets": list(FORBIDDEN_SELF_MODIFICATION_TARGETS),
    }


def forbidden_self_modification_targets() -> tuple[str, ...]:
    return FORBIDDEN_SELF_MODIFICATION_TARGETS


def get_self_modification_target(target_id: str) -> SelfModificationTargetMetadata:
    return _TARGET_BY_ID[str(target_id)]


def implemented_self_modification_target_ids() -> tuple[str, ...]:
    return tuple(
        t.target_id
        for t in _SELF_MODIFICATION_TARGETS
        if t.active_now
        and not t.future_reserved
        and t.modifiable
        and t.safe_for_agent_modification
        and not t.can_affect_genome
        and not t.can_affect_host_or_simulator
        and not t.can_affect_source_code
    )


def implemented_self_modification_target_mask() -> int:
    mask = 0
    for tid in implemented_self_modification_target_ids():
        mask |= 1 << int(_TARGET_INDEX[tid])
    return int(mask)


def sanitize_self_modification_target_mask(mask: Any) -> int:
    try:
        out = int(mask)
    except Exception:
        out = implemented_self_modification_target_mask()
    return int(out & implemented_self_modification_target_mask())


def self_modification_target_mask_contains(mask: int, target_id: str) -> bool:
    idx = int(_TARGET_INDEX[str(target_id)])
    return bool(int(mask) & (1 << idx))


def self_modification_target_ids_from_mask(mask: int) -> tuple[str, ...]:
    mask = sanitize_self_modification_target_mask(mask)
    return tuple(tid for tid in implemented_self_modification_target_ids() if self_modification_target_mask_contains(mask, tid))


def recombine_self_modification_target_mask(a: Any, b: Any, rng: Any) -> int:
    ma = sanitize_self_modification_target_mask(a)
    mb = sanitize_self_modification_target_mask(b)
    out = 0
    for tid in implemented_self_modification_target_ids():
        bit = 1 << int(_TARGET_INDEX[tid])
        if bool(rng.random() < 0.5):
            out |= ma & bit
        else:
            out |= mb & bit
    return sanitize_self_modification_target_mask(out)


def mutate_self_modification_target_mask(mask: Any, rng: Any, rate: float) -> int:
    out = sanitize_self_modification_target_mask(mask)
    p = max(0.0, min(1.0, float(rate)))
    if rng.random() >= p:
        return out
    ids = implemented_self_modification_target_ids()
    if not ids:
        return out
    tid = ids[int(rng.randrange(len(ids)))]
    out ^= 1 << int(_TARGET_INDEX[tid])
    return sanitize_self_modification_target_mask(out)


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


def _neutral_vector() -> tuple[float, ...]:
    return tuple(0.0 for _ in implemented_self_modification_target_ids())


def self_modification_ablation_flags(sim_cfg: Any = None) -> dict[str, bool]:
    ablate = _cfg_bool(sim_cfg, "ablate_self_modification", False)
    force_no = _cfg_bool(sim_cfg, "force_no_self_modification", False)
    force_all = _cfg_bool(sim_cfg, "force_all_self_modification_targets", False)
    force_default = _cfg_bool(sim_cfg, "force_default_self_modification_targets", False)
    mask_neutral = _cfg_bool(sim_cfg, "mask_self_modification_commands_to_neutral", False)
    freeze = _cfg_bool(sim_cfg, "freeze_self_modification_state", False)
    reset_each_tick = _cfg_bool(sim_cfg, "reset_self_modification_each_tick", False)
    reset_on_reproduction = _cfg_bool(sim_cfg, "reset_self_modification_on_reproduction", False)
    disable_decay = _cfg_bool(sim_cfg, "disable_self_modification_decay", False)
    return {
        "ablate": bool(ablate),
        "force_no": bool(force_no),
        "force_all": bool(force_all),
        "force_default": bool(force_default),
        "mask_to_neutral": bool(mask_neutral),
        "freeze": bool(freeze),
        "reset_each_tick": bool(reset_each_tick),
        "reset_on_reproduction": bool(reset_on_reproduction),
        "disable_decay": bool(disable_decay),
        "active": bool(ablate or force_no or force_all or force_default or mask_neutral or freeze or reset_each_tick or reset_on_reproduction or disable_decay),
    }


@dataclass(frozen=True)
class SelfModificationExpression:
    schema_version: str
    self_modification_enabled: bool
    self_modification_policy_output_enabled: bool
    active_self_modification_target_count: int
    potential_self_modification_target_count: int
    active_self_modification_target_mask: int
    active_self_modification_target_ids: tuple[str, ...]
    self_modification_capacity_fraction: float
    self_modification_cost_hypothetical: float
    self_modification_cost_charged: float
    self_modification_ablation_active: bool
    force_no_self_modification_active: bool
    force_all_self_modification_active: bool
    force_default_self_modification_active: bool
    mask_self_modification_commands_to_neutral_active: bool
    freeze_self_modification_state_active: bool
    reset_self_modification_each_tick_active: bool
    reset_self_modification_on_reproduction_active: bool
    disable_self_modification_decay_active: bool
    self_modification_runtime_state_available: bool
    self_modification_lifetime_persistent: bool
    self_modification_inheritable: bool
    genome_editing_active: bool

    def to_dict(self) -> dict[str, object]:
        out = asdict(self)
        out["active_self_modification_target_ids"] = list(self.active_self_modification_target_ids)
        return out


def express_self_modification(genome: Any, *, sim_cfg: Any = None) -> SelfModificationExpression:
    potential = len(implemented_self_modification_target_ids())
    feature_enabled = _cfg_bool(sim_cfg, "enable_self_modification", False)
    evolvable = _cfg_bool(sim_cfg, "enable_evolvable_self_modification_targets", False)
    flags = self_modification_ablation_flags(sim_cfg)
    default_mask = implemented_self_modification_target_mask() if _cfg_bool(sim_cfg, "default_self_modification_targets_enabled", False) else 0
    raw_mask = sanitize_self_modification_target_mask(_gget(genome, "self_modification_target_mask", default_mask))
    if flags["force_all"]:
        active_mask = implemented_self_modification_target_mask()
    elif flags["force_no"] or flags["ablate"]:
        active_mask = 0
    elif flags["force_default"] or not evolvable:
        active_mask = default_mask
    else:
        active_mask = raw_mask
    if not feature_enabled:
        active_mask = 0
    active_mask = sanitize_self_modification_target_mask(active_mask)
    active_ids = self_modification_target_ids_from_mask(active_mask)
    active_weight = sum(float(_TARGET_BY_ID[tid].cost_weight) for tid in active_ids)
    base = max(0.0, _cfg_float(sim_cfg, "self_modification_command_cost_base", 0.0))
    per_target = max(0.0, _cfg_float(sim_cfg, "self_modification_target_cost", 0.001))
    hypothetical = float(base + per_target * active_weight) if feature_enabled else 0.0
    charge = _cfg_bool(sim_cfg, "enable_experimental_self_modification_cost_charging", False)
    return SelfModificationExpression(
        schema_version=SELF_MODIFICATION_EXPRESSION_SCHEMA_VERSION,
        self_modification_enabled=bool(feature_enabled and len(active_ids) > 0 and not flags["mask_to_neutral"]),
        self_modification_policy_output_enabled=False,
        active_self_modification_target_count=int(len(active_ids)),
        potential_self_modification_target_count=int(potential),
        active_self_modification_target_mask=int(active_mask),
        active_self_modification_target_ids=tuple(active_ids),
        self_modification_capacity_fraction=float(len(active_ids)) / float(potential) if potential > 0 else 0.0,
        self_modification_cost_hypothetical=float(hypothetical),
        self_modification_cost_charged=float(hypothetical if charge else 0.0),
        self_modification_ablation_active=bool(flags["active"]),
        force_no_self_modification_active=bool(flags["force_no"] or flags["ablate"]),
        force_all_self_modification_active=bool(flags["force_all"]),
        force_default_self_modification_active=bool(flags["force_default"]),
        mask_self_modification_commands_to_neutral_active=bool(flags["mask_to_neutral"]),
        freeze_self_modification_state_active=bool(flags["freeze"]),
        reset_self_modification_each_tick_active=bool(flags["reset_each_tick"]),
        reset_self_modification_on_reproduction_active=bool(flags["reset_on_reproduction"]),
        disable_self_modification_decay_active=bool(flags["disable_decay"]),
        self_modification_runtime_state_available=bool(feature_enabled),
        self_modification_lifetime_persistent=bool(feature_enabled),
        self_modification_inheritable=False,
        genome_editing_active=False,
    )


@dataclass
class SelfModificationRuntimeState:
    schema_version: str = SELF_MODIFICATION_RUNTIME_SCHEMA_VERSION
    active_target_mask: int = 0
    offsets: dict[str, float] = field(default_factory=dict)
    lifetime_history: dict[str, float] = field(default_factory=dict)
    last_modification_command_vector: tuple[float, ...] = field(default_factory=_neutral_vector)
    applied_modification_deltas: dict[str, float] = field(default_factory=dict)
    clipped_modification_deltas: dict[str, float] = field(default_factory=dict)
    modification_event_count: int = 0
    modification_saturation_count: int = 0
    modification_reset_count: int = 0
    modification_decay_count: int = 0
    modification_reversal_count: int = 0
    modification_cost: float = 0.0
    baseline_phenotype_values: dict[str, float] = field(default_factory=dict)
    effective_modified_phenotype_values: dict[str, float] = field(default_factory=dict)
    ablated: bool = False
    forced: bool = False
    checkpoint_required: bool = True

    def to_dict(self) -> dict[str, object]:
        out = asdict(self)
        out["last_modification_command_vector"] = list(self.last_modification_command_vector)
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "SelfModificationRuntimeState":
        if not isinstance(data, Mapping):
            return cls()
        state = cls()
        state.active_target_mask = sanitize_self_modification_target_mask(data.get("active_target_mask", 0))
        state.offsets = {str(k): _finite(v, 0.0) for k, v in dict(data.get("offsets", {}) or {}).items()}
        state.lifetime_history = {str(k): _finite(v, 0.0) for k, v in dict(data.get("lifetime_history", {}) or {}).items()}
        raw_vec = list(data.get("last_modification_command_vector", []) or [])
        n = len(implemented_self_modification_target_ids())
        state.last_modification_command_vector = tuple(_clip(raw_vec[i], -1.0, 1.0, 0.0) if i < len(raw_vec) else 0.0 for i in range(n))
        state.applied_modification_deltas = {str(k): _finite(v, 0.0) for k, v in dict(data.get("applied_modification_deltas", {}) or {}).items()}
        state.clipped_modification_deltas = {str(k): _finite(v, 0.0) for k, v in dict(data.get("clipped_modification_deltas", {}) or {}).items()}
        for name in ("modification_event_count", "modification_saturation_count", "modification_reset_count", "modification_decay_count", "modification_reversal_count"):
            try:
                setattr(state, name, int(data.get(name, 0) or 0))
            except Exception:
                setattr(state, name, 0)
        state.modification_cost = max(0.0, _finite(data.get("modification_cost", 0.0), 0.0))
        state.baseline_phenotype_values = {str(k): _finite(v, 0.0) for k, v in dict(data.get("baseline_phenotype_values", {}) or {}).items()}
        state.effective_modified_phenotype_values = {str(k): _finite(v, 0.0) for k, v in dict(data.get("effective_modified_phenotype_values", {}) or {}).items()}
        state.ablated = bool(data.get("ablated", False))
        state.forced = bool(data.get("forced", False))
        state.checkpoint_required = bool(data.get("checkpoint_required", True))
        return state


def _baseline_values(genome: Any) -> dict[str, float]:
    return {
        "exploration_bias_lifetime_offset": _finite(_gget(genome, "exploration_bias", 0.0), 0.0),
        "action_temperature_lifetime_offset": _finite(_gget(genome, "action_temperature", 1.0), 1.0),
        "drive_gain_lifetime_offset": _finite(_gget(genome, "drive_logit_gain", 1.0), 1.0),
        "reproduction_threshold_lifetime_offset": _finite(_gget(genome, "reproduction_threshold", 30.0), 30.0),
        "prediction_lr_lifetime_offset": _finite(_gget(genome, "pred_lr", 0.08), 0.08),
    }


def _effective_bounds(target_id: str) -> tuple[float, float]:
    if target_id == "exploration_bias_lifetime_offset":
        return -1.0, 1.0
    if target_id == "action_temperature_lifetime_offset":
        return 0.05, 3.0
    if target_id == "drive_gain_lifetime_offset":
        return 0.0, 3.0
    if target_id == "reproduction_threshold_lifetime_offset":
        return 0.0, 120.0
    if target_id == "prediction_lr_lifetime_offset":
        return 0.0, 1.0
    return -1.0, 1.0


def _vector(command_vector: Any) -> tuple[float, ...]:
    ids = implemented_self_modification_target_ids()
    if command_vector is None:
        raw: list[Any] = []
    else:
        try:
            raw = list(command_vector)
        except Exception:
            raw = []
    return tuple(_clip(raw[i], -1.0, 1.0, 0.0) if i < len(raw) else 0.0 for i in range(len(ids)))


def apply_self_modification_commands(
    state: SelfModificationRuntimeState | Mapping[str, Any] | None,
    command_vector: Any,
    *,
    genome: Any = None,
    sim_cfg: Any = None,
) -> SelfModificationRuntimeState:
    current = state if isinstance(state, SelfModificationRuntimeState) else SelfModificationRuntimeState.from_dict(state)
    expression = express_self_modification(genome, sim_cfg=sim_cfg)
    flags = self_modification_ablation_flags(sim_cfg)
    ids = implemented_self_modification_target_ids()
    values = _neutral_vector() if flags["mask_to_neutral"] else _vector(command_vector)

    if flags["reset_each_tick"]:
        current.offsets = {}
        current.modification_reset_count += 1

    current.active_target_mask = int(expression.active_self_modification_target_mask)
    current.last_modification_command_vector = tuple(values)
    current.applied_modification_deltas = {}
    current.clipped_modification_deltas = {}
    current.baseline_phenotype_values = _baseline_values(genome)
    current.ablated = bool(expression.self_modification_ablation_active)
    current.forced = bool(expression.force_all_self_modification_active or expression.force_no_self_modification_active or expression.force_default_self_modification_active)
    current.checkpoint_required = True

    if not expression.self_modification_runtime_state_available or flags["force_no"] or flags["ablate"] or flags["freeze"]:
        current.effective_modified_phenotype_values = dict(current.baseline_phenotype_values)
        return current

    decay = 0.0 if flags["disable_decay"] else max(0.0, min(1.0, _cfg_float(sim_cfg, "self_modification_offset_decay", 0.01)))
    if decay > 0.0 and current.offsets:
        for tid, old in list(current.offsets.items()):
            new = _finite(old, 0.0) * (1.0 - decay)
            current.offsets[tid] = float(new)
        current.modification_decay_count += 1

    active = set(expression.active_self_modification_target_ids)
    event_cost = 0.0
    for idx, tid in enumerate(ids):
        meta = _TARGET_BY_ID[tid]
        raw = float(values[idx])
        if tid not in active:
            raw = float(meta.neutral_modification_value)
        bounded_raw = _clip(raw, -1.0, 1.0, 0.0)
        proposed_delta = bounded_raw * float(meta.max_delta_per_event)
        old_offset = _finite(current.offsets.get(tid, 0.0), 0.0)
        unclipped = old_offset + proposed_delta
        new_offset = _clip(unclipped, -float(meta.max_total_offset), float(meta.max_total_offset), 0.0)
        new_offset = _clip(new_offset, float(meta.min_value), float(meta.max_value), 0.0)
        clipped_delta = new_offset - old_offset
        current.offsets[tid] = float(new_offset)
        current.applied_modification_deltas[tid] = float(proposed_delta)
        current.clipped_modification_deltas[tid] = float(clipped_delta)
        current.lifetime_history[tid] = float(current.lifetime_history.get(tid, 0.0) + abs(clipped_delta))
        if abs(clipped_delta - proposed_delta) > 1e-12:
            current.modification_saturation_count += 1
        if old_offset != 0.0 and clipped_delta != 0.0 and (old_offset > 0.0) != (clipped_delta > 0.0):
            current.modification_reversal_count += 1
        event_cost += abs(clipped_delta) * float(meta.cost_weight)

    current.modification_event_count += 1
    current.modification_cost = float(max(0.0, _cfg_float(sim_cfg, "self_modification_update_cost_scale", 0.001)) * event_cost)
    effective: dict[str, float] = {}
    for tid, baseline in current.baseline_phenotype_values.items():
        lo, hi = _effective_bounds(tid)
        effective[tid] = _clip(float(baseline) + float(current.offsets.get(tid, 0.0)), lo, hi, float(baseline))
    current.effective_modified_phenotype_values = effective
    return current


def reset_self_modification_for_reproduction(
    state: SelfModificationRuntimeState | Mapping[str, Any] | None,
    *,
    sim_cfg: Any = None,
) -> SelfModificationRuntimeState:
    current = state if isinstance(state, SelfModificationRuntimeState) else SelfModificationRuntimeState.from_dict(state)
    if _cfg_bool(sim_cfg, "reset_self_modification_on_reproduction", False):
        current.offsets = {}
        current.modification_reset_count += 1
        current.effective_modified_phenotype_values = dict(current.baseline_phenotype_values)
    return current


def self_modification_runtime_summary(state: Any) -> dict[str, float]:
    current = state if isinstance(state, SelfModificationRuntimeState) else SelfModificationRuntimeState.from_dict(state)
    deltas = [abs(float(v)) for v in current.clipped_modification_deltas.values()]
    offsets = [abs(float(v)) for v in current.offsets.values()]
    active = max(1, len(implemented_self_modification_target_ids()))
    neutral = sum(1 for v in current.last_modification_command_vector if abs(float(v)) <= 1e-12)
    return {
        "delta_abs_mean": float(sum(deltas) / len(deltas)) if deltas else 0.0,
        "delta_abs_p90": float(np.percentile(np.asarray(deltas, dtype=np.float64), 90)) if deltas else 0.0,
        "offset_abs_mean": float(sum(offsets) / len(offsets)) if offsets else 0.0,
        "offset_abs_p90": float(np.percentile(np.asarray(offsets, dtype=np.float64), 90)) if offsets else 0.0,
        "clipped_fraction": float(current.modification_saturation_count) / float(max(1, current.modification_event_count * active)),
        "saturation_fraction": float(current.modification_saturation_count) / float(max(1, current.modification_event_count * active)),
        "event_count": float(current.modification_event_count),
        "neutral_fraction": float(neutral) / float(active),
        "reset_count": float(current.modification_reset_count),
        "decay_count": float(current.modification_decay_count),
        "lifetime_persistence": float(sum(offsets)),
        "reversal_count": float(current.modification_reversal_count),
        "cost": float(current.modification_cost),
        "modified_exploration_bias": float(current.effective_modified_phenotype_values.get("exploration_bias_lifetime_offset", current.baseline_phenotype_values.get("exploration_bias_lifetime_offset", 0.0))),
        "modified_action_temperature": float(current.effective_modified_phenotype_values.get("action_temperature_lifetime_offset", current.baseline_phenotype_values.get("action_temperature_lifetime_offset", 1.0))),
        "modified_drive_gain": float(current.effective_modified_phenotype_values.get("drive_gain_lifetime_offset", current.baseline_phenotype_values.get("drive_gain_lifetime_offset", 1.0))),
        "modified_reproduction_threshold": float(current.effective_modified_phenotype_values.get("reproduction_threshold_lifetime_offset", current.baseline_phenotype_values.get("reproduction_threshold_lifetime_offset", 30.0))),
        "modified_prediction_lr": float(current.effective_modified_phenotype_values.get("prediction_lr_lifetime_offset", current.baseline_phenotype_values.get("prediction_lr_lifetime_offset", 0.08))),
    }
