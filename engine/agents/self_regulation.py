from __future__ import annotations

"""Bounded self-regulation helpers for Theme 4 Stage 6.

This module defines a deterministic runtime control surface. It adjusts only
bounded simulated regulatory offsets and never mutates genomes, host state,
source code, simulator internals, RNG state, or checkpoint machinery.
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

import numpy as np


SELF_REGULATION_CONTROL_REGISTRY_SCHEMA_VERSION = "self_regulation_control_registry_v1"
SELF_REGULATION_EXPRESSION_SCHEMA_VERSION = "self_regulation_expression_v1"
SELF_REGULATION_RUNTIME_SCHEMA_VERSION = "self_regulation_runtime_state_v1"


FORBIDDEN_SELF_REGULATION_TARGETS: tuple[str, ...] = (
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
)


@dataclass(frozen=True)
class SelfRegulationControlMetadata:
    control_id: str
    display_name: str
    category: str
    active_now: bool
    future_reserved: bool
    default_enabled: bool
    maskable: bool
    cost_weight: float
    target_variable: str
    update_rule: str
    min_value: float
    max_value: float
    max_delta_per_tick: float
    neutral_control_value: float
    decay_or_recovery_rule: str
    persistence_scope: str
    metric_name_prefix: str
    safe_for_agent_control: bool
    can_affect_genome: bool
    can_affect_host_or_simulator: bool
    can_affect_source_code: bool
    direct_reward_risk: bool
    description: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _control(
    control_id: str,
    display_name: str,
    *,
    category: str,
    active_now: bool,
    target_variable: str,
    min_value: float,
    max_value: float,
    max_delta_per_tick: float,
    future_reserved: bool = False,
    default_enabled: bool = True,
    maskable: bool = True,
    cost_weight: float = 1.0,
    update_rule: str = "bounded_offset_add_with_clamp",
    neutral: float = 0.0,
    decay_rule: str = "linear_decay_toward_zero",
    persistence_scope: str = "lifetime_runtime_offset",
    safe: bool = True,
    direct_reward_risk: bool = False,
    description: str = "",
) -> SelfRegulationControlMetadata:
    return SelfRegulationControlMetadata(
        control_id=str(control_id),
        display_name=str(display_name),
        category=str(category),
        active_now=bool(active_now),
        future_reserved=bool(future_reserved),
        default_enabled=bool(default_enabled),
        maskable=bool(maskable),
        cost_weight=float(cost_weight),
        target_variable=str(target_variable),
        update_rule=str(update_rule),
        min_value=float(min_value),
        max_value=float(max_value),
        max_delta_per_tick=float(max_delta_per_tick),
        neutral_control_value=float(neutral),
        decay_or_recovery_rule=str(decay_rule),
        persistence_scope=str(persistence_scope),
        metric_name_prefix=f"self_regulation_control_{control_id}",
        safe_for_agent_control=bool(safe),
        can_affect_genome=False,
        can_affect_host_or_simulator=False,
        can_affect_source_code=False,
        direct_reward_risk=bool(direct_reward_risk),
        description=str(description),
    )


_SELF_REGULATION_CONTROLS: tuple[SelfRegulationControlMetadata, ...] = (
    _control(
        "exploration_pressure_adjustment",
        "Exploration Pressure Adjustment",
        category="exploration",
        active_now=True,
        target_variable="runtime_exploration_bias_offset",
        min_value=-0.5,
        max_value=0.5,
        max_delta_per_tick=0.05,
        description="Bounded lifetime runtime offset on exploration bias; not a reward.",
    ),
    _control(
        "action_temperature_adjustment",
        "Action Temperature Adjustment",
        category="action_policy",
        active_now=True,
        target_variable="runtime_action_temperature_offset",
        min_value=-0.5,
        max_value=0.5,
        max_delta_per_tick=0.05,
        description="Bounded runtime temperature offset; effective temperature remains clamped by policy helpers.",
    ),
    _control(
        "drive_gain_adjustment",
        "Drive Gain Adjustment",
        category="drive_regulation",
        active_now=True,
        target_variable="runtime_drive_logit_gain_offset",
        min_value=-0.75,
        max_value=0.75,
        max_delta_per_tick=0.05,
        description="Bounded runtime offset for drive logit gain.",
    ),
    _control(
        "reproduction_threshold_adjustment",
        "Reproduction Threshold Adjustment",
        category="reproduction_timing",
        active_now=True,
        target_variable="runtime_reproduction_threshold_offset",
        min_value=-10.0,
        max_value=10.0,
        max_delta_per_tick=1.0,
        description="Bounded runtime threshold offset; does not set reproduction success.",
    ),
    _control(
        "prediction_lr_adjustment",
        "Prediction Learning Rate Adjustment",
        category="prediction_update",
        active_now=True,
        target_variable="runtime_prediction_lr_offset",
        min_value=-0.25,
        max_value=0.25,
        max_delta_per_tick=0.025,
        description="Bounded runtime offset for effective prediction learning rate.",
    ),
    _control(
        "fear_weight_adjustment",
        "Fear Weight Adjustment",
        category="future_reserved",
        active_now=False,
        future_reserved=True,
        default_enabled=False,
        target_variable="future_fear_weight_offset",
        min_value=0.0,
        max_value=0.0,
        max_delta_per_tick=0.0,
        description="Reserved until drive weighting has a safe runtime hook.",
    ),
    _control(
        "hunger_weight_adjustment",
        "Hunger Weight Adjustment",
        category="future_reserved",
        active_now=False,
        future_reserved=True,
        default_enabled=False,
        target_variable="future_hunger_weight_offset",
        min_value=0.0,
        max_value=0.0,
        max_delta_per_tick=0.0,
        description="Reserved until drive weighting has a safe runtime hook.",
    ),
    _control(
        "memory_use_gate_adjustment",
        "Memory Use Gate Adjustment",
        category="future_reserved",
        active_now=False,
        future_reserved=True,
        default_enabled=False,
        target_variable="future_memory_use_gate_offset",
        min_value=0.0,
        max_value=0.0,
        max_delta_per_tick=0.0,
        description="Reserved; no active memory-use gate is exposed in Stage 6.",
    ),
    _control(
        "cognitive_budget_allocation",
        "Cognitive Budget Allocation",
        category="future_reserved",
        active_now=False,
        future_reserved=True,
        default_enabled=False,
        target_variable="future_cognitive_budget_allocation",
        min_value=0.0,
        max_value=0.0,
        max_delta_per_tick=0.0,
        description="Reserved for future bounded allocation; not active control.",
    ),
)

_CONTROL_INDEX = {c.control_id: i for i, c in enumerate(_SELF_REGULATION_CONTROLS)}
_CONTROL_BY_ID = {c.control_id: c for c in _SELF_REGULATION_CONTROLS}


def self_regulation_control_registry() -> tuple[SelfRegulationControlMetadata, ...]:
    return _SELF_REGULATION_CONTROLS


def self_regulation_control_registry_dict() -> dict[str, object]:
    return {
        "schema_version": SELF_REGULATION_CONTROL_REGISTRY_SCHEMA_VERSION,
        "controls": [c.to_dict() for c in _SELF_REGULATION_CONTROLS],
        "forbidden_targets": list(FORBIDDEN_SELF_REGULATION_TARGETS),
    }


def forbidden_self_regulation_targets() -> tuple[str, ...]:
    return FORBIDDEN_SELF_REGULATION_TARGETS


def get_self_regulation_control(control_id: str) -> SelfRegulationControlMetadata:
    return _CONTROL_BY_ID[str(control_id)]


def implemented_self_regulation_control_ids() -> tuple[str, ...]:
    return tuple(
        c.control_id
        for c in _SELF_REGULATION_CONTROLS
        if c.active_now
        and not c.future_reserved
        and c.safe_for_agent_control
        and not c.can_affect_genome
        and not c.can_affect_host_or_simulator
        and not c.can_affect_source_code
    )


def implemented_self_regulation_control_mask() -> int:
    mask = 0
    for cid in implemented_self_regulation_control_ids():
        mask |= 1 << int(_CONTROL_INDEX[cid])
    return int(mask)


def minimal_self_regulation_control_mask() -> int:
    return 0


def sanitize_self_regulation_control_mask(mask: Any) -> int:
    try:
        out = int(mask)
    except Exception:
        out = implemented_self_regulation_control_mask()
    return int(out & implemented_self_regulation_control_mask())


def self_regulation_control_mask_contains(mask: int, control_id: str) -> bool:
    idx = int(_CONTROL_INDEX[str(control_id)])
    return bool(int(mask) & (1 << idx))


def self_regulation_control_ids_from_mask(mask: int) -> tuple[str, ...]:
    mask = sanitize_self_regulation_control_mask(mask)
    return tuple(cid for cid in implemented_self_regulation_control_ids() if self_regulation_control_mask_contains(mask, cid))


def recombine_self_regulation_control_mask(a: Any, b: Any, rng: Any) -> int:
    ma = sanitize_self_regulation_control_mask(a)
    mb = sanitize_self_regulation_control_mask(b)
    out = 0
    for cid in implemented_self_regulation_control_ids():
        bit = 1 << int(_CONTROL_INDEX[cid])
        if bool(rng.random() < 0.5):
            out |= ma & bit
        else:
            out |= mb & bit
    return sanitize_self_regulation_control_mask(out)


def mutate_self_regulation_control_mask(mask: Any, rng: Any, rate: float) -> int:
    out = sanitize_self_regulation_control_mask(mask)
    p = max(0.0, min(1.0, float(rate)))
    if rng.random() >= p:
        return out
    ids = implemented_self_regulation_control_ids()
    if not ids:
        return out
    cid = ids[int(rng.randrange(len(ids)))]
    out ^= 1 << int(_CONTROL_INDEX[cid])
    return sanitize_self_regulation_control_mask(out)


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
    return tuple(0.0 for _ in implemented_self_regulation_control_ids())


def self_regulation_ablation_flags(sim_cfg: Any = None) -> dict[str, bool]:
    ablate = _cfg_bool(sim_cfg, "ablate_self_regulation", False)
    force_no = _cfg_bool(sim_cfg, "force_no_self_regulation", False)
    force_all = _cfg_bool(sim_cfg, "force_all_self_regulation_controls", False)
    force_default = _cfg_bool(sim_cfg, "force_default_self_regulation_controls", False)
    mask_neutral = _cfg_bool(sim_cfg, "mask_self_regulation_outputs_to_neutral", False)
    freeze = _cfg_bool(sim_cfg, "freeze_self_regulation_state", False)
    reset_each_tick = _cfg_bool(sim_cfg, "reset_self_regulation_each_tick", False)
    disable_decay = _cfg_bool(sim_cfg, "disable_self_regulation_decay", False)
    return {
        "ablate": bool(ablate),
        "force_no": bool(force_no),
        "force_all": bool(force_all),
        "force_default": bool(force_default),
        "mask_to_neutral": bool(mask_neutral),
        "freeze": bool(freeze),
        "reset_each_tick": bool(reset_each_tick),
        "disable_decay": bool(disable_decay),
        "active": bool(ablate or force_no or force_all or force_default or mask_neutral or freeze or reset_each_tick or disable_decay),
    }


@dataclass(frozen=True)
class SelfRegulationExpression:
    schema_version: str
    self_regulation_enabled: bool
    self_regulation_surface_enabled: bool
    self_regulation_policy_output_enabled: bool
    active_self_regulation_control_count: int
    potential_self_regulation_control_count: int
    active_self_regulation_control_mask: int
    active_self_regulation_control_ids: tuple[str, ...]
    self_regulation_capacity_fraction: float
    self_regulation_cost_hypothetical: float
    self_regulation_cost_charged: float
    self_regulation_ablation_active: bool
    force_no_self_regulation_active: bool
    force_all_self_regulation_active: bool
    force_default_self_regulation_active: bool
    mask_self_regulation_outputs_to_neutral_active: bool
    freeze_self_regulation_state_active: bool
    reset_self_regulation_each_tick_active: bool
    disable_self_regulation_decay_active: bool
    self_regulation_runtime_state_available: bool
    self_regulation_inheritable: bool
    self_modification_active: bool
    genome_editing_active: bool

    def to_dict(self) -> dict[str, object]:
        out = asdict(self)
        out["active_self_regulation_control_ids"] = list(self.active_self_regulation_control_ids)
        return out


def express_self_regulation(genome: Any, *, sim_cfg: Any = None) -> SelfRegulationExpression:
    potential = len(implemented_self_regulation_control_ids())
    feature_enabled = _cfg_bool(sim_cfg, "enable_self_regulation", False) or _cfg_bool(sim_cfg, "enable_self_regulation_surface", False)
    surface_enabled = _cfg_bool(sim_cfg, "enable_self_regulation_surface", False) or feature_enabled
    evolvable = _cfg_bool(sim_cfg, "enable_evolvable_self_regulation_controls", False)
    flags = self_regulation_ablation_flags(sim_cfg)
    default_mask = implemented_self_regulation_control_mask() if _cfg_bool(sim_cfg, "default_self_regulation_controls_enabled", False) else 0
    raw_mask = sanitize_self_regulation_control_mask(_gget(genome, "self_regulation_control_mask", default_mask))
    if flags["force_all"]:
        active_mask = implemented_self_regulation_control_mask()
    elif flags["force_no"] or flags["ablate"]:
        active_mask = 0
    elif flags["force_default"] or not evolvable:
        active_mask = default_mask
    else:
        active_mask = raw_mask
    if not surface_enabled:
        active_mask = 0
    active_mask = sanitize_self_regulation_control_mask(active_mask)
    active_ids = self_regulation_control_ids_from_mask(active_mask)
    active_weight = sum(float(_CONTROL_BY_ID[cid].cost_weight) for cid in active_ids)
    base = max(0.0, _cfg_float(sim_cfg, "self_regulation_control_cost_base", 0.0))
    per_control = max(0.0, _cfg_float(sim_cfg, "self_regulation_control_cost", 0.001))
    hypothetical = float(base + per_control * active_weight) if surface_enabled else 0.0
    charge = _cfg_bool(sim_cfg, "enable_experimental_self_regulation_cost_charging", False)
    return SelfRegulationExpression(
        schema_version=SELF_REGULATION_EXPRESSION_SCHEMA_VERSION,
        self_regulation_enabled=bool(surface_enabled and len(active_ids) > 0 and not flags["mask_to_neutral"]),
        self_regulation_surface_enabled=bool(surface_enabled),
        self_regulation_policy_output_enabled=False,
        active_self_regulation_control_count=int(len(active_ids)),
        potential_self_regulation_control_count=int(potential),
        active_self_regulation_control_mask=int(active_mask),
        active_self_regulation_control_ids=tuple(active_ids),
        self_regulation_capacity_fraction=float(len(active_ids)) / float(potential) if potential > 0 else 0.0,
        self_regulation_cost_hypothetical=float(hypothetical),
        self_regulation_cost_charged=float(hypothetical if charge else 0.0),
        self_regulation_ablation_active=bool(flags["active"]),
        force_no_self_regulation_active=bool(flags["force_no"] or flags["ablate"]),
        force_all_self_regulation_active=bool(flags["force_all"]),
        force_default_self_regulation_active=bool(flags["force_default"]),
        mask_self_regulation_outputs_to_neutral_active=bool(flags["mask_to_neutral"]),
        freeze_self_regulation_state_active=bool(flags["freeze"]),
        reset_self_regulation_each_tick_active=bool(flags["reset_each_tick"]),
        disable_self_regulation_decay_active=bool(flags["disable_decay"]),
        self_regulation_runtime_state_available=bool(surface_enabled),
        self_regulation_inheritable=False,
        self_modification_active=False,
        genome_editing_active=False,
    )


@dataclass
class SelfRegulationRuntimeState:
    schema_version: str = SELF_REGULATION_RUNTIME_SCHEMA_VERSION
    active_control_mask: int = 0
    offsets: dict[str, float] = field(default_factory=dict)
    last_control_vector: tuple[float, ...] = field(default_factory=_neutral_vector)
    applied_control_deltas: dict[str, float] = field(default_factory=dict)
    clipped_control_deltas: dict[str, float] = field(default_factory=dict)
    control_cost: float = 0.0
    control_application_count: int = 0
    control_saturation_count: int = 0
    control_reset_count: int = 0
    control_decay_count: int = 0
    baseline_values: dict[str, float] = field(default_factory=dict)
    effective_regulated_values: dict[str, float] = field(default_factory=dict)
    ablated: bool = False
    forced: bool = False
    policy_output_active: bool = False

    def to_dict(self) -> dict[str, object]:
        out = asdict(self)
        out["last_control_vector"] = list(self.last_control_vector)
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "SelfRegulationRuntimeState":
        if not isinstance(data, Mapping):
            return cls()
        state = cls()
        state.active_control_mask = sanitize_self_regulation_control_mask(data.get("active_control_mask", 0))
        state.offsets = {str(k): _finite(v, 0.0) for k, v in dict(data.get("offsets", {}) or {}).items()}
        raw_vec = list(data.get("last_control_vector", []) or [])
        n = len(implemented_self_regulation_control_ids())
        state.last_control_vector = tuple(_clip(raw_vec[i], -1.0, 1.0, 0.0) if i < len(raw_vec) else 0.0 for i in range(n))
        state.applied_control_deltas = {str(k): _finite(v, 0.0) for k, v in dict(data.get("applied_control_deltas", {}) or {}).items()}
        state.clipped_control_deltas = {str(k): _finite(v, 0.0) for k, v in dict(data.get("clipped_control_deltas", {}) or {}).items()}
        state.control_cost = max(0.0, _finite(data.get("control_cost", 0.0), 0.0))
        for name in ("control_application_count", "control_saturation_count", "control_reset_count", "control_decay_count"):
            try:
                setattr(state, name, int(data.get(name, 0) or 0))
            except Exception:
                setattr(state, name, 0)
        state.baseline_values = {str(k): _finite(v, 0.0) for k, v in dict(data.get("baseline_values", {}) or {}).items()}
        state.effective_regulated_values = {str(k): _finite(v, 0.0) for k, v in dict(data.get("effective_regulated_values", {}) or {}).items()}
        state.ablated = bool(data.get("ablated", False))
        state.forced = bool(data.get("forced", False))
        state.policy_output_active = bool(data.get("policy_output_active", False))
        return state


def _baseline_values(genome: Any) -> dict[str, float]:
    return {
        "exploration_pressure_adjustment": _finite(_gget(genome, "exploration_bias", 0.0), 0.0),
        "action_temperature_adjustment": _finite(_gget(genome, "action_temperature", 1.0), 1.0),
        "drive_gain_adjustment": _finite(_gget(genome, "drive_logit_gain", 1.0), 1.0),
        "reproduction_threshold_adjustment": _finite(_gget(genome, "reproduction_threshold", 30.0), 30.0),
        "prediction_lr_adjustment": _finite(_gget(genome, "pred_lr", 0.08), 0.08),
    }


def _effective_bounds(control_id: str) -> tuple[float, float]:
    if control_id == "exploration_pressure_adjustment":
        return -1.0, 1.0
    if control_id == "action_temperature_adjustment":
        return 0.05, 3.0
    if control_id == "drive_gain_adjustment":
        return 0.0, 3.0
    if control_id == "reproduction_threshold_adjustment":
        return 0.0, 120.0
    if control_id == "prediction_lr_adjustment":
        return 0.0, 1.0
    return -1.0, 1.0


def _vector(control_vector: Any) -> tuple[float, ...]:
    ids = implemented_self_regulation_control_ids()
    if control_vector is None:
        raw: list[Any] = []
    else:
        try:
            raw = list(control_vector)
        except Exception:
            raw = []
    return tuple(_clip(raw[i], -1.0, 1.0, 0.0) if i < len(raw) else 0.0 for i in range(len(ids)))


def apply_self_regulation_controls(
    state: SelfRegulationRuntimeState | Mapping[str, Any] | None,
    control_vector: Any,
    *,
    genome: Any = None,
    sim_cfg: Any = None,
) -> SelfRegulationRuntimeState:
    """Apply a fixed-shape scripted control vector to runtime offsets.

    The function returns the updated state. It never mutates the genome. The
    caller may attach the returned state to an agent if lifetime persistence is
    desired.
    """
    current = state if isinstance(state, SelfRegulationRuntimeState) else SelfRegulationRuntimeState.from_dict(state)
    expression = express_self_regulation(genome, sim_cfg=sim_cfg)
    flags = self_regulation_ablation_flags(sim_cfg)
    ids = implemented_self_regulation_control_ids()
    values = _neutral_vector() if flags["mask_to_neutral"] else _vector(control_vector)

    if flags["reset_each_tick"]:
        current.offsets = {}
        current.control_reset_count += 1

    current.active_control_mask = int(expression.active_self_regulation_control_mask)
    current.last_control_vector = tuple(values)
    current.applied_control_deltas = {}
    current.clipped_control_deltas = {}
    current.baseline_values = _baseline_values(genome)
    current.ablated = bool(expression.self_regulation_ablation_active)
    current.forced = bool(expression.force_all_self_regulation_active or expression.force_no_self_regulation_active or expression.force_default_self_regulation_active)
    current.policy_output_active = bool(expression.self_regulation_policy_output_enabled)

    if not expression.self_regulation_surface_enabled or flags["force_no"] or flags["ablate"] or flags["freeze"]:
        current.effective_regulated_values = dict(current.baseline_values)
        return current

    decay = 0.0 if flags["disable_decay"] else max(0.0, min(1.0, _cfg_float(sim_cfg, "self_regulation_offset_decay", 0.05)))
    if decay > 0.0 and current.offsets:
        for cid, old in list(current.offsets.items()):
            new = _finite(old, 0.0) * (1.0 - decay)
            current.offsets[cid] = float(new)
        current.control_decay_count += 1

    active = set(expression.active_self_regulation_control_ids)
    application_cost = 0.0
    for idx, cid in enumerate(ids):
        meta = _CONTROL_BY_ID[cid]
        raw = float(values[idx])
        if cid not in active:
            raw = float(meta.neutral_control_value)
        bounded_raw = _clip(raw, -1.0, 1.0, 0.0)
        proposed_delta = bounded_raw * float(meta.max_delta_per_tick)
        old_offset = _finite(current.offsets.get(cid, 0.0), 0.0)
        unclipped = old_offset + proposed_delta
        new_offset = _clip(unclipped, float(meta.min_value), float(meta.max_value), 0.0)
        clipped_delta = new_offset - old_offset
        current.offsets[cid] = float(new_offset)
        current.applied_control_deltas[cid] = float(proposed_delta)
        current.clipped_control_deltas[cid] = float(clipped_delta)
        if abs(clipped_delta - proposed_delta) > 1e-12:
            current.control_saturation_count += 1
        application_cost += abs(clipped_delta) * float(meta.cost_weight)

    current.control_application_count += 1
    current.control_cost = float(max(0.0, _cfg_float(sim_cfg, "self_regulation_update_cost_scale", 0.001)) * application_cost)
    effective: dict[str, float] = {}
    for cid, baseline in current.baseline_values.items():
        lo, hi = _effective_bounds(cid)
        effective[cid] = _clip(float(baseline) + float(current.offsets.get(cid, 0.0)), lo, hi, float(baseline))
    current.effective_regulated_values = effective
    return current


def self_regulation_runtime_summary(state: Any) -> dict[str, float]:
    current = state if isinstance(state, SelfRegulationRuntimeState) else SelfRegulationRuntimeState.from_dict(state)
    deltas = [abs(float(v)) for v in current.clipped_control_deltas.values()]
    offsets = [abs(float(v)) for v in current.offsets.values()]
    active = max(1, len(implemented_self_regulation_control_ids()))
    neutral = sum(1 for v in current.last_control_vector if abs(float(v)) <= 1e-12)
    return {
        "delta_abs_mean": float(sum(deltas) / len(deltas)) if deltas else 0.0,
        "delta_abs_p90": float(np.percentile(np.asarray(deltas, dtype=np.float64), 90)) if deltas else 0.0,
        "offset_abs_mean": float(sum(offsets) / len(offsets)) if offsets else 0.0,
        "offset_abs_p90": float(np.percentile(np.asarray(offsets, dtype=np.float64), 90)) if offsets else 0.0,
        "clipped_fraction": float(current.control_saturation_count) / float(max(1, current.control_application_count * active)),
        "saturation_fraction": float(current.control_saturation_count) / float(max(1, current.control_application_count * active)),
        "application_count": float(current.control_application_count),
        "neutral_fraction": float(neutral) / float(active),
        "reset_count": float(current.control_reset_count),
        "decay_count": float(current.control_decay_count),
        "cost": float(current.control_cost),
        "effective_exploration_pressure": float(current.effective_regulated_values.get("exploration_pressure_adjustment", current.baseline_values.get("exploration_pressure_adjustment", 0.0))),
        "effective_action_temperature": float(current.effective_regulated_values.get("action_temperature_adjustment", current.baseline_values.get("action_temperature_adjustment", 1.0))),
        "effective_drive_gain": float(current.effective_regulated_values.get("drive_gain_adjustment", current.baseline_values.get("drive_gain_adjustment", 1.0))),
        "effective_reproduction_threshold": float(current.effective_regulated_values.get("reproduction_threshold_adjustment", current.baseline_values.get("reproduction_threshold_adjustment", 30.0))),
        "effective_prediction_lr": float(current.effective_regulated_values.get("prediction_lr_adjustment", current.baseline_values.get("prediction_lr_adjustment", 0.08))),
    }
