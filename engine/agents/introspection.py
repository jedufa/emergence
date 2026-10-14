from __future__ import annotations

"""Deterministic internal-state channel helpers for Theme 4 Stage 5.

This module defines a bounded introspection surface. It exposes simulated
internal state as normalized scalar channels, but it does not grant control over
those state variables and it does not expose host or simulator internals.
"""

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np


INTROSPECTION_CHANNEL_REGISTRY_SCHEMA_VERSION = "introspection_channel_registry_v1"
INTROSPECTION_EXPRESSION_SCHEMA_VERSION = "introspection_expression_v1"


@dataclass(frozen=True)
class IntrospectionChannelMetadata:
    channel_id: str
    display_name: str
    category: str
    active_now: bool
    future_reserved: bool
    default_enabled: bool
    maskable: bool
    cost_weight: float
    source: str
    normalization: str
    neutral_masked_value: float
    metric_name_prefix: str
    safe_for_agent_observation: bool
    leaks_host_or_simulator_state: bool
    diagnostic_only: bool
    future_control_candidate: bool
    description: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _channel(
    channel_id: str,
    display_name: str,
    *,
    category: str,
    active_now: bool,
    future_reserved: bool = False,
    default_enabled: bool = True,
    maskable: bool = True,
    cost_weight: float = 1.0,
    source: str = "",
    normalization: str = "unit interval",
    neutral: float = 0.0,
    safe: bool = True,
    leaks: bool = False,
    diagnostic_only: bool = False,
    future_control_candidate: bool = True,
    description: str = "",
) -> IntrospectionChannelMetadata:
    return IntrospectionChannelMetadata(
        channel_id=str(channel_id),
        display_name=str(display_name),
        category=str(category),
        active_now=bool(active_now),
        future_reserved=bool(future_reserved),
        default_enabled=bool(default_enabled),
        maskable=bool(maskable),
        cost_weight=float(cost_weight),
        source=str(source),
        normalization=str(normalization),
        neutral_masked_value=float(neutral),
        metric_name_prefix=f"introspection_channel_{channel_id}",
        safe_for_agent_observation=bool(safe),
        leaks_host_or_simulator_state=bool(leaks),
        diagnostic_only=bool(diagnostic_only),
        future_control_candidate=bool(future_control_candidate),
        description=str(description),
    )


_INTROSPECTION_CHANNELS: tuple[IntrospectionChannelMetadata, ...] = (
    _channel("energy", "Energy", category="physiological", active_now=True, source="agent.energy / env_cfg.max_energy"),
    _channel("health", "Health", category="physiological", active_now=True, source="agent.health / env_cfg.start_health"),
    _channel("hunger", "Hunger Drive", category="drive", active_now=True, source="agent.drives[0]"),
    _channel("pain", "Pain Drive", category="drive", active_now=True, source="agent.drives[1]"),
    _channel("fatigue", "Fatigue Drive", category="drive", active_now=True, source="agent.drives[2]"),
    _channel("fear", "Fear Drive", category="drive", active_now=True, source="agent.drives[3]"),
    _channel(
        "drive_conflict_proxy",
        "Drive Conflict Proxy",
        category="drive",
        active_now=True,
        source="max(drive) - mean(drive)",
        description="Simple deterministic pressure-spread proxy; diagnostic, not a reward.",
    ),
    _channel(
        "age_normalized",
        "Age Normalized",
        category="developmental",
        active_now=True,
        source="stats.survival_steps / developmental_maturity_age",
        description="Age normalized to the configured maturity horizon.",
    ),
    _channel(
        "maturity_progress",
        "Maturity Progress",
        category="developmental",
        active_now=True,
        source="cognitive phenotype maturity progress",
    ),
    _channel(
        "developmental_expression_factor",
        "Developmental Expression Factor",
        category="developmental",
        active_now=True,
        source="cognitive phenotype developmental expression factor",
    ),
    _channel(
        "memory_capacity_fraction",
        "Memory Capacity Fraction",
        category="memory",
        active_now=True,
        source="cognitive phenotype memory_capacity_fraction",
    ),
    _channel(
        "memory_saturation_proxy",
        "Memory Saturation Proxy",
        category="memory",
        active_now=True,
        source="AgentStats memory telemetry when available",
        diagnostic_only=True,
    ),
    _channel(
        "prediction_error",
        "Prediction Error",
        category="prediction",
        active_now=True,
        source="stats.pred_err_sum / stats.pred_err_count",
        diagnostic_only=True,
        description="Prediction diagnostic; lower error is not intelligence.",
    ),
    _channel(
        "prediction_update_activity",
        "Prediction Update Activity",
        category="prediction",
        active_now=True,
        source="stats.prediction_update_count normalized by age",
        diagnostic_only=True,
    ),
    _channel(
        "cognitive_cost_fraction",
        "Cognitive Cost Fraction",
        category="cost_budget",
        active_now=True,
        source="cognitive budget ledger cost fraction",
        diagnostic_only=True,
    ),
    _channel(
        "uncertainty",
        "Uncertainty",
        category="future_reserved",
        active_now=False,
        future_reserved=True,
        default_enabled=False,
        source="not implemented",
        diagnostic_only=True,
        description="Reserved for future bounded uncertainty telemetry.",
    ),
    _channel(
        "stress",
        "Stress",
        category="future_reserved",
        active_now=False,
        future_reserved=True,
        default_enabled=False,
        source="not implemented",
        description="Reserved; no current stress state exists as a safe channel.",
    ),
    _channel(
        "self_coherence",
        "Self Coherence",
        category="future_reserved",
        active_now=False,
        future_reserved=True,
        default_enabled=False,
        source="not implemented",
        diagnostic_only=True,
        description="Reserved for future internal stability/coherence surfaces.",
    ),
)

_CHANNEL_INDEX = {c.channel_id: i for i, c in enumerate(_INTROSPECTION_CHANNELS)}
_CHANNEL_BY_ID = {c.channel_id: c for c in _INTROSPECTION_CHANNELS}


def introspection_channel_registry() -> tuple[IntrospectionChannelMetadata, ...]:
    return _INTROSPECTION_CHANNELS


def introspection_channel_registry_dict() -> dict[str, object]:
    return {
        "schema_version": INTROSPECTION_CHANNEL_REGISTRY_SCHEMA_VERSION,
        "channels": [c.to_dict() for c in _INTROSPECTION_CHANNELS],
    }


def get_introspection_channel(channel_id: str) -> IntrospectionChannelMetadata:
    return _CHANNEL_BY_ID[str(channel_id)]


def implemented_introspection_channel_ids() -> tuple[str, ...]:
    return tuple(
        c.channel_id
        for c in _INTROSPECTION_CHANNELS
        if c.active_now and not c.future_reserved and c.safe_for_agent_observation and not c.leaks_host_or_simulator_state
    )


def implemented_introspection_channel_mask() -> int:
    mask = 0
    for cid in implemented_introspection_channel_ids():
        mask |= 1 << int(_CHANNEL_INDEX[cid])
    return int(mask)


def minimal_introspection_channel_mask() -> int:
    return 0


def sanitize_introspection_channel_mask(mask: Any) -> int:
    try:
        out = int(mask)
    except Exception:
        out = implemented_introspection_channel_mask()
    return int(out & implemented_introspection_channel_mask())


def introspection_channel_mask_contains(mask: int, channel_id: str) -> bool:
    idx = int(_CHANNEL_INDEX[str(channel_id)])
    return bool(int(mask) & (1 << idx))


def introspection_channel_ids_from_mask(mask: int) -> tuple[str, ...]:
    mask = sanitize_introspection_channel_mask(mask)
    return tuple(cid for cid in implemented_introspection_channel_ids() if introspection_channel_mask_contains(mask, cid))


def recombine_introspection_channel_mask(a: Any, b: Any, rng: Any) -> int:
    ma = sanitize_introspection_channel_mask(a)
    mb = sanitize_introspection_channel_mask(b)
    out = 0
    for cid in implemented_introspection_channel_ids():
        bit = 1 << int(_CHANNEL_INDEX[cid])
        if bool(rng.random() < 0.5):
            out |= ma & bit
        else:
            out |= mb & bit
    return sanitize_introspection_channel_mask(out)


def mutate_introspection_channel_mask(mask: Any, rng: Any, rate: float) -> int:
    out = sanitize_introspection_channel_mask(mask)
    p = max(0.0, min(1.0, float(rate)))
    if rng.random() >= p:
        return out
    ids = implemented_introspection_channel_ids()
    if not ids:
        return out
    cid = ids[int(rng.randrange(len(ids)))]
    out ^= 1 << int(_CHANNEL_INDEX[cid])
    return sanitize_introspection_channel_mask(out)


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


def introspection_ablation_flags(sim_cfg: Any = None) -> dict[str, bool]:
    ablate = _cfg_bool(sim_cfg, "ablate_introspection", False)
    force_no = _cfg_bool(sim_cfg, "force_no_introspection_channels", False)
    force_min = _cfg_bool(sim_cfg, "force_minimal_introspection_channels", False)
    force_all = _cfg_bool(sim_cfg, "force_all_introspection_channels", False)
    force_default = _cfg_bool(sim_cfg, "force_default_introspection_channels", False)
    mask_neutral = _cfg_bool(sim_cfg, "mask_introspection_values_to_neutral", False)
    return {
        "ablate": bool(ablate),
        "force_no": bool(force_no),
        "force_minimal": bool(force_min),
        "force_all": bool(force_all),
        "force_default": bool(force_default),
        "mask_to_neutral": bool(mask_neutral),
        "active": bool(ablate or force_no or force_min or force_all or force_default or mask_neutral),
    }


@dataclass(frozen=True)
class IntrospectionExpression:
    schema_version: str
    introspection_enabled: bool
    introspection_channels_available: bool
    active_introspection_channel_count: int
    potential_introspection_channel_count: int
    active_introspection_channel_mask: int
    active_introspection_channel_ids: tuple[str, ...]
    introspection_capacity_fraction: float
    introspection_cost_hypothetical: float
    introspection_cost_charged: float
    introspection_ablation_active: bool
    force_all_introspection_channels_active: bool
    force_minimal_introspection_channels_active: bool
    force_no_introspection_channels_active: bool
    force_default_introspection_channels_active: bool
    mask_introspection_values_to_neutral_active: bool
    introspection_observation_shape: int
    introspection_policy_input_enabled: bool
    introspection_future_control_surface_ready: bool

    def to_dict(self) -> dict[str, object]:
        out = asdict(self)
        out["active_introspection_channel_ids"] = list(self.active_introspection_channel_ids)
        return out


def express_introspection(genome: Any, *, sim_cfg: Any = None) -> IntrospectionExpression:
    potential = len(implemented_introspection_channel_ids())
    feature_enabled = _cfg_bool(sim_cfg, "enable_introspection_channels", False)
    evolvable = _cfg_bool(sim_cfg, "enable_evolvable_introspection_channels", False)
    flags = introspection_ablation_flags(sim_cfg)
    default_mask = implemented_introspection_channel_mask() if _cfg_bool(sim_cfg, "default_introspection_channels_enabled", False) else 0
    raw_mask = sanitize_introspection_channel_mask(_gget(genome, "introspection_channel_mask", default_mask))
    if flags["force_all"]:
        active_mask = implemented_introspection_channel_mask()
    elif flags["force_no"] or flags["ablate"]:
        active_mask = 0
    elif flags["force_minimal"]:
        active_mask = minimal_introspection_channel_mask()
    elif flags["force_default"] or not evolvable:
        active_mask = default_mask
    else:
        active_mask = raw_mask
    if not feature_enabled:
        active_mask = 0
    active_mask = sanitize_introspection_channel_mask(active_mask)
    active_ids = introspection_channel_ids_from_mask(active_mask)
    active_weight = sum(float(_CHANNEL_BY_ID[cid].cost_weight) for cid in active_ids)
    per_channel = max(0.0, _cfg_float(sim_cfg, "introspection_channel_cost", 0.001))
    observation_base = max(0.0, _cfg_float(sim_cfg, "introspection_observation_cost_base", 0.0))
    hypothetical = float(observation_base + per_channel * active_weight) if feature_enabled else 0.0
    charge = _cfg_bool(sim_cfg, "enable_experimental_introspection_cost_charging", False)
    return IntrospectionExpression(
        schema_version=INTROSPECTION_EXPRESSION_SCHEMA_VERSION,
        introspection_enabled=bool(feature_enabled and len(active_ids) > 0 and not flags["mask_to_neutral"]),
        introspection_channels_available=bool(potential > 0),
        active_introspection_channel_count=int(len(active_ids)),
        potential_introspection_channel_count=int(potential),
        active_introspection_channel_mask=int(active_mask),
        active_introspection_channel_ids=tuple(active_ids),
        introspection_capacity_fraction=float(len(active_ids)) / float(potential) if potential > 0 else 0.0,
        introspection_cost_hypothetical=float(hypothetical),
        introspection_cost_charged=float(hypothetical if charge else 0.0),
        introspection_ablation_active=bool(flags["active"]),
        force_all_introspection_channels_active=bool(flags["force_all"]),
        force_minimal_introspection_channels_active=bool(flags["force_minimal"]),
        force_no_introspection_channels_active=bool(flags["force_no"] or flags["ablate"]),
        force_default_introspection_channels_active=bool(flags["force_default"]),
        mask_introspection_values_to_neutral_active=bool(flags["mask_to_neutral"]),
        introspection_observation_shape=int(potential),
        introspection_policy_input_enabled=False,
        introspection_future_control_surface_ready=True,
    )


def _unit(value: Any, default: float = 0.0) -> float:
    try:
        v = float(value)
    except Exception:
        v = float(default)
    if not np.isfinite(v):
        v = float(default)
    return float(max(0.0, min(1.0, v)))


def _drive(agent: Any, idx: int) -> float:
    try:
        d = np.asarray(getattr(agent, "drives", np.zeros((4,), dtype=np.float32)), dtype=np.float32)
        return _unit(float(d[int(idx)]), 0.0)
    except Exception:
        return 0.0


def _prediction_error(agent: Any) -> float:
    st = getattr(agent, "stats", None)
    try:
        n = int(getattr(st, "pred_err_count", 0) or 0)
        if n <= 0:
            return 0.0
        return _unit(float(getattr(st, "pred_err_sum", 0.0) or 0.0) / float(n), 0.0)
    except Exception:
        return 0.0


def introspection_channel_value(
    channel_id: str,
    *,
    agent: Any = None,
    env_cfg: Any = None,
    sim_cfg: Any = None,
    phenotype: Any = None,
    ledger: Any = None,
    memory_runtime_telemetry: Mapping[str, Any] | None = None,
) -> float:
    st = getattr(agent, "stats", None) if agent is not None else None
    cid = str(channel_id)
    if cid == "energy":
        denom = float(getattr(env_cfg, "max_energy", getattr(agent, "max_energy_norm", 160.0)) or 160.0)
        return _unit(float(getattr(agent, "energy", 0.0)) / max(1e-9, denom), 0.0)
    if cid == "health":
        denom = float(getattr(env_cfg, "start_health", getattr(agent, "start_health_norm", 12.0)) or 12.0)
        return _unit(float(getattr(agent, "health", 0.0)) / max(1e-9, denom), 0.0)
    if cid == "hunger":
        return _drive(agent, 0)
    if cid == "pain":
        return _drive(agent, 1)
    if cid == "fatigue":
        return _drive(agent, 2)
    if cid == "fear":
        return _drive(agent, 3)
    if cid == "drive_conflict_proxy":
        try:
            d = np.asarray(getattr(agent, "drives", np.zeros((4,), dtype=np.float32)), dtype=np.float32)
            return _unit(float(np.max(d) - np.mean(d)), 0.0)
        except Exception:
            return 0.0
    if cid == "age_normalized":
        age = float(getattr(st, "survival_steps", 0) or 0)
        maturity = float(getattr(sim_cfg, "developmental_maturity_age", 20) or 20)
        return _unit(age / max(1.0, maturity), 0.0)
    if cid == "maturity_progress":
        return _unit(getattr(phenotype, "cognitive_growth_progress", 0.0), 0.0)
    if cid == "developmental_expression_factor":
        return _unit(getattr(phenotype, "developmental_expression_factor", 1.0), 1.0)
    if cid == "memory_capacity_fraction":
        return _unit(getattr(phenotype, "memory_capacity_fraction", 0.0), 0.0)
    if cid == "memory_saturation_proxy":
        if memory_runtime_telemetry is not None:
            return _unit(memory_runtime_telemetry.get("memory_saturation_proxy_mean", 0.0), 0.0)
        return _unit(getattr(st, "memory_saturation_proxy", 0.0), 0.0)
    if cid == "prediction_error":
        return _prediction_error(agent)
    if cid == "prediction_update_activity":
        age = float(getattr(st, "survival_steps", 0) or 0)
        updates = float(getattr(st, "prediction_update_count", 0) or 0)
        return _unit(updates / max(1.0, age), 0.0)
    if cid == "cognitive_cost_fraction":
        return _unit(getattr(ledger, "cognitive_cost_fraction_of_energy_budget", 0.0), 0.0)
    return 0.0


def extract_introspection_observation(
    agent: Any,
    *,
    env_cfg: Any = None,
    sim_cfg: Any = None,
    phenotype: Any = None,
    ledger: Any = None,
    memory_runtime_telemetry: Mapping[str, Any] | None = None,
) -> np.ndarray:
    g = getattr(agent, "genome", None) if agent is not None else None
    expression = express_introspection(g, sim_cfg=sim_cfg)
    mask_to_neutral = bool(expression.mask_introspection_values_to_neutral_active)
    active = set(expression.active_introspection_channel_ids)
    values: list[float] = []
    for channel in _INTROSPECTION_CHANNELS:
        if not channel.active_now or channel.future_reserved:
            continue
        if channel.channel_id not in active or mask_to_neutral:
            values.append(float(channel.neutral_masked_value))
            continue
        values.append(
            introspection_channel_value(
                channel.channel_id,
                agent=agent,
                env_cfg=env_cfg,
                sim_cfg=sim_cfg,
                phenotype=phenotype,
                ledger=ledger,
                memory_runtime_telemetry=memory_runtime_telemetry,
            )
        )
    return np.asarray(values, dtype=np.float32)
