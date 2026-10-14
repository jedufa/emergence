from __future__ import annotations

"""Deterministic perception channel and radius helpers for Theme 4.

The observation tensor shape remains fixed. These helpers only describe and
apply masks over the existing layout.
"""

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping

import numpy as np


PERCEPTION_CHANNEL_REGISTRY_SCHEMA_VERSION = "perception_channel_registry_v1"
PERCEPTION_EXPRESSION_SCHEMA_VERSION = "perception_expression_v1"
PERCEPTION_ROUTING_CONTRACT_VERSION = "perception_routing_contract_v1"

SENSOR_RADIUS_LADDER: tuple[int, ...] = (1, 2, 3)
DEFAULT_SENSOR_RADIUS = 2


@dataclass(frozen=True)
class PerceptionChannelMetadata:
    channel_id: str
    display_name: str
    category: str
    active_now: bool
    future_reserved: bool
    default_enabled: bool
    maskable: bool
    cost_weight: float
    description: str
    source: str
    neutral_masked_value: float
    metric_name_prefix: str
    local_channel_index: int | None = None
    self_state_index: int | None = None
    spatial: bool = True
    patch_channel: bool = False
    value_type: str = "scalar"
    value_shape: str = "scalar"
    value_min: float = 0.0
    value_max: float = 1.0
    direct_policy_exposure: bool = True
    routing_eligible: bool = False
    routing_ineligible_reason: str = "not_admitted"
    feature_gate: str = "always"

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
    description: str = "",
    source: str = "",
    neutral: float = 0.0,
    local_channel_index: int | None = None,
    self_state_index: int | None = None,
    spatial: bool = True,
    patch_channel: bool = False,
    value_min: float = 0.0,
    value_max: float = 1.0,
    routing_eligible: bool = False,
    routing_ineligible_reason: str = "spatial_channel_requires_new_reduction",
    feature_gate: str = "always",
) -> PerceptionChannelMetadata:
    return PerceptionChannelMetadata(
        channel_id=str(channel_id),
        display_name=str(display_name),
        category=str(category),
        active_now=bool(active_now),
        future_reserved=bool(future_reserved),
        default_enabled=bool(default_enabled),
        maskable=bool(maskable),
        cost_weight=float(cost_weight),
        description=str(description),
        source=str(source),
        neutral_masked_value=float(neutral),
        metric_name_prefix=f"sensor_channel_{channel_id}",
        local_channel_index=local_channel_index,
        self_state_index=self_state_index,
        spatial=bool(spatial),
        patch_channel=bool(patch_channel),
        value_type="vector" if bool(spatial) else "scalar",
        value_shape="view_cells" if bool(spatial) else "scalar",
        value_min=float(value_min),
        value_max=float(value_max),
        direct_policy_exposure=bool(active_now and not future_reserved),
        routing_eligible=bool(routing_eligible),
        routing_ineligible_reason="" if routing_eligible else str(routing_ineligible_reason),
        feature_gate=str(feature_gate),
    )


_PERCEPTION_CHANNELS: tuple[PerceptionChannelMetadata, ...] = (
    _channel(
        "local_grid_state",
        "Local Grid State",
        category="legacy_spatial",
        active_now=True,
        cost_weight=1.0,
        source="legacy food/occupancy scalar local grid channel",
        description="Existing fixed local channel: -1 for occupied/blocked, 0 neutral, 1 food.",
        local_channel_index=0,
        patch_channel=False,
        value_min=-1.0,
        feature_gate="always",
    ),
    _channel(
        "patch_food_amount_norm",
        "Patch Food Amount",
        category="patch_world",
        active_now=True,
        cost_weight=1.0,
        source="patch_world_normalized_observation_maps()[0]",
        description="Existing Patch World normalized food amount channel, available only when patch observations are enabled.",
        local_channel_index=1,
        patch_channel=True,
        feature_gate="patch_world_observation_enable",
    ),
    _channel(
        "patch_cover",
        "Patch Cover",
        category="patch_world",
        active_now=True,
        cost_weight=1.0,
        source="patch_cover",
        description="Existing Patch World cover channel, available only when patch observations are enabled.",
        local_channel_index=2,
        patch_channel=True,
        feature_gate="patch_world_observation_enable",
    ),
    _channel(
        "patch_shelter",
        "Patch Shelter",
        category="patch_world",
        active_now=True,
        cost_weight=1.0,
        source="patch_shelter",
        description="Existing Patch World shelter channel, available only when patch observations are enabled.",
        local_channel_index=3,
        patch_channel=True,
        feature_gate="patch_world_observation_enable",
    ),
    _channel(
        "patch_scent_norm",
        "Patch Scent",
        category="patch_world",
        active_now=True,
        cost_weight=1.0,
        source="patch_scent normalized by patch_scent_max",
        description="Existing Patch World scent/activity trace channel, available only when patch observations are enabled.",
        local_channel_index=4,
        patch_channel=True,
        feature_gate="patch_world_observation_enable",
    ),
    _channel(
        "patch_waste_norm",
        "Patch Waste",
        category="patch_world",
        active_now=True,
        cost_weight=1.0,
        source="patch_waste normalized by patch_waste_max",
        description="Existing Patch World waste channel, available only when patch observations are enabled.",
        local_channel_index=5,
        patch_channel=True,
        feature_gate="patch_world_observation_enable",
    ),
    _channel(
        "patch_density_norm",
        "Patch Density",
        category="patch_world",
        active_now=True,
        cost_weight=1.0,
        source="local_density_map normalized by Patch World density settings",
        description="Existing Patch World local crowding density channel, available only when patch observations are enabled.",
        local_channel_index=6,
        patch_channel=True,
        feature_gate="patch_world_observation_enable",
    ),
    _channel(
        "patch_movement_cost_norm",
        "Patch Movement Cost",
        category="patch_world",
        active_now=True,
        cost_weight=1.0,
        source="patch_movement_cost normalized by configured movement-cost range",
        description="Existing Patch World terrain movement-cost channel, available only when patch observations are enabled.",
        local_channel_index=7,
        patch_channel=True,
        feature_gate="patch_world_observation_enable",
    ),
    _channel(
        "self_energy",
        "Self Energy",
        category="self_state",
        active_now=True,
        cost_weight=0.5,
        source="agent.energy / env_cfg.max_energy",
        description="Existing normalized self-energy scalar appended to the policy observation.",
        self_state_index=0,
        spatial=False,
        patch_channel=False,
        routing_eligible=True,
        routing_ineligible_reason="",
    ),
    _channel(
        "self_health",
        "Self Health",
        category="self_state",
        active_now=True,
        cost_weight=0.5,
        source="agent.health / env_cfg.start_health",
        description="Existing normalized self-health scalar appended to the policy observation.",
        self_state_index=1,
        spatial=False,
        patch_channel=False,
        routing_eligible=True,
        routing_ineligible_reason="",
    ),
    _channel(
        "patch_fertility_norm",
        "Patch Fertility",
        category="patch_world_future",
        active_now=False,
        future_reserved=True,
        default_enabled=False,
        cost_weight=1.0,
        source="patch_fertility exists but is not currently in the fixed observation vector",
        description="Future-reserved Patch World fertility perception channel.",
        patch_channel=True,
        routing_ineligible_reason="future_reserved_not_directly_observed",
        feature_gate="unsupported",
    ),
    _channel(
        "patch_recovery_activity_norm",
        "Patch Recovery Activity",
        category="patch_world_future",
        active_now=False,
        future_reserved=True,
        default_enabled=False,
        cost_weight=1.0,
        source="patch_recovery_activity exists but is not currently in the fixed observation vector",
        description="Future-reserved ecological recovery activity perception channel.",
        patch_channel=True,
        routing_ineligible_reason="future_reserved_not_directly_observed",
        feature_gate="unsupported",
    ),
    _channel(
        "jungle_predator_risk_norm",
        "Jungle Predator Risk",
        category="jungle_future",
        active_now=False,
        future_reserved=True,
        default_enabled=False,
        cost_weight=1.0,
        source="computed Jungle risk maps are not currently in the fixed observation vector",
        description="Future-reserved predator-risk perception channel.",
        patch_channel=True,
        routing_ineligible_reason="future_reserved_perfect_risk_sensor_prohibited",
        feature_gate="unsupported",
    ),
    _channel(
        "seasonal_signal",
        "Seasonal Signal",
        category="environment_future",
        active_now=False,
        future_reserved=True,
        default_enabled=False,
        cost_weight=1.0,
        source="not implemented",
        description="Future-reserved seasonal perception channel.",
        patch_channel=True,
        routing_ineligible_reason="future_reserved_not_implemented",
        feature_gate="unsupported",
    ),
)

_CHANNEL_BY_ID = {c.channel_id: c for c in _PERCEPTION_CHANNELS}
_CHANNEL_INDEX = {c.channel_id: i for i, c in enumerate(_PERCEPTION_CHANNELS)}


@dataclass(frozen=True)
class PerceptionExpression:
    schema_version: str
    active_sensor_channel_mask: int
    active_sensor_channel_ids: tuple[str, ...]
    active_sensor_channel_count: int
    potential_sensor_channel_count: int
    available_sensor_channel_count: int
    future_reserved_channel_count: int
    perception_enabled: bool
    perception_masking_enabled: bool
    perception_cost_enabled: bool
    sensor_channel_cost_total: float
    sensor_channel_cost_per_tick: float
    sensor_channel_mask_ablation_active: bool
    sensor_channel_force_all_active: bool
    sensor_channel_force_minimal_active: bool
    active_sensor_radius: int
    max_sensor_radius: int
    default_sensor_radius: int
    sensor_radius_masking_enabled: bool
    active_observed_cell_count: int
    max_observed_cell_count: int
    active_cell_fraction: float
    cell_channel_capacity: float
    spatial_perception_capacity: float
    perception_capacity_component: float
    sensor_radius_cost_hypothetical: float
    sensor_radius_cost_charged: float
    perception_cost_hypothetical: float
    perception_cost_charged: float
    sensor_radius_ablation_active: bool
    sensor_radius_force_min_active: bool
    sensor_radius_force_max_active: bool
    sensor_radius_force_default_active: bool

    def to_dict(self) -> dict[str, object]:
        out = asdict(self)
        out["active_sensor_channel_ids"] = list(self.active_sensor_channel_ids)
        return out


def perception_channel_registry() -> tuple[PerceptionChannelMetadata, ...]:
    return _PERCEPTION_CHANNELS


def perception_channel_ids() -> tuple[str, ...]:
    return tuple(c.channel_id for c in _PERCEPTION_CHANNELS)


def perception_channel_registry_dict() -> dict[str, object]:
    return {
        "schema_version": PERCEPTION_CHANNEL_REGISTRY_SCHEMA_VERSION,
        "routing_contract_version": PERCEPTION_ROUTING_CONTRACT_VERSION,
        "channels": [c.to_dict() for c in _PERCEPTION_CHANNELS],
    }


def get_perception_channel(channel_id: str) -> PerceptionChannelMetadata:
    return _CHANNEL_BY_ID[str(channel_id)]


def implemented_sensor_channel_ids() -> tuple[str, ...]:
    return tuple(c.channel_id for c in _PERCEPTION_CHANNELS if c.active_now and not c.future_reserved)


def implemented_sensor_channel_mask() -> int:
    mask = 0
    for cid in implemented_sensor_channel_ids():
        mask |= 1 << int(_CHANNEL_INDEX[cid])
    return int(mask)


def minimal_sensor_channel_mask() -> int:
    return int(1 << int(_CHANNEL_INDEX["local_grid_state"]))


def sanitize_sensor_channel_mask(mask: Any, *, preserve_basic: bool = True) -> int:
    try:
        out = int(mask)
    except Exception:
        out = implemented_sensor_channel_mask()
    out &= implemented_sensor_channel_mask()
    if preserve_basic and (out & minimal_sensor_channel_mask()) == 0:
        out |= minimal_sensor_channel_mask()
    return int(out)


def sensor_channel_mask_contains(mask: int, channel_id: str) -> bool:
    idx = int(_CHANNEL_INDEX[str(channel_id)])
    return bool(int(mask) & (1 << idx))


def sensor_channel_ids_from_mask(mask: int, *, env_cfg: Any = None) -> tuple[str, ...]:
    return tuple(
        c.channel_id
        for c in _PERCEPTION_CHANNELS
        if c.active_now and _channel_available(c, env_cfg) and sensor_channel_mask_contains(mask, c.channel_id)
    )


def snap_sensor_radius(value: Any, *, max_radius: int | None = None) -> int:
    try:
        raw = int(value)
    except Exception:
        raw = int(DEFAULT_SENSOR_RADIUS)
    radius = int(min(SENSOR_RADIUS_LADDER, key=lambda v: abs(int(v) - raw)))
    if max_radius is not None:
        radius = min(radius, max(1, int(max_radius)))
    return int(max(1, radius))


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


def _patch_observation_enabled(env_cfg: Any) -> bool:
    try:
        from env.gridworld_pkg.patch_world import patch_world_observation_enabled

        return bool(patch_world_observation_enabled(env_cfg))
    except Exception:
        return False


def _channel_available(channel: PerceptionChannelMetadata, env_cfg: Any) -> bool:
    if not channel.active_now or channel.future_reserved:
        return False
    if channel.patch_channel:
        return bool(_patch_observation_enabled(env_cfg))
    return True


def available_sensor_channel_ids(env_cfg: Any = None) -> tuple[str, ...]:
    return tuple(c.channel_id for c in _PERCEPTION_CHANNELS if _channel_available(c, env_cfg))


def available_sensor_channel_mask(env_cfg: Any = None) -> int:
    mask = 0
    for cid in available_sensor_channel_ids(env_cfg):
        mask |= 1 << int(_CHANNEL_INDEX[cid])
    return int(mask)


def express_perception(
    genome: Any,
    *,
    env_cfg: Any = None,
    sim_cfg: Any = None,
) -> PerceptionExpression:
    available_mask = available_sensor_channel_mask(env_cfg)
    potential_count = len(implemented_sensor_channel_ids())
    available_count = len(available_sensor_channel_ids(env_cfg))
    preserve_basic = _cfg_bool(sim_cfg, "sensor_channel_preserve_basic_channel", True)

    raw_mask = sanitize_sensor_channel_mask(
        _gget(genome, "sensor_channel_mask", implemented_sensor_channel_mask()),
        preserve_basic=preserve_basic,
    )
    channel_feature_enabled = _cfg_bool(sim_cfg, "enable_evolvable_sensor_channel_mask", False)
    ablate = _cfg_bool(sim_cfg, "ablate_sensor_channel_mask", False)
    force_all = _cfg_bool(sim_cfg, "force_all_sensor_channels_enabled", False)
    force_min = _cfg_bool(sim_cfg, "force_minimal_sensor_channels_enabled", False)

    if force_all:
        active_mask = implemented_sensor_channel_mask()
    elif force_min:
        active_mask = minimal_sensor_channel_mask()
    elif ablate or not channel_feature_enabled:
        active_mask = implemented_sensor_channel_mask()
    else:
        active_mask = raw_mask
    active_mask = sanitize_sensor_channel_mask(active_mask, preserve_basic=preserve_basic)
    effective_mask = int(active_mask & available_mask)
    if preserve_basic and (available_mask & minimal_sensor_channel_mask()) and (effective_mask & minimal_sensor_channel_mask()) == 0:
        effective_mask |= minimal_sensor_channel_mask()

    active_ids = sensor_channel_ids_from_mask(effective_mask, env_cfg=env_cfg)
    active_weight = 0.0
    for cid in active_ids:
        active_weight += float(_CHANNEL_BY_ID[cid].cost_weight)

    max_radius = int(max(1, getattr(env_cfg, "view_radius", DEFAULT_SENSOR_RADIUS) if env_cfg is not None else DEFAULT_SENSOR_RADIUS))
    default_radius = int(min(max_radius, max(1, DEFAULT_SENSOR_RADIUS)))
    radius_feature_enabled = _cfg_bool(sim_cfg, "enable_evolvable_sensor_radius", False)
    radius_ablate = _cfg_bool(sim_cfg, "ablate_sensor_radius", False)
    force_min_radius = _cfg_bool(sim_cfg, "force_min_sensor_radius", False)
    force_max_radius = _cfg_bool(sim_cfg, "force_max_sensor_radius", False)
    force_default_radius = _cfg_bool(sim_cfg, "force_default_sensor_radius", False)

    genome_radius = snap_sensor_radius(_gget(genome, "sensor_radius", default_radius), max_radius=max_radius)
    if force_min_radius:
        active_radius = int(min(SENSOR_RADIUS_LADDER))
    elif force_max_radius:
        active_radius = int(max_radius)
    elif force_default_radius or radius_ablate or not radius_feature_enabled:
        active_radius = int(max_radius)
    else:
        active_radius = int(genome_radius)
    active_radius = int(max(1, min(active_radius, max_radius)))

    active_cells = int((2 * active_radius + 1) ** 2)
    max_cells = int((2 * max_radius + 1) ** 2)
    active_cell_fraction = float(active_cells / max_cells) if max_cells > 0 else 0.0
    cell_channel_capacity = float(active_cells) * float(len(active_ids))

    channel_unit_cost = max(0.0, _cfg_float(sim_cfg, "sensor_channel_cost_per_weight", 0.001))
    channel_cost = float(active_weight) * float(channel_unit_cost)
    radius_base = max(0.0, _cfg_float(sim_cfg, "sensor_radius_cost_base", 0.0))
    radius_cell = max(0.0, _cfg_float(sim_cfg, "sensor_radius_cost_per_cell", 0.0001))
    radius_cell_channel = max(0.0, _cfg_float(sim_cfg, "sensor_radius_cost_per_cell_channel", 0.00005))
    radius_hyp = float(radius_base + radius_cell * active_cells + radius_cell_channel * cell_channel_capacity)

    charge_channels = _cfg_bool(sim_cfg, "enable_experimental_perception_cost_charging", False)
    charge_radius = _cfg_bool(sim_cfg, "enable_experimental_sensor_radius_cost_charging", False)
    perception_charged = float((channel_cost if charge_channels else 0.0) + (radius_hyp if charge_radius else 0.0))
    perception_hyp = float(channel_cost + radius_hyp)

    return PerceptionExpression(
        schema_version=PERCEPTION_EXPRESSION_SCHEMA_VERSION,
        active_sensor_channel_mask=int(effective_mask),
        active_sensor_channel_ids=tuple(active_ids),
        active_sensor_channel_count=int(len(active_ids)),
        potential_sensor_channel_count=int(potential_count),
        available_sensor_channel_count=int(available_count),
        future_reserved_channel_count=int(sum(1 for c in _PERCEPTION_CHANNELS if c.future_reserved)),
        perception_enabled=bool(len(active_ids) > 0),
        perception_masking_enabled=bool(channel_feature_enabled or ablate or force_all or force_min),
        perception_cost_enabled=bool(charge_channels or charge_radius),
        sensor_channel_cost_total=float(channel_cost),
        sensor_channel_cost_per_tick=float(channel_cost),
        sensor_channel_mask_ablation_active=bool(ablate),
        sensor_channel_force_all_active=bool(force_all),
        sensor_channel_force_minimal_active=bool(force_min),
        active_sensor_radius=int(active_radius),
        max_sensor_radius=int(max_radius),
        default_sensor_radius=int(default_radius),
        sensor_radius_masking_enabled=bool(radius_feature_enabled or radius_ablate or force_min_radius or force_max_radius or force_default_radius),
        active_observed_cell_count=int(active_cells),
        max_observed_cell_count=int(max_cells),
        active_cell_fraction=float(active_cell_fraction),
        cell_channel_capacity=float(cell_channel_capacity),
        spatial_perception_capacity=float(cell_channel_capacity),
        perception_capacity_component=float(cell_channel_capacity),
        sensor_radius_cost_hypothetical=float(radius_hyp),
        sensor_radius_cost_charged=float(radius_hyp if charge_radius else 0.0),
        perception_cost_hypothetical=float(perception_hyp),
        perception_cost_charged=float(perception_charged),
        sensor_radius_ablation_active=bool(radius_ablate),
        sensor_radius_force_min_active=bool(force_min_radius),
        sensor_radius_force_max_active=bool(force_max_radius),
        sensor_radius_force_default_active=bool(force_default_radius),
    )


def mutate_sensor_channel_mask(mask: Any, rng: Any, mut_rate: float, *, preserve_basic: bool = True) -> int:
    current = sanitize_sensor_channel_mask(mask, preserve_basic=preserve_basic)
    p = max(0.0, min(1.0, float(mut_rate)))
    if float(rng.random()) >= p:
        return int(current)
    ids = implemented_sensor_channel_ids()
    mutable_ids = tuple(cid for cid in ids if (not preserve_basic) or cid != "local_grid_state")
    if not mutable_ids:
        return int(current)
    cid = mutable_ids[int(rng.randrange(len(mutable_ids)))]
    idx = int(_CHANNEL_INDEX[cid])
    return sanitize_sensor_channel_mask(current ^ (1 << idx), preserve_basic=preserve_basic)


def recombine_sensor_channel_mask(mask_a: Any, mask_b: Any, rng: Any, *, preserve_basic: bool = True) -> int:
    a = sanitize_sensor_channel_mask(mask_a, preserve_basic=preserve_basic)
    b = sanitize_sensor_channel_mask(mask_b, preserve_basic=preserve_basic)
    out = 0
    for cid in implemented_sensor_channel_ids():
        idx = int(_CHANNEL_INDEX[cid])
        chosen = a if float(rng.random()) < 0.5 else b
        if chosen & (1 << idx):
            out |= 1 << idx
    return sanitize_sensor_channel_mask(out, preserve_basic=preserve_basic)


def mutate_sensor_radius(radius: Any, rng: Any, mut_rate: float, *, max_radius: int | None = None) -> int:
    current = snap_sensor_radius(radius, max_radius=max_radius)
    p = max(0.0, min(1.0, float(mut_rate)))
    if float(rng.random()) >= p:
        return int(current)
    ladder = tuple(v for v in SENSOR_RADIUS_LADDER if max_radius is None or int(v) <= int(max_radius))
    if not ladder:
        ladder = (1,)
    idx = min(range(len(ladder)), key=lambda i: abs(int(ladder[i]) - current))
    step = -1 if float(rng.random()) < 0.5 else 1
    idx = max(0, min(len(ladder) - 1, idx + step))
    return int(ladder[idx])


def recombine_sensor_radius(radius_a: Any, radius_b: Any, rng: Any, *, max_radius: int | None = None) -> int:
    chosen = radius_a if float(rng.random()) < 0.5 else radius_b
    return snap_sensor_radius(chosen, max_radius=max_radius)


def apply_perception_masks(
    *,
    local: np.ndarray,
    energy_norm: np.ndarray,
    health_norm: np.ndarray,
    agents: Iterable[Any],
    env_cfg: Any,
    sim_cfg: Any,
) -> None:
    """Apply deterministic in-place perception masks to fixed-shape observations."""
    if local.size <= 0:
        return
    agents_list = list(agents)
    if not agents_list:
        return
    patch_enabled = _patch_observation_enabled(env_cfg)
    channel_count = 8 if patch_enabled else 1
    max_radius = int(max(1, getattr(env_cfg, "view_radius", DEFAULT_SENSOR_RADIUS)))
    k = int(2 * max_radius + 1)
    cells = int(k * k)
    expected = int(cells * channel_count)
    if int(local.shape[1]) != expected:
        return
    cell_visible_cache: dict[int, np.ndarray] = {}

    for i, agent in enumerate(agents_list):
        expr = express_perception(getattr(agent, "genome", None), env_cfg=env_cfg, sim_cfg=sim_cfg)
        mask = int(expr.active_sensor_channel_mask)
        row = local[i, :]
        for channel in _PERCEPTION_CHANNELS:
            if not channel.active_now or channel.future_reserved:
                continue
            if channel.local_channel_index is not None:
                ci = int(channel.local_channel_index)
                if ci >= channel_count:
                    continue
                if not sensor_channel_mask_contains(mask, channel.channel_id):
                    start = ci * cells
                    row[start:start + cells] = np.float32(channel.neutral_masked_value)
            elif channel.self_state_index == 0:
                if not sensor_channel_mask_contains(mask, channel.channel_id):
                    energy_norm[i] = np.float32(channel.neutral_masked_value)
            elif channel.self_state_index == 1:
                if not sensor_channel_mask_contains(mask, channel.channel_id):
                    health_norm[i] = np.float32(channel.neutral_masked_value)

        active_radius = int(expr.active_sensor_radius)
        if active_radius < max_radius:
            cell_visible = cell_visible_cache.get(active_radius)
            if cell_visible is None:
                visible = np.zeros((k, k), dtype=bool)
                center = max_radius
                for yy in range(k):
                    for xx in range(k):
                        if max(abs(yy - center), abs(xx - center)) <= active_radius:
                            visible[yy, xx] = True
                cell_visible = visible.reshape(cells)
                cell_visible_cache[active_radius] = cell_visible
            for ci in range(channel_count):
                start = ci * cells
                block = row[start:start + cells]
                block[~cell_visible] = np.float32(0.0)
