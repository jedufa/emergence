"""Default-off PATH1-DRH0 delayed-refuge reporting helpers.

This module derives reporting-only metrics from existing WDL delayed-hazard
state plus patch cover/shelter maps. It does not alter rewards, actions,
hazard dynamics, or observations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy as np


PATH1_DRH0_METRIC_FIELDS: tuple[str, ...] = (
    "path1_refuge_cue_cell_count",
    "path1_refuge_active_cell_count",
    "path1_refuge_active_without_prior_cue_count",
    "path1_refuge_cue_without_active_count",
    "path1_refuge_zone_cell_count",
    "path1_refuge_reachable_zone_cell_count",
    "path1_refuge_reachable_before_active_fraction",
    "path1_refuge_agents_in_hazard_at_cue",
    "path1_refuge_agents_reachable_refuge_at_cue",
    "path1_refuge_agents_in_refuge_before_active",
    "path1_refuge_agents_stayed_in_hazard_until_active",
    "path1_refuge_shelter_occupancy_pre_active",
    "path1_refuge_cover_occupancy_pre_active",
    "path1_refuge_move_toward_refuge_pre_active",
    "path1_refuge_move_away_from_refuge_pre_active",
    "path1_refuge_disabled_control_leakage_flag",
    "path1_drh0_cue_exposure_count",
    "path1_drh0_cue_visible_steps",
    "path1_drh0_hidden_window_steps",
    "path1_drh0_hazard_active_steps",
    "path1_drh0_refuge_cell_count",
    "path1_drh0_reachable_refuge_count",
    "path1_drh0_distance_to_nearest_refuge_at_cue",
    "path1_drh0_movement_toward_refuge_after_cue",
    "path1_drh0_refuge_occupancy_before_hazard",
    "path1_drh0_refuge_occupancy_during_hazard",
    "path1_drh0_damage_in_refuge",
    "path1_drh0_damage_outside_refuge",
    "path1_drh0_damage_prevented_by_refuge",
    "path1_drh0_failed_reachability_cases",
    "path1_drh0_action_timing_before_consequence",
    "path1_drh0_outcome_delta_vs_disabled_controls",
    "path1_drh0_pre_consequence_refuge_arrival_rate",
    "path1_drh0_cue_conditioned_refuge_movement_delta",
    "path1_drh0_refuge_reachability_success_rate",
    "path1_drh0_early_move_cost_paid",
    "path1_drh0_refuge_over_static_shelter_delta",
    "path1_drh0_hazard_damage_reduction_given_pre_arrival",
    "path1_drh0_hidden_window_action_count",
    "path1_drh0_no_cue_false_positive_refuge_movement",
)


def zero_path1_drh0_metrics() -> dict[str, float]:
    """Return the all-zero disabled scaffold payload."""
    return {name: 0.0 for name in PATH1_DRH0_METRIC_FIELDS}


def path1_drh0_reporting_enabled(cfg: Any) -> bool:
    """Return true only for explicitly enabled reporting over WDL Patch World."""
    return (
        bool(getattr(cfg, "path1_drh0_reporting_enable", False))
        and bool(getattr(cfg, "wdl_delayed_hazard_enable", False))
        and bool(getattr(cfg, "patch_world_enable", False))
    )


def path1_drh0_mechanics_enabled(cfg: Any) -> bool:
    """Return true only for explicitly enabled DRH0 consequence mechanics."""
    return (
        path1_drh0_reporting_enabled(cfg)
        and bool(getattr(cfg, "path1_drh0_consequence_enable", False))
        and bool(getattr(cfg, "path1_drh0_refuge_damage_reduction_enable", False))
    )


@dataclass(frozen=True)
class Path1DRH0DamageAdjustment:
    raw_damage: float
    effective_damage: float
    prevented_damage: float
    in_refuge: bool
    protected: bool
    static_refuge_control: bool
    cue_conditioned: bool
    cue_observable: bool


def _bounded_float(value: Any, default: float, *, lo: float = 0.0, hi: float | None = None) -> float:
    try:
        out = float(value)
    except Exception:
        out = float(default)
    out = max(float(lo), float(out))
    if hi is not None:
        out = min(float(hi), out)
    return out


def _float_metric(metrics: Mapping[str, Any] | None, name: str, default: float = 0.0) -> float:
    if not metrics:
        return float(default)
    try:
        return float(metrics.get(name, default))
    except Exception:
        return float(default)


def _bounded_int(value: Any, default: int, *, lo: int = 0) -> int:
    try:
        out = int(value)
    except Exception:
        out = int(default)
    return max(int(lo), int(out))


def _array(env: Any, name: str, *, dtype: Any, shape: tuple[int, int] | None = None, default: float = 0.0) -> np.ndarray:
    if hasattr(env, name):
        return np.asarray(getattr(env, name), dtype=dtype)
    if shape is None:
        return np.zeros((0, 0), dtype=dtype)
    return np.full(shape, default, dtype=dtype)


def _refuge_mask(env: Any, event_mask: np.ndarray, radius: int) -> np.ndarray:
    cfg = getattr(env, "cfg", None)
    h, w = tuple(event_mask.shape)
    cover = _array(env, "patch_cover", dtype=np.float32, shape=(h, w), default=0.0)
    shelter = _array(env, "patch_shelter", dtype=np.float32, shape=(h, w), default=0.0)
    if bool(getattr(cfg, "path1_drh0_refuge_requires_shelter_or_cover", True)):
        refuge_source = (cover > 0.0) | (shelter > 0.0)
    else:
        refuge_source = np.ones((h, w), dtype=bool)

    zone = np.zeros((h, w), dtype=bool)
    event_cells = np.argwhere(event_mask)
    for y_raw, x_raw in event_cells.tolist():
        y = int(y_raw)
        x = int(x_raw)
        y0 = max(0, y - int(radius))
        y1 = min(h - 1, y + int(radius))
        x0 = max(0, x - int(radius))
        x1 = min(w - 1, x + int(radius))
        yy, xx = np.ogrid[y0 : y1 + 1, x0 : x1 + 1]
        zone[y0 : y1 + 1, x0 : x1 + 1] |= (np.abs(yy - y) + np.abs(xx - x)) <= int(radius)
    return zone & refuge_source


def path1_drh0_refuge_mask(env: Any) -> np.ndarray:
    """Return event-linked refuge cells for current WDL/DRH0 state."""
    cue = _array(env, "patch_delayed_hazard_cue", dtype=np.float32)
    if cue.size <= 0:
        return np.zeros((0, 0), dtype=bool)
    active = _array(env, "patch_delayed_hazard_active", dtype=np.float32, shape=tuple(cue.shape), default=0.0)
    prior = _array(env, "patch_delayed_hazard_prior_cue", dtype=np.int8, shape=tuple(cue.shape), default=0.0)
    cfg = getattr(env, "cfg", None)
    event_mask = (cue > 0.0) | (active > 0.0) | (prior > 0)
    radius = _bounded_int(getattr(cfg, "path1_drh0_refuge_radius", 3), 3, lo=0)
    refuge = _refuge_mask(env, event_mask, radius)
    if int(np.sum(refuge)) < _bounded_int(getattr(cfg, "path1_drh0_min_refuge_cells", 1), 1, lo=0):
        return np.zeros_like(refuge, dtype=bool)
    return refuge


def _reachable_refuge_mask(env: Any, cue_mask: np.ndarray, refuge: np.ndarray) -> np.ndarray:
    cfg = getattr(env, "cfg", None)
    if not bool(getattr(cfg, "path1_drh0_refuge_reachability_enable", False)):
        return np.zeros_like(refuge, dtype=bool)
    if not bool(np.any(cue_mask)) or not bool(np.any(refuge)):
        return np.zeros_like(refuge, dtype=bool)

    timer = _array(env, "patch_delayed_hazard_timer", dtype=np.int32, shape=cue_mask.shape, default=-1)
    default_steps = _bounded_int(getattr(cfg, "wdl_delayed_hazard_delay_steps", 8), 8, lo=0)
    reachable = np.zeros_like(refuge, dtype=bool)
    refuge_cells = np.argwhere(refuge)
    for y_raw, x_raw in np.argwhere(cue_mask).tolist():
        y = int(y_raw)
        x = int(x_raw)
        steps = int(timer[y, x]) if int(timer[y, x]) >= 0 else int(default_steps)
        if steps < 0:
            continue
        distances = np.abs(refuge_cells[:, 0] - y) + np.abs(refuge_cells[:, 1] - x)
        for ry, rx in refuge_cells[distances <= int(steps)].tolist():
            reachable[int(ry), int(rx)] = True
    return reachable


def path1_drh0_damage_adjustment(
    env: Any,
    *,
    agent_id: int,
    y: int,
    x: int,
    raw_damage: float,
    cued_agent_ids: Iterable[int] | None = None,
) -> Path1DRH0DamageAdjustment:
    """Return DRH0 refuge damage adjustment for one active-hazard exposure."""
    cfg = getattr(env, "cfg", None)
    raw = max(0.0, float(raw_damage))
    if raw <= 0.0 or not path1_drh0_mechanics_enabled(cfg):
        return Path1DRH0DamageAdjustment(raw, raw, 0.0, False, False, False, False, False)

    refuge = path1_drh0_refuge_mask(env)
    in_refuge = bool(refuge.size and 0 <= int(y) < refuge.shape[0] and 0 <= int(x) < refuge.shape[1] and refuge[int(y), int(x)])
    if not in_refuge:
        return Path1DRH0DamageAdjustment(raw, raw, 0.0, False, False, False, False, False)

    static_control = bool(getattr(cfg, "path1_drh0_static_refuge_control_enable", False))
    cue_required = bool(getattr(cfg, "path1_drh0_cue_required_for_refuge_protection", True))
    pre_arrival_required = bool(getattr(cfg, "path1_drh0_pre_arrival_required_for_refuge_protection", True))
    cue_observable = bool(getattr(cfg, "path1_drh0_observation_enable", False)) or bool(
        getattr(cfg, "wdl_delayed_hazard_observation_enable", False)
    )
    cued = int(agent_id) in {int(aid) for aid in list(cued_agent_ids or [])}
    cue_conditioned = (
        (not cue_required or (cue_observable and cued))
        and (not pre_arrival_required or cued)
        and not static_control
    )
    protected = bool(static_control or cue_conditioned)
    if not protected:
        return Path1DRH0DamageAdjustment(raw, raw, 0.0, True, False, static_control, False, cue_observable)

    frac = _bounded_float(getattr(cfg, "path1_drh0_damage_reduction_fraction", 0.50), 0.50, lo=0.0, hi=1.0)
    cap = _bounded_float(getattr(cfg, "path1_drh0_damage_reduction_cap", 1.0), 1.0, lo=0.0)
    prevented = raw * frac
    if cap > 0.0:
        prevented = min(prevented, cap)
    prevented = min(raw, max(0.0, prevented))
    effective = max(0.0, raw - prevented)
    return Path1DRH0DamageAdjustment(
        raw_damage=raw,
        effective_damage=effective,
        prevented_damage=prevented,
        in_refuge=True,
        protected=True,
        static_refuge_control=static_control,
        cue_conditioned=cue_conditioned,
        cue_observable=cue_observable,
    )


def _alive_agent_positions(env: Any) -> list[tuple[int, int]]:
    try:
        alive = np.asarray(getattr(env, "agent_alive"), dtype=bool)
        pos_y = np.asarray(getattr(env, "agent_pos_y"), dtype=np.int32)
        pos_x = np.asarray(getattr(env, "agent_pos_x"), dtype=np.int32)
    except Exception:
        return []
    out: list[tuple[int, int]] = []
    for aid in np.nonzero(alive)[0].astype(np.int32).tolist():
        y = int(pos_y[int(aid)])
        x = int(pos_x[int(aid)])
        if y >= 0 and x >= 0:
            out.append((y, x))
    return out


def path1_drh0_generation_metrics(
    env: Any,
    *,
    generation_metrics: Mapping[str, Any] | None = None,
) -> dict[str, float]:
    """Return PATH1-DRH0 derived metrics for one generation row."""
    cfg = getattr(env, "cfg", None)
    if not path1_drh0_reporting_enabled(cfg):
        return zero_path1_drh0_metrics()
    if int(getattr(cfg, "path1_drh0_schema_version", 1) or 1) != 1:
        raise ValueError("unsupported path1_drh0_schema_version")

    cue = _array(env, "patch_delayed_hazard_cue", dtype=np.float32)
    if cue.size <= 0:
        return zero_path1_drh0_metrics()
    active = _array(env, "patch_delayed_hazard_active", dtype=np.float32, shape=tuple(cue.shape), default=0.0)
    prior = _array(env, "patch_delayed_hazard_prior_cue", dtype=np.int8, shape=tuple(cue.shape), default=0.0)
    cover = _array(env, "patch_cover", dtype=np.float32, shape=tuple(cue.shape), default=0.0)
    shelter = _array(env, "patch_shelter", dtype=np.float32, shape=tuple(cue.shape), default=0.0)

    cue_mask = cue > 0.0
    active_mask = active > 0.0
    pre_active_mask = cue_mask | ((prior > 0) & ~active_mask)
    refuge = path1_drh0_refuge_mask(env)
    reachable = _reachable_refuge_mask(env, cue_mask, refuge)

    positions = _alive_agent_positions(env)
    agents_in_hazard_at_cue = 0
    agents_reachable_at_cue = 0
    agents_in_refuge_pre_active = 0
    agents_in_refuge_during_hazard = 0
    shelter_occ_pre_active = 0
    cover_occ_pre_active = 0
    reachable_cells = np.argwhere(reachable)
    default_steps = _bounded_int(getattr(cfg, "wdl_delayed_hazard_delay_steps", 8), 8, lo=0)
    timer = _array(env, "patch_delayed_hazard_timer", dtype=np.int32, shape=tuple(cue.shape), default=-1)
    has_pre_active_context = bool(np.any(pre_active_mask))

    for y, x in positions:
        if y >= cue.shape[0] or x >= cue.shape[1]:
            continue
        if bool(cue_mask[y, x]):
            agents_in_hazard_at_cue += 1
            steps = int(timer[y, x]) if int(timer[y, x]) >= 0 else int(default_steps)
            if reachable_cells.size and bool(np.any(np.abs(reachable_cells[:, 0] - y) + np.abs(reachable_cells[:, 1] - x) <= int(steps))):
                agents_reachable_at_cue += 1
        if has_pre_active_context and bool(refuge[y, x]):
            agents_in_refuge_pre_active += 1
            if float(shelter[y, x]) > 0.0:
                shelter_occ_pre_active += 1
            if float(cover[y, x]) > 0.0:
                cover_occ_pre_active += 1
        if bool(active_mask[y, x]) and bool(refuge[y, x]):
            agents_in_refuge_during_hazard += 1

    refuge_count = float(np.sum(refuge))
    reachable_count = float(np.sum(reachable))
    cue_exposure_count = _float_metric(generation_metrics, "wdl_cue_seen_agents", float(agents_in_hazard_at_cue))
    movement_toward = _float_metric(generation_metrics, "wdl_cue_exposed_next_move_away_count", 0.0)
    movement_away = _float_metric(generation_metrics, "wdl_cue_exposed_next_move_toward_count", 0.0)
    cue_visible_steps = _float_metric(
        generation_metrics,
        "path1_drh0_cue_visible_steps",
        1.0 if bool(np.any(cue_mask)) else 0.0,
    )
    hidden_window_steps = _float_metric(
        generation_metrics,
        "path1_drh0_hidden_window_steps",
        1.0 if bool(np.any((prior > 0) & ~cue_mask & ~active_mask)) else 0.0,
    )
    hazard_active_steps = _float_metric(
        generation_metrics,
        "path1_drh0_hazard_active_steps",
        1.0 if bool(np.any(active_mask)) else 0.0,
    )
    damage_in_refuge = _float_metric(generation_metrics, "path1_drh0_damage_in_refuge", 0.0)
    damage_outside_refuge = _float_metric(
        generation_metrics,
        "path1_drh0_damage_outside_refuge",
        _float_metric(generation_metrics, "wdl_delayed_hazard_damage_total", 0.0),
    )
    damage_prevented = _float_metric(generation_metrics, "path1_drh0_damage_prevented_by_refuge", 0.0)
    static_prevented = _float_metric(generation_metrics, "path1_drh0_static_refuge_damage_prevented", 0.0)
    cue_prevented = _float_metric(generation_metrics, "path1_drh0_cue_conditioned_damage_prevented", 0.0)
    false_positive = _float_metric(generation_metrics, "path1_drh0_no_cue_false_positive_refuge_movement", 0.0)
    nearest_distance = 0.0
    if bool(np.any(cue_mask)) and bool(np.any(refuge)):
        refuge_cells = np.argwhere(refuge)
        distances: list[float] = []
        for y_raw, x_raw in np.argwhere(cue_mask).tolist():
            y = int(y_raw)
            x = int(x_raw)
            d = np.abs(refuge_cells[:, 0] - y) + np.abs(refuge_cells[:, 1] - x)
            if d.size:
                distances.append(float(np.min(d)))
        nearest_distance = float(np.mean(np.asarray(distances, dtype=np.float32))) if distances else 0.0
    reachable_success_rate = (
        float(agents_reachable_at_cue) / float(agents_in_hazard_at_cue)
        if int(agents_in_hazard_at_cue) > 0
        else 0.0
    )
    failed_reachability = max(0.0, float(agents_in_hazard_at_cue - agents_reachable_at_cue))
    pre_arrival_rate = (
        float(agents_in_refuge_pre_active) / float(agents_reachable_at_cue)
        if int(agents_reachable_at_cue) > 0
        else 0.0
    )
    return {
        "path1_refuge_cue_cell_count": _float_metric(
            generation_metrics, "wdl_delayed_hazard_cue_cell_count", float(np.sum(cue_mask))
        ),
        "path1_refuge_active_cell_count": _float_metric(
            generation_metrics, "wdl_delayed_hazard_active_cell_count", float(np.sum(active_mask))
        ),
        "path1_refuge_active_without_prior_cue_count": _float_metric(
            generation_metrics,
            "wdl_delayed_hazard_active_without_prior_cue_count",
            float(np.sum(active_mask & (prior <= 0))),
        ),
        "path1_refuge_cue_without_active_count": _float_metric(
            generation_metrics,
            "wdl_delayed_hazard_cue_without_active_count",
            float(np.sum((prior > 0) & ~active_mask)),
        ),
        "path1_refuge_zone_cell_count": refuge_count,
        "path1_refuge_reachable_zone_cell_count": reachable_count,
        "path1_refuge_reachable_before_active_fraction": reachable_count / refuge_count if refuge_count > 0.0 else 0.0,
        "path1_refuge_agents_in_hazard_at_cue": float(agents_in_hazard_at_cue),
        "path1_refuge_agents_reachable_refuge_at_cue": float(agents_reachable_at_cue),
        "path1_refuge_agents_in_refuge_before_active": float(agents_in_refuge_pre_active),
        "path1_refuge_agents_stayed_in_hazard_until_active": _float_metric(
            generation_metrics, "wdl_hazard_exposure_after_cue_count", 0.0
        ),
        "path1_refuge_shelter_occupancy_pre_active": float(shelter_occ_pre_active),
        "path1_refuge_cover_occupancy_pre_active": float(cover_occ_pre_active),
        "path1_refuge_move_toward_refuge_pre_active": _float_metric(
            generation_metrics, "wdl_cue_exposed_next_move_away_count", 0.0
        ),
        "path1_refuge_move_away_from_refuge_pre_active": _float_metric(
            generation_metrics, "wdl_cue_exposed_next_move_toward_count", 0.0
        ),
        "path1_refuge_disabled_control_leakage_flag": 0.0,
        "path1_drh0_cue_exposure_count": cue_exposure_count,
        "path1_drh0_cue_visible_steps": cue_visible_steps,
        "path1_drh0_hidden_window_steps": hidden_window_steps,
        "path1_drh0_hazard_active_steps": hazard_active_steps,
        "path1_drh0_refuge_cell_count": refuge_count,
        "path1_drh0_reachable_refuge_count": reachable_count,
        "path1_drh0_distance_to_nearest_refuge_at_cue": nearest_distance,
        "path1_drh0_movement_toward_refuge_after_cue": movement_toward,
        "path1_drh0_refuge_occupancy_before_hazard": float(agents_in_refuge_pre_active),
        "path1_drh0_refuge_occupancy_during_hazard": float(agents_in_refuge_during_hazard),
        "path1_drh0_damage_in_refuge": damage_in_refuge,
        "path1_drh0_damage_outside_refuge": damage_outside_refuge,
        "path1_drh0_damage_prevented_by_refuge": damage_prevented,
        "path1_drh0_failed_reachability_cases": failed_reachability,
        "path1_drh0_action_timing_before_consequence": movement_toward,
        "path1_drh0_outcome_delta_vs_disabled_controls": cue_prevented,
        "path1_drh0_pre_consequence_refuge_arrival_rate": pre_arrival_rate,
        "path1_drh0_cue_conditioned_refuge_movement_delta": movement_toward - movement_away,
        "path1_drh0_refuge_reachability_success_rate": reachable_success_rate,
        "path1_drh0_early_move_cost_paid": 0.0,
        "path1_drh0_refuge_over_static_shelter_delta": cue_prevented - static_prevented,
        "path1_drh0_hazard_damage_reduction_given_pre_arrival": (
            damage_prevented / (damage_prevented + damage_in_refuge)
            if (damage_prevented + damage_in_refuge) > 0.0
            else 0.0
        ),
        "path1_drh0_hidden_window_action_count": movement_toward + movement_away,
        "path1_drh0_no_cue_false_positive_refuge_movement": false_positive,
    }
