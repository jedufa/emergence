"""Dormant resource-specific handling representation helpers.

REP-RH0 owns representation only. These helpers expose a bounded fixed-budget
Resource-A/Resource-B capacity map, but do not couple it to live resource
mechanics.
"""

from __future__ import annotations

import math
import random
from typing import Any

from agents.modular_genome_v1 import canon_float


RESOURCE_HANDLING_SCHEMA_VERSION = "rep_rh0_resource_handling_tradeoff_v1"
RESOURCE_HANDLING_AXIS_FIELD = "resource_handling_axis"
RESOURCE_HANDLING_AXIS_NEUTRAL = 0.0
RESOURCE_HANDLING_AXIS_MIN = -1.0
RESOURCE_HANDLING_AXIS_MAX = 1.0
RESOURCE_HANDLING_CAPACITY_TOTAL = 2.0
RESOURCE_HANDLING_MUTATION_SIGMA = 0.10


def clamp_resource_handling_axis(value: Any) -> float:
    """Return a canonical bounded resource-handling axis value."""

    try:
        x = float(value)
    except Exception:
        x = RESOURCE_HANDLING_AXIS_NEUTRAL
    if not math.isfinite(x):
        x = RESOURCE_HANDLING_AXIS_NEUTRAL
    x = min(RESOURCE_HANDLING_AXIS_MAX, max(RESOURCE_HANDLING_AXIS_MIN, x))
    return canon_float(x, field=RESOURCE_HANDLING_AXIS_FIELD)


def resource_handling_axis_is_neutral(axis: Any) -> bool:
    return clamp_resource_handling_axis(axis) == RESOURCE_HANDLING_AXIS_NEUTRAL


def resource_handling_capacities(axis: Any) -> dict[str, float | str]:
    """Map a signed axis to reciprocal fixed-budget capacities.

    A positive axis favors Resource A and reduces Resource B. A negative axis
    favors Resource B and reduces Resource A. The capacity sum is invariant, so
    no axis value can improve both resource capacities over the neutral midpoint.
    """

    x = clamp_resource_handling_axis(axis)
    return {
        "schema_version": RESOURCE_HANDLING_SCHEMA_VERSION,
        "axis": x,
        "resource_a_capacity": canon_float(1.0 + x, field="resource_a_capacity"),
        "resource_b_capacity": canon_float(1.0 - x, field="resource_b_capacity"),
        "capacity_total": canon_float(RESOURCE_HANDLING_CAPACITY_TOTAL, field="capacity_total"),
    }


def resource_handling_tradeoff_summary(axis: Any) -> dict[str, Any]:
    caps = resource_handling_capacities(axis)
    a_capacity = float(caps["resource_a_capacity"])
    b_capacity = float(caps["resource_b_capacity"])
    neutral = resource_handling_capacities(RESOURCE_HANDLING_AXIS_NEUTRAL)
    return {
        "schema_version": RESOURCE_HANDLING_SCHEMA_VERSION,
        "axis": float(caps["axis"]),
        "resource_a_capacity": a_capacity,
        "resource_b_capacity": b_capacity,
        "capacity_total": float(caps["capacity_total"]),
        "neutral_resource_a_capacity": float(neutral["resource_a_capacity"]),
        "neutral_resource_b_capacity": float(neutral["resource_b_capacity"]),
        "resource_a_delta_from_neutral": canon_float(
            a_capacity - float(neutral["resource_a_capacity"]),
            field="resource_a_delta_from_neutral",
        ),
        "resource_b_delta_from_neutral": canon_float(
            b_capacity - float(neutral["resource_b_capacity"]),
            field="resource_b_delta_from_neutral",
        ),
        "no_universal_upgrade": bool(
            a_capacity <= float(neutral["resource_a_capacity"])
            or b_capacity <= float(neutral["resource_b_capacity"])
        ),
        "mechanics_status": "dormant_representation_only",
    }


def mutate_resource_handling_axis(
    axis: Any,
    rng: random.Random,
    *,
    mutation_rate: float,
    sigma: float = RESOURCE_HANDLING_MUTATION_SIGMA,
) -> tuple[float, dict[str, Any]]:
    """Deterministically mutate the axis using the caller-owned RNG."""

    parent_axis = clamp_resource_handling_axis(axis)
    rate = min(1.0, max(0.0, float(mutation_rate)))
    mutation_draw = float(rng.random())
    gaussian_delta = 0.0
    attempted = mutation_draw < rate
    unclamped = parent_axis
    if attempted:
        gaussian_delta = float(rng.gauss(0.0, float(sigma)))
        unclamped = float(parent_axis) + gaussian_delta
    child_axis = clamp_resource_handling_axis(unclamped)
    record = {
        "schema_version": RESOURCE_HANDLING_SCHEMA_VERSION,
        "parent_axis": parent_axis,
        "child_pre_mutation_axis": parent_axis,
        "child_post_mutation_axis": child_axis,
        "mutation_rate": rate,
        "mutation_sigma": float(sigma),
        "mutation_draw": mutation_draw,
        "mutation_attempted": bool(attempted),
        "gaussian_delta": canon_float(gaussian_delta, field="resource_handling_gaussian_delta"),
        "mutation_delta": canon_float(child_axis - parent_axis, field="resource_handling_mutation_delta"),
        "clamped": bool(child_axis != canon_float(unclamped, field="resource_handling_unclamped_axis")),
        "capacity_after_mutation": resource_handling_capacities(child_axis),
    }
    return child_axis, record
