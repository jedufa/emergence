from __future__ import annotations

"""Primitive-array helpers for CostChargeSurface benchmark probes.

These helpers are deliberately benchmark-only. Normal simulation import and
runtime execution must not require a compiler or a native shared library.
"""

from collections.abc import Sequence
from typing import Any

import numpy as np


COST_SURFACE_COLUMN_NAMES = (
    "sensor_channel_cost_per_tick",
    "sensor_radius_cost_hypothetical",
    "prediction_cost_charged",
    "developmental_cost_charged",
    "maturation_cost_charged",
    "introspection_cost_charged",
    "self_regulation_cost_charged",
    "self_modification_cost_charged",
)

COST_SURFACE_COLUMN_COUNT = len(COST_SURFACE_COLUMN_NAMES)


def as_cost_surface_matrix(rows: Any) -> np.ndarray:
    arr = np.asarray(rows, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("cost surface rows must be a 2D array")
    if int(arr.shape[1]) != COST_SURFACE_COLUMN_COUNT:
        raise ValueError(f"cost surface rows must have exactly {COST_SURFACE_COLUMN_COUNT} columns")
    return np.ascontiguousarray(arr)


def pack_cost_charge_surfaces(surfaces: Sequence[Any]) -> np.ndarray:
    rows = np.zeros((len(surfaces), COST_SURFACE_COLUMN_COUNT), dtype=np.float64)
    for row, surface in enumerate(surfaces):
        for col, name in enumerate(COST_SURFACE_COLUMN_NAMES):
            rows[row, col] = float(getattr(surface, name, 0.0))
    return np.ascontiguousarray(rows)


def cost_surface_scalars_python(
    rows: Any,
    *,
    charge_channels: bool,
    charge_radius: bool,
    ledger_enabled: bool = True,
) -> np.ndarray:
    matrix = as_cost_surface_matrix(rows)
    out = np.zeros((int(matrix.shape[0]),), dtype=np.float64)
    if not ledger_enabled:
        return out
    for i in range(int(matrix.shape[0])):
        perception = 0.0
        if charge_channels:
            perception += float(matrix[i, 0])
        if charge_radius:
            perception += float(matrix[i, 1])
        out[i] = (
            perception
            + float(matrix[i, 2])
            + float(matrix[i, 3])
            + float(matrix[i, 4])
            + float(matrix[i, 5])
            + float(matrix[i, 6])
            + float(matrix[i, 7])
        )
    return out


def cost_surface_scalars_numpy(
    rows: Any,
    *,
    charge_channels: bool,
    charge_radius: bool,
    ledger_enabled: bool = True,
) -> np.ndarray:
    matrix = as_cost_surface_matrix(rows)
    out = np.zeros((int(matrix.shape[0]),), dtype=np.float64)
    if not ledger_enabled:
        return out
    if charge_channels:
        out += matrix[:, 0]
    if charge_radius:
        out += matrix[:, 1]
    out += matrix[:, 2]
    out += matrix[:, 3]
    out += matrix[:, 4]
    out += matrix[:, 5]
    out += matrix[:, 6]
    out += matrix[:, 7]
    return out
