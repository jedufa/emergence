from __future__ import annotations

"""Diagnostic native-style kernels for NATIVE-1.

This module intentionally does not integrate with the simulator. It exposes a
small primitive-array kernel that mirrors the arithmetic part of cognitive
budget cost aggregation, so the native feasibility track can test build/import,
parity, and microbenchmark speed without changing simulation behavior.
"""

from typing import Tuple

import numpy as np

try:  # pragma: no cover - exercised conditionally by local environment
    from numba import njit

    NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover - environment-dependent fallback
    njit = None
    NUMBA_AVAILABLE = False


CostResult = Tuple[np.ndarray, np.ndarray, np.ndarray]


def _as_cost_matrix(components: np.ndarray) -> np.ndarray:
    arr = np.asarray(components, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("components must be a 2D array")
    if int(arr.shape[1]) != 8:
        raise ValueError("components must have exactly 8 cost columns")
    return np.ascontiguousarray(arr)


def _as_energy_vector(energy: np.ndarray, n: int) -> np.ndarray:
    arr = np.asarray(energy, dtype=np.float64).reshape(-1)
    if int(arr.shape[0]) != int(n):
        raise ValueError("energy length must match component row count")
    return np.ascontiguousarray(arr)


def add_arrays(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Tiny import/build smoke function for optional native-style acceleration."""
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    if aa.shape != bb.shape:
        raise ValueError("array shapes must match")
    if NUMBA_AVAILABLE:
        return _add_arrays_numba(np.ascontiguousarray(aa), np.ascontiguousarray(bb))
    return aa + bb


def cognitive_costs_python(components: np.ndarray, energy: np.ndarray) -> CostResult:
    """Reference Python loop for cognitive budget cost aggregation.

    Columns are:
    memory, prediction, sensor/perception, developmental, introspection,
    self_regulation, self_modification, directed_inheritance.
    """
    comp = _as_cost_matrix(components)
    en = _as_energy_vector(energy, int(comp.shape[0]))
    n = int(comp.shape[0])
    k = int(comp.shape[1])
    totals = np.zeros((n,), dtype=np.float64)
    energy_fraction = np.zeros((n,), dtype=np.float64)
    fractions = np.zeros((n, k), dtype=np.float64)
    for i in range(n):
        total = 0.0
        for j in range(k):
            total += float(comp[i, j])
        totals[i] = total
        if total > 0.0 and float(en[i]) > 0.0:
            energy_fraction[i] = total / float(en[i])
        if total > 0.0:
            for j in range(k):
                value = float(comp[i, j])
                if value > 0.0:
                    fractions[i, j] = value / total
    return totals, energy_fraction, fractions


def cognitive_costs_numpy(components: np.ndarray, energy: np.ndarray) -> CostResult:
    """Vectorized NumPy comparison implementation."""
    comp = _as_cost_matrix(components)
    en = _as_energy_vector(energy, int(comp.shape[0]))
    totals = np.sum(comp, axis=1, dtype=np.float64)
    energy_fraction = np.divide(totals, en, out=np.zeros_like(totals), where=(totals > 0.0) & (en > 0.0))
    fractions = np.divide(comp, totals[:, None], out=np.zeros_like(comp), where=totals[:, None] > 0.0)
    fractions = np.where(comp > 0.0, fractions, 0.0)
    return totals, energy_fraction, fractions


def cognitive_costs_native(components: np.ndarray, energy: np.ndarray) -> CostResult:
    """Optional native-style candidate; falls back to NumPy if Numba is absent."""
    comp = _as_cost_matrix(components)
    en = _as_energy_vector(energy, int(comp.shape[0]))
    if NUMBA_AVAILABLE:
        return _cognitive_costs_numba(comp, en)
    return cognitive_costs_numpy(comp, en)


def backend_name() -> str:
    return "numba" if NUMBA_AVAILABLE else "numpy-fallback"


if NUMBA_AVAILABLE:

    @njit(cache=True)
    def _add_arrays_numba(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        out = np.empty_like(a)
        for i in range(a.size):
            out.flat[i] = a.flat[i] + b.flat[i]
        return out

    @njit(cache=True)
    def _cognitive_costs_numba(components: np.ndarray, energy: np.ndarray) -> CostResult:
        n = components.shape[0]
        k = components.shape[1]
        totals = np.zeros((n,), dtype=np.float64)
        energy_fraction = np.zeros((n,), dtype=np.float64)
        fractions = np.zeros((n, k), dtype=np.float64)
        for i in range(n):
            total = 0.0
            for j in range(k):
                total += components[i, j]
            totals[i] = total
            if total > 0.0 and energy[i] > 0.0:
                energy_fraction[i] = total / energy[i]
            if total > 0.0:
                for j in range(k):
                    value = components[i, j]
                    if value > 0.0:
                        fractions[i, j] = value / total
        return totals, energy_fraction, fractions

else:

    def _add_arrays_numba(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return a + b
