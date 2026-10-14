"""env/gridworld_pkg/energy.py

Effective max-energy calculation (vectorized).

Computes the maximum energy capacity per agent as a function of injury + optional scar.
Used by GridWorld stepping to clamp energy after costs and food gain.
"""

from __future__ import annotations

import numpy as np


class EnergyMixin:
    def _effective_max_energy_vec(self, injuries: np.ndarray, scars: np.ndarray | None = None) -> np.ndarray:
        max_e = float(self.cfg.max_energy)

        # Fast path: injury disabled
        if not getattr(self.cfg, "enable_injury", False):
            return np.full(injuries.shape, max_e, dtype=np.float32)

        inj_max = float(getattr(self.cfg, "injury_max", np.inf))
        if not np.isfinite(inj_max) or inj_max <= 0.0:
            inj_max = 1.0

        penalty = float(getattr(self.cfg, "injury_max_energy_penalty", 0.0))
        if penalty <= 0.0:
            return np.full(injuries.shape, max_e, dtype=np.float32)

        inj = np.clip(
            injuries.astype(np.float32, copy=False),
            0.0,
            inj_max,
        )

        # Include scar as additional impairment
        if scars is not None and getattr(self.cfg, "enable_scar", False):
            scar_max = float(getattr(self.cfg, "scar_max", inj_max))
            if not np.isfinite(scar_max) or scar_max <= 0.0:
                scar_max = inj_max
            sc = np.clip(scars.astype(np.float32, copy=False), 0.0, scar_max)
            w = float(getattr(self.cfg, "scar_cost_weight", 1.0))
            impairment = inj + w * sc
        else:
            impairment = inj

        # Normalize and apply concave curve so moderate injury bites hard
        x = np.clip(impairment / inj_max, 0.0, 1.0)
        p = float(getattr(self.cfg, "injury_penalty_curve_power", 1.0))
        if p <= 0.0:
            p = 1.0
        # NumPy's float32 power ufunc and C++ libm can differ by one ULP.
        # Define the shared route-neutral operation explicitly instead.
        curve = np.power(x.astype(np.float64), p).astype(np.float32, copy=False)

        # Minimum allowed fraction of max energy
        floor_frac = float(getattr(self.cfg, "injury_min_energy_frac", 0.20))

        # Amplitude: penalty * injury_max = max fractional reduction at full impairment
        amp = float(penalty) * float(inj_max)

        frac = 1.0 - amp * curve
        frac = np.maximum(frac, floor_frac)

        return (max_e * frac).astype(np.float32, copy=False)
