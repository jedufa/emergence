"""simulation/runner_pkg/shock.py

Shock scheduling mixin.

Chooses whether a generation is inside a shock window.

Modes (priority order):
1) Pulsed mode (Path A temporal specialization): long calm periods, short brutal windows.
   - shock_pulsed: bool
   - shock_calm_min / shock_calm_max
   - shock_brutal_min / shock_brutal_max
2) Stochastic mode: start probability with duration + cooldown.
   - shock_start_prob
   - shock_duration_min/max
   - shock_cooldown_min/max
3) Periodic legacy mode:
   - shock_period
   - shock_duration

Stateful: should be called once per generation.
"""

from __future__ import annotations

import random


class ShockMixin:
    def _ensure_shock_state(self) -> None:
        """Lazy state init (mixins may not run __init__)."""
        if not hasattr(self, "_shock_active_rem"):
            self._shock_active_rem = 0
        if not hasattr(self, "_shock_cooldown_rem"):
            self._shock_cooldown_rem = 0
        if not hasattr(self, "_shock_rng"):
            # Phase 1: use RNGManager policy stream (no new RNG island).
            rm = getattr(self, "rng_manager", None)
            if rm is not None:
                self._shock_rng = rm.policy_py
            else:
                raise RuntimeError("ShockMixin requires rng_manager (no RNG islands allowed)")

        if not hasattr(self, "_shock_last_active"):
            self._shock_last_active = False
        if not hasattr(self, "_shock_last_kind"):
            self._shock_last_kind = "none"

    def _shock_active(self, gen: int) -> bool:
        """Return True if shock is active for this generation."""
        self._ensure_shock_state()

        # Phase 1 determinism rule:
        # Shock logic must consume a fixed number of RNG draws per generation.
        # Draw a fixed block here and use the values conditionally.
        r1 = float(self._shock_rng.random())
        _ = float(self._shock_rng.random())
        _ = float(self._shock_rng.random())
        _ = float(self._shock_rng.random())
        i1 = int(self._shock_rng.randint(0, 2**31 - 1))
        i2 = int(self._shock_rng.randint(0, 2**31 - 1))
        i3 = int(self._shock_rng.randint(0, 2**31 - 1))
        i4 = int(self._shock_rng.randint(0, 2**31 - 1))

        def _pick_int(raw: int, low: int, high: int) -> int:
            low_i = int(low)
            high_i = int(high)
            if high_i < low_i:
                high_i = low_i
            span = (high_i - low_i) + 1
            return low_i + (int(raw) % int(span))

        # --- Pulsed calm/brutal mode (Path A) ---
        if bool(getattr(self.sim_cfg, "shock_pulsed", False)):
            self._shock_last_kind = "pulsed"

            if self._shock_active_rem > 0:
                self._shock_active_rem -= 1
                self._shock_last_active = True
                if self._shock_active_rem == 0:
                    cmin = int(getattr(self.sim_cfg, "shock_calm_min", 20))
                    cmax = int(getattr(self.sim_cfg, "shock_calm_max", 60))
                    self._shock_cooldown_rem = _pick_int(i1, cmin, cmax)
                return True

            if self._shock_cooldown_rem > 0:
                self._shock_cooldown_rem -= 1
                self._shock_last_active = False
                return False

            dmin = int(getattr(self.sim_cfg, "shock_brutal_min", 2))
            dmax = int(getattr(self.sim_cfg, "shock_brutal_max", 6))
            self._shock_active_rem = _pick_int(i2, dmin, dmax)
            self._shock_last_active = True
            return True

        # --- Stochastic mode ---
        p_start = getattr(self.sim_cfg, "shock_start_prob", None)
        if p_start is not None and float(p_start) > 0.0:
            self._shock_last_kind = "stochastic"

            if self._shock_active_rem > 0:
                self._shock_active_rem -= 1
                self._shock_last_active = True
                if self._shock_active_rem == 0:
                    cd_min = int(getattr(self.sim_cfg, "shock_cooldown_min", 0))
                    cd_max = int(getattr(self.sim_cfg, "shock_cooldown_max", 0))
                    if cd_max > 0:
                        self._shock_cooldown_rem = _pick_int(i3, cd_min, cd_max)
                return True

            if self._shock_cooldown_rem > 0:
                self._shock_cooldown_rem -= 1
                self._shock_last_active = False
                return False

            if r1 < float(p_start):
                dmin = int(getattr(self.sim_cfg, "shock_duration_min", 1))
                dmax = int(getattr(self.sim_cfg, "shock_duration_max", 1))
                self._shock_active_rem = _pick_int(i4, dmin, dmax)
                self._shock_last_active = True
                return True

            self._shock_last_active = False
            return False

        # --- Periodic legacy mode ---
        self._shock_last_kind = "periodic"
        p = int(getattr(self.sim_cfg, "shock_period", 0))
        d = int(getattr(self.sim_cfg, "shock_duration", 0))
        if p <= 0 or d <= 0:
            self._shock_last_active = False
            return False
        active = (gen % p) < d
        self._shock_last_active = bool(active)
        return bool(active)
