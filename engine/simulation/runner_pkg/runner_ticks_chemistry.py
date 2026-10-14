from __future__ import annotations

"""simulation/runner_pkg/runner_ticks_chemistry.py

Phase 1A deterministic chemistry tick update.
"""

from contextlib import nullcontext
from typing import Dict, Optional

import numpy as np

from agents.chemistry import (
    CHEM_SLOT_ACUTE_DAMAGE,
    CHEM_SLOT_CHRONIC_DAMAGE,
    CHEM_SLOT_ENERGY_STORE,
    CHEM_SLOT_REGULATOR_1,
    CHEM_SLOT_REGULATOR_2,
    CHEM_SLOT_REPAIR_RESERVE,
    CHEMISTRY_SIZE,
    chemistry_from_values,
    clamp01,
)


def chemistry_vector_step(
    chemistry: np.ndarray,
    energy: np.ndarray,
    max_energy: np.ndarray,
    health: np.ndarray,
    start_health: np.ndarray,
    injury: np.ndarray,
    drive_pain: np.ndarray,
    noise: np.ndarray,
) -> np.ndarray:
    """Vectorized equivalent of one scalar chemistry tick."""
    c = np.asarray(chemistry, dtype=np.float64)
    if c.ndim != 2 or int(c.shape[1]) != CHEMISTRY_SIZE:
        raise ValueError("chemistry must have shape (n, CHEMISTRY_SIZE)")
    n = int(c.shape[0])
    if n <= 0:
        return np.zeros((0, CHEMISTRY_SIZE), dtype=np.float32)

    e = np.asarray(energy, dtype=np.float64).reshape(n)
    emax = np.asarray(max_energy, dtype=np.float64).reshape(n)
    h = np.asarray(health, dtype=np.float64).reshape(n)
    hmax = np.asarray(start_health, dtype=np.float64).reshape(n)
    inj = np.asarray(injury, dtype=np.float64).reshape(n)
    pain = np.asarray(drive_pain, dtype=np.float64).reshape(n)
    nz = np.asarray(noise, dtype=np.float64).reshape(n, 2)

    emax = np.where(emax != 0.0, emax, 1.0)
    hmax = np.where(hmax != 0.0, hmax, 1.0)

    energy_n = np.clip(e / emax, 0.0, 1.0)
    health_n = np.clip(h / hmax, 0.0, 1.0)
    injury_n = np.clip(inj / hmax, 0.0, 1.0)
    stress = np.clip(0.55 * injury_n + 0.30 * pain + 0.15 * (1.0 - energy_n), 0.0, 1.0)

    energy_store = c[:, CHEM_SLOT_ENERGY_STORE].copy()
    repair = c[:, CHEM_SLOT_REPAIR_RESERVE].copy()
    acute = c[:, CHEM_SLOT_ACUTE_DAMAGE].copy()
    chronic = c[:, CHEM_SLOT_CHRONIC_DAMAGE].copy()
    reg1 = c[:, CHEM_SLOT_REGULATOR_1].copy()
    reg2 = c[:, CHEM_SLOT_REGULATOR_2].copy()

    acute_target = np.clip(1.25 * injury_n + 0.45 * stress + 0.20 * (1.0 - health_n), 0.0, 1.0)
    acute = np.clip(acute + 0.50 * (acute_target - acute) + 0.03 * reg1, 0.0, 1.0)

    chronic_target = np.clip(0.75 * chronic + 0.25 * acute, 0.0, 1.0)
    chronic = np.clip(chronic + 0.07 * (chronic_target - chronic), 0.0, 1.0)

    energy_store = np.clip(energy_store + 0.03 * energy_n - 0.02 * stress - 0.01 * acute, 0.0, 1.0)
    repair = np.clip(repair + 0.04 * energy_n - 0.03 * acute - 0.01 * chronic, 0.0, 1.0)

    reg1 = np.clip(0.92 * reg1 + 0.08 * (1.0 - acute) + 0.01 * nz[:, 0], 0.0, 1.0)
    reg2 = np.clip(0.97 * reg2 + 0.03 * chronic + 0.01 * nz[:, 1], 0.0, 1.0)

    acute_resolve = np.minimum(acute, 0.10 * repair + 0.04 * energy_store)
    acute = np.clip(acute - acute_resolve, 0.0, 1.0)
    chronic = np.clip(chronic + 0.25 * acute - 0.002 * repair, 0.0, 1.0)

    out = np.empty((n, CHEMISTRY_SIZE), dtype=np.float64)
    out[:, CHEM_SLOT_ENERGY_STORE] = energy_store
    out[:, CHEM_SLOT_REPAIR_RESERVE] = repair
    out[:, CHEM_SLOT_ACUTE_DAMAGE] = acute
    out[:, CHEM_SLOT_CHRONIC_DAMAGE] = chronic
    out[:, CHEM_SLOT_REGULATOR_1] = reg1
    out[:, CHEM_SLOT_REGULATOR_2] = reg2
    return out.astype(np.float32)


class TickChemistryMixin:
    def _update_chemistry_state_resident_arrays(self, alive, tick_index: int, buckets: Optional[Dict[str, float]] = None) -> bool:
        n = int(len(alive))
        if n <= 0:
            return True
        if not hasattr(self, "_ensure_chemistry_arrays"):
            return False
        if hasattr(self, "_ensure_drive_arrays"):
            self._ensure_drive_arrays()
        self._ensure_chemistry_arrays()
        if getattr(self, "_chemistry_state", None) is None or getattr(self, "_chemistry_slot_token", None) is None:
            return False

        _budget = nullcontext()
        rm = getattr(self, "rng_manager", None)
        if rm is not None:
            expected = rm.expected_delta_for_boundary(
                "chemistry.tick.update",
                ctx={"alive_count": int(n), "chemistry_draws": int(2 * n)},
            )
            _budget = rm.budget_scope("chemistry.tick.update", expected_delta=expected)

        with _budget:
            noise_all = None
            if rm is not None:
                noise_all = rm.policy_np.normal(0.0, 1.0, size=(n, 2)).astype(np.float32)
            if noise_all is None:
                raise RuntimeError(
                    "[DETERMINISM_VIOLATION]\n"
                    "code=DET_FORBIDDEN_RNG_API\n"
                    "context=missing_rng_manager_for_chemistry_tick_update\n"
                )

            valid_idx = []
            valid_slots = []
            for i, a in enumerate(alive):
                if not bool(getattr(a, "alive", False)):
                    continue
                slot = int(getattr(a, "slot", -1))
                if slot < 0:
                    return False
                if not self._sync_chemistry_slot(a):
                    return False
                valid_idx.append(int(i))
                valid_slots.append(int(slot))

            m = int(len(valid_slots))
            if m <= 0:
                return True

            slots_np = np.asarray(valid_slots, dtype=np.int32)
            idx_np = np.asarray(valid_idx, dtype=np.int32)
            need = int(np.max(slots_np)) + 1
            self._ensure_chemistry_arrays(required_slots=need)
            if hasattr(self, "_ensure_drive_arrays"):
                self._ensure_drive_arrays(required_slots=need)

            energy = np.empty((m,), dtype=np.float64)
            max_energy = np.empty((m,), dtype=np.float64)
            health = np.empty((m,), dtype=np.float64)
            start_health = np.empty((m,), dtype=np.float64)
            injury = np.empty((m,), dtype=np.float64)
            drive_pain = np.zeros((m,), dtype=np.float64)

            drive_tokens = getattr(self, "_drive_slot_token", None)
            drive_pain_arr = getattr(self, "_drive_pain", None)
            sync_drive_slot = getattr(self, "_sync_drive_slot", None)
            for j, i in enumerate(valid_idx):
                a = alive[i]
                slot = int(valid_slots[j])
                energy[j] = float(getattr(a, "energy", 0.0))
                max_energy[j] = float(getattr(a, "max_energy_norm", 1.0)) or 1.0
                health[j] = float(getattr(a, "health", 0.0))
                start_health[j] = float(getattr(a, "start_health_norm", 1.0)) or 1.0
                injury[j] = float(getattr(a, "injury", 0.0))
                token = int(getattr(a, "slot_token", -1))
                if drive_tokens is not None and drive_pain_arr is not None and slot < int(drive_tokens.shape[0]):
                    if int(drive_tokens[slot]) != token and callable(sync_drive_slot):
                        sync_drive_slot(a)
                    if int(drive_tokens[slot]) == token:
                        drive_pain[j] = float(drive_pain_arr[slot])
                        continue
                d = getattr(a, "drives", None)
                if d is not None:
                    try:
                        drive_pain[j] = float(np.asarray(d, dtype=np.float32).reshape(-1)[1])
                    except Exception:
                        drive_pain[j] = 0.0

            updated = chemistry_vector_step(
                self._chemistry_state[slots_np],
                energy,
                max_energy,
                health,
                start_health,
                injury,
                drive_pain,
                noise_all[idx_np],
            )
            self._chemistry_state[slots_np, :] = updated
            tokens = getattr(self, "_chemistry_slot_token", None)
            if tokens is not None:
                for j, i in enumerate(valid_idx):
                    tokens[int(valid_slots[j])] = np.int32(int(getattr(alive[i], "slot_token", 0)))
                    self._mirror_chemistry_slot_to_agent(alive[i])
        return True

    def _update_chemistry_state(self, alive, tick_index: int, buckets: Optional[Dict[str, float]] = None) -> None:
        """Deterministic chemistry update in fixed slot order.

        RNG schedule:
        - Exactly 2 NumPy draws per alive agent (regulator noise channels).
        - Draw budget asserted at boundary: chemistry.tick.update.
        """
        n = int(len(alive))
        if n <= 0:
            return
        sim_cfg = getattr(self, "sim_cfg", None)
        if bool(getattr(sim_cfg, "cxxrewrite_resident_generation_chemistry_arrays_enable", False)):
            try:
                if self._update_chemistry_state_resident_arrays(alive, tick_index=tick_index, buckets=buckets):
                    return
            except Exception:
                if bool(getattr(sim_cfg, "cxxrewrite_resident_generation_runner_fail_closed", False)):
                    raise
            if bool(getattr(sim_cfg, "cxxrewrite_resident_generation_runner_fail_closed", False)):
                raise RuntimeError(
                    "resident chemistry arrays were requested but the chemistry tick could not use them"
                )

        _budget = nullcontext()
        rm = getattr(self, "rng_manager", None)
        if rm is not None:
            expected = rm.expected_delta_for_boundary(
                "chemistry.tick.update",
                ctx={"alive_count": int(n), "chemistry_draws": int(2 * n)},
            )
            _budget = rm.budget_scope("chemistry.tick.update", expected_delta=expected)

        with _budget:
            noise = None
            if rm is not None:
                noise = rm.policy_np.normal(0.0, 1.0, size=(n, 2)).astype(np.float32)
            if noise is None:
                raise RuntimeError(
                    "[DETERMINISM_VIOLATION]\n"
                    "code=DET_FORBIDDEN_RNG_API\n"
                    "context=missing_rng_manager_for_chemistry_tick_update\n"
                )

            for i, a in enumerate(alive):
                if not bool(getattr(a, "alive", False)):
                    continue

                c_raw = getattr(a, "chemistry", None)
                if c_raw is None:
                    c = np.zeros((CHEMISTRY_SIZE,), dtype=np.float32)
                else:
                    c = chemistry_from_values(c_raw)

                e = float(getattr(a, "energy", 0.0))
                emax = float(getattr(a, "max_energy_norm", 1.0)) or 1.0
                h = float(getattr(a, "health", 0.0))
                hmax = float(getattr(a, "start_health_norm", 1.0)) or 1.0
                inj = float(getattr(a, "injury", 0.0))

                d = getattr(a, "drives", None)
                drive_pain = 0.0
                if d is not None:
                    try:
                        drive_pain = float(np.asarray(d, dtype=np.float32).reshape(-1)[1])
                    except Exception:
                        drive_pain = 0.0

                energy_n = clamp01(e / emax)
                health_n = clamp01(h / hmax)
                injury_n = clamp01(inj / hmax)
                stress = clamp01(0.55 * injury_n + 0.30 * drive_pain + 0.15 * (1.0 - energy_n))

                energy_store = float(c[CHEM_SLOT_ENERGY_STORE])
                repair = float(c[CHEM_SLOT_REPAIR_RESERVE])
                acute = float(c[CHEM_SLOT_ACUTE_DAMAGE])
                chronic = float(c[CHEM_SLOT_CHRONIC_DAMAGE])
                reg1 = float(c[CHEM_SLOT_REGULATOR_1])
                reg2 = float(c[CHEM_SLOT_REGULATOR_2])

                # 1) Acute (fast channel)
                acute_target = clamp01(1.25 * injury_n + 0.45 * stress + 0.20 * (1.0 - health_n))
                acute = clamp01(acute + 0.50 * (acute_target - acute) + 0.03 * reg1)

                # 2) Chronic (slow channel)
                chronic_target = clamp01(0.75 * chronic + 0.25 * acute)
                chronic = clamp01(chronic + 0.07 * (chronic_target - chronic))

                # 3) Energy / repair pools
                energy_store = clamp01(energy_store + 0.03 * energy_n - 0.02 * stress - 0.01 * acute)
                repair = clamp01(repair + 0.04 * energy_n - 0.03 * acute - 0.01 * chronic)

                # 4) Regulators (always consume exactly two RNG draws per agent).
                n1 = float(noise[i, 0])
                n2 = float(noise[i, 1])
                reg1 = clamp01(0.92 * reg1 + 0.08 * (1.0 - acute) + 0.01 * n1)
                reg2 = clamp01(0.97 * reg2 + 0.03 * chronic + 0.01 * n2)

                # 5) Repair coupling (bounded and deterministic).
                # Keep acute/chronic channels observable under realistic stress.
                # The previous coefficients could deterministically zero-out acute every tick
                # when energy/repair were moderate, collapsing intent signal to flat lines.
                acute_resolve = min(acute, 0.10 * repair + 0.04 * energy_store)
                acute = clamp01(acute - acute_resolve)
                chronic = clamp01(chronic + 0.25 * acute - 0.002 * repair)

                a.chemistry = chemistry_from_values(
                    (
                        energy_store,
                        repair,
                        acute,
                        chronic,
                        reg1,
                        reg2,
                    )
                )
