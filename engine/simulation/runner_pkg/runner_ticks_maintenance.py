from __future__ import annotations

"""simulation/runner_pkg/runner_ticks_maintenance.py

Phase A2 deterministic maintenance process.

Boundary labels:
- maintenance.tick.schedule
- maintenance.tick.apply_delayed
"""

from contextlib import nullcontext
from typing import Dict, Optional

import numpy as np

from agents.chemistry import (
    CHEM_SLOT_ACUTE_DAMAGE,
    CHEM_SLOT_CHRONIC_DAMAGE,
    CHEM_SLOT_REPAIR_RESERVE,
    chemistry_from_values,
)
from agents.maintenance import (
    MAINT_SLOT_DUE_TICK,
    MAINT_SLOT_LAST_SCHEDULE_TICK,
    MAINT_SLOT_PENDING_STRENGTH,
    MAINT_SLOT_RECENT_LOAD,
    maintenance_from_values,
    zeros_maintenance_state,
)
from env.gridworld_pkg.constants import FOOD
from perf.cxxrewrite4_upstream_resident import (
    resident_scalar_sync_fail_closed as cxxrewrite4_resident_scalar_sync_fail_closed,
    sync_agents_scalar_state as cxxrewrite4_sync_agents_scalar_state,
)
from .runner_ticks_common import ACT_REST, clamp01


class TickMaintenanceMixin:
    def _maintenance_opportunity_score(self, agent_id: int) -> float:
        """Deterministic local opportunity proxy from env food neighborhood."""
        env = getattr(self, "env", None)
        if env is None:
            return 0.0
        grid = getattr(env, "grid", None)
        ys = getattr(env, "agent_pos_y", None)
        xs = getattr(env, "agent_pos_x", None)
        if grid is None or ys is None or xs is None:
            return 0.0
        try:
            aid = int(agent_id)
            y = int(ys[aid])
            x = int(xs[aid])
            h, w = int(grid.shape[0]), int(grid.shape[1])
            if y < 0 or x < 0 or y >= h or x >= w:
                return 0.0
            y0 = max(0, y - 1)
            y1 = min(h, y + 2)
            x0 = max(0, x - 1)
            x1 = min(w, x + 2)
            patch = np.asarray(grid[y0:y1, x0:x1])
            if patch.size <= 0:
                return 0.0
            return float(np.count_nonzero(patch == FOOD)) / float(patch.size)
        except Exception:
            return 0.0

    def _update_maintenance_state_resident_arrays(
        self,
        alive,
        actions_np,
        tick_index: int,
        buckets: Optional[Dict[str, float]] = None,
    ) -> bool:
        n = int(len(alive))
        if n <= 0:
            return True
        if not hasattr(self, "_ensure_maintenance_arrays"):
            return False

        self._ensure_maintenance_arrays()
        if getattr(self, "_maintenance_state", None) is None or getattr(self, "_maintenance_slot_token", None) is None:
            return False

        valid: list[tuple[int, object, int]] = []
        max_slot = -1
        for i, a in enumerate(alive):
            if not bool(getattr(a, "alive", False)):
                continue
            slot = int(getattr(a, "slot", -1))
            if slot < 0:
                return False
            max_slot = max(max_slot, slot)
            valid.append((int(i), a, slot))

        if not valid:
            return True
        if max_slot >= int(self._maintenance_slot_token.shape[0]):
            self._ensure_maintenance_arrays(required_slots=max_slot + 1)
            if getattr(self, "_maintenance_state", None) is None or getattr(self, "_maintenance_slot_token", None) is None:
                return False

        for _, a, _ in valid:
            if not self._sync_maintenance_slot(a):
                return False

        delay_ticks = max(0, int(getattr(self.sim_cfg, "maintenance_delay_ticks", 0)))
        base_effect = clamp01(float(getattr(self.sim_cfg, "maintenance_base_effect", 0.0)))
        diminishing = max(0.0, float(getattr(self.sim_cfg, "maintenance_diminishing_factor", 0.0)))
        decay = clamp01(float(getattr(self.sim_cfg, "maintenance_recent_decay", 1.0)))
        energy_cost = max(0.0, float(getattr(self.sim_cfg, "maintenance_energy_cost", 0.0)))
        opp_coeff = max(0.0, float(getattr(self.sim_cfg, "maintenance_opportunity_cost_coeff", 0.0)))

        rm = getattr(self, "rng_manager", None)

        schedule_requests = 0
        schedule_energy_cost_sum = 0.0
        schedule_opp_cost_sum = 0.0
        mutated_agents = []

        schedule_budget = nullcontext()
        if rm is not None:
            expected = rm.expected_delta_for_boundary(
                "maintenance.tick.schedule",
                ctx={"alive_count": int(n), "maintenance_requests": int(0)},
            )
            schedule_budget = rm.budget_scope("maintenance.tick.schedule", expected_delta=expected)

        with schedule_budget:
            state = self._maintenance_state
            for i, a, slot in valid:
                m = state[slot]
                m[MAINT_SLOT_RECENT_LOAD] = np.float32(clamp01(float(m[MAINT_SLOT_RECENT_LOAD]) * decay))
                act = int(actions_np[i])
                if act == ACT_REST:
                    schedule_requests += 1
                    opportunity = self._maintenance_opportunity_score(int(getattr(a, "agent_id", -1)))
                    opp_cost = opp_coeff * opportunity
                    total_cost = energy_cost + opp_cost
                    if total_cost > 0.0:
                        a.energy = max(0.0, float(getattr(a, "energy", 0.0)) - float(total_cost))
                        mutated_agents.append(a)
                        schedule_energy_cost_sum += float(total_cost)
                        schedule_opp_cost_sum += float(opp_cost)
                    m[MAINT_SLOT_PENDING_STRENGTH] = np.float32(
                        clamp01(float(m[MAINT_SLOT_PENDING_STRENGTH]) + base_effect)
                    )
                    m[MAINT_SLOT_DUE_TICK] = np.float32(float(int(tick_index) + delay_ticks))
                    m[MAINT_SLOT_LAST_SCHEDULE_TICK] = np.float32(float(int(tick_index)))

                    st = getattr(a, "stats", None)
                    if st is not None:
                        st.maintenance_requests = int(getattr(st, "maintenance_requests", 0)) + 1
                        st.maintenance_energy_spent_total = float(getattr(st, "maintenance_energy_spent_total", 0.0)) + float(total_cost)
                        st.maintenance_opportunity_cost_total = float(
                            getattr(st, "maintenance_opportunity_cost_total", 0.0)
                        ) + float(opp_cost)

        apply_count = 0
        applied_effect_sum = 0.0
        apply_budget = nullcontext()
        if rm is not None:
            expected = rm.expected_delta_for_boundary(
                "maintenance.tick.apply_delayed",
                ctx={"alive_count": int(n), "maintenance_applications": int(0)},
            )
            apply_budget = rm.budget_scope("maintenance.tick.apply_delayed", expected_delta=expected)

        with apply_budget:
            state = self._maintenance_state
            mirror_maintenance_slot = getattr(self, "_mirror_maintenance_slot_to_agent", None)
            for _, a, slot in valid:
                m = state[slot]
                pending = float(m[MAINT_SLOT_PENDING_STRENGTH])
                due_tick = int(round(float(m[MAINT_SLOT_DUE_TICK])))
                if pending > 0.0 and due_tick >= 0 and int(tick_index) >= due_tick:
                    recent_load = float(m[MAINT_SLOT_RECENT_LOAD])
                    eff_mult = 1.0 / (1.0 + diminishing * recent_load)
                    applied = clamp01(pending * eff_mult)

                    c_raw = getattr(a, "chemistry", None)
                    c = chemistry_from_values(c_raw) if c_raw is not None else chemistry_from_values((0.0,) * 6)
                    c[CHEM_SLOT_REPAIR_RESERVE] = np.float32(clamp01(float(c[CHEM_SLOT_REPAIR_RESERVE]) + 0.65 * applied))
                    c[CHEM_SLOT_ACUTE_DAMAGE] = np.float32(clamp01(float(c[CHEM_SLOT_ACUTE_DAMAGE]) - 0.55 * applied))
                    c[CHEM_SLOT_CHRONIC_DAMAGE] = np.float32(clamp01(float(c[CHEM_SLOT_CHRONIC_DAMAGE]) - 0.25 * applied))
                    a.chemistry = chemistry_from_values(c)
                    sync_chemistry_slot = getattr(self, "_sync_chemistry_slot", None)
                    if callable(sync_chemistry_slot):
                        try:
                            sync_chemistry_slot(a, force_from_agent=True)
                        except Exception:
                            pass

                    m[MAINT_SLOT_PENDING_STRENGTH] = np.float32(0.0)
                    m[MAINT_SLOT_DUE_TICK] = np.float32(-1.0)
                    m[MAINT_SLOT_RECENT_LOAD] = np.float32(clamp01(recent_load + applied))
                    apply_count += 1
                    applied_effect_sum += float(applied)

                    st = getattr(a, "stats", None)
                    if st is not None:
                        st.maintenance_applied = int(getattr(st, "maintenance_applied", 0)) + 1
                        st.maintenance_applied_effect_total = float(
                            getattr(st, "maintenance_applied_effect_total", 0.0)
                        ) + float(applied)

                if callable(mirror_maintenance_slot):
                    mirror_maintenance_slot(a)

        self._maintenance_tick_count_gen = int(getattr(self, "_maintenance_tick_count_gen", 0)) + 1
        self._maintenance_action_opportunities_gen = int(getattr(self, "_maintenance_action_opportunities_gen", 0)) + int(n)
        self._maintenance_schedule_requests_gen = int(getattr(self, "_maintenance_schedule_requests_gen", 0)) + int(schedule_requests)
        self._maintenance_apply_count_gen = int(getattr(self, "_maintenance_apply_count_gen", 0)) + int(apply_count)
        self._maintenance_energy_cost_total_gen = float(
            getattr(self, "_maintenance_energy_cost_total_gen", 0.0)
        ) + float(schedule_energy_cost_sum)
        self._maintenance_opportunity_cost_total_gen = float(
            getattr(self, "_maintenance_opportunity_cost_total_gen", 0.0)
        ) + float(schedule_opp_cost_sum)
        self._maintenance_effect_total_gen = float(getattr(self, "_maintenance_effect_total_gen", 0.0)) + float(applied_effect_sum)

        opportunities = float(max(1, int(getattr(self, "_maintenance_action_opportunities_gen", 0))))
        ratio = float(getattr(self, "_maintenance_schedule_requests_gen", 0)) / opportunities
        threshold = float(getattr(self.sim_cfg, "maintenance_domination_threshold", 0.75))
        min_ticks = int(getattr(self.sim_cfg, "maintenance_domination_min_ticks", 8))
        triggered_now = int(getattr(self, "_maintenance_tick_count_gen", 0)) >= min_ticks and ratio >= threshold
        prev = dict(getattr(self, "_maintenance_domination_annotation", {}) or {})
        was_triggered = bool(prev.get("triggered", False))
        self._maintenance_domination_annotation = {
            "triggered": bool(was_triggered or triggered_now),
            "tick": int(tick_index),
            "ratio": float(ratio),
            "threshold": float(threshold),
            "requests": int(getattr(self, "_maintenance_schedule_requests_gen", 0)),
            "opportunities": int(getattr(self, "_maintenance_action_opportunities_gen", 0)),
        }

        if isinstance(buckets, dict):
            buckets["maintenance_schedule_requests"] = buckets.get("maintenance_schedule_requests", 0.0) + float(schedule_requests)
            buckets["maintenance_apply_count"] = buckets.get("maintenance_apply_count", 0.0) + float(apply_count)

        try:
            cxxrewrite4_sync_agents_scalar_state(self, mutated_agents, stage="maintenance_energy")
        except Exception:
            if cxxrewrite4_resident_scalar_sync_fail_closed(self.sim_cfg):
                raise
        return True

    def _update_maintenance_state(self, alive, actions_np, tick_index: int, buckets: Optional[Dict[str, float]] = None) -> None:
        if not bool(getattr(self.sim_cfg, "maintenance_enable", False)):
            return
        n = int(len(alive))
        if n <= 0:
            return
        sim_cfg = getattr(self, "sim_cfg", None)
        if bool(getattr(sim_cfg, "cxxrewrite_resident_generation_maintenance_arrays_enable", False)):
            try:
                if self._update_maintenance_state_resident_arrays(
                    alive,
                    actions_np,
                    tick_index=tick_index,
                    buckets=buckets,
                ):
                    return
            except Exception:
                if bool(getattr(sim_cfg, "cxxrewrite_resident_generation_runner_fail_closed", False)):
                    raise
            if bool(getattr(sim_cfg, "cxxrewrite_resident_generation_runner_fail_closed", False)):
                raise RuntimeError(
                    "resident maintenance arrays were requested but the maintenance tick could not use them"
                )

        delay_ticks = max(0, int(getattr(self.sim_cfg, "maintenance_delay_ticks", 0)))
        base_effect = clamp01(float(getattr(self.sim_cfg, "maintenance_base_effect", 0.0)))
        diminishing = max(0.0, float(getattr(self.sim_cfg, "maintenance_diminishing_factor", 0.0)))
        decay = clamp01(float(getattr(self.sim_cfg, "maintenance_recent_decay", 1.0)))
        energy_cost = max(0.0, float(getattr(self.sim_cfg, "maintenance_energy_cost", 0.0)))
        opp_coeff = max(0.0, float(getattr(self.sim_cfg, "maintenance_opportunity_cost_coeff", 0.0)))

        rm = getattr(self, "rng_manager", None)

        schedule_requests = 0
        schedule_energy_cost_sum = 0.0
        schedule_opp_cost_sum = 0.0
        mutated_agents = []

        schedule_budget = nullcontext()
        if rm is not None:
            expected = rm.expected_delta_for_boundary(
                "maintenance.tick.schedule",
                ctx={"alive_count": int(n), "maintenance_requests": int(0)},
            )
            schedule_budget = rm.budget_scope("maintenance.tick.schedule", expected_delta=expected)

        with schedule_budget:
            for i, a in enumerate(alive):
                if not bool(getattr(a, "alive", False)):
                    continue

                m_raw = getattr(a, "maintenance", None)
                if m_raw is None:
                    m = zeros_maintenance_state()
                else:
                    m = maintenance_from_values(m_raw)

                m[MAINT_SLOT_RECENT_LOAD] = np.float32(clamp01(float(m[MAINT_SLOT_RECENT_LOAD]) * decay))
                act = int(actions_np[i])
                if act == ACT_REST:
                    schedule_requests += 1
                    opportunity = self._maintenance_opportunity_score(int(getattr(a, "agent_id", -1)))
                    opp_cost = opp_coeff * opportunity
                    total_cost = energy_cost + opp_cost
                    if total_cost > 0.0:
                        a.energy = max(0.0, float(getattr(a, "energy", 0.0)) - float(total_cost))
                        mutated_agents.append(a)
                        schedule_energy_cost_sum += float(total_cost)
                        schedule_opp_cost_sum += float(opp_cost)
                    m[MAINT_SLOT_PENDING_STRENGTH] = np.float32(
                        clamp01(float(m[MAINT_SLOT_PENDING_STRENGTH]) + base_effect)
                    )
                    m[MAINT_SLOT_DUE_TICK] = np.float32(float(int(tick_index) + delay_ticks))
                    m[MAINT_SLOT_LAST_SCHEDULE_TICK] = np.float32(float(int(tick_index)))

                    st = getattr(a, "stats", None)
                    if st is not None:
                        st.maintenance_requests = int(getattr(st, "maintenance_requests", 0)) + 1
                        st.maintenance_energy_spent_total = float(getattr(st, "maintenance_energy_spent_total", 0.0)) + float(total_cost)
                        st.maintenance_opportunity_cost_total = float(
                            getattr(st, "maintenance_opportunity_cost_total", 0.0)
                        ) + float(opp_cost)

                a.maintenance = maintenance_from_values(m)

        apply_count = 0
        applied_effect_sum = 0.0
        apply_budget = nullcontext()
        if rm is not None:
            expected = rm.expected_delta_for_boundary(
                "maintenance.tick.apply_delayed",
                ctx={"alive_count": int(n), "maintenance_applications": int(0)},
            )
            apply_budget = rm.budget_scope("maintenance.tick.apply_delayed", expected_delta=expected)

        with apply_budget:
            for a in alive:
                if not bool(getattr(a, "alive", False)):
                    continue

                m_raw = getattr(a, "maintenance", None)
                if m_raw is None:
                    m = zeros_maintenance_state()
                else:
                    m = maintenance_from_values(m_raw)

                pending = float(m[MAINT_SLOT_PENDING_STRENGTH])
                due_tick = int(round(float(m[MAINT_SLOT_DUE_TICK])))
                if pending > 0.0 and due_tick >= 0 and int(tick_index) >= due_tick:
                    recent_load = float(m[MAINT_SLOT_RECENT_LOAD])
                    eff_mult = 1.0 / (1.0 + diminishing * recent_load)
                    applied = clamp01(pending * eff_mult)

                    c_raw = getattr(a, "chemistry", None)
                    c = chemistry_from_values(c_raw) if c_raw is not None else chemistry_from_values((0.0,) * 6)
                    c[CHEM_SLOT_REPAIR_RESERVE] = np.float32(clamp01(float(c[CHEM_SLOT_REPAIR_RESERVE]) + 0.65 * applied))
                    c[CHEM_SLOT_ACUTE_DAMAGE] = np.float32(clamp01(float(c[CHEM_SLOT_ACUTE_DAMAGE]) - 0.55 * applied))
                    c[CHEM_SLOT_CHRONIC_DAMAGE] = np.float32(clamp01(float(c[CHEM_SLOT_CHRONIC_DAMAGE]) - 0.25 * applied))
                    a.chemistry = chemistry_from_values(c)
                    sync_chemistry_slot = getattr(self, "_sync_chemistry_slot", None)
                    if callable(sync_chemistry_slot):
                        try:
                            sync_chemistry_slot(a, force_from_agent=True)
                        except Exception:
                            pass

                    m[MAINT_SLOT_PENDING_STRENGTH] = np.float32(0.0)
                    m[MAINT_SLOT_DUE_TICK] = np.float32(-1.0)
                    m[MAINT_SLOT_RECENT_LOAD] = np.float32(clamp01(recent_load + applied))
                    apply_count += 1
                    applied_effect_sum += float(applied)

                    st = getattr(a, "stats", None)
                    if st is not None:
                        st.maintenance_applied = int(getattr(st, "maintenance_applied", 0)) + 1
                        st.maintenance_applied_effect_total = float(
                            getattr(st, "maintenance_applied_effect_total", 0.0)
                        ) + float(applied)

                a.maintenance = maintenance_from_values(m)

        self._maintenance_tick_count_gen = int(getattr(self, "_maintenance_tick_count_gen", 0)) + 1
        self._maintenance_action_opportunities_gen = int(getattr(self, "_maintenance_action_opportunities_gen", 0)) + int(n)
        self._maintenance_schedule_requests_gen = int(getattr(self, "_maintenance_schedule_requests_gen", 0)) + int(schedule_requests)
        self._maintenance_apply_count_gen = int(getattr(self, "_maintenance_apply_count_gen", 0)) + int(apply_count)
        self._maintenance_energy_cost_total_gen = float(
            getattr(self, "_maintenance_energy_cost_total_gen", 0.0)
        ) + float(schedule_energy_cost_sum)
        self._maintenance_opportunity_cost_total_gen = float(
            getattr(self, "_maintenance_opportunity_cost_total_gen", 0.0)
        ) + float(schedule_opp_cost_sum)
        self._maintenance_effect_total_gen = float(getattr(self, "_maintenance_effect_total_gen", 0.0)) + float(applied_effect_sum)

        opportunities = float(max(1, int(getattr(self, "_maintenance_action_opportunities_gen", 0))))
        ratio = float(getattr(self, "_maintenance_schedule_requests_gen", 0)) / opportunities
        threshold = float(getattr(self.sim_cfg, "maintenance_domination_threshold", 0.75))
        min_ticks = int(getattr(self.sim_cfg, "maintenance_domination_min_ticks", 8))
        triggered_now = int(getattr(self, "_maintenance_tick_count_gen", 0)) >= min_ticks and ratio >= threshold
        prev = dict(getattr(self, "_maintenance_domination_annotation", {}) or {})
        was_triggered = bool(prev.get("triggered", False))
        self._maintenance_domination_annotation = {
            "triggered": bool(was_triggered or triggered_now),
            "tick": int(tick_index),
            "ratio": float(ratio),
            "threshold": float(threshold),
            "requests": int(getattr(self, "_maintenance_schedule_requests_gen", 0)),
            "opportunities": int(getattr(self, "_maintenance_action_opportunities_gen", 0)),
        }

        if isinstance(buckets, dict):
            buckets["maintenance_schedule_requests"] = buckets.get("maintenance_schedule_requests", 0.0) + float(schedule_requests)
            buckets["maintenance_apply_count"] = buckets.get("maintenance_apply_count", 0.0) + float(apply_count)

        try:
            cxxrewrite4_sync_agents_scalar_state(self, mutated_agents, stage="maintenance_energy")
        except Exception:
            if cxxrewrite4_resident_scalar_sync_fail_closed(self.sim_cfg):
                raise
