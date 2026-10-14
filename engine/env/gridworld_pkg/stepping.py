from __future__ import annotations

"""
env/gridworld_pkg/stepping.py

Numba-accelerated stepping kernels.

Public API unchanged: step_agents_batch(...)

Performance notes (why this file exists):
- Keep Python allocations out of the per-tick hot path.
- Avoid per-tick construction of Numba typed containers (very slow).
- Normalize dtypes once per call and reuse.
- Reuse scratch buffers on `self` where safe.

This file intentionally keeps behavior identical to the pre-numba version,
but with less overhead per tick.
"""

from typing import Tuple
import numpy as np

from env.gridworld_pkg.constants import FOOD, EMPTY, ACT_ATTACK, ACT_REST, ACT_GATHER
from env.gridworld_pkg.state import BIOME_WILD

try:
    from numba import njit
    _NUMBA_OK = True
except Exception:  # pragma: no cover
    _NUMBA_OK = False
    njit = None


if _NUMBA_OK:
    @njit(cache=True)
    def _sorted_agent_index(sorted_agent_ids, target_id):
        lo = 0
        hi = sorted_agent_ids.shape[0]
        while lo < hi:
            mid = lo + (hi - lo) // 2
            value = int(sorted_agent_ids[mid])
            if value < target_id:
                lo = mid + 1
            else:
                hi = mid
        if lo < sorted_agent_ids.shape[0] and int(sorted_agent_ids[lo]) == target_id:
            return lo
        return -1

    @njit(cache=True)
    def _combat_target_index(agent_grid, slot_to_agent_id, sorted_agent_ids, y, x):
        slot_handle = int(agent_grid[y, x])
        if slot_handle < 0 or slot_handle >= slot_to_agent_id.shape[0]:
            return -1
        target_id = int(slot_to_agent_id[slot_handle])
        if target_id < 0:
            return -1
        return _sorted_agent_index(sorted_agent_ids, target_id)

    @njit(cache=True)
    def _apply_costs(actions, costs, move_cost_mults, move_cost, gather_cost, rest_cost, attack_cost, action_cost_mult, rest_cost_mult):
        n = actions.shape[0]
        for i in range(n):
            a = int(actions[i])
            c = move_cost * float(move_cost_mults[i])
            if a == ACT_GATHER:
                c = gather_cost
            elif a == ACT_REST:
                c = rest_cost
            elif a == ACT_ATTACK:
                c = attack_cost
            if action_cost_mult != 1.0:
                c *= action_cost_mult
            if rest_cost_mult != 1.0 and a == ACT_REST:
                c *= rest_cost_mult
            costs[i] = c

    @njit(cache=True)
    def _apply_biome_cost_mult(costs, ys, xs, biome_grid, enable_biomes, wild_mult):
        if enable_biomes == 0 or wild_mult == 1.0:
            return
        n = costs.shape[0]
        for i in range(n):
            if biome_grid[int(ys[i]), int(xs[i])] == BIOME_WILD:
                costs[i] *= wild_mult

    @njit(cache=True)
    def _apply_injury_cost_mult(costs, injuries, scars, enable_injury, enable_scar,
                               injury_action_cost_penalty, injury_max, scar_max, scar_cost_weight, penalty_curve_power):
        if enable_injury == 0 or injury_action_cost_penalty <= 0.0:
            return
        if injury_max <= 0.0:
            injury_max = 1.0
        if penalty_curve_power <= 0.0:
            penalty_curve_power = 1.0

        n = costs.shape[0]
        amp = injury_action_cost_penalty * injury_max
        for i in range(n):
            inj = float(injuries[i])
            if inj < 0.0:
                inj = 0.0
            elif inj > injury_max:
                inj = injury_max

            if enable_scar != 0:
                sc = float(scars[i])
                if sc < 0.0:
                    sc = 0.0
                elif sc > scar_max:
                    sc = scar_max
                impairment = inj + scar_cost_weight * sc
                x = impairment / injury_max
            else:
                x = inj / injury_max

            if x < 0.0:
                x = 0.0
            elif x > 1.0:
                x = 1.0

            curve = x ** penalty_curve_power
            costs[i] *= (1.0 + amp * curve)

    @njit(cache=True)
    def _movement_kernel(world_slots, actions, ys, xs, agent_grid, height, width, moved_mask):
        n = world_slots.shape[0]
        for i in range(n):
            a = int(actions[i])
            # 0..3 are movement directions in this codebase
            if a < 0 or a > 3:
                continue

            slot_handle = int(world_slots[i])
            y = int(ys[i])
            x = int(xs[i])

            ny = y
            nx = x
            if a == 0:
                ny -= 1
            elif a == 1:
                ny += 1
            elif a == 2:
                nx += 1
            else:
                nx -= 1

            if ny < 0 or ny >= height or nx < 0 or nx >= width:
                continue
            if agent_grid[ny, nx] != -1:
                continue

            if agent_grid[y, x] == slot_handle:
                agent_grid[y, x] = -1
            agent_grid[ny, nx] = slot_handle
            ys[i] = ny
            xs[i] = nx
            moved_mask[i] = 1

    @njit(cache=True)
    def _gather_kernel(actions, ys, xs, grid, food_gained, food_energy):
        n = actions.shape[0]
        any_gain = 0
        for i in range(n):
            if int(actions[i]) != ACT_GATHER:
                continue
            y = int(ys[i])
            x = int(xs[i])
            if grid[y, x] == FOOD:
                grid[y, x] = EMPTY
                food_gained[i] = food_energy
                any_gain = 1
        return any_gain

    @njit(cache=True)
    def _patch_gather_kernel(
        actions,
        ys,
        xs,
        grid,
        patch_food_amount,
        scent_pending,
        waste_pending,
        food_gained,
        food_energy,
        gather_amount,
        threshold,
        scent_emit,
        waste_emit,
    ):
        n = actions.shape[0]
        any_gain = 0
        harvest_total = 0.0
        scent_total = 0.0
        waste_total = 0.0
        if gather_amount <= 0.0:
            gather_amount = 1.0
        for i in range(n):
            if int(actions[i]) != ACT_GATHER:
                continue
            y = int(ys[i])
            x = int(xs[i])
            amount = float(patch_food_amount[y, x])
            if amount <= threshold:
                grid[y, x] = EMPTY
                continue
            harvested = amount
            if harvested > gather_amount:
                harvested = gather_amount
            post = amount - harvested
            if post < 0.0:
                post = 0.0
            patch_food_amount[y, x] = post
            if post > threshold:
                grid[y, x] = FOOD
            else:
                grid[y, x] = EMPTY
            harvest_scale = harvested / gather_amount
            food_gained[i] = food_energy * harvest_scale
            if scent_emit > 0.0:
                scent_delta = scent_emit * harvest_scale
                scent_pending[y, x] += scent_delta
                scent_total += scent_delta
            if waste_emit > 0.0:
                waste_delta = waste_emit * harvest_scale
                waste_pending[y, x] += waste_delta
                waste_total += waste_delta
            harvest_total += harvested
            any_gain = 1
        return any_gain, harvest_total, scent_total, waste_total

    @njit(cache=True)
    def _hazard_kernel(ys, xs, biome_grid, hazard_map, tier3_hazard_mults, juvenile_hazard_mults, r, enable_biomes, p_h, dmg_h, enable_injury, inj_per, inj_max, new_health, new_injury):
        if enable_biomes == 0 or p_h <= 0.0 or dmg_h <= 0.0:
            return
        if inj_max <= 0.0:
            inj_max = 10.0

        n = ys.shape[0]
        for i in range(n):
            y = int(ys[i])
            x = int(xs[i])
            if biome_grid[y, x] != BIOME_WILD:
                continue
            # Tier 1 hazard semantics: multiplier on hazard probability/rate only.
            # Clamp only if the probability model requires it.
            p_eff = p_h * float(hazard_map[y, x]) * float(tier3_hazard_mults[i]) * float(juvenile_hazard_mults[i])
            if p_eff <= 0.0:
                continue
            if p_eff > 1.0:
                p_eff = 1.0
            if float(r[i]) >= p_eff:
                continue
            new_health[i] -= dmg_h
            if enable_injury != 0 and inj_per > 0.0:
                v = new_injury[i] + dmg_h * inj_per
                if v > inj_max:
                    v = inj_max
                new_injury[i] = v

    @njit(cache=True)
    def _combat_kernel(agent_ids, actions, ys, xs, agent_grid, slot_to_agent_id, sorted_agent_ids,
                       new_energy, new_health, new_injury, new_scar,
                       height, width, dmg, enable_injury, inj_per, inj_max,
                       enable_scar, scar_max, scar_thresh, scar_gain_per, scar_gain_base,
                       tape_target_id, tape_accepted, tape_reject_code, tape_attempted_damage,
                       tape_applied_damage, tape_target_health_before, tape_target_health_after):
        """Attack the weakest adjacent agent that is ALSO in the current batch.

        `sorted_agent_ids` maps biological IDs to their batch position using a
        bounded binary search. Its storage scales with resident batch size, not
        with the largest biological ID.
        """
        if dmg <= 0.0:
            return
        if inj_max <= 0.0:
            inj_max = 10.0
        if scar_max <= 0.0:
            scar_max = inj_max

        n = agent_ids.shape[0]
        for i in range(n):
            if int(actions[i]) != ACT_ATTACK:
                continue
            tape_attempted_damage[i] = dmg
            if new_energy[i] <= 0.0:
                tape_reject_code[i] = 2
                continue

            y = int(ys[i])
            x = int(xs[i])

            best_idx = -1
            best_h = 1e30

            # Check 4-neighborhood. We keep this explicit because numba likes it.
            if y > 0:
                t_idx = _combat_target_index(agent_grid, slot_to_agent_id, sorted_agent_ids, y - 1, x)
                if t_idx != -1 and t_idx != i and new_health[t_idx] > 0.0:
                    th = float(new_health[t_idx])
                    if th < best_h:
                        best_h = th
                        best_idx = t_idx

            if y + 1 < height:
                t_idx = _combat_target_index(agent_grid, slot_to_agent_id, sorted_agent_ids, y + 1, x)
                if t_idx != -1 and t_idx != i and new_health[t_idx] > 0.0:
                    th = float(new_health[t_idx])
                    if th < best_h:
                        best_h = th
                        best_idx = t_idx

            if x > 0:
                t_idx = _combat_target_index(agent_grid, slot_to_agent_id, sorted_agent_ids, y, x - 1)
                if t_idx != -1 and t_idx != i and new_health[t_idx] > 0.0:
                    th = float(new_health[t_idx])
                    if th < best_h:
                        best_h = th
                        best_idx = t_idx

            if x + 1 < width:
                t_idx = _combat_target_index(agent_grid, slot_to_agent_id, sorted_agent_ids, y, x + 1)
                if t_idx != -1 and t_idx != i and new_health[t_idx] > 0.0:
                    th = float(new_health[t_idx])
                    if th < best_h:
                        best_h = th
                        best_idx = t_idx

            if best_idx == -1:
                tape_reject_code[i] = 3
                continue

            target_id = int(agent_ids[best_idx])
            health_before = float(new_health[best_idx])
            new_health[best_idx] -= dmg
            health_after = float(new_health[best_idx])
            tape_target_id[i] = target_id
            tape_accepted[i] = 1
            tape_applied_damage[i] = dmg
            tape_target_health_before[i] = health_before
            tape_target_health_after[i] = health_after

            if enable_injury != 0 and inj_per > 0.0:
                inj_delta = dmg * inj_per
                if inj_delta > 0.0:
                    pre = float(new_injury[best_idx])
                    post = pre + inj_delta
                    if post > inj_max:
                        post = inj_max
                    if post < 0.0:
                        post = 0.0
                    new_injury[best_idx] = post

                    if enable_scar != 0:
                        add = scar_gain_base * inj_delta
                        if post >= scar_thresh:
                            add += scar_gain_per * inj_delta
                        if add > 0.0:
                            sc = float(new_scar[best_idx]) + add
                            if sc > scar_max:
                                sc = scar_max
                            if sc < 0.0:
                                sc = 0.0
                            new_scar[best_idx] = sc


class SteppingMixin:
    def _ensure_step_scratch(self, n: int) -> None:
        """Ensure reusable scratch buffers exist and are large enough.

        These buffers are internal-only and never returned to the caller.
        """
        # Position buffers (int32)
        if getattr(self, "_step_ys_buf", None) is None or self._step_ys_buf.size < n:
            self._step_ys_buf = np.empty((n,), dtype=np.int32)
            self._step_xs_buf = np.empty((n,), dtype=np.int32)

            # === Tier 3 change: scratch buffers ===
            self._step_prev_ys_buf = np.empty((n,), dtype=np.int32)
            self._step_prev_xs_buf = np.empty((n,), dtype=np.int32)
            self._step_tier3_hazard_mults_buf = np.ones((n,), dtype=np.float32)
            self._step_tier4_juvenile_hazard_mults_buf = np.ones((n,), dtype=np.float32)
            self._step_tier4_juvenile_harvest_mults_buf = np.ones((n,), dtype=np.float32)
            self._step_tier3_origin_regions_buf = np.zeros((n,), dtype=np.int32)
            self._step_tier3_dest_regions_buf = np.zeros((n,), dtype=np.int32)
            self._step_tier3_crossing_mask_buf = np.zeros((n,), dtype=np.uint8)
            # === Tier 3 change: end scratch buffers ===

        # Costs/flags/food
        if getattr(self, "_step_costs_buf", None) is None or self._step_costs_buf.size < n:
            self._step_costs_buf = np.empty((n,), dtype=np.float32)

        # Per-agent movement cost multipliers (default ones).
        if getattr(self, "_step_move_cost_mults_buf", None) is None or self._step_move_cost_mults_buf.size < n:
            self._step_move_cost_mults_buf = np.ones((n,), dtype=np.float32)
        else:
            self._step_move_cost_mults_buf[:n].fill(1.0)

        # === Tier 3 change: scratch buffers ===
        if getattr(self, "_step_prev_ys_buf", None) is None or self._step_prev_ys_buf.size < n:
            self._step_prev_ys_buf = np.empty((n,), dtype=np.int32)
            self._step_prev_xs_buf = np.empty((n,), dtype=np.int32)
        if getattr(self, "_step_tier3_hazard_mults_buf", None) is None or self._step_tier3_hazard_mults_buf.size < n:
            self._step_tier3_hazard_mults_buf = np.ones((n,), dtype=np.float32)
        else:
            self._step_tier3_hazard_mults_buf[:n].fill(1.0)
        if getattr(self, "_step_tier4_juvenile_hazard_mults_buf", None) is None or self._step_tier4_juvenile_hazard_mults_buf.size < n:
            self._step_tier4_juvenile_hazard_mults_buf = np.ones((n,), dtype=np.float32)
        else:
            self._step_tier4_juvenile_hazard_mults_buf[:n].fill(1.0)
        if getattr(self, "_step_tier4_juvenile_harvest_mults_buf", None) is None or self._step_tier4_juvenile_harvest_mults_buf.size < n:
            self._step_tier4_juvenile_harvest_mults_buf = np.ones((n,), dtype=np.float32)
        else:
            self._step_tier4_juvenile_harvest_mults_buf[:n].fill(1.0)
        if getattr(self, "_step_tier3_origin_regions_buf", None) is None or self._step_tier3_origin_regions_buf.size < n:
            self._step_tier3_origin_regions_buf = np.zeros((n,), dtype=np.int32)
            self._step_tier3_dest_regions_buf = np.zeros((n,), dtype=np.int32)
            self._step_tier3_crossing_mask_buf = np.zeros((n,), dtype=np.uint8)
        else:
            self._step_tier3_crossing_mask_buf[:n].fill(0)
        # === Tier 3 change: end scratch buffers ===
        if getattr(self, "_step_moved_buf", None) is None or self._step_moved_buf.size < n:
            self._step_moved_buf = np.zeros((n,), dtype=np.uint8)
        else:
            self._step_moved_buf[:n].fill(0)

        if getattr(self, "_step_food_buf", None) is None or self._step_food_buf.size < n:
            self._step_food_buf = np.zeros((n,), dtype=np.float32)
        else:
            self._step_food_buf[:n].fill(0.0)

        if getattr(self, "_step_attack_target_id_buf", None) is None or self._step_attack_target_id_buf.size < n:
            self._step_attack_target_id_buf = np.full((n,), -1, dtype=np.int32)
            self._step_attack_accepted_buf = np.zeros((n,), dtype=np.uint8)
            self._step_attack_reject_code_buf = np.zeros((n,), dtype=np.int16)
            self._step_attack_attempted_damage_buf = np.zeros((n,), dtype=np.float32)
            self._step_attack_applied_damage_buf = np.zeros((n,), dtype=np.float32)
            self._step_attack_target_health_before_buf = np.full((n,), np.nan, dtype=np.float32)
            self._step_attack_target_health_after_buf = np.full((n,), np.nan, dtype=np.float32)
        else:
            self._step_attack_target_id_buf[:n].fill(-1)
            self._step_attack_accepted_buf[:n].fill(0)
            self._step_attack_reject_code_buf[:n].fill(0)
            self._step_attack_attempted_damage_buf[:n].fill(0.0)
            self._step_attack_applied_damage_buf[:n].fill(0.0)
            self._step_attack_target_health_before_buf[:n].fill(np.nan)
            self._step_attack_target_health_after_buf[:n].fill(np.nan)

        # Hazard RNG buffer (only used when biomes enabled, but cheap to keep around)
        if getattr(self, "_step_rhaz_buf", None) is None or self._step_rhaz_buf.size < n:
            self._step_rhaz_buf = np.empty((n,), dtype=np.float32)

        # Scratch for "missing scars" case
        if getattr(self, "_step_zero_scars", None) is None or self._step_zero_scars.size < n:
            self._step_zero_scars = np.zeros((n,), dtype=np.float32)
        else:
            self._step_zero_scars[:n].fill(0.0)

        if (
            getattr(self, "_step_combat_lookup_ids", None) is None
            or self._step_combat_lookup_ids.size < n
        ):
            self._step_combat_lookup_ids = np.empty((n,), dtype=np.int32)

    def step_agents_batch(
        self,
        agent_ids: np.ndarray,
        actions: np.ndarray,
        energies: np.ndarray,
        healths: np.ndarray,
        injuries: np.ndarray,
        scars: np.ndarray | None = None,
        move_cost_mults: np.ndarray | None = None,
        juvenile_hazard_mults: np.ndarray | None = None,
        juvenile_harvest_mults: np.ndarray | None = None,
        resource_handling_axes: np.ndarray | None = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Step a batch of agents.

        Returns:
            died_mask, new_energy, new_health, food_gained, new_injury, new_scar
        """

        n = int(agent_ids.size)
        if n == 0:
            if bool(getattr(self, "patch_world_enable", False)):
                self.patch_food_harvest_total_last = 0.0
                self.patch_movement_energy_extra_last = 0.0
                self.patch_movement_energy_event_count_last = 0
            return (
                np.empty((0,), dtype=bool),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
            )

        if not _NUMBA_OK:
            raise RuntimeError("Numba is not available. Install it with: pip install numba")

        self._ensure_step_scratch(n)

        cfg = self.cfg

        # Public APIs/reporting remain biological-ID based. The occupancy grid
        # stores resident slot handles and resolves IDs through slot_to_agent_id.
        raw_ids = np.asarray(agent_ids)
        if not np.issubdtype(raw_ids.dtype, np.integer):
            raise ValueError("agent_ids must use an integer dtype")
        ids_i64 = raw_ids.astype(np.int64, copy=False)
        if bool(np.any(ids_i64 < 0)) or bool(
            np.any(ids_i64 > np.iinfo(np.int32).max)
        ):
            raise ValueError("agent_ids must be in signed int32 range")
        ids_i32 = ids_i64.astype(np.int32, copy=False)
        world_slots = np.empty((n,), dtype=np.int32)
        for index, agent_id in enumerate(ids_i64):
            world_slots[index] = self._resolve_world_slot(
                int(agent_id),
                require_alive=True,
            )[0]
        actions_i32 = actions.astype(np.int32, copy=False)

        injuries_f32 = injuries.astype(np.float32, copy=False)
        energies_f32 = energies.astype(np.float32, copy=False)
        healths_f32 = healths.astype(np.float32, copy=False)

        # Handle scars input without allocating per call
        if scars is None:
            scars_f32 = self._step_zero_scars[:n]
        else:
            scars_f32 = scars.astype(np.float32, copy=False)


        # ------------------------------------------------------------
        # Phase 3 determinism: canonicalize agent processing order
        # ------------------------------------------------------------
        # Env resolution (movement, collisions, attacks, gather) can be order-sensitive.
        # Make it invariant by sorting by agent_id and then un-permuting outputs.
        order = np.argsort(ids_i32, kind="mergesort")
        # IMPORTANT: endpoint checks are insufficient (a permuted order can still have
        # 0 at head and n-1 at tail). Use full equality to decide whether to permute.
        is_identity = bool(np.array_equal(order, np.arange(n, dtype=order.dtype)))
        if not is_identity:
            inv = np.empty_like(order)
            inv[order] = np.arange(n, dtype=order.dtype)

            ids_i32 = ids_i32[order]
            ids_i64 = ids_i64[order]
            world_slots = world_slots[order]
            actions_i32 = actions_i32[order]

            injuries_f32 = injuries_f32[order]
            energies_f32 = energies_f32[order]
            healths_f32 = healths_f32[order]
            scars_f32 = scars_f32[order]
        else:
            inv = None




        # Pull positions into reusable buffers (movement mutates these)
        ys = self._step_ys_buf[:n]
        xs = self._step_xs_buf[:n]
        ys[:] = self.agent_pos_y_by_slot[world_slots].astype(np.int32, copy=False)
        xs[:] = self.agent_pos_x_by_slot[world_slots].astype(np.int32, copy=False)

        costs = self._step_costs_buf[:n]
        moved_mask = self._step_moved_buf[:n]
        food_gained = self._step_food_buf[:n]


        # === Tier 3 change: per-tick init ===
        tier3_enabled = bool(getattr(cfg, "migration_pressure_enable", False))
        tier3_stats = {
            "overcrowd_mean": 0.0,
            "overcrowd_max": 0.0,
            "crossings": 0,
            "cross_events": 0,
            "extra_move_cost_sum": 0.0,
            "extra_move_cost_max": 0.0,
        }
        # Runner reads this each tick (even when Tier 3 is disabled).
        self.tier3_last_step_stats = tier3_stats
        # === Tier 3 change: end per-tick init ===


        enable_biomes = 1 if bool(getattr(cfg, "enable_biomes", False)) else 0
        enable_injury = 1 if bool(getattr(cfg, "enable_injury", False)) else 0
        enable_scar = 1 if bool(getattr(cfg, "enable_scar", False)) else 0

        # Hazard RNG only if enabled
        if enable_biomes:
            # Hazard RNG must align with the *current* canonicalized batch order.
            # Note: ids/actions/energies/... have already been permuted when inv is not None,
            # so we MUST NOT re-apply `order` to fresh random draws (that would misalign the
            # RNG samples with the agent vectors and reintroduce order-sensitivity).
            r_h = self.rng.random(n, dtype=np.float32)
        else:
            r_h = self._step_rhaz_buf[:n]
            r_h.fill(0.0)


        # Movement cost multipliers (Part B). If not provided, use ones.
        if move_cost_mults is None:
            move_cost_mults = self._step_move_cost_mults_buf[:n]
        else:
            move_cost_mults = move_cost_mults.astype(np.float32, copy=False)


        if inv is not None:
            move_cost_mults = move_cost_mults[order]


        # Tier 4 juvenile multipliers (optional, provided by runner). If not provided, use ones.
        if juvenile_hazard_mults is None:
            juvenile_hazard_mults = self._step_tier4_juvenile_hazard_mults_buf[:n]
        else:
            juvenile_hazard_mults = juvenile_hazard_mults.astype(np.float32, copy=False)


        if inv is not None:
            juvenile_hazard_mults = juvenile_hazard_mults[order]

        if juvenile_harvest_mults is None:
            juvenile_harvest_mults = self._step_tier4_juvenile_harvest_mults_buf[:n]
        else:
            juvenile_harvest_mults = juvenile_harvest_mults.astype(np.float32, copy=False)


        if inv is not None:
            juvenile_harvest_mults = juvenile_harvest_mults[order]

        if resource_handling_axes is not None:
            resource_handling_axes = resource_handling_axes.astype(np.float32, copy=False)
            if inv is not None:
                resource_handling_axes = resource_handling_axes[order]

        _apply_costs(
            actions_i32,
            costs,
            move_cost_mults,
            float(cfg.move_cost),
            float(cfg.gather_cost),
            float(cfg.rest_cost),
            float(cfg.attack_cost),
            float(self.action_cost_mult),
            float(self.rest_cost_mult),
        )

        _apply_biome_cost_mult(
            costs,
            ys,
            xs,
            self.biome_grid,
            enable_biomes,
            float(getattr(cfg, "biome_wild_action_cost_mult", 1.0)),
        )

        _apply_injury_cost_mult(
            costs,
            injuries_f32,
            scars_f32,
            enable_injury,
            enable_scar,
            float(getattr(cfg, "injury_action_cost_penalty", 0.0)),
            float(getattr(cfg, "injury_max", 10.0)),
            float(getattr(cfg, "scar_max", float(getattr(cfg, "injury_max", 10.0)))),
            float(getattr(cfg, "scar_cost_weight", 1.0)),
            float(getattr(cfg, "injury_penalty_curve_power", 1.0)),
        )


        # === Tier 3 change: transition risk kernel (capture origin positions) ===
        tier3_tr = getattr(cfg, "tier3_region_transition_risk", {}) or {}
        tier3_tr_enable = tier3_enabled and bool(tier3_tr.get("enable", False)) and float(tier3_tr.get("crossing_risk_base", 0.0)) > 0.0
        prev_ys = self._step_prev_ys_buf[:n]
        prev_xs = self._step_prev_xs_buf[:n]
        if tier3_tr_enable:
            prev_ys[:] = ys
            prev_xs[:] = xs
        # === Tier 3 change: end transition risk kernel (capture origin positions) ===

        _movement_kernel(
            world_slots,
            actions_i32,
            ys,
            xs,
            self.agent_grid,
            int(cfg.height),
            int(cfg.width),
            moved_mask,
        )

        moved_any = bool(moved_mask.any())
        if bool(getattr(self, "patch_world_enable", False)):
            self.patch_movement_energy_extra_last = 0.0
            self.patch_movement_energy_event_count_last = 0
            if bool(getattr(cfg, "patch_movement_cost_apply_enable", False)) and moved_any:
                from env.gridworld_pkg.patch_world import patch_movement_cost_energy_scale

                moved = moved_mask.astype(bool, copy=False)
                terrain_cost = np.asarray(self.patch_movement_cost[ys, xs], dtype=np.float32)
                scale = float(patch_movement_cost_energy_scale(cfg))
                mult = np.maximum(0.0, 1.0 + scale * (terrain_cost - 1.0)).astype(np.float32, copy=False)
                old_cost = costs[moved].astype(np.float32, copy=True)
                new_cost = np.maximum(0.0, old_cost * mult[moved]).astype(np.float32, copy=False)
                costs[moved] = new_cost
                self.patch_movement_energy_extra_last = float(np.sum(new_cost - old_cost, dtype=np.float64))
                self.patch_movement_energy_event_count_last = int(np.sum(moved))
            if moved_any:
                from env.gridworld_pkg.patch_world import queue_patch_world_movement_emissions

                queue_patch_world_movement_emissions(self, ys, xs, moved_mask)

        if bool(getattr(self, "patch_world_enable", False)):
            self.patch_food_harvest_total_last = 0.0
            from env.gridworld_pkg.patch_world import (
                _emit_value,
                apply_respec0b_resource_gather,
                patch_food_gather_amount,
                patch_food_presence_threshold,
                respec0b_resource_specialization_enabled,
            )

            if respec0b_resource_specialization_enabled(cfg):
                any_gain, harvest_total = apply_respec0b_resource_gather(
                    self,
                    ids_i32,
                    actions_i32,
                    ys,
                    xs,
                    food_gained,
                    resource_handling_axes=resource_handling_axes,
                )
            else:
                any_gain, harvest_total, _, _ = _patch_gather_kernel(
                    actions_i32,
                    ys,
                    xs,
                    self.grid,
                    self.patch_food_amount,
                    self.patch_scent_emit_pending,
                    self.patch_waste_emit_pending,
                    food_gained,
                    float(cfg.food_energy),
                    float(patch_food_gather_amount(cfg)),
                    float(patch_food_presence_threshold(cfg)),
                    float(_emit_value(cfg, "patch_scent_gather_emit", 0.05)),
                    float(_emit_value(cfg, "patch_waste_consume_emit", 0.05)),
                )
            self.patch_food_harvest_total_last = float(harvest_total)
        else:
            any_gain = _gather_kernel(
                actions_i32,
                ys,
                xs,
                self.grid,
                food_gained,
                float(cfg.food_energy),
            )

        # Avoid scanning moved_mask multiple times
        if moved_any or any_gain == 1:
            self._views_dirty = True


        # === Tier 3 change: compute per-step region signals (CC + move cost + crossings) ===
        tier3_hazard_mults = self._step_tier3_hazard_mults_buf[:n]
        if tier3_enabled:
            region_grid = getattr(self, "tier3_region_id_grid", None)
            region_count = int(getattr(self, "tier3_region_count", 1) or 1)

            if region_grid is None:
                region_ids = np.zeros((n,), dtype=np.int32)
                region_count = 1
            else:
                region_ids = region_grid[ys, xs].astype(np.int32, copy=False)

            # Carrying capacity (crowding multipliers)
            cc = getattr(cfg, "tier3_regional_carrying_capacity", {}) or {}
            cc_enable = bool(cc.get("enable", False))

            yield_mult_by_region = None
            hazard_mult_by_region = None

            if cc_enable and region_count > 0:
                K = cc.get("capacity_by_region", [])
                if not isinstance(K, (list, tuple)) or len(K) != region_count:
                    raise ValueError(f"Tier 3 carrying_capacity capacity_by_region must have length {region_count}")
                K = np.asarray(K, dtype=np.float32)
                if np.any(K <= 0.0):
                    raise ValueError("Tier 3 carrying_capacity forbids K<=0; use hazard/cost to make regions uninhabitable")

                N = np.bincount(region_ids, minlength=region_count).astype(np.float32)

                gamma = float(cc.get("ema_gamma", 1.0))
                if gamma <= 0.0:
                    gamma = 1.0
                if gamma > 1.0:
                    gamma = 1.0

                if not hasattr(self, "_tier3_cc_Neff") or getattr(self, "_tier3_cc_Neff").shape[0] != region_count:
                    self._tier3_cc_Neff = np.zeros((region_count,), dtype=np.float32)
                Neff = self._tier3_cc_Neff
                if gamma < 1.0:
                    Neff[:] = (1.0 - gamma) * Neff + gamma * N
                else:
                    Neff[:] = N

                over = (Neff - K) / K
                over = np.maximum(over, 0.0)

                tier3_stats["overcrowd_mean"] = float(np.mean(over)) if over.size else 0.0
                tier3_stats["overcrowd_max"] = float(np.max(over)) if over.size else 0.0

                scarcity_alpha = float(cc.get("scarcity_alpha", 0.0))
                yield_floor = float(cc.get("yield_floor", 0.0))
                if scarcity_alpha > 0.0:
                    yield_mult_by_region = 1.0 / (1.0 + scarcity_alpha * over)
                    if yield_floor > 0.0:
                        yield_mult_by_region = np.maximum(yield_mult_by_region, yield_floor).astype(np.float32)
                    else:
                        yield_mult_by_region = yield_mult_by_region.astype(np.float32)

                hazard_beta = float(cc.get("hazard_beta", 0.0))
                if hazard_beta > 0.0:
                    hazard_mult_by_region = (1.0 + hazard_beta * over).astype(np.float32)
                    cap = cc.get("hazard_mult_cap", None)
                    if cap is not None:
                        hazard_mult_by_region = np.minimum(hazard_mult_by_region, float(cap)).astype(np.float32)

            # Apply CC1 (consumption yield) at gather result (energy gained)
            if yield_mult_by_region is not None and any_gain == 1:
                food_gained[:] = food_gained * yield_mult_by_region[region_ids]

            # Hazard mults for hazard kernel (CC2)
            if hazard_mult_by_region is not None:
                tier3_hazard_mults[:] = hazard_mult_by_region[region_ids]
            else:
                tier3_hazard_mults.fill(1.0)

            # Movement cost (successful move only; destination-region lookup; extra beyond baseline)
            mc = getattr(cfg, "tier3_movement_cost_map", {}) or {}
            mc_enable = bool(mc.get("enable", False))
            if mc_enable and moved_any:
                cost_by_region = mc.get("cost_by_region", [])
                if not isinstance(cost_by_region, (list, tuple)) or len(cost_by_region) != region_count:
                    raise ValueError(f"Tier 3 movement_cost_map cost_by_region must have length {region_count}")
                base = float(mc.get("base_move_cost", 1.0))
                mult_by_region = np.asarray(cost_by_region, dtype=np.float32)
                dest_mult = mult_by_region[region_ids]
                extra = base * (dest_mult - 1.0)
                mm = moved_mask.astype(bool)
                extra_applied = extra * mm.astype(np.float32)
                costs[:] = costs + extra_applied
                tier3_stats["extra_move_cost_sum"] = float(np.sum(extra_applied)) if extra_applied.size else 0.0
                tier3_stats["extra_move_cost_max"] = float(np.max(extra_applied)) if np.any(mm) else 0.0

            # Crossing detection (no per-agent persistent state)
            if tier3_tr_enable and moved_any:
                origin_buf = self._step_tier3_origin_regions_buf[:n]
                dest_buf = self._step_tier3_dest_regions_buf[:n]
                cross_buf = self._step_tier3_crossing_mask_buf[:n]

                if region_grid is None:
                    origin_buf.fill(0)
                    dest_buf.fill(0)
                else:
                    origin_buf[:] = region_grid[prev_ys, prev_xs].astype(np.int32, copy=False)
                    dest_buf[:] = region_ids

                mm = moved_mask.astype(bool)
                crossing_mask = mm & (origin_buf != dest_buf)
                cross_buf[:] = crossing_mask.astype(np.uint8)
                tier3_stats["crossings"] = int(np.sum(crossing_mask))
        else:
            tier3_hazard_mults.fill(1.0)
        # === Tier 3 change: end compute per-step region signals (CC + move cost + crossings) ===

        # Tier 4: juvenile harvest inefficiency (age effect). Apply exactly once, after Tier 3 yield modifiers.
        if any_gain == 1:
            food_gained[:] = food_gained * juvenile_harvest_mults


        eff_max = self._effective_max_energy_vec(injuries_f32, scars_f32).astype(np.float32, copy=False)
        new_energy = energies_f32 - costs + food_gained
        new_energy = np.minimum(new_energy, eff_max)
        new_energy = np.maximum(new_energy, 0.0).astype(np.float32, copy=False)

        # These are returned -> keep as fresh arrays (callers may hold references)
        new_health = healths_f32.copy()
        new_injury = injuries_f32.copy()
        new_scar = scars_f32.astype(np.float32, copy=False).copy()


        # === Tier 3 change: transition risk kernel (apply crossing hazard events) ===
        if tier3_enabled and (getattr(cfg, "tier3_region_transition_risk", {}) or {}).get("enable", False):
            if tier3_stats.get("crossings", 0) > 0:
                tier3_tr = getattr(cfg, "tier3_region_transition_risk", {}) or {}
                base_risk = float(tier3_tr.get("crossing_risk_base", 0.0))
                dmg = float(tier3_tr.get("crossing_damage", 0.1))
                if base_risk > 0.0 and dmg > 0.0:
                    cross_buf = self._step_tier3_crossing_mask_buf[:n].astype(bool)
                    origin_buf = self._step_tier3_origin_regions_buf[:n]
                    dest_buf = self._step_tier3_dest_regions_buf[:n]
                    region_count = int(getattr(self, "tier3_region_count", 1) or 1)

                    mat = tier3_tr.get("transition_risk_matrix", None)
                    if mat is None:
                        p = np.full((n,), base_risk, dtype=np.float32)
                    else:
                        mat_np = np.asarray(mat, dtype=np.float32)
                        if mat_np.shape != (region_count, region_count):
                            raise ValueError(f"Tier 3 transition_risk_matrix must be shape {(region_count, region_count)}, got {mat_np.shape}")
                        p = base_risk * mat_np[origin_buf, dest_buf].astype(np.float32, copy=False)
                    p = np.clip(p, 0.0, 1.0)

                    r_cross = self.rng.random(n, dtype=np.float32)
                    event_mask = cross_buf & (r_cross < p)
                    events = int(np.sum(event_mask))
                    tier3_stats["cross_events"] = events
                    if events > 0:
                        new_health[event_mask] -= dmg
                        if enable_injury != 0:
                            inj_per = float(getattr(cfg, "injury_per_damage", 0.0))
                            inj_max = float(getattr(cfg, "injury_max", 10.0))
                            if inj_max <= 0.0:
                                inj_max = 10.0
                            if inj_per > 0.0:
                                v = new_injury[event_mask] + (dmg * inj_per)
                                v = np.minimum(v, inj_max)
                                new_injury[event_mask] = v.astype(np.float32, copy=False)
        # === Tier 3 change: end transition risk kernel (apply crossing hazard events) ===

        _hazard_kernel(
            ys,
            xs,
            self.biome_grid,
            self.hazard_rate_map,
            tier3_hazard_mults,
            juvenile_hazard_mults,
            r_h,
            enable_biomes,
            float(getattr(cfg, "biome_wild_hazard_prob", 0.0)),
            float(getattr(cfg, "biome_wild_hazard_damage", 0.0)),
            enable_injury,
            float(getattr(cfg, "biome_wild_hazard_injury_per_damage", 0.0)),
            float(getattr(cfg, "injury_max", 10.0)),
            new_health,
            new_injury,
        )

        if bool(getattr(cfg, "enable_combat", False)):
            dmg = float(getattr(cfg, "attack_damage", 0.0))
            inj_per = float(getattr(cfg, "injury_per_damage", 0.0)) if enable_injury else 0.0
            inj_max = float(getattr(cfg, "injury_max", 10.0))
            scar_max = float(getattr(cfg, "scar_max", inj_max))
            scar_thresh = float(getattr(cfg, "scar_start_threshold", 0.0))
            scar_gain_per = float(getattr(cfg, "scar_gain_per_injury", 0.0))
            scar_gain_base = float(getattr(cfg, "scar_gain_base", 0.0))

            lookup_ids = self._step_combat_lookup_ids[:n]
            lookup_ids[:] = ids_i32

            _combat_kernel(
                ids_i64,
                actions_i32,
                ys,
                xs,
                self.agent_grid,
                self.slot_to_agent_id,
                lookup_ids,
                new_energy,
                new_health,
                new_injury,
                new_scar,
                int(cfg.height),
                int(cfg.width),
                dmg,
                enable_injury,
                inj_per,
                inj_max,
                enable_scar,
                scar_max,
                scar_thresh,
                scar_gain_per,
                scar_gain_base,
                self._step_attack_target_id_buf[:n],
                self._step_attack_accepted_buf[:n],
                self._step_attack_reject_code_buf[:n],
                self._step_attack_attempted_damage_buf[:n],
                self._step_attack_applied_damage_buf[:n],
                self._step_attack_target_health_before_buf[:n],
                self._step_attack_target_health_after_buf[:n],
            )

        elif bool(np.any(actions_i32 == ACT_ATTACK)):
            self._step_attack_reject_code_buf[:n][actions_i32 == ACT_ATTACK] = 4

        if bool(getattr(self, "action_consequence_telemetry_enable", False)):
            rows = []
            for ii in range(n):
                if int(actions_i32[ii]) != ACT_ATTACK:
                    continue
                code = int(self._step_attack_reject_code_buf[ii])
                accepted = bool(int(self._step_attack_accepted_buf[ii]) == 1)
                target_id = int(self._step_attack_target_id_buf[ii])
                h0 = float(self._step_attack_target_health_before_buf[ii])
                h1 = float(self._step_attack_target_health_after_buf[ii])
                if accepted:
                    reason = ""
                    evaluable = True
                elif code == 2:
                    reason = "attacker_energy_nonpositive"
                    evaluable = True
                elif code == 3:
                    reason = "no_adjacent_batch_target"
                    evaluable = True
                elif code == 4:
                    reason = "combat_disabled"
                    evaluable = True
                else:
                    reason = "attack_not_processed"
                    evaluable = False
                rows.append(
                    {
                        "schema_version": "attacker_consequence_event_v1",
                        "attacker_id": int(ids_i32[ii]),
                        "target_id": target_id,
                        "attack_accepted": accepted,
                        "attack_rejected": not accepted,
                        "rejection_reason": reason,
                        "attempted_damage": float(self._step_attack_attempted_damage_buf[ii]),
                        "applied_damage": float(self._step_attack_applied_damage_buf[ii]),
                        "target_health_before": h0 if np.isfinite(h0) else None,
                        "target_health_after": h1 if np.isfinite(h1) else None,
                        "target_killed_by_event": bool(accepted and h0 > 0.0 and h1 <= 0.0),
                        "consequence_evaluable": bool(evaluable),
                        "not_evaluable_reason": "" if evaluable else reason,
                    }
                )
            self.last_attack_consequence_events = rows
        else:
            self.last_attack_consequence_events = []

        died_mask = ((new_energy <= 0.0) | (new_health <= 0.0)).astype(bool, copy=False)

        # Commit updated positions back into state arrays (only where moved)
        if moved_any:
            moved = moved_mask.astype(bool, copy=False)
            moved_ids = ids_i32[moved]
            moved_slots = world_slots[moved]
            self.agent_pos_y_by_slot[moved_slots] = ys[moved]
            self.agent_pos_x_by_slot[moved_slots] = xs[moved]
            shadow_mask = moved_ids < int(self.agent_alive.size)
            shadow_ids = moved_ids[shadow_mask]
            self.agent_pos_y[shadow_ids] = ys[moved][shadow_mask]
            self.agent_pos_x[shadow_ids] = xs[moved][shadow_mask]


        if inv is not None:
            died_mask = died_mask[inv]
            new_energy = new_energy[inv]
            new_health = new_health[inv]
            food_gained = food_gained[inv]
            new_injury = new_injury[inv]
            new_scar = new_scar[inv]

        return died_mask, new_energy, new_health, food_gained, new_injury, new_scar
