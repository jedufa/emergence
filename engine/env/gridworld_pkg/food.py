from __future__ import annotations

import numpy as np
from env.gridworld_pkg.constants import FOOD


class FoodMixin:
    def _sync_patch_world_after_spawn(self) -> None:
        if not bool(getattr(self, "patch_world_enable", False)):
            return
        from env.gridworld_pkg.patch_world import sync_binary_spawned_food_to_patch_amount

        sync_binary_spawned_food_to_patch_amount(self)

    def _ensure_biome_cells(self) -> None:
        """
        Lazily build biome cell caches if biomes are enabled.
        This avoids fragile init ordering dependencies.
        """
        if hasattr(self, "_biome_cells"):
            return

        self._biome_cells = {}

        # If biome grid does not exist yet, fall back to empty sets
        biome_grid = getattr(self, "biome_grid", None)
        if biome_grid is None:
            self._biome_cells["safe"] = (
                np.empty((0,), dtype=np.int32),
                np.empty((0,), dtype=np.int32),
            )
            self._biome_cells["wild"] = (
                np.empty((0,), dtype=np.int32),
                np.empty((0,), dtype=np.int32),
            )
            return

        # Expected convention: biome_grid[y, x] == 0 safe, 1 wild
        ys_safe, xs_safe = np.where(biome_grid == 0)
        ys_wild, xs_wild = np.where(biome_grid == 1)

        self._biome_cells["safe"] = (
            ys_safe.astype(np.int32, copy=False),
            xs_safe.astype(np.int32, copy=False),
        )
        self._biome_cells["wild"] = (
            ys_wild.astype(np.int32, copy=False),
            xs_wild.astype(np.int32, copy=False),
        )

    def spawn_food(self) -> None:
        current_food = int((self.grid == FOOD).sum())
        if current_food >= int(self.cfg.max_food):
            return

        # Base spawn intensity (count-based).
        # Tier 1 semantics:
        # - resource_regrowth_map acts as a GLOBAL rate proxy (mean multiplier)
        #   because the current food model spawns a fixed count per step.
        # - resource_spawn_density_map controls WHERE spawns go (sampling weights).
        base = float(self.cfg.food_spawn_per_step)
        reg_mean = 1.0
        try:
            # Tier 1: static regrowth map (uniform when disabled)
            reg_map = getattr(self, "resource_regrowth_map", None)
            if reg_map is None:
                reg_map = 1.0
            # Tier 2: drought multiplier (uniform when disabled)
            if bool(getattr(self, "spatial_droughts_enable", False)) and hasattr(self, "drought_regrowth_mult_map"):
                try:
                    reg_map = np.asarray(reg_map) * self.drought_regrowth_mult_map
                except Exception:
                    pass
            # Use mean as a global intensity proxy for the count-based spawn model.
            reg_mean = float(np.mean(reg_map)) if hasattr(reg_map, "shape") else float(reg_map)
        except Exception:
            reg_mean = 1.0

        eff = int(round(base * float(self.food_spawn_mult) * max(0.0, reg_mean)))
        if eff <= 0:
            return

        to_spawn = min(eff, int(self.cfg.max_food) - current_food)
        if to_spawn <= 0:
            return

        # -----------------------------
        # No biomes: legacy uniform spawn
        # -----------------------------
        if not bool(getattr(self.cfg, "enable_biomes", False)):
            # Weighted spawn using resource_spawn_density_map, falling back to uniform.
            spawned_any = False
            h = int(self.cfg.height)
            w = int(self.cfg.width)
            weights = None
            try:
                base_weights = None
                if hasattr(self, "resource_spawn_density_map"):
                    base_weights = self.resource_spawn_density_map
                if base_weights is not None:
                    weights = np.maximum(np.asarray(base_weights, dtype=np.float64), 0.0)
            except Exception:
                weights = None

            # Tier 2: drought spawn multiplier affects WHERE spawns go (density weights).
            if weights is not None and bool(getattr(self, "spatial_droughts_enable", False)) and hasattr(self, "drought_spawn_mult_map"):
                try:
                    weights = np.maximum(weights * self.drought_spawn_mult_map.astype(np.float64, copy=False), 0.0)
                except Exception:
                    pass

            if weights is None or float(weights.sum()) <= 1e-12:
                # Legacy uniform spawn
                for _ in range(to_spawn):
                    y = int(self.rng.integers(0, h))
                    x = int(self.rng.integers(0, w))
                    if self.grid[y, x] != FOOD:
                        self.grid[y, x] = FOOD
                        spawned_any = True
            else:
                p = (weights.reshape(-1) / float(weights.sum())).astype(np.float64, copy=False)
                placed = 0
                tries = 0
                max_tries = max(80, to_spawn * 10)
                fast_weighted = bool(getattr(self, "_runnercore11_fast_weighted_food_spawn_enable", False))
                if fast_weighted:
                    while placed < to_spawn and tries < max_tries:
                        need = min(int(to_spawn - placed), int(max_tries - tries))
                        if need <= 0:
                            break
                        idxs = np.asarray(self.rng.choice(h * w, size=need, p=p)).reshape(-1)
                        tries += int(idxs.size)
                        for raw_idx in idxs:
                            idx = int(raw_idx)
                            y = int(idx // w)
                            x = int(idx % w)
                            if self.grid[y, x] != FOOD:
                                self.grid[y, x] = FOOD
                                spawned_any = True
                                placed += 1
                    self._runnercore11_fast_weighted_food_spawn_calls = int(
                        getattr(self, "_runnercore11_fast_weighted_food_spawn_calls", 0) or 0
                    ) + 1
                    self._runnercore11_fast_weighted_food_spawn_draws = int(
                        getattr(self, "_runnercore11_fast_weighted_food_spawn_draws", 0) or 0
                    ) + int(tries)
                    self._runnercore11_fast_weighted_food_spawn_placements = int(
                        getattr(self, "_runnercore11_fast_weighted_food_spawn_placements", 0) or 0
                    ) + int(placed)
                    if placed < to_spawn:
                        self._runnercore11_fast_weighted_food_spawn_incomplete = int(
                            getattr(self, "_runnercore11_fast_weighted_food_spawn_incomplete", 0) or 0
                        ) + 1
                else:
                    while placed < to_spawn and tries < max_tries:
                        tries += 1
                        idx = int(self.rng.choice(h * w, p=p))
                        y = int(idx // w)
                        x = int(idx % w)
                        if self.grid[y, x] != FOOD:
                            self.grid[y, x] = FOOD
                            spawned_any = True
                            placed += 1

            if spawned_any:
                self._sync_patch_world_after_spawn()
                self._views_dirty = True
            return

        # -----------------------------
        # Biome-aware spawn (safe / wild)
        # -----------------------------
        self._ensure_biome_cells()

        safe_mult = float(getattr(self.cfg, "biome_safe_food_mult", 1.0))
        wild_mult = float(getattr(self.cfg, "biome_wild_food_mult", 1.0))

        # Optional spatial spawn density weights
        spawn_map = None
        if bool(getattr(self, "spatial_ecology_enable", False)) and hasattr(self, "resource_spawn_density_map"):
            try:
                spawn_map = np.maximum(self.resource_spawn_density_map.astype(np.float64, copy=False), 0.0)
            except Exception:
                spawn_map = None

        ys_safe, xs_safe = self._biome_cells.get(
            "safe",
            (np.empty((0,), dtype=np.int32), np.empty((0,), dtype=np.int32)),
        )
        ys_wild, xs_wild = self._biome_cells.get(
            "wild",
            (np.empty((0,), dtype=np.int32), np.empty((0,), dtype=np.int32)),
        )

        n_safe = int(ys_safe.size)
        n_wild = int(ys_wild.size)

        if n_safe == 0 and n_wild == 0:
            return

        if n_safe == 0:
            safe_mult = 0.0
        if n_wild == 0:
            wild_mult = 0.0

        # Biome weights (Tier 1: incorporate spawn density map if present).
        if spawn_map is None:
            w_safe = float(n_safe) * safe_mult
            w_wild = float(n_wild) * wild_mult
        else:
            w_safe = float(spawn_map[ys_safe, xs_safe].sum()) * safe_mult
            w_wild = float(spawn_map[ys_wild, xs_wild].sum()) * wild_mult
        denom = w_safe + w_wild
        if denom <= 1e-12:
            w_safe = float(n_safe)
            w_wild = float(n_wild)
            denom = w_safe + w_wild

        n_spawn_safe = int(round(float(to_spawn) * (w_safe / denom)))
        n_spawn_safe = max(0, min(to_spawn, n_spawn_safe))
        n_spawn_wild = int(to_spawn - n_spawn_safe)

        spawned_any = False

        def _spawn_in_cells(ys: np.ndarray, xs: np.ndarray, n: int, weights_1d: np.ndarray | None = None) -> None:
            nonlocal spawned_any
            if n <= 0 or ys.size == 0:
                return
            placed = 0
            tries = 0
            max_tries = max(50, n * 6)
            while placed < n and tries < max_tries:
                tries += 1
                if weights_1d is None or float(weights_1d.sum()) <= 1e-12:
                    j = int(self.rng.integers(0, ys.size))
                else:
                    p = (weights_1d / float(weights_1d.sum())).astype(np.float64, copy=False)
                    j = int(self.rng.choice(ys.size, p=p))
                y = int(ys[j])
                x = int(xs[j])
                if self.grid[y, x] != FOOD:
                    self.grid[y, x] = FOOD
                    spawned_any = True
                    placed += 1

        if spawn_map is None:
            _spawn_in_cells(ys_safe, xs_safe, n_spawn_safe)
            _spawn_in_cells(ys_wild, xs_wild, n_spawn_wild)
        else:
            w_safe_cells = spawn_map[ys_safe, xs_safe].reshape(-1).astype(np.float64, copy=False)
            w_wild_cells = spawn_map[ys_wild, xs_wild].reshape(-1).astype(np.float64, copy=False)
            _spawn_in_cells(ys_safe, xs_safe, n_spawn_safe, w_safe_cells)
            _spawn_in_cells(ys_wild, xs_wild, n_spawn_wild, w_wild_cells)

        if spawned_any:
            self._sync_patch_world_after_spawn()
            self._views_dirty = True
