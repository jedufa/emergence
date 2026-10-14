from __future__ import annotations

"""
env/gridworld_pkg/observations.py

Numba-accelerated batched local observations.
Falls back to numpy indexing if numba isn't available.
"""

import numpy as np
from env.gridworld_pkg.constants import FOOD
from env.gridworld_pkg.patch_world import (
    patch_world_normalized_observation_maps,
    patch_world_local_observation_channel_count,
    patch_world_observation_enabled,
)

try:
    from numba import njit
    _NUMBA_OK = True
except Exception:  # pragma: no cover
    _NUMBA_OK = False
    njit = None


if _NUMBA_OK:
    @njit(cache=True)
    def _obs_kernel(
        ys: np.ndarray,
        xs: np.ndarray,
        r: int,
        occ_p: np.ndarray,
        food_p: np.ndarray,
        out: np.ndarray,
    ) -> None:
        n = ys.shape[0]
        k = 2 * r + 1
        for i in range(n):
            yp = int(ys[i]) + r
            xp = int(xs[i]) + r
            base = i * (k * k)
            t = 0
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    oy = yp + dy
                    ox = xp + dx
                    if occ_p[oy, ox] != 0:
                        out[base + t] = -1.0
                    else:
                        out[base + t] = float(food_p[oy, ox])
                    t += 1


    @njit(cache=True)
    def _obs_kernel_with_patch_maps(
        ys: np.ndarray,
        xs: np.ndarray,
        r: int,
        occ_p: np.ndarray,
        food_p: np.ndarray,
        patch_maps: np.ndarray,
        out: np.ndarray,
    ) -> None:
        n = ys.shape[0]
        k = 2 * r + 1
        kk = k * k
        patch_c = patch_maps.shape[0]
        h = patch_maps.shape[1]
        w = patch_maps.shape[2]
        channel_count = 1 + patch_c
        for i in range(n):
            y0 = int(ys[i])
            x0 = int(xs[i])
            yp = y0 + r
            xp = x0 + r
            legacy_base = i * channel_count * kk
            t = 0
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    oy = yp + dy
                    ox = xp + dx
                    if occ_p[oy, ox] != 0:
                        out[legacy_base + t] = -1.0
                    else:
                        out[legacy_base + t] = float(food_p[oy, ox])
                    t += 1

            for c in range(patch_c):
                patch_base = legacy_base + (1 + c) * kk
                t = 0
                for dy in range(-r, r + 1):
                    yy = y0 + dy
                    for dx in range(-r, r + 1):
                        xx = x0 + dx
                        if yy < 0 or xx < 0 or yy >= h or xx >= w:
                            out[patch_base + t] = 0.0
                        else:
                            out[patch_base + t] = float(patch_maps[c, yy, xx])
                        t += 1


class ObservationsMixin:
    def _refresh_padded_views(self):
        r = self.cfg.view_radius
        occ = (self.agent_grid != -1).astype(np.uint8, copy=False)
        food = (self.grid == FOOD).astype(np.uint8, copy=False)

        self._occ_p = np.pad(occ, r, mode="constant", constant_values=0)
        self._food_p = np.pad(food, r, mode="constant", constant_values=0)
        self._views_dirty = False

    def get_local_observation(self, agent_id: int) -> np.ndarray:
        if patch_world_observation_enabled(self.cfg):
            ids = np.asarray([int(agent_id)], dtype=np.int32)
            return self.get_local_observations(ids)[0]

        if self._views_dirty or self._occ_p is None or self._food_p is None:
            self._refresh_padded_views()

        slot, _ = self._resolve_world_slot(int(agent_id), require_alive=True)
        y0 = int(self.agent_pos_y_by_slot[slot])
        x0 = int(self.agent_pos_x_by_slot[slot])
        r = int(self.cfg.view_radius)
        yp = y0 + r
        xp = x0 + r

        occ_view = self._occ_p[yp - r: yp + r + 1, xp - r: xp + r + 1]
        food_view = self._food_p[yp - r: yp + r + 1, xp - r: xp + r + 1]
        return np.where(occ_view, -1.0, food_view.astype(np.float32, copy=False))

    def get_local_observations(self, agent_ids: np.ndarray) -> np.ndarray:
        if self._views_dirty or self._occ_p is None or self._food_p is None:
            self._refresh_padded_views()

        n = int(agent_ids.size)
        r = int(self.cfg.view_radius)
        k = 2 * r + 1

        patch_obs_enable = patch_world_observation_enabled(self.cfg)
        channel_count = patch_world_local_observation_channel_count(self.cfg) if patch_obs_enable else 1

        if n == 0:
            return np.empty((0, k * k * channel_count), dtype=np.float32)

        raw_ids = np.asarray(agent_ids)
        if not np.issubdtype(raw_ids.dtype, np.integer):
            raise ValueError("agent_ids must use an integer dtype")
        ids = raw_ids.astype(np.int64, copy=False)
        slots = np.empty((n,), dtype=np.int32)
        for index, agent_id in enumerate(ids):
            slots[index] = self._resolve_world_slot(
                int(agent_id),
                require_alive=True,
            )[0]
        ys = self.agent_pos_y_by_slot[slots].astype(np.int32, copy=False)
        xs = self.agent_pos_x_by_slot[slots].astype(np.int32, copy=False)
        patch_maps = patch_world_normalized_observation_maps(self) if patch_obs_enable else []

        if (
            patch_obs_enable
            and _NUMBA_OK
            and bool(getattr(self.cfg, "patch_world_observation_numba_gather_enable", False))
            and patch_maps
        ):
            maps_stack = np.stack([np.asarray(arr, dtype=np.float32) for arr in patch_maps], axis=0)
            out_flat = np.empty((n * k * k * channel_count,), dtype=np.float32)
            _obs_kernel_with_patch_maps(ys, xs, r, self._occ_p, self._food_p, maps_stack, out_flat)
            return out_flat.reshape(n, k * k * channel_count)

        if _NUMBA_OK:
            legacy_flat = np.empty((n * k * k,), dtype=np.float32)
            _obs_kernel(ys, xs, r, self._occ_p, self._food_p, legacy_flat)
            legacy = legacy_flat.reshape(n, k * k)
        else:
            legacy = self._local_legacy_views_numpy(ys, xs, r, k)

        if not patch_obs_enable:
            return legacy.astype(np.float32, copy=False)

        yp = ys + r
        xp = xs + r
        dy = np.arange(-r, r + 1, dtype=np.int32).reshape(1, k, 1)
        dx = np.arange(-r, r + 1, dtype=np.int32).reshape(1, 1, k)
        y_idx = yp.reshape(-1, 1, 1).astype(np.int32, copy=False) + dy
        x_idx = xp.reshape(-1, 1, 1).astype(np.int32, copy=False) + dx

        channels = [legacy.reshape(n, k, k).astype(np.float32, copy=False)]
        if bool(getattr(self.cfg, "patch_world_observation_stacked_gather_enable", False)) and patch_maps:
            maps_stack = np.stack(
                [np.asarray(arr, dtype=np.float32) for arr in patch_maps],
                axis=0,
            )
            maps_p = np.pad(maps_stack, ((0, 0), (r, r), (r, r)), mode="constant", constant_values=0.0)
            gathered = maps_p[:, y_idx, x_idx].astype(np.float32, copy=False)
            channels.extend(gathered)
        else:
            for arr in patch_maps:
                arr_p = np.pad(np.asarray(arr, dtype=np.float32), r, mode="constant", constant_values=0.0)
                channels.append(arr_p[y_idx, x_idx].astype(np.float32, copy=False))

        stacked = np.stack(channels, axis=1)
        return stacked.reshape(n, -1).astype(np.float32, copy=False)

    def _local_legacy_views_numpy(self, ys: np.ndarray, xs: np.ndarray, r: int, k: int) -> np.ndarray:
        yp = ys + r
        xp = xs + r
        dy = np.arange(-r, r + 1, dtype=np.int32).reshape(1, k, 1)
        dx = np.arange(-r, r + 1, dtype=np.int32).reshape(1, 1, k)
        y_idx = yp.reshape(-1, 1, 1).astype(np.int32, copy=False) + dy
        x_idx = xp.reshape(-1, 1, 1).astype(np.int32, copy=False) + dx
        occ_view = self._occ_p[y_idx, x_idx]
        food_view = self._food_p[y_idx, x_idx]
        out = np.where(occ_view, -1.0, food_view.astype(np.float32, copy=False))
        return out.reshape(int(ys.size), -1).astype(np.float32, copy=False)
