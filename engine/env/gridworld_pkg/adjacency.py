from __future__ import annotations

from typing import List, Optional, Tuple
import numpy as np

try:
    from numba import njit
    _NUMBA_OK = True
except Exception:  # pragma: no cover
    _NUMBA_OK = False
    njit = None


if _NUMBA_OK:
    @njit(cache=True)
    def _append_adjacent_agent_id(slot_handle: int, agent_id: int, slot_to_agent_id, out_ids, n: int):
        if slot_handle < 0 or slot_handle >= slot_to_agent_id.shape[0]:
            return n
        other = int(slot_to_agent_id[slot_handle])
        if other != -1 and other != agent_id:
            out_ids[n] = other
            n += 1
        return n

    @njit(cache=True)
    def _adjacent_ids_kernel(y: int, x: int, agent_id: int, agent_grid, slot_to_agent_id, height: int, width: int, out_ids):
        n = 0
        # N
        ny = y - 1
        nx = x
        if 0 <= ny < height and 0 <= nx < width:
            n = _append_adjacent_agent_id(int(agent_grid[ny, nx]), agent_id, slot_to_agent_id, out_ids, n)
        # S
        ny = y + 1
        nx = x
        if 0 <= ny < height and 0 <= nx < width:
            n = _append_adjacent_agent_id(int(agent_grid[ny, nx]), agent_id, slot_to_agent_id, out_ids, n)
        # W
        ny = y
        nx = x - 1
        if 0 <= ny < height and 0 <= nx < width:
            n = _append_adjacent_agent_id(int(agent_grid[ny, nx]), agent_id, slot_to_agent_id, out_ids, n)
        # E
        ny = y
        nx = x + 1
        if 0 <= ny < height and 0 <= nx < width:
            n = _append_adjacent_agent_id(int(agent_grid[ny, nx]), agent_id, slot_to_agent_id, out_ids, n)
        return n

    @njit(cache=True)
    def _adjacent_empty_kernel(y: int, x: int, agent_grid, height: int, width: int, out_xy):
        n = 0
        coords = ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1))
        for j in range(4):
            ny, nx = coords[j]
            if 0 <= ny < height and 0 <= nx < width:
                if int(agent_grid[ny, nx]) == -1:
                    out_xy[n, 0] = ny
                    out_xy[n, 1] = nx
                    n += 1
        return n


class AdjacencyMixin:
    def adjacent_agent_ids(self, agent_id: int) -> List[int]:
        slot, _ = self._resolve_world_slot(int(agent_id), require_alive=True)
        y = int(self.agent_pos_y_by_slot[slot])
        x = int(self.agent_pos_x_by_slot[slot])

        if not _NUMBA_OK:
            adj: List[int] = []
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if self.in_bounds(ny, nx):
                    slot_handle = int(self.agent_grid[ny, nx])
                    if slot_handle < 0 or slot_handle >= int(self.slot_to_agent_id.size):
                        continue
                    other = int(self.slot_to_agent_id[slot_handle])
                    if other != -1 and other != agent_id:
                        adj.append(other)
            return adj

        out = getattr(self, "_adj_out_ids", None)
        if out is None or getattr(out, "dtype", None) != np.int32 or getattr(out, "shape", None) != (4,):
            out = np.empty((4,), dtype=np.int32)
            try:
                setattr(self, "_adj_out_ids", out)
            except Exception:
                pass
        n = _adjacent_ids_kernel(y, x, int(agent_id), self.agent_grid, self.slot_to_agent_id, int(self.cfg.height), int(self.cfg.width), out)
        # NOTE: out is a reusable scratch buffer. Only the first n slots are valid.
        return out[: int(n)].tolist()

    def find_adjacent_empty(self, agent_id: int) -> Optional[Tuple[int, int]]:
        slot, _ = self._resolve_world_slot(int(agent_id), require_alive=True)
        y = int(self.agent_pos_y_by_slot[slot])
        x = int(self.agent_pos_x_by_slot[slot])

        if not _NUMBA_OK:
            candidates = [(y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)]
            self.rng.shuffle(candidates)
            for ny, nx in candidates:
                if self.in_bounds(ny, nx) and self.agent_grid[ny, nx] == -1:
                    return (ny, nx)
            return None

        out_xy = getattr(self, "_adj_out_xy", None)
        if out_xy is None or getattr(out_xy, "dtype", None) != np.int32 or getattr(out_xy, "shape", None) != (4, 2):
            out_xy = np.empty((4, 2), dtype=np.int32)
            try:
                setattr(self, "_adj_out_xy", out_xy)
            except Exception:
                pass
        n = _adjacent_empty_kernel(y, x, self.agent_grid, int(self.cfg.height), int(self.cfg.width), out_xy)
        if int(n) <= 0:
            return None

        idx = getattr(self, "_adj_idx", None)
        if idx is None or getattr(idx, "dtype", None) != np.int32 or getattr(idx, "shape", None) != (4,):
            idx = np.empty((4,), dtype=np.int32)
            try:
                setattr(self, "_adj_idx", idx)
            except Exception:
                pass
        # Fill 0..n-1 (n <= 4)
        for j in range(int(n)):
            idx[j] = j
        self.rng.shuffle(idx[: int(n)])
        j = int(idx[0])
        return (int(out_xy[j, 0]), int(out_xy[j, 1]))
