from __future__ import annotations

"""Default-off CORRIDOR-CP0 central-place eligibility helpers."""

from typing import Any

import numpy as np

from env.gridworld_pkg.patch_world import CORRIDOR_CP0_ZONE_HOME_LOCAL


def corridor_cp0_central_place_eligibility_enabled(env_cfg: Any, sim_cfg: Any) -> bool:
    return bool(getattr(env_cfg, "patch_world_enable", False)) and bool(getattr(env_cfg, "corridor_cp0_enable", False)) and bool(
        getattr(sim_cfg, "corridor_cp0_central_place_eligibility_enable", False)
    )


def corridor_cp0_zone_at(env: Any, y: int, x: int) -> int:
    zones = getattr(env, "corridor_cp0_zone_class", None)
    if zones is None:
        return 0
    arr = np.asarray(zones)
    if y < 0 or x < 0 or y >= int(arr.shape[0]) or x >= int(arr.shape[1]):
        return 0
    return int(arr[int(y), int(x)])


def corridor_cp0_agent_position(env: Any, agent: Any) -> tuple[int, int] | None:
    try:
        aid = int(getattr(agent, "agent_id"))
    except Exception:
        return None
    pos_y = getattr(env, "agent_pos_y", None)
    pos_x = getattr(env, "agent_pos_x", None)
    alive = getattr(env, "agent_alive", None)
    if pos_y is None or pos_x is None:
        return None
    if aid < 0 or aid >= int(getattr(pos_y, "shape", (0,))[0]) or aid >= int(getattr(pos_x, "shape", (0,))[0]):
        return None
    if alive is not None and (aid >= int(getattr(alive, "shape", (0,))[0]) or not bool(alive[aid])):
        return None
    y = int(pos_y[aid])
    x = int(pos_x[aid])
    if y < 0 or x < 0:
        return None
    return y, x


def corridor_cp0_agent_home_eligible(env: Any, agent: Any) -> bool:
    pos = corridor_cp0_agent_position(env, agent)
    if pos is None:
        return False
    return corridor_cp0_zone_at(env, pos[0], pos[1]) == CORRIDOR_CP0_ZONE_HOME_LOCAL


def corridor_cp0_filter_home_eligible_agents(env: Any, agents: list[Any]) -> list[Any]:
    return [agent for agent in list(agents or []) if corridor_cp0_agent_home_eligible(env, agent)]
