from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .schema import validate_playback


@dataclass(frozen=True)
class PlaybackState:
    tick: int
    living_agents: tuple[dict[str, Any], ...]
    active_lineage_ids: tuple[str, ...]
    active_branch_ids: tuple[str, ...]
    largest_lineage_share: float
    lineage_counts: dict[str, int]
    branch_counts: dict[str, int]


class PlaybackModel:
    def __init__(self, data: dict[str, Any]):
        validate_playback(data)
        self.data = data

    def state_at(self, tick: int) -> PlaybackState:
        first_tick = self.data["metadata"]["first_tick"]
        last_tick = self.data["metadata"]["last_tick"]
        tick = max(first_tick, min(last_tick, tick))
        living = tuple(
            agent
            for agent in self.data["agents"]
            if agent["birth_tick"] <= tick
            and (agent.get("death_tick") is None or agent["death_tick"] > tick)
        )
        lineage_counts: dict[str, int] = {}
        branch_ids: set[str] = set()
        for agent in living:
            lineage_counts[agent["lineage_id"]] = lineage_counts.get(agent["lineage_id"], 0) + 1
            branch_ids.add(agent["branch_id"])
        largest = max(lineage_counts.values(), default=0)
        share = largest / len(living) if living else 0.0
        return PlaybackState(
            tick=tick,
            living_agents=living,
            active_lineage_ids=tuple(sorted(lineage_counts)),
            active_branch_ids=tuple(sorted(branch_ids)),
            largest_lineage_share=share,
            lineage_counts=dict(lineage_counts),
            branch_counts={branch_id: sum(1 for agent in living if agent["branch_id"] == branch_id) for branch_id in branch_ids},
        )
