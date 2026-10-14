from __future__ import annotations

"""
simulation/runner_pkg/envsync.py (EnvSyncMixin)

What this file does:
- Keeps the GridWorld environment's agent registry in sync with the Population:
  - Removes dead/missing agents from the env
  - Adds any living agents missing from the env
- Also supports a "place all agents" reset at the start of a run.

World membership is keyed by reusable population slot/token identity. Dense
low-ID arrays are compatibility shadows and are not membership authority.
"""

import numpy as np

from evolution.population_pkg.founder_spatial_separation import (
    founder_spatial_separation_enabled,
    place_initial_founders,
)

INITIAL_SPATIAL_PLACEMENT_HERITAGE_V1 = "heritage_random_v1"
INITIAL_SPATIAL_PLACEMENT_DISPERSED_LATTICE_V1 = "dispersed_lattice_v1"
INITIAL_SPATIAL_PLACEMENT_DISPERSED_LATTICE_FILL_V1 = "dispersed_lattice_fill_v1"
INITIAL_SPATIAL_PLACEMENT_CLUSTERED_QUADRANT_V1 = "clustered_quadrant_v1"
INITIAL_SPATIAL_PLACEMENT_PROFILES_V1 = frozenset(
    {
        INITIAL_SPATIAL_PLACEMENT_HERITAGE_V1,
        INITIAL_SPATIAL_PLACEMENT_DISPERSED_LATTICE_V1,
        INITIAL_SPATIAL_PLACEMENT_DISPERSED_LATTICE_FILL_V1,
        INITIAL_SPATIAL_PLACEMENT_CLUSTERED_QUADRANT_V1,
    }
)


def _initial_spatial_positions_v1(
    profile: str,
    *,
    width: int,
    height: int,
    count: int,
) -> tuple[tuple[int, int], ...] | None:
    profile = str(profile or INITIAL_SPATIAL_PLACEMENT_HERITAGE_V1)
    if profile == INITIAL_SPATIAL_PLACEMENT_HERITAGE_V1:
        return None
    if profile not in INITIAL_SPATIAL_PLACEMENT_PROFILES_V1:
        raise RuntimeError(f"unknown initial spatial placement profile: {profile}")
    width = int(width)
    height = int(height)
    count = int(count)
    if width <= 0 or height <= 0 or count < 0 or count > width * height:
        raise RuntimeError(
            "initial spatial placement dimensions are invalid: "
            f"profile={profile};width={width};height={height};count={count}"
        )

    if profile in {
        INITIAL_SPATIAL_PLACEMENT_DISPERSED_LATTICE_V1,
        INITIAL_SPATIAL_PLACEMENT_DISPERSED_LATTICE_FILL_V1,
    }:
        step_y = max(1, height // 8)
        step_x = max(1, width // 8)
        coordinates = [
            (y, x) for y in range(0, height, step_y) for x in range(0, width, step_x)
        ]
        if profile == INITIAL_SPATIAL_PLACEMENT_DISPERSED_LATTICE_FILL_V1:
            occupied = set(coordinates)
            coordinates.extend(
                (y, x)
                for y in range(height)
                for x in range(width)
                if (y, x) not in occupied
            )
    else:
        side = int(np.ceil(np.sqrt(count)))
        if side > width or side > height:
            raise RuntimeError(
                "clustered initial placement cannot fit the requested population"
            )
        coordinates = [(y, x) for y in range(side) for x in range(side)]

    unique = tuple(dict.fromkeys(coordinates[:count]))
    if len(unique) != count:
        raise RuntimeError(
            "initial spatial placement produced overlapping coordinates: "
            f"profile={profile};count={count};unique={len(unique)}"
        )
    return unique


class EnvSyncMixin:
    def _reject_unmigrated_world_authority(self) -> None:
        reverse = getattr(self.env, "_agent_id_to_world_slot", None)
        if not isinstance(reverse, dict) or reverse:
            return
        if not bool(np.any(np.asarray(self.env.agent_grid) != -1)):
            return
        self.env._world_authority_migration_required = True
        raise RuntimeError(
            "occupied agent_grid has no slot-keyed world authority; "
            "explicit world_state_v2_to_v3 migration is required"
        )

    def _configure_env_world_capacity(self) -> None:
        configure = getattr(self.env, "_configure_world_slot_capacity", None)
        slots = getattr(self.pop, "_slots", None)
        if not callable(configure) or slots is None:
            raise RuntimeError("envsync requires slot-authoritative world capacity")
        configure(int(getattr(slots, "max_slots", -1)))

    @staticmethod
    def _agent_world_identity(agent: object) -> tuple[int, int, int]:
        agent_id = int(getattr(agent, "agent_id", -1))
        slot = int(getattr(agent, "slot", -1))
        slot_token = int(getattr(agent, "slot_token", -1))
        if agent_id < 0 or slot < 0 or slot_token <= 0:
            raise RuntimeError(
                f"invalid population world identity: agent_id={agent_id};"
                f"slot={slot};slot_token={slot_token}"
            )
        return agent_id, slot, slot_token

    def _place_agents(self):
        """
        Reset env and place all currently-alive population agents.
        """
        self.env.reset()
        self._configure_env_world_capacity()
        if founder_spatial_separation_enabled(getattr(self, "sim_cfg", None)):
            bind = getattr(self.env, "_bind_world_slot", None)
            if not callable(bind):
                raise RuntimeError("founder placement requires world slot binding")
            for agent in self.pop.alive_agents():
                bind(*self._agent_world_identity(agent))
            place_initial_founders(
                self.pop, self.env, run_dir=str(getattr(self, "run_dir", "") or "")
            )
            return
        profile = str(
            getattr(
                getattr(self, "sim_cfg", None),
                "initial_spatial_placement_profile",
                INITIAL_SPATIAL_PLACEMENT_HERITAGE_V1,
            )
            or INITIAL_SPATIAL_PLACEMENT_HERITAGE_V1
        )
        raw_agents = list(self.pop.alive_agents())
        agents = (
            raw_agents
            if profile == INITIAL_SPATIAL_PLACEMENT_HERITAGE_V1
            else sorted(raw_agents, key=lambda item: int(getattr(item, "agent_id", -1)))
        )
        positions = _initial_spatial_positions_v1(
            profile,
            width=int(getattr(self.env.cfg, "width")),
            height=int(getattr(self.env.cfg, "height")),
            count=len(agents),
        )
        for index, a in enumerate(agents):
            # add_agent handles occupancy + random placement if needed
            agent_id, slot, slot_token = self._agent_world_identity(a)
            position = None if positions is None else positions[index]
            self.env.add_agent(
                agent_id,
                slot=slot,
                slot_token=slot_token,
                pos=position,
            )

    def _sync_env_with_population(self):
        """
        Ensure env and population agree on which agents are present.

        Rules:
        - If env has an agent id marked alive but population does not, remove from env.
        - If population has an alive agent but env is missing it, add to env.
        """
        self._reject_unmigrated_world_authority()
        self._configure_env_world_capacity()
        resolver = getattr(self.env, "_resolve_world_slot", None)
        reverse = getattr(self.env, "_agent_id_to_world_slot", None)
        if not callable(resolver) or not isinstance(reverse, dict):
            raise RuntimeError("envsync requires world membership resolver")

        residents = [
            (int(slot), int(agent_id), int(self.env.slot_tokens[slot]))
            for slot, agent_id in enumerate(self.env.slot_to_agent_id.tolist())
            if int(agent_id) != -1
        ]
        for slot, agent_id, slot_token in residents:
            resolver(agent_id, slot=slot, slot_token=slot_token)
            agent = self.pop.agents.get(agent_id)
            if agent is None or not bool(getattr(agent, "alive", False)):
                self.env.remove_agent(
                    agent_id,
                    slot=slot,
                    slot_token=slot_token,
                )

        for agent in self.pop.alive_agents():
            agent_id, slot, slot_token = self._agent_world_identity(agent)
            if agent_id not in reverse:
                self.env.add_agent(
                    agent_id,
                    slot=slot,
                    slot_token=slot_token,
                )
                continue
            resolved_slot, resolved_token = resolver(
                agent_id,
                slot=slot,
                slot_token=slot_token,
            )
            if not bool(self.env.agent_world_alive_by_slot[resolved_slot]):
                self.env.add_agent(
                    agent_id,
                    slot=resolved_slot,
                    slot_token=resolved_token,
                )
