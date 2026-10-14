from __future__ import annotations

"""
evolution/spawn.py (SpawnMixin)

What this file does:
- Creates new agents for the population (initial random spawns and children births).
- Allocates population "slots" used for dense policy banks (MLP bank and GRU bank).
- Constructs the agent policy (MLPPolicy or GRUPolicy) and attaches the Genome.

Important design invariants (banked execution):
- The GRU parameter bank stacks tensors across agents, so all GRU policies MUST share
  the same hidden_dim. We fix GRU hidden_dim at 64 for performance and simplicity.
- Evolution controls *effective* memory capacity via genome.memory_dim ∈ {0,8,16,32,64}.
  Memory capacity is enforced at inference time by masking the GRU hidden state in
  runner_policy.py, and it is paid for via per-step upkeep cost.
- Therefore: genome.memory_dim does NOT change GRU policy architecture shape.
"""

import copy
from typing import Optional

import torch
import numpy as np

from env.env_config import EnvConfig
from agents.agent import Agent, N_ACTIONS, assign_child_lineage, assign_founder_lineage
from agents.chemistry import init_chemistry_for_agent
from agents.genome import Genome
from agents.agent_policy import MLPPolicy
from agents.gru_policy import GRUPolicy
from evolution.reproduction import crossover_genome, crossover_policy


_FIXED_GRU_HIDDEN = 64


def _refresh_genome_payload_cache(genome: Genome) -> None:
    refresh = getattr(genome, "refresh_canonical_payload_cache", None)
    if callable(refresh):
        refresh()


def _copy_gru_into_fixed64(src: GRUPolicy, dst: GRUPolicy) -> None:
    """
    Copies parameters from src GRU into dst GRU, embedding smaller hidden sizes into 64.
    If src.hidden_dim == 64, this is an exact copy.
    If src.hidden_dim < 64, we copy the top-left blocks and leave the rest as dst init.
    """
    with torch.no_grad():
        hs = int(getattr(src, "hidden_dim", 0))
        hd = int(getattr(dst, "hidden_dim", 0))
        if hs <= 0 or hd <= 0:
            return
        k = min(hs, hd)

        # Encoder: (hidden, obs)
        dst.encoder.weight[:k, :].copy_(src.encoder.weight[:k, :])
        dst.encoder.bias[:k].copy_(src.encoder.bias[:k])

        # GRUCell weights:
        # weight_ih: (3*hidden, hidden)
        # weight_hh: (3*hidden, hidden)
        # biases:    (3*hidden,)
        dst.gru.weight_ih[: 3 * k, :k].copy_(src.gru.weight_ih[: 3 * k, :k])
        dst.gru.weight_hh[: 3 * k, :k].copy_(src.gru.weight_hh[: 3 * k, :k])
        dst.gru.bias_ih[: 3 * k].copy_(src.gru.bias_ih[: 3 * k])
        dst.gru.bias_hh[: 3 * k].copy_(src.gru.bias_hh[: 3 * k])

        # Head: (actions, hidden)
        dst.head.weight[:, :k].copy_(src.head.weight[:, :k])
        dst.head.bias.copy_(src.head.bias)




def _init_drives_for_agent(self, agent: Agent) -> None:
    """Initialize internal drives on spawn/childbirth.

    Drives are state, not genes. Genes affect how drives drift (inertia/noise) and
    how strongly they influence logits (drive_logit_gain).
    """
    try:
        e = float(getattr(agent, "energy", 0.0))
        emax = float(getattr(agent, "max_energy_norm", 1.0)) or 1.0
        hunger = float(np.clip(1.0 - (e / emax), 0.0, 1.0))
        base = np.asarray([hunger, 0.0, 0.0, 0.0], dtype=np.float32)

        # small spawn noise (bounded)
        noise = 0.03
        rng = getattr(self, "np_rng", None)
        if rng is None:
            raise RuntimeError(
                "[DETERMINISM_VIOLATION]\n"
                "code=DET_FORBIDDEN_RNG_API\n"
                "context=np_rng_missing_for_spawn_init_drives\n"
            )
        base = base + noise * rng.normal(0.0, 1.0, size=(4,)).astype(np.float32)

        agent.drives = np.clip(base, 0.0, 1.0).astype(np.float32, copy=False)
    except RuntimeError:
        raise
    except Exception:
        # worst case: leave default zeros
        pass


def _init_chemistry_for_agent(agent: Agent) -> None:
    try:
        agent.chemistry = init_chemistry_for_agent(agent)
    except Exception:
        pass


def _lineage_generation_context(pop: object) -> int:
    try:
        value = int(getattr(pop, "_eiel_current_generation_index", 0) or 0)
        return value if value >= 0 else 0
    except Exception:
        return 0


def _lineage_tick_context(pop: object) -> int:
    for name in ("_eiel_current_tick_index", "tick", "_pred_tick"):
        try:
            value = int(getattr(pop, name, 0) or 0)
            return value if value >= 0 else 0
        except Exception:
            continue
    return 0


class SpawnMixin:
    def _spawn_random_agent(self) -> Optional[Agent]:
        alloc = self._slots.alloc()
        if alloc is None:
            return None
        slot, token = alloc

        gid = self.next_id
        self.next_id += 1

        sim_cfg = getattr(self, "sim_cfg", None)
        founder_caps = None
        if sim_cfg is not None:
            founder_caps = {
                "max_modules_per_agent": int(getattr(sim_cfg, "genome_b2_max_modules_per_agent", 12)),
                "max_connections_per_agent": int(getattr(sim_cfg, "genome_b2_max_connections_per_agent", 32)),
                "max_module_fanin": int(getattr(sim_cfg, "genome_b2_max_module_fanin", 6)),
                "max_module_fanout": int(getattr(sim_cfg, "genome_b2_max_module_fanout", 6)),
            }
        genome = Genome.random(self.rng, b2_caps=founder_caps)

        # Treat GRU usage as derived from memory_dim/use_gru (backward compatible)
        use_gru = bool(getattr(genome, "use_gru", False)) or (int(getattr(genome, "memory_dim", 0)) > 0)

        if use_gru:
            # IMPORTANT: fixed GRU hidden size for banked execution
            policy = GRUPolicy(
                obs_size=self.obs_size,
                hidden_dim=_FIXED_GRU_HIDDEN,
                n_actions=N_ACTIONS,
            )
            # Ensure genome consistency
            if hasattr(genome, "use_gru"):
                genome.use_gru = True
        else:
            policy = MLPPolicy(
                obs_size=self.obs_size,
                hidden_size=int(getattr(genome, "hidden_size", 64)),
                n_actions=N_ACTIONS,
            )
            if hasattr(genome, "use_gru"):
                genome.use_gru = False
            if hasattr(genome, "memory_dim"):
                genome.memory_dim = 0
        _refresh_genome_payload_cache(genome)

        agent = Agent(
            agent_id=gid,
            genome=genome,
            policy=policy,
            energy=self.env_cfg.start_energy,
            health=self.env_cfg.start_health,
            max_energy_norm=self.env_cfg.max_energy,
            start_health_norm=self.env_cfg.start_health,
        )
        assign_founder_lineage(
            agent,
            generation=_lineage_generation_context(self),
            tick=_lineage_tick_context(self),
        )
        agent.slot = slot
        agent.slot_token = token

        _init_drives_for_agent(self, agent)
        _init_chemistry_for_agent(agent)

        # Reset GRU memory for this slot if GRU bank exists (safe no-op otherwise)
        if hasattr(self, "_reset_memory_slots"):
            slots_gpu = torch.tensor([int(slot)], device=getattr(self, "_bank_device", "cpu"), dtype=torch.long)
            self._reset_memory_slots(slots_gpu)

        self.agents[gid] = agent
        return agent

    def make_child(self, p1: Agent, p2: Agent, child_energy: float) -> Optional[Agent]:
        alloc = self._slots.alloc()
        if alloc is None:
            return None
        slot, token = alloc

        cid = self.next_id
        self.next_id += 1

        child_genome = crossover_genome(p1.genome, p2.genome, self.rng)

        # Determine child policy type from genome (or parents if legacy fields exist)
        child_use_gru = bool(getattr(child_genome, "use_gru", False)) or (int(getattr(child_genome, "memory_dim", 0)) > 0)

        p1_is_gru = isinstance(getattr(p1, "policy", None), GRUPolicy)
        p2_is_gru = isinstance(getattr(p2, "policy", None), GRUPolicy)

        if child_use_gru or p1_is_gru or p2_is_gru:
            # Choose a GRU parent if available; otherwise fallback to p1
            if p1_is_gru and p2_is_gru:
                chosen = p1 if self.rng.random() < 0.5 else p2
            elif p1_is_gru:
                chosen = p1
            elif p2_is_gru:
                chosen = p2
            else:
                chosen = p1

            # Build a fresh fixed-size GRU and embed chosen parent weights
            child_policy = GRUPolicy(self.obs_size, _FIXED_GRU_HIDDEN, N_ACTIONS)
            if isinstance(getattr(chosen, "policy", None), GRUPolicy):
                _copy_gru_into_fixed64(chosen.policy, child_policy)

            # Force genome flags consistent
            if hasattr(child_genome, "use_gru"):
                child_genome.use_gru = True
            # NOTE: child_genome.memory_dim remains evolvable and is NOT overwritten by policy shape.
            _refresh_genome_payload_cache(child_genome)
        else:
            # MLP child
            if p1.genome.hidden_size != p2.genome.hidden_size:
                chosen = p1 if self.rng.random() < 0.5 else p2
                child_policy = copy.deepcopy(chosen.policy)
                child_genome.hidden_size = int(chosen.genome.hidden_size)
            else:
                rm = getattr(self, "rng_manager", None)
                child_policy = crossover_policy(
                    p1.policy, p2.policy,
                    self.obs_size, int(child_genome.hidden_size), N_ACTIONS,
                    self.rng,
                    torch_rng=None,
                    rng_manager=rm,
                )
            if hasattr(child_genome, "use_gru"):
                child_genome.use_gru = False
            if hasattr(child_genome, "memory_dim"):
                child_genome.memory_dim = 0
            _refresh_genome_payload_cache(child_genome)

        child = Agent(
            agent_id=cid,
            genome=child_genome,
            policy=child_policy,
            energy=child_energy,
            health=self.env_cfg.start_health,
            max_energy_norm=self.env_cfg.max_energy,
            start_health_norm=self.env_cfg.start_health,
        )
        assign_child_lineage(
            child,
            p1,
            p2,
            generation=_lineage_generation_context(self),
            tick=_lineage_tick_context(self),
        )

        child.slot = slot
        child.slot_token = token

        _init_drives_for_agent(self, child)
        _init_chemistry_for_agent(child)

        # Reset GRU memory for this slot (safe no-op if GRU bank not initialized)
        if hasattr(self, "_reset_memory_slots"):
            slots_gpu = torch.tensor([int(slot)], device=getattr(self, "_bank_device", "cpu"), dtype=torch.long)
            self._reset_memory_slots(slots_gpu)

        self.agents[cid] = child
        return child
