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
import hashlib
import time
from dataclasses import dataclass, field
from contextlib import nullcontext
from typing import Any, Optional, Dict

import torch
import numpy as np

from agents.agent import Agent, AgentStats, N_ACTIONS, assign_child_lineage, assign_founder_lineage
from agents.chemistry import init_chemistry_for_agent, zeros_chemistry
from agents.maintenance import zeros_maintenance_state
from agents.genome import Genome
from agents.family_head_residuals import (
    FAMILY_HEAD_RESIDUAL_FIELD,
    FAMILY_HEAD_RESIDUAL_FAMILY_ORDER,
    canonical_family_head_residual_block,
)
from agents.structural_payload_activation import apply_structural_payload_activation_to_founder
from agents.agent_policy import MLPPolicy
from agents.gru_policy import GRUPolicy
from evolution.reproduction import (
    RunnerCoreC1TypedGenomeRow,
    consume_gru_policy_init_rng,
    consume_mlp_policy_init_rng,
    consume_self_policy_crossover_rng,
    crossover_genome,
    crossover_policy,
    mutate_policy_weights,
)
from evolution.inheritance_registry_v1 import (
    LINKAGE_BLOCK_IDS,
    active_inheritance_profile,
    canonical_hereditary_payload_sha256,
)
from evolution.diploid_inheritance_v1 import DIPLOID_INHERITANCE_PROFILE
from evolution.policy_lineage_telemetry import (
    cue_action_probe as policy_lineage_cue_action_probe,
    maybe_record_policy_lineage_event,
    policy_hash as policy_lineage_policy_hash,
    state_snapshot as policy_lineage_state_snapshot,
)
from knobs_override import apply_genome_overrides_to_instance


_FIXED_GRU_HIDDEN = 64


def _policy_controller_digest(policy: object) -> str:
    """Stable no-RNG digest used only to break equal-genome parent ties."""

    digest = hashlib.sha256(type(policy).__name__.encode("utf-8"))
    parameters = getattr(policy, "parameters", None)
    if callable(parameters):
        for parameter in parameters():
            tensor = parameter.detach().cpu().contiguous()
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(str(tuple(int(v) for v in tensor.shape)).encode("ascii"))
            digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _linked_memory_policy_donor(child_genome: Any, parent_a: Agent, parent_b: Agent) -> Agent:
    """Resolve the memory-block donor invariantly under caller parent swaps."""

    context = getattr(child_genome, "_recomb1_linkage_context", None)
    if not isinstance(context, dict):
        raise RuntimeError("RECOMB1-LINKAGE: child is missing transient donor context")
    block_index = LINKAGE_BLOCK_IDS.index("memory_controller_architecture_v1")
    donor_ordinals = tuple(context.get("donor_ordinals", ()))
    donor_digests = dict(context.get("donor_digests", {}))
    if len(donor_ordinals) != len(LINKAGE_BLOCK_IDS):
        raise RuntimeError("RECOMB1-LINKAGE: invalid transient donor draw schedule")
    donor_ordinal = int(donor_ordinals[block_index])
    wanted_digest = str(donor_digests.get("memory_controller_architecture_v1", ""))
    rows = [
        (canonical_hereditary_payload_sha256(parent_a.genome), parent_a),
        (canonical_hereditary_payload_sha256(parent_b.genome), parent_b),
    ]
    matching = [parent for digest, parent in rows if digest == wanted_digest]
    if len(matching) == 1:
        return matching[0]
    if len(matching) == 2:
        # Equal genome payloads can still carry different hereditary controller
        # weights.  Sort those weights, then apply the already-consumed canonical
        # donor ordinal; no additional draw is introduced.
        matching.sort(key=lambda parent: _policy_controller_digest(parent.policy))
        return matching[donor_ordinal]
    raise RuntimeError("RECOMB1-LINKAGE: policy donor digest does not match either parent")


def _clone_linked_policy_controller(policy: object) -> object:
    """Make an owned controller copy without constructor or crossover RNG."""

    child_policy = copy.deepcopy(policy)
    if child_policy is policy:
        raise RuntimeError("RECOMB1-LINKAGE: policy controller deepcopy remained aliased")
    return child_policy


def _policy_weight_mutation_effective(sim_cfg: Any) -> bool:
    return bool(
        sim_cfg is not None
        and bool(getattr(sim_cfg, "policy_weight_mutation_enable", False))
        and float(getattr(sim_cfg, "policy_weight_mutation_std", 0.0) or 0.0) > 0.0
    )


def _detach_aliased_policy_for_mutation(
    child_policy: object,
    parent_a_policy: object,
    parent_b_policy: object,
    sim_cfg: Any,
) -> tuple[object, bool]:
    if (
        _policy_weight_mutation_effective(sim_cfg)
        and (child_policy is parent_a_policy or child_policy is parent_b_policy)
    ):
        return copy.deepcopy(child_policy), True
    return child_policy, False


def _validate_linked_memory_controller(child_genome: Any, donor: Agent, policy: object) -> None:
    inherited_memory_dim = int(getattr(child_genome, "memory_dim", 0))
    inherited_use_gru = bool(getattr(child_genome, "use_gru", False))
    if isinstance(policy, GRUPolicy):
        if int(getattr(policy, "hidden_dim", 0)) != _FIXED_GRU_HIDDEN:
            raise RuntimeError(
                "RECOMB1-LINKAGE: linked GRU donor does not use the fixed-64 architecture"
            )
        if not inherited_use_gru or inherited_memory_dim <= 0:
            raise RuntimeError(
                "RECOMB1-LINKAGE: inherited GRU controller is inconsistent with the inherited memory block"
            )
        return
    if isinstance(policy, MLPPolicy):
        if int(getattr(donor.genome, "hidden_size", -1)) != int(getattr(child_genome, "hidden_size", -2)):
            raise RuntimeError(
                "RECOMB1-LINKAGE: linked MLP donor architecture does not match child genome"
            )
        if inherited_use_gru or inherited_memory_dim != 0:
            raise RuntimeError(
                "RECOMB1-LINKAGE: inherited MLP controller is inconsistent with the inherited memory block"
            )
        return
    raise RuntimeError("RECOMB1-LINKAGE: unsupported policy controller type")


def fixed_family_head_residual_block_for_arm(arm: str) -> dict[str, Any]:
    """Return the preregistered Stage A founder residual block for a fixed arm."""

    arm_name = str(arm or "").strip()
    if not arm_name or arm_name in {"historical_control", "neutral_authority_cost"}:
        return canonical_family_head_residual_block(None)

    prefix = "plus_"
    marker = "_minus_"
    if not arm_name.startswith(prefix) or marker not in arm_name:
        raise ValueError(f"unknown policy_family_head_residual_fixed_arm={arm_name!r}")
    positive, negative = arm_name[len(prefix) :].split(marker, 1)
    families = set(FAMILY_HEAD_RESIDUAL_FAMILY_ORDER)
    if positive not in families or negative not in families or positive == negative:
        raise ValueError(f"invalid policy_family_head_residual_fixed_arm={arm_name!r}")
    return canonical_family_head_residual_block({positive: 0.1, negative: -0.1})


@dataclass
class ResidentChildRowSpec:
    """Newborn state before compatibility Agent materialization."""

    agent_id: int
    genome: Genome
    policy: object
    energy: float
    health: float
    max_energy_norm: float
    start_health_norm: float
    slot: int
    slot_token: int
    alive: bool = True
    repro_cooldown: int = 0
    injury: float = 0.0
    scar: float = 0.0
    rest_streak: int = 0
    stats: AgentStats = field(default_factory=AgentStats)
    drives: np.ndarray = field(default_factory=lambda: np.zeros((4,), dtype=np.float32))
    chemistry: np.ndarray = field(default_factory=zeros_chemistry)
    maintenance: np.ndarray = field(default_factory=zeros_maintenance_state)
    _bank_token: int = -1
    _gru_bank_token: int = -1
    typed_contract_schema_version: str = "runnercore6_resident_child_row_v1"
    typed_materialization_state: str = "resident_row_spec"
    policy_kind: str = "unknown"
    policy_source_kind: str = "materialized_policy"
    policy_source_agent_id: int = -1
    policy_source_slot: int = -1
    parent_a_id: int = -1
    parent_b_id: int = -1
    rng_replay_marker: str = ""
    c1_typed_genome_row: RunnerCoreC1TypedGenomeRow | None = None
    c1_typed_projection_digest: str = ""
    c1_payload_hash: str = ""


def _refresh_genome_payload_cache(genome: Genome) -> None:
    refresh = getattr(genome, "refresh_canonical_payload_cache", None)
    if callable(refresh):
        refresh()


def _force_fixed_policy_family_genome_if_requested(genome: Genome, sim_cfg: Any) -> None:
    if not bool(getattr(sim_cfg, "native_runtime_cxx26_scope_lock_enable", False)):
        return
    family = str(
        getattr(sim_cfg, "native_runtime_fixed_policy_family", "") or ""
    ).strip().lower()
    if family == "mlp":
        if hasattr(genome, "use_gru"):
            genome.use_gru = False
        if hasattr(genome, "memory_dim"):
            genome.memory_dim = 0
    elif family == "gru" and hasattr(genome, "use_gru"):
        genome.use_gru = True


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


def _init_drives_for_agent(self, agent: Agent, parent_a: Optional[Agent] = None, parent_b: Optional[Agent] = None) -> None:
    """Initialize internal drives + prediction priors on spawn/childbirth.

    Cross-episode internal consistency:
    - Drives start near *heritable setpoints* (temperament baselines) instead of pure random reset.
    - A newborn can inherit a weak prior over per-action outcome expectations from parents.
    - Small birth noise remains evolvable (drive_birth_noise), so stability is selectable.

    This is NOT "memory". It's temperament + priors.
    """
    try:
        g = getattr(agent, "genome", None)

        # --- Drives (state) ---
        e = float(getattr(agent, "energy", 0.0))
        emax = float(getattr(agent, "max_energy_norm", 1.0)) or 1.0
        hunger_from_energy = float(np.clip(1.0 - (e / emax), 0.0, 1.0))

        sp_h = float(getattr(g, "drive_sp_hunger", 0.55))
        sp_p = float(getattr(g, "drive_sp_pain", 0.05))
        sp_f = float(getattr(g, "drive_sp_fatigue", 0.15))
        sp_r = float(getattr(g, "drive_sp_fear", 0.10))

        # Blend setpoint with immediate physiology (so starving newborns still feel hungry)
        base = np.asarray(
            [
                0.60 * sp_h + 0.40 * hunger_from_energy,
                sp_p,
                sp_f,
                sp_r,
            ],
            dtype=np.float32,
        )

        birth_noise = float(getattr(g, "drive_birth_noise", 0.03))
        birth_noise = float(np.clip(birth_noise, 0.0, 0.20))

        rng = getattr(self, "np_rng", None)
        rm = getattr(self, "rng_manager", None)
        birth_noise_draws = 4 if birth_noise > 0.0 else 0
        _budget = nullcontext()
        if rm is not None:
            expected = rm.expected_delta_for_boundary(
                "spawn.population.init_drives.birth_noise",
                ctx={"birth_noise_draws": int(birth_noise_draws)},
            )
            _budget = rm.budget_scope("spawn.population.init_drives.birth_noise", expected_delta=expected)

        with _budget:
            if birth_noise > 0.0:
                if rng is None:
                    raise RuntimeError(
                        "[DETERMINISM_VIOLATION]\n"
                        "code=DET_FORBIDDEN_RNG_API\n"
                        "context=np_rng_missing_for_population_spawn_birth_noise\n"
                    )
                # Performance: reuse small scratch buffers to avoid per-birth allocations.
                # NOTE: Generator.standard_normal(0,1) matches Generator.normal(0,1) stream exactly.
                try:
                    n = 4
                    n64 = getattr(self, "_eiel_spawn_noise64", None)
                    n32 = getattr(self, "_eiel_spawn_noise32", None)
                    if n64 is None or n32 is None or int(getattr(n64, "shape", [0])[0]) < n:
                        cap = max(n, 16)
                        n64 = np.empty((cap,), dtype=np.float64)
                        n32 = np.empty((cap,), dtype=np.float32)
                        setattr(self, "_eiel_spawn_noise64", n64)
                        setattr(self, "_eiel_spawn_noise32", n32)
                    view64 = n64[:n]
                    view32 = n32[:n]
                    rng.standard_normal(size=(n,), dtype=np.float64, out=view64)
                    view32[...] = view64  # float64 -> float32 (matches `.astype(np.float32)`)
                    base = base + birth_noise * view32
                except Exception:
                    # Fallback: original allocate+astype path.
                    base = base + birth_noise * rng.normal(0.0, 1.0, size=(4,)).astype(np.float32)

        agent.drives = np.clip(base, 0.0, 1.0).astype(np.float32, copy=False)

        # Record birth reset shock: mismatch between initial drives and heritable setpoints
        try:
            st = getattr(agent, "stats", None)
            if st is not None:
                sp = np.asarray([sp_h, sp_p, sp_f, sp_r], dtype=np.float32)
                st.birth_reset_shock = float(np.abs(agent.drives[:4] - sp).mean())
        except Exception:
            pass

        # --- Expectation priors (state) ---
        # Newborns can inherit parents' per-action expectations as weak priors.
        try:
            st = getattr(agent, "stats", None)
            if st is not None:
                # Reset lifetime accumulators
                st.pred_err_sum = 0.0
                st.pred_err_count = 0
                try:
                    st.pred_queue.clear()
                except Exception:
                    st.pred_queue = []

                # Reset drive volatility/jitter accumulators
                st.drive_volatility = 0.0
                st.drive_jitter = 0.0
                try:
                    st.prev_drive_delta[:] = 0.0
                except Exception:
                    pass

                s = float(getattr(g, "exp_prior_strength", 0.35))
                s = float(np.clip(s, 0.0, 1.0))
                pn = float(getattr(g, "exp_prior_noise", 0.05))
                pn = float(np.clip(pn, 0.0, 0.5))

                if parent_a is not None and parent_b is not None:
                    pa = getattr(getattr(parent_a, "stats", None), "exp_dE", None)
                    pb = getattr(getattr(parent_b, "stats", None), "exp_dE", None)
                    ia = getattr(getattr(parent_a, "stats", None), "exp_dInj", None)
                    ib = getattr(getattr(parent_b, "stats", None), "exp_dInj", None)
                    if pa is not None and pb is not None:
                        prior_de = 0.5 * (np.asarray(pa, dtype=np.float32) + np.asarray(pb, dtype=np.float32))
                    else:
                        prior_de = np.zeros((N_ACTIONS,), dtype=np.float32)
                    if ia is not None and ib is not None:
                        prior_dinj = 0.5 * (np.asarray(ia, dtype=np.float32) + np.asarray(ib, dtype=np.float32))
                    else:
                        prior_dinj = np.zeros((N_ACTIONS,), dtype=np.float32)
                else:
                    prior_de = np.zeros((N_ACTIONS,), dtype=np.float32)
                    prior_dinj = np.zeros((N_ACTIONS,), dtype=np.float32)

                # Blend inherited prior with neutral zero baseline.
                st.exp_dE[:] = (s * prior_de).astype(np.float32, copy=False)
                st.exp_dInj[:] = (s * prior_dinj).astype(np.float32, copy=False)

                if pn > 0.0:
                    if rng is None:
                        raise RuntimeError(
                            "[DETERMINISM_VIOLATION]\n"
                            "code=DET_FORBIDDEN_RNG_API\n"
                            "context=np_rng_missing_for_population_spawn_exp_prior_noise\n"
                        )
                    # Performance: reuse small scratch buffers to avoid per-birth allocations.
                    # Two RNG draws are required (ΔE prior noise, then ΔInj prior noise) to
                    # preserve the existing RNG consumption schedule.
                    try:
                        n = int(st.exp_dE.shape[0])
                        n64 = getattr(self, "_eiel_spawn_noise64", None)
                        n32 = getattr(self, "_eiel_spawn_noise32", None)
                        if n64 is None or n32 is None or int(getattr(n64, "shape", [0])[0]) < n:
                            cap = max(n, 16)
                            n64 = np.empty((cap,), dtype=np.float64)
                            n32 = np.empty((cap,), dtype=np.float32)
                            setattr(self, "_eiel_spawn_noise64", n64)
                            setattr(self, "_eiel_spawn_noise32", n32)

                        view64 = n64[:n]
                        view32 = n32[:n]

                        rng.standard_normal(size=(n,), dtype=np.float64, out=view64)
                        view32[...] = view64
                        st.exp_dE[:] = st.exp_dE + pn * view32

                        rng.standard_normal(size=(n,), dtype=np.float64, out=view64)
                        view32[...] = view64
                        st.exp_dInj[:] = st.exp_dInj + pn * view32
                    except Exception:
                        # Fallback: original allocate+astype path.
                        st.exp_dE[:] = st.exp_dE + pn * rng.normal(0.0, 1.0, size=st.exp_dE.shape).astype(np.float32)
                        st.exp_dInj[:] = st.exp_dInj + pn * rng.normal(0.0, 1.0, size=st.exp_dInj.shape).astype(np.float32)
        except RuntimeError:
            raise
        except Exception:
            pass
    except RuntimeError:
        raise
    except Exception:
        # worst case: leave defaults
        pass


def _init_chemistry_for_agent(agent: Agent) -> None:
    """Initialize deterministic 6-slot chemistry state."""
    try:
        agent.chemistry = init_chemistry_for_agent(agent)
    except Exception:
        pass


def _init_locality_accounting_for_agent(agent: Agent) -> None:
    """Initialize deterministic B3 locality accounting stats from genome."""
    try:
        st = getattr(agent, "stats", None)
        g = getattr(agent, "genome", None)
        if st is None or g is None:
            return
        st.structural_locality_cost_total = float(getattr(g, "b3_locality_cost_total", 0.0) or 0.0)
        st.structural_locality_mean_distance = float(getattr(g, "b3_locality_mean_distance", 0.0) or 0.0)
        st.structural_locality_max_distance = float(getattr(g, "b3_locality_max_distance", 0.0) or 0.0)
    except Exception:
        pass


def _runnercore11_stat_add_from_pop(pop: object, key: str, value: int | float = 1) -> None:
    cfg = getattr(pop, "sim_cfg", None)
    stats = getattr(cfg, "_runnercore11_c1_typed_genome_row_stats", None) if cfg is not None else None
    if not isinstance(stats, dict):
        return
    old = stats.get(str(key), 0)
    if isinstance(value, float):
        stats[str(key)] = float(old or 0.0) + float(value)
    else:
        stats[str(key)] = int(old or 0) + int(value)


def _runnercore11_reason_add_from_pop(pop: object, reason: str, value: int = 1) -> None:
    cfg = getattr(pop, "sim_cfg", None)
    stats = getattr(cfg, "_runnercore11_c1_typed_genome_row_stats", None) if cfg is not None else None
    if not isinstance(stats, dict):
        return
    reasons = stats.get("forced_materialization_reasons", {})
    if not isinstance(reasons, dict):
        reasons = {}
    reasons[str(reason)] = int(reasons.get(str(reason), 0) or 0) + int(value)
    stats["forced_materialization_reasons"] = reasons


def _set_genome_projection_gene(genome: object, name: str, value: object) -> None:
    set_gene = getattr(genome, "set_gene", None)
    if callable(set_gene):
        set_gene(str(name), value)
    else:
        setattr(genome, str(name), value)


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
    @classmethod
    def _env_capacity_array_names(cls, env: object) -> tuple[str, ...]:
        del cls, env
        return ("agent_pos_y", "agent_pos_x", "agent_alive")

    def _ensure_env_capacity_for_id(self, agent_id: int) -> None:
        """Validate identity/state capacity without growing arrays by agent ID."""
        from evolution.capacity_errors import CapacityIntegrityError, REASON_RUNTIME_ALLOCATION

        agent_id = int(agent_id)
        if agent_id < 0 or agent_id > np.iinfo(np.int32).max:
            raise CapacityIntegrityError(
                REASON_RUNTIME_ALLOCATION,
                f"agent_id_out_of_signed_int32_range={agent_id}",
            )
        env = getattr(self, "env", None)
        if env is None:
            return
        ensure = getattr(env, "_ensure_agent_capacity", None)
        if callable(ensure):
            ensure(agent_id)

        base_names = ("agent_pos_y", "agent_pos_x", "agent_alive")
        if any(getattr(env, name, None) is None for name in base_names):
            return

        array_names = self._env_capacity_array_names(env)
        missing = [name for name in array_names if getattr(env, name, None) is None]
        if missing:
            raise CapacityIntegrityError(
                REASON_RUNTIME_ALLOCATION,
                f"missing_capacity_arrays={missing}",
            )
        arrays = {name: getattr(env, name) for name in array_names}
        shapes = {name: tuple(getattr(array, "shape", ())) for name, array in arrays.items()}
        if any(len(shape) != 1 for shape in shapes.values()):
            raise CapacityIntegrityError(REASON_RUNTIME_ALLOCATION, f"invalid_array_shapes={shapes}")
        lengths = {int(shape[0]) for shape in shapes.values()}
        if len(lengths) != 1:
            raise CapacityIntegrityError(REASON_RUNTIME_ALLOCATION, f"mismatched_array_shapes={shapes}")
        _ = next(iter(lengths))

    def _reserve_env_world_slot(self, agent_id: int, slot: int, token: int) -> object:
        env = getattr(self, "env", None)
        if env is None:
            return None
        snapshot_fn = getattr(env, "_world_slot_snapshot", None)
        bind_fn = getattr(env, "_bind_world_slot", None)
        if not callable(snapshot_fn) or not callable(bind_fn):
            return None
        slot_capacity = int(getattr(getattr(self, "_slots", None), "max_slots", -1))
        configure_fn = getattr(env, "_configure_world_slot_capacity", None)
        if not callable(configure_fn):
            raise RuntimeError("slot-authoritative world cannot configure SlotPool capacity")
        configure_fn(slot_capacity)
        world_capacity = int(np.asarray(getattr(env, "slot_to_agent_id")).size)
        if world_capacity != slot_capacity:
            raise RuntimeError(
                "world/SlotPool capacity mismatch: "
                f"world_capacity={world_capacity};slotpool_capacity={slot_capacity}"
            )
        snapshot = snapshot_fn(int(slot))
        bind_fn(int(agent_id), int(slot), int(token))
        return snapshot

    def _restore_env_world_slot(self, slot: int, snapshot: object) -> None:
        if snapshot is None:
            return
        env = getattr(self, "env", None)
        restore_fn = getattr(env, "_restore_world_slot_snapshot", None)
        if not callable(restore_fn):
            raise RuntimeError("world slot snapshot cannot be restored")
        restore_fn(int(slot), snapshot)

    def _snapshot_memory_slot(self, slot: int) -> object:
        memory = getattr(self, "_Hmem", None)
        if not getattr(self, "_gru_inited", False) or memory is None:
            return None
        return memory[int(slot)].detach().clone()

    def _restore_memory_slot(self, slot: int, snapshot: object) -> None:
        if snapshot is None:
            return
        memory = getattr(self, "_Hmem", None)
        if memory is None:
            raise RuntimeError("GRU memory bank disappeared during birth rollback")
        memory[int(slot)].copy_(snapshot)

    def _spawn_random_agent(self) -> Optional[Agent]:
        """Spawn a brand-new random agent (initial population only).

        Note: this path is called during Population.init_random() and does not receive
        per-tick perf buckets. Keep it allocation-light and deterministic.
        """

        gid = int(self.next_id)
        # Validate fail-closed dimensions before consuming a reusable slot/token.
        self._ensure_env_capacity_for_id(gid)

        if not self._slots.free_slots:
            return None
        slot = int(self._slots.free_slots[0])
        token = int(self._slots.slot_tokens[slot]) + 1

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


        # Phase 1.5: apply optional genome.* overrides (from --knob genome.*=...)
        # The runner attaches sim_cfg onto Population (best-effort). If absent, no overrides apply.
        try:
            sim_cfg = getattr(self, "sim_cfg", None)
            overrides = getattr(sim_cfg, "genome_overrides", None) if sim_cfg is not None else None
            if isinstance(overrides, dict) and overrides:
                apply_genome_overrides_to_instance(genome, overrides)
        except Exception:
            # Never crash spawning due to override plumbing.
            pass
        genome = apply_structural_payload_activation_to_founder(genome, getattr(self, "sim_cfg", None))
        sim_cfg = getattr(self, "sim_cfg", None)
        if bool(getattr(sim_cfg, "policy_family_head_residual_genome_enable", False)):
            residual_arm = str(getattr(sim_cfg, "policy_family_head_residual_fixed_arm", "") or "")
            genome = Genome.from_dict(
                {
                    **genome.to_dict(),
                    FAMILY_HEAD_RESIDUAL_FIELD: fixed_family_head_residual_block_for_arm(residual_arm),
                }
            )
        _force_fixed_policy_family_genome_if_requested(
            genome, getattr(self, "sim_cfg", None)
        )

        # Treat GRU usage as derived from memory_dim/use_gru (backward compatible)
        use_gru = bool(getattr(genome, "use_gru", False)) or (int(getattr(genome, "memory_dim", 0)) > 0)

        if use_gru:
            policy = GRUPolicy(
                obs_size=self.obs_size,
                hidden_dim=_FIXED_GRU_HIDDEN,
                n_actions=N_ACTIONS,
            )
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
            slot=int(slot),
            slot_token=int(token),
        )
        assign_founder_lineage(
            agent,
            generation=_lineage_generation_context(self),
            tick=_lineage_tick_context(self),
        )
        agent.stats.patch_habitat_lineage_id = int(gid)
        agent.slot = slot
        agent.slot_token = token

        _init_drives_for_agent(self, agent, None, None)
        _init_chemistry_for_agent(agent)
        _init_locality_accounting_for_agent(agent)

        world_slot_before = None
        memory_slot_before = None
        allocated = None
        try:
            alloc = self._slots.alloc()
            allocated = alloc
            if alloc != (slot, token):
                raise RuntimeError("founder slot availability changed during construction")
            self.next_id = int(gid) + 1
            world_slot_before = self._reserve_env_world_slot(gid, int(slot), int(token))
            # Reset memory only after world reservation succeeds, so bind failure
            # cannot mutate memory-visible slot state.
            if hasattr(self, "_reset_memory_slots"):
                memory_slot_before = self._snapshot_memory_slot(int(slot))
                slots_gpu = torch.tensor(
                    [int(slot)],
                    device=getattr(self, "_bank_device", "cpu"),
                    dtype=torch.long,
                )
                self._reset_memory_slots(slots_gpu)
            self.agents[gid] = agent
        except BaseException:
            self._restore_memory_slot(int(slot), memory_slot_before)
            self._restore_env_world_slot(int(slot), world_slot_before)
            self.agents.pop(gid, None)
            self.next_id = int(gid)
            if allocated is not None:
                allocated_slot, allocated_token = allocated
                if (
                    int(self._slots.slot_tokens[int(allocated_slot)])
                    == int(allocated_token)
                ):
                    self._slots.slot_tokens[int(allocated_slot)] = (
                        int(allocated_token) - 1
                    )
                    self._slots.free_slots.appendleft(int(allocated_slot))
            raise
        return agent

    def make_child_row_spec(
        self,
        p1: Agent,
        p2: Agent,
        child_energy: float,
        buckets: Optional[Dict[str, float]] = None,
    ) -> Optional[ResidentChildRowSpec]:
        b = buckets if isinstance(buckets, dict) else None

        _t_alloc0 = time.perf_counter()
        if not self._slots.free_slots:
            if b is not None:
                b["make_child_alloc_slot"] = b.get("make_child_alloc_slot", 0.0) + (
                    time.perf_counter() - _t_alloc0
                )
            return None

        cid = self.next_id

        sim_cfg = getattr(self, "sim_cfg", None)
        if sim_cfg is not None:
            try:
                setattr(sim_cfg, "_mutation_weather_generation", int(getattr(self, "_eiel_current_generation_index", 0) or 0))
            except Exception:
                pass
        runnercore11_typed_projection = bool(
            sim_cfg is not None
            and getattr(sim_cfg, "runnercore11_candidate_enable", False)
            and getattr(sim_cfg, "runnercore11_c1_typed_rows_enable", False)
        )

        _t_genome0 = time.perf_counter()
        child_genome = crossover_genome(
            p1.genome,
            p2.genome,
            self.rng,
            rng_manager=getattr(self, "rng_manager", None),
            sim_cfg=sim_cfg,
            buckets=b,
            return_typed_projection=runnercore11_typed_projection,
            structural_identity_context={
                "campaign_id": str(getattr(sim_cfg, "campaign_id", "") or "") if sim_cfg is not None else "",
                "branch_id": str(getattr(sim_cfg, "branch_id", "default") or "default") if sim_cfg is not None else "default",
                "generation": int(getattr(self, "_eiel_current_generation_index", 0) or 0),
                "tick": int(_lineage_tick_context(self)),
                "child_agent_id": int(cid),
                "parent_agent_ids": [int(getattr(p1, "agent_id", -1)), int(getattr(p2, "agent_id", -1))],
            },
        )
        _force_fixed_policy_family_genome_if_requested(child_genome, sim_cfg)
        if b is not None:
            b["make_child_genome"] = b.get("make_child_genome", 0.0) + (time.perf_counter() - _t_genome0)

        _t_policy0 = time.perf_counter()

        child_use_gru = bool(getattr(child_genome, "use_gru", False)) or (int(getattr(child_genome, "memory_dim", 0)) > 0)

        p1_is_gru = isinstance(getattr(p1, "policy", None), GRUPolicy)
        p2_is_gru = isinstance(getattr(p2, "policy", None), GRUPolicy)
        policy_donor_parent = None
        policy_construction_kind = "unknown"

        linked_inheritance = (
            active_inheritance_profile(sim_cfg) == DIPLOID_INHERITANCE_PROFILE
        )
        if linked_inheritance:
            chosen = _linked_memory_policy_donor(child_genome, p1, p2)
            chosen_policy = getattr(chosen, "policy", None)
            if not isinstance(chosen_policy, (MLPPolicy, GRUPolicy)):
                raise RuntimeError(
                    "RECOMB1-LINKAGE: memory donor has no supported MLP/GRU policy controller"
                )
            child_policy = _clone_linked_policy_controller(chosen_policy)
            policy_donor_parent = chosen
            _validate_linked_memory_controller(child_genome, chosen, child_policy)
            if isinstance(child_policy, GRUPolicy):
                policy_construction_kind = "recomb1_linked_gru_controller_copy"
            else:
                policy_construction_kind = "recomb1_linked_mlp_controller_copy"
        elif child_use_gru or p1_is_gru or p2_is_gru:
            if p1_is_gru and p2_is_gru:
                chosen = p1 if self.rng.random() < 0.5 else p2
            elif p1_is_gru:
                chosen = p1
            elif p2_is_gru:
                chosen = p2
            else:
                chosen = p1
            policy_donor_parent = chosen
            policy_construction_kind = "gru_single_parent_copy"

            runnercore7_gru_alias = bool(
                p1 is p2
                and chosen is p1
                and isinstance(getattr(chosen, "policy", None), GRUPolicy)
                and sim_cfg is not None
                and getattr(sim_cfg, "runnercore7_self_gru_policy_source_alias_enable", False)
            )
            if runnercore7_gru_alias:
                consume_gru_policy_init_rng(
                    obs_size=self.obs_size,
                    hidden_size=_FIXED_GRU_HIDDEN,
                    n_actions=N_ACTIONS,
                    rng_manager=getattr(self, "rng_manager", None),
                    sim_cfg=sim_cfg,
                    buckets=b,
                )
                child_policy = chosen.policy
                policy_construction_kind = "gru_parent_alias"
            else:
                child_policy = GRUPolicy(self.obs_size, _FIXED_GRU_HIDDEN, N_ACTIONS)
                if isinstance(getattr(chosen, "policy", None), GRUPolicy):
                    _copy_gru_into_fixed64(chosen.policy, child_policy)

            if hasattr(child_genome, "use_gru"):
                _set_genome_projection_gene(child_genome, "use_gru", True)
            _refresh_genome_payload_cache(child_genome)
        else:
            if p1.genome.hidden_size != p2.genome.hidden_size:
                chosen = p1 if self.rng.random() < 0.5 else p2
                child_policy = copy.deepcopy(chosen.policy)
                _set_genome_projection_gene(child_genome, "hidden_size", int(chosen.genome.hidden_size))
                policy_donor_parent = chosen
                policy_construction_kind = "mlp_single_parent_copy_hidden_mismatch"
            else:
                rm = getattr(self, "rng_manager", None)
                runnercore7_alias = bool(
                    p1 is p2
                    and isinstance(getattr(p1, "policy", None), MLPPolicy)
                    and isinstance(getattr(p2, "policy", None), MLPPolicy)
                    and sim_cfg is not None
                    and getattr(sim_cfg, "runnercore7_self_mlp_policy_source_alias_enable", False)
                    and getattr(sim_cfg, "runnercore5_flat_self_policy_rng_enable", False)
                    and int(getattr(child_genome, "hidden_size", -1)) == int(getattr(p1.genome, "hidden_size", -2))
                    and int(getattr(child_genome, "hidden_size", -1)) == int(getattr(p2.genome, "hidden_size", -3))
                )
                if runnercore7_alias:
                    consume_mlp_policy_init_rng(
                        obs_size=self.obs_size,
                        hidden_size=int(child_genome.hidden_size),
                        n_actions=N_ACTIONS,
                        rng_manager=rm,
                        sim_cfg=sim_cfg,
                        buckets=b,
                    )
                    consume_self_policy_crossover_rng(
                        p1.policy,
                        rng_manager=rm,
                        sim_cfg=sim_cfg,
                        buckets=b,
                    )
                    child_policy = p1.policy
                    policy_donor_parent = p1
                    policy_construction_kind = "mlp_parent_alias"
                else:
                    child_policy = crossover_policy(
                        p1.policy, p2.policy,
                        self.obs_size, int(child_genome.hidden_size), N_ACTIONS,
                        self.rng,
                        torch_rng=None,
                        rng_manager=rm,
                        sim_cfg=sim_cfg,
                        self_fastpath=(
                            bool(getattr(sim_cfg, "runnercore3_self_recombination_fastpath_enable", False))
                            or bool(getattr(sim_cfg, "runnercore4_structural_payload_cache_enable", False))
                            or bool(getattr(sim_cfg, "runnercore5_flat_self_policy_rng_enable", False))
                        )
                        and p1 is p2,
                        buckets=b,
                    )
                    policy_donor_parent = p1
                    policy_construction_kind = "mlp_parameter_crossover_parent_a_reference"
            if hasattr(child_genome, "use_gru"):
                _set_genome_projection_gene(child_genome, "use_gru", False)
            if hasattr(child_genome, "memory_dim"):
                _set_genome_projection_gene(child_genome, "memory_dim", 0)
            _refresh_genome_payload_cache(child_genome)

        child_policy, detached_for_mutation = _detach_aliased_policy_for_mutation(
            child_policy,
            getattr(p1, "policy", None),
            getattr(p2, "policy", None),
            sim_cfg,
        )
        if detached_for_mutation:
            # Some legacy resident-row fast paths deliberately alias an exact
            # parent controller after replaying the historical RNG side effects.
            # Weight mutation is in-place, so detach without RNG before mutation.
            policy_construction_kind = f"{policy_construction_kind}_detached_for_mutation"

        telemetry_enabled = bool(getattr(sim_cfg, "policy_lineage_telemetry_enable", False)) if sim_cfg is not None else False
        pre_mutation_policy_hash = policy_lineage_policy_hash(child_policy) if telemetry_enabled else ""
        pre_mutation_policy_snapshot = policy_lineage_state_snapshot(child_policy) if telemetry_enabled else {}
        pre_mutation_cue_probe = (
            policy_lineage_cue_action_probe(child_policy, obs_size=int(getattr(self, "obs_size", 1)))
            if telemetry_enabled
            else {}
        )
        mutation_summary = mutate_policy_weights(
            child_policy,
            rng_manager=getattr(self, "rng_manager", None),
            sim_cfg=sim_cfg,
            buckets=b,
        )

        if b is not None:
            b["make_child_policy"] = b.get("make_child_policy", 0.0) + (time.perf_counter() - _t_policy0)

        _t_row0 = time.perf_counter()
        if isinstance(child_policy, GRUPolicy):
            policy_kind = "gru"
        elif isinstance(child_policy, MLPPolicy):
            policy_kind = "mlp"
        else:
            policy_kind = type(child_policy).__name__ if child_policy is not None else "none"
        if child_policy is getattr(p1, "policy", None):
            policy_source_kind = "parent_a_alias"
            policy_source_agent_id = int(getattr(p1, "agent_id", -1))
            policy_source_slot = int(getattr(p1, "slot", -1))
        elif child_policy is getattr(p2, "policy", None):
            policy_source_kind = "parent_b_alias"
            policy_source_agent_id = int(getattr(p2, "agent_id", -1))
            policy_source_slot = int(getattr(p2, "slot", -1))
        else:
            policy_source_kind = "materialized_policy"
            policy_source_agent_id = -1
            policy_source_slot = -1
        if isinstance(child_genome, RunnerCoreC1TypedGenomeRow):
            try:
                child_genome.policy_kind = str(policy_kind)
                child_genome.policy_source_kind = str(policy_source_kind)
            except Exception:
                pass
        child = ResidentChildRowSpec(
            agent_id=cid,
            genome=child_genome,
            policy=child_policy,
            energy=child_energy,
            health=self.env_cfg.start_health,
            max_energy_norm=self.env_cfg.max_energy,
            start_health_norm=self.env_cfg.start_health,
            slot=-1,
            slot_token=-1,
            typed_contract_schema_version="runnercore10_typed_child_contract_v1",
            policy_kind=str(policy_kind),
            policy_source_kind=str(policy_source_kind),
            policy_source_agent_id=int(policy_source_agent_id),
            policy_source_slot=int(policy_source_slot),
            parent_a_id=int(getattr(p1, "agent_id", -1)),
            parent_b_id=int(getattr(p2, "agent_id", -1)),
            rng_replay_marker=str(policy_source_kind),
            c1_typed_genome_row=(
                child_genome if isinstance(child_genome, RunnerCoreC1TypedGenomeRow) else None
            ),
            c1_typed_projection_digest=(
                str(getattr(child_genome, "projection_digest", ""))
                if isinstance(child_genome, RunnerCoreC1TypedGenomeRow)
                else ""
            ),
            c1_payload_hash=(
                str(getattr(child_genome, "c1_payload_hash", ""))
                if isinstance(child_genome, RunnerCoreC1TypedGenomeRow)
                else ""
            ),
        )
        assign_child_lineage(
            child,
            p1,
            p2,
            generation=_lineage_generation_context(self),
            tick=_lineage_tick_context(self),
        )
        try:
            l1 = int(getattr(getattr(p1, "stats", None), "patch_habitat_lineage_id", int(getattr(p1, "agent_id", cid))) or int(getattr(p1, "agent_id", cid)))
            l2 = int(getattr(getattr(p2, "stats", None), "patch_habitat_lineage_id", int(getattr(p2, "agent_id", cid))) or int(getattr(p2, "agent_id", cid)))
            child.stats.patch_habitat_lineage_id = int(min(l1, l2))
        except Exception:
            child.stats.patch_habitat_lineage_id = int(cid)
        _t_alloc0 = time.perf_counter()
        alloc = self._slots.alloc()
        if b is not None:
            b["make_child_alloc_slot"] = b.get("make_child_alloc_slot", 0.0) + (
                time.perf_counter() - _t_alloc0
            )
        if alloc is None:
            raise RuntimeError("child slot availability changed during construction")
        slot, token = alloc

        env = getattr(self, "env", None)
        env_arrays_before = {
            name: getattr(env, name, None)
            for name in self._env_capacity_array_names(env)
        } if env is not None else {}
        world_slot_before = None
        memory_slot_before = None
        try:
            self._ensure_env_capacity_for_id(cid)
            world_slot_before = self._reserve_env_world_slot(cid, int(slot), int(token))
            child.slot = slot
            child.slot_token = token

            _init_drives_for_agent(self, child, p1, p2)
            _init_chemistry_for_agent(child)
            _init_locality_accounting_for_agent(child)

            if hasattr(self, "_reset_memory_slots"):
                memory_slot_before = self._snapshot_memory_slot(int(slot))
                slots_gpu = torch.tensor(
                    [int(slot)],
                    device=getattr(self, "_bank_device", "cpu"),
                    dtype=torch.long,
                )
                self._reset_memory_slots(slots_gpu)

            self.next_id = int(cid) + 1
            maybe_record_policy_lineage_event(
                self,
                p1=p1,
                p2=p2,
                child=child,
                policy_donor=policy_donor_parent,
                construction_kind=policy_construction_kind,
                pre_mutation_hash=pre_mutation_policy_hash,
                pre_mutation_snapshot=pre_mutation_policy_snapshot,
                pre_mutation_probe=pre_mutation_cue_probe,
                mutation_summary=mutation_summary,
            )
            try:
                gen1_f1 = getattr(self, "gen1_f1_capture", None)
                if gen1_f1 is not None:
                    gen1_f1.record_birth(
                        parent_a=p1,
                        parent_b=p2,
                        child=child,
                        generation=_lineage_generation_context(self),
                        step=_lineage_tick_context(self),
                        population_size=len(self.agents) + 1,
                    )
            except Exception:
                # GEN1-F1 is passive observability and must never interrupt birth.
                pass
        except BaseException:
            self.next_id = int(cid)
            self._restore_memory_slot(int(slot), memory_slot_before)
            self._restore_env_world_slot(int(slot), world_slot_before)
            if env is not None:
                for name, value in env_arrays_before.items():
                    if value is not None:
                        setattr(env, name, value)
            self._slots.slot_tokens[int(slot)] = int(token) - 1
            self._slots.free_slots.appendleft(int(slot))
            raise

        if b is not None:
            b["make_child_row_spec_setup"] = b.get("make_child_row_spec_setup", 0.0) + (time.perf_counter() - _t_row0)
        return child

    def materialize_child_row_spec(self, spec: ResidentChildRowSpec, *, materialize_genome: bool = True) -> Agent:
        genome = spec.genome
        if isinstance(genome, RunnerCoreC1TypedGenomeRow):
            if bool(materialize_genome):
                genome = genome.materialize_genome()
                spec.genome = genome
                try:
                    spec.c1_typed_genome_row = None
                    spec.typed_materialization_state = "compatibility_genome_materialized"
                except Exception:
                    pass
                _runnercore11_stat_add_from_pop(self, "genome_from_dict_forced", 1)
                _runnercore11_stat_add_from_pop(self, "genome_materialization_forced", 1)
                _runnercore11_stat_add_from_pop(self, "compatibility_export_forced", 1)
                _runnercore11_reason_add_from_pop(self, "materialize_child_row_spec", 1)
            else:
                try:
                    spec.typed_materialization_state = "compatibility_proxy_agent_typed_genome"
                except Exception:
                    pass
                _runnercore11_stat_add_from_pop(self, "generation_end_proxy_agents_created", 1)
                _runnercore11_stat_add_from_pop(self, "generation_end_proxy_genome_materialization_avoided", 1)
        child = Agent(
            agent_id=int(spec.agent_id),
            genome=genome,
            policy=spec.policy,
            energy=float(spec.energy),
            health=float(spec.health),
            alive=bool(spec.alive),
            stats=spec.stats,
            repro_cooldown=int(spec.repro_cooldown),
            max_energy_norm=float(spec.max_energy_norm),
            start_health_norm=float(spec.start_health_norm),
            injury=float(spec.injury),
            rest_streak=int(spec.rest_streak),
            slot=int(spec.slot),
            slot_token=int(spec.slot_token),
            drives=np.asarray(spec.drives, dtype=np.float32).copy(),
            chemistry=np.asarray(spec.chemistry, dtype=np.float32).copy(),
            maintenance=np.asarray(spec.maintenance, dtype=np.float32).copy(),
        )
        child.scar = float(getattr(spec, "scar", 0.0))
        for attr in ("_bank_token", "_gru_bank_token", "_eiel_force_full_mlp_bank_sync_once"):
            if hasattr(spec, attr):
                try:
                    setattr(child, attr, getattr(spec, attr))
                except Exception:
                    pass
        for attr in (
            "typed_contract_schema_version",
            "typed_materialization_state",
            "policy_kind",
            "policy_source_kind",
            "policy_source_agent_id",
            "policy_source_slot",
            "parent_a_id",
            "parent_b_id",
            "rng_replay_marker",
            "c1_typed_projection_digest",
            "c1_payload_hash",
        ):
            if hasattr(spec, attr):
                try:
                    setattr(child, attr, getattr(spec, attr))
                except Exception:
                    pass
        if str(getattr(child, "typed_materialization_state", "")) == "resident_row_spec":
            child.typed_materialization_state = "compatibility_agent_materialized"
        self.agents[int(child.agent_id)] = child
        return child

    def make_child(self, p1: Agent, p2: Agent, child_energy: float, buckets: Optional[Dict[str, float]] = None) -> Optional[Agent]:
        b = buckets if isinstance(buckets, dict) else None
        _t_agent0 = time.perf_counter()
        spec = self.make_child_row_spec(p1, p2, child_energy, buckets=b)
        if spec is None:
            return None
        child = self.materialize_child_row_spec(spec)
        if b is not None:
            b["make_child_agent_setup"] = b.get("make_child_agent_setup", 0.0) + (time.perf_counter() - _t_agent0)
        return child
