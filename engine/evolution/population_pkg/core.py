from __future__ import annotations

"""evolution/population_pkg/core.py

Population container + evolutionary turnover.

This class:
- owns the agent dictionary
- spawns initial random agents
- removes dead agents and frees their bank slots
- performs reproduction + culling to maintain target population size

Identity Pressure integration
----------------------------
The simulation runner performs per-step bookkeeping for coherence signals:
- action thrashing (switches/oscillations)
- drive volatility (internal instability)
- self-harm events (injury gained after risky acts)
- prediction error (expected vs actual deltas)

Selection must actually *use* these signals, otherwise they're just pretty logs.

We therefore rank agents by an **effective fitness**:
    effective_fitness = base_fitness - coherence_penalties

Base fitness remains defined in evolution/selection.py and should stay comparable
across runs. Penalties are a pressure instrument, not a new "goal".

Note:
Population does not *require* sim_cfg, but if the runner sets self.sim_cfg on the
Population, penalty weights will be taken from it. Otherwise safe defaults apply.
"""

from typing import Any, Dict, List, Optional
import hashlib
import json
import random
import numpy as np

from env.env_config import EnvConfig
from env.gridworld_pkg.corridor_cp0 import corridor_cp0_central_place_eligibility_enabled, corridor_cp0_filter_home_eligible_agents
from env.gridworld_pkg.patch_world import patch_world_total_observation_size
from agents.agent import Agent, format_lineage_id, normalize_lineage_stats
from evolution.selection import fitness as base_fitness
from evolution.population_pkg.population_config import PopConfig
from evolution.population_pkg.slots import SlotPool
from evolution.population_pkg.spawn import SpawnMixin
from evolution.population_pkg.founder_balance import apply_founder_balance_initialization
from evolution.inheritance_registry_v1 import active_inheritance_profile


def _lineage_funnel_enabled(sim_cfg: Any) -> bool:
    return bool(getattr(sim_cfg, "lineage_reproduction_funnel_reporting_enable", False)) if sim_cfg is not None else False


def _structural_selection_funnel_enabled(sim_cfg: Any) -> bool:
    return bool(getattr(sim_cfg, "structural_selection_funnel_reporting_enable", False)) if sim_cfg is not None else False


def _lineage_id(agent: Any) -> str:
    try:
        aid = int(getattr(agent, "agent_id", 0))
    except Exception:
        aid = 0
    stats = normalize_lineage_stats(getattr(agent, "stats", None), aid, generation=0, tick=0)
    if stats is None:
        return format_lineage_id(aid)
    return str(getattr(stats, "lineage_id", "") or format_lineage_id(aid))


def _lineage_counts(agents: List[Any]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for agent in list(agents or []):
        if not bool(getattr(agent, "alive", True)):
            continue
        lid = _lineage_id(agent)
        out[lid] = out.get(lid, 0) + 1
    return dict(sorted(out.items()))


class _EIELTraitInjectionSimProxy:
    """Minimal sim proxy for trait injection integration.

    The trait injection system expects a `sim` object with:
    - sim.run_dir: str
    - sim.sim_cfg: config-like object (may define trait_injection_apply_phase)
    - sim.pop: population-like object exposing alive_agents() and agents dict

    We keep this proxy on the Population instance so the injection manager can
    store idempotency caches on it across generations (Phase 5B tests rely on this).
    """

    def __init__(self, *, run_dir: str, pop: Any, sim_cfg: Any):
        self.run_dir = str(run_dir)
        self.pop = pop
        self.sim_cfg = sim_cfg

class Population(SpawnMixin):
    def __init__(
        self,
        env_cfg: EnvConfig,
        pop_cfg: PopConfig,
        seed: int = 0,
        rng: Optional[random.Random] = None,
        np_rng: Optional[np.random.Generator] = None,
    ):
        self.env_cfg = env_cfg
        self.pop_cfg = pop_cfg
        # Phase 1: prefer RNG injection from RNGManager named streams.
        # Phase 1 determinism rule: no RNG islands outside RNGManager.
        # In debug mode, require explicit rng injection.
        if rng is None:
            raise RuntimeError("Population requires rng injection from RNGManager (no RNG islands allowed)")
        self.rng = rng
        # Avoid np.random.default_rng here when determinism guard is enabled.
        if np_rng is None:
            raise RuntimeError("Population requires np_rng injection from RNGManager (no RNG islands allowed)")
        self.np_rng = np_rng
        self.next_id = 1
        self.agents: Dict[int, Agent] = {}

        self.obs_size = patch_world_total_observation_size(env_cfg)

        self._slots = SlotPool.build(
            base_population=int(pop_cfg.population_size),
            slot_factor=int(getattr(pop_cfg, "slot_factor", 6)),
            max_slots_cap=int(getattr(pop_cfg, "max_slots_cap", 0) or 0),
        )
        self.max_slots = int(self._slots.max_slots)

        # Optional: runner may attach sim_cfg here for coherent penalty weights
        self.sim_cfg = None  # type: ignore[assignment]

    def init_random(self):
        # A founder reinitialization resets the SlotPool token epoch.  Keep the
        # slot-authoritative world in the same epoch before reserving the new
        # founder rows; otherwise a later bind observes the previous epoch's
        # token and correctly rejects the reset token as stale.
        env = getattr(self, "env", None)
        reset_env = getattr(env, "reset", None)
        if callable(reset_env):
            reset_env()
        self.agents.clear()
        self.next_id = 1
        self._slots.reset()

        for _ in range(self.pop_cfg.population_size):
            a = self._spawn_random_agent()
            if a is None:
                raise RuntimeError(
                    f"init_random: No free agent slots available (max_slots={self.max_slots}). "
                    f"Increase PopConfig.slot_factor."
                )

        apply_founder_balance_initialization(self)
        if bool(
            getattr(getattr(self, "sim_cfg", None), "recomb72_audit_enable", False)
        ):
            self.apply_recomb72_founder_packages()

        # Determinism: normalize dict insertion order by agent_id.
        self._canonicalize_agents_dict()

    def _canonicalize_agents_dict(self) -> None:
        """Rebuild the agents dict in ascending agent_id order.

        Determinism rationale
        --------------------
        Python dict iteration is insertion-ordered. If agents are inserted
        in a different order between fresh vs resume paths, downstream loops
        that iterate `self.agents.values()` (metrics, bookkeeping, etc.) can
        produce floating-point drift or consume RNG in a different order.
        """
        try:
            items = sorted(self.agents.items(), key=lambda kv: int(kv[0]))
            self.agents = {k: v for k, v in items}
        except Exception:
            try:
                self.agents = dict(sorted(self.agents.items(), key=lambda kv: int(kv[0])))
            except Exception:
                pass

    def alive_agents(self) -> List[Agent]:
        """Return alive agents in a stable order.

        Determinism rationale
        --------------------
        Many downstream systems iterate the alive list and consume RNG
        (tick loop, reproduction/mating, selection tie-breaks, metric
        aggregation). If iteration order depends on dict insertion order,
        then any upstream unordered insertion (directly or indirectly from
        an unordered container) can shift RNG consumption and cause run
        divergence.

        We therefore enforce a stable ordering by agent_id.
        """
        alive = [a for a in self.agents.values() if a.alive]
        alive.sort(key=lambda a: int(getattr(a, "agent_id", 0)))
        return alive

    def remove_dead(self):
        """Remove dead agents and free their bank slots in a stable order.

        Determinism rationale
        --------------------
        SlotPool.free() appends to a deque, so the *order* of frees becomes
        the future allocation order. If dead-agent removal order varies,
        future slot allocation order varies, which can change policy-bank
        packing and downstream decisions.

        We therefore free slots sorted by agent_id.
        """
        dead: List[Agent] = [a for a in self.agents.values() if not a.alive]
        dead.sort(key=lambda a: int(getattr(a, "agent_id", 0)))
        for a in dead:
            agent_id = int(getattr(a, "agent_id", -1))
            slot = int(getattr(a, "slot", -1))
            slot_token = int(getattr(a, "slot_token", -1))
            try:
                from simulation.membership_trace import note_write  # noqa: WPS433
                note_write(kind="pop.remove_dead", agent_id=agent_id, detail="")
            except Exception:
                pass
            env = getattr(self, "env", None)
            remove_agent = getattr(env, "remove_agent", None)
            if not callable(remove_agent):
                raise RuntimeError(
                    f"cannot unpublish dead agent before slot reuse: agent_id={agent_id}"
                )
            remove_agent(agent_id, slot=slot, slot_token=slot_token)
            self.agents.pop(agent_id, None)
            self._slots.free(slot)

    # -----------------------------
    # Selection ranking
    # -----------------------------
    def _effective_fitness(self, a: Agent) -> float:
        f = float(base_fitness(a))
        st = getattr(a, "stats", None)
        if st is None:
            return f

        sim_cfg = getattr(self, "sim_cfg", None)

        thrash_w = float(getattr(sim_cfg, "coherence_thrashing_weight", 0.15))
        drive_w = float(getattr(sim_cfg, "coherence_drive_volatility_weight", 0.08))
        selfharm_w = float(getattr(sim_cfg, "coherence_selfharm_weight", 3.0))
        pred_w = float(getattr(sim_cfg, "coherence_pred_err_weight", 1.25))

        switches = float(getattr(st, "action_switches", 0))
        osc = float(getattr(st, "action_oscillations", 0))
        thrash = 0.5 * switches + 1.0 * osc

        drive_vol = float(getattr(st, "drive_volatility", 0.0))
        selfharm = float(getattr(st, "selfharm_events", 0))

        pe_sum = float(getattr(st, "pred_err_sum", 0.0))
        pe_n = float(getattr(st, "pred_err_count", 0))
        pred_err_mean = pe_sum / max(1.0, pe_n)

        return f - thrash_w * thrash - drive_w * drive_vol - selfharm_w * selfharm - pred_w * pred_err_mean

    def _rank_agents_effective(self, agents: List[Agent]) -> List[Agent]:
        """Return agents ranked by effective fitness (best -> worst).

        Determinism note (FDP): Python sort is stable, so equal-fitness ties preserve input order.
        Fresh vs resume can reconstruct agent list order differently, which would otherwise change
        parent selection and cause generation-transition drift.
        """
        if not agents:
            return []

        # Key by agent_id to avoid per-process object id differences.
        scores: Dict[int, float] = {int(a.agent_id): float(self._effective_fitness(a)) for a in agents}

        # Sort by fitness descending, then agent_id ascending for a deterministic tie-break.
        return sorted(agents, key=lambda a: (-scores[int(a.agent_id)], int(a.agent_id)))

    def _maybe_apply_trait_injection_post_selection_pre_mutation(self) -> List[str]:
        """Apply Phase 5B trait injections at the post-selection-pre-mutation phase.

        This is a **best-effort** hook: any exception becomes a no-op.

        Returns
        -------
        List[str]
            The list of command UUIDs applied (may be empty).
        """
        try:
            gen_idx = int(getattr(self, "_eiel_current_generation_index", -1))
            run_dir = str(getattr(self, "run_dir", "") or "")
            if gen_idx < 0 or not run_dir:
                return []

            from simulation.trait_injection_pkg import apply_trait_injections  # noqa: WPS433

            # Reuse a stable proxy so the injection manager can keep idempotency caches.
            proxy = getattr(self, "_trait_injection_sim_proxy", None)
            if proxy is None or not hasattr(proxy, "pop"):
                proxy = _EIELTraitInjectionSimProxy(run_dir=run_dir, pop=self, sim_cfg=getattr(self, "sim_cfg", None))
                setattr(self, "_trait_injection_sim_proxy", proxy)
            else:
                # Keep pointers fresh (defensive).
                try:
                    proxy.run_dir = run_dir
                    proxy.pop = self
                    proxy.sim_cfg = getattr(self, "sim_cfg", None)
                except Exception:
                    pass

            applied = apply_trait_injections(
                sim=proxy,
                generation_index=gen_idx,
                apply_phase="post-selection-pre-mutation",
            )

            try:
                setattr(self, "_trait_injection_last_applied_uuids", list(applied))
            except Exception:
                pass

            return list(applied)
        except Exception:
            return []

    def apply_recomb72_founder_packages(self) -> dict[str, Any]:
        """Install the frozen 32/32 hereditary founder package without RNG."""

        from evolution.recomb72_inheritance import (
            AUDITED_LOCI,
            founder_packages,
            registry_sha256,
        )
        from agents.modular_genome_v1 import canon_float

        sim_cfg = getattr(self, "sim_cfg", None)
        if not bool(getattr(sim_cfg, "recomb72_audit_enable", False)):
            raise RuntimeError("RECOMB72_AUDIT_CONTRACT: audit mode is disabled")
        alive = self.alive_agents()
        if len(alive) != 64:
            raise RuntimeError(
                "RECOMB72_AUDIT_CONTRACT: founder initialization requires 64 agents"
            )
        self._validate_recomb72_audit_contract(alive)
        package_a, package_b = founder_packages(
            str(getattr(sim_cfg, "recomb72_founder_pattern", "") or "")
        )
        assignments: list[dict[str, Any]] = []
        for ordinal, agent in enumerate(alive):
            group = "a" if ordinal < 32 else "b"
            package = package_a if group == "a" else package_b
            for name in AUDITED_LOCI:
                setattr(
                    agent.genome,
                    name,
                    canon_float(float(package[name]), field=name),
                )
            agent.genome.mutation_rate = 0.0
            refresh = getattr(agent.genome, "refresh_canonical_payload_cache", None)
            if callable(refresh):
                refresh()
            setattr(agent, "_recomb72_founder_group", group)
            setattr(
                agent,
                "_recomb72_ancestry",
                {f"founder_{ordinal:02d}": 1.0},
            )
            assignments.append(
                {
                    "agent_id": int(agent.agent_id),
                    "founder_group": group,
                    "package": dict(package),
                }
            )
        receipt = {
            "schema_version": "recomb72_founder_assignment_v1",
            "pattern": str(sim_cfg.recomb72_founder_pattern),
            "registry_sha256": registry_sha256(),
            "founder_split": {"a": 32, "b": 32},
            "assignments": assignments,
        }
        self._recomb72_founder_receipt = receipt
        return receipt

    @staticmethod
    def _recomb72_schedule_word(*parts: Any) -> int:
        payload = json.dumps(
            list(parts),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
        return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")

    def _validate_recomb72_audit_contract(self, alive: List[Agent]) -> None:
        from evolution.recomb72_inheritance import INHERITANCE_MODES

        sim_cfg = getattr(self, "sim_cfg", None)
        dimensions = (
            int(getattr(self.env_cfg, "width", 0) or 0),
            int(getattr(self.env_cfg, "height", 0) or 0),
        )
        generations = int(getattr(sim_cfg, "generations", 0) or 0)
        ticks = int(getattr(sim_cfg, "steps_per_generation", 0) or 0)
        mode = str(getattr(sim_cfg, "recomb72_inheritance_mode", "") or "")
        if (
            dimensions != (16, 16)
            or int(self.pop_cfg.population_size) != 64
            or len(alive) != 64
            or not 1 <= generations <= 30
            or ticks != 80
            or mode not in INHERITANCE_MODES
            or not bool(getattr(sim_cfg, "recomb72_mutation_off", False))
            or not bool(
                getattr(sim_cfg, "recomb72_disable_ordinary_tick_births", False)
            )
            or not bool(
                getattr(sim_cfg, "recomb72_neutral_actions_enable", False)
            )
            or int(getattr(sim_cfg, "recomb72_neutral_action_id", -1)) != 5
            or not bool(
                getattr(
                    sim_cfg,
                    "recomb72_neutral_environment_enable",
                    False,
                )
            )
            or bool(getattr(sim_cfg, "policy_weight_mutation_enable", False))
            or bool(getattr(sim_cfg, "mutation_weather_enabled", False))
        ):
            raise RuntimeError(
                "RECOMB72_AUDIT_CONTRACT: requires w16p64, G1-G30, T80, "
                "a frozen mode, mutation-off, ordinary-birth-off, and neutral actions"
            )

    def _recomb72_reproduce_and_cull(self) -> None:
        """Run one genotype-blind 32-cull/32-refill audit boundary."""

        from evolution.recomb72_inheritance import (
            AUDITED_LOCI,
            deterministic_parent_pairs,
            merge_ancestry,
            registry_sha256,
        )

        alive = self.alive_agents()
        self._validate_recomb72_audit_contract(alive)
        sim_cfg = self.sim_cfg
        generation = (
            int(getattr(self, "_eiel_current_generation_index", 0) or 0) + 1
        )
        seed = int(getattr(sim_cfg, "recomb72_audit_seed", 0) or 0)
        ranked = sorted(
            alive,
            key=lambda agent: (
                self._recomb72_schedule_word(
                    seed,
                    generation,
                    int(agent.agent_id),
                    "neutral-cull",
                ),
                int(agent.agent_id),
            ),
        )
        survivors = sorted(ranked[:32], key=lambda agent: int(agent.agent_id))
        culled = sorted(ranked[32:], key=lambda agent: int(agent.agent_id))
        if len(survivors) != 32 or len(culled) != 32:
            raise RuntimeError("RECOMB72_AUDIT_CONTRACT: cull split is not 32/32")
        for agent in culled:
            agent.alive = False
        self.remove_dead()

        by_id = {int(agent.agent_id): agent for agent in survivors}
        pairs = deterministic_parent_pairs(
            tuple(by_id),
            seed=seed,
            generation=generation,
            child_count=32,
        )
        reproduction_events: list[dict[str, Any]] = []
        for child_ordinal, (parent_a_id, parent_b_id) in enumerate(pairs):
            parent_a = by_id[parent_a_id]
            parent_b = by_id[parent_b_id]
            inheritance_seed = self._recomb72_schedule_word(
                seed,
                generation,
                child_ordinal,
                "inheritance",
            )
            production_rng = self.rng
            self.rng = random.Random(inheritance_seed)
            try:
                child = self.make_child(
                    parent_a,
                    parent_b,
                    float(self.env_cfg.start_energy),
                )
            finally:
                self.rng = production_rng
            if child is None:
                raise RuntimeError(
                    "RECOMB72_AUDIT_CONTRACT: authoritative make_child failed"
                )
            parent_a.stats.offspring_count += 1
            parent_b.stats.offspring_count += 1
            ancestry_a = getattr(
                parent_a,
                "_recomb72_ancestry",
                {f"agent_{parent_a_id}": 1.0},
            )
            ancestry_b = getattr(
                parent_b,
                "_recomb72_ancestry",
                {f"agent_{parent_b_id}": 1.0},
            )
            child_ancestry = merge_ancestry(ancestry_a, ancestry_b)
            setattr(child, "_recomb72_ancestry", child_ancestry)
            child_group = "mixed"
            def _founder_side(founder: Any) -> str | None:
                suffix = str(founder).rsplit("_", 1)[-1]
                if suffix in {"a", "b"}:
                    return suffix
                try:
                    return "a" if int(suffix) < 32 else "b"
                except (TypeError, ValueError):
                    return None

            a_mass = sum(
                float(weight)
                for founder, weight in child_ancestry.items()
                if str(founder).startswith("founder_")
                and _founder_side(founder) == "a"
            )
            b_mass = sum(
                float(weight)
                for founder, weight in child_ancestry.items()
                if str(founder).startswith("founder_")
                and _founder_side(founder) == "b"
            )
            if a_mass > b_mass:
                child_group = "a"
            elif b_mass > a_mass:
                child_group = "b"
            setattr(child, "_recomb72_founder_group", child_group)
            inheritance = getattr(
                child.genome, "_recomb72_inheritance_event", None
            )
            if not isinstance(inheritance, dict):
                raise RuntimeError(
                    "RECOMB72_AUDIT_CONTRACT: crossover emitted no audit event"
                )
            reproduction_events.append(
                {
                    "child_ordinal": child_ordinal,
                    "parent_a_id": parent_a_id,
                    "parent_b_id": parent_b_id,
                    "child_id": int(child.agent_id),
                    "inheritance_seed": inheritance_seed,
                    "parent_a_package": {
                        name: float(getattr(parent_a.genome, name))
                        for name in AUDITED_LOCI
                    },
                    "parent_b_package": {
                        name: float(getattr(parent_b.genome, name))
                        for name in AUDITED_LOCI
                    },
                    "child_package": {
                        name: float(getattr(child.genome, name))
                        for name in AUDITED_LOCI
                    },
                    "inheritance": dict(inheritance),
                    "ancestry": child_ancestry,
                    "parent_a_ancestry": dict(ancestry_a),
                    "parent_b_ancestry": dict(ancestry_b),
                    "child_ancestry": dict(child_ancestry),
                    "parent_a_group": str(
                        getattr(parent_a, "_recomb72_founder_group", "mixed")
                    ),
                    "parent_b_group": str(
                        getattr(parent_b, "_recomb72_founder_group", "mixed")
                    ),
                    "child_group": child_group,
                }
            )
        if len(self.alive_agents()) != 64:
            raise RuntimeError(
                "RECOMB72_AUDIT_CONTRACT: refill did not restore 64 agents"
            )
        event = {
            "schema_version": "recomb72_generation_boundary_v1",
            "generation": generation,
            "seed": seed,
            "registry_sha256": registry_sha256(),
            "before_ids": [int(agent.agent_id) for agent in alive],
            "culled_ids": [int(agent.agent_id) for agent in culled],
            "survivor_ids": [int(agent.agent_id) for agent in survivors],
            "eligible_parent_ids": [int(agent.agent_id) for agent in survivors],
            "reproduction_events": reproduction_events,
            "mutation_events": 0,
            "ordinary_tick_births": 0,
            "population_after": 64,
        }
        self._recomb72_generation_boundary_event = event
        history = getattr(self, "_recomb72_generation_boundary_events", None)
        if not isinstance(history, list):
            history = []
            self._recomb72_generation_boundary_events = history
        history.append(event)

    def reproduce_and_cull(self):
        """Perform evolutionary turnover to maintain the target population size.

        This method does three things, in order:
        1) Hard-cap: if alive agents exceed the target size, kill the worst-ranked extras.
        2) Turnover: kill a fraction of the remaining population (cull_fraction).
        3) Refill: if below target, create new offspring via crossover+mutation.

        Phase 5B integration:
        - Applies trait injections at the post-selection-pre-mutation phase, even on
          the hard-cap early-return path (otherwise injections are silently skipped in
          high-birth regimes).
        """
        active_inheritance_profile(getattr(self, "sim_cfg", None))
        if bool(
            getattr(getattr(self, "sim_cfg", None), "recomb72_audit_enable", False)
        ):
            self._recomb72_reproduce_and_cull()
            return
        alive = self.alive_agents()
        funnel_on = _lineage_funnel_enabled(getattr(self, "sim_cfg", None))
        structural_funnel_on = _structural_selection_funnel_enabled(getattr(self, "sim_cfg", None))
        if funnel_on:
            self._lineage_reproduction_funnel_turnover_gen = {
                "schema_version": 1,
                "status": "started",
                "pre_cull_alive_counts": _lineage_counts(alive),
                "post_cull_alive_counts": {},
                "post_refill_alive_counts": {},
                "culled_counts": {},
                "hardcap_culled_counts": {},
                "refill_parent_source_counts": {},
                "refill_offspring_counts": {},
                "deaths_before_refill": 0,
                "deaths_after_refill": 0,
            }
        if structural_funnel_on:
            self._structural_selection_funnel_turnover_gen = {
                "schema_version": 1,
                "status": "started",
                "pre_cull_alive_agent_ids": [int(getattr(a, "agent_id", -1)) for a in alive],
                "hardcap_culled_agent_ids": [],
                "culled_agent_ids": [],
                "post_cull_alive_agent_ids": [],
                "actual_refill_parent_pool_ids": [],
                "parent_draw_events": [],
                "successful_reproduction_events": [],
                "post_refill_alive_agent_ids": [],
                "reporting_only": True,
            }
        if len(alive) < 2:
            self.init_random()
            if funnel_on:
                self._lineage_reproduction_funnel_turnover_gen.update(
                    {
                        "status": "low_population_reinitialized",
                        "post_cull_alive_counts": _lineage_counts(self.alive_agents()),
                        "post_refill_alive_counts": _lineage_counts(self.alive_agents()),
                    }
                )
            if structural_funnel_on:
                self._structural_selection_funnel_turnover_gen.update(
                    {
                        "status": "low_population_reinitialized",
                        "post_cull_alive_agent_ids": [int(getattr(a, "agent_id", -1)) for a in self.alive_agents()],
                        "post_refill_alive_agent_ids": [int(getattr(a, "agent_id", -1)) for a in self.alive_agents()],
                    }
                )
            return

        ranked = self._rank_agents_effective(alive)  # best -> worst
        target = int(self.pop_cfg.population_size)

        # 1) HARD CAP
        if len(ranked) > target:
            hardcap_culled = list(ranked[target:])
            for a in ranked[target:]:
                a.alive = False
            self.remove_dead()
            # Phase 5B: ensure trait injections run even on the hard-cap path.
            self._maybe_apply_trait_injection_post_selection_pre_mutation()
            if funnel_on:
                self._lineage_reproduction_funnel_turnover_gen.update(
                    {
                        "status": "hardcap",
                        "hardcap_culled_counts": _lineage_counts(hardcap_culled),
                        "culled_counts": _lineage_counts(hardcap_culled),
                        "deaths_before_refill": len(hardcap_culled),
                        "post_cull_alive_counts": _lineage_counts(self.alive_agents()),
                        "post_refill_alive_counts": _lineage_counts(self.alive_agents()),
                    }
                )
            if structural_funnel_on:
                self._structural_selection_funnel_turnover_gen.update(
                    {
                        "status": "hardcap",
                        "hardcap_culled_agent_ids": [int(getattr(a, "agent_id", -1)) for a in hardcap_culled],
                        "culled_agent_ids": [int(getattr(a, "agent_id", -1)) for a in hardcap_culled],
                        "post_cull_alive_agent_ids": [int(getattr(a, "agent_id", -1)) for a in self.alive_agents()],
                        "post_refill_alive_agent_ids": [int(getattr(a, "agent_id", -1)) for a in self.alive_agents()],
                    }
                )
            return

        # 2) SELECTION TURNOVER (always)
        cull_frac = float(self.pop_cfg.cull_fraction)
        n_cull = int(len(ranked) * cull_frac)
        culled_agents = list(ranked[-n_cull:]) if n_cull > 0 else []
        if n_cull > 0:
            for a in culled_agents:
                a.alive = False
            self.remove_dead()

        alive = self.alive_agents()
        if len(alive) < 2:
            self.init_random()
            if funnel_on:
                self._lineage_reproduction_funnel_turnover_gen.update(
                    {
                        "status": "post_cull_low_population_reinitialized",
                        "culled_counts": _lineage_counts(culled_agents),
                        "deaths_before_refill": len(culled_agents),
                        "post_cull_alive_counts": _lineage_counts(alive),
                        "post_refill_alive_counts": _lineage_counts(self.alive_agents()),
                    }
                )
            if structural_funnel_on:
                self._structural_selection_funnel_turnover_gen.update(
                    {
                        "status": "post_cull_low_population_reinitialized",
                        "culled_agent_ids": [int(getattr(a, "agent_id", -1)) for a in culled_agents],
                        "post_cull_alive_agent_ids": [int(getattr(a, "agent_id", -1)) for a in alive],
                        "post_refill_alive_agent_ids": [int(getattr(a, "agent_id", -1)) for a in self.alive_agents()],
                    }
                )
            return

        ranked = self._rank_agents_effective(alive)
        if funnel_on:
            self._lineage_reproduction_funnel_turnover_gen.update(
                {
                    "status": "post_cull",
                    "culled_counts": _lineage_counts(culled_agents),
                    "deaths_before_refill": len(culled_agents),
                    "post_cull_alive_counts": _lineage_counts(ranked),
                }
            )
        if structural_funnel_on:
            self._structural_selection_funnel_turnover_gen.update(
                {
                    "status": "post_cull",
                    "culled_agent_ids": [int(getattr(a, "agent_id", -1)) for a in culled_agents],
                    "post_cull_alive_agent_ids": [int(getattr(a, "agent_id", -1)) for a in ranked],
                }
            )


        # -----------------------------
        # Phase 5B: Trait Injection hook (post-selection-pre-mutation)
        # -----------------------------
        self._maybe_apply_trait_injection_post_selection_pre_mutation()

        # 3) OPTIONAL REFILL
        if len(ranked) < target:
            n_alive = len(ranked)
            n_parents = max(2, int(n_alive * float(self.pop_cfg.parent_fraction)))
            parents = ranked[:n_parents]
            if corridor_cp0_central_place_eligibility_enabled(self.env_cfg, getattr(self, "sim_cfg", None)):
                env = getattr(self, "env", None)
                parents = corridor_cp0_filter_home_eligible_agents(env, ranked)[:n_parents] if env is not None else []
            if structural_funnel_on:
                self._structural_selection_funnel_turnover_gen["actual_refill_parent_pool_ids"] = [
                    int(getattr(a, "agent_id", -1)) for a in parents
                ]
            if not parents:
                self.init_random()
                if funnel_on:
                    self._lineage_reproduction_funnel_turnover_gen.update(
                        {
                            "status": "no_refill_parents_reinitialized",
                            "post_refill_alive_counts": _lineage_counts(self.alive_agents()),
                        }
                    )
                if structural_funnel_on:
                    self._structural_selection_funnel_turnover_gen.update(
                        {
                            "status": "no_refill_parents_reinitialized",
                            "post_refill_alive_agent_ids": [int(getattr(a, "agent_id", -1)) for a in self.alive_agents()],
                        }
                    )
                return

            while len(self.agents) < target:
                p1 = self.rng.choice(parents)
                p2 = self.rng.choice(parents)
                if p1.agent_id == p2.agent_id:
                    continue
                if structural_funnel_on:
                    self._structural_selection_funnel_turnover_gen.setdefault("parent_draw_events", []).append(
                        {
                            "parent_a_id": int(getattr(p1, "agent_id", -1)),
                            "parent_b_id": int(getattr(p2, "agent_id", -1)),
                        }
                    )

                child_energy = float(self.env_cfg.start_energy)
                child = self.make_child(p1, p2, child_energy)
                if child is None:
                    break
                if structural_funnel_on:
                    self._structural_selection_funnel_turnover_gen.setdefault("successful_reproduction_events", []).append(
                        {
                            "parent_a_id": int(getattr(p1, "agent_id", -1)),
                            "parent_b_id": int(getattr(p2, "agent_id", -1)),
                            "child_id": int(getattr(child, "agent_id", -1)),
                        }
                    )

                if funnel_on:
                    parent_sources = self._lineage_reproduction_funnel_turnover_gen.setdefault(
                        "refill_parent_source_counts", {}
                    )
                    offspring_counts = self._lineage_reproduction_funnel_turnover_gen.setdefault(
                        "refill_offspring_counts", {}
                    )
                    for parent in (p1, p2):
                        lid = _lineage_id(parent)
                        parent_sources[lid] = int(parent_sources.get(lid, 0)) + 1
                    child_lid = _lineage_id(child)
                    offspring_counts[child_lid] = int(offspring_counts.get(child_lid, 0)) + 1

                p1.stats.offspring_count += 1
                p2.stats.offspring_count += 1
        if funnel_on:
            self._lineage_reproduction_funnel_turnover_gen.update(
                {
                    "status": "completed",
                    "post_refill_alive_counts": _lineage_counts(self.alive_agents()),
                }
            )
        if structural_funnel_on:
            self._structural_selection_funnel_turnover_gen.update(
                {
                    "status": "completed",
                    "post_refill_alive_agent_ids": [int(getattr(a, "agent_id", -1)) for a in self.alive_agents()],
                }
            )
