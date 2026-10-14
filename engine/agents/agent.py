from __future__ import annotations

"""agents/agent.py

Agent state container.

This file defines:
- Agent: the persistent entity that lives in the simulation.
- AgentStats: per-lifetime/per-generation counters used for metrics and selection.

Internal drives (Sprint: Identity Pressure via Internal Drives)
--------------------------------------------------------------
We maintain a small per-agent "drives" vector representing internal pressure
signals. Drives are NOT explicit goals. They drift, respond to physiology, and
bias action logits (neuromodulation). Incoherent regulation gets penalized
via selection (thrashing, self-harm loops, prediction error).

Drive indices:
0 = hunger   (bias toward GATHER)
1 = pain     (bias toward REST, away from ATTACK)
2 = fatigue  (bias toward REST; shaped by rest/action streaks)
3 = fear     (bias away from ATTACK; shaped by injury spikes / low health)

Outcome expectation model (cheap self-model pressure)
----------------------------------------------------
Each agent keeps a tiny per-action expectation of outcome deltas:
- expected Δenergy per action
- expected Δinjury per action

These are updated online (EMA) and we track prediction error:
- high prediction error over lifetime becomes a fitness penalty
- agents that settle into stable expectations survive longer

This is intentionally primitive. It's not "learning" in the RL sense.
It's a selection pressure for internal coherence.
"""

from dataclasses import dataclass, field
from typing import Any, Optional
from collections import deque

import numpy as np
import torch

from agents.genome import Genome
from agents.chemistry import zeros_chemistry
from agents.maintenance import zeros_maintenance_state

# Action indices used by GridWorld stepping.  Keep the vocabulary authoritative
# here so capacity/reporting code does not have to duplicate the count.
ACTION_VOCABULARY_SCHEMA_VERSION = "eiel_action_vocabulary_v1"
ACTION_VOCABULARY: tuple[str, ...] = (
    "move_north",
    "move_south",
    "move_east",
    "move_west",
    "gather",
    "rest",
    "attack",
)
N_ACTIONS = len(ACTION_VOCABULARY)

# Drives are stored as a fixed-size float32 vector.
DRIVE_VOCABULARY_SCHEMA_VERSION = "eiel_drive_vocabulary_v1"
DRIVE_VOCABULARY: tuple[str, ...] = ("hunger", "pain", "fatigue", "fear")
N_DRIVES = len(DRIVE_VOCABULARY)
DRIVE_HUNGER = 0
DRIVE_PAIN = 1
DRIVE_FATIGUE = 2
DRIVE_FEAR = 3


def _zeros_actions() -> np.ndarray:
    return np.zeros((N_ACTIONS,), dtype=np.float32)

def _zeros_drives() -> np.ndarray:
    return np.zeros((N_DRIVES,), dtype=np.float32)


def format_lineage_id(agent_id: int) -> str:
    """Return the deterministic founder lineage id for an agent id."""
    try:
        aid = int(agent_id)
    except Exception:
        aid = 0
    return f"L{max(0, aid):08d}"


def _lineage_parent_tuple(value: Any, *, as_int: bool) -> tuple:
    if value is None:
        return tuple()
    if isinstance(value, (str, bytes)):
        items = [value]
    else:
        try:
            items = list(value)
        except Exception:
            items = [value]
    out = []
    for item in items:
        try:
            out.append(int(item) if as_int else str(item))
        except Exception:
            continue
    return tuple(out)


def _lineage_context_value(value: Any, fallback: int = 0) -> int:
    try:
        iv = int(value)
        return iv if iv >= 0 else int(fallback)
    except Exception:
        return int(fallback)


def normalize_lineage_stats(stats: Any, agent_id: int, *, generation: int = 0, tick: int = 0) -> Any:
    """Fill missing Theme 5 lineage metadata without drawing RNG."""
    if stats is None:
        return stats
    aid = _lineage_context_value(agent_id, 0)
    lineage_id = str(getattr(stats, "lineage_id", "") or "").strip()
    if not lineage_id:
        lineage_id = format_lineage_id(aid)
    setattr(stats, "lineage_id", lineage_id)

    founder = getattr(stats, "founder_id", None)
    try:
        founder_i = int(founder)
    except Exception:
        founder_i = aid
    if founder_i < 0:
        founder_i = aid
    setattr(stats, "founder_id", founder_i)

    setattr(stats, "parent_ids", _lineage_parent_tuple(getattr(stats, "parent_ids", ()), as_int=True))
    setattr(stats, "parent_lineage_ids", _lineage_parent_tuple(getattr(stats, "parent_lineage_ids", ()), as_int=False))
    setattr(stats, "birth_generation", _lineage_context_value(getattr(stats, "birth_generation", None), generation))
    setattr(stats, "birth_tick", _lineage_context_value(getattr(stats, "birth_tick", None), tick))
    setattr(stats, "lineage_depth", _lineage_context_value(getattr(stats, "lineage_depth", None), 0))
    return stats


def assign_founder_lineage(agent: Any, *, generation: int = 0, tick: int = 0) -> None:
    """Assign deterministic founder lineage metadata to a newly spawned agent."""
    stats = normalize_lineage_stats(getattr(agent, "stats", None), getattr(agent, "agent_id", 0), generation=generation, tick=tick)
    if stats is None:
        return
    aid = _lineage_context_value(getattr(agent, "agent_id", 0), 0)
    stats.lineage_id = format_lineage_id(aid)
    stats.founder_id = aid
    stats.parent_ids = tuple()
    stats.parent_lineage_ids = tuple()
    stats.birth_generation = _lineage_context_value(generation, 0)
    stats.birth_tick = _lineage_context_value(tick, 0)
    stats.lineage_depth = 0


def assign_child_lineage(child: Any, parent_a: Any, parent_b: Any, *, generation: int = 0, tick: int = 0) -> None:
    """Assign deterministic biological lineage metadata to a child agent."""
    parents = sorted(
        [p for p in (parent_a, parent_b) if p is not None],
        key=lambda p: _lineage_context_value(getattr(p, "agent_id", 0), 0),
    )
    child_stats = normalize_lineage_stats(getattr(child, "stats", None), getattr(child, "agent_id", 0), generation=generation, tick=tick)
    if child_stats is None:
        return
    if not parents:
        assign_founder_lineage(child, generation=generation, tick=tick)
        return

    parent_ids = []
    parent_lineage_ids = []
    parent_depths = []
    for p in parents:
        pid = _lineage_context_value(getattr(p, "agent_id", 0), 0)
        pst = normalize_lineage_stats(getattr(p, "stats", None), pid, generation=0, tick=0)
        parent_ids.append(pid)
        parent_lineage_ids.append(str(getattr(pst, "lineage_id", format_lineage_id(pid)) if pst is not None else format_lineage_id(pid)))
        parent_depths.append(_lineage_context_value(getattr(pst, "lineage_depth", 0) if pst is not None else 0, 0))

    primary = parents[0]
    primary_id = _lineage_context_value(getattr(primary, "agent_id", 0), 0)
    primary_stats = normalize_lineage_stats(getattr(primary, "stats", None), primary_id, generation=0, tick=0)

    child_stats.lineage_id = str(getattr(primary_stats, "lineage_id", format_lineage_id(primary_id)) if primary_stats is not None else format_lineage_id(primary_id))
    child_stats.founder_id = _lineage_context_value(getattr(primary_stats, "founder_id", primary_id) if primary_stats is not None else primary_id, primary_id)
    child_stats.parent_ids = tuple(parent_ids)
    child_stats.parent_lineage_ids = tuple(parent_lineage_ids)
    child_stats.birth_generation = _lineage_context_value(generation, 0)
    child_stats.birth_tick = _lineage_context_value(tick, 0)
    child_stats.lineage_depth = int(max(parent_depths or [0])) + 1


@dataclass
class AgentStats:
    survival_steps: int = 0
    energy_collected: float = 0.0
    offspring_count: int = 0
    attacks: int = 0

    # First reproduction timing (analysis-only)
    # Age (in life-steps) when this agent first successfully produced offspring.
    # None means the agent has not reproduced yet.
    first_repro_step: Optional[int] = None

    # Patch World diagnostics only: founder-style habitat lineage id used for
    # reporting habitat use. It must not affect selection or behavior.
    patch_habitat_lineage_id: int = 0

    # Theme 5 biological lineage metadata. These fields are observability-only:
    # they must not influence selection, reproduction, policy execution, or fitness.
    lineage_id: str = ""
    founder_id: int = -1
    parent_ids: tuple[int, ...] = field(default_factory=tuple)
    parent_lineage_ids: tuple[str, ...] = field(default_factory=tuple)
    birth_generation: int = 0
    birth_tick: int = 0
    lineage_depth: int = 0

    # -----------------------------
    # Early-life fitness window accumulators
    # -----------------------------
    # These are only accumulated during the first K steps of the agent's life,
    # where K = sim_cfg.early_life_steps.
    early_survival_steps: int = 0
    early_energy_collected: float = 0.0
    early_injury_gained: float = 0.0

    # Used by the rest-charge system
    idle_charge_paid: float = 0.0
    maintenance_requests: int = 0
    maintenance_applied: int = 0
    maintenance_energy_spent_total: float = 0.0
    maintenance_opportunity_cost_total: float = 0.0
    maintenance_applied_effect_total: float = 0.0
    structural_locality_cost_total: float = 0.0
    structural_locality_mean_distance: float = 0.0
    structural_locality_max_distance: float = 0.0
    chemistry_action_cost_mult: float = 1.0
    chemistry_observation_fidelity: float = 1.0
    chemistry_observation_bias: float = 0.0
    chemistry_repro_viability: float = 1.0

    # -----------------------------
    # Coherence / thrashing signals
    # -----------------------------
    last_action: int = -1
    last_last_action: int = -1
    action_switches: int = 0
    action_oscillations: int = 0

    # Drive volatility accumulator (sum abs(delta drive) per step)
    drive_volatility: float = 0.0

    # High-frequency drive oscillation accumulator (jitter). Prefer this over volatility.
    drive_jitter: float = 0.0

    # Previous per-step drive delta (for jitter computation)
    prev_drive_delta: np.ndarray = field(default_factory=_zeros_drives)

    # Birth / episode reset mismatch (how far initial drives are from inherited setpoints)
    birth_reset_shock: float = 0.0

    # Episode-0 (early-life) accumulators for cross-episode consistency analysis
    episode0_pred_err_sum: float = 0.0
    episode0_pred_err_count: int = 0
    episode0_switches: int = 0
    episode0_oscillations: int = 0
    episode0_drive_jitter_sum: float = 0.0
    episode0_selfharm_events: int = 0

    # -----------------------------
    # Episode-0 FINALIZED fields (what MetricsLogger expects)
    # -----------------------------
    # These are computed at generation end (see runner_pkg/generation.py).
    episode0_pred_err_mean: float = 0.0
    episode0_drive_jitter: float = 0.0
    episode0_selfharm: float = 0.0

    # -----------------------------
    # Self-harm + prediction error
    # -----------------------------
    selfharm_events: int = 0  # crude: injury gained after ATTACK or repeated risky acts
    pred_err_sum: float = 0.0
    pred_err_count: int = 0

    # -----------------------------
    # Option C: Late-life prediction error window (post-first-reproduction)
    # -----------------------------
    # These accumulate ONLY after the agent's first successful reproduction step.
    # They are used for an analysis-clean selection penalty (applied at parent selection time).
    late_pred_err_sum: float = 0.0
    late_pred_err_count: int = 0

    # Cached (selection-time) derived values for logging/debug.
    late_pred_err_mean: float = 0.0
    late_pred_cost: float = 0.0

    # -----------------------------
    # Life-history tradeoff knob: late-life coherence efficiency bonus
    # -----------------------------
    # Cached per-tick efficiency multiplier applied to action costs in adulthood.
    # 1.0 means no bonus/penalty. Values <1.0 mean cheaper actions (more efficient).
    late_coherence_eff_mult: float = 1.0
    # Cached coherence score in [0,1] used to derive late_coherence_eff_mult (1 = coherent).
    late_coherence_score: float = 0.0

    # Exponential moving average of adult prediction error (Part B coupling).
    # Updated per tick after adult_move_penalty_start_step if enabled.
    adult_pred_err_ema: float = 0.0

    # Phase 4: identity drift metric (EMA of probe signature drift).
    # This is analysis + selection pressure plumbing (movement capability coupling).
    identity_drift_ema: float = 0.0

    # Expected per-action deltas (online updated)
    exp_dE: np.ndarray = field(default_factory=_zeros_actions)   # expected Δenergy
    exp_dInj: np.ndarray = field(default_factory=_zeros_actions) # expected Δinjury
    prediction_update_count: int = 0
    prediction_state_delta_abs_sum: float = 0.0
    prediction_state_delta_abs_max: float = 0.0
    prediction_reset_count: int = 0


    # Pending delayed predictions: (due_step, action, exp_dE, exp_dInj, energy_base, injury_base)
    # Used to score short-horizon prediction error (anticipation pressure).
    pred_queue: deque = field(default_factory=lambda: deque(maxlen=8))
    # Cached deltas from last env step (filled by runner)
    last_energy_delta: float = 0.0
    last_health_delta: float = 0.0
    last_injury_delta: float = 0.0


@dataclass
class Agent:
    agent_id: int
    genome: Genome
    policy: object  # MLPPolicy or GRUPolicy (or future policies)
    energy: float
    health: float
    alive: bool = True

    stats: AgentStats = field(default_factory=AgentStats)

    # reproduction
    repro_cooldown: int = 0

    # normalization (set by Population from EnvConfig)
    max_energy_norm: float = 160.0
    start_health_norm: float = 12.0

    # injury + rest streak (for recovery gating)
    injury: float = 0.0
    rest_streak: int = 0
    habitat_acclimation_state: float = 0.0

    # compact slot index for dense arrays / policy bank
    slot: int = -1
    slot_token: int = 0

    # (optional) per-agent cached memory on CPU side (we primarily store memory in the GPU bank)
    memory_cpu: Optional[torch.Tensor] = None

    # -----------------------------
    # Internal drives (persistent)
    # -----------------------------
    drives: np.ndarray = field(
        default_factory=lambda: np.zeros((N_DRIVES,), dtype=np.float32)
    )
    chemistry: np.ndarray = field(default_factory=zeros_chemistry)
    maintenance: np.ndarray = field(default_factory=zeros_maintenance_state)

    # ESCAP-3: owned runtime state is keyed by stable module_instance_id.  These
    # maps start empty for every newborn; hereditary architecture lives only in
    # the genome and lifetime state is never copied through reproduction.
    structural_live_module_state: dict[str, float] = field(default_factory=dict)
    structural_live_module_outputs: dict[str, float] = field(default_factory=dict)
    structural_live_module_telemetry: dict[str, object] = field(default_factory=dict)
    # ESCAP-6: irreversible developmental dependency gates are lifetime state,
    # keyed by stable module_instance_id.  Newborns start empty and C2 never
    # copies this map from either parent.
    structural_development_latch_by_instance_id: dict[str, bool] = field(default_factory=dict)
    structural_development_stage_id: str = ""
    structural_development_effective_gain_by_instance_id: dict[str, float] = field(default_factory=dict)

    # Optional tiny action history for debugging / future richer coherence metrics
    recent_actions: deque = field(default_factory=lambda: deque(maxlen=12))

    def observe_vector(self, local_obs: np.ndarray) -> np.ndarray:
        """Build a flat observation vector.

        NOTE: The runner currently builds observations in runner_policy.py for
        batched inference. This method is still useful for debugging, unit tests,
        and future non-banked policies.

        We append drives at the end as a forward-compatible extension, but the
        current banked policy path does NOT depend on this method.
        """
        flat = local_obs.astype(np.float32).reshape(-1)
        energy_n = np.array([self.energy / float(self.max_energy_norm)], dtype=np.float32)
        health_n = np.array([self.health / float(self.start_health_norm)], dtype=np.float32)

        d = self.drives.astype(np.float32, copy=False).reshape(-1)
        return np.concatenate([flat, energy_n, health_n, d], axis=0)
