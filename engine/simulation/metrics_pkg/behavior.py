from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union

import numpy as np

from simulation.output_cost_timing import current_flat_output_cost_timing


@dataclass
class AgentBehavior:
    move_n: int = 0
    move_s: int = 0
    move_e: int = 0
    move_w: int = 0
    gather: int = 0
    rest: int = 0
    attack: int = 0

    lifespan_steps: int = 0
    energy_collected: float = 0.0
    offspring: int = 0
    attacks_made: int = 0
    fitness: float = 0.0
    alive_end: int = 0

    eligible_for_selection: int = 0
    kept_next: int = 0

    # Injury at end-of-generation, and delta vs start-of-generation snapshot.
    injury_end: float = 0.0
    injury_delta: float = 0.0

    # NEW: genome trait (per-agent)
    aggression_bias: float = 0.0

    # Internal flag: if caller explicitly set injury_delta, don't overwrite it.
    _injury_delta_set: bool = False

    cohort: str = "start"


InjurySnapshot = Union[np.ndarray, Dict[int, float], None]


class BehaviorLogger:
    """
    behaviors.csv logger

    Fixes implemented:
    (1) Keep a reliable per-generation *start injury snapshot*.
    (2) Compute injury_delta correctly from (end - start) unless explicitly provided.
    (3) Snapshot newborns too (so their delta isn't accidentally wrong due to missing start).
    """

    def __init__(
        self,
        run_dir: str | Path,
        action_id_map: Dict[str, int],
        cohort_mode: str = "start",
    ):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.run_dir / "behaviors.csv"
        self.action_id_map = {k: int(v) for k, v in action_id_map.items()}

        if cohort_mode not in ("start", "acted", "full"):
            raise ValueError(f"Invalid cohort_mode={cohort_mode!r}. Use 'start', 'acted', or 'full'.")
        self.cohort_mode = cohort_mode

        self._gen: Optional[int] = None
        self._ids: List[int] = []
        self._start_ids: set[int] = set()
        self._data: Dict[int, AgentBehavior] = {}

        # Injury snapshots for correct delta computation (aid -> injury at gen start).
        self._injury_start: Dict[int, float] = {}

        self._fh = None
        self._writer = None
        self._ensure_open()

    def _ensure_open(self):
        new = not self.path.exists()
        self._fh = self.path.open("a", newline="", encoding="utf-8")
        self._writer = csv.writer(self._fh)
        if new:
            header = [
                "generation", "agent_id",
                "cohort",
                "move_n", "move_s", "move_e", "move_w",
                "gather", "rest", "attack",
                "lifespan_steps", "energy_collected",
                "offspring", "attacks_made",
                "fitness", "alive_end",
                "eligible_for_selection", "kept_next",
                "injury_end", "injury_delta",
                "aggression_bias",  # NEW
            ]
            timing = current_flat_output_cost_timing()
            if timing is None:
                self._writer.writerow(header)
                self._fh.flush()
            else:
                with timing.serialization_phase():
                    buffer = io.StringIO(newline="")
                    csv.writer(buffer).writerow(header)
                    encoded = buffer.getvalue()
                with timing.write_phase():
                    self._fh.write(encoded)
                with timing.flush_phase():
                    self._fh.flush()

    def injury_start_for(self, agent_id: int, default: float = 0.0) -> float:
        """Expose start injury snapshot (used by selection for missing agents)."""
        return float(self._injury_start.get(int(agent_id), default))

    def _ensure_agent(self, agent_id: int, cohort: str) -> Optional[AgentBehavior]:
        aid = int(agent_id)
        b = self._data.get(aid)
        if b is not None:
            return b

        if self.cohort_mode == "start":
            return None

        b = AgentBehavior(cohort=str(cohort))
        self._data[aid] = b
        self._ids.append(aid)
        return b

    def _snapshot_injury(self, aid: int, injuries: InjurySnapshot):
        """Record starting injury for agent aid from either array-indexed or dict-based snapshot."""
        if injuries is None:
            return

        if isinstance(injuries, dict):
            self._injury_start[aid] = float(injuries.get(aid, 0.0))
            return

        # numpy array case: index by agent id
        inj_arr = injuries.astype(np.float32, copy=False)
        if 0 <= aid < int(inj_arr.size):
            self._injury_start[aid] = float(inj_arr[aid])
        else:
            self._injury_start[aid] = 0.0

    def start_generation(
        self,
        gen: int,
        agent_ids: Iterable[int],
        injuries: InjurySnapshot = None,
    ):
        """
        Start a new generation and optionally snapshot starting injuries for delta computation.

        injuries may be:
          - np.ndarray indexed by agent_id
          - Dict[int, float] mapping agent_id -> injury
          - None (no snapshot; injury_delta will be left as whatever was set)
        """
        self._gen = int(gen)
        self._ids = []
        self._start_ids = set()
        self._data = {}
        self._injury_start = {}

        for a in agent_ids:
            aid = int(a)
            self._start_ids.add(aid)
            self._ids.append(aid)
            self._data[aid] = AgentBehavior(cohort="start")
            self._snapshot_injury(aid, injuries)

    def register_agent(
        self,
        agent_id: int,
        cohort: str = "born",
        injuries: InjurySnapshot = None,
    ):
        """
        Register a newly born agent (only in cohort_mode='full') and optionally snapshot injury_start.
        """
        if self._gen is None:
            return
        if self.cohort_mode != "full":
            return

        aid = int(agent_id)
        if aid in self._data:
            return

        self._data[aid] = AgentBehavior(cohort=str(cohort))
        self._ids.append(aid)
        self._snapshot_injury(aid, injuries)

    def record_actions(self, agent_ids: np.ndarray, actions: np.ndarray):
        if self._gen is None:
            return

        for aid, act in zip(agent_ids.tolist(), actions.tolist()):
            aid_i = int(aid)
            act_i = int(act)

            b = self._data.get(aid_i)
            if b is None:
                b = self._ensure_agent(aid_i, cohort="acted")
                if b is None:
                    continue

            if act_i == self.action_id_map["MOVE_N"]:
                b.move_n += 1
            elif act_i == self.action_id_map["MOVE_S"]:
                b.move_s += 1
            elif act_i == self.action_id_map["MOVE_E"]:
                b.move_e += 1
            elif act_i == self.action_id_map["MOVE_W"]:
                b.move_w += 1
            elif act_i == self.action_id_map["GATHER"]:
                b.gather += 1
            elif act_i == self.action_id_map["REST"]:
                b.rest += 1
            elif act_i == self.action_id_map["ATTACK"]:
                b.attack += 1
                b.attacks_made += 1

    def generation_proxy_metrics(self) -> dict[str, float]:
        """Return deterministic telemetry-only behavior proxy aggregates."""
        move_n = sum(int(b.move_n) for b in self._data.values())
        move_s = sum(int(b.move_s) for b in self._data.values())
        move_e = sum(int(b.move_e) for b in self._data.values())
        move_w = sum(int(b.move_w) for b in self._data.values())
        move_total = int(move_n + move_s + move_e + move_w)
        gather = sum(int(b.gather) for b in self._data.values())
        rest = sum(int(b.rest) for b in self._data.values())
        attack = sum(int(b.attack) for b in self._data.values())
        action_total = int(move_total + gather + rest + attack)
        denom = float(max(1, action_total))
        return {
            "behavior_action_total": float(action_total),
            "behavior_move_total": float(move_total),
            "behavior_move_n_total": float(move_n),
            "behavior_move_s_total": float(move_s),
            "behavior_move_e_total": float(move_e),
            "behavior_move_w_total": float(move_w),
            "behavior_gather_total": float(gather),
            "behavior_rest_total": float(rest),
            "behavior_attack_total": float(attack),
            "behavior_move_fraction": float(move_total) / denom,
            "behavior_gather_fraction": float(gather) / denom,
            "behavior_rest_fraction": float(rest) / denom,
            "behavior_attack_fraction": float(attack) / denom,
        }

    def set_outcomes(self, agent_id: int, **kwargs):
        """
        Backward compatible: existing callers can keep using keyword outcomes.

        Optional: you may pass agent=<Agent> and we'll snapshot genome.aggression_bias.
        """
        b = self._data.get(int(agent_id))
        if b is None:
            return

        # Optional agent object (do not require it)
        agent = kwargs.pop("agent", None)
        if agent is not None:
            try:
                b.aggression_bias = float(getattr(getattr(agent, "genome", None), "aggression_bias", 0.0))
            except Exception:
                b.aggression_bias = 0.0

        for k, v in kwargs.items():
            if not hasattr(b, k):
                continue

            if k in ("lifespan_steps", "offspring", "attacks_made", "alive_end", "eligible_for_selection", "kept_next"):
                setattr(b, k, int(v))
            elif k in ("energy_collected", "fitness", "injury_end", "aggression_bias"):
                # allow direct setting too
                setattr(b, k, float(v))
            elif k == "injury_delta":
                b._injury_delta_set = True
                setattr(b, k, float(v))
            elif k == "cohort":
                setattr(b, k, str(v))
            else:
                setattr(b, k, v)

    def end_generation(self):
        if self._gen is None:
            return

        timing = current_flat_output_cost_timing()
        for aid in self._ids:
            b = self._data.get(aid, AgentBehavior())

            # If we have a start snapshot and caller didn't explicitly set delta, compute it.
            if (aid in self._injury_start) and (not b._injury_delta_set):
                start_inj = float(self._injury_start.get(aid, 0.0))
                b.injury_delta = float(b.injury_end) - start_inj

            row = [
                self._gen, aid,
                b.cohort,
                b.move_n, b.move_s, b.move_e, b.move_w,
                b.gather, b.rest, b.attack,
                b.lifespan_steps, b.energy_collected,
                b.offspring, b.attacks_made,
                b.fitness, b.alive_end,
                b.eligible_for_selection, b.kept_next,
                b.injury_end, b.injury_delta,
                b.aggression_bias,  # NEW
            ]
            if timing is None:
                self._writer.writerow(row)
            else:
                with timing.serialization_phase():
                    buffer = io.StringIO(newline="")
                    csv.writer(buffer).writerow(row)
                    encoded = buffer.getvalue()
                with timing.write_phase():
                    self._fh.write(encoded)

        if timing is None:
            self._fh.flush()
        else:
            with timing.flush_phase():
                self._fh.flush()

        # -----------------------------
        # Phase 5d: optional viz/ behaviors aggregation summary (observability only)
        # -----------------------------
        try:
            from simulation.viz_pkg.viz_writer import get_viz_writer

            vw = get_viz_writer(self.run_dir)
            if bool(getattr(vw, "enabled", False)) and bool(getattr(vw.cfg, "enable_behaviors_agg", False)):
                traits = list(getattr(vw.cfg, "behaviors_traits_allowlist", ()) or ())
                if traits:
                    values_by_trait = {t: [] for t in traits}
                    for aid in self._ids:
                        b = self._data.get(aid)
                        if b is None:
                            continue
                        for t in traits:
                            if not hasattr(b, t):
                                continue
                            v = getattr(b, t)
                            if v is None:
                                continue
                            try:
                                values_by_trait[t].append(float(v))
                            except Exception:
                                pass
                    vw.append_behaviors_agg_summary(generation=int(self._gen), values_by_trait=values_by_trait)
        except Exception:
            # Never break determinism-critical behavior logging.
            pass

        self._data.clear()
        self._ids = []
        self._start_ids = set()
        self._injury_start = {}
        self._gen = None

    def close(self):
        if self._fh:
            self._fh.close()
            self._fh = None
            self._writer = None
