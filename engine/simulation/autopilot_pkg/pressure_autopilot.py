"""simulation/autopilot_pkg/pressure_autopilot.py

Pressure Autopilot (Phase 6 draft).

What this is
------------
This is a small, deterministic "PID-ish" controller that can steer a run by
scheduling ControlPlane PressureUpdateCommand interventions.

It is deliberately conservative:
- No RNG usage.
- No direct mutation of the sim outside ControlPlane commands.
- Operates only at SafeBoundary (after metrics are computed).

Why it exists
-------------
Humans are inconsistent and get bored. This module watches a handful of
high-level signals (EPI, AV, cohort survival, diversity collapse) and nudges
environmental knobs to keep the population inside an "evolvable" band:
not extinct, not stagnant.

Runtime control (activation/deactivation/modification)
------------------------------------------------------
The autopilot is controlled entirely via sim_cfg fields (which means you can
toggle it during a run using an existing PressureUpdateCommand):

  - sim_cfg.pressure_autopilot_enable
  - sim_cfg.pressure_autopilot_profile
  - sim_cfg.pressure_autopilot_tick_interval_gen
  - (and the other sim_cfg.pressure_autopilot_* knobs)

Branching notes
---------------
Each run directory has its own ControlPlane + autopilot state. If you fork from
a checkpoint into a new run_dir, the checkpoint payload will carry autopilot
state so the controller resumes equivalently. If you want a fork to behave
differently, change the sim_cfg autopilot knobs on the fork (or reset state).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import os
import uuid
from typing import Any, Dict, List, Optional, Tuple

from simulation.control_plane_pkg.commands import PressureUpdateCommand
from simulation.control_plane_pkg.util_json import canonical_json_bytes, canonical_json_bytes_lossless


AUTOPILOT_STATE_SCHEMA_VERSION = "pressure_autopilot_state_v1"
AUTOPILOT_DECISION_SCHEMA_VERSION = "pressure_autopilot_decision_v1"


# Deterministic namespace for uuid5 generation.
_AUTOPILOT_NAMESPACE_UUID = uuid.UUID("7b7b6c3b-3b7a-4b8a-9a9a-7c5b7c6a1e5f")


def _clamp(x: float, lo: float, hi: float) -> float:
    """Clamp x to [lo, hi] as float."""
    xf = float(x)
    if xf < float(lo):
        return float(lo)
    if xf > float(hi):
        return float(hi)
    return xf


def _safe_float(x: Any, default: float = 0.0) -> float:
    """Best-effort float cast with a deterministic fallback."""
    try:
        return float(x)
    except Exception:
        return float(default)


def _safe_int(x: Any, default: int = 0) -> int:
    """Best-effort int cast with a deterministic fallback."""
    try:
        return int(x)
    except Exception:
        return int(default)


def _atomic_write_bytes(path: str, data: bytes) -> None:
    """Atomic write helper: tmp + os.replace.

    This is not part of the determinism substrate; it's just to avoid partial
    files when something crashes mid-write.
    """
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)


def _read_json_file(path: str) -> Optional[Dict[str, Any]]:
    """Read a small JSON file; returns None if missing/unreadable."""
    try:
        import json

        if not os.path.isfile(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, dict):
            return obj
    except Exception:
        return None
    return None


def _uuid5(name: str) -> str:
    """Generate a deterministic UUID (uuid5) from name."""
    return str(uuid.uuid5(_AUTOPILOT_NAMESPACE_UUID, str(name)))


@dataclass
class _KnobSpec:
    """Internal knob mapping spec.

    Attributes
    ----------
    path:
        Dot-path used by PressureUpdateCommand (must be allowed by ControlPlane).
    axis:
        Which "axis" this knob belongs to (used by cycle profiles).
    kind:
        "int" or "float".
    low_mult / high_mult:
        Multipliers for pressure_level=0 (easy) and pressure_level=1 (hard).
        For baseline equivalence at pressure=0.5, choose low_mult + high_mult == 2.
    min_value / max_value:
        Hard clamps to avoid absurd values.
    allow_enable_if_zero:
        If baseline value is 0.0 (user disabled the feature), the autopilot will
        keep it 0 unless this flag is True.
    baseline_if_zero:
        Baseline substitute used when allow_enable_if_zero is True and baseline==0.
    """

    path: str
    axis: str
    kind: str
    low_mult: float
    high_mult: float
    min_value: float
    max_value: float
    allow_enable_if_zero: bool = False
    baseline_if_zero: float = 0.0


@dataclass
class PressureAutopilotState:
    """Persisted autopilot state.

    This state is required for checkpoint/resume equivalence.
    """

    schema_version: str = AUTOPILOT_STATE_SCHEMA_VERSION
    pressure_level: float = 0.50
    axis_index: int = 0
    stagnation_ticks: int = 0
    last_tick_generation: int = -1
    action_counter: int = 0
    last_epi_value: Optional[float] = None
    last_av_value: Optional[float] = None
    baseline: Dict[str, Any] = field(default_factory=dict)  # path -> baseline value
    script_cursor: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": str(self.schema_version),
            "pressure_level": float(self.pressure_level),
            "axis_index": int(self.axis_index),
            "stagnation_ticks": int(self.stagnation_ticks),
            "last_tick_generation": int(self.last_tick_generation),
            "action_counter": int(self.action_counter),
            "last_epi_value": None if self.last_epi_value is None else float(self.last_epi_value),
            "last_av_value": None if self.last_av_value is None else float(self.last_av_value),
            "baseline": dict(self.baseline),
            "script_cursor": int(self.script_cursor),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PressureAutopilotState":
        if not isinstance(d, dict):
            return cls()
        st = cls()
        try:
            st.pressure_level = float(d.get("pressure_level", st.pressure_level))
        except Exception:
            pass
        try:
            st.axis_index = int(d.get("axis_index", st.axis_index))
        except Exception:
            pass
        try:
            st.stagnation_ticks = int(d.get("stagnation_ticks", st.stagnation_ticks))
        except Exception:
            pass
        try:
            st.last_tick_generation = int(d.get("last_tick_generation", st.last_tick_generation))
        except Exception:
            pass
        try:
            st.action_counter = int(d.get("action_counter", st.action_counter))
        except Exception:
            pass
        try:
            st.last_epi_value = d.get("last_epi_value", None)
            if st.last_epi_value is not None:
                st.last_epi_value = float(st.last_epi_value)
        except Exception:
            st.last_epi_value = None
        try:
            st.last_av_value = d.get("last_av_value", None)
            if st.last_av_value is not None:
                st.last_av_value = float(st.last_av_value)
        except Exception:
            st.last_av_value = None
        try:
            b = d.get("baseline", {})
            if isinstance(b, dict):
                st.baseline = dict(b)
        except Exception:
            st.baseline = {}
        try:
            st.script_cursor = int(d.get("script_cursor", st.script_cursor))
        except Exception:
            st.script_cursor = 0
        return st


class PressureAutopilot:
    """Deterministic pressure autopilot.

    The autopilot does NOT mutate the simulation directly. It only schedules
    PressureUpdateCommand interventions via the run's ControlPlane.
    """

    def __init__(self, run_dir: str):
        self.run_dir = str(run_dir)
        self._state_path = os.path.join(self.run_dir, "autopilot", "pressure_autopilot_state.json")
        self._decisions_path = os.path.join(self.run_dir, "autopilot", "pressure_autopilot_decisions.jsonl")
        self.state: PressureAutopilotState = PressureAutopilotState()

        # Best-effort load (not relied on for resume equivalence; checkpoint carries state).
        disk = _read_json_file(self._state_path)
        if isinstance(disk, dict) and str(disk.get("schema_version", "")) == AUTOPILOT_STATE_SCHEMA_VERSION:
            try:
                self.state = PressureAutopilotState.from_dict(disk)
            except Exception:
                self.state = PressureAutopilotState()

        # Edge detector for enable/disable transitions (not persisted).
        # Seed from persisted state so resume does not re-trigger "on enable".
        self._was_enabled: bool = bool(getattr(self.state, "last_tick_generation", -1) >= 0)

    # -----------------------------
    # Checkpoint / resume integration
    # -----------------------------
    def export_state(self) -> Dict[str, Any]:
        """Export determinism-critical autopilot state for checkpoint payloads."""
        return self.state.to_dict()

    def restore_state(self, state: Dict[str, Any]) -> None:
        """Restore determinism-critical autopilot state from checkpoint payloads."""
        if not isinstance(state, dict):
            return
        # Best-effort: accept unknown schema without crashing resume.
        try:
            self.state = PressureAutopilotState.from_dict(state)
        except Exception:
            return

        # Persist to disk for observability.
        self._persist_state_best_effort()

    # -----------------------------
    # Main tick
    # -----------------------------
    def maybe_tick(self, sim: Any, generation_index: int) -> None:
        """Run one autopilot decision tick at SafeBoundary.

        This method is designed to be called once per SafeBoundary *after*
        metrics are computed.
        """

        sim_cfg = getattr(sim, "sim_cfg", None)
        if sim_cfg is None:
            return

        enabled = bool(getattr(sim_cfg, "pressure_autopilot_enable", False))
        hold = bool(getattr(sim_cfg, "pressure_autopilot_hold", False))
        if (not enabled) or hold:
            # Treat hold/disable as "off" for edge detection.
            self._was_enabled = False
            return

        # Enable edge: first SafeBoundary after turning it on.
        if not self._was_enabled:
            self._was_enabled = True
            if bool(getattr(sim_cfg, "pressure_autopilot_reset_on_enable", False)):
                # Reset baselines + controller memory, but DO NOT reset action_counter
                # (otherwise we will regenerate already-used command UUIDs).
                self.state.baseline = {}
                self.state.stagnation_ticks = 0
                self.state.axis_index = 0
                self.state.pressure_level = float(getattr(sim_cfg, "pressure_autopilot_pressure_init", 0.50) or 0.50)
                self.state.last_epi_value = None
                self.state.last_av_value = None
                self.state.last_tick_generation = -1

            # If pressure_level was never initialized (fresh state), seed it.
            if int(getattr(self.state, "last_tick_generation", -1)) < 0:
                try:
                    self.state.pressure_level = float(getattr(sim_cfg, "pressure_autopilot_pressure_init", 0.50) or 0.50)
                except Exception:
                    self.state.pressure_level = 0.50

        gi = int(generation_index)
        tick_interval = int(getattr(sim_cfg, "pressure_autopilot_tick_interval_gen", 0) or 0)
        if tick_interval <= 0:
            return
        start_gen = int(getattr(sim_cfg, "pressure_autopilot_start_gen", 0) or 0)
        if gi < start_gen:
            return
        if (gi % tick_interval) != 0:
            return
        if int(getattr(self.state, "last_tick_generation", -1)) == gi:
            return

        # Read signals
        signals = self._read_signals(sim=sim, generation_index=gi)

        # Compute decision
        decision = self._decide(sim=sim, generation_index=gi, signals=signals)

        # Apply (schedule) decision via ControlPlane
        self._schedule(sim=sim, generation_index=gi, decision=decision)

    # -----------------------------
    # Internal helpers
    # -----------------------------
    def _read_signals(self, sim: Any, generation_index: int) -> Dict[str, Any]:
        """Gather the input signals used by the controller."""

        logger = getattr(sim, "logger", None)
        last_row = None
        if logger is not None:
            try:
                last_row = getattr(logger, "last_row", None)
            except Exception:
                last_row = None

        survival_rate = None
        diversity = None
        collapse_streak = None
        alive = None

        if isinstance(last_row, dict):
            survival_rate = _safe_float(last_row.get("cohort_survival_rate", None), default=0.0)
            diversity = _safe_float(last_row.get("genome_diversity_norm_std_mean", None), default=0.0)
            collapse_streak = _safe_int(last_row.get("diversity_collapse_streak", None), default=0)
            alive = _safe_int(last_row.get("alive", None), default=0)
        else:
            # Fallback: compute from current population snapshot.
            try:
                pop = getattr(sim, "pop", None)
                pop_cfg = getattr(sim, "pop_cfg", None)
                alive_agents = list(pop.alive_agents()) if pop is not None else []
                alive = int(len(alive_agents))
                N0 = int(getattr(pop_cfg, "population_size", 0) or 0)
                survival_rate = (float(alive) / float(N0)) if N0 > 0 else 0.0
            except Exception:
                survival_rate = 0.0
                alive = 0
            diversity = 0.0
            collapse_streak = 0

        epi_value, av_value = self._read_epi_av(sim=sim, generation_index=generation_index)

        return {
            "survival_rate": float(survival_rate if survival_rate is not None else 0.0),
            "alive": int(alive if alive is not None else 0),
            "diversity": float(diversity if diversity is not None else 0.0),
            "collapse_streak": int(collapse_streak if collapse_streak is not None else 0),
            "epi_value": None if epi_value is None else float(epi_value),
            "av_value": None if av_value is None else float(av_value),
        }

    def _read_epi_av(self, sim: Any, generation_index: int) -> Tuple[Optional[float], Optional[float]]:
        """Best-effort read of the latest EPI and AV values."""

        # Prefer in-memory Phase5MetricsEngine (freshly computed at SafeBoundary).
        try:
            me = getattr(sim, "_phase5_metrics_engine", None)
            if me is not None:
                epi_rec = getattr(me, "last_epi_record", None)
                av_rec = getattr(me, "last_av_record", None)
                if isinstance(epi_rec, dict) and int(epi_rec.get("generation", -1)) == int(generation_index):
                    epi_value = _safe_float(epi_rec.get("epi_value", None), default=0.0)
                else:
                    epi_value = None
                if isinstance(av_rec, dict) and int(av_rec.get("generation", -1)) == int(generation_index):
                    avv = av_rec.get("av_value", None)
                    av_value = None if avv is None else _safe_float(avv, default=0.0)
                else:
                    av_value = None
                return epi_value, av_value
        except Exception:
            pass

        # Fallback: read from disk JSON artifacts.
        epi_path = os.path.join(self.run_dir, "epi_v2.json")
        av_path = os.path.join(self.run_dir, "av_v2.json")
        epi_obj = _read_json_file(epi_path)
        av_obj = _read_json_file(av_path)
        epi_value = None
        av_value = None
        try:
            if isinstance(epi_obj, dict):
                h = epi_obj.get("history", [])
                if isinstance(h, list) and h:
                    last = h[-1]
                    if isinstance(last, dict) and int(last.get("generation", -1)) == int(generation_index):
                        epi_value = _safe_float(last.get("epi_value", None), default=0.0)
        except Exception:
            epi_value = None
        try:
            if isinstance(av_obj, dict):
                h = av_obj.get("history", [])
                if isinstance(h, list) and h:
                    last = h[-1]
                    if isinstance(last, dict) and int(last.get("generation", -1)) == int(generation_index):
                        avv = last.get("av_value", None)
                        av_value = None if avv is None else _safe_float(avv, default=0.0)
        except Exception:
            av_value = None
        return epi_value, av_value

    def _knobs_for_profile(self, sim: Any) -> Tuple[List[_KnobSpec], List[str]]:
        """Return knob specs and axis ordering for the active profile."""

        sim_cfg = getattr(sim, "sim_cfg", None)
        profile = str(getattr(sim_cfg, "pressure_autopilot_profile", "balanced") or "balanced").strip().lower()

        # Default axis ordering for cycle profiles.
        axes = ["scarcity", "cost", "shock"]

        # Core knobs used by all profiles.
        knobs: List[_KnobSpec] = []

        # Scarcity axis: env food spawn count.
        knobs.append(
            _KnobSpec(
                path="env.cfg.food_spawn_per_step",
                axis="scarcity",
                kind="int",
                low_mult=1.30,
                high_mult=0.70,
                min_value=1.0,
                max_value=1000000.0,
            )
        )

        # Cost axis: baseline movement energy cost.
        knobs.append(
            _KnobSpec(
                path="env.cfg.move_cost",
                axis="cost",
                kind="float",
                low_mult=0.85,
                high_mult=1.15,
                min_value=0.0,
                max_value=1000000.0,
            )
        )

        # Shock axis: change how often shocks occur.
        allow_shock_enable = bool(getattr(sim_cfg, "pressure_autopilot_allow_shock_enable", False))
        knobs.append(
            _KnobSpec(
                path="sim_cfg.shock_start_prob",
                axis="shock",
                kind="float",
                low_mult=0.60,
                high_mult=1.40,
                min_value=0.0,
                max_value=1.0,
                allow_enable_if_zero=allow_shock_enable,
                baseline_if_zero=float(getattr(sim_cfg, "pressure_autopilot_shock_baseline_if_zero", 0.02) or 0.02),
            )
        )

        # Balanced profile uses all axes at once.
        if profile == "balanced":
            axes = ["all"]
            for k in knobs:
                k.axis = "all"  # type: ignore[misc]
            return knobs, axes

        if profile == "scripted":
            return [], ["scripted"]

        # Cycle profile: one axis active at a time.
        if profile in ("cycle", "cycle3"):
            return knobs, axes

        # Single-axis profiles: keep only one knob axis.
        if profile in ("scarcity", "cost", "shock"):
            axes = [profile]
            knobs = [k for k in knobs if k.axis == profile]
            return knobs, axes

        # Unknown profile: fail closed (balanced-like but explicit).
        axes = ["all"]
        for k in knobs:
            k.axis = "all"  # type: ignore[misc]
        return knobs, axes

    def _get_current_value(self, sim: Any, path: str) -> Any:
        """Resolve current value at dot-path (same traversal rules as commands)._"""

        parts = [p for p in str(path).split(".") if p]
        cur: Any = sim
        for p in parts:
            if isinstance(cur, dict):
                cur = cur.get(p)
            else:
                cur = getattr(cur, p, None)
        return cur

    def _capture_baseline_if_needed(self, sim: Any, knobs: List[_KnobSpec]) -> None:
        """Capture baseline values for knobs the first time the autopilot runs."""

        for k in knobs:
            if k.path in self.state.baseline:
                continue
            cur = self._get_current_value(sim, k.path)
            # Store JSON-serializable primitives only.
            if k.kind == "int":
                self.state.baseline[k.path] = int(_safe_int(cur, default=0))
            elif k.kind == "float":
                self.state.baseline[k.path] = float(_safe_float(cur, default=0.0))
            else:
                self.state.baseline[k.path] = cur

    def _script_entries(self, sim: Any) -> List[Dict[str, Any]]:
        """Return normalized scripted-autopilot entries.

        Expected config shape:
        {
          "schema_version": "pressure_autopilot_script_v1",
          "entries": [
            {
              "decision_generation": 300,
              "target_generation": 301,
              "label": "apply",
              "updates": [{"path": "sim_cfg.foo", "value": 1.23}],
            }
          ]
        }
        """

        sim_cfg = getattr(sim, "sim_cfg", None)
        spec = getattr(sim_cfg, "pressure_autopilot_script", {}) if sim_cfg is not None else {}
        if not isinstance(spec, dict):
            return []
        raw_entries = spec.get("entries", [])
        if not isinstance(raw_entries, list):
            return []

        out: List[Dict[str, Any]] = []
        for idx, raw in enumerate(raw_entries):
            if not isinstance(raw, dict):
                continue
            try:
                decision_generation = int(raw.get("decision_generation"))
                target_generation = int(raw.get("target_generation", decision_generation + 1))
            except Exception:
                continue
            updates = raw.get("updates", [])
            if not isinstance(updates, list):
                continue
            norm_updates: List[Dict[str, Any]] = []
            for upd in updates:
                if not isinstance(upd, dict):
                    continue
                path = str(upd.get("path", "") or "")
                if not path:
                    continue
                norm_updates.append({"path": path, "value": upd.get("value", None)})
            out.append(
                {
                    "entry_index": int(idx),
                    "decision_generation": int(decision_generation),
                    "target_generation": int(target_generation),
                    "label": str(raw.get("label", f"entry_{idx}") or f"entry_{idx}"),
                    "updates": norm_updates,
                }
            )

        out.sort(key=lambda item: (int(item["decision_generation"]), int(item["target_generation"]), int(item["entry_index"])))
        return out

    def _decide(self, sim: Any, generation_index: int, signals: Dict[str, Any]) -> Dict[str, Any]:
        """Compute the next controller decision.

        Returns a dict that is later turned into a PressureUpdateCommand.
        """

        sim_cfg = getattr(sim, "sim_cfg", None)
        if sim_cfg is None:
            return {}

        profile = str(getattr(sim_cfg, "pressure_autopilot_profile", "balanced") or "balanced").strip().lower()
        if profile == "scripted":
            entries = self._script_entries(sim)
            cursor = max(0, int(getattr(self.state, "script_cursor", 0) or 0))
            while cursor < len(entries) and int(entries[cursor]["decision_generation"]) < int(generation_index):
                cursor += 1
            self.state.script_cursor = int(cursor)
            if cursor >= len(entries):
                return {
                    "schema_version": AUTOPILOT_DECISION_SCHEMA_VERSION,
                    "generation": int(generation_index),
                    "profile": "scripted",
                    "axis": "scripted",
                    "pressure_level": float(self.state.pressure_level),
                    "signals": dict(signals),
                    "updates": [],
                }
            entry = entries[cursor]
            if int(entry["decision_generation"]) != int(generation_index):
                return {
                    "schema_version": AUTOPILOT_DECISION_SCHEMA_VERSION,
                    "generation": int(generation_index),
                    "profile": "scripted",
                    "axis": "scripted",
                    "pressure_level": float(self.state.pressure_level),
                    "signals": dict(signals),
                    "updates": [],
                }
            return {
                "schema_version": AUTOPILOT_DECISION_SCHEMA_VERSION,
                "generation": int(generation_index),
                "profile": "scripted",
                "axis": "scripted",
                "pressure_level": float(self.state.pressure_level),
                "signals": dict(signals),
                "label": str(entry["label"]),
                "entry_index": int(entry["entry_index"]),
                "target_generation": int(entry["target_generation"]),
                "updates": list(entry["updates"]),
            }

        knobs, axes = self._knobs_for_profile(sim)

        # Ensure baselines exist for knob paths.
        self._capture_baseline_if_needed(sim, knobs)

        # -------- Axis switching (cycle profiles) --------
        epi_value = signals.get("epi_value", None)
        collapse_streak = int(signals.get("collapse_streak", 0) or 0)

        if profile in ("cycle", "cycle3") and axes:
            improve_eps = float(getattr(sim_cfg, "pressure_autopilot_epi_improve_eps", 0.25) or 0.25)
            stagnation_required = int(getattr(sim_cfg, "pressure_autopilot_stagnation_ticks_required", 2) or 2)
            collapse_switch = int(getattr(sim_cfg, "pressure_autopilot_collapse_streak_switch", 3) or 3)

            improved = False
            if epi_value is not None:
                last_epi = self.state.last_epi_value
                if last_epi is None:
                    improved = True
                else:
                    improved = bool(float(epi_value) > float(last_epi) + float(improve_eps))

            if improved:
                self.state.stagnation_ticks = 0
            else:
                self.state.stagnation_ticks = int(self.state.stagnation_ticks) + 1

            should_switch = bool(self.state.stagnation_ticks >= stagnation_required)
            if collapse_streak >= collapse_switch:
                should_switch = True

            if should_switch:
                self.state.axis_index = int(self.state.axis_index + 1) % int(len(axes))
                self.state.stagnation_ticks = 0

        # Active axis
        axis = axes[int(self.state.axis_index) % int(len(axes))] if axes else "all"

        # -------- Pressure update (PID-ish) --------
        p = float(self.state.pressure_level)

        survival = float(signals.get("survival_rate", 0.0) or 0.0)
        diversity = float(signals.get("diversity", 0.0) or 0.0)
        av_value = signals.get("av_value", None)

        target_survival = float(getattr(sim_cfg, "pressure_autopilot_target_survival", 0.55) or 0.55)
        deadband = float(getattr(sim_cfg, "pressure_autopilot_survival_deadband", 0.05) or 0.05)
        delta_up = float(getattr(sim_cfg, "pressure_autopilot_delta_up", 0.05) or 0.05)
        delta_down = float(getattr(sim_cfg, "pressure_autopilot_delta_down", 0.10) or 0.10)

        survival_floor = float(getattr(sim_cfg, "pressure_autopilot_survival_floor", 0.20) or 0.20)
        survival_ceiling = float(getattr(sim_cfg, "pressure_autopilot_survival_ceiling", 0.80) or 0.80)
        diversity_floor = float(getattr(sim_cfg, "pressure_autopilot_diversity_floor", 0.0) or 0.0)

        # Hard safety first.
        if survival < survival_floor:
            p = p - (2.0 * delta_down)
        elif survival > survival_ceiling:
            p = p + (1.5 * delta_up)
        else:
            if survival > (target_survival + deadband):
                p = p + delta_up
            elif survival < (target_survival - deadband):
                p = p - delta_down

        # AV influence (only when non-transient)
        if av_value is not None:
            av_low = float(getattr(sim_cfg, "pressure_autopilot_av_low", -0.25) or -0.25)
            av_high = float(getattr(sim_cfg, "pressure_autopilot_av_high", 0.25) or 0.25)
            if float(av_value) < av_low:
                p = p - delta_down
            elif float(av_value) > av_high and survival >= target_survival:
                p = p + (0.5 * delta_up)
            else:
                # If AV is flat and survival is high, we can push.
                av_dead = float(getattr(sim_cfg, "pressure_autopilot_av_deadband", 0.05) or 0.05)
                if abs(float(av_value)) < av_dead and survival > (target_survival + deadband):
                    p = p + (0.25 * delta_up)

        # Diversity relief: low diversity usually means selection is too sharp.
        if diversity_floor > 0.0 and diversity < diversity_floor and survival < target_survival:
            p = p - (0.5 * delta_down)

        p = _clamp(p, 0.0, 1.0)

        # Store controller state
        self.state.pressure_level = float(p)
        self.state.last_epi_value = None if epi_value is None else float(epi_value)
        self.state.last_av_value = None if av_value is None else float(av_value)

        # -------- Translate pressure into knob targets --------
        updates: List[Dict[str, Any]] = []
        float_eps = float(getattr(sim_cfg, "pressure_autopilot_float_eps", 1e-6) or 1e-6)

        # In cycle mode, inactive knobs get baseline (pressure=0.5), so regimes truly switch.
        p_inactive = 0.50

        for k in knobs:
            base = self.state.baseline.get(k.path, None)
            if base is None:
                continue

            # If baseline is zero and enabling is disallowed, keep it zero.
            if k.kind == "float" and float(base) == 0.0 and (not k.allow_enable_if_zero):
                target = 0.0
            else:
                base_f = float(base)
                if base_f == 0.0 and k.allow_enable_if_zero:
                    base_f = float(k.baseline_if_zero)

                # Choose axis pressure.
                p_axis = p
                if profile in ("cycle", "cycle3") and k.axis != axis:
                    p_axis = p_inactive

                mult = float(k.low_mult) + (float(k.high_mult) - float(k.low_mult)) * float(p_axis)
                target = base_f * mult

            target = float(_clamp(float(target), float(k.min_value), float(k.max_value)))

            cur = self._get_current_value(sim, k.path)

            if k.kind == "int":
                tgt_i = int(round(float(target)))
                cur_i = int(_safe_int(cur, default=tgt_i))
                if cur_i != tgt_i:
                    updates.append({"path": str(k.path), "value": int(tgt_i)})
            else:
                tgt_f = float(target)
                cur_f = float(_safe_float(cur, default=tgt_f))
                if abs(cur_f - tgt_f) > float_eps:
                    updates.append({"path": str(k.path), "value": float(tgt_f)})

        return {
            "schema_version": AUTOPILOT_DECISION_SCHEMA_VERSION,
            "generation": int(generation_index),
            "profile": str(profile),
            "axis": str(axis),
            "pressure_level": float(p),
            "signals": dict(signals),
            "updates": updates,
        }

    def _schedule(self, sim: Any, generation_index: int, decision: Dict[str, Any]) -> None:
        """Schedule the computed updates via ControlPlane."""

        gi = int(generation_index)
        sim_cfg = getattr(sim, "sim_cfg", None)
        cp = getattr(sim, "control_plane", None)
        if sim_cfg is None or cp is None:
            return

        # Option C (advanced): suggestion-only mode.
        # When enabled, we log decisions but never schedule ControlPlane commands.
        if bool(getattr(sim_cfg, "pressure_autopilot_suggestion_only", False)):
            updates = decision.get("updates", [])
            if not isinstance(updates, list) or not updates:
                self._log_decision(decision, ack_status="NOOP", ack_reason="")
            else:
                self._log_decision(decision, ack_status="SUGGESTION_ONLY", ack_reason="suggestion-only mode")
            self._persist_state_best_effort()
            return


        # Mark tick even if we do nothing, so the controller doesn't re-run on the same boundary.
        self.state.last_tick_generation = gi

        updates = decision.get("updates", [])
        if not isinstance(updates, list) or not updates:
            self._log_decision(decision, ack_status="NOOP", ack_reason="")
            self._persist_state_best_effort()
            return

        target_generation_raw = decision.get("target_generation", None)
        if target_generation_raw is None:
            delay = int(getattr(sim_cfg, "pressure_autopilot_apply_delay_gen", 1) or 1)
            if delay < 1:
                delay = 1
            target_generation = gi + delay
        else:
            target_generation = int(target_generation_raw)
            if target_generation <= gi:
                target_generation = gi + 1

        # Deterministic command uuid.
        branch_id = str(getattr(sim_cfg, "branch_id", "default") or "default")
        salt = hashlib.sha256(canonical_json_bytes(decision)).hexdigest()[:12]
        name = f"pressure_autopilot|{branch_id}|g{gi}|n{int(self.state.action_counter)}|{salt}"
        cu = _uuid5(name)

        cmd = PressureUpdateCommand(
            schema_version="control_command_v1",
            command_uuid=str(cu),
            target_generation=int(target_generation),
            updates=list(updates),
        )

        ack_status = "UNKNOWN"
        ack_reason = ""
        try:
            ack = cp.submit_command(cmd)
            ack_status = str(getattr(ack, "status", "UNKNOWN") or "UNKNOWN")
            ack_reason = str(getattr(ack, "reason", "") or "")
        except Exception as e:
            ack_status = "ERROR"
            ack_reason = f"submit_command error: {type(e).__name__}"

        # Update counter only if accepted (keeps uuid chain stable across retries).
        if ack_status == "ACCEPTED":
            self.state.action_counter = int(self.state.action_counter) + 1
            if str(decision.get("profile", "") or "").strip().lower() == "scripted":
                self.state.script_cursor = int(getattr(self.state, "script_cursor", 0) or 0) + 1

        self._log_decision(decision, ack_status=ack_status, ack_reason=ack_reason)
        self._persist_state_best_effort()

    def _log_decision(self, decision: Dict[str, Any], *, ack_status: str, ack_reason: str) -> None:
        """Append one decision record to jsonl (observability only)."""

        rec = dict(decision)
        rec["ack_status"] = str(ack_status)
        rec["ack_reason"] = str(ack_reason or "")
        rec["state"] = self.state.to_dict()

        try:
            line = canonical_json_bytes(rec).decode("utf-8") + "\n"
            os.makedirs(os.path.dirname(self._decisions_path), exist_ok=True)
            with open(self._decisions_path, "a", encoding="utf-8") as f:
                f.write(line)
        except Exception:
            # Never crash the sim due to logging.
            pass

    def _persist_state_best_effort(self) -> None:
        """Persist state to disk (debug/observability only)."""

        try:
            data = canonical_json_bytes_lossless(self.state.to_dict())
            _atomic_write_bytes(self._state_path, data)
        except Exception:
            pass
