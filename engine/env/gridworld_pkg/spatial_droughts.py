from __future__ import annotations

"""
env/gridworld_pkg/spatial_droughts.py

Tier 2: Spatial Droughts (Regional Stress Events)
-------------------------------------------------

This module implements *time-varying* regional resource stress (droughts) on top of:
- Base environment parameters
- Tier 1 static spatial ecology maps (resource_regrowth_map, resource_spawn_density_map)

Design constraints (project principles):
- World-level only: depends only on time, space, and seed.
- No agent inspection: never reads genomes, NN state, fitness, etc.
- Reproducible: explicit schedules are deterministic; stochastic schedules are seeded and
  all realized events are logged. On resume, the realized log is authoritative.

Semantics:
- Regions are compiled into boolean masks using slicing bounds [x0:x1), [y0:y1).
- Each drought event contributes a multiplier per channel via:
    m_event(t) = 1 - intensity(t) * (1 - target_mult)
  where intensity(t) in [0,1] ramps in/out (linear by default) and target_mult is the
  multiplier at full drought intensity.

- Overlaps combine per-cell using a stack rule:
    - multiply: product of active multipliers
    - min:      minimum active multiplier

Logging (when run_dir is provided):
- run_dir/droughts/drought_events.jsonl          (one JSON record per realized event)
- run_dir/droughts/drought_intensity_timeseries.csv (per-step summary)
- run_dir/droughts/drought_stats_periodic.csv    (every N steps)
- run_dir/droughts/regions/<id>_mask.npy         (region masks)
- run_dir/droughts/regions/drought_region_id_map.npy (int32 id raster, -1 = none)

This file is intentionally independent of any agent implementation.

Last modified: 2026-02-24 03:45:00 CET (Europe/Vienna)
"""

import csv
import hashlib
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


logger = logging.getLogger(__name__)
_ONCE_WARNINGS: set[str] = set()


def _warn_once(*, key: str, msg: str) -> None:
    """Emit a warning at most once per process for a given key.

    Used to prevent console spam when spatial drought specs are compiled
    repeatedly in shadow-clone runners (EPI).
    """
    try:
        if key in _ONCE_WARNINGS:
            return
        _ONCE_WARNINGS.add(key)
        logger.warning(msg)
    except Exception:
        return


# -----------------------------
# Stable hashing / seed helpers
# -----------------------------


def stable_hash_uint64(text: str) -> int:
    """Return a stable uint64 hash of a string using SHA256."""
    h = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(h[:8], byteorder="little", signed=False)


# -----------------------------
# Data model
# -----------------------------


@dataclass
class DroughtRegion:
    """A compiled drought region."""
    region_id: str
    mask: np.ndarray  # bool (H,W)
    affects_regrowth: bool
    affects_spawn: bool
    target_regrowth: float
    target_spawn: float
    duration_steps: int
    recovery_steps: int
    ramp_steps: int


@dataclass
class DroughtEvent:
    """A realized drought event (explicit or stochastic)."""
    event_id: int
    region_id: str
    start_step: int
    peak_start: int
    peak_end: int
    end_step: int
    target_regrowth: float
    target_spawn: float
    ramp_steps: int
    duration_steps: int
    recovery_steps: int
    mode: str  # "explicit" | "stochastic"


# -----------------------------
# Core system
# -----------------------------


class SpatialDroughtSystem:
    """Tier 2 drought controller.

    Public contract:
    - step(abs_step) -> (regrowth_mult_map, spawn_mult_map)
      Returns current per-cell multipliers for each channel.

    This object owns all drought logging. It never inspects agents.
    """

    def __init__(
        self,
        *,
        width: int,
        height: int,
        enabled: bool,
        spec: Optional[Dict[str, Any]],
        base_seed: int,
        run_dir: Optional[str],
        periodic_stats_every: int = 100,
        timeseries_every: int = 1,
    ):
        """
        Args:
            width, height: Grid dimensions.
            enabled: Feature gate. If False, system is inert.
            spec: Drought spec dict (see docs). If None, treated as empty.
            base_seed: Run seed used for stochastic scheduling.
            run_dir: If provided, drought artifacts are written here.
            periodic_stats_every: Frequency for drought_stats_periodic.csv.
            timeseries_every: Frequency for drought_intensity_timeseries.csv.
        """
        self.width = int(width)
        self.height = int(height)
        self.enabled = bool(enabled)
        self.spec = dict(spec or {})
        self.base_seed = int(base_seed)
        self.run_dir = run_dir

        self._periodic_every = int(max(1, periodic_stats_every))
        self._timeseries_every = int(max(1, timeseries_every))

        self._regions: Dict[str, DroughtRegion] = {}
        self._events: List[DroughtEvent] = []
        self._next_event_id: int = 1

        # Cached per-cell multipliers (updated by step())
        self._regrowth_map = np.ones((self.height, self.width), dtype=np.float32)
        self._spawn_map = np.ones((self.height, self.width), dtype=np.float32)

        # Scheduling state (stochastic)
        self._stoch_cfg: Dict[str, Any] = {}
        self._last_start_by_region: Dict[str, int] = {}

        # IO
        self._paths = {}
        self._ts_fp = None
        self._ts_writer = None
        self._ps_fp = None
        self._ps_writer = None

        if not self.enabled:
            return

        self._validate_and_init()

    # -----------------------------
    # Initialization
    # -----------------------------

    def _validate_and_init(self) -> None:
        """Validate spec, compile regions, load any prior event log, and init writers."""
        # Composition defaults
        comp = dict(self.spec.get("composition") or {})
        self.stack_rule = str(comp.get("stack_rule", "multiply")).strip().lower()
        if self.stack_rule not in ("multiply", "min"):
            raise ValueError(f"spatial_droughts.composition.stack_rule must be multiply|min, got {self.stack_rule}")

        self.min_regrowth_mult = float(comp.get("min_regrowth_mult", 0.0))
        self.min_spawn_mult = float(comp.get("min_spawn_mult", 0.0))
        self.allow_gt1 = bool(comp.get("allow_gt1", False))

        dyn = dict(self.spec.get("dynamics") or {})
        self.ramp_shape = str(dyn.get("ramp_shape", "linear")).strip().lower()
        # Tier 2 implements linear only. Others are schema hooks.
        if self.ramp_shape not in ("linear", "cosine", "smoothstep"):
            raise ValueError(f"spatial_droughts.dynamics.ramp_shape invalid: {self.ramp_shape}")
        self.production_only = bool(dyn.get("production_only", True))

        # Compile regions
        regions = list(self.spec.get("regions") or [])
        self._compile_regions(regions)

        # IO paths
        if self.run_dir:
            root = Path(self.run_dir) / "droughts"
            regions_dir = root / "regions"
            regions_dir.mkdir(parents=True, exist_ok=True)
            self._paths = {
                "root": root,
                "regions": regions_dir,
                "events": root / "drought_events.jsonl",
                "timeseries": root / "drought_intensity_timeseries.csv",
                "periodic": root / "drought_stats_periodic.csv",
                "readme": root / "README.txt",
            }
            self._write_readme()

            # Save region rasters
            self._save_region_rasters()

            # Load existing realized events if present (authoritative on resume)
            self._load_existing_events()

            # Init writers
            self._init_writers()

        # Init schedule mode
        sched = dict(self.spec.get("schedule") or {})
        self.schedule_mode = str(sched.get("mode", "explicit")).strip().lower()
        if self.schedule_mode not in ("explicit", "stochastic"):
            raise ValueError(f"spatial_droughts.schedule.mode must be explicit|stochastic, got {self.schedule_mode}")

        if self.schedule_mode == "explicit":
            if not self._events:
                self._init_explicit_events(sched.get("explicit") or [])
        else:
            self._init_stochastic(sched.get("stochastic") or {})

    def _write_readme(self) -> None:
        """Write a short doc into the run dir for future debugging sanity."""
        if not self.run_dir:
            return
        p = self._paths.get("readme")
        if not p:
            return
        txt = (
            "Tier 2 Spatial Droughts\n"
            "======================\n\n"
            "Composition order with Tier 1:\n"
            "  effective = base * tier1_static_map * drought_multiplier\n\n"
            "Channels:\n"
            "  - regrowth: scales resource regrowth / global spawn intensity proxy\n"
            "  - spawn: scales spawn density weights (where spawns go)\n\n"
            "Notes:\n"
            "  - Tier 2 is production-only by default (existing resources do not decay).\n"
            "  - If stochastic mode is enabled, drought_events.jsonl is authoritative on resume.\n"
        )
        try:
            Path(p).write_text(txt, encoding="utf-8")
        except Exception:
            pass

    def _compile_regions(self, regions: Sequence[Dict[str, Any]]) -> None:
        """Compile region specs into masks + region objects."""
        self._regions = {}

        # Build region_id raster for debugging overlaps.
        # Policy: last-wins, but region_id_map is diagnostic only.
        region_id_map = np.full((self.height, self.width), -1, dtype=np.int32)

        for idx, r in enumerate(regions):
            region_id = str(r.get("id") or r.get("region_id") or "").strip()
            if not region_id:
                raise ValueError("Each drought region must have an 'id'")

            bounds = dict(r.get("bounds") or {})
            # Use Python slicing semantics [x0:x1), [y0:y1)
            x0 = int(bounds.get("x0", 0))
            x1 = int(bounds.get("x1", self.width))
            y0 = int(bounds.get("y0", 0))
            y1 = int(bounds.get("y1", self.height))

            # Normalize bounds
            if x1 < x0:
                x0, x1 = x1, x0
            if y1 < y0:
                y0, y1 = y1, y0
            x0 = max(0, min(self.width, x0))
            x1 = max(0, min(self.width, x1))
            y0 = max(0, min(self.height, y0))
            y1 = max(0, min(self.height, y1))

            mask = np.zeros((self.height, self.width), dtype=bool)
            if x0 != x1 and y0 != y1:
                mask[y0:y1, x0:x1] = True
                region_id_map[y0:y1, x0:x1] = int(idx)

            affects = list(r.get("affects") or ["regrowth"])
            affects = [str(a).strip().lower() for a in affects]
            affects_regrowth = "regrowth" in affects
            affects_spawn = "spawn" in affects

            tmult = dict(r.get("target_mult") or {})
            target_regrowth = float(tmult.get("regrowth", 1.0))
            target_spawn = float(tmult.get("spawn", 1.0))

            if not self.allow_gt1:
                if target_regrowth > 1.0 or target_spawn > 1.0:
                    # Do not hard fail: clamp to 1.0 to avoid surprise floods in Tier 2.
                    _warn_once(
                        key=f"spatial_droughts_clamp_gt1::{region_id}",
                        msg=(
                            "[spatial-droughts] target_mult > 1.0 not allowed (allow_gt1=false). "
                            f"Clamping to 1.0 for region={region_id}"
                        ),
                    )
                    target_regrowth = min(target_regrowth, 1.0)
                    target_spawn = min(target_spawn, 1.0)

            timing = dict(r.get("timing") or {})
            duration = int(timing.get("duration_steps", 0))
            recovery = int(timing.get("recovery_steps", 0))
            ramp = int(timing.get("ramp_steps", 0))
            duration = max(0, duration)
            recovery = max(0, recovery)
            ramp = max(0, ramp)

            self._regions[region_id] = DroughtRegion(
                region_id=region_id,
                mask=mask,
                affects_regrowth=bool(affects_regrowth),
                affects_spawn=bool(affects_spawn),
                target_regrowth=float(target_regrowth),
                target_spawn=float(target_spawn),
                duration_steps=duration,
                recovery_steps=recovery,
                ramp_steps=ramp,
            )

        self._region_id_map = region_id_map

    def _save_region_rasters(self) -> None:
        """Persist region masks and region_id map."""
        if not self.run_dir:
            return
        try:
            np.save(str(self._paths["regions"] / "drought_region_id_map.npy"), self._region_id_map.astype(np.int32, copy=False))
        except Exception:
            pass
        for rid, reg in self._regions.items():
            try:
                np.save(str(self._paths["regions"] / f"{rid}_mask.npy"), reg.mask.astype(np.uint8, copy=False))
            except Exception:
                pass

    def _load_existing_events(self) -> None:
        """Load existing realized events if event log exists (authoritative)."""
        p = self._paths.get("events")
        if not p or not Path(p).exists():
            return
        events = []
        try:
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    events.append(
                        DroughtEvent(
                            event_id=int(rec["event_id"]),
                            region_id=str(rec["region_id"]),
                            start_step=int(rec["start_step"]),
                            peak_start=int(rec["peak_start"]),
                            peak_end=int(rec["peak_end"]),
                            end_step=int(rec["end_step"]),
                            target_regrowth=float(rec.get("target_regrowth", 1.0)),
                            target_spawn=float(rec.get("target_spawn", 1.0)),
                            ramp_steps=int(rec.get("ramp_steps", 0)),
                            duration_steps=int(rec.get("duration_steps", 0)),
                            recovery_steps=int(rec.get("recovery_steps", 0)),
                            mode=str(rec.get("mode", "unknown")),
                        )
                    )
        except Exception:
            return

        if events:
            self._events = sorted(events, key=lambda e: (e.start_step, e.event_id))
            self._next_event_id = max(e.event_id for e in self._events) + 1

    def _init_writers(self) -> None:
        """Initialize CSV writers for time series and periodic stats."""
        if not self.run_dir:
            return

        # Timeseries
        ts_path = self._paths["timeseries"]
        # Always rewrite if not present; if present, append (resume).
        ts_exists = Path(ts_path).exists()
        self._ts_fp = open(ts_path, "a", newline="", encoding="utf-8")
        self._ts_writer = csv.writer(self._ts_fp)
        if not ts_exists:
            header = ["step", "global_mean_mult", "global_min_mult", "global_max_mult"]
            for rid in self._regions.keys():
                header.append(f"region:{rid}_mult")
            self._ts_writer.writerow(header)
            self._ts_fp.flush()

        # Periodic stats
        ps_path = self._paths["periodic"]
        ps_exists = Path(ps_path).exists()
        self._ps_fp = open(ps_path, "a", newline="", encoding="utf-8")
        self._ps_writer = csv.writer(self._ps_fp)
        if not ps_exists:
            self._ps_writer.writerow(["step", "mean_mult", "min_mult", "max_mult"])
            self._ps_fp.flush()

    def _close_writers(self) -> None:
        """Close file handles."""
        try:
            if self._ts_fp:
                self._ts_fp.flush()
                self._ts_fp.close()
        except Exception:
            pass
        try:
            if self._ps_fp:
                self._ps_fp.flush()
                self._ps_fp.close()
        except Exception:
            pass

    # -----------------------------
    # Schedule initialization
    # -----------------------------

    def _init_explicit_events(self, explicit: Sequence[Dict[str, Any]]) -> None:
        """Build events from explicit schedule list."""
        for item in list(explicit or []):
            region_id = str(item.get("region_id") or item.get("id") or "").strip()
            if not region_id:
                raise ValueError("Explicit drought schedule entries require region_id")
            if region_id not in self._regions:
                raise ValueError(f"Explicit drought schedule references unknown region_id={region_id}")

            start = int(item.get("start_step", 0))
            reg = self._regions[region_id]

            # Allow per-event overrides (optional)
            t_reg = float(item.get("target_regrowth", reg.target_regrowth))
            t_spa = float(item.get("target_spawn", reg.target_spawn))
            ramp = int(item.get("ramp_steps", reg.ramp_steps))
            duration = int(item.get("duration_steps", reg.duration_steps))
            recovery = int(item.get("recovery_steps", reg.recovery_steps))

            self._create_event(
                region_id=region_id,
                start_step=start,
                ramp_steps=ramp,
                duration_steps=duration,
                recovery_steps=recovery,
                target_regrowth=t_reg,
                target_spawn=t_spa,
                mode="explicit",
                write_log=True,
            )

        # Sort events once (evaluation is cheap)
        self._events.sort(key=lambda e: (e.start_step, e.event_id))

    def _init_stochastic(self, stoch: Dict[str, Any]) -> None:
        """Initialize stochastic schedule config.

        Algorithm: per-region independent Bernoulli-per-step with min_gap.

        Parameters:
        - expected_events_per_10k_steps: float
        - min_gap_steps: int
        """
        self._stoch_cfg = dict(stoch or {})
        algo = str(self._stoch_cfg.get("algorithm", "per_region_poisson")).strip()
        if algo not in ("per_region_poisson", "per_region_bernoulli"):
            raise ValueError(f"Unsupported stochastic algorithm: {algo}")

        self._stoch_cfg.setdefault("algorithm", algo)
        rate = float(self._stoch_cfg.get("expected_events_per_10k_steps", 0.0))
        self._stoch_cfg["expected_events_per_10k_steps"] = max(0.0, rate)
        self._stoch_cfg["min_gap_steps"] = int(max(0, int(self._stoch_cfg.get("min_gap_steps", 0))))
        # Optional separate seed override
        if "seed" in self._stoch_cfg and self._stoch_cfg["seed"] is not None:
            self._stoch_seed = int(self._stoch_cfg["seed"])
        else:
            self._stoch_seed = self.base_seed

        # Initialize last start times from existing log if present
        for e in self._events:
            self._last_start_by_region[e.region_id] = max(self._last_start_by_region.get(e.region_id, -10**18), e.start_step)

    # -----------------------------
    # Event creation & logging
    # -----------------------------

    def _create_event(
        self,
        *,
        region_id: str,
        start_step: int,
        ramp_steps: int,
        duration_steps: int,
        recovery_steps: int,
        target_regrowth: float,
        target_spawn: float,
        mode: str,
        write_log: bool,
    ) -> DroughtEvent:
        """Create an event instance and optionally append to event log immediately."""
        ramp_steps = int(max(0, ramp_steps))
        duration_steps = int(max(0, duration_steps))
        recovery_steps = int(max(0, recovery_steps))

        peak_start = start_step + ramp_steps
        peak_end = peak_start + duration_steps
        end_step = peak_end + recovery_steps

        ev = DroughtEvent(
            event_id=int(self._next_event_id),
            region_id=str(region_id),
            start_step=int(start_step),
            peak_start=int(peak_start),
            peak_end=int(peak_end),
            end_step=int(end_step),
            target_regrowth=float(target_regrowth),
            target_spawn=float(target_spawn),
            ramp_steps=int(ramp_steps),
            duration_steps=int(duration_steps),
            recovery_steps=int(recovery_steps),
            mode=str(mode),
        )
        self._next_event_id += 1
        self._events.append(ev)

        if write_log and self.run_dir:
            self._append_event_log(ev)

        return ev

    def _append_event_log(self, ev: DroughtEvent) -> None:
        """Append a single JSONL record for an event (immediate logging)."""
        p = self._paths.get("events")
        if not p:
            return
        rec = {
            "event_id": ev.event_id,
            "region_id": ev.region_id,
            "start_step": ev.start_step,
            "peak_start": ev.peak_start,
            "peak_end": ev.peak_end,
            "end_step": ev.end_step,
            "target_regrowth": ev.target_regrowth,
            "target_spawn": ev.target_spawn,
            "ramp_steps": ev.ramp_steps,
            "duration_steps": ev.duration_steps,
            "recovery_steps": ev.recovery_steps,
            "mode": ev.mode,
            "stochastic": (ev.mode == "stochastic"),
            "algorithm": str(self._stoch_cfg.get("algorithm", "")) if hasattr(self, "_stoch_cfg") else "",
            "base_seed": int(self.base_seed),
        }
        try:
            with open(p, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, sort_keys=True) + "\n")
        except Exception:
            pass

    # -----------------------------
    # Per-step update
    # -----------------------------

    def step(self, abs_step: int) -> Tuple[np.ndarray, np.ndarray]:
        """Advance drought state for an absolute step and return multiplier maps.

        Args:
            abs_step: Absolute step index since run start.

        Returns:
            (regrowth_mult_map, spawn_mult_map), both float32 arrays shape (H,W).
        """
        if not self.enabled:
            return self._regrowth_map, self._spawn_map

        t = int(abs_step)

        # If stochastic: possibly start new events (but ONLY if we are not replaying from existing log)
        if self.schedule_mode == "stochastic":
            self._maybe_sample_stochastic_events(t)

        # Recompute from scratch (Tier 2 simplicity)
        self._regrowth_map.fill(1.0)
        self._spawn_map.fill(1.0)

        # Compute current multipliers per region (scalar), then apply over mask
        region_mult_regrowth: Dict[str, float] = {rid: 1.0 for rid in self._regions.keys()}
        region_mult_spawn: Dict[str, float] = {rid: 1.0 for rid in self._regions.keys()}

        # Active events only (cheap filter)
        for ev in self._events:
            if t < ev.start_step or t >= ev.end_step:
                continue

            reg = self._regions.get(ev.region_id)
            if reg is None:
                continue

            intensity = self._event_intensity(ev, t)
            intensity = float(max(0.0, min(1.0, intensity)))

            if reg.affects_regrowth:
                m = 1.0 - intensity * (1.0 - float(ev.target_regrowth))
                region_mult_regrowth[reg.region_id] = self._combine(region_mult_regrowth[reg.region_id], m)

            if reg.affects_spawn:
                m = 1.0 - intensity * (1.0 - float(ev.target_spawn))
                region_mult_spawn[reg.region_id] = self._combine(region_mult_spawn[reg.region_id], m)

        # Apply region scalars to cells
        for rid, reg in self._regions.items():
            if reg.affects_regrowth:
                m = float(region_mult_regrowth.get(rid, 1.0))
                if self.stack_rule == "multiply":
                    self._regrowth_map[reg.mask] *= m
                else:
                    self._regrowth_map[reg.mask] = np.minimum(self._regrowth_map[reg.mask], m)

            if reg.affects_spawn:
                m = float(region_mult_spawn.get(rid, 1.0))
                if self.stack_rule == "multiply":
                    self._spawn_map[reg.mask] *= m
                else:
                    self._spawn_map[reg.mask] = np.minimum(self._spawn_map[reg.mask], m)

        # Floors
        if self.min_regrowth_mult > 0.0:
            np.maximum(self._regrowth_map, float(self.min_regrowth_mult), out=self._regrowth_map)
        if self.min_spawn_mult > 0.0:
            np.maximum(self._spawn_map, float(self.min_spawn_mult), out=self._spawn_map)

        # Log summaries
        self._log_step(t, region_mult_regrowth, region_mult_spawn)

        return self._regrowth_map, self._spawn_map

    def _combine(self, a: float, b: float) -> float:
        """Combine multipliers using stack_rule."""
        if self.stack_rule == "min":
            return float(min(a, b))
        return float(a * b)

    def _event_intensity(self, ev: DroughtEvent, t: int) -> float:
        """Compute intensity(t) in [0,1] for an event at time t."""
        if t < ev.start_step or t >= ev.end_step:
            return 0.0

        # Ramp-in
        if ev.ramp_steps > 0 and t < ev.peak_start:
            frac = (t - ev.start_step) / float(ev.ramp_steps)
            return float(max(0.0, min(1.0, frac)))

        # Plateau
        if t < ev.peak_end:
            return 1.0

        # Ramp-out / recovery
        if ev.recovery_steps > 0:
            frac = (t - ev.peak_end) / float(ev.recovery_steps)
            return float(max(0.0, min(1.0, 1.0 - frac)))

        return 0.0

    # -----------------------------
    # Stochastic sampling
    # -----------------------------

    def _maybe_sample_stochastic_events(self, t: int) -> None:
        """Sample stochastic events at step t (independent per region).

        IMPORTANT: If an event log already exists (resume), we do not "resample the past".
        We only allow sampling to add events strictly after the last logged step.
        """
        # Determine last logged start step (for safety on resume)
        last_logged_start = -10**18
        if self._events:
            last_logged_start = max(last_logged_start, max(ev.start_step for ev in self._events))

        # We only sample for t >= last_logged_start to avoid generating divergent history.
        # If resume starts mid-run, tick loop will revisit earlier t? It shouldn't, but be safe.
        if t < last_logged_start:
            return

        rate_10k = float(self._stoch_cfg.get("expected_events_per_10k_steps", 0.0))
        if rate_10k <= 0.0:
            return
        p = min(1.0, rate_10k / 10000.0)
        min_gap = int(self._stoch_cfg.get("min_gap_steps", 0))

        algo = str(self._stoch_cfg.get("algorithm", "per_region_poisson"))

        for rid, reg in self._regions.items():
            last_start = int(self._last_start_by_region.get(rid, -10**18))
            if (t - last_start) < min_gap:
                continue

            # Deterministic per-step sampling:
            # We derive a step-specific random value by hashing (seed, rid, t).
            # This avoids maintaining RNG state across checkpoints.
            u = float(stable_hash_uint64(f"{self._stoch_seed}|{rid}|{t}") % (2**53)) / float(2**53)

            fire = (u < p)
            if not fire:
                continue

            # Start event
            ev = self._create_event(
                region_id=rid,
                start_step=t,
                ramp_steps=reg.ramp_steps,
                duration_steps=reg.duration_steps,
                recovery_steps=reg.recovery_steps,
                target_regrowth=reg.target_regrowth,
                target_spawn=reg.target_spawn,
                mode="stochastic",
                write_log=True,
            )
            self._last_start_by_region[rid] = int(ev.start_step)

        # Keep events sorted for stable evaluation
        self._events.sort(key=lambda e: (e.start_step, e.event_id))

    # -----------------------------
    # Logging helpers
    # -----------------------------

    def _log_step(
        self,
        t: int,
        region_mult_regrowth: Dict[str, float],
        region_mult_spawn: Dict[str, float],
    ) -> None:
        """Write per-step summaries and periodic stats."""
        if not self.run_dir or not self._ts_writer:
            return

        # We log a single combined drought multiplier summary, using regrowth channel if present else spawn.
        # This is diagnostic, not a semantic truth.
        # Prefer regrowth map for global stats because it reflects scarcity.
        m = self._regrowth_map if np.any(self._regrowth_map != 1.0) else self._spawn_map
        mean_m = float(np.mean(m))
        min_m = float(np.min(m))
        max_m = float(np.max(m))

        if (t % self._timeseries_every) == 0:
            row = [t, mean_m, min_m, max_m]
            for rid in self._regions.keys():
                # Prefer regrowth multiplier if region affects regrowth else spawn
                reg = self._regions[rid]
                if reg.affects_regrowth:
                    row.append(float(region_mult_regrowth.get(rid, 1.0)))
                elif reg.affects_spawn:
                    row.append(float(region_mult_spawn.get(rid, 1.0)))
                else:
                    row.append(1.0)
            try:
                self._ts_writer.writerow(row)
                self._ts_fp.flush()
            except Exception:
                pass

        if (t % self._periodic_every) == 0 and self._ps_writer:
            try:
                self._ps_writer.writerow([t, mean_m, min_m, max_m])
                self._ps_fp.flush()
            except Exception:
                pass

    # -----------------------------
    # Cleanup
    # -----------------------------

    def close(self) -> None:
        """Close open file handles."""
        self._close_writers()
