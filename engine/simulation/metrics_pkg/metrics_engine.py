"""simulation/metrics_pkg/metrics_engine.py

Phase 5a: Experimental Capability Layer (Metrics Engine wrapper).

This module wires:
- EPI (Experimental Pressure Index)
- AV (Adaptation Velocity)

into a single post-SafeBoundary step that can be invoked by the runner loop.

Hard rules:
- Must not mutate main sim state
- Must not change selection/reproduction timing
- Canonical JSON outputs
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

from simulation.control_plane_pkg.util_json import canonical_json_bytes
from simulation.metrics_pkg.adaptation_velocity import AdaptationVelocityEngine
from simulation.metrics_pkg.epi_engine import EPIEngine
from simulation.metrics_pkg.stress_scenarios import StressScenarioV1, default_stress_scenarios_v1
from simulation.metrics_pkg.epi_av_timeseries import EPITimeSeriesWriter


def _atomic_write_json(path: str, payload: Dict) -> None:
    data = canonical_json_bytes(payload)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


@dataclass
class Phase5MetricsEngine:
    """Orchestrates Phase 5 metrics (EPI + AV) per SafeBoundary."""

    run_dir: str

    def __post_init__(self) -> None:
        self._epi = EPIEngine()
        self._av = None
        self._ts = None

        # Last computed records (best-effort observability helper).
        # Used by in-process consumers (e.g. PressureAutopilot) so they don't
        # have to re-read JSON from disk.
        self.last_epi_record = None
        self.last_av_record = None

    def compute_post_boundary(self, sim: Any, generation_index: int) -> None:
        sim_cfg = sim.sim_cfg
        if not bool(getattr(sim_cfg, "epi_enable", False)):
            return

        scenarios: Sequence[StressScenarioV1] = default_stress_scenarios_v1()
        filter_ids: List[str] = list(getattr(sim_cfg, "epi_scenario_ids", []))
        if filter_ids:
            wanted = set(filter_ids)
            scenarios = [s for s in scenarios if s.scenario_id in wanted]

        epi_record = self._epi.compute(sim=sim, generation_index=generation_index, scenarios=scenarios)
        try:
            self.last_epi_record = dict(epi_record)
        except Exception:
            self.last_epi_record = None
        epi_path = os.path.join(self.run_dir, "epi_v2.json")
        _atomic_write_json(epi_path, epi_record)

        av_value_opt = None
        av_record = None
        if bool(getattr(sim_cfg, "av_enable", True)):
            if self._av is None:
                self._av = AdaptationVelocityEngine(
                    smoothing_k=int(getattr(sim_cfg, "av_smoothing_k", 5)),
                    short_window=int(getattr(sim_cfg, "av_short_window", 20)),
                    long_window=int(getattr(sim_cfg, "av_long_window", 100)),
                )

            av_record = self._av.update(
                generation_index=int(generation_index),
                epi_value=float(epi_record["epi_value"]),
            )
            try:
                self.last_av_record = dict(av_record)
            except Exception:
                self.last_av_record = None
            av_value_opt = av_record.get("av_value", None)
            av_path = os.path.join(self.run_dir, "av_v2.json")
            _atomic_write_json(av_path, av_record)

        # Optional per-generation time series (human-friendly, non-critical artifact).
        if bool(getattr(sim_cfg, "epi_timeseries_enable", False)):
            if self._ts is None:
                self._ts = EPITimeSeriesWriter(run_dir=self.run_dir)
            self._ts.append_row(
                generation=int(generation_index),
                epi_value=float(epi_record["epi_value"]),
                av_value=(float(av_value_opt) if av_value_opt is not None else None),
            )

        # -----------------------------
        # Phase 5d: viz/ EPI+AV summary (observability only)
        # -----------------------------
        try:
            from simulation.viz_pkg.viz_writer import get_viz_writer
            get_viz_writer(self.run_dir).append_epi_av_summary(
                generation=int(generation_index),
                epi=float(epi_record["epi_value"]),
                av=(float(av_value_opt) if av_value_opt is not None else None),
            )
        except Exception:
            pass

        # Optional per-generation log line (human-friendly, console only).
        if bool(getattr(sim_cfg, "epi_print_each_gen", False)):
            av_s = "null" if av_value_opt is None else f"{float(av_value_opt):.6f}"
            try:
                print(f"[phase5a] gen={int(generation_index)} epi={float(epi_record['epi_value']):.6f} av={av_s}", flush=True)
            except Exception:
                pass
