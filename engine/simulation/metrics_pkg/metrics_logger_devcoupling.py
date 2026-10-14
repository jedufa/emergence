from __future__ import annotations

"""
simulation/metrics_pkg/metrics_logger_devcoupling.py

Developmental coupling analysis output (devcoupling.csv).

This is intentionally separated from metrics.csv logging because:
- It has its own schema and lifecycle (init once, append rows).
- It's analysis-only and optional (gated by sim_cfg.devcoupling_enable).
- It tends to grow with experiments, and should not bloat MetricsLogger core.

Step 3D: Reproductive timing instrumentation
--------------------------------------------
Adds devcoupling columns:
- first_repro_step_low_mean
- first_repro_step_high_mean
- delta_first_repro_step_high_minus_low

Definitions:
- first_repro_step: the agent's age in steps (stats.survival_steps) at the first
  time it successfully produces offspring (set elsewhere; logger only reads it).
- "low" and "high" cohorts: quantile bins based on episode-0 prediction error.

Determinism:
- No RNG calls.
- Deterministic Spearman implementation from utils.
"""

import os
import csv
from typing import Any, Dict, List, Optional

import numpy as np

from simulation.metrics_pkg.stats import mean
from simulation.metrics_pkg.metrics_logger_utils import (
    _unwrap_agent,
    _as_float,
    _as_int,
    _spearman,
)


class DevCouplingWriter:
    """Owns devcoupling.csv schema/init + per-generation row writing."""

    def __init__(self, run_dir: str):
        self.devc_path = os.path.join(run_dir, "devcoupling.csv")
        self._devc_inited = False

    def _init_devcoupling_csv(self) -> None:
        """Initialize devcoupling.csv with header (once)."""
        if self._devc_inited:
            return
        header = [
            "generation",
            "n_alive",
            "n_low",
            "n_high",
            "ep0_pred_low_mean",
            "ep0_pred_high_mean",
            "later_pred_low_mean",
            "later_pred_high_mean",
            "delta_later_pred_high_minus_low",
            "injury_low_mean",
            "injury_high_mean",
            "offspring_low_mean",
            "offspring_high_mean",
            "offspring_rate_low_mean",
            "offspring_rate_high_mean",
            "repro_success_low_frac",
            "repro_success_high_frac",
            "first_repro_step_low_mean",
            "first_repro_step_high_mean",
            "delta_first_repro_step_high_minus_low",
            "parent_later_pred_mean",
            "nonparent_later_pred_mean",
            "spearman_ep0_vs_later_pred",
            "spearman_ep0_vs_injury",
            "spearman_ep0_vs_offspring_rate",
        ]
        with open(self.devc_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(header)
        self._devc_inited = True

    def log_devcoupling(
        self,
        generation: int,
        agents: List,
        selected_parent_agents: Optional[List] = None,
        sim_cfg: Any = None,
    ) -> None:
        """Write one devcoupling.csv row for this generation (analysis only)."""
        if not agents:
            return

        min_n = int(getattr(sim_cfg, "devcoupling_min_agents", 32)) if sim_cfg is not None else 32
        if len(agents) < min_n:
            return

        low_q = float(getattr(sim_cfg, "devcoupling_low_q", 0.20)) if sim_cfg is not None else 0.20
        high_q = float(getattr(sim_cfg, "devcoupling_high_q", 0.80)) if sim_cfg is not None else 0.80
        low_q = float(np.clip(low_q, 0.0, 0.49))
        high_q = float(np.clip(high_q, 0.51, 1.0))

        ep0_list: List[float] = []
        later_list: List[float] = []
        inj_list: List[float] = []
        off_rate_list: List[float] = []

        recs: List[Dict[str, Any]] = []

        for raw in agents:
            a = _unwrap_agent(raw)
            if a is None:
                continue
            st = getattr(a, "stats", None)
            if st is None:
                continue

            ep0_sum = _as_float(getattr(st, "episode0_pred_err_sum", 0.0))
            ep0_n = _as_float(getattr(st, "episode0_pred_err_count", 0.0))
            ep0_mean = ep0_sum / max(1.0, ep0_n)

            tot_sum = _as_float(getattr(st, "pred_err_sum", 0.0))
            tot_n = _as_float(getattr(st, "pred_err_count", 0.0))

            later_sum = tot_sum - ep0_sum
            later_n = tot_n - ep0_n
            later_mean = (later_sum / later_n) if later_n > 0 else 0.0

            injury = _as_float(getattr(a, "injury", 0.0))
            offspring = _as_float(getattr(st, "offspring_count", 0.0))
            steps = float(max(1, _as_int(getattr(st, "survival_steps", 0), 0)))
            off_rate = offspring / steps

            # Step 3D: first reproduction step (logger reads; set elsewhere)
            frs = getattr(st, "first_repro_step", None)

            recs.append(
                {
                    "id": int(getattr(a, "agent_id", -1)),
                    "ep0": float(ep0_mean),
                    "later": float(later_mean),
                    "inj": float(injury),
                    "offspring": float(offspring),
                    "off_rate": float(off_rate),
                    "steps": float(steps),
                    "first_repro_step": None if frs is None else int(_as_int(frs, 0)),
                }
            )

            ep0_list.append(float(ep0_mean))
            later_list.append(float(later_mean))
            inj_list.append(float(injury))
            off_rate_list.append(float(off_rate))

        if len(recs) < min_n:
            return

        ep0_vals = np.asarray([r["ep0"] for r in recs], dtype=np.float64)
        lo_thr = float(np.quantile(ep0_vals, low_q))
        hi_thr = float(np.quantile(ep0_vals, high_q))

        low = [r for r in recs if r["ep0"] <= lo_thr]
        high = [r for r in recs if r["ep0"] >= hi_thr]

        if len(low) < max(5, min_n // 8) or len(high) < max(5, min_n // 8):
            return

        def _m(xs: List[float]) -> float:
            return float(mean(xs)) if xs else 0.0

        def _frac_true(xs: List[bool]) -> float:
            if not xs:
                return 0.0
            return float(sum(1 for b in xs if b)) / float(len(xs))

        ep0_low_m = _m([r["ep0"] for r in low])
        ep0_high_m = _m([r["ep0"] for r in high])

        later_low_m = _m([r["later"] for r in low])
        later_high_m = _m([r["later"] for r in high])

        inj_low_m = _m([r["inj"] for r in low])
        inj_high_m = _m([r["inj"] for r in high])

        off_low_m = _m([r["offspring"] for r in low])
        off_high_m = _m([r["offspring"] for r in high])

        offrate_low_m = _m([r["off_rate"] for r in low])
        offrate_high_m = _m([r["off_rate"] for r in high])

        repro_low_frac = _frac_true([r["offspring"] > 0.0 for r in low])
        repro_high_frac = _frac_true([r["offspring"] > 0.0 for r in high])

        # Step 3D: mean first reproduction step among reproducers only.
        frs_low = [float(r["first_repro_step"]) for r in low if (r["first_repro_step"] is not None)]
        frs_high = [float(r["first_repro_step"]) for r in high if (r["first_repro_step"] is not None)]
        frs_low_m = _m(frs_low)
        frs_high_m = _m(frs_high)

        parent_ids = set(int(getattr(_unwrap_agent(p), "agent_id", -1)) for p in (selected_parent_agents or []))
        parent_later = [r["later"] for r in recs if r["id"] in parent_ids]
        nonparent_later = [r["later"] for r in recs if r["id"] not in parent_ids]

        parent_later_m = _m(parent_later)
        nonparent_later_m = _m(nonparent_later)

        s_ep0_later = _spearman(ep0_list, later_list)
        s_ep0_inj = _spearman(ep0_list, inj_list)
        s_ep0_offrate = _spearman(ep0_list, off_rate_list)

        self._init_devcoupling_csv()
        with open(self.devc_path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    int(generation),
                    int(len(recs)),
                    int(len(low)),
                    int(len(high)),
                    float(ep0_low_m),
                    float(ep0_high_m),
                    float(later_low_m),
                    float(later_high_m),
                    float(later_high_m - later_low_m),
                    float(inj_low_m),
                    float(inj_high_m),
                    float(off_low_m),
                    float(off_high_m),
                    float(offrate_low_m),
                    float(offrate_high_m),
                    float(repro_low_frac),
                    float(repro_high_frac),
                    float(frs_low_m),
                    float(frs_high_m),
                    float(frs_high_m - frs_low_m),
                    float(parent_later_m),
                    float(nonparent_later_m),
                    float(s_ep0_later),
                    float(s_ep0_inj),
                    float(s_ep0_offrate),
                ]
            )
