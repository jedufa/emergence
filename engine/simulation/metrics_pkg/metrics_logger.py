from __future__ import annotations

"""
simulation/metrics_pkg/metrics_logger.py

Aggregates per-generation statistics and writes them to metrics.csv.

This file used to contain *everything* (helpers, genome parsing, devcoupling,
drive/coherence aggregation, CSV I/O). That scales about as well as a one-file
operating system.

As of this refactor, MetricsLogger is a thin orchestrator:
- CSV I/O lives in metrics_logger_io.py
- generic helpers live in metrics_logger_utils.py
- memory/GRU helpers live in metrics_logger_memory.py
- genome/gene stats + collapse tracking live in metrics_logger_genome.py
- devcoupling output lives in metrics_logger_devcoupling.py

Public API compatibility:
- MetricsLogger class still exists with the same constructor and methods:
  - log_generation(...)
  - log_devcoupling(...)
- Existing callers should not need changes.

Determinism:
- No new RNG calls.
- No reordering of agent iteration that would change results.
"""

import math
import os
import time
from dataclasses import asdict
from collections import Counter
from typing import List, Optional, Any, Dict

from evolution.selection import fitness as fitness_fn
from simulation.metrics_pkg.rows import MetricsRow
from simulation.metrics_pkg.stats import mean, std, percentile

from simulation.metrics_pkg.metrics_logger_io import init_metrics_csv, append_metrics_row
from simulation.metrics_pkg.metrics_logger_utils import (
    _unwrap_agent,
    _as_float,
    _as_int,
    _biome_fracs,
)
from simulation.metrics_pkg.metrics_logger_memory import (
    _gru_frac,
    _memory_dim_list,
    _memory_bucket_fracs,
)
from simulation.metrics_pkg.metrics_logger_genome import (
    safe_aggression_bias,
    safe_exploration_bias,
    safe_repro_threshold,
    safe_mutation_rate,
    safe_hidden_size,
    genome_stats_from_agents,
    update_collapse_state,
)
from simulation.metrics_pkg.metrics_logger_devcoupling import DevCouplingWriter
from simulation.metrics_pkg.lineage import LineageMetricsWriter
from simulation.metrics_pkg.lineage_forensics import (
    LineageCollapseForensicsState,
    compute_lineage_collapse_forensics,
    empty_lineage_forensics_metrics,
)
from simulation.metrics_pkg.cognitive_status import compute_cognitive_status_metrics


def _native_public_metrics_overlay_values(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    if str(payload.get("source", "") or "") != "native_owner_checkpoint_projection":
        return {}
    values = payload.get("values")
    return dict(values) if isinstance(values, dict) else {}


def _native_policy_cache_metrics_from_public_payloads(
    native_public_output_counts: Any,
    native_public_metrics_overlay: Any,
) -> dict[str, int]:
    counts: dict[str, Any] = {}
    if isinstance(native_public_output_counts, dict):
        if str(native_public_output_counts.get("source", "") or "") == "native_compact_result":
            counts = dict(native_public_output_counts)

    overlay_values = _native_public_metrics_overlay_values(native_public_metrics_overlay)

    if not counts and not overlay_values:
        return {}
    from native_runtime.public_metrics_overlay import native_policy_cache_metrics  # noqa: WPS433

    return native_policy_cache_metrics(counts, overlay_values)


def _policy_cache_metrics_from_public_counts(payload: Any) -> tuple[dict[str, int], str]:
    if payload is None:
        return {}, ""
    if not isinstance(payload, dict):
        raise TypeError("policy_cache_public_counts must be a mapping")
    source = str(payload.get("source", "") or "")
    if source not in {"native_compact_result", "reference_policy_cache_service"}:
        raise ValueError(f"unsupported policy-cache metric source: {source!r}")
    from native_runtime.public_metrics_overlay import native_policy_cache_metrics  # noqa: WPS433

    return native_policy_cache_metrics(payload), source


def _lineage_share_snapshot(agents: List[Any]) -> Dict[str, float]:
    counts: Counter[str] = Counter()
    for raw in agents or []:
        a = _unwrap_agent(raw)
        if a is None:
            continue
        st = getattr(a, "stats", None)
        if st is None:
            continue
        lineage_id = str(getattr(st, "lineage_id", "") or "").strip()
        if not lineage_id:
            continue
        counts[lineage_id] += 1
    total = sum(counts.values())
    if total <= 0 or not counts:
        return {"effective": 0.0, "entropy": 0.0, "top1": 0.0, "min": 0.0, "max": 0.0, "count": 0.0}
    shares = [float(v) / float(total) for v in counts.values()]
    entropy = -sum(p * math.log(p) for p in shares if p > 0.0)
    return {
        "effective": float(math.exp(entropy)) if entropy > 0.0 else 1.0,
        "entropy": float(entropy),
        "top1": float(max(shares)),
        "min": float(min(shares)),
        "max": float(max(shares)),
        "count": float(len(shares)),
    }


def _founder_balance_metric_defaults(sim_cfg: Any, agents: List[Any]) -> Dict[str, float | int]:
    enabled = bool(getattr(sim_cfg, "lineage_founder_balance_enable", False)) if sim_cfg is not None else False
    snapshot = _lineage_share_snapshot(agents)
    if not enabled:
        return {
            "founder_balance_enable": 0,
            "founder_group_count": 0,
            "founder_min_agents_per_group": 0,
            "founder_initial_effective_lineages": 0.0,
            "founder_initial_lineage_entropy": 0.0,
            "founder_initial_top1_share": 0.0,
            "founder_initial_min_lineage_share": 0.0,
            "founder_initial_max_lineage_share": 0.0,
            "founder_initial_lineage_count": 0.0,
            "founder_balance_violation_count": 0,
        }
    return {
        "founder_balance_enable": 1,
        "founder_group_count": int(getattr(sim_cfg, "lineage_founder_group_count", 0) or 0),
        "founder_min_agents_per_group": int(getattr(sim_cfg, "lineage_founder_min_agents_per_group", 0) or 0),
        "founder_initial_effective_lineages": float(getattr(sim_cfg, "_founder_initial_effective_lineages", snapshot["effective"]) or 0.0),
        "founder_initial_lineage_entropy": float(getattr(sim_cfg, "_founder_initial_lineage_entropy", snapshot["entropy"]) or 0.0),
        "founder_initial_top1_share": float(getattr(sim_cfg, "_founder_initial_top1_share", snapshot["top1"]) or 0.0),
        "founder_initial_min_lineage_share": float(getattr(sim_cfg, "_founder_initial_min_lineage_share", snapshot["min"]) or 0.0),
        "founder_initial_max_lineage_share": float(getattr(sim_cfg, "_founder_initial_max_lineage_share", snapshot["max"]) or 0.0),
        "founder_initial_lineage_count": float(getattr(sim_cfg, "_founder_initial_lineage_count", snapshot["count"]) or 0.0),
        "founder_balance_violation_count": int(getattr(sim_cfg, "_founder_balance_violation_count", 0) or 0),
    }



def _read_last_gen_from_metrics_csv(path: str) -> int:
    """Return last generation in an existing metrics.csv, or -1.

    Streaming-ish: reads from the end by scanning lines in reverse until it finds
    a parseable integer in the first column (generation).
    """
    try:
        if not path or (not os.path.exists(path)):
            return -1
        # Read a tail chunk to avoid loading huge files.
        with open(path, "rb") as f:
            try:
                f.seek(0, os.SEEK_END)
                size = f.tell()
                # Read last 256KB or full file if smaller.
                tail = min(size, 256 * 1024)
                f.seek(size - tail)
            except Exception:
                f.seek(0)
            data = f.read().decode("utf-8", errors="ignore")
        lines = [ln.strip() for ln in data.splitlines() if ln.strip()]
        # Walk backwards skipping header and malformed rows.
        for ln in reversed(lines):
            if ln.lower().startswith("generation,"):
                continue
            parts = ln.split(",")
            if not parts:
                continue
            try:
                return int(parts[0])
            except Exception:
                continue
        return -1
    except Exception:
        return -1

class MetricsLogger:
    """Aggregates metrics per generation and writes CSV outputs."""

    def __init__(self, run_dir: str):
        self.run_dir = run_dir
        os.makedirs(run_dir, exist_ok=True)

        self.path = os.path.join(run_dir, "metrics.csv")
        init_metrics_csv(self.path)

        # If this run directory already has carried-over history (branch/resume),
        # do not duplicate rows at the fork boundary.
        self._last_written_gen = _read_last_gen_from_metrics_csv(self.path)


        # Diversity collapse streak is stateful across generations.
        self._collapse_streak = 0

        # Last written metrics row (observability helper for controllers/autopilots).
        # Not used to generate metrics.csv, so it is not part of export_state().
        self.last_row: Optional[Dict[str, Any]] = None
        self.last_policy_cache_metrics: Dict[str, int] = {}
        self.last_policy_cache_metric_source: str = ""
        self.last_native_policy_cache_metrics: Dict[str, int] = {}
        self._last_perf_csv_write_s = 0.0
        self._last_perf_lineage_write_s = 0.0
        self._early_lineage_effective_gen_1: Optional[float] = None
        self._early_lineage_effective_gen_2: Optional[float] = None
        self._early_lineage_top1_share_gen_1: Optional[float] = None
        self._early_lineage_top1_share_gen_2: Optional[float] = None

        # Devcoupling writer (optional usage is gated by sim_cfg.devcoupling_enable elsewhere).
        self.devc = DevCouplingWriter(run_dir=run_dir)
        self.lineage = LineageMetricsWriter(run_dir=run_dir)
        self._lineage_forensics_state = LineageCollapseForensicsState()

    # -----------------------------
    # Checkpoint / resume state
    # -----------------------------
    def export_state(self) -> Dict[str, Any]:
        """Export determinism-critical internal metrics state for checkpoint/resume.

        Only includes state that directly affects metrics.csv outputs.
        """
        return {
            "schema_version": "metrics_state_v1",
            "collapse_streak": int(getattr(self, "_collapse_streak", 0) or 0),
            "theme5_lineage_state": self.lineage.export_state(),
            "lineage_collapse_forensics_state": self._lineage_forensics_state.to_dict(),
            "early_lineage_effective_gen_1": self._early_lineage_effective_gen_1,
            "early_lineage_effective_gen_2": self._early_lineage_effective_gen_2,
            "early_lineage_top1_share_gen_1": self._early_lineage_top1_share_gen_1,
            "early_lineage_top1_share_gen_2": self._early_lineage_top1_share_gen_2,
        }

    def restore_state(self, state: Dict[str, Any]) -> None:
        """Restore determinism-critical internal metrics state from checkpoint payload."""
        if not isinstance(state, dict):
            return
        # Best-effort: accept unknown schema without crashing resume.
        try:
            self._collapse_streak = int(state.get("collapse_streak", 0) or 0)
        except Exception:
            self._collapse_streak = 0
        try:
            lineage_state = state.get("theme5_lineage_state", {})
            if isinstance(lineage_state, dict):
                self.lineage.restore_state(lineage_state)
        except Exception:
            pass
        try:
            self._lineage_forensics_state = LineageCollapseForensicsState.from_dict(
                state.get("lineage_collapse_forensics_state", {})
            )
        except Exception:
            self._lineage_forensics_state = LineageCollapseForensicsState()
        for name in (
            "_early_lineage_effective_gen_1",
            "_early_lineage_effective_gen_2",
            "_early_lineage_top1_share_gen_1",
            "_early_lineage_top1_share_gen_2",
        ):
            key = name[1:]
            value = state.get(key)
            try:
                setattr(self, name, None if value is None else float(value))
            except Exception:
                setattr(self, name, None)


    # -----------------------------
    # Safe access helpers
    # -----------------------------
    def _safe_fitness(self, agent) -> float:
        """Compute fitness(agent) with a safe fallback."""
        try:
            return float(fitness_fn(agent))
        except Exception:
            return 0.0

    # -----------------------------
    # Developmental coupling CSV
    # -----------------------------
    def log_devcoupling(
        self,
        generation: int,
        agents: List,
        selected_parent_agents: Optional[List] = None,
        sim_cfg: Any = None,
    ) -> None:
        """Write one devcoupling.csv row for this generation (analysis only)."""
        self.devc.log_devcoupling(
            generation=generation,
            agents=agents,
            selected_parent_agents=selected_parent_agents,
            sim_cfg=sim_cfg,
        )

    # -----------------------------
    # Per-generation metrics CSV
    # -----------------------------
    def log_generation(
        self,
        generation: int,
        agents: List,
        deaths_this_gen: int,
        successful_births: int = 0,
        failed_births: int = 0,
        injury_blocked_births: int = 0,
        cohort_size: int = 0,
        cohort_survivors: int = 0,
        cohort_survival_rate: float = 0.0,
        shock_active: int = 0,
        idle_charge_total: float = 0.0,
        idle_charge_deaths: int = 0,
        peak_injury: float = 0.0,
        birth_attempts: int = 0,
        birth_pre_cooldown: int = 0,
        birth_pre_no_energy: int = 0,
        birth_pre_injury_gate: int = 0,
        birth_pre_no_space: int = 0,
        birth_pre_no_mate: int = 0,
        birth_attempt_conception_fail: int = 0,
        birth_gestation_fail: int = 0,
        birth_make_child_fail: int = 0,
        birth_fail_no_space: int = 0,
        birth_fail_no_mate: int = 0,
        birth_fail_buffer: int = 0,
        birth_fail_injury_gate: int = 0,
        birth_fail_scar_gate: int = 0,
        birth_fail_conception: int = 0,
        birth_fail_gestation: int = 0,
        # === Tier 3 change: metrics fields ===
        tier3_overcrowd_mean: float = 0.0,
        tier3_overcrowd_max: float = 0.0,
        tier3_crossings_per_step: float = 0.0,
        tier3_cross_events_per_step: float = 0.0,
        tier3_extra_move_cost_mean: float = 0.0,
        tier3_extra_move_cost_max: float = 0.0,
        # Tier 4: juvenile spatial vulnerability (optional, no-op by default)
        tier4_juvenile_exposure_steps: int = 0,
        tier4_adult_exposure_steps: int = 0,
        tier4_juvenile_deaths: int = 0,
        tier4_adult_deaths: int = 0,
        tier4_juvenile_death_rate_per_1k_steps: float = 0.0,
        tier4_adult_death_rate_per_1k_steps: float = 0.0,
        tier4_juvenile_fraction_mean: float = 0.0,
        tier4_fraction_reproduced_alive_end: float = 0.0,
        tier4_first_repro_step_mean_reproducers_alive_end: float = 0.0,
        tier4_first_repro_step_p50_reproducers_alive_end: float = 0.0,
        tier4_first_repro_step_p90_reproducers_alive_end: float = 0.0,
        tier4_bins_steps_json: str = "[]",
        tier4_alive_age_hist_json: str = "[]",
        tier4_deaths_age_hist_json: str = "[]",
        patch_food_amount_mean: float = 0.0,
        patch_food_amount_min: float = 0.0,
        patch_food_amount_max: float = 0.0,
        patch_food_amount_p10: float = 0.0,
        patch_food_amount_p90: float = 0.0,
        patch_food_depleted_frac: float = 0.0,
        patch_food_full_frac: float = 0.0,
        patch_food_regrowth_total: float = 0.0,
        patch_food_harvest_total: float = 0.0,
        patch_fertility_mean: float = 0.0,
        mosaic0a_burst_cell_frac: float = 0.0,
        mosaic0a_renewal_cell_frac: float = 0.0,
        mosaic0a_burst_food_amount_mean: float = 0.0,
        mosaic0a_renewal_food_amount_mean: float = 0.0,
        mosaic0a_burst_fertility_mean: float = 0.0,
        mosaic0a_renewal_fertility_mean: float = 0.0,
        respec0b_resource_a_cell_frac: float = 0.0,
        respec0b_resource_b_cell_frac: float = 0.0,
        respec0b_resource_gather_attempts: float = 0.0,
        respec0b_resource_gather_successes: float = 0.0,
        respec0b_resource_gather_failed_handling: float = 0.0,
        respec0b_resource_gather_resource_a_successes: float = 0.0,
        respec0b_resource_gather_resource_b_successes: float = 0.0,
        respec0b_resource_gather_energy_resource_a: float = 0.0,
        respec0b_resource_gather_energy_resource_b: float = 0.0,
        respec0b_resource_gather_handling_ticks: float = 0.0,
        respec0b_resource_gather_interruptions: float = 0.0,
        patch_cover_mean: float = 0.0,
        patch_cover_occupied_mean: float = 0.0,
        patch_shelter_mean: float = 0.0,
        patch_shelter_occupied_mean: float = 0.0,
        patch_visibility_mean: float = 0.0,
        patch_visibility_occupied_mean: float = 0.0,
        patch_movement_cost_mean: float = 0.0,
        patch_movement_cost_occupied_mean: float = 0.0,
        patch_movement_energy_extra_total: float = 0.0,
        patch_movement_energy_extra_mean: float = 0.0,
        patch_scent_mean: float = 0.0,
        patch_scent_max: float = 0.0,
        patch_scent_p90: float = 0.0,
        patch_scent_occupied_mean: float = 0.0,
        patch_scent_emitted_total: float = 0.0,
        patch_scent_decayed_total: float = 0.0,
        patch_waste_mean: float = 0.0,
        patch_waste_max: float = 0.0,
        patch_waste_p90: float = 0.0,
        patch_waste_occupied_mean: float = 0.0,
        patch_waste_emitted_total: float = 0.0,
        patch_waste_decayed_total: float = 0.0,
        patch_density_mean: float = 0.0,
        patch_density_max: float = 0.0,
        patch_density_p90: float = 0.0,
        patch_density_occupied_mean: float = 0.0,
        patch_density_hotspot_frac: float = 0.0,
        patch_observation_enable: int = 0,
        patch_observation_schema_version: int = 0,
        patch_observation_channel_count: int = 0,
        patch_observation_size: int = 0,
        patch_world_descriptor_schema_version: int = 0,
        patch_world_diagnostics_enable: int = 0,
        patch_world_lineage_descriptors_enable: int = 0,
        occupied_patch_food_amount_mean: float = 0.0,
        occupied_patch_fertility_mean: float = 0.0,
        occupied_patch_cover_mean: float = 0.0,
        occupied_patch_shelter_mean: float = 0.0,
        occupied_patch_visibility_mean: float = 0.0,
        occupied_patch_movement_cost_mean: float = 0.0,
        occupied_patch_scent_mean: float = 0.0,
        occupied_patch_waste_mean: float = 0.0,
        occupied_patch_density_mean: float = 0.0,
        birth_patch_count: float = 0.0,
        birth_patch_food_amount_mean: float = 0.0,
        birth_patch_fertility_mean: float = 0.0,
        birth_patch_cover_mean: float = 0.0,
        birth_patch_shelter_mean: float = 0.0,
        birth_patch_visibility_mean: float = 0.0,
        birth_patch_movement_cost_mean: float = 0.0,
        birth_patch_scent_mean: float = 0.0,
        birth_patch_waste_mean: float = 0.0,
        birth_patch_density_mean: float = 0.0,
        death_patch_count: float = 0.0,
        death_patch_food_amount_mean: float = 0.0,
        death_patch_fertility_mean: float = 0.0,
        death_patch_cover_mean: float = 0.0,
        death_patch_shelter_mean: float = 0.0,
        death_patch_visibility_mean: float = 0.0,
        death_patch_movement_cost_mean: float = 0.0,
        death_patch_scent_mean: float = 0.0,
        death_patch_waste_mean: float = 0.0,
        death_patch_density_mean: float = 0.0,
        patch_food_occupancy_bias: float = 0.0,
        patch_fertility_occupancy_bias: float = 0.0,
        patch_cover_occupancy_bias: float = 0.0,
        patch_shelter_occupancy_bias: float = 0.0,
        patch_visibility_occupancy_bias: float = 0.0,
        patch_movement_cost_occupancy_bias: float = 0.0,
        patch_scent_occupancy_bias: float = 0.0,
        patch_waste_occupancy_bias: float = 0.0,
        patch_density_occupancy_bias: float = 0.0,
        lineage_habitat_descriptor_count: float = 0.0,
        lineage_patch_food_amount_mean_std: float = 0.0,
        lineage_patch_cover_mean_std: float = 0.0,
        lineage_patch_shelter_mean_std: float = 0.0,
        lineage_patch_scent_mean_std: float = 0.0,
        lineage_patch_waste_mean_std: float = 0.0,
        lineage_patch_density_mean_std: float = 0.0,
        lineage_patch_recovery_activity_mean_std: float = 0.0,
        lineage_diversity_schema_version: int = 0,
        lineage_diversity_reporting_enable: int = 0,
        lineage_strategy_descriptors_enable: int = 0,
        lineage_branch_candidate_reporting_enable: int = 0,
        lineage_active_count: float = 0.0,
        lineage_extinct_count: float = 0.0,
        lineage_alive_count_total: float = 0.0,
        lineage_top1_alive_share: float = 0.0,
        lineage_top3_alive_share: float = 0.0,
        lineage_top5_alive_share: float = 0.0,
        lineage_top1_birth_share: float = 0.0,
        lineage_top3_birth_share: float = 0.0,
        lineage_top5_birth_share: float = 0.0,
        lineage_top1_death_share: float = 0.0,
        lineage_top3_death_share: float = 0.0,
        lineage_alive_share_entropy: float = 0.0,
        lineage_birth_share_entropy: float = 0.0,
        lineage_effective_alive_lineages: float = 0.0,
        lineage_effective_birth_lineages: float = 0.0,
        lineage_top1_dominance_warning: int = 0,
        lineage_top3_dominance_warning: int = 0,
        lineage_entropy_collapse_warning: int = 0,
        lineage_low_active_count_warning: int = 0,
        lineage_collapse_warning_any: int = 0,
        lineage_strategy_descriptor_count: float = 0.0,
        lineage_birth_patch_waste_mean_std: float = 0.0,
        lineage_birth_patch_shelter_mean_std: float = 0.0,
        lineage_birth_patch_recovery_activity_mean_std: float = 0.0,
        lineage_predator_damage_mean_std: float = 0.0,
        lineage_predator_damage_prevented_mean_std: float = 0.0,
        lineage_reproduction_rate_std: float = 0.0,
        lineage_branch_candidate_count: float = 0.0,
        lineage_habitat_descriptors_json: str = "[]",
        lineage_strategy_descriptors_json: str = "[]",
        lineage_branch_candidates_json: str = "[]",
        patch_observation_min: float = 0.0,
        patch_observation_max: float = 0.0,
        patch_observation_nan_count: float = 0.0,
        patch_world_waste_consequence_enable: int = 0,
        patch_regrowth_base_total: float = 0.0,
        patch_regrowth_effective_total: float = 0.0,
        patch_regrowth_suppressed_total: float = 0.0,
        patch_regrowth_multiplier_mean: float = 1.0,
        patch_regrowth_multiplier_min: float = 1.0,
        patch_regrowth_multiplier_p10: float = 1.0,
        patch_regrowth_multiplier_p90: float = 1.0,
        patch_waste_regrowth_penalty_mean: float = 0.0,
        patch_waste_regrowth_penalty_max: float = 0.0,
        patch_density_regrowth_penalty_mean: float = 0.0,
        patch_density_regrowth_penalty_max: float = 0.0,
        patch_high_waste_regrowth_suppression_frac: float = 0.0,
        patch_high_density_regrowth_suppression_frac: float = 0.0,
        jungle_ecological_recovery_enable: int = 0,
        patch_recovery_activity_mean: float = 0.0,
        patch_recovery_activity_min: float = 0.0,
        patch_recovery_activity_max: float = 0.0,
        patch_recovery_activity_p10: float = 0.0,
        patch_recovery_activity_p90: float = 0.0,
        patch_recovery_activity_occupied_mean: float = 0.0,
        patch_recovery_positive_total: float = 0.0,
        patch_recovery_negative_total: float = 0.0,
        patch_recovery_decay_total: float = 0.0,
        jungle_recovery_regrowth_boost_total: float = 0.0,
        jungle_recovery_regrowth_boost_mean: float = 0.0,
        jungle_recovery_regrowth_boost_max: float = 0.0,
        jungle_recovery_regrowth_boost_p90: float = 0.0,
        patch_regrowth_after_waste_total: float = 0.0,
        patch_regrowth_after_recovery_total: float = 0.0,
        patch_recovery_added_regrowth_total: float = 0.0,
        patch_fertility_damage_total: float = 0.0,
        patch_fertility_recovery_total: float = 0.0,
        jungle_predator_pressure_enable: int = 0,
        jungle_predator_risk_mean: float = 0.0,
        jungle_predator_risk_min: float = 0.0,
        jungle_predator_risk_max: float = 0.0,
        jungle_predator_risk_p10: float = 0.0,
        jungle_predator_risk_p90: float = 0.0,
        jungle_predator_occupied_risk_mean: float = 0.0,
        jungle_predator_occupied_risk_p90: float = 0.0,
        jungle_predator_events: float = 0.0,
        jungle_predator_event_rate: float = 0.0,
        jungle_predator_damage_total: float = 0.0,
        jungle_predator_damage_mean: float = 0.0,
        jungle_terrain_protection_enable: int = 0,
        jungle_predator_damage_raw_total: float = 0.0,
        jungle_predator_damage_effective_total: float = 0.0,
        jungle_predator_damage_prevented_total: float = 0.0,
        jungle_predator_damage_multiplier_mean: float = 1.0,
        jungle_predator_damage_multiplier_min: float = 1.0,
        jungle_predator_damage_multiplier_p10: float = 1.0,
        jungle_predator_damage_multiplier_p90: float = 1.0,
        jungle_predator_event_damage_multiplier_mean: float = 1.0,
        jungle_predator_event_damage_prevented_mean: float = 0.0,
        occupied_predator_damage_multiplier_mean: float = 1.0,
        jungle_predator_deaths: float = 0.0,
        jungle_predator_scent_component_mean: float = 0.0,
        jungle_predator_waste_component_mean: float = 0.0,
        jungle_predator_density_component_mean: float = 0.0,
        jungle_predator_visibility_component_mean: float = 0.0,
        jungle_predator_cover_reduction_mean: float = 0.0,
        jungle_predator_shelter_reduction_mean: float = 0.0,
        predator_event_patch_count: float = 0.0,
        predator_event_patch_scent_mean: float = 0.0,
        predator_event_patch_waste_mean: float = 0.0,
        predator_event_patch_density_mean: float = 0.0,
        predator_event_patch_cover_mean: float = 0.0,
        predator_event_patch_shelter_mean: float = 0.0,
        predator_event_patch_visibility_mean: float = 0.0,
        jungle_reproduction_emissions_enable: int = 0,
        jungle_birth_scent_emitted_total: float = 0.0,
        jungle_birth_waste_emitted_total: float = 0.0,
        jungle_birth_emission_count: float = 0.0,
        jungle_birth_emission_patch_scent_mean: float = 0.0,
        jungle_birth_emission_patch_waste_mean: float = 0.0,
        jungle_birth_emission_patch_density_mean: float = 0.0,
        jungle_birth_emission_patch_cover_mean: float = 0.0,
        jungle_birth_emission_patch_shelter_mean: float = 0.0,
        jungle_juvenile_predator_vulnerability_enable: int = 0,
        jungle_juvenile_predator_events: float = 0.0,
        jungle_juvenile_predator_damage_total: float = 0.0,
        jungle_juvenile_predator_deaths: float = 0.0,
        jungle_adult_predator_events: float = 0.0,
        jungle_adult_predator_damage_total: float = 0.0,
        jungle_juvenile_vs_adult_predator_event_ratio: float = 0.0,
        wdl_delayed_hazard_enable: int = 0,
        wdl_delayed_hazard_cue_cell_count: float = 0.0,
        wdl_delayed_hazard_active_cell_count: float = 0.0,
        wdl_delayed_hazard_event_count: float = 0.0,
        wdl_delayed_hazard_agent_cue_exposure_count: float = 0.0,
        wdl_delayed_hazard_agent_active_exposure_count: float = 0.0,
        wdl_delayed_hazard_damage_total: float = 0.0,
        wdl_delayed_hazard_damage_mean: float = 0.0,
        wdl_delayed_hazard_damage_max: float = 0.0,
        wdl_delayed_hazard_agents_damaged: float = 0.0,
        wdl_delayed_hazard_cue_to_active_delay_mean: float = 0.0,
        wdl_delayed_hazard_cue_without_active_count: float = 0.0,
        wdl_delayed_hazard_active_without_prior_cue_count: float = 0.0,
        wdl_cue_seen_agents: float = 0.0,
        wdl_cue_exposed_next_move_away_count: float = 0.0,
        wdl_cue_exposed_next_move_toward_count: float = 0.0,
        wdl_cue_exposed_stayed_count: float = 0.0,
        wdl_pre_hazard_exit_count: float = 0.0,
        wdl_hazard_enter_count: float = 0.0,
        wdl_hazard_avoidance_proxy: float = 0.0,
        wdl_hazard_exposure_after_cue_count: float = 0.0,
        wdl_hazard_damage_by_lineage_json: str = "{}",
        wdl_cue_exposure_by_lineage_json: str = "{}",
        wdl_hazard_exposure_by_lineage_json: str = "{}",
        path1_refuge_cue_cell_count: float = 0.0,
        path1_refuge_active_cell_count: float = 0.0,
        path1_refuge_active_without_prior_cue_count: float = 0.0,
        path1_refuge_cue_without_active_count: float = 0.0,
        path1_refuge_zone_cell_count: float = 0.0,
        path1_refuge_reachable_zone_cell_count: float = 0.0,
        path1_refuge_reachable_before_active_fraction: float = 0.0,
        path1_refuge_agents_in_hazard_at_cue: float = 0.0,
        path1_refuge_agents_reachable_refuge_at_cue: float = 0.0,
        path1_refuge_agents_in_refuge_before_active: float = 0.0,
        path1_refuge_agents_stayed_in_hazard_until_active: float = 0.0,
        path1_refuge_shelter_occupancy_pre_active: float = 0.0,
        path1_refuge_cover_occupancy_pre_active: float = 0.0,
        path1_refuge_move_toward_refuge_pre_active: float = 0.0,
        path1_refuge_move_away_from_refuge_pre_active: float = 0.0,
        path1_refuge_disabled_control_leakage_flag: float = 0.0,
        path1_drh0_cue_exposure_count: float = 0.0,
        path1_drh0_cue_visible_steps: float = 0.0,
        path1_drh0_hidden_window_steps: float = 0.0,
        path1_drh0_hazard_active_steps: float = 0.0,
        path1_drh0_refuge_cell_count: float = 0.0,
        path1_drh0_reachable_refuge_count: float = 0.0,
        path1_drh0_distance_to_nearest_refuge_at_cue: float = 0.0,
        path1_drh0_movement_toward_refuge_after_cue: float = 0.0,
        path1_drh0_refuge_occupancy_before_hazard: float = 0.0,
        path1_drh0_refuge_occupancy_during_hazard: float = 0.0,
        path1_drh0_damage_in_refuge: float = 0.0,
        path1_drh0_damage_outside_refuge: float = 0.0,
        path1_drh0_damage_prevented_by_refuge: float = 0.0,
        path1_drh0_failed_reachability_cases: float = 0.0,
        path1_drh0_action_timing_before_consequence: float = 0.0,
        path1_drh0_outcome_delta_vs_disabled_controls: float = 0.0,
        path1_drh0_pre_consequence_refuge_arrival_rate: float = 0.0,
        path1_drh0_cue_conditioned_refuge_movement_delta: float = 0.0,
        path1_drh0_refuge_reachability_success_rate: float = 0.0,
        path1_drh0_early_move_cost_paid: float = 0.0,
        path1_drh0_refuge_over_static_shelter_delta: float = 0.0,
        path1_drh0_hazard_damage_reduction_given_pre_arrival: float = 0.0,
        path1_drh0_hidden_window_action_count: float = 0.0,
        path1_drh0_no_cue_false_positive_refuge_movement: float = 0.0,
        prediction_demand_enable: int = 0,
        prediction_demand_observation_enable: int = 0,
        prediction_demand_cue_cell_count: float = 0.0,
        prediction_demand_active_cell_count: float = 0.0,
        prediction_demand_event_count: float = 0.0,
        prediction_demand_transition_count: float = 0.0,
        prediction_demand_cue_ambiguity_mean: float = 0.0,
        prediction_demand_cue_to_consequence_delay_mean: float = 0.0,
        prediction_demand_cue_without_consequence_count: float = 0.0,
        prediction_demand_consequence_without_prior_cue_count: float = 0.0,
        prediction_demand_pre_consequence_agent_context_count: float = 0.0,
        prediction_demand_consequence_agent_exposure_count: float = 0.0,
        prediction_demand_resource_consequence_total: float = 0.0,
        prediction_demand_direct_prediction_reward_active: int = 0,
        prediction_demand_direct_error_reward_active: int = 0,
        delayed_resource_depletion_enable: int = 0,
        delayed_resource_depletion_cue_cell_count: float = 0.0,
        delayed_resource_depletion_hidden_gap_cell_count: float = 0.0,
        delayed_resource_depletion_active_cell_count: float = 0.0,
        delayed_resource_depletion_event_count: float = 0.0,
        delayed_resource_depletion_agent_cue_exposure_count: float = 0.0,
        delayed_resource_depletion_agent_hidden_gap_exposure_count: float = 0.0,
        delayed_resource_depletion_agent_active_exposure_count: float = 0.0,
        delayed_resource_depletion_consequence_count: float = 0.0,
        delayed_resource_depletion_consequence_total: float = 0.0,
        delayed_resource_depletion_gather_cue_count: float = 0.0,
        delayed_resource_depletion_gather_hidden_gap_count: float = 0.0,
        delayed_resource_depletion_gather_active_count: float = 0.0,
        delayed_resource_depletion_revisit_after_cue_count: float = 0.0,
        delayed_resource_depletion_avoidance_proxy: float = 0.0,
        delayed_crowding_waste_enable: int = 0,
        delayed_crowding_waste_cue_cell_count: float = 0.0,
        delayed_crowding_waste_hidden_gap_cell_count: float = 0.0,
        delayed_crowding_waste_active_cell_count: float = 0.0,
        delayed_crowding_waste_event_count: float = 0.0,
        delayed_crowding_waste_agent_cue_exposure_count: float = 0.0,
        delayed_crowding_waste_agent_hidden_gap_exposure_count: float = 0.0,
        delayed_crowding_waste_agent_active_exposure_count: float = 0.0,
        delayed_crowding_waste_consequence_count: float = 0.0,
        delayed_crowding_waste_consequence_total: float = 0.0,
        delayed_crowding_waste_gather_cue_count: float = 0.0,
        delayed_crowding_waste_gather_hidden_gap_count: float = 0.0,
        delayed_crowding_waste_gather_active_count: float = 0.0,
        delayed_crowding_waste_revisit_after_cue_count: float = 0.0,
        delayed_crowding_waste_avoidance_proxy: float = 0.0,
        mem_max_delayed_choice_enable: int = 0,
        mem_max_delayed_choice_observation_enable: int = 0,
        mem_max_delayed_choice_event_count: float = 0.0,
        mem_max_delayed_choice_target_a_cue_count: float = 0.0,
        mem_max_delayed_choice_target_b_cue_count: float = 0.0,
        mem_max_delayed_choice_hidden_window_active: float = 0.0,
        mem_max_delayed_choice_consequence_step_active: float = 0.0,
        mem_max_delayed_choice_correct_arrival_count: float = 0.0,
        mem_max_delayed_choice_incorrect_arrival_count: float = 0.0,
        mem_max_delayed_choice_no_choice_count: float = 0.0,
        mem_max_delayed_choice_correct_choice_rate: float = 0.0,
        mem_max_delayed_choice_consequence_exposure_count: float = 0.0,
        mem_max_delayed_choice_damage_total: float = 0.0,
        mem_max_delayed_choice_damage_prevented_total: float = 0.0,
        mem_max_delayed_choice_expected_consequence_total: float = 0.0,
        mem_max_delayed_choice_applied_health_damage_total: float = 0.0,
        mem_max_delayed_choice_applied_energy_cost_total: float = 0.0,
        mem_max_delayed_choice_consequence_applied_count: float = 0.0,
        mem_max_delayed_choice_consequence_eligible_count: float = 0.0,
        mem_max_delayed_choice_duplicate_prevented_count: float = 0.0,
        mem_max_delayed_choice_correct_health_damage_total: float = 0.0,
        mem_max_delayed_choice_incorrect_health_damage_total: float = 0.0,
        mem_max_delayed_choice_no_choice_health_damage_total: float = 0.0,
        mem_max_delayed_choice_target_a_arrival_count: float = 0.0,
        mem_max_delayed_choice_target_b_arrival_count: float = 0.0,
        mem_max_delayed_choice_any_target_reachable_fraction: float = 0.0,
        mem_max_delayed_choice_shortcut_exposure: int = 0,
        mem_max_delayed_choice_behavior_metric_available: int = 0,
        mem_max_delayed_choice_behavior_coupling_supported: int = 0,
        mem_max_delayed_choice_direct_memory_reward_active: int = 0,
        mem_max_delayed_choice_direct_cognition_reward_active: int = 0,
        dev_stage_transition_count: int = 0,
        dev_stage_eval_stage_counts_json: str = "{}",
        # === Tier 3 change: end metrics fields ===
        scar_cost_weight: float | None = None,
        selected_parent_agents: Optional[List] = None,
        sim_cfg: Any = None,
        env_cfg: Any = None,
        memory_runtime_telemetry: Optional[dict] = None,
        behavior_proxy_metrics: Optional[dict] = None,
        birth_contexts: Optional[List[Dict[str, Any]]] = None,
        death_contexts: Optional[List[Dict[str, Any]]] = None,
        predator_event_contexts: Optional[List[Dict[str, Any]]] = None,
        native_public_output_counts: Optional[Dict[str, Any]] = None,
        native_public_metrics_overlay: Optional[Dict[str, Any]] = None,
        policy_cache_public_counts: Optional[Dict[str, Any]] = None,
        **_ignored_kwargs,
    ) -> None:
        """Write one row to metrics.csv for this generation."""

        # FDP v3 determinism hardening (metrics-only):
        # Metrics aggregation uses floating-point summations. Even if values are identical,
        # summation order differences can produce tiny (1e-9..1e-7) deltas due to
        # non-associativity of floating point arithmetic.
        #
        # To ensure strict byte-identical metrics.csv between fresh and resume runs,
        # we aggregate over a canonical, stable agent ordering.
        def _agent_sort_key(x) -> int:
            """Return a stable integer sort key for an agent-like object."""
            ax = _unwrap_agent(x)
            # Canonical runtime identity.
            if hasattr(ax, "agent_id"):
                try:
                    return int(getattr(ax, "agent_id"))
                except Exception:
                    pass
            # Legacy compatibility fields.
            if hasattr(ax, "id"):
                try:
                    return int(getattr(ax, "id"))
                except Exception:
                    pass
            # Fallbacks used in some older snapshots.
            if hasattr(ax, "uid"):
                try:
                    return int(getattr(ax, "uid"))
                except Exception:
                    pass
            # Last resort: keep original order (stable within a single run). This should
            # not happen in supported EIEL runs.
            return 0

        agents = sorted(list(agents or []), key=_agent_sort_key)
        if selected_parent_agents is not None:
            selected_parent_agents = sorted(list(selected_parent_agents or []), key=_agent_sort_key)
        native_overlay_values = _native_public_metrics_overlay_values(native_public_metrics_overlay)
        native_lineage_input = native_overlay_values.get("native_lineage_metrics_input")

        alive_n = int(len(agents))
        row_alive_n = alive_n
        row_birth_attempts = int(birth_attempts)
        row_birth_pre_cooldown = int(birth_pre_cooldown)
        row_birth_pre_no_energy = int(birth_pre_no_energy)
        row_birth_pre_injury_gate = int(birth_pre_injury_gate)
        row_birth_pre_no_space = int(birth_pre_no_space)
        row_birth_pre_no_mate = int(birth_pre_no_mate)
        row_birth_fail_no_space = int(birth_fail_no_space)
        row_birth_fail_no_mate = int(birth_fail_no_mate)
        row_birth_fail_buffer = int(birth_fail_buffer)
        row_birth_fail_injury_gate = int(birth_fail_injury_gate)
        row_birth_fail_scar_gate = int(birth_fail_scar_gate)
        row_birth_fail_conception = int(birth_fail_conception)
        row_birth_fail_gestation = int(birth_fail_gestation)
        if isinstance(native_public_output_counts, dict):
            try:
                if str(native_public_output_counts.get("source", "")) == "native_compact_result":
                    row_alive_n = max(0, int(native_public_output_counts.get("alive", alive_n)))
                    row_birth_attempts = max(0, int(native_public_output_counts.get("birth_attempts", birth_attempts)))
                    row_birth_pre_cooldown = max(0, int(native_public_output_counts.get("birth_pre_cooldown", birth_pre_cooldown)))
                    row_birth_pre_no_energy = max(0, int(native_public_output_counts.get("birth_pre_no_energy", birth_pre_no_energy)))
                    row_birth_pre_injury_gate = max(0, int(native_public_output_counts.get("birth_pre_injury_gate", birth_pre_injury_gate)))
                    row_birth_pre_no_space = max(0, int(native_public_output_counts.get("birth_pre_no_space", birth_pre_no_space)))
                    row_birth_pre_no_mate = max(0, int(native_public_output_counts.get("birth_pre_no_mate", birth_pre_no_mate)))
                    row_birth_fail_no_space = max(0, int(native_public_output_counts.get("birth_fail_no_space", birth_fail_no_space)))
                    row_birth_fail_no_mate = max(0, int(native_public_output_counts.get("birth_fail_no_mate", birth_fail_no_mate)))
                    row_birth_fail_buffer = max(0, int(native_public_output_counts.get("birth_fail_buffer", birth_fail_buffer)))
                    row_birth_fail_injury_gate = max(0, int(native_public_output_counts.get("birth_fail_injury_gate", birth_fail_injury_gate)))
                    row_birth_fail_scar_gate = max(0, int(native_public_output_counts.get("birth_fail_scar_gate", birth_fail_scar_gate)))
                    row_birth_fail_conception = max(0, int(native_public_output_counts.get("birth_fail_conception", birth_fail_conception)))
                    row_birth_fail_gestation = max(0, int(native_public_output_counts.get("birth_fail_gestation", birth_fail_gestation)))
            except Exception:
                row_alive_n = alive_n
                row_birth_attempts = int(birth_attempts)
                row_birth_pre_cooldown = int(birth_pre_cooldown)
                row_birth_pre_no_energy = int(birth_pre_no_energy)
                row_birth_pre_injury_gate = int(birth_pre_injury_gate)
                row_birth_pre_no_space = int(birth_pre_no_space)
                row_birth_pre_no_mate = int(birth_pre_no_mate)
                row_birth_fail_no_space = int(birth_fail_no_space)
                row_birth_fail_no_mate = int(birth_fail_no_mate)
                row_birth_fail_buffer = int(birth_fail_buffer)
                row_birth_fail_injury_gate = int(birth_fail_injury_gate)
                row_birth_fail_scar_gate = int(birth_fail_scar_gate)
                row_birth_fail_conception = int(birth_fail_conception)
                row_birth_fail_gestation = int(birth_fail_gestation)
        shock_active_i = int(shock_active)

        biome_safe_frac, biome_wild_frac = _biome_fracs(agents)

        fit = [self._safe_fitness(_unwrap_agent(a)) for a in agents] if agents else []
        mean_fit = mean(fit) if fit else 0.0
        std_fit = std(fit) if fit else 0.0
        p90_fit = percentile(fit, 90) if fit else 0.0
        max_fit = max(fit) if fit else 0.0

        surv_steps = [
            float(getattr(getattr(_unwrap_agent(a), "stats", None), "survival_steps", 0.0))
            for a in agents
        ] if agents else []
        energy_col = [
            float(getattr(getattr(_unwrap_agent(a), "stats", None), "energy_collected", 0.0))
            for a in agents
        ] if agents else []
        offspring = [
            float(getattr(getattr(_unwrap_agent(a), "stats", None), "offspring_count", 0.0))
            for a in agents
        ] if agents else []
        attacks = [
            float(getattr(getattr(_unwrap_agent(a), "stats", None), "attacks", 0.0))
            for a in agents
        ] if agents else []
        idle_paid = [
            float(getattr(getattr(_unwrap_agent(a), "stats", None), "idle_charge_paid", 0.0))
            for a in agents
        ] if agents else []

        mean_surv = mean(surv_steps) if surv_steps else 0.0
        mean_energy_col = mean(energy_col) if energy_col else 0.0
        std_energy_col = std(energy_col) if energy_col else 0.0
        mean_off = mean(offspring) if offspring else 0.0
        mean_att = mean(attacks) if attacks else 0.0
        mean_idle_paid = mean(idle_paid) if idle_paid else 0.0

        injuries = [float(getattr(_unwrap_agent(a), "injury", 0.0)) for a in agents] if agents else []
        scars = [float(getattr(_unwrap_agent(a), "scar", 0.0)) for a in agents] if agents else []

        mean_inj = mean(injuries) if injuries else 0.0
        p90_inj = percentile(injuries, 90) if injuries else 0.0
        max_inj = max(injuries) if injuries else 0.0

        mean_scar = mean(scars) if scars else 0.0
        p90_scar = percentile(scars, 90) if scars else 0.0
        max_scar = max(scars) if scars else 0.0

        w_scar = float(scar_cost_weight) if scar_cost_weight is not None else 1.0
        impair = [
            (float(getattr(_unwrap_agent(a), "injury", 0.0)) + w_scar * float(getattr(_unwrap_agent(a), "scar", 0.0)))
            for a in agents
        ] if agents else []
        mean_imp = mean(impair) if impair else 0.0

        # -----------------------------
        # Phase A chemistry observability (generation aggregates)
        # -----------------------------
        def _chem6(agent_obj: Any) -> List[float]:
            raw = getattr(agent_obj, "chemistry", None)
            vals = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            try:
                seq = list(raw) if raw is not None else []
            except Exception:
                seq = []
            n = min(6, len(seq))
            for i in range(n):
                try:
                    x = float(seq[i])
                    # deterministic finite-only fallback
                    if x == x and x != float("inf") and x != -float("inf"):
                        vals[i] = x
                except Exception:
                    pass
            return vals

        chem_energy_store: List[float] = []
        chem_repair_reserve: List[float] = []
        chem_acute_damage: List[float] = []
        chem_chronic_damage: List[float] = []
        chem_regulator_1: List[float] = []
        chem_regulator_2: List[float] = []

        for raw in agents:
            a = _unwrap_agent(raw)
            if a is None:
                continue
            c = _chem6(a)
            chem_energy_store.append(float(c[0]))
            chem_repair_reserve.append(float(c[1]))
            chem_acute_damage.append(float(c[2]))
            chem_chronic_damage.append(float(c[3]))
            chem_regulator_1.append(float(c[4]))
            chem_regulator_2.append(float(c[5]))

        acute_thr = float(getattr(sim_cfg, "chem_acute_high_threshold", 0.70)) if sim_cfg is not None else 0.70
        chronic_thr = float(getattr(sim_cfg, "chem_chronic_high_threshold", 0.70)) if sim_cfg is not None else 0.70
        acute_high_frac = float(sum(1 for x in chem_acute_damage if x >= acute_thr)) / float(max(1, len(chem_acute_damage)))
        chronic_high_frac = float(sum(1 for x in chem_chronic_damage if x >= chronic_thr)) / float(max(1, len(chem_chronic_damage)))

        chem_chronic_par: List[float] = []
        chem_repair_par: List[float] = []
        if selected_parent_agents:
            for raw in selected_parent_agents:
                a = _unwrap_agent(raw)
                if a is None:
                    continue
                c = _chem6(a)
                chem_repair_par.append(float(c[1]))
                chem_chronic_par.append(float(c[3]))

        ag_bias = [safe_aggression_bias(a) for a in agents] if agents else []
        mean_ag = mean(ag_bias) if ag_bias else 0.0
        std_ag = std(ag_bias) if ag_bias else 0.0
        p10_ag = percentile(ag_bias, 10) if ag_bias else 0.0
        p90_ag = percentile(ag_bias, 90) if ag_bias else 0.0

        gru_alive_frac = _gru_frac(agents)
        gru_parent_frac = _gru_frac(selected_parent_agents)

        alive_dims = _memory_dim_list(agents)
        parent_dims = _memory_dim_list(selected_parent_agents)

        mean_mem_alive = mean([float(x) for x in alive_dims]) if alive_dims else 0.0
        p90_mem_alive = percentile([float(x) for x in alive_dims], 90) if alive_dims else 0.0
        max_mem_alive = float(max(alive_dims)) if alive_dims else 0.0

        fr = _memory_bucket_fracs(alive_dims)
        mem0_frac = float(fr[0])
        mem8_frac = float(fr[8])
        mem16_frac = float(fr[16])
        mem32_frac = float(fr[32])
        mem64_frac = float(fr[64])

        mean_mem_parent = mean([float(x) for x in parent_dims]) if parent_dims else 0.0
        p90_mem_parent = percentile([float(x) for x in parent_dims], 90) if parent_dims else 0.0

        genome_pool = selected_parent_agents if (selected_parent_agents is not None) else []
        genome_stats = list(
            genome_stats_from_agents(genome_pool)
        )
        for index, name in enumerate(
            (
                "gen_hidden_size_mean",
                "gen_hidden_size_std",
                "gen_exploration_bias_mean",
                "gen_exploration_bias_std",
                "gen_aggression_bias_mean",
                "gen_aggression_bias_std",
                "gen_reproduction_threshold_mean",
                "gen_reproduction_threshold_std",
                "gen_mutation_rate_mean",
                "gen_mutation_rate_std",
                "genome_diversity_norm_std_mean",
            )
        ):
            if name in native_overlay_values and native_overlay_values[name] is not None:
                genome_stats[index] = float(native_overlay_values[name])
        (
            hs_m, hs_s,
            ex_m, ex_s,
            ag_m2, ag_s2,
            rt_m, rt_s,
            mr_m, mr_s,
            diversity_norm_std_mean,
        ) = genome_stats

        collapse_flag, self._collapse_streak = update_collapse_state(
            diversity_norm_std_mean=diversity_norm_std_mean,
            sim_cfg=sim_cfg,
            prev_streak=self._collapse_streak,
        )
        collapse_streak = self._collapse_streak

        # -----------------------------
        # Internal drives + coherence
        # -----------------------------
        hunger: List[float] = []
        pain: List[float] = []
        fatigue: List[float] = []
        fear: List[float] = []

        switches: List[float] = []
        oscs: List[float] = []
        drive_vol: List[float] = []
        drive_jitter: List[float] = []
        identity_drift: List[float] = []

        reset_shock: List[float] = []
        ep0_pred_mean: List[float] = []
        ep0_drive_jitter: List[float] = []
        ep0_switches: List[float] = []
        ep0_osc: List[float] = []
        ep0_selfharm: List[float] = []
        dev_active_module_count: List[float] = []

        coh_reset: List[float] = []
        coh_ep0_pred: List[float] = []

        coh_thrash: List[float] = []
        coh_selfharm: List[float] = []
        coh_pred_err: List[float] = []
        coh_penalty_total: List[float] = []

        for raw in agents:
            a = _unwrap_agent(raw)
            if a is None:
                continue

            d = getattr(a, "drives", None)
            if d is not None and len(d) >= 4:
                hunger.append(_as_float(d[0]))
                pain.append(_as_float(d[1]))
                fatigue.append(_as_float(d[2]))
                fear.append(_as_float(d[3]))

            st = getattr(a, "stats", None)
            if st is None:
                continue

            switches.append(_as_float(getattr(st, "action_switches", 0)))
            oscs.append(_as_float(getattr(st, "action_oscillations", 0)))
            drive_vol.append(_as_float(getattr(st, "drive_volatility", 0.0)))

            # Preferred jitter metric (high-frequency). Fallback to drive_volatility.
            drive_jitter.append(_as_float(getattr(st, "drive_jitter", getattr(st, "drive_volatility", 0.0))))
            identity_drift.append(_as_float(getattr(st, "identity_drift_ema", 0.0)))

            # Coherence penalty breakdown populated by SelectionMixin._effective_fitness.
            coh_thrash.append(_as_float(getattr(st, "coh_thrash", 0.0)))
            coh_selfharm.append(_as_float(getattr(st, "coh_selfharm", 0.0)))
            coh_pred_err.append(_as_float(getattr(st, "coh_pred_err_mean", 0.0)))
            coh_penalty_total.append(_as_float(getattr(st, "coh_penalty_total", 0.0)))

            # Cross-episode internal consistency (episode-0 signals)
            reset_shock.append(_as_float(getattr(st, "reset_shock", 0.0)))

            ep0_pred_mean.append(_as_float(getattr(st, "episode0_pred_err_mean", 0.0)))
            ep0_drive_jitter.append(_as_float(getattr(st, "episode0_drive_jitter", getattr(st, "episode0_drive_jitter_mean", 0.0))))
            ep0_switches.append(_as_float(getattr(st, "episode0_switches", getattr(st, "episode0_switches_mean", 0.0))))
            ep0_osc.append(_as_float(getattr(st, "episode0_oscillations", getattr(st, "episode0_oscillations_mean", 0.0))))
            ep0_selfharm.append(_as_float(getattr(st, "episode0_selfharm", getattr(st, "episode0_selfharm_mean", 0.0))))

            coh_reset.append(_as_float(getattr(st, "coh_reset_shock", 0.0)))
            coh_ep0_pred.append(_as_float(getattr(st, "coh_ep0_pred_err_mean", 0.0)))
            dev_active_module_count.append(_as_float(getattr(st, "_dev_active_module_count", 0.0)))


        # -----------------------------
        # Option C: Late-life predictive coherence pressure (analysis-clean)
        # -----------------------------
        late_enable = bool(getattr(sim_cfg, "late_pred_cost_enable", False)) if sim_cfg is not None else False
        late_w = float(getattr(sim_cfg, "late_pred_cost_weight", 0.0)) if sim_cfg is not None else 0.0

        late_err_means: List[float] = []
        late_costs: List[float] = []
        first_repro_steps: List[int] = []
        for raw in agents:
            a = _unwrap_agent(raw)
            if a is None:
                continue
            st = getattr(a, "stats", None)
            if st is None:
                continue
            # Late-life metric (post-first-reproduction window)
            try:
                lc = int(getattr(st, "late_pred_err_count", 0))
                if lc > 0:
                    lm = float(getattr(st, "late_pred_err_sum", 0.0)) / float(lc)
                    late_err_means.append(float(lm))
                    # Applied selection penalty magnitude (0 if disabled or weight=0)
                    cst = float(late_w) * float(lm) if (late_enable and late_w > 0.0 and lm > 0.0) else 0.0
                    late_costs.append(float(cst))
            except Exception:
                pass

            # Guardrail: time-to-first-reproduction
            fr = getattr(st, "first_repro_step", None)
            if fr is not None:
                try:
                    first_repro_steps.append(int(fr))
                except Exception:
                    pass

        late_active_frac = (float(len(late_err_means)) / float(max(1, alive_n)))
        late_pred_err_mean_alive = float(mean(late_err_means) if late_err_means else 0.0)
        late_pred_cost_mean_alive = float(mean(late_costs) if late_costs else 0.0)

        # Selected parents (pre-cull snapshot)
        late_err_means_par: List[float] = []
        late_costs_par: List[float] = []
        if selected_parent_agents:
            for raw in selected_parent_agents:
                a = _unwrap_agent(raw)
                if a is None:
                    continue
                st = getattr(a, "stats", None)
                if st is None:
                    continue
                try:
                    lc = int(getattr(st, "late_pred_err_count", 0))
                    if lc > 0:
                        lm = float(getattr(st, "late_pred_err_sum", 0.0)) / float(lc)
                        late_err_means_par.append(float(lm))
                        cst = float(late_w) * float(lm) if (late_enable and late_w > 0.0 and lm > 0.0) else 0.0
                        late_costs_par.append(float(cst))
                except Exception:
                    pass

        late_pred_err_mean_selected = float(mean(late_err_means_par) if late_err_means_par else 0.0)
        late_pred_cost_mean_selected = float(mean(late_costs_par) if late_costs_par else 0.0)

        first_repro_step_mean_alive = float(mean(first_repro_steps) if first_repro_steps else 0.0)
        first_repro_frac_alive = float(len(first_repro_steps)) / float(max(1, alive_n))
        founder_metrics = _founder_balance_metric_defaults(sim_cfg, agents)
        lineage_snapshot = _lineage_share_snapshot(agents)
        if int(generation) == 1 and self._early_lineage_effective_gen_1 is None:
            self._early_lineage_effective_gen_1 = float(
                native_overlay_values.get("early_lineage_effective_gen_1", lineage_snapshot["effective"])
            )
            self._early_lineage_top1_share_gen_1 = float(
                native_overlay_values.get("early_lineage_top1_share_gen_1", lineage_snapshot["top1"])
            )
        if int(generation) == 2 and self._early_lineage_effective_gen_2 is None:
            self._early_lineage_effective_gen_2 = float(
                native_overlay_values.get("early_lineage_effective_gen_2", lineage_snapshot["effective"])
            )
            self._early_lineage_top1_share_gen_2 = float(
                native_overlay_values.get("early_lineage_top1_share_gen_2", lineage_snapshot["top1"])
            )
        try:
            lineage_forensics_metrics, self._lineage_forensics_state = compute_lineage_collapse_forensics(
                generation=int(generation),
                agents=agents,
                sim_cfg=sim_cfg,
                state=self._lineage_forensics_state,
                birth_contexts=list(birth_contexts or []),
                death_contexts=list(death_contexts or []),
                lineage_strategy_descriptors_json=str(lineage_strategy_descriptors_json or "[]"),
                lineage_branch_candidates_json=str(lineage_branch_candidates_json or "[]"),
                failed_births=int(failed_births),
                birth_failure_counts={
                    "birth_fail_no_space": int(birth_fail_no_space),
                    "birth_fail_no_mate": int(birth_fail_no_mate),
                    "birth_fail_buffer": int(birth_fail_buffer),
                    "birth_fail_injury_gate": int(birth_fail_injury_gate),
                    "birth_fail_scar_gate": int(birth_fail_scar_gate),
                    "birth_fail_conception": int(birth_fail_conception),
                    "birth_fail_gestation": int(birth_fail_gestation),
                },
            )
        except ValueError:
            raise
        except Exception:
            lineage_forensics_metrics = empty_lineage_forensics_metrics()
        cog_metrics = compute_cognitive_status_metrics(
            agents,
            selected_parent_agents=selected_parent_agents,
            env_cfg=env_cfg,
            sim_cfg=sim_cfg,
            memory_runtime_telemetry=memory_runtime_telemetry,
            generation=int(generation),
        )

        behavior_proxy_metrics = dict(behavior_proxy_metrics or {})

        resource_handling_axis_mean = 0.0
        fixed_arm_axis_bin = 0.0
        if bool(getattr(sim_cfg, "resource_handling_representation_reporting_enable", False)):
            axes = []
            for agent in agents:
                genome = getattr(_unwrap_agent(agent), "genome", None)
                try:
                    axis = float(getattr(genome, "resource_handling_axis", 0.0))
                except Exception:
                    axis = 0.0
                if math.isfinite(axis):
                    axes.append(axis)
            if axes:
                resource_handling_axis_mean = float(sum(axes) / float(len(axes)))
                if resource_handling_axis_mean > 1e-9:
                    fixed_arm_axis_bin = 1.0
                elif resource_handling_axis_mean < -1e-9:
                    fixed_arm_axis_bin = -1.0

        def _behavior_metric(name: str, default: float = 0.0) -> float:
            try:
                value = float(behavior_proxy_metrics.get(name, default))
                return value if math.isfinite(value) else float(default)
            except Exception:
                return float(default)

        behavior_action_total = _behavior_metric("behavior_action_total")
        behavior_move_total = _behavior_metric("behavior_move_total")
        behavior_move_n_total = _behavior_metric("behavior_move_n_total")
        behavior_move_s_total = _behavior_metric("behavior_move_s_total")
        behavior_move_e_total = _behavior_metric("behavior_move_e_total")
        behavior_move_w_total = _behavior_metric("behavior_move_w_total")
        behavior_gather_total = _behavior_metric("behavior_gather_total")
        behavior_rest_total = _behavior_metric("behavior_rest_total")
        behavior_attack_total = _behavior_metric("behavior_attack_total")
        if isinstance(native_public_output_counts, dict):
            try:
                if str(native_public_output_counts.get("source", "")) == "native_compact_result":
                    behavior_action_total = float(native_public_output_counts.get("behavior_action_total", behavior_action_total))
                    behavior_move_total = float(native_public_output_counts.get("behavior_move_total", behavior_move_total))
                    behavior_move_n_total = float(native_public_output_counts.get("behavior_move_n_total", behavior_move_n_total))
                    behavior_move_s_total = float(native_public_output_counts.get("behavior_move_s_total", behavior_move_s_total))
                    behavior_move_e_total = float(native_public_output_counts.get("behavior_move_e_total", behavior_move_e_total))
                    behavior_move_w_total = float(native_public_output_counts.get("behavior_move_w_total", behavior_move_w_total))
                    behavior_gather_total = float(native_public_output_counts.get("behavior_gather_total", behavior_gather_total))
                    behavior_rest_total = float(native_public_output_counts.get("behavior_rest_total", behavior_rest_total))
                    behavior_attack_total = float(native_public_output_counts.get("behavior_attack_total", behavior_attack_total))
            except Exception:
                pass
        behavior_move_fraction = _behavior_metric("behavior_move_fraction")
        behavior_gather_fraction = _behavior_metric("behavior_gather_fraction")
        behavior_rest_fraction = _behavior_metric("behavior_rest_fraction")
        behavior_attack_fraction = _behavior_metric("behavior_attack_fraction")
        if behavior_action_total > 0.0:
            behavior_move_fraction = float(behavior_move_total) / float(behavior_action_total)
            behavior_gather_fraction = float(behavior_gather_total) / float(behavior_action_total)
            behavior_rest_fraction = float(behavior_rest_total) / float(behavior_action_total)
            behavior_attack_fraction = float(behavior_attack_total) / float(behavior_action_total)
        behavior_patch_interaction_proxy = float(
            behavior_gather_total + float(patch_food_harvest_total) + float(birth_patch_count)
        )
        behavior_risk_exposure_proxy = float(
            float(jungle_predator_events)
            + float(jungle_predator_damage_total)
            + float(jungle_predator_deaths)
            + float(patch_waste_occupied_mean)
            + float(death_patch_count)
        )
        behavior_protection_proxy = float(
            float(occupied_patch_shelter_mean)
            + float(patch_shelter_occupied_mean)
            + float(jungle_predator_damage_prevented_total)
        )
        behavior_reproduction_proxy = float(
            float(row_birth_attempts)
            + float(successful_births)
            + float(first_repro_frac_alive)
            + float(birth_patch_count)
        )
        behavior_visible_resource_exposure_proxy = float(
            float(patch_food_amount_max)
            * float(cog_metrics.get("sensor_channel_patch_food_amount_norm_enabled_fraction", 0.0))
        )
        behavior_visible_hazard_exposure_proxy = float(
            (float(patch_waste_max) + float(jungle_predator_risk_max))
            * max(
                float(cog_metrics.get("sensor_channel_patch_waste_norm_enabled_fraction", 0.0)),
                float(cog_metrics.get("sensor_channel_patch_movement_cost_norm_enabled_fraction", 0.0)),
            )
        )
        behavior_visible_shelter_exposure_proxy = float(
            float(patch_shelter_mean)
            * float(cog_metrics.get("sensor_channel_patch_shelter_enabled_fraction", 0.0))
        )
        behavior_visible_predator_exposure_proxy = float(
            float(jungle_predator_risk_max)
            * max(
                float(cog_metrics.get("sensor_channel_patch_scent_norm_enabled_fraction", 0.0)),
                float(cog_metrics.get("sensor_channel_patch_density_norm_enabled_fraction", 0.0)),
            )
        )

        row = MetricsRow(
            generation=int(generation),
            alive=int(row_alive_n),
            cohort_size=int(cohort_size),
            cohort_survivors=int(cohort_survivors),
            cohort_survival_rate=float(cohort_survival_rate),
            shock_active=int(shock_active_i),

            biome_safe_frac_alive=float(biome_safe_frac),
            biome_wild_frac_alive=float(biome_wild_frac),

            mean_fitness=float(mean_fit),
            std_fitness=float(std_fit),
            p90_fitness=float(p90_fit),
            max_fitness=float(max_fit),

            mean_survival_steps_total=float(mean_surv),
            deaths=int(deaths_this_gen),

            mean_energy_collected_total=float(mean_energy_col),
            std_energy_collected_total=float(std_energy_col),

            mean_offspring_total=float(mean_off),
            mean_attacks_total=float(mean_att),

            mean_aggression_bias=float(mean_ag),
            std_aggression_bias=float(std_ag),
            p10_aggression_bias=float(p10_ag),
            p90_aggression_bias=float(p90_ag),

            successful_births=int(successful_births),
            failed_births=int(failed_births),
            injury_blocked_births=int(injury_blocked_births),

            idle_charge_total=float(idle_charge_total),
            idle_charge_deaths=int(idle_charge_deaths),
            mean_idle_charge_paid_total=float(mean_idle_paid),

            mean_injury=float(mean_inj),
            p90_injury=float(p90_inj),
            max_injury=float(max_inj),
            peak_injury=float(peak_injury),

            mean_scar=float(mean_scar),
            p90_scar=float(p90_scar),
            max_scar=float(max_scar),

            mean_impairment=float(mean_imp),
            chem_energy_store_mean=float(mean(chem_energy_store) if chem_energy_store else 0.0),
            chem_repair_reserve_mean=float(mean(chem_repair_reserve) if chem_repair_reserve else 0.0),
            chem_acute_damage_mean=float(mean(chem_acute_damage) if chem_acute_damage else 0.0),
            chem_chronic_damage_mean=float(mean(chem_chronic_damage) if chem_chronic_damage else 0.0),
            chem_regulator_1_mean=float(mean(chem_regulator_1) if chem_regulator_1 else 0.0),
            chem_regulator_2_mean=float(mean(chem_regulator_2) if chem_regulator_2 else 0.0),
            chem_acute_damage_p90=float(percentile(chem_acute_damage, 90) if chem_acute_damage else 0.0),
            chem_chronic_damage_p90=float(percentile(chem_chronic_damage, 90) if chem_chronic_damage else 0.0),
            chem_acute_high_frac=float(acute_high_frac),
            chem_chronic_high_frac=float(chronic_high_frac),
            chem_chronic_damage_mean_selected_parents=float(mean(chem_chronic_par) if chem_chronic_par else 0.0),
            chem_repair_reserve_mean_selected_parents=float(mean(chem_repair_par) if chem_repair_par else 0.0),

            birth_attempts=int(row_birth_attempts),

            birth_pre_cooldown=int(row_birth_pre_cooldown),
            birth_pre_no_energy=int(row_birth_pre_no_energy),
            birth_pre_injury_gate=int(row_birth_pre_injury_gate),
            birth_pre_no_space=int(row_birth_pre_no_space),
            birth_pre_no_mate=int(row_birth_pre_no_mate),

            birth_attempt_conception_fail=int(birth_attempt_conception_fail),
            birth_gestation_fail=int(birth_gestation_fail),
            birth_make_child_fail=int(birth_make_child_fail),

            birth_fail_no_space=int(row_birth_fail_no_space),
            birth_fail_no_mate=int(row_birth_fail_no_mate),
            birth_fail_buffer=int(row_birth_fail_buffer),
            birth_fail_injury_gate=int(row_birth_fail_injury_gate),
            birth_fail_scar_gate=int(row_birth_fail_scar_gate),
            birth_fail_conception=int(row_birth_fail_conception),
            birth_fail_gestation=int(row_birth_fail_gestation),

            gen_hidden_size_mean=float(hs_m),
            gen_hidden_size_std=float(hs_s),

            gen_exploration_bias_mean=float(ex_m),
            gen_exploration_bias_std=float(ex_s),

            gen_aggression_bias_mean=float(ag_m2),
            gen_aggression_bias_std=float(ag_s2),

            gen_reproduction_threshold_mean=float(rt_m),
            gen_reproduction_threshold_std=float(rt_s),

            gen_mutation_rate_mean=float(mr_m),
            gen_mutation_rate_std=float(mr_s),

            genome_diversity_norm_std_mean=float(diversity_norm_std_mean),

            diversity_collapse=int(collapse_flag),
            diversity_collapse_streak=int(collapse_streak),

            gru_alive_frac=float(gru_alive_frac),
            gru_parent_frac=float(gru_parent_frac),

            mean_memory_dim_alive=float(mean_mem_alive),
            p90_memory_dim_alive=float(p90_mem_alive),
            max_memory_dim_alive=float(max_mem_alive),

            mem0_frac_alive=float(mem0_frac),
            mem8_frac_alive=float(mem8_frac),
            mem16_frac_alive=float(mem16_frac),
            mem32_frac_alive=float(mem32_frac),
            mem64_frac_alive=float(mem64_frac),

            mean_memory_dim_parent=float(mean_mem_parent),
            p90_memory_dim_parent=float(p90_mem_parent),

            drive_hunger_mean=float(mean(hunger) if hunger else 0.0),
            drive_hunger_std=float(std(hunger) if hunger else 0.0),
            drive_pain_mean=float(mean(pain) if pain else 0.0),
            drive_pain_std=float(std(pain) if pain else 0.0),
            drive_fatigue_mean=float(mean(fatigue) if fatigue else 0.0),
            drive_fatigue_std=float(std(fatigue) if fatigue else 0.0),
            drive_fear_mean=float(mean(fear) if fear else 0.0),
            drive_fear_std=float(std(fear) if fear else 0.0),

            coherence_switches_mean=float(mean(switches) if switches else 0.0),
            coherence_oscillations_mean=float(mean(oscs) if oscs else 0.0),
            drive_volatility_mean=float(mean(drive_vol) if drive_vol else 0.0),

            coh_thrash_mean=float(mean(coh_thrash) if coh_thrash else 0.0),
            coh_selfharm_mean=float(mean(coh_selfharm) if coh_selfharm else 0.0),
            coh_pred_err_mean=float(mean(coh_pred_err) if coh_pred_err else 0.0),
            coh_penalty_total_mean=float(mean(coh_penalty_total) if coh_penalty_total else 0.0),
            drive_jitter_mean=float(mean(drive_jitter) if drive_jitter else 0.0),

            identity_drift_mean=float(mean(identity_drift) if identity_drift else 0.0),
            identity_drift_p90=float(percentile(identity_drift, 90) if identity_drift else 0.0),
            identity_drift_max=float(max(identity_drift) if identity_drift else 0.0),

            reset_shock_mean=float(mean(reset_shock) if reset_shock else 0.0),
            episode0_pred_err_mean=float(mean(ep0_pred_mean) if ep0_pred_mean else 0.0),
            episode0_drive_jitter_mean=float(mean(ep0_drive_jitter) if ep0_drive_jitter else 0.0),
            episode0_switches_mean=float(mean(ep0_switches) if ep0_switches else 0.0),
            episode0_oscillations_mean=float(mean(ep0_osc) if ep0_osc else 0.0),
            episode0_selfharm_mean=float(mean(ep0_selfharm) if ep0_selfharm else 0.0),

            # Option C + guardrails
            late_pred_active_frac=float(late_active_frac),
            late_pred_err_mean_alive=float(late_pred_err_mean_alive),
            late_pred_cost_mean_alive=float(late_pred_cost_mean_alive),
            late_pred_err_mean_selected_parents=float(late_pred_err_mean_selected),
            late_pred_cost_mean_selected_parents=float(late_pred_cost_mean_selected),
            first_repro_step_mean_alive=float(first_repro_step_mean_alive),
            first_repro_frac_alive=float(first_repro_frac_alive),

            behavior_action_total=float(behavior_action_total),
            behavior_move_total=float(behavior_move_total),
            behavior_move_n_total=float(behavior_move_n_total),
            behavior_move_s_total=float(behavior_move_s_total),
            behavior_move_e_total=float(behavior_move_e_total),
            behavior_move_w_total=float(behavior_move_w_total),
            behavior_gather_total=float(behavior_gather_total),
            behavior_rest_total=float(behavior_rest_total),
            behavior_attack_total=float(behavior_attack_total),
            behavior_move_fraction=float(behavior_move_fraction),
            behavior_gather_fraction=float(behavior_gather_fraction),
            behavior_rest_fraction=float(behavior_rest_fraction),
            behavior_attack_fraction=float(behavior_attack_fraction),
            behavior_patch_interaction_proxy=float(behavior_patch_interaction_proxy),
            behavior_risk_exposure_proxy=float(behavior_risk_exposure_proxy),
            behavior_protection_proxy=float(behavior_protection_proxy),
            behavior_reproduction_proxy=float(behavior_reproduction_proxy),
            behavior_visible_resource_exposure_proxy=float(behavior_visible_resource_exposure_proxy),
            behavior_visible_hazard_exposure_proxy=float(behavior_visible_hazard_exposure_proxy),
            behavior_visible_shelter_exposure_proxy=float(behavior_visible_shelter_exposure_proxy),
            behavior_visible_predator_exposure_proxy=float(behavior_visible_predator_exposure_proxy),

            coh_reset_shock_mean=float(mean(coh_reset) if coh_reset else 0.0),
            coh_ep0_pred_err_mean=float(mean(coh_ep0_pred) if coh_ep0_pred else 0.0),
            dev_active_module_count_mean=float(mean(dev_active_module_count) if dev_active_module_count else 0.0),
            dev_stage_transition_count=int(dev_stage_transition_count),
            dev_stage_eval_stage_counts_json=str(dev_stage_eval_stage_counts_json or "{}"),

            # === Tier 3 change: migration pressure metrics ===
            tier3_overcrowd_mean=float(tier3_overcrowd_mean),
            tier3_overcrowd_max=float(tier3_overcrowd_max),
            tier3_crossings_per_step=float(tier3_crossings_per_step),
            tier3_cross_events_per_step=float(tier3_cross_events_per_step),
            tier3_extra_move_cost_mean=float(tier3_extra_move_cost_mean),
            tier3_extra_move_cost_max=float(tier3_extra_move_cost_max),
            tier4_juvenile_exposure_steps=int(tier4_juvenile_exposure_steps),
            tier4_adult_exposure_steps=int(tier4_adult_exposure_steps),
            tier4_juvenile_deaths=int(tier4_juvenile_deaths),
            tier4_adult_deaths=int(tier4_adult_deaths),
            tier4_juvenile_death_rate_per_1k_steps=float(tier4_juvenile_death_rate_per_1k_steps),
            tier4_adult_death_rate_per_1k_steps=float(tier4_adult_death_rate_per_1k_steps),
            tier4_juvenile_fraction_mean=float(tier4_juvenile_fraction_mean),
            tier4_fraction_reproduced_alive_end=float(tier4_fraction_reproduced_alive_end),
            tier4_first_repro_step_mean_reproducers_alive_end=float(tier4_first_repro_step_mean_reproducers_alive_end),
            tier4_first_repro_step_p50_reproducers_alive_end=float(tier4_first_repro_step_p50_reproducers_alive_end),
            tier4_first_repro_step_p90_reproducers_alive_end=float(tier4_first_repro_step_p90_reproducers_alive_end),
            tier4_bins_steps_json=str(tier4_bins_steps_json),
            tier4_alive_age_hist_json=str(tier4_alive_age_hist_json),
            tier4_deaths_age_hist_json=str(tier4_deaths_age_hist_json),
            patch_food_amount_mean=float(patch_food_amount_mean),
            patch_food_amount_min=float(patch_food_amount_min),
            patch_food_amount_max=float(patch_food_amount_max),
            patch_food_amount_p10=float(patch_food_amount_p10),
            patch_food_amount_p90=float(patch_food_amount_p90),
            patch_food_depleted_frac=float(patch_food_depleted_frac),
            patch_food_full_frac=float(patch_food_full_frac),
            patch_food_regrowth_total=float(patch_food_regrowth_total),
            patch_food_harvest_total=float(patch_food_harvest_total),
            patch_fertility_mean=float(patch_fertility_mean),
            mosaic0a_burst_cell_frac=float(mosaic0a_burst_cell_frac),
            mosaic0a_renewal_cell_frac=float(mosaic0a_renewal_cell_frac),
            mosaic0a_burst_food_amount_mean=float(mosaic0a_burst_food_amount_mean),
            mosaic0a_renewal_food_amount_mean=float(mosaic0a_renewal_food_amount_mean),
            mosaic0a_burst_fertility_mean=float(mosaic0a_burst_fertility_mean),
            mosaic0a_renewal_fertility_mean=float(mosaic0a_renewal_fertility_mean),
            respec0b_resource_a_cell_frac=float(respec0b_resource_a_cell_frac),
            respec0b_resource_b_cell_frac=float(respec0b_resource_b_cell_frac),
            respec0b_resource_gather_attempts=float(respec0b_resource_gather_attempts),
            respec0b_resource_gather_successes=float(respec0b_resource_gather_successes),
            respec0b_resource_gather_failed_handling=float(respec0b_resource_gather_failed_handling),
            respec0b_resource_gather_resource_a_successes=float(respec0b_resource_gather_resource_a_successes),
            respec0b_resource_gather_resource_b_successes=float(respec0b_resource_gather_resource_b_successes),
            respec0b_resource_gather_energy_resource_a=float(respec0b_resource_gather_energy_resource_a),
            respec0b_resource_gather_energy_resource_b=float(respec0b_resource_gather_energy_resource_b),
            respec0b_resource_gather_handling_ticks=float(respec0b_resource_gather_handling_ticks),
            respec0b_resource_gather_interruptions=float(respec0b_resource_gather_interruptions),
            resource_handling_axis_mean=float(resource_handling_axis_mean),
            fixed_arm_axis_bin=float(fixed_arm_axis_bin),
            patch_cover_mean=float(patch_cover_mean),
            patch_cover_occupied_mean=float(patch_cover_occupied_mean),
            patch_shelter_mean=float(patch_shelter_mean),
            patch_shelter_occupied_mean=float(patch_shelter_occupied_mean),
            patch_visibility_mean=float(patch_visibility_mean),
            patch_visibility_occupied_mean=float(patch_visibility_occupied_mean),
            patch_movement_cost_mean=float(patch_movement_cost_mean),
            patch_movement_cost_occupied_mean=float(patch_movement_cost_occupied_mean),
            patch_movement_energy_extra_total=float(patch_movement_energy_extra_total),
            patch_movement_energy_extra_mean=float(patch_movement_energy_extra_mean),
            patch_scent_mean=float(patch_scent_mean),
            patch_scent_max=float(patch_scent_max),
            patch_scent_p90=float(patch_scent_p90),
            patch_scent_occupied_mean=float(patch_scent_occupied_mean),
            patch_scent_emitted_total=float(patch_scent_emitted_total),
            patch_scent_decayed_total=float(patch_scent_decayed_total),
            patch_waste_mean=float(patch_waste_mean),
            patch_waste_max=float(patch_waste_max),
            patch_waste_p90=float(patch_waste_p90),
            patch_waste_occupied_mean=float(patch_waste_occupied_mean),
            patch_waste_emitted_total=float(patch_waste_emitted_total),
            patch_waste_decayed_total=float(patch_waste_decayed_total),
            patch_density_mean=float(patch_density_mean),
            patch_density_max=float(patch_density_max),
            patch_density_p90=float(patch_density_p90),
            patch_density_occupied_mean=float(patch_density_occupied_mean),
            patch_density_hotspot_frac=float(patch_density_hotspot_frac),
            patch_observation_enable=int(patch_observation_enable),
            patch_observation_schema_version=int(patch_observation_schema_version),
            patch_observation_channel_count=int(patch_observation_channel_count),
            patch_observation_size=int(patch_observation_size),
            patch_world_descriptor_schema_version=int(patch_world_descriptor_schema_version),
            patch_world_diagnostics_enable=int(patch_world_diagnostics_enable),
            patch_world_lineage_descriptors_enable=int(patch_world_lineage_descriptors_enable),
            occupied_patch_food_amount_mean=float(occupied_patch_food_amount_mean),
            occupied_patch_fertility_mean=float(occupied_patch_fertility_mean),
            occupied_patch_cover_mean=float(occupied_patch_cover_mean),
            occupied_patch_shelter_mean=float(occupied_patch_shelter_mean),
            occupied_patch_visibility_mean=float(occupied_patch_visibility_mean),
            occupied_patch_movement_cost_mean=float(occupied_patch_movement_cost_mean),
            occupied_patch_scent_mean=float(occupied_patch_scent_mean),
            occupied_patch_waste_mean=float(occupied_patch_waste_mean),
            occupied_patch_density_mean=float(occupied_patch_density_mean),
            birth_patch_count=float(birth_patch_count),
            birth_patch_food_amount_mean=float(birth_patch_food_amount_mean),
            birth_patch_fertility_mean=float(birth_patch_fertility_mean),
            birth_patch_cover_mean=float(birth_patch_cover_mean),
            birth_patch_shelter_mean=float(birth_patch_shelter_mean),
            birth_patch_visibility_mean=float(birth_patch_visibility_mean),
            birth_patch_movement_cost_mean=float(birth_patch_movement_cost_mean),
            birth_patch_scent_mean=float(birth_patch_scent_mean),
            birth_patch_waste_mean=float(birth_patch_waste_mean),
            birth_patch_density_mean=float(birth_patch_density_mean),
            death_patch_count=float(death_patch_count),
            death_patch_food_amount_mean=float(death_patch_food_amount_mean),
            death_patch_fertility_mean=float(death_patch_fertility_mean),
            death_patch_cover_mean=float(death_patch_cover_mean),
            death_patch_shelter_mean=float(death_patch_shelter_mean),
            death_patch_visibility_mean=float(death_patch_visibility_mean),
            death_patch_movement_cost_mean=float(death_patch_movement_cost_mean),
            death_patch_scent_mean=float(death_patch_scent_mean),
            death_patch_waste_mean=float(death_patch_waste_mean),
            death_patch_density_mean=float(death_patch_density_mean),
            patch_food_occupancy_bias=float(patch_food_occupancy_bias),
            patch_fertility_occupancy_bias=float(patch_fertility_occupancy_bias),
            patch_cover_occupancy_bias=float(patch_cover_occupancy_bias),
            patch_shelter_occupancy_bias=float(patch_shelter_occupancy_bias),
            patch_visibility_occupancy_bias=float(patch_visibility_occupancy_bias),
            patch_movement_cost_occupancy_bias=float(patch_movement_cost_occupancy_bias),
            patch_scent_occupancy_bias=float(patch_scent_occupancy_bias),
            patch_waste_occupancy_bias=float(patch_waste_occupancy_bias),
            patch_density_occupancy_bias=float(patch_density_occupancy_bias),
            lineage_habitat_descriptor_count=float(lineage_habitat_descriptor_count),
            lineage_patch_food_amount_mean_std=float(lineage_patch_food_amount_mean_std),
            lineage_patch_cover_mean_std=float(lineage_patch_cover_mean_std),
            lineage_patch_shelter_mean_std=float(lineage_patch_shelter_mean_std),
            lineage_patch_scent_mean_std=float(lineage_patch_scent_mean_std),
            lineage_patch_waste_mean_std=float(lineage_patch_waste_mean_std),
            lineage_patch_density_mean_std=float(lineage_patch_density_mean_std),
            lineage_patch_recovery_activity_mean_std=float(lineage_patch_recovery_activity_mean_std),
            founder_balance_enable=int(founder_metrics["founder_balance_enable"]),
            founder_group_count=int(founder_metrics["founder_group_count"]),
            founder_min_agents_per_group=int(founder_metrics["founder_min_agents_per_group"]),
            founder_initial_effective_lineages=float(founder_metrics["founder_initial_effective_lineages"]),
            founder_initial_lineage_entropy=float(founder_metrics["founder_initial_lineage_entropy"]),
            founder_initial_top1_share=float(founder_metrics["founder_initial_top1_share"]),
            founder_initial_min_lineage_share=float(founder_metrics["founder_initial_min_lineage_share"]),
            founder_initial_max_lineage_share=float(founder_metrics["founder_initial_max_lineage_share"]),
            founder_initial_lineage_count=float(founder_metrics["founder_initial_lineage_count"]),
            founder_balance_violation_count=int(founder_metrics["founder_balance_violation_count"]),
            early_lineage_effective_gen_1=float(self._early_lineage_effective_gen_1 or 0.0),
            early_lineage_effective_gen_2=float(self._early_lineage_effective_gen_2 or 0.0),
            early_lineage_top1_share_gen_1=float(self._early_lineage_top1_share_gen_1 or 0.0),
            early_lineage_top1_share_gen_2=float(self._early_lineage_top1_share_gen_2 or 0.0),
            lineage_diversity_schema_version=int(lineage_diversity_schema_version),
            lineage_diversity_reporting_enable=int(lineage_diversity_reporting_enable),
            lineage_strategy_descriptors_enable=int(lineage_strategy_descriptors_enable),
            lineage_branch_candidate_reporting_enable=int(lineage_branch_candidate_reporting_enable),
            lineage_active_count=float(lineage_active_count),
            lineage_extinct_count=float(lineage_extinct_count),
            lineage_alive_count_total=float(lineage_alive_count_total),
            lineage_top1_alive_share=float(lineage_top1_alive_share),
            lineage_top3_alive_share=float(lineage_top3_alive_share),
            lineage_top5_alive_share=float(lineage_top5_alive_share),
            lineage_top1_birth_share=float(lineage_top1_birth_share),
            lineage_top3_birth_share=float(lineage_top3_birth_share),
            lineage_top5_birth_share=float(lineage_top5_birth_share),
            lineage_top1_death_share=float(lineage_top1_death_share),
            lineage_top3_death_share=float(lineage_top3_death_share),
            lineage_alive_share_entropy=float(lineage_alive_share_entropy),
            lineage_birth_share_entropy=float(lineage_birth_share_entropy),
            lineage_effective_alive_lineages=float(lineage_effective_alive_lineages),
            lineage_effective_birth_lineages=float(lineage_effective_birth_lineages),
            lineage_top1_dominance_warning=int(lineage_top1_dominance_warning),
            lineage_top3_dominance_warning=int(lineage_top3_dominance_warning),
            lineage_entropy_collapse_warning=int(lineage_entropy_collapse_warning),
            lineage_low_active_count_warning=int(lineage_low_active_count_warning),
            lineage_collapse_warning_any=int(lineage_collapse_warning_any),
            lineage_strategy_descriptor_count=float(lineage_strategy_descriptor_count),
            lineage_birth_patch_waste_mean_std=float(lineage_birth_patch_waste_mean_std),
            lineage_birth_patch_shelter_mean_std=float(lineage_birth_patch_shelter_mean_std),
            lineage_birth_patch_recovery_activity_mean_std=float(lineage_birth_patch_recovery_activity_mean_std),
            lineage_predator_damage_mean_std=float(lineage_predator_damage_mean_std),
            lineage_predator_damage_prevented_mean_std=float(lineage_predator_damage_prevented_mean_std),
            lineage_reproduction_rate_std=float(lineage_reproduction_rate_std),
            lineage_branch_candidate_count=float(lineage_branch_candidate_count),
            lineage_habitat_descriptors_json=str(lineage_habitat_descriptors_json or "[]"),
            lineage_strategy_descriptors_json=str(lineage_strategy_descriptors_json or "[]"),
            lineage_branch_candidates_json=str(lineage_branch_candidates_json or "[]"),
            **lineage_forensics_metrics,
            patch_observation_min=float(patch_observation_min),
            patch_observation_max=float(patch_observation_max),
            patch_observation_nan_count=float(patch_observation_nan_count),
            patch_world_waste_consequence_enable=int(patch_world_waste_consequence_enable),
            patch_regrowth_base_total=float(patch_regrowth_base_total),
            patch_regrowth_effective_total=float(patch_regrowth_effective_total),
            patch_regrowth_suppressed_total=float(patch_regrowth_suppressed_total),
            patch_regrowth_multiplier_mean=float(patch_regrowth_multiplier_mean),
            patch_regrowth_multiplier_min=float(patch_regrowth_multiplier_min),
            patch_regrowth_multiplier_p10=float(patch_regrowth_multiplier_p10),
            patch_regrowth_multiplier_p90=float(patch_regrowth_multiplier_p90),
            patch_waste_regrowth_penalty_mean=float(patch_waste_regrowth_penalty_mean),
            patch_waste_regrowth_penalty_max=float(patch_waste_regrowth_penalty_max),
            patch_density_regrowth_penalty_mean=float(patch_density_regrowth_penalty_mean),
            patch_density_regrowth_penalty_max=float(patch_density_regrowth_penalty_max),
            patch_high_waste_regrowth_suppression_frac=float(patch_high_waste_regrowth_suppression_frac),
            patch_high_density_regrowth_suppression_frac=float(patch_high_density_regrowth_suppression_frac),
            jungle_ecological_recovery_enable=int(jungle_ecological_recovery_enable),
            patch_recovery_activity_mean=float(patch_recovery_activity_mean),
            patch_recovery_activity_min=float(patch_recovery_activity_min),
            patch_recovery_activity_max=float(patch_recovery_activity_max),
            patch_recovery_activity_p10=float(patch_recovery_activity_p10),
            patch_recovery_activity_p90=float(patch_recovery_activity_p90),
            patch_recovery_activity_occupied_mean=float(patch_recovery_activity_occupied_mean),
            patch_recovery_positive_total=float(patch_recovery_positive_total),
            patch_recovery_negative_total=float(patch_recovery_negative_total),
            patch_recovery_decay_total=float(patch_recovery_decay_total),
            jungle_recovery_regrowth_boost_total=float(jungle_recovery_regrowth_boost_total),
            jungle_recovery_regrowth_boost_mean=float(jungle_recovery_regrowth_boost_mean),
            jungle_recovery_regrowth_boost_max=float(jungle_recovery_regrowth_boost_max),
            jungle_recovery_regrowth_boost_p90=float(jungle_recovery_regrowth_boost_p90),
            patch_regrowth_after_waste_total=float(patch_regrowth_after_waste_total),
            patch_regrowth_after_recovery_total=float(patch_regrowth_after_recovery_total),
            patch_recovery_added_regrowth_total=float(patch_recovery_added_regrowth_total),
            patch_fertility_damage_total=float(patch_fertility_damage_total),
            patch_fertility_recovery_total=float(patch_fertility_recovery_total),
            jungle_predator_pressure_enable=int(jungle_predator_pressure_enable),
            jungle_predator_risk_mean=float(jungle_predator_risk_mean),
            jungle_predator_risk_min=float(jungle_predator_risk_min),
            jungle_predator_risk_max=float(jungle_predator_risk_max),
            jungle_predator_risk_p10=float(jungle_predator_risk_p10),
            jungle_predator_risk_p90=float(jungle_predator_risk_p90),
            jungle_predator_occupied_risk_mean=float(jungle_predator_occupied_risk_mean),
            jungle_predator_occupied_risk_p90=float(jungle_predator_occupied_risk_p90),
            jungle_predator_events=float(jungle_predator_events),
            jungle_predator_event_rate=float(jungle_predator_event_rate),
            jungle_predator_damage_total=float(jungle_predator_damage_total),
            jungle_predator_damage_mean=float(jungle_predator_damage_mean),
            jungle_terrain_protection_enable=int(jungle_terrain_protection_enable),
            jungle_predator_damage_raw_total=float(jungle_predator_damage_raw_total),
            jungle_predator_damage_effective_total=float(jungle_predator_damage_effective_total),
            jungle_predator_damage_prevented_total=float(jungle_predator_damage_prevented_total),
            jungle_predator_damage_multiplier_mean=float(jungle_predator_damage_multiplier_mean),
            jungle_predator_damage_multiplier_min=float(jungle_predator_damage_multiplier_min),
            jungle_predator_damage_multiplier_p10=float(jungle_predator_damage_multiplier_p10),
            jungle_predator_damage_multiplier_p90=float(jungle_predator_damage_multiplier_p90),
            jungle_predator_event_damage_multiplier_mean=float(jungle_predator_event_damage_multiplier_mean),
            jungle_predator_event_damage_prevented_mean=float(jungle_predator_event_damage_prevented_mean),
            occupied_predator_damage_multiplier_mean=float(occupied_predator_damage_multiplier_mean),
            jungle_predator_deaths=float(jungle_predator_deaths),
            jungle_predator_scent_component_mean=float(jungle_predator_scent_component_mean),
            jungle_predator_waste_component_mean=float(jungle_predator_waste_component_mean),
            jungle_predator_density_component_mean=float(jungle_predator_density_component_mean),
            jungle_predator_visibility_component_mean=float(jungle_predator_visibility_component_mean),
            jungle_predator_cover_reduction_mean=float(jungle_predator_cover_reduction_mean),
            jungle_predator_shelter_reduction_mean=float(jungle_predator_shelter_reduction_mean),
            predator_event_patch_count=float(predator_event_patch_count),
            predator_event_patch_scent_mean=float(predator_event_patch_scent_mean),
            predator_event_patch_waste_mean=float(predator_event_patch_waste_mean),
            predator_event_patch_density_mean=float(predator_event_patch_density_mean),
            predator_event_patch_cover_mean=float(predator_event_patch_cover_mean),
            predator_event_patch_shelter_mean=float(predator_event_patch_shelter_mean),
            predator_event_patch_visibility_mean=float(predator_event_patch_visibility_mean),
            jungle_reproduction_emissions_enable=int(jungle_reproduction_emissions_enable),
            jungle_birth_scent_emitted_total=float(jungle_birth_scent_emitted_total),
            jungle_birth_waste_emitted_total=float(jungle_birth_waste_emitted_total),
            jungle_birth_emission_count=float(jungle_birth_emission_count),
            jungle_birth_emission_patch_scent_mean=float(jungle_birth_emission_patch_scent_mean),
            jungle_birth_emission_patch_waste_mean=float(jungle_birth_emission_patch_waste_mean),
            jungle_birth_emission_patch_density_mean=float(jungle_birth_emission_patch_density_mean),
            jungle_birth_emission_patch_cover_mean=float(jungle_birth_emission_patch_cover_mean),
            jungle_birth_emission_patch_shelter_mean=float(jungle_birth_emission_patch_shelter_mean),
            jungle_juvenile_predator_vulnerability_enable=int(jungle_juvenile_predator_vulnerability_enable),
            jungle_juvenile_predator_events=float(jungle_juvenile_predator_events),
            jungle_juvenile_predator_damage_total=float(jungle_juvenile_predator_damage_total),
            jungle_juvenile_predator_deaths=float(jungle_juvenile_predator_deaths),
            jungle_adult_predator_events=float(jungle_adult_predator_events),
            jungle_adult_predator_damage_total=float(jungle_adult_predator_damage_total),
            jungle_juvenile_vs_adult_predator_event_ratio=float(jungle_juvenile_vs_adult_predator_event_ratio),
            wdl_delayed_hazard_enable=int(wdl_delayed_hazard_enable),
            wdl_delayed_hazard_cue_cell_count=float(wdl_delayed_hazard_cue_cell_count),
            wdl_delayed_hazard_active_cell_count=float(wdl_delayed_hazard_active_cell_count),
            wdl_delayed_hazard_event_count=float(wdl_delayed_hazard_event_count),
            wdl_delayed_hazard_agent_cue_exposure_count=float(wdl_delayed_hazard_agent_cue_exposure_count),
            wdl_delayed_hazard_agent_active_exposure_count=float(wdl_delayed_hazard_agent_active_exposure_count),
            wdl_delayed_hazard_damage_total=float(wdl_delayed_hazard_damage_total),
            wdl_delayed_hazard_damage_mean=float(wdl_delayed_hazard_damage_mean),
            wdl_delayed_hazard_damage_max=float(wdl_delayed_hazard_damage_max),
            wdl_delayed_hazard_agents_damaged=float(wdl_delayed_hazard_agents_damaged),
            wdl_delayed_hazard_cue_to_active_delay_mean=float(wdl_delayed_hazard_cue_to_active_delay_mean),
            wdl_delayed_hazard_cue_without_active_count=float(wdl_delayed_hazard_cue_without_active_count),
            wdl_delayed_hazard_active_without_prior_cue_count=float(wdl_delayed_hazard_active_without_prior_cue_count),
            wdl_cue_seen_agents=float(wdl_cue_seen_agents),
            wdl_cue_exposed_next_move_away_count=float(wdl_cue_exposed_next_move_away_count),
            wdl_cue_exposed_next_move_toward_count=float(wdl_cue_exposed_next_move_toward_count),
            wdl_cue_exposed_stayed_count=float(wdl_cue_exposed_stayed_count),
            wdl_pre_hazard_exit_count=float(wdl_pre_hazard_exit_count),
            wdl_hazard_enter_count=float(wdl_hazard_enter_count),
            wdl_hazard_avoidance_proxy=float(wdl_hazard_avoidance_proxy),
            wdl_hazard_exposure_after_cue_count=float(wdl_hazard_exposure_after_cue_count),
            wdl_hazard_damage_by_lineage_json=str(wdl_hazard_damage_by_lineage_json),
            wdl_cue_exposure_by_lineage_json=str(wdl_cue_exposure_by_lineage_json),
            wdl_hazard_exposure_by_lineage_json=str(wdl_hazard_exposure_by_lineage_json),
            path1_refuge_cue_cell_count=float(path1_refuge_cue_cell_count),
            path1_refuge_active_cell_count=float(path1_refuge_active_cell_count),
            path1_refuge_active_without_prior_cue_count=float(path1_refuge_active_without_prior_cue_count),
            path1_refuge_cue_without_active_count=float(path1_refuge_cue_without_active_count),
            path1_refuge_zone_cell_count=float(path1_refuge_zone_cell_count),
            path1_refuge_reachable_zone_cell_count=float(path1_refuge_reachable_zone_cell_count),
            path1_refuge_reachable_before_active_fraction=float(path1_refuge_reachable_before_active_fraction),
            path1_refuge_agents_in_hazard_at_cue=float(path1_refuge_agents_in_hazard_at_cue),
            path1_refuge_agents_reachable_refuge_at_cue=float(path1_refuge_agents_reachable_refuge_at_cue),
            path1_refuge_agents_in_refuge_before_active=float(path1_refuge_agents_in_refuge_before_active),
            path1_refuge_agents_stayed_in_hazard_until_active=float(path1_refuge_agents_stayed_in_hazard_until_active),
            path1_refuge_shelter_occupancy_pre_active=float(path1_refuge_shelter_occupancy_pre_active),
            path1_refuge_cover_occupancy_pre_active=float(path1_refuge_cover_occupancy_pre_active),
            path1_refuge_move_toward_refuge_pre_active=float(path1_refuge_move_toward_refuge_pre_active),
            path1_refuge_move_away_from_refuge_pre_active=float(path1_refuge_move_away_from_refuge_pre_active),
            path1_refuge_disabled_control_leakage_flag=float(path1_refuge_disabled_control_leakage_flag),
            path1_drh0_cue_exposure_count=float(path1_drh0_cue_exposure_count),
            path1_drh0_cue_visible_steps=float(path1_drh0_cue_visible_steps),
            path1_drh0_hidden_window_steps=float(path1_drh0_hidden_window_steps),
            path1_drh0_hazard_active_steps=float(path1_drh0_hazard_active_steps),
            path1_drh0_refuge_cell_count=float(path1_drh0_refuge_cell_count),
            path1_drh0_reachable_refuge_count=float(path1_drh0_reachable_refuge_count),
            path1_drh0_distance_to_nearest_refuge_at_cue=float(path1_drh0_distance_to_nearest_refuge_at_cue),
            path1_drh0_movement_toward_refuge_after_cue=float(path1_drh0_movement_toward_refuge_after_cue),
            path1_drh0_refuge_occupancy_before_hazard=float(path1_drh0_refuge_occupancy_before_hazard),
            path1_drh0_refuge_occupancy_during_hazard=float(path1_drh0_refuge_occupancy_during_hazard),
            path1_drh0_damage_in_refuge=float(path1_drh0_damage_in_refuge),
            path1_drh0_damage_outside_refuge=float(path1_drh0_damage_outside_refuge),
            path1_drh0_damage_prevented_by_refuge=float(path1_drh0_damage_prevented_by_refuge),
            path1_drh0_failed_reachability_cases=float(path1_drh0_failed_reachability_cases),
            path1_drh0_action_timing_before_consequence=float(path1_drh0_action_timing_before_consequence),
            path1_drh0_outcome_delta_vs_disabled_controls=float(path1_drh0_outcome_delta_vs_disabled_controls),
            path1_drh0_pre_consequence_refuge_arrival_rate=float(path1_drh0_pre_consequence_refuge_arrival_rate),
            path1_drh0_cue_conditioned_refuge_movement_delta=float(path1_drh0_cue_conditioned_refuge_movement_delta),
            path1_drh0_refuge_reachability_success_rate=float(path1_drh0_refuge_reachability_success_rate),
            path1_drh0_early_move_cost_paid=float(path1_drh0_early_move_cost_paid),
            path1_drh0_refuge_over_static_shelter_delta=float(path1_drh0_refuge_over_static_shelter_delta),
            path1_drh0_hazard_damage_reduction_given_pre_arrival=float(path1_drh0_hazard_damage_reduction_given_pre_arrival),
            path1_drh0_hidden_window_action_count=float(path1_drh0_hidden_window_action_count),
            path1_drh0_no_cue_false_positive_refuge_movement=float(path1_drh0_no_cue_false_positive_refuge_movement),
            prediction_demand_enable=int(prediction_demand_enable),
            prediction_demand_observation_enable=int(prediction_demand_observation_enable),
            prediction_demand_cue_cell_count=float(prediction_demand_cue_cell_count),
            prediction_demand_active_cell_count=float(prediction_demand_active_cell_count),
            prediction_demand_event_count=float(prediction_demand_event_count),
            prediction_demand_transition_count=float(prediction_demand_transition_count),
            prediction_demand_cue_ambiguity_mean=float(prediction_demand_cue_ambiguity_mean),
            prediction_demand_cue_to_consequence_delay_mean=float(prediction_demand_cue_to_consequence_delay_mean),
            prediction_demand_cue_without_consequence_count=float(prediction_demand_cue_without_consequence_count),
            prediction_demand_consequence_without_prior_cue_count=float(
                prediction_demand_consequence_without_prior_cue_count
            ),
            prediction_demand_pre_consequence_agent_context_count=float(
                prediction_demand_pre_consequence_agent_context_count
            ),
            prediction_demand_consequence_agent_exposure_count=float(
                prediction_demand_consequence_agent_exposure_count
            ),
            prediction_demand_resource_consequence_total=float(prediction_demand_resource_consequence_total),
            prediction_demand_direct_prediction_reward_active=int(prediction_demand_direct_prediction_reward_active),
            prediction_demand_direct_error_reward_active=int(prediction_demand_direct_error_reward_active),
            delayed_resource_depletion_enable=int(delayed_resource_depletion_enable),
            delayed_resource_depletion_cue_cell_count=float(delayed_resource_depletion_cue_cell_count),
            delayed_resource_depletion_hidden_gap_cell_count=float(
                delayed_resource_depletion_hidden_gap_cell_count
            ),
            delayed_resource_depletion_active_cell_count=float(delayed_resource_depletion_active_cell_count),
            delayed_resource_depletion_event_count=float(delayed_resource_depletion_event_count),
            delayed_resource_depletion_agent_cue_exposure_count=float(
                delayed_resource_depletion_agent_cue_exposure_count
            ),
            delayed_resource_depletion_agent_hidden_gap_exposure_count=float(
                delayed_resource_depletion_agent_hidden_gap_exposure_count
            ),
            delayed_resource_depletion_agent_active_exposure_count=float(
                delayed_resource_depletion_agent_active_exposure_count
            ),
            delayed_resource_depletion_consequence_count=float(delayed_resource_depletion_consequence_count),
            delayed_resource_depletion_consequence_total=float(delayed_resource_depletion_consequence_total),
            delayed_resource_depletion_gather_cue_count=float(delayed_resource_depletion_gather_cue_count),
            delayed_resource_depletion_gather_hidden_gap_count=float(
                delayed_resource_depletion_gather_hidden_gap_count
            ),
            delayed_resource_depletion_gather_active_count=float(delayed_resource_depletion_gather_active_count),
            delayed_resource_depletion_revisit_after_cue_count=float(
                delayed_resource_depletion_revisit_after_cue_count
            ),
            delayed_resource_depletion_avoidance_proxy=float(delayed_resource_depletion_avoidance_proxy),
            delayed_crowding_waste_enable=int(delayed_crowding_waste_enable),
            delayed_crowding_waste_cue_cell_count=float(delayed_crowding_waste_cue_cell_count),
            delayed_crowding_waste_hidden_gap_cell_count=float(delayed_crowding_waste_hidden_gap_cell_count),
            delayed_crowding_waste_active_cell_count=float(delayed_crowding_waste_active_cell_count),
            delayed_crowding_waste_event_count=float(delayed_crowding_waste_event_count),
            delayed_crowding_waste_agent_cue_exposure_count=float(
                delayed_crowding_waste_agent_cue_exposure_count
            ),
            delayed_crowding_waste_agent_hidden_gap_exposure_count=float(
                delayed_crowding_waste_agent_hidden_gap_exposure_count
            ),
            delayed_crowding_waste_agent_active_exposure_count=float(
                delayed_crowding_waste_agent_active_exposure_count
            ),
            delayed_crowding_waste_consequence_count=float(delayed_crowding_waste_consequence_count),
            delayed_crowding_waste_consequence_total=float(delayed_crowding_waste_consequence_total),
            delayed_crowding_waste_gather_cue_count=float(delayed_crowding_waste_gather_cue_count),
            delayed_crowding_waste_gather_hidden_gap_count=float(
                delayed_crowding_waste_gather_hidden_gap_count
            ),
            delayed_crowding_waste_gather_active_count=float(delayed_crowding_waste_gather_active_count),
            delayed_crowding_waste_revisit_after_cue_count=float(delayed_crowding_waste_revisit_after_cue_count),
            delayed_crowding_waste_avoidance_proxy=float(delayed_crowding_waste_avoidance_proxy),
            mem_max_delayed_choice_enable=int(mem_max_delayed_choice_enable),
            mem_max_delayed_choice_observation_enable=int(mem_max_delayed_choice_observation_enable),
            mem_max_delayed_choice_event_count=float(mem_max_delayed_choice_event_count),
            mem_max_delayed_choice_target_a_cue_count=float(mem_max_delayed_choice_target_a_cue_count),
            mem_max_delayed_choice_target_b_cue_count=float(mem_max_delayed_choice_target_b_cue_count),
            mem_max_delayed_choice_hidden_window_active=float(mem_max_delayed_choice_hidden_window_active),
            mem_max_delayed_choice_consequence_step_active=float(mem_max_delayed_choice_consequence_step_active),
            mem_max_delayed_choice_correct_arrival_count=float(mem_max_delayed_choice_correct_arrival_count),
            mem_max_delayed_choice_incorrect_arrival_count=float(mem_max_delayed_choice_incorrect_arrival_count),
            mem_max_delayed_choice_no_choice_count=float(mem_max_delayed_choice_no_choice_count),
            mem_max_delayed_choice_correct_choice_rate=float(mem_max_delayed_choice_correct_choice_rate),
            mem_max_delayed_choice_consequence_exposure_count=float(
                mem_max_delayed_choice_consequence_exposure_count
            ),
            mem_max_delayed_choice_damage_total=float(mem_max_delayed_choice_damage_total),
            mem_max_delayed_choice_damage_prevented_total=float(mem_max_delayed_choice_damage_prevented_total),
            mem_max_delayed_choice_expected_consequence_total=float(mem_max_delayed_choice_expected_consequence_total),
            mem_max_delayed_choice_applied_health_damage_total=float(
                mem_max_delayed_choice_applied_health_damage_total
            ),
            mem_max_delayed_choice_applied_energy_cost_total=float(
                mem_max_delayed_choice_applied_energy_cost_total
            ),
            mem_max_delayed_choice_consequence_applied_count=float(mem_max_delayed_choice_consequence_applied_count),
            mem_max_delayed_choice_consequence_eligible_count=float(mem_max_delayed_choice_consequence_eligible_count),
            mem_max_delayed_choice_duplicate_prevented_count=float(mem_max_delayed_choice_duplicate_prevented_count),
            mem_max_delayed_choice_correct_health_damage_total=float(
                mem_max_delayed_choice_correct_health_damage_total
            ),
            mem_max_delayed_choice_incorrect_health_damage_total=float(
                mem_max_delayed_choice_incorrect_health_damage_total
            ),
            mem_max_delayed_choice_no_choice_health_damage_total=float(
                mem_max_delayed_choice_no_choice_health_damage_total
            ),
            mem_max_delayed_choice_target_a_arrival_count=float(mem_max_delayed_choice_target_a_arrival_count),
            mem_max_delayed_choice_target_b_arrival_count=float(mem_max_delayed_choice_target_b_arrival_count),
            mem_max_delayed_choice_any_target_reachable_fraction=float(
                mem_max_delayed_choice_any_target_reachable_fraction
            ),
            mem_max_delayed_choice_shortcut_exposure=int(mem_max_delayed_choice_shortcut_exposure),
            mem_max_delayed_choice_behavior_metric_available=int(mem_max_delayed_choice_behavior_metric_available),
            mem_max_delayed_choice_behavior_coupling_supported=int(
                mem_max_delayed_choice_behavior_coupling_supported
            ),
            mem_max_delayed_choice_direct_memory_reward_active=int(mem_max_delayed_choice_direct_memory_reward_active),
            mem_max_delayed_choice_direct_cognition_reward_active=int(
                mem_max_delayed_choice_direct_cognition_reward_active
            ),
            # === Tier 3 change: end migration pressure metrics ===
            **cog_metrics,
        )

        fields = row.__dataclass_fields__
        for name, value in native_overlay_values.items():
            if name in fields and value is not None:
                setattr(row, name, value)

        policy_cache_metrics, policy_cache_source = _policy_cache_metrics_from_public_counts(
            policy_cache_public_counts
        )
        if not policy_cache_metrics:
            policy_cache_metrics = _native_policy_cache_metrics_from_public_payloads(
                native_public_output_counts,
                native_public_metrics_overlay,
            )
            if policy_cache_metrics:
                policy_cache_source = "native_compact_result"
        for name, value in policy_cache_metrics.items():
            if name in fields:
                setattr(row, name, value)

        if int(generation) <= int(getattr(self, '_last_written_gen', -1)):
            # Skip duplicates at fork boundary (history already present).
            self._last_perf_csv_write_s = 0.0
            self._last_perf_lineage_write_s = 0.0
            return

        t_csv0 = time.perf_counter()
        append_metrics_row(self.path, row)
        self._last_perf_csv_write_s = time.perf_counter() - t_csv0
        t_lineage0 = time.perf_counter()
        self.lineage.log_generation(
            generation=int(generation),
            agents=agents,
            sim_cfg=sim_cfg,
            genome_diversity=float(diversity_norm_std_mean),
            native_lineage_input=native_lineage_input,
        )
        self._last_perf_lineage_write_s = time.perf_counter() - t_lineage0

        self._last_written_gen = int(generation)

        # Store last row as a plain dict for in-process consumers.
        try:
            self.last_row = asdict(row)
            self.last_policy_cache_metrics = dict(policy_cache_metrics)
            self.last_policy_cache_metric_source = policy_cache_source
            self.last_native_policy_cache_metrics = (
                dict(policy_cache_metrics) if policy_cache_source == "native_compact_result" else {}
            )
        except Exception:
            self.last_row = None
            self.last_policy_cache_metrics = {}
            self.last_policy_cache_metric_source = ""
            self.last_native_policy_cache_metrics = {}
