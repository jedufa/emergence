from __future__ import annotations

"""CXXCORE-5 tick-island layout-cache diagnostic helpers.

Diagnostic-only. The cache is rebuildable from authoritative Python state and
is not checkpoint-authoritative simulation state.
"""

from collections import Counter
from dataclasses import dataclass, field
import hashlib
import json
import time
from typing import Any, Sequence

import numpy as np

from perf.cxxcore3_runner_backend_shadow import compare_runner_backend_states, compare_summaries, state_summary
from perf.cxxcore4_cost_layout_backend import (
    CostLayout,
    CostLayoutBackendInputs,
    as_cost_layout,
    as_cost_layout_backend_inputs,
    build_cost_layout_from_surfaces,
    compare_cost_vectors,
    evaluate_cost_layout_python,
    layout_summary,
    run_cost_layout_backend_python_reference,
)
from perf.cxxcore_env_apply_shadow import EnvApplyShadowConfig
from perf.native.cost_charge_surface_kernel import COST_SURFACE_COLUMN_NAMES, pack_cost_charge_surfaces


SCHEMA_VERSION = "cxxcore5_tick_island_layout_cache_v1"
CACHE_INVALIDATION_REASONS = (
    "agent_order_changed",
    "genome_changed",
    "age_or_developmental_state_changed",
    "prediction_effective_lr_changed",
    "perception_config_changed",
    "cognition_config_changed",
    "runner_config_changed",
    "unknown_dependency_full_rebuild",
)


@dataclass(frozen=True)
class CostLayoutCacheSnapshot:
    schema_version: str
    layout: CostLayout
    agent_id_order_digest: str
    flags_digest: str
    config_digest: str
    row_dependency_digests: tuple[str, ...]
    cache_hits: int
    cache_misses: int
    rows_rebuilt: int
    full_rebuilds: int
    row_updates: int
    invalidation_reasons: dict[str, int]
    build_s: float
    update_s: float
    cache_lookup_s: float
    digest_validation_s: float
    production_cost_s: float


@dataclass(frozen=True)
class TickIslandLayoutCacheInputs:
    backend: CostLayoutBackendInputs
    cache_snapshot: CostLayoutCacheSnapshot


@dataclass
class CostLayoutCache:
    env_cfg: Any
    sim_cfg: Any
    schema_version: str = SCHEMA_VERSION
    _snapshot: CostLayoutCacheSnapshot | None = None
    _rows: np.ndarray | None = None
    _surfaces: list[Any] = field(default_factory=list)
    _cache_hits: int = 0
    _cache_misses: int = 0
    _rows_rebuilt: int = 0
    _full_rebuilds: int = 0
    _row_updates: int = 0
    _invalidation_reasons: Counter[str] = field(default_factory=Counter)

    def build_or_update(
        self,
        agents: Sequence[Any],
        *,
        force_full_rebuild: bool = False,
        invalidation_reasons: Sequence[str] = (),
    ) -> tuple[CostLayoutCacheSnapshot, dict[str, Any]]:
        from agents.cognition import compute_env_cost_charge_scalar_from_surface, express_cost_charge_surface

        lookup_t0 = time.perf_counter()
        agent_ids = np.ascontiguousarray([int(getattr(agent, "agent_id", i)) for i, agent in enumerate(agents)], dtype=np.int32)
        agent_order_digest = _stable_digest({"agent_ids": agent_ids.tolist()})
        flags_digest = _flags_digest(self.sim_cfg)
        config_digest = _config_digest(self.env_cfg, self.sim_cfg)
        row_digests = tuple(_agent_row_dependency_digest(agent, self.sim_cfg) for agent in agents)
        lookup_s = time.perf_counter() - lookup_t0

        previous = self._snapshot
        reasons = Counter(str(reason) for reason in invalidation_reasons if reason)
        full_rebuild = bool(force_full_rebuild or previous is None)
        if previous is None:
            reasons["unknown_dependency_full_rebuild"] += 1
        elif previous.agent_id_order_digest != agent_order_digest:
            reasons["agent_order_changed"] += 1
            full_rebuild = True
        elif previous.config_digest != config_digest or previous.flags_digest != flags_digest:
            reasons["runner_config_changed"] += 1
            full_rebuild = True

        changed_rows: list[int] = []
        if not full_rebuild and previous is not None:
            for idx, (old, new) in enumerate(zip(previous.row_dependency_digests, row_digests)):
                if old != new:
                    changed_rows.append(idx)
            if changed_rows:
                reasons["age_or_developmental_state_changed"] += len(changed_rows)

        production_cost_s = 0.0
        build_s = 0.0
        update_s = 0.0
        digest_validation_s = 0.0
        production_scalars: list[float] = []
        surfaces: list[Any]
        if full_rebuild:
            build_t0 = time.perf_counter()
            surfaces = []
            for agent in agents:
                surface_t0 = time.perf_counter()
                surface = express_cost_charge_surface(
                    getattr(agent, "genome", None),
                    agent=agent,
                    env_cfg=self.env_cfg,
                    sim_cfg=self.sim_cfg,
                )
                production_cost_s += time.perf_counter() - surface_t0
                surfaces.append(surface)
                production_scalars.append(float(compute_env_cost_charge_scalar_from_surface(surface, sim_cfg=self.sim_cfg)))
            layout = build_cost_layout_from_surfaces(agent_ids=agent_ids, surfaces=surfaces, sim_cfg=self.sim_cfg)
            self._rows = layout.rows.copy()
            self._surfaces = list(surfaces)
            self._cache_misses += 1
            self._full_rebuilds += 1
            self._rows_rebuilt += int(len(agents))
            build_s = time.perf_counter() - build_t0
        elif changed_rows:
            update_t0 = time.perf_counter()
            rows = np.asarray(self._rows, dtype=np.float64).copy()
            surfaces = list(self._surfaces)
            for idx in changed_rows:
                agent = agents[idx]
                surface_t0 = time.perf_counter()
                surface = express_cost_charge_surface(
                    getattr(agent, "genome", None),
                    agent=agent,
                    env_cfg=self.env_cfg,
                    sim_cfg=self.sim_cfg,
                )
                production_cost_s += time.perf_counter() - surface_t0
                surfaces[idx] = surface
                rows[idx, :] = pack_cost_charge_surfaces([surface])[0]
            layout = as_cost_layout(
                agent_ids=agent_ids,
                rows=rows,
                charge_channels=bool(getattr(self.sim_cfg, "enable_experimental_perception_cost_charging", False)),
                charge_radius=bool(getattr(self.sim_cfg, "enable_experimental_sensor_radius_cost_charging", False)),
                ledger_enabled=bool(getattr(self.sim_cfg, "enable_cognitive_budget_ledger", True)),
            )
            production_scalars = [
                float(compute_env_cost_charge_scalar_from_surface(surface, sim_cfg=self.sim_cfg)) for surface in surfaces
            ]
            self._rows = layout.rows.copy()
            self._surfaces = list(surfaces)
            self._cache_misses += 1
            self._row_updates += len(changed_rows)
            self._rows_rebuilt += len(changed_rows)
            update_s = time.perf_counter() - update_t0
        else:
            digest_t0 = time.perf_counter()
            layout = as_cost_layout(
                agent_ids=agent_ids,
                rows=np.asarray(self._rows, dtype=np.float64),
                charge_channels=bool(getattr(self.sim_cfg, "enable_experimental_perception_cost_charging", False)),
                charge_radius=bool(getattr(self.sim_cfg, "enable_experimental_sensor_radius_cost_charging", False)),
                ledger_enabled=bool(getattr(self.sim_cfg, "enable_cognitive_budget_ledger", True)),
            )
            digest_validation_s = time.perf_counter() - digest_t0
            production_scalars = []
            self._cache_hits += 1

        self._invalidation_reasons.update(reasons)
        snapshot = CostLayoutCacheSnapshot(
            schema_version=SCHEMA_VERSION,
            layout=layout,
            agent_id_order_digest=agent_order_digest,
            flags_digest=flags_digest,
            config_digest=config_digest,
            row_dependency_digests=row_digests,
            cache_hits=int(self._cache_hits),
            cache_misses=int(self._cache_misses),
            rows_rebuilt=int(self._rows_rebuilt),
            full_rebuilds=int(self._full_rebuilds),
            row_updates=int(self._row_updates),
            invalidation_reasons=dict(sorted(self._invalidation_reasons.items())),
            build_s=float(build_s),
            update_s=float(update_s),
            cache_lookup_s=float(lookup_s),
            digest_validation_s=float(digest_validation_s),
            production_cost_s=float(production_cost_s),
        )
        self._snapshot = snapshot
        info = {
            "cache_mode": "full_rebuild" if full_rebuild else "row_update" if changed_rows else "hit",
            "changed_rows": [int(v) for v in changed_rows],
            "production_scalars": np.asarray(production_scalars, dtype=np.float64),
            "layout_scalars": evaluate_cost_layout_python(layout),
        }
        return snapshot, info


def as_tick_island_inputs(
    *,
    config: EnvApplyShadowConfig,
    agent_ids: Any,
    initial_energy: Any,
    initial_health: Any,
    initial_injury: Any,
    initial_scar: Any,
    pos_y_by_id: Any,
    pos_x_by_id: Any,
    agent_grid: Any,
    grid: Any,
    actions_by_tick: Any,
    cache_snapshot: CostLayoutCacheSnapshot,
    move_cost_mults: Any | None = None,
    juvenile_harvest_mults: Any | None = None,
) -> TickIslandLayoutCacheInputs:
    backend = as_cost_layout_backend_inputs(
        config=config,
        agent_ids=agent_ids,
        initial_energy=initial_energy,
        initial_health=initial_health,
        initial_injury=initial_injury,
        initial_scar=initial_scar,
        pos_y_by_id=pos_y_by_id,
        pos_x_by_id=pos_x_by_id,
        agent_grid=agent_grid,
        grid=grid,
        actions_by_tick=actions_by_tick,
        layout=cache_snapshot.layout,
        move_cost_mults=move_cost_mults,
        juvenile_harvest_mults=juvenile_harvest_mults,
    )
    return TickIslandLayoutCacheInputs(backend=backend, cache_snapshot=cache_snapshot)


def run_tick_island_python_reference(inputs: TickIslandLayoutCacheInputs) -> dict[str, np.ndarray]:
    return run_cost_layout_backend_python_reference(inputs.backend)


def cache_summary(snapshot: CostLayoutCacheSnapshot) -> dict[str, Any]:
    return {
        "schema_version": snapshot.schema_version,
        "agent_count": int(snapshot.layout.agent_ids.shape[0]),
        "agent_id_order_digest": snapshot.agent_id_order_digest,
        "layout_digest": snapshot.layout.digest,
        "field_names": list(COST_SURFACE_COLUMN_NAMES),
        "flags_digest": snapshot.flags_digest,
        "config_digest": snapshot.config_digest,
        "cache_hits": int(snapshot.cache_hits),
        "cache_misses": int(snapshot.cache_misses),
        "rows_rebuilt": int(snapshot.rows_rebuilt),
        "full_rebuilds": int(snapshot.full_rebuilds),
        "row_updates": int(snapshot.row_updates),
        "invalidation_reasons": dict(snapshot.invalidation_reasons),
        "build_s": float(snapshot.build_s),
        "update_s": float(snapshot.update_s),
        "cache_lookup_s": float(snapshot.cache_lookup_s),
        "digest_validation_s": float(snapshot.digest_validation_s),
        "production_cost_s": float(snapshot.production_cost_s),
        "layout": layout_summary(snapshot.layout),
    }


def checkpoint_metrics_boundary_classification() -> list[dict[str, str]]:
    return [
        {"state": "Python Agent objects/genomes/stats", "classification": "authoritative Python state"},
        {"state": "GridWorld arrays and runner energy/health arrays", "classification": "authoritative Python state"},
        {"state": "CostLayoutCacheSnapshot rows/digests/counters", "classification": "rebuildable shadow/cache state"},
        {"state": "CXXCORE-5 native backend copied arrays", "classification": "rebuildable shadow/cache state"},
        {"state": "native sparse summaries and compare reports", "classification": "runtime diagnostic output"},
        {"state": "probe_summary.json and evidence markdown", "classification": "report-only evidence"},
        {"state": "checkpoint-required state", "classification": "none added by CXXCORE-5"},
        {"state": "metrics-required state", "classification": "none added by CXXCORE-5"},
    ]


def audit_payload() -> dict[str, Any]:
    return {
        "schema_version": "cxxcore5_tick_island_layout_cache_audit_v1",
        "finding": "cache_or_incremental_layout_feasible_for_shadow_no_rng_slice",
        "stable_across_ticks": [
            "prediction_cost_charged when effective LR/config are stable",
            "introspection_cost_charged for stable genome/config",
            "self_regulation_cost_charged for stable genome/config",
            "self_modification_cost_charged for stable genome/config",
            "perception costs when developmental factor/perception config are stable",
        ],
        "may_change": [
            "developmental_cost_charged with age/developmental state",
            "maturation_cost_charged with age/maturity state",
            "perception cost fields when developmental scaling or perception config changes",
            "prediction cost when effective LR changes",
            "all rows when agent order/genome/config changes",
        ],
        "invalidation_is_conservative_and_deterministic": True,
        "cache_checkpoint_semantics": "cache rows are rebuildable shadow state and are not checkpoint-authoritative",
        "rng_boundary": "included CXXCORE-5 path requires no independent C++ RNG",
        "metrics_boundary": "native/cache outputs are diagnostics/report-only and do not change metrics definitions",
        "checkpoint_metrics_classification": checkpoint_metrics_boundary_classification(),
        "decision_gate": "cache_or_incremental_layout_feasible_for_shadow_no_rng_slice",
    }


def parity_payload(
    *,
    shape: str,
    arm: str,
    agents: int,
    ticks: int,
    cells: int,
    report: dict[str, Any],
    snapshot: CostLayoutCacheSnapshot | None,
    cache_mode: str,
) -> dict[str, Any]:
    first = report.get("first_divergence") or {}
    return {
        "shape": str(shape),
        "arm": str(arm),
        "agents": int(agents),
        "ticks": int(ticks),
        "cells": int(cells),
        "parity": "pass" if bool(report.get("ok")) else "fail",
        "mismatch_count": int(report.get("mismatch_count", 0) or 0),
        "max_abs_diff": float(report.get("max_abs_diff", 0.0) or 0.0),
        "first_divergence": report.get("first_divergence"),
        "first_divergence_generation": first.get("generation"),
        "first_divergence_tick": first.get("tick"),
        "first_divergence_agent_id": first.get("agent_id"),
        "first_divergence_grid_coordinate": first.get("index") if first.get("field") == "grid" else None,
        "first_divergence_field": first.get("field"),
        "python_value": first.get("python", first.get("production")),
        "native_value": first.get("candidate", first.get("layout")),
        "abs_diff": first.get("abs_diff"),
        "cache_mode": str(cache_mode),
        "cache_hits": int(snapshot.cache_hits) if snapshot is not None else 0,
        "cache_misses": int(snapshot.cache_misses) if snapshot is not None else 0,
        "rows_rebuilt": int(snapshot.rows_rebuilt) if snapshot is not None else 0,
        "full_rebuilds": int(snapshot.full_rebuilds) if snapshot is not None else 0,
    }


def _stable_digest(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")).hexdigest()


def _flags_digest(sim_cfg: Any) -> str:
    names = (
        "enable_cognitive_budget_ledger",
        "enable_experimental_perception_cost_charging",
        "enable_experimental_sensor_radius_cost_charging",
        "enable_experimental_prediction_cost_charging",
        "enable_experimental_developmental_cost_charging",
        "enable_experimental_introspection_cost_charging",
        "enable_experimental_self_regulation_cost_charging",
        "enable_experimental_self_modification_cost_charging",
        "env_hotpath_inactive_phenotype_defaults_enable",
    )
    return _stable_digest({name: getattr(sim_cfg, name, None) for name in names})


def _config_digest(env_cfg: Any, sim_cfg: Any) -> str:
    names = (
        "developmental_cognition_mode",
        "developmental_maturity_age",
        "juvenile_cognition_scale",
        "adult_cognition_scale",
        "prediction_cost_base",
        "prediction_cost_per_lr",
        "prediction_state_cost",
        "developmental_cognition_cost_base",
        "developmental_cognition_cost_per_unexpressed_capacity",
        "maturation_cost_per_tick",
    )
    env_names = ("width", "height", "view_radius")
    return _stable_digest(
        {
            "sim": {name: getattr(sim_cfg, name, None) for name in names},
            "env": {name: getattr(env_cfg, name, None) for name in env_names},
        }
    )


def _agent_row_dependency_digest(agent: Any, sim_cfg: Any) -> str:
    genome = getattr(agent, "genome", None)
    stats = getattr(agent, "stats", None)
    genome_names = (
        "memory_dim",
        "use_gru",
        "sensor_channel_mask",
        "sensor_radius",
        "pred_lr",
        "drive_logit_gain",
        "introspection_channel_mask",
        "self_regulation_control_mask",
        "self_modification_target_mask",
    )
    payload = {
        "agent_id": int(getattr(agent, "agent_id", 0) or 0),
        "alive": bool(getattr(agent, "alive", True)),
        "survival_steps": int(getattr(stats, "survival_steps", 0) or 0) if stats is not None else 0,
        "genome": {name: getattr(genome, name, None) for name in genome_names},
        "flags": _flags_digest(sim_cfg),
        "directed_inheritance_payload": getattr(agent, "directed_inheritance_payload", None),
    }
    return _stable_digest(payload)


__all__ = [
    "CACHE_INVALIDATION_REASONS",
    "SCHEMA_VERSION",
    "CostLayoutCache",
    "CostLayoutCacheSnapshot",
    "TickIslandLayoutCacheInputs",
    "as_tick_island_inputs",
    "audit_payload",
    "cache_summary",
    "checkpoint_metrics_boundary_classification",
    "compare_cost_vectors",
    "compare_runner_backend_states",
    "compare_summaries",
    "evaluate_cost_layout_python",
    "parity_payload",
    "run_tick_island_python_reference",
    "state_summary",
]
