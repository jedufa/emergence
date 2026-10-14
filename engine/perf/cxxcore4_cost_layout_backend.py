from __future__ import annotations

"""CXXCORE-4 cost-layout/backend diagnostic helpers.

This module is diagnostic-only. It builds and evaluates numeric
CostChargeSurface rows for shadow/parity probes; normal simulation remains
Python-authoritative.
"""

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Sequence

import numpy as np

from perf.cxxcore3_runner_backend_shadow import (
    RunnerBackendShadowInputs,
    as_runner_backend_shadow_inputs,
    compare_runner_backend_states,
    compare_summaries,
    run_runner_backend_python_reference,
    state_summary,
)
from perf.cxxcore_env_apply_shadow import EnvApplyShadowConfig
from perf.native.cost_charge_surface_kernel import (
    COST_SURFACE_COLUMN_NAMES,
    as_cost_surface_matrix,
    cost_surface_scalars_numpy,
    pack_cost_charge_surfaces,
)


SCHEMA_VERSION = "cxxcore_cost_layout_v1"
BACKEND_SCHEMA_VERSION = "cxxcore4_cost_layout_backend_v1"
ACTION_CODE_ORDER = ("move_north", "move_south", "move_east", "move_west", "gather", "rest", "attack")


@dataclass(frozen=True)
class CostLayout:
    schema_version: str
    agent_ids: np.ndarray
    rows: np.ndarray
    charge_channels: bool
    charge_radius: bool
    ledger_enabled: bool
    action_code_order: tuple[str, ...]
    omitted_fields: tuple[str, ...]
    digest: str


@dataclass(frozen=True)
class CostLayoutBackendInputs:
    runner: RunnerBackendShadowInputs
    layout: CostLayout


def _layout_digest(
    *,
    agent_ids: np.ndarray,
    rows: np.ndarray,
    charge_channels: bool,
    charge_radius: bool,
    ledger_enabled: bool,
    omitted_fields: Sequence[str],
) -> str:
    h = hashlib.sha256()
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "fields": list(COST_SURFACE_COLUMN_NAMES),
        "agent_ids_dtype": str(agent_ids.dtype),
        "rows_dtype": str(rows.dtype),
        "rows_shape": [int(v) for v in rows.shape],
        "charge_channels": bool(charge_channels),
        "charge_radius": bool(charge_radius),
        "ledger_enabled": bool(ledger_enabled),
        "action_code_order": list(ACTION_CODE_ORDER),
        "omitted_fields": list(omitted_fields),
    }
    h.update(json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    h.update(np.ascontiguousarray(agent_ids, dtype=np.int32).tobytes())
    h.update(np.ascontiguousarray(rows, dtype=np.float64).tobytes())
    return h.hexdigest()


def as_cost_layout(
    *,
    agent_ids: Any,
    rows: Any,
    charge_channels: bool,
    charge_radius: bool,
    ledger_enabled: bool = True,
    omitted_fields: Sequence[str] = (),
) -> CostLayout:
    ids = np.ascontiguousarray(agent_ids, dtype=np.int32)
    matrix = as_cost_surface_matrix(rows)
    if int(matrix.shape[0]) != int(ids.shape[0]):
        raise ValueError("cost layout rows must match agent_ids")
    omitted = tuple(str(name) for name in omitted_fields)
    digest = _layout_digest(
        agent_ids=ids,
        rows=matrix,
        charge_channels=bool(charge_channels),
        charge_radius=bool(charge_radius),
        ledger_enabled=bool(ledger_enabled),
        omitted_fields=omitted,
    )
    return CostLayout(
        schema_version=SCHEMA_VERSION,
        agent_ids=ids,
        rows=matrix,
        charge_channels=bool(charge_channels),
        charge_radius=bool(charge_radius),
        ledger_enabled=bool(ledger_enabled),
        action_code_order=ACTION_CODE_ORDER,
        omitted_fields=omitted,
        digest=digest,
    )


def build_cost_layout_from_surfaces(
    *,
    agent_ids: Any,
    surfaces: Sequence[Any],
    sim_cfg: Any,
    omitted_fields: Sequence[str] = (),
) -> CostLayout:
    rows = pack_cost_charge_surfaces(surfaces)
    return as_cost_layout(
        agent_ids=agent_ids,
        rows=rows,
        charge_channels=bool(getattr(sim_cfg, "enable_experimental_perception_cost_charging", False)),
        charge_radius=bool(getattr(sim_cfg, "enable_experimental_sensor_radius_cost_charging", False)),
        ledger_enabled=bool(getattr(sim_cfg, "enable_cognitive_budget_ledger", True)),
        omitted_fields=omitted_fields,
    )


def build_cost_layout_from_agents(
    *,
    agents: Sequence[Any],
    env_cfg: Any,
    sim_cfg: Any,
) -> tuple[CostLayout, dict[str, Any]]:
    from agents.cognition import compute_env_cost_charge_scalar_from_surface, express_cost_charge_surface

    ids: list[int] = []
    surfaces: list[Any] = []
    production_scalars: list[float] = []
    fallback_count = 0
    for agent in agents:
        ids.append(int(getattr(agent, "agent_id", len(ids))))
        try:
            surface = express_cost_charge_surface(
                getattr(agent, "genome", None),
                agent=agent,
                env_cfg=env_cfg,
                sim_cfg=sim_cfg,
            )
        except Exception:
            fallback_count += 1
            raise
        surfaces.append(surface)
        production_scalars.append(float(compute_env_cost_charge_scalar_from_surface(surface, sim_cfg=sim_cfg)))
    layout = build_cost_layout_from_surfaces(agent_ids=np.asarray(ids, dtype=np.int32), surfaces=surfaces, sim_cfg=sim_cfg)
    info = {
        "agents": int(len(ids)),
        "fallback_count": int(fallback_count),
        "production_scalars": np.ascontiguousarray(production_scalars, dtype=np.float64),
    }
    return layout, info


def evaluate_cost_layout_python(layout: CostLayout) -> np.ndarray:
    return np.ascontiguousarray(
        cost_surface_scalars_numpy(
            layout.rows,
            charge_channels=layout.charge_channels,
            charge_radius=layout.charge_radius,
            ledger_enabled=layout.ledger_enabled,
        ),
        dtype=np.float64,
    )


def cost_matrix_from_layout(layout: CostLayout, actions_by_tick: Any) -> np.ndarray:
    actions = np.ascontiguousarray(actions_by_tick, dtype=np.int32)
    if actions.ndim != 2:
        raise ValueError("actions_by_tick must be a 2D matrix")
    if int(actions.shape[1]) != int(layout.agent_ids.shape[0]):
        raise ValueError("actions_by_tick columns must match cost layout agent_ids")
    scalar = evaluate_cost_layout_python(layout).astype(np.float32)
    return np.ascontiguousarray(np.broadcast_to(scalar.reshape((1, -1)), actions.shape).copy(), dtype=np.float32)


def as_cost_layout_backend_inputs(
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
    layout: CostLayout,
    move_cost_mults: Any | None = None,
    juvenile_harvest_mults: Any | None = None,
) -> CostLayoutBackendInputs:
    actions = np.ascontiguousarray(actions_by_tick, dtype=np.int32)
    if not np.array_equal(np.ascontiguousarray(agent_ids, dtype=np.int32), layout.agent_ids):
        raise ValueError("cost layout agent_ids must match backend agent_ids")
    runner = as_runner_backend_shadow_inputs(
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
        actions_by_tick=actions,
        cost_scalars_by_tick=cost_matrix_from_layout(layout, actions),
        move_cost_mults=move_cost_mults,
        juvenile_harvest_mults=juvenile_harvest_mults,
    )
    return CostLayoutBackendInputs(runner=runner, layout=layout)


def run_cost_layout_backend_python_reference(inputs: CostLayoutBackendInputs) -> dict[str, np.ndarray]:
    return run_runner_backend_python_reference(inputs.runner)


def compare_cost_vectors(
    *,
    agent_ids: Any,
    production: Any,
    layout: Any,
    actions_by_tick: Any | None = None,
    atol: float = 1.0e-12,
) -> dict[str, Any]:
    ids = np.ascontiguousarray(agent_ids, dtype=np.int32)
    prod = np.asarray(production, dtype=np.float64)
    cand = np.asarray(layout, dtype=np.float64)
    report: dict[str, Any] = {
        "schema_version": BACKEND_SCHEMA_VERSION,
        "ok": True,
        "mismatch_count": 0,
        "max_abs_diff": 0.0,
        "first_divergence": None,
        "fields_compared": ["cost_scalar_by_agent"],
    }
    if prod.shape != cand.shape:
        report["ok"] = False
        report["mismatch_count"] = 1
        report["first_divergence"] = {
            "field": "cost_scalar_by_agent",
            "agent_id": None,
            "tick": None,
            "action": None,
            "component": "shape",
            "production": list(prod.shape),
            "layout": list(cand.shape),
            "abs_diff": None,
        }
        return report
    diff = np.abs(prod - cand)
    report["max_abs_diff"] = float(np.max(diff)) if diff.size else 0.0
    bad = np.nonzero(diff > float(atol))[0]
    if bad.size:
        idx = int(bad[0])
        action = None
        if actions_by_tick is not None:
            actions = np.ascontiguousarray(actions_by_tick, dtype=np.int32)
            if actions.ndim == 2 and actions.shape[0] and idx < int(actions.shape[1]):
                action = int(actions[0, idx])
        report["ok"] = False
        report["mismatch_count"] = int(bad.size)
        report["first_divergence"] = {
            "field": "cost_scalar_by_agent",
            "agent_id": int(ids[idx]) if idx < int(ids.shape[0]) else None,
            "tick": 0,
            "action": action,
            "component": "env_cost_charge_scalar",
            "production": float(prod[idx]),
            "layout": float(cand[idx]),
            "abs_diff": float(diff[idx]),
        }
    return report


def layout_summary(layout: CostLayout) -> dict[str, Any]:
    return {
        "schema_version": layout.schema_version,
        "field_names": list(COST_SURFACE_COLUMN_NAMES),
        "dtype": str(layout.rows.dtype),
        "shape": [int(v) for v in layout.rows.shape],
        "agent_count": int(layout.agent_ids.shape[0]),
        "charge_channels": bool(layout.charge_channels),
        "charge_radius": bool(layout.charge_radius),
        "ledger_enabled": bool(layout.ledger_enabled),
        "action_code_order": list(layout.action_code_order),
        "omitted_fields": list(layout.omitted_fields),
        "digest": str(layout.digest),
        "array_memory_bytes": int(layout.rows.nbytes + layout.agent_ids.nbytes),
    }


def cost_layout_audit_payload() -> dict[str, Any]:
    fields = [
        {
            "field": "sensor_channel_cost_per_tick",
            "cadence": "per-agent; changes when genome/perception/developmental expression changes",
            "requires_python_object_walking_now": True,
            "numeric_array_representable": True,
            "cacheable": "yes across ticks while genome, age-developmental factor, and config are unchanged",
            "rng_dependency": False,
            "checkpoint_or_metrics_only": False,
            "object_identity_dependency": False,
            "contract": "engine/agents/cognition.py: CostChargeSurface.sensor_channel_cost_per_tick and compute_env_cost_charge_scalar_from_surface",
        },
        {
            "field": "sensor_radius_cost_hypothetical",
            "cadence": "per-agent; changes when genome/perception/developmental expression changes",
            "requires_python_object_walking_now": True,
            "numeric_array_representable": True,
            "cacheable": "yes across ticks while genome, age-developmental factor, and config are unchanged",
            "rng_dependency": False,
            "checkpoint_or_metrics_only": False,
            "object_identity_dependency": False,
            "contract": "engine/agents/cognition.py: CostChargeSurface.sensor_radius_cost_hypothetical and compute_env_cost_charge_scalar_from_surface",
        },
        {
            "field": "prediction_cost_charged",
            "cadence": "per-agent/per-generation or on developmental effective lr/config changes",
            "requires_python_object_walking_now": True,
            "numeric_array_representable": True,
            "cacheable": "yes while prediction flags, effective lr, and config are unchanged",
            "rng_dependency": False,
            "checkpoint_or_metrics_only": False,
            "object_identity_dependency": False,
            "contract": "engine/agents/cognition.py: prediction_cost_components and express_cost_charge_surface",
        },
        {
            "field": "developmental_cost_charged",
            "cadence": "per-agent; age/developmental state can change by tick",
            "requires_python_object_walking_now": True,
            "numeric_array_representable": True,
            "cacheable": "generation/tick-window cache only when age-developmental factor is stable",
            "rng_dependency": False,
            "checkpoint_or_metrics_only": False,
            "object_identity_dependency": False,
            "contract": "engine/agents/cognition.py: developmental_expression_state and developmental_cost_components",
        },
        {
            "field": "maturation_cost_charged",
            "cadence": "per-agent; age/developmental state can change by tick",
            "requires_python_object_walking_now": True,
            "numeric_array_representable": True,
            "cacheable": "generation/tick-window cache only when maturity state is stable",
            "rng_dependency": False,
            "checkpoint_or_metrics_only": False,
            "object_identity_dependency": False,
            "contract": "engine/agents/cognition.py: developmental_expression_state and developmental_cost_components",
        },
        {
            "field": "introspection_cost_charged",
            "cadence": "per-agent/per-generation unless introspection config or genome changes",
            "requires_python_object_walking_now": True,
            "numeric_array_representable": True,
            "cacheable": "yes across ticks for stable genome/config",
            "rng_dependency": False,
            "checkpoint_or_metrics_only": False,
            "object_identity_dependency": False,
            "contract": "engine/agents/cognition.py: express_introspection and express_cost_charge_surface",
        },
        {
            "field": "self_regulation_cost_charged",
            "cadence": "per-agent/per-generation unless self-regulation config or genome changes",
            "requires_python_object_walking_now": True,
            "numeric_array_representable": True,
            "cacheable": "yes across ticks for stable genome/config",
            "rng_dependency": False,
            "checkpoint_or_metrics_only": False,
            "object_identity_dependency": False,
            "contract": "engine/agents/cognition.py: express_self_regulation and express_cost_charge_surface",
        },
        {
            "field": "self_modification_cost_charged",
            "cadence": "per-agent/per-generation unless self-modification config or genome changes",
            "requires_python_object_walking_now": True,
            "numeric_array_representable": True,
            "cacheable": "yes across ticks for stable genome/config",
            "rng_dependency": False,
            "checkpoint_or_metrics_only": False,
            "object_identity_dependency": False,
            "contract": "engine/agents/cognition.py: express_self_modification and express_cost_charge_surface",
        },
    ]
    return {
        "schema_version": "cxxcore4_cost_layout_audit_v1",
        "preliminary_finding": "layout_contract_feasible_with_limited_omissions",
        "represented_fields": fields,
        "runtime_flags": [
            "enable_cognitive_budget_ledger",
            "enable_experimental_perception_cost_charging",
            "enable_experimental_sensor_radius_cost_charging",
        ],
        "stable_action_code_order": list(ACTION_CODE_ORDER),
        "omitted_or_python_authoritative_fields": [
            "Python object/phenotype construction remains the producer for CXXCORE-4",
            "directed_inheritance cost remains visible but excluded from env cost charge scalar",
            "memory upkeep is charged outside this env cost scalar",
            "diagnostic counters, checkpoint serialization, and metrics summaries remain Python-owned",
        ],
        "rng_boundary": "no RNG source is used by scalar evaluation once rows exist",
        "checkpoint_metrics_boundary": "layout rows and scalar totals are diagnostic/shadow data only",
        "production_contract_functions": [
            "engine/agents/cognition.py: CostChargeSurface",
            "engine/agents/cognition.py: express_cost_charge_surface",
            "engine/agents/cognition.py: compute_env_cost_charge_scalar_from_surface",
            "engine/perf/native/cost_charge_surface_kernel.py: COST_SURFACE_COLUMN_NAMES",
            "engine/perf/costsurface_soa.py: costsurface_soa_scalar_from_row",
            "engine/simulation/runner_pkg/runner_ticks_env_apply.py: env_hotpath_cost_charge_surface_enable branch",
        ],
    }


__all__ = [
    "ACTION_CODE_ORDER",
    "BACKEND_SCHEMA_VERSION",
    "SCHEMA_VERSION",
    "CostLayout",
    "CostLayoutBackendInputs",
    "as_cost_layout",
    "as_cost_layout_backend_inputs",
    "build_cost_layout_from_agents",
    "build_cost_layout_from_surfaces",
    "compare_cost_vectors",
    "compare_runner_backend_states",
    "compare_summaries",
    "cost_layout_audit_payload",
    "cost_matrix_from_layout",
    "evaluate_cost_layout_python",
    "layout_summary",
    "run_cost_layout_backend_python_reference",
    "state_summary",
]
