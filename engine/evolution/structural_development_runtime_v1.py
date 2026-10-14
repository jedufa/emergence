from __future__ import annotations

"""ESCAP-6 deterministic lifetime-expression resolver.

The resolver changes no physical shape and owns no RNG.  Hereditary
dependencies remain keyed by homology_id; the returned plan is keyed by the
stable module_instance_id used by the ESCAP-3 lifetime state maps.
"""

from dataclasses import dataclass
from typing import Any, Iterable

from evolution.structural_development_v1 import (
    DEVELOPMENTAL_EXPRESSION_BLOCK_KEY,
    normalize_agent_developmental_modules,
    normalize_developmental_expression_record,
    resolve_structural_stage,
)


@dataclass(frozen=True)
class DevelopmentRuntimePlan:
    stage_id: str
    stage_index: int
    requested_gain_by_instance_id: dict[str, float]
    dependency_ready_by_instance_id: dict[str, bool]
    effective_gain_by_instance_id: dict[str, float]
    latch_by_instance_id: dict[str, bool]
    prerequisite_instances_by_instance_id: dict[str, tuple[str, ...]]
    topological_instance_ids: tuple[str, ...]
    telemetry: dict[str, Any]


def _identity(module: dict[str, Any], field: str) -> str:
    value = str(module.get(field, "") or "").strip()
    if not value:
        raise RuntimeError(
            f"ESCAP6_MISSING_IDENTITY field={field};module_id={module.get('module_id', '')}"
        )
    return value


def _topological_homology_order(
    homologies: Iterable[str],
    prerequisites_by_homology: dict[str, tuple[str, ...]],
) -> tuple[str, ...]:
    nodes = sorted({str(value) for value in homologies})
    indegree = {node: 0 for node in nodes}
    outgoing: dict[str, list[str]] = {node: [] for node in nodes}
    for target in nodes:
        for source in prerequisites_by_homology.get(target, ()):
            if source not in indegree:
                raise RuntimeError(
                    f"ESCAP6_DEPENDENCY_MISSING_ENDPOINT target={target};prerequisite={source}"
                )
            indegree[target] += 1
            outgoing[source].append(target)
    ready = sorted(node for node, degree in indegree.items() if degree == 0)
    ordered: list[str] = []
    while ready:
        node = ready.pop(0)
        ordered.append(node)
        for target in sorted(outgoing[node]):
            indegree[target] -= 1
            if indegree[target] == 0:
                ready.append(target)
                ready.sort()
    if len(ordered) != len(nodes):
        blocked = sorted(node for node, degree in indegree.items() if degree > 0)
        raise RuntimeError(f"ESCAP6_DEPENDENCY_CYCLE homology_ids={','.join(blocked)}")
    return tuple(ordered)


def resolve_agent_development_runtime(
    agent: Any,
    *,
    sim_cfg: Any,
    dependency_lesion_instance_ids: Iterable[str] = (),
    expression_lesion_instance_ids: Iterable[str] = (),
    latch_lesion_instance_ids: Iterable[str] = (),
) -> DevelopmentRuntimePlan:
    """Resolve one atomic, JSON-ready developmental plan for the current tick."""

    genome = getattr(agent, "genome", None)
    modules = normalize_agent_developmental_modules(
        list(getattr(genome, "modules", []) or [])
    )
    generic = [
        dict(module)
        for module in modules
        if str(module.get("module_type", "")) == "generic"
    ]
    age_steps = int(
        getattr(getattr(agent, "stats", None), "survival_steps", 0) or 0
    )
    stage_id, stage_index = resolve_structural_stage(
        age_steps=age_steps,
        sim_cfg=sim_cfg,
    )

    by_homology: dict[str, dict[str, Any]] = {}
    instance_by_homology: dict[str, str] = {}
    record_by_homology: dict[str, dict[str, Any] | None] = {}
    prerequisites_by_homology: dict[str, tuple[str, ...]] = {}
    for module in generic:
        homology = _identity(module, "homology_id")
        instance = _identity(module, "module_instance_id")
        if homology in by_homology:
            raise RuntimeError(f"ESCAP6_AMBIGUOUS_HOMOLOGY homology_id={homology}")
        by_homology[homology] = module
        instance_by_homology[homology] = instance
        params = module.get("params", {})
        if not isinstance(params, dict):
            raise RuntimeError(
                f"ESCAP6_MODULE_PARAMS_MALFORMED module_id={module.get('module_id', '')}"
            )
        raw = params.get(DEVELOPMENTAL_EXPRESSION_BLOCK_KEY)
        record = (
            None
            if DEVELOPMENTAL_EXPRESSION_BLOCK_KEY not in params
            else normalize_developmental_expression_record(raw)
        )
        record_by_homology[homology] = record
        prerequisites_by_homology[homology] = tuple(
            str(value)
            for value in (record or {}).get("prerequisite_homology_ids", [])
        )

    order = _topological_homology_order(
        by_homology,
        prerequisites_by_homology,
    )
    dependency_lesions = {str(value) for value in dependency_lesion_instance_ids}
    expression_lesions = {str(value) for value in expression_lesion_instance_ids}
    latch_lesions = {str(value) for value in latch_lesion_instance_ids}
    valid_instances = set(instance_by_homology.values())
    old_latches = {
        str(key): bool(value)
        for key, value in dict(
            getattr(agent, "structural_development_latch_by_instance_id", {}) or {}
        ).items()
        if str(key) in valid_instances and bool(value)
    }
    next_latches = dict(sorted(old_latches.items()))

    requested: dict[str, float] = {}
    dependency_ready: dict[str, bool] = {}
    effective: dict[str, float] = {}
    prerequisite_instances: dict[str, tuple[str, ...]] = {}
    rows: list[dict[str, Any]] = []
    for homology in order:
        instance = instance_by_homology[homology]
        record = record_by_homology[homology]
        gains = [1.0, 1.0, 1.0] if record is None else list(record["stage_gains"])
        requested_gain = float(gains[int(stage_index)])
        if instance in expression_lesions:
            requested_gain = 0.0
        prerequisite_homologies = prerequisites_by_homology[homology]
        resolved_prerequisites = tuple(
            instance_by_homology[value] for value in prerequisite_homologies
        )
        prerequisite_instances[instance] = resolved_prerequisites
        raw_ready = all(
            float(effective.get(prerequisite_instance, 0.0)) > 0.0
            for prerequisite_instance in resolved_prerequisites
        )
        dependency_lesioned = instance in dependency_lesions
        latch_lesioned = instance in latch_lesions
        latch_enabled = bool(
            record is not None
            and record.get("latch_dependency_readiness", False)
        )
        previously_latched = bool(next_latches.get(instance, False)) and not latch_lesioned
        gate_ready = bool(raw_ready or previously_latched or dependency_lesioned)
        if latch_enabled and raw_ready and not latch_lesioned and not dependency_lesioned:
            next_latches[instance] = True
            gate_ready = True
        if not latch_enabled:
            next_latches.pop(instance, None)
        final_gain = requested_gain if gate_ready else 0.0
        requested[instance] = requested_gain
        dependency_ready[instance] = gate_ready
        effective[instance] = final_gain
        rows.append(
            {
                "module_id": str(by_homology[homology].get("module_id", "")),
                "homology_id": homology,
                "module_instance_id": instance,
                "stage_id": stage_id,
                "requested_expression_gain": requested_gain,
                "prerequisite_homology_ids": list(prerequisite_homologies),
                "prerequisite_module_instance_ids": list(resolved_prerequisites),
                "raw_dependency_ready": bool(raw_ready),
                "dependency_lesioned": bool(dependency_lesioned),
                "dependency_gate_latched": bool(next_latches.get(instance, False)),
                "latch_lesioned": bool(latch_lesioned),
                "final_effective_expression_gain": final_gain,
            }
        )

    topological_instances = tuple(instance_by_homology[value] for value in order)
    transition_count = int(
        sum(
            1
            for row in rows
            if float(row["requested_expression_gain"]) != 1.0
        )
    )
    telemetry = {
        "schema_version": "escap6_development_runtime_v1",
        "age_steps": age_steps,
        "stage_id": stage_id,
        "stage_index": int(stage_index),
        "allocated_developmental_records": int(
            sum(record is not None for record in record_by_homology.values())
        ),
        "dependency_edges": int(
            sum(len(values) for values in prerequisites_by_homology.values())
        ),
        "dependency_ready_modules": int(sum(bool(value) for value in dependency_ready.values())),
        "dependency_blocked_modules": int(sum(not bool(value) for value in dependency_ready.values())),
        "expressed_modules": int(sum(float(value) > 0.0 for value in effective.values())),
        "inactive_modules": int(sum(float(value) == 0.0 for value in effective.values())),
        "irreversible_commitments": int(sum(bool(value) for value in next_latches.values())),
        "expression_transitions": transition_count,
        "topological_homology_order": list(order),
        "topological_module_instance_order": list(topological_instances),
        "module_rows": rows,
        "random_draws": 0,
        "reporting_only": True,
    }
    return DevelopmentRuntimePlan(
        stage_id=stage_id,
        stage_index=int(stage_index),
        requested_gain_by_instance_id=dict(sorted(requested.items())),
        dependency_ready_by_instance_id=dict(sorted(dependency_ready.items())),
        effective_gain_by_instance_id=dict(sorted(effective.items())),
        latch_by_instance_id=dict(sorted(next_latches.items())),
        prerequisite_instances_by_instance_id=dict(sorted(prerequisite_instances.items())),
        topological_instance_ids=topological_instances,
        telemetry=telemetry,
    )
