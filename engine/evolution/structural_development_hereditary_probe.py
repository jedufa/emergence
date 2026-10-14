from __future__ import annotations

"""Deterministic ESCAP-6 hereditary-mechanics probe.

The probe exercises only the existing fixed-budget C1 developmental operator
and C2 homology recombination.  Developmental records may change, but physical
module identities, interfaces, connections, and tensor-relevant shapes do not.
"""

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from agents.modular_genome_v1 import canonical_payload_from_genome_dict
from agents.structural_identity_v1 import new_native_identity
from evolution.operators_c1 import apply_development_operator_attempt
from evolution.recombination_c2 import recombine_lineage_payloads
from evolution.structural_development_v1 import (
    DEVELOPMENTAL_EXPRESSION_BLOCK_KEY,
    DEVELOPMENTAL_STAGE_IDS,
    neutral_developmental_expression_record,
    normalize_agent_developmental_modules,
    resolve_developmental_expression,
    validate_agent_developmental_graph,
)


PROBE_SCHEMA_VERSION = "escap6_hereditary_mechanics_probe_v1"


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


class _CountingRng:
    def __init__(self, values: tuple[float, ...]) -> None:
        self._values = iter(values)
        self.count = 0

    def random(self) -> float:
        self.count += 1
        return float(next(self._values))


def _identity(index: int) -> dict[str, Any]:
    return new_native_identity(
        context={
            "campaign_id": "escap6-hereditary-mechanics-probe",
            "branch_id": "bounded-development",
            "generation": 1,
            "tick": 0,
            "child_agent_id": int(index),
            "parent_agent_ids": [101, 202],
            "lineage_token": "escap6-hereditary-probe-lineage",
        },
        operator_id="escap6_hereditary_probe_fixture",
        attempt_index=int(index),
        operator_role=f"module-{int(index)}",
    )


def _module(index: int) -> dict[str, Any]:
    identity = _identity(index)
    return {
        "module_id": f"m{int(index):03d}",
        "module_type": "generic",
        "params": {
            "live_operator": "bounded_scalar_integrator_v1",
            "expression_scale": 1.0,
            "live_retention": 0.5,
            "live_input_gain": 1.0,
            "live_output_gain": 1.0,
        },
        "io": {
            "inputs": [{"slot_id": "in", "value_type": "scalar"}],
            "outputs": [{"slot_id": "out", "value_type": "scalar"}],
            "caps": {"max_fanin": 6, "max_fanout": 6},
        },
        **identity,
        "latent_pos": list(identity["topology_pos"]),
    }


def _connection(source: Mapping[str, Any], target: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "connection_id": "c000",
        "src_module_id": str(source["module_id"]),
        "src_slot": "out",
        "dst_module_id": str(target["module_id"]),
        "dst_slot": "in",
        "value_type": "scalar",
        "weight": 0.5,
        "enabled": True,
    }


def _payload(
    modules: list[dict[str, Any]],
    connections: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": "modular_genome_v1",
        "b2_invalid_policy": "reject",
        "genes": {},
        "modules": deepcopy(modules),
        "connections": deepcopy(connections),
    }


def _module_sort_key(module: Mapping[str, Any]) -> tuple[str, str, str]:
    return (
        str(module.get("homology_id", "")),
        str(module.get("module_instance_id", "")),
        str(module.get("module_id", "")),
    )


def _records_by_homology(modules: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for module in sorted(modules, key=_module_sort_key):
        params = module.get("params", {})
        record = (
            params.get(DEVELOPMENTAL_EXPRESSION_BLOCK_KEY)
            if isinstance(params, Mapping)
            else None
        )
        out[str(module.get("homology_id", ""))] = deepcopy(record)
    return out


def _physical_projection(
    modules: list[dict[str, Any]],
    connections: list[dict[str, Any]],
) -> dict[str, Any]:
    local_to_homology = {
        str(module.get("module_id", "")): str(module.get("homology_id", ""))
        for module in modules
    }
    module_rows: list[dict[str, Any]] = []
    for module in sorted(modules, key=_module_sort_key):
        params = deepcopy(module.get("params", {}))
        if not isinstance(params, dict):
            raise ValueError("ESCAP6_HEREDITARY_PROBE_PARAMS_NOT_MAPPING")
        params.pop(DEVELOPMENTAL_EXPRESSION_BLOCK_KEY, None)
        module_rows.append(
            {
                "homology_id": str(module.get("homology_id", "")),
                "module_instance_id": str(module.get("module_instance_id", "")),
                "module_type": str(module.get("module_type", "")),
                "identity_origin": str(module.get("identity_origin", "")),
                "topology_pos": deepcopy(module.get("topology_pos", [])),
                "latent_pos": deepcopy(module.get("latent_pos", [])),
                "io": deepcopy(module.get("io", {})),
                "nondevelopmental_params": params,
            }
        )
    connection_rows = sorted(
        (
            {
                "src_homology_id": local_to_homology.get(
                    str(connection.get("src_module_id", "")), ""
                ),
                "src_slot": str(connection.get("src_slot", "")),
                "dst_homology_id": local_to_homology.get(
                    str(connection.get("dst_module_id", "")), ""
                ),
                "dst_slot": str(connection.get("dst_slot", "")),
                "value_type": str(connection.get("value_type", "")),
                "weight": float(connection.get("weight", 0.0)),
                "enabled": bool(connection.get("enabled", True)),
            }
            for connection in connections
        ),
        key=lambda row: (
            row["src_homology_id"],
            row["dst_homology_id"],
            row["src_slot"],
            row["dst_slot"],
        ),
    )
    return {
        "module_count": len(module_rows),
        "connection_count": len(connection_rows),
        "modules": module_rows,
        "connections": connection_rows,
    }


def _operator_attempt(
    modules: list[dict[str, Any]],
    *,
    draws: tuple[float, float],
    enabled: bool,
) -> tuple[Any, int]:
    rng = _CountingRng(draws)
    result = apply_development_operator_attempt(
        deepcopy(modules),
        rng=rng,
        enabled=bool(enabled),
    )
    return result, int(rng.count)


def _run_sequence() -> dict[str, Any]:
    base_modules = sorted(
        normalize_agent_developmental_modules([_module(1), _module(2)]),
        key=_module_sort_key,
    )
    base_connections = [_connection(base_modules[0], base_modules[1])]
    physical_before = _physical_projection(base_modules, base_connections)

    neutral, neutral_draws = _operator_attempt(
        base_modules,
        draws=(0.0, 0.0),
        enabled=True,
    )
    neutral_records = _records_by_homology(neutral.modules)
    allocated_records = [value for value in neutral_records.values() if value is not None]

    stage_gain, stage_gain_draws = _operator_attempt(
        neutral.modules,
        draws=(0.25, 0.2),
        enabled=True,
    )
    stage_gain_records = _records_by_homology(stage_gain.modules)

    dependency, dependency_draws = _operator_attempt(
        stage_gain.modules,
        draws=(0.5, 0.5),
        enabled=True,
    )
    validate_agent_developmental_graph(dependency.modules)
    dependency_records = _records_by_homology(dependency.modules)
    dependency_edges = [
        (owner, str(target))
        for owner, record in sorted(dependency_records.items())
        if isinstance(record, Mapping)
        for target in list(record.get("prerequisite_homology_ids", []))
    ]

    canonical_once = sorted(
        normalize_agent_developmental_modules(dependency.modules),
        key=_module_sort_key,
    )
    canonical_twice = sorted(
        normalize_agent_developmental_modules(canonical_once),
        key=_module_sort_key,
    )
    payload_once = canonical_payload_from_genome_dict(
        _payload(canonical_once, base_connections)
    )
    payload_twice = canonical_payload_from_genome_dict(deepcopy(payload_once))
    canonical_parent_modules = list(payload_once.get("modules", []) or [])
    canonical_parent_connections = list(payload_once.get("connections", []) or [])
    canonical_parent_records = _records_by_homology(canonical_parent_modules)
    canonical_parent_physical = _physical_projection(
        canonical_parent_modules,
        canonical_parent_connections,
    )

    c2_rng = _CountingRng(tuple(0.125 for _ in range(16)))
    child, c2_summary = recombine_lineage_payloads(
        parent_a_payload=deepcopy(payload_once),
        parent_b_payload=deepcopy(payload_once),
        rng=c2_rng,
    )
    child_modules = list(child.get("modules", []) or [])
    child_connections = list(child.get("connections", []) or [])
    child_records = _records_by_homology(child_modules)

    disabled, disabled_draws = _operator_attempt(
        base_modules,
        draws=(0.0, 0.0),
        enabled=False,
    )

    resolution = resolve_developmental_expression(
        canonical_once,
        stage_id="early",
    )
    physical_after_c1 = _physical_projection(canonical_once, base_connections)
    physical_after_c2 = _physical_projection(child_modules, child_connections)
    attempt_rows = [
        {
            "name": "neutral_record_creation",
            "supplied_draws": [0.0, 0.0],
            "consumed_draws": neutral_draws,
            "telemetry": deepcopy(neutral.telemetry),
        },
        {
            "name": "later_stage_gain_mutation",
            "supplied_draws": [0.25, 0.2],
            "consumed_draws": stage_gain_draws,
            "telemetry": deepcopy(stage_gain.telemetry),
        },
        {
            "name": "dependency_acquisition",
            "supplied_draws": [0.5, 0.5],
            "consumed_draws": dependency_draws,
            "telemetry": deepcopy(dependency.telemetry),
        },
        {
            "name": "disabled_control",
            "supplied_draws": [0.0, 0.0],
            "consumed_draws": disabled_draws,
            "telemetry": deepcopy(disabled.telemetry),
        },
    ]

    checks = {
        "neutral_record_created": bool(
            neutral.telemetry.get("status") == "accepted"
            and neutral.telemetry.get("operation") == "add_neutral_record"
            and allocated_records == [neutral_developmental_expression_record()]
        ),
        "later_legal_stage_gain_mutation": bool(
            stage_gain.telemetry.get("status") == "accepted"
            and stage_gain.telemetry.get("operation") == "mutate_stage_gain"
            and stage_gain.telemetry.get("selected_stage_id") in DEVELOPMENTAL_STAGE_IDS
            and 0.0 <= float(stage_gain.telemetry.get("gain_after")) <= 1.0
            and stage_gain.telemetry.get("gain_before")
            != stage_gain.telemetry.get("gain_after")
        ),
        "one_valid_dependency_acquired": bool(
            dependency.telemetry.get("status") == "accepted"
            and dependency.telemetry.get("operation") == "add_or_remove_dependency"
            and dependency.telemetry.get("dependency_detail") == "add"
            and len(dependency_edges) == 1
        ),
        "canonical_persistence": bool(
            _canonical_bytes(canonical_once) == _canonical_bytes(canonical_twice)
            and _canonical_bytes(payload_once) == _canonical_bytes(payload_twice)
        ),
        "c2_preserves_developmental_records": bool(
            child_records == canonical_parent_records
            and c2_summary.budget_draws == c2_rng.count
        ),
        "disabled_control_parity": bool(
            disabled.telemetry.get("status") == "disabled"
            and _canonical_bytes(disabled.modules) == _canonical_bytes(base_modules)
        ),
        "exact_two_supplied_draws_per_attempt": all(
            int(row["consumed_draws"]) == 2
            and row["telemetry"].get("draw_budget")
            == {"declared_supplied_draws": 2, "internal_rng_draws": 0}
            for row in attempt_rows
        ),
        "no_internal_execution_rng": bool(
            all(
                int(row["telemetry"]["draw_budget"]["internal_rng_draws"]) == 0
                for row in attempt_rows
            )
        ),
        "physical_architecture_and_shapes_unchanged": bool(
            physical_before == physical_after_c1
            and canonical_parent_physical == physical_after_c2
            and physical_before["module_count"] == 2
            and physical_before["connection_count"] == 1
            and canonical_parent_physical["module_count"] == 2
            and canonical_parent_physical["connection_count"] == 1
        ),
        "dependency_resolves_canonically": bool(
            len(resolution.topological_homology_ids) == 2
            and len(resolution.rows_by_module_instance_id) == 2
        ),
    }
    return {
        "attempts": attempt_rows,
        "developmental_records": {
            "after_neutral_creation": neutral_records,
            "after_stage_gain_mutation": stage_gain_records,
            "after_dependency_acquisition": dependency_records,
            "after_c2": child_records,
        },
        "dependency_edges": [list(edge) for edge in dependency_edges],
        "canonical": {
            "modules_sha256": _sha256(canonical_once),
            "payload_sha256": _sha256(payload_once),
        },
        "c2": {
            "budget_draws": int(c2_summary.budget_draws),
            "supplied_draws_consumed": int(c2_rng.count),
            "matched_modules": int(c2_summary.matched_modules),
            "matched_connections": int(c2_summary.matched_connections),
        },
        "resolution": {
            "stage_id": resolution.stage_id,
            "topological_homology_ids": list(resolution.topological_homology_ids),
            "rows_by_module_instance_id": deepcopy(
                resolution.rows_by_module_instance_id
            ),
        },
        "physical_projection_sha256": _sha256(physical_before),
        "checks": checks,
    }


def run_hereditary_probe() -> dict[str, Any]:
    first = _run_sequence()
    second = _run_sequence()
    repeat_equal = _canonical_bytes(first) == _canonical_bytes(second)
    checks = dict(first["checks"])
    checks["repeat_byte_parity"] = bool(repeat_equal)
    payload: dict[str, Any] = {
        "schema_version": PROBE_SCHEMA_VERSION,
        "purpose": "bounded hereditary developmental mechanics only",
        "sequence": first,
        "checks": checks,
        "repeat_sequence_sha256": [_sha256(first), _sha256(second)],
        "interpretation_boundary": (
            "technical hereditary mechanics only; no tuning, ecology, "
            "adaptiveness, cognition, emergence, intelligence, fitness, or "
            "selection claim"
        ),
    }
    payload["probe_hash"] = _sha256(payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    payload = run_hereditary_probe()
    text = json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n"
    if args.output is None:
        print(text, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    return 0 if all(bool(value) for value in payload["checks"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
