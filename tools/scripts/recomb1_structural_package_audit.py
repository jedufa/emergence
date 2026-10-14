#!/usr/bin/env python3
"""Deterministic CPU audit of C2 structural package coherence.

This probe changes no production rule.  It demonstrates the difference between
the current independent module/connection donor schedule and a shadow
whole-package linkage control.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENGINE_ROOT = PROJECT_ROOT / "engine"
for root in (PROJECT_ROOT, ENGINE_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from agents.structural_identity_v1 import new_native_identity  # noqa: E402
from evolution.recombination_c2 import recombine_lineage_payloads  # noqa: E402
from evolution.structural_development_v1 import (  # noqa: E402
    DEVELOPMENTAL_EXPRESSION_BLOCK_KEY,
    DEVELOPMENTAL_MODULE_SCHEMA_VERSION,
    DEVELOPMENTAL_STAGE_PROFILE_VERSION,
)


class SequenceRng:
    def __init__(self, values: list[float]) -> None:
        self._values = iter(values)
        self.draw_count = 0

    def random(self) -> float:
        self.draw_count += 1
        return float(next(self._values))


def _identity(index: int) -> dict[str, Any]:
    return new_native_identity(
        context={
            "campaign_id": "recomb1-structural-package-audit",
            "branch_id": "inheritance",
            "generation": 0,
            "tick": 0,
            "child_agent_id": index,
            "parent_agent_ids": [1, 2],
            "lineage_token": "recomb1-shared-founder",
        },
        operator_id="fixture",
        attempt_index=index,
        operator_role=f"module-{index}",
    )


def _development(prerequisites: list[str]) -> dict[str, Any]:
    return {
        "schema_version": DEVELOPMENTAL_MODULE_SCHEMA_VERSION,
        "stage_profile_version": DEVELOPMENTAL_STAGE_PROFILE_VERSION,
        "stage_gains": [1.0, 1.0, 1.0],
        "prerequisite_homology_ids": list(prerequisites),
        "latch_dependency_readiness": False,
    }


def _module(index: int, *, marker: float, prerequisites: list[str] | None = None) -> dict[str, Any]:
    identity = _identity(index)
    params: dict[str, Any] = {
        "live_operator": "bounded_scalar_integrator_v1",
        "expression_scale": float(marker),
        "live_retention": 0.5,
        "live_input_gain": 1.0,
        "live_output_gain": 1.0,
    }
    if prerequisites is not None:
        params[DEVELOPMENTAL_EXPRESSION_BLOCK_KEY] = _development(prerequisites)
    return {
        "module_id": f"m{index:03d}",
        "module_type": "generic",
        "params": params,
        "io": {
            "inputs": [{"slot_id": "in", "value_type": "scalar"}],
            "outputs": [{"slot_id": "out", "value_type": "scalar"}],
            "caps": {"max_fanin": 6, "max_fanout": 6},
        },
        **identity,
        "latent_pos": list(identity["topology_pos"]),
    }


def _payload(marker: float, *, cycle_side: str = "") -> dict[str, Any]:
    identities = [_identity(0), _identity(1)]
    homology = [str(item["homology_id"]) for item in identities]
    if cycle_side == "left":
        dependencies = [[homology[1]], []]
    elif cycle_side == "right":
        dependencies = [[], [homology[0]]]
    else:
        dependencies = [None, None]
    modules = [
        _module(index, marker=marker, prerequisites=dependencies[index])
        for index in range(2)
    ]
    connection = {
        "connection_id": "c000",
        "src_module_id": "m000",
        "src_slot": "out",
        "dst_module_id": "m001",
        "dst_slot": "in",
        "value_type": "scalar",
        "channel": "generic",
        "weight": float(marker),
        "enabled": True,
    }
    return {
        "schema_version": "modular_genome_v1",
        "b2_invalid_policy": "reject",
        "genes": {},
        "modules": modules,
        "connections": [connection],
    }


def _package_markers(payload: dict[str, Any]) -> list[float]:
    markers = [float(module["params"]["expression_scale"]) for module in payload["modules"]]
    markers.extend(float(connection["weight"]) for connection in payload["connections"])
    return markers


def _digest(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def build_audit() -> dict[str, Any]:
    parent_positive = _payload(1.0)
    parent_negative = _payload(-1.0)
    mixed_rng = SequenceRng([0.1, 0.9, 0.1])
    mixed, summary = recombine_lineage_payloads(
        parent_a_payload=deepcopy(parent_positive),
        parent_b_payload=deepcopy(parent_negative),
        rng=mixed_rng,
    )
    mixed_markers = _package_markers(mixed)
    current_coherent = len(set(mixed_markers)) == 1

    linked_parent = min(
        (parent_positive, parent_negative),
        key=lambda payload: _digest(payload),
    )
    linked = deepcopy(linked_parent)
    linked_markers = _package_markers(linked)

    hybrid_failure = ""
    hybrid_rng = SequenceRng([0.1, 0.9, 0.1])
    try:
        recombine_lineage_payloads(
            parent_a_payload=_payload(1.0, cycle_side="left"),
            parent_b_payload=_payload(-1.0, cycle_side="right"),
            rng=hybrid_rng,
        )
    except Exception as exc:  # deterministic fail-closed behavior is the evidence
        hybrid_failure = f"{type(exc).__name__}: {exc}"

    checks = {
        "current_c2_consumes_one_draw_per_module_and_connection": mixed_rng.draw_count == 3,
        "current_c2_can_break_parent_coherent_motif": not current_coherent,
        "shadow_linkage_control_preserves_parent_coherent_motif": len(set(linked_markers)) == 1,
        "hybrid_development_conflict_is_rejected": bool(hybrid_failure),
        "hybrid_conflict_consumes_frozen_current_budget_before_rejection": hybrid_rng.draw_count == 3,
        "parent_inputs_remain_unchanged": _package_markers(parent_positive) == [1.0, 1.0, 1.0]
        and _package_markers(parent_negative) == [-1.0, -1.0, -1.0],
    }
    return {
        "schema_version": "recomb1_structural_package_audit_v1",
        "scope": {
            "selection": "none",
            "mutation": "none",
            "ecology": "none",
            "cpu_reference_only": True,
            "production_behavior_changed": False,
            "cuda_or_accelerated_work_started": False,
        },
        "current_c2_fixture": {
            "matched_modules": int(summary.matched_modules),
            "matched_connections": int(summary.matched_connections),
            "draw_count": mixed_rng.draw_count,
            "child_package_markers": mixed_markers,
            "parent_coherent": current_coherent,
        },
        "shadow_linkage_control": {
            "rule": "one canonical-parent donor for the complete matched two-module plus connection package",
            "draws": 1,
            "child_package_markers": linked_markers,
            "parent_coherent": len(set(linked_markers)) == 1,
        },
        "hybrid_development_fixture": {
            "parents_individually_acyclic": True,
            "current_draw_count": hybrid_rng.draw_count,
            "failure": hybrid_failure,
        },
        "checks": checks,
        "all_checks_pass": all(checks.values()),
        "decision_signal": "bounded_linkage_block_required" if all(checks.values()) else "incomplete",
        "claims_boundary": (
            "This mechanics fixture demonstrates package breakup and deterministic rejection only; "
            "it does not establish fitness, adaptation, cognition, or accelerated-lane support."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    payload = build_audit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not payload["all_checks_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
