from __future__ import annotations

"""Bounded CPU-only hereditary reachability probe for ESCAP-2.

This is a mechanics probe.  It has no environment, reward, fitness, or
selection input and makes no claim about functional or ecological value.
"""

from copy import deepcopy
import hashlib
import json
import random
from typing import Any

from agents.genome import Genome
from evolution.operators_c1 import apply_operator_pipeline
from evolution.recombination_c2 import recombine_lineage_payloads, recombine_self_lineage_payload


PROBE_SCHEMA_VERSION = "eiel_escap2_structural_bootstrap_probe_v1"
PREREGISTERED_SEEDS = (72001, 72002, 72003, 72004)
MAX_STOCHASTIC_OFFSPRING_EVENTS = 64


class _SequenceRng:
    def __init__(self, values: list[float]) -> None:
        self._values = iter(values)
        self.draw_count = 0

    def random(self) -> float:
        self.draw_count += 1
        return float(next(self._values))


def _canonical_digest(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _empty() -> dict[str, Any]:
    return {
        "schema_version": "modular_genome_v1",
        "b2_invalid_policy": "reject",
        "genes": {},
        "modules": [],
        "connections": [],
    }


def _context(*, seed: int, child_id: int, generation: int) -> dict[str, Any]:
    return {
        "campaign_id": f"escap2-probe-seed-{int(seed)}",
        "branch_id": "escap2-technical-probe",
        "generation": int(generation),
        "tick": 0,
        "child_agent_id": int(child_id),
        "parent_agent_ids": [max(0, int(child_id) - 2), max(0, int(child_id) - 1)],
    }


def _forced_step(
    payload: dict[str, Any],
    *,
    schedule_draw: float,
    child_id: int,
    generation: int,
) -> tuple[dict[str, Any], dict[str, Any], int]:
    rng = _SequenceRng([float(schedule_draw), 0.1, 0.2, 0.3])
    out, telemetry = apply_operator_pipeline(
        payload=deepcopy(payload),
        rng=rng,
        max_ops=1,
        structural_bootstrap_enable=True,
        structural_identity_context=_context(seed=0, child_id=child_id, generation=generation),
    )
    return out, telemetry, int(rng.draw_count)


def forced_reachability_fixture() -> dict[str, Any]:
    payload = _empty()
    events: list[dict[str, Any]] = []
    schedule = (
        (0.20, "first_module"),
        (0.20, "second_module"),
        (0.05, "first_connection"),
    )
    for generation, (schedule_draw, expected) in enumerate(schedule, start=1):
        before = deepcopy(payload)
        payload, telemetry, draws = _forced_step(
            payload,
            schedule_draw=schedule_draw,
            child_id=100 + generation,
            generation=generation,
        )
        events.append(
            {
                "generation": generation,
                "expected_transition": expected,
                "draws": draws,
                "module_count_before": len(before["modules"]),
                "module_count_after": len(payload["modules"]),
                "connection_count_before": len(before["connections"]),
                "connection_count_after": len(payload["connections"]),
                "operator_reason_counts": telemetry["operator_reason_counts"],
                "blocked_count": telemetry["capacity_blocked_count"],
                "noop_count": telemetry["noop_count"],
            }
        )
    identities = {
        str(module["homology_id"]): {
            "module_instance_id": str(module["module_instance_id"]),
            "topology_pos": list(module["topology_pos"]),
        }
        for module in payload["modules"]
    }
    inherited = deepcopy(payload)
    continuity = True
    for generation in range(4, 7):
        inherited, _ = recombine_self_lineage_payload(
            parent_payload=inherited,
            rng=random.Random(81000 + generation),
        )
        observed = {
            str(module["homology_id"]): {
                "module_instance_id": str(module["module_instance_id"]),
                "topology_pos": list(module["topology_pos"]),
            }
            for module in inherited["modules"]
        }
        continuity = continuity and observed == identities
    return {
        "status": "passed"
        if len(payload["modules"]) == 2
        and len(payload["connections"]) == 1
        and all(event["draws"] == 4 for event in events)
        and continuity
        else "failed",
        "events": events,
        "final_payload_digest": _canonical_digest(payload),
        "module_count": len(payload["modules"]),
        "connection_count": len(payload["connections"]),
        "identity_continuity_generations": 3,
        "identity_continuity_passed": bool(continuity),
        "payload": payload,
    }


def _stochastic_cell(seed: int, *, bootstrap_enabled: bool) -> dict[str, Any]:
    rng = random.Random(int(seed))
    payload = _empty()
    first_module_event = None
    first_additional_module_event = None
    first_connection_event = None
    accepted = 0
    rejected = 0
    blocked = 0
    noops = 0
    reason_counts: dict[str, int] = {}
    for event_index in range(MAX_STOCHASTIC_OFFSPRING_EVENTS):
        before_modules = len(payload["modules"])
        before_connections = len(payload["connections"])
        payload, telemetry = apply_operator_pipeline(
            payload=payload,
            rng=rng,
            max_ops=2,
            structural_bootstrap_enable=bool(bootstrap_enabled),
            structural_identity_context=_context(
                seed=int(seed),
                child_id=1000 + event_index,
                generation=event_index + 1,
            ),
        )
        after_modules = len(payload["modules"])
        after_connections = len(payload["connections"])
        noops += int(telemetry["noop_count"])
        blocked += int(telemetry["capacity_blocked_count"])
        accepted += max(0, int(telemetry["max_ops"]) - int(telemetry["noop_count"]))
        rejected += int(telemetry["capacity_blocked_count"])
        for reason, count in dict(telemetry.get("noop_reason_counts", {})).items():
            reason_counts[str(reason)] = int(reason_counts.get(str(reason), 0)) + int(count)
        if first_module_event is None and before_modules == 0 and after_modules >= 1:
            first_module_event = event_index
        if first_additional_module_event is None and before_modules < 2 <= after_modules:
            first_additional_module_event = event_index
        if first_connection_event is None and before_connections == 0 and after_connections >= 1:
            first_connection_event = event_index
        if first_connection_event is not None and after_modules >= 2:
            break
    return {
        "seed": int(seed),
        "bootstrap_enabled": bool(bootstrap_enabled),
        "events_executed": event_index + 1,
        "first_module_event": first_module_event,
        "first_additional_module_event": first_additional_module_event,
        "first_connection_event": first_connection_event,
        "accepted_attempts": int(accepted),
        "rejected_attempts": int(rejected),
        "blocked_attempts": int(blocked),
        "noop_attempts": int(noops),
        "noop_reason_counts": dict(sorted(reason_counts.items())),
        "module_count": len(payload["modules"]),
        "connection_count": len(payload["connections"]),
        "payload_digest": _canonical_digest(payload),
        "reached": bool(first_connection_event is not None and len(payload["modules"]) >= 2),
    }


def run_structural_bootstrap_technical_probe() -> dict[str, Any]:
    forced = forced_reachability_fixture()
    treatment = [_stochastic_cell(seed, bootstrap_enabled=True) for seed in PREREGISTERED_SEEDS]
    controls = [_stochastic_cell(seed, bootstrap_enabled=False) for seed in PREREGISTERED_SEEDS]
    repeats = [_stochastic_cell(seed, bootstrap_enabled=True) for seed in PREREGISTERED_SEEDS]
    repeat_parity = treatment == repeats
    passed = bool(
        forced["status"] == "passed"
        and all(cell["reached"] for cell in treatment)
        and all(not cell["reached"] for cell in controls)
        and repeat_parity
    )
    return {
        "schema_version": PROBE_SCHEMA_VERSION,
        "status": "passed" if passed else "failed",
        "purpose": "technical hereditary reachability and identity continuity only",
        "preregistered_seeds": list(PREREGISTERED_SEEDS),
        "max_stochastic_offspring_events": MAX_STOCHASTIC_OFFSPRING_EVENTS,
        "forced_fixture": {key: value for key, value in forced.items() if key != "payload"},
        "treatment_cells": treatment,
        "bootstrap_disabled_controls": controls,
        "repeat_parity": bool(repeat_parity),
        "claims": {
            "legal_reachability": passed,
            "ecological_value": "not_tested",
            "selection_benefit": "not_tested",
            "useful_computation": "not_tested",
        },
    }
