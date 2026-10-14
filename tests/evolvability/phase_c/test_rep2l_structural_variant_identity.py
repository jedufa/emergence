from __future__ import annotations

import json

from agents.genome import Genome
from evolution.operators_c1 import apply_operator_pipeline
from evolution.structural_mutation_viability import DEFAULT_DRAWS, seeded_fixture_genome
from evolution.structural_variant_identity import (
    agent_identity,
    birth_identity_context,
    c1_attempt_count,
    full_identity_payload,
    global_event_id,
    graph_identity_digest,
    graph_identity_summary,
    graph_semantic_digest,
    mutation_trace_identity,
    operator_origin_set,
    semantic_identity_payload,
    stable_json,
)
from simulation.runner_pkg.runner_config import SimConfig


def _canonical_digest() -> str:
    return graph_semantic_digest(seeded_fixture_genome())


def _payload_copy() -> dict:
    return json.loads(stable_json(seeded_fixture_genome().canonical_payload()))


def _agent(agent_id: int, genome: Genome) -> object:
    return type("Agent", (), {"agent_id": int(agent_id), "genome": genome, "alive": True})()


class _SequenceRng:
    def __init__(self, values: tuple[float, ...]) -> None:
        self._values = tuple(float(v) for v in values)
        self._idx = 0

    def random(self) -> float:
        value = self._values[self._idx % len(self._values)]
        self._idx += 1
        return float(value)


def test_rep2l_reporting_flag_is_default_off() -> None:
    assert SimConfig().structural_variant_identity_reporting_enable is False
    assert SimConfig().structural_variant_event_join_reporting_enable is False


def test_rep2m_global_event_id_is_run_scoped_and_stable() -> None:
    a = global_event_id(
        run_id="run_a",
        local_event_id="g000000:e00000001",
        event_kind="birth_success",
        generation=0,
        tick=2,
        event_sequence=1,
    )
    assert a == global_event_id(
        run_id="run_a",
        local_event_id="g000000:e00000001",
        event_kind="birth_success",
        generation=0,
        tick=2,
        event_sequence=1,
    )
    assert a != global_event_id(
        run_id="run_b",
        local_event_id="g000000:e00000001",
        event_kind="birth_success",
        generation=0,
        tick=2,
        event_sequence=1,
    )


def test_rep2l_semantic_digest_stable_for_record_order() -> None:
    payload = _payload_copy()
    reordered = json.loads(stable_json(payload))
    reordered["modules"] = list(reversed(reordered["modules"]))
    reordered["connections"] = list(reversed(reordered["connections"]))
    assert graph_semantic_digest(payload) == graph_semantic_digest(reordered)


def test_rep2l_semantic_digest_changes_for_live_fields() -> None:
    payload = _payload_copy()
    changed = json.loads(stable_json(payload))
    changed["modules"][1]["params"]["aggression_bias"] = 1.25
    assert graph_semantic_digest(payload) != graph_semantic_digest(changed)

    changed = json.loads(stable_json(payload))
    changed["connections"][0]["weight"] = 0.5
    assert graph_semantic_digest(payload) != graph_semantic_digest(changed)

    changed = json.loads(stable_json(payload))
    changed["connections"] = []
    assert graph_semantic_digest(payload) != graph_semantic_digest(changed)


def test_rep2l_origin_metadata_changes_full_identity_only() -> None:
    payload = _payload_copy()
    changed = json.loads(stable_json(payload))
    changed["genes"]["c1_operator_applied"] = "gate_drift"
    changed["genes"]["c1_op_count__gate_drift"] = 1
    assert graph_semantic_digest(payload) == graph_semantic_digest(changed)
    assert graph_identity_digest(payload) != graph_identity_digest(changed)
    assert operator_origin_set(changed) == ["mutation:gate_drift"]


def test_rep2l_canonical_seed_identity_and_history_only_variant() -> None:
    canonical = _canonical_digest()
    agent = _agent(1, seeded_fixture_genome())
    ident = agent_identity(agent, canonical_semantic_digest=canonical)
    assert ident["canonical_seed_exact"] is True
    assert ident["variant_class"] == "canonical_seed_exact"

    payload = _payload_copy()
    payload["genes"]["c1_operator_applied"] = "gate_drift"
    payload["genes"]["c1_op_count__gate_drift"] = 1
    hist = agent_identity(_agent(2, Genome.from_dict(payload)), canonical_semantic_digest=canonical)
    assert hist["canonical_seed_semantic_match"] is True
    assert hist["canonical_seed_exact"] is False
    assert hist["variant_class"] == "canonical_seed_history_only_variant"


def test_rep2l_mutation_operator_capture_without_guessing_from_diff() -> None:
    payload = _payload_copy()
    trace_records = []
    out, telemetry = apply_operator_pipeline(
        payload=payload,
        rng=_SequenceRng(DEFAULT_DRAWS[0] * 8),
        max_ops=1,
        payload_is_canonical=True,
        trace_records=trace_records,
        trace_payload_identity_fn=mutation_trace_identity,
    )
    assert telemetry["schedule_ids"]
    assert c1_attempt_count(out) == 1
    origins = operator_origin_set(out)
    assert len(origins) == 1
    assert origins[0].startswith("mutation:")
    assert len(trace_records) == 1
    assert trace_records[0]["operator_id"] == telemetry["schedule_ids"][0]
    assert trace_records[0]["pre_graph_semantic_digest"]
    assert trace_records[0]["normalized_graph_semantic_digest"]
    assert "semantic_changed" in trace_records[0]


def test_rep2l_birth_context_links_parent_child_digests() -> None:
    canonical = _canonical_digest()
    parent = _agent(10, seeded_fixture_genome())
    child_payload = _payload_copy()
    child_payload["modules"][1]["params"]["aggression_bias"] = 1.25
    child_payload["genes"]["c1_operator_applied"] = "gate_drift"
    child_payload["genes"]["c1_op_count__gate_drift"] = 1
    child = _agent(11, Genome.from_dict(child_payload))
    ctx = birth_identity_context(parent, None, child, canonical_semantic_digest=canonical)
    assert ctx["parent_graph_semantic_digest"] == canonical
    assert ctx["child_graph_semantic_digest"] != canonical
    assert ctx["structural_variant_event_kind"] == "mutation"
    assert ctx["child_semantic_graph_changed_vs_parent"] == 1
    assert ctx["child_operator_origin_available"] is True
    assert "mutation:gate_drift" in ctx["child_operator_origin_set"]


def test_rep2m_birth_context_reports_parent_child_boundary_fields() -> None:
    canonical = _canonical_digest()
    parent = _agent(10, seeded_fixture_genome())
    mate = _agent(12, seeded_fixture_genome())
    child_payload = _payload_copy()
    post_c2 = graph_identity_summary(child_payload)
    child_payload["modules"][1]["params"]["aggression_bias"] = 1.25
    final = graph_identity_summary(child_payload)
    child = _agent(11, Genome.from_dict(child_payload))
    setattr(child.genome, "_rep2m_child_post_c2_identity", post_c2)
    setattr(child.genome, "_rep2m_final_child_identity", final)
    setattr(
        child.genome,
        "_rep2m_mutation_delta_events",
        [
            {
                "mutation_attempt_event_id": "attempt-1",
                "operator_status": "applied",
                "historical_changed": 1,
                "semantic_changed": 1,
            }
        ],
    )
    ctx = birth_identity_context(
        parent,
        mate,
        child,
        canonical_semantic_digest=canonical,
        run_id="run_a",
        local_birth_event_id="g000000:e00000001",
        generation=0,
        tick=1,
        event_sequence=2,
    )
    assert ctx["global_birth_event_id"]
    assert ctx["parent_graph_semantic_digest"] == canonical
    assert ctx["mate_graph_semantic_digest"] == canonical
    assert ctx["child_post_recombination_graph_semantic_digest"] == post_c2["graph_semantic_digest"]
    assert ctx["child_final_graph_semantic_digest"] == final["graph_semantic_digest"]
    assert ctx["attached_newborn_matches_final_child"] == 1
    assert ctx["mutation_delta_event_count"] == 1
    assert ctx["mutation_delta_semantic_change_count"] == 1


def test_rep2l_identity_payloads_are_stable_json_objects() -> None:
    payload = _payload_copy()
    assert stable_json(semantic_identity_payload(payload)) == stable_json(semantic_identity_payload(payload))
    assert stable_json(full_identity_payload(payload)) == stable_json(full_identity_payload(payload))
