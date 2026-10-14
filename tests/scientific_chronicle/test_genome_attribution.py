from __future__ import annotations

import json
from pathlib import Path
import random
import sys

import pytest
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[5]
ENGINE_ROOT = REPO_ROOT / "experiments" / "projects" / "emergence" / "engine"
while str(ENGINE_ROOT) in sys.path:
    sys.path.remove(str(ENGINE_ROOT))
sys.path.insert(0, str(ENGINE_ROOT))

from agents.genome import Genome
from simulation.chronicle.genome_attribution import (
    DEFAULT_MATCH_FIELDS,
    GenomeAttributionAccumulator,
    GenomeAttributionError,
    make_success_profile,
    storage_preflight,
    validate_evidence_strength,
    write_attribution_outputs,
)


def _genome(exploration: float, aggression: float = 0.0) -> dict:
    return Genome(
        hidden_size=8,
        exploration_bias=exploration,
        aggression_bias=aggression,
        reproduction_threshold=20.0,
        mutation_rate=0.05,
        action_temperature=1.0,
        use_gru=False,
        memory_dim=0,
        drive_inertia=0.85,
        drive_noise=0.02,
        drive_logit_gain=1.0,
        pred_lr=0.08,
        developmental_program={},
    ).to_dict()


def _source(
    *,
    generation: int,
    parent_id: int,
    child_id: int,
    parent: dict,
    pre: dict,
    post: dict,
    ref: str,
    lineage: str = "L1",
) -> dict:
    return {
        "run_id": "run-a",
        "campaign_id": "campaign-a",
        "branch_id": "branch-a",
        "generation": generation,
        "step": 9,
        "reproduction_phase": "birth_commit",
        "parent_agent_ids": [parent_id],
        "child_agent_id": child_id,
        "lineage_id": lineage,
        "parent_genomes": [parent],
        "child_pre_mutation_genome": pre,
        "child_post_mutation_genome": post,
        "final_attached_child_genome": post,
        "mutation_operator": "scalar_gaussian",
        "source_completeness": "full_canonical_genome_snapshots",
        "source_record_refs": [ref],
        "chronicle_event_id": f"evt_{child_id}",
        "birth_chronicle_event_id": f"evt_birth_{child_id}",
    }


def test_mutation_ids_delta_order_and_hash_reconciliation_are_deterministic() -> None:
    source = _source(
        generation=1,
        parent_id=1,
        child_id=2,
        parent=_genome(0.0),
        pre=_genome(0.1),
        post=_genome(0.2, 0.1),
        ref="births.jsonl:1",
    )
    first = GenomeAttributionAccumulator()
    second = GenomeAttributionAccumulator()
    first.add(source)
    second.add(json.loads(json.dumps(source)))
    result_a = first.build()
    result_b = second.build()
    assert result_a == result_b
    assert [row["changed_path"] for row in result_a["mutation_events"]] == [
        "/genes/aggression_bias",
        "/genes/exploration_bias",
    ]
    assert all(row["mutation_event_id"].startswith("mut_") for row in result_a["mutation_events"])
    assert all(row["genome_delta_id"].startswith("gdel_") for row in result_a["genome_deltas"])
    assert all(row["hash_verification"]["reconstructable_from_source_payloads"] for row in result_a["genome_deltas"])


def test_hash_mismatch_and_missing_source_identity_fail_closed() -> None:
    source = _source(
        generation=1,
        parent_id=1,
        child_id=2,
        parent=_genome(0.0),
        pre=_genome(0.1),
        post=_genome(0.2),
        ref="births.jsonl:1",
    )
    source["child_post_mutation_genome_hash"] = "0" * 64
    with pytest.raises(GenomeAttributionError, match="does not reconcile"):
        GenomeAttributionAccumulator().add(source)
    source.pop("child_post_mutation_genome_hash")
    source["source_record_refs"] = []
    with pytest.raises(GenomeAttributionError, match="source_record_refs"):
        GenomeAttributionAccumulator().add(source)


def test_inheritance_persistence_and_independent_similar_mutations_remain_distinct() -> None:
    first_source = _source(
        generation=1,
        parent_id=1,
        child_id=2,
        parent=_genome(0.0),
        pre=_genome(0.1),
        post=_genome(0.2),
        ref="births.jsonl:1",
    )
    descendant = _source(
        generation=2,
        parent_id=2,
        child_id=3,
        parent=_genome(0.2),
        pre=_genome(0.2),
        post=_genome(0.2, 0.1),
        ref="births.jsonl:2",
    )
    independent = _source(
        generation=3,
        parent_id=10,
        child_id=11,
        parent=_genome(0.0),
        pre=_genome(0.1),
        post=_genome(0.2),
        ref="births.jsonl:3",
        lineage="L2",
    )
    accumulator = GenomeAttributionAccumulator()
    for source in (first_source, descendant, independent):
        accumulator.add(source)
    result = accumulator.build()
    inherited = [row for row in result["mutation_inheritance"] if row["status"] == "inherited"]
    assert any(row["carrier_agent_id"] == 3 and row["path"] == "/genes/exploration_bias" for row in inherited)
    persistence = result["mutation_persistence_summary"]["mutations"]
    assert any(row["persisted_generations"] == 1 and row["evidence_strength"] == "persisted" for row in persistence)
    similar = [
        row for row in result["mutation_events"]
        if row["changed_path"] == "/genes/exploration_bias"
    ]
    assert len(similar) == 2
    assert len({row["mutation_event_id"] for row in similar}) == 2
    assert all(row["mutation_identity_relation"] == "independent_similar_mutation" for row in similar)


def _outcome(agent_id: int, value: float) -> dict:
    row = {
        "agent_id": agent_id,
        "outcomes": {"lifespan": value},
        "generation": 1,
        "lineage_id": "L1",
        "parent_or_sibling_group": "siblings-1",
        "age": 0,
        "scenario": "scenario-a",
        "seed": 11,
        "region_id": "global",
        "opportunity_exposure": "same-opportunity-bin",
        "starting_state": "same-start-bin",
    }
    assert set(DEFAULT_MATCH_FIELDS).issubset(row)
    return row


def test_matched_association_is_available_only_with_complete_controls() -> None:
    accumulator = GenomeAttributionAccumulator()
    accumulator.add(_source(
        generation=1,
        parent_id=1,
        child_id=2,
        parent=_genome(0.0),
        pre=_genome(0.1),
        post=_genome(0.2),
        ref="births.jsonl:1",
    ))
    unavailable = accumulator.build(outcomes=[{"agent_id": 2, "outcomes": {"lifespan": 12}}])
    assert unavailable["mutation_outcome_associations"][0]["status"] == "unavailable"
    available = accumulator.build(outcomes=[_outcome(2, 12), _outcome(9, 8)])
    association = available["mutation_outcome_associations"][0]
    assert association["status"] == "available"
    assert association["difference_carrier_minus_control"] == 4.0
    assert association["causal_status"] == "association_only"


def test_evidence_ladder_success_profile_storage_and_resume(tmp_path: Path) -> None:
    validate_evidence_strength({"evidence_strength": "observed"})
    validate_evidence_strength({
        "evidence_strength": "inherited",
        "inheritance_record_id": "inh_1",
    })
    with pytest.raises(GenomeAttributionError, match="branch comparison"):
        validate_evidence_strength({"evidence_strength": "branch_supported"})
    with pytest.raises(GenomeAttributionError, match="unsupported causal"):
        make_success_profile(
            subject_type="agent",
            subject_id="2",
            run_id="run-a",
            measured_outcomes={"summary": "Mutation caused success"},
        )
    profile = make_success_profile(
        subject_type="agent",
        subject_id="2",
        run_id="run-a",
        measured_outcomes={"surviving_offspring": 8},
        mutation_event_ids=["mut_a"],
        limitations=["No single causal explanation was established."],
    )
    assert profile["causal_status"] == "not_assessed"
    with pytest.raises(GenomeAttributionError, match="acknowledge_large_output"):
        storage_preflight(source_record_count=1, estimated_delta_count=1, include_full_genomes=True)

    source = _source(
        generation=1,
        parent_id=1,
        child_id=2,
        parent=_genome(0.0),
        pre=_genome(0.1),
        post=_genome(0.2),
        ref="births.jsonl:1",
    )
    uninterrupted = GenomeAttributionAccumulator()
    uninterrupted.add(source)
    python_rng = random.getstate()
    numpy_rng = json.dumps(np.random.get_state()[1].tolist())
    torch_rng = torch.random.get_rng_state().clone()
    uninterrupted.build()
    assert random.getstate() == python_rng
    assert json.dumps(np.random.get_state()[1].tolist()) == numpy_rng
    assert torch.equal(torch.random.get_rng_state(), torch_rng)
    resumed = GenomeAttributionAccumulator()
    checkpoint_state = json.loads(json.dumps(uninterrupted.export_state()))
    resumed.import_state(checkpoint_state)
    assert resumed.build() == uninterrupted.build()
    resumed.add(source)
    assert len(resumed.source_records) == 1
    checkpoint_state["source_records"][0]["generation"] = 99
    with pytest.raises(GenomeAttributionError, match="identity mismatch"):
        GenomeAttributionAccumulator().import_state(checkpoint_state)

    output = tmp_path / "attribution"
    result = write_attribution_outputs(resumed, output)
    assert set(result["files"]) == {
        "genome_deltas.jsonl",
        "mutation_attribution_manifest.json",
        "mutation_attribution_report.md",
        "mutation_carrier_summary.json",
        "mutation_events.jsonl",
        "mutation_inheritance.jsonl",
        "mutation_outcome_associations.csv",
        "mutation_persistence_summary.json",
    }
    manifest = json.loads((output / "mutation_attribution_manifest.json").read_text(encoding="utf-8"))
    assert manifest["include_full_genomes"] is False
    assert manifest["evidence_strength_ladder"][-1] == "experimentally_supported"
