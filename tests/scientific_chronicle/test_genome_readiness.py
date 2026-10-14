from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import random
import shutil
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[5]
ENGINE_ROOT = REPO_ROOT / "experiments" / "projects" / "emergence" / "engine"
for required in (str(REPO_ROOT), str(ENGINE_ROOT)):
    while required in sys.path:
        sys.path.remove(required)
sys.path.insert(0, str(ENGINE_ROOT))
sys.path.insert(1, str(REPO_ROOT))

from agents.genome import Genome
from infra.webui.server_chronicle import discover_runs, event_detail, subject_detail
from simulation.chronicle.genome_attribution import GenomeAttributionAccumulator, write_attribution_outputs
from simulation.chronicle.genome_readiness import (
    GenomeReadinessError,
    MATCH_HIERARCHY,
    OUTPUT_NAMES,
    validate_no_causal_wording,
    write_readiness_outputs,
)
from simulation.chronicle.recorder import ScientificChronicleRecorder


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
    chronicle_event_id: str | None = None,
) -> dict:
    return {
        "run_id": "run-a",
        "campaign_id": "campaign-a",
        "branch_id": "branch-a",
        "source_commit": "test-commit",
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
        "chronicle_event_id": chronicle_event_id or f"evt_{child_id}",
        "birth_chronicle_event_id": f"evt_birth_{child_id}",
    }


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
    row.update({field: "same-r1-stratum" for field in MATCH_HIERARCHY})
    return row


def _f0_fixture(root: Path, *, chronicle_event_id: str | None = None) -> Path:
    sources = (
        _source(
            generation=1,
            parent_id=1,
            child_id=2,
            parent=_genome(0.0),
            pre=_genome(0.1),
            post=_genome(0.2),
            ref="births.jsonl:1",
            chronicle_event_id=chronicle_event_id,
        ),
        _source(
            generation=2,
            parent_id=2,
            child_id=3,
            parent=_genome(0.2),
            pre=_genome(0.2),
            post=_genome(0.2, 0.1),
            ref="births.jsonl:2",
        ),
        _source(
            generation=3,
            parent_id=10,
            child_id=11,
            parent=_genome(0.0),
            pre=_genome(0.1),
            post=_genome(0.2),
            ref="births.jsonl:3",
            lineage="L2",
        ),
    )
    accumulator = GenomeAttributionAccumulator()
    for source in sources:
        accumulator.add(source)
    write_attribution_outputs(
        accumulator,
        root,
        outcomes=[_outcome(2, 12), _outcome(9, 8)],
        required_match_fields=MATCH_HIERARCHY,
    )
    return root


def _policy_row(*, trace: str = "trace-1", child: int = 5) -> dict:
    return {
        "schema_version": "eiel_policy_lineage_event_v1",
        "seed": 101,
        "event_id": 1,
        "mutation_trace_id": trace,
        "mutation_mutated": True,
        "mutation_mutated_parameter_count": 4,
        "mutation_parameter_count": 10,
        "full_policy_tensor_delta_norm": 0.5,
        "generation": 1,
        "biological_parent_ids": [1, 2],
        "child_agent_id": child,
        "child_biological_lineage_id": "L1",
        "parent_policy_hash": "1" * 64,
        "child_pre_mutation_policy_hash": "2" * 64,
        "child_post_mutation_policy_hash": "3" * 64,
        "child_final_policy_hash": "3" * 64,
        "policy_mutation_mode": "sparse",
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _tree_bytes(root: Path) -> dict[str, bytes]:
    return {path.relative_to(root).as_posix(): path.read_bytes() for path in sorted(root.iterdir()) if path.is_file()}


def _rng_state() -> tuple[object, tuple, torch.Tensor]:
    numpy_state = np.random.get_state()
    copied_numpy = (numpy_state[0], numpy_state[1].copy(), numpy_state[2], numpy_state[3], numpy_state[4])
    return random.getstate(), copied_numpy, torch.random.get_rng_state().clone()


def _assert_rng_equal(before: tuple[object, tuple, torch.Tensor], after: tuple[object, tuple, torch.Tensor]) -> None:
    assert before[0] == after[0]
    assert before[1][0] == after[1][0]
    assert np.array_equal(before[1][1], after[1][1])
    assert before[1][2:] == after[1][2:]
    assert torch.equal(before[2], after[2])


def test_real_policy_identity_is_separate_by_selection_and_outputs_are_byte_deterministic(tmp_path: Path) -> None:
    first_source = tmp_path / "a" / "active_policy_stage_lineage_events.jsonl"
    second_source = tmp_path / "b" / "active_policy_stage_lineage_events.jsonl"
    _write_jsonl(first_source, [_policy_row()])
    _write_jsonl(second_source, [_policy_row()])
    source_before = (first_source.read_bytes(), second_source.read_bytes())
    rng_before = _rng_state()
    first = tmp_path / "output-a"
    second = tmp_path / "output-b"
    result = write_readiness_outputs(
        [first_source, second_source],
        first,
        selection_reason="two explicitly selected policy records",
    )
    write_readiness_outputs(
        [second_source, first_source],
        second,
        selection_reason="two explicitly selected policy records",
    )
    rng_after = _rng_state()
    assert result["decision"] == "CHRONICLE-GEN1-R1-C"
    rows = [json.loads(value) for value in (first / "mutation_event_inventory.jsonl").read_text().splitlines()]
    assert len(rows) == 2
    assert len({row["mutation_event_id"] for row in rows}) == 2
    assert all(row["source_scope"] == "policy_parameters" for row in rows)
    assert _tree_bytes(first) == _tree_bytes(second)
    assert source_before == (first_source.read_bytes(), second_source.read_bytes())
    _assert_rng_equal(rng_before, rng_after)
    assert sorted(_tree_bytes(first)) == sorted(OUTPUT_NAMES)


def test_canonical_fixture_preserves_inheritance_persistence_identity_and_association(tmp_path: Path) -> None:
    fixture = _f0_fixture(tmp_path / "f0")
    output = tmp_path / "readiness"
    result = write_readiness_outputs(
        [fixture],
        output,
        selection_reason="explicit canonical evidence test",
    )
    assert result["decision"] == "CHRONICLE-GEN1-R1-A"
    carriers = [json.loads(value) for value in (output / "carrier_records.jsonl").read_text().splitlines()]
    assert any(row["evidence_strength"] == "inherited" and row["agent_id"] == "3" for row in carriers)
    persistence = json.loads((output / "mutation_persistence_summary.json").read_text())["mutations"]
    assert any(row["status"] == "persisted" and row["duration_generations"] == 1 for row in persistence)
    identities = json.loads((output / "mutation_identity_map.json").read_text())["mutations"]
    similar = [row for row in identities if row["identity_relation"] == "independent_similar_mutation"]
    assert len(similar) == 2
    assert all(len(row["independent_similar_mutation_event_ids"]) == 1 for row in similar)
    associations = list(csv_rows(output / "mutation_outcome_associations.csv"))
    assert any(row["status"] == "available" and row["causal_status"] == "association_only" for row in associations)


def csv_rows(path: Path) -> list[dict[str, str]]:
    import csv

    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_fail_closed_matching_native_coverage_causal_wording_and_path_safety(tmp_path: Path) -> None:
    policy = tmp_path / "active_policy_stage_lineage_events.jsonl"
    _write_jsonl(policy, [_policy_row()])
    output = tmp_path / "policy-output"
    write_readiness_outputs([policy], output, selection_reason="explicit policy source")
    association = csv_rows(output / "mutation_outcome_associations.csv")[0]
    assert association["status"] == "unavailable"
    assert json.loads(association["matching_dimensions"]) == list(MATCH_HIERARCHY)
    assert json.loads(association["unmatched_dimensions"]) == list(MATCH_HIERARCHY)

    native = tmp_path / "lineage_metrics.jsonl"
    _write_jsonl(native, [{"schema_version": "native_lineage_metrics_v1", "mutation_count": 4}])
    native_output = tmp_path / "native-output"
    native_result = write_readiness_outputs([native], native_output, selection_reason="explicit aggregate source")
    assert native_result["decision"] == "CHRONICLE-GEN1-R1-D"
    completeness = json.loads((native_output / "source_completeness.json").read_text())
    assert completeness["sources"][0]["capabilities"] == ["aggregate_or_outcome_context_only"]

    with pytest.raises(GenomeReadinessError, match="unsupported causal"):
        validate_no_causal_wording({"summary": "This mutation caused success"})
    with pytest.raises(GenomeReadinessError, match="unsafe source path"):
        write_readiness_outputs([tmp_path / ".." / tmp_path.name / policy.name], tmp_path / "bad", selection_reason="bad")
    with pytest.raises(GenomeReadinessError, match="outside selected source roots"):
        write_readiness_outputs([tmp_path], tmp_path / "nested-output", selection_reason="bad output")
    with pytest.raises(GenomeReadinessError, match="maximum_records"):
        write_readiness_outputs([policy], tmp_path / "too-large", selection_reason="bad bound", maximum_records_per_source=1_000_001)


@dataclass
class _Stats:
    lineage_id: str
    founder_id: int


@dataclass
class _Agent:
    agent_id: int
    stats: _Stats


def _chronicle(run_root: Path) -> tuple[Path, str]:
    config = SimpleNamespace(
        chronicle_enable=True,
        chronicle_profile="notable",
        chronicle_run_id="run-a",
        chronicle_max_stored_events_per_generation=64,
        chronicle_minimum_persistence_generations=1,
        campaign_id="campaign-a",
        branch_id="branch-a",
        generations=1,
        allow_large_output_profile=False,
    )
    recorder = ScientificChronicleRecorder(SimpleNamespace(sim_cfg=config, run_dir=str(run_root)))
    recorder.observe_generation(
        generation=0,
        step=9,
        alive_agents=[_Agent(2, _Stats("L1", 1))],
        deaths=0,
        births=0,
        failed_births=0,
        cohort_size=1,
        patch_metrics={"patch_food_depleted_frac": 0.0},
    )
    recorder.finalize()
    event = json.loads((recorder.root / "formal_events.jsonl").read_text().splitlines()[0])
    return recorder.root, str(event["event_id"])


def test_external_companion_links_do_not_modify_frozen_chronicle(tmp_path: Path) -> None:
    chronicle_root, event_id = _chronicle(tmp_path / "runs" / "run-a")
    before = {path.name: path.read_bytes() for path in chronicle_root.iterdir() if path.is_file()}
    fixture = _f0_fixture(tmp_path / "f0", chronicle_event_id=event_id)
    readiness = tmp_path / "readiness"
    write_readiness_outputs([fixture], readiness, selection_reason="companion link test")
    companion_dir = chronicle_root / "gen1_readiness"
    companion_dir.mkdir()
    shutil.copyfile(readiness / "chronicle_companion_index.json", companion_dir / "chronicle_companion_index.json")
    cfg = SimpleNamespace(runs_root=tmp_path / "runs", chronicle_roots=(tmp_path / "runs",), chronicle_scan_depth=5)
    run_key = discover_runs(cfg)[0]["run_key"]
    subject = subject_detail(cfg, run_key, "agent", "2")
    event = event_detail(cfg, run_key, event_id)
    assert subject["gen1_readiness_links"]["status"] == "available"
    assert subject["gen1_readiness_links"]["mutation_event_ids"]
    assert event["gen1_readiness_links"]["status"] == "available"
    assert event["gen1_readiness_links"]["mutation_event_ids"]
    after = {path.name: path.read_bytes() for path in chronicle_root.iterdir() if path.is_file()}
    assert before == after
