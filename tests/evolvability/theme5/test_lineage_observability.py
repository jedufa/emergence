from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from agents.agent import (
    AgentStats,
    assign_child_lineage,
    assign_founder_lineage,
    format_lineage_id,
    normalize_lineage_stats,
)
from simulation.checkpoint import _apply_agents_and_genomes
from simulation.checkpoint_bundle_v2 import checkpoint_bundle_v2_apply, checkpoint_bundle_v2_load, checkpoint_bundle_v2_save
from simulation.metrics_pkg.lineage import (
    ARCHIVE_ENTRY_SCHEMA_VERSION,
    LineageMetricsState,
    LineageMetricsWriter,
    compute_branch_candidates,
    compute_lineage_descriptors,
    compute_lineage_snapshot,
    compute_qd_archive_entries,
    descriptor_distance,
    lineage_entropy,
    top_n_share,
    upgrade_qd_archive_entry,
    validate_qd_archive_entry,
    write_qd_archive_entries,
)
from tools.bin.theme5_lineage_experiment import create_manifest, summarize_manifest, validate_manifest
from tests.evolvability.phase_e.test_e1_metrics_engine import _make_sim
from tools.bin.theme5_lineage_report import render_report


def _agent(
    aid: int,
    *,
    lineage_id: str = "",
    founder_id: int = -1,
    offspring: int = 0,
    depth: int = 0,
    survival: int = 0,
    energy: float = 10.0,
    health: float = 10.0,
    injury: float = 0.0,
    memory_dim: int = 0,
    use_gru: bool = False,
    aggression_bias: float = 0.0,
    exploration_bias: float = 0.0,
    hunger: float | None = None,
    fear: float | None = None,
):
    stats = AgentStats()
    stats.offspring_count = int(offspring)
    stats.lineage_depth = int(depth)
    stats.survival_steps = int(survival)
    genome = SimpleNamespace(
        memory_dim=int(memory_dim),
        use_gru=bool(use_gru),
        hidden_size=64,
        aggression_bias=float(aggression_bias),
        exploration_bias=float(exploration_bias),
    )
    obj = SimpleNamespace(agent_id=int(aid), stats=stats, energy=float(energy), health=float(health), injury=float(injury), genome=genome)
    if hunger is not None or fear is not None:
        drives = np.zeros((4,), dtype=np.float32)
        if hunger is not None:
            drives[0] = float(hunger)
        if fear is not None:
            drives[3] = float(fear)
        obj.drives = drives
    if lineage_id:
        stats.lineage_id = lineage_id
    if founder_id >= 0:
        stats.founder_id = int(founder_id)
    normalize_lineage_stats(stats, int(aid))
    return obj


def test_founder_lineage_assignment_is_deterministic() -> None:
    a = _agent(7)
    assign_founder_lineage(a, generation=3, tick=11)

    assert a.stats.lineage_id == "L00000007"
    assert a.stats.founder_id == 7
    assert a.stats.parent_ids == ()
    assert a.stats.parent_lineage_ids == ()
    assert a.stats.birth_generation == 3
    assert a.stats.birth_tick == 11
    assert a.stats.lineage_depth == 0
    assert format_lineage_id(7) == "L00000007"


def test_child_lineage_records_parents_and_inherits_lower_agent_parent() -> None:
    high = _agent(20, lineage_id="L00000020", founder_id=20, depth=4)
    low = _agent(10, lineage_id="L00000010", founder_id=10, depth=2)
    child = _agent(30)

    assign_child_lineage(child, high, low, generation=5, tick=0)

    assert child.stats.parent_ids == (10, 20)
    assert child.stats.parent_lineage_ids == ("L00000010", "L00000020")
    assert child.stats.lineage_id == "L00000010"
    assert child.stats.founder_id == 10
    assert child.stats.lineage_depth == 5
    assert child.stats.birth_generation == 5


def test_lineage_helpers_do_not_advance_common_rngs() -> None:
    random.seed(1234)
    np_rng = np.random.default_rng(1234)
    py_before = random.getstate()
    np_before = np_rng.bit_generator.state

    p1 = _agent(2)
    p2 = _agent(1)
    child = _agent(3)
    assign_founder_lineage(p1)
    assign_founder_lineage(p2)
    assign_child_lineage(child, p1, p2)

    assert random.getstate() == py_before
    assert np_rng.bit_generator.state == np_before


def test_lineage_metric_math_and_collapse_warning() -> None:
    counts = {"L00000001": 3, "L00000002": 1}
    assert top_n_share(counts, 1) == 0.75
    assert top_n_share(counts, 3) == 1.0
    assert lineage_entropy(counts) > 0.0

    mixed_agents = [_agent(1, lineage_id="L00000001"), _agent(2, lineage_id="L00000002")]
    cfg = SimpleNamespace(
        lineage_dominance_threshold=0.80,
        lineage_dominance_streak_required=2,
        lineage_min_active_lineages=2,
        lineage_entropy_minimum=0.10,
        lineage_reproduction_dominance_threshold=0.80,
        lineage_reproduction_dominance_streak_required=2,
    )
    summary, rows, events, state = compute_lineage_snapshot(generation=0, agents=mixed_agents, sim_cfg=cfg)
    assert summary["active_lineages"] == 2
    assert summary["lineage_collapse_warning"] == 0
    assert len(rows) == 2
    assert events == []

    dominated = [_agent(i, lineage_id="L00000001") for i in range(1, 6)] + [_agent(6, lineage_id="L00000002")]
    summary, _, _, state = compute_lineage_snapshot(generation=1, agents=dominated, sim_cfg=cfg, state=state)
    assert summary["dominant_lineage_population_share"] > 0.80
    assert summary["lineage_collapse_warning"] == 0
    summary, _, events, _ = compute_lineage_snapshot(generation=2, agents=dominated, sim_cfg=cfg, state=state)
    assert summary["lineage_collapse_warning"] == 1
    assert any(e["event"] == "lineage_collapse_warning_triggered" for e in events)


def test_lineage_writer_is_repeatable_for_same_seeded_inputs(tmp_path: Path) -> None:
    agents = [_agent(1, lineage_id="L00000001", offspring=2), _agent(2, lineage_id="L00000002", offspring=1)]
    cfg = SimpleNamespace(lineage_observability_enable=True)

    run_a = tmp_path / "a"
    run_b = tmp_path / "b"
    for run_dir in (run_a, run_b):
        writer = LineageMetricsWriter(str(run_dir))
        writer.log_generation(generation=0, agents=agents, sim_cfg=cfg, genome_diversity=0.25)

    assert (run_a / "metrics" / "lineage_summary.csv").read_text(encoding="utf-8") == (
        run_b / "metrics" / "lineage_summary.csv"
    ).read_text(encoding="utf-8")
    assert (run_a / "metrics" / "lineage_metrics.jsonl").read_text(encoding="utf-8") == (
        run_b / "metrics" / "lineage_metrics.jsonl"
    ).read_text(encoding="utf-8")
    assert (run_a / "metrics" / "lineage_events.jsonl").exists()
    assert (run_a / "metrics" / "lineage_descriptors.jsonl").read_text(encoding="utf-8") == (
        run_b / "metrics" / "lineage_descriptors.jsonl"
    ).read_text(encoding="utf-8")
    assert (run_a / "metrics" / "branch_candidates.jsonl").read_text(encoding="utf-8") == (
        run_b / "metrics" / "branch_candidates.jsonl"
    ).read_text(encoding="utf-8")


def test_checkpoint_bundle_roundtrip_and_old_payload_fallback(tmp_path: Path) -> None:
    run_dir = tmp_path / "bundle"
    sim = _make_sim(run_dir, seed=515, generations=1, steps=1, pop_size=4)
    sim.run(start_gen=0, skip_init=False)

    bundle = run_dir / "checkpoint_bundles_v2" / "bundle_gen000001_theme5"
    checkpoint_bundle_v2_save(sim, str(bundle), generation_index=1, strict_replay_mode=True)
    payload = checkpoint_bundle_v2_load(str(bundle), strict=True)

    agents = payload["population"]["agents"]
    assert agents
    assert all("lineage_id" in a.get("stats", {}) for a in agents)

    for agent in agents:
        stats = agent.get("stats", {})
        for key in ("lineage_id", "founder_id", "parent_ids", "parent_lineage_ids", "birth_generation", "birth_tick", "lineage_depth"):
            stats.pop(key, None)

    restore = _make_sim(tmp_path / "restore", seed=515, generations=1, steps=1, pop_size=4)
    checkpoint_bundle_v2_apply(restore, payload, restore_rng=True)
    restored_agents = sorted(restore.pop.agents.values(), key=lambda a: int(a.agent_id))
    assert restored_agents
    for agent in restored_agents:
        assert agent.stats.lineage_id == format_lineage_id(agent.agent_id)
        assert agent.stats.founder_id == agent.agent_id
        assert agent.stats.parent_ids == ()
        assert agent.stats.parent_lineage_ids == ()
        assert agent.stats.lineage_depth == 0


def test_legacy_checkpoint_v1_old_stats_fallback_is_deterministic() -> None:
    sim = SimpleNamespace(pop=SimpleNamespace(next_id=0, agents={}), env=SimpleNamespace())
    agents_npz = {
        "population_next_id": np.asarray([8], dtype=np.int32),
        "agent_id": np.asarray([7], dtype=np.int32),
        "alive": np.asarray([True]),
        "energy": np.asarray([1.0], dtype=np.float32),
        "health": np.asarray([1.0], dtype=np.float32),
        "injury": np.asarray([0.0], dtype=np.float32),
        "rest_streak": np.asarray([0], dtype=np.int32),
        "repro_cooldown": np.asarray([0], dtype=np.int32),
        "slot": np.asarray([-1], dtype=np.int32),
        "slot_token": np.asarray([0], dtype=np.int32),
        "stats_json": np.asarray(["{}"], dtype=object),
    }

    _apply_agents_and_genomes(sim, agents_npz, {7: {}})

    restored = sim.pop.agents[7]
    assert restored.stats.lineage_id == "L00000007"
    assert restored.stats.founder_id == 7
    assert restored.stats.parent_ids == ()
    assert restored.stats.parent_lineage_ids == ()
    assert restored.stats.lineage_depth == 0


def test_report_generation_from_lineage_outputs(tmp_path: Path) -> None:
    run_dir = tmp_path / "report_run"
    writer = LineageMetricsWriter(str(run_dir))
    cfg = SimpleNamespace(lineage_observability_enable=True)
    writer.log_generation(
        generation=0,
        agents=[_agent(1, lineage_id="L00000001", offspring=2), _agent(2, lineage_id="L00000002", offspring=0)],
        sim_cfg=cfg,
    )

    report = render_report(run_dir)
    assert "Theme 5 Lineage Report" in report
    assert "L00000001" in report
    assert "Behavioral Descriptors" in report
    assert "Branch Candidates" in report
    assert "Quality-Diversity Archive" in report
    assert "Branch/Restart Manifests" in report
    assert "Co-evolution Readiness" in report

    rows = list(csv.DictReader((run_dir / "metrics" / "lineage_summary.csv").open(newline="", encoding="utf-8")))
    metrics = [json.loads(line) for line in (run_dir / "metrics" / "lineage_metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rows
    assert metrics


def test_lineage_descriptors_group_by_biological_lineage_and_order_deterministically() -> None:
    agents = [
        _agent(3, lineage_id="L00000002", survival=2, offspring=1, memory_dim=8, use_gru=True, hunger=0.4, fear=0.2),
        _agent(1, lineage_id="L00000001", survival=10, offspring=3, memory_dim=0, aggression_bias=0.9, exploration_bias=0.1),
        _agent(2, lineage_id="L00000001", survival=8, offspring=1, memory_dim=0, aggression_bias=0.7, exploration_bias=0.2),
    ]

    record = compute_lineage_descriptors(generation=4, agents=agents)

    assert record["schema_version"] == "theme5_lineage_descriptors_v1"
    assert [d["lineage_id"] for d in record["lineages"]] == ["L00000001", "L00000002"]
    first = record["lineages"][0]
    assert first["population_count"] == 2
    assert first["mean_survival_steps"] == 9.0
    assert first["mean_offspring_count"] == 2.0
    assert "mean_memory_dim" in record["global_descriptor_summary"]["available_numeric_fields"]


def test_descriptor_distance_is_deterministic_and_missing_fields_do_not_crash() -> None:
    a = {"lineage_id": "L00000001", "mean_survival_steps": 10.0, "mean_memory_dim": 0.0}
    b = {"lineage_id": "L00000002", "mean_survival_steps": 1.0, "mean_memory_dim": 64.0, "mean_injury": None}

    d1 = descriptor_distance(a, b)
    d2 = descriptor_distance(a, b)

    assert d1 == d2
    assert d1 > 0.0
    assert descriptor_distance({"lineage_id": "x"}, {"lineage_id": "y"}) == 0.0


def test_branch_candidates_detect_expected_synthetic_types() -> None:
    agents = [
        _agent(1, lineage_id="L00000001", offspring=8, survival=10, memory_dim=0, injury=0.1),
        _agent(2, lineage_id="L00000001", offspring=7, survival=9, memory_dim=0, injury=0.2),
        _agent(3, lineage_id="L00000001", offspring=6, survival=9, memory_dim=0, injury=0.2),
        _agent(4, lineage_id="L00000002", offspring=1, survival=3, memory_dim=64, use_gru=True, injury=0.5, fear=0.9),
        _agent(5, lineage_id="L00000003", offspring=0, survival=15, memory_dim=16, injury=0.0),
    ]
    state = LineageMetricsState()
    summary, rows, _, state = compute_lineage_snapshot(generation=0, agents=agents, state=state)
    summary, rows, _, state = compute_lineage_snapshot(generation=1, agents=agents, state=state)
    descriptors = compute_lineage_descriptors(generation=1, agents=agents)

    candidates = compute_branch_candidates(
        generation=1,
        summary=summary,
        lineage_rows=rows,
        descriptor_record=descriptors,
        state=state,
    )
    pairs = {(c["lineage_id"], c["candidate_type"]) for c in candidates["candidates"]}

    assert ("L00000001", "dominant") in pairs
    assert ("L00000002", "rare_persistent") in pairs
    assert ("L00000002", "behaviorally_divergent") in pairs
    assert ("L00000002", "high_cognition_viable") in pairs
    assert ("L00000001", "low_cognition_strong_baseline") in pairs
    assert any(t == "survival_specialist" for _, t in pairs)
    scores = [c["candidate_score"] for c in candidates["candidates"]]
    assert scores == sorted(scores, reverse=True)
    assert all(c["reasons"] for c in candidates["candidates"])


def test_lineage_candidate_state_export_import_roundtrip() -> None:
    state = LineageMetricsState()
    agents = [_agent(1, lineage_id="L00000001"), _agent(2, lineage_id="L00000002")]
    _, _, _, state = compute_lineage_snapshot(generation=0, agents=agents, state=state)
    payload = state.to_dict()

    restored = LineageMetricsState.from_dict(payload)

    assert restored.lineage_presence_counts == state.lineage_presence_counts
    assert restored.lineage_first_seen_generation == state.lineage_first_seen_generation


def _candidate_fixture():
    agents = [
        _agent(1, lineage_id="L00000001", offspring=8, survival=10, memory_dim=0, injury=0.1),
        _agent(2, lineage_id="L00000001", offspring=7, survival=9, memory_dim=0, injury=0.2),
        _agent(3, lineage_id="L00000001", offspring=6, survival=9, memory_dim=0, injury=0.2),
        _agent(4, lineage_id="L00000002", offspring=1, survival=3, memory_dim=64, use_gru=True, injury=0.5, fear=0.9),
        _agent(5, lineage_id="L00000003", offspring=0, survival=15, memory_dim=16, injury=0.0),
    ]
    state = LineageMetricsState()
    summary, rows, _, state = compute_lineage_snapshot(generation=0, agents=agents, state=state)
    summary, rows, _, state = compute_lineage_snapshot(generation=1, agents=agents, state=state)
    descriptors = compute_lineage_descriptors(generation=1, agents=agents)
    candidates = compute_branch_candidates(generation=1, summary=summary, lineage_rows=rows, descriptor_record=descriptors, state=state)
    return agents, state, descriptors, candidates


def test_qd_archive_entries_are_deterministic_and_metadata_only(tmp_path: Path) -> None:
    agents, state_a, descriptors, candidates = _candidate_fixture()
    cfg = SimpleNamespace(
        lineage_qd_archive_enable=True,
        lineage_qd_archive_top_n=3,
        lineage_qd_archive_min_score=0.25,
        lineage_qd_archive_cooldown_generations=5,
    )
    entries_a = compute_qd_archive_entries(
        generation=1,
        candidate_record=candidates,
        descriptor_record=descriptors,
        agents=agents,
        state=state_a,
        run_dir=str(tmp_path),
        sim_cfg=cfg,
    )
    agents_b, state_b, descriptors_b, candidates_b = _candidate_fixture()
    entries_b = compute_qd_archive_entries(
        generation=1,
        candidate_record=candidates_b,
        descriptor_record=descriptors_b,
        agents=agents_b,
        state=state_b,
        run_dir=str(tmp_path),
        sim_cfg=cfg,
    )

    assert entries_a == entries_b
    assert entries_a
    assert len(entries_a) <= 3
    assert all(str(e["archive_entry_id"]).startswith("t5qd_") for e in entries_a)
    assert all(e["created_by"] == "theme5_quality_diversity_archive" for e in entries_a)
    assert all(e["ecological_role"] == "general_agent" for e in entries_a)
    assert all(e["schema_version"] == ARCHIVE_ENTRY_SCHEMA_VERSION for e in entries_a)
    assert all("claims_allowed" in e and "claims_not_allowed" in e for e in entries_a)
    assert all("branch_replay_eligibility" in e for e in entries_a)
    assert all(e["live_population_context"]["live_diversity_not_archive_diversity"] is True for e in entries_a)
    assert compute_qd_archive_entries(
        generation=2,
        candidate_record=candidates,
        descriptor_record=descriptors,
        agents=agents,
        state=state_a,
        run_dir=str(tmp_path),
        sim_cfg=cfg,
    ) == []


def test_qd_archive_writer_outputs_index_and_entries(tmp_path: Path) -> None:
    agents, state, descriptors, candidates = _candidate_fixture()
    entries = compute_qd_archive_entries(
        generation=1,
        candidate_record=candidates,
        descriptor_record=descriptors,
        agents=agents,
        state=state,
        run_dir=str(tmp_path),
        sim_cfg=SimpleNamespace(lineage_qd_archive_enable=True, lineage_qd_archive_top_n=2, lineage_qd_archive_min_score=0.25),
    )
    write_qd_archive_entries(str(tmp_path), entries)

    index = tmp_path / "archives" / "lineage_qd" / "archive_index.jsonl"
    assert index.exists()
    rows = [json.loads(line) for line in index.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == len(entries)
    assert [r["archive_entry_id"] for r in rows] == [e["archive_entry_id"] for e in entries]
    for row in rows:
        assert (tmp_path / "archives" / "lineage_qd" / "entries" / f"{row['archive_entry_id']}.json").exists()


def test_qd_archive_old_metadata_entry_upgrades_without_checkpoint() -> None:
    old = {
        "schema_version": "theme5_lineage_qd_archive_entry_v1",
        "archive_entry_id": "old",
        "generation": 2,
        "lineage_id": "L00000001",
        "candidate_type": "dominant_lineage",
        "candidate_score": 0.8,
    }

    upgraded = upgrade_qd_archive_entry(old)
    validation = validate_qd_archive_entry(old)

    assert upgraded["schema_version"] == ARCHIVE_ENTRY_SCHEMA_VERSION
    assert upgraded["branch_replay_eligibility"]["mode"] == "metadata_only"
    assert validation["ok"] is True


def test_qd_archive_replay_eligible_entry_requires_checkpoint_provenance() -> None:
    entry = {
        "schema_version": ARCHIVE_ENTRY_SCHEMA_VERSION,
        "archive_entry_id": "replay",
        "generation": 2,
        "seed": 123,
        "lineage_id": "L00000001",
        "candidate_type": "dominant_lineage",
        "claims_allowed": [],
        "claims_not_allowed": [],
        "branch_replay_eligibility": {"eligible": True, "mode": "checkpoint_backed"},
    }

    validation = validate_qd_archive_entry(entry)

    assert validation["ok"] is False
    assert {err["field"] for err in validation["errors"]} >= {
        "checkpoint_reference",
        "checkpoint_hash",
        "scenario_config_hash",
    }


def test_qd_archive_capture_is_passive_for_agent_state(tmp_path: Path) -> None:
    agents = [_agent(1, lineage_id="L00000001", offspring=2), _agent(2, lineage_id="L00000002", offspring=0)]
    before = [(a.agent_id, a.stats.lineage_id, a.stats.offspring_count, a.energy) for a in agents]
    writer = LineageMetricsWriter(str(tmp_path))

    writer.log_generation(
        generation=0,
        agents=agents,
        sim_cfg=SimpleNamespace(lineage_observability_enable=True, lineage_qd_archive_enable=True),
    )

    after = [(a.agent_id, a.stats.lineage_id, a.stats.offspring_count, a.energy) for a in agents]
    assert after == before


def test_archive_state_export_import_roundtrip() -> None:
    agents, state, descriptors, candidates = _candidate_fixture()
    compute_qd_archive_entries(
        generation=1,
        candidate_record=candidates,
        descriptor_record=descriptors,
        agents=agents,
        state=state,
        sim_cfg=SimpleNamespace(lineage_qd_archive_enable=True),
    )
    restored = LineageMetricsState.from_dict(state.to_dict())

    assert restored.archived_entry_ids == state.archived_entry_ids
    assert restored.archive_last_generation_by_key == state.archive_last_generation_by_key


def test_manifest_creation_from_lineage_and_archive_entry(tmp_path: Path) -> None:
    writer = LineageMetricsWriter(str(tmp_path))
    writer.log_generation(generation=0, agents=[_agent(1, lineage_id="L00000001", offspring=2), _agent(2, lineage_id="L00000002")], sim_cfg=SimpleNamespace(lineage_observability_enable=True))
    agents, state, descriptors, candidates = _candidate_fixture()
    entries = compute_qd_archive_entries(
        generation=1,
        candidate_record=candidates,
        descriptor_record=descriptors,
        agents=agents,
        state=state,
        run_dir=str(tmp_path),
        sim_cfg=SimpleNamespace(lineage_qd_archive_enable=True, lineage_qd_archive_top_n=1, lineage_qd_archive_min_score=0.25),
    )
    write_qd_archive_entries(str(tmp_path), entries)

    manifest = create_manifest(
        run_dir=tmp_path,
        manifest_type="archive_candidate_branch",
        archive_entry_ids=[entries[0]["archive_entry_id"]],
    )
    repeat = create_manifest(
        run_dir=tmp_path,
        manifest_type="archive_candidate_branch",
        archive_entry_ids=[entries[0]["archive_entry_id"]],
    )

    assert manifest == repeat
    assert manifest["selected_archive_entry_ids"] == [entries[0]["archive_entry_id"]]
    assert entries[0]["lineage_id"] in manifest["selected_lineage_ids"]
    assert validate_manifest(manifest, run_dir=tmp_path)["valid"] is True
    summary = summarize_manifest(manifest)
    assert summary["execution_implemented"] is False


def test_manifest_creation_from_candidate_filter_and_validation_errors(tmp_path: Path) -> None:
    writer = LineageMetricsWriter(str(tmp_path))
    writer.log_generation(
        generation=0,
        agents=[
            _agent(1, lineage_id="L00000001", offspring=5, survival=5),
            _agent(2, lineage_id="L00000001", offspring=3, survival=5),
            _agent(3, lineage_id="L00000002", offspring=0, survival=10, memory_dim=64, use_gru=True),
        ],
        sim_cfg=SimpleNamespace(lineage_observability_enable=True),
    )

    manifest = create_manifest(run_dir=tmp_path, manifest_type="single_lineage_branch", candidate_type="dominant", top_n=1)
    assert manifest["selected_lineage_ids"]
    assert validate_manifest(manifest, run_dir=tmp_path)["valid"] is True

    invalid = dict(manifest)
    invalid["selected_lineage_ids"] = ["L99999999"]
    result = validate_manifest(invalid, run_dir=tmp_path)
    assert result["valid"] is False
    assert any("unknown lineage_id" in issue for issue in result["issues"])


def test_stage7_9_report_sections_with_archive_and_manifest(tmp_path: Path) -> None:
    writer = LineageMetricsWriter(str(tmp_path))
    agents = [
        _agent(1, lineage_id="L00000001", offspring=5, survival=5),
        _agent(2, lineage_id="L00000001", offspring=3, survival=5),
        _agent(3, lineage_id="L00000002", offspring=0, survival=10, memory_dim=64, use_gru=True),
    ]
    writer.log_generation(generation=0, agents=agents, sim_cfg=SimpleNamespace(lineage_observability_enable=True))
    archive_rows = [
        json.loads(line)
        for line in (tmp_path / "archives" / "lineage_qd" / "archive_index.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    manifest = create_manifest(
        run_dir=tmp_path,
        manifest_type="archive_candidate_branch",
        archive_entry_ids=[archive_rows[0]["archive_entry_id"]],
    )
    manifest_dir = tmp_path / "archives" / "lineage_qd" / "branch_manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    (manifest_dir / f"{manifest['manifest_id']}.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    report = render_report(tmp_path)

    assert "Quality-Diversity Archive" in report
    assert "live diversity != archive diversity" in report
    assert "Replay-eligible checkpoint-backed entries" in report
    assert "Branch/Restart Manifests" in report
    assert "Co-evolution Readiness" in report
    assert "general_agent" in report
    assert manifest["manifest_id"] in report
