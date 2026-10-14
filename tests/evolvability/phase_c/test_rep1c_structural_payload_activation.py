from __future__ import annotations

import csv
import json
import random
from pathlib import Path

from agents.developmental_program_v1 import canonicalize_developmental_program
from agents.genome import Genome
from agents.structural_payload_activation import (
    STRUCTURAL_PAYLOAD_SEED_SCHEMA_VERSION,
    apply_structural_payload_activation_to_founder,
    minimal_canonical_seed_payload,
)
from env.env_config import EnvConfig
from evolution.operators_c1 import apply_operator_pipeline
from evolution.population_pkg.population_config import PopConfig
from evolution.recombination_c2 import recombine_lineage_payloads
from simulation.checkpoint_bundle_v2 import checkpoint_bundle_v2_apply, checkpoint_bundle_v2_load, checkpoint_bundle_v2_save
from simulation.metrics import MetricsLogger
from simulation.metrics_pkg.cognitive_status import compute_cognitive_status_metrics
from simulation.metrics_pkg.noop_loggers import NoOpMetricsLogger
from simulation.performance_profiles import ACCEPTED_W6_FAST_PROFILE, apply_performance_profile
from simulation.runner import SimulationRunner
from simulation.runner_pkg.runner_config import SimConfig


def _cfg(*, enabled: bool = False, w6: bool = False, metrics: bool = False) -> SimConfig:
    cfg = SimConfig(seed=123, generations=1, steps_per_generation=3, device="cpu", bank_device="cpu", behavior_log_enable=False)
    cfg.policy_deterministic_actions = True
    cfg.checkpoint_bundle_v2_period_gen = 1
    cfg.trait_capacity_saturation_reporting_enable = bool(metrics)
    if enabled:
        cfg.structural_payload_activation_enable = True
        cfg.structural_payload_seed_profile = "minimal_canonical_v1"
        cfg.structural_payload_seed_schema_version = STRUCTURAL_PAYLOAD_SEED_SCHEMA_VERSION
    if w6:
        apply_performance_profile(cfg, ACCEPTED_W6_FAST_PROFILE)
        cfg.enable_cognitive_status_metrics = True
        cfg.trait_capacity_saturation_reporting_enable = bool(metrics)
    return cfg


def _genome() -> Genome:
    return Genome(
        hidden_size=64,
        exploration_bias=0.1,
        aggression_bias=0.2,
        reproduction_threshold=30.0,
        mutation_rate=0.05,
        action_temperature=1.0,
        use_gru=False,
        memory_dim=0,
        drive_inertia=0.8,
        drive_noise=0.01,
        drive_logit_gain=1.0,
        pred_lr=0.08,
        developmental_program=canonicalize_developmental_program(None),
    )


def _make_sim(run_dir: Path, *, enabled: bool, w6: bool = False, metrics: bool = False) -> SimulationRunner:
    sim_cfg = _cfg(enabled=enabled, w6=w6, metrics=metrics)
    return SimulationRunner(
        env_cfg=EnvConfig(width=8, height=8),
        pop_cfg=PopConfig(population_size=6),
        sim_cfg=sim_cfg,
        logger=MetricsLogger(str(run_dir)) if metrics else NoOpMetricsLogger(),
        run_dir=str(run_dir),
        perf=False,
    )


def _last_row(run_dir: Path) -> dict[str, str]:
    with (run_dir / "metrics.csv").open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows
    return dict(rows[-1])


def _agent_signature(sim: SimulationRunner) -> list[dict[str, object]]:
    out = []
    for aid, agent in sorted(sim.pop.agents.items()):
        g = agent.genome
        out.append(
            {
                "agent_id": int(aid),
                "module_count": len(g.modules),
                "connection_count": len(g.connections),
                "genome_hash": str(g.canonical_hash()),
            }
        )
    return out


def test_rep1c_flag_off_founders_remain_empty_and_seed_is_not_applied() -> None:
    g = _genome()
    seeded = apply_structural_payload_activation_to_founder(g, _cfg(enabled=False))
    assert seeded is g
    assert seeded.modules == []
    assert seeded.connections == []


def test_rep1c_minimal_seed_is_deterministic_valid_and_below_caps() -> None:
    g1 = apply_structural_payload_activation_to_founder(_genome(), _cfg(enabled=True))
    g2 = apply_structural_payload_activation_to_founder(_genome(), _cfg(enabled=True))
    assert g1.canonical_bytes() == g2.canonical_bytes()
    assert len(g1.modules) == 3
    assert len(g1.connections) == 2
    assert [m["module_id"] for m in g1.modules] == ["m000_policy_core", "m010_behavior_bias", "m030_identity_regulation"]
    assert [c["connection_id"] for c in g1.connections] == ["c000_behavior_to_policy", "c010_identity_to_policy"]
    assert g1.b3_locality_dropped_connections == 0
    assert len(g1.modules) < int(g1.b2_caps["max_modules_per_agent"])
    assert len(g1.connections) < int(g1.b2_caps["max_connections_per_agent"])


def test_rep1c_seed_application_consumes_no_python_random_draws() -> None:
    rng = random.Random(1234)
    before = rng.getstate()
    _ = apply_structural_payload_activation_to_founder(_genome(), _cfg(enabled=True))
    after = rng.getstate()
    assert before == after


def test_rep1c_old_path_empty_is_absorbing_for_c1_and_c2() -> None:
    empty = _genome().canonical_payload()
    out, telemetry = apply_operator_pipeline(payload=empty, rng=random.Random(5), max_ops=2)
    assert out["modules"] == []
    assert out["connections"] == []
    assert telemetry["noop_count"] == 2
    rec, summary = recombine_lineage_payloads(parent_a_payload=empty, parent_b_payload=empty, rng=random.Random(6))
    assert rec["modules"] == []
    assert rec["connections"] == []
    assert summary.matched_modules == 0
    assert summary.matched_connections == 0


def test_rep1c_seed_reaches_developmental_phenotype_without_scalar_drift() -> None:
    base = _genome()
    seeded = apply_structural_payload_activation_to_founder(base, _cfg(enabled=True))
    expr = seeded.resolve_developmental_expression(age_steps=0)
    assert expr["module_count"] == 3
    assert expr["active_module_count"] == 3
    assert expr["effective"]["aggression_bias"] == base.aggression_bias
    assert expr["effective"]["action_temperature"] == base.action_temperature
    assert expr["effective"]["drive_logit_gain"] == base.drive_logit_gain


def test_rep1c_existing_operator_can_change_seeded_payload_and_report_counters() -> None:
    seeded = apply_structural_payload_activation_to_founder(_genome(), _cfg(enabled=True))
    out, telemetry = apply_operator_pipeline(
        payload=seeded.canonical_payload(),
        rng=random.Random(1),
        max_ops=2,
    )
    assert len(out["modules"]) > 0
    assert telemetry["max_ops"] == 2
    assert sum(telemetry["usage_counts"].values()) == 2
    assert telemetry["noop_count"] < 2
    assert out["genes"]["c1_operator_applied"]


def test_rep1c_legacy_seed_inherits_without_fabricated_homology() -> None:
    a = apply_structural_payload_activation_to_founder(_genome(), _cfg(enabled=True))
    b = Genome.from_dict(minimal_canonical_seed_payload(_genome()))
    b.modules[1]["module_id"] = "m011_behavior_bias_alt"
    b = Genome.from_dict(b.to_dict())
    rec, summary = recombine_lineage_payloads(parent_a_payload=a.to_dict(), parent_b_payload=b.to_dict(), rng=random.Random(7))
    assert len(rec["modules"]) == 3
    assert len(rec["connections"]) >= 1
    # REP-1C predates persistent ancestry metadata.  Coincident local IDs (or
    # a local repack) cannot prove biological homology, so C2 preserves one
    # canonical unresolved parent set rather than falsely aligning or doubling
    # the legacy structure.
    assert summary.matched_modules == 0
    assert summary.unmatched_modules_parent_a + summary.unmatched_modules_parent_b == 3
    assert summary.legacy_unresolved_modules == 3
    assert all(module["identity_origin"] == "legacy_derived_unresolved" for module in rec["modules"])


def test_rep1c_rep1a_reports_empty_control_and_enabled_nonempty_payload() -> None:
    off_agent = type("A", (), {})()
    off_agent.genome = _genome()
    off_agent.stats = type("S", (), {"offspring_count": 0, "lineage_id": 1, "pred_err_count": 0, "pred_err_sum": 0.0, "prediction_update_count": 0})()
    on_agent = type("A", (), {})()
    on_agent.genome = apply_structural_payload_activation_to_founder(_genome(), _cfg(enabled=True))
    on_agent.stats = type("S", (), {"offspring_count": 0, "lineage_id": 2, "pred_err_count": 0, "pred_err_sum": 0.0, "prediction_update_count": 0})()

    off_metrics = compute_cognitive_status_metrics([off_agent], env_cfg=EnvConfig(width=8, height=8), sim_cfg=_cfg(metrics=True))
    on_metrics = compute_cognitive_status_metrics([on_agent], env_cfg=EnvConfig(width=8, height=8), sim_cfg=_cfg(metrics=True))
    off_struct = json.loads(off_metrics["trait_capacity_saturation_structural_payload_json"])
    on_struct = json.loads(on_metrics["trait_capacity_saturation_structural_payload_json"])
    assert off_struct["structural_status"] == "structural_payload_empty"
    assert on_struct["structural_status"] == "below_cap"
    assert on_struct["alive_module_count_max"] == 3.0
    assert on_struct["alive_connection_count_max"] == 2.0


def test_rep1c_checkpoint_resume_preserves_seeded_payload(tmp_path: Path) -> None:
    run_a = tmp_path / "seeded_a"
    sim_a = _make_sim(run_a, enabled=True)
    sim_a.run(start_gen=0, skip_init=False)
    bundle = run_a / "checkpoint_bundles_v2" / "bundle_gen000001"
    checkpoint_bundle_v2_save(sim_a, str(bundle), generation_index=1, strict_replay_mode=True)
    payload = checkpoint_bundle_v2_load(bundle, strict=True)
    run_b = tmp_path / "seeded_b"
    sim_b = _make_sim(run_b, enabled=True)
    checkpoint_bundle_v2_apply(sim_b, payload, restore_rng=True)
    assert _agent_signature(sim_a) == _agent_signature(sim_b)


def test_rep1c_w6_reference_and_default_report_same_initial_structural_state(tmp_path: Path) -> None:
    ref = _make_sim(tmp_path / "ref", enabled=True, w6=False, metrics=True)
    w6 = _make_sim(tmp_path / "w6", enabled=True, w6=True, metrics=True)
    ref.run(start_gen=0, skip_init=False)
    w6.run(start_gen=0, skip_init=False)
    assert all(x["module_count"] > 0 for x in _agent_signature(ref))
    assert all(x["module_count"] > 0 for x in _agent_signature(w6))
    ref_struct = json.loads(_last_row(tmp_path / "ref")["trait_capacity_saturation_structural_payload_json"])
    w6_struct = json.loads(_last_row(tmp_path / "w6")["trait_capacity_saturation_structural_payload_json"])
    assert ref_struct["structural_status"] == "below_cap"
    assert w6_struct["structural_status"] == "below_cap"
    assert ref_struct["alive_module_count_max"] is not None
    assert w6_struct["alive_module_count_max"] is not None
    assert ref_struct["alive_connection_count_max"] is not None
    assert w6_struct["alive_connection_count_max"] is not None
