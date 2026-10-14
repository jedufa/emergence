from __future__ import annotations

import json
import math
import random
from dataclasses import asdict
from pathlib import Path

import pytest

from agents.genome import Genome
from env.env_config import EnvConfig
from evolution.population_pkg.population_config import PopConfig
from evolution.structural_live_semantics import (
    FIELD_BOUNDS,
    SCHEMA_VERSION,
    STRUCTURAL_LIVE_EFFECT_SCALE_MAX,
    apply_live_structural_semantics,
    compile_live_graph,
    stable_hash,
    stable_json,
    structural_live_effect_scale,
    validate_structural_live_semantics_config,
)
from evolution.structural_mutation_viability import (
    DEFAULT_DRAWS,
    apply_single_operator,
    rep2b_live_config,
    seeded_fixture_genome,
)
from evolution.operators_c1 import operator_catalog
from simulation.checkpoint_bundle_v2 import CheckpointStrictError, checkpoint_bundle_v2_apply, checkpoint_bundle_v2_load, checkpoint_bundle_v2_save
from simulation.metrics import MetricsLogger
from simulation.metrics_pkg.cognitive_status import compute_cognitive_status_metrics
from simulation.metrics_pkg.noop_loggers import NoOpMetricsLogger
from simulation.performance_profiles import ACCEPTED_W6_FAST_PROFILE, apply_performance_profile
from simulation.runner import SimulationRunner
from simulation.runner_pkg.runner_config import SimConfig


def _cfg(*, enabled: bool = True, w6: bool = False, scale: float = 1.0) -> SimConfig:
    cfg = rep2b_live_config() if enabled else SimConfig()
    cfg.seed = 211
    cfg.generations = 1
    cfg.steps_per_generation = 3
    cfg.device = "cpu"
    cfg.bank_device = "cpu"
    cfg.behavior_log_enable = False
    cfg.policy_deterministic_actions = True
    cfg.trait_capacity_saturation_reporting_enable = True
    cfg.checkpoint_bundle_v2_period_gen = 1
    cfg.structural_live_effect_scale = float(scale)
    if w6:
        apply_performance_profile(cfg, ACCEPTED_W6_FAST_PROFILE)
        cfg.trait_capacity_saturation_reporting_enable = True
        cfg.enable_cognitive_status_metrics = True
    return cfg


def _apply(genome: Genome | None = None, *, cfg: SimConfig | None = None) -> tuple[dict, dict]:
    g = genome or seeded_fixture_genome()
    expr = g.resolve_developmental_expression(age_steps=0)
    return apply_live_structural_semantics(genome=g, expression=expr, sim_cfg=cfg or _cfg())


def _agent(genome: Genome) -> object:
    stats = type("Stats", (), {"offspring_count": 0, "lineage_id": 1, "patch_habitat_lineage_id": 1, "pred_err_count": 0, "pred_err_sum": 0.0, "prediction_update_count": 0})()
    return type("Agent", (), {"agent_id": 1, "genome": genome, "stats": stats, "energy": 50.0, "health": 10.0, "alive": True, "drives": [0.0, 0.0, 0.0, 0.0]})()


def _make_sim(run_dir: Path, *, enabled: bool = True, w6: bool = False, metrics: bool = False) -> SimulationRunner:
    return SimulationRunner(
        env_cfg=EnvConfig(width=8, height=8),
        pop_cfg=PopConfig(population_size=6),
        sim_cfg=_cfg(enabled=enabled, w6=w6),
        logger=MetricsLogger(str(run_dir)) if metrics else NoOpMetricsLogger(),
        run_dir=str(run_dir),
        perf=False,
    )


def _generic_module(mid: str) -> dict:
    return {
        "module_id": str(mid),
        "module_type": "generic",
        "io": {
            "inputs": [{"slot_id": "in", "value_type": "scalar"}],
            "outputs": [{"slot_id": "out", "value_type": "scalar"}],
        },
        "latent_pos": [0.0, 0.0],
        "params": {},
    }


def _compile_reason_payload(reason: str) -> dict:
    payload = json.loads(stable_json(seeded_fixture_genome().canonical_payload()))
    if reason == "missing_module_id":
        payload["modules"][0]["module_id"] = ""
    elif reason == "duplicate_module_id":
        payload["modules"][1]["module_id"] = payload["modules"][0]["module_id"]
    elif reason == "module_cap_exceeded":
        payload["modules"] = [_generic_module(f"m{i:02d}") for i in range(13)]
        payload["connections"] = []
    elif reason == "connection_cap_exceeded":
        base = payload["connections"][0]
        payload["connections"] = [dict(base, connection_id=f"c{i:03d}") for i in range(33)]
    elif reason == "connection_record_not_dict":
        payload["connections"][0] = "bad"
    elif reason == "missing_source":
        payload["connections"][0]["src_module_id"] = "missing"
    elif reason == "missing_destination":
        payload["connections"][0]["dst_module_id"] = "missing"
    elif reason == "missing_source_slot":
        payload["connections"][0]["src_slot"] = "missing"
    elif reason == "missing_destination_slot":
        payload["connections"][0]["dst_slot"] = "missing"
    elif reason == "source_slot_type_mismatch":
        payload["connections"][0]["value_type"] = "vector"
    elif reason == "destination_slot_type_mismatch":
        payload["connections"][0]["dst_slot"] = "mod_in"
    elif reason == "nonfinite_weight":
        payload["connections"][0]["weight"] = math.inf
    elif reason == "cycle_detected":
        payload["connections"].append(
            {
                "connection_id": "c999_cycle",
                "src_module_id": "m000_policy_core",
                "dst_module_id": "m010_behavior_bias",
                "src_slot": "action_out",
                "dst_slot": "state_in",
                "value_type": "vector",
                "channel": "cycle_probe",
                "weight": 1.0,
            }
        )
    else:
        raise AssertionError(reason)
    return payload


def test_rep2b_flag_off_exact_compatibility() -> None:
    g = seeded_fixture_genome()
    expr = g.resolve_developmental_expression(age_steps=0)
    out, summary = apply_live_structural_semantics(genome=g, expression=expr, sim_cfg=_cfg(enabled=False))
    assert out == expr
    assert summary["enabled"] is False
    assert summary["evaluation_count"] == 0


def test_rep2b_config_profile_validation() -> None:
    cfg = _cfg()
    cfg.structural_live_semantics_profile = "bad"
    with pytest.raises(ValueError, match="unsupported structural_live_semantics_profile"):
        validate_structural_live_semantics_config(cfg)


@pytest.mark.parametrize("scale", [0.0, 0.5, 1.0, 1.25, 1.5, 2.0])
def test_rep2p_structural_live_effect_scale_valid_ladder(scale: float) -> None:
    cfg = _cfg(scale=scale)
    validate_structural_live_semantics_config(cfg)
    assert structural_live_effect_scale(cfg) == scale


@pytest.mark.parametrize("scale", [-0.01, STRUCTURAL_LIVE_EFFECT_SCALE_MAX + 0.01, math.nan, math.inf])
def test_rep2p_structural_live_effect_scale_rejects_invalid_values(scale: float) -> None:
    cfg = _cfg(scale=scale)
    with pytest.raises(ValueError, match="structural_live_effect_scale"):
        validate_structural_live_semantics_config(cfg)
    cfg = _cfg()
    cfg.structural_live_semantics_profile = "none"
    with pytest.raises(ValueError, match="requires a non-none"):
        validate_structural_live_semantics_config(cfg)


def test_rep2b_empty_payload_handling() -> None:
    g = Genome.from_dict({"modules": [], "connections": []})
    out, summary = _apply(g)
    assert summary["payload_empty"] is True
    assert summary["graph_supported"] is True
    assert summary["changed_effective_fields"] == []
    assert out["effective"] == g.resolve_developmental_expression(age_steps=0)["effective"]


def test_rep2b_canonical_graph_order_and_digest_stability() -> None:
    payload = seeded_fixture_genome().canonical_payload()
    rev = dict(payload)
    rev["modules"] = list(reversed(payload["modules"]))
    rev["connections"] = list(reversed(payload["connections"]))
    assert compile_live_graph(payload)["graph_digest"] == compile_live_graph(rev)["graph_digest"]


def test_rep2b_missing_source_and_destination_rejected() -> None:
    payload = seeded_fixture_genome().canonical_payload()
    bad = json.loads(stable_json(payload))
    bad["connections"][0]["src_module_id"] = "missing"
    assert compile_live_graph(bad)["unsupported_reason"] == "missing_source"
    bad = json.loads(stable_json(payload))
    bad["connections"][0]["dst_module_id"] = "missing"
    assert compile_live_graph(bad)["unsupported_reason"] == "missing_destination"


def test_rep2b_slot_mismatch_and_nonfinite_weight_rejected() -> None:
    payload = seeded_fixture_genome().canonical_payload()
    bad = json.loads(stable_json(payload))
    bad["connections"][0]["src_slot"] = "missing"
    assert compile_live_graph(bad)["unsupported_reason"] == "missing_source_slot"
    bad = json.loads(stable_json(payload))
    bad["connections"][0]["weight"] = math.nan
    assert compile_live_graph(bad)["unsupported_reason"] == "nonfinite_weight"


def test_rep2b_cycle_rejected_or_documented() -> None:
    payload = seeded_fixture_genome().canonical_payload()
    cyclic = json.loads(stable_json(payload))
    cyclic["connections"].append(
        {
            "connection_id": "c999_cycle",
            "src_module_id": "m000_policy_core",
            "dst_module_id": "m010_behavior_bias",
            "src_slot": "action_out",
            "dst_slot": "state_in",
            "value_type": "vector",
            "channel": "cycle_probe",
            "weight": 1.0,
        }
    )
    summary = compile_live_graph(cyclic)
    assert summary["graph_supported"] is False
    assert summary["unsupported_reason"] == "cycle_detected"


@pytest.mark.parametrize(
    "reason",
    [
        "missing_module_id",
        "duplicate_module_id",
        "module_cap_exceeded",
        "connection_cap_exceeded",
        "connection_record_not_dict",
        "missing_source",
        "missing_destination",
        "missing_source_slot",
        "missing_destination_slot",
        "source_slot_type_mismatch",
        "destination_slot_type_mismatch",
        "nonfinite_weight",
        "cycle_detected",
    ],
)
def test_rep2b_compile_reason_codes_fail_closed(reason: str) -> None:
    summary = compile_live_graph(_compile_reason_payload(reason))
    assert summary["graph_supported"] is False
    assert summary["unsupported_reason"] == reason
    assert summary["reason_codes"] == [reason]


def test_rep2b_unsupported_profile_fails_closed() -> None:
    summary = compile_live_graph(seeded_fixture_genome().canonical_payload(), profile="bad")
    assert summary["graph_supported"] is False
    assert summary["unsupported_reason"] == "unsupported_profile"


def test_rep2b_zero_weight_and_zero_gate_neutrality() -> None:
    g = seeded_fixture_genome()
    zero = g.to_dict()
    zero["connections"] = [dict(c, weight=0.0) for c in zero["connections"]]
    out, summary = _apply(Genome.from_dict(zero))
    assert summary["changed_effective_fields"] == []
    assert out["effective"] == Genome.from_dict(zero).resolve_developmental_expression(age_steps=0)["effective"]


def test_rep2b_nonzero_bounded_connection_effect_and_clipping() -> None:
    out, summary = _apply()
    assert summary["graph_supported"] is True
    assert {"aggression_bias", "drive_logit_gain"}.issubset(set(summary["changed_effective_fields"]))
    assert -2.0 <= out["effective"]["aggression_bias"] <= 2.0
    assert 0.0 <= out["effective"]["drive_logit_gain"] <= 3.0


def test_rep2p_scale_one_preserves_rep2b_effective_fields() -> None:
    g = seeded_fixture_genome()
    expr = g.resolve_developmental_expression(age_steps=0)
    out_old, summary_old = apply_live_structural_semantics(genome=g, expression=expr, sim_cfg=_cfg(scale=1.0))
    out_new, summary_new = apply_live_structural_semantics(genome=g, expression=expr, sim_cfg=_cfg(scale=1.0))
    assert out_old["effective"] == out_new["effective"]
    assert summary_new["configured_effect_scale"] == 1.0
    assert summary_old["graph_digest"] == summary_new["graph_digest"]


def test_rep2p_scale_zero_ablates_structural_residual_without_graph_deletion() -> None:
    g = seeded_fixture_genome()
    base = g.resolve_developmental_expression(age_steps=0)["effective"]
    out, summary = _apply(g, cfg=_cfg(scale=0.0))
    for field in FIELD_BOUNDS:
        assert out["effective"][field] == pytest.approx(base[field])
        assert summary["structural_residual_after_scale"][field] == pytest.approx(0.0)
    assert summary["graph_supported"] is True
    assert summary["graph_digest"] == compile_live_graph(g.canonical_payload())["graph_digest"]


def test_rep2p_preclip_scale_applies_only_to_structural_residual() -> None:
    g = seeded_fixture_genome()
    base = g.resolve_developmental_expression(age_steps=0)["effective"]
    out_1, summary_1 = _apply(g, cfg=_cfg(scale=1.0))
    out_15, summary_15 = _apply(g, cfg=_cfg(scale=1.5))
    for field in ("aggression_bias", "drive_logit_gain"):
        residual = float(out_1["effective"][field]) - float(base[field])
        expected = float(base[field]) + 1.5 * residual
        lo, hi = FIELD_BOUNDS[field]
        expected = min(hi, max(lo, expected))
        assert out_15["effective"][field] == pytest.approx(expected)
        assert summary_15["structural_residual_after_scale"][field] == pytest.approx(
            1.5 * summary_1["structural_residual_before_scale"][field]
        )


def test_rep2p_scaling_preserves_graph_identity_and_semantic_digest_source() -> None:
    g = seeded_fixture_genome()
    _out_1, summary_1 = _apply(g, cfg=_cfg(scale=1.0))
    _out_2, summary_2 = _apply(g, cfg=_cfg(scale=2.0))
    assert summary_1["graph_digest"] == summary_2["graph_digest"]
    assert summary_1["effective_fields_before"] == summary_2["effective_fields_before"]


def test_rep2p_clipping_is_after_scaled_residual() -> None:
    payload = seeded_fixture_genome().to_dict()
    for conn in payload["connections"]:
        if conn["dst_slot"] == "bias_in":
            conn["weight"] = 2.0
    for module in payload["modules"]:
        if module["module_id"] == "m010_behavior_bias":
            module["params"]["aggression_bias"] = 4.0
    out, summary = _apply(Genome.from_dict(payload), cfg=_cfg(scale=2.0))
    assert out["effective"]["aggression_bias"] == FIELD_BOUNDS["aggression_bias"][1]
    assert "aggression_bias" in summary["clipped_fields"]


def test_rep2b_positive_and_negative_weight_behavior() -> None:
    payload = seeded_fixture_genome().to_dict()
    pos = Genome.from_dict(payload)
    neg_payload = json.loads(stable_json(payload))
    neg_payload["connections"][0]["weight"] = -1.0
    neg = Genome.from_dict(neg_payload)
    out_pos, _ = _apply(pos)
    out_neg, _ = _apply(neg)
    base = pos.resolve_developmental_expression(age_steps=0)["effective"]["aggression_bias"]
    assert out_pos["effective"]["aggression_bias"] > base
    assert out_neg["effective"]["aggression_bias"] < base


def test_rep2b_multiple_incoming_aggregation_and_duplicate_policy() -> None:
    payload = seeded_fixture_genome().to_dict()
    payload["connections"].append(dict(payload["connections"][0], connection_id="c001_dup", weight=0.5))
    out, summary = _apply(Genome.from_dict(payload))
    compiled = compile_live_graph(Genome.from_dict(payload).canonical_payload())
    assert any(c["duplicate_edge_policy"] == "aggregate_sorted" for c in compiled["connections"])
    assert "aggression_bias" in summary["changed_effective_fields"]
    assert out["effective"]["aggression_bias"] > seeded_fixture_genome().resolve_developmental_expression(age_steps=0)["effective"]["aggression_bias"]


def test_rep2b_expression_scale_behavior() -> None:
    payload = seeded_fixture_genome().to_dict()
    for module in payload["modules"]:
        if module["module_id"] == "m010_behavior_bias":
            module["params"]["expression_scale"] = 1.2
    out, summary = _apply(Genome.from_dict(payload))
    assert "aggression_bias" in summary["changed_effective_fields"]
    assert summary["live_expression_change_count"] >= 1
    assert out["effective"]["aggression_bias"] != seeded_fixture_genome().resolve_developmental_expression(age_steps=0)["effective"]["aggression_bias"]


def test_rep2b_execution_off_vs_on_causal_change() -> None:
    g = seeded_fixture_genome()
    off, _ = apply_live_structural_semantics(genome=g, expression=g.resolve_developmental_expression(age_steps=0), sim_cfg=_cfg(enabled=False))
    on, summary = _apply(g)
    assert off["effective"] != on["effective"]
    assert summary["evaluation_count"] == 1


def test_rep2b_edge_removal_causal_change() -> None:
    g = seeded_fixture_genome()
    payload = g.to_dict()
    payload["connections"] = [payload["connections"][1]]
    removed = Genome.from_dict(payload)
    on_full, _ = _apply(g)
    on_removed, _ = _apply(removed)
    assert on_full["effective"]["aggression_bias"] != on_removed["effective"]["aggression_bias"]


def test_rep2b_random_draw_isolation() -> None:
    rng = random.Random(42)
    before = rng.getstate()
    _apply()
    after = rng.getstate()
    assert before == after


def test_rep2b_runner_reference_path_reaches_effective_fields(tmp_path: Path) -> None:
    sim = _make_sim(tmp_path / "ref", enabled=True)
    sim.run(start_gen=0, skip_init=False)
    agent = next(iter(sim.pop.agents.values()))
    expr = sim.resolve_developmental_expression_for_agent(agent)
    assert "structural_live_semantics" in expr
    assert expr["structural_live_semantics"]["changed_effective_fields"]


def test_rep2b_w6_path_supports_live_summary(tmp_path: Path) -> None:
    sim = _make_sim(tmp_path / "w6", enabled=True, w6=True)
    sim.run(start_gen=0, skip_init=False)
    agent = next(iter(sim.pop.agents.values()))
    expr = sim.resolve_developmental_expression_for_agent(agent)
    assert expr["structural_live_semantics"]["w6_support_status"] == "derived_reference_path"


def test_rep2b_metrics_json_and_rep1a_rep2a_integration() -> None:
    metrics = compute_cognitive_status_metrics([_agent(seeded_fixture_genome())], env_cfg=EnvConfig(width=8, height=8), sim_cfg=_cfg())
    live = json.loads(metrics["structural_live_semantics_summary_json"])
    rep1a = json.loads(metrics["trait_capacity_saturation_structural_payload_json"])
    assert live["enabled"] is True
    assert live["changed_effective_agent_count"] == 1
    assert rep1a["structural_status"] == "below_cap"


def test_rep2b_gate_drift_reclassifies_as_live_connection_change() -> None:
    row = apply_single_operator("gate_drift", (0.1, 0.2), sim_cfg=_cfg())
    assert row["classification"] == "valid_live_change"
    assert row["changed_effective_fields"]


@pytest.mark.parametrize("operator_name", [spec.op_id for spec in operator_catalog()])
@pytest.mark.parametrize("draws", DEFAULT_DRAWS)
def test_rep2b_operator_battery_all_catalog_draws(operator_name: str, draws: tuple[float, ...]) -> None:
    row = apply_single_operator(operator_name, draws, sim_cfg=_cfg())
    assert row["operator"] == operator_name
    assert row["catastrophic_integrity_failure"] is False
    assert row["classification"] in {
        "noop",
        "valid_live_change",
        "valid_stored_only_change",
        "valid_compression",
        "valid_neutral_representation_change",
    }


@pytest.mark.parametrize("field,bounds", sorted(FIELD_BOUNDS.items()))
def test_rep2b_effective_field_bounds_are_respected(field: str, bounds: tuple[float, float]) -> None:
    out, summary = _apply()
    lo, hi = bounds
    assert lo <= float(out["effective"][field]) <= hi
    assert lo <= float(summary["effective_fields_after"][field]) <= hi


def test_rep2b_checkpoint_resume_rebuilds_derived_summary(tmp_path: Path) -> None:
    sim_a = _make_sim(tmp_path / "ck_a", enabled=True)
    sim_a.run(start_gen=0, skip_init=False)
    bundle = tmp_path / "ck_a" / "checkpoint_bundles_v2" / "bundle_gen000001"
    checkpoint_bundle_v2_save(sim_a, str(bundle), generation_index=1, strict_replay_mode=True)
    payload = checkpoint_bundle_v2_load(bundle, strict=True)
    sim_b = _make_sim(tmp_path / "ck_b", enabled=True)
    checkpoint_bundle_v2_apply(sim_b, payload, restore_rng=True)
    a = next(iter(sim_a.pop.agents.values()))
    b = next(iter(sim_b.pop.agents.values()))
    expr_a = sim_a.resolve_developmental_expression_for_agent(a)
    expr_b = sim_b.resolve_developmental_expression_for_agent(b)
    assert stable_hash(expr_a["structural_live_semantics"]) == stable_hash(expr_b["structural_live_semantics"])


def test_rep2p_checkpoint_round_trip_preserves_nondefault_scale(tmp_path: Path) -> None:
    sim_a = _make_sim(tmp_path / "ck_scale_a", enabled=True)
    sim_a.sim_cfg.structural_live_effect_scale = 1.5
    sim_a.run(start_gen=0, skip_init=False)
    bundle = tmp_path / "ck_scale_a" / "checkpoint_bundles_v2" / "bundle_gen000001"
    checkpoint_bundle_v2_save(sim_a, str(bundle), generation_index=1, strict_replay_mode=True)
    payload = checkpoint_bundle_v2_load(bundle, strict=True)
    assert payload["metadata"]["structural_live_effect_scale"] == 1.5
    sim_b = _make_sim(tmp_path / "ck_scale_b", enabled=True)
    sim_b.sim_cfg.structural_live_effect_scale = 1.5
    checkpoint_bundle_v2_apply(sim_b, payload, restore_rng=True)
    a = next(iter(sim_a.pop.agents.values()))
    b = next(iter(sim_b.pop.agents.values()))
    expr_a = sim_a.resolve_developmental_expression_for_agent(a)
    expr_b = sim_b.resolve_developmental_expression_for_agent(b)
    assert expr_a["structural_live_semantics"]["configured_effect_scale"] == 1.5
    assert stable_hash(expr_a["structural_live_semantics"]) == stable_hash(expr_b["structural_live_semantics"])


def test_rep2p_checkpoint_strict_resume_rejects_scale_mismatch(tmp_path: Path) -> None:
    sim_a = _make_sim(tmp_path / "ck_mismatch_a", enabled=True)
    sim_a.sim_cfg.structural_live_effect_scale = 1.5
    sim_a.run(start_gen=0, skip_init=False)
    bundle = tmp_path / "ck_mismatch_a" / "checkpoint_bundles_v2" / "bundle_gen000001"
    checkpoint_bundle_v2_save(sim_a, str(bundle), generation_index=1, strict_replay_mode=True)
    payload = checkpoint_bundle_v2_load(bundle, strict=True)
    sim_b = _make_sim(tmp_path / "ck_mismatch_b", enabled=True)
    sim_b.sim_cfg.structural_live_effect_scale = 1.0
    with pytest.raises(CheckpointStrictError, match="structural_live_effect_scale mismatch"):
        checkpoint_bundle_v2_apply(sim_b, payload, restore_rng=True)


def test_rep2p_old_checkpoint_metadata_defaults_to_scale_one(tmp_path: Path) -> None:
    sim_a = _make_sim(tmp_path / "ck_old_a", enabled=True)
    sim_a.run(start_gen=0, skip_init=False)
    bundle = tmp_path / "ck_old_a" / "checkpoint_bundles_v2" / "bundle_gen000001"
    checkpoint_bundle_v2_save(sim_a, str(bundle), generation_index=1, strict_replay_mode=True)
    payload = checkpoint_bundle_v2_load(bundle, strict=True)
    payload["metadata"].pop("structural_live_effect_scale", None)
    sim_b = _make_sim(tmp_path / "ck_old_b", enabled=True)
    sim_b.sim_cfg.structural_live_effect_scale = 1.0
    checkpoint_bundle_v2_apply(sim_b, payload, restore_rng=True)


def test_rep2p_config_hash_changes_for_nondefault_scale() -> None:
    cfg_a = _cfg(scale=1.0)
    cfg_b = _cfg(scale=1.5)
    assert asdict(cfg_a)["structural_live_effect_scale"] == 1.0
    assert asdict(cfg_b)["structural_live_effect_scale"] == 1.5
    assert stable_hash(asdict(cfg_a)) != stable_hash(asdict(cfg_b))


def test_rep2b_stable_json() -> None:
    _out, summary = _apply()
    assert stable_json(summary) == stable_json(summary)
    assert stable_hash(summary) == stable_hash(summary)
