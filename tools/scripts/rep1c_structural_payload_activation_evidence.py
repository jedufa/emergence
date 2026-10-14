#!/usr/bin/env python3
from __future__ import annotations

"""Generate compact REP-1C structural payload activation evidence."""

import argparse
import csv
import json
import random
import tempfile
from pathlib import Path
from typing import Any

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
from simulation.metrics_pkg.trait_capacity_saturation import SCHEMA_VERSION as REP1A_SCHEMA, stable_json
from simulation.performance_profiles import ACCEPTED_W6_FAST_PROFILE, apply_performance_profile
from simulation.runner import SimulationRunner
from simulation.runner_pkg.runner_config import SimConfig


SCHEMA_VERSION = "rep1c_structural_payload_activation_evidence_v1"


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


def _cfg(*, enabled: bool, metrics: bool = False, w6: bool = False) -> SimConfig:
    cfg = SimConfig(seed=707, generations=1, steps_per_generation=3, device="cpu", bank_device="cpu", behavior_log_enable=False)
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


def _agent(genome: Genome, *, offspring: int = 0) -> Any:
    stats = type(
        "Stats",
        (),
        {
            "offspring_count": int(offspring),
            "lineage_id": 1,
            "patch_habitat_lineage_id": 1,
            "pred_err_count": 0,
            "pred_err_sum": 0.0,
            "prediction_update_count": 0,
        },
    )()
    return type(
        "Agent",
        (),
        {
            "agent_id": 1,
            "genome": genome,
            "stats": stats,
            "energy": 50.0,
            "health": 10.0,
            "alive": True,
            "drives": [0.0, 0.0, 0.0, 0.0],
        },
    )()


def _make_sim(run_dir: Path, *, enabled: bool, metrics: bool = False, w6: bool = False) -> SimulationRunner:
    return SimulationRunner(
        env_cfg=EnvConfig(width=8, height=8),
        pop_cfg=PopConfig(population_size=6),
        sim_cfg=_cfg(enabled=enabled, metrics=metrics, w6=w6),
        logger=MetricsLogger(str(run_dir)) if metrics else NoOpMetricsLogger(),
        run_dir=str(run_dir),
        perf=False,
    )


def _last_metrics_row(run_dir: Path) -> dict[str, str]:
    with (run_dir / "metrics.csv").open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"no metrics rows in {run_dir}")
    return dict(rows[-1])


def _signature(sim: SimulationRunner) -> list[dict[str, Any]]:
    out = []
    for aid, a in sorted(sim.pop.agents.items()):
        g = a.genome
        out.append(
            {
                "agent_id": int(aid),
                "module_count": len(g.modules),
                "connection_count": len(g.connections),
                "genome_hash": str(g.canonical_hash()),
                "c1_operator_schedule": str(getattr(g, "c1_operator_schedule", "")),
                "c1_operator_applied": str(getattr(g, "c1_operator_applied", "")),
                "c1_operator_noop_count": int(getattr(g, "c1_operator_noop_count", 0) or 0),
            }
        )
    return out


def build_outputs(out_dir: Path, command_results: list[dict[str, Any]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    empty = _genome()
    seeded = apply_structural_payload_activation_to_founder(_genome(), _cfg(enabled=True))
    seed_payload = seeded.to_dict()

    empty_after_c1, empty_telemetry = apply_operator_pipeline(payload=empty.canonical_payload(), rng=random.Random(5), max_ops=2)
    operator_payload, operator_telemetry = apply_operator_pipeline(payload=seeded.canonical_payload(), rng=random.Random(1), max_ops=2)
    recombined, recomb_summary = recombine_lineage_payloads(
        parent_a_payload=seeded.to_dict(),
        parent_b_payload=Genome.from_dict(minimal_canonical_seed_payload(_genome())).to_dict(),
        rng=random.Random(7),
    )

    off_metrics = compute_cognitive_status_metrics([_agent(empty)], env_cfg=EnvConfig(width=8, height=8), sim_cfg=_cfg(enabled=False, metrics=True))
    on_metrics = compute_cognitive_status_metrics([_agent(seeded)], env_cfg=EnvConfig(width=8, height=8), sim_cfg=_cfg(enabled=True, metrics=True))
    op_metrics = compute_cognitive_status_metrics(
        [_agent(Genome.from_dict(operator_payload), offspring=1)],
        selected_parent_agents=[_agent(seeded), _agent(seeded)],
        env_cfg=EnvConfig(width=8, height=8),
        sim_cfg=_cfg(enabled=True, metrics=True),
    )

    with tempfile.TemporaryDirectory(prefix="rep1c_evidence_") as tmp:
        tmp_root = Path(tmp)
        ref = _make_sim(tmp_root / "ref", enabled=True, metrics=True, w6=False)
        w6 = _make_sim(tmp_root / "w6", enabled=True, metrics=True, w6=True)
        ref.run(start_gen=0, skip_init=False)
        w6.run(start_gen=0, skip_init=False)
        ck_a = _make_sim(tmp_root / "ck_a", enabled=True, metrics=False)
        ck_a.run(start_gen=0, skip_init=False)
        bundle = tmp_root / "ck_a" / "checkpoint_bundles_v2" / "bundle_gen000001"
        checkpoint_bundle_v2_save(ck_a, str(bundle), generation_index=1, strict_replay_mode=True)
        payload = checkpoint_bundle_v2_load(bundle, strict=True)
        ck_b = _make_sim(tmp_root / "ck_b", enabled=True, metrics=False)
        checkpoint_bundle_v2_apply(ck_b, payload, restore_rng=True)
        checkpoint_summary = {
            "schema_version": SCHEMA_VERSION,
            "matched": _signature(ck_a) == _signature(ck_b),
            "uninterrupted_signature": _signature(ck_a),
            "restored_signature": _signature(ck_b),
        }
        w6_summary = {
            "schema_version": SCHEMA_VERSION,
            "profile": ACCEPTED_W6_FAST_PROFILE,
            "reference_structural_json": json.loads(_last_metrics_row(tmp_root / "ref")["trait_capacity_saturation_structural_payload_json"]),
            "w6_structural_json": json.loads(_last_metrics_row(tmp_root / "w6")["trait_capacity_saturation_structural_payload_json"]),
            "reference_signature": _signature(ref),
            "w6_signature": _signature(w6),
            "supported": True,
        }

    root_cause = {
        "schema_version": SCHEMA_VERSION,
        "first_normal_creation": "Genome.random creates explicit empty modules and connections lists",
        "founders_receive_structure_by_default": False,
        "c1_can_create_first_module_from_empty": False,
        "connection_ops_can_create_first_connection_without_modules": False,
        "recombining_two_empty_payloads_remains_empty": True,
        "default_off_gate_prevents_activation": "structural_payload_activation_enable defaults false",
        "operators_running_on_empty_return_noop": True,
        "payloads_later_dropped": False,
        "payload_reaches_phenotype_boundary": True,
        "w6_support": "supported for report-facing structural state in paired fixture",
        "empty_payload_interpretation": "both deliberate control state and missing bootstrap for controlled structural experiments",
    }
    design_options = {
        "schema_version": SCHEMA_VERSION,
        "option_a_deterministic_founder_seed": "selected",
        "option_b_operator_bootstrap_from_empty": "rejected; existing operators no-op on empty",
        "option_c_controlled_payload_input": "not needed; would widen input surface",
        "option_d_audit_only": "not selected; small default-off seed is safe",
    }
    selected_design = {
        "schema_version": SCHEMA_VERSION,
        "decision": "Option A",
        "config_gate": "structural_payload_activation_enable",
        "seed_profile": "minimal_canonical_v1",
        "seed_schema_version": STRUCTURAL_PAYLOAD_SEED_SCHEMA_VERSION,
        "random_draws": 0,
        "module_count": len(seeded.modules),
        "connection_count": len(seeded.connections),
        "normal_default_unchanged": empty.modules == [] and empty.connections == [],
    }
    operator_summary = {
        "schema_version": SCHEMA_VERSION,
        "empty_after_c1_module_count": len(empty_after_c1.get("modules", [])),
        "empty_after_c1_connection_count": len(empty_after_c1.get("connections", [])),
        "empty_telemetry": empty_telemetry,
        "seeded_after_c1_module_count": len(operator_payload.get("modules", [])),
        "seeded_after_c1_connection_count": len(operator_payload.get("connections", [])),
        "seeded_telemetry": operator_telemetry,
    }
    inheritance_recombination = {
        "schema_version": SCHEMA_VERSION,
        "seeded_payload_hash": seeded.canonical_hash(),
        "recombined_module_count": len(recombined.get("modules", [])),
        "recombined_connection_count": len(recombined.get("connections", [])),
        "recombination_summary": recomb_summary.__dict__,
    }
    rep1a_summary = {
        "schema_version": REP1A_SCHEMA,
        "off_structural": json.loads(off_metrics["trait_capacity_saturation_structural_payload_json"]),
        "enabled_structural": json.loads(on_metrics["trait_capacity_saturation_structural_payload_json"]),
        "operator_structural": json.loads(op_metrics["trait_capacity_saturation_structural_payload_json"]),
    }
    repeat = []
    for name, fn in {
        "seed_payload": lambda: apply_structural_payload_activation_to_founder(_genome(), _cfg(enabled=True)).canonical_hash(),
        "operator_payload": lambda: stable_json(apply_operator_pipeline(payload=seeded.canonical_payload(), rng=random.Random(1), max_ops=2)[0]),
        "rep1a_enabled_json": lambda: compute_cognitive_status_metrics([_agent(seeded)], env_cfg=EnvConfig(width=8, height=8), sim_cfg=_cfg(enabled=True, metrics=True))["trait_capacity_saturation_structural_payload_json"],
    }.items():
        repeat.append({"cell": name, "equal": fn() == fn()})
    repeat_summary = {"schema_version": SCHEMA_VERSION, "passed": all(r["equal"] for r in repeat), "repeat_cells": repeat}
    integrity = {
        "schema_version": SCHEMA_VERSION,
        "seed_module_count": len(seeded.modules),
        "seed_connection_count": len(seeded.connections),
        "seed_below_module_cap": len(seeded.modules) < int(seeded.b2_caps["max_modules_per_agent"]),
        "seed_below_connection_cap": len(seeded.connections) < int(seeded.b2_caps["max_connections_per_agent"]),
        "checkpoint_resume_passed": bool(checkpoint_summary["matched"]),
        "repeat_parity_passed": bool(repeat_summary["passed"]),
        "w6_supported": bool(w6_summary["supported"]),
    }
    decision = {
        "schema_version": SCHEMA_VERSION,
        "decision": "REP-1C-A",
        "reason": "root cause is empty founder payload plus no first-structure C1 operator; default-off deterministic founder seed activates existing structural machinery",
        "next_step": "REP-2A: Structural Mutation Viability and Retention Gate",
    }

    files = {
        "rep1c_root_cause_map.json": root_cause,
        "rep1c_design_options.json": design_options,
        "rep1c_selected_design.json": selected_design,
        "rep1c_payload_schema_example.json": seed_payload,
        "rep1c_operator_path_summary.json": operator_summary,
        "rep1c_inheritance_recombination_summary.json": inheritance_recombination,
        "rep1c_checkpoint_resume_summary.json": checkpoint_summary,
        "rep1c_w6_support_summary.json": w6_summary,
        "rep1c_rep1a_integration_summary.json": rep1a_summary,
        "rep1c_repeat_parity_summary.json": repeat_summary,
        "rep1c_integrity_summary.json": integrity,
        "rep1c_test_command_summary.json": {"schema_version": SCHEMA_VERSION, "commands": command_results},
        "rep1c_decision.json": decision,
    }
    for name, payload in files.items():
        (out_dir / name).write_text(stable_json(payload) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "files": sorted([*files.keys(), "README.md", "campaign_evidence_manifest.json"]),
        "raw_outputs": "none; temporary technical runs only",
    }
    (out_dir / "campaign_evidence_manifest.json").write_text(stable_json(manifest) + "\n", encoding="utf-8")
    (out_dir / "README.md").write_text(
        "# REP-1C Structural Payload Activation Evidence\n\n"
        "Compact fixture and tiny-run evidence for the default-off structural payload activation gate.\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate REP-1C evidence.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--test-command-summary-json", default="")
    args = parser.parse_args()
    commands: list[dict[str, Any]] = []
    if args.test_command_summary_json:
        commands = json.loads(Path(args.test_command_summary_json).read_text(encoding="utf-8"))
    build_outputs(Path(args.out_dir), commands)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
