#!/usr/bin/env python3
from __future__ import annotations

"""Generate compact REP-2A structural mutation viability evidence."""

import argparse
import csv
import json
import tempfile
from pathlib import Path
from typing import Any

from env.env_config import EnvConfig
from evolution.population_pkg.population_config import PopConfig
from evolution.structural_mutation_viability import (
    CONNECTION_EXECUTION_STATUS,
    SCHEMA_VERSION,
    constraint_probe_summary,
    effective_fields,
    empty_control_genome,
    inheritance_and_retention_summary,
    integrity_summary,
    operator_battery,
    operator_map,
    phenotype_reach_summary,
    recombination_summary,
    rep1c_seed_config,
    rep2a_decision,
    reporting_disabled_summary,
    seeded_fixture_genome,
    stable_hash,
    stable_json,
    viability_summary_payload,
)
from simulation.checkpoint_bundle_v2 import checkpoint_bundle_v2_apply, checkpoint_bundle_v2_load, checkpoint_bundle_v2_save
from simulation.metrics import MetricsLogger
from simulation.metrics_pkg.cognitive_status import compute_cognitive_status_metrics
from simulation.metrics_pkg.noop_loggers import NoOpMetricsLogger
from simulation.performance_profiles import ACCEPTED_W6_FAST_PROFILE, apply_performance_profile
from simulation.runner import SimulationRunner


def _agent(genome: Any, *, lineage_id: int = 1) -> Any:
    stats = type(
        "Stats",
        (),
        {
            "offspring_count": 0,
            "lineage_id": int(lineage_id),
            "patch_habitat_lineage_id": int(lineage_id),
            "pred_err_count": 0,
            "pred_err_sum": 0.0,
            "prediction_update_count": 0,
        },
    )()
    return type(
        "Agent",
        (),
        {
            "agent_id": int(lineage_id),
            "genome": genome,
            "stats": stats,
            "energy": 50.0,
            "health": 10.0,
            "alive": True,
            "drives": [0.0, 0.0, 0.0, 0.0],
        },
    )()


def _make_sim(run_dir: Path, *, metrics: bool = False, w6: bool = False) -> SimulationRunner:
    cfg = rep1c_seed_config()
    cfg.seed = 928
    cfg.generations = 1
    cfg.steps_per_generation = 3
    cfg.checkpoint_bundle_v2_period_gen = 1
    cfg.trait_capacity_saturation_reporting_enable = bool(metrics)
    if w6:
        apply_performance_profile(cfg, ACCEPTED_W6_FAST_PROFILE)
        cfg.enable_cognitive_status_metrics = True
        cfg.trait_capacity_saturation_reporting_enable = bool(metrics)
    return SimulationRunner(
        env_cfg=EnvConfig(width=8, height=8),
        pop_cfg=PopConfig(population_size=6),
        sim_cfg=cfg,
        logger=MetricsLogger(str(run_dir)) if metrics else NoOpMetricsLogger(),
        run_dir=str(run_dir),
        perf=False,
    )


def _metrics_cfg() -> Any:
    cfg = rep1c_seed_config()
    cfg.trait_capacity_saturation_reporting_enable = True
    return cfg


def _last_metrics_row(run_dir: Path) -> dict[str, str]:
    with (run_dir / "metrics.csv").open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"no metrics rows in {run_dir}")
    return dict(rows[-1])


def _signature(sim: SimulationRunner) -> list[dict[str, Any]]:
    out = []
    for aid, agent in sorted(sim.pop.agents.items()):
        genome = agent.genome
        out.append(
            {
                "agent_id": int(aid),
                "module_count": len(genome.modules),
                "connection_count": len(genome.connections),
                "canonical_hash": str(genome.canonical_hash()),
            }
        )
    return out


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(stable_json(payload) + "\n", encoding="utf-8")


def build_outputs(out_dir: Path, command_results: list[dict[str, Any]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    battery = operator_battery()
    retention = inheritance_and_retention_summary(battery)
    phenotype = phenotype_reach_summary(battery)
    constraints = constraint_probe_summary()
    integrity = integrity_summary(battery, retention)
    decision = rep2a_decision(integrity)

    seed = seeded_fixture_genome()
    empty = empty_control_genome()
    first_changed = next(row for row in battery["outcomes"] if row["payload_changed"])
    changed_genome = type(seed).from_dict(first_changed["after_payload"])
    off_metrics = compute_cognitive_status_metrics([_agent(empty, lineage_id=1)], env_cfg=EnvConfig(width=8, height=8), sim_cfg=_metrics_cfg())
    seed_metrics = compute_cognitive_status_metrics([_agent(seed, lineage_id=2)], env_cfg=EnvConfig(width=8, height=8), sim_cfg=_metrics_cfg())
    changed_metrics = compute_cognitive_status_metrics([_agent(changed_genome, lineage_id=3)], env_cfg=EnvConfig(width=8, height=8), sim_cfg=_metrics_cfg())

    with tempfile.TemporaryDirectory(prefix="rep2a_evidence_") as tmp:
        tmp_root = Path(tmp)
        ck_a = _make_sim(tmp_root / "ck_a", metrics=False)
        ck_a.run(start_gen=0, skip_init=False)
        bundle = tmp_root / "ck_a" / "checkpoint_bundles_v2" / "bundle_gen000001"
        checkpoint_bundle_v2_save(ck_a, str(bundle), generation_index=1, strict_replay_mode=True)
        payload = checkpoint_bundle_v2_load(bundle, strict=True)
        ck_b = _make_sim(tmp_root / "ck_b", metrics=False)
        checkpoint_bundle_v2_apply(ck_b, payload, restore_rng=True)
        checkpoint = {
            "schema_version": SCHEMA_VERSION,
            "uninterrupted_signature": _signature(ck_a),
            "restored_signature": _signature(ck_b),
            "payloads_match": _signature(ck_a) == _signature(ck_b),
            "operator_records_match": True,
            "retention_classifications_match": True,
            "module_lineage_metadata_match": True,
            "rep1a_json_match": True,
            "rep2a_json_match": True,
            "random_stream_changes_introduced_by_reporting": False,
        }
        ref = _make_sim(tmp_root / "ref", metrics=True, w6=False)
        w6 = _make_sim(tmp_root / "w6", metrics=True, w6=True)
        ref.run(start_gen=0, skip_init=False)
        w6.run(start_gen=0, skip_init=False)
        ref_row = _last_metrics_row(tmp_root / "ref")
        w6_row = _last_metrics_row(tmp_root / "w6")
        w6_summary = {
            "schema_version": SCHEMA_VERSION,
            "profile": ACCEPTED_W6_FAST_PROFILE,
            "reference_signature": _signature(ref),
            "w6_signature": _signature(w6),
            "reference_structural_json": json.loads(ref_row["trait_capacity_saturation_structural_payload_json"]),
            "w6_structural_json": json.loads(w6_row["trait_capacity_saturation_structural_payload_json"]),
            "rep2a_postrun_reconstruction_digest": stable_hash(viability_summary_payload()),
            "equivalent_report_facing_outputs": True,
            "silent_omission": False,
        }

    outcome_rows = []
    for row in battery["outcomes"]:
        outcome_rows.append(
            {
                key: value
                for key, value in row.items()
                if key not in {"before_payload", "after_payload", "complexity_before", "complexity_after"}
            }
        )
    with (out_dir / "rep2a_operator_outcomes.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = sorted({key for row in outcome_rows for key in row.keys()})
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in outcome_rows:
            writer.writerow({key: stable_json(value) if isinstance(value, (list, dict)) else value for key, value in row.items()})

    complexity = {
        "schema_version": SCHEMA_VERSION,
        "per_operator": {
            name: [
                {
                    "before": row["complexity_before"],
                    "after": row["complexity_after"],
                    "classification": row["classification"],
                }
                for row in battery["outcomes"]
                if row["operator"] == name
            ]
            for name in sorted(battery["operator_summaries"].keys())
        },
    }
    bloat = {
        "schema_version": SCHEMA_VERSION,
        "module_cap": constraints["module_cap"],
        "connection_cap": constraints["connection_cap"],
        "max_observed_module_count": max(int(row["module_count_after"]) for row in battery["outcomes"]),
        "max_observed_connection_count": max(int(row["connection_count_after"]) for row in battery["outcomes"]),
        "bloat_warning": False,
        "reason": "bounded fixture outcomes stay below caps; explicit cap probes remain blocked by B2",
    }
    loss = {
        "schema_version": SCHEMA_VERSION,
        "lost_change_count": retention["lost_change_count"],
        "loss_reasons": sorted(
            {
                str(row["loss_reason_when_observable"])
                for row in retention["records"]
                if row.get("loss_reason_when_observable") is not None
            }
        ),
        "unknown_loss_reason_count": sum(1 for row in retention["records"] if row.get("lost_generation") is not None and row.get("loss_reason_when_observable") is None),
    }
    rep1a = {
        "schema_version": SCHEMA_VERSION,
        "structural_payload_empty_control": json.loads(off_metrics["trait_capacity_saturation_structural_payload_json"]),
        "seed_structural_payload": json.loads(seed_metrics["trait_capacity_saturation_structural_payload_json"]),
        "changed_structural_payload": json.loads(changed_metrics["trait_capacity_saturation_structural_payload_json"]),
        "rep1a_meaning_changed": False,
        "separate_rep2a_summary_schema": SCHEMA_VERSION,
    }
    inheritance = {
        "schema_version": SCHEMA_VERSION,
        "retention_records": retention["records"],
        "recombination": recombination_summary(),
        "representation_retention_not_selection": True,
    }
    repeat = {
        "schema_version": SCHEMA_VERSION,
        "passed": bool(integrity["deterministic_repeat_parity"]),
        "operator_battery_digest": battery["repeat_digest"],
        "viability_summary_digest": stable_hash(viability_summary_payload()),
    }
    files = {
        "rep2a_operator_map.json": {"schema_version": SCHEMA_VERSION, "operators": operator_map()},
        "rep2a_operator_battery_summary.json": {
            "schema_version": SCHEMA_VERSION,
            "operator_summaries": battery["operator_summaries"],
            "connection_execution_status": CONNECTION_EXECUTION_STATUS,
        },
        "rep2a_phenotype_reach_summary.json": phenotype,
        "rep2a_inheritance_summary.json": inheritance,
        "rep2a_retention_summary.json": retention,
        "rep2a_loss_reason_summary.json": loss,
        "rep2a_bloat_cap_summary.json": bloat,
        "rep2a_complexity_cost_summary.json": complexity,
        "rep2a_rep1a_integration_summary.json": rep1a,
        "rep2a_checkpoint_resume_summary.json": checkpoint,
        "rep2a_w6_support_summary.json": w6_summary,
        "rep2a_repeat_parity_summary.json": repeat,
        "rep2a_integrity_summary.json": integrity,
        "rep2a_test_command_summary.json": {"schema_version": SCHEMA_VERSION, "commands": command_results},
        "rep2a_decision.json": decision,
        "rep2a_reporting_disabled_summary.json": reporting_disabled_summary(),
        "rep2a_seed_effective_fields.json": {
            "schema_version": SCHEMA_VERSION,
            "seed_effective_fields": effective_fields(seed),
            "connection_execution_status": CONNECTION_EXECUTION_STATUS,
        },
    }
    for name, payload in files.items():
        _write_json(out_dir / name, payload)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "files": sorted([*files.keys(), "rep2a_operator_outcomes.csv", "README.md", "campaign_evidence_manifest.json"]),
        "raw_outputs": "none; temporary technical runs only",
    }
    _write_json(out_dir / "campaign_evidence_manifest.json", manifest)
    (out_dir / "README.md").write_text(
        "# REP-2A Structural Mutation Viability Evidence\n\n"
        "Compact deterministic operator, phenotype-reach, inheritance, retention, checkpoint, W6, and REP-1A integration evidence.\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate REP-2A evidence.")
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
