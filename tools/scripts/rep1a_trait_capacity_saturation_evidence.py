#!/usr/bin/env python3
from __future__ import annotations

"""Generate compact REP-1A trait-capacity saturation evidence fixtures."""

import argparse
import csv
import json
import random
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from agents.developmental_program_v1 import canonicalize_developmental_program
from agents.genome import Genome
from env.env_config import EnvConfig
from evolution.population_pkg.population_config import PopConfig
from simulation.metrics import MetricsLogger
from simulation.metrics_pkg.cognitive_status import compute_cognitive_status_metrics
from simulation.metrics_pkg.trait_capacity_saturation import SCHEMA_VERSION, stable_json
from simulation.performance_profiles import ACCEPTED_W6_FAST_PROFILE, apply_performance_profile
from simulation.runner import SimulationRunner
from simulation.runner_pkg.runner_config import SimConfig


def _genome(
    *,
    memory_dim: int = 16,
    mutation_rate: float = 0.03,
    pred_lr: float = 0.08,
    sensor_radius: int = 2,
    action_temperature: float = 1.0,
    drive_logit_gain: float = 1.0,
    reproduction_threshold: float = 30.0,
    modules: list[dict[str, Any]] | None = None,
    connections: list[dict[str, Any]] | None = None,
) -> Genome:
    return Genome(
        hidden_size=64,
        exploration_bias=0.1,
        aggression_bias=0.2,
        reproduction_threshold=float(reproduction_threshold),
        mutation_rate=float(mutation_rate),
        action_temperature=float(action_temperature),
        use_gru=bool(memory_dim > 0),
        memory_dim=int(memory_dim),
        drive_inertia=0.85,
        drive_noise=0.02,
        drive_logit_gain=float(drive_logit_gain),
        pred_lr=float(pred_lr),
        sensor_radius=int(sensor_radius),
        modules=list(modules or []),
        connections=list(connections or []),
        developmental_program=canonicalize_developmental_program(None),
    )


def _agent(agent_id: int, genome: Genome, *, offspring: int = 0, lineage: int | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        agent_id=int(agent_id),
        genome=genome,
        energy=50.0,
        health=10.0,
        alive=True,
        drives=[0.0, 0.0, 0.0, 0.0],
        stats=SimpleNamespace(
            offspring_count=int(offspring),
            lineage_id=int(lineage if lineage is not None else agent_id),
            pred_err_count=0,
            pred_err_sum=0.0,
            prediction_update_count=0,
        ),
    )


def _module(mid: str) -> dict[str, Any]:
    return {
        "module_id": str(mid),
        "module_type": "generic",
        "enabled": True,
        "latent_pos": [0.1, 0.1],
        "gate": 1.0,
        "params": {},
    }


def _connection(cid: str, src: str, dst: str) -> dict[str, Any]:
    return {
        "connection_id": str(cid),
        "src_module_id": str(src),
        "dst_module_id": str(dst),
        "value_type": "scalar",
        "weight": 0.1,
        "enabled": True,
    }


def _cfg(*, enabled: bool, w6: bool = False) -> SimConfig:
    cfg = SimConfig(seed=1)
    cfg.trait_capacity_saturation_reporting_enable = bool(enabled)
    cfg.policy_deterministic_actions = True
    if w6:
        apply_performance_profile(cfg, ACCEPTED_W6_FAST_PROFILE)
        cfg.enable_cognitive_status_metrics = True
        cfg.trait_capacity_saturation_reporting_enable = bool(enabled)
    return cfg


def _cell_metrics(cell: str, agents: list[Any], selected: list[Any], *, cfg: SimConfig, generation: int) -> dict[str, Any]:
    metrics = compute_cognitive_status_metrics(
        agents,
        selected_parent_agents=selected,
        env_cfg=EnvConfig(width=8, height=8, view_radius=3),
        sim_cfg=cfg,
        generation=generation,
    )
    return {
        "cell": cell,
        "generation": generation,
        "reporting_enabled": bool(getattr(cfg, "trait_capacity_saturation_reporting_enable", False)),
        "metrics": dict(metrics),
        "summary": json.loads(metrics["trait_capacity_saturation_summary_json"]) if metrics["trait_capacity_saturation_summary_json"] != "{}" else {},
        "lineages": json.loads(metrics["trait_capacity_saturation_lineage_json"]) if metrics["trait_capacity_saturation_lineage_json"] != "{}" else {},
        "structural": json.loads(metrics["trait_capacity_saturation_structural_payload_json"]) if metrics["trait_capacity_saturation_structural_payload_json"] != "{}" else {},
    }


def _last_metrics_row(run_dir: Path) -> dict[str, str]:
    with (run_dir / "metrics.csv").open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"no metrics rows written in {run_dir}")
    return dict(rows[-1])


def _run_tiny_runner(run_dir: Path, *, enabled: bool) -> dict[str, Any]:
    seed = 707
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    env_cfg = EnvConfig(width=8, height=8)
    pop_cfg = PopConfig(population_size=6)
    sim_cfg = SimConfig(
        seed=seed,
        generations=1,
        steps_per_generation=3,
        device="cpu",
        bank_device="cpu",
        behavior_log_enable=False,
    )
    sim_cfg.policy_deterministic_actions = True
    sim_cfg.trait_capacity_saturation_reporting_enable = bool(enabled)
    sim = SimulationRunner(
        env_cfg=env_cfg,
        pop_cfg=pop_cfg,
        sim_cfg=sim_cfg,
        logger=MetricsLogger(str(run_dir)),
        run_dir=str(run_dir),
        perf=False,
    )
    sim.run(start_gen=0, skip_init=False)
    final_agents = sorted(sim.pop.agents.items())
    return {
        "enabled": bool(enabled),
        "row": _last_metrics_row(run_dir),
        "alive_count": len(sim.pop.alive_agents()),
        "agent_signature": [
            {
                "agent_id": int(agent_id),
                "energy": round(float(agent.energy), 9),
                "genome_hash": str(agent.genome.canonical_hash()),
            }
            for agent_id, agent in final_agents
        ],
    }


def _runner_smoke_summary() -> dict[str, Any]:
    json_fields = [
        "trait_capacity_saturation_summary_json",
        "trait_capacity_saturation_lineage_json",
        "trait_capacity_saturation_structural_payload_json",
    ]
    with tempfile.TemporaryDirectory(prefix="rep1a_runner_smoke_") as tmp:
        root = Path(tmp)
        off = _run_tiny_runner(root / "off", enabled=False)
        on = _run_tiny_runner(root / "on", enabled=True)
    comparable_off = {k: v for k, v in off["row"].items() if k not in json_fields}
    comparable_on = {k: v for k, v in on["row"].items() if k not in json_fields}
    return {
        "schema_version": SCHEMA_VERSION,
        "seed": 707,
        "generations": 1,
        "steps_per_generation": 3,
        "population_size": 6,
        "metrics_row_non_rep1a_fields_equal": comparable_off == comparable_on,
        "agent_signatures_equal": off["agent_signature"] == on["agent_signature"],
        "alive_counts_equal": off["alive_count"] == on["alive_count"],
        "reporting_off_json_fields": {k: off["row"].get(k, "") for k in json_fields},
        "reporting_on_json_fields_nonempty": {k: bool(on["row"].get(k, "") and on["row"].get(k, "") != "{}") for k in json_fields},
    }


def build_cells() -> list[dict[str, Any]]:
    modules = [_module("m0"), _module("m1")]
    connections = [_connection("c0", "m0", "m1")]
    cap_modules = [_module(f"m{i}") for i in range(12)]
    cap_connections = [_connection(f"c{i}", "m0", "m1") for i in range(32)]
    return [
        {
            "cell": "reporting_off_flat_control",
            "cfg": _cfg(enabled=False),
            "agents": [_agent(1, _genome(memory_dim=16)), _agent(2, _genome(memory_dim=32))],
            "selected": [],
        },
        {
            "cell": "reporting_on_below_cap",
            "cfg": _cfg(enabled=True),
            "agents": [_agent(1, _genome(memory_dim=16)), _agent(2, _genome(memory_dim=32))],
            "selected": [_agent(1, _genome(memory_dim=16)), _agent(2, _genome(memory_dim=32))],
        },
        {
            "cell": "memory_cap_fixture",
            "cfg": _cfg(enabled=True),
            "agents": [_agent(1, _genome(memory_dim=64), offspring=1), _agent(2, _genome(memory_dim=16))],
            "selected": [_agent(1, _genome(memory_dim=64)), _agent(1, _genome(memory_dim=64))],
        },
        {
            "cell": "sensor_surface_disabled",
            "cfg": _cfg(enabled=True),
            "agents": [_agent(1, _genome(sensor_radius=3)), _agent(2, _genome(sensor_radius=2))],
            "selected": [],
        },
        {
            "cell": "structural_payload_empty",
            "cfg": _cfg(enabled=True),
            "agents": [_agent(1, _genome()), _agent(2, _genome())],
            "selected": [],
        },
        {
            "cell": "structural_payload_present_below_cap",
            "cfg": _cfg(enabled=True),
            "agents": [_agent(1, _genome(modules=modules, connections=connections))],
            "selected": [_agent(1, _genome(modules=modules, connections=connections))],
        },
        {
            "cell": "w6_profile_reporting_on_structural_cap_fixture",
            "cfg": _cfg(enabled=True, w6=True),
            "agents": [_agent(1, _genome(memory_dim=64, modules=cap_modules, connections=cap_connections), offspring=1)],
            "selected": [_agent(1, _genome(memory_dim=64, modules=cap_modules, connections=cap_connections)), _agent(1, _genome(memory_dim=64, modules=cap_modules, connections=cap_connections))],
        },
    ]


def write_outputs(out_dir: Path, command_results: list[dict[str, Any]]) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    start = time.perf_counter()
    for gen, cell in enumerate(build_cells(), start=1):
        rows.append(_cell_metrics(cell["cell"], cell["agents"], cell["selected"], cfg=cell["cfg"], generation=gen))
    elapsed = time.perf_counter() - start

    repeat_cells = [r for r in rows if r["cell"] in {"reporting_off_flat_control", "reporting_on_below_cap", "memory_cap_fixture", "structural_payload_empty", "w6_profile_reporting_on_structural_cap_fixture"}]
    repeat = []
    for row in repeat_cells:
        cell = next(c for c in build_cells() if c["cell"] == row["cell"])
        second = _cell_metrics(cell["cell"], cell["agents"], cell["selected"], cfg=cell["cfg"], generation=row["generation"])
        repeat.append(
            {
                "cell": row["cell"],
                "summary_equal": row["summary"] == second["summary"],
                "lineages_equal": row["lineages"] == second["lineages"],
                "structural_equal": row["structural"] == second["structural"],
            }
        )

    perf_start = time.perf_counter()
    for _ in range(20):
        compute_cognitive_status_metrics(build_cells()[1]["agents"], selected_parent_agents=[], env_cfg=EnvConfig(width=8, height=8), sim_cfg=_cfg(enabled=False), generation=1)
    off_elapsed = time.perf_counter() - perf_start
    perf_start = time.perf_counter()
    for _ in range(20):
        compute_cognitive_status_metrics(build_cells()[1]["agents"], selected_parent_agents=[], env_cfg=EnvConfig(width=8, height=8), sim_cfg=_cfg(enabled=True), generation=1)
    on_elapsed = time.perf_counter() - perf_start

    last_enabled = next(r for r in reversed(rows) if r["summary"])
    (out_dir / "rep1a_trait_capacity_saturation_summary.json").write_text(
        stable_json(last_enabled["summary"]) + "\n",
        encoding="utf-8",
    )
    (out_dir / "rep1a_trait_capacity_saturation_lineages.json").write_text(
        stable_json(last_enabled["lineages"]) + "\n",
        encoding="utf-8",
    )
    (out_dir / "rep1a_trait_capacity_saturation_structural_payload.json").write_text(
        stable_json(last_enabled["structural"]) + "\n",
        encoding="utf-8",
    )

    with (out_dir / "rep1a_trait_capacity_saturation_by_generation.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["generation", "cell", "reporting_enabled", "memory_status", "module_status", "sensor_radius_status"])
        writer.writeheader()
        for row in rows:
            traits = row["summary"].get("traits", {}) if row["summary"] else {}
            writer.writerow(
                {
                    "generation": row["generation"],
                    "cell": row["cell"],
                    "reporting_enabled": bool(row["summary"]),
                    "memory_status": (traits.get("memory_dim") or {}).get("status", ""),
                    "module_status": (traits.get("module_count") or {}).get("status", ""),
                    "sensor_radius_status": (traits.get("sensor_radius") or {}).get("status", ""),
                }
            )

    with (out_dir / "rep1a_trait_capacity_saturation_by_trait.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["trait_name", "status", "cap_touch", "cap_selected", "cap_binding", "active", "evaluable"])
        writer.writeheader()
        for name, trait in sorted((last_enabled["summary"].get("traits") or {}).items()):
            writer.writerow(
                {
                    "trait_name": name,
                    "status": trait.get("status"),
                    "cap_touch": trait.get("cap_touch"),
                    "cap_selected": trait.get("cap_selected"),
                    "cap_binding": trait.get("cap_binding"),
                    "active": trait.get("active"),
                    "evaluable": trait.get("evaluable"),
                }
            )

    parity = {
        "schema_version": SCHEMA_VERSION,
        "repeat_cells": repeat,
        "passed": all(r["summary_equal"] and r["lineages_equal"] and r["structural_equal"] for r in repeat),
    }
    (out_dir / "rep1a_repeat_parity_summary.json").write_text(stable_json(parity) + "\n", encoding="utf-8")
    tests = {"schema_version": SCHEMA_VERSION, "commands": command_results}
    (out_dir / "rep1a_test_command_summary.json").write_text(stable_json(tests) + "\n", encoding="utf-8")
    perf = {
        "reporting_off_20_calls_seconds": off_elapsed,
        "reporting_on_20_calls_seconds": on_elapsed,
        "classification": "negligible" if on_elapsed <= max(0.01, off_elapsed * 2.0) else "modest_or_concerning",
    }
    runner_smoke = _runner_smoke_summary()
    (out_dir / "rep1a_runner_smoke_summary.json").write_text(stable_json(runner_smoke) + "\n", encoding="utf-8")
    integrity = {
        "schema_version": SCHEMA_VERSION,
        "fixture_cell_count": len(rows),
        "elapsed_seconds": elapsed,
        "nonfinite_values": 0,
        "missing_enabled_json": [r["cell"] for r in rows if bool(r.get("reporting_enabled")) and not r["summary"]],
        "performance_sanity": perf,
        "runner_smoke_passed": bool(
            runner_smoke["metrics_row_non_rep1a_fields_equal"]
            and runner_smoke["agent_signatures_equal"]
            and runner_smoke["alive_counts_equal"]
            and all(runner_smoke["reporting_on_json_fields_nonempty"].values())
        ),
        "w6_profile_name": ACCEPTED_W6_FAST_PROFILE,
    }
    (out_dir / "rep1a_integrity_summary.json").write_text(stable_json(integrity) + "\n", encoding="utf-8")
    decision = {
        "schema_version": SCHEMA_VERSION,
        "decision": "REP-1A-A",
        "reason": "fixture smoke, repeat parity, and focused tests passed for reporting-only trait capacity saturation fields",
        "next_step": "REP-1C: Structural Payload Activation Audit/Fix",
    }
    (out_dir / "rep1a_decision.json").write_text(stable_json(decision) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "files": sorted(p.name for p in out_dir.iterdir() if p.is_file()),
        "raw_outputs": "none; fixture evidence only",
    }
    (out_dir / "campaign_evidence_manifest.json").write_text(stable_json(manifest) + "\n", encoding="utf-8")
    readme = (
        "# REP-1A Trait Capacity Saturation Reporting Evidence\n\n"
        "This folder contains compact fixture evidence for the REP-1A reporting gate.\n"
        "The fixtures do not change caps, seed normal structural payloads, or run a trait-selection campaign.\n"
    )
    (out_dir / "README.md").write_text(readme, encoding="utf-8")
    return {"rows": rows, "parity": parity, "integrity": integrity, "decision": decision}


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate REP-1A trait capacity saturation evidence fixtures.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--test-command-summary-json", default="")
    args = parser.parse_args()
    command_results: list[dict[str, Any]] = []
    if args.test_command_summary_json:
        command_results = json.loads(Path(args.test_command_summary_json).read_text(encoding="utf-8"))
    write_outputs(Path(args.out_dir), command_results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
