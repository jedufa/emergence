from __future__ import annotations

import csv
import importlib.util
import json
import sys
from types import SimpleNamespace
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
TOOL_PATH = PROJECT_ROOT / "tools" / "bin" / "mem_evo_2_memory_evolution_observability.py"
ROWS_PATH = PROJECT_ROOT / "engine" / "simulation" / "metrics_pkg" / "rows.py"


def _load_metrics_row():
    spec = importlib.util.spec_from_file_location("mem_evo_2_rows", ROWS_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.MetricsRow


def _load_tool():
    spec = importlib.util.spec_from_file_location("mem_evo_2_memory_evolution_observability", TOOL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _hist(mean: float, buckets: dict[str, int]) -> str:
    total = float(sum(buckets.values()))
    return json.dumps(
        {
            "buckets": buckets,
            "count": int(total),
            "fractions": {k: (float(v) / total if total > 0.0 else 0.0) for k, v in buckets.items()},
            "mean": float(mean),
            "other": 0,
            "p90": float(mean),
        },
        sort_keys=True,
    )


def test_mem_evo_2_required_fields_are_in_metrics_schema() -> None:
    MetricsRow = _load_metrics_row()
    fields = set(MetricsRow.__annotations__)
    required = {
        "mem_evo_population_memory_dim_histogram",
        "mem_evo_selected_parent_memory_dim_histogram",
        "mem_evo_descendant_memory_dim_histogram",
        "mem_evo_lineage_memory_dim_summary_json",
        "mem_evo_effective_memory_dim_histogram",
        "mem_evo_use_gru_share",
        "mem_evo_memory_cost_mean",
        "mem_evo_memory_cost_total",
        "mem_evo_memory_dim_selected_minus_population_mean",
        "mem_evo_memory_dim_descendant_minus_initial_mean",
        "mem_evo_memory_dim_wdl_minus_flat_delta",
        "mem_evo_memory_dim_shift_interpretable",
    }
    assert required <= fields


def test_mem_evo_2_extractor_summarizes_selection_and_descendant_shift(tmp_path: Path) -> None:
    tool = _load_tool()
    metrics_path = tmp_path / "metrics.csv"
    fieldnames = [
        "generation",
        "run_id",
        "world_context",
        "memory_arm",
        "init_arm",
        "seed",
        "memory_dim_mean",
        "memory_dim_p90",
        "memory_dim_max",
        "effective_memory_mean",
        "memory_cost_mean",
        "memory_cost_total",
        "mem_evo_population_memory_dim_histogram",
        "mem_evo_selected_parent_memory_dim_histogram",
        "mem_evo_descendant_memory_dim_histogram",
        "mem_evo_lineage_memory_dim_summary_json",
        "mem_evo_effective_memory_dim_histogram",
        "mem_evo_use_gru_share",
        "mem_evo_memory_dim_selected_minus_population_mean",
        "mem_evo_memory_dim_descendant_minus_initial_mean",
        "mem_evo_memory_dim_wdl_minus_flat_delta",
        "mem_evo_memory_dim_shift_interpretable",
    ]
    lineage_payload = {
        "lineage_count": 1,
        "lineages": [{"lineage": 7, "count": 4, "memory_dim_mean": 16.0, "effective_memory_dim_mean": 16.0}],
        "mean_lineage_memory_dim_mean": 16.0,
    }
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "generation": 0,
                "run_id": "fixture",
                "world_context": "wdl_delayed_hazard",
                "memory_arm": "memory_normal",
                "init_arm": "founder_balanced",
                "seed": 101,
                "memory_dim_mean": 8.0,
                "memory_dim_p90": 8.0,
                "memory_dim_max": 8.0,
                "effective_memory_mean": 8.0,
                "memory_cost_mean": 0.1,
                "memory_cost_total": 0.4,
                "mem_evo_population_memory_dim_histogram": _hist(8.0, {"0": 0, "8": 4, "16": 0, "32": 0, "64": 0}),
                "mem_evo_selected_parent_memory_dim_histogram": _hist(16.0, {"0": 0, "8": 0, "16": 2, "32": 0, "64": 0}),
                "mem_evo_descendant_memory_dim_histogram": _hist(8.0, {"0": 0, "8": 4, "16": 0, "32": 0, "64": 0}),
                "mem_evo_lineage_memory_dim_summary_json": json.dumps(lineage_payload, sort_keys=True),
                "mem_evo_effective_memory_dim_histogram": _hist(8.0, {"0": 0, "8": 4, "16": 0, "32": 0, "64": 0}),
                "mem_evo_use_gru_share": 1.0,
                "mem_evo_memory_dim_selected_minus_population_mean": 8.0,
                "mem_evo_memory_dim_descendant_minus_initial_mean": 0.0,
                "mem_evo_memory_dim_wdl_minus_flat_delta": 0.0,
                "mem_evo_memory_dim_shift_interpretable": 0.0,
            }
        )
        writer.writerow(
            {
                "generation": 1,
                "run_id": "fixture",
                "world_context": "wdl_delayed_hazard",
                "memory_arm": "memory_normal",
                "init_arm": "founder_balanced",
                "seed": 101,
                "memory_dim_mean": 16.0,
                "memory_dim_p90": 16.0,
                "memory_dim_max": 16.0,
                "effective_memory_mean": 16.0,
                "memory_cost_mean": 0.2,
                "memory_cost_total": 0.8,
                "mem_evo_population_memory_dim_histogram": _hist(16.0, {"0": 0, "8": 0, "16": 4, "32": 0, "64": 0}),
                "mem_evo_selected_parent_memory_dim_histogram": _hist(16.0, {"0": 0, "8": 0, "16": 2, "32": 0, "64": 0}),
                "mem_evo_descendant_memory_dim_histogram": _hist(16.0, {"0": 0, "8": 0, "16": 4, "32": 0, "64": 0}),
                "mem_evo_lineage_memory_dim_summary_json": json.dumps(lineage_payload, sort_keys=True),
                "mem_evo_effective_memory_dim_histogram": _hist(16.0, {"0": 0, "8": 0, "16": 4, "32": 0, "64": 0}),
                "mem_evo_use_gru_share": 1.0,
                "mem_evo_memory_dim_selected_minus_population_mean": 0.0,
                "mem_evo_memory_dim_descendant_minus_initial_mean": 8.0,
                "mem_evo_memory_dim_wdl_minus_flat_delta": 0.0,
                "mem_evo_memory_dim_shift_interpretable": 1.0,
            }
        )

    summary = tool.build_summaries([metrics_path])
    assert summary["decision"]["decision"] == "MEM-EVO-2A"
    assert summary["decision"]["current_schema_complete"] is True
    assert summary["decision"]["existing_shift_interpretable"] is True
    assert summary["parent_rows"][0]["selected_minus_population_mean"] == 8.0
    assert summary["descendant_rows"][0]["descendant_minus_initial_mean"] == 8.0
    assert summary["lineage_rows"][0]["lineage_count"] == 1

    out_dir = tmp_path / "out"
    tool.write_outputs(out_dir, summary)
    assert (out_dir / "memory_evolution_field_inventory.json").exists()
    assert (out_dir / "memory_dim_distribution_summary.csv").exists()
    assert json.loads((out_dir / "mem_evo_2_decision.json").read_text(encoding="utf-8"))["decision"] == "MEM-EVO-2A"


def test_mem_evo_histograms_report_genome_memory_under_ablation() -> None:
    from agents.developmental_program_v1 import canonicalize_developmental_program
    from agents.genome import Genome
    from env.env_config import EnvConfig
    from simulation.metrics_pkg.cognitive_status import compute_cognitive_status_metrics
    from simulation.runner_pkg.runner_config import SimConfig

    genome = Genome(
        hidden_size=64,
        exploration_bias=0.1,
        aggression_bias=0.2,
        reproduction_threshold=30.0,
        mutation_rate=0.03,
        action_temperature=1.0,
        use_gru=True,
        memory_dim=16,
        drive_inertia=0.85,
        drive_noise=0.02,
        drive_logit_gain=1.1,
        pred_lr=0.08,
        developmental_program=canonicalize_developmental_program(None),
    )
    agent = SimpleNamespace(genome=genome, stats=SimpleNamespace(lineage_id=7), energy=100.0, alive=True)
    sim_cfg = SimConfig(seed=7)
    sim_cfg.force_memory_disabled = True
    sim_cfg.reset_memory_each_tick = True
    sim_cfg.enable_experimental_memory_cost_charging = True
    metrics = compute_cognitive_status_metrics(
        [agent],
        selected_parent_agents=[agent],
        env_cfg=EnvConfig(width=8, height=8),
        sim_cfg=sim_cfg,
    )

    population = json.loads(metrics["mem_evo_population_memory_dim_histogram"])
    parent = json.loads(metrics["mem_evo_selected_parent_memory_dim_histogram"])
    lineage = json.loads(metrics["mem_evo_lineage_memory_dim_summary_json"])
    effective = json.loads(metrics["mem_evo_effective_memory_dim_histogram"])

    assert population["mean"] == 16.0
    assert parent["mean"] == 16.0
    assert lineage["lineages"][0]["memory_dim_mean"] == 16.0
    assert lineage["lineages"][0]["active_memory_dim_mean"] == 0.0
    assert effective["mean"] == 0.0
