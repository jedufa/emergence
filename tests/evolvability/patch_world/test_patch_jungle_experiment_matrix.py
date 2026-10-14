from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from dataclasses import fields
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

from env.env_config import EnvConfig
from evolution.population_pkg.population_config import PopConfig
from experiments.harness.schema import load_and_validate_registry
from experiments.patch_jungle_eval import (
    MATRIX_VERSION,
    PATCH_JUNGLE_FLAGS,
    all_metric_names,
    build_paired_manifest,
    compare_metric_summaries,
    default_comparison_pairs,
    metric_groups,
    scenario_by_id,
    scenario_cli_tokens,
    scenario_knob_pairs,
    scenario_registry,
    summarize_batch,
    summarize_metrics_csv,
    validate_metric_groups_against_schema,
    write_summary_json,
)
from knobs_override import apply_knob_overrides
from simulation.metrics import MetricsLogger
from simulation.metrics_pkg.rows import MetricsRow
from simulation.runner import SimulationRunner
from simulation.runner_pkg.runner_config import SimConfig


REGISTRY_PATH = Path("_spec/registry/v1/patch_jungle_v0_smoke.yaml")


def _base_env_cfg() -> EnvConfig:
    cfg = EnvConfig(width=6, height=6, view_radius=1)
    cfg.enable_biomes = False
    cfg.food_spawn_per_step = 2
    cfg.max_food = 12
    return cfg


def _base_sim_cfg(seed: int = 43) -> SimConfig:
    cfg = SimConfig(
        seed=int(seed),
        generations=1,
        steps_per_generation=2,
        device="cpu",
        bank_device="cpu",
        behavior_log_enable=False,
        policy_deterministic_actions=True,
        policy_tick_stride=1,
    )
    cfg.phase_e_e1_enable = False
    cfg.epi_enable = False
    cfg.checkpoint_bundle_v2_period_gen = 0
    return cfg


def _run_smoke(tmp_path: Path, scenario_id: str) -> Path:
    run_dir = tmp_path / scenario_id
    env_cfg = _base_env_cfg()
    pop_cfg = PopConfig(population_size=4)
    sim_cfg = _base_sim_cfg(seed=101)
    applied, genome_overrides = apply_knob_overrides(
        knob_pairs=list(scenario_knob_pairs(scenario_id)),
        env_cfg=env_cfg,
        pop_cfg=pop_cfg,
        sim_cfg=sim_cfg,
    )
    assert applied
    assert genome_overrides == {}

    sim = SimulationRunner(
        env_cfg=env_cfg,
        pop_cfg=pop_cfg,
        sim_cfg=sim_cfg,
        logger=MetricsLogger(str(run_dir)),
        run_dir=str(run_dir),
        perf=False,
    )
    sim.run(start_gen=0, skip_init=False)
    return run_dir


def _read_last_metrics_row(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows
    return rows[-1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_step14_scenario_registry_is_stable_and_explicit() -> None:
    scenarios = scenario_registry()
    ids = [scenario.scenario_id for scenario in scenarios]

    assert MATRIX_VERSION == 1
    assert ids == [
        "s00_flat_baseline",
        "s01_patch_food_only",
        "s02_patch_food_terrain",
        "s03_patch_observed",
        "s04_patch_waste_crowding",
        "s05_predator_pressure",
        "s06_predator_reproduction_emissions",
        "s07_predator_terrain_protection",
        "s08_full_jungle_no_recovery",
        "s09_full_jungle_recovery",
    ]

    for scenario in scenarios:
        override_keys = {key for key, _ in scenario.overrides}
        for flag in PATCH_JUNGLE_FLAGS:
            assert flag in override_keys
        assert "patch_movement_cost_apply_enable" in override_keys

    flat = dict(scenario_by_id("s00_flat_baseline").overrides)
    assert flat["patch_world_enable"] is False
    for flag in PATCH_JUNGLE_FLAGS:
        assert flat[flag] is False

    observed = dict(scenario_by_id("s03_patch_observed").overrides)
    assert observed["patch_world_enable"] is True
    assert observed["patch_world_observation_enable"] is True
    assert observed["jungle_predator_pressure_enable"] is False

    full = dict(scenario_by_id("s09_full_jungle_recovery").overrides)
    assert full["patch_world_enable"] is True
    assert full["patch_world_observation_enable"] is True
    assert full["patch_world_waste_consequence_enable"] is True
    assert full["jungle_predator_pressure_enable"] is True
    assert full["jungle_reproduction_emissions_enable"] is True
    assert full["jungle_terrain_protection_enable"] is True
    assert full["jungle_ecological_recovery_enable"] is True
    assert full["jungle_juvenile_predator_vulnerability_enable"] is False


def test_step14_registry_does_not_change_env_defaults() -> None:
    before = EnvConfig()
    _ = scenario_registry()
    _ = metric_groups()
    after = EnvConfig()

    assert before == after
    assert after.patch_world_enable is False
    assert after.patch_world_observation_enable is False
    assert after.patch_world_waste_consequence_enable is False
    assert after.jungle_predator_pressure_enable is False
    assert after.jungle_reproduction_emissions_enable is False
    assert after.jungle_terrain_protection_enable is False
    assert after.jungle_ecological_recovery_enable is False


def test_step14_registry_yaml_matches_python_scenarios_and_expands_deterministically() -> None:
    doc = yaml.safe_load(REGISTRY_PATH.read_text(encoding="utf-8"))
    registry = load_and_validate_registry(doc)
    scenarios = scenario_registry()

    assert registry.name == "patch_jungle_v0_smoke"
    assert registry.matrix_order == ["scenario", "seed"]
    assert registry.seeds == [43]

    axis = next(axis for axis in registry.axes if axis.name == "scenario")
    assert axis.values == [scenario.scenario_id for scenario in scenarios]
    for scenario in scenarios:
        assert axis.flags[scenario.scenario_id] == list(scenario_cli_tokens(scenario))


def test_step14_registry_batch_dry_run_entrypoint_plans_matrix(tmp_path: Path) -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "_runner.run_registry_batch",
            "--registry",
            "v1/patch_jungle_v0_smoke.yaml",
            "--runs_root",
            str(tmp_path / "runs"),
            "--batch_tag",
            "step14_test",
            "--dry_run",
            "--overwrite",
        ],
        cwd=Path.cwd(),
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    output = proc.stdout + proc.stderr
    assert "Planned runs:" in output
    assert "10" in output
    for scenario in scenario_registry():
        assert scenario.scenario_id in output
    assert "dry_run" not in proc.stderr.lower()


def test_step14_local_experiments_shared_compat_imports() -> None:
    from experiments_shared.checkpoint_branching import branch_from_checkpoint
    from experiments_shared.runner_utils import resolve_registry_path

    assert callable(branch_from_checkpoint)
    resolved = resolve_registry_path(project_root=Path.cwd(), raw="v1/patch_jungle_v0_smoke.yaml")
    assert resolved == REGISTRY_PATH.resolve()


def test_step14_manifest_generation_is_deterministic() -> None:
    manifest_a = build_paired_manifest(
        scenario_ids=("s00_flat_baseline", "s09_full_jungle_recovery"),
        seeds=(43, 44),
        run_length={"generations": 2, "steps": 12},
        output_dir="runs/experiments/patch_jungle_v0",
    )
    manifest_b = build_paired_manifest(
        scenario_ids=("s00_flat_baseline", "s09_full_jungle_recovery"),
        seeds=(43, 44),
        run_length={"generations": 2, "steps": 12},
        output_dir="runs/experiments/patch_jungle_v0",
    )

    assert manifest_a == manifest_b
    assert "created_timestamp" not in manifest_a
    assert manifest_a["experiment_matrix_version"] == 1
    assert manifest_a["scenario_ids"] == ["s00_flat_baseline", "s09_full_jungle_recovery"]
    assert manifest_a["seeds"] == [43, 44]
    assert len(manifest_a["scenarios"][0]["config_hash"]) == 12


def test_step14_metric_groups_reference_metrics_row_schema() -> None:
    schema_fields = {field.name for field in fields(MetricsRow)}
    validation = validate_metric_groups_against_schema(schema_fields)

    assert validation["ok"] is True
    assert validation["missing_metrics"] == []
    assert len(all_metric_names()) == len(set(all_metric_names()))
    assert [group.name for group in metric_groups()] == [
        "population_survival",
        "reproduction",
        "patch_use",
        "food_ecology",
        "waste_crowding",
        "predator_pressure",
        "terrain_protection",
        "recovery",
        "lineage_diversity",
        "existing_cognition_diagnostics",
    ]


def test_step14_summary_helpers_compute_raw_deltas_and_missing_metrics(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.csv"
    variant = tmp_path / "variant.csv"
    baseline.write_text(
        "generation,alive,patch_waste_mean,jungle_predator_events\n"
        "0,4,0.1,0\n"
        "1,5,0.2,1\n",
        encoding="utf-8",
    )
    variant.write_text(
        "generation,alive,patch_waste_mean,jungle_predator_events\n"
        "0,6,0.3,2\n"
        "1,7,0.5,4\n",
        encoding="utf-8",
    )

    metric_names = ("alive", "patch_waste_mean", "jungle_predator_events", "missing_metric")
    base_summary = summarize_metrics_csv(baseline, metric_names)
    variant_summary = summarize_metrics_csv(variant, metric_names)
    comparison = compare_metric_summaries(
        baseline_id="baseline",
        baseline=base_summary,
        variant_id="variant",
        variant=variant_summary,
        metric_names=metric_names,
    )

    assert base_summary["metrics"]["alive"] == 5.0
    assert variant_summary["metrics"]["patch_waste_mean"] == 0.5
    assert comparison["deltas"]["alive"] == 2.0
    assert comparison["deltas"]["patch_waste_mean"] == pytest.approx(0.3)
    assert comparison["deltas"]["jungle_predator_events"] == 3.0
    assert comparison["missing_metrics"] == ["missing_metric"]


def test_step14_summary_handles_large_lineage_json_fields(tmp_path: Path) -> None:
    metrics = tmp_path / "metrics.csv"
    large_json = json.dumps({"lineages": [{"id": "lineage_a", "payload": "x" * 140_000}]})
    with metrics.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=("generation", "alive", "lineage_strategy_descriptors_json"),
        )
        writer.writeheader()
        writer.writerow(
            {
                "generation": 0,
                "alive": 4,
                "lineage_strategy_descriptors_json": large_json,
            }
        )

    summary = summarize_metrics_csv(metrics, ("alive",))

    assert summary["rows"] == 1
    assert summary["metrics"]["alive"] == 4.0


def test_step14_batch_summary_is_deterministic_with_fixture_outputs(tmp_path: Path) -> None:
    batch_dir = tmp_path / "batch"
    run_a = batch_dir / "runs" / "a"
    run_b = batch_dir / "runs" / "b"
    run_a.mkdir(parents=True)
    run_b.mkdir(parents=True)
    (batch_dir / "matrix.csv").write_text(
        "run_id,runtime_run_id,run_tag,seed,scenario,runs_dir,run_dir,cli\n"
        f"a,a,tag,43,s00_flat_baseline,{batch_dir / 'runs'},{run_a},x\n"
        f"b,b,tag,43,s01_patch_food_only,{batch_dir / 'runs'},{run_b},x\n",
        encoding="utf-8",
    )
    (run_a / "metrics.csv").write_text("generation,alive,patch_food_amount_mean\n0,4,0.0\n", encoding="utf-8")
    (run_b / "metrics.csv").write_text("generation,alive,patch_food_amount_mean\n0,5,0.2\n", encoding="utf-8")

    summary = summarize_batch(batch_dir)
    out = tmp_path / "summary.json"
    write_summary_json(out, summary)
    loaded = json.loads(out.read_text(encoding="utf-8"))

    assert loaded == summary
    assert loaded["scenario_summaries"]["s00_flat_baseline"]["metrics"]["alive"] == 4.0
    assert loaded["scenario_summaries"]["s01_patch_food_only"]["metrics"]["patch_food_amount_mean"] == 0.2
    assert loaded["comparisons"][0]["comparison_id"] == "s01_patch_food_only_minus_s00_flat_baseline"
    assert loaded["comparisons"][0]["deltas"]["alive"] == 1.0


def test_campaign_c_report_groups_variant_summaries_and_paired_deltas(tmp_path: Path) -> None:
    batch_dir = tmp_path / "batch"
    runs_dir = batch_dir / "runs"
    runs_dir.mkdir(parents=True)
    (batch_dir / "manifest.json").write_text(
        json.dumps(
            {
                "registry": {
                    "name": "patch_jungle_v0_campaign_c_fixture",
                    "version": 1,
                    "sha256_short": "fixture",
                }
            }
        ),
        encoding="utf-8",
    )
    rows = [
        ("low_101", "low", 101, 10, 1.0),
        ("low_202", "low", 202, 12, 1.2),
        ("high_101", "high", 101, 14, 2.0),
        ("high_202", "high", 202, 18, 2.2),
    ]
    with (batch_dir / "matrix.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "run_id",
                "runtime_run_id",
                "run_tag",
                "seed",
                "variant",
                "scenario",
                "runs_dir",
                "run_dir",
                "cli",
            ],
        )
        writer.writeheader()
        for run_id, variant, seed, alive, predator_events in rows:
            run_dir = runs_dir / run_id
            run_dir.mkdir()
            writer.writerow(
                {
                    "run_id": run_id,
                    "runtime_run_id": run_id,
                    "run_tag": f"fixture_{variant}",
                    "seed": seed,
                    "variant": variant,
                    "scenario": "s05_predator_pressure",
                    "runs_dir": runs_dir,
                    "run_dir": run_dir,
                    "cli": "",
                }
            )
            (run_dir / "metrics.csv").write_text(
                "generation,alive,jungle_predator_events,patch_world_enable,patch_world_observation_enable,"
                "patch_world_waste_consequence_enable,jungle_predator_pressure_enable,"
                "jungle_reproduction_emissions_enable,jungle_juvenile_predator_vulnerability_enable,"
                "jungle_terrain_protection_enable,jungle_ecological_recovery_enable,"
                "lineage_strategy_descriptors_json,lineage_branch_candidates_json\n"
                f"9,{alive},{predator_events},1,1,0,1,0,0,0,0,\"[]\",\"[]\"\n",
                encoding="utf-8",
            )

    out_json = tmp_path / "campaign_c_report.json"
    summary_csv = tmp_path / "campaign_c_summary.csv"
    deltas_csv = tmp_path / "campaign_c_deltas.csv"
    report_md = tmp_path / "campaign_c.md"
    proc = subprocess.run(
        [
            sys.executable,
            "tools/bin/patch_jungle_campaign_c_report.py",
            "--batch-dir",
            str(batch_dir),
            "--out-json",
            str(out_json),
            "--summary-csv",
            str(summary_csv),
            "--deltas-csv",
            str(deltas_csv),
            "--report-md",
            str(report_md),
        ],
        cwd=Path.cwd(),
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    report = json.loads(out_json.read_text(encoding="utf-8"))
    assert report["integrity"]["planned_runs"] == 4
    assert report["integrity"]["completed_runs"] == 4
    assert report["integrity"]["critical_nonfinite_issues"] == 0
    assert report["integrity"]["scenario_flag_mismatches"] == 0
    summaries = {
        item["variant"]: item
        for item in report["variant_summaries"]
        if item["scenario"] == "s05_predator_pressure"
    }
    assert summaries["low"]["metrics"]["alive"] == 11.0
    assert summaries["high"]["metrics"]["alive"] == 16.0
    delta = report["paired_deltas"][0]
    assert delta["baseline_variant"] == "low"
    assert delta["variant"] == "high"
    assert delta["deltas"]["alive"] == 5.0
    assert summary_csv.exists()
    assert deltas_csv.exists()
    assert report_md.exists()


def test_campaign_e_cuda_registry_is_deterministic_and_uses_tuned_overrides() -> None:
    registry_path = Path("_spec/registry/v1/patch_jungle_v0_campaign_e_lineage_strategy.yaml")
    doc = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    registry = load_and_validate_registry(doc)

    assert registry.name == "patch_jungle_v0_campaign_e_lineage_strategy_triage"
    assert registry.matrix_order == ["scenario", "seed"]
    assert registry.seeds == [
        101,
        202,
        303,
        404,
        505,
        606,
        707,
        808,
        909,
        1001,
        1102,
        1203,
        1304,
        1405,
        1506,
        1607,
        1708,
        1809,
        19010,
        20011,
    ]
    axis = next(axis for axis in registry.axes if axis.name == "scenario")
    assert axis.values == [
        "e00_flat_baseline",
        "e01_patch_observed",
        "e02_default_full_jungle",
        "e03_tuned_full_jungle_no_recovery",
        "e04_tuned_full_jungle_recovery",
    ]
    assert ["--device", "cuda"] in registry.cli_defaults
    assert ["--bank_device", "cuda"] in registry.cli_defaults
    assert ["--generations", "24"] in registry.cli_defaults
    assert ["--steps", "120"] in registry.cli_defaults
    assert ["--pop", "64"] in registry.cli_defaults
    tuned_flags = axis.flags["e04_tuned_full_jungle_recovery"]
    assert "env_cfg.jungle_predator_risk_cap=0.05" in tuned_flags
    assert "env_cfg.jungle_terrain_damage_multiplier_min=0.15" in tuned_flags
    assert "env_cfg.jungle_recovery_regrowth_boost_scale=0.05" in tuned_flags
    no_recovery_flags = axis.flags["e03_tuned_full_jungle_no_recovery"]
    assert "env_cfg.jungle_ecological_recovery_enable=false" in no_recovery_flags
    assert "env_cfg.jungle_recovery_regrowth_boost_scale=0.05" not in no_recovery_flags


def test_campaign_e_lineage_report_computes_collapse_and_branch_diagnostics(tmp_path: Path) -> None:
    batch_dir = tmp_path / "batch"
    runs_dir = batch_dir / "runs"
    runs_dir.mkdir(parents=True)

    def write_config(run_dir: Path, scenario: str) -> None:
        env = {
            "patch_world_enable": True,
            "patch_world_observation_enable": True,
            "patch_world_waste_consequence_enable": True,
            "jungle_predator_pressure_enable": True,
            "jungle_reproduction_emissions_enable": True,
            "jungle_terrain_protection_enable": True,
            "jungle_ecological_recovery_enable": True,
        }
        if scenario == "e04_tuned_full_jungle_recovery":
            env.update(
                {
                    "jungle_predator_base_risk": 0.0,
                    "jungle_predator_scent_weight": 0.02,
                    "jungle_predator_waste_weight": 0.02,
                    "jungle_predator_density_weight": 0.02,
                    "jungle_predator_visibility_weight": 0.02,
                    "jungle_predator_risk_cap": 0.05,
                    "jungle_predator_health_damage": 0.25,
                    "patch_waste_regrowth_penalty_scale": 0.5,
                    "patch_waste_regrowth_penalty_max": 0.5,
                    "patch_density_regrowth_penalty_scale": 0.5,
                    "patch_density_regrowth_penalty_max": 0.5,
                    "patch_regrowth_min_multiplier": 0.1,
                    "jungle_cover_damage_reduction_weight": 0.4,
                    "jungle_shelter_damage_reduction_weight": 0.4,
                    "jungle_terrain_damage_multiplier_min": 0.15,
                    "jungle_recovery_regrowth_boost_scale": 0.05,
                    "jungle_recovery_regrowth_boost_max": 0.1,
                    "jungle_recovery_fertility_enable": False,
                    "jungle_resource_spread_enable": False,
                }
            )
        (run_dir / "config.json").write_text(
            json.dumps({"env_cfg": env, "sim_cfg": {"device": "cuda", "bank_device": "cuda"}}),
            encoding="utf-8",
        )

    def descriptor(
        lineage_id: str,
        alive_share: float,
        alive_count: int,
        birth_count: int,
        cover: float,
        waste: float,
        damage: float,
        recovery: float,
    ) -> dict[str, float | str]:
        return {
            "lineage_id": lineage_id,
            "lineage_alive_share": alive_share,
            "lineage_alive_count": alive_count,
            "lineage_birth_count": birth_count,
            "lineage_birth_share": birth_count / 10.0,
            "lineage_patch_cover_mean": cover,
            "lineage_patch_shelter_mean": cover / 2.0,
            "lineage_patch_waste_mean": waste,
            "lineage_patch_density_mean": 0.2 + waste,
            "lineage_patch_recovery_activity_mean": recovery,
            "lineage_predator_event_count": damage,
            "lineage_predator_damage_total": damage,
            "lineage_predator_damage_mean": damage / max(alive_count, 1),
            "lineage_predator_damage_prevented_total": cover,
            "lineage_birth_patch_waste_mean": waste,
            "lineage_birth_patch_recovery_activity_mean": recovery,
            "lineage_reproduction_rate": birth_count / max(alive_count, 1),
        }

    rows = [
        ("e02_run", "e02_default_full_jungle", 101),
        ("e04_run", "e04_tuned_full_jungle_recovery", 101),
    ]
    with (batch_dir / "matrix.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["run_id", "runtime_run_id", "run_tag", "seed", "scenario", "runs_dir", "run_dir", "cli"],
        )
        writer.writeheader()
        for run_id, scenario, seed in rows:
            run_dir = runs_dir / run_id
            run_dir.mkdir()
            write_config(run_dir, scenario)
            writer.writerow(
                {
                    "run_id": run_id,
                    "runtime_run_id": run_id,
                    "run_tag": scenario,
                    "seed": seed,
                    "scenario": scenario,
                    "runs_dir": runs_dir,
                    "run_dir": run_dir,
                    "cli": "",
                }
            )
            with (run_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as mf:
                fieldnames = [
                    "generation",
                    "alive",
                    "deaths",
                    "cohort_survival_rate",
                    "successful_births",
                    "patch_food_amount_mean",
                    "patch_regrowth_suppressed_total",
                    "jungle_predator_events",
                    "jungle_predator_damage_total",
                    "jungle_predator_damage_prevented_total",
                    "patch_recovery_activity_mean",
                    "patch_recovery_added_regrowth_total",
                    "lineage_active_count",
                    "lineage_top1_alive_share",
                    "lineage_alive_share_entropy",
                    "lineage_effective_alive_lineages",
                    "lineage_branch_candidate_count",
                    "lineage_strategy_descriptors_json",
                    "lineage_branch_candidates_json",
                ]
                metrics_writer = csv.DictWriter(mf, fieldnames=fieldnames)
                metrics_writer.writeheader()
                for gen, top1, eff, entropy in [(0, 0.45, 2.2, 0.9), (1, 0.82, 1.4, 0.4), (2, 1.0, 1.0, 0.0)]:
                    descriptors = [
                        descriptor("lin_dom", top1, 10, 7, 0.4, 0.1, 0.5, 0.2),
                        descriptor("lin_rare", 0.12 if gen > 0 else 0.2, 2, 1, 0.9, 0.02, 0.05, 0.6),
                    ]
                    candidates = [
                        {
                            "candidate_reason": "dominant_lineage",
                            "candidate_lineage_id": "lin_dom",
                            "candidate_alive_share": top1,
                        },
                        {
                            "candidate_reason": "rare_persistent_lineage",
                            "candidate_lineage_id": "lin_rare",
                            "candidate_alive_share": 0.12,
                        },
                    ]
                    if scenario == "e04_tuned_full_jungle_recovery" and gen == 0:
                        descriptors[0]["lineage_large_diagnostic_payload"] = "x" * 140000
                    metrics_writer.writerow(
                        {
                            "generation": gen,
                            "alive": 20,
                            "deaths": 1,
                            "cohort_survival_rate": 0.9,
                            "successful_births": 8,
                            "patch_food_amount_mean": 0.4,
                            "patch_regrowth_suppressed_total": 1.0,
                            "jungle_predator_events": 2.0,
                            "jungle_predator_damage_total": 0.5,
                            "jungle_predator_damage_prevented_total": 0.2,
                            "patch_recovery_activity_mean": 0.3,
                            "patch_recovery_added_regrowth_total": 0.1,
                            "lineage_active_count": 2 if gen < 2 else 1,
                            "lineage_top1_alive_share": top1,
                            "lineage_alive_share_entropy": entropy,
                            "lineage_effective_alive_lineages": eff,
                            "lineage_branch_candidate_count": 2,
                            "lineage_strategy_descriptors_json": json.dumps(descriptors, sort_keys=True),
                            "lineage_branch_candidates_json": json.dumps(candidates, sort_keys=True),
                        }
                    )

    out_json = tmp_path / "campaign_e_report.json"
    analysis_json = tmp_path / "campaign_e_analysis.json"
    summary_csv = tmp_path / "campaign_e_summary.csv"
    deltas_csv = tmp_path / "campaign_e_deltas.csv"
    lineage_json = tmp_path / "campaign_e_lineage.json"
    branch_json = tmp_path / "campaign_e_branch.json"
    report_md = tmp_path / "campaign_e.md"
    proc = subprocess.run(
        [
            sys.executable,
            "tools/bin/patch_jungle_campaign_e_lineage_report.py",
            "--batch-dir",
            str(batch_dir),
            "--out-json",
            str(out_json),
            "--analysis-json",
            str(analysis_json),
            "--summary-csv",
            str(summary_csv),
            "--deltas-csv",
            str(deltas_csv),
            "--lineage-json",
            str(lineage_json),
            "--branch-json",
            str(branch_json),
            "--report-md",
            str(report_md),
        ],
        cwd=Path.cwd(),
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    report = json.loads(out_json.read_text(encoding="utf-8"))
    assert report["integrity"]["planned_runs"] == 2
    assert report["integrity"]["completed_runs"] == 2
    assert report["integrity"]["critical_nonfinite_issues"] == 0
    lineage = json.loads(lineage_json.read_text(encoding="utf-8"))
    e04 = lineage["scenario_summaries"]["e04_tuned_full_jungle_recovery"]
    assert e04["collapse_top1_generation_mean"] == 1.0
    assert e04["rare_persistent_candidate_count"] >= 1
    assert e04["branch_candidate_counts"]["rare_persistent_lineage"] == 3
    assert e04["branch_candidate_non_dominant_fraction_mean"] == 0.5
    assert summary_csv.exists()
    assert deltas_csv.exists()
    assert branch_json.exists()
    assert report_md.exists()


def test_step14_default_comparison_pairs_are_valid() -> None:
    known = {scenario.scenario_id for scenario in scenario_registry()}
    for baseline_id, variant_id in default_comparison_pairs():
        assert baseline_id in known
        assert variant_id in known


def test_step14_tiny_smoke_runs_emit_metrics(tmp_path: Path) -> None:
    for scenario_id in ("s00_flat_baseline", "s01_patch_food_only", "s09_full_jungle_recovery"):
        run_dir = _run_smoke(tmp_path, scenario_id)
        metrics_path = run_dir / "metrics.csv"
        assert metrics_path.exists()
        last = _read_last_metrics_row(metrics_path)
        assert "alive" in last
        assert "lineage_active_count" in last
        if scenario_id == "s00_flat_baseline":
            assert int(float(last["patch_observation_enable"])) == 0
            assert int(float(last["jungle_predator_pressure_enable"])) == 0
        if scenario_id == "s09_full_jungle_recovery":
            assert int(float(last["patch_observation_enable"])) == 1
            assert int(float(last["jungle_predator_pressure_enable"])) == 1
            assert int(float(last["jungle_ecological_recovery_enable"])) == 1


@pytest.mark.parametrize(
    "scenario_id",
    ("s00_flat_baseline", "s05_predator_pressure", "s09_full_jungle_recovery"),
)
def test_step14_engine_repeat_parity_for_campaign_probe_scenarios(tmp_path: Path, scenario_id: str) -> None:
    """Same scenario+seed engine runs must emit identical raw metrics.

    This guards Campaign B repeat parity. The settings are intentionally tiny,
    but the test uses the real `engine.main` entrypoint so it covers bootstrap
    seeding and policy module initialization, not only in-process helpers.
    """

    hashes: list[tuple[str, str, str]] = []
    for label in ("a", "b"):
        runs_dir = tmp_path / label
        cmd = [
            sys.executable,
            "-m",
            "engine.main",
            "--generations",
            "2",
            "--steps",
            "8",
            "--pop",
            "8",
            "--grid_size",
            "6",
            "--policy_tick_stride",
            "1",
            "--device",
            "cpu",
            "--bank_device",
            "cpu",
            "--knob",
            "sim_cfg.behavior_log_enable=false",
            "--knob",
            "sim_cfg.checkpoint_bundle_v2_period_gen=0",
            *scenario_cli_tokens(scenario_id),
            "--seed",
            "101",
            "--runs_dir",
            str(runs_dir),
            "--run_id",
            "repeat",
            "--run_tag",
            f"repeat_{scenario_id}",
        ]
        proc = subprocess.run(
            cmd,
            cwd=Path.cwd(),
            text=True,
            capture_output=True,
            check=False,
        )
        assert proc.returncode == 0, proc.stdout + proc.stderr
        run_dir = runs_dir / "repeat"
        metrics_path = run_dir / "metrics.csv"
        assert metrics_path.exists()
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["cli_args"]["seed"] == 101
        assert manifest["determinism"]["torch_seed_u32"] == 101
        hashes.append(
            (
                _sha256(metrics_path),
                _sha256(run_dir / "config.json"),
                _sha256(run_dir / "run_config.csv"),
            )
        )

    assert hashes[0] == hashes[1]
