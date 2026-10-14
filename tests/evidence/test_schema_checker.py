from __future__ import annotations

import csv
import json
from pathlib import Path

from tools.evidence.contracts import check_campaign_schema
from tools.evidence.profiles import COG_P1_ECOLOGIES, COG_P1_INIT_ARMS, COG_P1_PERCEPTION_CONDITIONS, COG_P1_SEEDS


METRIC_COLUMNS = [
    "generation",
    "alive",
    "mean_offspring_total",
    "lineage_effective_alive_lineages",
    "lineage_alive_share_entropy",
    "lineage_active_count",
    "founder_initial_effective_lineages",
    "founder_initial_top1_share",
    "perception_cost_charged_mean",
    "sensor_radius_cost_charged_mean",
    "sensor_channel_active_count_mean",
    "sensor_channel_force_minimal_active_flag",
    "sensor_radius_force_default_active_flag",
    "behavior_action_total",
    "behavior_patch_interaction_proxy",
    "behavior_risk_exposure_proxy",
    "direct_cognition_reward_present_flag",
    "direct_internal_selection_pressure_present_flag",
    "direct_prediction_error_selection_pressure_present_flag",
    "prediction_direct_selection_pressure_disabled_flag",
    "direct_development_reward_present_flag",
    "direct_introspection_reward_present_flag",
    "direct_internal_state_reward_present_flag",
    "direct_self_regulation_reward_present_flag",
    "direct_self_modification_reward_present_flag",
    "direct_directed_inheritance_reward_present_flag",
    "direct_parental_investment_reward_present_flag",
    "direct_offspring_quality_reward_present_flag",
]


def _config(ecology: str, init_arm: str, perception: str) -> dict:
    pressure = ecology != "cogp1_flat_control_clean"
    return {
        "env_cfg": {
            "patch_world_enable": pressure,
            "patch_world_observation_enable": pressure,
            "patch_world_waste_consequence_enable": pressure,
            "jungle_predator_pressure_enable": pressure,
            "jungle_reproduction_emissions_enable": pressure,
            "jungle_terrain_protection_enable": pressure,
            "jungle_ecological_recovery_enable": pressure,
        },
        "sim_cfg": {
            "enable_cognitive_trait_registry": True,
            "enable_cognitive_status_metrics": True,
            "enable_cognitive_budget_ledger": True,
            "enable_evolvable_sensor_channel_mask": True,
            "force_default_sensor_radius": True,
            "enable_experimental_perception_cost_charging": True,
            "disable_prediction_error_selection_pressure": True,
            "episode0_fitness_weight": 0.0,
            "episode0_pred_err_fitness_weight": 0.0,
            "episode0_thrashing_fitness_weight": 0.0,
            "episode0_drive_jitter_fitness_weight": 0.0,
            "episode0_selfharm_fitness_weight": 0.0,
            "late_pred_cost_enable": False,
            "late_coherence_eff_enable": False,
            "adult_move_penalty_enable": False,
            "identity_drift_enable": False,
            "lineage_founder_balance_enable": init_arm == "init_founder_balanced",
            "force_minimal_sensor_channels_enabled": perception == "perception_ablated",
        },
    }


def _metric_row(seed: int, perception: str) -> dict[str, str]:
    return {
        "generation": "15",
        "alive": str(100 + seed % 17),
        "mean_offspring_total": "2.5",
        "lineage_effective_alive_lineages": "4.0",
        "lineage_alive_share_entropy": "1.2",
        "lineage_active_count": "5",
        "founder_initial_effective_lineages": "8",
        "founder_initial_top1_share": "0.2",
        "perception_cost_charged_mean": "0.01",
        "sensor_radius_cost_charged_mean": "0.01",
        "sensor_channel_active_count_mean": "3",
        "sensor_channel_force_minimal_active_flag": "1" if perception == "perception_ablated" else "0",
        "sensor_radius_force_default_active_flag": "1",
        "behavior_action_total": "12",
        "behavior_patch_interaction_proxy": "1",
        "behavior_risk_exposure_proxy": "2",
        "direct_cognition_reward_present_flag": "0",
        "direct_internal_selection_pressure_present_flag": "0",
        "direct_prediction_error_selection_pressure_present_flag": "0",
        "prediction_direct_selection_pressure_disabled_flag": "1",
        "direct_development_reward_present_flag": "0",
        "direct_introspection_reward_present_flag": "0",
        "direct_internal_state_reward_present_flag": "0",
        "direct_self_regulation_reward_present_flag": "0",
        "direct_self_modification_reward_present_flag": "0",
        "direct_directed_inheritance_reward_present_flag": "0",
        "direct_parental_investment_reward_present_flag": "0",
        "direct_offspring_quality_reward_present_flag": "0",
    }


def build_cogp1_fixture(tmp_path: Path) -> tuple[Path, list[dict]]:
    run_root = tmp_path / "runs"
    runs = []
    counter = 0
    for ecology in COG_P1_ECOLOGIES:
        for init_arm in COG_P1_INIT_ARMS:
            for perception in COG_P1_PERCEPTION_CONDITIONS:
                for seed in COG_P1_SEEDS:
                    counter += 1
                    run_dir = run_root / f"r{counter:04d}"
                    run_dir.mkdir(parents=True)
                    with (run_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
                        writer = csv.DictWriter(handle, fieldnames=METRIC_COLUMNS)
                        writer.writeheader()
                        writer.writerow(_metric_row(seed, perception))
                    (run_dir / "config.json").write_text(
                        json.dumps(_config(ecology, init_arm, perception), sort_keys=True),
                        encoding="utf-8",
                    )
                    runs.append(
                        {
                            "run_id": f"run-{counter:04d}",
                            "run_dir": str(run_dir),
                            "seed": seed,
                            "status": "ok",
                            "returncode": 0,
                            "axes": {
                                "ecology_context": ecology,
                                "init_arm": init_arm,
                                "perception_condition": perception,
                            },
                        }
                    )
    summary_path = run_root / "summary.json"
    summary_path.write_text(json.dumps({"status": "ok", "planned_runs": 120, "runs": runs}), encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    manifest = {
        "campaign_id": "phaseIV_cog_p1_perception_selection_probe",
        "profile": "COG-P1",
        "git_commit": "abc",
        "branch": "fixture",
        "worktree": ".",
        "run_root": str(run_root),
        "registry_path": "registry.yaml",
        "scenario_matrix": {
            "dimensions": [
                {"name": "ecology_context", "values": list(COG_P1_ECOLOGIES)},
                {"name": "init_arm", "values": list(COG_P1_INIT_ARMS)},
                {"name": "perception_condition", "values": list(COG_P1_PERCEPTION_CONDITIONS)},
            ]
        },
        "seed_list": list(COG_P1_SEEDS),
        "generations": 16,
        "steps_per_generation": 80,
        "population_size": 64,
        "grid_size": 16,
        "device": "cuda",
        "planned_run_count": 120,
        "completed_run_count": 120,
        "failed_run_count": 0,
        "config_files": [],
        "output_paths": [],
        "metrics_glob": "**/metrics.csv",
        "config_glob": "**/config.json",
        "harness_manifest_paths": [],
        "harness_summary_paths": [str(summary_path)],
        "timestamp_utc": "2026-07-02T00:00:00Z",
    }
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    return manifest_path, runs


def test_valid_campaign_output_passes(tmp_path: Path) -> None:
    manifest_path, _runs = build_cogp1_fixture(tmp_path)
    result = check_campaign_schema(manifest_path)

    assert result["ok"] is True
    assert result["run_count_observed"] == 120
    assert result["run_count_expected"] == 120
    assert result["scenario_count_observed"] == 12
    assert result["seed_count_observed"] == 10


def test_missing_required_column_fails(tmp_path: Path) -> None:
    manifest_path, runs = build_cogp1_fixture(tmp_path)
    metrics = Path(runs[0]["run_dir"]) / "metrics.csv"
    rows = list(csv.DictReader(metrics.open(encoding="utf-8")))
    columns = [column for column in METRIC_COLUMNS if column != "alive"]
    with metrics.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerow({column: rows[0][column] for column in columns})

    result = check_campaign_schema(manifest_path)
    assert result["ok"] is False
    assert any("missing required metrics columns: alive" in error for error in result["errors"])


def test_malformed_json_field_fails(tmp_path: Path) -> None:
    manifest_path, runs = build_cogp1_fixture(tmp_path)
    (Path(runs[0]["run_dir"]) / "config.json").write_text("{bad json", encoding="utf-8")

    result = check_campaign_schema(manifest_path)
    assert result["ok"] is False
    assert any("config JSON unreadable" in error for error in result["errors"])


def test_nan_inf_metric_fails(tmp_path: Path) -> None:
    manifest_path, runs = build_cogp1_fixture(tmp_path)
    metrics = Path(runs[0]["run_dir"]) / "metrics.csv"
    row = _metric_row(COG_P1_SEEDS[0], "perception_normal")
    row["alive"] = "inf"
    with metrics.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_COLUMNS)
        writer.writeheader()
        writer.writerow(row)

    result = check_campaign_schema(manifest_path)
    assert result["ok"] is False
    assert any("non-finite metric values" in error for error in result["errors"])


def test_missing_seed_and_scenario_are_reported(tmp_path: Path) -> None:
    manifest_path, runs = build_cogp1_fixture(tmp_path)
    summary = json.loads((Path(runs[0]["run_dir"]).parents[0] / "summary.json").read_text(encoding="utf-8"))
    summary["runs"] = summary["runs"][:-1]
    summary["runs"][0]["seed"] = 999
    (Path(runs[0]["run_dir"]).parents[0] / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

    result = check_campaign_schema(manifest_path)
    assert result["ok"] is False
    assert any("unexpected seed: 999" in error for error in result["errors"])
    assert result["anomaly_summary"]["counts"]["missing_run"] >= 1


def test_duplicate_empty_metrics_and_failed_runs_are_data(tmp_path: Path) -> None:
    manifest_path, runs = build_cogp1_fixture(tmp_path)
    summary_path = Path(runs[0]["run_dir"]).parents[0] / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["runs"].append(dict(summary["runs"][0]))
    summary["runs"][1]["status"] = "failed"
    summary["runs"][1]["returncode"] = 2
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    (Path(runs[2]["run_dir"]) / "metrics.csv").write_text("", encoding="utf-8")

    result = check_campaign_schema(manifest_path)
    assert result["ok"] is False
    counts = result["anomaly_summary"]["counts"]
    assert counts["duplicate_run"] >= 1
    assert counts["duplicate_scenario_seed"] >= 1
    assert counts["empty_metrics"] >= 1
    assert counts["failed_run"] >= 1


def test_large_metrics_field_does_not_break_csv_reader(tmp_path: Path) -> None:
    manifest_path, runs = build_cogp1_fixture(tmp_path)
    metrics = Path(runs[0]["run_dir"]) / "metrics.csv"
    row = _metric_row(COG_P1_SEEDS[0], "perception_normal")
    row["large_debug_payload"] = "x" * 200_000
    columns = METRIC_COLUMNS + ["large_debug_payload"]
    with metrics.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerow(row)

    result = check_campaign_schema(manifest_path)
    assert result["ok"] is True
