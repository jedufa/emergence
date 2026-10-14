from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[3]
REGISTRY = ROOT / "_spec/registry/v1/phaseIV_cog_p1_perception_selection_probe.yaml"
HELPER = ROOT / "tools/bin/phaseIV_cog_p1_perception_selection_report.py"


def _load_helper():
    spec = importlib.util.spec_from_file_location("phaseIV_cog_p1_perception_selection_report", HELPER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _knobs(flags: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for idx, item in enumerate(flags):
        if item == "--knob" and idx + 1 < len(flags):
            key, _, value = str(flags[idx + 1]).partition("=")
            out[key] = value
    return out


def _default_knobs(registry: dict) -> dict[str, str]:
    out: dict[str, str] = {}
    for pair in registry["cli_defaults"]:
        if pair[0] == "--knob":
            key, _, value = str(pair[1]).partition("=")
            out[key] = value
    return out


def test_cogp1_registry_matrix_and_cuda_defaults() -> None:
    registry = yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))
    axes = registry["axes"]
    assert registry["matrix_order"] == ["ecology_context", "init_arm", "perception_condition", "seed"]
    assert axes["ecology_context"]["values"] == [
        "cogp1_flat_control_clean",
        "cogp1_tuned_full_jungle_recovery",
        "cogp1_theme12_clean_pressure",
    ]
    assert axes["init_arm"]["values"] == ["init_legacy", "init_founder_balanced"]
    assert axes["perception_condition"]["values"] == ["perception_normal", "perception_ablated"]
    assert registry["seeds"] == [101, 202, 303, 404, 505, 606, 707, 808, 909, 1001]
    assert len(axes["ecology_context"]["values"]) * len(axes["init_arm"]["values"]) * len(
        axes["perception_condition"]["values"]
    ) * len(registry["seeds"]) == 120
    flattened_defaults = [value for pair in registry["cli_defaults"] for value in pair]
    assert "--device" in flattened_defaults
    assert "cuda" in flattened_defaults
    assert "sim_cfg.device=cuda" in flattened_defaults
    assert "sim_cfg.bank_device=cuda" in flattened_defaults
    assert ["--generations", "16"] in registry["cli_defaults"]
    assert ["--steps", "80"] in registry["cli_defaults"]


def test_cogp1_b1_flags_are_only_in_founder_balanced_arm() -> None:
    registry = yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))
    flags = registry["axes"]["init_arm"]["flags"]
    legacy = _knobs(flags["init_legacy"])
    balanced = _knobs(flags["init_founder_balanced"])
    assert legacy["sim_cfg.lineage_founder_balance_enable"] == "false"
    assert balanced["sim_cfg.lineage_founder_balance_enable"] == "true"
    assert balanced["sim_cfg.lineage_founder_group_count"] == "8"
    assert balanced["sim_cfg.lineage_founder_min_agents_per_group"] == "8"
    assert balanced["sim_cfg.lineage_founder_assignment_mode"] == "deterministic_round_robin"


def test_cogp1_perception_ablation_flags_are_condition_scoped() -> None:
    registry = yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))
    flags = registry["axes"]["perception_condition"]["flags"]
    assert flags["perception_normal"] == []
    ablated = _knobs(flags["perception_ablated"])
    assert ablated["sim_cfg.force_minimal_sensor_channels_enabled"] == "true"
    defaults = _default_knobs(registry)
    assert defaults["sim_cfg.force_default_sensor_radius"] == "true"
    assert defaults["sim_cfg.enable_evolvable_sensor_channel_mask"] == "true"
    assert defaults["sim_cfg.enable_experimental_perception_cost_charging"] == "true"


def test_cogp1_direct_reward_guards_are_explicit() -> None:
    registry = yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))
    defaults = _default_knobs(registry)
    assert defaults["sim_cfg.disable_prediction_error_selection_pressure"] == "true"
    assert defaults["sim_cfg.episode0_fitness_weight"] == "0.0"
    assert defaults["sim_cfg.episode0_pred_err_fitness_weight"] == "0.0"
    assert defaults["sim_cfg.episode0_thrashing_fitness_weight"] == "0.0"
    assert defaults["sim_cfg.episode0_drive_jitter_fitness_weight"] == "0.0"
    assert defaults["sim_cfg.episode0_selfharm_fitness_weight"] == "0.0"
    assert defaults["sim_cfg.late_pred_cost_enable"] == "false"
    assert defaults["sim_cfg.late_coherence_eff_enable"] == "false"
    assert defaults["sim_cfg.adult_move_penalty_enable"] == "false"
    assert defaults["sim_cfg.identity_drift_enable"] == "false"


def test_cogp1_reward_audit_and_decision_schema_for_synthetic_no_go(tmp_path: Path) -> None:
    helper = _load_helper()
    audit = helper.build_reward_audit(tmp_path)
    assert audit["valid"] is True
    assert audit["direct_cognition_reward_active"] is False
    assert audit["direct_perception_reward_active"] is False
    assert audit["direct_diversity_reward_active"] is False

    rows = []
    for ecology in helper.ECOLOGIES:
        for init in helper.INIT_ARMS:
            for perception in helper.PERCEPTION_CONDITIONS:
                for seed in helper.SEEDS:
                    rows.append(
                        {
                            "ecology_context": ecology,
                            "init_arm": init,
                            "perception_condition": perception,
                            "seed": seed,
                            "status": "ok",
                            "perception_cost_charged_mean": 0.01,
                            "sensor_channel_force_minimal_active_flag": 1.0
                            if perception == "perception_ablated"
                            else 0.0,
                            "sensor_radius_force_default_active_flag": 1.0,
                        }
                    )
    summary = {
        "integrity": {
            "valid": True,
            "expected_runs": 120,
            "observed_runs": 120,
            "completed_runs": 120,
            "failed_runs": 0,
        },
        "reward_audit": audit,
        "runs": rows,
        "cost_sanity": {
            "perception_cost_available_runs": 120,
            "perception_cost_positive_runs": 120,
            "ablated_force_minimal_runs": 60,
            "normal_force_minimal_runs": 0,
            "force_default_sensor_radius_runs": 120,
        },
        "pressure_specificity": [
            {
                "pressure_context": pressure,
                "init_arm": init,
                "pressure_minus_flat_delta_alive": 0.0,
                "pressure_minus_flat_delta_mean_offspring_total": 0.0,
                "pressure_mean_delta_alive": 0.0,
                "pressure_mean_delta_mean_offspring_total": 0.0,
                "seed_alive_pressure_exceeds_flat_fraction": 0.0,
                "seed_births_pressure_exceeds_flat_fraction": 0.0,
            }
            for pressure in helper.PRESSURE_ECOLOGIES
            for init in helper.INIT_ARMS
        ],
    }
    decision = helper.decide(summary)
    assert decision["decision"] == "COG-P1-B"
    json.dumps(decision)
