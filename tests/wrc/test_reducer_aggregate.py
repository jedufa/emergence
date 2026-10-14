from __future__ import annotations

from engine.wrc.reducer import ReducerConfig
from engine.wrc.reducer.aggregate import classify_response_curve, reduce_parameter


def _metric(value: float) -> dict:
    return {
        "last": {"value": value},
        "time_weighted_mean": value,
        "maximum": {"value": value},
        "minimum": {"value": value},
        "slope_per_tick": 0.0,
        "net_change": 0.0,
    }


def _world(level: object, seed: int, value: float) -> dict:
    return {
        "identity": {
            "world_id": f"w-{level}-{seed}",
            "parameter_id": "env_cfg.switch",
            "parameter_value": level,
            "seed": seed,
        },
        "metrics": {
            "switch": _metric(1.0 if level is True else 0.0),
            "activation_signal": _metric(value),
            "alive": _metric(100.0 - value),
        },
    }


def test_boolean_parameter_reduction_is_paired_and_active() -> None:
    worlds = [
        *[_world(False, seed, 0.0) for seed in (101, 202, 303, 404)],
        *[_world(True, seed, 10.0 + index) for index, seed in enumerate((101, 202, 303, 404))],
    ]
    selected = {
        "ID": "PTEST",
        "Parameter": "env_cfg.switch",
        "Type": "bool",
        "Reference default": "false",
        "Family": "test",
        "World base": "patch",
        "Primary response metrics": "activation signal; alive",
    }
    plan = {
        "id": "PTEST",
        "parameter": "env_cfg.switch",
        "runtime_parameter": "env_cfg.switch",
        "type": "bool",
        "levels": [False, True],
        "seeds": [101, 202, 303, 404],
        "planned_runs": 8,
        "challenge": {"assertions": [{"metric": "activation_signal"}]},
        "world_overlays": [],
    }

    result = reduce_parameter(
        selected_row=selected,
        launch_plan=plan,
        world_results=worlds,
        config=ReducerConfig(),
    )

    assert result["coverage"]["complete"] is True
    assert result["screening"]["classification"] == "ACTIVE"
    assert result["screening"]["activation_status"] == "STRONG"
    assert result["screening"]["top_metric"] == "activation_signal"
    assert result["standardized_summary"]["control_transport"]["status"] == "CONFIRMED"
    assert result["standardized_summary"]["control_transport"]["metric"] == "switch"
    assert result["standardized_summary"]["layers"]["activation"]["status"] == "STRONG"
    assert all(row["metric"] != "switch" for row in result["ranked_metrics"])
    assert any(
        row["shape"] == "BOOLEAN_CONTRAST"
        for row in result["response_curves"]
        if row["metric"] == "activation_signal"
    )


def test_response_curve_classifies_monotonic_and_threshold_shapes() -> None:
    config = ReducerConfig(noisy_signal_to_noise_threshold=0.0)
    monotonic = classify_response_curve(
        [
            {"level": 0.0, "mean": 0.0, "standard_deviation": 0.01},
            {"level": 1.0, "mean": 1.0, "standard_deviation": 0.01},
            {"level": 2.0, "mean": 2.0, "standard_deviation": 0.01},
            {"level": 3.0, "mean": 3.0, "standard_deviation": 0.01},
        ],
        domain="ecology",
        config=config,
    )
    threshold = classify_response_curve(
        [
            {"level": 0.0, "mean": 10.0, "standard_deviation": 0.01},
            {"level": 1.0, "mean": 10.0, "standard_deviation": 0.01},
            {"level": 2.0, "mean": 9.9, "standard_deviation": 0.01},
            {"level": 3.0, "mean": 1.0, "standard_deviation": 0.01},
        ],
        domain="population",
        config=config,
    )

    assert monotonic["shape"] == "MONOTONIC_POSITIVE"
    assert threshold["shape"] == "CATASTROPHIC_THRESHOLD_NEGATIVE"
    assert threshold["threshold_level"] == 3.0
    assert threshold["collapse_level"] == 3.0
    assert threshold["first_effective_level"] == 3.0


def test_scientific_ranking_downweights_tiny_denominator_artifact() -> None:
    levels = [0.0, 0.1]
    seeds = [101, 202, 303, 404, 505, 606, 707, 808]
    worlds = []
    for level in levels:
        for seed in seeds:
            direct = 0.55 if level == 0.0 else 0.05
            tiny = 1.0e-5 if level == 0.0 else 5.0e-5
            worlds.append(
                {
                    "identity": {
                        "world_id": f"w-{level}-{seed}",
                        "parameter_id": "env_cfg.jungle_recovery_activity_density_penalty_weight",
                        "parameter_value": level,
                        "seed": seed,
                    },
                    "metrics": {
                        "patch_recovery_activity_mean": _metric(direct),
                        "prediction_by_age_slope": _metric(tiny),
                    },
                }
            )
    selected = {
        "ID": "P015",
        "Parameter": "env_cfg.jungle_recovery_activity_density_penalty_weight",
        "Type": "float",
        "Reference default": "0",
        "Family": "Jungle ecological recovery",
        "Primary response metrics": "patch_recovery_activity_mean",
    }
    plan = {
        "id": "P015",
        "parameter": selected["Parameter"],
        "runtime_parameter": selected["Parameter"],
        "type": "float",
        "levels": levels,
        "seeds": seeds,
        "planned_runs": len(worlds),
        "challenge": {"assertions": []},
    }

    result = reduce_parameter(
        selected_row=selected,
        launch_plan=plan,
        world_results=worlds,
        config=ReducerConfig(),
    )

    assert result["screening"]["top_metric"] == "patch_recovery_activity_mean"
    effects = {row["metric"]: row for row in result["metric_effects"] if row["feature"] == "terminal"}
    assert effects["prediction_by_age_slope"]["effect_score"] > 0.5
    assert effects["prediction_by_age_slope"]["scientific_materiality"] < 0.1
    assert (
        effects["patch_recovery_activity_mean"]["scientific_effect_score"]
        > effects["prediction_by_age_slope"]["scientific_effect_score"]
    )


def test_interpretation_semantics_separate_control_mechanism_and_downstream() -> None:
    seeds = [101, 202, 303, 404]
    worlds = []
    for enabled in (False, True):
        for seed in seeds:
            worlds.append(
                {
                    "identity": {
                        "world_id": f"w-{enabled}-{seed}",
                        "parameter_id": "env_cfg.regrowth_boost_enable",
                        "parameter_value": enabled,
                        "seed": seed,
                    },
                    "last_tick": 200,
                    "metrics": {
                        # Exact readback proves transport only.
                        "regrowth_boost_enable": _metric(1.0 if enabled else 0.0),
                        # Declared activation metric intentionally does not move.
                        "patch_recovery_activity_mean": _metric(0.2),
                        # Direct mechanism consequence moves strongly.
                        "regrowth_boost_total": _metric(100.0 if enabled else 0.0),
                        "alive": _metric(100.0),
                    },
                }
            )
    selected = {
        "ID": "PSEM",
        "Parameter": "env_cfg.regrowth_boost_enable",
        "Type": "bool",
        "Reference default": "false",
        "Family": "ecological recovery",
        "Primary response metrics": "regrowth_boost_total",
    }
    plan = {
        "id": "PSEM",
        "parameter": selected["Parameter"],
        "runtime_parameter": selected["Parameter"],
        "type": "bool",
        "levels": [False, True],
        "seeds": seeds,
        "planned_runs": 8,
        "challenge": {"assertions": [{"metric": "patch_recovery_activity_mean"}]},
    }

    result = reduce_parameter(
        selected_row=selected,
        launch_plan=plan,
        world_results=worlds,
        config=ReducerConfig(),
    )

    semantics = result["standardized_summary"]["interpretation_semantics"]
    assert semantics["control_path_activation"] == "CONFIRMED"
    assert result["screening"]["activation_status"] == "NO_EFFECT_DETECTED"
    assert semantics["mechanism_response"] == "STRONG"
    assert semantics["mechanism_metric"] == "regrowth_boost_total"
    assert result["screening"]["classification"] == "ACTIVE"


def test_supported_effect_without_onset_tick_is_not_reported_as_not_detected() -> None:
    seeds = [101, 202, 303, 404]
    worlds = []
    for level in (0.0, 1.0, 2.0):
        for seed in seeds:
            worlds.append(
                {
                    "identity": {
                        "world_id": f"w-{level}-{seed}",
                        "parameter_id": "env_cfg.recovery_gain",
                        "parameter_value": level,
                        "seed": seed,
                    },
                    "last_tick": 200,
                    "metrics": {
                        "recovery_gain": _metric(level),
                        "patch_recovery_activity_mean": _metric(level * 10.0),
                    },
                }
            )
    selected = {
        "ID": "PTIME",
        "Parameter": "env_cfg.recovery_gain",
        "Type": "float",
        "Reference default": "0",
        "Family": "ecological recovery",
        "Primary response metrics": "patch_recovery_activity_mean",
    }
    plan = {
        "id": "PTIME",
        "parameter": selected["Parameter"],
        "runtime_parameter": selected["Parameter"],
        "type": "float",
        "levels": [0.0, 1.0, 2.0],
        "seeds": seeds,
        "planned_runs": 12,
        "challenge": {"assertions": [{"metric": "patch_recovery_activity_mean"}]},
    }

    result = reduce_parameter(
        selected_row=selected,
        launch_plan=plan,
        world_results=worlds,
        config=ReducerConfig(noisy_signal_to_noise_threshold=0.0),
    )

    dynamics = result["standardized_summary"]["dynamics"]
    assert dynamics["pattern"] in {"PERSISTENT", "THRESHOLD_OR_COLLAPSE"}
    assert dynamics["timing"] == "EFFECT_PRESENT_ONSET_UNRESOLVED"
    assert dynamics["causal_onset_claim"] is False
