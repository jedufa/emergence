"""Decision-layer profile contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class ConfigExpectation:
    section: str
    field: str
    expected: Any
    message: str


@dataclass(frozen=True)
class ProfileContract:
    name: str
    required_dimensions: tuple[str, ...]
    expected_values: dict[str, tuple[Any, ...]]
    required_metric_columns: tuple[str, ...]
    finite_metric_columns: tuple[str, ...]
    reward_leakage_columns: tuple[str, ...]
    config_expectations: tuple[ConfigExpectation, ...] = ()
    primary_comparisons: tuple[str, ...] = ()
    forbidden_claims: tuple[str, ...] = ()
    implemented_decision_logic: bool = False


COG_P1_ECOLOGIES = (
    "cogp1_flat_control_clean",
    "cogp1_tuned_full_jungle_recovery",
    "cogp1_theme12_clean_pressure",
)
COG_P1_INIT_ARMS = ("init_legacy", "init_founder_balanced")
COG_P1_PERCEPTION_CONDITIONS = ("perception_normal", "perception_ablated")
COG_P1_SEEDS = (101, 202, 303, 404, 505, 606, 707, 808, 909, 1001)

COG_P1_REQUIRED_COLUMNS = (
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
    "direct_development_reward_present_flag",
    "direct_introspection_reward_present_flag",
    "direct_self_regulation_reward_present_flag",
    "direct_self_modification_reward_present_flag",
    "direct_directed_inheritance_reward_present_flag",
)

COG_P1_CONFIG_EXPECTATIONS = (
    ConfigExpectation("sim_cfg", "enable_cognitive_trait_registry", True, "cognitive trait registry must be enabled"),
    ConfigExpectation("sim_cfg", "enable_cognitive_status_metrics", True, "cognitive status metrics must be enabled"),
    ConfigExpectation("sim_cfg", "enable_cognitive_budget_ledger", True, "cognitive budget ledger must be enabled"),
    ConfigExpectation("sim_cfg", "enable_evolvable_sensor_channel_mask", True, "sensor channel mask must be enabled"),
    ConfigExpectation("sim_cfg", "force_default_sensor_radius", True, "default sensor radius must be forced"),
    ConfigExpectation("sim_cfg", "enable_experimental_perception_cost_charging", True, "perception cost charging must be enabled"),
    ConfigExpectation("sim_cfg", "disable_prediction_error_selection_pressure", True, "prediction-error selection pressure must be disabled"),
    ConfigExpectation("sim_cfg", "episode0_fitness_weight", 0.0, "episode0 fitness weight must be zero"),
    ConfigExpectation("sim_cfg", "episode0_pred_err_fitness_weight", 0.0, "episode0 prediction-error weight must be zero"),
    ConfigExpectation("sim_cfg", "episode0_thrashing_fitness_weight", 0.0, "episode0 thrashing weight must be zero"),
    ConfigExpectation("sim_cfg", "episode0_drive_jitter_fitness_weight", 0.0, "episode0 drive-jitter weight must be zero"),
    ConfigExpectation("sim_cfg", "episode0_selfharm_fitness_weight", 0.0, "episode0 self-harm weight must be zero"),
    ConfigExpectation("sim_cfg", "late_pred_cost_enable", False, "late prediction cost must be disabled"),
    ConfigExpectation("sim_cfg", "late_coherence_eff_enable", False, "late coherence efficiency must be disabled"),
    ConfigExpectation("sim_cfg", "adult_move_penalty_enable", False, "adult move penalty must be disabled"),
    ConfigExpectation("sim_cfg", "identity_drift_enable", False, "identity drift must be disabled"),
)

MEMORY_ASSAY_CONFIG_EXPECTATIONS = (
    ConfigExpectation("sim_cfg", "disable_prediction_error_selection_pressure", True, "prediction-error selection pressure must be disabled"),
    ConfigExpectation("sim_cfg", "episode0_fitness_weight", 0.0, "episode0 fitness weight must be zero"),
    ConfigExpectation("sim_cfg", "episode0_pred_err_fitness_weight", 0.0, "episode0 prediction-error weight must be zero"),
    ConfigExpectation("sim_cfg", "episode0_thrashing_fitness_weight", 0.0, "episode0 thrashing weight must be zero"),
    ConfigExpectation("sim_cfg", "episode0_drive_jitter_fitness_weight", 0.0, "episode0 drive-jitter weight must be zero"),
    ConfigExpectation("sim_cfg", "episode0_selfharm_fitness_weight", 0.0, "episode0 self-harm weight must be zero"),
    ConfigExpectation("sim_cfg", "late_pred_cost_enable", False, "late prediction cost must be disabled"),
    ConfigExpectation("sim_cfg", "late_coherence_eff_enable", False, "late coherence efficiency must be disabled"),
    ConfigExpectation("sim_cfg", "adult_move_penalty_enable", False, "adult move penalty must be disabled"),
    ConfigExpectation("sim_cfg", "identity_drift_enable", False, "identity drift must be disabled"),
)

QUAL_DELTA_1_REQUIRED_COLUMNS = (
    "generation",
    "qf_route_receipt_present_flag",
    "qf_route_receipt_sha256_match_flag",
    "qf_route_binding_exact_flag",
    "qf_source_binding_exact_flag",
    "qf_coverage_complete_flag",
    "qf_repeat_parity_pass_flag",
    "qf_checkpoint_resume_pass_flag",
    "qf_finite_outputs_flag",
    "qf_no_integrity_failure_flag",
    "qf_no_posthoc_widening_flag",
    "qf_region_supported_flag",
)


PROFILES: dict[str, ProfileContract] = {
    "COG-P1": ProfileContract(
        name="COG-P1",
        required_dimensions=("ecology_context", "init_arm", "perception_condition", "seed"),
        expected_values={
            "ecology_context": COG_P1_ECOLOGIES,
            "init_arm": COG_P1_INIT_ARMS,
            "perception_condition": COG_P1_PERCEPTION_CONDITIONS,
            "seed": COG_P1_SEEDS,
        },
        required_metric_columns=COG_P1_REQUIRED_COLUMNS,
        finite_metric_columns=COG_P1_REQUIRED_COLUMNS,
        reward_leakage_columns=(
            "direct_cognition_reward_present_flag",
            "direct_internal_selection_pressure_present_flag",
            "direct_prediction_error_selection_pressure_present_flag",
            "direct_development_reward_present_flag",
            "direct_introspection_reward_present_flag",
            "direct_internal_state_reward_present_flag",
            "direct_self_regulation_reward_present_flag",
            "direct_self_modification_reward_present_flag",
            "direct_directed_inheritance_reward_present_flag",
            "direct_parental_investment_reward_present_flag",
            "direct_offspring_quality_reward_present_flag",
        ),
        config_expectations=COG_P1_CONFIG_EXPECTATIONS,
        primary_comparisons=(
            "normal_vs_ablated",
            "pressure_specificity_over_flat",
            "founder_balanced_vs_legacy",
        ),
        forbidden_claims=(
            "cognition emergence",
            "open-ended evolution",
            "ecological generality beyond the campaign matrix",
            "statistical significance",
        ),
        implemented_decision_logic=True,
    ),
    "COG-P2": ProfileContract(
        name="COG-P2",
        required_dimensions=("ecology_context", "init_arm", "cognition_condition", "seed"),
        expected_values={},
        required_metric_columns=("generation", "alive", "mean_offspring_total"),
        finite_metric_columns=("alive", "mean_offspring_total"),
        reward_leakage_columns=(),
        primary_comparisons=("reserved_cog_p2_primary_comparison",),
        forbidden_claims=("cognition emergence", "open-ended evolution", "scientific validation"),
        implemented_decision_logic=False,
    ),
    "CRB-1": ProfileContract(
        name="CRB-1",
        required_dimensions=("ecology_context", "init_arm", "perception_condition", "seed"),
        expected_values={},
        required_metric_columns=("generation", "alive", "mean_offspring_total", "perception_cost_charged_mean"),
        finite_metric_columns=("alive", "mean_offspring_total", "perception_cost_charged_mean"),
        reward_leakage_columns=("direct_cognition_reward_present_flag",),
        primary_comparisons=("normal_vs_ablated",),
        forbidden_claims=("scientific validation", "ecological generality beyond the campaign matrix"),
        implemented_decision_logic=False,
    ),
    "CRB-M1": ProfileContract(
        name="CRB-M1",
        required_dimensions=("ecology_context", "init_arm", "memory_condition", "seed"),
        expected_values={},
        required_metric_columns=("generation", "alive", "mean_offspring_total"),
        finite_metric_columns=("alive", "mean_offspring_total"),
        reward_leakage_columns=("direct_cognition_reward_present_flag",),
        config_expectations=MEMORY_ASSAY_CONFIG_EXPECTATIONS,
        primary_comparisons=("reserved_memory_condition_comparison",),
        forbidden_claims=("memory emergence", "open-ended evolution", "scientific validation"),
        implemented_decision_logic=False,
    ),
    "QUAL-DELTA-1": ProfileContract(
        name="QUAL-DELTA-1",
        required_dimensions=("ecology_context", "init_arm", "perception_condition", "seed"),
        expected_values={},
        required_metric_columns=QUAL_DELTA_1_REQUIRED_COLUMNS,
        finite_metric_columns=QUAL_DELTA_1_REQUIRED_COLUMNS,
        reward_leakage_columns=(),
        primary_comparisons=("qualification_gate_vector",),
        forbidden_claims=(
            "intelligence",
            "emergence",
            "cognition",
            "ecological generality",
            "campaign-specific science",
            "post-hoc envelope expansion",
        ),
        implemented_decision_logic=True,
    ),
}


def get_profile(name: str) -> ProfileContract:
    try:
        return PROFILES[name]
    except KeyError as exc:
        known = ", ".join(sorted(PROFILES))
        raise KeyError(f"unknown evidence profile {name!r}; known profiles: {known}") from exc


def expected_dimension_values(profile: ProfileContract, manifest_values: Mapping[str, list[Any]]) -> dict[str, tuple[Any, ...]]:
    values: dict[str, tuple[Any, ...]] = {}
    for dimension in profile.required_dimensions:
        if dimension == "seed":
            continue
        profile_values = profile.expected_values.get(dimension)
        values[dimension] = tuple(profile_values or manifest_values.get(dimension, ()))
    return values
