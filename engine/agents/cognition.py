from __future__ import annotations

"""Stage 0 cognitive trait registry, phenotype expression, and budget ledger.

This module is intentionally read-only with respect to agent behavior. It gives
existing cognitive loci stable metadata and deterministic summaries so later
Theme 4 stages have one place to extend expression, costs, and control surfaces.
"""

from dataclasses import asdict, dataclass
import time
from typing import Any, Iterable, Mapping, Sequence

from agents.perception import (
    SENSOR_RADIUS_LADDER,
    express_perception,
    implemented_sensor_channel_mask,
    minimal_sensor_channel_mask,
    perception_channel_registry,
    sensor_channel_mask_contains,
)
from agents.introspection import (
    INTROSPECTION_EXPRESSION_SCHEMA_VERSION,
    IntrospectionExpression,
    express_introspection,
    implemented_introspection_channel_mask,
    introspection_ablation_flags,
    introspection_channel_registry,
)
from agents.self_regulation import (
    SELF_REGULATION_EXPRESSION_SCHEMA_VERSION,
    SelfRegulationExpression,
    express_self_regulation,
    implemented_self_regulation_control_mask,
    self_regulation_ablation_flags,
    self_regulation_control_registry,
)
from agents.self_modification import (
    SELF_MODIFICATION_EXPRESSION_SCHEMA_VERSION,
    SelfModificationExpression,
    express_self_modification,
    implemented_self_modification_target_mask,
    self_modification_ablation_flags,
    self_modification_target_registry,
)
from agents.directed_inheritance import (
    DIRECTED_INHERITANCE_EXPRESSION_SCHEMA_VERSION,
    DirectedInheritancePayload,
    DirectedInheritanceExpression,
    directed_inheritance_payload_summary,
    directed_inheritance_channel_registry,
    directed_inheritance_ablation_flags,
    express_directed_inheritance,
    implemented_directed_inheritance_channel_mask,
)


REGISTRY_SCHEMA_VERSION = "cognitive_trait_registry_v1"
PHENOTYPE_SCHEMA_VERSION = "cognitive_phenotype_v1"
BUDGET_LEDGER_SCHEMA_VERSION = "cognitive_budget_ledger_v1"
NATIVE_BUDGET_AGGREGATION_SCHEMA_VERSION = "native_cognitive_budget_cost_aggregation_v1"
COST_CHARGE_SURFACE_SCHEMA_VERSION = "cost_charge_surface_v1"

MEMORY_LADDER: tuple[int, ...] = (0, 8, 16, 32, 64)
FIXED_POLICY_HIDDEN_SIZE = 64
_BUDGET_COST_COMPONENT_COLUMNS = 8


@dataclass(frozen=True)
class CognitiveTraitMetadata:
    trait_id: str
    display_name: str
    category: str
    genetic_surface_present: bool
    phenotype_surface_present: bool
    developmental_ready: bool
    introspection_ready: bool
    control_ready: bool
    heritable: bool
    mutable: bool
    directly_mutable: bool
    structurally_mutable: bool
    recombinable: bool
    costed: bool
    ablatable: bool
    metric_enabled: bool
    checkpoint_required: bool
    default_value: object
    min_value: object = None
    max_value: object = None
    allowed_ladder: tuple[object, ...] = ()
    units: str = ""
    interpretation: str = ""
    status: str = "active"
    description: str = ""

    def to_dict(self) -> dict[str, object]:
        out = asdict(self)
        out["allowed_ladder"] = list(self.allowed_ladder)
        return out


def _trait(
    trait_id: str,
    display_name: str,
    category: str,
    *,
    default_value: object,
    min_value: object = None,
    max_value: object = None,
    allowed_ladder: Iterable[object] = (),
    genetic: bool = True,
    phenotype: bool = True,
    developmental: bool = False,
    introspection: bool = False,
    control: bool = False,
    heritable: bool = True,
    mutable: bool = True,
    directly_mutable: bool | None = None,
    structurally_mutable: bool = False,
    recombinable: bool = True,
    costed: bool = False,
    ablatable: bool = False,
    metric: bool = True,
    checkpoint_required: bool = True,
    units: str = "",
    interpretation: str = "",
    status: str = "active",
    description: str = "",
) -> CognitiveTraitMetadata:
    return CognitiveTraitMetadata(
        trait_id=str(trait_id),
        display_name=str(display_name),
        category=str(category),
        genetic_surface_present=bool(genetic),
        phenotype_surface_present=bool(phenotype),
        developmental_ready=bool(developmental),
        introspection_ready=bool(introspection),
        control_ready=bool(control),
        heritable=bool(heritable),
        mutable=bool(mutable),
        directly_mutable=bool(mutable if directly_mutable is None else directly_mutable),
        structurally_mutable=bool(structurally_mutable),
        recombinable=bool(recombinable),
        costed=bool(costed),
        ablatable=bool(ablatable),
        metric_enabled=bool(metric),
        checkpoint_required=bool(checkpoint_required),
        default_value=default_value,
        min_value=min_value,
        max_value=max_value,
        allowed_ladder=tuple(allowed_ladder),
        units=str(units),
        interpretation=str(interpretation),
        status=str(status),
        description=str(description),
    )


_COGNITIVE_TRAITS: tuple[CognitiveTraitMetadata, ...] = (
    _trait(
        "memory_dim",
        "Memory Dimension",
        "memory",
        default_value=0,
        allowed_ladder=MEMORY_LADDER,
        costed=True,
        ablatable=True,
        units="hidden units",
        interpretation="Discrete genome ladder; values greater than zero enable policy memory.",
        description="Existing evolvable memory capacity gene.",
    ),
    _trait(
        "policy_memory_enabled",
        "Policy Memory Enabled",
        "memory",
        default_value=False,
        costed=True,
        ablatable=True,
        units="boolean",
        interpretation="Derived from use_gru or positive memory_dim.",
        description="Existing recurrent policy activation surface.",
        mutable=False,
        directly_mutable=False,
    ),
    _trait(
        "policy_hidden_size",
        "Policy Hidden Size",
        "action_policy",
        default_value=FIXED_POLICY_HIDDEN_SIZE,
        min_value=FIXED_POLICY_HIDDEN_SIZE,
        max_value=FIXED_POLICY_HIDDEN_SIZE,
        mutable=False,
        recombinable=False,
        units="hidden units",
        interpretation="Currently fixed at runtime; not an active evolutionary axis.",
        status="fixed",
        description="Registered for future variable architecture support without changing tensor shapes now.",
    ),
    _trait(
        "action_temperature",
        "Action Temperature",
        "action_policy",
        default_value=1.0,
        min_value=0.25,
        max_value=2.5,
        units="softmax temperature",
        description="Existing evolvable decision stochasticity gene.",
    ),
    _trait(
        "exploration_bias",
        "Exploration Bias",
        "action_policy",
        default_value=0.0,
        min_value=-1.0,
        max_value=1.0,
        description="Existing behavior-bias gene.",
    ),
    _trait(
        "memory_enabled",
        "Memory Enabled",
        "memory",
        default_value=False,
        genetic=False,
        phenotype=True,
        heritable=False,
        mutable=False,
        directly_mutable=False,
        recombinable=False,
        costed=True,
        ablatable=True,
        units="boolean",
        interpretation="Derived phenotype flag for positive effective memory capacity.",
        description="Stage 2 runtime expression of whether temporal memory capacity is active.",
    ),
    _trait(
        "effective_memory_dim",
        "Effective Memory Dimension",
        "memory",
        default_value=0,
        allowed_ladder=MEMORY_LADDER,
        genetic=False,
        phenotype=True,
        heritable=False,
        mutable=False,
        directly_mutable=False,
        recombinable=False,
        costed=True,
        ablatable=True,
        units="hidden units",
        interpretation="Runtime memory dimension after deterministic ablation overrides.",
        description="Stage 2 behavior-facing memory capacity; genome memory_dim is not mutated by overrides.",
    ),
    _trait(
        "memory_capacity",
        "Memory Capacity",
        "memory",
        default_value=0.0,
        min_value=0.0,
        max_value=float(max(MEMORY_LADDER)),
        genetic=False,
        phenotype=True,
        heritable=False,
        mutable=False,
        directly_mutable=False,
        recombinable=False,
        costed=True,
        ablatable=True,
        units="hidden units",
        interpretation="Raw effective memory capacity component; not an intelligence score.",
        description="Stage 2 capacity surface dominated by the fixed memory ladder.",
    ),
    _trait(
        "memory_cost",
        "Memory Cost",
        "cognition_budget",
        default_value=0.0,
        min_value=0.0,
        genetic=False,
        phenotype=True,
        heritable=False,
        mutable=False,
        directly_mutable=False,
        recombinable=False,
        costed=True,
        ablatable=True,
        units="energy per tick",
        interpretation="Existing GRU upkeep cost, plus future gated memory-cost experiments if added.",
        description="Stage 2 separates hypothetical and charged memory upkeep without double-charging.",
    ),
    _trait(
        "recurrent_state_enabled",
        "Recurrent State Enabled",
        "memory",
        default_value=False,
        genetic=False,
        phenotype=True,
        heritable=False,
        mutable=False,
        directly_mutable=False,
        recombinable=False,
        costed=True,
        ablatable=True,
        units="boolean",
        interpretation="Whether the fixed GRU policy path exists for this agent; not the same as useful memory.",
        description="Stage 2 architecture-status flag. Runtime tensor shape remains fixed.",
    ),
    _trait(
        "memory_ladder_index",
        "Memory Ladder Index",
        "memory",
        default_value=0,
        min_value=0,
        max_value=len(MEMORY_LADDER) - 1,
        genetic=False,
        phenotype=True,
        heritable=False,
        mutable=False,
        directly_mutable=False,
        recombinable=False,
        costed=False,
        ablatable=True,
        units="index",
        interpretation="Canonical index of the effective memory dimension in the fixed memory ladder.",
        description="Stage 2 deterministic serialization and metrics helper.",
    ),
    _trait(
        "memory_ablation_state",
        "Memory Ablation State",
        "memory",
        default_value="normal",
        genetic=False,
        phenotype=True,
        heritable=False,
        mutable=False,
        directly_mutable=False,
        recombinable=False,
        costed=False,
        ablatable=True,
        metric=True,
        checkpoint_required=False,
        units="state",
        interpretation="Runtime-only memory ablation mode; genomes remain unchanged.",
        description="Stage 2 memory ablation surface for paired usefulness tests.",
    ),
    _trait(
        "memory_use_signal",
        "Memory Use Signal",
        "memory",
        default_value=0.0,
        genetic=False,
        phenotype=True,
        heritable=False,
        mutable=False,
        directly_mutable=False,
        recombinable=False,
        costed=False,
        ablatable=False,
        metric=True,
        checkpoint_required=False,
        units="diagnostic",
        interpretation="Hidden-state telemetry when the fixed GRU bank is available; not intelligence.",
        status="partially_active",
        description="Stage 2 diagnostic surface for memory activity and leakage checks.",
    ),
    _trait(
        "memory_pressure_test_ready",
        "Memory Pressure Test Ready",
        "memory",
        default_value=False,
        genetic=False,
        phenotype=True,
        heritable=False,
        mutable=False,
        directly_mutable=False,
        recombinable=False,
        costed=False,
        ablatable=False,
        metric=True,
        checkpoint_required=False,
        units="boolean",
        interpretation="Scenario metadata flag for controlled memory-pressure scaffolds.",
        status="partially_active",
        description="Stage 2 fixture/scenario hook. It does not reward memory directly.",
    ),
    _trait(
        "aggression_bias",
        "Aggression Bias",
        "action_policy",
        default_value=0.0,
        min_value=-2.0,
        max_value=2.0,
        description="Existing behavior-bias gene.",
    ),
    _trait(
        "reproduction_threshold",
        "Reproduction Threshold",
        "drive_regulation",
        default_value=30.0,
        min_value=2.0,
        max_value=90.0,
        units="energy",
        interpretation="Life-history regulation trait, not a cognition reward.",
        description="Existing evolvable reproduction regulation gene.",
    ),
    _trait(
        "drive_logit_gain",
        "Drive Logit Gain",
        "drive_regulation",
        default_value=1.0,
        min_value=0.0,
        max_value=3.0,
        developmental=True,
        description="Existing drive-to-policy gain gene.",
    ),
    _trait(
        "drive_inertia",
        "Drive Inertia",
        "drive_regulation",
        default_value=0.85,
        min_value=0.0,
        max_value=0.999,
        developmental=True,
        description="Existing drive smoothing gene.",
    ),
    _trait(
        "drive_noise",
        "Drive Noise",
        "drive_regulation",
        default_value=0.02,
        min_value=0.0,
        max_value=0.10,
        developmental=True,
        description="Existing internal-drive volatility gene.",
    ),
    _trait(
        "drive_sp_hunger",
        "Hunger Sensitivity Prior",
        "drive_regulation",
        default_value=0.55,
        min_value=0.0,
        max_value=1.0,
        description="Existing inherited hunger drive setpoint prior.",
        mutable=False,
        directly_mutable=False,
    ),
    _trait(
        "drive_sp_pain",
        "Pain Sensitivity Prior",
        "drive_regulation",
        default_value=0.05,
        min_value=0.0,
        max_value=1.0,
        description="Existing inherited pain drive setpoint prior.",
        mutable=False,
        directly_mutable=False,
    ),
    _trait(
        "drive_sp_fatigue",
        "Fatigue Sensitivity Prior",
        "drive_regulation",
        default_value=0.15,
        min_value=0.0,
        max_value=1.0,
        description="Existing inherited fatigue drive setpoint prior.",
        mutable=False,
        directly_mutable=False,
    ),
    _trait(
        "drive_sp_fear",
        "Fear Sensitivity Prior",
        "drive_regulation",
        default_value=0.10,
        min_value=0.0,
        max_value=1.0,
        description="Existing inherited fear drive setpoint prior.",
        mutable=False,
        directly_mutable=False,
    ),
    _trait(
        "prediction_learning_rate",
        "Prediction Learning Rate",
        "prediction",
        default_value=0.08,
        min_value=0.0,
        max_value=1.0,
        developmental=True,
        units="EMA alpha",
        description="Existing prediction expectation update rate gene.",
    ),
    _trait(
        "prediction_enabled",
        "Prediction Enabled",
        "prediction",
        default_value=True,
        genetic=False,
        phenotype=True,
        heritable=False,
        mutable=False,
        directly_mutable=False,
        recombinable=False,
        costed=True,
        ablatable=True,
        units="boolean",
        interpretation="Derived from effective prediction learning rate and ablation state.",
        description="Stage 3 runtime prediction availability flag.",
    ),
    _trait(
        "prediction_update_enabled",
        "Prediction Update Enabled",
        "prediction",
        default_value=True,
        genetic=False,
        phenotype=True,
        heritable=False,
        mutable=False,
        directly_mutable=False,
        recombinable=False,
        costed=True,
        ablatable=True,
        units="boolean",
        interpretation="Whether per-action expectation state is allowed to update.",
        description="Stage 3 update-control surface for prediction ablation tests.",
    ),
    _trait(
        "prediction_error_tracking",
        "Prediction Error Tracking",
        "prediction",
        default_value=True,
        genetic=False,
        phenotype=True,
        heritable=False,
        mutable=False,
        directly_mutable=False,
        recombinable=False,
        costed=False,
        ablatable=True,
        metric=True,
        checkpoint_required=False,
        units="boolean",
        interpretation="Diagnostic prediction-error accumulation; not intelligence.",
        description="Existing per-generation prediction-error telemetry.",
    ),
    _trait(
        "prediction_error_mean",
        "Prediction Error Mean",
        "prediction",
        default_value=0.0,
        genetic=False,
        phenotype=True,
        heritable=False,
        mutable=False,
        directly_mutable=False,
        recombinable=False,
        costed=False,
        ablatable=False,
        metric=True,
        checkpoint_required=False,
        units="error",
        interpretation="Diagnostic expected-vs-actual outcome error; lower is not automatically intelligence.",
        description="Stage 3 explicit prediction diagnostic surface.",
    ),
    _trait(
        "prediction_capacity",
        "Prediction Capacity",
        "prediction",
        default_value=0.0,
        min_value=0.0,
        max_value=1.0,
        genetic=False,
        phenotype=True,
        heritable=False,
        mutable=False,
        directly_mutable=False,
        recombinable=False,
        costed=True,
        ablatable=True,
        units="proxy",
        interpretation="Approximate proxy based on effective learning rate; no variable prediction model exists yet.",
        status="partially_active",
        description="Stage 3 honest proxy until real prediction capacity is introduced.",
    ),
    _trait(
        "prediction_cost",
        "Prediction Cost",
        "cognition_budget",
        default_value=0.0,
        min_value=0.0,
        genetic=False,
        phenotype=True,
        heritable=False,
        mutable=False,
        directly_mutable=False,
        recombinable=False,
        costed=True,
        ablatable=True,
        units="energy per tick",
        interpretation="Telemetry-only unless explicit experimental prediction cost charging is enabled.",
        description="Stage 3 budget surface for prediction updates and state.",
    ),
    _trait(
        "prediction_ablation_state",
        "Prediction Ablation State",
        "prediction",
        default_value="normal",
        genetic=False,
        phenotype=True,
        heritable=False,
        mutable=False,
        directly_mutable=False,
        recombinable=False,
        costed=False,
        ablatable=True,
        metric=True,
        checkpoint_required=False,
        units="state",
        interpretation="Runtime-only prediction ablation mode; genomes remain unchanged.",
        description="Stage 3 prediction ablation surface.",
    ),
    _trait(
        "prediction_state_available",
        "Prediction State Available",
        "prediction",
        default_value=True,
        genetic=False,
        phenotype=True,
        heritable=False,
        mutable=False,
        directly_mutable=False,
        recombinable=False,
        costed=False,
        ablatable=True,
        metric=True,
        checkpoint_required=False,
        units="boolean",
        interpretation="Whether action-outcome expectation arrays are behaviorally available.",
        description="Stage 3 state-availability surface over existing exp_dE/exp_dInj arrays.",
    ),
    _trait(
        "prediction_pressure_test_ready",
        "Prediction Pressure Test Ready",
        "prediction",
        default_value=False,
        genetic=False,
        phenotype=True,
        heritable=False,
        mutable=False,
        directly_mutable=False,
        recombinable=False,
        costed=False,
        ablatable=False,
        metric=True,
        checkpoint_required=False,
        units="boolean",
        interpretation="Scenario metadata flag for controlled prediction-pressure scaffolds.",
        status="partially_active",
        description="Stage 3 fixture hook. It does not reward prediction directly.",
    ),
    _trait(
        "prediction_prior_strength",
        "Prediction Prior Strength",
        "prediction",
        default_value=0.35,
        min_value=0.0,
        max_value=1.0,
        description="Existing inherited expectation prior strength.",
        mutable=False,
        directly_mutable=False,
    ),
    _trait(
        "prediction_prior_noise",
        "Prediction Prior Noise",
        "prediction",
        default_value=0.05,
        min_value=0.0,
        max_value=0.5,
        description="Existing inherited expectation prior noise.",
        mutable=False,
        directly_mutable=False,
    ),
    _trait(
        "gru_memory_upkeep",
        "GRU Memory Upkeep",
        "cognition_budget",
        default_value=0.0,
        min_value=0.0,
        genetic=False,
        phenotype=True,
        heritable=False,
        mutable=False,
        recombinable=False,
        costed=True,
        checkpoint_required=False,
        units="energy per tick",
        interpretation="Existing environment-configured memory upkeep cost.",
        description="Budget ledger component derived from env_cfg.",
    ),
    _trait(
        "late_prediction_selection_cost",
        "Late Prediction Selection Cost",
        "learning",
        default_value=0.0,
        min_value=0.0,
        genetic=False,
        phenotype=True,
        heritable=False,
        mutable=False,
        recombinable=False,
        costed=True,
        checkpoint_required=False,
        units="selection score",
        interpretation="Existing selection-time internal diagnostic penalty; fake-evolution risk surface.",
        status="partially_active",
        description="Registered for visibility only; Stage 0 does not add or alter this pressure.",
    ),
    _trait(
        "developmental_program",
        "Developmental Program",
        "development",
        default_value="developmental_program_v1",
        developmental=True,
        mutable=False,
        directly_mutable=False,
        status="partially_active",
        interpretation="Existing age/stage expression scaffold; active for module gates, not new cognition.",
        description="Future surface for age, energy, stress, and stage-based cognition expression.",
    ),
    _trait(
        "developmental_cognition_enabled",
        "Developmental Cognition Enabled",
        "development",
        default_value=False,
        genetic=False,
        phenotype=True,
        heritable=False,
        mutable=False,
        directly_mutable=False,
        recombinable=False,
        costed=True,
        ablatable=True,
        units="boolean",
        interpretation="Feature-gated runtime expression flag; default legacy full expression preserves behavior.",
        status="active",
        description="Stage 4 enables deterministic age-gated cognitive expression without changing runtime tensor shapes.",
    ),
    _trait(
        "developmental_stage",
        "Developmental Stage",
        "development",
        default_value="default",
        genetic=False,
        phenotype=True,
        heritable=False,
        mutable=False,
        directly_mutable=False,
        recombinable=False,
        costed=False,
        ablatable=True,
        units="stage id",
        interpretation="Existing developmental-program stage label, derived from age and program windows.",
        status="active",
        description="Stage 4 uses this for juvenile/adult cognitive telemetry; it is not a reward.",
    ),
    _trait(
        "maturity_age",
        "Maturity Age",
        "development",
        default_value=20,
        min_value=0,
        genetic=False,
        phenotype=True,
        heritable=False,
        mutable=False,
        directly_mutable=False,
        recombinable=False,
        costed=True,
        ablatable=True,
        units="ticks",
        interpretation="Config-controlled maturity threshold in Stage 4; not yet an evolvable gene.",
        status="fixed",
        description="Future surface for evolvable maturity timing.",
    ),
    _trait(
        "juvenile_cognition_scale",
        "Juvenile Cognition Scale",
        "development",
        default_value=0.5,
        min_value=0.0,
        max_value=1.0,
        genetic=False,
        phenotype=True,
        heritable=False,
        mutable=False,
        directly_mutable=False,
        recombinable=False,
        costed=True,
        ablatable=True,
        units="scale",
        interpretation="Config-controlled multiplier for juvenile effective cognition when Stage 4 is enabled.",
        status="fixed",
        description="Telemetry and ablation surface for juvenile cognitive expression.",
    ),
    _trait(
        "adult_cognition_scale",
        "Adult Cognition Scale",
        "development",
        default_value=1.0,
        min_value=0.0,
        max_value=1.0,
        genetic=False,
        phenotype=True,
        heritable=False,
        mutable=False,
        directly_mutable=False,
        recombinable=False,
        costed=True,
        ablatable=True,
        units="scale",
        interpretation="Config-controlled adult multiplier; default is full legacy cognition.",
        status="fixed",
        description="Stage 4 adult expression surface.",
    ),
    _trait(
        "cognitive_growth_rate",
        "Cognitive Growth Rate",
        "development",
        default_value=1.0,
        min_value=0.0,
        max_value=1.0,
        genetic=False,
        phenotype=True,
        heritable=False,
        mutable=False,
        directly_mutable=False,
        recombinable=False,
        costed=True,
        ablatable=True,
        units="fraction per maturity window",
        interpretation="Derived from age divided by maturity age in linear mode; not yet genetically evolvable.",
        status="fixed",
        description="Stage 4 deterministic growth-progress surface.",
    ),
    _trait(
        "cognitive_growth_curve",
        "Cognitive Growth Curve",
        "development",
        default_value="legacy_full_expression",
        genetic=False,
        phenotype=True,
        heritable=False,
        mutable=False,
        directly_mutable=False,
        recombinable=False,
        costed=True,
        ablatable=True,
        units="mode",
        interpretation="Config mode: legacy full, step juvenile/adult, or linear growth to maturity.",
        status="fixed",
        description="Stage 4 keeps the growth curve config-controlled and deterministic.",
    ),
    _trait(
        "age_gated_memory_expression",
        "Age-Gated Memory Expression",
        "development",
        default_value=False,
        genetic=False,
        phenotype=True,
        heritable=False,
        mutable=False,
        directly_mutable=False,
        recombinable=False,
        costed=True,
        ablatable=True,
        units="boolean",
        interpretation="Runtime expression can reduce effective memory dimension under Stage 4.",
        status="active",
        description="Uses fixed GRU shape and effective-memory masking only.",
    ),
    _trait(
        "age_gated_perception_expression",
        "Age-Gated Perception Expression",
        "development",
        default_value=False,
        genetic=False,
        phenotype=True,
        heritable=False,
        mutable=False,
        directly_mutable=False,
        recombinable=False,
        costed=True,
        ablatable=True,
        units="boolean",
        interpretation="Runtime expression can reduce effective sensor radius and channel count under Stage 4.",
        status="active",
        description="Observation shape remains fixed; later runtime masking can consume the expressed phenotype.",
    ),
    _trait(
        "age_gated_prediction_expression",
        "Age-Gated Prediction Expression",
        "development",
        default_value=False,
        genetic=False,
        phenotype=True,
        heritable=False,
        mutable=False,
        directly_mutable=False,
        recombinable=False,
        costed=True,
        ablatable=True,
        units="boolean",
        interpretation="Runtime expression can scale effective prediction learning rate under Stage 4.",
        status="active",
        description="Prediction state shape remains fixed.",
    ),
    _trait(
        "developmental_cognition_cost",
        "Developmental Cognition Cost",
        "cognition_budget",
        default_value=0.0,
        min_value=0.0,
        genetic=False,
        phenotype=True,
        heritable=False,
        mutable=False,
        directly_mutable=False,
        recombinable=False,
        costed=True,
        ablatable=True,
        units="energy per tick",
        interpretation="Telemetry-only unless explicit experimental developmental cost charging is enabled.",
        status="active",
        description="Stage 4 additional growth-expression cost surface.",
    ),
    _trait(
        "maturation_cost",
        "Maturation Cost",
        "cognition_budget",
        default_value=0.0,
        min_value=0.0,
        genetic=False,
        phenotype=True,
        heritable=False,
        mutable=False,
        directly_mutable=False,
        recombinable=False,
        costed=True,
        ablatable=True,
        units="energy per tick",
        interpretation="Config-controlled maturation cost; not charged by default.",
        status="active",
        description="Stage 4 maturation-cost telemetry and charging gate.",
    ),
    _trait(
        "developmental_ablation_state",
        "Developmental Ablation State",
        "development",
        default_value="normal",
        genetic=False,
        phenotype=True,
        heritable=False,
        mutable=False,
        directly_mutable=False,
        recombinable=False,
        costed=False,
        ablatable=True,
        metric=True,
        checkpoint_required=False,
        units="state",
        interpretation="Runtime-only developmental cognition override; genomes remain unchanged.",
        status="active",
        description="Stage 4 paired-test ablation surface.",
    ),
    _trait(
        "developmental_expression_active",
        "Developmental Expression Active",
        "development",
        default_value=False,
        genetic=False,
        phenotype=True,
        heritable=False,
        mutable=False,
        directly_mutable=False,
        recombinable=False,
        costed=False,
        ablatable=True,
        units="boolean",
        interpretation="Whether cognitive expression differs from legacy full expression.",
        status="active",
        description="Stage 4 developmental cognition status metric surface.",
    ),
    _trait(
        "developmental_pressure_test_ready",
        "Developmental Pressure Test Ready",
        "development",
        default_value=False,
        genetic=False,
        phenotype=True,
        heritable=False,
        mutable=False,
        directly_mutable=False,
        recombinable=False,
        costed=False,
        ablatable=False,
        metric=True,
        checkpoint_required=False,
        units="boolean",
        interpretation="Scenario metadata flag for controlled developmental-pressure scaffolds.",
        status="partially_active",
        description="Stage 4 fixture hook. It does not reward development directly.",
    ),
    _trait(
        "module_count",
        "Module Count",
        "modularity",
        default_value=0,
        min_value=0,
        mutable=False,
        directly_mutable=False,
        structurally_mutable=True,
        costed=False,
        status="partially_active",
        units="modules",
        description="Existing modular genome count; structural operators may populate it.",
    ),
    _trait(
        "connection_count",
        "Connection Count",
        "modularity",
        default_value=0,
        min_value=0,
        mutable=False,
        directly_mutable=False,
        structurally_mutable=True,
        costed=False,
        status="partially_active",
        units="connections",
        description="Existing modular genome connection count; structural operators may populate it.",
    ),
    _trait(
        "sensor_observation_capacity",
        "Sensor Observation Capacity",
        "perception",
        default_value=0,
        min_value=0,
        genetic=False,
        heritable=False,
        mutable=False,
        recombinable=False,
        costed=False,
        status="fixed",
        units="observation slots",
        interpretation="Derived from fixed observation shape; not evolvable in Stage 0.",
        description="Registered as the perception surface for later evolvable sensors.",
    ),
    _trait(
        "sensor_channel_mask",
        "Sensor Channel Mask",
        "perception",
        default_value=implemented_sensor_channel_mask(),
        min_value=0,
        genetic=True,
        phenotype=True,
        developmental=True,
        heritable=True,
        mutable=True,
        directly_mutable=True,
        recombinable=True,
        costed=True,
        ablatable=True,
        units="bitmask",
        interpretation="Genetic fixed-order bitmask over implemented perception channels; active behavior only when sensor masking is enabled.",
        status="active",
        description="Stage 1A evolvable perception-channel budget gene.",
    ),
    _trait(
        "active_sensor_channel_count",
        "Active Sensor Channel Count",
        "perception",
        default_value=0,
        min_value=0,
        genetic=False,
        phenotype=True,
        heritable=False,
        mutable=False,
        directly_mutable=False,
        recombinable=False,
        costed=True,
        ablatable=True,
        units="channels",
        interpretation="Derived phenotype count of enabled channels currently available in the fixed observation layout.",
        status="active",
        description="Stage 1A phenotype summary derived from the sensor channel mask and config.",
    ),
    _trait(
        "perception_cost",
        "Perception Cost",
        "cognition_budget",
        default_value=0.0,
        min_value=0.0,
        genetic=False,
        phenotype=True,
        heritable=False,
        mutable=False,
        directly_mutable=False,
        recombinable=False,
        costed=True,
        ablatable=True,
        units="energy per tick",
        interpretation="Budget-ledger cost; telemetry-only unless explicit experimental charging flags are enabled.",
        status="active",
        description="Stage 1A/1B perception budget cost surface.",
    ),
    _trait(
        "perception_capacity_component",
        "Perception Capacity Component",
        "perception",
        default_value=0.0,
        min_value=0.0,
        genetic=False,
        phenotype=True,
        heritable=False,
        mutable=False,
        directly_mutable=False,
        recombinable=False,
        costed=True,
        ablatable=True,
        units="cell-channel slots",
        interpretation="Raw component index for active cell-channel capacity; not an intelligence score.",
        status="active",
        description="Stage 1A/1B raw perception component used for telemetry and future selection diagnostics.",
    ),
    _trait(
        "sensor_radius",
        "Sensor Radius",
        "perception",
        default_value=2,
        allowed_ladder=SENSOR_RADIUS_LADDER,
        genetic=True,
        phenotype=True,
        developmental=True,
        heritable=True,
        mutable=True,
        directly_mutable=True,
        recombinable=True,
        costed=True,
        ablatable=True,
        units="Chebyshev radius",
        interpretation="Genetic bounded radius ladder applied as a mask within the fixed maximum observation window.",
        status="active",
        description="Stage 1B evolvable spatial perception radius gene.",
    ),
    _trait(
        "sensor_radius_cost",
        "Sensor Radius Cost",
        "cognition_budget",
        default_value=0.0,
        min_value=0.0,
        genetic=False,
        phenotype=True,
        heritable=False,
        mutable=False,
        directly_mutable=False,
        recombinable=False,
        costed=True,
        ablatable=True,
        units="energy per tick",
        interpretation="Telemetry-only spatial perception cost unless explicit experimental radius charging is enabled.",
        status="active",
        description="Stage 1B spatial perception cost surface.",
    ),
    _trait(
        "spatial_perception_capacity",
        "Spatial Perception Capacity",
        "perception",
        default_value=0.0,
        min_value=0.0,
        genetic=False,
        phenotype=True,
        heritable=False,
        mutable=False,
        directly_mutable=False,
        recombinable=False,
        costed=True,
        ablatable=True,
        units="cell-channel slots",
        interpretation="Observed cell count times active channel count; raw telemetry only.",
        status="active",
        description="Stage 1B spatial perception capacity summary.",
    ),
    _trait(
        "perception_cell_count",
        "Perception Cell Count",
        "perception",
        default_value=0,
        min_value=0,
        genetic=False,
        phenotype=True,
        heritable=False,
        mutable=False,
        directly_mutable=False,
        recombinable=False,
        costed=True,
        ablatable=True,
        units="cells",
        interpretation="Number of local cells visible within active sensor radius.",
        status="active",
        description="Stage 1B fixed-window radius-mask cell count.",
    ),
    _trait(
        "cell_channel_perception_capacity",
        "Cell-Channel Perception Capacity",
        "perception",
        default_value=0.0,
        min_value=0.0,
        genetic=False,
        phenotype=True,
        heritable=False,
        mutable=False,
        directly_mutable=False,
        recombinable=False,
        costed=True,
        ablatable=True,
        units="cell-channel slots",
        interpretation="Observed cells multiplied by active implemented channels.",
        status="active",
        description="Stage 1B combined channel/radius capacity surface.",
    ),
    _trait(
        "introspection_channel_mask",
        "Introspection Channel Mask",
        "introspection_future",
        default_value=implemented_introspection_channel_mask(),
        min_value=0,
        genetic=True,
        phenotype=True,
        heritable=True,
        mutable=True,
        directly_mutable=True,
        recombinable=True,
        costed=True,
        ablatable=True,
        units="bitmask",
        interpretation="Optional fixed-order internal-state access mask; active only when introspection channels are enabled.",
        status="active",
        description="Stage 5 bounded internal-state observation access surface.",
    ),
    _trait(
        "active_introspection_channel_count",
        "Active Introspection Channel Count",
        "introspection_future",
        default_value=0,
        min_value=0,
        genetic=False,
        phenotype=True,
        heritable=False,
        mutable=False,
        directly_mutable=False,
        recombinable=False,
        costed=True,
        ablatable=True,
        units="channels",
        interpretation="Phenotype count of enabled safe internal-state channels.",
        status="active",
        description="Stage 5 internal-state access count; not self-control.",
    ),
    _trait(
        "introspection_cost",
        "Introspection Cost",
        "cognition_budget",
        default_value=0.0,
        min_value=0.0,
        genetic=False,
        phenotype=True,
        heritable=False,
        mutable=False,
        directly_mutable=False,
        recombinable=False,
        costed=True,
        ablatable=True,
        units="energy per tick",
        interpretation="Telemetry-only introspection cost unless explicit experimental charging is enabled.",
        status="active",
        description="Stage 5 budget surface for internal-state observation access.",
    ),
    _trait(
        "introspection_observation_surface",
        "Introspection Observation Surface",
        "introspection_future",
        default_value=False,
        genetic=False,
        phenotype=True,
        introspection=True,
        control=False,
        heritable=False,
        mutable=False,
        directly_mutable=False,
        recombinable=False,
        costed=True,
        ablatable=True,
        checkpoint_required=False,
        status="active",
        interpretation="Fixed-shape internal-state extraction surface; not self-regulation.",
        description="Stage 5 exposes selected simulated internal state as bounded normalized values.",
    ),
    _trait(
        "future_introspection_channel",
        "Future Introspection Channel",
        "introspection_future",
        default_value=False,
        genetic=False,
        phenotype=False,
        introspection=True,
        heritable=False,
        mutable=False,
        recombinable=False,
        metric=True,
        checkpoint_required=False,
        status="future_reserved",
        interpretation="No agent introspection exists in Stage 0.",
        description="Reserved metadata surface for bounded future introspection.",
    ),
    _trait(
        "self_regulation_control_mask",
        "Self-Regulation Control Mask",
        "control_future",
        default_value=implemented_self_regulation_control_mask(),
        min_value=0,
        genetic=True,
        phenotype=True,
        control=True,
        heritable=True,
        mutable=True,
        directly_mutable=True,
        recombinable=True,
        costed=True,
        ablatable=True,
        units="bitmask",
        interpretation="Optional fixed-order access mask for bounded runtime regulatory controls.",
        status="active",
        description="Stage 6 self-regulation access mask; inert unless self-regulation is explicitly enabled.",
    ),
    _trait(
        "active_self_regulation_control_count",
        "Active Self-Regulation Control Count",
        "control_future",
        default_value=0,
        min_value=0,
        genetic=False,
        phenotype=True,
        control=True,
        heritable=False,
        mutable=False,
        directly_mutable=False,
        recombinable=False,
        costed=True,
        ablatable=True,
        units="controls",
        interpretation="Phenotype count of enabled safe bounded runtime control channels.",
        status="active",
        description="Stage 6 control surface capacity; not self-modification.",
    ),
    _trait(
        "self_regulation_cost",
        "Self-Regulation Cost",
        "cognition_budget",
        default_value=0.0,
        min_value=0.0,
        genetic=False,
        phenotype=True,
        control=True,
        heritable=False,
        mutable=False,
        directly_mutable=False,
        recombinable=False,
        costed=True,
        ablatable=True,
        units="energy per tick",
        interpretation="Telemetry-only self-regulation cost unless explicit experimental charging is enabled.",
        status="active",
        description="Stage 6 budget surface for bounded runtime regulation.",
    ),
    _trait(
        "self_regulation_runtime_state",
        "Self-Regulation Runtime State",
        "control_future",
        default_value=False,
        genetic=False,
        phenotype=True,
        control=True,
        heritable=False,
        mutable=False,
        directly_mutable=False,
        recombinable=False,
        costed=True,
        ablatable=True,
        checkpoint_required=False,
        status="active",
        interpretation="Runtime offsets and control telemetry; not inherited and not genome editing.",
        description="Stage 6 bounded state surface for scripted or future policy control vectors.",
    ),
    _trait(
        "self_modification_target_mask",
        "Self-Modification Target Mask",
        "control_future",
        default_value=implemented_self_modification_target_mask(),
        min_value=0,
        genetic=True,
        phenotype=True,
        control=True,
        heritable=True,
        mutable=True,
        directly_mutable=True,
        recombinable=True,
        costed=True,
        ablatable=True,
        units="bitmask",
        interpretation="Optional fixed-order access mask for bounded lifetime phenotype-offset modification targets.",
        status="active",
        description="Stage 7 access mask; inert unless self-modification is explicitly enabled and never grants genome editing.",
    ),
    _trait(
        "active_self_modification_target_count",
        "Active Self-Modification Target Count",
        "control_future",
        default_value=0,
        min_value=0,
        genetic=False,
        phenotype=True,
        control=True,
        heritable=False,
        mutable=False,
        directly_mutable=False,
        recombinable=False,
        costed=True,
        ablatable=True,
        units="targets",
        interpretation="Phenotype count of enabled safe bounded lifetime modification targets.",
        status="active",
        description="Stage 7 command surface capacity; not self-regulation and not genome editing.",
    ),
    _trait(
        "self_modification_cost",
        "Self-Modification Cost",
        "cognition_budget",
        default_value=0.0,
        min_value=0.0,
        genetic=False,
        phenotype=True,
        control=True,
        heritable=False,
        mutable=False,
        directly_mutable=False,
        recombinable=False,
        costed=True,
        ablatable=True,
        units="energy per tick",
        interpretation="Telemetry-only self-modification cost unless explicit experimental charging is enabled.",
        status="active",
        description="Stage 7 budget surface for bounded lifetime-scoped phenotype offsets.",
    ),
    _trait(
        "self_modification_runtime_state",
        "Self-Modification Runtime State",
        "control_future",
        default_value=False,
        genetic=False,
        phenotype=True,
        control=True,
        heritable=False,
        mutable=False,
        directly_mutable=False,
        recombinable=False,
        costed=True,
        ablatable=True,
        checkpoint_required=True,
        status="active",
        interpretation="Lifecycle-scoped offsets and command telemetry; not inherited and not genome editing.",
        description="Stage 7 bounded runtime phenotype-offset state.",
    ),
    _trait(
        "directed_inheritance_channel_mask",
        "Directed-Inheritance Channel Mask",
        "directed_inheritance_future",
        default_value=implemented_directed_inheritance_channel_mask(),
        min_value=0,
        genetic=True,
        phenotype=True,
        control=True,
        heritable=True,
        mutable=True,
        directly_mutable=True,
        recombinable=True,
        costed=True,
        ablatable=True,
        units="bitmask",
        interpretation="Optional fixed-order access mask for bounded non-genomic parent-to-offspring influence channels.",
        status="active",
        description="Stage 8 access mask; inert unless directed inheritance is explicitly enabled and never grants genome editing.",
    ),
    _trait(
        "active_directed_inheritance_channel_count",
        "Active Directed-Inheritance Channel Count",
        "directed_inheritance_future",
        default_value=0,
        min_value=0,
        genetic=False,
        phenotype=True,
        control=True,
        heritable=False,
        mutable=False,
        directly_mutable=False,
        recombinable=False,
        costed=True,
        ablatable=True,
        units="channels",
        interpretation="Phenotype count of enabled safe bounded non-genomic inheritance channels.",
        status="active",
        description="Stage 8 command surface capacity; not genome editing, mutation-rate control, or recombination control.",
    ),
    _trait(
        "directed_inheritance_cost",
        "Directed-Inheritance Cost",
        "cognition_budget",
        default_value=0.0,
        min_value=0.0,
        genetic=False,
        phenotype=True,
        control=True,
        heritable=False,
        mutable=False,
        directly_mutable=False,
        recombinable=False,
        costed=True,
        ablatable=True,
        units="energy per reproduction",
        interpretation="Telemetry-only directed-inheritance cost unless explicit experimental charging is enabled.",
        status="active",
        description="Stage 8 budget surface for bounded non-genomic inheritance influence.",
    ),
    _trait(
        "non_genomic_inheritance_payload",
        "Non-Genomic Inheritance Payload",
        "directed_inheritance_future",
        default_value=False,
        genetic=False,
        phenotype=True,
        control=True,
        heritable=False,
        mutable=False,
        directly_mutable=False,
        recombinable=False,
        costed=True,
        ablatable=True,
        checkpoint_required=True,
        status="active",
        interpretation="Bounded offspring phenotype-bias payload; separate from genome and not inherited as self-modification state.",
        description="Stage 8 payload state for scripted or future policy influence vectors.",
    ),
    _trait(
        "future_self_regulation_channel",
        "Future Self Regulation Channel",
        "control_future",
        default_value=False,
        genetic=False,
        phenotype=False,
        control=True,
        heritable=False,
        mutable=False,
        recombinable=False,
        costed=True,
        metric=True,
        checkpoint_required=False,
        status="future_reserved",
        interpretation="No self-regulation control exists in Stage 0.",
        description="Reserved metadata surface for costly future self-regulation.",
    ),
)

_TRAITS_BY_ID = {t.trait_id: t for t in _COGNITIVE_TRAITS}


def cognitive_trait_registry() -> tuple[CognitiveTraitMetadata, ...]:
    return _COGNITIVE_TRAITS


def cognitive_trait_ids() -> tuple[str, ...]:
    return tuple(t.trait_id for t in _COGNITIVE_TRAITS)


def cognitive_trait_registry_dict() -> dict[str, object]:
    return {
        "schema_version": REGISTRY_SCHEMA_VERSION,
        "traits": [t.to_dict() for t in _COGNITIVE_TRAITS],
    }


def get_cognitive_trait(trait_id: str) -> CognitiveTraitMetadata:
    return _TRAITS_BY_ID[str(trait_id)]


@dataclass(frozen=True)
class CognitivePhenotype:
    schema_version: str
    age: int
    memory_enabled: bool
    policy_memory_enabled: bool
    active_memory_capacity: int
    effective_memory_capacity: int
    genetic_memory_dim: int
    active_memory_dim: int
    effective_memory_dim: int
    max_memory_dim: int
    memory_ladder_index: int
    memory_capacity_fraction: float
    recurrent_state_enabled: bool
    memory_masking_enabled: bool
    memory_cost_hypothetical: float
    memory_cost_charged: float
    memory_ablation_active: bool
    force_memory_disabled_active: bool
    force_memory_min_active: bool
    force_memory_max_active: bool
    force_memory_default_active: bool
    reset_memory_each_tick_active: bool
    memory_use_measurement_available: bool
    maturity_age: int
    is_juvenile: bool
    is_mature: bool
    developmental_cognition_enabled: bool
    developmental_expression_factor: float
    juvenile_cognition_scale: float
    adult_cognition_scale: float
    cognitive_growth_progress: float
    active_memory_dim_before_development: int
    active_memory_dim_after_development: int
    active_sensor_radius_before_development: int
    active_sensor_radius_after_development: int
    active_sensor_channel_count_before_development: int
    active_sensor_channel_count_after_development: int
    effective_prediction_lr_before_development: float
    effective_prediction_lr_after_development: float
    developmental_cost_hypothetical: float
    developmental_cost_charged: float
    maturation_cost_hypothetical: float
    maturation_cost_charged: float
    developmental_ablation_active: bool
    force_adult_cognition_active: bool
    force_juvenile_cognition_active: bool
    disable_developmental_cognition_active: bool
    force_legacy_full_cognition_active: bool
    developmental_growth_disabled_active: bool
    developmental_cost_charging_enabled: bool
    policy_hidden_size: int
    active_prediction_capability: bool
    prediction_enabled: bool
    prediction_update_enabled: bool
    prediction_learning_rate: float
    effective_prediction_learning_rate: float
    prediction_capacity: float
    prediction_state_available: bool
    prediction_error_tracking_enabled: bool
    prediction_cost_hypothetical: float
    prediction_cost_charged: float
    prediction_ablation_active: bool
    force_prediction_disabled_active: bool
    force_prediction_frozen_active: bool
    force_prediction_zero_lr_active: bool
    force_prediction_default_active: bool
    prediction_reset_each_tick_active: bool
    prediction_use_measurement_available: bool
    direct_prediction_selection_pressure_present: bool
    drive_logit_gain: float
    drive_inertia: float
    drive_noise: float
    hunger_sensitivity: float
    pain_sensitivity: float
    fatigue_sensitivity: float
    fear_sensitivity: float
    action_temperature: float
    exploration_bias: float
    aggression_bias: float
    reproduction_threshold: float
    developmental_stage: str
    developmental_expression_active: bool
    module_count: int
    active_module_count: int
    connection_count: int
    sensor_observation_capacity: int
    active_sensor_channel_count: int
    potential_sensor_channel_count: int
    active_sensor_channel_mask: int
    active_sensor_channel_ids: tuple[str, ...]
    perception_enabled: bool
    perception_masking_enabled: bool
    perception_cost_enabled: bool
    sensor_channel_cost_total: float
    sensor_channel_cost_per_tick: float
    sensor_channel_mask_ablation_active: bool
    sensor_channel_force_all_active: bool
    sensor_channel_force_minimal_active: bool
    active_sensor_radius: int
    max_sensor_radius: int
    default_sensor_radius: int
    sensor_radius_masking_enabled: bool
    active_observed_cell_count: int
    max_observed_cell_count: int
    active_cell_fraction: float
    cell_channel_capacity: float
    spatial_perception_capacity: float
    perception_capacity_component: float
    sensor_radius_cost_hypothetical: float
    sensor_radius_cost_charged: float
    perception_cost_hypothetical: float
    perception_cost_charged: float
    sensor_radius_ablation_active: bool
    sensor_radius_force_min_active: bool
    sensor_radius_force_max_active: bool
    sensor_radius_force_default_active: bool
    introspection_enabled: bool
    introspection_channels_available: bool
    active_introspection_channel_count: int
    potential_introspection_channel_count: int
    active_introspection_channel_mask: int
    active_introspection_channel_ids: tuple[str, ...]
    introspection_capacity_fraction: float
    introspection_cost_hypothetical: float
    introspection_cost_charged: float
    introspection_ablation_active: bool
    force_all_introspection_channels_active: bool
    force_minimal_introspection_channels_active: bool
    force_no_introspection_channels_active: bool
    force_default_introspection_channels_active: bool
    mask_introspection_values_to_neutral_active: bool
    introspection_observation_shape: int
    introspection_policy_input_enabled: bool
    introspection_future_control_surface_ready: bool
    self_regulation_enabled: bool
    self_regulation_surface_enabled: bool
    self_regulation_policy_output_enabled: bool
    active_self_regulation_control_count: int
    potential_self_regulation_control_count: int
    active_self_regulation_control_mask: int
    active_self_regulation_control_ids: tuple[str, ...]
    self_regulation_capacity_fraction: float
    self_regulation_cost_hypothetical: float
    self_regulation_cost_charged: float
    self_regulation_ablation_active: bool
    force_no_self_regulation_active: bool
    force_all_self_regulation_active: bool
    force_default_self_regulation_active: bool
    mask_self_regulation_outputs_to_neutral_active: bool
    freeze_self_regulation_state_active: bool
    reset_self_regulation_each_tick_active: bool
    disable_self_regulation_decay_active: bool
    self_regulation_runtime_state_available: bool
    self_regulation_inheritable: bool
    self_modification_active: bool
    self_modification_enabled: bool
    self_modification_policy_output_enabled: bool
    active_self_modification_target_count: int
    potential_self_modification_target_count: int
    active_self_modification_target_mask: int
    active_self_modification_target_ids: tuple[str, ...]
    self_modification_capacity_fraction: float
    self_modification_cost_hypothetical: float
    self_modification_cost_charged: float
    self_modification_ablation_active: bool
    force_no_self_modification_active: bool
    force_all_self_modification_active: bool
    force_default_self_modification_active: bool
    mask_self_modification_commands_to_neutral_active: bool
    freeze_self_modification_state_active: bool
    reset_self_modification_each_tick_active: bool
    reset_self_modification_on_reproduction_active: bool
    disable_self_modification_decay_active: bool
    self_modification_runtime_state_available: bool
    self_modification_lifetime_persistent: bool
    self_modification_inheritable: bool
    directed_inheritance_enabled: bool
    directed_inheritance_surface_enabled: bool
    directed_inheritance_policy_output_enabled: bool
    active_directed_inheritance_channel_count: int
    potential_directed_inheritance_channel_count: int
    active_directed_inheritance_channel_mask: int
    active_directed_inheritance_channel_ids: tuple[str, ...]
    directed_inheritance_capacity_fraction: float
    directed_inheritance_cost_hypothetical: float
    directed_inheritance_cost_charged: float
    directed_inheritance_ablation_active: bool
    force_no_directed_inheritance_active: bool
    force_all_directed_inheritance_active: bool
    force_default_directed_inheritance_active: bool
    mask_directed_inheritance_commands_to_neutral_active: bool
    freeze_directed_inheritance_payload_active: bool
    reset_directed_inheritance_payload_on_birth_active: bool
    reset_directed_inheritance_payload_on_maturity_active: bool
    non_genomic_payload_available: bool
    non_genomic_payload_applied: bool
    genome_editing_active: bool
    mutation_rate_control_active: bool
    recombination_control_active: bool
    inherited_self_modification_state_active: bool
    offspring_developmental_expression_bias: float
    offspring_juvenile_cognition_scale_bias: float
    offspring_memory_expression_bias: float
    offspring_prediction_lr_bias: float
    offspring_exploration_bias_marker: float
    offspring_reproduction_threshold_bias: float
    cognitive_capacity_memory_component: float
    cognitive_capacity_prediction_component: float
    cognitive_capacity_drive_component: float
    cognitive_capacity_perception_component: float
    cognitive_capacity: float
    potential_cognitive_capacity: float

    def to_dict(self) -> dict[str, object]:
        out = asdict(self)
        out["active_sensor_channel_ids"] = list(self.active_sensor_channel_ids)
        out["active_introspection_channel_ids"] = list(self.active_introspection_channel_ids)
        out["active_self_regulation_control_ids"] = list(self.active_self_regulation_control_ids)
        out["active_self_modification_target_ids"] = list(self.active_self_modification_target_ids)
        out["active_directed_inheritance_channel_ids"] = list(self.active_directed_inheritance_channel_ids)
        return out


@dataclass(frozen=True)
class CognitiveBudgetLedger:
    schema_version: str
    cognitive_upkeep_total: float
    memory_upkeep_cost: float
    memory_upkeep_cost_hypothetical: float
    memory_upkeep_cost_charged: float
    memory_cost_per_tick: float
    memory_cost_total_component: float
    memory_cost_fraction_of_cog_cost: float
    memory_capacity_component: float
    effective_memory_capacity_component: float
    recurrent_policy_memory_component: float
    memory_cost_charging_enabled: bool
    memory_cost_telemetry_only: bool
    policy_memory_cost: float
    prediction_update_cost: float
    prediction_update_cost_hypothetical: float
    prediction_update_cost_charged: float
    prediction_state_cost_hypothetical: float
    prediction_state_cost_charged: float
    prediction_cost_per_tick: float
    prediction_cost_total_component: float
    prediction_cost_fraction_of_cog_cost: float
    prediction_capacity_component: float
    prediction_cost_charging_enabled: bool
    prediction_cost_telemetry_only: bool
    developmental_cognition_cost_hypothetical: float
    developmental_cognition_cost_charged: float
    maturation_cost_hypothetical: float
    maturation_cost_charged: float
    juvenile_cognitive_cost_meaning: str
    adult_cognitive_cost_meaning: str
    cognitive_growth_cost: float
    developmental_cost_fraction_of_cog_cost: float
    developmental_cost_charging_enabled: bool
    developmental_cost_telemetry_only: bool
    introspection_channel_cost_hypothetical: float
    introspection_channel_cost_charged: float
    introspection_observation_cost_hypothetical: float
    introspection_observation_cost_charged: float
    introspection_cost_per_tick: float
    introspection_cost_total_component: float
    introspection_cost_fraction_of_cog_cost: float
    introspection_capacity_component: float
    introspection_cost_charging_enabled: bool
    introspection_cost_telemetry_only: bool
    self_regulation_control_cost_hypothetical: float
    self_regulation_control_cost_charged: float
    self_regulation_update_cost_hypothetical: float
    self_regulation_update_cost_charged: float
    self_regulation_maintenance_cost_hypothetical: float
    self_regulation_maintenance_cost_charged: float
    self_regulation_cost_per_tick: float
    self_regulation_cost_total_component: float
    self_regulation_cost_fraction_of_cog_cost: float
    self_regulation_capacity_component: float
    self_regulation_cost_charging_enabled: bool
    self_regulation_cost_telemetry_only: bool
    self_modification_command_cost_hypothetical: float
    self_modification_command_cost_charged: float
    self_modification_update_cost_hypothetical: float
    self_modification_update_cost_charged: float
    self_modification_maintenance_cost_hypothetical: float
    self_modification_maintenance_cost_charged: float
    self_modification_cost_per_tick: float
    self_modification_cost_total_component: float
    self_modification_cost_fraction_of_cog_cost: float
    self_modification_capacity_component: float
    self_modification_cost_charging_enabled: bool
    self_modification_cost_telemetry_only: bool
    directed_inheritance_command_cost_hypothetical: float
    directed_inheritance_command_cost_charged: float
    directed_inheritance_payload_cost_hypothetical: float
    directed_inheritance_payload_cost_charged: float
    directed_inheritance_parent_cost_hypothetical: float
    directed_inheritance_parent_cost_charged: float
    directed_inheritance_offspring_burden_hypothetical: float
    directed_inheritance_offspring_burden_charged: float
    directed_inheritance_cost_per_reproduction: float
    directed_inheritance_cost_total_component: float
    directed_inheritance_cost_fraction_of_cog_cost: float
    directed_inheritance_capacity_component: float
    directed_inheritance_cost_charging_enabled: bool
    directed_inheritance_cost_telemetry_only: bool
    sensor_observation_cost: float
    drive_regulation_cost: float
    learning_or_plasticity_cost: float
    future_self_regulation_cost: float
    sensor_channel_cost: float
    perception_cost_total: float
    perception_cost_hypothetical: float
    perception_cost_charged: float
    sensor_radius_cost_hypothetical: float
    sensor_radius_cost_charged: float
    spatial_perception_cost: float
    active_sensor_channel_count: int
    potential_sensor_channel_count: int
    active_sensor_radius: int
    active_observed_cell_count: int
    cell_channel_capacity: float
    total_active_cognitive_capacity: float
    total_potential_cognitive_capacity: float
    cognitive_cost_per_tick: float
    cognitive_cost_fraction_of_energy_budget: float
    memory_status: str
    policy_memory_status: str
    prediction_status: str
    sensor_status: str
    drive_regulation_status: str
    learning_status: str
    introspection_status: str
    self_regulation_status: str
    self_modification_status: str
    future_self_regulation_status: str
    experimental_charge_enabled: bool
    charged_energy_delta: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class CostChargeSurface:
    """Minimal scalar surface for the env energy-charge path only."""

    schema_version: str
    sensor_channel_cost_per_tick: float
    sensor_radius_cost_hypothetical: float
    prediction_cost_charged: float
    developmental_cost_charged: float
    maturation_cost_charged: float
    introspection_cost_charged: float
    self_regulation_cost_charged: float
    self_modification_cost_charged: float
    directed_inheritance_visible_but_not_env_charged: bool = True


@dataclass(frozen=True)
class _CognitiveBudgetCostAggregation:
    total: float
    energy_fraction: float
    fractions: tuple[float, ...]


def _gget(g: Any, name: str, default: Any) -> Any:
    if g is None:
        return default
    if hasattr(g, name):
        try:
            return getattr(g, name)
        except Exception:
            return default
    if isinstance(g, Mapping):
        return g.get(name, default)
    return default


def _aget(a: Any, name: str, default: Any) -> Any:
    if a is None:
        return default
    if hasattr(a, name):
        try:
            return getattr(a, name)
        except Exception:
            return default
    if isinstance(a, Mapping):
        return a.get(name, default)
    return default


def _as_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return int(default)


def _as_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _as_bool(x: Any, default: bool = False) -> bool:
    if isinstance(x, bool):
        return bool(x)
    if isinstance(x, str):
        s = x.strip().lower()
        if s in {"1", "true", "t", "yes", "y", "on"}:
            return True
        if s in {"0", "false", "f", "no", "n", "off"}:
            return False
    try:
        return bool(x)
    except Exception:
        return bool(default)


def _snap_memory_dim(x: Any) -> int:
    d = _as_int(x, 0)
    return int(min(MEMORY_LADDER, key=lambda v: abs(int(v) - d)))


def memory_ablation_flags(sim_cfg: Any = None) -> dict[str, bool]:
    ablate = bool(_aget(sim_cfg, "ablate_memory", False))
    force_disabled = bool(_aget(sim_cfg, "force_memory_disabled", False))
    force_min = bool(_aget(sim_cfg, "force_min_memory_dim", False))
    force_max = bool(_aget(sim_cfg, "force_max_memory_dim", False))
    force_default = bool(_aget(sim_cfg, "force_default_memory_dim", False))
    reset_each_tick = bool(_aget(sim_cfg, "reset_memory_each_tick", False))
    return {
        "ablate": bool(ablate),
        "force_disabled": bool(force_disabled),
        "force_min": bool(force_min),
        "force_max": bool(force_max),
        "force_default": bool(force_default),
        "reset_each_tick": bool(reset_each_tick),
        "active": bool(ablate or force_disabled or force_min or force_max or force_default or reset_each_tick),
    }


def effective_memory_dim_for_genome(genome: Any, *, sim_cfg: Any = None) -> int:
    raw_dim = _snap_memory_dim(_gget(genome, "memory_dim", 0))
    recurrent_state = _as_bool(_gget(genome, "use_gru", False), False) or raw_dim > 0
    flags = memory_ablation_flags(sim_cfg)
    if flags["ablate"] or flags["force_disabled"]:
        return 0
    if flags["force_min"]:
        return int(min(MEMORY_LADDER))
    if flags["force_max"]:
        return int(max(MEMORY_LADDER)) if recurrent_state else 0
    if flags["force_default"]:
        return 0
    return int(raw_dim if recurrent_state else 0)


def memory_cost_components(genome: Any, *, env_cfg: Any = None, sim_cfg: Any = None) -> dict[str, float | bool | int]:
    raw_dim = _snap_memory_dim(_gget(genome, "memory_dim", 0))
    recurrent_state = _as_bool(_gget(genome, "use_gru", False), False) or raw_dim > 0
    eff_dim = effective_memory_dim_for_genome(genome, sim_cfg=sim_cfg)
    flags = memory_ablation_flags(sim_cfg)
    memory_unavailable = bool(flags["ablate"] or flags["force_disabled"])
    gru_base = _as_float(_aget(env_cfg, "gru_cost_base", 0.0), 0.0)
    gru_per_dim = _as_float(_aget(env_cfg, "gru_cost_per_dim", 0.0), 0.0)
    cost = 0.0
    if recurrent_state and not memory_unavailable:
        cost = max(0.0, float(gru_base) + float(gru_per_dim) * float(eff_dim))
    return {
        "raw_dim": int(raw_dim),
        "effective_dim": int(eff_dim),
        "recurrent_state": bool(recurrent_state),
        "memory_unavailable": bool(memory_unavailable),
        "hypothetical": float(cost),
        "charged": float(cost),
        "charging_enabled": bool(cost > 0.0),
        "telemetry_only": False,
    }


def prediction_ablation_flags(sim_cfg: Any = None) -> dict[str, bool]:
    ablate = bool(_aget(sim_cfg, "ablate_prediction", False))
    force_disabled = bool(_aget(sim_cfg, "force_prediction_disabled", False))
    force_frozen = bool(_aget(sim_cfg, "force_prediction_frozen", False))
    force_zero_lr = bool(_aget(sim_cfg, "force_prediction_zero_lr", False))
    force_default = bool(_aget(sim_cfg, "force_prediction_default", False))
    reset_each_tick = bool(_aget(sim_cfg, "reset_prediction_each_tick", False))
    return {
        "ablate": bool(ablate),
        "force_disabled": bool(force_disabled),
        "force_frozen": bool(force_frozen),
        "force_zero_lr": bool(force_zero_lr),
        "force_default": bool(force_default),
        "reset_each_tick": bool(reset_each_tick),
        "active": bool(ablate or force_disabled or force_frozen or force_zero_lr or force_default or reset_each_tick),
    }


def effective_prediction_learning_rate_for_genome(genome: Any, *, sim_cfg: Any = None, developmental_effective: Mapping[str, Any] | None = None) -> float:
    flags = prediction_ablation_flags(sim_cfg)
    raw = _as_float(
        (developmental_effective or {}).get("pred_lr", _gget(genome, "pred_lr", 0.08)),
        0.08,
    )
    raw = max(0.0, min(1.0, float(raw)))
    if flags["ablate"] or flags["force_disabled"] or flags["force_frozen"] or flags["force_zero_lr"]:
        return 0.0
    if flags["force_default"]:
        return 0.08
    return float(raw)


def direct_prediction_selection_pressure_present(sim_cfg: Any = None) -> bool:
    if sim_cfg is None:
        return True
    if bool(_aget(sim_cfg, "disable_prediction_error_selection_pressure", False)):
        return False
    names = (
        "coherence_pred_err_weight",
        "coherence_episode0_pred_err_weight",
        "episode0_pred_err_fitness_weight",
        "episode0_fitness_weight",
        "late_pred_cost_enable",
        "late_coherence_eff_enable",
        "late_pred_energy_tax_enable",
        "adult_move_penalty_enable",
    )
    for name in names:
        try:
            v = getattr(sim_cfg, name)
        except Exception:
            continue
        if isinstance(v, bool):
            if v:
                return True
        else:
            try:
                if float(v) != 0.0:
                    return True
            except Exception:
                pass
    return False


def prediction_cost_components(genome: Any, *, sim_cfg: Any = None, developmental_effective: Mapping[str, Any] | None = None) -> dict[str, float | bool]:
    eff_lr = effective_prediction_learning_rate_for_genome(genome, sim_cfg=sim_cfg, developmental_effective=developmental_effective)
    flags = prediction_ablation_flags(sim_cfg)
    state_available = not bool(flags["ablate"] or flags["force_disabled"])
    update_enabled = bool(state_available and eff_lr > 0.0 and not flags["force_frozen"])
    base = _as_float(_aget(sim_cfg, "prediction_cost_base", 0.0), 0.0)
    per_lr = _as_float(_aget(sim_cfg, "prediction_cost_per_lr", 0.001), 0.001)
    state = _as_float(_aget(sim_cfg, "prediction_state_cost", 0.0), 0.0)
    update_cost = max(0.0, float(base) + float(per_lr) * float(eff_lr)) if update_enabled else 0.0
    state_cost = max(0.0, float(state)) if state_available else 0.0
    hypothetical = float(update_cost + state_cost)
    charge = bool(_aget(sim_cfg, "enable_experimental_prediction_cost_charging", False))
    charged = float(hypothetical if charge else 0.0)
    return {
        "effective_lr": float(eff_lr),
        "state_available": bool(state_available),
        "update_enabled": bool(update_enabled),
        "update_hypothetical": float(update_cost),
        "state_hypothetical": float(state_cost),
        "hypothetical": float(hypothetical),
        "charged": float(charged),
        "charging_enabled": bool(charge),
        "telemetry_only": bool(not charge),
    }


def developmental_ablation_flags(sim_cfg: Any = None) -> dict[str, bool]:
    ablate = bool(_aget(sim_cfg, "ablate_developmental_cognition", False))
    force_adult = bool(_aget(sim_cfg, "force_adult_cognition", False))
    force_juvenile = bool(_aget(sim_cfg, "force_juvenile_cognition", False))
    force_legacy = bool(_aget(sim_cfg, "force_legacy_full_cognition", False))
    disable_growth = bool(_aget(sim_cfg, "disable_cognitive_growth", False))
    force_zero_cost = bool(_aget(sim_cfg, "force_zero_maturation_cost", False))
    force_max_cost = bool(_aget(sim_cfg, "force_max_maturation_cost", False))
    return {
        "ablate": bool(ablate),
        "force_adult": bool(force_adult),
        "force_juvenile": bool(force_juvenile),
        "force_legacy": bool(force_legacy),
        "disable_growth": bool(disable_growth),
        "force_zero_cost": bool(force_zero_cost),
        "force_max_cost": bool(force_max_cost),
        "active": bool(ablate or force_adult or force_juvenile or force_legacy or disable_growth or force_zero_cost or force_max_cost),
    }


def _clamp_unit(value: Any, default: float = 1.0) -> float:
    v = _as_float(value, default)
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return float(v)


def developmental_expression_state(*, age_steps: int, sim_cfg: Any = None) -> dict[str, float | bool | int | str]:
    flags = developmental_ablation_flags(sim_cfg)
    maturity_age = max(0, _as_int(_aget(sim_cfg, "developmental_maturity_age", 20), 20))
    age = max(0, int(age_steps))
    juvenile_scale = _clamp_unit(_aget(sim_cfg, "juvenile_cognition_scale", 0.5), 0.5)
    adult_scale = _clamp_unit(_aget(sim_cfg, "adult_cognition_scale", 1.0), 1.0)
    mode = str(_aget(sim_cfg, "developmental_cognition_mode", "legacy_full_expression") or "legacy_full_expression")
    enabled = bool(_aget(sim_cfg, "enable_developmental_cognition", False))
    is_mature = bool(age >= maturity_age) if maturity_age > 0 else True

    if flags["ablate"] or flags["force_legacy"] or not enabled or mode == "legacy_full_expression":
        factor = 1.0
        progress = 1.0
        effective_mode = "legacy_full_expression"
    elif flags["force_adult"]:
        factor = adult_scale
        progress = 1.0
        effective_mode = "force_adult"
    elif flags["force_juvenile"]:
        factor = juvenile_scale
        progress = 0.0
        effective_mode = "force_juvenile"
    elif flags["disable_growth"]:
        factor = juvenile_scale
        progress = 0.0
        effective_mode = "growth_disabled"
    elif mode == "linear_growth_to_maturity":
        progress = 1.0 if maturity_age <= 0 else max(0.0, min(1.0, float(age) / float(maturity_age)))
        factor = juvenile_scale + (adult_scale - juvenile_scale) * progress
        effective_mode = mode
    elif mode == "step_juvenile_adult":
        progress = 1.0 if is_mature else 0.0
        factor = adult_scale if is_mature else juvenile_scale
        effective_mode = mode
    else:
        factor = 1.0
        progress = 1.0
        effective_mode = "legacy_full_expression"

    factor = _clamp_unit(factor, 1.0)
    return {
        "age": int(age),
        "maturity_age": int(maturity_age),
        "is_juvenile": bool(not is_mature),
        "is_mature": bool(is_mature),
        "enabled": bool(enabled and not flags["ablate"] and not flags["force_legacy"] and mode != "legacy_full_expression"),
        "factor": float(factor),
        "juvenile_scale": float(juvenile_scale),
        "adult_scale": float(adult_scale),
        "progress": float(progress),
        "mode": str(effective_mode),
        "ablation_active": bool(flags["active"]),
        "force_adult": bool(flags["force_adult"]),
        "force_juvenile": bool(flags["force_juvenile"]),
        "disable_developmental": bool(flags["ablate"]),
        "force_legacy": bool(flags["force_legacy"]),
        "growth_disabled": bool(flags["disable_growth"]),
        "force_zero_cost": bool(flags["force_zero_cost"]),
        "force_max_cost": bool(flags["force_max_cost"]),
    }


def _scale_memory_dim(dim: int, factor: float) -> int:
    d = _snap_memory_dim(dim)
    if d <= 0:
        return 0
    target = float(d) * _clamp_unit(factor, 1.0)
    allowed = [int(v) for v in MEMORY_LADDER if int(v) <= int(target + 1e-9)]
    return int(max(allowed) if allowed else 0)


def _scale_sensor_radius(radius: int, factor: float) -> int:
    r = max(1, _as_int(radius, 1))
    if float(factor) >= 0.999:
        return int(r)
    target = max(1, int(float(r) * _clamp_unit(factor, 1.0)))
    allowed = [int(v) for v in SENSOR_RADIUS_LADDER if int(v) <= target]
    return int(max(allowed) if allowed else 1)


def _scale_sensor_channel_mask(mask: int, count: int, factor: float) -> tuple[int, tuple[str, ...]]:
    ids = [
        str(c.channel_id)
        for c in perception_channel_registry()
        if c.active_now and not c.future_reserved and sensor_channel_mask_contains(int(mask), str(c.channel_id))
    ]
    if not ids:
        return 0, ()
    keep_n = int(len(ids)) if float(factor) >= 0.999 else max(1, int(float(len(ids)) * _clamp_unit(factor, 1.0)))
    keep_n = min(int(keep_n), int(count), len(ids))
    kept = tuple(ids[:keep_n])
    out_mask = 0
    for c in perception_channel_registry():
        if str(c.channel_id) in kept:
            try:
                out_mask |= int(minimal_sensor_channel_mask()) if str(c.channel_id) == "local_grid_state" else 0
            except Exception:
                pass
    # Rebuild from canonical index by reusing contains against singleton masks.
    out_mask = 0
    for idx, c in enumerate(perception_channel_registry()):
        if str(c.channel_id) in kept:
            out_mask |= 1 << int(idx)
    return int(out_mask), kept


def developmental_cost_components(
    *,
    before_capacity: float,
    after_capacity: float,
    state: Mapping[str, Any],
    sim_cfg: Any = None,
) -> dict[str, float | bool]:
    charge = bool(_aget(sim_cfg, "enable_experimental_developmental_cost_charging", False))
    force_zero = bool(state.get("force_zero_cost", False))
    force_max = bool(state.get("force_max_cost", False))
    base = max(0.0, _as_float(_aget(sim_cfg, "developmental_cognition_cost_base", 0.0), 0.0))
    per_unexpressed = max(0.0, _as_float(_aget(sim_cfg, "developmental_cognition_cost_per_unexpressed_capacity", 0.001), 0.001))
    maturation = max(0.0, _as_float(_aget(sim_cfg, "maturation_cost_per_tick", 0.0), 0.0))
    unexpressed = max(0.0, float(before_capacity) - float(after_capacity))
    dev_cost = 0.0 if force_zero else max(0.0, float(base) + per_unexpressed * unexpressed)
    maturation_cost = 0.0 if force_zero else (0.0 if bool(state.get("is_mature", True)) else float(maturation))
    if force_max:
        dev_cost = max(dev_cost, 1.0)
        maturation_cost = max(maturation_cost, 1.0)
    hypothetical = float(dev_cost + maturation_cost)
    charged = float(hypothetical if charge else 0.0)
    return {
        "developmental": float(dev_cost),
        "maturation": float(maturation_cost),
        "hypothetical": float(hypothetical),
        "charged": float(charged),
        "charging_enabled": bool(charge),
        "telemetry_only": bool(not charge),
    }


def _cognitive_budget_cost_components(phenotype: CognitivePhenotype, *, sim_cfg: Any = None) -> tuple[float, ...]:
    charge_channels = bool(_aget(sim_cfg, "enable_experimental_perception_cost_charging", False))
    charge_radius = bool(_aget(sim_cfg, "enable_experimental_sensor_radius_cost_charging", False))
    perception_charged_cost = float(
        (float(phenotype.sensor_channel_cost_per_tick) if charge_channels else 0.0)
        + (float(phenotype.sensor_radius_cost_hypothetical) if charge_radius else 0.0)
    )
    return (
        float(phenotype.memory_cost_charged),
        float(phenotype.prediction_cost_charged),
        float(perception_charged_cost),
        float(phenotype.developmental_cost_charged) + float(phenotype.maturation_cost_charged),
        float(phenotype.introspection_cost_charged),
        float(phenotype.self_regulation_cost_charged),
        float(phenotype.self_modification_cost_charged),
        float(phenotype.directed_inheritance_cost_charged),
    )


def compute_env_cost_charge_scalar(phenotype: CognitivePhenotype, *, sim_cfg: Any = None) -> float:
    """Return the scalar energy charge used by env cost charging.

    This intentionally covers only the per-tick environment charge path.
    Memory upkeep is charged earlier in the env tick, and directed inheritance
    is not part of the current env cost-charging sum.
    """
    if sim_cfg is not None and not bool(_aget(sim_cfg, "enable_cognitive_budget_ledger", True)):
        return 0.0
    charge_channels = bool(_aget(sim_cfg, "enable_experimental_perception_cost_charging", False))
    charge_radius = bool(_aget(sim_cfg, "enable_experimental_sensor_radius_cost_charging", False))
    perception = (
        (float(phenotype.sensor_channel_cost_per_tick) if charge_channels else 0.0)
        + (float(phenotype.sensor_radius_cost_hypothetical) if charge_radius else 0.0)
    )
    return float(
        float(perception)
        + float(phenotype.prediction_cost_charged)
        + float(phenotype.developmental_cost_charged)
        + float(phenotype.maturation_cost_charged)
        + float(phenotype.introspection_cost_charged)
        + float(phenotype.self_regulation_cost_charged)
        + float(phenotype.self_modification_cost_charged)
    )


def compute_env_cost_charge_scalar_from_surface(surface: CostChargeSurface, *, sim_cfg: Any = None) -> float:
    """Return the env cost-charging scalar from a minimal cost surface."""
    if sim_cfg is not None and not bool(_aget(sim_cfg, "enable_cognitive_budget_ledger", True)):
        return 0.0
    charge_channels = bool(_aget(sim_cfg, "enable_experimental_perception_cost_charging", False))
    charge_radius = bool(_aget(sim_cfg, "enable_experimental_sensor_radius_cost_charging", False))
    perception = (
        (float(surface.sensor_channel_cost_per_tick) if charge_channels else 0.0)
        + (float(surface.sensor_radius_cost_hypothetical) if charge_radius else 0.0)
    )
    return float(
        float(perception)
        + float(surface.prediction_cost_charged)
        + float(surface.developmental_cost_charged)
        + float(surface.maturation_cost_charged)
        + float(surface.introspection_cost_charged)
        + float(surface.self_regulation_cost_charged)
        + float(surface.self_modification_cost_charged)
    )


def compute_env_cost_charge_scalar_from_ledger(ledger: CognitiveBudgetLedger) -> float:
    """Return the env cost-charging scalar from the reference ledger fields."""
    return float(
        float(getattr(ledger, "perception_cost_charged", 0.0))
        + float(getattr(ledger, "prediction_update_cost_charged", 0.0))
        + float(getattr(ledger, "developmental_cognition_cost_charged", 0.0))
        + float(getattr(ledger, "maturation_cost_charged", 0.0))
        + float(getattr(ledger, "introspection_cost_total_component", 0.0))
        + float(getattr(ledger, "self_regulation_cost_total_component", 0.0))
        + float(getattr(ledger, "self_modification_cost_total_component", 0.0))
    )


def _python_budget_cost_aggregation(
    components: Sequence[float],
    *,
    energy_budget: float,
) -> _CognitiveBudgetCostAggregation:
    values = tuple(float(value) for value in components)
    total = 0.0
    for value in values:
        total += float(value)
    energy_fraction = float(total / energy_budget) if energy_budget > 0.0 and total > 0.0 else 0.0
    fractions = tuple(float(value / total) if total > 0.0 and value > 0.0 else 0.0 for value in values)
    return _CognitiveBudgetCostAggregation(total=float(total), energy_fraction=float(energy_fraction), fractions=fractions)


def _native_budget_cost_aggregation_batch(
    component_rows: Sequence[Sequence[float]],
    energy_budgets: Sequence[float],
) -> tuple[list[_CognitiveBudgetCostAggregation], dict[str, object]]:
    import time as _time

    import numpy as np

    rows = list(component_rows)
    energies = [float(value) for value in energy_budgets]
    if len(rows) != len(energies):
        raise ValueError("component row count must match energy budget count")
    if not rows:
        return [], {
            "schema_version": NATIVE_BUDGET_AGGREGATION_SCHEMA_VERSION,
            "backend": "empty",
            "hit_count": 0,
            "fallback_count": 0,
            "mismatch_count": 0,
            "pack_rows": 0,
            "native_kernel_s": 0.0,
            "input_packing_s": 0.0,
        }

    pack_t0 = _time.perf_counter()
    comp = np.asarray(rows, dtype=np.float64)
    energy = np.asarray(energies, dtype=np.float64)
    if comp.ndim != 2 or int(comp.shape[1]) != _BUDGET_COST_COMPONENT_COLUMNS:
        raise ValueError("component rows must have exactly 8 cost columns")
    comp = np.ascontiguousarray(comp)
    energy = np.ascontiguousarray(energy).reshape(-1)
    packing_s = float(_time.perf_counter() - pack_t0)

    kernel_t0 = _time.perf_counter()
    try:
        from engine.perf.native import NUMBA_AVAILABLE, backend_name, cognitive_costs_native

        totals, energy_fractions, fractions = cognitive_costs_native(comp, energy)
        backend = backend_name()
        fallback_count = 0 if bool(NUMBA_AVAILABLE) else int(comp.shape[0])
    except Exception:
        totals = np.sum(comp, axis=1, dtype=np.float64)
        energy_fractions = np.divide(totals, energy, out=np.zeros_like(totals), where=(totals > 0.0) & (energy > 0.0))
        fractions = np.divide(comp, totals[:, None], out=np.zeros_like(comp), where=totals[:, None] > 0.0)
        fractions = np.where(comp > 0.0, fractions, 0.0)
        backend = "numpy-exception-fallback"
        fallback_count = int(comp.shape[0])
    kernel_s = float(_time.perf_counter() - kernel_t0)

    aggregations = [
        _CognitiveBudgetCostAggregation(
            total=float(totals[i]),
            energy_fraction=float(energy_fractions[i]),
            fractions=tuple(float(value) for value in fractions[i].tolist()),
        )
        for i in range(int(comp.shape[0]))
    ]
    return aggregations, {
        "schema_version": NATIVE_BUDGET_AGGREGATION_SCHEMA_VERSION,
        "backend": str(backend),
        "hit_count": int(comp.shape[0] - fallback_count),
        "fallback_count": int(fallback_count),
        "mismatch_count": 0,
        "pack_rows": int(comp.shape[0]),
        "native_kernel_s": float(kernel_s),
        "input_packing_s": float(packing_s),
    }


def _record_envhot_phenotype_inactive_default(diag: Any, name: str, *, count: int = 1) -> None:
    if diag is None:
        return
    try:
        diag.record_phenotype_inactive_default(str(name), count=int(count))
    except Exception:
        pass


def _envhot_current_diagnostics() -> Any:
    try:
        from simulation.env_hotpath_expression_diagnostics import current_diagnostics  # noqa: WPS433

        return current_diagnostics()
    except Exception:
        return None


def _record_envhot_phenotype_bucket(diag: Any, name: str, started: float, *, count: int = 1) -> None:
    if diag is None:
        return
    try:
        diag.record_phenotype_decomposition_bucket(str(name), time.perf_counter() - float(started), count=int(count))
    except Exception:
        pass


def _record_envhot_phenotype_inactive(diag: Any, name: str, *, count: int = 1) -> None:
    if diag is None:
        return
    try:
        diag.record_phenotype_inactive_path(str(name), count=int(count))
    except Exception:
        pass


def _envhot4_inactive_defaults_enabled(sim_cfg: Any) -> bool:
    return bool(_aget(sim_cfg, "env_hotpath_inactive_phenotype_defaults_enable", False))


def _envhot4_no_active_flags(flags: Mapping[str, Any]) -> bool:
    return not bool(flags.get("active", False))


def _envhot4_inactive_introspection_default() -> IntrospectionExpression:
    potential = len(
        tuple(
            c.channel_id
            for c in introspection_channel_registry()
            if c.active_now
            and not c.future_reserved
            and c.safe_for_agent_observation
            and not c.leaks_host_or_simulator_state
        )
    )
    return IntrospectionExpression(
        schema_version=INTROSPECTION_EXPRESSION_SCHEMA_VERSION,
        introspection_enabled=False,
        introspection_channels_available=bool(potential > 0),
        active_introspection_channel_count=0,
        potential_introspection_channel_count=int(potential),
        active_introspection_channel_mask=0,
        active_introspection_channel_ids=(),
        introspection_capacity_fraction=0.0,
        introspection_cost_hypothetical=0.0,
        introspection_cost_charged=0.0,
        introspection_ablation_active=False,
        force_all_introspection_channels_active=False,
        force_minimal_introspection_channels_active=False,
        force_no_introspection_channels_active=False,
        force_default_introspection_channels_active=False,
        mask_introspection_values_to_neutral_active=False,
        introspection_observation_shape=int(potential),
        introspection_policy_input_enabled=False,
        introspection_future_control_surface_ready=True,
    )


def _envhot4_try_inactive_introspection_default(sim_cfg: Any) -> IntrospectionExpression | None:
    if _aget(sim_cfg, "enable_introspection_channels", False):
        return None
    if not _envhot4_no_active_flags(introspection_ablation_flags(sim_cfg)):
        return None
    return _envhot4_inactive_introspection_default()


def _envhot4_inactive_self_regulation_default() -> SelfRegulationExpression:
    potential = len(
        tuple(
            c.control_id
            for c in self_regulation_control_registry()
            if c.active_now
            and not c.future_reserved
            and c.safe_for_agent_control
            and not c.can_affect_genome
            and not c.can_affect_host_or_simulator
            and not c.can_affect_source_code
        )
    )
    return SelfRegulationExpression(
        schema_version=SELF_REGULATION_EXPRESSION_SCHEMA_VERSION,
        self_regulation_enabled=False,
        self_regulation_surface_enabled=False,
        self_regulation_policy_output_enabled=False,
        active_self_regulation_control_count=0,
        potential_self_regulation_control_count=int(potential),
        active_self_regulation_control_mask=0,
        active_self_regulation_control_ids=(),
        self_regulation_capacity_fraction=0.0,
        self_regulation_cost_hypothetical=0.0,
        self_regulation_cost_charged=0.0,
        self_regulation_ablation_active=False,
        force_no_self_regulation_active=False,
        force_all_self_regulation_active=False,
        force_default_self_regulation_active=False,
        mask_self_regulation_outputs_to_neutral_active=False,
        freeze_self_regulation_state_active=False,
        reset_self_regulation_each_tick_active=False,
        disable_self_regulation_decay_active=False,
        self_regulation_runtime_state_available=False,
        self_regulation_inheritable=False,
        self_modification_active=False,
        genome_editing_active=False,
    )


def _envhot4_try_inactive_self_regulation_default(sim_cfg: Any) -> SelfRegulationExpression | None:
    if _aget(sim_cfg, "enable_self_regulation", False) or _aget(sim_cfg, "enable_self_regulation_surface", False):
        return None
    if not _envhot4_no_active_flags(self_regulation_ablation_flags(sim_cfg)):
        return None
    return _envhot4_inactive_self_regulation_default()


def _envhot4_inactive_self_modification_default() -> SelfModificationExpression:
    potential = len(
        tuple(
            t.target_id
            for t in self_modification_target_registry()
            if t.active_now
            and not t.future_reserved
            and t.modifiable
            and t.safe_for_agent_modification
            and not t.can_affect_genome
            and not t.can_affect_host_or_simulator
            and not t.can_affect_source_code
        )
    )
    return SelfModificationExpression(
        schema_version=SELF_MODIFICATION_EXPRESSION_SCHEMA_VERSION,
        self_modification_enabled=False,
        self_modification_policy_output_enabled=False,
        active_self_modification_target_count=0,
        potential_self_modification_target_count=int(potential),
        active_self_modification_target_mask=0,
        active_self_modification_target_ids=(),
        self_modification_capacity_fraction=0.0,
        self_modification_cost_hypothetical=0.0,
        self_modification_cost_charged=0.0,
        self_modification_ablation_active=False,
        force_no_self_modification_active=False,
        force_all_self_modification_active=False,
        force_default_self_modification_active=False,
        mask_self_modification_commands_to_neutral_active=False,
        freeze_self_modification_state_active=False,
        reset_self_modification_each_tick_active=False,
        reset_self_modification_on_reproduction_active=False,
        disable_self_modification_decay_active=False,
        self_modification_runtime_state_available=False,
        self_modification_lifetime_persistent=False,
        self_modification_inheritable=False,
        genome_editing_active=False,
    )


def _envhot4_try_inactive_self_modification_default(sim_cfg: Any) -> SelfModificationExpression | None:
    if _aget(sim_cfg, "enable_self_modification", False):
        return None
    if not _envhot4_no_active_flags(self_modification_ablation_flags(sim_cfg)):
        return None
    return _envhot4_inactive_self_modification_default()


def _envhot4_empty_directed_payload_default() -> DirectedInheritancePayload:
    return DirectedInheritancePayload()


def _envhot4_empty_directed_payload_summary_default() -> dict[str, float]:
    return {
        "delta_abs_mean": 0.0,
        "delta_abs_p90": 0.0,
        "payload_abs_mean": 0.0,
        "payload_abs_p90": 0.0,
        "clipped_fraction": 0.0,
        "saturation_fraction": 0.0,
        "event_count": 0.0,
        "neutral_command_fraction": 1.0,
        "payload_reset_count": 0.0,
        "payload_applied_fraction": 0.0,
        "payload_frozen_fraction": 0.0,
        "cost": 0.0,
        "offspring_developmental_expression_bias": 0.0,
        "offspring_juvenile_cognition_scale_bias": 0.0,
        "offspring_memory_expression_bias": 0.0,
        "offspring_prediction_lr_bias": 0.0,
        "offspring_exploration_bias_marker": 0.0,
        "offspring_reproduction_threshold_bias": 0.0,
    }


def _envhot4_inactive_directed_inheritance_default() -> DirectedInheritanceExpression:
    potential = len(
        tuple(
            c.influence_id
            for c in directed_inheritance_channel_registry()
            if c.active_now
            and not c.future_reserved
            and c.maskable
            and c.safe_for_directed_inheritance
            and not c.can_affect_genome
            and not c.can_affect_mutation_rate
            and not c.can_affect_recombination
            and not c.can_affect_host_or_simulator
            and not c.can_affect_source_code
        )
    )
    return DirectedInheritanceExpression(
        schema_version=DIRECTED_INHERITANCE_EXPRESSION_SCHEMA_VERSION,
        directed_inheritance_enabled=False,
        directed_inheritance_surface_enabled=False,
        directed_inheritance_policy_output_enabled=False,
        active_directed_inheritance_channel_count=0,
        potential_directed_inheritance_channel_count=int(potential),
        active_directed_inheritance_channel_mask=0,
        active_directed_inheritance_channel_ids=(),
        directed_inheritance_capacity_fraction=0.0,
        directed_inheritance_cost_hypothetical=0.0,
        directed_inheritance_cost_charged=0.0,
        directed_inheritance_ablation_active=False,
        force_no_directed_inheritance_active=False,
        force_all_directed_inheritance_active=False,
        force_default_directed_inheritance_active=False,
        mask_directed_inheritance_commands_to_neutral_active=False,
        freeze_directed_inheritance_payload_active=False,
        reset_directed_inheritance_payload_on_birth_active=False,
        reset_directed_inheritance_payload_on_maturity_active=False,
        directed_inheritance_runtime_state_available=False,
        non_genomic_payload_available=True,
        non_genomic_payload_applied=False,
        genome_editing_active=False,
        mutation_rate_control_active=False,
        recombination_control_active=False,
        inherited_self_modification_state_active=False,
    )


def _envhot4_try_inactive_directed_inheritance_default(sim_cfg: Any) -> DirectedInheritanceExpression | None:
    if _aget(sim_cfg, "enable_directed_inheritance", False) or _aget(sim_cfg, "enable_directed_inheritance_surface", False):
        return None
    if not _envhot4_no_active_flags(directed_inheritance_ablation_flags(sim_cfg)):
        return None
    return _envhot4_inactive_directed_inheritance_default()


DEVELOPMENTAL_EFFECTIVE_PROJECTION_FIELDS = (
    "aggression_bias",
    "action_temperature",
    "drive_logit_gain",
    "drive_inertia",
    "drive_noise",
    "pred_lr",
    "reproduction_threshold",
)


def developmental_effective_projection(expression: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(expression, Mapping):
        return {}
    raw = expression.get("effective", expression)
    if not isinstance(raw, Mapping):
        return {}
    return {
        str(key): raw[key]
        for key in DEVELOPMENTAL_EFFECTIVE_PROJECTION_FIELDS
        if key in raw
    }


def _developmental_effective(agent: Any, age_steps: int) -> dict[str, Any]:
    _diag = None
    _dev_t0 = 0.0
    try:
        from simulation.env_hotpath_expression_diagnostics import current_diagnostics  # noqa: WPS433

        _diag = current_diagnostics()
        if _diag is not None:
            _dev_t0 = time.perf_counter()
    except Exception:
        _diag = None

    if _diag is not None:
        _dev_t0 = time.perf_counter()

    def _record_source(source: str) -> None:
        if _diag is None:
            return
        try:
            _diag.record_developmental_effective_lookup(
                source=str(source),
                agent_id=int(_aget(agent, "agent_id", 0) or 0),
                age_steps=int(age_steps),
                seconds=time.perf_counter() - _dev_t0,
            )
        except Exception:
            pass

    g = _aget(agent, "genome", None)
    if g is None:
        _record_source("no_genome")
        return {}
    if agent is not None and hasattr(agent, "_eiel_cached_cognitive_expression"):
        cached = getattr(agent, "_eiel_cached_cognitive_expression", None)
        if isinstance(cached, dict):
            if _diag is not None:
                try:
                    _diag.record_decomposition_bucket("env_apply_developmental_expression_total", time.perf_counter() - _dev_t0)
                except Exception:
                    pass
            _record_source("agent_cached_cognitive_expression")
            return cached
    if hasattr(g, "resolve_developmental_expression"):
        try:
            out = dict(g.resolve_developmental_expression(age_steps=int(age_steps)))
            if _diag is not None and bool(getattr(_diag, "developmental_effective_projection_enabled", False)):
                try:
                    _proj_t0 = time.perf_counter()
                    _direct_projection = developmental_effective_projection(out)
                    _proj_construct_s = time.perf_counter() - _proj_t0
                    _diag.record_developmental_effective_projection_check(
                        candidate=getattr(agent, "_eiel_dev_expression_cache_candidate", None),
                        direct=out,
                        direct_projection=_direct_projection,
                        agent_id=int(_aget(agent, "agent_id", 0) or 0),
                        age_steps=int(age_steps),
                        seconds=time.perf_counter() - _proj_t0,
                        projection_seconds=_proj_construct_s,
                    )
                except Exception:
                    pass
            if _diag is not None and bool(getattr(_diag, "developmental_cache_equality_enabled", False)):
                try:
                    _eq_t0 = time.perf_counter()
                    _diag.record_developmental_cache_equality_check(
                        candidate=getattr(agent, "_eiel_dev_expression_cache_candidate", None),
                        direct=out,
                        agent_id=int(_aget(agent, "agent_id", 0) or 0),
                        age_steps=int(age_steps),
                        seconds=time.perf_counter() - _eq_t0,
                    )
                except Exception:
                    pass
            if _diag is not None:
                try:
                    _diag.record_decomposition_bucket("env_apply_developmental_expression_total", time.perf_counter() - _dev_t0)
                except Exception:
                    pass
            _record_source("genome_resolve")
            return out
        except Exception:
            _record_source("genome_resolve_exception")
            return {}
    _record_source("no_resolver")
    return {}


def express_cognitive_phenotype(
    genome: Any,
    *,
    agent: Any = None,
    env_cfg: Any = None,
    sim_cfg: Any = None,
    age_steps: int | None = None,
) -> CognitivePhenotype:
    _phenotype_total_t0 = time.perf_counter()
    _scalar_t0 = _phenotype_total_t0
    st = _aget(agent, "stats", None)
    if age_steps is None:
        age_steps = _as_int(_aget(st, "survival_steps", 0), 0)

    eff = _developmental_effective(agent, int(age_steps)) if agent is not None else {}

    mem_dim = _snap_memory_dim(_gget(genome, "memory_dim", 0))
    use_gru = _as_bool(_gget(genome, "use_gru", False), False) or mem_dim > 0
    flags = memory_ablation_flags(sim_cfg)
    eff_mem_dim_before_dev = effective_memory_dim_for_genome(genome, sim_cfg=sim_cfg)
    eff_mem_dim = int(eff_mem_dim_before_dev)
    memory_enabled = bool(eff_mem_dim > 0)
    policy_memory_enabled = bool(use_gru and eff_mem_dim > 0)
    max_memory_dim = int(max(MEMORY_LADDER))

    hidden_size = _as_int(_gget(genome, "hidden_size", FIXED_POLICY_HIDDEN_SIZE), FIXED_POLICY_HIDDEN_SIZE)

    pred_lr = _as_float(eff.get("pred_lr", _gget(genome, "pred_lr", 0.08)), 0.08)
    pred_lr = max(0.0, min(1.0, pred_lr))
    pred_flags = prediction_ablation_flags(sim_cfg)
    pred_costs = prediction_cost_components(genome, sim_cfg=sim_cfg, developmental_effective=eff)
    eff_pred_lr = float(pred_costs["effective_lr"])
    prediction_state_available = bool(pred_costs["state_available"])
    prediction_update_enabled = bool(pred_costs["update_enabled"])
    direct_pred_pressure = direct_prediction_selection_pressure_present(sim_cfg)

    drive_gain = _as_float(eff.get("drive_logit_gain", _gget(genome, "drive_logit_gain", 1.0)), 1.0)
    drive_inertia = _as_float(eff.get("drive_inertia", _gget(genome, "drive_inertia", 0.85)), 0.85)
    drive_noise = _as_float(eff.get("drive_noise", _gget(genome, "drive_noise", 0.02)), 0.02)
    action_temperature = _as_float(eff.get("action_temperature", _gget(genome, "action_temperature", 1.0)), 1.0)
    aggression_bias = _as_float(eff.get("aggression_bias", _gget(genome, "aggression_bias", 0.0)), 0.0)
    reproduction_threshold = _as_float(
        eff.get("reproduction_threshold", _gget(genome, "reproduction_threshold", 30.0)),
        30.0,
    )

    module_count = len(list(_gget(genome, "modules", []) or []))
    connection_count = len(list(_gget(genome, "connections", []) or []))
    active_module_count = _as_int(_aget(st, "_dev_active_module_count", module_count), module_count)

    stage = ""
    if st is not None:
        stage = str(_aget(st, "_dev_stage_id", "") or "")
    if not stage and hasattr(genome, "developmental_stage_id"):
        try:
            stage = str(genome.developmental_stage_id(age_steps=int(age_steps)))
        except Exception:
            stage = "default"
    if not stage:
        stage = "default"

    sensor_capacity = 0
    if env_cfg is not None:
        try:
            from env.gridworld_pkg.patch_world import patch_world_total_observation_size

            sensor_capacity = int(patch_world_total_observation_size(env_cfg))
        except Exception:
            sensor_capacity = 0

    _diag = None
    try:
        from simulation.env_hotpath_expression_diagnostics import current_diagnostics  # noqa: WPS433

        _diag = current_diagnostics()
    except Exception:
        _diag = None
    _record_envhot_phenotype_bucket(_diag, "phenotype_scalar_extraction_total", _scalar_t0)
    _perception_t0 = time.perf_counter()
    perception = express_perception(genome, env_cfg=env_cfg, sim_cfg=sim_cfg)
    if _diag is not None:
        try:
            _diag.record_decomposition_bucket("env_apply_perception_total", time.perf_counter() - _perception_t0)
        except Exception:
            pass
    _record_envhot_phenotype_bucket(_diag, "phenotype_perception_total", _perception_t0)
    _envhot4_enabled = _envhot4_inactive_defaults_enabled(sim_cfg)
    _introspection_t0 = time.perf_counter()
    introspection = None
    if _envhot4_enabled:
        introspection = _envhot4_try_inactive_introspection_default(sim_cfg)
        if introspection is not None:
            _record_envhot_phenotype_inactive_default(_diag, "factory:introspection")
        else:
            _record_envhot_phenotype_inactive_default(_diag, "fallback:introspection")
    if introspection is None:
        introspection = express_introspection(genome, sim_cfg=sim_cfg)
    _record_envhot_phenotype_bucket(_diag, "phenotype_introspection_total", _introspection_t0)
    if not bool(introspection.introspection_enabled or introspection.introspection_ablation_active):
        _record_envhot_phenotype_inactive(_diag, "introspection_inactive")
    _self_regulation_t0 = time.perf_counter()
    self_regulation = None
    if _envhot4_enabled:
        self_regulation = _envhot4_try_inactive_self_regulation_default(sim_cfg)
        if self_regulation is not None:
            _record_envhot_phenotype_inactive_default(_diag, "factory:self_regulation")
        else:
            _record_envhot_phenotype_inactive_default(_diag, "fallback:self_regulation")
    if self_regulation is None:
        self_regulation = express_self_regulation(genome, sim_cfg=sim_cfg)
    _record_envhot_phenotype_bucket(_diag, "phenotype_self_regulation_total", _self_regulation_t0)
    if not bool(self_regulation.self_regulation_enabled or self_regulation.self_regulation_ablation_active):
        _record_envhot_phenotype_inactive(_diag, "self_regulation_inactive")
    _self_modification_t0 = time.perf_counter()
    self_modification = None
    if _envhot4_enabled:
        self_modification = _envhot4_try_inactive_self_modification_default(sim_cfg)
        if self_modification is not None:
            _record_envhot_phenotype_inactive_default(_diag, "factory:self_modification")
        else:
            _record_envhot_phenotype_inactive_default(_diag, "fallback:self_modification")
    if self_modification is None:
        self_modification = express_self_modification(genome, sim_cfg=sim_cfg)
    _record_envhot_phenotype_bucket(_diag, "phenotype_self_modification_total", _self_modification_t0)
    if not bool(self_modification.self_modification_enabled or self_modification.self_modification_ablation_active):
        _record_envhot_phenotype_inactive(_diag, "self_modification_inactive")
    _directed_payload_t0 = time.perf_counter()
    _raw_directed_payload = _aget(agent, "directed_inheritance_payload", None)
    _directed_payload_factory_used = False
    if (
        _envhot4_enabled
        and not isinstance(_raw_directed_payload, Mapping)
        and not isinstance(_raw_directed_payload, DirectedInheritancePayload)
    ):
        directed_payload = _envhot4_empty_directed_payload_default()
        _directed_payload_factory_used = True
        _record_envhot_phenotype_inactive_default(_diag, "factory:directed_payload_empty")
    else:
        if _envhot4_enabled:
            _record_envhot_phenotype_inactive_default(_diag, "fallback:directed_payload")
        directed_payload = DirectedInheritancePayload.from_dict(_raw_directed_payload)
    _record_envhot_phenotype_bucket(_diag, "phenotype_directed_payload_from_dict_total", _directed_payload_t0)
    if not isinstance(_raw_directed_payload, Mapping):
        _record_envhot_phenotype_inactive(_diag, "directed_payload_non_dict_or_none")
    _directed_expression_t0 = time.perf_counter()
    directed_inheritance = None
    if _envhot4_enabled:
        directed_inheritance = _envhot4_try_inactive_directed_inheritance_default(sim_cfg)
        if directed_inheritance is not None:
            _record_envhot_phenotype_inactive_default(_diag, "factory:directed_inheritance")
        else:
            _record_envhot_phenotype_inactive_default(_diag, "fallback:directed_inheritance")
    if directed_inheritance is None:
        directed_inheritance = express_directed_inheritance(genome, payload=directed_payload, sim_cfg=sim_cfg)
    _record_envhot_phenotype_bucket(_diag, "phenotype_directed_inheritance_total", _directed_expression_t0)
    if not bool(directed_inheritance.directed_inheritance_enabled or directed_inheritance.directed_inheritance_ablation_active):
        _record_envhot_phenotype_inactive(_diag, "directed_inheritance_inactive")
    _directed_summary_t0 = time.perf_counter()
    if _envhot4_enabled and _directed_payload_factory_used:
        directed_summary = _envhot4_empty_directed_payload_summary_default()
        _record_envhot_phenotype_inactive_default(_diag, "factory:directed_payload_summary_empty")
    else:
        if _envhot4_enabled:
            _record_envhot_phenotype_inactive_default(_diag, "fallback:directed_payload_summary")
        directed_summary = directed_inheritance_payload_summary(directed_payload)
    _record_envhot_phenotype_bucket(_diag, "phenotype_directed_payload_summary_total", _directed_summary_t0)

    _developmental_t0 = time.perf_counter()
    _dev_state_t0 = _developmental_t0
    dev_state = developmental_expression_state(age_steps=int(age_steps), sim_cfg=sim_cfg)
    if _diag is not None:
        try:
            _diag.record_decomposition_bucket("env_apply_developmental_state_total", time.perf_counter() - _dev_state_t0)
        except Exception:
            pass
    dev_factor = float(dev_state["factor"])
    sensor_radius_before_dev = int(perception.active_sensor_radius)
    sensor_channel_count_before_dev = int(perception.active_sensor_channel_count)
    pred_lr_before_dev = float(eff_pred_lr)
    active_channel_mask = int(perception.active_sensor_channel_mask)
    active_channel_ids = tuple(perception.active_sensor_channel_ids)
    active_channel_count = int(perception.active_sensor_channel_count)
    active_radius = int(perception.active_sensor_radius)
    observed_cells = int(perception.active_observed_cell_count)
    active_cell_fraction = float(perception.active_cell_fraction)
    cell_channel_capacity = float(perception.cell_channel_capacity)
    spatial_perception_capacity = float(perception.spatial_perception_capacity)
    perception_capacity_component = float(perception.perception_capacity_component)
    sensor_radius_hyp_cost = float(perception.sensor_radius_cost_hypothetical)
    sensor_channel_cost = float(perception.sensor_channel_cost_per_tick)
    perception_hyp_cost = float(perception.perception_cost_hypothetical)

    if bool(dev_state["enabled"]):
        eff_mem_dim = _scale_memory_dim(eff_mem_dim_before_dev, dev_factor)
        active_radius = _scale_sensor_radius(sensor_radius_before_dev, dev_factor)
        active_channel_mask, active_channel_ids = _scale_sensor_channel_mask(active_channel_mask, active_channel_count, dev_factor)
        active_channel_count = int(len(active_channel_ids))
        observed_cells = int((2 * int(active_radius) + 1) ** 2)
        max_cells = max(1, int(perception.max_observed_cell_count))
        active_cell_fraction = float(observed_cells) / float(max_cells)
        cell_channel_capacity = float(observed_cells * active_channel_count)
        spatial_perception_capacity = float(cell_channel_capacity)
        perception_capacity_component = float(cell_channel_capacity)
        eff_pred_lr = float(eff_pred_lr) * dev_factor
        prediction_update_enabled = bool(prediction_state_available and eff_pred_lr > 0.0 and not pred_flags["force_frozen"])
        sensor_channel_cost = float(sensor_channel_cost) * dev_factor
        sensor_radius_hyp_cost = float(sensor_radius_hyp_cost) * dev_factor
        perception_hyp_cost = float(sensor_channel_cost + sensor_radius_hyp_cost)

    memory_enabled = bool(eff_mem_dim > 0)
    policy_memory_enabled = bool(use_gru and eff_mem_dim > 0)
    memory_ladder_index = int(MEMORY_LADDER.index(_snap_memory_dim(eff_mem_dim)))
    memory_capacity_fraction = float(eff_mem_dim) / float(max_memory_dim) if max_memory_dim > 0 else 0.0

    memory_unavailable = bool(flags["ablate"] or flags["force_disabled"])
    gru_base = _as_float(_aget(env_cfg, "gru_cost_base", 0.0), 0.0)
    gru_per_dim = _as_float(_aget(env_cfg, "gru_cost_per_dim", 0.0), 0.0)
    memory_cost = 0.0
    if use_gru and not memory_unavailable:
        memory_cost = max(0.0, float(gru_base) + float(gru_per_dim) * float(eff_mem_dim))
    memory_costs = {
        "hypothetical": float(memory_cost),
        "charged": float(memory_cost),
    }

    pred_base = _as_float(_aget(sim_cfg, "prediction_cost_base", 0.0), 0.0)
    pred_per_lr = _as_float(_aget(sim_cfg, "prediction_cost_per_lr", 0.001), 0.001)
    pred_state_cost = _as_float(_aget(sim_cfg, "prediction_state_cost", 0.0), 0.0)
    pred_update_hyp = max(0.0, float(pred_base) + float(pred_per_lr) * float(eff_pred_lr)) if prediction_update_enabled else 0.0
    pred_state_hyp = max(0.0, float(pred_state_cost)) if prediction_state_available else 0.0
    pred_hyp_total = float(pred_update_hyp + pred_state_hyp)
    pred_charged_total = float(pred_hyp_total if _as_bool(_aget(sim_cfg, "enable_experimental_prediction_cost_charging", False), False) else 0.0)

    prediction_enabled = bool(prediction_state_available and eff_pred_lr > 0.0)
    memory_component = float(eff_mem_dim)
    prediction_component = float(eff_pred_lr if prediction_enabled else 0.0)
    drive_component = float(max(0.0, drive_gain))
    perception_component = (
        float(perception_capacity_component)
        if bool(perception.perception_masking_enabled or perception.sensor_radius_masking_enabled)
        else 0.0
    )
    introspection_component = (
        float(introspection.active_introspection_channel_count)
        if bool(introspection.introspection_enabled or introspection.introspection_ablation_active)
        else 0.0
    )
    self_regulation_component = (
        float(self_regulation.active_self_regulation_control_count)
        if bool(self_regulation.self_regulation_enabled or self_regulation.self_regulation_ablation_active)
        else 0.0
    )
    self_modification_component = (
        float(self_modification.active_self_modification_target_count)
        if bool(self_modification.self_modification_enabled or self_modification.self_modification_ablation_active)
        else 0.0
    )
    directed_inheritance_component = (
        float(directed_inheritance.active_directed_inheritance_channel_count)
        if bool(directed_inheritance.directed_inheritance_enabled or directed_inheritance.directed_inheritance_ablation_active)
        else 0.0
    )
    perception_component_before = (
        float(perception.perception_capacity_component)
        if bool(perception.perception_masking_enabled or perception.sensor_radius_masking_enabled)
        else 0.0
    )
    before_capacity = float(eff_mem_dim_before_dev)
    if bool(use_gru and eff_mem_dim_before_dev > 0):
        before_capacity += 1.0
    before_capacity += float(pred_lr_before_dev if prediction_state_available and pred_lr_before_dev > 0.0 else 0.0)
    before_capacity += float(drive_component)
    before_capacity += float(max(0, active_module_count))
    before_capacity += float(perception_component_before)

    active_capacity = float(memory_component)
    if policy_memory_enabled:
        active_capacity += 1.0
    active_capacity += float(prediction_component)
    active_capacity += float(drive_component)
    active_capacity += float(max(0, active_module_count))
    active_capacity += float(perception_component)
    active_capacity += float(introspection_component)
    active_capacity += float(self_regulation_component)
    active_capacity += float(self_modification_component)
    active_capacity += float(directed_inheritance_component)

    potential_capacity = float(max(MEMORY_LADDER))
    potential_capacity += 1.0
    potential_capacity += 1.0
    potential_capacity += 3.0
    potential_capacity += float(max(module_count, active_module_count))
    if bool(perception.perception_masking_enabled or perception.sensor_radius_masking_enabled):
        potential_capacity += float(perception.max_observed_cell_count * max(1, perception.potential_sensor_channel_count))
    if bool(introspection.introspection_channels_available):
        potential_capacity += float(introspection.potential_introspection_channel_count)
    if bool(self_regulation.self_regulation_runtime_state_available):
        potential_capacity += float(self_regulation.potential_self_regulation_control_count)
    if bool(self_modification.self_modification_runtime_state_available):
        potential_capacity += float(self_modification.potential_self_modification_target_count)
    if bool(directed_inheritance.directed_inheritance_runtime_state_available):
        potential_capacity += float(directed_inheritance.potential_directed_inheritance_channel_count)

    dev_costs = developmental_cost_components(
        before_capacity=float(before_capacity),
        after_capacity=float(active_capacity),
        state=dev_state,
        sim_cfg=sim_cfg,
    )
    _record_envhot_phenotype_bucket(_diag, "phenotype_developmental_expression_total", _developmental_t0)

    _dataclass_t0 = time.perf_counter()
    phenotype = CognitivePhenotype(
        schema_version=PHENOTYPE_SCHEMA_VERSION,
        age=int(age_steps),
        memory_enabled=memory_enabled,
        policy_memory_enabled=policy_memory_enabled,
        active_memory_capacity=int(eff_mem_dim if memory_enabled else 0),
        effective_memory_capacity=int(eff_mem_dim if use_gru else 0),
        genetic_memory_dim=int(mem_dim),
        active_memory_dim=int(eff_mem_dim if memory_enabled else 0),
        effective_memory_dim=int(eff_mem_dim if use_gru else 0),
        max_memory_dim=int(max_memory_dim),
        memory_ladder_index=int(memory_ladder_index),
        memory_capacity_fraction=float(memory_capacity_fraction),
        recurrent_state_enabled=bool(use_gru),
        memory_masking_enabled=bool(use_gru),
        memory_cost_hypothetical=float(memory_costs["hypothetical"]),
        memory_cost_charged=float(memory_costs["charged"]),
        memory_ablation_active=bool(flags["active"]),
        force_memory_disabled_active=bool(flags["ablate"] or flags["force_disabled"]),
        force_memory_min_active=bool(flags["force_min"]),
        force_memory_max_active=bool(flags["force_max"]),
        force_memory_default_active=bool(flags["force_default"]),
        reset_memory_each_tick_active=bool(flags["reset_each_tick"]),
        memory_use_measurement_available=bool(use_gru),
        maturity_age=int(dev_state["maturity_age"]),
        is_juvenile=bool(dev_state["is_juvenile"]),
        is_mature=bool(dev_state["is_mature"]),
        developmental_cognition_enabled=bool(dev_state["enabled"]),
        developmental_expression_factor=float(dev_state["factor"]),
        juvenile_cognition_scale=float(dev_state["juvenile_scale"]),
        adult_cognition_scale=float(dev_state["adult_scale"]),
        cognitive_growth_progress=float(dev_state["progress"]),
        active_memory_dim_before_development=int(eff_mem_dim_before_dev),
        active_memory_dim_after_development=int(eff_mem_dim),
        active_sensor_radius_before_development=int(sensor_radius_before_dev),
        active_sensor_radius_after_development=int(active_radius),
        active_sensor_channel_count_before_development=int(sensor_channel_count_before_dev),
        active_sensor_channel_count_after_development=int(active_channel_count),
        effective_prediction_lr_before_development=float(pred_lr_before_dev),
        effective_prediction_lr_after_development=float(eff_pred_lr),
        developmental_cost_hypothetical=float(dev_costs["developmental"]),
        developmental_cost_charged=float(dev_costs["developmental"] if bool(dev_costs["charging_enabled"]) else 0.0),
        maturation_cost_hypothetical=float(dev_costs["maturation"]),
        maturation_cost_charged=float(dev_costs["maturation"] if bool(dev_costs["charging_enabled"]) else 0.0),
        developmental_ablation_active=bool(dev_state["ablation_active"]),
        force_adult_cognition_active=bool(dev_state["force_adult"]),
        force_juvenile_cognition_active=bool(dev_state["force_juvenile"]),
        disable_developmental_cognition_active=bool(dev_state["disable_developmental"]),
        force_legacy_full_cognition_active=bool(dev_state["force_legacy"]),
        developmental_growth_disabled_active=bool(dev_state["growth_disabled"]),
        developmental_cost_charging_enabled=bool(dev_costs["charging_enabled"]),
        policy_hidden_size=int(hidden_size),
        active_prediction_capability=prediction_enabled,
        prediction_enabled=prediction_enabled,
        prediction_update_enabled=bool(prediction_update_enabled),
        prediction_learning_rate=float(pred_lr),
        effective_prediction_learning_rate=float(eff_pred_lr),
        prediction_capacity=float(eff_pred_lr if prediction_state_available else 0.0),
        prediction_state_available=bool(prediction_state_available),
        prediction_error_tracking_enabled=bool(prediction_state_available),
        prediction_cost_hypothetical=float(pred_hyp_total),
        prediction_cost_charged=float(pred_charged_total),
        prediction_ablation_active=bool(pred_flags["active"]),
        force_prediction_disabled_active=bool(pred_flags["ablate"] or pred_flags["force_disabled"]),
        force_prediction_frozen_active=bool(pred_flags["force_frozen"]),
        force_prediction_zero_lr_active=bool(pred_flags["force_zero_lr"]),
        force_prediction_default_active=bool(pred_flags["force_default"]),
        prediction_reset_each_tick_active=bool(pred_flags["reset_each_tick"]),
        prediction_use_measurement_available=bool(prediction_state_available),
        direct_prediction_selection_pressure_present=bool(direct_pred_pressure),
        drive_logit_gain=float(drive_gain),
        drive_inertia=float(drive_inertia),
        drive_noise=float(drive_noise),
        hunger_sensitivity=_as_float(_gget(genome, "drive_sp_hunger", 0.55), 0.55),
        pain_sensitivity=_as_float(_gget(genome, "drive_sp_pain", 0.05), 0.05),
        fatigue_sensitivity=_as_float(_gget(genome, "drive_sp_fatigue", 0.15), 0.15),
        fear_sensitivity=_as_float(_gget(genome, "drive_sp_fear", 0.10), 0.10),
        action_temperature=float(action_temperature),
        exploration_bias=_as_float(_gget(genome, "exploration_bias", 0.0), 0.0),
        aggression_bias=float(aggression_bias),
        reproduction_threshold=float(reproduction_threshold),
        developmental_stage=str(stage),
        developmental_expression_active=bool(active_module_count != module_count or stage != "default" or dev_state["enabled"]),
        module_count=int(module_count),
        active_module_count=int(active_module_count),
        connection_count=int(connection_count),
        sensor_observation_capacity=int(sensor_capacity),
        active_sensor_channel_count=int(active_channel_count),
        potential_sensor_channel_count=int(perception.potential_sensor_channel_count),
        active_sensor_channel_mask=int(active_channel_mask),
        active_sensor_channel_ids=tuple(active_channel_ids),
        perception_enabled=bool(perception.perception_enabled),
        perception_masking_enabled=bool(perception.perception_masking_enabled),
        perception_cost_enabled=bool(perception.perception_cost_enabled),
        sensor_channel_cost_total=float(sensor_channel_cost),
        sensor_channel_cost_per_tick=float(sensor_channel_cost),
        sensor_channel_mask_ablation_active=bool(perception.sensor_channel_mask_ablation_active),
        sensor_channel_force_all_active=bool(perception.sensor_channel_force_all_active),
        sensor_channel_force_minimal_active=bool(perception.sensor_channel_force_minimal_active),
        active_sensor_radius=int(active_radius),
        max_sensor_radius=int(perception.max_sensor_radius),
        default_sensor_radius=int(perception.default_sensor_radius),
        sensor_radius_masking_enabled=bool(perception.sensor_radius_masking_enabled),
        active_observed_cell_count=int(observed_cells),
        max_observed_cell_count=int(perception.max_observed_cell_count),
        active_cell_fraction=float(active_cell_fraction),
        cell_channel_capacity=float(cell_channel_capacity),
        spatial_perception_capacity=float(spatial_perception_capacity),
        perception_capacity_component=float(perception_capacity_component),
        sensor_radius_cost_hypothetical=float(sensor_radius_hyp_cost),
        sensor_radius_cost_charged=float(perception.sensor_radius_cost_charged),
        perception_cost_hypothetical=float(perception_hyp_cost),
        perception_cost_charged=float(perception.perception_cost_charged),
        sensor_radius_ablation_active=bool(perception.sensor_radius_ablation_active),
        sensor_radius_force_min_active=bool(perception.sensor_radius_force_min_active),
        sensor_radius_force_max_active=bool(perception.sensor_radius_force_max_active),
        sensor_radius_force_default_active=bool(perception.sensor_radius_force_default_active),
        introspection_enabled=bool(introspection.introspection_enabled),
        introspection_channels_available=bool(introspection.introspection_channels_available),
        active_introspection_channel_count=int(introspection.active_introspection_channel_count),
        potential_introspection_channel_count=int(introspection.potential_introspection_channel_count),
        active_introspection_channel_mask=int(introspection.active_introspection_channel_mask),
        active_introspection_channel_ids=tuple(introspection.active_introspection_channel_ids),
        introspection_capacity_fraction=float(introspection.introspection_capacity_fraction),
        introspection_cost_hypothetical=float(introspection.introspection_cost_hypothetical),
        introspection_cost_charged=float(introspection.introspection_cost_charged),
        introspection_ablation_active=bool(introspection.introspection_ablation_active),
        force_all_introspection_channels_active=bool(introspection.force_all_introspection_channels_active),
        force_minimal_introspection_channels_active=bool(introspection.force_minimal_introspection_channels_active),
        force_no_introspection_channels_active=bool(introspection.force_no_introspection_channels_active),
        force_default_introspection_channels_active=bool(introspection.force_default_introspection_channels_active),
        mask_introspection_values_to_neutral_active=bool(introspection.mask_introspection_values_to_neutral_active),
        introspection_observation_shape=int(introspection.introspection_observation_shape),
        introspection_policy_input_enabled=bool(introspection.introspection_policy_input_enabled),
        introspection_future_control_surface_ready=bool(introspection.introspection_future_control_surface_ready),
        self_regulation_enabled=bool(self_regulation.self_regulation_enabled),
        self_regulation_surface_enabled=bool(self_regulation.self_regulation_surface_enabled),
        self_regulation_policy_output_enabled=bool(self_regulation.self_regulation_policy_output_enabled),
        active_self_regulation_control_count=int(self_regulation.active_self_regulation_control_count),
        potential_self_regulation_control_count=int(self_regulation.potential_self_regulation_control_count),
        active_self_regulation_control_mask=int(self_regulation.active_self_regulation_control_mask),
        active_self_regulation_control_ids=tuple(self_regulation.active_self_regulation_control_ids),
        self_regulation_capacity_fraction=float(self_regulation.self_regulation_capacity_fraction),
        self_regulation_cost_hypothetical=float(self_regulation.self_regulation_cost_hypothetical),
        self_regulation_cost_charged=float(self_regulation.self_regulation_cost_charged),
        self_regulation_ablation_active=bool(self_regulation.self_regulation_ablation_active),
        force_no_self_regulation_active=bool(self_regulation.force_no_self_regulation_active),
        force_all_self_regulation_active=bool(self_regulation.force_all_self_regulation_active),
        force_default_self_regulation_active=bool(self_regulation.force_default_self_regulation_active),
        mask_self_regulation_outputs_to_neutral_active=bool(self_regulation.mask_self_regulation_outputs_to_neutral_active),
        freeze_self_regulation_state_active=bool(self_regulation.freeze_self_regulation_state_active),
        reset_self_regulation_each_tick_active=bool(self_regulation.reset_self_regulation_each_tick_active),
        disable_self_regulation_decay_active=bool(self_regulation.disable_self_regulation_decay_active),
        self_regulation_runtime_state_available=bool(self_regulation.self_regulation_runtime_state_available),
        self_regulation_inheritable=bool(self_regulation.self_regulation_inheritable),
        self_modification_active=bool(self_modification.self_modification_enabled),
        self_modification_enabled=bool(self_modification.self_modification_enabled),
        self_modification_policy_output_enabled=bool(self_modification.self_modification_policy_output_enabled),
        active_self_modification_target_count=int(self_modification.active_self_modification_target_count),
        potential_self_modification_target_count=int(self_modification.potential_self_modification_target_count),
        active_self_modification_target_mask=int(self_modification.active_self_modification_target_mask),
        active_self_modification_target_ids=tuple(self_modification.active_self_modification_target_ids),
        self_modification_capacity_fraction=float(self_modification.self_modification_capacity_fraction),
        self_modification_cost_hypothetical=float(self_modification.self_modification_cost_hypothetical),
        self_modification_cost_charged=float(self_modification.self_modification_cost_charged),
        self_modification_ablation_active=bool(self_modification.self_modification_ablation_active),
        force_no_self_modification_active=bool(self_modification.force_no_self_modification_active),
        force_all_self_modification_active=bool(self_modification.force_all_self_modification_active),
        force_default_self_modification_active=bool(self_modification.force_default_self_modification_active),
        mask_self_modification_commands_to_neutral_active=bool(self_modification.mask_self_modification_commands_to_neutral_active),
        freeze_self_modification_state_active=bool(self_modification.freeze_self_modification_state_active),
        reset_self_modification_each_tick_active=bool(self_modification.reset_self_modification_each_tick_active),
        reset_self_modification_on_reproduction_active=bool(self_modification.reset_self_modification_on_reproduction_active),
        disable_self_modification_decay_active=bool(self_modification.disable_self_modification_decay_active),
        self_modification_runtime_state_available=bool(self_modification.self_modification_runtime_state_available),
        self_modification_lifetime_persistent=bool(self_modification.self_modification_lifetime_persistent),
        self_modification_inheritable=bool(self_modification.self_modification_inheritable),
        directed_inheritance_enabled=bool(directed_inheritance.directed_inheritance_enabled),
        directed_inheritance_surface_enabled=bool(directed_inheritance.directed_inheritance_surface_enabled),
        directed_inheritance_policy_output_enabled=bool(directed_inheritance.directed_inheritance_policy_output_enabled),
        active_directed_inheritance_channel_count=int(directed_inheritance.active_directed_inheritance_channel_count),
        potential_directed_inheritance_channel_count=int(directed_inheritance.potential_directed_inheritance_channel_count),
        active_directed_inheritance_channel_mask=int(directed_inheritance.active_directed_inheritance_channel_mask),
        active_directed_inheritance_channel_ids=tuple(directed_inheritance.active_directed_inheritance_channel_ids),
        directed_inheritance_capacity_fraction=float(directed_inheritance.directed_inheritance_capacity_fraction),
        directed_inheritance_cost_hypothetical=float(directed_inheritance.directed_inheritance_cost_hypothetical),
        directed_inheritance_cost_charged=float(directed_inheritance.directed_inheritance_cost_charged),
        directed_inheritance_ablation_active=bool(directed_inheritance.directed_inheritance_ablation_active),
        force_no_directed_inheritance_active=bool(directed_inheritance.force_no_directed_inheritance_active),
        force_all_directed_inheritance_active=bool(directed_inheritance.force_all_directed_inheritance_active),
        force_default_directed_inheritance_active=bool(directed_inheritance.force_default_directed_inheritance_active),
        mask_directed_inheritance_commands_to_neutral_active=bool(directed_inheritance.mask_directed_inheritance_commands_to_neutral_active),
        freeze_directed_inheritance_payload_active=bool(directed_inheritance.freeze_directed_inheritance_payload_active),
        reset_directed_inheritance_payload_on_birth_active=bool(directed_inheritance.reset_directed_inheritance_payload_on_birth_active),
        reset_directed_inheritance_payload_on_maturity_active=bool(directed_inheritance.reset_directed_inheritance_payload_on_maturity_active),
        non_genomic_payload_available=bool(directed_inheritance.non_genomic_payload_available),
        non_genomic_payload_applied=bool(directed_inheritance.non_genomic_payload_applied),
        genome_editing_active=bool(self_modification.genome_editing_active or directed_inheritance.genome_editing_active),
        mutation_rate_control_active=bool(directed_inheritance.mutation_rate_control_active),
        recombination_control_active=bool(directed_inheritance.recombination_control_active),
        inherited_self_modification_state_active=bool(directed_inheritance.inherited_self_modification_state_active),
        offspring_developmental_expression_bias=float(directed_summary["offspring_developmental_expression_bias"]),
        offspring_juvenile_cognition_scale_bias=float(directed_summary["offspring_juvenile_cognition_scale_bias"]),
        offspring_memory_expression_bias=float(directed_summary["offspring_memory_expression_bias"]),
        offspring_prediction_lr_bias=float(directed_summary["offspring_prediction_lr_bias"]),
        offspring_exploration_bias_marker=float(directed_summary["offspring_exploration_bias_marker"]),
        offspring_reproduction_threshold_bias=float(directed_summary["offspring_reproduction_threshold_bias"]),
        cognitive_capacity_memory_component=float(memory_component),
        cognitive_capacity_prediction_component=float(prediction_component),
        cognitive_capacity_drive_component=float(drive_component),
        cognitive_capacity_perception_component=float(perception_component),
        cognitive_capacity=float(active_capacity),
        potential_cognitive_capacity=float(potential_capacity),
    )
    _record_envhot_phenotype_bucket(_diag, "phenotype_dataclass_assembly_total", _dataclass_t0)
    _record_envhot_phenotype_bucket(_diag, "phenotype_total", _phenotype_total_t0)
    return phenotype


def express_cost_charge_surface(
    genome: Any,
    *,
    agent: Any = None,
    env_cfg: Any = None,
    sim_cfg: Any = None,
    age_steps: int | None = None,
) -> CostChargeSurface:
    """Build only the scalar inputs needed for env energy charging."""
    if sim_cfg is not None and not bool(_aget(sim_cfg, "enable_cognitive_budget_ledger", True)):
        return CostChargeSurface(
            schema_version=COST_CHARGE_SURFACE_SCHEMA_VERSION,
            sensor_channel_cost_per_tick=0.0,
            sensor_radius_cost_hypothetical=0.0,
            prediction_cost_charged=0.0,
            developmental_cost_charged=0.0,
            maturation_cost_charged=0.0,
            introspection_cost_charged=0.0,
            self_regulation_cost_charged=0.0,
            self_modification_cost_charged=0.0,
        )

    st = _aget(agent, "stats", None)
    if age_steps is None:
        age_steps = _as_int(_aget(st, "survival_steps", 0), 0)

    eff = _developmental_effective(agent, int(age_steps)) if agent is not None else {}
    mem_dim = _snap_memory_dim(_gget(genome, "memory_dim", 0))
    use_gru = _as_bool(_gget(genome, "use_gru", False), False) or mem_dim > 0
    flags = memory_ablation_flags(sim_cfg)
    eff_mem_dim_before_dev = effective_memory_dim_for_genome(genome, sim_cfg=sim_cfg)
    eff_mem_dim = int(eff_mem_dim_before_dev)

    pred_lr = _as_float(eff.get("pred_lr", _gget(genome, "pred_lr", 0.08)), 0.08)
    pred_lr = max(0.0, min(1.0, pred_lr))
    pred_flags = prediction_ablation_flags(sim_cfg)
    pred_costs = prediction_cost_components(genome, sim_cfg=sim_cfg, developmental_effective=eff)
    eff_pred_lr = float(pred_costs["effective_lr"])
    prediction_state_available = bool(pred_costs["state_available"])
    prediction_update_enabled = bool(pred_costs["update_enabled"])

    drive_gain = _as_float(eff.get("drive_logit_gain", _gget(genome, "drive_logit_gain", 1.0)), 1.0)
    module_count = len(list(_gget(genome, "modules", []) or []))
    active_module_count = _as_int(_aget(st, "_dev_active_module_count", module_count), module_count)

    perception = express_perception(genome, env_cfg=env_cfg, sim_cfg=sim_cfg)
    _envhot4_enabled = _envhot4_inactive_defaults_enabled(sim_cfg)

    introspection = _envhot4_try_inactive_introspection_default(sim_cfg) if _envhot4_enabled else None
    if introspection is None:
        introspection = express_introspection(genome, sim_cfg=sim_cfg)

    self_regulation = _envhot4_try_inactive_self_regulation_default(sim_cfg) if _envhot4_enabled else None
    if self_regulation is None:
        self_regulation = express_self_regulation(genome, sim_cfg=sim_cfg)

    self_modification = _envhot4_try_inactive_self_modification_default(sim_cfg) if _envhot4_enabled else None
    if self_modification is None:
        self_modification = express_self_modification(genome, sim_cfg=sim_cfg)

    _raw_directed_payload = _aget(agent, "directed_inheritance_payload", None)
    if (
        _envhot4_enabled
        and not isinstance(_raw_directed_payload, Mapping)
        and not isinstance(_raw_directed_payload, DirectedInheritancePayload)
    ):
        directed_payload = _envhot4_empty_directed_payload_default()
    else:
        directed_payload = DirectedInheritancePayload.from_dict(_raw_directed_payload)
    directed_inheritance = _envhot4_try_inactive_directed_inheritance_default(sim_cfg) if _envhot4_enabled else None
    if directed_inheritance is None:
        directed_inheritance = express_directed_inheritance(genome, payload=directed_payload, sim_cfg=sim_cfg)

    dev_state = developmental_expression_state(age_steps=int(age_steps), sim_cfg=sim_cfg)
    dev_factor = float(dev_state["factor"])
    sensor_radius_before_dev = int(perception.active_sensor_radius)
    active_channel_count = int(perception.active_sensor_channel_count)
    active_channel_mask = int(perception.active_sensor_channel_mask)
    active_radius = int(perception.active_sensor_radius)
    observed_cells = int(perception.active_observed_cell_count)
    cell_channel_capacity = float(perception.cell_channel_capacity)
    perception_capacity_component = float(perception.perception_capacity_component)
    sensor_radius_hyp_cost = float(perception.sensor_radius_cost_hypothetical)
    sensor_channel_cost = float(perception.sensor_channel_cost_per_tick)
    pred_lr_before_dev = float(eff_pred_lr)

    if bool(dev_state["enabled"]):
        eff_mem_dim = _scale_memory_dim(eff_mem_dim_before_dev, dev_factor)
        active_radius = _scale_sensor_radius(sensor_radius_before_dev, dev_factor)
        _active_channel_mask, active_channel_ids = _scale_sensor_channel_mask(
            active_channel_mask,
            active_channel_count,
            dev_factor,
        )
        active_channel_count = int(len(active_channel_ids))
        observed_cells = int((2 * int(active_radius) + 1) ** 2)
        cell_channel_capacity = float(observed_cells * active_channel_count)
        perception_capacity_component = float(cell_channel_capacity)
        eff_pred_lr = float(eff_pred_lr) * dev_factor
        prediction_update_enabled = bool(prediction_state_available and eff_pred_lr > 0.0 and not pred_flags["force_frozen"])
        sensor_channel_cost = float(sensor_channel_cost) * dev_factor
        sensor_radius_hyp_cost = float(sensor_radius_hyp_cost) * dev_factor

    pred_base = _as_float(_aget(sim_cfg, "prediction_cost_base", 0.0), 0.0)
    pred_per_lr = _as_float(_aget(sim_cfg, "prediction_cost_per_lr", 0.001), 0.001)
    pred_state_cost = _as_float(_aget(sim_cfg, "prediction_state_cost", 0.0), 0.0)
    pred_update_hyp = max(0.0, float(pred_base) + float(pred_per_lr) * float(eff_pred_lr)) if prediction_update_enabled else 0.0
    pred_state_hyp = max(0.0, float(pred_state_cost)) if prediction_state_available else 0.0
    pred_hyp_total = float(pred_update_hyp + pred_state_hyp)
    pred_charged_total = float(pred_hyp_total if _as_bool(_aget(sim_cfg, "enable_experimental_prediction_cost_charging", False), False) else 0.0)

    memory_component = float(eff_mem_dim)
    prediction_enabled = bool(prediction_state_available and eff_pred_lr > 0.0)
    prediction_component = float(eff_pred_lr if prediction_enabled else 0.0)
    drive_component = float(max(0.0, drive_gain))
    perception_component = (
        float(perception_capacity_component)
        if bool(perception.perception_masking_enabled or perception.sensor_radius_masking_enabled)
        else 0.0
    )
    introspection_component = (
        float(introspection.active_introspection_channel_count)
        if bool(introspection.introspection_enabled or introspection.introspection_ablation_active)
        else 0.0
    )
    self_regulation_component = (
        float(self_regulation.active_self_regulation_control_count)
        if bool(self_regulation.self_regulation_enabled or self_regulation.self_regulation_ablation_active)
        else 0.0
    )
    self_modification_component = (
        float(self_modification.active_self_modification_target_count)
        if bool(self_modification.self_modification_enabled or self_modification.self_modification_ablation_active)
        else 0.0
    )
    directed_inheritance_component = (
        float(directed_inheritance.active_directed_inheritance_channel_count)
        if bool(directed_inheritance.directed_inheritance_enabled or directed_inheritance.directed_inheritance_ablation_active)
        else 0.0
    )
    perception_component_before = (
        float(perception.perception_capacity_component)
        if bool(perception.perception_masking_enabled or perception.sensor_radius_masking_enabled)
        else 0.0
    )
    before_capacity = float(eff_mem_dim_before_dev)
    if bool(use_gru and eff_mem_dim_before_dev > 0):
        before_capacity += 1.0
    before_capacity += float(pred_lr_before_dev if prediction_state_available and pred_lr_before_dev > 0.0 else 0.0)
    before_capacity += float(drive_component)
    before_capacity += float(max(0, active_module_count))
    before_capacity += float(perception_component_before)

    active_capacity = float(memory_component)
    if bool(use_gru and eff_mem_dim > 0):
        active_capacity += 1.0
    active_capacity += float(prediction_component)
    active_capacity += float(drive_component)
    active_capacity += float(max(0, active_module_count))
    active_capacity += float(perception_component)
    active_capacity += float(introspection_component)
    active_capacity += float(self_regulation_component)
    active_capacity += float(self_modification_component)
    active_capacity += float(directed_inheritance_component)

    dev_costs = developmental_cost_components(
        before_capacity=float(before_capacity),
        after_capacity=float(active_capacity),
        state=dev_state,
        sim_cfg=sim_cfg,
    )

    return CostChargeSurface(
        schema_version=COST_CHARGE_SURFACE_SCHEMA_VERSION,
        sensor_channel_cost_per_tick=float(sensor_channel_cost),
        sensor_radius_cost_hypothetical=float(sensor_radius_hyp_cost),
        prediction_cost_charged=float(pred_charged_total),
        developmental_cost_charged=float(dev_costs["developmental"] if bool(dev_costs["charging_enabled"]) else 0.0),
        maturation_cost_charged=float(dev_costs["maturation"] if bool(dev_costs["charging_enabled"]) else 0.0),
        introspection_cost_charged=float(introspection.introspection_cost_charged),
        self_regulation_cost_charged=float(self_regulation.self_regulation_cost_charged),
        self_modification_cost_charged=float(self_modification.self_modification_cost_charged),
    )


def compute_cognitive_budget_ledger(
    phenotype: CognitivePhenotype,
    *,
    agent: Any = None,
    env_cfg: Any = None,
    sim_cfg: Any = None,
    _aggregation: _CognitiveBudgetCostAggregation | None = None,
) -> CognitiveBudgetLedger:
    if sim_cfg is not None and not bool(_aget(sim_cfg, "enable_cognitive_budget_ledger", True)):
        return CognitiveBudgetLedger(
            schema_version=BUDGET_LEDGER_SCHEMA_VERSION,
            cognitive_upkeep_total=0.0,
            memory_upkeep_cost=0.0,
            memory_upkeep_cost_hypothetical=0.0,
            memory_upkeep_cost_charged=0.0,
            memory_cost_per_tick=0.0,
            memory_cost_total_component=0.0,
            memory_cost_fraction_of_cog_cost=0.0,
            memory_capacity_component=0.0,
            effective_memory_capacity_component=0.0,
            recurrent_policy_memory_component=0.0,
            memory_cost_charging_enabled=False,
            memory_cost_telemetry_only=False,
            policy_memory_cost=0.0,
            prediction_update_cost=0.0,
            prediction_update_cost_hypothetical=0.0,
            prediction_update_cost_charged=0.0,
            prediction_state_cost_hypothetical=0.0,
            prediction_state_cost_charged=0.0,
            prediction_cost_per_tick=0.0,
            prediction_cost_total_component=0.0,
            prediction_cost_fraction_of_cog_cost=0.0,
            prediction_capacity_component=0.0,
            prediction_cost_charging_enabled=False,
            prediction_cost_telemetry_only=True,
            developmental_cognition_cost_hypothetical=0.0,
            developmental_cognition_cost_charged=0.0,
            maturation_cost_hypothetical=0.0,
            maturation_cost_charged=0.0,
            juvenile_cognitive_cost_meaning="disabled",
            adult_cognitive_cost_meaning="disabled",
            cognitive_growth_cost=0.0,
            developmental_cost_fraction_of_cog_cost=0.0,
            developmental_cost_charging_enabled=False,
            developmental_cost_telemetry_only=True,
            introspection_channel_cost_hypothetical=0.0,
            introspection_channel_cost_charged=0.0,
            introspection_observation_cost_hypothetical=0.0,
            introspection_observation_cost_charged=0.0,
            introspection_cost_per_tick=0.0,
            introspection_cost_total_component=0.0,
            introspection_cost_fraction_of_cog_cost=0.0,
            introspection_capacity_component=0.0,
            introspection_cost_charging_enabled=False,
            introspection_cost_telemetry_only=True,
            self_regulation_control_cost_hypothetical=0.0,
            self_regulation_control_cost_charged=0.0,
            self_regulation_update_cost_hypothetical=0.0,
            self_regulation_update_cost_charged=0.0,
            self_regulation_maintenance_cost_hypothetical=0.0,
            self_regulation_maintenance_cost_charged=0.0,
            self_regulation_cost_per_tick=0.0,
            self_regulation_cost_total_component=0.0,
            self_regulation_cost_fraction_of_cog_cost=0.0,
            self_regulation_capacity_component=0.0,
            self_regulation_cost_charging_enabled=False,
            self_regulation_cost_telemetry_only=True,
            self_modification_command_cost_hypothetical=0.0,
            self_modification_command_cost_charged=0.0,
            self_modification_update_cost_hypothetical=0.0,
            self_modification_update_cost_charged=0.0,
            self_modification_maintenance_cost_hypothetical=0.0,
            self_modification_maintenance_cost_charged=0.0,
            self_modification_cost_per_tick=0.0,
            self_modification_cost_total_component=0.0,
            self_modification_cost_fraction_of_cog_cost=0.0,
            self_modification_capacity_component=0.0,
            self_modification_cost_charging_enabled=False,
            self_modification_cost_telemetry_only=True,
            directed_inheritance_command_cost_hypothetical=0.0,
            directed_inheritance_command_cost_charged=0.0,
            directed_inheritance_payload_cost_hypothetical=0.0,
            directed_inheritance_payload_cost_charged=0.0,
            directed_inheritance_parent_cost_hypothetical=0.0,
            directed_inheritance_parent_cost_charged=0.0,
            directed_inheritance_offspring_burden_hypothetical=0.0,
            directed_inheritance_offspring_burden_charged=0.0,
            directed_inheritance_cost_per_reproduction=0.0,
            directed_inheritance_cost_total_component=0.0,
            directed_inheritance_cost_fraction_of_cog_cost=0.0,
            directed_inheritance_capacity_component=0.0,
            directed_inheritance_cost_charging_enabled=False,
            directed_inheritance_cost_telemetry_only=True,
            sensor_observation_cost=0.0,
            drive_regulation_cost=0.0,
            learning_or_plasticity_cost=0.0,
            future_self_regulation_cost=0.0,
            sensor_channel_cost=0.0,
            perception_cost_total=0.0,
            perception_cost_hypothetical=0.0,
            perception_cost_charged=0.0,
            sensor_radius_cost_hypothetical=0.0,
            sensor_radius_cost_charged=0.0,
            spatial_perception_cost=0.0,
            active_sensor_channel_count=0,
            potential_sensor_channel_count=0,
            active_sensor_radius=0,
            active_observed_cell_count=0,
            cell_channel_capacity=0.0,
            total_active_cognitive_capacity=float(phenotype.cognitive_capacity),
            total_potential_cognitive_capacity=float(phenotype.potential_cognitive_capacity),
            cognitive_cost_per_tick=0.0,
            cognitive_cost_fraction_of_energy_budget=0.0,
            memory_status="unavailable",
            policy_memory_status="unavailable",
            prediction_status="unavailable",
            sensor_status="unavailable",
            drive_regulation_status="unavailable",
            learning_status="unavailable",
            introspection_status="unavailable",
            self_regulation_status="unavailable",
            self_modification_status="unavailable",
            future_self_regulation_status="future_reserved",
            experimental_charge_enabled=False,
            charged_energy_delta=0.0,
        )

    memory_hyp_cost = float(phenotype.memory_cost_hypothetical)
    memory_charged_cost = float(phenotype.memory_cost_charged)

    prediction_hyp_cost = float(phenotype.prediction_cost_hypothetical)
    prediction_charged_cost = float(phenotype.prediction_cost_charged)
    developmental_hyp_cost = float(phenotype.developmental_cost_hypothetical)
    maturation_hyp_cost = float(phenotype.maturation_cost_hypothetical)
    developmental_charged_cost = float(phenotype.developmental_cost_charged)
    maturation_charged_cost = float(phenotype.maturation_cost_charged)
    introspection_hyp_cost = float(phenotype.introspection_cost_hypothetical)
    introspection_charged_cost = float(phenotype.introspection_cost_charged)
    self_regulation_hyp_cost = float(phenotype.self_regulation_cost_hypothetical)
    self_regulation_charged_cost = float(phenotype.self_regulation_cost_charged)
    self_modification_hyp_cost = float(phenotype.self_modification_cost_hypothetical)
    self_modification_charged_cost = float(phenotype.self_modification_cost_charged)
    directed_hyp_cost = float(phenotype.directed_inheritance_cost_hypothetical)
    directed_charged_cost = float(phenotype.directed_inheritance_cost_charged)

    components = _cognitive_budget_cost_components(phenotype, sim_cfg=sim_cfg)
    if _aggregation is None:
        energy_budget = _as_float(_aget(agent, "energy", 0.0), 0.0)
        aggregation = _python_budget_cost_aggregation(components, energy_budget=energy_budget)
    else:
        aggregation = _aggregation
    if len(aggregation.fractions) != _BUDGET_COST_COMPONENT_COLUMNS:
        raise ValueError("budget aggregation must have exactly 8 fractions")
    memory_cost = float(components[0])
    prediction_update_cost = float(components[1])
    perception_charged_cost = float(components[2])
    sensor_cost = float(perception_charged_cost)
    developmental_cost = float(components[3])
    introspection_cost = float(components[4])
    self_regulation_cost = float(components[5])
    self_modification_cost = float(components[6])
    directed_cost = float(components[7])
    drive_cost = 0.0
    learning_cost = 0.0
    future_cost = 0.0
    total = float(aggregation.total)
    frac = float(aggregation.energy_fraction)
    component_fractions = tuple(float(value) for value in aggregation.fractions)

    experimental_charging = bool(_aget(sim_cfg, "enable_experimental_cognitive_cost_charging", False))
    charge_channels = bool(_aget(sim_cfg, "enable_experimental_perception_cost_charging", False))
    charge_radius = bool(_aget(sim_cfg, "enable_experimental_sensor_radius_cost_charging", False))
    perception_charging = bool(charge_channels or charge_radius)
    prediction_charging = bool(_as_bool(_aget(sim_cfg, "enable_experimental_prediction_cost_charging", False), False))
    developmental_charging = bool(_as_bool(_aget(sim_cfg, "enable_experimental_developmental_cost_charging", False), False))
    introspection_charging = bool(_as_bool(_aget(sim_cfg, "enable_experimental_introspection_cost_charging", False), False))
    self_regulation_charging = bool(_as_bool(_aget(sim_cfg, "enable_experimental_self_regulation_cost_charging", False), False))
    self_modification_charging = bool(_as_bool(_aget(sim_cfg, "enable_experimental_self_modification_cost_charging", False), False))
    directed_inheritance_charging = bool(_as_bool(_aget(sim_cfg, "enable_experimental_directed_inheritance_cost_charging", False), False))

    return CognitiveBudgetLedger(
        schema_version=BUDGET_LEDGER_SCHEMA_VERSION,
        cognitive_upkeep_total=float(total),
        memory_upkeep_cost=float(memory_cost),
        memory_upkeep_cost_hypothetical=float(memory_hyp_cost),
        memory_upkeep_cost_charged=float(memory_charged_cost),
        memory_cost_per_tick=float(memory_charged_cost),
        memory_cost_total_component=float(memory_charged_cost),
        memory_cost_fraction_of_cog_cost=float(component_fractions[0]),
        memory_capacity_component=float(phenotype.active_memory_dim),
        effective_memory_capacity_component=float(phenotype.effective_memory_dim),
        recurrent_policy_memory_component=1.0 if phenotype.recurrent_state_enabled else 0.0,
        memory_cost_charging_enabled=bool(memory_charged_cost > 0.0 or _as_bool(_aget(sim_cfg, "enable_experimental_memory_cost_charging", False), False)),
        memory_cost_telemetry_only=False,
        policy_memory_cost=float(memory_cost),
        prediction_update_cost=float(prediction_update_cost),
        prediction_update_cost_hypothetical=float(prediction_hyp_cost),
        prediction_update_cost_charged=float(prediction_charged_cost),
        prediction_state_cost_hypothetical=0.0,
        prediction_state_cost_charged=0.0,
        prediction_cost_per_tick=float(prediction_update_cost),
        prediction_cost_total_component=float(prediction_update_cost),
        prediction_cost_fraction_of_cog_cost=float(component_fractions[1]),
        prediction_capacity_component=float(phenotype.prediction_capacity),
        prediction_cost_charging_enabled=bool(prediction_charging),
        prediction_cost_telemetry_only=not bool(prediction_charging),
        developmental_cognition_cost_hypothetical=float(developmental_hyp_cost),
        developmental_cognition_cost_charged=float(developmental_charged_cost),
        maturation_cost_hypothetical=float(maturation_hyp_cost),
        maturation_cost_charged=float(maturation_charged_cost),
        juvenile_cognitive_cost_meaning="telemetry_only_when_not_charged",
        adult_cognitive_cost_meaning="full_expression_cost_when_developmental_factor_reaches_adult_scale",
        cognitive_growth_cost=float(developmental_hyp_cost),
        developmental_cost_fraction_of_cog_cost=float(component_fractions[3]),
        developmental_cost_charging_enabled=bool(developmental_charging),
        developmental_cost_telemetry_only=not bool(developmental_charging),
        introspection_channel_cost_hypothetical=float(introspection_hyp_cost),
        introspection_channel_cost_charged=float(introspection_charged_cost),
        introspection_observation_cost_hypothetical=float(introspection_hyp_cost),
        introspection_observation_cost_charged=float(introspection_charged_cost),
        introspection_cost_per_tick=float(introspection_charged_cost),
        introspection_cost_total_component=float(introspection_charged_cost),
        introspection_cost_fraction_of_cog_cost=float(component_fractions[4]),
        introspection_capacity_component=float(phenotype.active_introspection_channel_count),
        introspection_cost_charging_enabled=bool(introspection_charging),
        introspection_cost_telemetry_only=not bool(introspection_charging),
        self_regulation_control_cost_hypothetical=float(self_regulation_hyp_cost),
        self_regulation_control_cost_charged=float(self_regulation_charged_cost),
        self_regulation_update_cost_hypothetical=0.0,
        self_regulation_update_cost_charged=0.0,
        self_regulation_maintenance_cost_hypothetical=float(self_regulation_hyp_cost),
        self_regulation_maintenance_cost_charged=float(self_regulation_charged_cost),
        self_regulation_cost_per_tick=float(self_regulation_charged_cost),
        self_regulation_cost_total_component=float(self_regulation_charged_cost),
        self_regulation_cost_fraction_of_cog_cost=float(component_fractions[5]),
        self_regulation_capacity_component=float(phenotype.active_self_regulation_control_count),
        self_regulation_cost_charging_enabled=bool(self_regulation_charging),
        self_regulation_cost_telemetry_only=not bool(self_regulation_charging),
        self_modification_command_cost_hypothetical=float(self_modification_hyp_cost),
        self_modification_command_cost_charged=float(self_modification_charged_cost),
        self_modification_update_cost_hypothetical=0.0,
        self_modification_update_cost_charged=0.0,
        self_modification_maintenance_cost_hypothetical=float(self_modification_hyp_cost),
        self_modification_maintenance_cost_charged=float(self_modification_charged_cost),
        self_modification_cost_per_tick=float(self_modification_charged_cost),
        self_modification_cost_total_component=float(self_modification_charged_cost),
        self_modification_cost_fraction_of_cog_cost=float(component_fractions[6]),
        self_modification_capacity_component=float(phenotype.active_self_modification_target_count),
        self_modification_cost_charging_enabled=bool(self_modification_charging),
        self_modification_cost_telemetry_only=not bool(self_modification_charging),
        directed_inheritance_command_cost_hypothetical=float(directed_hyp_cost),
        directed_inheritance_command_cost_charged=float(directed_charged_cost),
        directed_inheritance_payload_cost_hypothetical=float(directed_hyp_cost),
        directed_inheritance_payload_cost_charged=float(directed_charged_cost),
        directed_inheritance_parent_cost_hypothetical=float(directed_hyp_cost),
        directed_inheritance_parent_cost_charged=float(directed_charged_cost),
        directed_inheritance_offspring_burden_hypothetical=0.0,
        directed_inheritance_offspring_burden_charged=0.0,
        directed_inheritance_cost_per_reproduction=float(directed_charged_cost),
        directed_inheritance_cost_total_component=float(directed_charged_cost),
        directed_inheritance_cost_fraction_of_cog_cost=float(component_fractions[7]),
        directed_inheritance_capacity_component=float(phenotype.active_directed_inheritance_channel_count),
        directed_inheritance_cost_charging_enabled=bool(directed_inheritance_charging),
        directed_inheritance_cost_telemetry_only=not bool(directed_inheritance_charging),
        sensor_observation_cost=float(sensor_cost),
        drive_regulation_cost=float(drive_cost),
        learning_or_plasticity_cost=float(learning_cost),
        future_self_regulation_cost=float(future_cost),
        sensor_channel_cost=float(phenotype.sensor_channel_cost_per_tick),
        perception_cost_total=float(sensor_cost),
        perception_cost_hypothetical=float(phenotype.perception_cost_hypothetical),
        perception_cost_charged=float(perception_charged_cost),
        sensor_radius_cost_hypothetical=float(phenotype.sensor_radius_cost_hypothetical),
        sensor_radius_cost_charged=float(phenotype.sensor_radius_cost_charged),
        spatial_perception_cost=float(phenotype.sensor_radius_cost_hypothetical),
        active_sensor_channel_count=int(phenotype.active_sensor_channel_count),
        potential_sensor_channel_count=int(phenotype.potential_sensor_channel_count),
        active_sensor_radius=int(phenotype.active_sensor_radius),
        active_observed_cell_count=int(phenotype.active_observed_cell_count),
        cell_channel_capacity=float(phenotype.cell_channel_capacity),
        total_active_cognitive_capacity=float(phenotype.cognitive_capacity),
        total_potential_cognitive_capacity=float(phenotype.potential_cognitive_capacity),
        cognitive_cost_per_tick=float(total),
        cognitive_cost_fraction_of_energy_budget=float(frac),
        memory_status="active" if phenotype.memory_enabled else "inactive",
        policy_memory_status="active" if phenotype.policy_memory_enabled else "inactive",
        prediction_status="active" if phenotype.active_prediction_capability else "inactive",
        sensor_status="active" if phenotype.perception_enabled else ("fixed" if phenotype.sensor_observation_capacity > 0 else "unavailable"),
        drive_regulation_status="active",
        learning_status="active" if phenotype.effective_prediction_learning_rate > 0.0 else "inactive",
        introspection_status="active" if phenotype.introspection_enabled else ("ablated" if phenotype.introspection_ablation_active else "inactive"),
        self_regulation_status="active" if phenotype.self_regulation_enabled else ("ablated" if phenotype.self_regulation_ablation_active else "inactive"),
        self_modification_status="active" if phenotype.self_modification_enabled else ("ablated" if phenotype.self_modification_ablation_active else "inactive"),
        future_self_regulation_status="future_reserved",
        experimental_charge_enabled=bool(
            experimental_charging
            or perception_charging
            or prediction_charging
            or developmental_charging
            or introspection_charging
            or self_regulation_charging
            or self_modification_charging
            or directed_inheritance_charging
        ),
        charged_energy_delta=float(
            total
            if (
                experimental_charging
                or perception_charging
                or prediction_charging
                or developmental_charging
                or introspection_charging
                or self_regulation_charging
                or self_modification_charging
                or directed_inheritance_charging
            )
            else 0.0
        ),
    )


def compute_cognitive_budget_ledgers_batch(
    phenotypes: Sequence[CognitivePhenotype],
    *,
    agents: Sequence[Any],
    env_cfg: Any = None,
    sim_cfg: Any = None,
) -> tuple[list[CognitiveBudgetLedger], dict[str, object]]:
    """Build ledgers using one optional native-style aggregation call."""
    if len(phenotypes) != len(agents):
        raise ValueError("phenotype count must match agent count")

    component_rows = [_cognitive_budget_cost_components(phenotype, sim_cfg=sim_cfg) for phenotype in phenotypes]
    energy_budgets = [_as_float(_aget(agent, "energy", 0.0), 0.0) for agent in agents]
    aggregations, info = _native_budget_cost_aggregation_batch(component_rows, energy_budgets)

    mismatch_count = 0
    if bool(_aget(sim_cfg, "native_cognitive_budget_cost_aggregation_mismatch_guard", False)):
        guarded: list[_CognitiveBudgetCostAggregation] = []
        for components, energy_budget, aggregation in zip(component_rows, energy_budgets, aggregations):
            expected = _python_budget_cost_aggregation(components, energy_budget=float(energy_budget))
            if (
                float(expected.total) != float(aggregation.total)
                or float(expected.energy_fraction) != float(aggregation.energy_fraction)
                or tuple(float(v) for v in expected.fractions) != tuple(float(v) for v in aggregation.fractions)
            ):
                mismatch_count += 1
                guarded.append(expected)
            else:
                guarded.append(aggregation)
        aggregations = guarded

    ledgers = [
        compute_cognitive_budget_ledger(
            phenotype,
            agent=agent,
            env_cfg=env_cfg,
            sim_cfg=sim_cfg,
            _aggregation=aggregation,
        )
        for phenotype, agent, aggregation in zip(phenotypes, agents, aggregations)
    ]
    info = dict(info)
    info["mismatch_count"] = int(mismatch_count)
    if mismatch_count:
        info["fallback_count"] = int(info.get("fallback_count", 0)) + int(mismatch_count)
    return ledgers, info


def compute_env_apply_cognitive_cost_scalar(
    phenotype: CognitivePhenotype,
    *,
    agent: Any = None,
    env_cfg: Any = None,
    sim_cfg: Any = None,
) -> float:
    """Return the existing env-apply charged cost sum without ledger assembly."""

    _ = agent, env_cfg
    _diag = _envhot_current_diagnostics()
    _scalar_t0 = time.perf_counter()
    try:
        if sim_cfg is not None and not bool(_aget(sim_cfg, "enable_cognitive_budget_ledger", True)):
            _record_envhot_phenotype_inactive(_diag, "budget_ledger_disabled")
            return 0.0

        charge_channels = bool(_aget(sim_cfg, "enable_experimental_perception_cost_charging", False))
        charge_radius = bool(_aget(sim_cfg, "enable_experimental_sensor_radius_cost_charging", False))
        perception_charged_cost = float(
            (float(phenotype.sensor_channel_cost_per_tick) if charge_channels else 0.0)
            + (float(phenotype.sensor_radius_cost_hypothetical) if charge_radius else 0.0)
        )
        return float(
            perception_charged_cost
            + float(phenotype.prediction_cost_charged)
            + float(phenotype.developmental_cost_charged)
            + float(phenotype.maturation_cost_charged)
            + float(phenotype.introspection_cost_charged)
            + float(phenotype.self_regulation_cost_charged)
            + float(phenotype.self_modification_cost_charged)
        )
    finally:
        _record_envhot_phenotype_bucket(_diag, "phenotype_budget_ledger_scalar_fastpath_total", _scalar_t0)


def express_and_ledger_for_agent(agent: Any, *, env_cfg: Any = None, sim_cfg: Any = None) -> tuple[CognitivePhenotype, CognitiveBudgetLedger]:
    g = _aget(agent, "genome", None)
    phenotype = express_cognitive_phenotype(g, agent=agent, env_cfg=env_cfg, sim_cfg=sim_cfg)
    ledger = compute_cognitive_budget_ledger(phenotype, agent=agent, env_cfg=env_cfg, sim_cfg=sim_cfg)
    return phenotype, ledger
