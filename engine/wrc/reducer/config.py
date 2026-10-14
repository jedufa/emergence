"""Configuration and frozen vocabulary for the general WRC reducer.

The reducer is intentionally source-neutral.  The same configuration is used
when observations arrive live from a simulator and when a completed metrics or
Chronicle source is replayed later.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

from .canonical import canonical_sha256


REDUCER_VERSION = "eiel_wrc_reducer_v1_1"
CONFIG_SCHEMA_VERSION = "eiel_wrc_reducer_config_v1"
WORLD_RESULT_SCHEMA_VERSION = "eiel_wrc_world_reduction_v1"
PARAMETER_RESULT_SCHEMA_VERSION = "eiel_wrc_parameter_reduction_v1"
ATLAS_RESULT_SCHEMA_VERSION = "eiel_wrc_atlas_reduction_v1"
STATE_SCHEMA_VERSION = "eiel_wrc_reducer_state_v1"

REDUCTION_STAGES = (
    "anchors",
    "local_change_and_turning_points",
    "cumulative_and_baseline_drift",
    "extrema_outliers_and_trajectory_summary",
)

RESPONSE_SHAPES = (
    "BOOLEAN_CONTRAST",
    "FLAT",
    "MONOTONIC_POSITIVE",
    "MONOTONIC_NEGATIVE",
    "THRESHOLD_POSITIVE",
    "THRESHOLD_NEGATIVE",
    "SATURATING_POSITIVE",
    "SATURATING_NEGATIVE",
    "U_SHAPED",
    "INVERTED_U",
    "NON_MONOTONIC",
    "CATASTROPHIC_THRESHOLD_NEGATIVE",
    "NOISY_UNRESOLVED",
)

SCREENING_CLASSIFICATIONS = (
    "ACTIVE",
    "WEAK",
    "NO_EFFECT_DETECTED",
    "UNRESOLVED",
    "INCOMPLETE",
)


@dataclass(frozen=True, slots=True)
class ReducerConfig:
    """Deterministic thresholds and bounded candidate budgets.

    Local changes catch abrupt events.  Cumulative drift compares the current
    value with the last drift anchor, so repeated sub-threshold changes are not
    lost.  Baseline drift always compares with the first value and therefore
    captures slow secular movement even after intermediate anchors are reset.
    """

    schema_version: str = CONFIG_SCHEMA_VERSION
    local_relative_threshold: float = 0.02
    local_absolute_threshold: float = 0.0
    stable_relative_threshold: float = 0.002
    stable_absolute_threshold: float = 0.0
    cumulative_relative_threshold: float = 0.05
    cumulative_absolute_threshold: float = 0.0
    baseline_relative_threshold: float = 0.10
    baseline_absolute_threshold: float = 0.0
    outlier_z_threshold: float = 3.5
    outlier_min_samples: int = 5
    scale_floor: float = 1.0e-12
    zero_epsilon: float = 1.0e-12
    local_candidate_budget: int = 8
    turning_candidate_budget: int = 6
    cumulative_candidate_budget: int = 8
    baseline_candidate_budget: int = 6
    outlier_candidate_budget: int = 6
    stable_span_budget: int = 8
    max_selected_points_per_metric: int = 24
    max_ranked_metrics_per_parameter: int = 160
    max_detail_metrics_per_parameter: int = 24
    minimum_metric_world_coverage: float = 0.75
    scientific_materiality_floor: float = 1.0e-3
    scientific_seed_target: int = 8
    flat_relative_range_threshold: float = 0.01
    flat_absolute_range_threshold: float = 1.0e-12
    noisy_signal_to_noise_threshold: float = 0.35
    monotonic_tolerance_fraction: float = 0.10
    threshold_step_fraction: float = 0.60
    saturation_tail_fraction: float = 0.25
    catastrophic_relative_drop: float = 0.50
    metric_include_patterns: tuple[str, ...] = field(default_factory=tuple)
    metric_exclude_patterns: tuple[str, ...] = field(
        default_factory=lambda: (
            r"(^|_)json$",
            r"_json$",
            r"_sha256$",
            r"(^|_)id$",
            r"(^|_)path$",
        )
    )

    def __post_init__(self) -> None:
        if self.schema_version != CONFIG_SCHEMA_VERSION:
            raise ValueError("unsupported WRC reducer configuration schema")
        for name in (
            "local_relative_threshold",
            "local_absolute_threshold",
            "stable_relative_threshold",
            "stable_absolute_threshold",
            "cumulative_relative_threshold",
            "cumulative_absolute_threshold",
            "baseline_relative_threshold",
            "baseline_absolute_threshold",
            "outlier_z_threshold",
            "scale_floor",
            "zero_epsilon",
            "minimum_metric_world_coverage",
            "scientific_materiality_floor",
            "flat_relative_range_threshold",
            "flat_absolute_range_threshold",
            "noisy_signal_to_noise_threshold",
            "monotonic_tolerance_fraction",
            "threshold_step_fraction",
            "saturation_tail_fraction",
            "catastrophic_relative_drop",
        ):
            value = float(getattr(self, name))
            if value < 0.0:
                raise ValueError(f"{name} must be nonnegative")
        if not 0.0 <= self.minimum_metric_world_coverage <= 1.0:
            raise ValueError("minimum_metric_world_coverage must be in [0, 1]")
        for name in (
            "outlier_min_samples",
            "local_candidate_budget",
            "turning_candidate_budget",
            "cumulative_candidate_budget",
            "baseline_candidate_budget",
            "outlier_candidate_budget",
            "stable_span_budget",
            "max_selected_points_per_metric",
            "max_ranked_metrics_per_parameter",
            "max_detail_metrics_per_parameter",
            "scientific_seed_target",
        ):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be positive")

    def payload(self) -> dict[str, Any]:
        value = asdict(self)
        value["metric_include_patterns"] = list(self.metric_include_patterns)
        value["metric_exclude_patterns"] = list(self.metric_exclude_patterns)
        return value

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.payload())

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ReducerConfig":
        data = dict(value)
        for name in ("metric_include_patterns", "metric_exclude_patterns"):
            if name in data:
                raw = data[name]
                if not isinstance(raw, (list, tuple)) or any(
                    not isinstance(item, str) for item in raw
                ):
                    raise ValueError(f"{name} must be an array of strings")
                data[name] = tuple(raw)
        return cls(**data)


def reducer_schema(config: ReducerConfig) -> dict[str, Any]:
    """Return a compact machine-readable explanation of reducer semantics."""

    return {
        "schema_version": "eiel_wrc_reducer_schema_v1",
        "reducer_version": REDUCER_VERSION,
        "config": config.payload(),
        "config_sha256": config.sha256,
        "stages": [
            {
                "ordinal": 1,
                "name": REDUCTION_STAGES[0],
                "purpose": "always preserve first and terminal observations",
            },
            {
                "ordinal": 2,
                "name": REDUCTION_STAGES[1],
                "purpose": "rank abrupt changes and local extrema",
            },
            {
                "ordinal": 3,
                "name": REDUCTION_STAGES[2],
                "purpose": (
                    "retain accumulated drift even when every individual step "
                    "is below the local threshold"
                ),
            },
            {
                "ordinal": 4,
                "name": REDUCTION_STAGES[3],
                "purpose": (
                    "retain extrema and online statistical surprises while "
                    "summarizing stable spans, trend, variation and integral"
                ),
            },
        ],
        "response_shapes": list(RESPONSE_SHAPES),
        "screening_classifications": list(SCREENING_CLASSIFICATIONS),
        "limitations": [
            "Screening classifications are descriptive, not decision authority.",
            "Metrics CSV is generation-boundary data unless the provider declares another cadence.",
            "Chronicle generation summaries expose a compact subset of full metrics.",
            "Raw sources remain authoritative and are referenced by drill-back locators.",
        ],
    }


__all__ = [
    "ATLAS_RESULT_SCHEMA_VERSION",
    "CONFIG_SCHEMA_VERSION",
    "PARAMETER_RESULT_SCHEMA_VERSION",
    "REDUCER_VERSION",
    "REDUCTION_STAGES",
    "RESPONSE_SHAPES",
    "ReducerConfig",
    "SCREENING_CLASSIFICATIONS",
    "STATE_SCHEMA_VERSION",
    "WORLD_RESULT_SCHEMA_VERSION",
    "reducer_schema",
]
