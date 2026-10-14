"""Typed contracts for the deterministic WRC reducer.

The reducer deliberately admits a provider-neutral observation stream.  Live
simulation callbacks, metrics CSV replay, Chronicle replay, and archived
material can therefore use the same reduction core rather than maintaining
parallel formulas.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import fnmatch
import hashlib
import json
import math
from typing import Any, Mapping, Sequence


REDUCER_CONFIG_SCHEMA_V1 = "eiel_wrc_reducer_config_v1"
OBSERVATION_SCHEMA_V1 = "eiel_wrc_observation_v1"
WORLD_SUMMARY_SCHEMA_V1 = "eiel_wrc_world_summary_v1"
PARAMETER_SUMMARY_SCHEMA_V1 = "eiel_wrc_parameter_summary_v1"
ATLAS_SUMMARY_SCHEMA_V1 = "eiel_wrc_atlas_summary_v1"
REDUCER_STATE_SCHEMA_V1 = "eiel_wrc_reducer_state_v1"


class WrcReducerError(ValueError):
    """Stable fail-closed reducer error."""


def canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise WrcReducerError("value is not canonical JSON data") from exc


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def pretty_json(value: Any) -> str:
    return json.dumps(
        value,
        indent=2,
        sort_keys=True,
        ensure_ascii=True,
        allow_nan=False,
    ) + "\n"


Scalar = bool | int | float | str | None


def finite_number(value: Any, *, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise WrcReducerError(f"{path} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise WrcReducerError(f"{path} must be finite")
    return result


def canonical_level_key(value: Scalar) -> str:
    """Return a type-preserving deterministic level identity."""

    if value is None:
        return "null:null"
    if isinstance(value, bool):
        return f"bool:{str(value).lower()}"
    if isinstance(value, int):
        return f"int:{value}"
    if isinstance(value, float):
        if not math.isfinite(value):
            raise WrcReducerError("level must be finite")
        return f"float:{format(value, '.17g')}"
    if isinstance(value, str):
        return f"str:{value}"
    raise WrcReducerError(f"unsupported level type: {type(value).__name__}")


def scalar_to_json(value: Scalar) -> Scalar:
    if isinstance(value, float) and not math.isfinite(value):
        raise WrcReducerError("nonfinite scalar is not valid JSON evidence")
    return value


@dataclass(frozen=True, slots=True)
class ReducerConfig:
    """Materiality and bounded-retention policy.

    Four deterministic retention stages are applied per numeric metric:

    1. anchors: first and terminal observations;
    2. local events: material point-to-point moves and running outliers;
    3. cumulative drift: repeated sub-threshold changes that accumulate;
    4. shape preservation: turning points and extrema.

    The core never deletes or mutates raw simulation material.  It only emits a
    compact index back to retained source rows.
    """

    schema_version: str = REDUCER_CONFIG_SCHEMA_V1
    local_relative_threshold: float = 0.05
    cumulative_relative_threshold: float = 0.01
    turning_relative_threshold: float = 0.005
    effect_relative_threshold: float = 0.01
    absolute_threshold: float = 1.0e-12
    relative_floor: float = 1.0e-9
    outlier_z_threshold: float = 4.0
    effect_noise_multiplier: float = 1.0
    minimum_seed_consistency: float = 0.625
    max_retained_points_per_metric: int = 64
    top_metrics_per_parameter: int = 50
    top_metrics_per_layer: int = 8
    drilldown_metrics: int = 16
    include_patterns: tuple[str, ...] = ()
    exclude_patterns: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.schema_version != REDUCER_CONFIG_SCHEMA_V1:
            raise WrcReducerError("unsupported reducer config schema")
        for name in (
            "local_relative_threshold",
            "cumulative_relative_threshold",
            "turning_relative_threshold",
            "effect_relative_threshold",
            "absolute_threshold",
            "relative_floor",
            "outlier_z_threshold",
            "effect_noise_multiplier",
            "minimum_seed_consistency",
        ):
            value = finite_number(getattr(self, name), path=f"config.{name}")
            if value < 0:
                raise WrcReducerError(f"config.{name} must be nonnegative")
        if not 0 <= self.minimum_seed_consistency <= 1:
            raise WrcReducerError(
                "config.minimum_seed_consistency must be between 0 and 1"
            )
        for name in (
            "max_retained_points_per_metric",
            "top_metrics_per_parameter",
            "top_metrics_per_layer",
            "drilldown_metrics",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise WrcReducerError(f"config.{name} must be a positive integer")
        if self.max_retained_points_per_metric < 4:
            raise WrcReducerError(
                "config.max_retained_points_per_metric must be at least 4"
            )
        for name in ("include_patterns", "exclude_patterns"):
            values = getattr(self, name)
            if not isinstance(values, tuple) or any(
                not isinstance(item, str) or not item for item in values
            ):
                raise WrcReducerError(f"config.{name} must be nonempty strings")

    @property
    def identity_sha256(self) -> str:
        return canonical_sha256(self.to_dict())

    def accepts_metric(self, name: str) -> bool:
        if not isinstance(name, str) or not name:
            return False
        if self.include_patterns and not any(
            fnmatch.fnmatchcase(name, pattern) for pattern in self.include_patterns
        ):
            return False
        return not any(
            fnmatch.fnmatchcase(name, pattern) for pattern in self.exclude_patterns
        )

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["include_patterns"] = list(self.include_patterns)
        value["exclude_patterns"] = list(self.exclude_patterns)
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ReducerConfig":
        data = dict(value)
        data["include_patterns"] = tuple(data.get("include_patterns", ()))
        data["exclude_patterns"] = tuple(data.get("exclude_patterns", ()))
        return cls(**data)


@dataclass(frozen=True, slots=True)
class WorldIdentity:
    run_id: str
    parameter_id: str
    parameter_name: str
    level: Scalar
    seed: int
    ticks_per_generation: int | None = None
    source_kind: str = "live"
    source_path: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("run_id", "parameter_id", "parameter_name", "source_kind"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise WrcReducerError(f"world.{name} must be a nonempty string")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise WrcReducerError("world.seed must be an integer")
        if self.ticks_per_generation is not None and (
            isinstance(self.ticks_per_generation, bool)
            or not isinstance(self.ticks_per_generation, int)
            or self.ticks_per_generation < 1
        ):
            raise WrcReducerError(
                "world.ticks_per_generation must be a positive integer or null"
            )
        scalar_to_json(self.level)
        if not isinstance(self.metadata, Mapping):
            raise WrcReducerError("world.metadata must be an object")

    @property
    def level_key(self) -> str:
        return canonical_level_key(self.level)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "parameter_id": self.parameter_id,
            "parameter_name": self.parameter_name,
            "level": scalar_to_json(self.level),
            "level_key": self.level_key,
            "seed": self.seed,
            "ticks_per_generation": self.ticks_per_generation,
            "source_kind": self.source_kind,
            "source_path": self.source_path,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "WorldIdentity":
        data = dict(value)
        data.pop("level_key", None)
        return cls(**data)


@dataclass(frozen=True, slots=True)
class Observation:
    sample_index: int
    coordinate: float
    coordinate_kind: str
    metrics: Mapping[str, float]
    raw_ref: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if (
            isinstance(self.sample_index, bool)
            or not isinstance(self.sample_index, int)
            or self.sample_index < 0
        ):
            raise WrcReducerError("observation.sample_index must be nonnegative")
        finite_number(self.coordinate, path="observation.coordinate")
        if not isinstance(self.coordinate_kind, str) or not self.coordinate_kind:
            raise WrcReducerError(
                "observation.coordinate_kind must be a nonempty string"
            )
        if not isinstance(self.metrics, Mapping):
            raise WrcReducerError("observation.metrics must be an object")
        for name, value in self.metrics.items():
            if not isinstance(name, str) or not name:
                raise WrcReducerError("observation metric names must be nonempty")
            finite_number(value, path=f"observation.metrics.{name}")
        if not isinstance(self.raw_ref, Mapping):
            raise WrcReducerError("observation.raw_ref must be an object")

    def digest_payload(self) -> dict[str, Any]:
        return {
            "schema_version": OBSERVATION_SCHEMA_V1,
            "sample_index": self.sample_index,
            "coordinate": float(self.coordinate),
            "coordinate_kind": self.coordinate_kind,
            "metrics": {
                name: float(value) for name, value in sorted(self.metrics.items())
            },
        }


@dataclass(frozen=True, slots=True)
class RunDescriptor:
    identity: WorldIdentity
    metrics_path: Path
    expected_samples: int | None = None
    status: str = "PASS"


@dataclass(frozen=True, slots=True)
class ParameterDescriptor:
    parameter_id: str
    parameter_name: str
    family: str
    value_type: str
    reference_default: Scalar
    levels: tuple[Scalar, ...]
    seeds: tuple[int, ...]
    expected_worlds: int
    primary_response_metrics: str
    assertions: tuple[Mapping[str, Any], ...]
    expected_timescale: str
    launcher_profile: str
    world_base: str
    world_overlays: tuple[str, ...]
    row_metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def reference_level_key(self) -> str:
        return canonical_level_key(self.reference_default)

    def to_dict(self) -> dict[str, Any]:
        return {
            "parameter_id": self.parameter_id,
            "parameter_name": self.parameter_name,
            "family": self.family,
            "value_type": self.value_type,
            "reference_default": scalar_to_json(self.reference_default),
            "reference_level_key": self.reference_level_key,
            "levels": [scalar_to_json(value) for value in self.levels],
            "seeds": list(self.seeds),
            "expected_worlds": self.expected_worlds,
            "primary_response_metrics": self.primary_response_metrics,
            "assertions": [dict(value) for value in self.assertions],
            "expected_timescale": self.expected_timescale,
            "launcher_profile": self.launcher_profile,
            "world_base": self.world_base,
            "world_overlays": list(self.world_overlays),
            "row_metadata": dict(self.row_metadata),
        }


def cast_atlas_scalar(value: Any, value_type: str) -> Scalar:
    """Cast one Atlas value without accepting ambiguous booleans."""

    if value is None:
        return None
    if value_type == "bool":
        if isinstance(value, bool):
            return value
        normalized = str(value).strip().lower()
        if normalized == "true":
            return True
        if normalized == "false":
            return False
        raise WrcReducerError(f"invalid bool Atlas value: {value!r}")
    if value_type == "int":
        if isinstance(value, bool):
            raise WrcReducerError("bool is not an Atlas int")
        number = float(value)
        if not math.isfinite(number) or not number.is_integer():
            raise WrcReducerError(f"invalid int Atlas value: {value!r}")
        return int(number)
    if value_type == "float":
        if isinstance(value, bool):
            raise WrcReducerError("bool is not an Atlas float")
        number = float(value)
        if not math.isfinite(number):
            raise WrcReducerError(f"invalid float Atlas value: {value!r}")
        return number
    return str(value)


def matching_level(value: Scalar, levels: Sequence[Scalar]) -> Scalar | None:
    key = canonical_level_key(value)
    for candidate in levels:
        if canonical_level_key(candidate) == key:
            return candidate
    return None


__all__ = [
    "ATLAS_SUMMARY_SCHEMA_V1",
    "OBSERVATION_SCHEMA_V1",
    "PARAMETER_SUMMARY_SCHEMA_V1",
    "REDUCER_CONFIG_SCHEMA_V1",
    "REDUCER_STATE_SCHEMA_V1",
    "WORLD_SUMMARY_SCHEMA_V1",
    "Observation",
    "ParameterDescriptor",
    "ReducerConfig",
    "RunDescriptor",
    "Scalar",
    "WorldIdentity",
    "WrcReducerError",
    "canonical_bytes",
    "canonical_level_key",
    "canonical_sha256",
    "cast_atlas_scalar",
    "finite_number",
    "matching_level",
    "pretty_json",
]
