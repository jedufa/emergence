"""Generic resolved-world projection for the CUDA V2 Science Instrument.

This module is deliberately campaign-neutral.  It converts one already
materialized ``EIEL_SCIENCE_WORLD_BUNDLE_V1`` world into the closed payload
consumed by the native owner.  Physical slot, batch size, test family, and
panel names never participate in the projection.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final

from .canonical import canonical_sha256


SCIENCE_CUDA_V2_NATIVE_WORLD_SCHEMA_VERSION: Final[str] = (
    "eiel_science_cuda_v2_native_world_v1"
)
SCIENCE_CUDA_V2_NATIVE_WORLD_CONTROL_SET_SCHEMA_VERSION: Final[str] = (
    "eiel_science_cuda_v2_native_world_control_set_v1"
)

_CAPABILITY_CONTROL_BITS: Final[dict[str, int]] = {
    "environment.patch.enabled": 1 << 0,
    "environment.wdl.enabled": 1 << 1,
    "environment.jungle.enabled": 1 << 2,
    "environment.eed.enabled": 1 << 3,
    "reproduction.enabled": 1 << 4,
    "mutation.enabled": 1 << 5,
    "recombination.enabled": 1 << 6,
}
_FULL_CAPABILITY_MASK: Final[int] = (1 << 7) - 1

# Numeric controls map directly to fields in the reusable native world-control
# ABI.  Ranges are deliberately bounded to integer arithmetic that is stable
# across host and CUDA implementations.
_NUMERIC_CONTROL_RANGES: Final[dict[str, tuple[type, float, float]]] = {
    "mutation.rate": (float, 0.0, 1.0),
    "mutation.step": (int, 1, 127),
    "environment.resource.carrying_base": (int, 0, 65535),
    "environment.resource.carrying_variation": (int, 0, 65535),
    "environment.resource.growth_base": (int, 0, 65535),
    "environment.resource.growth_variation": (int, 0, 65535),
    "environment.resource.multiplier": (float, 0.0, 16.0),
    "environment.resource_multiplier": (float, 0.0, 16.0),
    "environment.movement.cost": (int, 0, 32767),
    "environment.resource.uptake_base": (int, 0, 32767),
    "environment.wdl.damage": (int, 0, 32767),
    "environment.jungle.damage_threshold": (int, 0, 255),
    "environment.eed.loss_shift": (int, 0, 15),
    "reproduction.energy_threshold": (int, 1, 32767),
    "reproduction.energy_cost": (int, 0, 32767),
    "selection.pressure": (float, 0.0, 16.0),
}
SCIENCE_CUDA_V2_SUPPORTED_NATIVE_CONTROL_IDS_V1: Final[tuple[str, ...]] = tuple(
    sorted((*_CAPABILITY_CONTROL_BITS, *_NUMERIC_CONTROL_RANGES))
)


class ScienceCudaV2NativeWorldError(ValueError):
    """A resolved science world cannot be represented by the native ABI."""


def _mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if hasattr(value, "to_value"):
        value = value.to_value()
    if not isinstance(value, Mapping):
        raise ScienceCudaV2NativeWorldError(f"{field} must be an object")
    return value


def _digest(value: object, *, field: str) -> str:
    if (
        not isinstance(value, Mapping)
        or set(value) != {"algorithm", "hex"}
        or value.get("algorithm") != "sha256"
    ):
        raise ScienceCudaV2NativeWorldError(f"{field} is not a digest object")
    digest = value.get("hex")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ScienceCudaV2NativeWorldError(f"{field} is not a SHA-256 digest")
    return digest


def _resolved_inputs(value: object) -> Mapping[str, Any]:
    if hasattr(value, "semantic_inputs"):
        value = getattr(value, "semantic_inputs")
    value = _mapping(value, field="native world input")
    candidate = value.get("semantic_inputs")
    if isinstance(candidate, Mapping):
        value = candidate
    required = {
        "world",
        "initialization",
        "dimension_bindings",
        "control_bindings",
        "seed_bindings",
        "time",
        "initial_state_identity",
    }
    if set(value) != required:
        raise ScienceCudaV2NativeWorldError(
            "resolved semantic input fields are not the exact materialized contract"
        )
    return value


def _science_ownership(value: object) -> Mapping[str, Any]:
    if hasattr(value, "science"):
        value = getattr(value, "science")
    value = _mapping(value, field="science ownership")
    candidate = value.get("science")
    if isinstance(candidate, Mapping):
        value = candidate
    required = {"test_id", "test_version", "study_id", "arm_id", "replicate"}
    if set(value) != required:
        raise ScienceCudaV2NativeWorldError(
            "science ownership is not the exact materialized contract"
        )
    for name in required - {"replicate"}:
        if not isinstance(value[name], str) or not value[name]:
            raise ScienceCudaV2NativeWorldError(f"science.{name} must be text")
    if not isinstance(value["replicate"], (str, int)) or isinstance(
        value["replicate"], bool
    ):
        raise ScienceCudaV2NativeWorldError(
            "science.replicate must be text or integer"
        )
    return value


def _validate_schedule(value: object, *, field: str) -> dict[str, Any]:
    schedule = dict(_mapping(value, field=field))
    kind = schedule.get("kind")
    if kind == "static" and set(schedule) == {"kind"}:
        return schedule
    if kind == "piecewise_constant_absolute_ticks" and set(schedule) == {
        "kind",
        "segments",
    }:
        segments = schedule["segments"]
        if not isinstance(segments, list) or not segments:
            raise ScienceCudaV2NativeWorldError(f"{field}.segments must be nonempty")
        starts: list[int] = []
        for index, segment in enumerate(segments):
            if not isinstance(segment, Mapping) or set(segment) != {
                "start_absolute_tick",
                "value",
            }:
                raise ScienceCudaV2NativeWorldError(
                    f"{field}.segments[{index}] is invalid"
                )
            start = segment["start_absolute_tick"]
            if type(start) is not int or start < 0:
                raise ScienceCudaV2NativeWorldError(
                    f"{field}.segments[{index}] tick is invalid"
                )
            starts.append(start)
        if starts != sorted(set(starts)):
            raise ScienceCudaV2NativeWorldError(
                f"{field}.segments must have unique ascending absolute ticks"
            )
        return schedule
    if kind == "pulse_absolute_ticks" and set(schedule) == {
        "kind",
        "start_absolute_tick",
        "duration_ticks",
    }:
        if (
            type(schedule["start_absolute_tick"]) is not int
            or schedule["start_absolute_tick"] < 0
            or type(schedule["duration_ticks"]) is not int
            or schedule["duration_ticks"] < 1
        ):
            raise ScienceCudaV2NativeWorldError(f"{field} pulse is invalid")
        return schedule
    if kind == "switch_absolute_tick" and set(schedule) == {
        "kind",
        "absolute_tick",
    }:
        if type(schedule["absolute_tick"]) is not int or schedule["absolute_tick"] < 0:
            raise ScienceCudaV2NativeWorldError(f"{field} switch is invalid")
        return schedule
    raise ScienceCudaV2NativeWorldError(
        f"{field} is not a native absolute-tick schedule"
    )


def _validate_control_value(control_id: str, value: object, *, field: str) -> None:
    if control_id in _CAPABILITY_CONTROL_BITS:
        if type(value) is not bool:
            raise ScienceCudaV2NativeWorldError(f"{field} must be boolean")
        return
    rule = _NUMERIC_CONTROL_RANGES.get(control_id)
    if rule is None:
        raise ScienceCudaV2NativeWorldError(
            f"unsupported native science control {control_id!r}"
        )
    expected_type, minimum, maximum = rule
    if expected_type is int:
        valid_type = type(value) is int
    else:
        valid_type = isinstance(value, (int, float)) and not isinstance(value, bool)
    if not valid_type or not minimum <= value <= maximum:
        raise ScienceCudaV2NativeWorldError(
            f"{field} must be in [{minimum}, {maximum}]"
        )


def _control_rows(value: object) -> tuple[dict[str, Any], ...]:
    rows = value
    if not isinstance(rows, list):
        raise ScienceCudaV2NativeWorldError("control_bindings must be an array")
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(rows):
        if not isinstance(raw, Mapping) or set(raw) != {
            "control_id",
            "resolved_value",
            "schedule",
            "descriptor_sha256",
        }:
            raise ScienceCudaV2NativeWorldError(
                f"control_bindings[{index}] fields are not exact"
            )
        control_id = raw.get("control_id")
        if not isinstance(control_id, str) or not control_id:
            raise ScienceCudaV2NativeWorldError(
                f"control_bindings[{index}].control_id is invalid"
            )
        if control_id in seen:
            raise ScienceCudaV2NativeWorldError(
                f"native science control {control_id!r} is duplicated"
            )
        seen.add(control_id)
        _digest(raw.get("descriptor_sha256"), field=f"{control_id}.descriptor_sha256")
        schedule = _validate_schedule(
            raw.get("schedule"), field=f"{control_id}.schedule"
        )
        resolved = raw.get("resolved_value")
        _validate_control_value(control_id, resolved, field=control_id)
        if schedule.get("kind") == "piecewise_constant_absolute_ticks":
            for segment in schedule["segments"]:
                _validate_control_value(
                    control_id,
                    segment["value"],
                    field=f"{control_id}.schedule.value",
                )
        result.append(
            {
                "control_id": control_id,
                "resolved_value": resolved,
                "schedule": schedule,
                "descriptor_sha256": dict(raw["descriptor_sha256"]),
            }
        )
    return tuple(sorted(result, key=lambda row: row["control_id"]))


def _initial_capability_mask(
    controls: tuple[dict[str, Any], ...], *, start_next_tick: int
) -> int:
    mask = _FULL_CAPABILITY_MASK
    for row in controls:
        control_id = row["control_id"]
        bit = _CAPABILITY_CONTROL_BITS.get(control_id)
        if bit is None:
            continue
        value = row["resolved_value"]
        schedule = row["schedule"]
        if schedule["kind"] == "piecewise_constant_absolute_ticks":
            for segment in schedule["segments"]:
                if segment["start_absolute_tick"] <= start_next_tick:
                    value = segment["value"]
                else:
                    break
        elif schedule["kind"] == "switch_absolute_tick" and (
            start_next_tick >= schedule["absolute_tick"]
        ):
            value = not value
        elif schedule["kind"] == "pulse_absolute_ticks":
            pulse_start = schedule["start_absolute_tick"]
            value = bool(value) if pulse_start <= start_next_tick < (
                pulse_start + schedule["duration_ticks"]
            ) else True
        mask = mask | bit if value else mask & ~bit
    return mask


@dataclass(frozen=True, slots=True)
class ScienceCudaV2NativeWorldV1:
    """One exact logical-world configuration, independent of physical slot."""

    payload: dict[str, Any]

    def __post_init__(self) -> None:
        parsed = self.from_dict(self.payload)
        if parsed.payload != self.payload:
            raise ScienceCudaV2NativeWorldError("native world payload is not canonical")

    @property
    def control_digest(self) -> str:
        return str(self.payload["control_digest"]["hex"])

    @property
    def capability_mask_at_start(self) -> int:
        return int(self.payload["capability_mask_at_start"])

    def to_dict(self) -> dict[str, Any]:
        return dict(self.payload)

    @classmethod
    def project(cls, value: object) -> "ScienceCudaV2NativeWorldV1":
        semantic = _resolved_inputs(value)
        science = _science_ownership(value)
        world = dict(_mapping(semantic["world"], field="world"))
        initialization = dict(
            _mapping(semantic["initialization"], field="initialization")
        )
        seeds = dict(_mapping(semantic["seed_bindings"], field="seed_bindings"))
        selected_world_initialization = initialization.get("schema_version") == (
            "eiel_science_selected_world_initialization_v1"
        )
        if (
            (not seeds and not selected_world_initialization)
            or (selected_world_initialization and seeds)
            or any(
            not isinstance(name, str)
            or not name
            or type(seed) is not int
            or seed < 0
            or seed >= 1 << 63
            for name, seed in seeds.items()
            )
        ):
            raise ScienceCudaV2NativeWorldError(
                "seed bindings must match founder or selected-world initialization"
            )
        time = dict(_mapping(semantic["time"], field="time"))
        if time.get("authority") != "absolute_tick_v1":
            raise ScienceCudaV2NativeWorldError("absolute-tick authority is required")
        start = time.get("start_next_tick", 0)
        total = time.get("total_ticks")
        if type(start) is not int or start < 0 or type(total) is not int or total < 1:
            raise ScienceCudaV2NativeWorldError("absolute-tick horizon is invalid")
        _digest(semantic["initial_state_identity"], field="initial_state_identity")
        controls = _control_rows(semantic["control_bindings"])
        body = {
            "schema_version": SCIENCE_CUDA_V2_NATIVE_WORLD_SCHEMA_VERSION,
            "semantic_inputs_sha256": {
                "algorithm": "sha256",
                "hex": canonical_sha256(semantic),
            },
            "science_ownership_sha256": {
                "algorithm": "sha256",
                "hex": canonical_sha256(science),
            },
            "capability_mask_at_start": _initial_capability_mask(
                controls, start_next_tick=start
            ),
            "world": world,
            "initialization": initialization,
            "seed_bindings": seeds,
            "control_bindings": list(controls),
            "time": time,
        }
        return object.__new__(cls)._from_body(body)

    def _from_body(self, body: dict[str, Any]) -> "ScienceCudaV2NativeWorldV1":
        payload = {
            **body,
            "control_digest": {
                "algorithm": "sha256",
                "hex": canonical_sha256(body),
            },
        }
        object.__setattr__(self, "payload", payload)
        return self

    @classmethod
    def from_dict(cls, value: object) -> "ScienceCudaV2NativeWorldV1":
        raw = dict(_mapping(value, field="native world control"))
        required = {
            "schema_version",
            "semantic_inputs_sha256",
            "science_ownership_sha256",
            "capability_mask_at_start",
            "world",
            "initialization",
            "seed_bindings",
            "control_bindings",
            "time",
            "control_digest",
        }
        if set(raw) != required or raw.get("schema_version") != (
            SCIENCE_CUDA_V2_NATIVE_WORLD_SCHEMA_VERSION
        ):
            raise ScienceCudaV2NativeWorldError(
                "native world control fields or schema are unsupported"
            )
        _digest(raw["semantic_inputs_sha256"], field="semantic_inputs_sha256")
        _digest(raw["science_ownership_sha256"], field="science_ownership_sha256")
        _digest(raw["control_digest"], field="control_digest")
        mask = raw["capability_mask_at_start"]
        if type(mask) is not int or not 0 <= mask <= _FULL_CAPABILITY_MASK:
            raise ScienceCudaV2NativeWorldError(
                "capability_mask_at_start is invalid"
            )
        _mapping(raw["world"], field="world")
        _mapping(raw["initialization"], field="initialization")
        _mapping(raw["seed_bindings"], field="seed_bindings")
        _mapping(raw["time"], field="time")
        _control_rows(raw["control_bindings"])
        body = {key: raw[key] for key in required - {"control_digest"}}
        if canonical_sha256(body) != raw["control_digest"]["hex"]:
            raise ScienceCudaV2NativeWorldError(
                "native world control digest does not match its body"
            )
        instance = object.__new__(cls)
        object.__setattr__(instance, "payload", raw)
        return instance


def project_science_cuda_v2_native_world_v1(
    value: object,
) -> ScienceCudaV2NativeWorldV1:
    return ScienceCudaV2NativeWorldV1.project(value)


def native_world_control_set_sha256_v1(
    values: tuple[ScienceCudaV2NativeWorldV1, ...] | list[ScienceCudaV2NativeWorldV1],
) -> str:
    if not values:
        raise ScienceCudaV2NativeWorldError("native world control set cannot be empty")
    return canonical_sha256(
        {
            "schema_version": SCIENCE_CUDA_V2_NATIVE_WORLD_CONTROL_SET_SCHEMA_VERSION,
            "ordered_control_digests": [value.control_digest for value in values],
        }
    )


__all__ = (
    "SCIENCE_CUDA_V2_NATIVE_WORLD_CONTROL_SET_SCHEMA_VERSION",
    "SCIENCE_CUDA_V2_NATIVE_WORLD_SCHEMA_VERSION",
    "SCIENCE_CUDA_V2_SUPPORTED_NATIVE_CONTROL_IDS_V1",
    "ScienceCudaV2NativeWorldError",
    "ScienceCudaV2NativeWorldV1",
    "native_world_control_set_sha256_v1",
    "project_science_cuda_v2_native_world_v1",
)
