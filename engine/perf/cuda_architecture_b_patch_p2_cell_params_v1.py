from __future__ import annotations

"""Canonical CPU-side codec for the fixed Patch P2 per-cell parameter ABI."""

from dataclasses import asdict, dataclass, fields
from hashlib import sha256
import math
import struct
from typing import Any, Final, Mapping


ABI_SCHEMA_V1: Final[str] = "eiel_cuda_architecture_b_patch_p2_cell_params_v1"
DIGEST_SCHEMA_V1: Final[str] = (
    "sha256_canonical_little_endian_patch_p2_cell_params_v1"
)
ABI_VERSION_V1: Final[int] = 1
RECORD_BYTES_V1: Final[int] = 160
RECORD_ALIGNMENT_V1: Final[int] = 16

FLAG_FOOD_REGROW_USE_FERTILITY_V1: Final[int] = 1 << 0
FLAG_FOOD_REGROW_CAP_BY_FERTILITY_V1: Final[int] = 1 << 1
FLAG_DENSITY_INCLUDE_SELF_V1: Final[int] = 1 << 2
FLAG_DENSITY_NORMALIZE_V1: Final[int] = 1 << 3
FLAG_WORLD_WASTE_CONSEQUENCE_ENABLE_V1: Final[int] = 1 << 4
FLAG_WASTE_REGROWTH_PENALTY_ENABLE_V1: Final[int] = 1 << 5
FLAG_DENSITY_REGROWTH_PENALTY_ENABLE_V1: Final[int] = 1 << 6
FLAG_MOVEMENT_COST_APPLY_ENABLE_V1: Final[int] = 1 << 7
DEFINED_FLAGS_V1: Final[int] = 0xFF
REQUIRED_P2_PROFILE_FLAGS_V1: Final[int] = (
    FLAG_WORLD_WASTE_CONSEQUENCE_ENABLE_V1
    | FLAG_MOVEMENT_COST_APPLY_ENABLE_V1
)

FLAG_CONFIG_FIELDS_V1: Final[tuple[tuple[str, int], ...]] = (
    ("patch_food_regrow_use_fertility", FLAG_FOOD_REGROW_USE_FERTILITY_V1),
    (
        "patch_food_regrow_cap_by_fertility",
        FLAG_FOOD_REGROW_CAP_BY_FERTILITY_V1,
    ),
    ("patch_density_include_self", FLAG_DENSITY_INCLUDE_SELF_V1),
    ("patch_density_normalize", FLAG_DENSITY_NORMALIZE_V1),
    (
        "patch_world_waste_consequence_enable",
        FLAG_WORLD_WASTE_CONSEQUENCE_ENABLE_V1,
    ),
    (
        "patch_waste_regrowth_penalty_enable",
        FLAG_WASTE_REGROWTH_PENALTY_ENABLE_V1,
    ),
    (
        "patch_density_regrowth_penalty_enable",
        FLAG_DENSITY_REGROWTH_PENALTY_ENABLE_V1,
    ),
    ("patch_movement_cost_apply_enable", FLAG_MOVEMENT_COST_APPLY_ENABLE_V1),
)

FLOAT32_FIELDS_V1: Final[tuple[str, ...]] = (
    "patch_scent_decay_factor",
    "patch_waste_decay_factor",
    "patch_waste_density_emit_scale",
    "patch_scent_max",
    "patch_waste_max",
    "patch_food_amount_max",
    "patch_food_regrow_per_step",
    "patch_food_presence_threshold",
    "patch_waste_regrowth_penalty_scale",
    "patch_waste_regrowth_penalty_max",
    "patch_density_regrowth_penalty_scale",
    "patch_density_regrowth_penalty_max",
    "patch_regrowth_min_multiplier",
)
FLOAT64_FIELDS_V1: Final[tuple[str, ...]] = (
    "patch_scent_presence_emit",
    "patch_waste_presence_emit",
    "patch_food_amount_initial_food_cell_value",
    "patch_food_gather_amount",
    "patch_scent_move_emit",
    "patch_waste_move_emit",
    "patch_scent_gather_emit",
    "patch_waste_consume_emit",
    "patch_movement_cost_energy_scale",
    "patch_movement_cost_min",
    "patch_movement_cost_max",
)

SOURCE_CONFIG_FIELDS_V1: Final[tuple[str, ...]] = (
    "patch_scent_decay_per_step",
    "patch_waste_decay_per_step",
    "patch_scent_presence_emit",
    "patch_waste_presence_emit",
    "patch_waste_density_emit_scale",
    "patch_scent_max",
    "patch_waste_max",
    "patch_food_amount_max",
    "patch_food_regrow_per_step",
    "patch_food_regrow_use_fertility",
    "patch_food_regrow_cap_by_fertility",
    "patch_food_presence_threshold",
    "patch_density_radius",
    "patch_density_include_self",
    "patch_density_normalize",
    "patch_world_waste_consequence_enable",
    "patch_waste_regrowth_penalty_enable",
    "patch_waste_regrowth_penalty_scale",
    "patch_waste_regrowth_penalty_max",
    "patch_density_regrowth_penalty_enable",
    "patch_density_regrowth_penalty_scale",
    "patch_density_regrowth_penalty_max",
    "patch_regrowth_min_multiplier",
    "patch_food_amount_initial_food_cell_value",
    "patch_food_gather_amount",
    "patch_scent_move_emit",
    "patch_waste_move_emit",
    "patch_scent_gather_emit",
    "patch_waste_consume_emit",
    "patch_movement_cost_apply_enable",
    "patch_movement_cost_energy_scale",
    "patch_movement_cost_min",
    "patch_movement_cost_max",
)

CANONICAL_STRUCT_V1: Final[struct.Struct] = struct.Struct("<IIq13fI11d")
if CANONICAL_STRUCT_V1.size != RECORD_BYTES_V1:
    raise RuntimeError("Patch P2 cell parameter ABI must be exactly 160 bytes")

FIELD_OFFSETS_V1: Final[dict[str, int]] = {
    "abi_version": 0,
    "flags": 4,
    "patch_density_radius": 8,
    **{name: 16 + index * 4 for index, name in enumerate(FLOAT32_FIELDS_V1)},
    "reserved_zero": 68,
    **{name: 72 + index * 8 for index, name in enumerate(FLOAT64_FIELDS_V1)},
}


class PatchP2CellParamsError(ValueError):
    """The source config or canonical ABI record violates the v1 contract."""


@dataclass(frozen=True)
class PatchP2CellParamsV1:
    abi_version: int
    flags: int
    patch_density_radius: int

    patch_scent_decay_factor: float
    patch_waste_decay_factor: float
    patch_waste_density_emit_scale: float
    patch_scent_max: float
    patch_waste_max: float
    patch_food_amount_max: float
    patch_food_regrow_per_step: float
    patch_food_presence_threshold: float
    patch_waste_regrowth_penalty_scale: float
    patch_waste_regrowth_penalty_max: float
    patch_density_regrowth_penalty_scale: float
    patch_density_regrowth_penalty_max: float
    patch_regrowth_min_multiplier: float

    reserved_zero: int

    patch_scent_presence_emit: float
    patch_waste_presence_emit: float
    patch_food_amount_initial_food_cell_value: float
    patch_food_gather_amount: float
    patch_scent_move_emit: float
    patch_waste_move_emit: float
    patch_scent_gather_emit: float
    patch_waste_consume_emit: float
    patch_movement_cost_energy_scale: float
    patch_movement_cost_min: float
    patch_movement_cost_max: float

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


NORMALIZED_FIELDS_V1: Final[tuple[str, ...]] = tuple(
    field.name for field in fields(PatchP2CellParamsV1)
)


def _exact_int(value: Any, *, name: str) -> int:
    if type(value) is not int:
        raise PatchP2CellParamsError(f"{name} must be an exact int")
    return value


def _exact_bool(value: Any, *, name: str) -> bool:
    if type(value) is not bool:
        raise PatchP2CellParamsError(f"{name} must be an exact bool")
    return value


def _finite_float64(value: Any, *, name: str) -> float:
    if isinstance(value, bool):
        raise PatchP2CellParamsError(f"{name} must be numeric, not bool")
    try:
        normalized = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise PatchP2CellParamsError(f"{name} must be a finite float64") from exc
    if not math.isfinite(normalized):
        raise PatchP2CellParamsError(f"{name} must be finite")
    return 0.0 if normalized == 0.0 else normalized


def _finite_float32(value: Any, *, name: str) -> float:
    normalized = _finite_float64(value, name=name)
    try:
        normalized = struct.unpack("<f", struct.pack("<f", normalized))[0]
    except (OverflowError, struct.error) as exc:
        raise PatchP2CellParamsError(f"{name} is outside finite float32") from exc
    if not math.isfinite(normalized):
        raise PatchP2CellParamsError(f"{name} is outside finite float32")
    return 0.0 if normalized == 0.0 else normalized


def _clamp(value: float, low: float, high: float) -> float:
    return min(high, max(low, value))


def _require_exact_fields(
    value: Mapping[str, Any], expected: tuple[str, ...], *, name: str
) -> None:
    actual = set(value)
    expected_set = set(expected)
    if actual != expected_set:
        missing = sorted(expected_set - actual)
        unexpected = sorted(actual - expected_set)
        raise PatchP2CellParamsError(
            f"{name} field mismatch: missing={missing}, unexpected={unexpected}"
        )


def _canonicalize_record(value: PatchP2CellParamsV1) -> PatchP2CellParamsV1:
    if not isinstance(value, PatchP2CellParamsV1):
        raise PatchP2CellParamsError("value must be PatchP2CellParamsV1")
    raw = value.to_dict()
    canonical = PatchP2CellParamsV1(
        abi_version=_exact_int(raw["abi_version"], name="abi_version"),
        flags=_exact_int(raw["flags"], name="flags"),
        patch_density_radius=_exact_int(
            raw["patch_density_radius"], name="patch_density_radius"
        ),
        **{
            name: _finite_float32(raw[name], name=name)
            for name in FLOAT32_FIELDS_V1
        },
        reserved_zero=_exact_int(raw["reserved_zero"], name="reserved_zero"),
        **{
            name: _finite_float64(raw[name], name=name)
            for name in FLOAT64_FIELDS_V1
        },
    )
    _validate_canonical_record(canonical)
    return canonical


def _validate_canonical_record(value: PatchP2CellParamsV1) -> None:
    if value.abi_version != ABI_VERSION_V1:
        raise PatchP2CellParamsError("unsupported Patch P2 cell parameter ABI version")
    if value.flags < 0 or value.flags > 0xFFFFFFFF:
        raise PatchP2CellParamsError("flags must fit uint32")
    if value.flags & ~DEFINED_FLAGS_V1:
        raise PatchP2CellParamsError("flags contain reserved bits")
    if (
        value.flags & REQUIRED_P2_PROFILE_FLAGS_V1
    ) != REQUIRED_P2_PROFILE_FLAGS_V1:
        raise PatchP2CellParamsError(
            "waste consequence and movement-cost P2 profile flags must be true"
        )
    if not -(1 << 63) <= value.patch_density_radius < (1 << 63):
        raise PatchP2CellParamsError("patch_density_radius must fit int64")
    if value.patch_density_radius < 0:
        raise PatchP2CellParamsError("patch_density_radius must be nonnegative")
    if value.reserved_zero != 0:
        raise PatchP2CellParamsError("reserved storage must be zero")

    unit_interval = (
        "patch_scent_decay_factor",
        "patch_waste_decay_factor",
        "patch_waste_regrowth_penalty_max",
        "patch_density_regrowth_penalty_max",
        "patch_regrowth_min_multiplier",
    )
    for name in unit_interval:
        if not 0.0 <= getattr(value, name) <= 1.0:
            raise PatchP2CellParamsError(f"{name} must be in [0, 1]")
    nonnegative = (
        "patch_waste_density_emit_scale",
        "patch_scent_max",
        "patch_waste_max",
        "patch_food_amount_max",
        "patch_food_regrow_per_step",
        "patch_food_presence_threshold",
        "patch_waste_regrowth_penalty_scale",
        "patch_density_regrowth_penalty_scale",
        "patch_scent_presence_emit",
        "patch_waste_presence_emit",
        "patch_scent_move_emit",
        "patch_waste_move_emit",
        "patch_scent_gather_emit",
        "patch_waste_consume_emit",
        "patch_movement_cost_energy_scale",
        "patch_movement_cost_min",
        "patch_movement_cost_max",
    )
    for name in nonnegative:
        if getattr(value, name) < 0.0:
            raise PatchP2CellParamsError(f"{name} must be nonnegative")
    if value.patch_food_presence_threshold > value.patch_food_amount_max:
        raise PatchP2CellParamsError(
            "patch_food_presence_threshold must not exceed patch_food_amount_max"
        )
    if not (
        value.patch_food_presence_threshold
        < value.patch_food_amount_initial_food_cell_value
        <= value.patch_food_amount_max
    ):
        raise PatchP2CellParamsError(
            "initial food value must be above threshold and no greater than maximum"
        )
    if value.patch_food_gather_amount <= 0.0:
        raise PatchP2CellParamsError("patch_food_gather_amount must be positive")
    if value.patch_movement_cost_max < value.patch_movement_cost_min:
        raise PatchP2CellParamsError(
            "patch_movement_cost_max must be at least patch_movement_cost_min"
        )


def normalize_patch_p2_cell_params_v1(
    config: Mapping[str, Any],
) -> PatchP2CellParamsV1:
    """Normalize the exact current 33-field P2 config into the frozen v1 POD."""

    if not isinstance(config, Mapping):
        raise PatchP2CellParamsError("config must be a mapping")
    _require_exact_fields(config, SOURCE_CONFIG_FIELDS_V1, name="source config")

    flags = 0
    for name, bit in FLAG_CONFIG_FIELDS_V1:
        if _exact_bool(config[name], name=name):
            flags |= bit

    radius = _exact_int(config["patch_density_radius"], name="patch_density_radius")
    f64 = {
        name: _finite_float64(config[name], name=name)
        for name in SOURCE_CONFIG_FIELDS_V1
        if name not in {field for field, _ in FLAG_CONFIG_FIELDS_V1}
        and name != "patch_density_radius"
    }
    food_max = max(0.0, f64["patch_food_amount_max"])

    value = PatchP2CellParamsV1(
        abi_version=ABI_VERSION_V1,
        flags=flags,
        patch_density_radius=radius,
        patch_scent_decay_factor=_finite_float32(
            1.0 - _clamp(f64["patch_scent_decay_per_step"], 0.0, 1.0),
            name="patch_scent_decay_factor",
        ),
        patch_waste_decay_factor=_finite_float32(
            1.0 - _clamp(f64["patch_waste_decay_per_step"], 0.0, 1.0),
            name="patch_waste_decay_factor",
        ),
        patch_waste_density_emit_scale=max(
            0.0, f64["patch_waste_density_emit_scale"]
        ),
        patch_scent_max=max(0.0, f64["patch_scent_max"]),
        patch_waste_max=max(0.0, f64["patch_waste_max"]),
        patch_food_amount_max=food_max,
        patch_food_regrow_per_step=max(
            0.0, f64["patch_food_regrow_per_step"]
        ),
        patch_food_presence_threshold=_clamp(
            f64["patch_food_presence_threshold"], 0.0, food_max
        ),
        patch_waste_regrowth_penalty_scale=max(
            0.0, f64["patch_waste_regrowth_penalty_scale"]
        ),
        patch_waste_regrowth_penalty_max=_clamp(
            f64["patch_waste_regrowth_penalty_max"], 0.0, 1.0
        ),
        patch_density_regrowth_penalty_scale=max(
            0.0, f64["patch_density_regrowth_penalty_scale"]
        ),
        patch_density_regrowth_penalty_max=_clamp(
            f64["patch_density_regrowth_penalty_max"], 0.0, 1.0
        ),
        patch_regrowth_min_multiplier=_clamp(
            f64["patch_regrowth_min_multiplier"], 0.0, 1.0
        ),
        reserved_zero=0,
        patch_scent_presence_emit=max(0.0, f64["patch_scent_presence_emit"]),
        patch_waste_presence_emit=max(0.0, f64["patch_waste_presence_emit"]),
        patch_food_amount_initial_food_cell_value=f64[
            "patch_food_amount_initial_food_cell_value"
        ],
        patch_food_gather_amount=f64["patch_food_gather_amount"],
        patch_scent_move_emit=f64["patch_scent_move_emit"],
        patch_waste_move_emit=f64["patch_waste_move_emit"],
        patch_scent_gather_emit=f64["patch_scent_gather_emit"],
        patch_waste_consume_emit=f64["patch_waste_consume_emit"],
        patch_movement_cost_energy_scale=f64[
            "patch_movement_cost_energy_scale"
        ],
        patch_movement_cost_min=f64["patch_movement_cost_min"],
        patch_movement_cost_max=f64["patch_movement_cost_max"],
    )
    return _canonicalize_record(value)


def validate_patch_p2_cell_params_v1(value: PatchP2CellParamsV1) -> None:
    _canonicalize_record(value)


def _coerce_record(
    value: PatchP2CellParamsV1 | Mapping[str, Any],
) -> PatchP2CellParamsV1:
    if isinstance(value, PatchP2CellParamsV1):
        return _canonicalize_record(value)
    if not isinstance(value, Mapping):
        raise PatchP2CellParamsError(
            "value must be PatchP2CellParamsV1 or an exact config mapping"
        )
    if set(value) == set(SOURCE_CONFIG_FIELDS_V1):
        return normalize_patch_p2_cell_params_v1(value)
    _require_exact_fields(value, NORMALIZED_FIELDS_V1, name="normalized record")
    return _canonicalize_record(PatchP2CellParamsV1(**value))


def pack_patch_p2_cell_params_v1(
    value: PatchP2CellParamsV1 | Mapping[str, Any],
) -> bytes:
    canonical = _coerce_record(value)
    packed = CANONICAL_STRUCT_V1.pack(
        canonical.abi_version,
        canonical.flags,
        canonical.patch_density_radius,
        *(getattr(canonical, name) for name in FLOAT32_FIELDS_V1),
        canonical.reserved_zero,
        *(getattr(canonical, name) for name in FLOAT64_FIELDS_V1),
    )
    if len(packed) != RECORD_BYTES_V1:
        raise AssertionError("canonical Patch P2 cell parameter size drift")
    return packed


def unpack_patch_p2_cell_params_v1(
    payload: bytes | bytearray | memoryview,
) -> PatchP2CellParamsV1:
    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise PatchP2CellParamsError("payload must be bytes-like")
    canonical = bytes(payload)
    if len(canonical) != RECORD_BYTES_V1:
        raise PatchP2CellParamsError(
            f"payload must contain exactly {RECORD_BYTES_V1} bytes"
        )
    unpacked = CANONICAL_STRUCT_V1.unpack(canonical)
    index = 0
    values: dict[str, int | float] = {
        "abi_version": unpacked[index],
        "flags": unpacked[index + 1],
        "patch_density_radius": unpacked[index + 2],
    }
    index += 3
    for name in FLOAT32_FIELDS_V1:
        values[name] = unpacked[index]
        index += 1
    values["reserved_zero"] = unpacked[index]
    index += 1
    for name in FLOAT64_FIELDS_V1:
        values[name] = unpacked[index]
        index += 1
    if index != len(unpacked):
        raise AssertionError("canonical Patch P2 cell parameter field drift")
    return _canonicalize_record(PatchP2CellParamsV1(**values))


def patch_p2_cell_params_sha256_v1(
    value: PatchP2CellParamsV1 | Mapping[str, Any] | bytes | bytearray | memoryview,
) -> str:
    if isinstance(value, (bytes, bytearray, memoryview)):
        canonical = pack_patch_p2_cell_params_v1(
            unpack_patch_p2_cell_params_v1(value)
        )
    else:
        canonical = pack_patch_p2_cell_params_v1(value)
    return sha256(canonical).hexdigest()


def patch_p2_cell_params_digest_v1(
    value: PatchP2CellParamsV1 | Mapping[str, Any] | bytes | bytearray | memoryview,
) -> dict[str, str | int]:
    return {
        "schema": DIGEST_SCHEMA_V1,
        "algorithm": "sha256",
        "record_abi": ABI_SCHEMA_V1,
        "record_bytes": RECORD_BYTES_V1,
        "sha256": patch_p2_cell_params_sha256_v1(value),
    }


def patch_p2_cell_params_digest_schema_v1() -> dict[str, Any]:
    return {
        "schema": DIGEST_SCHEMA_V1,
        "algorithm": "sha256",
        "canonical_input": "one validated 160-byte little-endian ABI v1 record",
        "record_abi": ABI_SCHEMA_V1,
        "record_bytes": RECORD_BYTES_V1,
        "record_alignment": RECORD_ALIGNMENT_V1,
        "field_offsets": dict(FIELD_OFFSETS_V1),
    }


__all__ = [
    "ABI_SCHEMA_V1",
    "ABI_VERSION_V1",
    "CANONICAL_STRUCT_V1",
    "DEFINED_FLAGS_V1",
    "DIGEST_SCHEMA_V1",
    "FIELD_OFFSETS_V1",
    "FLAG_MOVEMENT_COST_APPLY_ENABLE_V1",
    "FLAG_WORLD_WASTE_CONSEQUENCE_ENABLE_V1",
    "FLOAT32_FIELDS_V1",
    "FLOAT64_FIELDS_V1",
    "NORMALIZED_FIELDS_V1",
    "PatchP2CellParamsError",
    "PatchP2CellParamsV1",
    "RECORD_ALIGNMENT_V1",
    "RECORD_BYTES_V1",
    "REQUIRED_P2_PROFILE_FLAGS_V1",
    "SOURCE_CONFIG_FIELDS_V1",
    "normalize_patch_p2_cell_params_v1",
    "pack_patch_p2_cell_params_v1",
    "patch_p2_cell_params_digest_schema_v1",
    "patch_p2_cell_params_digest_v1",
    "patch_p2_cell_params_sha256_v1",
    "unpack_patch_p2_cell_params_v1",
    "validate_patch_p2_cell_params_v1",
]
