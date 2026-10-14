from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
from pathlib import Path
import shutil
import struct
import subprocess

import pytest

from engine.perf.cuda_architecture_b_patch_p2_cell_params_v1 import (
    ABI_SCHEMA_V1,
    ABI_VERSION_V1,
    CANONICAL_STRUCT_V1,
    DIGEST_SCHEMA_V1,
    FIELD_OFFSETS_V1,
    FLAG_MOVEMENT_COST_APPLY_ENABLE_V1,
    FLAG_WORLD_WASTE_CONSEQUENCE_ENABLE_V1,
    FLOAT32_FIELDS_V1,
    FLOAT64_FIELDS_V1,
    PatchP2CellParamsError,
    RECORD_ALIGNMENT_V1,
    RECORD_BYTES_V1,
    REQUIRED_P2_PROFILE_FLAGS_V1,
    normalize_patch_p2_cell_params_v1,
    pack_patch_p2_cell_params_v1,
    patch_p2_cell_params_digest_schema_v1,
    patch_p2_cell_params_digest_v1,
    patch_p2_cell_params_sha256_v1,
    unpack_patch_p2_cell_params_v1,
    validate_patch_p2_cell_params_v1,
)


ROOT = Path(__file__).resolve().parents[1]
HEADER = ROOT / "engine" / "perf" / "cuda_architecture_b_patch_p2_cell_params_v1.hpp"


def _config() -> dict[str, object]:
    return {
        "patch_scent_decay_per_step": 0.05,
        "patch_waste_decay_per_step": 0.02,
        "patch_scent_presence_emit": 0.01,
        "patch_waste_presence_emit": 0.005,
        "patch_waste_density_emit_scale": 0.125,
        "patch_scent_max": 1.0,
        "patch_waste_max": 1.0,
        "patch_food_amount_max": 1.0,
        "patch_food_regrow_per_step": 0.02,
        "patch_food_regrow_use_fertility": True,
        "patch_food_regrow_cap_by_fertility": False,
        "patch_food_presence_threshold": 0.5,
        "patch_density_radius": 1,
        "patch_density_include_self": True,
        "patch_density_normalize": False,
        "patch_world_waste_consequence_enable": True,
        "patch_waste_regrowth_penalty_enable": True,
        "patch_waste_regrowth_penalty_scale": 0.5,
        "patch_waste_regrowth_penalty_max": 0.8,
        "patch_density_regrowth_penalty_enable": True,
        "patch_density_regrowth_penalty_scale": 0.25,
        "patch_density_regrowth_penalty_max": 0.7,
        "patch_regrowth_min_multiplier": 0.1,
        "patch_food_amount_initial_food_cell_value": 1.0,
        "patch_food_gather_amount": 1.0,
        "patch_scent_move_emit": 0.02,
        "patch_waste_move_emit": 0.03,
        "patch_scent_gather_emit": 0.05,
        "patch_waste_consume_emit": 0.05,
        "patch_movement_cost_apply_enable": True,
        "patch_movement_cost_energy_scale": 1.0,
        "patch_movement_cost_min": 0.0,
        "patch_movement_cost_max": 4.0,
    }


def test_cpp_pod_contract_compiles_with_frozen_size_alignment_and_offsets() -> None:
    compiler = shutil.which("g++") or shutil.which("c++")
    if compiler is None:
        pytest.skip("a C++ compiler is not available")
    source = """
#include "engine/perf/cuda_architecture_b_patch_p2_cell_params_v1.hpp"
using namespace eiel::cuda_architecture_b::patch_p2_cell_params_v1;
static_assert(sizeof(PatchP2CellParamsV1) == 160);
static_assert(alignof(PatchP2CellParamsV1) == 16);
static_assert(offsetof(PatchP2CellParamsV1, patch_scent_presence_emit) == 72);
static_assert(offsetof(PatchP2CellParamsV1, patch_movement_cost_max) == 152);
int main() { return 0; }
"""
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-fsyntax-only",
            "-I",
            str(ROOT),
            "-x",
            "c++",
            "-",
        ],
        input=source,
        text=True,
        check=True,
        capture_output=True,
    )


def test_python_layout_is_exactly_the_cxx_little_endian_layout() -> None:
    assert HEADER.is_file()
    assert CANONICAL_STRUCT_V1.format == "<IIq13fI11d"
    assert CANONICAL_STRUCT_V1.size == RECORD_BYTES_V1 == 160
    assert RECORD_ALIGNMENT_V1 == 16
    assert FIELD_OFFSETS_V1 == {
        "abi_version": 0,
        "flags": 4,
        "patch_density_radius": 8,
        **{name: 16 + index * 4 for index, name in enumerate(FLOAT32_FIELDS_V1)},
        "reserved_zero": 68,
        **{name: 72 + index * 8 for index, name in enumerate(FLOAT64_FIELDS_V1)},
    }


def test_config_normalization_roundtrips_exact_canonical_bytes() -> None:
    params = normalize_patch_p2_cell_params_v1(_config())
    packed = pack_patch_p2_cell_params_v1(params)
    unpacked = unpack_patch_p2_cell_params_v1(packed)

    assert unpacked == params
    assert pack_patch_p2_cell_params_v1(unpacked) == packed
    assert pack_patch_p2_cell_params_v1(_config()) == packed
    assert len(packed) == 160
    assert struct.unpack_from("<I", packed, 0)[0] == ABI_VERSION_V1
    assert struct.unpack_from("<I", packed, 4)[0] == params.flags
    assert struct.unpack_from("<q", packed, 8)[0] == 1
    assert struct.unpack_from("<I", packed, 68)[0] == 0
    assert struct.unpack_from("<d", packed, 72)[0] == 0.01
    assert struct.unpack_from("<d", packed, 152)[0] == 4.0


def test_normalization_matches_host_clamps_and_float32_rounding() -> None:
    config = _config()
    config.update(
        patch_scent_decay_per_step=-2.0,
        patch_waste_decay_per_step=2.0,
        patch_waste_density_emit_scale=-4.0,
        patch_food_presence_threshold=-3.0,
        patch_waste_regrowth_penalty_max=2.0,
        patch_density_regrowth_penalty_max=-1.0,
        patch_regrowth_min_multiplier=1.5,
    )
    params = normalize_patch_p2_cell_params_v1(config)

    assert params.patch_scent_decay_factor == 1.0
    assert params.patch_waste_decay_factor == 0.0
    assert params.patch_waste_density_emit_scale == 0.0
    assert params.patch_food_presence_threshold == 0.0
    assert params.patch_waste_regrowth_penalty_max == 1.0
    assert params.patch_density_regrowth_penalty_max == 0.0
    assert params.patch_regrowth_min_multiplier == 1.0
    expected_f32 = struct.unpack("<f", struct.pack("<f", 0.95))[0]
    assert (
        normalize_patch_p2_cell_params_v1(_config()).patch_scent_decay_factor
        == expected_f32
    )


@pytest.mark.parametrize(
    "field,bit",
    [
        (
            "patch_world_waste_consequence_enable",
            FLAG_WORLD_WASTE_CONSEQUENCE_ENABLE_V1,
        ),
        ("patch_movement_cost_apply_enable", FLAG_MOVEMENT_COST_APPLY_ENABLE_V1),
    ],
)
def test_both_complete_p2_profile_flags_are_required(field: str, bit: int) -> None:
    config = _config()
    config[field] = False
    with pytest.raises(PatchP2CellParamsError, match="profile flags must be true"):
        normalize_patch_p2_cell_params_v1(config)

    params = normalize_patch_p2_cell_params_v1(_config())
    assert params.flags & REQUIRED_P2_PROFILE_FLAGS_V1 == REQUIRED_P2_PROFILE_FLAGS_V1
    with pytest.raises(PatchP2CellParamsError, match="profile flags must be true"):
        validate_patch_p2_cell_params_v1(replace(params, flags=params.flags & ~bit))


def test_reserved_flag_bits_and_reserved_storage_are_rejected() -> None:
    params = normalize_patch_p2_cell_params_v1(_config())
    with pytest.raises(PatchP2CellParamsError, match="reserved bits"):
        pack_patch_p2_cell_params_v1(replace(params, flags=params.flags | (1 << 31)))
    with pytest.raises(PatchP2CellParamsError, match="reserved storage"):
        pack_patch_p2_cell_params_v1(replace(params, reserved_zero=1))

    packed = bytearray(pack_patch_p2_cell_params_v1(params))
    struct.pack_into("<I", packed, 4, params.flags | (1 << 8))
    with pytest.raises(PatchP2CellParamsError, match="reserved bits"):
        unpack_patch_p2_cell_params_v1(packed)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_source_and_wire_values_are_rejected(value: float) -> None:
    config = _config()
    config["patch_scent_presence_emit"] = value
    with pytest.raises(PatchP2CellParamsError, match="must be finite"):
        normalize_patch_p2_cell_params_v1(config)

    packed = bytearray(pack_patch_p2_cell_params_v1(_config()))
    struct.pack_into("<d", packed, FIELD_OFFSETS_V1["patch_scent_presence_emit"], value)
    with pytest.raises(PatchP2CellParamsError, match="must be finite"):
        unpack_patch_p2_cell_params_v1(packed)


def test_digest_schema_hashes_only_validated_canonical_record_bytes() -> None:
    packed = pack_patch_p2_cell_params_v1(_config())
    expected = sha256(packed).hexdigest()
    digest = patch_p2_cell_params_digest_v1(packed)
    schema = patch_p2_cell_params_digest_schema_v1()

    assert patch_p2_cell_params_sha256_v1(_config()) == expected
    assert digest == {
        "schema": DIGEST_SCHEMA_V1,
        "algorithm": "sha256",
        "record_abi": ABI_SCHEMA_V1,
        "record_bytes": 160,
        "sha256": expected,
    }
    assert schema["schema"] == DIGEST_SCHEMA_V1
    assert schema["canonical_input"] == (
        "one validated 160-byte little-endian ABI v1 record"
    )
    assert schema["field_offsets"] == FIELD_OFFSETS_V1


def test_signed_zero_is_canonicalized_before_digesting() -> None:
    positive = _config()
    negative = _config()
    negative["patch_scent_move_emit"] = -0.0
    negative["patch_waste_density_emit_scale"] = -0.0

    assert pack_patch_p2_cell_params_v1(negative) == pack_patch_p2_cell_params_v1(
        positive | {
            "patch_scent_move_emit": 0.0,
            "patch_waste_density_emit_scale": 0.0,
        }
    )
    assert patch_p2_cell_params_sha256_v1(negative) == (
        patch_p2_cell_params_sha256_v1(
            positive | {
                "patch_scent_move_emit": 0.0,
                "patch_waste_density_emit_scale": 0.0,
            }
        )
    )


def test_negative_movement_cost_bounds_are_rejected() -> None:
    config = _config()
    config["patch_movement_cost_min"] = -2.0
    config["patch_movement_cost_max"] = -1.0

    with pytest.raises(PatchP2CellParamsError, match="must be nonnegative"):
        normalize_patch_p2_cell_params_v1(config)


def test_wrong_size_version_fields_and_boolean_types_fail_closed() -> None:
    with pytest.raises(PatchP2CellParamsError, match="bytes-like"):
        unpack_patch_p2_cell_params_v1([0] * 160)  # type: ignore[arg-type]
    with pytest.raises(PatchP2CellParamsError, match="exactly 160 bytes"):
        unpack_patch_p2_cell_params_v1(b"\0" * 159)

    packed = bytearray(pack_patch_p2_cell_params_v1(_config()))
    struct.pack_into("<I", packed, 0, 2)
    with pytest.raises(PatchP2CellParamsError, match="unsupported.*version"):
        unpack_patch_p2_cell_params_v1(packed)

    config = _config()
    config.pop("patch_scent_max")
    with pytest.raises(PatchP2CellParamsError, match="field mismatch"):
        normalize_patch_p2_cell_params_v1(config)

    config = _config()
    config["patch_density_include_self"] = 1
    with pytest.raises(PatchP2CellParamsError, match="exact bool"):
        normalize_patch_p2_cell_params_v1(config)
