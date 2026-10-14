"""Route-pinned physical encoding for qualified Torch CPU RNG states."""

from __future__ import annotations

from functools import lru_cache
from hashlib import sha256
import json
import platform
import struct
import sys
from typing import Final, Mapping

import torch


TORCH_CPU_RNG_STATE_ENCODING_V1: Final[str] = (
    "torch_cpu_rng_state_5056_u32_middle_v1"
)
TORCH_CPU_RNG_STATE_FULL_BYTES_V1: Final[int] = 5_056
TORCH_CPU_RNG_STATE_PACKED_BYTES_V1: Final[int] = 2_560
TORCH_CPU_RNG_STATE_ABI_SCHEMA_VERSION_V1: Final[str] = (
    "eiel_torch_cpu_mt19937_state_2560_abi_v1"
)
TORCH_CPU_RNG_STATE_ABI_FINGERPRINT_SHA256_V1: Final[str] = (
    "dd3966ffb664fc6bb7b6c99a42436a8945e8b0fc4a0adf15238de16d07ddf5b4"
)

_LANE_BYTES: Final[int] = 8
_LOW_HALF_BYTES: Final[int] = 4
_HEAD_LANES: Final[int] = 3
_MIDDLE_LANES: Final[int] = 624
_TAIL_LANES: Final[int] = 5
_HEAD_BYTES: Final[int] = _HEAD_LANES * _LANE_BYTES
_MIDDLE_FULL_BYTES: Final[int] = _MIDDLE_LANES * _LANE_BYTES
_MIDDLE_PACKED_BYTES: Final[int] = _MIDDLE_LANES * _LOW_HALF_BYTES
_TAIL_FULL_OFFSET: Final[int] = _HEAD_BYTES + _MIDDLE_FULL_BYTES
_TAIL_PACKED_OFFSET: Final[int] = _HEAD_BYTES + _MIDDLE_PACKED_BYTES
_QUALIFIED_TORCH_VERSION: Final[str] = "2.10.0+cu128"
_QUALIFIED_TORCH_GIT_VERSION: Final[str] = (
    "449b1768410104d3ed79d3bcfe4ba1d65c7f22c0"
)
_QUALIFIED_TORCH_CUDA_BUILD: Final[str] = "12.8"
_QUALIFIED_TORCH_CXX11_ABI: Final[bool] = True
_QUALIFIED_TORCH_BUILD_CONFIG_SHA256: Final[str] = (
    "62a3328b6570991ff3dd369f547254c82ca716de947cd8a66374b52903f6793e"
)
_QUALIFIED_PLATFORM_MACHINE: Final[str] = "x86_64"
_GOLDEN_SEED: Final[int] = 0x1234_5678_ABCD_EF01
_GOLDEN_FLOAT32_UNIFORM_DRAWS: Final[int] = 9_415
_GOLDEN_STATE_SHA256: Final[str] = (
    "961fb072efe82dcf2bd9aed93642492b7d3503c66b06adf6771e910620151a1d"
)


class TorchCpuRngStateAbiError(RuntimeError):
    """The active Torch build does not match the qualified state ABI."""


def _expected_abi_descriptor_v1() -> dict[str, object]:
    return {
        "abi_schema_version": TORCH_CPU_RNG_STATE_ABI_SCHEMA_VERSION_V1,
        "codec_schema_version": TORCH_CPU_RNG_STATE_ENCODING_V1,
        "torch_version": _QUALIFIED_TORCH_VERSION,
        "torch_git_version": _QUALIFIED_TORCH_GIT_VERSION,
        "torch_cuda_build": _QUALIFIED_TORCH_CUDA_BUILD,
        "torch_cxx11_abi": _QUALIFIED_TORCH_CXX11_ABI,
        "torch_build_config_sha256": _QUALIFIED_TORCH_BUILD_CONFIG_SHA256,
        "platform_machine": _QUALIFIED_PLATFORM_MACHINE,
        "byteorder": "little",
        "uint64_bytes": 8,
        "uint32_bytes": 4,
        "full_state_bytes": TORCH_CPU_RNG_STATE_FULL_BYTES_V1,
        "packed_state_bytes": TORCH_CPU_RNG_STATE_PACKED_BYTES_V1,
        "leading_full_uint64_lanes": _HEAD_LANES,
        "middle_low_uint32_lanes": _MIDDLE_LANES,
        "trailing_full_uint64_lanes": _TAIL_LANES,
        "leading_full_bytes": _HEAD_BYTES,
        "trailing_full_byte_offset": _TAIL_FULL_OFFSET,
        "trailing_packed_byte_offset": _TAIL_PACKED_OFFSET,
        "golden_seed": _GOLDEN_SEED,
        "golden_float32_uniform_draws": _GOLDEN_FLOAT32_UNIFORM_DRAWS,
        "golden_state_sha256": _GOLDEN_STATE_SHA256,
        "source_identity": "torch.Generator(device=cpu).get_state",
        "physical_profile_identity": (
            "eiel_cuda_architecture_b_fixed_mlp_policy_post_state_v1"
        ),
        "route_binding": "execution_route_sha256",
    }


def _descriptor_sha256_v1(value: Mapping[str, object]) -> str:
    payload = json.dumps(
        dict(value), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return sha256(payload).hexdigest()


if (
    _descriptor_sha256_v1(_expected_abi_descriptor_v1())
    != TORCH_CPU_RNG_STATE_ABI_FINGERPRINT_SHA256_V1
):
    raise RuntimeError("Torch CPU RNG ABI descriptor fingerprint is stale")


def _runtime_abi_observation_v1() -> dict[str, object]:
    generator = torch.Generator(device="cpu").manual_seed(_GOLDEN_SEED)
    torch.rand(
        (_GOLDEN_FLOAT32_UNIFORM_DRAWS,),
        generator=generator,
        dtype=torch.float32,
    )
    golden_state = generator.get_state()
    return {
        "torch_version": str(torch.__version__),
        "torch_git_version": str(torch.version.git_version),
        "torch_cuda_build": str(torch.version.cuda),
        "torch_cxx11_abi": bool(torch.compiled_with_cxx11_abi()),
        "torch_build_config_sha256": sha256(
            torch.__config__.show().encode("utf-8")
        ).hexdigest(),
        "platform_machine": platform.machine(),
        "byteorder": sys.byteorder,
        "uint64_bytes": struct.calcsize("Q"),
        "uint32_bytes": struct.calcsize("I"),
        "full_state_bytes": int(golden_state.numel()),
        "golden_state_sha256": sha256(
            golden_state.numpy().tobytes()
        ).hexdigest(),
    }


@lru_cache(maxsize=1)
def validate_torch_cpu_rng_state_abi_v1() -> None:
    """Fail closed unless the active process matches the qualified ABI."""

    expected = _expected_abi_descriptor_v1()
    observed = _runtime_abi_observation_v1()
    mismatches = [
        f"{name}:expected={expected[name]!r},actual={actual!r}"
        for name, actual in observed.items()
        if type(actual) is not type(expected[name]) or actual != expected[name]
    ]
    if mismatches:
        raise TorchCpuRngStateAbiError(
            "unsupported Torch CPU RNG state ABI: " + "; ".join(mismatches)
        )


def torch_cpu_rng_state_abi_receipt_v1() -> dict[str, object]:
    """Return the validated route-pinned ABI and codec receipt."""

    validate_torch_cpu_rng_state_abi_v1()
    return {
        **_expected_abi_descriptor_v1(),
        "abi_fingerprint_sha256": (
            TORCH_CPU_RNG_STATE_ABI_FINGERPRINT_SHA256_V1
        ),
        "portable_checkpoint_format": False,
        "execution_tape_encoding": True,
    }


def require_torch_cpu_rng_state_abi_receipt_v1(
    value: object,
) -> dict[str, object]:
    """Validate route metadata before accepting compact state rows."""

    expected = torch_cpu_rng_state_abi_receipt_v1()
    if not isinstance(value, Mapping):
        raise TorchCpuRngStateAbiError("Torch CPU RNG ABI receipt must be a mapping")
    if set(value) != set(expected):
        raise TorchCpuRngStateAbiError(
            "Torch CPU RNG ABI receipt fields do not match the qualified schema"
        )
    mismatches = [
        f"{name}:expected={expected_value!r},actual={value[name]!r}"
        for name, expected_value in expected.items()
        if type(value[name]) is not type(expected_value)
        or value[name] != expected_value
    ]
    if mismatches:
        raise TorchCpuRngStateAbiError(
            "Torch CPU RNG ABI receipt mismatch: " + "; ".join(mismatches)
        )
    return dict(expected)


def _require_uint8_state_tensor(
    value: object,
    *,
    label: str,
    extent: int,
    allowed_devices: tuple[str, ...],
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.dtype != torch.uint8:
        raise ValueError(f"{label} must be a uint8 tensor")
    if value.ndim < 1:
        raise ValueError(f"{label} must have rank at least 1")
    if int(value.shape[-1]) != extent:
        raise ValueError(f"{label} must have last extent {extent}")
    if value.device.type not in allowed_devices:
        devices = " or ".join(name.upper() for name in allowed_devices)
        raise ValueError(f"{label} must be on {devices}")
    return value


def _require_canonical_serialized_metadata_v1(state: torch.Tensor) -> None:
    rows = state.contiguous().numpy().reshape(
        -1, TORCH_CPU_RNG_STATE_FULL_BYTES_V1
    )
    for row_index, row in enumerate(rows):
        _, left, seeded, next_index = struct.unpack_from("<QiiQ", row, 0)
        current_position = (
            (left == 1 and next_index == 0)
            or (
                1 <= left <= 624
                and 1 <= next_index <= 624
                and left + next_index == 625
            )
        )
        double_normal_valid = struct.unpack_from("<i", row, 5_040)[0]
        float_normal_valid = int(row[5_052])
        if seeded != 1 or not current_position:
            raise ValueError(
                f"Torch CPU RNG state row {row_index} has noncanonical "
                "seeded/left/next metadata"
            )
        if double_normal_valid not in (0, 1) or float_normal_valid not in (0, 1):
            raise ValueError(
                f"Torch CPU RNG state row {row_index} has noncanonical "
                "normal-cache validity metadata"
            )
        if bool(row[5_044:5_048].any()) or bool(row[5_053:5_056].any()):
            raise ValueError(
                f"Torch CPU RNG state row {row_index} has nonzero ABI padding"
            )


def pack_torch_cpu_rng_state_v1(state: torch.Tensor) -> torch.Tensor:
    """Pack canonical 5,056-byte Torch CPU RNG state rows losslessly."""

    validate_torch_cpu_rng_state_abi_v1()
    state = _require_uint8_state_tensor(
        state,
        label="Torch CPU RNG state",
        extent=TORCH_CPU_RNG_STATE_FULL_BYTES_V1,
        allowed_devices=("cpu",),
    )
    state = state.contiguous()
    _require_canonical_serialized_metadata_v1(state)
    leading_shape = tuple(state.shape[:-1])
    middle = state[..., _HEAD_BYTES:_TAIL_FULL_OFFSET].reshape(
        *leading_shape,
        _MIDDLE_LANES,
        _LANE_BYTES,
    )
    if bool(torch.any(middle[..., _LOW_HALF_BYTES:])):
        raise ValueError(
            "Torch CPU RNG state has nonzero omitted upper 32-bit lane bytes"
        )
    middle_low = middle[..., :_LOW_HALF_BYTES].reshape(
        *leading_shape,
        _MIDDLE_PACKED_BYTES,
    )
    return torch.cat(
        (
            state[..., :_HEAD_BYTES],
            middle_low,
            state[..., _TAIL_FULL_OFFSET:],
        ),
        dim=-1,
    ).contiguous()


def unpack_torch_cpu_rng_state_v1(packed: torch.Tensor) -> torch.Tensor:
    """Reconstruct canonical 5,056-byte state rows on CPU or CUDA."""

    validate_torch_cpu_rng_state_abi_v1()
    packed = _require_uint8_state_tensor(
        packed,
        label="Packed Torch CPU RNG state",
        extent=TORCH_CPU_RNG_STATE_PACKED_BYTES_V1,
        allowed_devices=("cpu", "cuda"),
    )
    leading_shape = tuple(packed.shape[:-1])
    middle = torch.zeros(
        (*leading_shape, _MIDDLE_LANES, _LANE_BYTES),
        dtype=torch.uint8,
        device=packed.device,
    )
    middle[..., :_LOW_HALF_BYTES].copy_(
        packed[..., _HEAD_BYTES:_TAIL_PACKED_OFFSET].reshape(
            *leading_shape,
            _MIDDLE_LANES,
            _LOW_HALF_BYTES,
        )
    )
    return torch.cat(
        (
            packed[..., :_HEAD_BYTES],
            middle.reshape(*leading_shape, _MIDDLE_FULL_BYTES),
            packed[..., _TAIL_PACKED_OFFSET:],
        ),
        dim=-1,
    ).contiguous()


__all__ = [
    "TORCH_CPU_RNG_STATE_ABI_FINGERPRINT_SHA256_V1",
    "TORCH_CPU_RNG_STATE_ABI_SCHEMA_VERSION_V1",
    "TORCH_CPU_RNG_STATE_ENCODING_V1",
    "TORCH_CPU_RNG_STATE_FULL_BYTES_V1",
    "TORCH_CPU_RNG_STATE_PACKED_BYTES_V1",
    "TorchCpuRngStateAbiError",
    "pack_torch_cpu_rng_state_v1",
    "require_torch_cpu_rng_state_abi_receipt_v1",
    "torch_cpu_rng_state_abi_receipt_v1",
    "unpack_torch_cpu_rng_state_v1",
    "validate_torch_cpu_rng_state_abi_v1",
]
