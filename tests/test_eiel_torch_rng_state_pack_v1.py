from __future__ import annotations

from hashlib import sha256

import pytest
import torch

import engine.perf.cuda_architecture_b_torch_rng_state_pack_v1 as state_pack
from engine.perf.cuda_architecture_b_torch_rng_state_pack_v1 import (
    TORCH_CPU_RNG_STATE_ABI_FINGERPRINT_SHA256_V1,
    TORCH_CPU_RNG_STATE_ABI_SCHEMA_VERSION_V1,
    TORCH_CPU_RNG_STATE_ENCODING_V1,
    TORCH_CPU_RNG_STATE_FULL_BYTES_V1,
    TORCH_CPU_RNG_STATE_PACKED_BYTES_V1,
    TorchCpuRngStateAbiError,
    pack_torch_cpu_rng_state_v1,
    require_torch_cpu_rng_state_abi_receipt_v1,
    torch_cpu_rng_state_abi_receipt_v1,
    unpack_torch_cpu_rng_state_v1,
    validate_torch_cpu_rng_state_abi_v1,
)


def _post_state(seed: int, *, offset: int = 0) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    torch.rand((11 + offset,), generator=generator, dtype=torch.float64)
    torch.randn((5 + offset,), generator=generator, dtype=torch.float64)
    return generator.get_state().clone()


def test_torch_cpu_rng_state_encoding_constants_and_layout_boundaries() -> None:
    assert (
        TORCH_CPU_RNG_STATE_ENCODING_V1
        == "torch_cpu_rng_state_5056_u32_middle_v1"
    )
    assert TORCH_CPU_RNG_STATE_FULL_BYTES_V1 == 5_056
    assert TORCH_CPU_RNG_STATE_PACKED_BYTES_V1 == 2_560

    state = _post_state(0x1234_5678_ABCD_EF01, offset=17)
    expected = torch.cat(
        (
            state[:24],
            state[24:5_016].reshape(624, 8)[:, :4].reshape(-1),
            state[5_016:],
        )
    )

    packed = pack_torch_cpu_rng_state_v1(state)
    assert torch.equal(packed, expected)
    assert torch.equal(unpack_torch_cpu_rng_state_v1(packed), state)


@pytest.mark.parametrize(
    "seed",
    [0, 1, (1 << 32) + 17, 0x1234_5678_ABCD_EF01, (1 << 64) - 1],
)
def test_torch_cpu_rng_state_exact_roundtrip_and_rand_replay(seed: int) -> None:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    torch.rand((13,), generator=generator, dtype=torch.float64)
    torch.randn((7,), generator=generator, dtype=torch.float64)
    state = generator.get_state().clone()

    packed = pack_torch_cpu_rng_state_v1(state)
    restored = unpack_torch_cpu_rng_state_v1(packed)

    assert tuple(packed.shape) == (2_560,)
    assert packed.dtype == torch.uint8
    assert packed.device.type == "cpu"
    assert packed.is_contiguous()
    assert restored.is_contiguous()
    assert torch.equal(restored, state)

    expected_rand = torch.rand((19,), generator=generator, dtype=torch.float64)
    expected_randn = torch.randn((21,), generator=generator, dtype=torch.float64)
    replay = torch.Generator(device="cpu")
    replay.set_state(restored)
    actual_rand = torch.rand((19,), generator=replay, dtype=torch.float64)
    actual_randn = torch.randn((21,), generator=replay, dtype=torch.float64)

    assert torch.equal(actual_rand, expected_rand)
    assert torch.equal(actual_randn, expected_randn)


def test_torch_cpu_rng_state_mixed_batch_preserves_leading_dimensions() -> None:
    states = torch.stack(
        [_post_state(10_000 + index, offset=index) for index in range(6)]
    ).reshape(2, 3, 5_056)
    noncontiguous_states = states.transpose(0, 1)
    assert not noncontiguous_states.is_contiguous()

    packed = pack_torch_cpu_rng_state_v1(noncontiguous_states)
    restored = unpack_torch_cpu_rng_state_v1(packed)

    assert tuple(packed.shape) == (3, 2, 2_560)
    assert packed.is_contiguous()
    assert restored.is_contiguous()
    assert torch.equal(restored, noncontiguous_states)

    noncontiguous_packed = pack_torch_cpu_rng_state_v1(states).transpose(0, 1)
    assert not noncontiguous_packed.is_contiguous()
    assert torch.equal(
        unpack_torch_cpu_rng_state_v1(noncontiguous_packed),
        noncontiguous_states,
    )


@pytest.mark.parametrize(
    ("state", "message"),
    [
        (torch.zeros((5_056,), dtype=torch.float32), "uint8 tensor"),
        (torch.tensor(0, dtype=torch.uint8), "rank at least 1"),
        (torch.zeros((5_055,), dtype=torch.uint8), "last extent 5056"),
        (torch.zeros((2, 5_057), dtype=torch.uint8), "last extent 5056"),
    ],
    ids=("dtype", "rank", "short-extent", "long-extent"),
)
def test_pack_torch_cpu_rng_state_rejects_malformed_input(
    state: torch.Tensor,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        pack_torch_cpu_rng_state_v1(state)


@pytest.mark.parametrize("offset", [28, 31, 5_012, 5_015])
def test_pack_torch_cpu_rng_state_rejects_nonzero_omitted_bytes(
    offset: int,
) -> None:
    state = torch.stack((_post_state(100), _post_state(101)))
    state[1, offset] = 1

    with pytest.raises(ValueError, match="nonzero omitted upper 32-bit"):
        pack_torch_cpu_rng_state_v1(state)


@pytest.mark.parametrize(
    ("packed", "message"),
    [
        (torch.zeros((2_560,), dtype=torch.bool), "uint8 tensor"),
        (torch.tensor(0, dtype=torch.uint8), "rank at least 1"),
        (torch.zeros((2_559,), dtype=torch.uint8), "last extent 2560"),
        (torch.zeros((2, 2_561), dtype=torch.uint8), "last extent 2560"),
    ],
    ids=("dtype", "rank", "short-extent", "long-extent"),
)
def test_unpack_torch_cpu_rng_state_rejects_malformed_input(
    packed: torch.Tensor,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        unpack_torch_cpu_rng_state_v1(packed)


def test_torch_cpu_rng_state_pack_rejects_unsupported_devices() -> None:
    with pytest.raises(ValueError, match="must be on CPU"):
        pack_torch_cpu_rng_state_v1(
            torch.empty((5_056,), dtype=torch.uint8, device="meta")
        )
    with pytest.raises(ValueError, match="must be on CPU or CUDA"):
        unpack_torch_cpu_rng_state_v1(
            torch.empty((2_560,), dtype=torch.uint8, device="meta")
        )


def test_torch_cpu_rng_state_empty_leading_dimension_round_trips() -> None:
    states = torch.empty((2, 0, 5_056), dtype=torch.uint8)

    packed = pack_torch_cpu_rng_state_v1(states)
    restored = unpack_torch_cpu_rng_state_v1(packed)

    assert tuple(packed.shape) == (2, 0, 2_560)
    assert packed.is_contiguous()
    assert tuple(restored.shape) == tuple(states.shape)
    assert restored.is_contiguous()
    assert torch.equal(restored, states)


@pytest.mark.parametrize(
    "draws",
    [0, 1, 2, 623, 624, 625, 9_415, 18_830],
)
def test_torch_cpu_rng_state_twist_boundary_corpus(draws: int) -> None:
    generator = torch.Generator(device="cpu").manual_seed(
        0x1234_5678_ABCD_EF01
    )
    torch.rand((draws,), generator=generator, dtype=torch.float32)
    state = generator.get_state()

    packed = pack_torch_cpu_rng_state_v1(state)
    restored = unpack_torch_cpu_rng_state_v1(packed)

    assert torch.equal(restored, state)
    expected = torch.rand((1_003,), generator=generator, dtype=torch.float32)
    replay = torch.Generator(device="cpu")
    replay.set_state(restored)
    assert torch.equal(
        torch.rand((1_003,), generator=replay, dtype=torch.float32),
        expected,
    )


@pytest.mark.parametrize(
    ("dtype", "normal_draws"),
    [
        (torch.float32, 1),
        (torch.float32, 2),
        (torch.float32, 3),
        (torch.float64, 1),
        (torch.float64, 2),
        (torch.float64, 3),
    ],
    ids=("f32-odd", "f32-even", "f32-cached", "f64-odd", "f64-even", "f64-cached"),
)
def test_torch_cpu_rng_state_normal_cache_corpus(
    dtype: torch.dtype,
    normal_draws: int,
) -> None:
    generator = torch.Generator(device="cpu").manual_seed(0xCAFE_BABE)
    torch.rand((625,), generator=generator, dtype=dtype)
    torch.randn((normal_draws,), generator=generator, dtype=dtype)
    state = generator.get_state()

    restored = unpack_torch_cpu_rng_state_v1(
        pack_torch_cpu_rng_state_v1(state)
    )
    expected = torch.randn((17,), generator=generator, dtype=dtype)
    replay = torch.Generator(device="cpu")
    replay.set_state(restored)

    assert torch.equal(restored, state)
    assert torch.equal(
        torch.randn((17,), generator=replay, dtype=dtype),
        expected,
    )


def test_torch_cpu_rng_state_mixed_uniform_normal_corpus() -> None:
    generator = torch.Generator(device="cpu").manual_seed((1 << 64) - 1)
    torch.rand((623,), generator=generator, dtype=torch.float32)
    torch.randn((7,), generator=generator, dtype=torch.float64)
    torch.rand((9_415,), generator=generator, dtype=torch.float64)
    torch.randn((8,), generator=generator, dtype=torch.float32)
    state = generator.get_state()

    restored = unpack_torch_cpu_rng_state_v1(
        pack_torch_cpu_rng_state_v1(state)
    )

    assert torch.equal(restored, state)


@pytest.mark.parametrize(
    ("offset", "value", "message"),
    [
        (8, 0, "seeded/left/next"),
        (12, 0, "seeded/left/next"),
        (16, 0xFF, "seeded/left/next"),
        (5_040, 2, "normal-cache validity"),
        (5_044, 1, "nonzero ABI padding"),
        (5_052, 2, "normal-cache validity"),
        (5_055, 1, "nonzero ABI padding"),
    ],
)
def test_pack_torch_cpu_rng_state_rejects_noncanonical_metadata(
    offset: int,
    value: int,
    message: str,
) -> None:
    state = torch.Generator(device="cpu").manual_seed(77).get_state()
    state[offset] = value

    with pytest.raises(ValueError, match=message):
        pack_torch_cpu_rng_state_v1(state)


def test_torch_cpu_rng_state_abi_receipt_is_exact_and_deterministic() -> None:
    validate_torch_cpu_rng_state_abi_v1()
    first = torch_cpu_rng_state_abi_receipt_v1()
    second = torch_cpu_rng_state_abi_receipt_v1()

    assert first == second
    assert first["abi_schema_version"] == TORCH_CPU_RNG_STATE_ABI_SCHEMA_VERSION_V1
    assert (
        first["abi_fingerprint_sha256"]
        == TORCH_CPU_RNG_STATE_ABI_FINGERPRINT_SHA256_V1
    )
    assert first["torch_version"] == "2.10.0+cu128"
    assert first["byteorder"] == "little"
    assert first["full_state_bytes"] == 5_056
    assert first["packed_state_bytes"] == 2_560
    assert first["portable_checkpoint_format"] is False
    assert first["execution_tape_encoding"] is True
    assert require_torch_cpu_rng_state_abi_receipt_v1(first) == first

    state = _post_state(88)
    packed_digest = sha256(
        pack_torch_cpu_rng_state_v1(state).numpy().tobytes()
    ).hexdigest()
    assert packed_digest == (
        "6f7414299ee073d16b2cec3abb9924493d1230ec93b0170725384e9f193e51e1"
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("byteorder", "big"),
        ("torch_version", "2.11.0"),
        ("full_state_bytes", 5_057),
        ("abi_fingerprint_sha256", "0" * 64),
    ],
)
def test_torch_cpu_rng_state_abi_receipt_rejects_mismatch(
    field: str,
    value: object,
) -> None:
    receipt = torch_cpu_rng_state_abi_receipt_v1()
    receipt[field] = value

    with pytest.raises(TorchCpuRngStateAbiError, match="receipt mismatch"):
        require_torch_cpu_rng_state_abi_receipt_v1(receipt)


def test_torch_cpu_rng_state_runtime_abi_rejects_other_build(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed = state_pack._runtime_abi_observation_v1()
    observed["torch_git_version"] = "0" * 40
    monkeypatch.setattr(
        state_pack,
        "_runtime_abi_observation_v1",
        lambda: dict(observed),
    )
    validate_torch_cpu_rng_state_abi_v1.cache_clear()
    try:
        with pytest.raises(TorchCpuRngStateAbiError, match="torch_git_version"):
            validate_torch_cpu_rng_state_abi_v1()
    finally:
        validate_torch_cpu_rng_state_abi_v1.cache_clear()
