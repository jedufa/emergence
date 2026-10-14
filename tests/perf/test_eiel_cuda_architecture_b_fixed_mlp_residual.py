from __future__ import annotations

import copy
from dataclasses import fields
import inspect
from pathlib import Path

import numpy as np
import pytest
import torch

from perf.cuda_architecture_b_fixed_mlp import (
    CUDA_FIXED_MLP_CAPABILITIES,
    CudaFixedMlpResult,
    fixed_mlp_refill_counted_cuda,
    fixed_mlp_refill_counted_with_residual_cuda,
)


OBSERVATION_WIDTH = 27
HIDDEN_SIZE = 4
ACTION_COUNT = 7
CAPACITY = 3
ROW_COUNT = 2
CACHE_WIDTH = 1

pytestmark = pytest.mark.filterwarnings("ignore:.*CUDA.*")


def _cuda_state() -> dict[str, torch.Tensor]:
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(19_072_026)
    packed_width = (
        HIDDEN_SIZE * OBSERVATION_WIDTH
        + HIDDEN_SIZE
        + HIDDEN_SIZE * HIDDEN_SIZE
        + HIDDEN_SIZE
        + ACTION_COUNT * HIDDEN_SIZE
        + ACTION_COUNT
    )
    return {
        "observations": torch.randn(
            (ROW_COUNT, OBSERVATION_WIDTH),
            generator=generator,
            device=device,
            dtype=torch.float32,
        ).contiguous(),
        "slots": torch.tensor([2, 0], device=device, dtype=torch.int32),
        "active_count": torch.tensor([ROW_COUNT], device=device, dtype=torch.int64),
        "mlp_pack": (
            torch.randn(
                (CAPACITY, packed_width),
                generator=generator,
                device=device,
                dtype=torch.float32,
            )
            * np.float32(0.125)
        ).contiguous(),
        "drives": torch.rand(
            (CAPACITY, 4),
            generator=generator,
            device=device,
            dtype=torch.float32,
        ).contiguous(),
        "attack_bias": torch.tensor(
            [-0.25, 0.0, 0.375], device=device, dtype=torch.float32
        ),
        "action_temperature": torch.tensor(
            [0.75, 1.0, 1.25], device=device, dtype=torch.float32
        ),
        "drive_logit_gain": torch.tensor(
            [0.5, 0.75, 1.0], device=device, dtype=torch.float32
        ),
        "slot_tokens": torch.tensor([70, 80, 90], device=device, dtype=torch.int64),
        "action_cache_actions": torch.full(
            (CAPACITY, CACHE_WIDTH), -1, device=device, dtype=torch.int32
        ),
        "action_cache_token": torch.full(
            (CAPACITY,), -1, device=device, dtype=torch.int64
        ),
        "action_cache_cursor": torch.full(
            (CAPACITY,), -1, device=device, dtype=torch.int32
        ),
        "action_cache_next": torch.full(
            (CAPACITY,), -1, device=device, dtype=torch.int32
        ),
    }


def _clone_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: tensor.clone() for name, tensor in state.items()}


def _pcg64_state() -> dict[str, object]:
    return copy.deepcopy(np.random.PCG64(303).state)


def _common_arguments(state: dict[str, torch.Tensor], build: Path) -> dict[str, object]:
    return {
        **state,
        "hidden_size": HIDDEN_SIZE,
        "action_count": ACTION_COUNT,
        "cache_width": CACHE_WIDTH,
        "deterministic": False,
        "rng_seed": 0x1234_5678,
        "rng_counter": 901,
        "pcg64_state": _pcg64_state(),
        "build_directory": build,
    }


def _assert_result_bit_exact(left: CudaFixedMlpResult, right: CudaFixedMlpResult) -> None:
    for field in fields(CudaFixedMlpResult):
        left_tensor = getattr(left, field.name)
        right_tensor = getattr(right, field.name)
        assert left_tensor.dtype == right_tensor.dtype, field.name
        assert tuple(left_tensor.shape) == tuple(right_tensor.shape), field.name
        assert torch.equal(left_tensor, right_tensor), field.name


def test_legacy_public_api_and_capability_receipt_remain_unchanged() -> None:
    assert "logit_residuals" not in inspect.signature(
        fixed_mlp_refill_counted_cuda
    ).parameters
    assert [field.name for field in fields(CudaFixedMlpResult)] == [
        "logits",
        "actions",
        "action_cache_actions",
        "action_cache_token",
        "action_cache_cursor",
        "action_cache_next",
        "next_rng_counter",
        "pcg64_state_words",
        "pcg64_buffer_words",
    ]
    assert CUDA_FIXED_MLP_CAPABILITIES == {
        "schema_version": "eiel_cuda_architecture_b_fixed_mlp_capabilities_v1",
        "decision": "ARCHB-CUDA-STAGE1A/FIXED-MLP-DEVICE-RESIDENT",
        "fixed_mlp_policy_resident": True,
        "policy_input_cuda": True,
        "policy_cache_cuda": True,
        "fixed_capacity_counted_active_rows": True,
        "device_resident_active_count": True,
        "host_sampling": False,
        "python_row_loop": False,
        "deterministic_argmax_parity": "exact_actions_bounded_float_logits",
        "stochastic_sampling": (
            "numpy_pcg64_gumbel_exact_or_explicit_splitmix64_prototype"
        ),
        "stochastic_cross_backend_parity": "exact_when_pcg64_state_is_provided",
        "gru_supported": False,
        "full_generation_resident": False,
        "promotion_qualified": False,
    }


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_zero_residual_is_bit_exact_to_legacy_outputs_state_and_rng(
    tmp_path: Path,
) -> None:
    source = _cuda_state()
    legacy = fixed_mlp_refill_counted_cuda(
        **_common_arguments(_clone_state(source), tmp_path / "build")
    )
    residual = fixed_mlp_refill_counted_with_residual_cuda(
        **_common_arguments(_clone_state(source), tmp_path / "build"),
        logit_residuals=torch.zeros(
            (ROW_COUNT, ACTION_COUNT), device="cuda", dtype=torch.float32
        ),
    )

    _assert_result_bit_exact(legacy, residual)
    assert legacy.pcg64_state_dict() == residual.pcg64_state_dict()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_controlled_residual_changes_logits_before_action_selection(
    tmp_path: Path,
) -> None:
    source = _cuda_state()
    legacy_arguments = _common_arguments(_clone_state(source), tmp_path / "build")
    legacy_arguments["deterministic"] = True
    legacy_arguments.pop("pcg64_state")
    legacy = fixed_mlp_refill_counted_cuda(**legacy_arguments)
    targets = torch.remainder(legacy.actions.to(torch.int64) + 1, ACTION_COUNT)
    residuals = torch.zeros_like(legacy.logits)
    row_indices = torch.arange(ROW_COUNT, device="cuda")
    deltas = (
        legacy.logits.max(dim=1).values
        - legacy.logits[row_indices, targets]
        + torch.ones(ROW_COUNT, device="cuda", dtype=torch.float32)
    )
    residuals[row_indices, targets] = deltas
    changed_arguments = _common_arguments(_clone_state(source), tmp_path / "build")
    changed_arguments["deterministic"] = True
    changed_arguments.pop("pcg64_state")
    changed = fixed_mlp_refill_counted_with_residual_cuda(
        **changed_arguments, logit_residuals=residuals
    )
    expected_logits = legacy.logits + residuals
    expected_actions = expected_logits.argmax(dim=1).to(torch.int32)

    assert torch.equal(changed.logits, expected_logits)
    assert torch.equal(changed.actions, expected_actions)
    assert torch.equal(changed.actions, targets.to(torch.int32))
    assert not torch.equal(changed.actions, legacy.actions)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize("damage", ("shape", "dtype", "device"))
def test_invalid_residual_shape_dtype_and_device_fail_closed(
    tmp_path: Path,
    damage: str,
) -> None:
    state = _cuda_state()
    residuals = torch.zeros(
        (ROW_COUNT, ACTION_COUNT), device="cuda", dtype=torch.float32
    )
    if damage == "shape":
        residuals = residuals[:, :-1].contiguous()
    elif damage == "dtype":
        residuals = residuals.to(torch.float64)
    else:
        residuals = residuals.cpu()

    with pytest.raises((RuntimeError, ValueError), match="logit_residuals"):
        fixed_mlp_refill_counted_with_residual_cuda(
            **_common_arguments(state, tmp_path / "build"),
            logit_residuals=residuals,
        )
