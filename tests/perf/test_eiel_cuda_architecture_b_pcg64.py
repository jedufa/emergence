from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
import torch

from perf.cuda_architecture_b_pcg64 import (
    CUDA_PCG64_CAPABILITIES,
    pcg64_state_tensors,
    pcg64_uniform_float32_cuda,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize("seed", [0, 101, 202, 303, 910_202_607])
def test_cuda_pcg64_float32_draws_and_state_are_numpy_exact(seed: int, tmp_path: Path) -> None:
    bitgen = np.random.PCG64(seed)
    initial = copy.deepcopy(bitgen.state)
    expected = np.random.Generator(bitgen).random(17, dtype=np.float32)
    expected_state = copy.deepcopy(bitgen.state)
    state_words, buffer_words = pcg64_state_tensors(initial)

    result = pcg64_uniform_float32_cuda(
        state_words=state_words,
        buffer_words=buffer_words,
        count=17,
        build_directory=tmp_path / "build",
    )

    assert torch.equal(result.values.cpu().view(torch.int32), torch.from_numpy(expected.view(np.int32)))
    assert result.state_dict() == expected_state
    assert result.state_words.data_ptr() == state_words.data_ptr()
    assert result.buffer_words.data_ptr() == buffer_words.data_ptr()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cuda_pcg64_odd_even_continuation_preserves_buffer(tmp_path: Path) -> None:
    bitgen = np.random.PCG64(303)
    initial = copy.deepcopy(bitgen.state)
    generator = np.random.Generator(bitgen)
    expected_first = generator.random(7, dtype=np.float32)
    midpoint = copy.deepcopy(bitgen.state)
    expected_second = generator.random(10, dtype=np.float32)
    final = copy.deepcopy(bitgen.state)
    state_words, buffer_words = pcg64_state_tensors(initial)

    first = pcg64_uniform_float32_cuda(
        state_words=state_words,
        buffer_words=buffer_words,
        count=7,
        build_directory=tmp_path / "build",
    )
    assert torch.equal(first.values.cpu().view(torch.int32), torch.from_numpy(expected_first.view(np.int32)))
    assert first.state_dict() == midpoint

    second = pcg64_uniform_float32_cuda(
        state_words=first.state_words,
        buffer_words=first.buffer_words,
        count=10,
        build_directory=tmp_path / "build",
    )
    assert torch.equal(second.values.cpu().view(torch.int32), torch.from_numpy(expected_second.view(np.int32)))
    assert second.state_dict() == final


def test_cuda_pcg64_capability_is_bounded_and_state_validation_is_fail_closed() -> None:
    assert CUDA_PCG64_CAPABILITIES["numpy_pcg64_float32_draws"] == "bit_exact"
    assert CUDA_PCG64_CAPABILITIES["promotion_qualified"] is False
    with pytest.raises(ValueError, match="only NumPy PCG64"):
        pcg64_state_tensors(
            {"bit_generator": "PCG64DXSM", "state": {"state": 1, "inc": 3}},
            device="cpu",
        )
