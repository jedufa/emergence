from __future__ import annotations

import random

import pytest

from perf.cuda_architecture_b_generation_birth_state import (
    CudaGenerationBirthStateError,
    _python_random_state,
)
from perf.cuda_architecture_b_lifecycle import pack_cpython_random_state_cuda


def test_python_random_checkpoint_payload_restores_absolute_raw_draw_count() -> None:
    payload = {
        "version": 3,
        "state": [0] * 624,
        "index": 0,
        "gauss_next": None,
        "raw_draws": 77,
    }
    state, raw_draws = _python_random_state(
        {"native_runtime_python_random_state": payload}
    )
    assert state == (3, tuple([0] * 625), None)
    assert raw_draws == 77

    legacy_state, legacy_raw_draws = _python_random_state(
        {
            "generation": 0,
            "native_runtime_python_random_state": {
                key: value for key, value in payload.items() if key != "raw_draws"
            },
        }
    )
    assert legacy_state == state
    assert legacy_raw_draws == 0

    with pytest.raises(
        CudaGenerationBirthStateError,
        match="nonzero-generation.*omitted raw draw count",
    ):
        _python_random_state(
            {
                "generation": 6,
                "native_runtime_python_random_state": {
                    key: value for key, value in payload.items() if key != "raw_draws"
                },
            }
        )

    payload["raw_draws"] = -1
    with pytest.raises(CudaGenerationBirthStateError, match="raw draw count"):
        _python_random_state({"native_runtime_python_random_state": payload})


def test_pack_cpython_random_state_preserves_absolute_raw_draw_count() -> None:
    state = random.Random(17).getstate()
    packed = pack_cpython_random_state_cuda(state, device="cpu", raw_draws=41)
    assert int(packed["python_random_raw_draws"][0]) == 41

    with pytest.raises(ValueError, match="non-negative integer"):
        pack_cpython_random_state_cuda(state, device="cpu", raw_draws=-1)
    with pytest.raises(ValueError, match="non-negative integer"):
        pack_cpython_random_state_cuda(state, device="cpu", raw_draws=True)
    with pytest.raises(ValueError, match="raw draw base"):
        pack_cpython_random_state_cuda(state, device="cpu", raw_draw_base=-1)
    with pytest.raises(ValueError, match="raw draw base"):
        pack_cpython_random_state_cuda(
            state,
            device="cpu",
            raw_draws=40,
            raw_draw_base=41,
        )


def test_gaussian_tape_uses_absolute_raw_draw_index_after_checkpoint() -> None:
    source = random.Random(23)
    state = source.getstate()
    expected_primary = source.gauss(0.0, 1.0)
    expected_secondary = source.getstate()[2]

    packed = pack_cpython_random_state_cuda(
        state,
        device="cpu",
        raw_draws=37,
        gauss_tape_raw_capacity=8,
    )

    primary = packed["python_random_gauss_primary_tape"]
    secondary = packed["python_random_gauss_secondary_tape"]
    assert primary.shape == secondary.shape == (45,)
    assert float(primary[37]) == expected_primary
    assert float(secondary[37]) == expected_secondary


def test_gaussian_tape_can_be_staged_at_checkpoint_raw_draw_base() -> None:
    source = random.Random(23)
    state = source.getstate()
    expected_primary = source.gauss(0.0, 1.0)
    expected_secondary = source.getstate()[2]

    packed = pack_cpython_random_state_cuda(
        state,
        device="cpu",
        raw_draws=37,
        raw_draw_base=37,
        gauss_tape_raw_capacity=8,
    )

    primary = packed["python_random_gauss_primary_tape"]
    secondary = packed["python_random_gauss_secondary_tape"]
    assert primary.shape == secondary.shape == (8,)
    assert float(primary[0]) == expected_primary
    assert float(secondary[0]) == expected_secondary
