from __future__ import annotations

import pytest
import torch

from engine.option_d import (
    DrawOperation,
    DrawPredicate,
    DrawSchedule,
    OptionDContractError,
    RNGStateBundle,
    execute_cpu,
    state_from_seed,
)
from engine.option_d.cuda import execute_cuda


SOURCE = "test-current-source-cuda-raw"
CUDA_AVAILABLE = torch.cuda.is_available()


def _bundle(*seeds: object) -> RNGStateBundle:
    return RNGStateBundle(
        tuple(state_from_seed(seed) for seed in seeds),
        source_binding=SOURCE,
    )


def _assert_cuda_equals_cpu(bundle: RNGStateBundle, schedule: DrawSchedule) -> None:
    before = bundle.digest()
    expected = execute_cpu(bundle, schedule)
    first = execute_cuda(bundle, schedule)
    second = execute_cuda(bundle, schedule)

    assert bundle.digest() == before
    assert first.material == expected.material
    assert first.successor == expected.successor
    assert second.material == first.material
    assert second.successor == first.successor


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA unavailable")
@pytest.mark.parametrize("count", [0, 1, 623, 624, 625, 1249, 9415])
def test_raw_vectors_and_long_continuation(count: int) -> None:
    bundle = _bundle(42)
    schedule = DrawSchedule((tuple(DrawOperation.raw32() for _ in range(count)),))
    _assert_cuda_equals_cpu(bundle, schedule)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA unavailable")
def test_non_gaussian_mixed_schedule_exactness() -> None:
    operations: list[DrawOperation] = []
    for _ in range(257):
        random_index = len(operations)
        operations.extend(
            (
                DrawOperation.random(),
                DrawOperation.getrandbits(0),
                DrawOperation.getrandbits(1),
                DrawOperation.getrandbits(32),
                DrawOperation.getrandbits(33),
                DrawOperation.getrandbits(64),
                DrawOperation.randbelow(1),
                DrawOperation.randbelow(2),
                DrawOperation.randbelow(3),
                DrawOperation.randbelow(
                    65537,
                    predicate=DrawPredicate.random_lt(random_index, 0.25),
                ),
                DrawOperation.randbelow((1 << 63) - 1),
            )
        )
    _assert_cuda_equals_cpu(_bundle(20260818), DrawSchedule((tuple(operations),)))


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA unavailable")
def test_multiworld_independence_and_variable_schedule_lengths() -> None:
    bundle = _bundle(*range(56))
    worlds = tuple(
        tuple(
            operation
            for _ in range(1 + world % 17)
            for operation in (
                DrawOperation.raw32(),
                DrawOperation.random(),
                DrawOperation.getrandbits(world % 65),
                DrawOperation.randbelow(1 + (world * 7919) % 100000),
            )
        )
        for world in range(56)
    )
    schedule = DrawSchedule(worlds)
    _assert_cuda_equals_cpu(bundle, schedule)

    combined = execute_cuda(bundle, schedule)
    for world in (0, 1, 17, 47, 55):
        isolated = execute_cuda(
            RNGStateBundle((bundle.worlds[world],), source_binding=SOURCE),
            DrawSchedule((schedule.worlds[world],)),
        )
        assert combined.material.worlds[world] == isolated.material.worlds[0]
        assert combined.successor.worlds[world] == isolated.successor.worlds[0]


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA unavailable")
def test_gauss_is_fail_closed_when_raw_backend_selected() -> None:
    bundle = _bundle(7)
    before = bundle.digest()
    schedule = DrawSchedule(((DrawOperation.raw32(), DrawOperation.gauss()),))

    with pytest.raises(OptionDContractError, match="explicit Gaussian strategy"):
        execute_cuda(bundle, schedule)

    assert bundle.digest() == before
