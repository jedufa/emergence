from __future__ import annotations

import random

import pytest

from engine.option_d import (
    CpythonRNGState,
    DrawOperation,
    DrawPredicate,
    DrawSchedule,
    Operator,
    OptionDContractError,
    RNGStateBundle,
    execute_cpu,
    float64_bits,
    state_from_seed,
)


SOURCE_BINDING = "test-cpython-3.14.4-current-source"


def _bundle(*seeds: object) -> RNGStateBundle:
    return RNGStateBundle(
        tuple(state_from_seed(seed) for seed in seeds),
        source_binding=SOURCE_BINDING,
    )


class _CountingRandom(random.Random):
    def __init__(self) -> None:
        self.getrandbits_calls = 0
        super().__init__()

    def getrandbits(self, bit_count: int) -> int:
        self.getrandbits_calls += 1
        return super().getrandbits(bit_count)


def _official_schedule(
    state: CpythonRNGState,
    operations: tuple[DrawOperation, ...],
) -> tuple[tuple[tuple[Operator, int], ...], tuple[object, ...], int]:
    # Kept separate from _official_values so gauss parameters are reconstructed
    # without any decimal conversion.
    import struct

    rng = _CountingRandom()
    rng.setstate(state.to_python_state())
    outputs: list[tuple[Operator, int]] = []
    for operation in operations:
        if operation.operator is Operator.RAW32:
            value = rng.getrandbits(32)
            bits = value
        elif operation.operator is Operator.RANDOM:
            bits = float64_bits(rng.random())
        elif operation.operator is Operator.GETRANDBITS:
            bits = rng.getrandbits(operation.argument)
        elif operation.operator is Operator.RANDBELOW:
            bits = rng._randbelow(operation.argument)
        else:
            mu = struct.unpack("<d", struct.pack("<Q", operation.mu_bits))[0]
            sigma = struct.unpack("<d", struct.pack("<Q", operation.sigma_bits))[0]
            bits = float64_bits(rng.gauss(mu, sigma))
        outputs.append((operation.operator, bits))
    return tuple(outputs), rng.getstate(), rng.getrandbits_calls


@pytest.mark.parametrize("seed", [0, 1, 42, -7, "option-d", b"rng"])
def test_raw_words_and_twist_boundaries_match_current_cpython(seed: object) -> None:
    bundle = _bundle(seed)
    operations = tuple(DrawOperation.raw32() for _ in range(2000))
    schedule = DrawSchedule((operations,))

    result = execute_cpu(bundle, schedule)
    official, official_state, getrandbits_calls = _official_schedule(
        bundle.worlds[0], operations
    )

    assert tuple((value.operator, value.value_bits) for value in result.material.worlds[0]) == official
    assert result.successor.worlds[0].to_python_state() == official_state
    assert result.successor.worlds[0].counters.raw_words == 2000
    assert result.successor.worlds[0].counters.semantic_draws == 2000
    assert getrandbits_calls == 2000


def test_mixed_schedule_matches_current_cpython_bits_state_and_cache() -> None:
    bundle = _bundle(0xD0D0)
    operations = (
        DrawOperation.random(),
        DrawOperation.getrandbits(0),
        DrawOperation.getrandbits(1),
        DrawOperation.getrandbits(31),
        DrawOperation.getrandbits(32),
        DrawOperation.getrandbits(33),
        DrawOperation.getrandbits(64),
        DrawOperation.randbelow(2),
        DrawOperation.randbelow(3),
        DrawOperation.randbelow((1 << 63) - 1),
        DrawOperation.gauss(),
        DrawOperation.gauss(),
        DrawOperation.random(),
        DrawOperation.gauss(mu=-3.25, sigma=0.125),
    )
    schedule = DrawSchedule((operations,))

    result = execute_cpu(bundle, schedule)
    official, official_state, getrandbits_calls = _official_schedule(
        bundle.worlds[0], operations
    )

    assert tuple((value.operator, value.value_bits) for value in result.material.worlds[0]) == official
    assert result.successor.worlds[0].to_python_state() == official_state
    counters = result.successor.worlds[0].counters
    assert counters.semantic_draws == len(operations)
    assert counters.getrandbits_calls == getrandbits_calls
    assert counters.gauss_calls == 3
    assert counters.gauss_cache_hits == 1
    assert counters.gauss_cache_misses == 2
    assert counters.random_calls == 6  # two direct plus four inside cache misses


def test_getrandbits_zero_advances_only_typed_and_semantic_counters() -> None:
    bundle = _bundle(123)
    initial = bundle.worlds[0]
    result = execute_cpu(
        bundle,
        DrawSchedule(((DrawOperation.getrandbits(0),),)),
    )
    successor = result.successor.worlds[0]

    assert result.material.worlds[0][0].value_bits == 0
    assert successor.words == initial.words
    assert successor.index == initial.index
    assert successor.gauss_next_bits == initial.gauss_next_bits
    assert successor.counters.raw_words == 0
    assert successor.counters.getrandbits_calls == 1
    assert successor.counters.semantic_draws == 1


def test_gauss_cache_hit_consumes_no_raw_words() -> None:
    bundle = _bundle(999)
    first = execute_cpu(bundle, DrawSchedule(((DrawOperation.gauss(),),)))
    first_state = first.successor.worlds[0]
    assert first_state.gauss_next_bits is not None
    assert first.receipt.worlds[0].counters_delta.raw_words == 4

    second = execute_cpu(
        first.successor,
        DrawSchedule(((DrawOperation.gauss(mu=2.0, sigma=4.0),),)),
    )
    assert second.successor.worlds[0].gauss_next_bits is None
    assert second.receipt.worlds[0].counters_delta.raw_words == 0
    assert second.receipt.worlds[0].counters_delta.random_calls == 0
    assert second.receipt.worlds[0].counters_delta.gauss_cache_hits == 1


def test_split_continuation_equals_uninterrupted_execution() -> None:
    bundle = _bundle("continuation")
    operations = tuple(
        operation
        for _ in range(37)
        for operation in (
            DrawOperation.random(),
            DrawOperation.randbelow(17),
            DrawOperation.gauss(mu=0.25, sigma=2.0),
            DrawOperation.getrandbits(0),
            DrawOperation.getrandbits(64),
        )
    )
    midpoint = 83
    full = execute_cpu(bundle, DrawSchedule((operations,)))
    prefix = execute_cpu(bundle, DrawSchedule((operations[:midpoint],)))
    suffix = execute_cpu(prefix.successor, DrawSchedule((operations[midpoint:],)))

    assert prefix.material.worlds[0] + suffix.material.worlds[0] == full.material.worlds[0]
    assert suffix.successor == full.successor


def test_multiworld_schedules_are_independent_and_branch_shaped() -> None:
    bundle = _bundle(11, 22, 33)
    schedule = DrawSchedule(
        (
            (DrawOperation.gauss(), DrawOperation.gauss()),
            (DrawOperation.getrandbits(0), DrawOperation.randbelow(2)),
            tuple(DrawOperation.raw32() for _ in range(700)),
        )
    )
    combined = execute_cpu(bundle, schedule)

    for world, operations in enumerate(schedule.worlds):
        isolated_bundle = RNGStateBundle(
            (bundle.worlds[world],),
            source_binding=SOURCE_BINDING,
        )
        isolated = execute_cpu(isolated_bundle, DrawSchedule((operations,)))
        assert combined.material.worlds[world] == isolated.material.worlds[0]
        assert combined.successor.worlds[world] == isolated.successor.worlds[0]


def test_invalid_schedule_is_rejected_without_touching_input() -> None:
    bundle = _bundle(7)
    before_digest = bundle.digest()
    schedule = DrawSchedule(((DrawOperation(Operator.GETRANDBITS, argument=65),),))

    with pytest.raises(OptionDContractError, match=r"\[0, 64\]"):
        execute_cpu(bundle, schedule)

    assert bundle.digest() == before_digest


def test_receipt_binds_all_transaction_products() -> None:
    bundle = _bundle(1, 2)
    schedule = DrawSchedule.same_for_all(
        (DrawOperation.random(), DrawOperation.gauss(), DrawOperation.randbelow(5)),
        world_count=2,
    )
    result = execute_cpu(bundle, schedule)

    result.validate(initial=bundle, schedule=schedule)
    assert result.receipt.initial_bundle_sha256 == bundle.digest()
    assert result.receipt.successor_bundle_sha256 == result.successor.digest()
    assert result.receipt.material_sha256 == result.material.digest()
    assert result.receipt.schedule_sha256 == schedule.digest()


def test_output_dependent_predicates_match_current_cpython_without_host_refill() -> None:
    bundle = _bundle("resident-branch")
    operations: list[DrawOperation] = []
    for _ in range(128):
        random_index = len(operations)
        operations.append(DrawOperation.random())
        operations.append(
            DrawOperation.gauss(
                sigma=0.005,
                predicate=DrawPredicate.random_lt(random_index, 0.2),
            )
        )
        memory_index = len(operations)
        operations.append(DrawOperation.random())
        operations.append(
            DrawOperation.randbelow(
                2,
                predicate=DrawPredicate.random_lt(memory_index, 0.05),
            )
        )
    schedule = DrawSchedule((tuple(operations),))
    result = execute_cpu(bundle, schedule)

    official = random.Random()
    official.setstate(bundle.worlds[0].to_python_state())
    expected: list[tuple[bool, int]] = []
    for _ in range(128):
        decision = official.random()
        expected.append((True, float64_bits(decision)))
        if decision < 0.2:
            expected.append((True, float64_bits(official.gauss(0.0, 0.005))))
        else:
            expected.append((False, 0))
        memory_decision = official.random()
        expected.append((True, float64_bits(memory_decision)))
        if memory_decision < 0.05:
            expected.append((True, official._randbelow(2)))
        else:
            expected.append((False, 0))

    assert [
        (value.executed, value.value_bits)
        for value in result.material.worlds[0]
    ] == expected
    assert result.successor.worlds[0].to_python_state() == official.getstate()
    assert result.successor.worlds[0].counters.semantic_draws == sum(
        executed for executed, _ in expected
    )


def test_predicates_must_reference_an_earlier_unpredicated_random() -> None:
    bundle = _bundle(8)
    forward = DrawSchedule(
        ((DrawOperation.raw32(predicate=DrawPredicate.random_lt(0, 0.5)),),)
    )
    with pytest.raises(OptionDContractError, match="earlier operation"):
        execute_cpu(bundle, forward)

    wrong_source = DrawSchedule(
        ((
            DrawOperation.raw32(),
            DrawOperation.randbelow(
                2,
                predicate=DrawPredicate.random_lt(0, 0.5),
            ),
        ),)
    )
    with pytest.raises(OptionDContractError, match="unpredicated random"):
        execute_cpu(bundle, wrong_source)
