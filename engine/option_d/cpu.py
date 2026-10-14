"""Current-source CPython MT19937 and ``Random.gauss`` CPU oracle."""

from __future__ import annotations

from dataclasses import replace
import math
import random
from typing import Any

from .contracts import (
    CPYTHON_STATE_WORDS,
    CpythonRNGState,
    DrawOperation,
    DrawSchedule,
    ExecutionReceipt,
    GeneratedMaterial,
    GeneratedValue,
    GenerationResult,
    Operator,
    OptionDExecutionError,
    RNGCounters,
    RNGStateBundle,
    WorldExecutionReceipt,
    float64_bits,
    float64_from_bits,
)


_MATRIX_A = 0x9908B0DF
_UPPER_MASK = 0x80000000
_LOWER_MASK = 0x7FFFFFFF
_M = 397
_TWOPI = 2.0 * math.pi


def state_from_seed(seed: Any) -> CpythonRNGState:
    """Create an API state from the current authoritative CPython runtime."""

    rng = random.Random(seed)
    return CpythonRNGState.from_python_state(rng.getstate())


class _MutableCpythonMT:
    def __init__(self, state: CpythonRNGState):
        state.validate()
        self.words = list(state.words)
        self.index = state.index
        self.gauss_next_bits = state.gauss_next_bits
        self.counters = state.counters

    def _counter(self, **changes: int) -> None:
        values = {
            name: getattr(self.counters, name)
            for name in self.counters.__dataclass_fields__
        }
        for name, amount in changes.items():
            values[name] += amount
        self.counters = RNGCounters(**values)

    def twist(self) -> None:
        words = self.words
        for index in range(CPYTHON_STATE_WORDS - _M):
            y = (words[index] & _UPPER_MASK) | (words[index + 1] & _LOWER_MASK)
            words[index] = (
                words[index + _M] ^ (y >> 1) ^ (_MATRIX_A if y & 1 else 0)
            ) & 0xFFFFFFFF
        for index in range(CPYTHON_STATE_WORDS - _M, CPYTHON_STATE_WORDS - 1):
            y = (words[index] & _UPPER_MASK) | (words[index + 1] & _LOWER_MASK)
            words[index] = (
                words[index + (_M - CPYTHON_STATE_WORDS)]
                ^ (y >> 1)
                ^ (_MATRIX_A if y & 1 else 0)
            ) & 0xFFFFFFFF
        y = (words[-1] & _UPPER_MASK) | (words[0] & _LOWER_MASK)
        words[-1] = (words[_M - 1] ^ (y >> 1) ^ (_MATRIX_A if y & 1 else 0)) & 0xFFFFFFFF
        self.index = 0
        self._counter(twists=1)

    def raw32(self) -> int:
        if self.index >= CPYTHON_STATE_WORDS:
            self.twist()
        value = self.words[self.index]
        self.index += 1
        value ^= value >> 11
        value ^= (value << 7) & 0x9D2C5680
        value ^= (value << 15) & 0xEFC60000
        value ^= value >> 18
        self._counter(raw_words=1)
        return value & 0xFFFFFFFF

    def random_double(self) -> float:
        first = self.raw32() >> 5
        second = self.raw32() >> 6
        self._counter(random_calls=1)
        return (first * 67108864.0 + second) * (1.0 / 9007199254740992.0)

    def getrandbits(self, bit_count: int) -> int:
        self._counter(getrandbits_calls=1)
        if bit_count == 0:
            return 0
        if bit_count <= 32:
            return self.raw32() >> (32 - bit_count)
        word_count = (bit_count - 1) // 32 + 1
        value = 0
        remaining = bit_count
        for index in range(word_count):
            word = self.raw32()
            if remaining < 32:
                word >>= 32 - remaining
            value |= word << (32 * index)
            remaining -= 32
        return value

    def randbelow(self, upper: int) -> int:
        bit_count = upper.bit_length()
        rejected = 0
        value = self.getrandbits(bit_count)
        while value >= upper:
            rejected += 1
            value = self.getrandbits(bit_count)
        if rejected:
            self._counter(randbelow_rejections=rejected)
        return value

    def gauss(self, mu_bits: int, sigma_bits: int) -> float:
        mu = float64_from_bits(mu_bits)
        sigma = float64_from_bits(sigma_bits)
        self._counter(gauss_calls=1)
        if self.gauss_next_bits is None:
            self._counter(gauss_cache_misses=1)
            x2pi = self.random_double() * _TWOPI
            g2rad = math.sqrt(-2.0 * math.log(1.0 - self.random_double()))
            z = math.cos(x2pi) * g2rad
            self.gauss_next_bits = float64_bits(math.sin(x2pi) * g2rad)
        else:
            self._counter(gauss_cache_hits=1)
            z = float64_from_bits(self.gauss_next_bits)
            self.gauss_next_bits = None
        return mu + z * sigma

    def execute(self, operation: DrawOperation) -> GeneratedValue:
        operator = operation.operator
        if operator is Operator.RAW32:
            value_bits = self.raw32()
        elif operator is Operator.RANDOM:
            value_bits = float64_bits(self.random_double())
        elif operator is Operator.GETRANDBITS:
            value_bits = self.getrandbits(operation.argument)
        elif operator is Operator.RANDBELOW:
            value_bits = self.randbelow(operation.argument)
        elif operator is Operator.GAUSS:
            value_bits = float64_bits(self.gauss(operation.mu_bits, operation.sigma_bits))
        else:  # pragma: no cover - contract validation closes this path
            raise OptionDExecutionError(f"unsupported draw opcode {operator!r}")
        self._counter(semantic_draws=1)
        return GeneratedValue(operator, value_bits)

    def export_state(self) -> CpythonRNGState:
        state = CpythonRNGState(
            words=tuple(self.words),
            index=self.index,
            gauss_next_bits=self.gauss_next_bits,
            counters=self.counters,
        )
        state.validate()
        return state


def execute_cpu(
    bundle: RNGStateBundle,
    schedule: DrawSchedule,
    *,
    backend_id: str = "option_d_cpu_oracle_v1",
) -> GenerationResult:
    """Execute a schedule transactionally against independent world clones."""

    bundle.validate()
    schedule.validate(expected_worlds=len(bundle.worlds))
    if not isinstance(backend_id, str) or not backend_id.strip():
        raise OptionDExecutionError("backend_id must be a nonempty string")

    successor_worlds: list[CpythonRNGState] = []
    material_worlds: list[tuple[GeneratedValue, ...]] = []
    world_receipts: list[WorldExecutionReceipt] = []
    try:
        for world_index, (initial, operations) in enumerate(
            zip(bundle.worlds, schedule.worlds, strict=True)
        ):
            engine = _MutableCpythonMT(initial)
            generated: list[GeneratedValue] = []
            for operation in operations:
                predicate = operation.predicate
                if predicate is not None:
                    source = generated[predicate.source_operation]
                    if not source.executed or source.operator is not Operator.RANDOM:
                        raise OptionDExecutionError(
                            "predicate source was not an executed random output"
                        )
                    if not (
                        float64_from_bits(source.value_bits)
                        < float64_from_bits(predicate.threshold_bits)
                    ):
                        generated.append(GeneratedValue(operation.operator, 0, False))
                        continue
                generated.append(engine.execute(operation))
            values = tuple(generated)
            successor = engine.export_state()
            successor_worlds.append(successor)
            material_worlds.append(values)
            world_receipts.append(
                WorldExecutionReceipt(
                    world=world_index,
                    operations=len(operations),
                    initial_state_sha256=initial.digest(),
                    successor_state_sha256=successor.digest(),
                    counters_before=initial.counters,
                    counters_after=successor.counters,
                    counters_delta=successor.counters.delta(initial.counters),
                )
            )
    except Exception as error:
        raise OptionDExecutionError(
            "CPU schedule failed; no successor state was published"
        ) from error

    successor_bundle = replace(bundle, worlds=tuple(successor_worlds))
    material = GeneratedMaterial(tuple(material_worlds))
    receipt = ExecutionReceipt(
        backend_id=backend_id,
        source_binding=bundle.source_binding,
        schedule_sha256=schedule.digest(),
        initial_bundle_sha256=bundle.digest(),
        successor_bundle_sha256=successor_bundle.digest(),
        material_sha256=material.digest(),
        worlds=tuple(world_receipts),
    )
    result = GenerationResult(material, successor_bundle, receipt)
    result.validate(initial=bundle, schedule=schedule)
    return result
