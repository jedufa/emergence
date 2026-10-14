"""Immutable standalone Option D API and canonical receipt contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from hashlib import sha256
import json
import math
import struct
from typing import Any, Final, Iterable, Mapping, Sequence


OPTION_D_API_SCHEMA_V1: Final[str] = "eiel_option_d_standalone_api_v1"
OPTION_D_STATE_SCHEMA_V1: Final[str] = "eiel_option_d_cpython_state_v1"
OPTION_D_SCHEDULE_SCHEMA_V1: Final[str] = "eiel_option_d_draw_schedule_v1"
OPTION_D_MATERIAL_SCHEMA_V1: Final[str] = "eiel_option_d_generated_material_v1"
OPTION_D_RECEIPT_SCHEMA_V1: Final[str] = "eiel_option_d_execution_receipt_v1"
OPTION_D_CHECKPOINT_SCHEMA_V1: Final[str] = "eiel_option_d_checkpoint_v1"
CPYTHON_ALGORITHM_ID_V1: Final[str] = "cpython.random.mt19937.v3.14.4"
CPYTHON_STATE_WORDS: Final[int] = 624
MAX_SCHEDULE_OPERATIONS_PER_WORLD_V1: Final[int] = 1_000_000
MAX_DEVICE_GETRANDBITS_V1: Final[int] = 64
MAX_DEVICE_RANDBELOW_V1: Final[int] = (1 << 63) - 1


class OptionDContractError(ValueError):
    """An input cannot be represented by the frozen standalone contract."""


class OptionDExecutionError(RuntimeError):
    """Execution failed without publishing a successor state."""


def float64_bits(value: float) -> int:
    """Return the exact IEEE-754 binary64 payload for ``value``."""

    return struct.unpack("<Q", struct.pack("<d", float(value)))[0]


def float64_from_bits(bits: int) -> float:
    """Reconstruct a binary64 from its unsigned payload."""

    if type(bits) is not int or not 0 <= bits <= 0xFFFFFFFFFFFFFFFF:
        raise OptionDContractError("float64 payload must be a uint64")
    return struct.unpack("<d", struct.pack("<Q", bits))[0]


def canonical_json_bytes(payload: Any) -> bytes:
    """Encode a contract payload without platform-dependent whitespace."""

    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(payload: Any) -> str:
    return sha256(canonical_json_bytes(payload)).hexdigest()


def _require_nonnegative_int(name: str, value: int) -> None:
    if type(value) is not int or value < 0:
        raise OptionDContractError(f"{name} must be a nonnegative integer")


@dataclass(frozen=True, slots=True)
class RNGCounters:
    """Persistent accounting for every semantic and primitive RNG boundary.

    ``semantic_draws`` counts externally scheduled values.  Primitive calls
    made inside ``gauss`` and ``randbelow`` are instead visible in their typed
    counters, while every consumed MT word is reflected in ``raw_words``.
    """

    raw_words: int = 0
    semantic_draws: int = 0
    random_calls: int = 0
    getrandbits_calls: int = 0
    randbelow_rejections: int = 0
    gauss_calls: int = 0
    gauss_cache_hits: int = 0
    gauss_cache_misses: int = 0
    twists: int = 0

    def validate(self) -> None:
        for name in self.__dataclass_fields__:
            _require_nonnegative_int(name, getattr(self, name))
        if self.gauss_cache_hits + self.gauss_cache_misses != self.gauss_calls:
            raise OptionDContractError(
                "gauss cache hit and miss counters must sum to gauss_calls"
            )

    def to_dict(self) -> dict[str, int]:
        self.validate()
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RNGCounters":
        expected = set(cls.__dataclass_fields__)
        if set(payload) != expected:
            raise OptionDContractError("checkpoint RNG counter fields are invalid")
        result = cls(**{name: payload[name] for name in expected})
        result.validate()
        return result

    def delta(self, before: "RNGCounters") -> "RNGCounters":
        values = {
            name: getattr(self, name) - getattr(before, name)
            for name in self.__dataclass_fields__
        }
        result = RNGCounters(**values)
        result.validate()
        return result


@dataclass(frozen=True, slots=True)
class CpythonRNGState:
    """One complete CPython Random state plus Option D accounting."""

    words: tuple[int, ...]
    index: int
    gauss_next_bits: int | None = None
    counters: RNGCounters = RNGCounters()
    schema_version: str = OPTION_D_STATE_SCHEMA_V1
    algorithm_id: str = CPYTHON_ALGORITHM_ID_V1

    def validate(self) -> None:
        if self.schema_version != OPTION_D_STATE_SCHEMA_V1:
            raise OptionDContractError("unsupported Option D CPython state schema")
        if self.algorithm_id != CPYTHON_ALGORITHM_ID_V1:
            raise OptionDContractError("unsupported Option D RNG algorithm")
        if not isinstance(self.words, tuple) or len(self.words) != CPYTHON_STATE_WORDS:
            raise OptionDContractError("CPython MT state must contain 624 words")
        if any(type(word) is not int or not 0 <= word <= 0xFFFFFFFF for word in self.words):
            raise OptionDContractError("CPython MT words must be uint32 values")
        if type(self.index) is not int or not 0 <= self.index <= CPYTHON_STATE_WORDS:
            raise OptionDContractError("CPython MT index must be in [0, 624]")
        if self.gauss_next_bits is not None:
            if (
                type(self.gauss_next_bits) is not int
                or not 0 <= self.gauss_next_bits <= 0xFFFFFFFFFFFFFFFF
            ):
                raise OptionDContractError("gauss cache payload must be uint64 or null")
            if not math.isfinite(float64_from_bits(self.gauss_next_bits)):
                raise OptionDContractError("Option D V1 gauss cache must be finite")
        self.counters.validate()

    @classmethod
    def from_python_state(
        cls,
        state: Sequence[Any],
        *,
        counters: RNGCounters | None = None,
    ) -> "CpythonRNGState":
        if not isinstance(state, (tuple, list)) or len(state) != 3:
            raise OptionDContractError("CPython state must be a three-item sequence")
        version, internal, gauss_next = state
        if version != 3:
            raise OptionDContractError("Option D V1 requires CPython state version 3")
        if not isinstance(internal, (tuple, list)) or len(internal) != 625:
            raise OptionDContractError("CPython internal state must contain 625 integers")
        result = cls(
            words=tuple(internal[:-1]),
            index=internal[-1],
            gauss_next_bits=None if gauss_next is None else float64_bits(gauss_next),
            counters=counters or RNGCounters(),
        )
        result.validate()
        return result

    def to_python_state(self) -> tuple[int, tuple[int, ...], float | None]:
        self.validate()
        gauss_next = (
            None
            if self.gauss_next_bits is None
            else float64_from_bits(self.gauss_next_bits)
        )
        return (3, self.words + (self.index,), gauss_next)

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "algorithm_id": self.algorithm_id,
            "counters": self.counters.to_dict(),
            "gauss_next_bits": self.gauss_next_bits,
            "index": self.index,
            "schema_version": self.schema_version,
            "words": list(self.words),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CpythonRNGState":
        expected = {
            "algorithm_id",
            "counters",
            "gauss_next_bits",
            "index",
            "schema_version",
            "words",
        }
        if set(payload) != expected or not isinstance(payload["counters"], Mapping):
            raise OptionDContractError("checkpoint CPython state fields are invalid")
        words = payload["words"]
        if not isinstance(words, list):
            raise OptionDContractError("checkpoint CPython words must be an array")
        result = cls(
            words=tuple(words),
            index=payload["index"],
            gauss_next_bits=payload["gauss_next_bits"],
            counters=RNGCounters.from_dict(payload["counters"]),
            schema_version=payload["schema_version"],
            algorithm_id=payload["algorithm_id"],
        )
        result.validate()
        return result

    def digest(self) -> str:
        return canonical_sha256(self.to_dict())


@dataclass(frozen=True, slots=True)
class RNGStateBundle:
    """Independent authoritative state for every scheduled world."""

    worlds: tuple[CpythonRNGState, ...]
    source_binding: str
    schema_version: str = OPTION_D_API_SCHEMA_V1

    def validate(self) -> None:
        if self.schema_version != OPTION_D_API_SCHEMA_V1:
            raise OptionDContractError("unsupported Option D API schema")
        if not isinstance(self.worlds, tuple) or not self.worlds:
            raise OptionDContractError("Option D requires at least one world")
        if not isinstance(self.source_binding, str) or not self.source_binding.strip():
            raise OptionDContractError("source_binding must be a nonempty string")
        for state in self.worlds:
            if not isinstance(state, CpythonRNGState):
                raise OptionDContractError("every world must own a CPython RNG state")
            state.validate()

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "source_binding": self.source_binding,
            "worlds": [world.to_dict() for world in self.worlds],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RNGStateBundle":
        if set(payload) != {"schema_version", "source_binding", "worlds"}:
            raise OptionDContractError("checkpoint state bundle fields are invalid")
        worlds = payload["worlds"]
        if not isinstance(worlds, list):
            raise OptionDContractError("checkpoint worlds must be an array")
        result = cls(
            worlds=tuple(CpythonRNGState.from_dict(world) for world in worlds),
            source_binding=payload["source_binding"],
            schema_version=payload["schema_version"],
        )
        result.validate()
        return result

    def digest(self) -> str:
        return canonical_sha256(self.to_dict())


class Operator(IntEnum):
    RAW32 = 1
    RANDOM = 2
    GETRANDBITS = 3
    RANDBELOW = 4
    GAUSS = 5


class PredicateKind(IntEnum):
    RANDOM_LT = 1


@dataclass(frozen=True, slots=True)
class DrawPredicate:
    """Device-evaluated branch over an earlier unpredicated random output."""

    kind: PredicateKind
    source_operation: int
    threshold_bits: int

    @classmethod
    def random_lt(cls, source_operation: int, threshold: float) -> "DrawPredicate":
        return cls(PredicateKind.RANDOM_LT, source_operation, float64_bits(threshold))

    def validate(self) -> None:
        if type(self.kind) is not PredicateKind:
            raise OptionDContractError("draw predicate kind is invalid")
        if type(self.source_operation) is not int or self.source_operation < 0:
            raise OptionDContractError("predicate source operation must be nonnegative")
        if (
            type(self.threshold_bits) is not int
            or not 0 <= self.threshold_bits <= 0xFFFFFFFFFFFFFFFF
            or not math.isfinite(float64_from_bits(self.threshold_bits))
        ):
            raise OptionDContractError("predicate threshold must be finite binary64")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "kind": self.kind.name.lower(),
            "kind_code": int(self.kind),
            "source_operation": self.source_operation,
            "threshold_bits": self.threshold_bits,
        }


@dataclass(frozen=True, slots=True)
class DrawOperation:
    """One typed output in a per-world sequential draw schedule."""

    operator: Operator
    argument: int = 0
    mu_bits: int = 0
    sigma_bits: int = 0x3FF0000000000000
    predicate: DrawPredicate | None = None

    @classmethod
    def raw32(cls, *, predicate: DrawPredicate | None = None) -> "DrawOperation":
        return cls(Operator.RAW32, predicate=predicate)

    @classmethod
    def random(cls, *, predicate: DrawPredicate | None = None) -> "DrawOperation":
        return cls(Operator.RANDOM, predicate=predicate)

    @classmethod
    def getrandbits(
        cls,
        bit_count: int,
        *,
        predicate: DrawPredicate | None = None,
    ) -> "DrawOperation":
        return cls(Operator.GETRANDBITS, argument=bit_count, predicate=predicate)

    @classmethod
    def randbelow(
        cls,
        upper: int,
        *,
        predicate: DrawPredicate | None = None,
    ) -> "DrawOperation":
        return cls(Operator.RANDBELOW, argument=upper, predicate=predicate)

    @classmethod
    def gauss(
        cls,
        mu: float = 0.0,
        sigma: float = 1.0,
        *,
        predicate: DrawPredicate | None = None,
    ) -> "DrawOperation":
        return cls(
            Operator.GAUSS,
            mu_bits=float64_bits(mu),
            sigma_bits=float64_bits(sigma),
            predicate=predicate,
        )

    def validate(self) -> None:
        if type(self.operator) is not Operator:
            raise OptionDContractError("draw operator must be a frozen Operator value")
        if type(self.argument) is not int:
            raise OptionDContractError("draw operation argument must be an integer")
        for name, bits in (("mu_bits", self.mu_bits), ("sigma_bits", self.sigma_bits)):
            if type(bits) is not int or not 0 <= bits <= 0xFFFFFFFFFFFFFFFF:
                raise OptionDContractError(f"{name} must be a uint64 payload")
        if self.operator is Operator.GETRANDBITS:
            if not 0 <= self.argument <= MAX_DEVICE_GETRANDBITS_V1:
                raise OptionDContractError("Option D V1 getrandbits supports k in [0, 64]")
        elif self.operator is Operator.RANDBELOW:
            if not 1 <= self.argument <= MAX_DEVICE_RANDBELOW_V1:
                raise OptionDContractError(
                    "Option D V1 randbelow supports n in [1, 2**63 - 1]"
                )
        elif self.argument != 0:
            raise OptionDContractError("this operator does not accept an integer argument")
        if self.operator is Operator.GAUSS:
            if not math.isfinite(float64_from_bits(self.mu_bits)) or not math.isfinite(
                float64_from_bits(self.sigma_bits)
            ):
                raise OptionDContractError("Option D V1 gauss parameters must be finite")
        elif self.mu_bits != 0 or self.sigma_bits != 0x3FF0000000000000:
            raise OptionDContractError("only gauss accepts mu and sigma payloads")
        if self.predicate is not None:
            if not isinstance(self.predicate, DrawPredicate):
                raise OptionDContractError("operation predicate must be DrawPredicate or null")
            self.predicate.validate()

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "argument": self.argument,
            "mu_bits": self.mu_bits,
            "operator": self.operator.name.lower(),
            "opcode": int(self.operator),
            "predicate": None if self.predicate is None else self.predicate.to_dict(),
            "sigma_bits": self.sigma_bits,
        }


@dataclass(frozen=True, slots=True)
class DrawSchedule:
    """Deterministic, independently variable operation stream per world."""

    worlds: tuple[tuple[DrawOperation, ...], ...]
    schema_version: str = OPTION_D_SCHEDULE_SCHEMA_V1

    def validate(self, *, expected_worlds: int | None = None) -> None:
        if self.schema_version != OPTION_D_SCHEDULE_SCHEMA_V1:
            raise OptionDContractError("unsupported Option D schedule schema")
        if not isinstance(self.worlds, tuple) or not self.worlds:
            raise OptionDContractError("draw schedule requires at least one world")
        if expected_worlds is not None and len(self.worlds) != expected_worlds:
            raise OptionDContractError("draw schedule world count does not match state")
        for operations in self.worlds:
            if not isinstance(operations, tuple):
                raise OptionDContractError("each world schedule must be an immutable tuple")
            if len(operations) > MAX_SCHEDULE_OPERATIONS_PER_WORLD_V1:
                raise OptionDContractError("world schedule exceeds the V1 bounded limit")
            for operation_index, operation in enumerate(operations):
                if not isinstance(operation, DrawOperation):
                    raise OptionDContractError("schedule entries must be DrawOperation values")
                operation.validate()
                predicate = operation.predicate
                if predicate is not None:
                    if predicate.source_operation >= operation_index:
                        raise OptionDContractError(
                            "predicate must reference an earlier operation"
                        )
                    source = operations[predicate.source_operation]
                    if source.operator is not Operator.RANDOM or source.predicate is not None:
                        raise OptionDContractError(
                            "V1 predicate source must be an unpredicated random operation"
                        )

    @classmethod
    def same_for_all(
        cls,
        operations: Iterable[DrawOperation],
        *,
        world_count: int,
    ) -> "DrawSchedule":
        if type(world_count) is not int or world_count <= 0:
            raise OptionDContractError("world_count must be positive")
        frozen = tuple(operations)
        result = cls(tuple(frozen for _ in range(world_count)))
        result.validate()
        return result

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "worlds": [
                [operation.to_dict() for operation in operations]
                for operations in self.worlds
            ],
        }

    def digest(self) -> str:
        return canonical_sha256(self.to_dict())


@dataclass(frozen=True, slots=True)
class GeneratedValue:
    """Canonical output payload; floats are always stored by exact bits."""

    operator: Operator
    value_bits: int
    executed: bool = True

    def validate(self) -> None:
        if type(self.operator) is not Operator:
            raise OptionDContractError("generated value operator is invalid")
        if type(self.executed) is not bool:
            raise OptionDContractError("generated value executed flag must be bool")
        if type(self.value_bits) is not int or not 0 <= self.value_bits <= 0xFFFFFFFFFFFFFFFF:
            raise OptionDContractError("generated value payload must be uint64")
        if not self.executed and self.value_bits != 0:
            raise OptionDContractError("skipped generated value payload must be zero")
        if self.operator is Operator.RAW32 and self.value_bits > 0xFFFFFFFF:
            raise OptionDContractError("raw32 output exceeds uint32")

    @property
    def value(self) -> int | float | None:
        self.validate()
        if not self.executed:
            return None
        if self.operator in {Operator.RANDOM, Operator.GAUSS}:
            return float64_from_bits(self.value_bits)
        return self.value_bits

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "executed": self.executed,
            "operator": self.operator.name.lower(),
            "opcode": int(self.operator),
            "value_bits": self.value_bits,
        }


@dataclass(frozen=True, slots=True)
class GeneratedMaterial:
    worlds: tuple[tuple[GeneratedValue, ...], ...]
    schema_version: str = OPTION_D_MATERIAL_SCHEMA_V1

    def validate(self, schedule: DrawSchedule | None = None) -> None:
        if self.schema_version != OPTION_D_MATERIAL_SCHEMA_V1:
            raise OptionDContractError("unsupported generated material schema")
        if not isinstance(self.worlds, tuple) or not self.worlds:
            raise OptionDContractError("generated material requires at least one world")
        if schedule is not None and len(self.worlds) != len(schedule.worlds):
            raise OptionDContractError("material world count does not match schedule")
        for world_index, values in enumerate(self.worlds):
            if not isinstance(values, tuple):
                raise OptionDContractError("generated values must be immutable tuples")
            if schedule is not None and len(values) != len(schedule.worlds[world_index]):
                raise OptionDContractError("material value count does not match schedule")
            for value_index, value in enumerate(values):
                if not isinstance(value, GeneratedValue):
                    raise OptionDContractError("material entry is not a GeneratedValue")
                value.validate()
                if schedule is not None and (
                    value.operator is not schedule.worlds[world_index][value_index].operator
                ):
                    raise OptionDContractError("material operator does not match schedule")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "worlds": [[value.to_dict() for value in world] for world in self.worlds],
        }

    def digest(self) -> str:
        return canonical_sha256(self.to_dict())


@dataclass(frozen=True, slots=True)
class WorldExecutionReceipt:
    world: int
    operations: int
    initial_state_sha256: str
    successor_state_sha256: str
    counters_before: RNGCounters
    counters_after: RNGCounters
    counters_delta: RNGCounters

    def to_dict(self) -> dict[str, Any]:
        return {
            "counters_after": self.counters_after.to_dict(),
            "counters_before": self.counters_before.to_dict(),
            "counters_delta": self.counters_delta.to_dict(),
            "initial_state_sha256": self.initial_state_sha256,
            "operations": self.operations,
            "successor_state_sha256": self.successor_state_sha256,
            "world": self.world,
        }


@dataclass(frozen=True, slots=True)
class ExecutionReceipt:
    backend_id: str
    source_binding: str
    schedule_sha256: str
    initial_bundle_sha256: str
    successor_bundle_sha256: str
    material_sha256: str
    worlds: tuple[WorldExecutionReceipt, ...]
    schema_version: str = OPTION_D_RECEIPT_SCHEMA_V1
    status: str = "success"

    def to_dict(self) -> dict[str, Any]:
        if self.schema_version != OPTION_D_RECEIPT_SCHEMA_V1 or self.status != "success":
            raise OptionDContractError("execution receipt schema or status is invalid")
        if not self.backend_id or not self.source_binding:
            raise OptionDContractError("execution receipt identity is incomplete")
        return {
            "backend_id": self.backend_id,
            "initial_bundle_sha256": self.initial_bundle_sha256,
            "material_sha256": self.material_sha256,
            "schedule_sha256": self.schedule_sha256,
            "schema_version": self.schema_version,
            "source_binding": self.source_binding,
            "status": self.status,
            "successor_bundle_sha256": self.successor_bundle_sha256,
            "worlds": [world.to_dict() for world in self.worlds],
        }

    def digest(self) -> str:
        return canonical_sha256(self.to_dict())


@dataclass(frozen=True, slots=True)
class GenerationResult:
    material: GeneratedMaterial
    successor: RNGStateBundle
    receipt: ExecutionReceipt

    def validate(self, *, initial: RNGStateBundle, schedule: DrawSchedule) -> None:
        initial.validate()
        schedule.validate(expected_worlds=len(initial.worlds))
        self.material.validate(schedule)
        self.successor.validate()
        if self.successor.source_binding != initial.source_binding:
            raise OptionDContractError("successor changed the frozen source binding")
        expected = {
            "initial_bundle_sha256": initial.digest(),
            "material_sha256": self.material.digest(),
            "schedule_sha256": schedule.digest(),
            "source_binding": initial.source_binding,
            "successor_bundle_sha256": self.successor.digest(),
        }
        receipt = self.receipt.to_dict()
        for field, value in expected.items():
            if receipt[field] != value:
                raise OptionDContractError(f"execution receipt {field} does not bind output")
