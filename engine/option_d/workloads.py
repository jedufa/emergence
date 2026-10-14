"""Current-source-shaped standalone schedules without Tensor World imports."""

from __future__ import annotations

from collections.abc import Sequence
import math

from .contracts import (
    DrawOperation,
    DrawPredicate,
    DrawSchedule,
    OptionDContractError,
)


_C4_SCALAR_SIGMAS = (0.10, 0.15, 2.0, 0.10, 0.05, 0.01, 0.20, 0.03)


def _probability(name: str, value: float) -> float:
    frozen = float(value)
    if not math.isfinite(frozen) or not 0.0 <= frozen <= 1.0:
        raise OptionDContractError(f"{name} must be a finite probability")
    return frozen


def c4_scalar_birth_operations(
    *,
    scalar_mutation_rate: float,
    memory_mutation_probability: float,
) -> tuple[DrawOperation, ...]:
    """Encode one current C4 scalar-birth RNG control-flow shape.

    This is a standalone schedule description only.  It deliberately imports
    no Tensor World runtime and emits no genome integration adapter.
    """

    scalar_rate = _probability("scalar_mutation_rate", scalar_mutation_rate)
    memory_rate = _probability(
        "memory_mutation_probability", memory_mutation_probability
    )
    operations: list[DrawOperation] = [
        DrawOperation.random(),  # hidden-size decision
    ]
    for sigma in _C4_SCALAR_SIGMAS:
        decision = len(operations)
        operations.append(DrawOperation.random())
        operations.append(
            DrawOperation.gauss(
                sigma=sigma,
                predicate=DrawPredicate.random_lt(decision, scalar_rate),
            )
        )
    child_rate_decision = len(operations)
    operations.append(DrawOperation.random())
    operations.append(
        DrawOperation.gauss(
            sigma=0.005,
            predicate=DrawPredicate.random_lt(child_rate_decision, 0.2),
        )
    )
    memory_decision = len(operations)
    operations.append(DrawOperation.random())
    operations.append(
        DrawOperation.randbelow(
            2,
            predicate=DrawPredicate.random_lt(memory_decision, memory_rate),
        )
    )
    return tuple(operations)


def c4_scalar_schedule(
    world_birth_rates: Sequence[Sequence[float]],
    world_memory_probabilities: Sequence[Sequence[float]],
) -> DrawSchedule:
    """Build independent variable-length C4-shaped schedules per world."""

    if len(world_birth_rates) != len(world_memory_probabilities):
        raise OptionDContractError("C4 workload world counts must match")
    if not world_birth_rates:
        raise OptionDContractError("C4 workload requires at least one world")
    worlds: list[tuple[DrawOperation, ...]] = []
    for world, (rates, memory_rates) in enumerate(
        zip(world_birth_rates, world_memory_probabilities, strict=True)
    ):
        if len(rates) != len(memory_rates):
            raise OptionDContractError(
                f"C4 workload birth count mismatch for world {world}"
            )
        operations: list[DrawOperation] = []
        for rate, memory_rate in zip(rates, memory_rates, strict=True):
            birth = c4_scalar_birth_operations(
                scalar_mutation_rate=rate,
                memory_mutation_probability=memory_rate,
            )
            base = len(operations)
            for operation in birth:
                predicate = operation.predicate
                if predicate is not None:
                    predicate = DrawPredicate(
                        predicate.kind,
                        predicate.source_operation + base,
                        predicate.threshold_bits,
                    )
                operations.append(
                    DrawOperation(
                        operator=operation.operator,
                        argument=operation.argument,
                        mu_bits=operation.mu_bits,
                        sigma_bits=operation.sigma_bits,
                        predicate=predicate,
                    )
                )
        worlds.append(tuple(operations))
    schedule = DrawSchedule(tuple(worlds))
    schedule.validate()
    return schedule


__all__ = ["c4_scalar_birth_operations", "c4_scalar_schedule"]
