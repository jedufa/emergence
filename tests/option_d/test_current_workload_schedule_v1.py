from __future__ import annotations

import pytest

from engine.option_d import OptionDContractError
from engine.option_d.workloads import (
    c4_scalar_birth_operations,
    c4_scalar_schedule,
)


def test_one_c4_birth_has_current_source_operator_shape() -> None:
    operations = c4_scalar_birth_operations(
        scalar_mutation_rate=0.1,
        memory_mutation_probability=0.025,
    )

    assert len(operations) == 21
    assert sum(operation.operator.name == "RANDOM" for operation in operations) == 11
    assert sum(operation.operator.name == "GAUSS" for operation in operations) == 9
    assert sum(operation.operator.name == "RANDBELOW" for operation in operations) == 1
    assert sum(operation.predicate is not None for operation in operations) == 10


def test_multi_birth_predicate_offsets_remain_local_to_each_birth() -> None:
    schedule = c4_scalar_schedule(
        ((0.01, 0.2, 1.0), (0.05,)),
        ((0.002, 0.05, 1.0), (0.01,)),
    )

    assert tuple(len(world) for world in schedule.worlds) == (63, 21)
    for birth in range(3):
        base = birth * 21
        predicates = [
            operation.predicate
            for operation in schedule.worlds[0][base : base + 21]
            if operation.predicate is not None
        ]
        assert predicates
        assert all(base <= predicate.source_operation < base + 21 for predicate in predicates)


@pytest.mark.parametrize("probability", [-0.1, 1.1, float("nan")])
def test_invalid_probabilities_fail_contract(probability: float) -> None:
    with pytest.raises(OptionDContractError, match="finite probability"):
        c4_scalar_birth_operations(
            scalar_mutation_rate=probability,
            memory_mutation_probability=0.01,
        )
