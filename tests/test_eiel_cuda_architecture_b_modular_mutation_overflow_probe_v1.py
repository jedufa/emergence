from __future__ import annotations

from typing import Any

import pytest
import torch

from engine.perf import cuda_architecture_b_direct_resident_modular_v1 as modular
from tests.test_eiel_cuda_architecture_b_direct_resident_modular_v1 import (
    CAPACITY,
    CELLS,
    MUTATION_SEQUENCE_CAPACITY,
    _cell_tensors,
    _patch,
)


def _qualified() -> modular.PreparedDirectResidentModularV1:
    prepared = modular.prepare_direct_resident_modular_lifecycle_v2(
        _patch(),
        [_cell_tensors()],
    )
    return modular.with_modular_turnover_mutation_qualification_v1(
        prepared,
        mutation_draws=torch.full(
            (CELLS, MUTATION_SEQUENCE_CAPACITY, 2),
            0.6,
            dtype=torch.float64,
        ),
        duplicate_descendant_mask=torch.zeros(
            (CELLS, CAPACITY, 16),
            dtype=torch.bool,
        ),
    )


def test_normal_g4_mutation_request_fails_before_capability_or_dispatch() -> None:
    class Module:
        capability_called = False
        dispatch_called = False

        @classmethod
        def eiel_cuda_architecture_b_flattened_modular_lifecycle_capability_v2(
            cls,
        ) -> dict[str, Any]:
            cls.capability_called = True
            return {}

        @classmethod
        def eiel_cuda_architecture_b_execute_direct_resident_modular_lifecycle_v2(
            cls,
            *_arguments: Any,
        ) -> dict[str, Any]:
            cls.dispatch_called = True
            return {}

    with pytest.raises(
        modular.CudaArchitectureBDirectResidentModularV1Error,
        match="successful execution only through G3",
    ):
        modular.execute_direct_resident_modular_lifecycle_v2(
            Module(),
            _qualified(),
            logical_generation_count=4,
            run_id="normal-g4-must-not-dispatch",
        )

    assert Module.capability_called is False
    assert Module.dispatch_called is False


@pytest.mark.parametrize(
    ("generation_count", "qualified"),
    [(3, True), (4, False)],
)
def test_overflow_probe_requires_exact_qualified_g4(
    generation_count: int,
    qualified: bool,
) -> None:
    prepared = (
        _qualified()
        if qualified
        else modular.prepare_direct_resident_modular_lifecycle_v2(
            _patch(),
            [_cell_tensors()],
        )
    )

    with pytest.raises(
        modular.CudaArchitectureBDirectResidentModularV1Error,
        match="overflow probe requires the qualified G4 fixture",
    ):
        modular.execute_direct_resident_modular_lifecycle_v2(
            object(),
            prepared,
            logical_generation_count=generation_count,
            run_id="invalid-overflow-probe",
            overflow_probe=True,
        )
