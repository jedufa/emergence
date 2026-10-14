from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from engine.perf import cuda_architecture_b_direct_resident_modular_v1 as modular
from tests.test_eiel_cuda_architecture_b_direct_resident_modular_v1 import (
    _cell_tensors,
    _lifecycle_capability,
    _patch,
)


TURNOVER_MUTATION_CAPABILITY_FIELDS = (
    ("turnover_refine_descendant_parameter_qualification", True),
    (
        "turnover_mutation_receipt_raw_layout_version",
        modular.MODULAR_TURNOVER_MUTATION_RAW_LAYOUT_VERSION_V1,
    ),
    (
        "turnover_mutation_receipt_sequence_capacity",
        modular.MODULAR_TURNOVER_MUTATION_SEQUENCE_CAPACITY_V1,
    ),
    ("mutation_receipt_sequence_v1", True),
    ("mutated_checkpoint_fresh_owner", True),
    ("turnover_mutation_qualified_generation_horizon", 3),
    ("turnover_mutation_overflow_test_generation", 4),
    ("ordinary_birth_mutation", True),
    ("turnover_mutation_g8_g32", False),
    ("turnover_mutation_default_production", False),
)


def _exact_capability() -> dict[str, Any]:
    return {
        **_lifecycle_capability(),
        **dict(TURNOVER_MUTATION_CAPABILITY_FIELDS),
    }


@pytest.fixture(scope="module")
def mutation_prepared() -> modular.PreparedDirectResidentModularV1:
    lifecycle = modular.prepare_direct_resident_modular_lifecycle_v2(
        _patch(),
        [_cell_tensors()],
    )
    draws = torch.full(
        (
            1,
            modular.MODULAR_TURNOVER_MUTATION_SEQUENCE_CAPACITY_V1,
            2,
        ),
        0.6,
        dtype=torch.float64,
    )
    duplicate_descendant_mask = torch.zeros((1, 4, 16), dtype=torch.bool)
    return modular.with_modular_turnover_mutation_qualification_v1(
        lifecycle,
        mutation_draws=draws,
        duplicate_descendant_mask=duplicate_descendant_mask,
    )


def _mock_module(
    capability: dict[str, Any],
    native_calls: list[tuple[Any, ...]],
) -> SimpleNamespace:
    def capability_entrypoint() -> dict[str, Any]:
        return capability

    def native_entrypoint(*arguments: Any) -> dict[str, Any]:
        native_calls.append(arguments)
        return {
            "modular_native_owner": True,
            "modular_native_controller_abi_version": 2,
            "modular_physical_gate_entrypoint": True,
            "modular_python_tick_or_generation_ownership": False,
            "modular_lifecycle_supported": True,
            "hidden_fallback": False,
            "runtime_jit": False,
        }

    module = SimpleNamespace()
    setattr(
        module,
        modular.MODULAR_LIFECYCLE_CAPABILITY_ENTRYPOINT_V2,
        capability_entrypoint,
    )
    setattr(
        module,
        modular.MODULAR_LIFECYCLE_PHYSICAL_ENTRYPOINT_V2,
        native_entrypoint,
    )
    return module


def _wrong_value(expected: Any) -> Any:
    if type(expected) is bool:
        return not expected
    return expected + 1


def _wrong_type(expected: Any) -> Any:
    if type(expected) is bool:
        return int(expected)
    return str(expected)


@pytest.mark.parametrize(
    ("field", "expected"),
    TURNOVER_MUTATION_CAPABILITY_FIELDS,
    ids=[field for field, _expected in TURNOVER_MUTATION_CAPABILITY_FIELDS],
)
@pytest.mark.parametrize("case", ("missing", "wrong-value", "wrong-type"))
def test_turnover_mutation_capability_rejects_before_native_entrypoint(
    mutation_prepared: modular.PreparedDirectResidentModularV1,
    field: str,
    expected: Any,
    case: str,
) -> None:
    capability = _exact_capability()
    if case == "missing":
        capability.pop(field)
    elif case == "wrong-value":
        capability[field] = _wrong_value(expected)
    else:
        capability[field] = _wrong_type(expected)
    native_calls: list[tuple[Any, ...]] = []

    with pytest.raises(modular.CudaArchitectureBDirectResidentModularV1Error):
        modular.execute_direct_resident_modular_lifecycle_physical_v2(
            _mock_module(capability, native_calls),
            mutation_prepared,
            generation_count=1,
            ticks_per_generation=1,
        )

    assert native_calls == []


def test_turnover_mutation_exact_capability_reaches_mock_native_entrypoint(
    mutation_prepared: modular.PreparedDirectResidentModularV1,
) -> None:
    native_calls: list[tuple[Any, ...]] = []

    result = modular.execute_direct_resident_modular_lifecycle_physical_v2(
        _mock_module(_exact_capability(), native_calls),
        mutation_prepared,
        generation_count=1,
        ticks_per_generation=1,
    )

    assert result["modular_native_owner"] is True
    assert len(native_calls) == 1
    assert all(tensor.device.type == "cpu" for tensor in native_calls[0][0])
