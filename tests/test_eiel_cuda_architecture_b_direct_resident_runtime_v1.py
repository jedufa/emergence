from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from engine.perf.cuda_architecture_b_direct_resident_runtime_v1 import (
    DIRECT_ARGUMENT_COUNT,
    DIRECT_ENTRYPOINT,
    GENERATION_COUNT_ARGUMENT_INDEX,
    GENERATION_ARGUMENT_INDEX,
    CudaArchitectureBDirectRuntimeError,
    execute_direct_resident_windows_v1,
    resident_window_count_v1,
)


def _result(
    *,
    logical_generation_count: int = 4,
    physical_window_generations: int = 2,
    **overrides: object,
) -> dict[str, object]:
    result: dict[str, object] = {
        "schema_version": (
            "eiel_cuda_architecture_b_native_resident_window_controller_v1"
        ),
        "owner": "native_extension",
        "physical_window_generations": physical_window_generations,
        "logical_generation_count": logical_generation_count,
        "window_count": (
            logical_generation_count + physical_window_generations - 1
        )
        // physical_window_generations,
        "physical_window_alignment": "absolute_generation_lattice_v1",
        "completed_generations": logical_generation_count,
        "external_python_window_boundaries": 0,
        "direct_cpp_window_core_dispatch": True,
        "production_native_chunk_loop": True,
        "hidden_fallback": False,
        "runtime_jit": False,
    }
    result.update(overrides)
    return result


class _FakeModule(SimpleNamespace):
    def __init__(self, result: object) -> None:
        super().__init__()
        self.calls: list[tuple[object, ...]] = []

        def entrypoint(*args: object) -> object:
            self.calls.append(args)
            return result

        setattr(self, DIRECT_ENTRYPOINT, entrypoint)


def _execute(
    module: object,
    *,
    argument_count: int = DIRECT_ARGUMENT_COUNT,
    logical_generation_count: int = 4,
    physical_window_generations: int = 2,
) -> dict[str, Any]:
    arguments = [object() for _index in range(argument_count)]
    if argument_count > GENERATION_ARGUMENT_INDEX:
        arguments[GENERATION_ARGUMENT_INDEX] = 0
    if argument_count > GENERATION_COUNT_ARGUMENT_INDEX:
        arguments[GENERATION_COUNT_ARGUMENT_INDEX] = physical_window_generations
    return execute_direct_resident_windows_v1(
        module,
        physical_window_arguments=arguments,
        logical_generation_count=logical_generation_count,
        policy_mutation_enable=True,
        policy_mutation_mode="gaussian",
        policy_mutation_rate=0.2,
        policy_mutation_std=0.05,
    )


def test_exact_abi_invokes_direct_entrypoint_once() -> None:
    arguments = [object() for _index in range(DIRECT_ARGUMENT_COUNT)]
    arguments[GENERATION_ARGUMENT_INDEX] = 0
    arguments[GENERATION_COUNT_ARGUMENT_INDEX] = 2
    module = _FakeModule(_result())

    result = execute_direct_resident_windows_v1(
        module,
        physical_window_arguments=arguments,
        logical_generation_count=4,
        policy_mutation_enable=True,
        policy_mutation_mode="gaussian",
        policy_mutation_rate=0.2,
        policy_mutation_std=0.05,
    )

    assert result == _result()
    assert len(module.calls) == 1
    call = module.calls[0]
    assert len(call) == DIRECT_ARGUMENT_COUNT + 5
    assert list(call[:DIRECT_ARGUMENT_COUNT]) == arguments
    assert call[DIRECT_ARGUMENT_COUNT:] == (4, True, "gaussian", 0.2, 0.05)


@pytest.mark.parametrize("argument_count", (0, 156, 158))
def test_wrong_physical_argument_count_rejects_before_call(
    argument_count: int,
) -> None:
    module = _FakeModule(_result())

    with pytest.raises(CudaArchitectureBDirectRuntimeError):
        _execute(module, argument_count=argument_count)

    assert module.calls == []


@pytest.mark.parametrize("logical_generation_count", (0, 1, 2, 129, 130, True))
def test_invalid_logical_generation_count_rejects_before_call(
    logical_generation_count: int,
) -> None:
    module = _FakeModule(_result())

    with pytest.raises(CudaArchitectureBDirectRuntimeError):
        _execute(module, logical_generation_count=logical_generation_count)

    assert module.calls == []


@pytest.mark.parametrize("logical_generation_count", (3, 7, 14, 37, 128))
def test_irregular_logical_generation_count_uses_one_native_dispatch(
    logical_generation_count: int,
) -> None:
    module = _FakeModule(_result(logical_generation_count=logical_generation_count))

    result = _execute(
        module,
        logical_generation_count=logical_generation_count,
    )

    assert result["window_count"] == (logical_generation_count + 1) // 2
    assert len(module.calls) == 1
    assert module.calls[0][DIRECT_ARGUMENT_COUNT] == logical_generation_count


def test_g1_physical_window_count_uses_explicit_abi_value() -> None:
    module = _FakeModule(
        _result(logical_generation_count=4, physical_window_generations=1)
    )

    result = _execute(module, physical_window_generations=1)

    assert result["physical_window_generations"] == 1
    assert result["window_count"] == 4
    assert len(module.calls) == 1


@pytest.mark.parametrize(
    ("initial_generation", "logical_generation_count", "expected"),
    ((0, 7, 4), (6, 7, 4), (7, 7, 4), (7, 8, 5), (8, 8, 4)),
)
def test_resident_window_count_is_anchored_to_absolute_generation(
    initial_generation: int,
    logical_generation_count: int,
    expected: int,
) -> None:
    assert resident_window_count_v1(
        initial_generation=initial_generation,
        logical_generation_count=logical_generation_count,
    ) == expected


def test_missing_direct_entrypoint_rejects() -> None:
    with pytest.raises(CudaArchitectureBDirectRuntimeError):
        _execute(SimpleNamespace())


def test_non_mapping_result_rejects() -> None:
    module = _FakeModule([])

    with pytest.raises(CudaArchitectureBDirectRuntimeError):
        _execute(module)

    assert len(module.calls) == 1


def test_expected_result_metadata_succeeds() -> None:
    expected = _result()

    assert _execute(_FakeModule(expected)) == expected


@pytest.mark.parametrize(
    "overrides",
    (
        {"completed_generations": 2},
        {"hidden_fallback": True},
        {"runtime_jit": True},
        {"external_python_window_boundaries": 1},
    ),
)
def test_invalid_result_metadata_rejects(overrides: dict[str, object]) -> None:
    module = _FakeModule(_result(**overrides))

    with pytest.raises(CudaArchitectureBDirectRuntimeError):
        _execute(module)

    assert len(module.calls) == 1
