"""Engine-owned invocation boundary for the Stage 3 direct CUDA controller."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Final


DIRECT_ENTRYPOINT: Final[str] = (
    "eiel_cuda_architecture_b_execute_direct_resident_windows_v1"
)
DIRECT_ARGUMENT_COUNT: Final[int] = 172
GENERATION_ARGUMENT_INDEX: Final[int] = 131
RESULT_SCHEMA_VERSION: Final[str] = (
    "eiel_cuda_architecture_b_native_resident_window_controller_v1"
)
PHYSICAL_WINDOW_GENERATIONS: Final[int] = 2
MIN_PHYSICAL_WINDOW_GENERATIONS: Final[int] = 1
MAX_PHYSICAL_WINDOW_GENERATIONS: Final[int] = 2
MIN_LOGICAL_GENERATIONS: Final[int] = 3
MAX_LOGICAL_GENERATIONS: Final[int] = 128
GENERATION_COUNT_ARGUMENT_INDEX: Final[int] = 155


class CudaArchitectureBDirectRuntimeError(RuntimeError):
    """The direct resident CUDA ABI or returned ownership receipt is invalid."""


def _fail(message: str) -> CudaArchitectureBDirectRuntimeError:
    return CudaArchitectureBDirectRuntimeError(message)


def _validate_generation_count(value: Any) -> int:
    if (
        type(value) is not int
        or value < MIN_LOGICAL_GENERATIONS
        or value > MAX_LOGICAL_GENERATIONS
    ):
        raise _fail("logical_generation_count must be an integer in [3, 128]")
    return value


def validate_physical_window_generations_v1(value: Any) -> int:
    if (
        type(value) is not int
        or value < MIN_PHYSICAL_WINDOW_GENERATIONS
        or value > MAX_PHYSICAL_WINDOW_GENERATIONS
    ):
        raise _fail("physical_window_generations must be an exact integer in [1, 2]")
    return value


def resident_window_count_v1(
    *,
    initial_generation: int,
    logical_generation_count: int,
    physical_window_generations: int = PHYSICAL_WINDOW_GENERATIONS,
) -> int:
    """Count physical windows on the absolute-generation lattice."""

    if type(initial_generation) is not int or initial_generation < 0:
        raise _fail("initial_generation must be a nonnegative exact integer")
    if (
        type(logical_generation_count) is not int
        or logical_generation_count < 1
        or logical_generation_count > MAX_LOGICAL_GENERATIONS
    ):
        raise _fail("window logical_generation_count must be an integer in [1, 128]")
    window = validate_physical_window_generations_v1(physical_window_generations)
    generations = logical_generation_count
    return (
        initial_generation % window
        + generations
        + window
        - 1
    ) // window


def _validate_result(
    value: Any,
    *,
    initial_generation: int,
    physical_window_generations: int,
    logical_generation_count: int,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise _fail("direct resident CUDA result must be a mapping")
    result = dict(value)
    expected = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "owner": "native_extension",
        "physical_window_generations": physical_window_generations,
        "logical_generation_count": logical_generation_count,
        "window_count": resident_window_count_v1(
            initial_generation=initial_generation,
            logical_generation_count=logical_generation_count,
            physical_window_generations=physical_window_generations,
        ),
        "physical_window_alignment": "absolute_generation_lattice_v1",
        "completed_generations": logical_generation_count,
        "external_python_window_boundaries": 0,
        "direct_cpp_window_core_dispatch": True,
        "production_native_chunk_loop": True,
        "hidden_fallback": False,
        "runtime_jit": False,
    }
    for name, expected_value in expected.items():
        if name not in result:
            raise _fail(f"direct resident CUDA result is missing {name}")
        actual = result[name]
        if type(actual) is not type(expected_value) or actual != expected_value:
            raise _fail(
                f"direct resident CUDA result {name} mismatch: "
                f"expected={expected_value!r} actual={actual!r}"
            )
    return result


def execute_direct_resident_windows_v1(
    module: Any,
    *,
    physical_window_arguments: Sequence[Any],
    logical_generation_count: int,
    policy_mutation_enable: Any,
    policy_mutation_mode: Any,
    policy_mutation_rate: Any,
    policy_mutation_std: Any,
) -> dict[str, Any]:
    """Invoke one direct resident trajectory and validate its ownership receipt.

    This function does not prepare state, select a backend, retry, or fall back.
    Once the extension call begins, any exception is terminal for its caller.
    """

    generation_count = _validate_generation_count(logical_generation_count)
    arguments = tuple(physical_window_arguments)
    if len(arguments) != DIRECT_ARGUMENT_COUNT:
        raise _fail(
            "direct resident CUDA dispatch requires exactly "
            f"{DIRECT_ARGUMENT_COUNT} prepared arguments"
        )
    initial_generation = arguments[GENERATION_ARGUMENT_INDEX]
    if type(initial_generation) is not int or initial_generation < 0:
        raise _fail("prepared initial generation must be a nonnegative exact integer")
    physical_window_generations = validate_physical_window_generations_v1(
        arguments[GENERATION_COUNT_ARGUMENT_INDEX]
    )
    entrypoint = getattr(module, DIRECT_ENTRYPOINT, None)
    if not callable(entrypoint):
        raise _fail(f"CUDA extension is missing callable {DIRECT_ENTRYPOINT}")

    result = entrypoint(
        *arguments,
        generation_count,
        policy_mutation_enable,
        policy_mutation_mode,
        policy_mutation_rate,
        policy_mutation_std,
    )
    return _validate_result(
        result,
        initial_generation=initial_generation,
        physical_window_generations=physical_window_generations,
        logical_generation_count=generation_count,
    )


__all__ = [
    "DIRECT_ARGUMENT_COUNT",
    "DIRECT_ENTRYPOINT",
    "GENERATION_ARGUMENT_INDEX",
    "CudaArchitectureBDirectRuntimeError",
    "execute_direct_resident_windows_v1",
    "resident_window_count_v1",
]
