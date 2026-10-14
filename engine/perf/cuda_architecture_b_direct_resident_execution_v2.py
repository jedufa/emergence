"""Engine-owned invocation boundary for the native V2 CUDA run controller."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

from engine.perf.cuda_architecture_b_direct_resident_runtime_v1 import (
    DIRECT_ARGUMENT_COUNT,
    GENERATION_COUNT_ARGUMENT_INDEX,
    RESULT_SCHEMA_VERSION as NATIVE_RESULT_SCHEMA_VERSION,
    resident_window_count_v1,
    validate_physical_window_generations_v1,
)


DIRECT_ENTRYPOINT_V2: Final[str] = (
    "eiel_cuda_architecture_b_execute_direct_resident_windows_v2"
)
RUN_CONTROL_TYPE_V2: Final[str] = "RunControlV2"
LOGICAL_GENERATIONS_V2: Final[int] = 4


class CudaArchitectureBDirectExecutionV2Error(RuntimeError):
    """The V2 native controller or its immediate ownership receipt is invalid."""


@dataclass(frozen=True, slots=True)
class DirectResidentNativeExecutionV2:
    run_control: Any
    direct_result: dict[str, Any]
    manifest: dict[str, Any]
    completed_generations: int


def _fail(message: str) -> CudaArchitectureBDirectExecutionV2Error:
    return CudaArchitectureBDirectExecutionV2Error(message)


def _mapping(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise _fail(f"{name} must be a mapping")
    return dict(value)


def create_run_control_v2(
    module: Any,
    *,
    run_id: str,
    owner_epoch: int = 1,
) -> Any:
    """Create one fresh single-use native controller before trajectory dispatch."""

    if (
        type(run_id) is not str
        or not run_id
        or run_id.strip() != run_id
        or not run_id.isascii()
        or not run_id.isprintable()
    ):
        raise _fail("run_id must be a nonempty stable ASCII identifier")
    if type(owner_epoch) is not int or owner_epoch < 1:
        raise _fail("owner_epoch must be a positive exact integer")
    constructor = getattr(module, RUN_CONTROL_TYPE_V2, None)
    if not callable(constructor):
        raise _fail(f"CUDA extension is missing callable {RUN_CONTROL_TYPE_V2}")
    control = constructor(run_id, owner_epoch)
    for method in (
        "request_cancel",
        "acknowledge_cancellation_checkpoint",
        "acknowledge_run_complete",
        "publish_failure",
        "manifest",
        "final_result",
    ):
        if not callable(getattr(control, method, None)):
            raise _fail(f"native run control is missing callable {method}")
    manifest = _mapping(control.manifest(), name="fresh run-control manifest")
    expected = {
        "run_id": run_id,
        "owner_epoch": owner_epoch,
        "status": "initialized",
        "run_started": False,
        "sealed": False,
    }
    for name, expected_value in expected.items():
        actual = manifest.get(name)
        if type(actual) is not type(expected_value) or actual != expected_value:
            raise _fail(
                f"fresh run-control manifest {name} mismatch: "
                f"expected={expected_value!r} actual={actual!r}"
            )
    return control


def _validate_native_result_v2(
    value: Any,
    *,
    physical_window_generations: int,
    logical_generation_count: int,
    manifest: Mapping[str, Any],
) -> tuple[dict[str, Any], int]:
    result = _mapping(value, name="direct V2 CUDA result")
    completed = result.get("completed_generations")
    if (
        type(completed) is not int
        or completed < 1
        or completed > logical_generation_count
    ):
        raise _fail("direct V2 result completed_generations is not an exact prefix")
    expected = {
        "schema_version": NATIVE_RESULT_SCHEMA_VERSION,
        "owner": "native_extension",
        "physical_window_generations": physical_window_generations,
        "logical_generation_count": logical_generation_count,
        "window_count": resident_window_count_v1(
            initial_generation=0,
            logical_generation_count=logical_generation_count,
            physical_window_generations=physical_window_generations,
        ),
        "external_python_window_boundaries": 0,
        "direct_cpp_window_core_dispatch": True,
        "production_native_chunk_loop": True,
        "hidden_fallback": False,
        "runtime_jit": False,
        "native_run_control_v2_trajectory_owner": True,
        "internal_pybind_trajectory_entry_trampoline": True,
        "python_tick_or_generation_ownership": False,
    }
    for name, expected_value in expected.items():
        actual = result.get(name)
        if type(actual) is not type(expected_value) or actual != expected_value:
            raise _fail(
                f"direct V2 result {name} mismatch: "
                f"expected={expected_value!r} actual={actual!r}"
            )
    embedded = _mapping(
        result.get("native_run_control_v2"),
        name="embedded run-control manifest",
    )
    for name in (
        "run_id",
        "controller_id",
        "owner_epoch",
        "status",
        "active_generation",
        "requested_generations",
        "target_next_generation",
        "cancel_requested",
        "awaiting_cancellation_checkpoint",
        "awaiting_final_output_ack",
        "sealed",
    ):
        if embedded.get(name) != manifest.get(name):
            raise _fail(f"embedded run-control manifest drifted at {name}")
    if manifest.get("status") != "generation_complete" or manifest.get("sealed"):
        raise _fail("native trajectory did not stop at an unsealed generation boundary")
    if manifest.get("requested_generations") != logical_generation_count:
        raise _fail("native run control changed the frozen generation target")
    if completed == logical_generation_count:
        if (
            manifest.get("awaiting_final_output_ack") is not True
            or manifest.get("awaiting_cancellation_checkpoint") is not False
            or manifest.get("cancel_requested") is not False
        ):
            raise _fail("complete native trajectory is not awaiting final output ack")
    elif (
        manifest.get("cancel_requested") is not True
        or manifest.get("awaiting_cancellation_checkpoint") is not True
        or manifest.get("awaiting_final_output_ack") is not False
    ):
        raise _fail("partial native trajectory is not awaiting cancellation checkpoint")
    return result, completed


def execute_direct_resident_windows_v2(
    module: Any,
    *,
    run_control: Any,
    physical_window_arguments: Sequence[Any],
    logical_generation_count: int,
    policy_mutation_enable: Any,
    policy_mutation_mode: Any,
    policy_mutation_rate: Any,
    policy_mutation_std: Any,
) -> DirectResidentNativeExecutionV2:
    """Submit one native-owned g4 trajectory without retry or fallback."""

    if type(logical_generation_count) is not int or (
        logical_generation_count != LOGICAL_GENERATIONS_V2
    ):
        raise _fail("the qualified V2 production envelope requires exactly g4")
    arguments = tuple(physical_window_arguments)
    if len(arguments) != DIRECT_ARGUMENT_COUNT:
        raise _fail(
            f"direct V2 dispatch requires exactly {DIRECT_ARGUMENT_COUNT} arguments"
        )
    physical_window_generations = validate_physical_window_generations_v1(
        arguments[GENERATION_COUNT_ARGUMENT_INDEX]
    )
    entrypoint = getattr(module, DIRECT_ENTRYPOINT_V2, None)
    if not callable(entrypoint):
        raise _fail(f"CUDA extension is missing callable {DIRECT_ENTRYPOINT_V2}")
    fresh = _mapping(run_control.manifest(), name="pre-dispatch run-control manifest")
    if fresh.get("run_started") is not False or fresh.get("sealed") is not False:
        raise _fail("direct V2 dispatch requires a fresh single-use run control")

    raw_result = entrypoint(
        run_control,
        *arguments,
        logical_generation_count,
        policy_mutation_enable,
        policy_mutation_mode,
        policy_mutation_rate,
        policy_mutation_std,
    )
    manifest = _mapping(run_control.manifest(), name="post-dispatch run-control manifest")
    result, completed = _validate_native_result_v2(
        raw_result,
        physical_window_generations=physical_window_generations,
        logical_generation_count=logical_generation_count,
        manifest=manifest,
    )
    return DirectResidentNativeExecutionV2(
        run_control=run_control,
        direct_result=result,
        manifest=manifest,
        completed_generations=completed,
    )


__all__ = [
    "DIRECT_ENTRYPOINT_V2",
    "LOGICAL_GENERATIONS_V2",
    "CudaArchitectureBDirectExecutionV2Error",
    "DirectResidentNativeExecutionV2",
    "create_run_control_v2",
    "execute_direct_resident_windows_v2",
]
