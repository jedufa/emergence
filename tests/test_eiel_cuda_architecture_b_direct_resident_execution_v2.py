from __future__ import annotations

from typing import Any

import pytest

from engine.perf.cuda_architecture_b_direct_resident_execution_v2 import (
    CudaArchitectureBDirectExecutionV2Error,
    create_run_control_v2,
    execute_direct_resident_windows_v2,
)
from engine.perf.cuda_architecture_b_direct_resident_runtime_v1 import (
    DIRECT_ARGUMENT_COUNT,
    GENERATION_COUNT_ARGUMENT_INDEX,
    RESULT_SCHEMA_VERSION,
)


class _Control:
    def __init__(self, run_id: str, owner_epoch: int) -> None:
        self.run_id = run_id
        self.owner_epoch = owner_epoch
        self.state = {
            "run_id": run_id,
            "controller_id": f"controller:{run_id}",
            "owner_epoch": owner_epoch,
            "status": "initialized",
            "run_started": False,
            "sealed": False,
            "active_generation": -1,
            "requested_generations": 0,
            "target_next_generation": -1,
            "cancel_requested": False,
            "awaiting_cancellation_checkpoint": False,
            "awaiting_final_output_ack": False,
        }

    def request_cancel(self, _request: Any) -> None: ...

    def acknowledge_cancellation_checkpoint(self, *_args: Any) -> None: ...

    def acknowledge_run_complete(self, *_args: Any) -> None: ...

    def publish_failure(self, *_args: Any) -> None: ...

    def final_result(self) -> None: ...

    def manifest(self) -> dict[str, Any]:
        return dict(self.state)


class _Module:
    RunControlV2 = _Control

    def __init__(
        self, *, completed: int = 4, physical_window_generations: int = 2
    ) -> None:
        self.completed = completed
        self.physical_window_generations = physical_window_generations
        self.calls: list[tuple[Any, ...]] = []

    def eiel_cuda_architecture_b_execute_direct_resident_windows_v2(
        self, control: _Control, *arguments: Any
    ) -> dict[str, Any]:
        self.calls.append((control, *arguments))
        control.state.update(
            {
                "status": "generation_complete",
                "run_started": True,
                "active_generation": self.completed - 1,
                "requested_generations": 4,
                "target_next_generation": 4,
                "cancel_requested": self.completed < 4,
                "awaiting_cancellation_checkpoint": self.completed < 4,
                "awaiting_final_output_ack": self.completed == 4,
            }
        )
        return {
            "schema_version": RESULT_SCHEMA_VERSION,
            "owner": "native_extension",
            "physical_window_generations": self.physical_window_generations,
            "logical_generation_count": 4,
            "window_count": 4 // self.physical_window_generations,
            "completed_generations": self.completed,
            "external_python_window_boundaries": 0,
            "direct_cpp_window_core_dispatch": True,
            "production_native_chunk_loop": True,
            "hidden_fallback": False,
            "runtime_jit": False,
            "native_run_control_v2": control.manifest(),
            "native_run_control_v2_trajectory_owner": True,
            "internal_pybind_trajectory_entry_trampoline": True,
            "python_tick_or_generation_ownership": False,
        }


@pytest.mark.parametrize("completed", [1, 3, 4])
def test_execute_accepts_only_exact_complete_or_cancelled_prefix(completed: int) -> None:
    module = _Module(completed=completed)
    control = create_run_control_v2(module, run_id="run-101", owner_epoch=3)
    arguments = [object() for _index in range(DIRECT_ARGUMENT_COUNT)]
    arguments[GENERATION_COUNT_ARGUMENT_INDEX] = 2
    execution = execute_direct_resident_windows_v2(
        module,
        run_control=control,
        physical_window_arguments=arguments,
        logical_generation_count=4,
        policy_mutation_enable=False,
        policy_mutation_mode="disabled",
        policy_mutation_rate=0.0,
        policy_mutation_std=0.0,
    )

    assert execution.completed_generations == completed
    assert execution.run_control is control
    assert len(module.calls) == 1
    assert len(module.calls[0]) == DIRECT_ARGUMENT_COUNT + 6
    assert execution.manifest["awaiting_final_output_ack"] is (completed == 4)
    assert execution.manifest["awaiting_cancellation_checkpoint"] is (
        completed < 4
    )


def test_execute_accepts_g1_physical_route_identity() -> None:
    module = _Module(physical_window_generations=1)
    control = create_run_control_v2(module, run_id="run-g1")
    arguments = [object() for _index in range(DIRECT_ARGUMENT_COUNT)]
    arguments[GENERATION_COUNT_ARGUMENT_INDEX] = 1

    execution = execute_direct_resident_windows_v2(
        module,
        run_control=control,
        physical_window_arguments=arguments,
        logical_generation_count=4,
        policy_mutation_enable=False,
        policy_mutation_mode="disabled",
        policy_mutation_rate=0.0,
        policy_mutation_std=0.0,
    )

    assert execution.direct_result["physical_window_generations"] == 1
    assert execution.direct_result["window_count"] == 4


def test_execute_rejects_nonqualified_shape_before_dispatch() -> None:
    module = _Module()
    control = create_run_control_v2(module, run_id="run-202")
    arguments = [object() for _index in range(DIRECT_ARGUMENT_COUNT)]
    arguments[GENERATION_COUNT_ARGUMENT_INDEX] = 2

    with pytest.raises(CudaArchitectureBDirectExecutionV2Error, match="exactly g4"):
        execute_direct_resident_windows_v2(
            module,
            run_control=control,
            physical_window_arguments=arguments,
            logical_generation_count=8,
            policy_mutation_enable=False,
            policy_mutation_mode="disabled",
            policy_mutation_rate=0.0,
            policy_mutation_std=0.0,
        )
    assert module.calls == []


def test_execute_fails_closed_on_false_ownership_metadata() -> None:
    module = _Module()
    original = module.eiel_cuda_architecture_b_execute_direct_resident_windows_v2

    def corrupt(control: _Control, *arguments: Any) -> dict[str, Any]:
        result = original(control, *arguments)
        result["python_tick_or_generation_ownership"] = True
        return result

    module.eiel_cuda_architecture_b_execute_direct_resident_windows_v2 = corrupt  # type: ignore[method-assign]
    control = create_run_control_v2(module, run_id="run-303")
    arguments = [object() for _index in range(DIRECT_ARGUMENT_COUNT)]
    arguments[GENERATION_COUNT_ARGUMENT_INDEX] = 2
    with pytest.raises(
        CudaArchitectureBDirectExecutionV2Error,
        match="python_tick_or_generation_ownership mismatch",
    ):
        execute_direct_resident_windows_v2(
            module,
            run_control=control,
            physical_window_arguments=arguments,
            logical_generation_count=4,
            policy_mutation_enable=False,
            policy_mutation_mode="disabled",
            policy_mutation_rate=0.0,
            policy_mutation_std=0.0,
        )
