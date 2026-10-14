from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HEADER = ROOT / "engine/perf/cuda_architecture_b_flattened_controller_v1.hpp"
KERNELS = ROOT / "engine/perf/cuda_architecture_b_flattened_controller_v1_kernels.cu"
BINDING = ROOT / "engine/perf/cuda_architecture_b_flattened_controller_v1_binding.cpp"
WINDOW_CONTROLLER = (
    ROOT / "engine/perf/cuda_architecture_b_resident_window_controller_v1.hpp"
)


def _function_body(source: str, name: str) -> str:
    start = source.index(name)
    opening = source.index("{", start)
    depth = 1
    cursor = opening + 1
    while depth:
        if source[cursor] == "{":
            depth += 1
        elif source[cursor] == "}":
            depth -= 1
        cursor += 1
    return source[opening + 1 : cursor - 1]


def test_v2_inter_tick_hook_is_optional_and_v1_remains_unsynchronized() -> None:
    header = HEADER.read_text(encoding="utf-8")
    kernels = KERNELS.read_text(encoding="utf-8")
    binding = BINDING.read_text(encoding="utf-8")

    callback_type = (
        "const std::function<void(int64_t, int64_t, int64_t)>& inter_tick_control"
    )
    assert callback_type in header
    assert callback_type in kernels

    helper = _function_body(kernels, "maybe_open_inter_tick_control_v2(")
    assert "tick != 0" in helper
    assert "tick + 1 >= ticks" in helper
    assert "!inter_tick_control" in helper
    assert helper.index("cudaStreamSynchronize(stream)") < helper.index(
        "inter_tick_control(generation"
    )
    assert "const int64_t completed_tick = tick;" in helper
    assert "start_tick + tick" not in helper

    controller = _function_body(
        kernels,
        "run_flat_dynamic_policy_food_flattened_controller_v1(",
    )
    hook_call = controller.index("maybe_open_inter_tick_control_v2(")
    tick_loop_start = controller.rindex(
        "for (int64_t tick = 0; tick < ticks; ++tick)",
        0,
        hook_call,
    )
    assert "maybe_open_inter_tick_control_v2(" in controller[tick_loop_start:]
    hook_dispatch = controller[hook_call : controller.index(");", hook_call)]
    assert "cudaStreamSynchronize" not in hook_dispatch

    assert "no_inter_tick_control" in binding
    assert re.search(
        r"static const std::function<void\(int64_t, int64_t, int64_t\)>\s+"
        r"no_inter_tick_control;",
        binding,
    )
    assert re.search(
        r"inter_tick_control\s*=\s*no_inter_tick_control",
        binding,
    )
    assert re.search(
        r"static_cast<float>\(w_fear_attack\),\s*"
        r"exact_cpu_policy_refill,\s*"
        r"hapb_ordinary_birth_commit,\s*"
        r"policy_birth_segment_refill,\s*"
        r"gaussian_segment_refill,\s*"
        r"inter_tick_control,\s*stream,\s*false",
        binding,
    )


def test_hapb_birth_bridge_signature_and_dispatch_match_the_public_contract() -> None:
    header = HEADER.read_text(encoding="utf-8")
    kernels = KERNELS.read_text(encoding="utf-8")

    callback_type = "const std::function<void(int64_t, int64_t)>& hapb_ordinary_birth_commit"
    assert callback_type in header
    assert callback_type in kernels

    controller = _function_body(
        kernels,
        "run_flat_dynamic_policy_food_flattened_controller_v1(",
    )
    prepare = controller.index("flattened_controller_birth_prepare_kernel_v1<<<")
    bridge = controller.index("hapb_ordinary_birth_commit(", prepare)
    commit = controller.index(
        "flattened_controller_birth_transaction_and_settlement_kernel_v1<<<",
        prepare,
    )
    assert prepare < bridge < commit


def test_terminal_generation_prefix_stops_both_native_controller_loops() -> None:
    binding = BINDING.read_text(encoding="utf-8")
    window = WINDOW_CONTROLLER.read_text(encoding="utf-8")

    marker = "native_run_control_v2_stop_after_completed_prefix"
    assert f'receipt["{marker}"]' in binding
    assert "manifest.awaiting_cancellation_checkpoint || manifest.sealed" in binding
    assert re.search(rf'receipt\.contains\(\s*"{marker}"\s*\)', window)
    assert "|| stop_after_completed_prefix" in window
    assert window.index("|| stop_after_completed_prefix") < window.index(
        "event_archive::drain_event_window_v1"
    )
    assert "if (archived_events.cells.empty())" in window
    assert window.index("if (archived_events.cells.empty())") < window.index(
        "event_archive::materialize_event_history_v1"
    )
