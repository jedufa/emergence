from __future__ import annotations

from pathlib import Path

import pytest
import torch

from engine.perf import cuda_architecture_b_direct_resident_gmnw_gru_v1 as gmnw
from engine.perf.cuda_architecture_b_direct_resident_source_allocator_v1 import (
    _allocate_gmnw_qualification_trace_outputs,
)


ROOT = Path(__file__).resolve().parents[1]
BINDING = ROOT / "engine/perf/cuda_architecture_b_flattened_wdl_controller_v2_binding.cpp"
CONTEXT = ROOT / "engine/perf/cuda_architecture_b_flattened_gmnw_controller_v1.hpp"
KERNELS = ROOT / "engine/perf/cuda_architecture_b_flattened_wdl_controller_v2_kernels.cu"


def test_trace_allocator_has_exact_bounded_shapes_and_sentinels() -> None:
    outputs = _allocate_gmnw_qualification_trace_outputs(
        capacity=3,
        hidden_width=64,
        task_state_width=9,
        device=torch.device("cpu"),
        output_ticks=80,
        batch_cells=2,
    )

    assert tuple(outputs) == (
        "gmnw_trace_hidden_before",
        "gmnw_trace_hidden_after",
        "gmnw_trace_task_state_before",
        "gmnw_trace_task_state_after",
    )
    for name in ("gmnw_trace_hidden_before", "gmnw_trace_hidden_after"):
        assert outputs[name].shape == (2, 80, 3, 64)
        assert outputs[name].dtype == torch.float32
        assert torch.isnan(outputs[name]).all()
    for name in (
        "gmnw_trace_task_state_before",
        "gmnw_trace_task_state_after",
    ):
        assert outputs[name].shape == (2, 80, 3, 9)
        assert outputs[name].dtype == torch.int64
        assert torch.all(outputs[name] == -1)


@pytest.mark.parametrize("output_ticks", (0, 161))
def test_trace_allocator_rejects_out_of_bounds_tick_shapes(output_ticks: int) -> None:
    with pytest.raises(ValueError, match=r"output_ticks must be an exact integer in \[1,160\]"):
        _allocate_gmnw_qualification_trace_outputs(
            capacity=1,
            hidden_width=64,
            task_state_width=9,
            device=torch.device("cpu"),
            output_ticks=output_ticks,
            batch_cells=1,
        )


def test_trace_contract_names_columns_and_optional_suffix_are_versioned() -> None:
    binding = BINDING.read_text(encoding="utf-8")
    context = CONTEXT.read_text(encoding="utf-8")

    assert gmnw.GMNW_QUALIFICATION_TRACE_VERSION_V1 == 1
    assert gmnw.GMNW_QUALIFICATION_TRACE_TASK_STATE_COLUMNS_V1 == (
        "cue_identity",
        "cue_phase",
        "episode_index",
        "phase_tick",
        "delay_remaining",
        "decision_due",
        "correct_action",
        "selected_private_action",
        "correctness",
    )
    assert len(gmnw.GMNW_QUALIFICATION_TRACE_TENSOR_NAMES_V1) == 7
    assert "const std::size_t expected_tensor_count = 23U" in binding
    assert "(has_control_tensors ? 4U : 0U)" in binding
    assert "(has_trace_tensors ? 7U : 0U)" in binding
    assert "(has_decision_tape_tensors ? 3U : 0U)" in binding
    assert "tensors.size() == expected_tensor_count" in binding
    assert "trace_ticks == 80 || trace_ticks == 160" in binding
    assert "kQualificationTraceTaskStateColumnsV1 = 9" in context
    assert 'out["qualification_trace_version"]' in binding


def test_gmnw_row_shape_storage_outlives_int_array_ref() -> None:
    binding = BINDING.read_text(encoding="utf-8")
    start = binding.index(
        "gmnw_controller::FlatGmnwContextV1 validated_gmnw_context_v1("
    )
    end = binding.index(
        "pybind11::dict execute_direct_resident_gru_v3(",
        start,
    )
    gmnw_validation = binding[start:end]

    assert (
        "const std::array<std::int64_t, 2> row_shape{cells, capacity};"
        in gmnw_validation
    )
    assert "torch::IntArrayRef(row_shape)" in gmnw_validation
    assert "torch::IntArrayRef({cells, capacity})" not in gmnw_validation


def test_kernel_records_hidden_around_inference_and_task_after_advance() -> None:
    source = KERNELS.read_text(encoding="utf-8")
    policy_start = source.index(
        "if (thread == 0 && status_out[cell] == 0 && full_gru_v3)"
    )
    policy_end = source.index("__syncthreads();", policy_start)
    policy = source[policy_start:policy_end]
    before = policy.index("gmnw_trace_context.hidden_before")
    inference = policy.index("service_stable_gru_gumbel_policy_serial(")
    after = policy.index("gmnw_trace_context.hidden_after")
    assert before < inference < after

    advance = source.index("task_lethal = persistent_gmnw::advance_slot(")
    post_task = source.index("gmnw_trace_context.task_state_after", advance)
    neutralize = source.index(
        "actions_by_slot_out[tick_capacity_offset + slot] = kRestAction", advance
    )
    assert advance < post_task < neutralize

    trace_helpers = source[
        source.index("write_gmnw_qualification_task_state_v1") :
        source.index("__global__ void flattened_controller_tick_core_and_birth_plan_kernel_v1")
    ]
    assert "gumbel" not in trace_helpers.lower()


def test_kernel_records_qualification_decision_before_task_advance() -> None:
    source = KERNELS.read_text(encoding="utf-8")
    loop = source[source.index("if (thread == 0 && full_gmnw_v1)") :]
    record = loop.index("record_gmnw_qualification_decision_v1(")
    advance = loop.index("persistent_gmnw::advance_slot(")

    assert record < advance
    helper_start = source.index("record_gmnw_qualification_decision_v1(")
    helper = source[
        helper_start : source.index(
            "__global__ void flattened_controller_tick_core", helper_start
        )
    ]
    assert "const int64_t row = stream" in helper
    assert "kQualificationEpisodesPerStreamV1 + episode" in helper
    assert "tape.present[cell_row] != 0" in helper
    assert "cue >= context.cue_alphabet[cell] || cue >= 4" in helper
    assert "selected_private_action == expected_actions[cue] ? 1 : 0" in helper


def test_trace_aliases_existing_active_order_and_stays_out_of_checkpoint_banks() -> None:
    source = Path(gmnw.__file__).read_text(encoding="utf-8")

    assert '"gmnw_trace_active_count": outputs["active_count"]' in source
    assert '"gmnw_trace_ordered_agent_ids": outputs["ordered_agent_ids"]' in source
    assert '"gmnw_trace_ordered_slots": outputs["ordered_slots"]' in source
    assert "qualification_trace=qualification_trace" in source
    trace_block = source[
        source.index("qualification_trace: dict[str, torch.Tensor] = {}") :
        source.index("initial_projection = base.initial_projection")
    ]
    assert "live_banks" not in trace_block
    assert "outputs.update" not in trace_block
