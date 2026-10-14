"""Static contract checks for tick-core status-1 predicate diagnostics."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from engine.perf.cuda_architecture_b_direct_resident_wdl_jungle_v1 import (
    CudaArchitectureBDirectResidentWdlJungleV1Error,
    _require_no_native_failure_records_v1,
)


ROOT = Path(__file__).resolve().parents[1]
HEADER = ROOT / "engine/perf/cuda_architecture_b_flattened_controller_v1.hpp"
KERNELS = ROOT / (
    "engine/perf/cuda_architecture_b_flattened_wdl_controller_v2_kernels.cu"
)
BINDING = ROOT / "engine/perf/cuda_architecture_b_flattened_controller_v1_binding.cpp"
PYTHON_FAILURE_SURFACE = ROOT / (
    "engine/perf/cuda_architecture_b_direct_resident_wdl_jungle_v1.py"
)


def _predicate_constants(header: str) -> list[str]:
    body = header[
        header.index("enum TickCorePredicateBitV1") : header.index(
            "kTickCorePredicateCountV1"
        )
    ]
    return re.findall(r"\b(kTickCorePredicate[A-Za-z0-9]+)\s*(?:,|=)", body)


def _tick_core_validation_block(kernels: str) -> str:
    start = kernels.index(
        "    if (\n        TICK_CORE_PREDICATE(",
        kernels.index("flattened_controller_tick_core_and_birth_plan_kernel_v1"),
    )
    end = kernels.index(
        "\n    if (thread == 0)",
        start,
    )
    return kernels[start:end]


def test_every_tick_core_predicate_has_one_short_circuit_wrapper() -> None:
    header = HEADER.read_text(encoding="utf-8")
    kernels = KERNELS.read_text(encoding="utf-8")
    constants = _predicate_constants(header)
    assert len(constants) == 36
    block = _tick_core_validation_block(kernels)
    assert block.count("TICK_CORE_PREDICATE(") == len(constants)
    for constant in constants:
        assert len(re.findall(rf"TICK_CORE_PREDICATE\(\s*{constant}\b", block)) == 1
        assert header.count(f"case {constant}:") == 1


def test_tick_core_validation_anchors_are_not_dropped() -> None:
    block = _tick_core_validation_block(KERNELS.read_text(encoding="utf-8"))
    anchors = (
        ("hidden_size > kMaxHiddenSize", 1),
        ("observation_size > kMaxObservationSize", 1),
        ("action_count > kMaxActionCount", 1),
        ("observation_size != expected_observation_size", 1),
        ("persistent_gmnw::context_valid(", 1),
        ("gru_context.gru_packed_width", 1),
        ("gmnw_control_context.policy_arm", 1),
        ("gmnw_trace_context.output_ticks_per_cell", 1),
        ("gmnw_decision_tape_context.present", 1),
        ("gmnw_selection_context.counts", 1),
        ("full_patch_p2 && view_radius != 1", 1),
        ("arguments.modular_context.mature_start_step", 1),
        ("full_wdl_v2 && !full_patch_p2", 1),
        ("full_eed_v1 && !full_patch_p2", 1),
        ("eed_context.resource_a_amount", 1),
        ("wdl_context.stats_ticks_per_cell", 4),
        ("!wdl_context_valid_v2(", 1),
        ("kPatchWdlRadiusOneObservationSize", 1),
        ("drive_width < 4", 1),
        ("cache_width <= 0", 1),
        ("cache_width > cache_capacity", 1),
        ("pending_capacity < capacity", 1),
        ("full_tick_state.chemistry_width < 6", 1),
        ("mechanics_config.start_tick < 0", 2),
        ("full_tick_state.drive_noise_sample_ticks", 2),
        ("ordinary_birth_config.rng_semantics_version != 1", 1),
        ("ordinary_birth_config.rng_semantic_epoch", 1),
        ("arguments.output_ticks_per_cell != 1", 1),
        ("arguments.resident_ticks_per_cell < ticks", 1),
        ("patch_p2_context.stats_ticks_per_cell < ticks", 1),
        ("event_state.capacity < 0", 1),
        ("event_state.reporting_rows < 0", 1),
        ("ordinary_birth_config.max_births_per_tick < 0", 1),
        ("!runtime_birth_layout_valid", 1),
        ("ordinary_birth_config.max_births_per_tick > capacity", 1),
        ("ordinary_birth_config.gauss_tape_capacity <= 0", 1),
    )
    assert len(anchors) == 36
    assert all(block.count(anchor) == expected for anchor, expected in anchors)


def test_diagnostic_receipt_and_failure_text_are_explicitly_non_authoritative() -> None:
    binding = BINDING.read_text(encoding="utf-8")
    failure_surface = PYTHON_FAILURE_SURFACE.read_text(encoding="utf-8")
    assert '"tick_core_predicate_mask"' in binding
    assert '"tick_core_predicate_names"' in binding
    assert '"tick_core_predicate_diagnostic_schema_version"' in binding
    assert '"diagnostic_only_first_short_circuited_predicate"' in binding
    assert '"tick_core_predicate_mask_is_first_failure"' in binding
    assert "tick_core_predicate_mask=" in failure_surface
    assert "tick_core_predicate_names=" in failure_surface


def test_python_failure_text_preserves_decoded_first_predicate() -> None:
    with pytest.raises(
        CudaArchitectureBDirectResidentWdlJungleV1Error,
        match=(
            r"status=1 tick_core_predicate_mask=65536 "
            r"tick_core_predicate_names=\['wdl_context_invalid'\]"
        ),
    ):
        _require_no_native_failure_records_v1(
            [
                {
                    "cell": 0,
                    "generation": 0,
                    "tick": 3359,
                    "stage": 1,
                    "status": 1,
                    "tick_core_predicate_mask": 65536,
                    "tick_core_predicate_names": ["wdl_context_invalid"],
                }
            ],
            sentinel=-1,
        )
