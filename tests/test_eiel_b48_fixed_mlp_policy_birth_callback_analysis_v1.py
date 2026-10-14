from __future__ import annotations

import json
from pathlib import Path

from tools.scripts.eiel_b48_fixed_mlp_policy_birth_callback_analysis_v1 import (
    analyze_execution_receipt,
)


def test_policy_birth_callback_analysis_closes_rows_and_avoids_h2d_double_count(
    tmp_path: Path,
) -> None:
    rows = []
    for scale in (1, 2):
        rows.append(
            {
                "operation": "refill_fixed_mlp_policy_window_v1",
                "host_wall_time_ns": 1_000 * scale,
                "device_to_host_time_ns": 10 * scale,
                "torch_random_generation_time_ns": 300 * scale,
                "torch_compare_pack_time_ns": 500 * scale,
                "policy_allocation_stack_copy_time_ns": 100 * scale,
                "host_to_device_time_ns": 25 * scale,
                "batch_cells": 48,
                "physical_capacity": 124,
                "policy_row_width": 9_991,
                "policy_crossover_mask_encoding": "bitpack_lsb0_v1",
                "policy_crossover_packed_row_bytes": 1_249,
                "option_d_m4_device_policy_refill_enabled": False,
                "option_d_m4_device_policy_refill_authority": "disabled",
            }
        )
    execution = {
        "native_result": {
            "policy_birth_segment_refill_calls_inside_tick_loop": 2,
            "policy_birth_segment_refill_receipts": rows,
            "policy_birth_segment_host_wall_time_ns": 3_000,
            "policy_birth_segment_device_to_host_time_ns": 30,
            "policy_birth_segment_torch_random_generation_time_ns": 900,
            "policy_birth_segment_torch_compare_pack_time_ns": 1_500,
            "policy_birth_segment_allocation_stack_copy_time_ns": 300,
            "policy_birth_segment_host_to_device_time_ns": 75,
        }
    }
    path = tmp_path / "execution_receipt.json"
    path.write_text(json.dumps(execution))

    result = analyze_execution_receipt(path)

    assert result["timing_ns"]["unattributed_residual_ns"]["sum_ns"] == 270
    assert result["closure"]["exact_row_to_aggregate_closure"] is True
    assert result["closure"]["host_to_device_time_is_included_in_policy_allocation_stack_copy_time"] is True
    assert result["architecture_gate"]["selection_prohibited_by_this_receipt_alone"] is True
