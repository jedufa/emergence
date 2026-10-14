from __future__ import annotations

import re
from pathlib import Path

import pytest
import torch

from engine.perf.cuda_architecture_b_checkpoint import (
    CudaCheckpointProjectionError,
    _ResidentCpuSnapshot,
)


ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "engine" / "perf"
HEADER = ENGINE / "cuda_architecture_b_flattened_controller_v1.hpp"
BINDING = ENGINE / "cuda_architecture_b_flattened_controller_v1_binding.cpp"
KERNELS = ENGINE / "cuda_architecture_b_flattened_controller_v1_kernels.cu"
TURNOVER = ENGINE / "cuda_architecture_b_flattened_turnover_v1_kernels.cuh"


def _code_only(source: str) -> str:
    def blank(match: re.Match[str]) -> str:
        return "\n" * match.group(0).count("\n")

    source = re.sub(r"/\*.*?\*/", blank, source, flags=re.DOTALL)
    source = re.sub(r"//[^\n]*", "", source)
    source = re.sub(r'"(?:\\.|[^"\\])*"', '""', source)
    return source


def _function_body(source: str, name: str) -> str:
    code = _code_only(source)
    match = re.search(rf"\b{re.escape(name)}\s*\(", code)
    assert match is not None, f"missing function {name}"
    opening = code.find("{", match.end())
    assert opening >= 0
    depth = 1
    for index in range(opening + 1, len(code)):
        if code[index] == "{":
            depth += 1
        elif code[index] == "}":
            depth -= 1
            if depth == 0:
                return code[opening + 1 : index]
    raise AssertionError(f"unbalanced function {name}")


def test_resident_window_bases_extend_abis_without_reindexing() -> None:
    header = HEADER.read_text(encoding="utf-8")
    binding = BINDING.read_text(encoding="utf-8")

    assert re.search(
        r"int64_t\* active_count;\s*int64_t\* sequence_base;",
        header,
    )
    assert "event_state_tensors.size() == 31" in binding
    assert "event_state_tensors.at(30).scalar_type() == torch::kInt64" in binding
    assert "event_state_tensors.at(30).data_ptr<int64_t>()" in binding

    assert "int64_t* policy_birth_tape_cursor_base;" in header
    assert "int64_t* python_random_raw_draw_base;" in header
    assert "const bool genome_upgrade_v2_state_enabled =" in binding
    assert "lifecycle_state_enabled && ordinary_birth_state_tensors.size() == 139" in binding
    assert re.search(r"genome_upgrade_v2_state_enabled\s*\?\s*139", binding)
    assert "for (const int64_t index : {111, 112})" in binding
    assert "state.at(111).data_ptr<int64_t>()" in binding
    assert "state.at(112).data_ptr<int64_t>()" in binding


def test_absolute_rng_counters_use_bounded_physical_scratch_cursors() -> None:
    kernels = KERNELS.read_text(encoding="utf-8")
    conversion = _function_body(kernels, "physical_cursor_from_absolute_v1")
    restoration = _function_body(kernels, "absolute_cursor_from_physical_v1")

    assert "absolute_cursor < sequence_base" in conversion
    assert "candidate > physical_capacity" in conversion
    assert "physical_cursor > physical_capacity" in restoration
    assert "sequence_base > INT64_MAX - physical_cursor" in restoration

    prepare = _function_body(
        kernels, "flattened_controller_birth_prepare_kernel_v1"
    )
    assert "physical_python_raw_draws" in prepare
    assert "physical_policy_cursor" in prepare
    assert re.search(
        r"kPythonRawDraws\s*\)\s*\[0\]\s*=\s*physical_python_raw_draws",
        prepare,
    )
    assert "mini_policy_cursor[0] = physical_policy_cursor" in prepare
    zero_birth_return = prepare.find("planned_count == 0")
    assert 0 <= prepare.find("physical_python_raw_draws", 0, zero_birth_return)
    assert 0 <= prepare.find("kMiniPolicyCursor", 0, zero_birth_return)

    commit = _function_body(kernels, "commit_flattened_birth_transaction_v1")
    conversion_position = commit.find("absolute_cursor_from_physical_v1")
    mutation_position = commit.find("commit_flattened_birth_plan_rng_v1")
    assert 0 <= conversion_position < mutation_position
    assert "return 26" in commit
    assert "absolute_python_raw_draws" in commit
    assert "absolute_policy_cursor" in commit

    ordinary = _function_body(kernels, "execute_ordinary_birth_phase_v1")
    assert "physical_python_raw_draws" in ordinary
    assert "physical_policy_cursor" in ordinary
    assert "birth_state.python_random_raw_draws + cell" not in ordinary
    assert "birth_state.policy_birth_tape_cursor + cell" not in ordinary
    assert ordinary.count("absolute_cursor_from_physical_v1(") == 2

    payload = _function_body(kernels, "commit_flattened_birth_payload_v1")
    assert re.search(
        r"python_random_raw_draws\[cell\]\s*=\s*absolute_python_raw_draws",
        payload,
    )
    assert re.search(
        r"policy_birth_tape_cursor\[cell\]\s*=\s*absolute_policy_cursor",
        payload,
    )

    turnover = _function_body(
        TURNOVER.read_text(encoding="utf-8"),
        "flattened_turnover_commit_kernel_v1",
    )
    assert "python_random_raw_draw_base[cell]" in turnover
    assert "policy_birth_tape_cursor_base[cell]" in turnover
    assert "turnover_python_raw_draws[0] -= python_raw_draw_base" in turnover
    assert "turnover_python_raw_draws[0] += python_raw_draw_base" in turnover
    assert "turnover_policy_cursor[0] -= policy_cursor_base" in turnover
    assert "turnover_policy_cursor[0] += policy_cursor_base" in turnover


def test_every_flattened_event_append_republishes_absolute_sequence() -> None:
    kernels = KERNELS.read_text(encoding="utf-8")
    death = _function_body(kernels, "append_death_event_class_v4")
    ordinary = _function_body(kernels, "execute_ordinary_birth_phase_v1")
    stackless = _function_body(kernels, "commit_flattened_birth_transaction_v1")

    assert kernels.count("append_death_lifecycle_events(") == 1
    assert kernels.count("append_preflighted_ordinary_birth_events_serial(") == 2
    for body in (death, ordinary):
        assert "event_sequence_contract_valid_v1" in body
        assert "publish_absolute_event_sequences_v1" in body
    # The staged transaction already writes the absolute sequence while it
    # materializes the event rows; it does not need a second helper pass.
    assert "event_sequence_contract_valid_v1" in stackless
    assert "event_sequence_base + row" in stackless
    assert "typed_sequence[event_cell_offset + row]" in stackless

    turnover = TURNOVER.read_text(encoding="utf-8")
    assert "append_turnover_lifecycle_events_serial" not in turnover
    assert "typed_sequence" not in turnover
    assert "run_flattened_turnover_v1(" in kernels


def test_public_event_cursors_are_absolute_and_compact_reduction_accepts_frontier() -> None:
    binding = BINDING.read_text(encoding="utf-8")
    kernels = KERNELS.read_text(encoding="utf-8")

    assert "absolute_event_cursor_v1" in kernels
    # Recycled windows intentionally carry an absolute frontier.  The
    # reduction converts that frontier back to the bounded physical ring
    # instead of rejecting every resumed window as non-zero based.
    assert "event_state.sequence_base[cell] != 0" not in kernels
    assert "physical_cursor_from_absolute_v1(" in _function_body(
        kernels, "finalize_generation_summary_scalar_v1"
    )
    assert (
        "metadata_inputs.event_cursor = event_state.active_count" not in binding
    )
    assert (
        "generation_compact_summary_context.event_cursor_end"
        in binding
    )


def test_terminal_event_archive_is_the_only_mixed_device_snapshot_exception() -> None:
    snapshot = _ResidentCpuSnapshot()
    snapshot.source_device = torch.device("cuda:0")
    host_event_column = torch.arange(4, dtype=torch.int64)

    with pytest.raises(CudaCheckpointProjectionError, match="share one device"):
        snapshot.copy(host_event_column, name="ordinary_resident_field")

    copied = snapshot.copy(
        host_event_column,
        name="resident_birth_state.typed_lifecycle_event_sequence",
        allow_pre_materialized_cpu=True,
    )
    assert torch.equal(copied, host_event_column)
    assert copied.data_ptr() != host_event_column.data_ptr()
    assert snapshot.source_device == torch.device("cuda:0")
    assert snapshot.device_to_host_operations == 0
    assert snapshot.device_to_host_bytes == 0
