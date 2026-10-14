"""Bounded resource and residency preparation over the executable C4 R9 route."""

from __future__ import annotations

import ast
import os
from pathlib import Path

import pytest
import torch

from engine.tensor_world.c4_resident_birth_r2 import (
    _half_tensors,
    _inherit_tensors,
    _minus_one_tensors,
    _one_tensors,
    _zero_tensors,
)
from engine.tensor_world.c4_resident_multitick_r10 import (
    make_c4_resident_multitick_r10_fixture_v1,
)
from engine.tensor_world.c4_resident_population_r9 import (
    make_c4_resident_population_r9_staging_v1,
)
from engine.tensor_world.c4_resident_resource_r10_r11 import (
    C4_RESIDENT_RESOURCE_R10_R11_DECISION_V1,
    c4_resident_r9_staging_budget_rows_v1,
    c4_resident_resource_r10_r11_contract_v1,
    measure_c4_resident_r9_staging_v1,
)


ROOT = Path(__file__).resolve().parents[1]
R9_ADAPTER = ROOT / "engine/tensor_world/c4_resident_population_r9.py"
R9_KERNEL = ROOT / "engine/tensor_world/native/tensor_world_c4_resident_population_r9.cu"
R2_KERNEL = ROOT / "engine/tensor_world/native/tensor_world_c4_resident_birth_r2.cu"


def _function_node_and_source(path: Path, symbol: str) -> tuple[ast.FunctionDef, str]:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == symbol:
            assert node.end_lineno is not None
            lines = source.splitlines()
            return node, "\n".join(lines[node.lineno - 1 : node.end_lineno])
    raise AssertionError(f"missing function {symbol}")


def _native_function_source(path: Path, symbol: str) -> str:
    source = path.read_text(encoding="utf-8")
    start = source.index(f"void {symbol}(")
    brace = source.index("{", start)
    depth = 0
    for index in range(brace, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[start : index + 1]
    raise AssertionError(f"unbounded native function {symbol}")


def test_r10_r11_preparation_freezes_exact_r9_staging_budget_without_gate() -> None:
    rows = c4_resident_r9_staging_budget_rows_v1()
    contract = c4_resident_resource_r10_r11_contract_v1()

    assert len(rows) == 90
    assert len({row.path for row in rows}) == 90
    assert sum(row.bytes for row in rows) == 794_736
    assert sum(row.bytes for row in rows if row.path.startswith("turnover.birth.")) == 113_088
    assert contract["route"] == (
        "R9_B48_C32_R10_RESOURCE_RESIDENCY_PREPARATION_ONLY"
    )
    assert contract["decision"] == C4_RESIDENT_RESOURCE_R10_R11_DECISION_V1
    assert contract["staging_tensor_count"] == 90
    assert contract["staging_payload_bytes"] == 794_736
    assert contract["unused_nested_turnover_birth_payload_bytes"] == 113_088
    assert contract["measured_steady_state_counters"] is None
    assert contract["resident_total_vram_claim"] is False
    assert contract["timing_claim"] is False
    assert contract["gate_claim"] is False


def test_r10_r11_static_source_sentinels_bound_launches_and_host_authority() -> None:
    node, execute_source = _function_node_and_source(
        R9_ADAPTER, "execute_c4_resident_population_r9_cuda_v1"
    )
    native_source = _native_function_source(
        R9_KERNEL, "tensor_world_c4_resident_population_r9_cuda"
    )
    dispatch_inherit = _native_function_source(R2_KERNEL, "dispatch_inherit")
    dispatch_fill = _native_function_source(R2_KERNEL, "dispatch_fill")
    launch_inherit = _native_function_source(R2_KERNEL, "launch_inherit")
    launch_fill = _native_function_source(R2_KERNEL, "launch_fill")

    forbidden_boundary_tokens = (
        ".cpu(",
        ".cuda(",
        ".to(",
        ".item(",
        ".tolist(",
        "cudaMemcpy",
        "torch.cuda.synchronize",
        "cudaDeviceSynchronize",
        "cudaStreamSynchronize",
        "cudaLaunchHostFunc",
        "cudaStreamAddCallback",
    )
    forbidden_hot_allocator_tokens = (
        "torch.empty(",
        "torch.zeros(",
        "torch.full(",
        "cudaMalloc",
        "cudaFree",
    )
    audited_hot_source = execute_source + native_source
    assert all(token not in audited_hot_source for token in forbidden_boundary_tokens)
    assert all(token not in audited_hot_source for token in forbidden_hot_allocator_tokens)

    loops = [child for child in ast.walk(node) if isinstance(child, ast.For)]
    assert len(loops) == 1
    assert ast.unparse(loops[0].target) == "tensor"
    assert ast.unparse(loops[0].iter) == "scalar_tensors"
    assert "load_tensor_world_c4_resident_population_r9_extension(" in execute_source
    assert "module.population_transaction(" in execute_source

    fixture = make_c4_resident_multitick_r10_fixture_v1()
    bundle = fixture.bundle
    group_counts = (
        len(_inherit_tensors(bundle)),
        len(_zero_tensors(bundle)),
        len(_minus_one_tensors(bundle)),
        len(_half_tensors(bundle)),
        len(_one_tensors(bundle)),
    )
    assert group_counts == (29, 55, 11, 1, 1)
    assert native_source.count("<<<") == 8
    assert launch_inherit.count("<<<") == 1
    assert launch_fill.count("<<<") == 1
    assert "<<<" not in dispatch_inherit
    assert "<<<" not in dispatch_fill
    assert 8 + sum(group_counts) == 105

    contract = c4_resident_resource_r10_r11_contract_v1()
    assert contract["native_kernel_launches_per_transaction"] == 105
    assert contract["component_execution_classification"] == (
        "device_local_multi_kernel_host_invocation_graph_binding_unproven"
    )
    assert contract["cuda_graph_capture_claim"] is False
    assert contract["source_observed_execute_h2d_or_d2h_tokens"] == 0
    assert contract["source_observed_execute_sync_or_materialization_tokens"] == 0
    assert contract["source_observed_execute_device_allocator_tokens"] == 0
    assert contract["source_observed_execute_per_world_python_loops"] == 0
    assert contract["source_observed_execute_per_agent_python_loops"] == 0
    assert contract["source_observed_host_validation_loops"] == 1
    assert contract["source_observed_host_extension_load_calls"] == 1
    assert contract["source_observed_host_native_invocations"] == 1
    assert contract["preallocated_staging_lifetime_bound_to_production_owner"] is False


def test_r10_r11_reports_current_exact_rung_blockers() -> None:
    contract = c4_resident_resource_r10_r11_contract_v1()

    assert contract["batch"] == 48
    assert contract["capacity"] == 32
    assert contract["r10_blocker"] is None
    assert contract["r10_execution_scope"] == (
        "five_host_launched_R9_transactions_not_graph_bound"
    )
    assert contract["r11_blocker"] == (
        "no_composed_C1_C2_C3_C4_resident_tick_seam"
    )
    assert "TORCH_CHECK(batch == 48" in R9_KERNEL.read_text(encoding="utf-8")
    production_callsites = []
    for path in (ROOT / "engine").rglob("*.py"):
        if path.name.startswith("c4_resident_") or path.name.startswith("c4_residency_"):
            continue
        if "execute_c4_resident_population_r9_cuda_v1" in path.read_text(
            encoding="utf-8"
        ):
            production_callsites.append(path.relative_to(ROOT).as_posix())
    assert production_callsites == []
    for name in (
        "h2d_bytes",
        "d2h_bytes",
        "host_callbacks",
        "synchronizations",
        "materializations",
        "device_allocator_calls",
        "per_world_python_loops",
        "per_agent_python_loops",
    ):
        assert contract[f"required_steady_state_{name}"] == 0


@pytest.mark.skipif(
    os.environ.get("EIEL_C4_R10_R11_RESOURCE_PROBE") != "1",
    reason="requires explicit bounded CUDA resource-probe admission",
)
def test_r10_r11_cuda_staging_payload_matches_allocator_measurement() -> None:
    assert torch.cuda.is_available()
    device = torch.device("cuda")
    torch.cuda.empty_cache()
    baseline_allocated = torch.cuda.memory_allocated(device)
    baseline_reserved = torch.cuda.memory_reserved(device)
    torch.cuda.reset_peak_memory_stats(device)

    staging = make_c4_resident_population_r9_staging_v1(
        batch=48, capacity=32, grid_size=8, streams=2, device=device
    )
    measurement = measure_c4_resident_r9_staging_v1(staging)
    allocated = torch.cuda.memory_allocated(device)
    reserved = torch.cuda.memory_reserved(device)
    peak_allocated = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)

    assert measurement == {
        "tensor_count": 90,
        "payload_bytes": 794_736,
        "unique_storage_bytes": 794_736,
        "device": str(staging.device),
        "synchronized": False,
        "allocator_counters_included": False,
    }
    assert allocated - baseline_allocated >= measurement["unique_storage_bytes"]
    assert reserved >= allocated
    assert peak_allocated >= allocated
    assert peak_reserved >= reserved
    assert baseline_reserved >= baseline_allocated
