"""Exact-parity and bounded benchmark seam for parallel C2 policy logits."""

from __future__ import annotations

import ast
import inspect
import os
from pathlib import Path
import re
from statistics import median

import pytest
import torch

from engine.tensor_world.c2_backend import (
    _config_tensors_v1,
    _native_state_v1,
    clone_c2_bundle_to_device_v1,
)
from engine.tensor_world.c2_fixture import make_c2_fixture_v1
from engine.tensor_world.c2_native_loader import load_tensor_world_c2_extension
from engine.tensor_world.c2_policy_parallel_r12 import (
    C2_POLICY_PARALLEL_R12_DECISION_V1,
    C2_POLICY_PARALLEL_R12_SCHEMA_VERSION_V1,
    C2_POLICY_PARALLEL_R12_THREADS_PER_ROW_V1,
    c2_policy_parallel_r12_source_receipt_v1,
    load_tensor_world_c2_policy_parallel_r12_full_extension,
    render_tensor_world_c2_policy_parallel_r12_full_source_v1,
)


ROOT = Path(__file__).resolve().parents[1]
SCALAR_SOURCE = ROOT / "engine/tensor_world/native/tensor_world_c2_kernels.cu"
PARALLEL_SOURCE = (
    ROOT / "engine/tensor_world/native/tensor_world_c2_policy_parallel_r12.cu"
)
PROBE_ENV = "EIEL_C2_POLICY_PARALLEL_R12_PROBE"
BENCHMARK_ENV = "EIEL_C2_POLICY_PARALLEL_R12_BENCHMARK"


def _enum_names(source: str, enum_name: str) -> tuple[str, ...]:
    match = re.search(
        rf"enum {enum_name} : int64_t \{{(?P<body>.*?)\n\}};",
        source,
        flags=re.DOTALL,
    )
    assert match is not None
    body = re.sub(r"//.*", "", match.group("body"))
    return tuple(
        item.split("=", 1)[0].strip()
        for item in body.split(",")
        if item.strip()
    )


def _make_policy_inputs(
    *, batch: int, capacity: int, active: int
) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor]:
    source, config = make_c2_fixture_v1(
        batch=batch,
        capacity=capacity,
        active=active,
        width=32,
    )
    observations = source.transactions.observations
    pattern = (
        torch.arange(observations.shape[-1], dtype=torch.float32)
        .remainder(17)
        .sub_(8)
        .mul_(1.0e-3)
    )
    observations.copy_(pattern.view(1, 1, -1))
    state = clone_c2_bundle_to_device_v1(source, torch.device("cuda:0"))
    tensors = _native_state_v1(state)
    config_ints, config_floats = _config_tensors_v1(config, state)
    return tensors, config_ints, config_floats


def _time_policy_ms(
    operation,
    tensors: list[torch.Tensor],
    config_ints: torch.Tensor,
    config_floats: torch.Tensor,
    *,
    warmup: int = 5,
    repeats: int = 21,
) -> float:
    for _ in range(warmup):
        operation(tensors, config_ints, config_floats)
    torch.cuda.synchronize()
    samples: list[float] = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        operation(tensors, config_ints, config_floats)
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)))
    return median(samples)


def test_parallel_policy_source_freezes_c2_abi_and_serial_neuron_order() -> None:
    scalar = SCALAR_SOURCE.read_text(encoding="utf-8")
    parallel = PARALLEL_SOURCE.read_text(encoding="utf-8")
    assert _enum_names(parallel, "T") == _enum_names(scalar, "T")
    assert _enum_names(parallel, "CF") == _enum_names(scalar, "CF")
    assert "constexpr int kThreadsPerRow = 64;" in parallel
    assert "rows, kThreadsPerRow, 0" in parallel
    assert "__shared__ float h1[kHiddenWidth];" in parallel
    assert "__shared__ float h2[kHiddenWidth];" in parallel
    assert parallel.count("__syncthreads();") == 2
    assert "feature = 0; feature < kObservationWidth; ++feature" in parallel
    assert "input = 0; input < kHiddenWidth; ++input" in parallel
    assert "hidden = 0; hidden < kHiddenWidth; ++hidden" in parallel
    assert not any(
        token in parallel
        for token in ("atomicAdd", "cub::", "cooperative_groups", "--use_fast_math")
    )
    generated = render_tensor_world_c2_policy_parallel_r12_full_source_v1()
    assert "policy_parallel_kernel<<<" in generated
    assert "policy_kernel<<<rows,1" not in generated
    assert generated.count('module.def("policy_logits", &c2_policy);') == 1
    assert generated.count("PYBIND11_MODULE(TORCH_EXTENSION_NAME, module)") == 1


def test_parallel_policy_receipt_and_loader_are_isolated_from_r11() -> None:
    receipt = c2_policy_parallel_r12_source_receipt_v1()
    assert receipt["schema_version"] == C2_POLICY_PARALLEL_R12_SCHEMA_VERSION_V1
    assert receipt["decision"] == C2_POLICY_PARALLEL_R12_DECISION_V1
    assert receipt["threads_per_row"] == C2_POLICY_PARALLEL_R12_THREADS_PER_ROW_V1
    assert receipt["arithmetic_order"]["cross_thread_reduction"] is False
    assert receipt["gate_claim"] is False
    assert all(len(row["sha256"]) == 64 for row in receipt["sources"].values())

    loader = inspect.getsource(
        load_tensor_world_c2_policy_parallel_r12_full_extension
    )
    assert "load_tensor_world_c2_extension(configured)" in loader
    assert "C2_POLICY_PARALLEL_R12_PREBUILT_ENV_V1" in loader
    assert "c4_resident_composed_r11" not in loader

    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    gpu_tests = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name.startswith("test_parallel_policy_cuda")
    ]
    assert {node.name for node in gpu_tests} == {
        "test_parallel_policy_cuda_is_bit_exact",
        "test_parallel_policy_cuda_b48_c512_benchmark",
    }


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.skipif(
    os.environ.get(PROBE_ENV) != "1",
    reason="parallel C2 policy probe is explicitly opt-in",
)
def test_parallel_policy_cuda_is_bit_exact() -> None:
    tensors, config_ints, config_floats = _make_policy_inputs(
        batch=4,
        capacity=32,
        active=17,
    )
    scalar = load_tensor_world_c2_extension()
    parallel = load_tensor_world_c2_policy_parallel_r12_full_extension()
    scalar.policy_logits(tensors, config_ints, config_floats)
    torch.cuda.synchronize()
    expected = tensors[75].clone()
    tensors[75].fill_(float("nan"))
    parallel.policy_logits(tensors, config_ints, config_floats)
    torch.cuda.synchronize()
    actual = tensors[75]
    assert torch.equal(actual, expected)
    assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))
    assert torch.equal(actual[~tensors[0]], torch.zeros_like(actual[~tensors[0]]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.skipif(
    os.environ.get(BENCHMARK_ENV) != "1",
    reason="parallel C2 policy benchmark is explicitly opt-in",
)
def test_parallel_policy_cuda_b48_c512_benchmark() -> None:
    tensors, config_ints, config_floats = _make_policy_inputs(
        batch=48,
        capacity=512,
        active=256,
    )
    scalar = load_tensor_world_c2_extension()
    parallel = load_tensor_world_c2_policy_parallel_r12_full_extension()
    scalar_ms = _time_policy_ms(
        scalar.policy_logits, tensors, config_ints, config_floats
    )
    parallel_ms = _time_policy_ms(
        parallel.policy_logits, tensors, config_ints, config_floats
    )
    print(
        {
            "geometry": "B48_C512_P256",
            "scalar_policy_median_ms": scalar_ms,
            "parallel_policy_median_ms": parallel_ms,
            "speedup": scalar_ms / parallel_ms,
        }
    )
    assert parallel_ms < scalar_ms
