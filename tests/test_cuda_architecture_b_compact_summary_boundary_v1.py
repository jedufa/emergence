from __future__ import annotations

import importlib.util
import re
import shutil
import subprocess
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
HEADER = ROOT / "engine/perf/cuda_architecture_b_compact_summary_boundary_v1.cuh"
CUDA_SOURCE = ROOT / "engine/perf/cuda_architecture_b_compact_summary_boundary_v1.cu"
BINDING = (
    ROOT / "tests/perf/cuda_architecture_b_compact_summary_boundary_v1_binding.cpp"
)
COMPILE_TEST = (
    ROOT / "tests/perf/cuda_architecture_b_compact_summary_boundary_v1_compile.cu"
)
PROBE = (
    ROOT / "tools/scripts/cuda_architecture_b_compact_summary_boundary_v1_probe.py"
)


def _load_probe() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "cuda_architecture_b_compact_summary_boundary_v1_probe_test", PROBE
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_contract_is_standalone_allocation_free_and_host_digest_honest() -> None:
    header = HEADER.read_text(encoding="utf-8")
    source = CUDA_SOURCE.read_text(encoding="utf-8")
    combined = header + source

    assert "kBoundaryHostDigestFinalizationRequiredV1" in header
    assert "kRequiredHostCanonicalDigestMaskV1 == 0x7FU" in header
    assert "canonical-JSON or manifest-envelope preimages" in header
    assert "runtime_capability_promoted" in header
    assert "kUnresolvedLiveMetadataMaskV1" in header
    assert "RetainedHostCanonicalizationOutputsV1" in header
    for fragment in (
        "GenerationAuthorityHeaderV1* headers",
        "BehaviorAuthorityRowV1* behavior_rows",
        "LineageAuthorityRowV1* lineage_rows",
        "ReductionResultV1* lifecycle_identity",
        "SnapshotOutputsV1 checkpoint_outputs",
    ):
        assert fragment in header
    assert re.search(
        r"\b(?:malloc|calloc|cudaMalloc|cudaFree|new|torch::empty|at::empty)\b",
        combined,
    ) is None
    for forbidden in (
        "cudaDeviceSynchronize",
        "cudaStreamSynchronize",
        "cudaMemcpy",
        "cudaMemset",
    ):
        assert forbidden not in combined


def test_launch_order_and_receipts_cover_zero_to_n_checkpoint_batches() -> None:
    source = CUDA_SOURCE.read_text(encoding="utf-8")
    reset = "compact_summary_boundary_validity_reset_v1<<<"
    reduction = "reductions::launch_compact_summary_reductions_v1("
    checkpoint = "checkpoint_hash::launch_checkpoint_tensor_hash_snapshot_v1("
    gate = "compact_summary_boundary_prerequisite_gate_v1<<<"
    pipeline = "eiel_cuda_architecture_b_compact_summary_pipeline_v1_launch("

    definition_end = source.index("BoundaryLaunchReceiptV1 launch_compact_summary_boundary_v1(")
    body = source[definition_end:]
    assert body.index(reset) < body.index(reduction)
    assert body.index(reduction) < body.index(checkpoint)
    assert body.index(checkpoint) < body.index(gate)
    assert body.index(gate) < body.rindex(pipeline)
    assert "batch < request.checkpoint_batch_count" in body
    assert "accepted_pipeline_kernel_launches = 3" in body
    assert "intermediate_synchronizations" not in body
    assert source.count("<<<") == 2


def test_sm75_sass_only_compile_proof(tmp_path: Path) -> None:
    nvcc = shutil.which("nvcc")
    cuobjdump = shutil.which("cuobjdump")
    if nvcc is None or cuobjdump is None:
        pytest.skip("CUDA compiler tools are unavailable")
    output = tmp_path / "cuda_architecture_b_compact_summary_boundary_v1.o"
    completed = subprocess.run(
        [
            nvcc,
            "-std=c++17",
            "-O2",
            "--fmad=false",
            "--prec-div=true",
            "--prec-sqrt=true",
            "-lineinfo",
            "-Xptxas=-v",
            "-gencode=arch=compute_75,code=sm_75",
            "-I",
            str(ROOT / "engine/perf"),
            "-c",
            str(COMPILE_TEST),
            "-o",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert output.is_file()
    assert "ptxas info" in completed.stderr

    elf = subprocess.run(
        [cuobjdump, "--list-elf", str(output)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    ptx = subprocess.run(
        [cuobjdump, "--list-ptx", str(output)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert "sm_75" in elf
    assert "ptx" not in ptx.lower()


@pytest.fixture(scope="module")
def qualification(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    if shutil.which("nvcc") is None or not torch.cuda.is_available():
        pytest.skip("CUDA compiler and device are required")
    probe = _load_probe()
    module = probe.build_extension(
        tmp_path_factory.mktemp("compact_summary_boundary_v1")
    )
    return probe.qualify(module)


def test_g2_b3_isolation_and_exact_launch_receipts(
    qualification: dict[str, Any],
) -> None:
    assert qualification["ok"] is True
    g2_b3 = qualification["g2_b3"]
    assert g2_b3["exact_planes"] is True
    assert g2_b3["checkpoint_hashes_exact"] is True
    assert g2_b3["reduction_rows"][0][0]["generation_counts"] == (1, 0, 0, 0)
    assert g2_b3["reduction_rows"][1][0]["generation_counts"] == (0, 0, 0, 1)
    for generation_rows in g2_b3["reduction_rows"]:
        assert all(row["status"] == 0 and row["valid"] == 1 for row in generation_rows)
    for receipt in g2_b3["receipts"]:
        assert receipt["status"] == 0
        assert receipt["canonical_row_submission_capable"] is True
        assert receipt["runtime_capability_promoted"] is False
        assert receipt["validity_reset_kernel_launches"] == 1
        assert receipt["reduction_kernel_launches"] == 1
        assert receipt["checkpoint_hash_kernel_launches"] == 1
        assert receipt["prerequisite_gate_kernel_launches"] == 1
        assert receipt["accepted_pipeline_kernel_launches"] == 3
        assert receipt["total_kernel_launches"] == 7
        assert receipt["intermediate_synchronizations"] == 0
        assert receipt["device_allocations"] == 0
        assert receipt["host_allocations"] == 0
        assert receipt["retained_lifecycle_identity"] is True
        assert receipt["retained_checkpoint_outputs"] is True
        assert receipt["retained_completed_host_digest_values"] is True


def test_missing_host_digests_returns_explicit_non_capability(
    qualification: dict[str, Any],
) -> None:
    case = qualification["host_digest_finalization_required"]
    receipt = case["receipt"]
    assert receipt["status"] == 1
    assert receipt["missing_host_canonical_digest_mask"] == 0x7F
    assert receipt["canonical_row_submission_capable"] is False
    assert receipt["runtime_capability_promoted"] is False
    assert receipt["checkpoint_hash_kernel_launches"] == 0
    assert receipt["accepted_pipeline_kernel_launches"] == 0
    assert receipt["total_kernel_launches"] == 3
    assert case["producer_status"] == [102, 102, 102]
    assert case["output_validity"] == [[0, 0, 0], [0, 0, 0]]


def test_malformed_sources_fail_closed_without_cross_cell_damage(
    qualification: dict[str, Any],
) -> None:
    case = qualification["malformed_fail_closed"]
    assert case["reduction_receipt"]["status"] == 0
    assert case["reduction_producer_status"] == [0, 100, 0]
    assert case["reduction_output_validity"][0] == [1, 0, 1]
    assert case["hash_receipt"]["status"] == 0
    assert case["hash_producer_status"] == [101, 101, 101]
    assert case["hash_output_validity"][0] == [0, 0, 0]


def test_terminal_failure_is_sticky_and_tail_remains_invalid(
    qualification: dict[str, Any],
) -> None:
    case = qualification["sticky_tail"]
    assert case["producer_status"] == [0, 16, 0]
    assert case["generation_zero_validity"] == [1, 1, 1]
    assert case["generation_one_validity"] == [1, 0, 1]


def test_runtime_resource_receipt_is_sm75_and_sync_free(
    qualification: dict[str, Any],
) -> None:
    resources = qualification["resources"]
    for kernel in (
        "validity_reset",
        "reduction",
        "checkpoint_hash",
        "prerequisite_gate",
    ):
        assert resources[kernel]["binary_version"] == 75
        assert resources[kernel]["registers_per_thread"] > 0
        assert resources[kernel]["local_bytes_per_thread"] >= 0
        assert resources[kernel]["shared_bytes_per_block"] == 0
    assert resources["intermediate_synchronizations"] == 0
    assert resources["device_allocations"] == 0
    assert resources["host_allocations"] == 0
    assert resources["runtime_capability_promoted"] is False
