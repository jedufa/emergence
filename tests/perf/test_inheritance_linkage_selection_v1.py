from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import textwrap

import pytest
import torch

from evolution.inheritance_registry_v1 import (
    INHERITANCE_REGISTRY_VERSION,
    LEGACY_INHERITANCE_PROFILE,
    LINKAGE_BLOCKS,
    LINKED_INHERITANCE_PROFILE,
    LINKED_PROFILE_DRAW_COUNT,
)


ROOT = Path(__file__).resolve().parents[2]
HEADER = (
    ROOT
    / "engine"
    / "perf"
    / "inheritance_linkage_selection_v1.cuh"
)


CPU_SOURCE = r"""
#include <cassert>
#include <cstdint>
#include <iostream>
#include "engine/perf/inheritance_linkage_selection_v1.cuh"

namespace inheritance = eiel::inheritance_linkage_v1;

int main() {
    static_assert(inheritance::kLinkageGroupCount == 7);
    static_assert(inheritance::kLinkedDonorDrawCount == 7);
    static_assert(inheritance::kLegacyAddedDonorDrawCount == 0);

    const std::uint8_t donors[7] = {0, 1, 1, 0, 1, 0, 1};
    for (std::uint32_t group = 0; group < 7; ++group) {
        const auto selected = inheritance::select_linkage_v1(
            inheritance::InheritanceModeV1::linked_parent_copy_v1,
            group,
            donors,
            7);
        assert(selected.status == inheritance::SelectionStatusV1::ok);
        assert(selected.kind == inheritance::SelectionKindV1::parent_donor);
        assert(selected.donor_ordinal == donors[group]);
        assert(selected.consumed_donor_draws == 7);
    }

    const auto legacy = inheritance::select_linkage_v1(
        inheritance::InheritanceModeV1::legacy_midpoint_v1,
        0,
        nullptr,
        0);
    assert(legacy.status == inheritance::SelectionStatusV1::ok);
    assert(legacy.kind == inheritance::SelectionKindV1::legacy_dispatch);
    assert(legacy.donor_ordinal == inheritance::kNoDonorOrdinal);
    assert(legacy.consumed_donor_draws == 0);

    const auto midpoint = inheritance::select_scalar_v1<double>(
        inheritance::InheritanceModeV1::legacy_midpoint_v1,
        0,
        nullptr,
        0,
        -2.0,
        6.0);
    assert(midpoint.status == inheritance::SelectionStatusV1::ok);
    assert(midpoint.value == 2.0);

    const auto copied = inheritance::select_scalar_v1<double>(
        inheritance::InheritanceModeV1::linked_parent_copy_v1,
        1,
        donors,
        7,
        -2.0,
        6.0);
    assert(copied.status == inheritance::SelectionStatusV1::ok);
    assert(copied.value == 6.0);
    assert(copied.donor_ordinal == 1);

    for (const auto group : {4u, 6u}) {
        const auto unsupported = inheritance::select_scalar_v1<double>(
            inheritance::InheritanceModeV1::legacy_midpoint_v1,
            group,
            nullptr,
            0,
            1.0,
            3.0);
        assert(
            unsupported.status ==
            inheritance::SelectionStatusV1::
                unsupported_legacy_field_operator);
    }

    assert(
        inheritance::validate_donor_schedule_v1(
            inheritance::InheritanceModeV1::linked_parent_copy_v1,
            donors,
            6) ==
        inheritance::SelectionStatusV1::invalid_donor_budget);
    assert(
        inheritance::validate_donor_schedule_v1(
            inheritance::InheritanceModeV1::legacy_midpoint_v1,
            donors,
            7) ==
        inheritance::SelectionStatusV1::invalid_donor_budget);
    std::uint8_t invalid_bits[7] = {0, 1, 0, 2, 0, 1, 0};
    assert(
        inheritance::validate_donor_schedule_v1(
            inheritance::InheritanceModeV1::linked_parent_copy_v1,
            invalid_bits,
            7) ==
        inheritance::SelectionStatusV1::invalid_donor_bit);
    assert(
        inheritance::validate_donor_schedule_v1(
            static_cast<inheritance::InheritanceModeV1>(99),
            nullptr,
            0) ==
        inheritance::SelectionStatusV1::invalid_mode);
    assert(
        inheritance::select_linkage_v1(
            inheritance::InheritanceModeV1::linked_parent_copy_v1,
            7,
            donors,
            7).status ==
        inheritance::SelectionStatusV1::invalid_linkage_group);

    std::cout
        << inheritance::kRegistryVersion << "|"
        << inheritance::kLegacyProfileId << "|"
        << inheritance::kLinkedProfileId << "\n";
    for (std::uint32_t group = 0; group < 7; ++group) {
        const auto* descriptor = inheritance::group_descriptor_v1(group);
        assert(descriptor != nullptr);
        std::cout
            << descriptor->block_id << "|"
            << static_cast<unsigned>(descriptor->legacy_dispatch) << "|";
        for (std::uint32_t field = 0; field < descriptor->field_count; ++field) {
            if (field != 0) {
                std::cout << ",";
            }
            std::cout << descriptor->fields[field];
        }
        std::cout << "\n";
    }
    assert(inheritance::group_descriptor_v1(7) == nullptr);
    return 0;
}
"""


CUDA_SOURCE = r"""
#include <cuda_runtime.h>
#include <cstdint>
#include <iostream>
#include "engine/perf/inheritance_linkage_selection_v1.cuh"

namespace inheritance = eiel::inheritance_linkage_v1;

__global__ void select_kernel(
    const std::uint8_t* donors,
    std::uint8_t* statuses,
    std::uint8_t* selected,
    double* values) {
    const auto group = static_cast<std::uint32_t>(threadIdx.x);
    if (group >= inheritance::kLinkageGroupCount) {
        return;
    }
    const auto result = inheritance::select_linkage_v1(
        inheritance::InheritanceModeV1::linked_parent_copy_v1,
        group,
        donors,
        inheritance::kLinkedDonorDrawCount);
    statuses[group] = static_cast<std::uint8_t>(result.status);
    selected[group] = result.donor_ordinal;
    const auto scalar = inheritance::select_scalar_v1<double>(
        inheritance::InheritanceModeV1::linked_parent_copy_v1,
        group,
        donors,
        inheritance::kLinkedDonorDrawCount,
        -4.0,
        9.0);
    values[group] = scalar.value;
    if (group == 0) {
        statuses[7] = static_cast<std::uint8_t>(
            inheritance::validate_donor_schedule_v1(
                inheritance::InheritanceModeV1::linked_parent_copy_v1,
                donors,
                inheritance::kLinkedDonorDrawCount - 1));
        statuses[8] = static_cast<std::uint8_t>(
            inheritance::select_linkage_v1(
                inheritance::InheritanceModeV1::legacy_midpoint_v1,
                0,
                nullptr,
                0).status);
    }
}

int main() {
    const std::uint8_t host_donors[7] = {1, 0, 1, 0, 0, 1, 1};
    std::uint8_t* device_donors = nullptr;
    std::uint8_t* device_statuses = nullptr;
    std::uint8_t* device_selected = nullptr;
    double* device_values = nullptr;
    if (cudaMalloc(&device_donors, 7) != cudaSuccess ||
        cudaMalloc(&device_statuses, 9) != cudaSuccess ||
        cudaMalloc(&device_selected, 7) != cudaSuccess ||
        cudaMalloc(&device_values, 7 * sizeof(double)) != cudaSuccess) {
        return 2;
    }
    cudaMemcpy(device_donors, host_donors, 7, cudaMemcpyHostToDevice);
    cudaMemset(device_statuses, 0xff, 9);
    select_kernel<<<1, 7>>>(
        device_donors,
        device_statuses,
        device_selected,
        device_values);
    if (cudaDeviceSynchronize() != cudaSuccess) {
        return 3;
    }
    std::uint8_t statuses[9] = {};
    std::uint8_t selected[7] = {};
    double values[7] = {};
    cudaMemcpy(statuses, device_statuses, 9, cudaMemcpyDeviceToHost);
    cudaMemcpy(selected, device_selected, 7, cudaMemcpyDeviceToHost);
    cudaMemcpy(values, device_values, 7 * sizeof(double), cudaMemcpyDeviceToHost);
    for (std::uint32_t group = 0; group < 7; ++group) {
        if (statuses[group] !=
                static_cast<std::uint8_t>(
                    inheritance::SelectionStatusV1::ok) ||
            selected[group] != host_donors[group] ||
            values[group] != (host_donors[group] == 0 ? -4.0 : 9.0)) {
            return 4;
        }
    }
    if (statuses[7] !=
            static_cast<std::uint8_t>(
                inheritance::SelectionStatusV1::invalid_donor_budget) ||
        statuses[8] !=
            static_cast<std::uint8_t>(
                inheritance::SelectionStatusV1::ok)) {
        return 5;
    }
    cudaFree(device_values);
    cudaFree(device_selected);
    cudaFree(device_statuses);
    cudaFree(device_donors);
    std::cout << "cuda_linkage_selection_v1_ok\n";
    return 0;
}
"""


def _compile_and_run(
    compiler: str,
    source: str,
    tmp_path: Path,
    *,
    suffix: str,
) -> subprocess.CompletedProcess[str]:
    source_path = tmp_path / f"probe{suffix}"
    executable = tmp_path / "probe"
    source_path.write_text(textwrap.dedent(source), encoding="utf-8")
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-I",
            str(ROOT),
            str(source_path),
            "-o",
            str(executable),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return subprocess.run(
        [str(executable)],
        check=True,
        capture_output=True,
        text=True,
    )


def test_cpu_primitive_and_compiled_registry_match_python_authority(
    tmp_path: Path,
) -> None:
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("g++ is unavailable")
    completed = _compile_and_run(
        compiler,
        CPU_SOURCE,
        tmp_path,
        suffix=".cpp",
    )
    lines = completed.stdout.splitlines()

    assert lines[0].split("|") == [
        INHERITANCE_REGISTRY_VERSION,
        LEGACY_INHERITANCE_PROFILE,
        LINKED_INHERITANCE_PROFILE,
    ]
    assert LINKED_PROFILE_DRAW_COUNT == 7
    observed = []
    for line in lines[1:]:
        block_id, legacy_dispatch, raw_fields = line.split("|")
        observed.append(
            (
                block_id,
                tuple(raw_fields.split(",")),
                int(legacy_dispatch),
            )
        )
    expected_dispatch = {
        "memory_controller_architecture_v1": 1,
        "family_head_strategy_v1": 2,
    }
    assert observed == [
        (
            block_id,
            fields,
            expected_dispatch.get(block_id, 0),
        )
        for block_id, fields in LINKAGE_BLOCKS
    ]


@pytest.mark.skipif(
    shutil.which("nvcc") is None or not torch.cuda.is_available(),
    reason="CUDA compiler/runtime is unavailable",
)
def test_cuda_primitive_compiles_and_runs_exact_donor_selection(
    tmp_path: Path,
) -> None:
    completed = _compile_and_run(
        shutil.which("nvcc") or "nvcc",
        CUDA_SOURCE,
        tmp_path,
        suffix=".cu",
    )
    assert completed.stdout.strip() == "cuda_linkage_selection_v1_ok"
