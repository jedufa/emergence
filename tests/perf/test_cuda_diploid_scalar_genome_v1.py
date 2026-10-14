from __future__ import annotations

from pathlib import Path
import random
import shutil
import subprocess
import textwrap

import pytest

from evolution.diploid_inheritance_v1 import (
    DIPLOID_LINKAGE_BLOCKS,
    DIPLOID_SCALAR_LOCI,
    LINKAGE_MAP_SHA256,
    LOCUS_BOUNDS,
    LOCUS_REGISTRY_SHA256,
    MEIOSIS_DRAW_COUNT,
    MUTATION_SEMANTIC_DRAW_COUNT,
    MUTATION_SIGMA,
)


ROOT = Path(__file__).resolve().parents[2]
HEADER = ROOT / "engine" / "perf" / "cuda_diploid_scalar_genome_v1.cuh"


CPU_SOURCE = r"""
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include "engine/perf/cuda_diploid_scalar_genome_v1.cuh"

namespace diploid = eiel::cuda_diploid_scalar_genome_v1;

struct TapeRandom {
    double draws[128];
    int64_t cursor = 0;
    double random_double() { return draws[cursor++]; }
    bool consume_gauss(double sigma, double* value) {
        *value = draws[cursor++] * sigma;
        return true;
    }
};

int main() {
    static_assert(diploid::kLocusCount == 17);
    static_assert(diploid::kAllelePlaneCount == 34);
    static_assert(diploid::kLinkageBlockCount == 5);
    static_assert(diploid::kMeiosisDrawCount == 10);
    static_assert(diploid::kMutationDrawCount == 51);

    constexpr int64_t slots = 3;
    float allele_0[diploid::kLocusCount * slots] = {};
    float allele_1[diploid::kLocusCount * slots] = {};
    float phenotype_storage[diploid::kLocusCount][slots] = {};
    diploid::PhenotypeArraysV1 phenotype = {};
    for (int32_t locus = 0; locus < diploid::kLocusCount; ++locus) {
        phenotype.locus[locus] = phenotype_storage[locus];
        allele_0[locus * slots] = static_cast<float>(
            diploid::kLowerBounds[locus]
            + 0.2 * (
                diploid::kUpperBounds[locus]
                - diploid::kLowerBounds[locus]
            )
        );
        allele_1[locus * slots] = static_cast<float>(
            diploid::kLowerBounds[locus]
            + 0.4 * (
                diploid::kUpperBounds[locus]
                - diploid::kLowerBounds[locus]
            )
        );
        allele_0[locus * slots + 1] = static_cast<float>(
            diploid::kLowerBounds[locus]
            + 0.6 * (
                diploid::kUpperBounds[locus]
                - diploid::kLowerBounds[locus]
            )
        );
        allele_1[locus * slots + 1] = static_cast<float>(
            diploid::kLowerBounds[locus]
            + 0.8 * (
                diploid::kUpperBounds[locus]
                - diploid::kLowerBounds[locus]
            )
        );
    }
    TapeRandom rng = {};
    for (int64_t index = 0; index < 128; ++index) {
        rng.draws[index] = (index % 7) * 0.13;
    }
    diploid::StagedChildV1 staged = {};
    auto status = diploid::stage_child_v1(
        {allele_0, allele_1, slots, slots},
        0,
        1,
        rng,
        false,
        true,
        1.0,
        &staged
    );
    assert(status == diploid::StatusV1::ok);
    assert(staged.semantic_draws == 10);
    assert(rng.cursor == 10);
    status = diploid::commit_staged_child_v1(
        {allele_0, allele_1, slots, slots},
        phenotype,
        2,
        staged
    );
    assert(status == diploid::StatusV1::ok);
    for (int32_t locus = 0; locus < diploid::kLocusCount; ++locus) {
        assert(
            phenotype_storage[locus][2]
            == diploid::canon_float(
                0.5 * allele_0[locus * slots + 2]
                + 0.5 * allele_1[locus * slots + 2]
            )
        );
    }

    TapeRandom mutation_rng = {};
    for (int64_t index = 0; index < 128; ++index) {
        mutation_rng.draws[index] = (index % 5) * 0.11;
    }
    status = diploid::stage_child_v1(
        {allele_0, allele_1, slots, slots},
        0,
        1,
        mutation_rng,
        true,
        false,
        1.0,
        &staged
    );
    assert(status == diploid::StatusV1::ok);
    assert(staged.semantic_draws == 61);
    assert(mutation_rng.cursor == 61);
    assert(staged.mutation_changed[diploid::kResourceHandlingAxis] == 0);

    for (int32_t locus = 0; locus < diploid::kLocusCount; ++locus) {
        phenotype_storage[locus][0] = staged.phenotype[locus];
    }
    diploid::StagedChildV1 migrated = {};
    status = diploid::stage_homozygous_migration_v1(
        phenotype, 0, &migrated
    );
    assert(status == diploid::StatusV1::ok);
    assert(migrated.semantic_draws == 0);
    for (int32_t locus = 0; locus < diploid::kLocusCount; ++locus) {
        assert(migrated.allele_0[locus] == migrated.allele_1[locus]);
        assert(migrated.allele_0[locus] == migrated.phenotype[locus]);
    }

    std::cout
        << diploid::kProfileId << "\n"
        << diploid::kSchemaVersion << "\n"
        << diploid::kLinkageMapSha256 << "\n"
        << diploid::kAdmittedLocusSha256 << "\n";
    for (int32_t locus = 0; locus < diploid::kLocusCount; ++locus) {
        std::cout
            << diploid::kLocusNames[locus] << "|"
            << diploid::kLinkageBlockByLocus[locus] << "|"
            << diploid::kLowerBounds[locus] << "|"
            << diploid::kUpperBounds[locus] << "|"
            << diploid::kMutationSigma[locus] << "\n";
    }
    return 0;
}
"""


CUDA_SOURCE = r"""
#include <cuda_runtime.h>
#include <cstdint>
#include <iostream>
#include "engine/perf/cuda_diploid_scalar_genome_v1.cuh"

namespace diploid = eiel::cuda_diploid_scalar_genome_v1;

struct DeviceTapeRandom {
    const double* draws;
    int64_t cursor;
    __device__ double random_double() { return draws[cursor++]; }
    __device__ bool consume_gauss(double sigma, double* value) {
        *value = draws[cursor++] * sigma;
        return true;
    }
};

__global__ void probe(
    float* allele_0,
    float* allele_1,
    float* phenotype,
    const double* draws,
    int32_t* result
) {
    diploid::PhenotypeArraysV1 phenotype_arrays = {};
    for (int32_t locus = 0; locus < diploid::kLocusCount; ++locus) {
        phenotype_arrays.locus[locus] = phenotype + locus * 3;
    }
    DeviceTapeRandom rng{draws, 0};
    diploid::StagedChildV1 staged = {};
    auto status = diploid::stage_child_v1(
        {allele_0, allele_1, 3, 3},
        0,
        1,
        rng,
        true,
        true,
        1.0,
        &staged
    );
    if (status == diploid::StatusV1::ok) {
        status = diploid::commit_staged_child_v1(
            {allele_0, allele_1, 3, 3},
            phenotype_arrays,
            2,
            staged
        );
    }
    result[0] = static_cast<int32_t>(status);
    result[1] = static_cast<int32_t>(staged.semantic_draws);
    result[2] = static_cast<int32_t>(rng.cursor);
}

int main() {
    constexpr int64_t values = diploid::kLocusCount * 3;
    float host_0[values] = {};
    float host_1[values] = {};
    double host_draws[128] = {};
    for (int32_t locus = 0; locus < diploid::kLocusCount; ++locus) {
        host_0[locus * 3] = static_cast<float>(
            diploid::kLowerBounds[locus]
            + 0.2 * (
                diploid::kUpperBounds[locus]
                - diploid::kLowerBounds[locus]
            )
        );
        host_1[locus * 3] = static_cast<float>(
            diploid::kLowerBounds[locus]
            + 0.4 * (
                diploid::kUpperBounds[locus]
                - diploid::kLowerBounds[locus]
            )
        );
        host_0[locus * 3 + 1] = static_cast<float>(
            diploid::kLowerBounds[locus]
            + 0.6 * (
                diploid::kUpperBounds[locus]
                - diploid::kLowerBounds[locus]
            )
        );
        host_1[locus * 3 + 1] = static_cast<float>(
            diploid::kLowerBounds[locus]
            + 0.8 * (
                diploid::kUpperBounds[locus]
                - diploid::kLowerBounds[locus]
            )
        );
    }
    for (int64_t index = 0; index < 128; ++index) {
        host_draws[index] = (index % 7) * 0.13;
    }
    float *device_0 = nullptr, *device_1 = nullptr, *device_p = nullptr;
    double* device_draws = nullptr;
    int32_t* device_result = nullptr;
    if (
        cudaMalloc(&device_0, values * sizeof(float)) != cudaSuccess
        || cudaMalloc(&device_1, values * sizeof(float)) != cudaSuccess
        || cudaMalloc(&device_p, values * sizeof(float)) != cudaSuccess
        || cudaMalloc(&device_draws, 128 * sizeof(double)) != cudaSuccess
        || cudaMalloc(&device_result, 3 * sizeof(int32_t)) != cudaSuccess
    ) {
        return 2;
    }
    cudaMemcpy(
        device_0, host_0, values * sizeof(float), cudaMemcpyHostToDevice
    );
    cudaMemcpy(
        device_1, host_1, values * sizeof(float), cudaMemcpyHostToDevice
    );
    cudaMemcpy(
        device_draws,
        host_draws,
        128 * sizeof(double),
        cudaMemcpyHostToDevice
    );
    cudaMemset(device_p, 0, values * sizeof(float));
    probe<<<1, 1>>>(
        device_0, device_1, device_p, device_draws, device_result
    );
    if (cudaDeviceSynchronize() != cudaSuccess) {
        return 3;
    }
    int32_t result[3] = {};
    cudaMemcpy(
        result, device_result, 3 * sizeof(int32_t), cudaMemcpyDeviceToHost
    );
    if (result[0] != 0 || result[1] != 61 || result[2] != 61) {
        return 4;
    }
    std::cout << "cuda_diploid_scalar_genome_v1_ok\n";
    cudaFree(device_result);
    cudaFree(device_draws);
    cudaFree(device_p);
    cudaFree(device_1);
    cudaFree(device_0);
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


def test_cpu_contract_matches_python_authority(tmp_path: Path) -> None:
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("g++ is unavailable")
    completed = _compile_and_run(
        compiler, CPU_SOURCE, tmp_path, suffix=".cpp"
    )
    lines = completed.stdout.splitlines()
    assert lines[:4] == [
        "diploid_linkage_v1",
        "diploid_genome_bundle_v1",
        LINKAGE_MAP_SHA256,
        LOCUS_REGISTRY_SHA256,
    ]
    observed = {}
    for row in lines[4:]:
        locus, block, lower, upper, sigma = row.split("|")
        observed[locus] = (
            int(block),
            float(lower),
            float(upper),
            float(sigma),
        )
    block_by_locus = {
        locus: block
        for block, (_block_id, loci) in enumerate(DIPLOID_LINKAGE_BLOCKS)
        for locus in loci
    }
    assert tuple(observed) == DIPLOID_SCALAR_LOCI
    for locus in DIPLOID_SCALAR_LOCI:
        block, lower, upper, sigma = observed[locus]
        assert block == block_by_locus[locus]
        assert (lower, upper) == LOCUS_BOUNDS[locus]
        assert sigma == MUTATION_SIGMA.get(locus, 0.0)
    assert MEIOSIS_DRAW_COUNT == 10
    assert MUTATION_SEMANTIC_DRAW_COUNT == 51


def test_sm75_cuda_primitive_executes(tmp_path: Path) -> None:
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        pytest.skip("nvcc is unavailable")
    source_path = tmp_path / "probe.cu"
    executable = tmp_path / "probe"
    source_path.write_text(textwrap.dedent(CUDA_SOURCE), encoding="utf-8")
    subprocess.run(
        [
            nvcc,
            "-std=c++17",
            "-arch=sm_75",
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
    completed = subprocess.run(
        [str(executable)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.strip() == "cuda_diploid_scalar_genome_v1_ok"


def test_header_declares_soa_and_zero_draw_migration() -> None:
    text = HEADER.read_text(encoding="utf-8")
    assert "kAllelePlaneCount = 34" not in text
    assert "kLocusCount * kHomologCount" in text
    assert "stage_homozygous_migration_v1" in text
    assert "staged->semantic_draws = 0" in text
