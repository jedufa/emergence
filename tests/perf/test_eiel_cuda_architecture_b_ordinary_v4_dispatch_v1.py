from __future__ import annotations

from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[2]
HEADER = ROOT / "engine/perf/cuda_architecture_b_cross_path_resident_control_v1.cuh"
KERNELS = ROOT / "engine/perf/cuda_architecture_b_flattened_controller_v1_kernels.cu"


def test_cross_path_control_host_contract_compiles_and_gates(tmp_path: Path) -> None:
    compiler = shutil.which("g++") or shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler unavailable")
    source = tmp_path / "cross_path_control.cpp"
    executable = tmp_path / "cross_path_control"
    source.write_text(
        r'''
#include "engine/perf/cuda_architecture_b_cross_path_resident_control_v1.cuh"

#include <cstdint>

namespace control =
    eiel::cuda_architecture_b::cross_path_resident_control_v1;

int main() {
    control::FlatCrossPathResidentControlV1 legacy{};
    if (!control::ordinary_live_birth_cell_v1(legacy, 0)) return 1;
    if (control::has_cross_path_authority_v1(legacy)) return 2;

    std::int8_t paths[2] = {
        control::kReproductionPathOrdinaryLiveBirth,
        control::kReproductionPathGenerationBoundaryRefill,
    };
    std::int8_t selections[2] = {};
    std::int8_t modes[2] = {};
    bool mutation[2] = {};
    bool neutral_fixture[2] = {true, true};
    bool trait_authority[2] = {};
    std::uint64_t selection_seeds[2] = {101, 202};
    double neutral_thresholds[2] = {5.0, 5.0};
    std::int64_t codes[4] = {};
    std::int64_t ranks[4] = {};
    std::int64_t canonical_0[2] = {};
    std::int64_t canonical_1[2] = {};
    std::int64_t child_slots[2] = {};
    std::uint8_t donors[14] = {};
    std::uint8_t donor_statuses[14] = {};
    std::int64_t donor_draws[2] = {};
    std::int64_t staged_counts[2] = {};
    std::int64_t committed_counts[2] = {};
    std::int32_t statuses[2] = {};
    control::FlatCrossPathResidentControlV1 resident{
        paths,
        selections,
        modes,
        mutation,
        neutral_fixture,
        trait_authority,
        selection_seeds,
        neutral_thresholds,
        codes,
        ranks,
        2,
        1,
        canonical_0,
        canonical_1,
        child_slots,
        donors,
        donor_statuses,
        staged_counts,
        committed_counts,
        donor_draws,
        statuses,
    };
    if (!control::has_complete_ordinary_authority_v1(resident)) return 3;
    if (!control::ordinary_live_birth_cell_v1(resident, 0)) return 4;
    if (control::ordinary_live_birth_cell_v1(resident, 1)) return 5;
    return 0;
}
''',
        encoding="utf-8",
    )
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-I",
            str(ROOT),
            str(source),
            "-o",
            str(executable),
        ],
        check=True,
        cwd=ROOT,
    )
    subprocess.run([str(executable)], check=True, cwd=ROOT)


def test_cross_path_control_cuda_contract_compiles(tmp_path: Path) -> None:
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        pytest.skip("nvcc unavailable")
    source = tmp_path / "cross_path_control.cu"
    output = tmp_path / "cross_path_control.o"
    source.write_text(
        r'''
#include "engine/perf/cuda_architecture_b_cross_path_resident_control_v1.cuh"

namespace control =
    eiel::cuda_architecture_b::cross_path_resident_control_v1;

__global__ void probe(
    control::FlatCrossPathResidentControlV1 authority,
    bool* output
) {
    output[0] = control::ordinary_live_birth_cell_v1(authority, 0);
    output[1] = control::has_complete_ordinary_authority_v1(authority);
}
''',
        encoding="utf-8",
    )
    subprocess.run(
        [
            nvcc,
            "-std=c++17",
            "-c",
            "-I",
            str(ROOT),
            str(source),
            "-o",
            str(output),
        ],
        check=True,
        cwd=ROOT,
    )


def test_real_flattened_transaction_dispatches_v4_and_preserves_legacy() -> None:
    text = KERNELS.read_text(encoding="utf-8")
    assert (
        "stage_ordinary_cross_path_parent_authority_v1(" in text
        and "stage_ordinary_fixed_mlp_child_scalar_genomes_with_mates_external_scratch_v4("
        in text
        and "stage_ordinary_fixed_mlp_child_scalar_genomes_with_mates_external_scratch_v3("
        in text
    )
    assert text.count("ordinary_live_birth_cell_v1(") >= 3
    assert "control.mutation_enabled_by_cell[cell]" in text
    assert "control.donor_draws_added[cell] = linked" in text
    assert "control.inheritance_status[cell]" in text
    assert "parent_rank <= mate_rank" in text
