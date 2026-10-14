from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from evolution.genome_upgrade_v2 import (
    GenomeUpgradeV2V1,
    assemble_child_from_tape_v1,
    build_runtime_context_q10_v1,
    express_reference_v1,
    project_runtime_traits_v1,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
HEADER_DIR = PROJECT_ROOT / "engine" / "perf"


def _fixture() -> tuple[GenomeUpgradeV2V1, tuple[int, ...]]:
    value = GenomeUpgradeV2V1.neutral()
    alleles = [
        [list(parameters) for parameters in homolog]
        for homolog in value.alleles
    ]
    modifiers = [list(homolog) for homolog in value.modifiers]
    alleles[0][9] = [701, 823, 991, -317]
    alleles[1][9] = [-257, 719, 877, 401]
    modifiers[0][9] = 211
    modifiers[1][9] = -99
    return (
        GenomeUpgradeV2V1(
            alleles=tuple(
                tuple(tuple(parameters) for parameters in homolog)
                for homolog in alleles
            ),
            modifiers=tuple(tuple(homolog) for homolog in modifiers),
        ),
        (113, -227, 331, -449, 557, -661, 773, -881),
    )


def _assignments(
    value: GenomeUpgradeV2V1,
    context: tuple[int, ...],
    *,
    genotype: str,
) -> str:
    rows = []
    for homolog in range(2):
        for product in range(12):
            for parameter in range(4):
                rows.append(
                    f"{genotype}alleles"
                    f"[{homolog}][{product}][{parameter}]="
                    f"{value.alleles[homolog][product][parameter]};"
                )
            rows.append(
                f"{genotype}modifiers[{homolog}][{product}]="
                f"{value.modifiers[homolog][product]};"
            )
    rows.extend(f"context[{index}]={item};" for index, item in enumerate(context))
    return "\n".join(rows)


def _parse_output(text: str) -> dict[str, tuple[int, ...] | int]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    assert len(lines) == 3
    return {
        "products": tuple(int(value) for value in lines[0].split()),
        "phenotypes": tuple(int(value) for value in lines[1].split()),
        "summary": tuple(int(value) for value in lines[2].split()),
    }


def _assert_expected(output: dict[str, tuple[int, ...] | int]) -> None:
    value, context = _fixture()
    expected = express_reference_v1(value, context)
    assert output["products"] == expected.products_q10
    assert output["phenotypes"] == expected.phenotypes_q10
    assert output["summary"] == (
        expected.expression_cost_q10,
        expected.active_product_count,
        expected.saturated_product_count,
    )


def test_native_cpp_matches_reference_exactly(tmp_path: Path) -> None:
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("g++ is unavailable")
    value, context = _fixture()
    source = tmp_path / "gv2_native.cpp"
    binary = tmp_path / "gv2_native"
    source.write_text(
        f"""
#include <iostream>
#include "genome_upgrade_v2_q10_v1.cuh"
using namespace eiel::genome_upgrade_v2_q10_v1;
int main() {{
    GenotypeV1 g{{}};
    int64_t context[kContextCount]{{}};
    {_assignments(value, context, genotype="g.")}
    ExpressionV1 out{{}};
    express_v1(g, context, &out);
    for (int i=0;i<kProductCount;++i) std::cout << out.products[i] << ' ';
    std::cout << '\\n';
    for (int i=0;i<kPhenotypeCount;++i) std::cout << out.phenotypes[i] << ' ';
    std::cout << '\\n' << out.expression_cost << ' '
              << out.active_product_count << ' '
              << out.saturated_product_count << '\\n';
}}
""",
        encoding="ascii",
    )
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O2",
            "-I",
            str(HEADER_DIR),
            str(source),
            "-o",
            str(binary),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    completed = subprocess.run(
        [str(binary)], check=True, capture_output=True, text=True
    )
    _assert_expected(_parse_output(completed.stdout))


def test_native_runtime_projection_matches_reference_exactly(
    tmp_path: Path,
) -> None:
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("g++ is unavailable")
    value, context = _fixture()
    expression = express_reference_v1(value, context)
    expected = project_runtime_traits_v1(expression)
    source = tmp_path / "gv2_runtime_projection.cpp"
    binary = tmp_path / "gv2_runtime_projection"
    source.write_text(
        f"""
#include <iostream>
#include "genome_upgrade_v2_q10_v1.cuh"
using namespace eiel::genome_upgrade_v2_q10_v1;
int main() {{
    GenotypeV1 g{{}};
    int64_t context[kContextCount]{{}};
    {_assignments(value, context, genotype="g.")}
    ExpressionV1 expression{{}};
    express_v1(g, context, &expression);
    const RuntimeProjectionV1 out = project_runtime_traits_v1(expression);
    std::cout
        << out.body_size_factor_q10 << ' '
        << out.movement_capacity_factor_q10 << ' '
        << out.damage_received_factor_q10 << ' '
        << out.metabolic_efficiency_factor_q10 << ' '
        << out.recovery_factor_q10 << ' '
        << out.sensory_fidelity_factor_q10 << ' '
        << out.reproduction_threshold_factor_q10 << ' '
        << out.offspring_energy_factor_q10 << ' '
        << out.threat_response_q10 << ' '
        << out.memory_allocation_q10 << ' '
        << out.memory_dimension << ' '
        << out.expression_energy_debit_q10 << '\\n';
}}
""",
        encoding="ascii",
    )
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O2",
            "-I",
            str(HEADER_DIR),
            str(source),
            "-o",
            str(binary),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    completed = subprocess.run(
        [str(binary)], check=True, capture_output=True, text=True
    )
    observed = tuple(int(value) for value in completed.stdout.split())
    assert observed == (
        expected.body_size_factor_q10,
        expected.movement_capacity_factor_q10,
        expected.damage_received_factor_q10,
        expected.metabolic_efficiency_factor_q10,
        expected.recovery_factor_q10,
        expected.sensory_fidelity_factor_q10,
        expected.reproduction_threshold_factor_q10,
        expected.offspring_energy_factor_q10,
        expected.threat_response_q10,
        expected.memory_allocation_q10,
        expected.memory_dimension,
        expected.expression_energy_debit_q10,
    )


def test_native_runtime_context_matches_reference_exactly(tmp_path: Path) -> None:
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("g++ is unavailable")
    inputs = {
        "energy": 25.0,
        "max_energy": 100.0,
        "injury": 12.5,
        "start_health": 50.0,
        "crowding": 0.5,
        "resource_a": 1.5,
        "resource_b": -0.25,
        "hazard": 0.125,
        "age": 7,
        "maturity_age": 8,
    }
    expected = build_runtime_context_q10_v1(**inputs)
    source = tmp_path / "gv2_runtime_context.cpp"
    binary = tmp_path / "gv2_runtime_context"
    source.write_text(
        """
#include <iostream>
#include "genome_upgrade_v2_q10_v1.cuh"
using namespace eiel::genome_upgrade_v2_q10_v1;
int main() {
    RuntimeContextInputsV1 inputs{
        25.0, 100.0, 12.5, 50.0, 0.5, 1.5, -0.25, 0.125, 7, 8
    };
    int64_t context[kContextCount]{};
    if (!build_runtime_context_q10_v1(inputs, context)) return 2;
    for (int i = 0; i < kContextCount; ++i) {
        std::cout << context[i] << ' ';
    }
    std::cout << '\\n';
}
""",
        encoding="ascii",
    )
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O2",
            "-I",
            str(HEADER_DIR),
            str(source),
            "-o",
            str(binary),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    completed = subprocess.run(
        [str(binary)], check=True, capture_output=True, text=True
    )
    assert tuple(int(value) for value in completed.stdout.split()) == expected


def test_native_offspring_transfer_preserves_debit_credit_contract(
    tmp_path: Path,
) -> None:
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("g++ is unavailable")
    source = tmp_path / "gv2_offspring_transfer.cpp"
    binary = tmp_path / "gv2_offspring_transfer"
    source.write_text(
        """
#include <iostream>
#include "genome_upgrade_v2_q10_v1.cuh"
using namespace eiel::genome_upgrade_v2_q10_v1;
int main() {
    double low = 0.0;
    double neutral = 0.0;
    double high = 0.0;
    if (!scale_offspring_transfer_v1(20.0, 768, &low)) return 2;
    if (!scale_offspring_transfer_v1(20.0, 1024, &neutral)) return 3;
    if (!scale_offspring_transfer_v1(20.0, 1280, &high)) return 4;
    double rejected = 0.0;
    if (scale_offspring_transfer_v1(20.0, 1281, &rejected)) return 5;
    std::cout << low << ' ' << neutral << ' ' << high << '\\n';
}
""",
        encoding="ascii",
    )
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O2",
            "-I",
            str(HEADER_DIR),
            str(source),
            "-o",
            str(binary),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    completed = subprocess.run(
        [str(binary)], check=True, capture_output=True, text=True
    )
    assert tuple(float(value) for value in completed.stdout.split()) == (
        15.0,
        20.0,
        25.0,
    )


def test_native_consequence_helpers_match_frozen_contract(tmp_path: Path) -> None:
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("g++ is unavailable")
    source = tmp_path / "gv2_consequences.cpp"
    binary = tmp_path / "gv2_consequences"
    source.write_text(
        """
#include <iostream>
#include "genome_upgrade_v2_q10_v1.cuh"
using namespace eiel::genome_upgrade_v2_q10_v1;
int main() {
    RuntimeProjectionV1 runtime{
        1280, 1280, 768, 1280, 1280, 1024,
        768, 1280, 0, 0, 32, 256
    };
    double movement = 0.0;
    double expression = 0.0;
    double threshold = 0.0;
    double damage = 0.0;
    double recovery = 0.0;
    if (!successful_movement_cost_v1(4.0, runtime, &movement)) return 2;
    if (!expression_energy_debit_v1(runtime, &expression)) return 3;
    if (!reproduction_threshold_v1(40.0, runtime, &threshold)) return 4;
    if (!health_after_delta_v1(80.0, 60.0, 100.0, runtime, &damage)) return 5;
    if (!health_after_delta_v1(80.0, 90.0, 100.0, runtime, &recovery)) return 6;
    std::cout
        << movement << ' ' << expression << ' ' << threshold << ' '
        << damage << ' ' << recovery << '\\n';
}
""",
        encoding="ascii",
    )
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O2",
            "-I",
            str(HEADER_DIR),
            str(source),
            "-o",
            str(binary),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    completed = subprocess.run(
        [str(binary)], check=True, capture_output=True, text=True
    )
    assert tuple(float(value) for value in completed.stdout.split()) == (
        3.2,
        0.25,
        30.0,
        65.0,
        92.5,
    )


def test_native_cuda_kernel_matches_reference_exactly(tmp_path: Path) -> None:
    compiler = shutil.which("nvcc")
    if compiler is None:
        pytest.skip("nvcc is unavailable")
    try:
        subprocess.run(
            ["nvidia-smi", "-L"], check=True, capture_output=True, text=True
        )
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("CUDA device is unavailable")
    value, context = _fixture()
    source = tmp_path / "gv2_cuda.cu"
    binary = tmp_path / "gv2_cuda"
    source.write_text(
        f"""
#include <cuda_runtime.h>
#include <iostream>
#include "genome_upgrade_v2_q10_v1.cuh"
using namespace eiel::genome_upgrade_v2_q10_v1;
__global__ void run(const GenotypeV1* g, const int64_t* context, ExpressionV1* out) {{
    express_v1(*g, context, out);
}}
int main() {{
    GenotypeV1* g = nullptr;
    int64_t* context = nullptr;
    ExpressionV1* out = nullptr;
    cudaMallocManaged(&g, sizeof(GenotypeV1));
    cudaMallocManaged(&context, sizeof(int64_t) * kContextCount);
    cudaMallocManaged(&out, sizeof(ExpressionV1));
    *g = GenotypeV1{{}};
    {_assignments(value, context, genotype="g->")}
    run<<<1,1>>>(g, context, out);
    if (cudaDeviceSynchronize() != cudaSuccess) return 2;
    for (int i=0;i<kProductCount;++i) std::cout << out->products[i] << ' ';
    std::cout << '\\n';
    for (int i=0;i<kPhenotypeCount;++i) std::cout << out->phenotypes[i] << ' ';
    std::cout << '\\n' << out->expression_cost << ' '
              << out->active_product_count << ' '
              << out->saturated_product_count << '\\n';
    cudaFree(g); cudaFree(context); cudaFree(out);
}}
""",
        encoding="ascii",
    )
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O2",
            "-arch=sm_75",
            "-I",
            str(HEADER_DIR),
            str(source),
            "-o",
            str(binary),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    completed = subprocess.run(
        [str(binary)], check=True, capture_output=True, text=True
    )
    _assert_expected(_parse_output(completed.stdout))


def test_cuda_runtime_context_matches_reference_exactly(tmp_path: Path) -> None:
    compiler = shutil.which("nvcc")
    if compiler is None:
        pytest.skip("nvcc is unavailable")
    try:
        subprocess.run(
            ["nvidia-smi", "-L"], check=True, capture_output=True, text=True
        )
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("CUDA device is unavailable")
    expected = build_runtime_context_q10_v1(
        energy=25.0,
        max_energy=100.0,
        injury=12.5,
        start_health=50.0,
        crowding=0.5,
        resource_a=1.5,
        resource_b=-0.25,
        hazard=0.125,
        age=7,
        maturity_age=8,
    )
    source = tmp_path / "gv2_runtime_context_cuda.cu"
    binary = tmp_path / "gv2_runtime_context_cuda"
    source.write_text(
        """
#include <cuda_runtime.h>
#include <iostream>
#include "genome_upgrade_v2_q10_v1.cuh"
using namespace eiel::genome_upgrade_v2_q10_v1;
__global__ void run(int64_t* context, int* success) {
    const RuntimeContextInputsV1 inputs{
        25.0, 100.0, 12.5, 50.0, 0.5, 1.5, -0.25, 0.125, 7, 8
    };
    *success = build_runtime_context_q10_v1(inputs, context) ? 1 : 0;
}
int main() {
    int64_t* context = nullptr;
    int* success = nullptr;
    cudaMallocManaged(&context, sizeof(int64_t) * kContextCount);
    cudaMallocManaged(&success, sizeof(int));
    run<<<1,1>>>(context, success);
    if (cudaDeviceSynchronize() != cudaSuccess || *success != 1) return 2;
    for (int i=0;i<kContextCount;++i) std::cout << context[i] << ' ';
    std::cout << '\\n';
    cudaFree(context);
    cudaFree(success);
}
""",
        encoding="ascii",
    )
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O2",
            "-arch=sm_75",
            "-I",
            str(HEADER_DIR),
            str(source),
            "-o",
            str(binary),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    completed = subprocess.run(
        [str(binary)], check=True, capture_output=True, text=True
    )
    assert tuple(int(value) for value in completed.stdout.split()) == expected


def test_native_child_assembly_matches_fixed_python_tape(tmp_path: Path) -> None:
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("g++ is unavailable")
    parent_a, _context = _fixture()
    base = GenomeUpgradeV2V1.neutral()
    alleles = [
        [list(parameters) for parameters in homolog]
        for homolog in base.alleles
    ]
    modifiers = [list(homolog) for homolog in base.modifiers]
    for homolog in range(2):
        for product in range(12):
            alleles[homolog][product] = (
                [-(product + 1) * 31, 700 + product, 900 - product, 17 * product]
                if homolog == 0
                else [23 * product, 600 + product, 800 - product, -19 * product]
            )
            modifiers[homolog][product] = (homolog * 2 - 1) * (product + 3) * 11
    parent_b = GenomeUpgradeV2V1(
        alleles=tuple(
            tuple(tuple(parameters) for parameters in homolog)
            for homolog in alleles
        ),
        modifiers=tuple(tuple(homolog) for homolog in modifiers),
    )
    donors = tuple(index % 2 for index in range(12))
    mutation_homologs = tuple((index // 3) % 2 for index in range(60))
    mutation_enabled = tuple(int(index % 4 == 0) for index in range(60))
    mutation_deltas = tuple((index % 129) - 64 for index in range(60))
    expected, receipt = assemble_child_from_tape_v1(
        parent_a,
        parent_b,
        donor_homologs=donors,
        mutation_homologs=mutation_homologs,
        mutation_enabled=mutation_enabled,
        mutation_delta_q10=mutation_deltas,
    )
    donor_values = ", ".join(str(value) for value in donors)
    homolog_values = ", ".join(str(value) for value in mutation_homologs)
    enabled_values = ", ".join(str(value) for value in mutation_enabled)
    delta_values = ", ".join(str(value) for value in mutation_deltas)
    source = tmp_path / "gv2_inheritance.cpp"
    binary = tmp_path / "gv2_inheritance"
    source.write_text(
        f"""
#include <cstdint>
#include <iostream>
#include "genome_upgrade_v2_q10_v1.cuh"
using namespace eiel::genome_upgrade_v2_q10_v1;
struct TapeRandom {{
    int donors[12]{{{donor_values}}};
    int homologs[60]{{{homolog_values}}};
    int enabled[60]{{{enabled_values}}};
    int deltas[60]{{{delta_values}}};
    int donor_cursor = 0;
    int homolog_cursor = 0;
    int decision_cursor = 0;
    int delta_cursor = 0;
    uint64_t randbelow(uint64_t n) {{
        if (n == 2 && donor_cursor < 12) return donors[donor_cursor++];
        if (n == 2) return homologs[homolog_cursor++];
        return static_cast<uint64_t>(deltas[delta_cursor++] + 64);
    }}
    double random_double() {{
        return enabled[decision_cursor++] ? 0.25 : 0.75;
    }}
}};
int main() {{
    constexpr int64_t capacity = 3;
    int64_t alleles[2 * 12 * 4 * capacity]{{}};
    int64_t modifiers[2 * 12 * capacity]{{}};
    int64_t products[12 * capacity]{{}};
    int64_t phenotypes[10 * capacity]{{}};
    int64_t costs[capacity]{{}};
    int64_t active[capacity]{{}};
    int64_t saturated[capacity]{{}};
    ResidentPlanesV1 planes{{
        alleles, modifiers, products, phenotypes, costs, active, saturated,
        capacity
    }};
    StagedChildV1 a{{}}, b{{}}, child{{}};
    int64_t context[kContextCount]{{}};
    {_assignments(parent_a, (0,) * 8, genotype="a.genotype.")}
    {_assignments(parent_b, (0,) * 8, genotype="b.genotype.")}
    commit_staged_child_v1(planes, 0, context, a);
    commit_staged_child_v1(planes, 1, context, b);
    TapeRandom random{{}};
    if (!stage_child_v1(planes, 0, 1, random, 512, 64, &child)) return 2;
    for (int h=0;h<2;++h) for (int p=0;p<12;++p) {{
        for (int q=0;q<4;++q)
            std::cout << child.genotype.alleles[h][p][q] << ' ';
        std::cout << child.genotype.modifiers[h][p] << ' ';
    }}
    std::cout << '\\n' << child.semantic_draws << ' '
              << child.changed_loci << '\\n';
}}
""",
        encoding="ascii",
    )
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O2",
            "-I",
            str(HEADER_DIR),
            str(source),
            "-o",
            str(binary),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    completed = subprocess.run(
        [str(binary)], check=True, capture_output=True, text=True
    )
    lines = completed.stdout.splitlines()
    observed = tuple(int(value) for value in lines[0].split())
    expected_values = tuple(
        value
        for homolog in range(2)
        for product in range(12)
        for value in (
            *expected.alleles[homolog][product],
            expected.modifiers[homolog][product],
        )
    )
    assert observed == expected_values
    assert tuple(int(value) for value in lines[1].split()) == (
        192,
        receipt["changed_loci"],
    )
