from __future__ import annotations

from pathlib import Path
import shutil
import subprocess

import pytest

from engine.evolution.genome_upgrade_v2_c1_executor_v1 import (
    reference_dimensions_for_fixture_v1,
)
from engine.evolution.genome_upgrade_v2_causal_gate_v1 import (
    MANDATORY_FAMILIES,
    REQUIRED_DIMENSIONS,
    fixture_input_dimensions_v1,
)


ROOT = Path(__file__).resolve().parents[2]
PERF = ROOT / "engine" / "perf"
HEADER = PERF / "genome_upgrade_v2_q10_v1.cuh"
BINDING = PERF / "cuda_architecture_b_flattened_controller_v1_binding.cpp"
INTERFACE = PERF / "cuda_architecture_b_flattened_controller_v1.hpp"
BASE_CUDA = PERF / "cuda_architecture_b_flattened_controller_v1_kernels.cu"
WDL_CUDA = PERF / "cuda_architecture_b_flattened_wdl_controller_v2_kernels.cu"


def _compile_probe(tmp_path: Path) -> Path:
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("g++ is unavailable")
    source = tmp_path / "c1_endpoint_probe.cpp"
    executable = tmp_path / "c1_endpoint_probe"
    source.write_text(
        r"""
#include "genome_upgrade_v2_q10_v1.cuh"

#include <cstdint>
#include <iostream>

int main() {
    namespace gv2 = eiel::genome_upgrade_v2_q10_v1;
    while (true) {
        int64_t genotype_flat[gv2::kC1GenotypeWidth]{};
        int64_t context[gv2::kContextCount]{};
        if (!(std::cin >> genotype_flat[0])) {
            break;
        }
        for (int32_t index = 1; index < gv2::kC1GenotypeWidth; ++index) {
            std::cin >> genotype_flat[index];
        }
        for (int32_t index = 0; index < gv2::kContextCount; ++index) {
            std::cin >> context[index];
        }
        gv2::GenotypeV1 genotype{};
        int32_t flat_index = 0;
        for (int32_t homolog = 0; homolog < gv2::kHomologCount; ++homolog) {
            for (int32_t product = 0; product < gv2::kProductCount; ++product) {
                for (
                    int32_t parameter = 0;
                    parameter < gv2::kParameterCount;
                    ++parameter
                ) {
                    genotype.alleles[homolog][product][parameter] =
                        genotype_flat[flat_index++];
                }
            }
        }
        for (int32_t homolog = 0; homolog < gv2::kHomologCount; ++homolog) {
            for (int32_t product = 0; product < gv2::kProductCount; ++product) {
                genotype.modifiers[homolog][product] =
                    genotype_flat[flat_index++];
            }
        }
        if (gv2::c1_input_status_v1(genotype, context) != 0) {
            return 2;
        }
        gv2::ExpressionV1 expression{};
        gv2::RegulatoryDiagnosticsV1 diagnostics{};
        gv2::express_v1(genotype, context, &expression, &diagnostics);
        const auto runtime = gv2::project_runtime_traits_v1(expression);
        int64_t genotype_out[gv2::kC1GenotypeWidth]{};
        int64_t context_out[gv2::kContextCount]{};
        int64_t product_out[gv2::kC1ProductWidth]{};
        int64_t regulatory_out[gv2::kC1RegulatoryWidth]{};
        int64_t phenotype_out[gv2::kPhenotypeCount]{};
        int64_t runtime_out[gv2::kC1RuntimeWidth]{};
        int64_t cost_out[gv2::kC1CostWidth]{};
        gv2::write_c1_dimensions_v1(
            genotype,
            context,
            expression,
            diagnostics,
            runtime,
            genotype_out,
            context_out,
            product_out,
            regulatory_out,
            phenotype_out,
            runtime_out,
            cost_out
        );
        const auto emit = [](const int64_t* values, const int32_t count) {
            for (int32_t index = 0; index < count; ++index) {
                std::cout << values[index] << ' ';
            }
        };
        emit(genotype_out, gv2::kC1GenotypeWidth);
        emit(context_out, gv2::kContextCount);
        emit(product_out, gv2::kC1ProductWidth);
        emit(regulatory_out, gv2::kC1RegulatoryWidth);
        emit(phenotype_out, gv2::kPhenotypeCount);
        emit(runtime_out, gv2::kC1RuntimeWidth);
        emit(cost_out, gv2::kC1CostWidth);
        std::cout << '\n';
    }
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
            str(PERF),
            str(source),
            "-o",
            str(executable),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return executable


def test_host_header_c1_batch_matches_all_frozen_reference_arms(
    tmp_path: Path,
) -> None:
    executable = _compile_probe(tmp_path)
    inputs = []
    expected = []
    for family in MANDATORY_FAMILIES:
        for arm in ("control", "treatment"):
            genotype, context = fixture_input_dimensions_v1(family, arm)
            inputs.append(" ".join(str(value) for value in (*genotype, *context)))
            dimensions = reference_dimensions_for_fixture_v1(family, arm)
            expected.append(
                tuple(
                    value
                    for name in REQUIRED_DIMENSIONS
                    for value in dimensions[name]
                )
            )
    result = subprocess.run(
        [str(executable)],
        input="\n".join(inputs) + "\n",
        check=True,
        capture_output=True,
        text=True,
    )
    actual = [
        tuple(int(value) for value in line.split())
        for line in result.stdout.splitlines()
    ]
    assert actual == expected


def test_production_binding_freezes_strict_native_and_cuda_endpoints() -> None:
    source = BINDING.read_text(encoding="utf-8")
    assert (
        '"eiel_cuda_architecture_b_genome_upgrade_v2_c1_native_batch_v1"'
        in source
    )
    assert (
        '"eiel_cuda_architecture_b_genome_upgrade_v2_c1_cuda_batch_v1"'
        in source
    )
    for contract in (
        "must use int64",
        "must have shape [batch, 120]",
        "must have shape [batch, 8]",
        "must be contiguous",
        "must not require gradients",
        'result["no_jit"] = true',
        'result["no_fallback"] = true',
        "c1_input_status_v1",
        "C10_CUDA_KERNEL_LAUNCH_CHECK",
        "cudaStreamSynchronize",
    ):
        assert contract in source


@pytest.mark.parametrize(
    "defect",
    ["activity", "quantity", "context", "stage"],
)
def test_host_header_c1_batch_rejects_out_of_contract_inputs(
    tmp_path: Path,
    defect: str,
) -> None:
    executable = _compile_probe(tmp_path)
    genotype, context = fixture_input_dimensions_v1(
        MANDATORY_FAMILIES[0],
        "control",
    )
    genotype_values = list(genotype)
    context_values = list(context)
    if defect == "activity":
        genotype_values[0] = 1025
    elif defect == "quantity":
        genotype_values[1] = -1
    elif defect == "context":
        context_values[0] = 1025
    else:
        context_values[6] = 1
        context_values[7] = 1
    result = subprocess.run(
        [str(executable)],
        input=" ".join(
            str(value) for value in (*genotype_values, *context_values)
        )
        + "\n",
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2


def test_c1_diagnostics_do_not_expand_production_expression_layout() -> None:
    source = HEADER.read_text(encoding="utf-8")
    expression = source.split("struct ExpressionV1 {", 1)[1].split("};", 1)[0]
    assert "regulator_nodes" not in expression
    assert "epistatic_delta" not in expression
    assert "sizeof(ExpressionV1) == 49 * sizeof(int64_t)" in source
    assert "struct RegulatoryDiagnosticsV1" in source
    assert "sizeof(RegulatoryDiagnosticsV1) == 20 * sizeof(int64_t)" in source


def test_shared_cuda_launcher_is_emitted_for_base_and_unified_wdl_artifacts() -> None:
    interface = INTERFACE.read_text(encoding="utf-8")
    assert "#if defined(__CUDACC__)" in interface
    assert "void launch_genome_upgrade_v2_c1_batch_v1(" in interface
    assert "inline void launch_genome_upgrade_v2_c1_batch_v1(" not in interface
    include = '#include "cuda_architecture_b_flattened_controller_v1.hpp"'
    assert include in BASE_CUDA.read_text(encoding="utf-8")
    assert include in WDL_CUDA.read_text(encoding="utf-8")
    includers = {
        path.name
        for path in PERF.glob("*.cu")
        if include in path.read_text(encoding="utf-8")
    }
    assert includers == {BASE_CUDA.name, WDL_CUDA.name}
