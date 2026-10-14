#include <cuda_runtime.h>

#include <cstdint>
#include <iostream>

#include "genome_upgrade_v2_q10_v1.cuh"

namespace genome_v2 = eiel::genome_upgrade_v2_q10_v1;

namespace {

constexpr int64_t kCells = 72;
constexpr int64_t kSlots = 256;

struct DeviceRandomV1 {
    uint64_t state;

    __device__ uint64_t randbelow(const uint64_t bound) {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        return bound == 0 ? 0 : state % bound;
    }

    __device__ double random_double() {
        return 0.0;
    }
};

__device__ bool genotype_equal(
    const genome_v2::GenotypeV1& first,
    const genome_v2::GenotypeV1& second
) {
    for (int32_t homolog = 0; homolog < genome_v2::kHomologCount; ++homolog) {
        for (int32_t product = 0; product < genome_v2::kProductCount; ++product) {
            for (
                int32_t parameter = 0;
                parameter < genome_v2::kParameterCount;
                ++parameter
            ) {
                if (
                    first.alleles[homolog][product][parameter]
                    != second.alleles[homolog][product][parameter]
                ) {
                    return false;
                }
            }
            if (
                first.modifiers[homolog][product]
                != second.modifiers[homolog][product]
            ) {
                return false;
            }
        }
    }
    return true;
}

__global__ void run_probe(
    int64_t* const alleles,
    int64_t* const modifiers,
    int64_t* const products,
    int64_t* const phenotypes,
    int64_t* const expression_cost,
    int64_t* const active_products,
    int64_t* const saturated_products,
    int32_t* const status
) {
    const int64_t cell = blockIdx.x;
    const int64_t slot = threadIdx.x;
    if (cell >= kCells || slot >= kSlots) {
        return;
    }
    const int64_t allele_stride =
        genome_v2::kHomologCount * genome_v2::kProductCount
        * genome_v2::kParameterCount * kSlots;
    const int64_t modifier_stride =
        genome_v2::kHomologCount * genome_v2::kProductCount * kSlots;
    const int64_t product_stride = genome_v2::kProductCount * kSlots;
    const int64_t phenotype_stride = genome_v2::kPhenotypeCount * kSlots;
    const genome_v2::ResidentPlanesV1 planes{
        alleles + cell * allele_stride,
        modifiers + cell * modifier_stride,
        products + cell * product_stride,
        phenotypes + cell * phenotype_stride,
        expression_cost + cell * kSlots,
        active_products + cell * kSlots,
        saturated_products + cell * kSlots,
        kSlots,
    };
    int64_t context[genome_v2::kContextCount]{};
    for (int32_t index = 0; index < genome_v2::kContextCount; ++index) {
        context[index] = (slot & (1LL << index)) != 0
            ? genome_v2::kScale
            : 0;
    }
    if (!genome_v2::express_resident_slot_v1(planes, slot, context)) {
        atomicExch(status + cell, 1);
    }
    const genome_v2::RuntimeProjectionV1 runtime =
        genome_v2::load_runtime_projection_v1(planes, slot);
    double movement = 0.0;
    double expression = 0.0;
    double threshold = 0.0;
    double offspring = 0.0;
    double health = 0.0;
    if (
        !genome_v2::successful_movement_cost_v1(
            4.0, runtime, &movement
        )
        || !genome_v2::expression_energy_debit_v1(runtime, &expression)
        || !genome_v2::reproduction_threshold_v1(
            40.0, runtime, &threshold
        )
        || !genome_v2::scale_offspring_transfer_v1(
            20.0, runtime.offspring_energy_factor_q10, &offspring
        )
        || !genome_v2::health_after_delta_v1(
            80.0, 60.0, 100.0, runtime, &health
        )
        || movement < 0.0 || expression < 0.0 || threshold < 0.0
        || offspring < 0.0 || health < 0.0 || health > 100.0
        || runtime.memory_dimension < 0 || runtime.memory_dimension > 64
    ) {
        atomicExch(status + cell, 2);
    }
    __syncthreads();
    if (status[cell] != 0 || slot != 0) {
        return;
    }

    DeviceRandomV1 birth_random{
        static_cast<uint64_t>(cell + 1)
    };
    genome_v2::StagedChildV1 birth_child{};
    if (
        genome_v2::stage_child_v1(
            planes, -1, 1, birth_random, genome_v2::kScale, 64, &birth_child
        )
        || !genome_v2::stage_child_v1(
            planes, 0, 1, birth_random, genome_v2::kScale, 64, &birth_child
        )
        || birth_child.semantic_draws != genome_v2::kInheritanceSemanticDrawCount
        || birth_child.changed_loci <= 0
    ) {
        atomicExch(status + cell, 3);
        return;
    }
    genome_v2::GenotypeV1 rollback_before{};
    genome_v2::GenotypeV1 rollback_after{};
    genome_v2::load_genotype_v1(planes, 2, &rollback_before);
    if (
        genome_v2::commit_staged_child_v1(
            planes, kSlots, context, birth_child
        )
    ) {
        atomicExch(status + cell, 4);
        return;
    }
    genome_v2::load_genotype_v1(planes, 2, &rollback_after);
    if (!genotype_equal(rollback_before, rollback_after)) {
        atomicExch(status + cell, 5);
        return;
    }
    if (
        !genome_v2::commit_staged_child_v1(
            planes, 2, context, birth_child
        )
    ) {
        atomicExch(status + cell, 6);
        return;
    }
    genome_v2::GenotypeV1 committed_birth{};
    genome_v2::load_genotype_v1(planes, 2, &committed_birth);
    if (
        !genotype_equal(committed_birth, birth_child.genotype)
        || planes.active_product_count[2] <= 0
    ) {
        atomicExch(status + cell, 7);
        return;
    }

    DeviceRandomV1 turnover_random{
        static_cast<uint64_t>(cell + 1001)
    };
    genome_v2::StagedChildV1 turnover_child{};
    if (
        !genome_v2::stage_child_v1(
            planes,
            0,
            1,
            turnover_random,
            genome_v2::kScale,
            64,
            &turnover_child
        )
        || !genome_v2::commit_staged_child_v1(
            planes, 3, context, turnover_child
        )
    ) {
        atomicExch(status + cell, 8);
        return;
    }
    genome_v2::GenotypeV1 committed_turnover{};
    genome_v2::load_genotype_v1(planes, 3, &committed_turnover);
    if (!genotype_equal(committed_turnover, turnover_child.genotype)) {
        atomicExch(status + cell, 9);
    }
}

template <typename T>
bool managed_allocate(T** const pointer, const int64_t count) {
    return cudaMallocManaged(pointer, sizeof(T) * count) == cudaSuccess;
}

}  // namespace

int main() {
    const int64_t allele_count =
        kCells * genome_v2::kHomologCount * genome_v2::kProductCount
        * genome_v2::kParameterCount * kSlots;
    const int64_t modifier_count =
        kCells * genome_v2::kHomologCount * genome_v2::kProductCount * kSlots;
    const int64_t product_count =
        kCells * genome_v2::kProductCount * kSlots;
    const int64_t phenotype_count =
        kCells * genome_v2::kPhenotypeCount * kSlots;
    const int64_t scalar_count = kCells * kSlots;
    int64_t* alleles = nullptr;
    int64_t* modifiers = nullptr;
    int64_t* products = nullptr;
    int64_t* phenotypes = nullptr;
    int64_t* expression_cost = nullptr;
    int64_t* active_products = nullptr;
    int64_t* saturated_products = nullptr;
    int32_t* status = nullptr;
    if (
        !managed_allocate(&alleles, allele_count)
        || !managed_allocate(&modifiers, modifier_count)
        || !managed_allocate(&products, product_count)
        || !managed_allocate(&phenotypes, phenotype_count)
        || !managed_allocate(&expression_cost, scalar_count)
        || !managed_allocate(&active_products, scalar_count)
        || !managed_allocate(&saturated_products, scalar_count)
        || !managed_allocate(&status, kCells)
    ) {
        return 2;
    }
    if (
        cudaMemset(alleles, 0, sizeof(int64_t) * allele_count) != cudaSuccess
        || cudaMemset(modifiers, 0, sizeof(int64_t) * modifier_count)
            != cudaSuccess
        || cudaMemset(products, 0, sizeof(int64_t) * product_count)
            != cudaSuccess
        || cudaMemset(phenotypes, 0, sizeof(int64_t) * phenotype_count)
            != cudaSuccess
        || cudaMemset(expression_cost, 0, sizeof(int64_t) * scalar_count)
            != cudaSuccess
        || cudaMemset(active_products, 0, sizeof(int64_t) * scalar_count)
            != cudaSuccess
        || cudaMemset(saturated_products, 0, sizeof(int64_t) * scalar_count)
            != cudaSuccess
        || cudaMemset(status, 0, sizeof(int32_t) * kCells) != cudaSuccess
    ) {
        return 3;
    }
    const genome_v2::ResidentPlanesV1 all_planes{
        alleles,
        modifiers,
        products,
        phenotypes,
        expression_cost,
        active_products,
        saturated_products,
        kSlots,
    };
    for (int64_t cell = 0; cell < kCells; ++cell) {
        genome_v2::ResidentPlanesV1 planes = all_planes;
        planes.alleles += cell
            * genome_v2::kHomologCount * genome_v2::kProductCount
            * genome_v2::kParameterCount * kSlots;
        for (int32_t homolog = 0; homolog < genome_v2::kHomologCount; ++homolog) {
            for (
                int32_t product = 0;
                product < genome_v2::kProductCount;
                ++product
            ) {
                for (int64_t slot = 0; slot < kSlots; ++slot) {
                    planes.alleles[
                        genome_v2::allele_index(
                            planes, homolog, product, 1, slot
                        )
                    ] = genome_v2::kScale;
                    planes.alleles[
                        genome_v2::allele_index(
                            planes, homolog, product, 2, slot
                        )
                    ] = genome_v2::kScale;
                }
            }
        }
    }
    run_probe<<<kCells, kSlots>>>(
        alleles,
        modifiers,
        products,
        phenotypes,
        expression_cost,
        active_products,
        saturated_products,
        status
    );
    if (cudaDeviceSynchronize() != cudaSuccess) {
        return 4;
    }
    for (int64_t cell = 0; cell < kCells; ++cell) {
        if (status[cell] != 0) {
            return 10 + status[cell];
        }
    }
    std::cout << "GENOME_V2_SANITIZER_PROBE_PASS b72_capacity256\n";
    cudaFree(alleles);
    cudaFree(modifiers);
    cudaFree(products);
    cudaFree(phenotypes);
    cudaFree(expression_cost);
    cudaFree(active_products);
    cudaFree(saturated_products);
    cudaFree(status);
    return 0;
}
