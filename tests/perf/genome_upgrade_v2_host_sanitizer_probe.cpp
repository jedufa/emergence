#include <array>
#include <cstdint>
#include <iostream>
#include <vector>

#include "genome_upgrade_v2_q10_v1.cuh"
#include "cuda_architecture_b_resident_run_control_v2.hpp"

namespace genome_v2 = eiel::genome_upgrade_v2_q10_v1;
namespace run_control =
    eiel::cuda_architecture_b::resident_run_control_v2;

namespace {

struct HostRandomV1 {
    uint64_t state = 1;

    uint64_t randbelow(const uint64_t bound) {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        return bound == 0 ? 0 : state % bound;
    }

    double random_double() {
        return 0.0;
    }
};

}  // namespace

int main() {
    constexpr int64_t capacity = 256;
    const int64_t allele_count = genome_v2::kHomologCount
        * genome_v2::kProductCount * genome_v2::kParameterCount * capacity;
    const int64_t modifier_count =
        genome_v2::kHomologCount * genome_v2::kProductCount * capacity;
    const int64_t product_count = genome_v2::kProductCount * capacity;
    const int64_t phenotype_count = genome_v2::kPhenotypeCount * capacity;
    std::vector<int64_t> alleles(allele_count);
    std::vector<int64_t> modifiers(modifier_count);
    std::vector<int64_t> products(product_count);
    std::vector<int64_t> phenotypes(phenotype_count);
    std::vector<int64_t> expression_cost(capacity);
    std::vector<int64_t> active_products(capacity);
    std::vector<int64_t> saturated_products(capacity);
    const genome_v2::ResidentPlanesV1 planes{
        alleles.data(),
        modifiers.data(),
        products.data(),
        phenotypes.data(),
        expression_cost.data(),
        active_products.data(),
        saturated_products.data(),
        capacity,
    };
    for (int32_t homolog = 0; homolog < genome_v2::kHomologCount; ++homolog) {
        for (int32_t product = 0; product < genome_v2::kProductCount; ++product) {
            for (int64_t slot = 0; slot < capacity; ++slot) {
                alleles[genome_v2::allele_index(
                    planes, homolog, product, 1, slot
                )] = genome_v2::kScale;
                alleles[genome_v2::allele_index(
                    planes, homolog, product, 2, slot
                )] = genome_v2::kScale;
            }
        }
    }
    std::array<int64_t, genome_v2::kContextCount> context{};
    for (int64_t slot = 0; slot < capacity; ++slot) {
        context[slot % context.size()] = genome_v2::kScale;
        if (!genome_v2::express_resident_slot_v1(
                planes, slot, context.data()
            )) {
            return 2;
        }
    }
    HostRandomV1 random{};
    genome_v2::StagedChildV1 child{};
    if (
        genome_v2::stage_child_v1(
            planes, -1, 1, random, genome_v2::kScale, 64, &child
        )
        || !genome_v2::stage_child_v1(
            planes, 0, 1, random, genome_v2::kScale, 64, &child
        )
        || genome_v2::commit_staged_child_v1(
            planes, capacity, context.data(), child
        )
        || !genome_v2::commit_staged_child_v1(
            planes, capacity - 1, context.data(), child
        )
    ) {
        return 3;
    }
    run_control::RunControlV2 controller("host-sanitizer", 7);
    const auto start = controller.start_run(0, 2);
    const run_control::CancelRequestV2 request{
        run_control::kCancelRequestApiVersionV2,
        "request-1",
        start.identity.controller_id,
        start.identity.owner_epoch,
        "operator_requested",
    };
    if (controller.request_cancel(request).disposition != "accepted") {
        return 4;
    }
    const run_control::BoundaryMetadataV2 boundary{
        0,
        1,
        1,
        0,
        0,
        0,
        1,
        "checkpoint-1",
        std::string(64, 'a'),
        true,
        true,
    };
    const auto terminal = controller.consume_cancellation_at_boundary(boundary);
    if (
        !terminal.terminal
        || terminal.terminal_status != "cancelled"
        || controller.request_cancel(request).disposition != "already_terminal"
    ) {
        return 5;
    }
    std::cout << "GENOME_V2_HOST_SANITIZER_PROBE_PASS capacity256\n";
    return 0;
}
