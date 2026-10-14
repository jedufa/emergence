#include "../../engine/perf/cuda_architecture_b_compact_summary_reporting_authority_v1.cuh"

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <type_traits>

namespace authority =
    eiel::cuda_architecture_b::compact_summary_reporting_authority_v1;

static_assert(sizeof(authority::GenerationAuthorityHeaderV1) == 16);
static_assert(sizeof(authority::BehaviorAuthorityRowV1) == 56);
static_assert(sizeof(authority::LineageAuthorityRowV1) == 136);
static_assert(
    std::string_view(authority::kRetainedLayoutVersionV2)
    == "eiel_cuda_architecture_b_compact_summary_reporting_retained_layout_v2"
);
static_assert(alignof(authority::LineageAuthorityRowV1) == 8);
static_assert(offsetof(authority::LineageAuthorityRowV1, agent_id) == 0);
static_assert(offsetof(authority::LineageAuthorityRowV1, identity_drift_ema) == 32);
static_assert(offsetof(authority::LineageAuthorityRowV1, drives) == 48);
static_assert(offsetof(authority::LineageAuthorityRowV1, y) == 80);
static_assert(offsetof(authority::LineageAuthorityRowV1, use_gru) == 132);
static_assert(offsetof(authority::LineageAuthorityRowV1, reserved) == 133);
static_assert(
    authority::retained_bytes_per_generation_cell_v1(64) == 12304
);
static_assert(
    authority::retained_bytes_per_generation_cell_v1(224) == 43024
);
static_assert(std::is_standard_layout_v<authority::BehaviorAuthorityRowV1>);
static_assert(std::is_trivially_copyable_v<authority::LineageAuthorityRowV1>);

__global__ void compile_reporting_authority_v1(
    authority::GenerationAuthorityHeaderV1* header,
    authority::BehaviorAuthorityRowV1* behavior,
    authority::LineageAuthorityRowV1* lineage,
    std::int32_t* slot_to_row,
    double* effective
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    authority::GenerationAuthorityAccessV1 access{
        header,
        behavior,
        lineage,
        slot_to_row,
        effective,
        authority::kStrictRowCapacityV1,
        authority::kStrictRowCapacityV1,
    };
    authority::initialize_generation_v1(access, 0, nullptr, nullptr, nullptr, 0);
}
