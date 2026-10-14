#include "../../engine/perf/cuda_architecture_b_compact_summary_source_projection_v1.cuh"

#include <cstddef>
#include <type_traits>

namespace projection =
    eiel::cuda_architecture_b::compact_summary_source_projection_v1;
namespace abi = eiel::cuda_architecture_b::compact_summary_v2;

using Authority = projection::CompactSummaryGenerationCellAuthorityV1;

static_assert(projection::kAuthoritySourceVersionV1 == 1);
static_assert(projection::kAuthoritySourcePlaneLogicalBytesV1 == 1218);
static_assert(projection::kAuthorityByteWidthV1 == 1256);
static_assert(abi::kPlaneCountV2 == 7);
static_assert(std::is_standard_layout_v<Authority>);
static_assert(std::is_trivially_copyable_v<Authority>);
static_assert(sizeof(Authority) == 1256);
static_assert(alignof(Authority) == 8);
static_assert(offsetof(Authority, status_codes) == 40);
static_assert(offsetof(Authority, integer_values) == 88);
static_assert(offsetof(Authority, float_values) == 696);
static_assert(offsetof(Authority, digest_values) == 864);
static_assert(offsetof(Authority, text_values) == 1184);
static_assert(projection::kProjectionSuccess == 0);
static_assert(projection::kProjectionFailClosed != projection::kProjectionSuccess);

__global__ void compile_authority_layout_v1(
    const Authority* const authority,
    std::int64_t* const out
) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        out[0] = authority[0].generation_index;
        out[1] = authority[0].cell_index;
        out[2] = authority[0].integer_values[abi::kIntegerAlive];
    }
}
