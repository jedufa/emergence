#pragma once

#include <cstdint>

namespace eiel::snapshot::projection_v1 {

// G2 intentionally implements only the synchronous, contiguous host core.
// The enum is part of the capsule header so a later CUDA projection owner can
// be added without pretending that this artifact performed device work.
enum class ProjectionBackend : std::uint32_t {
    HOST_CONTIGUOUS_SYNC_V1 = 1,
    CUDA_PACKED_ASYNC_V1 = 2,
};

constexpr ProjectionBackend kImplementedBackend =
    ProjectionBackend::HOST_CONTIGUOUS_SYNC_V1;
constexpr bool kSupportsCudaProjection = false;

}  // namespace eiel::snapshot::projection_v1
