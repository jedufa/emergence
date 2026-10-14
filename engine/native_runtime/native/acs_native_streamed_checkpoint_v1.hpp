#pragma once

#include <torch/extension.h>

#include <cstdint>
#include <array>
#include <string>

namespace eiel::native::acs_native_streamed_checkpoint_v1 {

// Copy one detached tensor to a Python sink in bounded field-major chunks.
// The sink owns hashing and durable-file writes; this native boundary owns
// contiguous conversion, CUDA-to-host staging, and the two reusable slabs.
std::uint64_t copy_tensor_chunks(
    torch::Tensor tensor,
    pybind11::object sink,
    std::int64_t chunk_bytes
);

// Reusable native capture owner for one bounded cell group.  The pool is
// deliberately independent of the file/container writer: it owns exactly two
// host slabs and rotates them in field-major order while the Python sink owns
// hashing and durable publication.
class SlabPool final {
public:
    SlabPool(std::int64_t chunk_bytes, bool pinned);

    std::uint64_t copy_tensor_chunks(
        torch::Tensor tensor,
        pybind11::object sink
    );

    pybind11::dict telemetry() const;

private:
    std::int64_t chunk_bytes_;
    bool pinned_;
    std::array<torch::Tensor, 2> slabs_;
    std::uint64_t next_slab_ = 0;
    std::uint64_t chunk_count_ = 0;
    std::uint64_t copy_operations_ = 0;
    std::uint64_t copied_bytes_ = 0;
    std::uint64_t field_count_ = 0;
    std::uint64_t last_slab_index_ = 0;
    std::uint64_t cuda_event_wait_count_ = 0;
    std::uint64_t final_cuda_fence_count_ = 0;
    std::string source_device_ = "unknown";
};

}  // namespace eiel::native::acs_native_streamed_checkpoint_v1
