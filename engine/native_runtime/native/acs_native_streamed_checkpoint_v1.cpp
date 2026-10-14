#include "acs_native_streamed_checkpoint_v1.hpp"

#include <array>
#include <algorithm>
#include <cstring>

namespace eiel::native::acs_native_streamed_checkpoint_v1 {

std::uint64_t copy_tensor_chunks(
    torch::Tensor tensor,
    pybind11::object sink,
    const std::int64_t chunk_bytes
) {
    TORCH_CHECK(chunk_bytes > 0, "native checkpoint chunk_bytes must be positive");
    TORCH_CHECK(tensor.defined(), "native checkpoint tensor must be defined");
    TORCH_CHECK(
        sink && pybind11::hasattr(sink, "write"),
        "native checkpoint sink must expose write"
    );

    auto source = tensor.detach();
    if (!source.is_contiguous()) {
        source = source.contiguous();
    }
    auto bytes = source.view(torch::kUInt8).reshape({-1});
    const auto total_bytes = bytes.numel();
    if (total_bytes == 0) {
        return 0;
    }

    const auto slab_options = torch::TensorOptions()
        .dtype(torch::kUInt8)
        .device(torch::kCPU)
        .pinned_memory(source.device().is_cuda());
    std::array<torch::Tensor, 2> slabs{
        torch::empty({chunk_bytes}, slab_options),
        torch::empty({chunk_bytes}, slab_options),
    };
    std::uint64_t chunk_count = 0;
    for (std::int64_t offset = 0; offset < total_bytes; offset += chunk_bytes) {
        const auto length = std::min<std::int64_t>(chunk_bytes, total_bytes - offset);
        auto slab = slabs[chunk_count % slabs.size()];
        auto source_slice = bytes.slice(0, offset, offset + length);
        if (source.device().is_cuda()) {
            slab.slice(0, 0, length).copy_(source_slice, false);
        } else {
            std::memcpy(
                slab.data_ptr<std::uint8_t>(),
                source_slice.data_ptr<std::uint8_t>(),
                static_cast<std::size_t>(length)
            );
        }
        auto raw = pybind11::bytes(
            reinterpret_cast<const char*>(slab.data_ptr<std::uint8_t>()),
            static_cast<pybind11::ssize_t>(length)
        );
        sink.attr("write")(raw);
        ++chunk_count;
    }
    return chunk_count;
}

SlabPool::SlabPool(const std::int64_t chunk_bytes, const bool pinned)
    : chunk_bytes_(chunk_bytes), pinned_(pinned) {
    TORCH_CHECK(chunk_bytes > 0, "native checkpoint slab chunk_bytes must be positive");
    auto options = torch::TensorOptions()
        .dtype(torch::kUInt8)
        .device(torch::kCPU);
    if (pinned) {
        options = options.pinned_memory(true);
    }
    slabs_[0] = torch::empty({chunk_bytes}, options);
    slabs_[1] = torch::empty({chunk_bytes}, options);
    if (pinned) {
        TORCH_CHECK(
            slabs_[0].is_pinned() && slabs_[1].is_pinned(),
            "native checkpoint capture requires two pinned slabs"
        );
    }
}

std::uint64_t SlabPool::copy_tensor_chunks(
    torch::Tensor tensor,
    pybind11::object sink
) {
    TORCH_CHECK(tensor.defined(), "native checkpoint tensor must be defined");
    TORCH_CHECK(
        sink && pybind11::hasattr(sink, "write"),
        "native checkpoint sink must expose write"
    );
    auto source = tensor.detach();
    if (!source.is_contiguous()) {
        source = source.contiguous();
    }
    if (source.device().is_cuda() && !pinned_) {
        TORCH_CHECK(
            false,
            "native checkpoint CUDA capture requires pinned slab ownership"
        );
    }
    source_device_ = source.device().str();
    auto bytes = source.view(torch::kUInt8).reshape({-1});
    const auto total_bytes = bytes.numel();
    if (total_bytes == 0) {
        ++field_count_;
        return 0;
    }

    ++field_count_;
    if (source.device().is_cuda()) {
        // The extension is intentionally built as ordinary C++ and links no
        // private CUDA runtime ABI.  Torch's public Python CUDA event objects
        // still bind the exact current stream and provide slab-local waits.
        auto torch_cuda = pybind11::module_::import("torch").attr("cuda");
        auto stream = torch_cuda.attr("current_stream")(
            pybind11::arg("device") = source.device().str()
        );
        std::array<pybind11::object, 2> ready_events{
            torch_cuda.attr("Event")(),
            torch_cuda.attr("Event")(),
        };
        std::array<bool, 2> pending{false, false};
        std::array<std::int64_t, 2> pending_lengths{0, 0};

        const auto flush_slab = [&](const std::size_t slab_index) {
            if (!pending[slab_index]) {
                return;
            }
            ready_events[slab_index].attr("synchronize")();
            ++cuda_event_wait_count_;
            const auto length = pending_lengths[slab_index];
            auto& slab = slabs_[slab_index];
            auto raw = pybind11::bytes(
                reinterpret_cast<const char*>(slab.data_ptr<std::uint8_t>()),
                static_cast<pybind11::ssize_t>(length)
            );
            sink.attr("write")(raw);
            pending[slab_index] = false;
            pending_lengths[slab_index] = 0;
            last_slab_index_ = slab_index;
            ++chunk_count_;
            ++copy_operations_;
            copied_bytes_ += static_cast<std::uint64_t>(length);
        };

        std::size_t most_recent_slab = 0;
        for (std::int64_t offset = 0; offset < total_bytes; offset += chunk_bytes_) {
            const auto length = std::min<std::int64_t>(
                chunk_bytes_,
                total_bytes - offset
            );
            const auto slab_index = static_cast<std::size_t>(
                next_slab_ % slabs_.size()
            );
            // The slab's prior file write completed before it can be reused.
            TORCH_CHECK(
                !pending[slab_index],
                "native checkpoint attempted to reuse a pending slab"
            );
            auto source_slice = bytes.slice(0, offset, offset + length);
            slabs_[slab_index]
                .slice(0, 0, length)
                .copy_(source_slice, true);
            ready_events[slab_index].attr("record")(stream);
            pending[slab_index] = true;
            pending_lengths[slab_index] = length;
            most_recent_slab = slab_index;
            next_slab_ = (next_slab_ + 1) % slabs_.size();

            // Once the current D2H is queued, the preceding slab can be
            // written while that transfer progresses.  The event wait is
            // slab-local; it never synchronizes the producer/device stream.
            const auto preceding = (slab_index + 1) % slabs_.size();
            flush_slab(preceding);
        }
        flush_slab(most_recent_slab);
        ++final_cuda_fence_count_;
    } else {
        for (std::int64_t offset = 0; offset < total_bytes; offset += chunk_bytes_) {
            const auto length = std::min<std::int64_t>(
                chunk_bytes_,
                total_bytes - offset
            );
            const auto slab_index = static_cast<std::size_t>(
                next_slab_ % slabs_.size()
            );
            auto& slab = slabs_[slab_index];
            auto source_slice = bytes.slice(0, offset, offset + length);
            std::memcpy(
                slab.data_ptr<std::uint8_t>(),
                source_slice.data_ptr<std::uint8_t>(),
                static_cast<std::size_t>(length)
            );
            auto raw = pybind11::bytes(
                reinterpret_cast<const char*>(slab.data_ptr<std::uint8_t>()),
                static_cast<pybind11::ssize_t>(length)
            );
            sink.attr("write")(raw);
            last_slab_index_ = slab_index;
            next_slab_ = (next_slab_ + 1) % slabs_.size();
            ++chunk_count_;
            ++copy_operations_;
            copied_bytes_ += static_cast<std::uint64_t>(length);
        }
    }
    return (total_bytes + chunk_bytes_ - 1) / chunk_bytes_;
}

pybind11::dict SlabPool::telemetry() const {
    pybind11::dict out;
    out["schema_version"] = "eiel_acs_native_slab_pool_v1";
    out["chunk_bytes"] = chunk_bytes_;
    out["slab_count"] = static_cast<std::int64_t>(slabs_.size());
    out["pinned"] = pinned_;
    out["preallocated"] = true;
    out["chunk_count"] = static_cast<std::int64_t>(chunk_count_);
    out["copy_operations"] = static_cast<std::int64_t>(copy_operations_);
    out["copied_bytes"] = static_cast<std::int64_t>(copied_bytes_);
    out["field_count"] = static_cast<std::int64_t>(field_count_);
    out["last_slab_index"] = static_cast<std::int64_t>(last_slab_index_);
    out["cuda_event_wait_count"] = static_cast<std::int64_t>(cuda_event_wait_count_);
    out["final_cuda_fence_count"] = static_cast<std::int64_t>(final_cuda_fence_count_);
    out["per_chunk_stream_synchronizations"] = static_cast<std::int64_t>(0);
    out["source_device"] = source_device_;
    return out;
}

}  // namespace eiel::native::acs_native_streamed_checkpoint_v1
