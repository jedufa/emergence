#include "cpu_compact_summary_torch_adapter_v1.hpp"

#include <c10/util/Exception.h>

#include <cstdint>
#include <limits>

namespace eiel::native::cpu_compact_summary_torch_adapter_v1 {
namespace {

namespace digests = cpu_compact_summary_digests_v1;

struct ScalarMappingV1 {
    digests::ScalarTypeV1 dtype;
    std::uint64_t item_bytes;
};

ScalarMappingV1 scalar_mapping(const c10::ScalarType dtype) {
    switch (dtype) {
        case torch::kBool:
            return {digests::ScalarTypeV1::kBool, 1U};
        case torch::kInt8:
            return {digests::ScalarTypeV1::kInt8, 1U};
        case torch::kUInt8:
            return {digests::ScalarTypeV1::kUInt8, 1U};
        case torch::kInt32:
            return {digests::ScalarTypeV1::kInt32, 4U};
        case torch::kFloat32:
            return {digests::ScalarTypeV1::kFloat32, 4U};
        case torch::kInt64:
            return {digests::ScalarTypeV1::kInt64, 8U};
        default:
            TORCH_CHECK(
                false,
                "cpu_compact_summary_torch_adapter_v1: unsupported dtype"
            );
    }
}

std::uint64_t checked_multiply(
    const std::uint64_t left,
    const std::uint64_t right
) {
    TORCH_CHECK(
        right == 0U
            || left <= std::numeric_limits<std::uint64_t>::max() / right,
        "cpu_compact_summary_torch_adapter_v1: logical byte count overflow"
    );
    return left * right;
}

}  // namespace

digests::TensorViewV1 tensor_view(const torch::Tensor& tensor) {
    TORCH_CHECK(
        tensor.defined(),
        "cpu_compact_summary_torch_adapter_v1: tensor must be defined"
    );
    TORCH_CHECK(
        tensor.device().is_cpu(),
        "cpu_compact_summary_torch_adapter_v1: tensor must be on CPU"
    );
    TORCH_CHECK(
        tensor.layout() == torch::kStrided,
        "cpu_compact_summary_torch_adapter_v1: tensor must use strided layout"
    );
    TORCH_CHECK(
        !tensor.is_conj() && !tensor.is_neg(),
        "cpu_compact_summary_torch_adapter_v1: lazy conjugate or negative views "
        "are unsupported"
    );

    const ScalarMappingV1 mapping = scalar_mapping(tensor.scalar_type());
    const std::int64_t rank = tensor.dim();
    TORCH_CHECK(
        rank == 1 || rank == 2,
        "cpu_compact_summary_torch_adapter_v1: tensor rank must be 1 or 2"
    );

    const std::int64_t extent0 = tensor.size(0);
    const std::int64_t extent1 = rank == 2 ? tensor.size(1) : 0;
    TORCH_CHECK(
        extent0 >= 0 && extent1 >= 0,
        "cpu_compact_summary_torch_adapter_v1: tensor extents must be nonnegative"
    );

    const std::uint64_t elements = checked_multiply(
        static_cast<std::uint64_t>(extent0),
        rank == 2 ? static_cast<std::uint64_t>(extent1) : 1U
    );
    const std::uint64_t logical_bytes = checked_multiply(
        elements, mapping.item_bytes
    );
    const std::int64_t tensor_item_bytes = tensor.element_size();
    TORCH_CHECK(
        tensor_item_bytes > 0
            && static_cast<std::uint64_t>(tensor_item_bytes) == mapping.item_bytes,
        "cpu_compact_summary_torch_adapter_v1: scalar byte width mismatch"
    );
    TORCH_CHECK(
        tensor.numel() >= 0
            && static_cast<std::uint64_t>(tensor.numel()) == elements,
        "cpu_compact_summary_torch_adapter_v1: tensor shape/numel mismatch"
    );
    TORCH_CHECK(
        tensor.is_contiguous(),
        "cpu_compact_summary_torch_adapter_v1: tensor must be contiguous"
    );

    return {
        tensor.data_ptr(),
        mapping.dtype,
        static_cast<std::int32_t>(rank),
        extent0,
        extent1,
        logical_bytes,
        true,
    };
}

}  // namespace eiel::native::cpu_compact_summary_torch_adapter_v1
