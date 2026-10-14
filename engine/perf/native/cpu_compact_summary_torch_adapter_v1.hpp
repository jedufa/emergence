#pragma once

#include "cpu_compact_summary_digests_v1.hpp"

#include <torch/types.h>

namespace eiel::native::cpu_compact_summary_torch_adapter_v1 {

[[nodiscard]] cpu_compact_summary_digests_v1::TensorViewV1 tensor_view(
    const torch::Tensor& tensor
);

}  // namespace eiel::native::cpu_compact_summary_torch_adapter_v1
