#include "cpu_compact_summary_torch_adapter_v1.hpp"

#include <torch/extension.h>

#include <cstdint>

namespace py = pybind11;
namespace adapter = eiel::native::cpu_compact_summary_torch_adapter_v1;
namespace digests = eiel::native::cpu_compact_summary_digests_v1;

namespace {

const char* scalar_type_name(const digests::ScalarTypeV1 dtype) {
    switch (dtype) {
        case digests::ScalarTypeV1::kBool:
            return "bool";
        case digests::ScalarTypeV1::kInt8:
            return "int8";
        case digests::ScalarTypeV1::kUInt8:
            return "uint8";
        case digests::ScalarTypeV1::kInt32:
            return "int32";
        case digests::ScalarTypeV1::kFloat32:
            return "float32";
        case digests::ScalarTypeV1::kInt64:
            return "int64";
        case digests::ScalarTypeV1::kUnknown:
            return "unknown";
    }
    return "unknown";
}

py::dict inspect(const torch::Tensor& tensor) {
    const digests::TensorViewV1 view = adapter::tensor_view(tensor);
    py::dict output;
    output["data_ptr"] = reinterpret_cast<std::uintptr_t>(view.data);
    output["dtype"] = scalar_type_name(view.dtype);
    output["dtype_code"] = static_cast<std::uint8_t>(view.dtype);
    output["rank"] = view.rank;
    output["extent0"] = view.extent0;
    output["extent1"] = view.extent1;
    output["storage_bytes"] = view.storage_bytes;
    output["c_contiguous"] = view.c_contiguous;
    return output;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("inspect", &inspect);
}
