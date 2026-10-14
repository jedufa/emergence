#include "cpu_compact_summary_lifecycle_reduction_v1.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>

namespace py = pybind11;
namespace reduction =
    eiel::native::cpu_compact_summary_lifecycle_reduction_v1;
namespace digests = eiel::native::cpu_compact_summary_digests_v1;

namespace {

digests::ScalarTypeV1 scalar_type(const py::array& array) {
    const py::dtype dtype = array.dtype();
    if (dtype.is(py::dtype::of<bool>())) {
        return digests::ScalarTypeV1::kBool;
    }
    if (dtype.is(py::dtype::of<std::int8_t>())) {
        return digests::ScalarTypeV1::kInt8;
    }
    if (dtype.is(py::dtype::of<std::uint8_t>())) {
        return digests::ScalarTypeV1::kUInt8;
    }
    if (dtype.is(py::dtype::of<std::int32_t>())) {
        return digests::ScalarTypeV1::kInt32;
    }
    if (dtype.is(py::dtype::of<float>())) {
        return digests::ScalarTypeV1::kFloat32;
    }
    if (dtype.is(py::dtype::of<std::int64_t>())) {
        return digests::ScalarTypeV1::kInt64;
    }
    return digests::ScalarTypeV1::kUnknown;
}

digests::TensorViewV1 tensor_view(const py::array& array) {
    return {
        array.size() == 0 ? nullptr : array.data(),
        scalar_type(array),
        static_cast<std::int32_t>(array.ndim()),
        array.ndim() >= 1 ? static_cast<std::int64_t>(array.shape(0)) : 0,
        array.ndim() >= 2 ? static_cast<std::int64_t>(array.shape(1)) : 0,
        static_cast<std::uint64_t>(array.nbytes()),
        (array.flags() & py::array::c_style) != 0,
    };
}

py::array required_array(const py::dict& mapping, const char* const name) {
    const py::str key(name);
    if (!mapping.contains(key)) {
        throw py::key_error(std::string("missing field: ") + name);
    }
    return py::cast<py::array>(mapping[key]);
}

std::string digest_hex(const digests::DigestV1& digest) {
    constexpr char kHex[] = "0123456789abcdef";
    std::string output(digest.bytes.size() * 2U, '0');
    for (std::size_t index = 0U; index < digest.bytes.size(); ++index) {
        output[index * 2U] = kHex[digest.bytes[index] >> 4U];
        output[index * 2U + 1U] = kHex[digest.bytes[index] & 0x0FU];
    }
    return output;
}

py::tuple counts_tuple(const reduction::EventTypeCountsV1& counts) {
    py::tuple output(counts.size());
    for (std::size_t index = 0U; index < counts.size(); ++index) {
        output[index] = counts[index];
    }
    return output;
}

bool output_is_clear(const reduction::LifecycleReductionV1& output) {
    const auto all_zero = [](const auto& values) {
        return std::all_of(values.begin(), values.end(), [](const auto value) {
            return value == 0;
        });
    };
    return output.event_row_count == 0
        && output.generation_event_count == 0
        && output.turnover_event_count == 0
        && all_zero(output.generation_counts_by_type)
        && all_zero(output.turnover_counts_by_type)
        && output.deaths_energy_depleted == 0
        && output.deaths_health_depleted == 0
        && all_zero(output.active_prefix_digest.bytes);
}

py::dict reduce_lifecycle(
    const py::dict& mapping,
    const std::int64_t event_cursor_start,
    const std::int64_t event_cursor_pre_turnover,
    const std::int64_t event_cursor_end
) {
    if (py::len(mapping) != 16) {
        throw py::value_error("lifecycle mapping must contain exactly 16 fields");
    }
    const digests::TypedLifecycleEventsViewV1 events{
        tensor_view(required_array(mapping, "event_type")),
        tensor_view(required_array(mapping, "generation")),
        tensor_view(required_array(mapping, "tick")),
        tensor_view(required_array(mapping, "sequence")),
        tensor_view(required_array(mapping, "actor_agent_id")),
        tensor_view(required_array(mapping, "target_agent_id")),
        tensor_view(required_array(mapping, "actor_slot")),
        tensor_view(required_array(mapping, "target_slot")),
        tensor_view(required_array(mapping, "parent_agent_id")),
        tensor_view(required_array(mapping, "child_agent_id")),
        tensor_view(required_array(mapping, "lineage_founder_id")),
        tensor_view(required_array(mapping, "lineage_depth")),
        tensor_view(required_array(mapping, "subtype_code")),
        tensor_view(required_array(mapping, "reason_code")),
        tensor_view(required_array(mapping, "success")),
        tensor_view(required_array(mapping, "value")),
        event_cursor_end,
    };

    reduction::LifecycleReductionV1 output;
    std::memset(&output, 0xA5, sizeof(output));
    const reduction::ReductionStatusV1 status =
        reduction::reduce_typed_lifecycle_active_prefix(
            events,
            event_cursor_start,
            event_cursor_pre_turnover,
            event_cursor_end,
            output
        );

    py::dict result;
    result["status"] = reduction::reduction_status_name(status);
    result["status_code"] = static_cast<std::uint32_t>(status);
    result["event_row_count"] = output.event_row_count;
    result["generation_event_count"] = output.generation_event_count;
    result["turnover_event_count"] = output.turnover_event_count;
    result["generation_counts_by_type"] = counts_tuple(
        output.generation_counts_by_type
    );
    result["turnover_counts_by_type"] = counts_tuple(
        output.turnover_counts_by_type
    );
    result["deaths_energy_depleted"] = output.deaths_energy_depleted;
    result["deaths_health_depleted"] = output.deaths_health_depleted;
    result["digest"] = py::bytes(
        reinterpret_cast<const char*>(output.active_prefix_digest.bytes.data()),
        static_cast<py::ssize_t>(output.active_prefix_digest.bytes.size())
    );
    result["hex"] = digest_hex(output.active_prefix_digest);
    result["is_cleared"] = output_is_clear(output);
    return result;
}

}  // namespace

PYBIND11_MODULE(
    eiel_cpu_compact_summary_lifecycle_reduction_v1_test,
    module
) {
    module.def("reduce", &reduce_lifecycle);
}
