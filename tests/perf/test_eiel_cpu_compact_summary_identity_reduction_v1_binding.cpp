#include "cpu_compact_summary_identity_reduction_v1.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cstddef>
#include <cstdint>
#include <string>

namespace py = pybind11;
namespace reduction =
    eiel::native::cpu_compact_summary_identity_reduction_v1;
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

reduction::CheckedInt64V1 checked_int64(const py::handle value) {
    if (!PyLong_Check(value.ptr())) {
        throw py::type_error("identity scalar must be an integer");
    }
    int overflow = 0;
    const long long parsed = PyLong_AsLongLongAndOverflow(
        value.ptr(), &overflow
    );
    if (PyErr_Occurred() != nullptr) {
        throw py::error_already_set();
    }
    if (overflow != 0) {
        return {0, false};
    }
    return {static_cast<std::int64_t>(parsed), true};
}

bool is_cleared(const reduction::IdentityReductionResultV1& value) {
    bool cleared = value.active_agent_count == 0
        && value.active_agent_id_min == 0
        && value.active_agent_id_max == 0
        && value.free_slot_count == 0
        && value.population_next_id == 0
        && value.active_agent_id_min_defined == 0U
        && value.active_agent_id_max_defined == 0U;
    for (const std::uint8_t byte : value.active_agent_ids_digest.bytes) {
        cleared = cleared && byte == 0U;
    }
    for (const std::uint8_t byte : value.reserved) {
        cleared = cleared && byte == 0U;
    }
    return cleared;
}

py::dict result(
    const reduction::IdentityReductionStatusV1 status,
    const reduction::IdentityReductionResultV1& value
) {
    constexpr char kHex[] = "0123456789abcdef";
    std::string hex(value.active_agent_ids_digest.bytes.size() * 2U, '0');
    for (
        std::size_t index = 0U;
        index < value.active_agent_ids_digest.bytes.size();
        ++index
    ) {
        const std::uint8_t byte = value.active_agent_ids_digest.bytes[index];
        hex[index * 2U] = kHex[byte >> 4U];
        hex[index * 2U + 1U] = kHex[byte & 0x0FU];
    }
    py::dict output;
    output["status"] = reduction::identity_reduction_status_name(status);
    output["status_code"] = static_cast<std::uint32_t>(status);
    output["active_agent_count"] = value.active_agent_count;
    output["active_agent_id_min_defined"] =
        value.active_agent_id_min_defined == 1U;
    output["active_agent_id_min"] = value.active_agent_id_min;
    output["active_agent_id_max_defined"] =
        value.active_agent_id_max_defined == 1U;
    output["active_agent_id_max"] = value.active_agent_id_max;
    output["free_slot_count"] = value.free_slot_count;
    output["population_next_id"] = value.population_next_id;
    output["active_agent_ids_digest"] = py::bytes(
        reinterpret_cast<const char*>(
            value.active_agent_ids_digest.bytes.data()
        ),
        static_cast<py::ssize_t>(value.active_agent_ids_digest.bytes.size())
    );
    output["active_agent_ids_digest_hex"] = hex;
    output["is_cleared"] = is_cleared(value);
    return output;
}

py::dict identity(
    const py::array& active_agent_ids,
    const py::handle capacity,
    const py::handle population_next_id
) {
    reduction::IdentityReductionResultV1 output{};
    output.active_agent_count = 91;
    output.active_agent_id_min = 92;
    output.active_agent_id_max = 93;
    output.free_slot_count = 94;
    output.population_next_id = 95;
    output.active_agent_id_min_defined = 1U;
    output.active_agent_id_max_defined = 1U;
    output.active_agent_ids_digest.bytes.fill(0xA5U);
    output.reserved.fill(0xA5U);
    const reduction::IdentityReductionStatusV1 status =
        reduction::reduce_identity(
            tensor_view(active_agent_ids),
            checked_int64(capacity),
            checked_int64(population_next_id),
            output
        );
    return result(status, output);
}

py::dict identity_with_storage_bytes(
    const py::array& active_agent_ids,
    const py::handle capacity,
    const py::handle population_next_id,
    const std::uint64_t storage_bytes
) {
    digests::TensorViewV1 view = tensor_view(active_agent_ids);
    view.storage_bytes = storage_bytes;
    reduction::IdentityReductionResultV1 output{};
    output.active_agent_ids_digest.bytes.fill(0xA5U);
    const reduction::IdentityReductionStatusV1 status =
        reduction::reduce_identity(
            view,
            checked_int64(capacity),
            checked_int64(population_next_id),
            output
        );
    return result(status, output);
}

py::dict validate_candidate(
    const std::int64_t capacity,
    const std::int64_t active_count,
    const std::uint8_t minimum_defined,
    const std::int64_t minimum,
    const std::uint8_t maximum_defined,
    const std::int64_t maximum,
    const std::int64_t free_slots,
    const std::int64_t population_next_id
) {
    reduction::IdentityReductionResultV1 candidate{};
    candidate.active_agent_count = active_count;
    candidate.active_agent_id_min_defined = minimum_defined;
    candidate.active_agent_id_min = minimum;
    candidate.active_agent_id_max_defined = maximum_defined;
    candidate.active_agent_id_max = maximum;
    candidate.free_slot_count = free_slots;
    candidate.population_next_id = population_next_id;
    candidate.active_agent_ids_digest.bytes.fill(0xA5U);
    candidate.reserved.fill(0xA5U);
    const reduction::IdentityReductionStatusV1 status =
        reduction::validate_identity_reduction_result(capacity, candidate);
    return result(status, candidate);
}

}  // namespace

PYBIND11_MODULE(
    eiel_cpu_compact_summary_identity_reduction_v1_test,
    module
) {
    module.def("identity", &identity);
    module.def("identity_with_storage_bytes", &identity_with_storage_bytes);
    module.def("validate_candidate", &validate_candidate);
}
