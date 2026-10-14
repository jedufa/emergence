#include "cpu_compact_summary_digests_v1.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>

namespace py = pybind11;
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
    if (dtype.is(py::dtype::of<double>())) {
        return digests::ScalarTypeV1::kFloat64;
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

py::dict result(
    const digests::DigestStatusV1 status,
    const digests::DigestV1& digest
) {
    constexpr char kHex[] = "0123456789abcdef";
    std::string hex(digest.bytes.size() * 2U, '0');
    bool zero = true;
    for (std::size_t index = 0U; index < digest.bytes.size(); ++index) {
        const std::uint8_t value = digest.bytes[index];
        zero = zero && value == 0U;
        hex[index * 2U] = kHex[value >> 4U];
        hex[index * 2U + 1U] = kHex[value & 0x0FU];
    }
    py::dict output;
    output["status"] = digests::digest_status_name(status);
    output["status_code"] = static_cast<std::uint32_t>(status);
    output["digest"] = py::bytes(
        reinterpret_cast<const char*>(digest.bytes.data()),
        static_cast<py::ssize_t>(digest.bytes.size())
    );
    output["hex"] = hex;
    output["is_zero"] = zero;
    return output;
}

std::string digest_hex(const digests::DigestV1& digest) {
    constexpr char kHex[] = "0123456789abcdef";
    std::string hex(digest.bytes.size() * 2U, '0');
    for (std::size_t index = 0U; index < digest.bytes.size(); ++index) {
        const std::uint8_t value = digest.bytes[index];
        hex[index * 2U] = kHex[value >> 4U];
        hex[index * 2U + 1U] = kHex[value & 0x0FU];
    }
    return hex;
}

bool digest_is_zero(const digests::DigestV1& digest) {
    for (const std::uint8_t value : digest.bytes) {
        if (value != 0U) {
            return false;
        }
    }
    return true;
}

py::bytes digest_bytes(const digests::DigestV1& digest) {
    return py::bytes(
        reinterpret_cast<const char*>(digest.bytes.data()),
        static_cast<py::ssize_t>(digest.bytes.size())
    );
}

py::dict checkpoint_tensor_descriptor(
    const py::array& tensor,
    const std::int32_t mode,
    const py::object& alive_mask,
    const std::int64_t active_length,
    const py::object& extent0_override,
    const py::object& extent1_override,
    const py::object& storage_bytes_override,
    const bool null_data
) {
    digests::TensorViewV1 tensor_metadata = tensor_view(tensor);
    if (!extent0_override.is_none()) {
        tensor_metadata.extent0 = py::cast<std::int64_t>(extent0_override);
    }
    if (!extent1_override.is_none()) {
        tensor_metadata.extent1 = py::cast<std::int64_t>(extent1_override);
    }
    if (!storage_bytes_override.is_none()) {
        tensor_metadata.storage_bytes = py::cast<std::uint64_t>(
            storage_bytes_override
        );
    }
    if (null_data) {
        tensor_metadata.data = nullptr;
    }

    py::array mask_array;
    digests::TensorViewV1 mask_metadata{};
    const digests::TensorViewV1* mask_pointer = nullptr;
    if (!alive_mask.is_none()) {
        mask_array = py::cast<py::array>(alive_mask);
        mask_metadata = tensor_view(mask_array);
        mask_pointer = &mask_metadata;
    }
    const digests::CheckpointTensorDigestDescriptorV1 descriptor{
        tensor_metadata,
        static_cast<digests::TensorCanonicalizationModeV1>(mode),
        mask_pointer,
        active_length,
    };
    digests::CheckpointTensorDigestResultV1 output{};
    output.raw.bytes.fill(0xA5U);
    output.canonical.bytes.fill(0x5AU);
    output.status = digests::DigestStatusV1::kBooleanValueInvalid;
    output.valid = 0x7FU;
    const digests::DigestStatusV1 returned_status =
        digests::checkpoint_tensor_descriptor_digests(descriptor, output);

    py::dict result;
    result["status"] = digests::digest_status_name(output.status);
    result["status_code"] = static_cast<std::uint32_t>(output.status);
    result["returned_status_code"] = static_cast<std::uint32_t>(returned_status);
    result["valid"] = output.valid != 0U;
    result["raw_digest"] = digest_bytes(output.raw);
    result["raw_hex"] = digest_hex(output.raw);
    result["raw_is_zero"] = digest_is_zero(output.raw);
    result["canonical_digest"] = digest_bytes(output.canonical);
    result["canonical_hex"] = digest_hex(output.canonical);
    result["canonical_is_zero"] = digest_is_zero(output.canonical);
    return result;
}

py::dict active_agent_ids(const py::array& ids) {
    digests::DigestV1 digest{};
    digest.bytes.fill(0xA5U);
    const digests::DigestStatusV1 status = digests::active_agent_ids_digest(
        tensor_view(ids), digest
    );
    return result(status, digest);
}

py::dict lifecycle(const py::dict& mapping, const std::int64_t active_count) {
    if (py::len(mapping) != 16) {
        throw py::value_error("lifecycle mapping must contain exactly 16 fields");
    }
    digests::TypedLifecycleEventsViewV1 events{
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
        active_count,
    };
    digests::DigestV1 digest{};
    digest.bytes.fill(0xA5U);
    const digests::DigestStatusV1 status =
        digests::typed_lifecycle_active_prefix_digest(events, digest);
    return result(status, digest);
}

std::int64_t policy_cursor(const py::handle value) {
    if (py::isinstance<py::int_>(value)) {
        return py::cast<std::int64_t>(value);
    }
    const py::array array = py::cast<py::array>(value);
    if (!array.dtype().is(py::dtype::of<std::int64_t>())
        || array.ndim() != 1
        || array.shape(0) != 1
        || (array.flags() & py::array::c_style) == 0) {
        throw py::value_error("policy_birth_tape_cursor must be int or contiguous int64[1]");
    }
    return *static_cast<const std::int64_t*>(array.data());
}

py::dict policy_tape(const py::dict& mapping) {
    if (py::len(mapping) != 6) {
        throw py::value_error("policy tape mapping must contain exactly 6 fields");
    }
    const py::str cursor_key("policy_birth_tape_cursor");
    if (!mapping.contains(cursor_key)) {
        throw py::key_error("missing field: policy_birth_tape_cursor");
    }
    const bool post_state_is_packed =
        mapping.contains(py::str("policy_birth_post_rng_state_packed_tape"));
    digests::PolicyMutationTapeViewV1 tape{
        tensor_view(required_array(mapping, "policy_birth_crossover_mask_tape")),
        tensor_view(required_array(mapping, "policy_birth_delta_tape")),
        tensor_view(required_array(mapping, "policy_birth_mutation_mask_tape")),
        tensor_view(
            required_array(
                mapping,
                post_state_is_packed
                    ? "policy_birth_post_rng_state_packed_tape"
                    : "policy_birth_post_rng_state_tape"
            )
        ),
        tensor_view(required_array(mapping, "policy_birth_semantic_draws_tape")),
        policy_cursor(mapping[cursor_key]),
        post_state_is_packed,
    };
    digests::DigestV1 digest{};
    digest.bytes.fill(0xA5U);
    const digests::DigestStatusV1 status =
        digests::policy_mutation_tape_consumed_prefix_digest(tape, digest);
    return result(status, digest);
}

}  // namespace

PYBIND11_MODULE(eiel_cpu_compact_summary_digests_v1_test, module) {
    module.def(
        "checkpoint_tensor_descriptor",
        &checkpoint_tensor_descriptor,
        py::arg("tensor"),
        py::arg("mode"),
        py::arg("alive_mask") = py::none(),
        py::arg("active_length") = 0,
        py::arg("extent0_override") = py::none(),
        py::arg("extent1_override") = py::none(),
        py::arg("storage_bytes_override") = py::none(),
        py::arg("null_data") = false
    );
    module.def("active_agent_ids", &active_agent_ids);
    module.def("lifecycle", &lifecycle);
    module.def("policy_tape", &policy_tape);
    module.attr("mode_exact") = static_cast<std::int32_t>(
        digests::TensorCanonicalizationModeV1::kExactStateRawBytes
    );
    module.attr("mode_dead_row_zero") = static_cast<std::int32_t>(
        digests::TensorCanonicalizationModeV1::kLiveRowsDeadZero
    );
    module.attr("mode_active_prefix") = static_cast<std::int32_t>(
        digests::TensorCanonicalizationModeV1::kActivePrefix
    );
}
