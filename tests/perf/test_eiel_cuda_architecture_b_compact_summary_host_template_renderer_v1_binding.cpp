#include "cuda_architecture_b_compact_summary_host_template_renderer_v1.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

namespace py = pybind11;
namespace renderer =
    eiel::cuda_architecture_b::compact_summary_host_template_renderer_v1;

namespace {

std::uint32_t surface_code(const std::string& value) {
    if (value == "checkpoint_trajectory") {
        return static_cast<std::uint32_t>(
            renderer::HostTemplateSurfaceV1::kCheckpointTrajectory
        );
    }
    if (value == "canonical_checkpoint_storage") {
        return static_cast<std::uint32_t>(
            renderer::HostTemplateSurfaceV1::kCanonicalCheckpointStorage
        );
    }
    if (value == "next_generation_source_projection") {
        return static_cast<std::uint32_t>(
            renderer::HostTemplateSurfaceV1::kNextGenerationSource
        );
    }
    if (value == "rng") {
        return static_cast<std::uint32_t>(renderer::HostTemplateSurfaceV1::kRng);
    }
    if (value == "active_agent_ids") {
        return static_cast<std::uint32_t>(
            renderer::HostTemplateSurfaceV1::kActiveAgentIds
        );
    }
    if (value == "policy_mutation_tape") {
        return static_cast<std::uint32_t>(
            renderer::HostTemplateSurfaceV1::kPolicyMutationTape
        );
    }
    return UINT32_MAX;
}

std::uint32_t token_type_code(const std::string& value) {
    if (value == "integer") {
        return static_cast<std::uint32_t>(
            renderer::HostTemplateTokenTypeV1::kInteger
        );
    }
    if (value == "boolean") {
        return static_cast<std::uint32_t>(
            renderer::HostTemplateTokenTypeV1::kBoolean
        );
    }
    if (value == "pcg_uint128") {
        return static_cast<std::uint32_t>(
            renderer::HostTemplateTokenTypeV1::kPcgUint128
        );
    }
    if (value == "python_mt_word") {
        return static_cast<std::uint32_t>(
            renderer::HostTemplateTokenTypeV1::kPythonMtWord
        );
    }
    if (value == "tensor_shape") {
        return static_cast<std::uint32_t>(
            renderer::HostTemplateTokenTypeV1::kTensorShape
        );
    }
    if (value == "sha256_hex") {
        return static_cast<std::uint32_t>(
            renderer::HostTemplateTokenTypeV1::kSha256Hex
        );
    }
    if (value == "nullable_float64") {
        return static_cast<std::uint32_t>(
            renderer::HostTemplateTokenTypeV1::kNullableFloat64
        );
    }
    return UINT32_MAX;
}

std::uint32_t validation_kind_code(const std::string& value) {
    if (value == "integer_range") {
        return static_cast<std::uint32_t>(
            renderer::HostTemplateValidationKindV1::kIntegerRange
        );
    }
    if (value == "exact_boolean") {
        return static_cast<std::uint32_t>(
            renderer::HostTemplateValidationKindV1::kExactBoolean
        );
    }
    if (value == "tensor_shape") {
        return static_cast<std::uint32_t>(
            renderer::HostTemplateValidationKindV1::kTensorShape
        );
    }
    if (value == "lowercase_sha256_hex") {
        return static_cast<std::uint32_t>(
            renderer::HostTemplateValidationKindV1::kLowercaseSha256Hex
        );
    }
    if (value == "finite_optional_float64") {
        return static_cast<std::uint32_t>(
            renderer::HostTemplateValidationKindV1::kFiniteOptionalFloat64
        );
    }
    return UINT32_MAX;
}

bool parse_uint128(
    const py::handle value,
    std::uint64_t* low,
    std::uint64_t* high
) {
    const std::string text = py::str(value).cast<std::string>();
    if (text.empty() || text[0] == '-') {
        return false;
    }
    std::uint64_t words[2] = {0, 0};
    for (const char character : text) {
        if (character < '0' || character > '9') {
            return false;
        }
        const std::uint64_t digit = static_cast<std::uint64_t>(character - '0');
        const std::uint64_t high_before = words[1];
        const std::uint64_t low_before = words[0];
        // Multiplication by ten can carry more than one bit. Use four base-2^32
        // words so the decimal parser remains exact without compiler extensions.
        std::uint32_t base_words[4] = {
            static_cast<std::uint32_t>(high_before >> 32U),
            static_cast<std::uint32_t>(high_before),
            static_cast<std::uint32_t>(low_before >> 32U),
            static_cast<std::uint32_t>(low_before),
        };
        std::uint64_t propagated = digit;
        for (int index = 3; index >= 0; --index) {
            const std::uint64_t product =
                static_cast<std::uint64_t>(base_words[index]) * 10U + propagated;
            base_words[index] = static_cast<std::uint32_t>(product);
            propagated = product >> 32U;
        }
        if (propagated != 0U) {
            return false;
        }
        words[1] = (static_cast<std::uint64_t>(base_words[0]) << 32U)
            | base_words[1];
        words[0] = (static_cast<std::uint64_t>(base_words[2]) << 32U)
            | base_words[3];
    }
    *low = words[0];
    *high = words[1];
    return true;
}

std::uint32_t part_kind_code(const std::string& value) {
    if (value == "literal_utf8") {
        return static_cast<std::uint32_t>(
            renderer::HostTemplatePartKindV1::kLiteralUtf8
        );
    }
    if (value == "dynamic_token") {
        return static_cast<std::uint32_t>(
            renderer::HostTemplatePartKindV1::kDynamicToken
        );
    }
    return UINT32_MAX;
}

std::uint32_t constraint_kind_code(const std::string& value) {
    if (value == "values_equal") {
        return static_cast<std::uint32_t>(
            renderer::HostTemplateConstraintKindV1::kValuesEqual
        );
    }
    if (value == "shape_first_dimension_equals_integer") {
        return static_cast<std::uint32_t>(
            renderer::HostTemplateConstraintKindV1::
                kShapeFirstDimensionEqualsInteger
        );
    }
    return UINT32_MAX;
}

const char* status_name(const std::uint32_t status) {
    using Status = renderer::HostTemplateStatusV1;
    switch (static_cast<Status>(status)) {
        case Status::kSuccess: return "success";
        case Status::kNullArgument: return "null_argument";
        case Status::kUnsupportedAbiVersion: return "unsupported_abi_version";
        case Status::kUnsupportedEncoding: return "unsupported_encoding";
        case Status::kUnsupportedSurface: return "unsupported_surface";
        case Status::kReservedFieldNonzero: return "reserved_field_nonzero";
        case Status::kProgramBounds: return "program_bounds";
        case Status::kMalformedProgram: return "malformed_program";
        case Status::kInvalidUtf8: return "invalid_utf8";
        case Status::kUnsupportedTokenType: return "unsupported_token_type";
        case Status::kUnsupportedValidationRule: return "unsupported_validation_rule";
        case Status::kTokenDeclarationContradiction:
            return "token_declaration_contradiction";
        case Status::kTokenValueCountMismatch: return "token_value_count_mismatch";
        case Status::kTokenValueTypeMismatch: return "token_value_type_mismatch";
        case Status::kTokenValueBounds: return "token_value_bounds";
        case Status::kTokenShapeContradiction: return "token_shape_contradiction";
        case Status::kTokenSha256Contradiction: return "token_sha256_contradiction";
        case Status::kUnsupportedPartKind: return "unsupported_part_kind";
        case Status::kPartBounds: return "part_bounds";
        case Status::kUnsupportedConstraintKind:
            return "unsupported_constraint_kind";
        case Status::kConstraintPathContradiction:
            return "constraint_path_contradiction";
        case Status::kConstraintTypeContradiction:
            return "constraint_type_contradiction";
        case Status::kConstraintValueContradiction:
            return "constraint_value_contradiction";
        case Status::kRenderedBoundContradiction:
            return "rendered_bound_contradiction";
        case Status::kOutputOverrun: return "output_overrun";
        case Status::kOutputModeContradiction: return "output_mode_contradiction";
    }
    return "unknown";
}

template <typename T>
T option_or(const py::dict& options, const char* key, const T fallback) {
    return options.contains(key) ? py::cast<T>(options[key]) : fallback;
}

struct LoweredProgramV1 {
    renderer::HostTemplateProgramV1 program{};
    std::vector<renderer::HostTemplateTokenSpecV1> tokens;
    std::vector<renderer::HostTemplateTokenValueV1> values;
    std::vector<renderer::HostTemplatePartV1> parts;
    std::vector<renderer::HostTemplateConstraintV1> constraints;
    std::vector<std::string> token_paths;
    std::vector<std::string> sha256_values;
    std::vector<std::vector<std::uint64_t>> shape_values;
    std::vector<std::string> literal_values;
    std::vector<std::string> constraint_left_paths;
    std::vector<std::string> constraint_right_paths;
};

void lower_token_rule(
    const py::dict& source,
    renderer::HostTemplateTokenSpecV1* token
) {
    const py::dict rule = py::cast<py::dict>(source["validation_rule"]);
    token->validation_kind = validation_kind_code(
        py::cast<std::string>(rule["kind"])
    );
    token->rank = rule.contains("rank")
        ? py::cast<std::uint32_t>(rule["rank"])
        : 0U;
    token->render_pack_depth = rule.contains("render_pack_depth")
        ? py::cast<std::uint32_t>(rule["render_pack_depth"])
        : 0U;
    if (rule.contains("minimum")) {
        if (!parse_uint128(
                rule["minimum"],
                &token->minimum_low,
                &token->minimum_high
            )) {
            token->reserved = 1U;
        }
    }
    if (rule.contains("maximum")) {
        if (!parse_uint128(
                rule["maximum"],
                &token->maximum_low,
                &token->maximum_high
            )) {
            token->reserved = 1U;
        }
    }
    if (rule.contains("minimum_dimension")) {
        if (!parse_uint128(
                rule["minimum_dimension"],
                &token->minimum_low,
                &token->minimum_high
            )) {
            token->reserved = 1U;
        }
    }
    if (rule.contains("maximum_dimension")) {
        if (!parse_uint128(
                rule["maximum_dimension"],
                &token->maximum_low,
                &token->maximum_high
            )) {
            token->reserved = 1U;
        }
    }
    token->rule_parameter = rule.contains("length")
        ? py::cast<std::uint32_t>(rule["length"])
        : 0U;
}

LoweredProgramV1 lower_program(
    const py::dict& source,
    const py::sequence& source_values
) {
    LoweredProgramV1 lowered;
    const py::list source_tokens = py::cast<py::list>(source["tokens"]);
    const py::list source_parts = py::cast<py::list>(source["parts"]);
    const py::list source_constraints = py::cast<py::list>(source["constraints"]);
    const std::size_t token_count = py::len(source_tokens);
    const std::size_t part_count = py::len(source_parts);
    const std::size_t constraint_count = py::len(source_constraints);
    if (py::len(source_values) != token_count) {
        throw py::value_error("binding input values must match the Python template");
    }

    lowered.tokens.resize(token_count);
    lowered.values.resize(token_count);
    lowered.token_paths.resize(token_count);
    lowered.sha256_values.resize(token_count);
    lowered.shape_values.resize(token_count);
    for (std::size_t index = 0; index < token_count; ++index) {
        const py::dict raw = py::cast<py::dict>(source_tokens[index]);
        auto& token = lowered.tokens[index];
        token.token_index = py::cast<std::uint32_t>(raw["token_index"]);
        token.token_type = token_type_code(py::cast<std::string>(raw["token_type"]));
        token.maximum_utf8_bytes = py::cast<std::uint32_t>(
            raw["maximum_utf8_bytes"]
        );
        lowered.token_paths[index] = py::cast<std::string>(raw["source_path"]);
        lower_token_rule(raw, &token);

        auto& value = lowered.values[index];
        const py::handle raw_value = source_values[index];
        if (raw_value.is_none()) {
            value.token_type = static_cast<std::uint32_t>(
                renderer::HostTemplateTokenTypeV1::kNullableFloat64
            );
        } else if (py::isinstance<py::float_>(raw_value)) {
            value.token_type = static_cast<std::uint32_t>(
                renderer::HostTemplateTokenTypeV1::kNullableFloat64
            );
            const double scalar = py::cast<double>(raw_value);
            std::memcpy(&value.scalar_low, &scalar, sizeof(scalar));
            value.scalar_high = 1U;
        } else if (py::isinstance<py::bool_>(raw_value)) {
            value.token_type = static_cast<std::uint32_t>(
                renderer::HostTemplateTokenTypeV1::kBoolean
            );
            value.scalar_low = py::cast<bool>(raw_value) ? 1U : 0U;
        } else if (py::isinstance<py::int_>(raw_value)) {
            value.token_type = token.token_type;
            if (!parse_uint128(raw_value, &value.scalar_low, &value.scalar_high)) {
                value.reserved = 1U;
            }
        } else if (py::isinstance<py::list>(raw_value)) {
            value.token_type = static_cast<std::uint32_t>(
                renderer::HostTemplateTokenTypeV1::kTensorShape
            );
            const py::list shape = py::cast<py::list>(raw_value);
            auto& dimensions = lowered.shape_values[index];
            dimensions.resize(py::len(shape));
            for (std::size_t dimension = 0; dimension < dimensions.size(); ++dimension) {
                std::uint64_t high = 0;
                if (py::isinstance<py::bool_>(shape[dimension])
                    || !parse_uint128(shape[dimension], &dimensions[dimension], &high)
                    || high != 0U) {
                    value.reserved = 1U;
                }
            }
            value.shape_rank = static_cast<std::uint32_t>(dimensions.size());
        } else if (py::isinstance<py::str>(raw_value)) {
            value.token_type = static_cast<std::uint32_t>(
                renderer::HostTemplateTokenTypeV1::kSha256Hex
            );
            lowered.sha256_values[index] = py::cast<std::string>(raw_value);
            value.sha256_hex_bytes = static_cast<std::uint32_t>(
                lowered.sha256_values[index].size()
            );
        } else {
            value.token_type = UINT32_MAX;
        }
    }

    lowered.parts.resize(part_count);
    lowered.literal_values.resize(part_count);
    for (std::size_t index = 0; index < part_count; ++index) {
        const py::dict raw = py::cast<py::dict>(source_parts[index]);
        auto& part = lowered.parts[index];
        const std::string kind = py::cast<std::string>(raw["kind"]);
        part.kind = part_kind_code(kind);
        if (kind == "literal_utf8") {
            part.token_index = renderer::kNoIndexV1;
            lowered.literal_values[index] = py::cast<std::string>(raw["value"]);
            part.literal_utf8_bytes = lowered.literal_values[index].size();
        } else if (kind == "dynamic_token") {
            part.token_index = py::cast<std::uint32_t>(raw["token_index"]);
        }
    }

    lowered.constraints.resize(constraint_count);
    lowered.constraint_left_paths.resize(constraint_count);
    lowered.constraint_right_paths.resize(constraint_count);
    for (std::size_t index = 0; index < constraint_count; ++index) {
        const py::dict raw = py::cast<py::dict>(source_constraints[index]);
        auto& constraint = lowered.constraints[index];
        const std::string kind = py::cast<std::string>(raw["kind"]);
        constraint.kind = constraint_kind_code(kind);
        if (kind == "values_equal") {
            lowered.constraint_left_paths[index] = py::cast<std::string>(
                raw["left_source_path"]
            );
            lowered.constraint_right_paths[index] = py::cast<std::string>(
                raw["right_source_path"]
            );
        } else if (kind == "shape_first_dimension_equals_integer") {
            lowered.constraint_left_paths[index] = py::cast<std::string>(
                raw["shape_source_path"]
            );
            lowered.constraint_right_paths[index] = py::cast<std::string>(
                raw["integer_source_path"]
            );
        }
    }

    lowered.program.abi_version =
        py::cast<std::string>(source["schema_version"])
                == "eiel_compact_summary_host_template_v1"
            ? renderer::kHostTemplateRendererAbiVersionV1
            : 0U;
    lowered.program.encoding = py::cast<std::string>(source["encoding"]) == "utf-8"
        ? static_cast<std::uint32_t>(renderer::HostTemplateEncodingV1::kUtf8)
        : 0U;
    lowered.program.surface = surface_code(py::cast<std::string>(source["surface"]));
    lowered.program.generation = py::cast<std::int64_t>(source["generation"]);
    lowered.program.cell = py::cast<std::int64_t>(source["cell"]);
    lowered.program.token_count = py::cast<std::uint32_t>(source["token_count"]);
    lowered.program.part_count = static_cast<std::uint32_t>(part_count);
    lowered.program.constraint_count = static_cast<std::uint32_t>(constraint_count);
    lowered.program.literal_utf8_bytes = py::cast<std::uint64_t>(
        source["literal_utf8_bytes"]
    );
    lowered.program.maximum_rendered_utf8_bytes = py::cast<std::uint64_t>(
        source["maximum_rendered_utf8_bytes"]
    );
    return lowered;
}

void apply_options(LoweredProgramV1* lowered, const py::dict& options) {
    auto& program = lowered->program;
    program.abi_version = option_or<std::uint32_t>(
        options, "program_abi_version", program.abi_version
    );
    program.encoding = option_or<std::uint32_t>(options, "encoding", program.encoding);
    program.surface = option_or<std::uint32_t>(options, "surface", program.surface);
    program.generation = option_or<std::int64_t>(
        options, "generation", program.generation
    );
    program.cell = option_or<std::int64_t>(options, "cell", program.cell);
    program.literal_utf8_bytes = option_or<std::uint64_t>(
        options, "literal_utf8_bytes", program.literal_utf8_bytes
    );
    program.maximum_rendered_utf8_bytes = option_or<std::uint64_t>(
        options,
        "maximum_rendered_utf8_bytes",
        program.maximum_rendered_utf8_bytes
    );
    program.reserved0 = option_or<std::uint32_t>(options, "program_reserved", 0U);
    program.token_count = option_or<std::uint32_t>(
        options, "program_token_count", program.token_count
    );

    if (!lowered->tokens.empty()) {
        const std::size_t index = option_or<std::size_t>(options, "token_index", 0U);
        auto& token = lowered->tokens.at(index);
        token.token_type = option_or<std::uint32_t>(
            options, "token_type", token.token_type
        );
        token.validation_kind = option_or<std::uint32_t>(
            options, "validation_kind", token.validation_kind
        );
        token.rank = option_or<std::uint32_t>(options, "token_rank", token.rank);
        token.render_pack_depth = option_or<std::uint32_t>(
            options, "pack_depth", token.render_pack_depth
        );
        token.maximum_utf8_bytes = option_or<std::uint32_t>(
            options, "token_maximum_utf8_bytes", token.maximum_utf8_bytes
        );
        token.reserved = option_or<std::uint64_t>(
            options, "token_reserved", token.reserved
        );
        if (options.contains("duplicate_source_path_with")) {
            lowered->token_paths[index] = lowered->token_paths.at(
                py::cast<std::size_t>(options["duplicate_source_path_with"])
            );
        }

        const std::size_t value_index = option_or<std::size_t>(
            options, "value_index", index
        );
        auto& value = lowered->values.at(value_index);
        value.token_type = option_or<std::uint32_t>(
            options, "value_type", value.token_type
        );
        value.scalar_low = option_or<std::uint64_t>(
            options, "scalar_low", value.scalar_low
        );
        value.scalar_high = option_or<std::uint64_t>(
            options, "scalar_high", value.scalar_high
        );
        value.shape_rank = option_or<std::uint32_t>(
            options, "shape_rank", value.shape_rank
        );
        value.sha256_hex_bytes = option_or<std::uint32_t>(
            options, "sha256_hex_bytes", value.sha256_hex_bytes
        );
        value.reserved = option_or<std::uint32_t>(
            options, "value_reserved", value.reserved
        );
    }

    if (!lowered->parts.empty()) {
        const std::size_t index = option_or<std::size_t>(options, "part_index", 0U);
        auto& part = lowered->parts.at(index);
        part.kind = option_or<std::uint32_t>(options, "part_kind", part.kind);
        part.token_index = option_or<std::uint32_t>(
            options, "part_token_index", part.token_index
        );
        part.reserved = option_or<std::uint64_t>(
            options, "part_reserved", part.reserved
        );
        if (option_or<bool>(options, "invalid_literal_utf8", false)) {
            lowered->literal_values[index] = std::string("\xc0\xaf", 2);
            part.literal_utf8_bytes = 2U;
        }
    }

    if (!lowered->constraints.empty()) {
        const std::size_t index = option_or<std::size_t>(
            options, "constraint_index", 0U
        );
        auto& constraint = lowered->constraints.at(index);
        constraint.kind = option_or<std::uint32_t>(
            options, "constraint_kind", constraint.kind
        );
        constraint.reserved0 = option_or<std::uint32_t>(
            options, "constraint_reserved", constraint.reserved0
        );
        if (options.contains("constraint_left_path")) {
            lowered->constraint_left_paths[index] = py::cast<std::string>(
                options["constraint_left_path"]
            );
        }
        if (options.contains("constraint_right_path")) {
            lowered->constraint_right_paths[index] = py::cast<std::string>(
                options["constraint_right_path"]
            );
        }
    }
}

void bind_pointers(LoweredProgramV1* lowered) {
    for (std::size_t index = 0; index < lowered->tokens.size(); ++index) {
        auto& token = lowered->tokens[index];
        token.source_path_utf8 = reinterpret_cast<const std::uint8_t*>(
            lowered->token_paths[index].data()
        );
        token.source_path_utf8_bytes = static_cast<std::uint32_t>(
            lowered->token_paths[index].size()
        );
        auto& value = lowered->values[index];
        value.shape_dimensions = lowered->shape_values[index].empty()
            ? nullptr
            : lowered->shape_values[index].data();
        value.sha256_hex = lowered->sha256_values[index].empty()
            ? nullptr
            : reinterpret_cast<const std::uint8_t*>(
                lowered->sha256_values[index].data()
            );
    }
    for (std::size_t index = 0; index < lowered->parts.size(); ++index) {
        auto& part = lowered->parts[index];
        part.literal_utf8 = lowered->literal_values[index].empty()
            ? nullptr
            : reinterpret_cast<const std::uint8_t*>(
                lowered->literal_values[index].data()
            );
        if (part.kind == static_cast<std::uint32_t>(
                renderer::HostTemplatePartKindV1::kLiteralUtf8
            )) {
            part.literal_utf8_bytes = lowered->literal_values[index].size();
        }
    }
    for (std::size_t index = 0; index < lowered->constraints.size(); ++index) {
        auto& constraint = lowered->constraints[index];
        constraint.left_source_path_utf8 = reinterpret_cast<const std::uint8_t*>(
            lowered->constraint_left_paths[index].data()
        );
        constraint.left_source_path_utf8_bytes = static_cast<std::uint32_t>(
            lowered->constraint_left_paths[index].size()
        );
        constraint.right_source_path_utf8 = reinterpret_cast<const std::uint8_t*>(
            lowered->constraint_right_paths[index].data()
        );
        constraint.right_source_path_utf8_bytes = static_cast<std::uint32_t>(
            lowered->constraint_right_paths[index].size()
        );
    }
    lowered->program.tokens = lowered->tokens.empty() ? nullptr : lowered->tokens.data();
    lowered->program.parts = lowered->parts.empty() ? nullptr : lowered->parts.data();
    lowered->program.constraints = lowered->constraints.empty()
        ? nullptr
        : lowered->constraints.data();
}

py::dict render_template(
    const py::dict& template_program,
    const py::sequence& values,
    const py::dict& options,
    const bool callback
) {
    LoweredProgramV1 lowered = lower_program(template_program, values);
    apply_options(&lowered, options);
    bind_pointers(&lowered);

    const std::uint32_t mode = option_or<std::uint32_t>(
        options,
        "mode",
        static_cast<std::uint32_t>(
            renderer::HostTemplateRenderModeV1::kRenderAndHash
        )
    );
    const std::uint64_t output_capacity = option_or<std::uint64_t>(
        options,
        "output_capacity",
        lowered.program.maximum_rendered_utf8_bytes
    );
    std::vector<std::uint8_t> output;
    if (mode == static_cast<std::uint32_t>(
            renderer::HostTemplateRenderModeV1::kRenderAndHash
        )) {
        output.resize(static_cast<std::size_t>(output_capacity));
    }
    renderer::HostTemplateResultV1 result{};
    renderer::HostTemplateRequestV1 request{
        option_or<std::uint32_t>(
            options,
            "request_abi_version",
            renderer::kHostTemplateRendererAbiVersionV1
        ),
        mode,
        &lowered.program,
        lowered.values.empty() ? nullptr : lowered.values.data(),
        option_or<std::uint32_t>(
            options,
            "token_value_count",
            static_cast<std::uint32_t>(lowered.values.size())
        ),
        option_or<std::uint32_t>(options, "request_reserved", 0U),
        output.empty() ? nullptr : output.data(),
        output.empty() ? 0U : output_capacity,
        &result,
        option_or<std::uint64_t>(options, "request_reserved_tail", 0U),
    };
    if (callback) {
        renderer::render_and_hash_host_template_callback_v1(&request);
    } else {
        static_cast<void>(renderer::render_and_hash_host_template_v1(&request));
    }

    py::dict response;
    response["status"] = result.status;
    response["status_name"] = status_name(result.status);
    response["failed_component"] = result.failed_component;
    response["failed_index"] = result.failed_index;
    response["rendered_utf8_bytes"] = result.rendered_utf8_bytes;
    response["digest"] = py::bytes(
        reinterpret_cast<const char*>(result.digest_sha256),
        renderer::kDigestBytesV1
    );
    response["rendered"] = result.status == 0U && !output.empty()
        ? py::bytes(
            reinterpret_cast<const char*>(output.data()),
            static_cast<py::ssize_t>(result.rendered_utf8_bytes)
        )
        : py::bytes();
    return response;
}

py::dict abi_layout() {
    py::dict result;
    result["abi_version"] = renderer::kHostTemplateRendererAbiVersionV1;
    result["token_spec_bytes"] = sizeof(renderer::HostTemplateTokenSpecV1);
    result["token_value_bytes"] = sizeof(renderer::HostTemplateTokenValueV1);
    result["part_bytes"] = sizeof(renderer::HostTemplatePartV1);
    result["constraint_bytes"] = sizeof(renderer::HostTemplateConstraintV1);
    result["program_bytes"] = sizeof(renderer::HostTemplateProgramV1);
    result["request_bytes"] = sizeof(renderer::HostTemplateRequestV1);
    result["result_bytes"] = sizeof(renderer::HostTemplateResultV1);
    return result;
}

}  // namespace

PYBIND11_MODULE(
    cuda_architecture_b_compact_summary_host_template_renderer_v1_probe,
    module
) {
    module.def(
        "render",
        &render_template,
        py::arg("template_program"),
        py::arg("values"),
        py::arg("options") = py::dict(),
        py::arg("callback") = false
    );
    module.def("abi_layout", &abi_layout);
}
