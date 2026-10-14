#include "cuda_architecture_b_compact_summary_host_template_renderer_v1.hpp"

#include <charconv>
#include <cmath>
#include <cstring>
#include <limits>

namespace eiel::cuda_architecture_b::compact_summary_host_template_renderer_v1 {
namespace {

struct Sha256ContextV1 {
    std::uint64_t total_bytes;
    std::uint32_t state[8];
    std::uint32_t buffered_bytes;
    std::uint8_t buffer[64];
};

constexpr std::uint32_t kSha256RoundConstantsV1[64] = {
    0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U,
    0x3956c25bU, 0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U,
    0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U,
    0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U,
    0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU,
    0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
    0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U,
    0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U,
    0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U,
    0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
    0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U,
    0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
    0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U,
    0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
    0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
    0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U,
};

constexpr std::uint32_t rotate_right(
    const std::uint32_t value,
    const std::uint32_t distance
) noexcept {
    return (value >> distance) | (value << (32U - distance));
}

std::uint32_t load_big_endian_u32(const std::uint8_t* bytes) noexcept {
    return (static_cast<std::uint32_t>(bytes[0]) << 24U)
        | (static_cast<std::uint32_t>(bytes[1]) << 16U)
        | (static_cast<std::uint32_t>(bytes[2]) << 8U)
        | static_cast<std::uint32_t>(bytes[3]);
}

void store_big_endian_u32(
    std::uint8_t* bytes,
    const std::uint32_t value
) noexcept {
    bytes[0] = static_cast<std::uint8_t>(value >> 24U);
    bytes[1] = static_cast<std::uint8_t>(value >> 16U);
    bytes[2] = static_cast<std::uint8_t>(value >> 8U);
    bytes[3] = static_cast<std::uint8_t>(value);
}

void sha256_transform(
    Sha256ContextV1* context,
    const std::uint8_t* block
) noexcept {
    std::uint32_t words[64];
    for (std::uint32_t index = 0; index < 16U; ++index) {
        words[index] = load_big_endian_u32(block + 4U * index);
    }
    for (std::uint32_t index = 16U; index < 64U; ++index) {
        const std::uint32_t value15 = words[index - 15U];
        const std::uint32_t value2 = words[index - 2U];
        const std::uint32_t sigma0 = rotate_right(value15, 7U)
            ^ rotate_right(value15, 18U) ^ (value15 >> 3U);
        const std::uint32_t sigma1 = rotate_right(value2, 17U)
            ^ rotate_right(value2, 19U) ^ (value2 >> 10U);
        words[index] = words[index - 16U] + sigma0
            + words[index - 7U] + sigma1;
    }

    std::uint32_t a = context->state[0];
    std::uint32_t b = context->state[1];
    std::uint32_t c = context->state[2];
    std::uint32_t d = context->state[3];
    std::uint32_t e = context->state[4];
    std::uint32_t f = context->state[5];
    std::uint32_t g = context->state[6];
    std::uint32_t h = context->state[7];

    for (std::uint32_t round = 0; round < 64U; ++round) {
        const std::uint32_t sum1 = rotate_right(e, 6U)
            ^ rotate_right(e, 11U) ^ rotate_right(e, 25U);
        const std::uint32_t choose = (e & f) ^ (~e & g);
        const std::uint32_t temporary1 = h + sum1 + choose
            + kSha256RoundConstantsV1[round] + words[round];
        const std::uint32_t sum0 = rotate_right(a, 2U)
            ^ rotate_right(a, 13U) ^ rotate_right(a, 22U);
        const std::uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
        const std::uint32_t temporary2 = sum0 + majority;
        h = g;
        g = f;
        f = e;
        e = d + temporary1;
        d = c;
        c = b;
        b = a;
        a = temporary1 + temporary2;
    }

    context->state[0] += a;
    context->state[1] += b;
    context->state[2] += c;
    context->state[3] += d;
    context->state[4] += e;
    context->state[5] += f;
    context->state[6] += g;
    context->state[7] += h;
}

void sha256_init(Sha256ContextV1* context) noexcept {
    context->total_bytes = 0;
    context->state[0] = 0x6a09e667U;
    context->state[1] = 0xbb67ae85U;
    context->state[2] = 0x3c6ef372U;
    context->state[3] = 0xa54ff53aU;
    context->state[4] = 0x510e527fU;
    context->state[5] = 0x9b05688cU;
    context->state[6] = 0x1f83d9abU;
    context->state[7] = 0x5be0cd19U;
    context->buffered_bytes = 0;
}

void sha256_update(
    Sha256ContextV1* context,
    const std::uint8_t* input,
    std::uint64_t input_bytes
) noexcept {
    context->total_bytes += input_bytes;
    if (context->buffered_bytes != 0U) {
        const std::uint32_t available = 64U - context->buffered_bytes;
        const std::uint32_t copied = input_bytes < available
            ? static_cast<std::uint32_t>(input_bytes)
            : available;
        std::memcpy(
            context->buffer + context->buffered_bytes,
            input,
            copied
        );
        context->buffered_bytes += copied;
        input += copied;
        input_bytes -= copied;
        if (context->buffered_bytes == 64U) {
            sha256_transform(context, context->buffer);
            context->buffered_bytes = 0;
        }
    }
    while (input_bytes >= 64U) {
        sha256_transform(context, input);
        input += 64U;
        input_bytes -= 64U;
    }
    if (input_bytes != 0U) {
        std::memcpy(context->buffer, input, static_cast<std::size_t>(input_bytes));
        context->buffered_bytes = static_cast<std::uint32_t>(input_bytes);
    }
}

void sha256_final(
    Sha256ContextV1* context,
    std::uint8_t output[kDigestBytesV1]
) noexcept {
    const std::uint64_t message_bits = context->total_bytes * 8U;
    std::uint32_t cursor = context->buffered_bytes;
    context->buffer[cursor++] = 0x80U;
    if (cursor > 56U) {
        while (cursor < 64U) {
            context->buffer[cursor++] = 0;
        }
        sha256_transform(context, context->buffer);
        cursor = 0;
    }
    while (cursor < 56U) {
        context->buffer[cursor++] = 0;
    }
    for (std::uint32_t index = 0; index < 8U; ++index) {
        context->buffer[56U + index] = static_cast<std::uint8_t>(
            message_bits >> (56U - 8U * index)
        );
    }
    sha256_transform(context, context->buffer);
    for (std::uint32_t index = 0; index < 8U; ++index) {
        store_big_endian_u32(output + 4U * index, context->state[index]);
    }
}

struct ByteSinkV1 {
    std::uint8_t* output;
    std::uint64_t capacity;
    std::uint64_t cursor;
    Sha256ContextV1* sha256;
};

bool append_bytes(
    ByteSinkV1* sink,
    const std::uint8_t* bytes,
    const std::uint64_t byte_count
) noexcept {
    if (byte_count > sink->capacity - sink->cursor) {
        return false;
    }
    if (byte_count != 0U) {
        if (sink->output != nullptr) {
            std::memcpy(
                sink->output + sink->cursor,
                bytes,
                static_cast<std::size_t>(byte_count)
            );
        }
        if (sink->sha256 != nullptr) {
            sha256_update(sink->sha256, bytes, byte_count);
        }
    }
    sink->cursor += byte_count;
    return true;
}

template <std::size_t Size>
bool append_ascii(ByteSinkV1* sink, const char (&value)[Size]) noexcept {
    return append_bytes(
        sink,
        reinterpret_cast<const std::uint8_t*>(value),
        Size - 1U
    );
}

bool append_repeated_ascii(
    ByteSinkV1* sink,
    const std::uint8_t value,
    const std::uint32_t count
) noexcept {
    for (std::uint32_t index = 0; index < count; ++index) {
        if (!append_bytes(sink, &value, 1U)) {
            return false;
        }
    }
    return true;
}

bool emit_uint128_decimal(
    ByteSinkV1* sink,
    const std::uint64_t low,
    const std::uint64_t high
) noexcept {
    std::uint32_t words[4] = {
        static_cast<std::uint32_t>(high >> 32U),
        static_cast<std::uint32_t>(high),
        static_cast<std::uint32_t>(low >> 32U),
        static_cast<std::uint32_t>(low),
    };
    std::uint8_t reversed[39];
    std::uint32_t digit_count = 0;
    do {
        std::uint64_t remainder = 0;
        bool quotient_nonzero = false;
        for (std::uint32_t index = 0; index < 4U; ++index) {
            const std::uint64_t dividend = (remainder << 32U) | words[index];
            words[index] = static_cast<std::uint32_t>(dividend / 10U);
            remainder = dividend % 10U;
            quotient_nonzero = quotient_nonzero || words[index] != 0U;
        }
        reversed[digit_count++] = static_cast<std::uint8_t>('0' + remainder);
        if (!quotient_nonzero) {
            break;
        }
    } while (digit_count < 39U);
    for (std::uint32_t index = 0; index < digit_count / 2U; ++index) {
        const std::uint8_t temporary = reversed[index];
        reversed[index] = reversed[digit_count - 1U - index];
        reversed[digit_count - 1U - index] = temporary;
    }
    return append_bytes(sink, reversed, digit_count);
}

double float64_from_bits(const std::uint64_t bits) noexcept {
    double value = 0.0;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

bool emit_python_float64(ByteSinkV1* sink, const double value) noexcept {
    char encoded[64];
    const double magnitude = std::fabs(value);
    const std::chars_format format =
        ((magnitude != 0.0 && magnitude < 1.0e-4) || magnitude >= 1.0e16)
        ? std::chars_format::scientific
        : std::chars_format::fixed;
    const auto conversion = std::to_chars(
        encoded,
        encoded + sizeof(encoded),
        value,
        format
    );
    if (conversion.ec != std::errc{}) {
        return false;
    }
    char* end = conversion.ptr;
    bool has_fraction_or_exponent = false;
    for (char* cursor = encoded; cursor != end; ++cursor) {
        has_fraction_or_exponent = has_fraction_or_exponent
            || *cursor == '.' || *cursor == 'e' || *cursor == 'E';
    }
    if (!has_fraction_or_exponent) {
        *end++ = '.';
        *end++ = '0';
    }
    return append_bytes(
        sink,
        reinterpret_cast<const std::uint8_t*>(encoded),
        static_cast<std::uint64_t>(end - encoded)
    );
}

bool valid_utf8(const std::uint8_t* bytes, const std::uint64_t size) noexcept {
    if (size != 0U && bytes == nullptr) {
        return false;
    }
    std::uint64_t cursor = 0;
    while (cursor < size) {
        const std::uint8_t first = bytes[cursor++];
        if (first <= 0x7fU) {
            continue;
        }
        std::uint32_t remaining = 0;
        std::uint32_t codepoint = 0;
        std::uint32_t minimum = 0;
        if (first >= 0xc2U && first <= 0xdfU) {
            remaining = 1;
            codepoint = first & 0x1fU;
            minimum = 0x80U;
        } else if (first >= 0xe0U && first <= 0xefU) {
            remaining = 2;
            codepoint = first & 0x0fU;
            minimum = 0x800U;
        } else if (first >= 0xf0U && first <= 0xf4U) {
            remaining = 3;
            codepoint = first & 0x07U;
            minimum = 0x10000U;
        } else {
            return false;
        }
        if (remaining > size - cursor) {
            return false;
        }
        for (std::uint32_t index = 0; index < remaining; ++index) {
            const std::uint8_t continuation = bytes[cursor++];
            if ((continuation & 0xc0U) != 0x80U) {
                return false;
            }
            codepoint = (codepoint << 6U) | (continuation & 0x3fU);
        }
        if (codepoint < minimum || codepoint > 0x10ffffU
            || (codepoint >= 0xd800U && codepoint <= 0xdfffU)) {
            return false;
        }
    }
    return true;
}

bool valid_source_path(
    const std::uint8_t* bytes,
    const std::uint32_t size
) noexcept {
    return size != 0U && size <= kMaximumSourcePathUtf8BytesV1
        && bytes != nullptr && bytes[0] == static_cast<std::uint8_t>('/')
        && valid_utf8(bytes, size);
}

bool bytes_equal(
    const std::uint8_t* left,
    const std::uint32_t left_size,
    const std::uint8_t* right,
    const std::uint32_t right_size
) noexcept {
    return left_size == right_size
        && (left_size == 0U || std::memcmp(left, right, left_size) == 0);
}

bool is_supported_surface(const std::uint32_t surface) noexcept {
    return surface >= static_cast<std::uint32_t>(
        HostTemplateSurfaceV1::kCheckpointTrajectory
    ) && surface <= static_cast<std::uint32_t>(
        HostTemplateSurfaceV1::kPolicyMutationTape
    );
}

const char* packed_kind(
    const std::uint32_t token_type,
    const HostTemplateTokenValueV1* const value,
    const bool representative
) noexcept {
    switch (static_cast<HostTemplateTokenTypeV1>(token_type)) {
        case HostTemplateTokenTypeV1::kInteger:
        case HostTemplateTokenTypeV1::kPcgUint128:
        case HostTemplateTokenTypeV1::kPythonMtWord:
            return "int";
        case HostTemplateTokenTypeV1::kBoolean:
            return "bool";
        case HostTemplateTokenTypeV1::kTensorShape:
            return "list";
        case HostTemplateTokenTypeV1::kSha256Hex:
            return "str";
        case HostTemplateTokenTypeV1::kNullableFloat64:
            return representative || value->scalar_high == 1U
                ? "float"
                : "NoneType";
    }
    return nullptr;
}

bool append_c_string(ByteSinkV1* sink, const char* value) noexcept {
    std::uint64_t size = 0;
    while (value[size] != '\0') {
        ++size;
    }
    return append_bytes(
        sink,
        reinterpret_cast<const std::uint8_t*>(value),
        size
    );
}

bool emit_base_value(
    ByteSinkV1* sink,
    const HostTemplateTokenSpecV1& spec,
    const HostTemplateTokenValueV1* value,
    const bool representative
) noexcept {
    switch (static_cast<HostTemplateTokenTypeV1>(spec.token_type)) {
        case HostTemplateTokenTypeV1::kInteger:
            return emit_uint128_decimal(
                sink,
                representative ? UINT64_MAX : value->scalar_low,
                0U
            );
        case HostTemplateTokenTypeV1::kBoolean:
            return representative || value->scalar_low == 0U
                ? append_ascii(sink, "false")
                : append_ascii(sink, "true");
        case HostTemplateTokenTypeV1::kPcgUint128:
            return emit_uint128_decimal(
                sink,
                representative ? UINT64_MAX : value->scalar_low,
                representative ? UINT64_MAX : value->scalar_high
            );
        case HostTemplateTokenTypeV1::kPythonMtWord:
            return emit_uint128_decimal(
                sink,
                representative ? UINT32_MAX : value->scalar_low,
                0U
            );
        case HostTemplateTokenTypeV1::kTensorShape: {
            if (!append_ascii(sink, "[")) {
                return false;
            }
            for (std::uint32_t index = 0; index < spec.rank; ++index) {
                if ((index != 0U && !append_ascii(sink, ","))
                    || !emit_uint128_decimal(
                        sink,
                        representative ? UINT64_MAX : value->shape_dimensions[index],
                        0U
                    )) {
                    return false;
                }
            }
            return append_ascii(sink, "]");
        }
        case HostTemplateTokenTypeV1::kSha256Hex:
            return append_ascii(sink, "\"")
                && (representative
                    ? append_repeated_ascii(
                        sink,
                        static_cast<std::uint8_t>('f'),
                        64U
                    )
                    : append_bytes(sink, value->sha256_hex, 64U))
                && append_ascii(sink, "\"");
        case HostTemplateTokenTypeV1::kNullableFloat64:
            if (!representative && value->scalar_high == 0U) {
                return append_ascii(sink, "null");
            }
            return emit_python_float64(
                sink,
                representative
                    ? -std::numeric_limits<double>::max()
                    : float64_from_bits(value->scalar_low)
            );
    }
    return false;
}

bool emit_packed_string(ByteSinkV1* sink, const char* value) noexcept {
    return append_ascii(sink, "{\"kind\":\"str\",\"value\":\"")
        && append_c_string(sink, value)
        && append_ascii(sink, "\"}");
}

bool emit_depth_one(
    ByteSinkV1* sink,
    const HostTemplateTokenSpecV1& spec,
    const HostTemplateTokenValueV1* value,
    const bool representative
) noexcept {
    const char* kind = packed_kind(spec.token_type, value, representative);
    if (kind == nullptr) {
        return false;
    }
    if (static_cast<HostTemplateTokenTypeV1>(spec.token_type)
        == HostTemplateTokenTypeV1::kTensorShape) {
        if (!append_ascii(sink, "{\"items\":[")) {
            return false;
        }
        for (std::uint32_t index = 0; index < spec.rank; ++index) {
            if ((index != 0U && !append_ascii(sink, ","))
                || !append_ascii(sink, "{\"kind\":\"int\",\"value\":")
                || !emit_uint128_decimal(
                    sink,
                    representative ? UINT64_MAX : value->shape_dimensions[index],
                    0U
                )
                || !append_ascii(sink, "}")) {
                return false;
            }
        }
        return append_ascii(sink, "],\"kind\":\"list\"}");
    }
    return append_ascii(sink, "{\"kind\":\"")
        && append_c_string(sink, kind)
        && append_ascii(sink, "\",\"value\":")
        && emit_base_value(sink, spec, value, representative)
        && append_ascii(sink, "}");
}

bool emit_depth_two_scalar(
    ByteSinkV1* sink,
    const HostTemplateTokenSpecV1& spec,
    const HostTemplateTokenValueV1* value,
    const bool representative
) noexcept {
    const char* kind = packed_kind(spec.token_type, value, representative);
    return kind != nullptr
        && append_ascii(sink, "{\"items\":[[")
        && emit_packed_string(sink, "kind")
        && append_ascii(sink, ",")
        && emit_packed_string(sink, kind)
        && append_ascii(sink, "],[")
        && emit_packed_string(sink, "value")
        && append_ascii(sink, ",")
        && emit_depth_one(sink, spec, value, representative)
        && append_ascii(sink, "]],\"kind\":\"mapping\"}");
}

bool emit_depth_two_shape(
    ByteSinkV1* sink,
    const HostTemplateTokenSpecV1& spec,
    const HostTemplateTokenValueV1* value,
    const bool representative
) noexcept {
    if (!append_ascii(sink, "{\"items\":[[")
        || !emit_packed_string(sink, "items")
        || !append_ascii(sink, ",{\"items\":[")) {
        return false;
    }
    HostTemplateTokenSpecV1 scalar_spec{};
    scalar_spec.token_type = static_cast<std::uint32_t>(
        HostTemplateTokenTypeV1::kInteger
    );
    HostTemplateTokenValueV1 scalar_value{};
    scalar_value.token_type = scalar_spec.token_type;
    for (std::uint32_t index = 0; index < spec.rank; ++index) {
        scalar_value.scalar_low = representative
            ? UINT64_MAX
            : value->shape_dimensions[index];
        if ((index != 0U && !append_ascii(sink, ","))
            || !emit_depth_two_scalar(
                sink,
                scalar_spec,
                &scalar_value,
                representative
            )) {
            return false;
        }
    }
    return append_ascii(sink, "],\"kind\":\"list\"}],[")
        && emit_packed_string(sink, "kind")
        && append_ascii(sink, ",")
        && emit_packed_string(sink, "list")
        && append_ascii(sink, "]],\"kind\":\"mapping\"}");
}

bool emit_token(
    ByteSinkV1* sink,
    const HostTemplateTokenSpecV1& spec,
    const HostTemplateTokenValueV1* value,
    const bool representative
) noexcept {
    if (spec.render_pack_depth == 0U) {
        return emit_base_value(sink, spec, value, representative);
    }
    if (spec.render_pack_depth == 1U) {
        return emit_depth_one(sink, spec, value, representative);
    }
    if (spec.render_pack_depth == 2U) {
        return static_cast<HostTemplateTokenTypeV1>(spec.token_type)
                == HostTemplateTokenTypeV1::kTensorShape
            ? emit_depth_two_shape(sink, spec, value, representative)
            : emit_depth_two_scalar(sink, spec, value, representative);
    }
    return false;
}

struct FailureV1 {
    HostTemplateStatusV1 status;
    HostTemplateFailureComponentV1 component;
    std::uint32_t index;
};

constexpr FailureV1 success() noexcept {
    return {
        HostTemplateStatusV1::kSuccess,
        HostTemplateFailureComponentV1::kNone,
        kNoIndexV1,
    };
}

constexpr FailureV1 failure(
    const HostTemplateStatusV1 status,
    const HostTemplateFailureComponentV1 component,
    const std::uint32_t index = kNoIndexV1
) noexcept {
    return {status, component, index};
}

bool checked_add(
    const std::uint64_t left,
    const std::uint64_t right,
    std::uint64_t* output
) noexcept {
    if (right > std::numeric_limits<std::uint64_t>::max() - left) {
        return false;
    }
    *output = left + right;
    return true;
}

FailureV1 validate_token_spec(
    const HostTemplateTokenSpecV1& spec,
    const std::uint32_t index
) noexcept {
    if (spec.reserved != 0U) {
        return failure(
            HostTemplateStatusV1::kReservedFieldNonzero,
            HostTemplateFailureComponentV1::kTokenDeclaration,
            index
        );
    }
    if (spec.token_index != index || spec.render_pack_depth > 2U
        || !valid_source_path(spec.source_path_utf8, spec.source_path_utf8_bytes)) {
        return failure(
            HostTemplateStatusV1::kTokenDeclarationContradiction,
            HostTemplateFailureComponentV1::kTokenDeclaration,
            index
        );
    }
    const auto type = static_cast<HostTemplateTokenTypeV1>(spec.token_type);
    bool valid_rule = false;
    switch (type) {
        case HostTemplateTokenTypeV1::kInteger:
            valid_rule = spec.validation_kind == static_cast<std::uint32_t>(
                    HostTemplateValidationKindV1::kIntegerRange
                )
                && spec.rank == 0U && spec.rule_parameter == 0U
                && spec.minimum_low == 0U && spec.minimum_high == 0U
                && spec.maximum_low == UINT64_MAX && spec.maximum_high == 0U;
            break;
        case HostTemplateTokenTypeV1::kBoolean:
            valid_rule = spec.validation_kind == static_cast<std::uint32_t>(
                    HostTemplateValidationKindV1::kExactBoolean
                )
                && spec.rank == 0U && spec.rule_parameter == 0U
                && spec.minimum_low == 0U && spec.minimum_high == 0U
                && spec.maximum_low == 0U && spec.maximum_high == 0U;
            break;
        case HostTemplateTokenTypeV1::kPcgUint128:
            valid_rule = spec.validation_kind == static_cast<std::uint32_t>(
                    HostTemplateValidationKindV1::kIntegerRange
                )
                && spec.rank == 0U && spec.rule_parameter == 0U
                && spec.minimum_low == 0U && spec.minimum_high == 0U
                && spec.maximum_low == UINT64_MAX
                && spec.maximum_high == UINT64_MAX;
            break;
        case HostTemplateTokenTypeV1::kPythonMtWord:
            valid_rule = spec.validation_kind == static_cast<std::uint32_t>(
                    HostTemplateValidationKindV1::kIntegerRange
                )
                && spec.rank == 0U && spec.rule_parameter == 0U
                && spec.minimum_low == 0U && spec.minimum_high == 0U
                && spec.maximum_low == UINT32_MAX && spec.maximum_high == 0U;
            break;
        case HostTemplateTokenTypeV1::kTensorShape:
            valid_rule = spec.validation_kind == static_cast<std::uint32_t>(
                    HostTemplateValidationKindV1::kTensorShape
                )
                && spec.rule_parameter == 0U
                && spec.minimum_low == 0U && spec.minimum_high == 0U
                && spec.maximum_low == UINT64_MAX && spec.maximum_high == 0U;
            break;
        case HostTemplateTokenTypeV1::kSha256Hex:
            valid_rule = spec.validation_kind == static_cast<std::uint32_t>(
                    HostTemplateValidationKindV1::kLowercaseSha256Hex
                )
                && spec.rank == 0U && spec.rule_parameter == 64U
                && spec.minimum_low == 0U && spec.minimum_high == 0U
                && spec.maximum_low == 0U && spec.maximum_high == 0U;
            break;
        case HostTemplateTokenTypeV1::kNullableFloat64:
            valid_rule = spec.validation_kind == static_cast<std::uint32_t>(
                    HostTemplateValidationKindV1::kFiniteOptionalFloat64
                )
                && spec.rank == 0U && spec.rule_parameter == 0U
                && spec.minimum_low == 0U && spec.minimum_high == 0U
                && spec.maximum_low == 0U && spec.maximum_high == 0U;
            break;
        default:
            return failure(
                HostTemplateStatusV1::kUnsupportedTokenType,
                HostTemplateFailureComponentV1::kTokenDeclaration,
                index
            );
    }
    if (!valid_rule) {
        return failure(
            HostTemplateStatusV1::kUnsupportedValidationRule,
            HostTemplateFailureComponentV1::kTokenDeclaration,
            index
        );
    }
    ByteSinkV1 counter{
        nullptr,
        kMaximumRenderedUtf8BytesPerTemplateV1 + 1U,
        0,
        nullptr,
    };
    if (!emit_token(&counter, spec, nullptr, true)
        || counter.cursor != spec.maximum_utf8_bytes) {
        return failure(
            HostTemplateStatusV1::kTokenDeclarationContradiction,
            HostTemplateFailureComponentV1::kTokenDeclaration,
            index
        );
    }
    return success();
}

FailureV1 validate_token_value(
    const HostTemplateTokenSpecV1& spec,
    const HostTemplateTokenValueV1& value,
    const std::uint32_t index,
    std::uint32_t* rendered_bytes
) noexcept {
    if (value.reserved != 0U) {
        return failure(
            HostTemplateStatusV1::kReservedFieldNonzero,
            HostTemplateFailureComponentV1::kTokenValue,
            index
        );
    }
    if (value.token_type != spec.token_type) {
        return failure(
            HostTemplateStatusV1::kTokenValueTypeMismatch,
            HostTemplateFailureComponentV1::kTokenValue,
            index
        );
    }
    const auto type = static_cast<HostTemplateTokenTypeV1>(spec.token_type);
    switch (type) {
        case HostTemplateTokenTypeV1::kInteger:
            if (value.scalar_high != 0U || value.shape_rank != 0U
                || value.shape_dimensions != nullptr || value.sha256_hex != nullptr
                || value.sha256_hex_bytes != 0U) {
                return failure(
                    HostTemplateStatusV1::kTokenValueBounds,
                    HostTemplateFailureComponentV1::kTokenValue,
                    index
                );
            }
            break;
        case HostTemplateTokenTypeV1::kBoolean:
            if (value.scalar_low > 1U || value.scalar_high != 0U
                || value.shape_rank != 0U || value.shape_dimensions != nullptr
                || value.sha256_hex != nullptr || value.sha256_hex_bytes != 0U) {
                return failure(
                    HostTemplateStatusV1::kTokenValueBounds,
                    HostTemplateFailureComponentV1::kTokenValue,
                    index
                );
            }
            break;
        case HostTemplateTokenTypeV1::kPcgUint128:
            if (value.shape_rank != 0U || value.shape_dimensions != nullptr
                || value.sha256_hex != nullptr || value.sha256_hex_bytes != 0U) {
                return failure(
                    HostTemplateStatusV1::kTokenValueBounds,
                    HostTemplateFailureComponentV1::kTokenValue,
                    index
                );
            }
            break;
        case HostTemplateTokenTypeV1::kPythonMtWord:
            if (value.scalar_low > UINT32_MAX || value.scalar_high != 0U
                || value.shape_rank != 0U || value.shape_dimensions != nullptr
                || value.sha256_hex != nullptr || value.sha256_hex_bytes != 0U) {
                return failure(
                    HostTemplateStatusV1::kTokenValueBounds,
                    HostTemplateFailureComponentV1::kTokenValue,
                    index
                );
            }
            break;
        case HostTemplateTokenTypeV1::kTensorShape:
            if (value.scalar_low != 0U || value.scalar_high != 0U
                || value.shape_rank != spec.rank
                || (value.shape_rank != 0U && value.shape_dimensions == nullptr)
                || (value.shape_rank == 0U && value.shape_dimensions != nullptr)
                || value.sha256_hex != nullptr || value.sha256_hex_bytes != 0U) {
                return failure(
                    HostTemplateStatusV1::kTokenShapeContradiction,
                    HostTemplateFailureComponentV1::kTokenValue,
                    index
                );
            }
            break;
        case HostTemplateTokenTypeV1::kSha256Hex:
            if (value.scalar_low != 0U || value.scalar_high != 0U
                || value.shape_rank != 0U || value.shape_dimensions != nullptr
                || value.sha256_hex == nullptr || value.sha256_hex_bytes != 64U) {
                return failure(
                    HostTemplateStatusV1::kTokenSha256Contradiction,
                    HostTemplateFailureComponentV1::kTokenValue,
                    index
                );
            }
            for (std::uint32_t byte = 0; byte < 64U; ++byte) {
                const std::uint8_t character = value.sha256_hex[byte];
                if (!((character >= static_cast<std::uint8_t>('0')
                        && character <= static_cast<std::uint8_t>('9'))
                    || (character >= static_cast<std::uint8_t>('a')
                        && character <= static_cast<std::uint8_t>('f')))) {
                    return failure(
                        HostTemplateStatusV1::kTokenSha256Contradiction,
                        HostTemplateFailureComponentV1::kTokenValue,
                        index
                    );
                }
            }
            break;
        case HostTemplateTokenTypeV1::kNullableFloat64:
            if (value.scalar_high > 1U
                || (value.scalar_high == 0U && value.scalar_low != 0U)
                || (value.scalar_high == 1U
                    && !std::isfinite(float64_from_bits(value.scalar_low)))
                || value.shape_rank != 0U
                || value.shape_dimensions != nullptr
                || value.sha256_hex != nullptr
                || value.sha256_hex_bytes != 0U) {
                return failure(
                    HostTemplateStatusV1::kTokenValueBounds,
                    HostTemplateFailureComponentV1::kTokenValue,
                    index
                );
            }
            break;
        default:
            return failure(
                HostTemplateStatusV1::kUnsupportedTokenType,
                HostTemplateFailureComponentV1::kTokenValue,
                index
            );
    }

    ByteSinkV1 counter{nullptr, spec.maximum_utf8_bytes, 0, nullptr};
    if (!emit_token(&counter, spec, &value, false)
        || counter.cursor > spec.maximum_utf8_bytes) {
        return failure(
            HostTemplateStatusV1::kTokenValueBounds,
            HostTemplateFailureComponentV1::kTokenValue,
            index
        );
    }
    *rendered_bytes = static_cast<std::uint32_t>(counter.cursor);
    return success();
}

std::uint32_t find_token_by_path(
    const HostTemplateProgramV1& program,
    const std::uint8_t* path,
    const std::uint32_t path_bytes
) noexcept {
    for (std::uint32_t index = 0; index < program.token_count; ++index) {
        const HostTemplateTokenSpecV1& token = program.tokens[index];
        if (bytes_equal(
                token.source_path_utf8,
                token.source_path_utf8_bytes,
                path,
                path_bytes
            )) {
            return index;
        }
    }
    return kNoIndexV1;
}

bool token_values_equal(
    const HostTemplateTokenSpecV1& left_spec,
    const HostTemplateTokenValueV1& left,
    const HostTemplateTokenSpecV1& right_spec,
    const HostTemplateTokenValueV1& right
) noexcept {
    if (left_spec.token_type != right_spec.token_type
        || left_spec.rank != right_spec.rank) {
        return false;
    }
    switch (static_cast<HostTemplateTokenTypeV1>(left_spec.token_type)) {
        case HostTemplateTokenTypeV1::kInteger:
        case HostTemplateTokenTypeV1::kBoolean:
        case HostTemplateTokenTypeV1::kPcgUint128:
        case HostTemplateTokenTypeV1::kPythonMtWord:
        case HostTemplateTokenTypeV1::kNullableFloat64:
            return left.scalar_low == right.scalar_low
                && left.scalar_high == right.scalar_high;
        case HostTemplateTokenTypeV1::kTensorShape:
            for (std::uint32_t index = 0; index < left_spec.rank; ++index) {
                if (left.shape_dimensions[index] != right.shape_dimensions[index]) {
                    return false;
                }
            }
            return true;
        case HostTemplateTokenTypeV1::kSha256Hex:
            return std::memcmp(left.sha256_hex, right.sha256_hex, 64U) == 0;
    }
    return false;
}

FailureV1 validate_constraints(
    const HostTemplateProgramV1& program,
    const HostTemplateTokenValueV1* values
) noexcept {
    const auto surface = static_cast<HostTemplateSurfaceV1>(program.surface);
    if (program.constraint_count != 0U
        && surface != HostTemplateSurfaceV1::kRng
        && surface != HostTemplateSurfaceV1::kPolicyMutationTape) {
        return failure(
            HostTemplateStatusV1::kMalformedProgram,
            HostTemplateFailureComponentV1::kProgram
        );
    }
    for (std::uint32_t index = 0; index < program.constraint_count; ++index) {
        const HostTemplateConstraintV1& constraint = program.constraints[index];
        if (constraint.reserved0 != 0U || constraint.reserved1 != 0U
            || constraint.reserved2 != 0U || constraint.reserved3 != 0U) {
            return failure(
                HostTemplateStatusV1::kReservedFieldNonzero,
                HostTemplateFailureComponentV1::kConstraint,
                index
            );
        }
        if (!valid_source_path(
                constraint.left_source_path_utf8,
                constraint.left_source_path_utf8_bytes
            ) || !valid_source_path(
                constraint.right_source_path_utf8,
                constraint.right_source_path_utf8_bytes
            )) {
            return failure(
                HostTemplateStatusV1::kConstraintPathContradiction,
                HostTemplateFailureComponentV1::kConstraint,
                index
            );
        }
        const std::uint32_t left_index = find_token_by_path(
            program,
            constraint.left_source_path_utf8,
            constraint.left_source_path_utf8_bytes
        );
        const std::uint32_t right_index = find_token_by_path(
            program,
            constraint.right_source_path_utf8,
            constraint.right_source_path_utf8_bytes
        );
        if (left_index == kNoIndexV1 || right_index == kNoIndexV1
            || left_index == right_index) {
            return failure(
                HostTemplateStatusV1::kConstraintPathContradiction,
                HostTemplateFailureComponentV1::kConstraint,
                index
            );
        }
        const auto kind = static_cast<HostTemplateConstraintKindV1>(
            constraint.kind
        );
        if (kind == HostTemplateConstraintKindV1::kValuesEqual) {
            if (!token_values_equal(
                    program.tokens[left_index],
                    values[left_index],
                    program.tokens[right_index],
                    values[right_index]
                )) {
                return failure(
                    program.tokens[left_index].token_type
                            == program.tokens[right_index].token_type
                        ? HostTemplateStatusV1::kConstraintValueContradiction
                        : HostTemplateStatusV1::kConstraintTypeContradiction,
                    HostTemplateFailureComponentV1::kConstraint,
                    index
                );
            }
        } else if (
            kind
            == HostTemplateConstraintKindV1::kShapeFirstDimensionEqualsInteger
        ) {
            if (surface != HostTemplateSurfaceV1::kPolicyMutationTape
                || program.tokens[left_index].token_type
                    != static_cast<std::uint32_t>(
                        HostTemplateTokenTypeV1::kTensorShape
                    )
                || program.tokens[left_index].rank == 0U
                || program.tokens[right_index].token_type
                    != static_cast<std::uint32_t>(
                        HostTemplateTokenTypeV1::kInteger
                    )) {
                return failure(
                    HostTemplateStatusV1::kConstraintTypeContradiction,
                    HostTemplateFailureComponentV1::kConstraint,
                    index
                );
            }
            if (values[left_index].shape_dimensions[0]
                != values[right_index].scalar_low) {
                return failure(
                    HostTemplateStatusV1::kConstraintValueContradiction,
                    HostTemplateFailureComponentV1::kConstraint,
                    index
                );
            }
        } else {
            return failure(
                HostTemplateStatusV1::kUnsupportedConstraintKind,
                HostTemplateFailureComponentV1::kConstraint,
                index
            );
        }
    }
    return success();
}

FailureV1 validate_program_and_values(
    const HostTemplateRequestV1& request,
    std::uint32_t token_rendered_bytes[kMaximumTokensPerTemplateV1],
    std::uint64_t* rendered_bytes
) noexcept {
    const HostTemplateProgramV1& program = *request.program;
    if (program.abi_version != kHostTemplateRendererAbiVersionV1) {
        return failure(
            HostTemplateStatusV1::kUnsupportedAbiVersion,
            HostTemplateFailureComponentV1::kProgram
        );
    }
    if (program.encoding != static_cast<std::uint32_t>(
            HostTemplateEncodingV1::kUtf8
        )) {
        return failure(
            HostTemplateStatusV1::kUnsupportedEncoding,
            HostTemplateFailureComponentV1::kProgram
        );
    }
    if (!is_supported_surface(program.surface)) {
        return failure(
            HostTemplateStatusV1::kUnsupportedSurface,
            HostTemplateFailureComponentV1::kProgram
        );
    }
    if (program.reserved0 != 0U || program.reserved1 != 0U
        || program.reserved2 != 0U) {
        return failure(
            HostTemplateStatusV1::kReservedFieldNonzero,
            HostTemplateFailureComponentV1::kProgram
        );
    }
    if (program.generation < 0 || program.cell < 0
        || static_cast<std::uint64_t>(program.generation) > UINT32_MAX
        || static_cast<std::uint64_t>(program.cell) > UINT32_MAX
        || program.token_count > kMaximumTokensPerTemplateV1
        || program.part_count == 0U
        || program.part_count > kMaximumRenderedUtf8BytesPerTemplateV1
        || program.constraint_count > program.token_count
        || program.literal_utf8_bytes
            > kMaximumLiteralUtf8BytesPerTemplateV1
        || program.maximum_rendered_utf8_bytes
            > kMaximumRenderedUtf8BytesPerTemplateV1
        || program.parts == nullptr
        || (program.token_count == 0U) != (program.tokens == nullptr)
        || (program.constraint_count == 0U) != (program.constraints == nullptr)) {
        return failure(
            HostTemplateStatusV1::kProgramBounds,
            HostTemplateFailureComponentV1::kProgram
        );
    }
    if (request.token_value_count != program.token_count
        || (program.token_count != 0U && request.token_values == nullptr)) {
        return failure(
            HostTemplateStatusV1::kTokenValueCountMismatch,
            HostTemplateFailureComponentV1::kRequest
        );
    }

    std::uint64_t compiled_maximum = program.literal_utf8_bytes;
    for (std::uint32_t index = 0; index < program.token_count; ++index) {
        FailureV1 outcome = validate_token_spec(program.tokens[index], index);
        if (outcome.status != HostTemplateStatusV1::kSuccess) {
            return outcome;
        }
        for (std::uint32_t prior = 0; prior < index; ++prior) {
            if (bytes_equal(
                    program.tokens[index].source_path_utf8,
                    program.tokens[index].source_path_utf8_bytes,
                    program.tokens[prior].source_path_utf8,
                    program.tokens[prior].source_path_utf8_bytes
                )) {
                return failure(
                    HostTemplateStatusV1::kTokenDeclarationContradiction,
                    HostTemplateFailureComponentV1::kTokenDeclaration,
                    index
                );
            }
        }
        outcome = validate_token_value(
            program.tokens[index],
            request.token_values[index],
            index,
            &token_rendered_bytes[index]
        );
        if (outcome.status != HostTemplateStatusV1::kSuccess) {
            return outcome;
        }
        if (!checked_add(
                compiled_maximum,
                program.tokens[index].maximum_utf8_bytes,
                &compiled_maximum
            )) {
            return failure(
                HostTemplateStatusV1::kProgramBounds,
                HostTemplateFailureComponentV1::kProgram
            );
        }
    }
    if (compiled_maximum != program.maximum_rendered_utf8_bytes) {
        return failure(
            HostTemplateStatusV1::kRenderedBoundContradiction,
            HostTemplateFailureComponentV1::kProgram
        );
    }

    std::uint64_t literal_bytes = 0;
    std::uint64_t actual_bytes = 0;
    std::uint64_t seen_tokens[
        (kMaximumTokensPerTemplateV1 + 63U) / 64U
    ] = {};
    bool previous_literal = false;
    for (std::uint32_t index = 0; index < program.part_count; ++index) {
        const HostTemplatePartV1& part = program.parts[index];
        if (part.reserved != 0U) {
            return failure(
                HostTemplateStatusV1::kReservedFieldNonzero,
                HostTemplateFailureComponentV1::kPart,
                index
            );
        }
        const auto kind = static_cast<HostTemplatePartKindV1>(part.kind);
        if (kind == HostTemplatePartKindV1::kLiteralUtf8) {
            if (previous_literal || part.token_index != kNoIndexV1
                || part.literal_utf8 == nullptr || part.literal_utf8_bytes == 0U
                || !valid_utf8(part.literal_utf8, part.literal_utf8_bytes)
                || !checked_add(
                    literal_bytes,
                    part.literal_utf8_bytes,
                    &literal_bytes
                )
                || !checked_add(
                    actual_bytes,
                    part.literal_utf8_bytes,
                    &actual_bytes
                )) {
                return failure(
                    !valid_utf8(part.literal_utf8, part.literal_utf8_bytes)
                        ? HostTemplateStatusV1::kInvalidUtf8
                        : HostTemplateStatusV1::kPartBounds,
                    HostTemplateFailureComponentV1::kPart,
                    index
                );
            }
            previous_literal = true;
        } else if (kind == HostTemplatePartKindV1::kDynamicToken) {
            if (part.token_index >= program.token_count
                || part.literal_utf8 != nullptr || part.literal_utf8_bytes != 0U
                || !checked_add(
                    actual_bytes,
                    token_rendered_bytes[part.token_index],
                    &actual_bytes
                )) {
                return failure(
                    HostTemplateStatusV1::kPartBounds,
                    HostTemplateFailureComponentV1::kPart,
                    index
                );
            }
            seen_tokens[part.token_index / 64U]
                |= 1ULL << (part.token_index % 64U);
            previous_literal = false;
        } else {
            return failure(
                HostTemplateStatusV1::kUnsupportedPartKind,
                HostTemplateFailureComponentV1::kPart,
                index
            );
        }
    }
    if (literal_bytes != program.literal_utf8_bytes) {
        return failure(
            HostTemplateStatusV1::kRenderedBoundContradiction,
            HostTemplateFailureComponentV1::kProgram
        );
    }
    for (std::uint32_t index = 0; index < program.token_count; ++index) {
        if ((seen_tokens[index / 64U] & (1ULL << (index % 64U))) == 0U) {
            return failure(
                HostTemplateStatusV1::kTokenDeclarationContradiction,
                HostTemplateFailureComponentV1::kTokenDeclaration,
                index
            );
        }
    }
    if (actual_bytes > program.maximum_rendered_utf8_bytes
        || actual_bytes > kMaximumRenderedUtf8BytesPerTemplateV1) {
        return failure(
            HostTemplateStatusV1::kRenderedBoundContradiction,
            HostTemplateFailureComponentV1::kProgram
        );
    }
    const FailureV1 constraint_outcome = validate_constraints(
        program,
        request.token_values
    );
    if (constraint_outcome.status != HostTemplateStatusV1::kSuccess) {
        return constraint_outcome;
    }
    *rendered_bytes = actual_bytes;
    return success();
}

void record_failure(
    HostTemplateResultV1* result,
    const FailureV1& outcome
) noexcept {
    result->status = static_cast<std::uint32_t>(outcome.status);
    result->failed_component = static_cast<std::uint32_t>(outcome.component);
    result->failed_index = outcome.index;
}

}  // namespace

HostTemplateStatusV1 render_and_hash_host_template_v1(
    const HostTemplateRequestV1* request
) noexcept {
    if (request == nullptr || request->result == nullptr) {
        return HostTemplateStatusV1::kNullArgument;
    }
    HostTemplateResultV1 clean_result{};
    clean_result.failed_index = kNoIndexV1;
    *request->result = clean_result;

    if (request->abi_version != kHostTemplateRendererAbiVersionV1) {
        const FailureV1 outcome = failure(
            HostTemplateStatusV1::kUnsupportedAbiVersion,
            HostTemplateFailureComponentV1::kRequest
        );
        record_failure(request->result, outcome);
        return outcome.status;
    }
    if (request->reserved0 != 0U || request->reserved1 != 0U) {
        const FailureV1 outcome = failure(
            HostTemplateStatusV1::kReservedFieldNonzero,
            HostTemplateFailureComponentV1::kRequest
        );
        record_failure(request->result, outcome);
        return outcome.status;
    }
    if (request->program == nullptr) {
        const FailureV1 outcome = failure(
            HostTemplateStatusV1::kNullArgument,
            HostTemplateFailureComponentV1::kRequest
        );
        record_failure(request->result, outcome);
        return outcome.status;
    }
    const auto mode = static_cast<HostTemplateRenderModeV1>(request->mode);
    if (mode != HostTemplateRenderModeV1::kHashOnly
        && mode != HostTemplateRenderModeV1::kRenderAndHash) {
        const FailureV1 outcome = failure(
            HostTemplateStatusV1::kOutputModeContradiction,
            HostTemplateFailureComponentV1::kRequest
        );
        record_failure(request->result, outcome);
        return outcome.status;
    }
    if (mode == HostTemplateRenderModeV1::kHashOnly
        && (request->output_utf8 != nullptr || request->output_capacity != 0U)) {
        const FailureV1 outcome = failure(
            HostTemplateStatusV1::kOutputModeContradiction,
            HostTemplateFailureComponentV1::kOutput
        );
        record_failure(request->result, outcome);
        return outcome.status;
    }

    std::uint32_t token_rendered_bytes[kMaximumTokensPerTemplateV1] = {};
    std::uint64_t rendered_bytes = 0;
    FailureV1 outcome = validate_program_and_values(
        *request,
        token_rendered_bytes,
        &rendered_bytes
    );
    if (outcome.status != HostTemplateStatusV1::kSuccess) {
        record_failure(request->result, outcome);
        return outcome.status;
    }
    if (mode == HostTemplateRenderModeV1::kRenderAndHash
        && (request->output_utf8 == nullptr
            || request->output_capacity < rendered_bytes)) {
        outcome = failure(
            HostTemplateStatusV1::kOutputOverrun,
            HostTemplateFailureComponentV1::kOutput
        );
        record_failure(request->result, outcome);
        return outcome.status;
    }

    Sha256ContextV1 sha256{};
    sha256_init(&sha256);
    ByteSinkV1 sink{
        mode == HostTemplateRenderModeV1::kRenderAndHash
            ? request->output_utf8
            : nullptr,
        rendered_bytes,
        0,
        &sha256,
    };
    for (std::uint32_t index = 0; index < request->program->part_count; ++index) {
        const HostTemplatePartV1& part = request->program->parts[index];
        const bool emitted = static_cast<HostTemplatePartKindV1>(part.kind)
                == HostTemplatePartKindV1::kLiteralUtf8
            ? append_bytes(&sink, part.literal_utf8, part.literal_utf8_bytes)
            : emit_token(
                &sink,
                request->program->tokens[part.token_index],
                &request->token_values[part.token_index],
                false
            );
        if (!emitted) {
            outcome = failure(
                HostTemplateStatusV1::kOutputOverrun,
                HostTemplateFailureComponentV1::kOutput,
                index
            );
            record_failure(request->result, outcome);
            return outcome.status;
        }
    }
    if (sink.cursor != rendered_bytes) {
        outcome = failure(
            HostTemplateStatusV1::kRenderedBoundContradiction,
            HostTemplateFailureComponentV1::kOutput
        );
        record_failure(request->result, outcome);
        return outcome.status;
    }

    sha256_final(&sha256, request->result->digest_sha256);
    request->result->rendered_utf8_bytes = rendered_bytes;
    request->result->status = static_cast<std::uint32_t>(
        HostTemplateStatusV1::kSuccess
    );
    request->result->failed_component = static_cast<std::uint32_t>(
        HostTemplateFailureComponentV1::kNone
    );
    request->result->failed_index = kNoIndexV1;
    return HostTemplateStatusV1::kSuccess;
}

void render_and_hash_host_template_callback_v1(void* user_data) noexcept {
    static_cast<void>(render_and_hash_host_template_v1(
        static_cast<const HostTemplateRequestV1*>(user_data)
    ));
}

}  // namespace eiel::cuda_architecture_b::compact_summary_host_template_renderer_v1
