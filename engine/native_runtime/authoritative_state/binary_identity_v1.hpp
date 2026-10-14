#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace eiel::authoritative_state {

inline constexpr std::string_view kBinaryIdentitySchemaV1 =
    "eiel_authoritative_state_binary_identity_v1";

enum class IdentityErrorCodeV1 : std::uint8_t {
    kOk = 0,
    kInvalidToken,
    kInvalidSha256,
    kInvalidSourceCommit,
    kInvalidAbiCount,
    kInvalidOrder,
    kInvalidUint64Decimal,
};

struct AbiVariantIdentityV1 {
    std::string_view entrypoint;
    std::string_view abi_version;
    std::uint32_t argument_count;
    std::string_view argument_order_sha256;
};

struct BinaryIdentityV1 {
    std::string_view contract_schema_version;
    std::string_view state_schema_version;
    std::string_view state_schema_sha256;
    std::string_view checkpoint_schema_version;
    std::string_view checkpoint_schema_sha256;
    const AbiVariantIdentityV1* abi_variants;
    std::size_t abi_variant_count;
    std::string_view abi_set_sha256;
    std::string_view source_commit;
    std::string_view source_inventory_sha256;
    std::string_view build_identity_sha256;
    const std::string_view* supported_route_contracts;
    std::size_t supported_route_contract_count;
    const std::string_view* capabilities;
    std::size_t capability_count;
};

constexpr bool is_lower_hex(const std::string_view value) noexcept {
    for (const char c : value) {
        if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'))) {
            return false;
        }
    }
    return true;
}

constexpr bool is_sha256(const std::string_view value) noexcept {
    return value.size() == 64U && is_lower_hex(value);
}

constexpr bool is_source_commit(const std::string_view value) noexcept {
    return value.size() == 40U && is_lower_hex(value);
}

constexpr bool is_token(const std::string_view value) noexcept {
    if (
        value.empty()
        || !(
            (value.front() >= 'A' && value.front() <= 'Z')
            || (value.front() >= 'a' && value.front() <= 'z')
            || (value.front() >= '0' && value.front() <= '9')
        )
    ) {
        return false;
    }
    for (const char c : value) {
        const bool accepted =
            (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z')
            || (c >= '0' && c <= '9') || c == '_' || c == '.'
            || c == ':' || c == '-';
        if (!accepted) {
            return false;
        }
    }
    return true;
}

constexpr bool are_sorted_unique_tokens(
    const std::string_view* const values,
    const std::size_t count
) noexcept {
    if (count > 0U && values == nullptr) {
        return false;
    }
    for (std::size_t index = 0; index < count; ++index) {
        if (!is_token(values[index])) {
            return false;
        }
        if (index > 0U && !(values[index - 1U] < values[index])) {
            return false;
        }
    }
    return true;
}

constexpr bool is_canonical_uint64_decimal(
    const std::string_view value
) noexcept {
    if (value.empty() || (value.size() > 1U && value.front() == '0')) {
        return false;
    }
    std::uint64_t parsed = 0U;
    for (const char c : value) {
        if (c < '0' || c > '9') {
            return false;
        }
        const std::uint64_t digit = static_cast<std::uint64_t>(c - '0');
        if (
            parsed > (UINT64_MAX - digit) / UINT64_C(10)
        ) {
            return false;
        }
        parsed = parsed * UINT64_C(10) + digit;
    }
    return true;
}

constexpr IdentityErrorCodeV1 validate_binary_identity_v1(
    const BinaryIdentityV1& identity
) noexcept {
    if (
        identity.contract_schema_version != kBinaryIdentitySchemaV1
        || !is_token(identity.state_schema_version)
        || !is_token(identity.checkpoint_schema_version)
    ) {
        return IdentityErrorCodeV1::kInvalidToken;
    }
    if (
        !is_sha256(identity.state_schema_sha256)
        || !is_sha256(identity.checkpoint_schema_sha256)
        || !is_sha256(identity.abi_set_sha256)
        || !is_sha256(identity.source_inventory_sha256)
        || !is_sha256(identity.build_identity_sha256)
    ) {
        return IdentityErrorCodeV1::kInvalidSha256;
    }
    if (!is_source_commit(identity.source_commit)) {
        return IdentityErrorCodeV1::kInvalidSourceCommit;
    }
    if (identity.abi_variants == nullptr || identity.abi_variant_count == 0U) {
        return IdentityErrorCodeV1::kInvalidAbiCount;
    }
    for (std::size_t index = 0; index < identity.abi_variant_count; ++index) {
        const auto& variant = identity.abi_variants[index];
        if (!is_token(variant.entrypoint) || !is_token(variant.abi_version)) {
            return IdentityErrorCodeV1::kInvalidToken;
        }
        if (!is_sha256(variant.argument_order_sha256)) {
            return IdentityErrorCodeV1::kInvalidSha256;
        }
        if (index > 0U) {
            const auto& previous = identity.abi_variants[index - 1U];
            const bool ordered =
                previous.entrypoint < variant.entrypoint
                || (
                    previous.entrypoint == variant.entrypoint
                    && previous.abi_version < variant.abi_version
                );
            if (!ordered) {
                return IdentityErrorCodeV1::kInvalidOrder;
            }
        }
    }
    if (
        !are_sorted_unique_tokens(
            identity.supported_route_contracts,
            identity.supported_route_contract_count
        )
        || !are_sorted_unique_tokens(
            identity.capabilities,
            identity.capability_count
        )
    ) {
        return IdentityErrorCodeV1::kInvalidOrder;
    }
    return IdentityErrorCodeV1::kOk;
}

}  // namespace eiel::authoritative_state
