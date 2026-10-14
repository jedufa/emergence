#include "modular_recurrent_cpu_oracle_c_api_v1.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>

namespace {

using Row = eiel_modular_recurrent_row_v1;
using Status = eiel_modular_recurrent_status_v1;

constexpr std::int32_t kModules = EIEL_MODULAR_RECURRENT_MODULE_SLOTS_V1;
constexpr std::int32_t kConnections = EIEL_MODULAR_RECURRENT_CONNECTION_SLOTS_V1;
constexpr std::int32_t kBindings = EIEL_MODULAR_RECURRENT_BINDINGS_V1;
constexpr std::int32_t kActions = EIEL_MODULAR_RECURRENT_ACTION_WIDTH_V1;
constexpr std::int32_t kObservationWidth = EIEL_MODULAR_RECURRENT_OBSERVATION_WIDTH_V1;
constexpr std::int32_t kCostFields = EIEL_MODULAR_RECURRENT_COST_FIELDS_V1;
constexpr std::int32_t kGenericModule = 1;
constexpr std::int32_t kIntegrator = 1;
constexpr std::int32_t kGated = 2;
constexpr std::int32_t kRestAction = 5;
constexpr double kCostCoefficients[kCostFields] = {1.0, 4.0, 32.0, 2.0, 1.0,
                                                   2.0, 1.0, 0.0, 0.0, 0.0};
constexpr std::size_t kReceiptJsonCapacity = 4096U;

static_assert(std::is_standard_layout_v<Row>);
static_assert(std::is_trivially_copyable_v<Row>);
static_assert(std::is_standard_layout_v<eiel_modular_recurrent_mutation_v1>);
static_assert(std::is_trivially_copyable_v<eiel_modular_recurrent_mutation_v1>);

[[nodiscard]] bool finite(const float value) noexcept { return std::isfinite(value); }
[[nodiscard]] bool finite(const double value) noexcept { return std::isfinite(value); }

[[nodiscard]] std::uint32_t float32_bits(const float value) noexcept {
    std::uint32_t bits = 0U;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

[[nodiscard]] bool digest_words_present(const std::uint64_t (&words)[4]) noexcept {
    return words[0] != 0U || words[1] != 0U || words[2] != 0U || words[3] != 0U;
}

void copy_digest_words(
    const std::uint64_t (&source)[4], std::uint64_t (&destination)[4]) noexcept {
    for (std::size_t index = 0; index < 4U; ++index) {
        destination[index] = source[index];
    }
}

[[nodiscard]] bool digest_words_equal(
    const std::uint64_t (&left)[4], const std::uint64_t (&right)[4]) noexcept {
    for (std::size_t index = 0; index < 4U; ++index) {
        if (left[index] != right[index]) return false;
    }
    return true;
}

struct Sha256Context {
    std::uint32_t state[8];
    std::uint64_t bit_count;
    std::uint8_t buffer[64];
};

constexpr std::uint32_t kSha256Initial[8] = {
    0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
    0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U,
};

constexpr std::uint32_t kSha256Round[64] = {
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

[[nodiscard]] std::uint32_t rotr32(
    const std::uint32_t value, const std::uint32_t shift) noexcept {
    return (value >> shift) | (value << (32U - shift));
}

void sha256_transform(Sha256Context* const context, const std::uint8_t block[64]) noexcept {
    std::uint32_t schedule[64]{};
    for (std::size_t index = 0; index < 16U; ++index) {
        const std::size_t offset = index * 4U;
        schedule[index] = (static_cast<std::uint32_t>(block[offset]) << 24U)
            | (static_cast<std::uint32_t>(block[offset + 1U]) << 16U)
            | (static_cast<std::uint32_t>(block[offset + 2U]) << 8U)
            | static_cast<std::uint32_t>(block[offset + 3U]);
    }
    for (std::size_t index = 16U; index < 64U; ++index) {
        const std::uint32_t s0 = rotr32(schedule[index - 15U], 7U)
            ^ rotr32(schedule[index - 15U], 18U) ^ (schedule[index - 15U] >> 3U);
        const std::uint32_t s1 = rotr32(schedule[index - 2U], 17U)
            ^ rotr32(schedule[index - 2U], 19U) ^ (schedule[index - 2U] >> 10U);
        schedule[index] = schedule[index - 16U] + s0 + schedule[index - 7U] + s1;
    }
    std::uint32_t a = context->state[0];
    std::uint32_t b = context->state[1];
    std::uint32_t c = context->state[2];
    std::uint32_t d = context->state[3];
    std::uint32_t e = context->state[4];
    std::uint32_t f = context->state[5];
    std::uint32_t g = context->state[6];
    std::uint32_t h = context->state[7];
    for (std::size_t index = 0; index < 64U; ++index) {
        const std::uint32_t s1 = rotr32(e, 6U) ^ rotr32(e, 11U) ^ rotr32(e, 25U);
        const std::uint32_t ch = (e & f) ^ ((~e) & g);
        const std::uint32_t temp1 = h + s1 + ch + kSha256Round[index] + schedule[index];
        const std::uint32_t s0 = rotr32(a, 2U) ^ rotr32(a, 13U) ^ rotr32(a, 22U);
        const std::uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        const std::uint32_t temp2 = s0 + maj;
        h = g;
        g = f;
        f = e;
        e = d + temp1;
        d = c;
        c = b;
        b = a;
        a = temp1 + temp2;
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

void sha256_init(Sha256Context* const context) noexcept {
    for (std::size_t index = 0; index < 8U; ++index) {
        context->state[index] = kSha256Initial[index];
    }
    context->bit_count = 0U;
    std::memset(context->buffer, 0, sizeof(context->buffer));
}

void sha256_update(
    Sha256Context* const context, const void* const bytes,
    const std::size_t byte_count) noexcept {
    const auto* input = static_cast<const std::uint8_t*>(bytes);
    std::size_t buffer_index = static_cast<std::size_t>((context->bit_count / 8U) % 64U);
    context->bit_count += static_cast<std::uint64_t>(byte_count) * 8U;
    std::size_t consumed = 0U;
    if (buffer_index != 0U) {
        const std::size_t to_copy = std::min(64U - buffer_index, byte_count);
        std::memcpy(context->buffer + buffer_index, input, to_copy);
        buffer_index += to_copy;
        consumed += to_copy;
        if (buffer_index == 64U) {
            sha256_transform(context, context->buffer);
        }
    }
    while (consumed + 64U <= byte_count) {
        sha256_transform(context, input + consumed);
        consumed += 64U;
    }
    if (consumed < byte_count) {
        std::memcpy(context->buffer, input + consumed, byte_count - consumed);
    }
}

void sha256_final(Sha256Context* const context, std::uint8_t digest[32]) noexcept {
    std::size_t buffer_index = static_cast<std::size_t>((context->bit_count / 8U) % 64U);
    context->buffer[buffer_index++] = 0x80U;
    if (buffer_index > 56U) {
        std::memset(context->buffer + buffer_index, 0, 64U - buffer_index);
        sha256_transform(context, context->buffer);
        buffer_index = 0U;
    }
    std::memset(context->buffer + buffer_index, 0, 56U - buffer_index);
    for (std::size_t index = 0; index < 8U; ++index) {
        context->buffer[63U - index] =
            static_cast<std::uint8_t>((context->bit_count >> (index * 8U)) & 0xffU);
    }
    sha256_transform(context, context->buffer);
    for (std::size_t index = 0; index < 8U; ++index) {
        digest[index * 4U] = static_cast<std::uint8_t>(context->state[index] >> 24U);
        digest[index * 4U + 1U] = static_cast<std::uint8_t>(context->state[index] >> 16U);
        digest[index * 4U + 2U] = static_cast<std::uint8_t>(context->state[index] >> 8U);
        digest[index * 4U + 3U] = static_cast<std::uint8_t>(context->state[index]);
    }
}

void sha256_words_from_digest(
    const std::uint8_t digest[32], std::uint64_t (&words)[4]) noexcept {
    for (std::size_t word = 0; word < 4U; ++word) {
        std::uint64_t value = 0U;
        for (std::size_t byte = 0; byte < 8U; ++byte) {
            value = (value << 8U) | digest[word * 8U + byte];
        }
        words[word] = value;
    }
}

void structural_update(Sha256Context* const context, const void* const bytes,
                       const std::size_t byte_count) noexcept {
    sha256_update(context, bytes, byte_count);
}

void structural_payload_sha256_words(
    const Row& row, std::uint64_t (&words)[4]) noexcept {
    Sha256Context context{};
    sha256_init(&context);
    static constexpr char kDomain[] = "eiel_modular_recurrent_row_structural_payload_v1";
    structural_update(&context, kDomain, sizeof(kDomain) - 1U);
    structural_update(&context, &row.module_count, sizeof(row.module_count));
    structural_update(&context, &row.connection_count, sizeof(row.connection_count));
    structural_update(&context, row.module_occupied, sizeof(row.module_occupied));
    structural_update(&context, row.module_local_id_index, sizeof(row.module_local_id_index));
    structural_update(&context, row.module_instance_id_index, sizeof(row.module_instance_id_index));
    structural_update(&context, row.module_homology_id_index, sizeof(row.module_homology_id_index));
    structural_update(&context, row.module_type_index, sizeof(row.module_type_index));
    structural_update(&context, row.module_type_code, sizeof(row.module_type_code));
    structural_update(&context, row.identity_origin_code, sizeof(row.identity_origin_code));
    structural_update(&context, row.topology_position, sizeof(row.topology_position));
    structural_update(&context, row.module_execution_order, sizeof(row.module_execution_order));
    structural_update(&context, row.dependency_topological_order, sizeof(row.dependency_topological_order));
    structural_update(&context, row.operator_code, sizeof(row.operator_code));
    structural_update(&context, row.integrator_parameters, sizeof(row.integrator_parameters));
    structural_update(&context, row.gated_parameters, sizeof(row.gated_parameters));
    structural_update(&context, row.module_dormant, sizeof(row.module_dormant));
    structural_update(&context, row.input_binding_count, sizeof(row.input_binding_count));
    structural_update(&context, row.input_semantic_code, sizeof(row.input_semantic_code));
    structural_update(&context, row.input_semantic_id_index, sizeof(row.input_semantic_id_index));
    structural_update(&context, row.input_gain, sizeof(row.input_gain));
    structural_update(&context, row.input_enabled, sizeof(row.input_enabled));
    structural_update(&context, row.output_binding_count, sizeof(row.output_binding_count));
    structural_update(&context, row.output_semantic_code, sizeof(row.output_semantic_code));
    structural_update(&context, row.output_semantic_id_index, sizeof(row.output_semantic_id_index));
    structural_update(&context, row.output_action_index, sizeof(row.output_action_index));
    structural_update(&context, row.output_gain, sizeof(row.output_gain));
    structural_update(&context, row.output_enabled, sizeof(row.output_enabled));
    structural_update(&context, row.development_explicit, sizeof(row.development_explicit));
    structural_update(&context, row.development_stage_gains, sizeof(row.development_stage_gains));
    structural_update(&context, row.development_prerequisite_count, sizeof(row.development_prerequisite_count));
    structural_update(&context, row.development_prerequisite_homology_index, sizeof(row.development_prerequisite_homology_index));
    structural_update(&context, row.development_prerequisite_module_slot, sizeof(row.development_prerequisite_module_slot));
    structural_update(&context, row.development_latch_enabled, sizeof(row.development_latch_enabled));
    structural_update(&context, row.development_latch_present, sizeof(row.development_latch_present));
    structural_update(&context, row.development_latch_state, sizeof(row.development_latch_state));
    structural_update(&context, row.development_effective_gain_present, sizeof(row.development_effective_gain_present));
    structural_update(&context, row.development_effective_gain, sizeof(row.development_effective_gain));
    structural_update(&context, row.connection_occupied, sizeof(row.connection_occupied));
    structural_update(&context, row.connection_local_id_index, sizeof(row.connection_local_id_index));
    structural_update(&context, row.connection_src_module_slot, sizeof(row.connection_src_module_slot));
    structural_update(&context, row.connection_dst_module_slot, sizeof(row.connection_dst_module_slot));
    structural_update(&context, row.connection_src_slot_index, sizeof(row.connection_src_slot_index));
    structural_update(&context, row.connection_dst_slot_index, sizeof(row.connection_dst_slot_index));
    structural_update(&context, row.connection_value_type_index, sizeof(row.connection_value_type_index));
    structural_update(&context, row.connection_channel_index, sizeof(row.connection_channel_index));
    structural_update(&context, row.connection_weight, sizeof(row.connection_weight));
    structural_update(&context, row.connection_execution_order, sizeof(row.connection_execution_order));
    structural_update(&context, row.structural_cost_counts, sizeof(row.structural_cost_counts));
    structural_update(&context, row.structural_cost_components, sizeof(row.structural_cost_components));
    structural_update(&context, &row.structural_cost_total, sizeof(row.structural_cost_total));
    std::uint8_t digest[32]{};
    sha256_final(&context, digest);
    sha256_words_from_digest(digest, words);
}

[[nodiscard]] bool structural_payload_matches_binding(
    const Row& row,
    const eiel_modular_recurrent_canonical_receipt_binding_v1& binding) noexcept {
    std::uint64_t native_words[4]{};
    structural_payload_sha256_words(row, native_words);
    return digest_words_equal(native_words, binding.structural_payload_sha256_words);
}

void digest_bytes_to_hex(const std::uint8_t digest[32], char hex_out[65]) noexcept {
    static constexpr char kHex[] = "0123456789abcdef";
    for (std::size_t index = 0; index < 32U; ++index) {
        hex_out[index * 2U] = kHex[(digest[index] >> 4U) & 0x0fU];
        hex_out[index * 2U + 1U] = kHex[digest[index] & 0x0fU];
    }
    hex_out[64] = '\0';
}

void digest_words_to_hex(
    const std::uint64_t (&words)[4], char hex_out[65]) noexcept {
    static constexpr char kHex[] = "0123456789abcdef";
    for (std::size_t word = 0; word < 4U; ++word) {
        const std::uint64_t value = words[word];
        for (std::size_t nibble = 0; nibble < 16U; ++nibble) {
            const std::size_t shift = (15U - nibble) * 4U;
            hex_out[word * 16U + nibble] =
                kHex[static_cast<std::size_t>((value >> shift) & 0x0fU)];
        }
    }
    hex_out[64] = '\0';
}

[[nodiscard]] const char* profile_id_for_native_receipt(
    const std::uint32_t profile) noexcept {
    if (profile == EIEL_MODULAR_RECURRENT_PROFILE_12X32_V1) return "b2-12x32-v1";
    if (profile == EIEL_MODULAR_RECURRENT_PROFILE_16X48_V1) return "b2-16x48-v1";
    return nullptr;
}

[[nodiscard]] const char* operation_name_for_code(const std::int32_t code) noexcept {
    switch (code) {
        case 0: return "toggle_dormancy";
        case 1: return "prune_eligible_module";
        case 2: return "refine_descendant_parameter";
        case 3: return "refine_descendant_routing_or_expression";
        default: return nullptr;
    }
}

[[nodiscard]] const char* status_name_for_code(const std::int32_t code) noexcept {
    switch (code) {
        case 1: return "accepted";
        case 2: return "dependency_blocked";
        case 3: return "disabled";
        case 4: return "ineligible";
        case 5: return "noop";
        case 6: return "rejected";
        default: return nullptr;
    }
}

[[nodiscard]] const char* reason_name_for_code(const std::int32_t code) noexcept {
    switch (code) {
        case 1: return "proposal_valid";
        case 2: return "native_core_committed";
        case 3: return "capacity_unavailable";
        case 4: return "stale_slot_token";
        case 5: return "invalid_draw";
        case 6: return "child_publication_failed";
        default: return nullptr;
    }
}

[[nodiscard]] const char* changed_dimension_name_for_code(
    const std::int32_t code) noexcept {
    switch (code) {
        case 0: return "none";
        case 1: return "expression_scale";
        case 2: return "live_retention";
        case 3: return "live_input_gain";
        case 4: return "live_output_gain";
        case 5: return "reset_bias";
        case 6: return "reset_input_gain";
        case 7: return "reset_state_gain";
        case 8: return "update_bias";
        case 9: return "update_input_gain";
        case 10: return "update_state_gain";
        case 11: return "candidate_bias";
        case 12: return "candidate_input_gain";
        case 13: return "candidate_state_gain";
        case 14: return "output_gain";
        case 15: return "input_binding_gain";
        case 16: return "output_binding_gain";
        case 17: return "module";
        case 18: return "dormant";
        default: return nullptr;
    }
}

[[nodiscard]] bool write_transaction_json(
    const eiel_modular_recurrent_canonical_receipt_binding_v1& binding,
    char* const out, const std::size_t capacity) noexcept {
    const char* const operation = operation_name_for_code(binding.operation_code);
    const char* const status = status_name_for_code(binding.status_code);
    const char* const reason = reason_name_for_code(binding.reason_code);
    const char* const dimension =
        changed_dimension_name_for_code(binding.changed_dimension_code);
    if (operation == nullptr || status == nullptr || reason == nullptr
        || dimension == nullptr) {
        return false;
    }
    const char* const changed = binding.changed != 0U ? "true" : "false";
    const int written = std::snprintf(
        out, capacity,
        "{\"attempt_ordinal\":0,\"changed\":%s,\"changed_dimension\":\"%s\","
        "\"child_agent_id\":%lld,\"child_slot\":%lld,\"child_slot_token\":%lld,"
        "\"draw0_float64_bits\":\"%016llx\",\"draw1_float64_bits\":\"%016llx\","
        "\"operation\":\"%s\",\"parent_agent_id\":%lld,\"parent_slot\":%lld,"
        "\"parent_slot_token\":%lld,\"reason\":\"%s\","
        "\"selected_homology_id_index\":%d,\"selected_instance_id_index\":%d,"
        "\"status\":\"%s\",\"transaction_kind\":\"ordinary_birth\","
        "\"transaction_ordinal\":%lld}",
        changed, dimension, static_cast<long long>(binding.child_agent_id),
        static_cast<long long>(binding.child_slot),
        static_cast<long long>(binding.child_slot_token),
        static_cast<unsigned long long>(binding.draw0_float64_bits),
        static_cast<unsigned long long>(binding.draw1_float64_bits), operation,
        static_cast<long long>(binding.parent_agent_id),
        static_cast<long long>(binding.parent_slot),
        static_cast<long long>(binding.parent_slot_token), reason,
        binding.selected_homology_id_index, binding.selected_instance_id_index,
        status, static_cast<long long>(binding.transaction_ordinal));
    return written > 0 && static_cast<std::size_t>(written) < capacity;
}

[[nodiscard]] bool native_canonical_receipt_sha256_words(
    const std::uint32_t profile,
    const eiel_modular_recurrent_canonical_receipt_source_v1& source,
    const eiel_modular_recurrent_canonical_receipt_binding_v1& binding,
    std::uint64_t (&canonical_words)[4],
    std::uint64_t (&canonical_bytes_words)[4]) noexcept {
    if (source.source_version != 1U || source.reserved0 != 0U
        || !digest_words_present(source.source_packed_sha256_words)
        || !digest_words_present(source.ancestry_tape_sha256_words)
        || binding.binding_version < 4U
        || binding.canonical_receipt_contract_version != 1U
        || binding.native_receipt_construction_required != 1U
        || binding.transaction_ordinal != 0 || binding.changed > 1U
        || (binding.changed != 0U) != (binding.status_code == 1)
        || ((binding.changed != 0U) == (binding.changed_dimension_code == 0))
        || binding.parent_agent_id < 0 || binding.parent_slot < 0
        || binding.parent_slot_token < 0 || binding.child_agent_id < 0
        || binding.child_slot < 0 || binding.child_slot_token < 0
        || binding.selected_instance_id_index < -1
        || binding.selected_homology_id_index < -1) {
        return false;
    }
    for (const auto value : binding.reserved) {
        if (value != 0U) return false;
    }
    for (const auto value : binding.reserved_tail) {
        if (value != 0U) return false;
    }
    const char* const profile_id = profile_id_for_native_receipt(profile);
    if (profile_id == nullptr) return false;
    char source_hex[65]{};
    char ancestry_hex[65]{};
    digest_words_to_hex(source.source_packed_sha256_words, source_hex);
    digest_words_to_hex(source.ancestry_tape_sha256_words, ancestry_hex);
    char transaction[kReceiptJsonCapacity]{};
    if (!write_transaction_json(binding, transaction, sizeof(transaction))) return false;
    char body[kReceiptJsonCapacity]{};
    int written = std::snprintf(
        body, sizeof(body),
        "{\"ancestry_tape_sha256\":\"%s\",\"contract_id\":\"modular_two_draws_v1\","
        "\"profile_id\":\"%s\",\"schema_version\":"
        "\"eiel_modular_recurrent_mutation_receipt_v1\","
        "\"source_packed_sha256\":\"%s\",\"transactions\":[%s]}",
        ancestry_hex, profile_id, source_hex, transaction);
    if (written <= 0 || static_cast<std::size_t>(written) >= sizeof(body)) {
        return false;
    }
    Sha256Context context{};
    sha256_init(&context);
    sha256_update(&context, body, static_cast<std::size_t>(written));
    std::uint8_t canonical_digest[32]{};
    sha256_final(&context, canonical_digest);
    sha256_words_from_digest(canonical_digest, canonical_words);
    char canonical_hex[65]{};
    digest_bytes_to_hex(canonical_digest, canonical_hex);
    char receipt[kReceiptJsonCapacity]{};
    written = std::snprintf(
        receipt, sizeof(receipt),
        "{\"ancestry_tape_sha256\":\"%s\",\"canonical_sha256\":\"%s\","
        "\"contract_id\":\"modular_two_draws_v1\",\"profile_id\":\"%s\","
        "\"schema_version\":\"eiel_modular_recurrent_mutation_receipt_v1\","
        "\"source_packed_sha256\":\"%s\",\"transactions\":[%s]}",
        ancestry_hex, canonical_hex, profile_id, source_hex, transaction);
    if (written <= 0 || static_cast<std::size_t>(written) >= sizeof(receipt)) {
        return false;
    }
    sha256_init(&context);
    sha256_update(&context, receipt, static_cast<std::size_t>(written));
    std::uint8_t bytes_digest[32]{};
    sha256_final(&context, bytes_digest);
    sha256_words_from_digest(bytes_digest, canonical_bytes_words);
    return true;
}

[[nodiscard]] bool native_canonical_receipt_matches_binding(
    const std::uint32_t profile,
    const eiel_modular_recurrent_canonical_receipt_source_v1* const source,
    const eiel_modular_recurrent_canonical_receipt_binding_v1& binding) noexcept {
    if (source == nullptr) return false;
    std::uint64_t canonical_words[4]{};
    std::uint64_t canonical_bytes_words[4]{};
    if (!native_canonical_receipt_sha256_words(
            profile, *source, binding, canonical_words, canonical_bytes_words)) {
        return false;
    }
    return digest_words_equal(canonical_words, binding.canonical_receipt_sha256_words)
        && digest_words_equal(
            canonical_bytes_words, binding.canonical_receipt_bytes_sha256_words);
}

void rollback_publication(
    Row* const rows, eiel_modular_recurrent_owner_view_v1* const owners,
    const std::int64_t row_index, const Row& previous_row,
    const std::int64_t previous_agent_id, const std::int64_t previous_token,
    const std::uint8_t previous_alive) noexcept {
    rows[row_index] = previous_row;
    owners->slot_to_agent_id[row_index] = previous_agent_id;
    owners->slot_tokens[row_index] = previous_token;
    owners->alive_mask[row_index] = previous_alive;
}

void mark_structural_mismatch(
    eiel_modular_recurrent_birth_receipt_v1* const receipt) noexcept {
    receipt->lifecycle_status = EIEL_MODULAR_RECURRENT_OWNER_MISMATCH_V1;
    receipt->committed = 0U;
    receipt->parent_preserved = 0U;
    receipt->child_runtime_reset = 0U;
    receipt->canonical_receipt_bound = 0U;
    receipt->canonical_receipt_identity_checked = 0U;
    receipt->recurrent_receipt_bytes_bound = 0U;
    receipt->structural_payload_sha256_bound = 0U;
    receipt->structural_payload_identity_checked = 0U;
    for (std::size_t index = 0; index < 4U; ++index) {
        receipt->canonical_recurrent_receipt_sha256_words[index] = 0U;
        receipt->canonical_recurrent_receipt_bytes_sha256_words[index] = 0U;
        receipt->structural_payload_sha256_words[index] = 0U;
    }
}

[[nodiscard]] std::uint32_t receipt_version_for_binding(
    const std::uint32_t binding_version) noexcept {
    return binding_version >= 3U ? 4U : (binding_version >= 2U ? 3U : 2U);
}

void bind_canonical_receipt(
    const eiel_modular_recurrent_canonical_receipt_binding_v1& binding,
    eiel_modular_recurrent_birth_receipt_v1* const receipt) noexcept {
    receipt->canonical_receipt_bound = 1U;
    receipt->canonical_receipt_identity_checked = 1U;
    receipt->recurrent_receipt_bytes_bound = 1U;
    receipt->draw0_float64_bits = binding.draw0_float64_bits;
    receipt->draw1_float64_bits = binding.draw1_float64_bits;
    copy_digest_words(
        binding.canonical_receipt_sha256_words,
        receipt->canonical_recurrent_receipt_sha256_words);
    copy_digest_words(
        binding.canonical_receipt_bytes_sha256_words,
        receipt->canonical_recurrent_receipt_bytes_sha256_words);
    if (binding.binding_version >= 2U
        && digest_words_present(binding.structural_payload_sha256_words)) {
        copy_digest_words(
            binding.structural_payload_sha256_words,
            receipt->structural_payload_sha256_words);
        receipt->structural_payload_sha256_bound = 1U;
        receipt->structural_payload_identity_checked = 1U;
    }
    receipt->transaction_ordinal = binding.transaction_ordinal;
    receipt->selected_instance_id_index = binding.selected_instance_id_index;
    receipt->selected_homology_id_index = binding.selected_homology_id_index;
    receipt->operation_code = binding.operation_code;
    receipt->status_code = binding.status_code;
    receipt->reason_code = binding.reason_code;
    receipt->changed = binding.changed;
}

[[nodiscard]] float clipped(const float value, const float low, const float high) noexcept {
    return std::min(high, std::max(low, value));
}

[[nodiscard]] float sigmoid(const float value) noexcept {
    const float x = clipped(value, -16.0F, 16.0F);
    if (x >= 0.0F) {
        const float term = std::exp(-x);
        return 1.0F / (1.0F + term);
    }
    const float term = std::exp(x);
    return term / (1.0F + term);
}

[[nodiscard]] bool binary(const std::uint8_t value) noexcept { return value <= 1U; }

struct BufferRange {
    std::uintptr_t begin;
    std::uintptr_t end;
    bool writable;
};

[[nodiscard]] bool make_range(
    const void* const pointer, const std::size_t bytes, const bool writable,
    BufferRange* const out) noexcept {
    if (pointer == nullptr || out == nullptr || bytes == 0U) return false;
    const auto begin = reinterpret_cast<std::uintptr_t>(pointer);
    if (bytes > std::numeric_limits<std::uintptr_t>::max() - begin) return false;
    *out = {begin, begin + bytes, writable};
    return true;
}

[[nodiscard]] bool ranges_overlap(
    const BufferRange& left, const BufferRange& right) noexcept {
    return left.begin < right.end && right.begin < left.end;
}

[[nodiscard]] bool checked_bytes(
    const std::int64_t count, const std::size_t element_size,
    std::size_t* const out) noexcept {
    if (count <= 0 || out == nullptr) return false;
    const auto unsigned_count = static_cast<std::uint64_t>(count);
    if (unsigned_count > std::numeric_limits<std::size_t>::max() / element_size) {
        return false;
    }
    *out = static_cast<std::size_t>(unsigned_count) * element_size;
    return true;
}

[[nodiscard]] Status profile_limits(
    const std::uint32_t profile, std::int32_t* const modules,
    std::int32_t* const connections) noexcept {
    if (modules == nullptr || connections == nullptr) {
        return EIEL_MODULAR_RECURRENT_NULL_POINTER_V1;
    }
    if (profile == EIEL_MODULAR_RECURRENT_PROFILE_12X32_V1) {
        *modules = 12;
        *connections = 32;
        return EIEL_MODULAR_RECURRENT_SUCCESS_V1;
    }
    if (profile == EIEL_MODULAR_RECURRENT_PROFILE_16X48_V1) {
        *modules = 16;
        *connections = 48;
        return EIEL_MODULAR_RECURRENT_SUCCESS_V1;
    }
    return EIEL_MODULAR_RECURRENT_INVALID_PROFILE_V1;
}

void clear_row(Row* const row) noexcept {
    std::memset(row, 0, sizeof(*row));
    row->agent_id = -1;
    row->agent_slot_token = -1;
    for (std::int32_t slot = 0; slot < kModules; ++slot) {
        row->module_local_id_index[slot] = -1;
        row->module_instance_id_index[slot] = -1;
        row->module_homology_id_index[slot] = -1;
        row->module_type_index[slot] = -1;
        row->module_execution_order[slot] = -1;
        row->dependency_topological_order[slot] = -1;
        for (std::int32_t binding = 0; binding < kBindings; ++binding) {
            const std::int32_t index = slot * kBindings + binding;
            row->input_semantic_code[index] = -1;
            row->input_semantic_id_index[index] = -1;
            row->output_semantic_code[index] = -1;
            row->output_semantic_id_index[index] = -1;
            row->output_action_index[index] = -1;
            row->development_prerequisite_homology_index[index] = -1;
            row->development_prerequisite_module_slot[index] = -1;
        }
    }
    for (std::int32_t slot = 0; slot < kConnections; ++slot) {
        row->connection_local_id_index[slot] = -1;
        row->connection_src_module_slot[slot] = -1;
        row->connection_dst_module_slot[slot] = -1;
        row->connection_src_slot_index[slot] = -1;
        row->connection_dst_slot_index[slot] = -1;
        row->connection_value_type_index[slot] = -1;
        row->connection_channel_index[slot] = -1;
        row->connection_execution_order[slot] = -1;
    }
}

void reset_runtime(Row* const row) noexcept {
    for (std::int32_t slot = 0; slot < kModules; ++slot) {
        row->module_state_present[slot] = 0U;
        row->module_state[slot] = 0.0F;
        row->module_output_present[slot] = 0U;
        row->module_previous_output[slot] = 0.0F;
        row->development_latch_present[slot] = 0U;
        row->development_latch_state[slot] = 0U;
        row->development_effective_gain_present[slot] = 0U;
        row->development_effective_gain[slot] = 0.0F;
    }
}

[[nodiscard]] bool empty_module_slot(const Row& row, const std::int32_t slot) noexcept {
    if (row.module_local_id_index[slot] != -1 || row.module_instance_id_index[slot] != -1
        || row.module_homology_id_index[slot] != -1 || row.module_type_index[slot] != -1
        || row.module_type_code[slot] != 0 || row.identity_origin_code[slot] != 0
        || row.operator_code[slot] != 0 || row.module_dormant[slot] != 0U
        || row.module_state_present[slot] != 0U || row.module_state[slot] != 0.0F
        || row.module_output_present[slot] != 0U || row.module_previous_output[slot] != 0.0F
        || row.input_binding_count[slot] != 0 || row.output_binding_count[slot] != 0
        || row.development_explicit[slot] != 0U
        || row.development_prerequisite_count[slot] != 0
        || row.development_latch_enabled[slot] != 0U
        || row.development_latch_present[slot] != 0U
        || row.development_latch_state[slot] != 0U
        || row.development_effective_gain_present[slot] != 0U
        || row.development_effective_gain[slot] != 0.0F) {
        return false;
    }
    for (std::int32_t value = 0; value < 2; ++value) {
        if (row.topology_position[slot * 2 + value] != 0.0F) return false;
    }
    for (std::int32_t value = 0; value < 4; ++value) {
        if (row.integrator_parameters[slot * 4 + value] != 0.0F) return false;
    }
    for (std::int32_t value = 0; value < 10; ++value) {
        if (row.gated_parameters[slot * 10 + value] != 0.0F) return false;
    }
    for (std::int32_t value = 0; value < 3; ++value) {
        if (row.development_stage_gains[slot * 3 + value] != 0.0F) return false;
    }
    for (std::int32_t binding = 0; binding < kBindings; ++binding) {
        const std::int32_t index = slot * kBindings + binding;
        if (row.input_semantic_code[index] != -1
            || row.input_semantic_id_index[index] != -1 || row.input_gain[index] != 0.0F
            || row.input_enabled[index] != 0U || row.output_semantic_code[index] != -1
            || row.output_semantic_id_index[index] != -1
            || row.output_action_index[index] != -1 || row.output_gain[index] != 0.0F
            || row.output_enabled[index] != 0U
            || row.development_prerequisite_homology_index[index] != -1
            || row.development_prerequisite_module_slot[index] != -1) {
            return false;
        }
    }
    return true;
}

[[nodiscard]] bool empty_connection_slot(const Row& row, const std::int32_t slot) noexcept {
    return row.connection_local_id_index[slot] == -1
        && row.connection_src_module_slot[slot] == -1
        && row.connection_dst_module_slot[slot] == -1
        && row.connection_src_slot_index[slot] == -1
        && row.connection_dst_slot_index[slot] == -1
        && row.connection_value_type_index[slot] == -1
        && row.connection_channel_index[slot] == -1
        && row.connection_weight[slot] == 0.0F
        && row.connection_execution_order[slot] == -1;
}

[[nodiscard]] Status compute_cost(
    const Row& row, eiel_modular_recurrent_cost_capacity_receipt_v1* const out,
    const std::uint32_t profile) noexcept {
    if (out == nullptr) return EIEL_MODULAR_RECURRENT_NULL_POINTER_V1;
    std::int32_t module_limit = 0;
    std::int32_t connection_limit = 0;
    Status status = profile_limits(profile, &module_limit, &connection_limit);
    if (status != EIEL_MODULAR_RECURRENT_SUCCESS_V1) return status;
    eiel_modular_recurrent_cost_capacity_receipt_v1 receipt{};
    receipt.logical_module_slots = module_limit;
    receipt.logical_connection_slots = connection_limit;
    for (std::int32_t slot = 0; slot < kModules; ++slot) {
        if (row.module_occupied[slot] == 0U) continue;
        ++receipt.occupied_modules;
        if (row.module_type_code[slot] != kGenericModule) continue;
        ++receipt.counts[0];
        if (row.module_dormant[slot] != 0U) {
            ++receipt.counts[7];
            continue;
        }
        if (row.operator_code[slot] == kIntegrator) ++receipt.counts[1];
        if (row.operator_code[slot] == kGated) ++receipt.counts[2];
        for (std::int32_t binding = 0; binding < row.input_binding_count[slot]; ++binding) {
            receipt.counts[4] += row.input_enabled[slot * kBindings + binding] != 0U ? 1 : 0;
        }
        for (std::int32_t binding = 0; binding < row.output_binding_count[slot]; ++binding) {
            receipt.counts[5] += row.output_enabled[slot * kBindings + binding] != 0U ? 1 : 0;
        }
        receipt.counts[6] += row.development_prerequisite_count[slot];
    }
    for (std::int32_t slot = 0; slot < kConnections; ++slot) {
        if (row.connection_occupied[slot] != 0U) ++receipt.occupied_connections;
    }
    receipt.counts[3] = receipt.occupied_connections;
    receipt.counts[8] = row.structural_cost_counts[8];
    receipt.counts[9] = row.structural_cost_counts[9];
    for (std::int32_t index = 0; index < kCostFields; ++index) {
        receipt.components[index] = static_cast<double>(receipt.counts[index])
            * kCostCoefficients[index];
        receipt.total += receipt.components[index];
    }
    receipt.reporting_only = 1U;
    receipt.at_module_capacity = receipt.occupied_modules == module_limit ? 1U : 0U;
    receipt.at_connection_capacity = receipt.occupied_connections == connection_limit ? 1U : 0U;
    *out = receipt;
    return EIEL_MODULAR_RECURRENT_SUCCESS_V1;
}

[[nodiscard]] Status validate_row(const Row& row, const std::uint32_t profile) noexcept {
    std::int32_t module_limit = 0;
    std::int32_t connection_limit = 0;
    Status status = profile_limits(profile, &module_limit, &connection_limit);
    if (status != EIEL_MODULAR_RECURRENT_SUCCESS_V1) return status;
    if (!binary(row.agent_occupied)) return EIEL_MODULAR_RECURRENT_INVALID_PLANE_V1;
    if (row.agent_occupied == 0U) {
        if (row.agent_id != -1 || row.agent_slot_token != -1 || row.module_count != 0
            || row.connection_count != 0 || row.structural_cost_total != 0.0) {
            return EIEL_MODULAR_RECURRENT_INVALID_PLANE_V1;
        }
    } else if (row.agent_id < 0 || row.agent_slot_token < 0) {
        return EIEL_MODULAR_RECURRENT_INVALID_PLANE_V1;
    }
    if (row.module_count < 0 || row.module_count > module_limit
        || row.connection_count < 0 || row.connection_count > connection_limit) {
        return EIEL_MODULAR_RECURRENT_CAPACITY_EXCEEDED_V1;
    }
    std::int32_t occupied_modules = 0;
    std::int32_t generic_modules = 0;
    bool execution_seen[kModules]{};
    bool dependency_seen[kModules]{};
    for (std::int32_t slot = 0; slot < kModules; ++slot) {
        if (!binary(row.module_occupied[slot]) || !binary(row.module_dormant[slot])
            || !binary(row.module_state_present[slot])
            || !binary(row.module_output_present[slot])
            || !binary(row.development_explicit[slot])
            || !binary(row.development_latch_enabled[slot])
            || !binary(row.development_latch_present[slot])
            || !binary(row.development_latch_state[slot])
            || !binary(row.development_effective_gain_present[slot])) {
            return EIEL_MODULAR_RECURRENT_INVALID_PLANE_V1;
        }
        if (slot >= module_limit && row.module_occupied[slot] != 0U) {
            return EIEL_MODULAR_RECURRENT_CAPACITY_EXCEEDED_V1;
        }
        if (row.module_occupied[slot] == 0U) {
            if (!empty_module_slot(row, slot)) return EIEL_MODULAR_RECURRENT_INVALID_PLANE_V1;
            continue;
        }
        ++occupied_modules;
        if (row.agent_occupied == 0U || row.module_local_id_index[slot] < 0
            || row.module_instance_id_index[slot] < 0
            || row.module_homology_id_index[slot] < 0 || row.module_type_index[slot] < 0
            || row.module_type_code[slot] < 1 || row.module_type_code[slot] > 6
            || row.identity_origin_code[slot] < 1 || row.identity_origin_code[slot] > 2) {
            return EIEL_MODULAR_RECURRENT_INVALID_PLANE_V1;
        }
        for (std::int32_t other = 0; other < slot; ++other) {
            if (row.module_occupied[other] != 0U
                && (row.module_local_id_index[other] == row.module_local_id_index[slot]
                    || row.module_instance_id_index[other] == row.module_instance_id_index[slot]
                    || row.module_homology_id_index[other] == row.module_homology_id_index[slot])) {
                return EIEL_MODULAR_RECURRENT_INVALID_PLANE_V1;
            }
        }
        for (std::int32_t value = 0; value < 2; ++value) {
            if (!finite(row.topology_position[slot * 2 + value])) {
                return EIEL_MODULAR_RECURRENT_NONFINITE_V1;
            }
        }
        const std::int32_t op = row.operator_code[slot];
        if (row.module_type_code[slot] == kGenericModule) {
            ++generic_modules;
            if (op != kIntegrator && op != kGated) {
                return EIEL_MODULAR_RECURRENT_INVALID_PLANE_V1;
            }
        }
        for (std::int32_t value = 0; value < 4; ++value) {
            const float parameter = row.integrator_parameters[slot * 4 + value];
            if (!finite(parameter)) return EIEL_MODULAR_RECURRENT_NONFINITE_V1;
        }
        const float expression = row.integrator_parameters[slot * 4];
        const float retention = row.integrator_parameters[slot * 4 + 1];
        const float input_gain = row.integrator_parameters[slot * 4 + 2];
        const float live_output_gain = row.integrator_parameters[slot * 4 + 3];
        if (row.module_type_code[slot] == kGenericModule
            && (expression < -2.0F || expression > 2.0F || retention < 0.0F
                || retention > 1.0F || input_gain < -2.0F || input_gain > 2.0F
                || live_output_gain < -2.0F || live_output_gain > 2.0F)) {
            return EIEL_MODULAR_RECURRENT_INVALID_PLANE_V1;
        }
        for (std::int32_t value = 0; value < 10; ++value) {
            const float parameter = row.gated_parameters[slot * 10 + value];
            if (!finite(parameter)) return EIEL_MODULAR_RECURRENT_NONFINITE_V1;
            if (op == kGated && (parameter < (value == 9 ? -2.0F : -4.0F)
                                 || parameter > (value == 9 ? 2.0F : 4.0F))) {
                return EIEL_MODULAR_RECURRENT_INVALID_PLANE_V1;
            }
        }
        if (!finite(row.module_state[slot]) || !finite(row.module_previous_output[slot])
            || row.module_state[slot] < -1.0F || row.module_state[slot] > 1.0F
            || row.module_previous_output[slot] < -1.0F
            || row.module_previous_output[slot] > 1.0F
            || (row.module_state_present[slot] == 0U && row.module_state[slot] != 0.0F)
            || (row.module_output_present[slot] == 0U
                && row.module_previous_output[slot] != 0.0F)) {
            return EIEL_MODULAR_RECURRENT_INVALID_PLANE_V1;
        }
        if (row.input_binding_count[slot] < 0 || row.input_binding_count[slot] > kBindings
            || row.output_binding_count[slot] < 0
            || row.output_binding_count[slot] > kBindings
            || row.development_prerequisite_count[slot] < 0
            || row.development_prerequisite_count[slot] > kBindings) {
            return EIEL_MODULAR_RECURRENT_INVALID_PLANE_V1;
        }
        for (std::int32_t binding = 0; binding < kBindings; ++binding) {
            const std::int32_t index = slot * kBindings + binding;
            if (!binary(row.input_enabled[index]) || !binary(row.output_enabled[index])) {
                return EIEL_MODULAR_RECURRENT_INVALID_PLANE_V1;
            }
            if (binding < row.input_binding_count[slot]) {
                if (row.input_semantic_code[index] < 0 || row.input_semantic_code[index] > 1
                    || row.input_semantic_id_index[index] < 0 || !finite(row.input_gain[index])
                    || row.input_gain[index] < -1.0F || row.input_gain[index] > 1.0F) {
                    return EIEL_MODULAR_RECURRENT_INVALID_PLANE_V1;
                }
            } else if (row.input_semantic_code[index] != -1
                       || row.input_semantic_id_index[index] != -1
                       || row.input_gain[index] != 0.0F || row.input_enabled[index] != 0U) {
                return EIEL_MODULAR_RECURRENT_INVALID_PLANE_V1;
            }
            if (binding < row.output_binding_count[slot]) {
                const std::int32_t semantic = row.output_semantic_code[index];
                const std::int32_t expected_action = semantic == 4 ? kRestAction : semantic;
                if (semantic < 0 || semantic > 4 || row.output_semantic_id_index[index] < 0
                    || row.output_action_index[index] != expected_action
                    || !finite(row.output_gain[index]) || row.output_gain[index] < -1.0F
                    || row.output_gain[index] > 1.0F) {
                    return EIEL_MODULAR_RECURRENT_INVALID_PLANE_V1;
                }
            } else if (row.output_semantic_code[index] != -1
                       || row.output_semantic_id_index[index] != -1
                       || row.output_action_index[index] != -1 || row.output_gain[index] != 0.0F
                       || row.output_enabled[index] != 0U) {
                return EIEL_MODULAR_RECURRENT_INVALID_PLANE_V1;
            }
            if (binding < row.development_prerequisite_count[slot]) {
                const std::int32_t dependency = row.development_prerequisite_module_slot[index];
                if (row.development_prerequisite_homology_index[index] < 0 || dependency < 0
                    || dependency >= module_limit || dependency == slot
                    || row.module_occupied[dependency] == 0U
                    || row.module_type_code[dependency] != kGenericModule
                    || row.module_homology_id_index[dependency]
                        != row.development_prerequisite_homology_index[index]) {
                    return EIEL_MODULAR_RECURRENT_INVALID_PLANE_V1;
                }
            } else if (row.development_prerequisite_homology_index[index] != -1
                       || row.development_prerequisite_module_slot[index] != -1) {
                return EIEL_MODULAR_RECURRENT_INVALID_PLANE_V1;
            }
        }
        for (std::int32_t stage = 0; stage < 3; ++stage) {
            const float gain = row.development_stage_gains[slot * 3 + stage];
            if (!finite(gain) || gain < 0.0F || gain > 1.0F) {
                return EIEL_MODULAR_RECURRENT_INVALID_PLANE_V1;
            }
        }
        if (!finite(row.development_effective_gain[slot])
            || row.development_effective_gain[slot] < 0.0F
            || row.development_effective_gain[slot] > 1.0F
            || (row.development_effective_gain_present[slot] == 0U
                && row.development_effective_gain[slot] != 0.0F)
            || (row.development_latch_present[slot] == 0U
                && row.development_latch_state[slot] != 0U)) {
            return EIEL_MODULAR_RECURRENT_INVALID_PLANE_V1;
        }
    }
    if (occupied_modules != row.module_count) return EIEL_MODULAR_RECURRENT_INVALID_PLANE_V1;
    for (std::int32_t order = 0; order < kModules; ++order) {
        const std::int32_t execution = row.module_execution_order[order];
        const std::int32_t dependency = row.dependency_topological_order[order];
        if (order < generic_modules) {
            if (execution < 0 || execution >= module_limit || dependency < 0
                || dependency >= module_limit || row.module_occupied[execution] == 0U
                || row.module_type_code[execution] != kGenericModule
                || row.module_occupied[dependency] == 0U
                || row.module_type_code[dependency] != kGenericModule
                || execution_seen[execution] || dependency_seen[dependency]) {
                return EIEL_MODULAR_RECURRENT_INVALID_ORDER_V1;
            }
            execution_seen[execution] = true;
            dependency_seen[dependency] = true;
            for (std::int32_t binding = 0;
                 binding < row.development_prerequisite_count[dependency]; ++binding) {
                const std::int32_t required = row.development_prerequisite_module_slot[
                    dependency * kBindings + binding];
                if (!dependency_seen[required]) return EIEL_MODULAR_RECURRENT_INVALID_ORDER_V1;
            }
        } else if (execution != -1 || dependency != -1) {
            return EIEL_MODULAR_RECURRENT_INVALID_ORDER_V1;
        }
    }
    std::int32_t occupied_connections = 0;
    bool connection_order_seen[kConnections]{};
    for (std::int32_t slot = 0; slot < kConnections; ++slot) {
        if (!binary(row.connection_occupied[slot])) {
            return EIEL_MODULAR_RECURRENT_INVALID_PLANE_V1;
        }
        if (slot >= connection_limit && row.connection_occupied[slot] != 0U) {
            return EIEL_MODULAR_RECURRENT_CAPACITY_EXCEEDED_V1;
        }
        if (row.connection_occupied[slot] == 0U) {
            if (!empty_connection_slot(row, slot)) {
                return EIEL_MODULAR_RECURRENT_INVALID_PLANE_V1;
            }
            continue;
        }
        ++occupied_connections;
        const std::int32_t source = row.connection_src_module_slot[slot];
        const std::int32_t destination = row.connection_dst_module_slot[slot];
        const std::int32_t order = row.connection_execution_order[slot];
        if (row.agent_occupied == 0U || row.connection_local_id_index[slot] < 0
            || row.connection_src_slot_index[slot] < 0
            || row.connection_dst_slot_index[slot] < 0
            || row.connection_value_type_index[slot] < 0
            || row.connection_channel_index[slot] < 0 || source < 0 || source >= module_limit
            || destination < 0 || destination >= module_limit
            || row.module_occupied[source] == 0U || row.module_occupied[destination] == 0U
            || order < 0 || order >= row.connection_count || connection_order_seen[order]
            || !finite(row.connection_weight[slot])) {
            return EIEL_MODULAR_RECURRENT_INVALID_PLANE_V1;
        }
        connection_order_seen[order] = true;
    }
    if (occupied_connections != row.connection_count) {
        return EIEL_MODULAR_RECURRENT_INVALID_PLANE_V1;
    }
    for (std::int32_t order = 0; order < occupied_connections; ++order) {
        if (!connection_order_seen[order]) return EIEL_MODULAR_RECURRENT_INVALID_ORDER_V1;
    }
    eiel_modular_recurrent_cost_capacity_receipt_v1 expected{};
    status = compute_cost(row, &expected, profile);
    if (status != EIEL_MODULAR_RECURRENT_SUCCESS_V1) return status;
    for (std::int32_t index = 0; index < kCostFields; ++index) {
        if (row.structural_cost_counts[index] < 0
            || row.structural_cost_counts[index] != expected.counts[index]
            || !finite(row.structural_cost_components[index])
            || row.structural_cost_components[index] != expected.components[index]) {
            return EIEL_MODULAR_RECURRENT_INVALID_PLANE_V1;
        }
    }
    if (!finite(row.structural_cost_total) || row.structural_cost_total != expected.total) {
        return EIEL_MODULAR_RECURRENT_INVALID_PLANE_V1;
    }
    return EIEL_MODULAR_RECURRENT_SUCCESS_V1;
}

[[nodiscard]] Status validate_owners(
    const Row* const rows, const std::int64_t row_count,
    const eiel_modular_recurrent_const_owner_view_v1* const owners) noexcept {
    if (owners == nullptr || owners->slot_to_agent_id == nullptr || owners->slot_tokens == nullptr
        || owners->alive_mask == nullptr) {
        return EIEL_MODULAR_RECURRENT_NULL_POINTER_V1;
    }
    if (owners->row_count != row_count) return EIEL_MODULAR_RECURRENT_EXTENT_MISMATCH_V1;
    for (std::int64_t row = 0; row < row_count; ++row) {
        if (!binary(owners->alive_mask[row])
            || owners->alive_mask[row] != rows[row].agent_occupied
            || owners->slot_to_agent_id[row] != rows[row].agent_id
            || (owners->alive_mask[row] != 0U
                && owners->slot_tokens[row] != rows[row].agent_slot_token)) {
            return EIEL_MODULAR_RECURRENT_OWNER_MISMATCH_V1;
        }
        if (owners->alive_mask[row] == 0U && owners->slot_to_agent_id[row] != -1) {
            return EIEL_MODULAR_RECURRENT_OWNER_MISMATCH_V1;
        }
    }
    return EIEL_MODULAR_RECURRENT_SUCCESS_V1;
}

[[nodiscard]] Status validate_bank(
    const Row* const rows, const std::int64_t row_count, const std::uint32_t profile,
    const eiel_modular_recurrent_const_owner_view_v1* const owners) noexcept {
    if (rows == nullptr) return EIEL_MODULAR_RECURRENT_NULL_POINTER_V1;
    if (row_count <= 0) return EIEL_MODULAR_RECURRENT_EXTENT_MISMATCH_V1;
    for (std::int64_t row = 0; row < row_count; ++row) {
        const Status status = validate_row(rows[row], profile);
        if (status != EIEL_MODULAR_RECURRENT_SUCCESS_V1) return status;
        if (rows[row].agent_occupied == 0U) continue;
        for (std::int64_t other = 0; other < row; ++other) {
            if (rows[other].agent_occupied != 0U && rows[other].agent_id == rows[row].agent_id) {
                return EIEL_MODULAR_RECURRENT_OWNER_MISMATCH_V1;
            }
        }
    }
    return validate_owners(rows, row_count, owners);
}

[[nodiscard]] eiel_modular_recurrent_const_owner_view_v1 const_owner(
    const eiel_modular_recurrent_owner_view_v1& owner) noexcept {
    return {owner.slot_to_agent_id, owner.slot_tokens, owner.alive_mask, owner.row_count};
}

[[nodiscard]] Status validate_mutations(
    Row* const candidate, const eiel_modular_recurrent_mutation_v1* const mutations,
    const std::int64_t mutation_count) noexcept {
    if (mutation_count < 0 || (mutation_count > 0 && mutations == nullptr)) {
        return EIEL_MODULAR_RECURRENT_INVALID_MUTATION_V1;
    }
    for (std::int64_t index = 0; index < mutation_count; ++index) {
        const auto& mutation = mutations[index];
        if (!finite(mutation.additive_delta)) return EIEL_MODULAR_RECURRENT_INVALID_MUTATION_V1;
        float* target = nullptr;
        if (mutation.field == EIEL_MODULAR_MUTATE_CONNECTION_WEIGHT_V1) {
            if (mutation.outer_index < 0 || mutation.outer_index >= kConnections
                || mutation.inner_index != 0
                || candidate->connection_occupied[mutation.outer_index] == 0U) {
                return EIEL_MODULAR_RECURRENT_INVALID_MUTATION_V1;
            }
            target = &candidate->connection_weight[mutation.outer_index];
        } else {
            if (mutation.outer_index < 0 || mutation.outer_index >= kModules
                || candidate->module_occupied[mutation.outer_index] == 0U
                || candidate->module_type_code[mutation.outer_index] != kGenericModule) {
                return EIEL_MODULAR_RECURRENT_INVALID_MUTATION_V1;
            }
            std::int32_t width = 0;
            float* base = nullptr;
            switch (mutation.field) {
                case EIEL_MODULAR_MUTATE_INTEGRATOR_PARAMETER_V1:
                    width = 4; base = candidate->integrator_parameters; break;
                case EIEL_MODULAR_MUTATE_GATED_PARAMETER_V1:
                    width = 10; base = candidate->gated_parameters; break;
                case EIEL_MODULAR_MUTATE_INPUT_GAIN_V1:
                    width = 2; base = candidate->input_gain; break;
                case EIEL_MODULAR_MUTATE_OUTPUT_GAIN_V1:
                    width = 2; base = candidate->output_gain; break;
                case EIEL_MODULAR_MUTATE_DEVELOPMENT_GAIN_V1:
                    width = 3; base = candidate->development_stage_gains; break;
                default: return EIEL_MODULAR_RECURRENT_INVALID_MUTATION_V1;
            }
            if (mutation.inner_index < 0 || mutation.inner_index >= width) {
                return EIEL_MODULAR_RECURRENT_INVALID_MUTATION_V1;
            }
            target = &base[mutation.outer_index * width + mutation.inner_index];
        }
        *target += mutation.additive_delta;
        if (!finite(*target)) return EIEL_MODULAR_RECURRENT_INVALID_MUTATION_V1;
    }
    return EIEL_MODULAR_RECURRENT_SUCCESS_V1;
}

[[nodiscard]] Status execute_row_tick(
    Row* const row, const float* const observation, const float fatigue,
    const std::int64_t stage, float* const residuals,
    eiel_modular_recurrent_tick_receipt_v1* const receipt) noexcept {
    if (row->agent_occupied == 0U) {
        for (std::int32_t action = 0; action < kActions; ++action) residuals[action] = 0.0F;
        return EIEL_MODULAR_RECURRENT_SUCCESS_V1;
    }
    float effective_gain[kModules]{};
    bool dependency_ready[kModules]{};
    const std::uint8_t old_latch_present[kModules] = {
        row->development_latch_present[0], row->development_latch_present[1],
        row->development_latch_present[2], row->development_latch_present[3],
        row->development_latch_present[4], row->development_latch_present[5],
        row->development_latch_present[6], row->development_latch_present[7],
        row->development_latch_present[8], row->development_latch_present[9],
        row->development_latch_present[10], row->development_latch_present[11],
        row->development_latch_present[12], row->development_latch_present[13],
        row->development_latch_present[14], row->development_latch_present[15]};
    const std::uint8_t old_latch_state[kModules] = {
        row->development_latch_state[0], row->development_latch_state[1],
        row->development_latch_state[2], row->development_latch_state[3],
        row->development_latch_state[4], row->development_latch_state[5],
        row->development_latch_state[6], row->development_latch_state[7],
        row->development_latch_state[8], row->development_latch_state[9],
        row->development_latch_state[10], row->development_latch_state[11],
        row->development_latch_state[12], row->development_latch_state[13],
        row->development_latch_state[14], row->development_latch_state[15]};
    std::int32_t generic_count = 0;
    for (std::int32_t slot = 0; slot < kModules; ++slot) {
        generic_count += row->module_occupied[slot] != 0U
            && row->module_type_code[slot] == kGenericModule ? 1 : 0;
    }
    for (std::int32_t order = 0; order < generic_count; ++order) {
        const std::int32_t slot = row->dependency_topological_order[order];
        bool ready = true;
        for (std::int32_t binding = 0;
             binding < row->development_prerequisite_count[slot]; ++binding) {
            const std::int32_t dependency = row->development_prerequisite_module_slot[
                slot * kBindings + binding];
            ready = ready && effective_gain[dependency] > 0.0F;
        }
        const bool previously_latched = old_latch_present[slot] != 0U
            && old_latch_state[slot] != 0U;
        const bool gate_ready = ready || previously_latched;
        const bool next_latched = row->development_latch_enabled[slot] != 0U
            && (previously_latched || ready);
        effective_gain[slot] = gate_ready
            ? row->development_stage_gains[slot * 3 + static_cast<std::int32_t>(stage)] : 0.0F;
        dependency_ready[slot] = gate_ready;
        row->development_latch_present[slot] = next_latched ? 1U : 0U;
        row->development_latch_state[slot] = next_latched ? 1U : 0U;
        row->development_effective_gain_present[slot] = 1U;
        row->development_effective_gain[slot] = effective_gain[slot];
    }
    float routed_observation[kModules]{};
    for (std::int32_t slot = 0; slot < kModules; ++slot) {
        if (row->module_occupied[slot] == 0U || row->module_type_code[slot] != kGenericModule) {
            continue;
        }
        float routed = 0.0F;
        for (std::int32_t binding = 0; binding < row->input_binding_count[slot]; ++binding) {
            const std::int32_t index = slot * kBindings + binding;
            if (row->input_enabled[index] == 0U) continue;
            const float source = observation[25 + row->input_semantic_code[index]];
            routed += row->input_gain[index] * source;
        }
        routed_observation[slot] = clipped(routed, -1.0F, 1.0F);
    }
    float routed_previous[kModules]{};
    std::int64_t evaluated_connections = 0;
    for (std::int32_t order = 0; order < row->connection_count; ++order) {
        std::int32_t connection = -1;
        for (std::int32_t slot = 0; slot < kConnections; ++slot) {
            if (row->connection_occupied[slot] != 0U
                && row->connection_execution_order[slot] == order) {
                connection = slot;
                break;
            }
        }
        const std::int32_t source = row->connection_src_module_slot[connection];
        const std::int32_t destination = row->connection_dst_module_slot[connection];
        const bool source_available = row->module_type_code[source] == kGenericModule
            && row->module_dormant[source] == 0U;
        const bool destination_available = row->module_type_code[destination] == kGenericModule
            && row->module_dormant[destination] == 0U && effective_gain[destination] > 0.0F;
        if (source_available && destination_available) {
            const float previous = effective_gain[source] > 0.0F
                ? row->module_previous_output[source] : 0.0F;
            routed_previous[destination] += row->connection_weight[connection] * previous;
            ++evaluated_connections;
        }
    }
    float next_output[kModules]{};
    std::int64_t executing_modules = 0;
    std::int64_t changing_modules = 0;
    for (std::int32_t order = 0; order < generic_count; ++order) {
        const std::int32_t slot = row->module_execution_order[order];
        const float before = row->module_state_present[slot] != 0U ? row->module_state[slot] : 0.0F;
        const float external = routed_observation[slot] == 0.0F
            ? fatigue : fatigue + routed_observation[slot];
        const bool executing = row->module_dormant[slot] == 0U && effective_gain[slot] > 0.0F;
        float after = before;
        float output = 0.0F;
        if (executing && row->operator_code[slot] == kIntegrator) {
            const float* const p = &row->integrator_parameters[slot * 4];
            after = clipped(p[1] * before + p[2] * external + routed_previous[slot], -1.0F, 1.0F);
            output = clipped(p[0] * p[3] * after, -1.0F, 1.0F);
            if (effective_gain[slot] != 1.0F) {
                output = clipped(effective_gain[slot] * output, -1.0F, 1.0F);
            }
        } else if (executing && row->operator_code[slot] == kGated) {
            const float* const p = &row->gated_parameters[slot * 10];
            const float aggregate = clipped(external + routed_previous[slot], -1.0F, 1.0F);
            const float reset = sigmoid(p[0] + p[1] * aggregate + p[2] * before);
            const float update = sigmoid(p[3] + p[4] * aggregate + p[5] * before);
            const float candidate = std::tanh(clipped(
                p[6] + p[7] * aggregate + p[8] * (reset * before), -16.0F, 16.0F));
            after = clipped((1.0F - update) * before + update * candidate, -1.0F, 1.0F);
            const float base = clipped(p[9] * after, -1.0F, 1.0F);
            output = clipped(row->integrator_parameters[slot * 4] * base, -1.0F, 1.0F);
            if (effective_gain[slot] != 1.0F) {
                output = clipped(effective_gain[slot] * output, -1.0F, 1.0F);
            }
        }
        row->module_state_present[slot] = 1U;
        row->module_state[slot] = after;
        row->module_output_present[slot] = 1U;
        row->module_previous_output[slot] = output;
        next_output[slot] = output;
        executing_modules += executing ? 1 : 0;
        changing_modules += after != before ? 1 : 0;
    }
    float legacy_rest = 0.0F;
    float routed_action[kActions]{};
    for (std::int32_t order = 0; order < generic_count; ++order) {
        const std::int32_t slot = row->module_execution_order[order];
        legacy_rest += next_output[slot];
        for (std::int32_t binding = 0; binding < row->output_binding_count[slot]; ++binding) {
            const std::int32_t index = slot * kBindings + binding;
            if (row->output_enabled[index] != 0U) {
                routed_action[row->output_action_index[index]] += row->output_gain[index]
                    * next_output[slot];
            }
        }
    }
    legacy_rest = clipped(legacy_rest, -0.25F, 0.25F);
    for (std::int32_t action = 0; action < kActions; ++action) {
        residuals[action] = clipped(routed_action[action], -0.25F, 0.25F);
    }
    residuals[kRestAction] = routed_action[kRestAction] == 0.0F
        ? legacy_rest : clipped(legacy_rest + routed_action[kRestAction], -0.25F, 0.25F);
    receipt->executing_modules += executing_modules;
    receipt->changing_state_modules += changing_modules;
    (void)evaluated_connections;
    (void)dependency_ready;
    return EIEL_MODULAR_RECURRENT_SUCCESS_V1;
}

[[nodiscard]] bool valid_index(const std::int64_t index, const std::int64_t count) noexcept {
    return index >= 0 && index < count;
}

[[nodiscard]] bool unique_agent_id_after_replace(
    const Row* const rows, const std::int64_t row_count,
    const std::int64_t destination, const std::int64_t agent_id) noexcept {
    for (std::int64_t row = 0; row < row_count; ++row) {
        if (row != destination && rows[row].agent_occupied != 0U
            && rows[row].agent_id == agent_id) {
            return false;
        }
    }
    return true;
}

[[nodiscard]] bool canonical_binding_is_valid(
    const eiel_modular_recurrent_canonical_receipt_binding_v1* const binding,
    const eiel_modular_recurrent_owner_view_v1* const owners,
    const std::int64_t row_count, const std::int64_t selected_parent_row,
    const std::int64_t expected_parent_token, const std::int64_t child_row,
    const std::int64_t child_agent_id, const std::int64_t publication_token) noexcept {
    if (binding == nullptr || owners == nullptr
        || (binding->binding_version != 1U && binding->binding_version != 2U
            && binding->binding_version != 3U && binding->binding_version != 4U)
        || binding->canonical_receipt_contract_version != 1U
        || !digest_words_present(binding->canonical_receipt_sha256_words)
        || !digest_words_present(binding->canonical_receipt_bytes_sha256_words)
        || (binding->binding_version >= 2U
            && !digest_words_present(binding->structural_payload_sha256_words))
        || binding->transaction_ordinal < 0 || binding->selected_instance_id_index < -1
        || binding->selected_homology_id_index < -1 || binding->operation_code < 0
        || binding->status_code < 0 || binding->reason_code < 0 || binding->changed > 1U
        || !valid_index(selected_parent_row, row_count) || !valid_index(child_row, row_count)
        || owners->alive_mask[selected_parent_row] == 0U
        || owners->slot_to_agent_id[selected_parent_row] != binding->parent_agent_id
        || owners->slot_tokens[selected_parent_row] != expected_parent_token
        || binding->parent_slot != selected_parent_row
        || binding->parent_slot_token != expected_parent_token
        || binding->child_agent_id != child_agent_id || binding->child_slot != child_row
        || binding->child_slot_token != publication_token) {
        return false;
    }
    for (const auto value : binding->reserved) {
        if (value != 0U) return false;
    }
    for (const auto value : binding->reserved_tail) {
        if (value != 0U) return false;
    }
    if (binding->binding_version >= 4U
        && (binding->native_receipt_construction_required != 1U
            || binding->changed_dimension_code < 0)) {
        return false;
    }
    return true;
}

}  // namespace

extern "C" {

uint32_t eiel_modular_recurrent_cpu_oracle_abi_version_v1(void) {
    return EIEL_MODULAR_RECURRENT_CPU_ORACLE_ABI_VERSION_V1;
}

int32_t eiel_modular_recurrent_cpu_oracle_plane_count_v1(void) {
    return EIEL_MODULAR_RECURRENT_PLANE_COUNT_V1;
}

int64_t eiel_modular_recurrent_cpu_oracle_row_size_v1(void) {
    return static_cast<int64_t>(sizeof(Row));
}

int32_t eiel_modular_recurrent_row_structural_payload_sha256_v1(
    const Row* const row, std::uint64_t digest_words_out[4]) {
    if (row == nullptr || digest_words_out == nullptr) {
        return EIEL_MODULAR_RECURRENT_NULL_POINTER_V1;
    }
    std::uint64_t words[4]{};
    structural_payload_sha256_words(*row, words);
    for (std::size_t index = 0; index < 4U; ++index) {
        digest_words_out[index] = words[index];
    }
    return EIEL_MODULAR_RECURRENT_SUCCESS_V1;
}

int32_t eiel_modular_recurrent_canonical_receipt_sha256_v1(
    const std::uint32_t profile,
    const eiel_modular_recurrent_canonical_receipt_source_v1* const source,
    const eiel_modular_recurrent_canonical_receipt_binding_v1* const binding,
    std::uint64_t canonical_receipt_sha256_words[4],
    std::uint64_t canonical_receipt_bytes_sha256_words[4]) {
    if (source == nullptr || binding == nullptr
        || canonical_receipt_sha256_words == nullptr
        || canonical_receipt_bytes_sha256_words == nullptr) {
        return EIEL_MODULAR_RECURRENT_NULL_POINTER_V1;
    }
    std::uint64_t canonical_words[4]{};
    std::uint64_t canonical_bytes_words[4]{};
    if (!native_canonical_receipt_sha256_words(
            profile, *source, *binding, canonical_words, canonical_bytes_words)) {
        return EIEL_MODULAR_RECURRENT_OWNER_MISMATCH_V1;
    }
    for (std::size_t index = 0; index < 4U; ++index) {
        canonical_receipt_sha256_words[index] = canonical_words[index];
        canonical_receipt_bytes_sha256_words[index] = canonical_bytes_words[index];
    }
    return EIEL_MODULAR_RECURRENT_SUCCESS_V1;
}

int32_t eiel_modular_recurrent_validate_v1(
    const Row* const rows, const int64_t row_count, const uint32_t profile,
    const eiel_modular_recurrent_const_owner_view_v1* const owners) {
    return validate_bank(rows, row_count, profile, owners);
}

int32_t eiel_modular_recurrent_tick_v1(
    Row* const rows, const int64_t row_count, const uint32_t profile,
    const eiel_modular_recurrent_const_owner_view_v1* const owners,
    const float* const observations, const int64_t observation_element_count,
    const float* const fatigue, const int64_t fatigue_count,
    const int64_t* const stage_indices, const int64_t stage_count,
    Row* const scratch_rows, const int64_t scratch_row_count,
    float* const scratch_action_residuals, const int64_t scratch_action_count,
    float* const action_residuals, const int64_t action_count,
    eiel_modular_recurrent_tick_receipt_v1* const receipt_out) {
    if (rows == nullptr || observations == nullptr || fatigue == nullptr
        || stage_indices == nullptr || scratch_rows == nullptr
        || scratch_action_residuals == nullptr || action_residuals == nullptr
        || receipt_out == nullptr) {
        return EIEL_MODULAR_RECURRENT_NULL_POINTER_V1;
    }
    if (row_count <= 0
        || row_count > std::numeric_limits<std::int64_t>::max() / kObservationWidth
        || row_count > std::numeric_limits<std::int64_t>::max() / kActions) {
        return EIEL_MODULAR_RECURRENT_EXTENT_MISMATCH_V1;
    }
    const std::int64_t required_observations = row_count * kObservationWidth;
    const std::int64_t required_actions = row_count * kActions;
    if (observation_element_count != required_observations || fatigue_count != row_count
        || stage_count != row_count || scratch_row_count < row_count
        || scratch_action_count < required_actions || action_count < required_actions) {
        return EIEL_MODULAR_RECURRENT_EXTENT_MISMATCH_V1;
    }
    Status status = validate_bank(rows, row_count, profile, owners);
    if (status != EIEL_MODULAR_RECURRENT_SUCCESS_V1) return status;
    std::size_t row_bytes = 0U;
    std::size_t observation_bytes = 0U;
    std::size_t fatigue_bytes = 0U;
    std::size_t stage_bytes = 0U;
    std::size_t action_bytes = 0U;
    std::size_t owner_id_bytes = 0U;
    std::size_t owner_alive_bytes = 0U;
    if (!checked_bytes(row_count, sizeof(Row), &row_bytes)
        || !checked_bytes(required_observations, sizeof(float), &observation_bytes)
        || !checked_bytes(row_count, sizeof(float), &fatigue_bytes)
        || !checked_bytes(row_count, sizeof(std::int64_t), &stage_bytes)
        || !checked_bytes(required_actions, sizeof(float), &action_bytes)
        || !checked_bytes(row_count, sizeof(std::int64_t), &owner_id_bytes)
        || !checked_bytes(row_count, sizeof(std::uint8_t), &owner_alive_bytes)) {
        return EIEL_MODULAR_RECURRENT_EXTENT_MISMATCH_V1;
    }
    BufferRange ranges[11]{};
    const bool ranges_valid =
        make_range(rows, row_bytes, true, &ranges[0])
        && make_range(scratch_rows, row_bytes, true, &ranges[1])
        && make_range(scratch_action_residuals, action_bytes, true, &ranges[2])
        && make_range(action_residuals, action_bytes, true, &ranges[3])
        && make_range(receipt_out, sizeof(*receipt_out), true, &ranges[4])
        && make_range(observations, observation_bytes, false, &ranges[5])
        && make_range(fatigue, fatigue_bytes, false, &ranges[6])
        && make_range(stage_indices, stage_bytes, false, &ranges[7])
        && make_range(owners->slot_to_agent_id, owner_id_bytes, false, &ranges[8])
        && make_range(owners->slot_tokens, owner_id_bytes, false, &ranges[9])
        && make_range(owners->alive_mask, owner_alive_bytes, false, &ranges[10]);
    if (!ranges_valid) return EIEL_MODULAR_RECURRENT_EXTENT_MISMATCH_V1;
    for (std::size_t left = 0U; left < 11U; ++left) {
        for (std::size_t right = left + 1U; right < 11U; ++right) {
            if ((ranges[left].writable || ranges[right].writable)
                && ranges_overlap(ranges[left], ranges[right])) {
                return EIEL_MODULAR_RECURRENT_EXTENT_MISMATCH_V1;
            }
        }
    }
    for (std::int64_t row = 0; row < row_count; ++row) {
        if (!finite(fatigue[row]) || fatigue[row] < 0.0F || fatigue[row] > 1.0F
            || stage_indices[row] < 0 || stage_indices[row] > 2) {
            return !finite(fatigue[row]) ? EIEL_MODULAR_RECURRENT_NONFINITE_V1
                                        : EIEL_MODULAR_RECURRENT_INVALID_PLANE_V1;
        }
        for (std::int32_t column = 0; column < kObservationWidth; ++column) {
            if (!finite(observations[row * kObservationWidth + column])) {
                return EIEL_MODULAR_RECURRENT_NONFINITE_V1;
            }
        }
    }
    eiel_modular_recurrent_tick_receipt_v1 receipt{};
    for (std::int64_t row = 0; row < row_count; ++row) {
        scratch_rows[row] = rows[row];
        if (rows[row].agent_occupied != 0U) {
            std::int64_t generic = 0;
            for (std::int32_t slot = 0; slot < kModules; ++slot) {
                generic += rows[row].module_occupied[slot] != 0U
                    && rows[row].module_type_code[slot] == kGenericModule ? 1 : 0;
            }
            receipt.structured_agents += generic > 0 ? 1 : 0;
            receipt.neutral_agents += generic == 0 ? 1 : 0;
        }
        status = execute_row_tick(
            &scratch_rows[row], observations + row * kObservationWidth, fatigue[row],
            stage_indices[row], scratch_action_residuals + row * kActions, &receipt);
        if (status != EIEL_MODULAR_RECURRENT_SUCCESS_V1) return status;
        status = validate_row(scratch_rows[row], profile);
        if (status != EIEL_MODULAR_RECURRENT_SUCCESS_V1) return status;
    }
    std::memcpy(rows, scratch_rows, static_cast<std::size_t>(row_count) * sizeof(Row));
    std::memcpy(action_residuals, scratch_action_residuals,
                static_cast<std::size_t>(row_count * kActions) * sizeof(float));
    receipt.committed_rows = row_count;
    receipt.rng_draws = 0;
    receipt.production_routing_qualified = 0U;
    receipt.publication_qualified = 0U;
    *receipt_out = receipt;
    return EIEL_MODULAR_RECURRENT_SUCCESS_V1;
}

int32_t eiel_modular_recurrent_founder_init_v1(
    Row* const rows, const int64_t row_count, const uint32_t profile,
    eiel_modular_recurrent_owner_view_v1* const owners, const int64_t destination_row,
    const int64_t expected_destination_token, const int64_t founder_agent_id,
    const int64_t publication_token, const Row* const founder_structure) {
    if (rows == nullptr || owners == nullptr || founder_structure == nullptr) {
        return EIEL_MODULAR_RECURRENT_NULL_POINTER_V1;
    }
    const auto view = const_owner(*owners);
    Status status = validate_bank(rows, row_count, profile, &view);
    if (status != EIEL_MODULAR_RECURRENT_SUCCESS_V1) return status;
    if (!valid_index(destination_row, row_count) || owners->alive_mask[destination_row] != 0U
        || owners->slot_tokens[destination_row] != expected_destination_token
        || founder_agent_id < 0 || publication_token < 0
        || !unique_agent_id_after_replace(rows, row_count, destination_row, founder_agent_id)) {
        return EIEL_MODULAR_RECURRENT_OWNER_MISMATCH_V1;
    }
    Row candidate = *founder_structure;
    candidate.agent_occupied = 1U;
    candidate.agent_id = founder_agent_id;
    candidate.agent_slot_token = publication_token;
    reset_runtime(&candidate);
    status = validate_row(candidate, profile);
    if (status != EIEL_MODULAR_RECURRENT_SUCCESS_V1) return status;
    rows[destination_row] = candidate;
    owners->slot_to_agent_id[destination_row] = founder_agent_id;
    owners->slot_tokens[destination_row] = publication_token;
    owners->alive_mask[destination_row] = 1U;
    return EIEL_MODULAR_RECURRENT_SUCCESS_V1;
}

int32_t eiel_modular_recurrent_newborn_inherit_v1(
    Row* const rows, const int64_t row_count, const uint32_t profile,
    eiel_modular_recurrent_owner_view_v1* const owners, const int64_t destination_row,
    const int64_t expected_destination_token, const int64_t parent_a_row,
    const int64_t expected_parent_a_token, const int64_t parent_b_row,
    const int64_t expected_parent_b_token, const float donor_draw,
    const int64_t child_agent_id, const int64_t publication_token,
    const eiel_modular_recurrent_mutation_v1* const mutations,
    const int64_t mutation_count) {
    if (rows == nullptr || owners == nullptr) return EIEL_MODULAR_RECURRENT_NULL_POINTER_V1;
    const auto view = const_owner(*owners);
    Status status = validate_bank(rows, row_count, profile, &view);
    if (status != EIEL_MODULAR_RECURRENT_SUCCESS_V1) return status;
    if (!valid_index(destination_row, row_count) || !valid_index(parent_a_row, row_count)
        || !valid_index(parent_b_row, row_count) || owners->alive_mask[destination_row] != 0U
        || owners->slot_tokens[destination_row] != expected_destination_token
        || owners->alive_mask[parent_a_row] == 0U || owners->alive_mask[parent_b_row] == 0U
        || owners->slot_tokens[parent_a_row] != expected_parent_a_token
        || owners->slot_tokens[parent_b_row] != expected_parent_b_token
        || !finite(donor_draw) || donor_draw < 0.0F || donor_draw >= 1.0F
        || child_agent_id < 0 || publication_token < 0
        || !unique_agent_id_after_replace(rows, row_count, destination_row, child_agent_id)) {
        return EIEL_MODULAR_RECURRENT_OWNER_MISMATCH_V1;
    }
    Row candidate = donor_draw < 0.5F ? rows[parent_a_row] : rows[parent_b_row];
    candidate.agent_id = child_agent_id;
    candidate.agent_slot_token = publication_token;
    reset_runtime(&candidate);
    status = validate_mutations(&candidate, mutations, mutation_count);
    if (status != EIEL_MODULAR_RECURRENT_SUCCESS_V1) return status;
    status = validate_row(candidate, profile);
    if (status != EIEL_MODULAR_RECURRENT_SUCCESS_V1) return status;
    rows[destination_row] = candidate;
    owners->slot_to_agent_id[destination_row] = child_agent_id;
    owners->slot_tokens[destination_row] = publication_token;
    owners->alive_mask[destination_row] = 1U;
    return EIEL_MODULAR_RECURRENT_SUCCESS_V1;
}

int32_t eiel_modular_recurrent_newborn_inherit_with_receipt_v1(
    Row* const rows, const int64_t row_count, const uint32_t profile,
    eiel_modular_recurrent_owner_view_v1* const owners, const int64_t destination_row,
    const int64_t expected_destination_token, const int64_t parent_a_row,
    const int64_t expected_parent_a_token, const int64_t parent_b_row,
    const int64_t expected_parent_b_token, const float donor_draw,
    const int64_t child_agent_id, const int64_t publication_token,
    const eiel_modular_recurrent_mutation_v1* const mutations,
    const int64_t mutation_count,
    eiel_modular_recurrent_birth_receipt_v1* const receipt_out) {
    if (receipt_out == nullptr) return EIEL_MODULAR_RECURRENT_NULL_POINTER_V1;
    eiel_modular_recurrent_birth_receipt_v1 receipt{};
    receipt.receipt_version = 1U;
    receipt.lifecycle_status = EIEL_MODULAR_RECURRENT_NULL_POINTER_V1;
    receipt.destination_row = destination_row;
    receipt.parent_a_row = parent_a_row;
    receipt.parent_b_row = parent_b_row;
    receipt.selected_parent_row = donor_draw < 0.5F ? parent_a_row : parent_b_row;
    receipt.expected_destination_token = expected_destination_token;
    receipt.expected_parent_a_token = expected_parent_a_token;
    receipt.expected_parent_b_token = expected_parent_b_token;
    receipt.child_agent_id = child_agent_id;
    receipt.publication_token = publication_token;
    receipt.donor_draw_float32_bits = float32_bits(donor_draw);
    receipt.mutation_count = mutation_count;

    if (rows != nullptr && owners != nullptr && valid_index(parent_a_row, row_count)
        && valid_index(parent_b_row, row_count)) {
        receipt.parent_a_agent_id = owners->slot_to_agent_id[parent_a_row];
        receipt.parent_b_agent_id = owners->slot_to_agent_id[parent_b_row];
    } else {
        receipt.parent_a_agent_id = -1;
        receipt.parent_b_agent_id = -1;
    }
    receipt.owner_checked = 1U;

    const int32_t status = eiel_modular_recurrent_newborn_inherit_v1(
        rows, row_count, profile, owners, destination_row, expected_destination_token,
        parent_a_row, expected_parent_a_token, parent_b_row, expected_parent_b_token,
        donor_draw, child_agent_id, publication_token, mutations, mutation_count);
    receipt.lifecycle_status = status;
    if (status == EIEL_MODULAR_RECURRENT_SUCCESS_V1) {
        receipt.committed = 1U;
        receipt.parent_preserved = 1U;
        receipt.child_runtime_reset = 1U;
    }
    *receipt_out = receipt;
    return status;
}

int32_t eiel_modular_recurrent_newborn_inherit_with_canonical_receipt_v1(
    Row* const rows, const int64_t row_count, const uint32_t profile,
    eiel_modular_recurrent_owner_view_v1* const owners, const int64_t destination_row,
    const int64_t expected_destination_token, const int64_t parent_a_row,
    const int64_t expected_parent_a_token, const int64_t parent_b_row,
    const int64_t expected_parent_b_token, const float donor_draw,
    const int64_t child_agent_id, const int64_t publication_token,
    const eiel_modular_recurrent_mutation_v1* const mutations,
    const int64_t mutation_count,
    const eiel_modular_recurrent_canonical_receipt_source_v1* const canonical_source,
    const eiel_modular_recurrent_canonical_receipt_binding_v1* const canonical_binding,
    eiel_modular_recurrent_birth_receipt_v1* const receipt_out) {
    if (receipt_out == nullptr || canonical_binding == nullptr) {
        return EIEL_MODULAR_RECURRENT_NULL_POINTER_V1;
    }
    const std::int64_t selected_parent_row = donor_draw < 0.5F ? parent_a_row : parent_b_row;
    const std::int64_t expected_selected_token = donor_draw < 0.5F
        ? expected_parent_a_token : expected_parent_b_token;
    if (!canonical_binding_is_valid(
            canonical_binding, owners, row_count, selected_parent_row,
            expected_selected_token, destination_row, child_agent_id, publication_token)) {
        eiel_modular_recurrent_birth_receipt_v1 receipt{};
        receipt.receipt_version =
            receipt_version_for_binding(canonical_binding->binding_version);
        receipt.lifecycle_status = EIEL_MODULAR_RECURRENT_OWNER_MISMATCH_V1;
        receipt.destination_row = destination_row;
        receipt.parent_a_row = parent_a_row;
        receipt.parent_b_row = parent_b_row;
        receipt.selected_parent_row = selected_parent_row;
        receipt.expected_destination_token = expected_destination_token;
        receipt.expected_parent_a_token = expected_parent_a_token;
        receipt.expected_parent_b_token = expected_parent_b_token;
        receipt.child_agent_id = child_agent_id;
        receipt.publication_token = publication_token;
        receipt.donor_draw_float32_bits = float32_bits(donor_draw);
        receipt.mutation_count = mutation_count;
        receipt.owner_checked = 1U;
        *receipt_out = receipt;
        return EIEL_MODULAR_RECURRENT_OWNER_MISMATCH_V1;
    }
    if (rows == nullptr || owners == nullptr) {
        return EIEL_MODULAR_RECURRENT_NULL_POINTER_V1;
    }
    const Row previous_row = rows[destination_row];
    const std::int64_t previous_agent_id = owners->slot_to_agent_id[destination_row];
    const std::int64_t previous_token = owners->slot_tokens[destination_row];
    const std::uint8_t previous_alive = owners->alive_mask[destination_row];
    const int32_t status = eiel_modular_recurrent_newborn_inherit_with_receipt_v1(
        rows, row_count, profile, owners, destination_row, expected_destination_token,
        parent_a_row, expected_parent_a_token, parent_b_row, expected_parent_b_token,
        donor_draw, child_agent_id, publication_token, mutations, mutation_count,
        receipt_out);
    receipt_out->receipt_version = receipt_version_for_binding(canonical_binding->binding_version);
    if (status == EIEL_MODULAR_RECURRENT_SUCCESS_V1) {
        if (canonical_binding->binding_version >= 3U
            && !structural_payload_matches_binding(rows[destination_row], *canonical_binding)) {
            rollback_publication(
                rows, owners, destination_row, previous_row, previous_agent_id,
                previous_token, previous_alive);
            mark_structural_mismatch(receipt_out);
            return EIEL_MODULAR_RECURRENT_OWNER_MISMATCH_V1;
        }
        if (canonical_binding->binding_version >= 4U
            && !native_canonical_receipt_matches_binding(
                profile, canonical_source, *canonical_binding)) {
            rollback_publication(
                rows, owners, destination_row, previous_row, previous_agent_id,
                previous_token, previous_alive);
            mark_structural_mismatch(receipt_out);
            return EIEL_MODULAR_RECURRENT_OWNER_MISMATCH_V1;
        }
        bind_canonical_receipt(*canonical_binding, receipt_out);
    }
    return status;
}

int32_t eiel_modular_recurrent_death_cleanup_v1(
    Row* const rows, const int64_t row_count, const uint32_t profile,
    eiel_modular_recurrent_owner_view_v1* const owners, const int64_t row_index,
    const int64_t expected_agent_id, const int64_t expected_token) {
    if (rows == nullptr || owners == nullptr) return EIEL_MODULAR_RECURRENT_NULL_POINTER_V1;
    const auto view = const_owner(*owners);
    Status status = validate_bank(rows, row_count, profile, &view);
    if (status != EIEL_MODULAR_RECURRENT_SUCCESS_V1) return status;
    if (!valid_index(row_index, row_count) || owners->alive_mask[row_index] == 0U
        || owners->slot_to_agent_id[row_index] != expected_agent_id
        || owners->slot_tokens[row_index] != expected_token) {
        return EIEL_MODULAR_RECURRENT_OWNER_MISMATCH_V1;
    }
    Row candidate{};
    clear_row(&candidate);
    rows[row_index] = candidate;
    owners->slot_to_agent_id[row_index] = -1;
    owners->alive_mask[row_index] = 0U;
    return EIEL_MODULAR_RECURRENT_SUCCESS_V1;
}

int32_t eiel_modular_recurrent_same_slot_reuse_v1(
    Row* const rows, const int64_t row_count, const uint32_t profile,
    eiel_modular_recurrent_owner_view_v1* const owners, const int64_t row_index,
    const int64_t expected_agent_id, const int64_t expected_token,
    const int64_t structural_parent_row, const int64_t expected_parent_token,
    const int64_t child_agent_id, const int64_t publication_token,
    const eiel_modular_recurrent_mutation_v1* const mutations,
    const int64_t mutation_count) {
    if (rows == nullptr || owners == nullptr) return EIEL_MODULAR_RECURRENT_NULL_POINTER_V1;
    const auto view = const_owner(*owners);
    Status status = validate_bank(rows, row_count, profile, &view);
    if (status != EIEL_MODULAR_RECURRENT_SUCCESS_V1) return status;
    if (!valid_index(row_index, row_count) || !valid_index(structural_parent_row, row_count)
        || owners->alive_mask[row_index] == 0U
        || owners->slot_to_agent_id[row_index] != expected_agent_id
        || owners->slot_tokens[row_index] != expected_token
        || owners->alive_mask[structural_parent_row] == 0U
        || owners->slot_tokens[structural_parent_row] != expected_parent_token
        || child_agent_id < 0 || publication_token < 0
        || !unique_agent_id_after_replace(rows, row_count, row_index, child_agent_id)) {
        return EIEL_MODULAR_RECURRENT_OWNER_MISMATCH_V1;
    }
    Row candidate = rows[structural_parent_row];
    candidate.agent_id = child_agent_id;
    candidate.agent_slot_token = publication_token;
    reset_runtime(&candidate);
    status = validate_mutations(&candidate, mutations, mutation_count);
    if (status != EIEL_MODULAR_RECURRENT_SUCCESS_V1) return status;
    status = validate_row(candidate, profile);
    if (status != EIEL_MODULAR_RECURRENT_SUCCESS_V1) return status;
    rows[row_index] = candidate;
    owners->slot_to_agent_id[row_index] = child_agent_id;
    owners->slot_tokens[row_index] = publication_token;
    owners->alive_mask[row_index] = 1U;
    return EIEL_MODULAR_RECURRENT_SUCCESS_V1;
}

int32_t eiel_modular_recurrent_same_slot_reuse_with_receipt_v1(
    Row* const rows, const int64_t row_count, const uint32_t profile,
    eiel_modular_recurrent_owner_view_v1* const owners, const int64_t row_index,
    const int64_t expected_agent_id, const int64_t expected_token,
    const int64_t structural_parent_row, const int64_t expected_parent_token,
    const int64_t child_agent_id, const int64_t publication_token,
    const eiel_modular_recurrent_mutation_v1* const mutations,
    const int64_t mutation_count,
    eiel_modular_recurrent_birth_receipt_v1* const receipt_out) {
    if (receipt_out == nullptr) return EIEL_MODULAR_RECURRENT_NULL_POINTER_V1;
    eiel_modular_recurrent_birth_receipt_v1 receipt{};
    receipt.receipt_version = 1U;
    receipt.lifecycle_status = EIEL_MODULAR_RECURRENT_NULL_POINTER_V1;
    receipt.destination_row = row_index;
    receipt.parent_a_row = structural_parent_row;
    receipt.parent_b_row = -1;
    receipt.selected_parent_row = structural_parent_row;
    receipt.expected_destination_token = expected_token;
    receipt.expected_parent_a_token = expected_parent_token;
    receipt.expected_parent_b_token = -1;
    receipt.child_agent_id = child_agent_id;
    receipt.publication_token = publication_token;
    receipt.donor_draw_float32_bits = 0U;
    receipt.mutation_count = mutation_count;

    if (rows != nullptr && owners != nullptr && valid_index(row_index, row_count)
        && valid_index(structural_parent_row, row_count)) {
        receipt.parent_a_agent_id = owners->slot_to_agent_id[structural_parent_row];
        receipt.parent_b_agent_id = -1;
    } else {
        receipt.parent_a_agent_id = -1;
        receipt.parent_b_agent_id = -1;
    }
    receipt.owner_checked = 1U;

    const int32_t status = eiel_modular_recurrent_same_slot_reuse_v1(
        rows, row_count, profile, owners, row_index, expected_agent_id, expected_token,
        structural_parent_row, expected_parent_token, child_agent_id, publication_token,
        mutations, mutation_count);
    receipt.lifecycle_status = status;
    if (status == EIEL_MODULAR_RECURRENT_SUCCESS_V1) {
        receipt.committed = 1U;
        receipt.parent_preserved = 1U;
        receipt.child_runtime_reset = 1U;
    }
    *receipt_out = receipt;
    return status;
}

int32_t eiel_modular_recurrent_same_slot_reuse_with_canonical_receipt_v1(
    Row* const rows, const int64_t row_count, const uint32_t profile,
    eiel_modular_recurrent_owner_view_v1* const owners, const int64_t row_index,
    const int64_t expected_agent_id, const int64_t expected_token,
    const int64_t structural_parent_row, const int64_t expected_parent_token,
    const int64_t child_agent_id, const int64_t publication_token,
    const eiel_modular_recurrent_mutation_v1* const mutations,
    const int64_t mutation_count,
    const eiel_modular_recurrent_canonical_receipt_source_v1* const canonical_source,
    const eiel_modular_recurrent_canonical_receipt_binding_v1* const canonical_binding,
    eiel_modular_recurrent_birth_receipt_v1* const receipt_out) {
    if (receipt_out == nullptr || canonical_binding == nullptr) {
        return EIEL_MODULAR_RECURRENT_NULL_POINTER_V1;
    }
    (void)expected_agent_id;
    (void)expected_token;
    if (!canonical_binding_is_valid(
            canonical_binding, owners, row_count, structural_parent_row,
            expected_parent_token, row_index, child_agent_id, publication_token)) {
        eiel_modular_recurrent_birth_receipt_v1 receipt{};
        receipt.receipt_version =
            receipt_version_for_binding(canonical_binding->binding_version);
        receipt.lifecycle_status = EIEL_MODULAR_RECURRENT_OWNER_MISMATCH_V1;
        receipt.destination_row = row_index;
        receipt.parent_a_row = structural_parent_row;
        receipt.parent_b_row = -1;
        receipt.selected_parent_row = structural_parent_row;
        receipt.expected_destination_token = expected_token;
        receipt.expected_parent_a_token = expected_parent_token;
        receipt.expected_parent_b_token = -1;
        receipt.child_agent_id = child_agent_id;
        receipt.publication_token = publication_token;
        receipt.mutation_count = mutation_count;
        receipt.owner_checked = 1U;
        *receipt_out = receipt;
        return EIEL_MODULAR_RECURRENT_OWNER_MISMATCH_V1;
    }
    if (rows == nullptr || owners == nullptr) {
        return EIEL_MODULAR_RECURRENT_NULL_POINTER_V1;
    }
    const Row previous_row = rows[row_index];
    const std::int64_t previous_agent_id = owners->slot_to_agent_id[row_index];
    const std::int64_t previous_token = owners->slot_tokens[row_index];
    const std::uint8_t previous_alive = owners->alive_mask[row_index];
    const int32_t status = eiel_modular_recurrent_same_slot_reuse_with_receipt_v1(
        rows, row_count, profile, owners, row_index, expected_agent_id, expected_token,
        structural_parent_row, expected_parent_token, child_agent_id, publication_token,
        mutations, mutation_count, receipt_out);
    receipt_out->receipt_version = receipt_version_for_binding(canonical_binding->binding_version);
    if (status == EIEL_MODULAR_RECURRENT_SUCCESS_V1) {
        if (canonical_binding->binding_version >= 3U
            && !structural_payload_matches_binding(rows[row_index], *canonical_binding)) {
            rollback_publication(
                rows, owners, row_index, previous_row, previous_agent_id,
                previous_token, previous_alive);
            mark_structural_mismatch(receipt_out);
            return EIEL_MODULAR_RECURRENT_OWNER_MISMATCH_V1;
        }
        if (canonical_binding->binding_version >= 4U
            && !native_canonical_receipt_matches_binding(
                profile, canonical_source, *canonical_binding)) {
            rollback_publication(
                rows, owners, row_index, previous_row, previous_agent_id,
                previous_token, previous_alive);
            mark_structural_mismatch(receipt_out);
            return EIEL_MODULAR_RECURRENT_OWNER_MISMATCH_V1;
        }
        bind_canonical_receipt(*canonical_binding, receipt_out);
    }
    return status;
}

int32_t eiel_modular_recurrent_checkpoint_project_v1(
    const Row* const rows, const int64_t row_count, const uint32_t profile,
    const eiel_modular_recurrent_const_owner_view_v1* const owners,
    const int64_t row_index, const int64_t expected_token, Row* const checkpoint_out) {
    if (checkpoint_out == nullptr) return EIEL_MODULAR_RECURRENT_NULL_POINTER_V1;
    Status status = validate_bank(rows, row_count, profile, owners);
    if (status != EIEL_MODULAR_RECURRENT_SUCCESS_V1) return status;
    if (!valid_index(row_index, row_count) || rows[row_index].agent_occupied == 0U
        || rows[row_index].agent_slot_token != expected_token) {
        return EIEL_MODULAR_RECURRENT_OWNER_MISMATCH_V1;
    }
    *checkpoint_out = rows[row_index];
    return EIEL_MODULAR_RECURRENT_SUCCESS_V1;
}

int32_t eiel_modular_recurrent_checkpoint_restore_v1(
    const Row* const checkpoint, Row* const rows, const int64_t row_count,
    const uint32_t profile, eiel_modular_recurrent_owner_view_v1* const owners,
    const int64_t destination_row, const int64_t expected_destination_token) {
    if (checkpoint == nullptr || rows == nullptr || owners == nullptr) {
        return EIEL_MODULAR_RECURRENT_NULL_POINTER_V1;
    }
    const auto view = const_owner(*owners);
    Status status = validate_bank(rows, row_count, profile, &view);
    if (status != EIEL_MODULAR_RECURRENT_SUCCESS_V1) return status;
    if (!valid_index(destination_row, row_count)
        || owners->alive_mask[destination_row] != 0U
        || owners->slot_tokens[destination_row] != expected_destination_token
        || !unique_agent_id_after_replace(
            rows, row_count, destination_row, checkpoint->agent_id)) {
        return EIEL_MODULAR_RECURRENT_OWNER_MISMATCH_V1;
    }
    status = validate_row(*checkpoint, profile);
    if (status != EIEL_MODULAR_RECURRENT_SUCCESS_V1 || checkpoint->agent_occupied == 0U) {
        return EIEL_MODULAR_RECURRENT_INVALID_CHECKPOINT_V1;
    }
    rows[destination_row] = *checkpoint;
    owners->slot_to_agent_id[destination_row] = checkpoint->agent_id;
    owners->slot_tokens[destination_row] = checkpoint->agent_slot_token;
    owners->alive_mask[destination_row] = 1U;
    return EIEL_MODULAR_RECURRENT_SUCCESS_V1;
}

int32_t eiel_modular_recurrent_cost_capacity_v1(
    const Row* const row, const uint32_t profile,
    eiel_modular_recurrent_cost_capacity_receipt_v1* const receipt_out) {
    if (row == nullptr || receipt_out == nullptr) return EIEL_MODULAR_RECURRENT_NULL_POINTER_V1;
    const Status status = validate_row(*row, profile);
    if (status != EIEL_MODULAR_RECURRENT_SUCCESS_V1) return status;
    return compute_cost(*row, receipt_out, profile);
}

}  // extern "C"
