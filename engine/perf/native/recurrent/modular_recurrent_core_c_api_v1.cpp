#include "modular_recurrent_core_c_api_v1.h"

#include "modular_recurrent_v1.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>

namespace {

namespace modular = eiel::recurrent::modular_v1;

using Envelope = modular::FixedEnvelope;
using Status = modular::Status;

static_assert(std::is_standard_layout_v<Envelope>);
static_assert(std::is_trivially_copyable_v<Envelope>);
static_assert(std::is_standard_layout_v<eiel_modular_recurrent_core_mutation_receipt_v1>);
static_assert(std::is_trivially_copyable_v<eiel_modular_recurrent_core_mutation_receipt_v1>);
static_assert(std::is_standard_layout_v<eiel_modular_recurrent_core_ordinary_birth_receipt_v1>);
static_assert(
    std::is_trivially_copyable_v<eiel_modular_recurrent_core_ordinary_birth_receipt_v1>);
static_assert(std::is_standard_layout_v<eiel_modular_recurrent_core_ordinary_birth_receipt_v4>);
static_assert(
    std::is_trivially_copyable_v<eiel_modular_recurrent_core_ordinary_birth_receipt_v4>);

[[nodiscard]] int32_t code(const Status status) noexcept {
    return static_cast<int32_t>(status);
}

[[nodiscard]] modular::CapacityProfile profile_from_code(const uint32_t profile) noexcept {
    return profile == EIEL_MODULAR_RECURRENT_CORE_PROFILE_16X48_V1
        ? modular::CapacityProfile::kExpanded16x48
        : modular::CapacityProfile::kLegacy12x32;
}

[[nodiscard]] bool valid_profile(const uint32_t profile) noexcept {
    return profile == EIEL_MODULAR_RECURRENT_CORE_PROFILE_12X32_V1
        || profile == EIEL_MODULAR_RECURRENT_CORE_PROFILE_16X48_V1;
}

[[nodiscard]] bool valid_disabled_reason_code(const uint32_t reason_code) noexcept {
    return reason_code <= EIEL_MODULAR_RECURRENT_CORE_DISABLED_REASON_MUTATION_ATTEMPT_BUDGET_ZERO_V1;
}

[[nodiscard]] bool valid_owner_transition(
    const int64_t parent_agent_id,
    const int64_t parent_slot,
    const int64_t parent_slot_token,
    const int64_t child_agent_id,
    const int64_t child_slot,
    const int64_t expected_child_slot_token,
    const int64_t child_publication_token) noexcept {
    if (parent_agent_id < 0 || parent_slot < 0 || parent_slot_token <= 0
        || child_agent_id < 0 || child_slot < 0 || expected_child_slot_token <= 0
        || child_publication_token <= 0) {
        return false;
    }
    if (child_publication_token <= expected_child_slot_token) {
        return false;
    }
    if (parent_slot == child_slot && child_publication_token <= parent_slot_token) {
        return false;
    }
    return true;
}

[[nodiscard]] bool decode_unit_float64_bits(
    const uint64_t bits,
    float* const out) noexcept {
    if (out == nullptr) {
        return false;
    }
    double value = 0.0;
    static_assert(sizeof(value) == sizeof(bits));
    std::memcpy(&value, &bits, sizeof(value));
    if (!std::isfinite(value) || value < 0.0 || value >= 1.0) {
        return false;
    }
    *out = static_cast<float>(value);
    return std::isfinite(*out) && *out >= 0.0F && *out < 1.0F;
}

[[nodiscard]] bool publishable_native_mutation_status(const Status status) noexcept {
    return status == Status::kOk || status == Status::kNoEligibleModule
        || status == Status::kDependencyBlocked;
}

[[nodiscard]] Envelope* mutable_envelope(
    void* const envelope,
    const int64_t envelope_bytes) noexcept {
    if (envelope == nullptr || envelope_bytes != static_cast<int64_t>(sizeof(Envelope))) {
        return nullptr;
    }
    return static_cast<Envelope*>(envelope);
}

[[nodiscard]] const Envelope* const_envelope(
    const void* const envelope,
    const int64_t envelope_bytes) noexcept {
    if (envelope == nullptr || envelope_bytes != static_cast<int64_t>(sizeof(Envelope))) {
        return nullptr;
    }
    return static_cast<const Envelope*>(envelope);
}

[[nodiscard]] modular::ModuleRecord integrator_module(
    const uint64_t module_instance_id,
    const uint64_t homology_id,
    const uint64_t topology_key,
    const float expression_scale,
    const float retention,
    const float input_gain,
    const float output_gain) noexcept {
    modular::ModuleRecord module{};
    module.module_instance_id = module_instance_id;
    module.homology_id = homology_id;
    module.topology_key = topology_key;
    module.op = modular::Operator::kIntegrator;
    module.integrator = {expression_scale, retention, input_gain, output_gain};
    module.input_binding_count = 1U;
    module.input_bindings[0] = {modular::InputSemantic::kSelfEnergy, 0.5F, true};
    module.output_binding_count = 1U;
    module.output_bindings[0] = {modular::OutputSemantic::kRest, 0.5F, true};
    return module;
}

}  // namespace

uint32_t eiel_modular_recurrent_core_abi_version_v1(void) {
    return EIEL_MODULAR_RECURRENT_CORE_ABI_VERSION_V1;
}

int64_t eiel_modular_recurrent_core_fixed_envelope_size_v1(void) {
    return static_cast<int64_t>(sizeof(Envelope));
}

int64_t eiel_modular_recurrent_core_fixed_envelope_alignment_v1(void) {
    return static_cast<int64_t>(alignof(Envelope));
}

int32_t eiel_modular_recurrent_core_initialize_empty_v1(
    void* const envelope,
    const int64_t envelope_bytes,
    const uint32_t profile) {
    Envelope* const target = mutable_envelope(envelope, envelope_bytes);
    if (target == nullptr) {
        return code(Status::kInvalidArgument);
    }
    if (!valid_profile(profile)) {
        return code(Status::kInvalidProfile);
    }
    modular::initialize_empty(target, profile_from_code(profile));
    return code(Status::kOk);
}

int32_t eiel_modular_recurrent_core_validate_v1(
    const void* const envelope,
    const int64_t envelope_bytes) {
    const Envelope* const target = const_envelope(envelope, envelope_bytes);
    if (target == nullptr) {
        return code(Status::kInvalidArgument);
    }
    return code(modular::validate(*target));
}

uint64_t eiel_modular_recurrent_core_canonical_digest_v1(
    const void* const envelope,
    const int64_t envelope_bytes) {
    const Envelope* const target = const_envelope(envelope, envelope_bytes);
    if (target == nullptr || modular::validate(*target) != Status::kOk) {
        return 0U;
    }
    return modular::canonical_digest(*target);
}

int32_t eiel_modular_recurrent_core_module_count_v1(
    const void* const envelope,
    const int64_t envelope_bytes) {
    const Envelope* const target = const_envelope(envelope, envelope_bytes);
    if (target == nullptr || modular::validate(*target) != Status::kOk) {
        return -1;
    }
    return static_cast<int32_t>(target->module_count);
}

int32_t eiel_modular_recurrent_core_activate_integrator_v1(
    void* const envelope,
    const int64_t envelope_bytes,
    const uint64_t module_instance_id,
    const uint64_t homology_id,
    const uint64_t topology_key,
    const float expression_scale,
    const float retention,
    const float input_gain,
    const float output_gain,
    const uint8_t dormant,
    uint16_t* const slot_out) {
    Envelope* const target = mutable_envelope(envelope, envelope_bytes);
    if (target == nullptr || slot_out == nullptr || dormant > 1U) {
        return code(Status::kInvalidArgument);
    }
    const modular::ModuleRecord module = integrator_module(
        module_instance_id,
        homology_id,
        topology_key,
        expression_scale,
        retention,
        input_gain,
        output_gain);
    return code(modular::activate_module(target, module, dormant != 0U, slot_out));
}

int32_t eiel_modular_recurrent_core_neutral_duplicate_v1(
    void* const envelope,
    const int64_t envelope_bytes,
    const uint16_t source_slot,
    const uint64_t new_instance_id,
    const uint64_t new_homology_id,
    const uint64_t new_topology_key,
    uint16_t* const slot_out) {
    Envelope* const target = mutable_envelope(envelope, envelope_bytes);
    if (target == nullptr || slot_out == nullptr) {
        return code(Status::kInvalidArgument);
    }
    return code(modular::neutral_duplicate(
        target,
        source_slot,
        new_instance_id,
        new_homology_id,
        new_topology_key,
        slot_out));
}

int32_t eiel_modular_recurrent_core_mutate_two_draws_v1(
    void* const envelope,
    const int64_t envelope_bytes,
    const float draw0,
    const float draw1,
    eiel_modular_recurrent_core_mutation_receipt_v1* const receipt_out) {
    Envelope* const target = mutable_envelope(envelope, envelope_bytes);
    if (target == nullptr || receipt_out == nullptr) {
        return code(Status::kInvalidArgument);
    }
    eiel_modular_recurrent_core_mutation_receipt_v1 receipt{};
    receipt.digest_before = modular::validate(*target) == Status::kOk
        ? modular::canonical_digest(*target)
        : 0U;
    const modular::MutationResult mutation = modular::mutate_two_draws(target, draw0, draw1);
    receipt.status = code(mutation.status);
    receipt.operation = static_cast<int32_t>(mutation.operation);
    receipt.selected_slot = static_cast<int32_t>(mutation.selected_slot);
    receipt.changed_dimension = mutation.changed_dimension;
    receipt.changed = mutation.changed ? 1U : 0U;
    receipt.digest_after = modular::validate(*target) == Status::kOk
        ? modular::canonical_digest(*target)
        : 0U;
    *receipt_out = receipt;
    return receipt.status;
}

int32_t eiel_modular_recurrent_core_ordinary_birth_two_draws_v1(
    const void* const parent_envelope,
    const int64_t parent_envelope_bytes,
    void* const child_envelope,
    const int64_t child_envelope_bytes,
    const int64_t parent_agent_id,
    const int64_t parent_slot,
    const int64_t parent_slot_token,
    const int64_t child_agent_id,
    const int64_t child_slot,
    const int64_t expected_child_slot_token,
    const int64_t child_publication_token,
    const uint64_t draw0_float64_bits,
    const uint64_t draw1_float64_bits,
    const uint8_t capacity_available,
    const uint8_t publish_success,
    eiel_modular_recurrent_core_ordinary_birth_receipt_v1* const receipt_out) {
    const Envelope* const parent = const_envelope(parent_envelope, parent_envelope_bytes);
    Envelope* const child = mutable_envelope(child_envelope, child_envelope_bytes);
    if (parent == nullptr || child == nullptr || receipt_out == nullptr
        || capacity_available > 1U || publish_success > 1U) {
        return code(Status::kInvalidArgument);
    }

    eiel_modular_recurrent_core_ordinary_birth_receipt_v1 receipt{};
    receipt.transaction_status = EIEL_MODULAR_RECURRENT_CORE_BIRTH_NATIVE_MUTATION_REJECTED_V1;
    receipt.native_status = code(Status::kOk);
    receipt.operation = static_cast<int32_t>(modular::MutationOperation::kToggleDormancy);
    receipt.selected_slot = -1;
    receipt.parent_agent_id = parent_agent_id;
    receipt.parent_slot = parent_slot;
    receipt.parent_slot_token = parent_slot_token;
    receipt.child_agent_id = child_agent_id;
    receipt.child_slot = child_slot;
    receipt.expected_child_slot_token = expected_child_slot_token;
    receipt.child_publication_token = child_publication_token;
    receipt.draw0_float64_bits = draw0_float64_bits;
    receipt.draw1_float64_bits = draw1_float64_bits;
    receipt.capacity_available = capacity_available;
    receipt.publish_success = publish_success;

    const Status parent_status = modular::validate(*parent);
    const Status child_status = modular::validate(*child);
    receipt.parent_digest_before = parent_status == Status::kOk
        ? modular::canonical_digest(*parent)
        : 0U;
    receipt.child_digest_before = child_status == Status::kOk
        ? modular::canonical_digest(*child)
        : 0U;
    receipt.child_digest_after = receipt.child_digest_before;
    if (parent_status != Status::kOk) {
        receipt.native_status = code(parent_status);
        *receipt_out = receipt;
        return code(Status::kOk);
    }
    if (child_status != Status::kOk) {
        receipt.native_status = code(child_status);
        *receipt_out = receipt;
        return code(Status::kOk);
    }

    float owner_draw0 = 0.0F;
    float owner_draw1 = 0.0F;
    if (decode_unit_float64_bits(draw0_float64_bits, &owner_draw0)
        && decode_unit_float64_bits(draw1_float64_bits, &owner_draw1)) {
        const uint8_t operation_index = static_cast<uint8_t>(owner_draw0 * 4.0F);
        receipt.operation = static_cast<int32_t>(
            operation_index < 4U ? operation_index : 3U);
    }

    if (!valid_owner_transition(
            parent_agent_id,
            parent_slot,
            parent_slot_token,
            child_agent_id,
            child_slot,
            expected_child_slot_token,
            child_publication_token)) {
        receipt.transaction_status = EIEL_MODULAR_RECURRENT_CORE_BIRTH_OWNER_MISMATCH_V1;
        *receipt_out = receipt;
        return code(Status::kOk);
    }

    float draw0 = 0.0F;
    float draw1 = 0.0F;
    if (!decode_unit_float64_bits(draw0_float64_bits, &draw0)
        || !decode_unit_float64_bits(draw1_float64_bits, &draw1)) {
        receipt.transaction_status = EIEL_MODULAR_RECURRENT_CORE_BIRTH_INVALID_DRAW_V1;
        receipt.native_status = code(Status::kInvalidNumeric);
        *receipt_out = receipt;
        return code(Status::kOk);
    }
    const uint8_t operation_index = static_cast<uint8_t>(draw0 * 4.0F);
    receipt.operation = static_cast<int32_t>(
        operation_index < 4U ? operation_index : 3U);

    if (capacity_available == 0U) {
        receipt.transaction_status =
            EIEL_MODULAR_RECURRENT_CORE_BIRTH_CAPACITY_UNAVAILABLE_V1;
        *receipt_out = receipt;
        return code(Status::kOk);
    }

    Envelope proposal = *parent;
    const modular::MutationResult mutation =
        modular::mutate_two_draws(&proposal, draw0, draw1);
    receipt.native_status = code(mutation.status);
    receipt.operation = static_cast<int32_t>(mutation.operation);
    receipt.selected_slot = static_cast<int32_t>(mutation.selected_slot);
    receipt.changed_dimension = mutation.changed_dimension;
    receipt.changed = mutation.changed ? 1U : 0U;
    receipt.selected_module_instance_id = mutation.selected_module_instance_id;
    receipt.selected_homology_id = mutation.selected_homology_id;
    receipt.value_before_float32 = mutation.value_before;
    receipt.value_after_float32 = mutation.value_after;
    receipt.proposal_digest = modular::validate(proposal) == Status::kOk
        ? modular::canonical_digest(proposal)
        : 0U;

    if (!publishable_native_mutation_status(mutation.status)) {
        receipt.transaction_status =
            EIEL_MODULAR_RECURRENT_CORE_BIRTH_NATIVE_MUTATION_REJECTED_V1;
        *receipt_out = receipt;
        return code(Status::kOk);
    }

    if (publish_success == 0U) {
        receipt.transaction_status =
            EIEL_MODULAR_RECURRENT_CORE_BIRTH_PUBLICATION_REJECTED_V1;
        receipt.selected_slot = -1;
        receipt.changed_dimension = 0U;
        receipt.changed = 0U;
        receipt.selected_module_instance_id = 0U;
        receipt.selected_homology_id = 0U;
        receipt.value_before_float32 = 0.0F;
        receipt.value_after_float32 = 0.0F;
        *receipt_out = receipt;
        return code(Status::kOk);
    }

    *child = proposal;
    receipt.committed = 1U;
    receipt.transaction_status = EIEL_MODULAR_RECURRENT_CORE_BIRTH_COMMITTED_V1;
    receipt.child_digest_after = modular::canonical_digest(*child);
    *receipt_out = receipt;
    return code(Status::kOk);
}

uint32_t eiel_modular_recurrent_core_ordinary_birth_receipt_layout_version_v2(void) {
    return EIEL_MODULAR_RECURRENT_CORE_ORDINARY_BIRTH_RECEIPT_LAYOUT_VERSION_V2;
}

int64_t eiel_modular_recurrent_core_ordinary_birth_receipt_size_v2(void) {
    return static_cast<int64_t>(
        sizeof(eiel_modular_recurrent_core_ordinary_birth_receipt_v2));
}

int64_t eiel_modular_recurrent_core_ordinary_birth_receipt_alignment_v2(void) {
    return static_cast<int64_t>(
        alignof(eiel_modular_recurrent_core_ordinary_birth_receipt_v2));
}

int32_t eiel_modular_recurrent_core_ordinary_birth_two_draws_v2(
    const void* const parent_envelope,
    const int64_t parent_envelope_bytes,
    void* const child_envelope,
    const int64_t child_envelope_bytes,
    const int64_t parent_agent_id,
    const int64_t parent_slot,
    const int64_t parent_slot_token,
    const int64_t child_agent_id,
    const int64_t child_slot,
    const int64_t expected_child_slot_token,
    const int64_t child_publication_token,
    const uint64_t draw0_float64_bits,
    const uint64_t draw1_float64_bits,
    const uint8_t capacity_available,
    const uint8_t publish_success,
    eiel_modular_recurrent_core_ordinary_birth_receipt_v2* const receipt_out) {
    return eiel_modular_recurrent_core_ordinary_birth_two_draws_v1(
        parent_envelope,
        parent_envelope_bytes,
        child_envelope,
        child_envelope_bytes,
        parent_agent_id,
        parent_slot,
        parent_slot_token,
        child_agent_id,
        child_slot,
        expected_child_slot_token,
        child_publication_token,
        draw0_float64_bits,
        draw1_float64_bits,
        capacity_available,
        publish_success,
        receipt_out);
}

int32_t eiel_modular_recurrent_core_ordinary_birth_two_draws_v3(
    const void* const parent_envelope,
    const int64_t parent_envelope_bytes,
    void* const child_envelope,
    const int64_t child_envelope_bytes,
    const int64_t parent_agent_id,
    const int64_t parent_slot,
    const int64_t parent_slot_token,
    const int64_t child_agent_id,
    const int64_t child_slot,
    const int64_t expected_child_slot_token,
    const int64_t child_publication_token,
    const uint64_t draw0_float64_bits,
    const uint64_t draw1_float64_bits,
    const uint8_t mutation_enabled,
    const uint8_t capacity_available,
    const uint8_t publish_success,
    eiel_modular_recurrent_core_ordinary_birth_receipt_v2* const receipt_out) {
    if (mutation_enabled > 1U) {
        return code(Status::kInvalidArgument);
    }
    if (mutation_enabled == 1U) {
        return eiel_modular_recurrent_core_ordinary_birth_two_draws_v1(
            parent_envelope,
            parent_envelope_bytes,
            child_envelope,
            child_envelope_bytes,
            parent_agent_id,
            parent_slot,
            parent_slot_token,
            child_agent_id,
            child_slot,
            expected_child_slot_token,
            child_publication_token,
            draw0_float64_bits,
            draw1_float64_bits,
            capacity_available,
            publish_success,
            receipt_out);
    }

    const Envelope* const parent = const_envelope(parent_envelope, parent_envelope_bytes);
    Envelope* const child = mutable_envelope(child_envelope, child_envelope_bytes);
    if (parent == nullptr || child == nullptr || receipt_out == nullptr
        || capacity_available > 1U || publish_success > 1U) {
        return code(Status::kInvalidArgument);
    }

    eiel_modular_recurrent_core_ordinary_birth_receipt_v2 receipt{};
    receipt.transaction_status =
        EIEL_MODULAR_RECURRENT_CORE_BIRTH_NATIVE_MUTATION_REJECTED_V1;
    receipt.native_status = code(Status::kOk);
    receipt.operation = static_cast<int32_t>(modular::MutationOperation::kToggleDormancy);
    receipt.selected_slot = -1;
    receipt.parent_agent_id = parent_agent_id;
    receipt.parent_slot = parent_slot;
    receipt.parent_slot_token = parent_slot_token;
    receipt.child_agent_id = child_agent_id;
    receipt.child_slot = child_slot;
    receipt.expected_child_slot_token = expected_child_slot_token;
    receipt.child_publication_token = child_publication_token;
    receipt.draw0_float64_bits = draw0_float64_bits;
    receipt.draw1_float64_bits = draw1_float64_bits;
    receipt.capacity_available = capacity_available;
    receipt.publish_success = publish_success;

    const Status parent_status = modular::validate(*parent);
    const Status child_status = modular::validate(*child);
    receipt.parent_digest_before = parent_status == Status::kOk
        ? modular::canonical_digest(*parent)
        : 0U;
    receipt.child_digest_before = child_status == Status::kOk
        ? modular::canonical_digest(*child)
        : 0U;
    receipt.child_digest_after = receipt.child_digest_before;
    if (parent_status != Status::kOk) {
        receipt.native_status = code(parent_status);
        *receipt_out = receipt;
        return code(Status::kOk);
    }
    if (child_status != Status::kOk) {
        receipt.native_status = code(child_status);
        *receipt_out = receipt;
        return code(Status::kOk);
    }

    float draw0 = 0.0F;
    float draw1 = 0.0F;
    if (!decode_unit_float64_bits(draw0_float64_bits, &draw0)
        || !decode_unit_float64_bits(draw1_float64_bits, &draw1)) {
        receipt.transaction_status = EIEL_MODULAR_RECURRENT_CORE_BIRTH_INVALID_DRAW_V1;
        receipt.native_status = code(Status::kInvalidNumeric);
        *receipt_out = receipt;
        return code(Status::kOk);
    }
    const uint8_t operation_index = static_cast<uint8_t>(draw0 * 4.0F);
    receipt.operation = static_cast<int32_t>(
        operation_index < 4U ? operation_index : 3U);

    if (!valid_owner_transition(
            parent_agent_id,
            parent_slot,
            parent_slot_token,
            child_agent_id,
            child_slot,
            expected_child_slot_token,
            child_publication_token)) {
        receipt.transaction_status = EIEL_MODULAR_RECURRENT_CORE_BIRTH_OWNER_MISMATCH_V1;
        *receipt_out = receipt;
        return code(Status::kOk);
    }

    if (capacity_available == 0U) {
        receipt.transaction_status =
            EIEL_MODULAR_RECURRENT_CORE_BIRTH_CAPACITY_UNAVAILABLE_V1;
        *receipt_out = receipt;
        return code(Status::kOk);
    }

    receipt.transaction_status =
        EIEL_MODULAR_RECURRENT_CORE_BIRTH_MUTATION_DISABLED_V1;
    *receipt_out = receipt;
    return code(Status::kOk);
}

uint32_t eiel_modular_recurrent_core_ordinary_birth_receipt_layout_version_v4(void) {
    return EIEL_MODULAR_RECURRENT_CORE_ORDINARY_BIRTH_RECEIPT_LAYOUT_VERSION_V4;
}

int64_t eiel_modular_recurrent_core_ordinary_birth_receipt_size_v4(void) {
    return static_cast<int64_t>(
        sizeof(eiel_modular_recurrent_core_ordinary_birth_receipt_v4));
}

int64_t eiel_modular_recurrent_core_ordinary_birth_receipt_alignment_v4(void) {
    return static_cast<int64_t>(
        alignof(eiel_modular_recurrent_core_ordinary_birth_receipt_v4));
}

int32_t eiel_modular_recurrent_core_ordinary_birth_two_draws_v4(
    const void* const parent_envelope,
    const int64_t parent_envelope_bytes,
    void* const child_envelope,
    const int64_t child_envelope_bytes,
    const int64_t parent_agent_id,
    const int64_t parent_slot,
    const int64_t parent_slot_token,
    const int64_t child_agent_id,
    const int64_t child_slot,
    const int64_t expected_child_slot_token,
    const int64_t child_publication_token,
    const uint64_t draw0_float64_bits,
    const uint64_t draw1_float64_bits,
    const uint32_t disabled_reason_code,
    const uint8_t capacity_available,
    const uint8_t publish_success,
    eiel_modular_recurrent_core_ordinary_birth_receipt_v4* const receipt_out) {
    if (!valid_disabled_reason_code(disabled_reason_code)) {
        return code(Status::kInvalidArgument);
    }
    if (disabled_reason_code == EIEL_MODULAR_RECURRENT_CORE_DISABLED_REASON_NONE_V1) {
        if (receipt_out == nullptr) {
            return code(Status::kInvalidArgument);
        }
        eiel_modular_recurrent_core_ordinary_birth_receipt_v2 base_receipt{};
        const int32_t status = eiel_modular_recurrent_core_ordinary_birth_two_draws_v2(
            parent_envelope,
            parent_envelope_bytes,
            child_envelope,
            child_envelope_bytes,
            parent_agent_id,
            parent_slot,
            parent_slot_token,
            child_agent_id,
            child_slot,
            expected_child_slot_token,
            child_publication_token,
            draw0_float64_bits,
            draw1_float64_bits,
            capacity_available,
            publish_success,
            &base_receipt);
        eiel_modular_recurrent_core_ordinary_birth_receipt_v4 receipt{};
        std::memcpy(&receipt, &base_receipt, sizeof(base_receipt));
        receipt.disabled_reason_code =
            EIEL_MODULAR_RECURRENT_CORE_DISABLED_REASON_NONE_V1;
        *receipt_out = receipt;
        return status;
    }

    const Envelope* const parent = const_envelope(parent_envelope, parent_envelope_bytes);
    Envelope* const child = mutable_envelope(child_envelope, child_envelope_bytes);
    if (parent == nullptr || child == nullptr || receipt_out == nullptr
        || capacity_available > 1U || publish_success > 1U) {
        return code(Status::kInvalidArgument);
    }

    eiel_modular_recurrent_core_ordinary_birth_receipt_v4 receipt{};
    receipt.transaction_status =
        EIEL_MODULAR_RECURRENT_CORE_BIRTH_NATIVE_MUTATION_REJECTED_V1;
    receipt.native_status = code(Status::kOk);
    receipt.operation = static_cast<int32_t>(modular::MutationOperation::kToggleDormancy);
    receipt.selected_slot = -1;
    receipt.parent_agent_id = parent_agent_id;
    receipt.parent_slot = parent_slot;
    receipt.parent_slot_token = parent_slot_token;
    receipt.child_agent_id = child_agent_id;
    receipt.child_slot = child_slot;
    receipt.expected_child_slot_token = expected_child_slot_token;
    receipt.child_publication_token = child_publication_token;
    receipt.draw0_float64_bits = draw0_float64_bits;
    receipt.draw1_float64_bits = draw1_float64_bits;
    receipt.capacity_available = capacity_available;
    receipt.publish_success = publish_success;
    receipt.disabled_reason_code = static_cast<int32_t>(disabled_reason_code);

    const Status parent_status = modular::validate(*parent);
    const Status child_status = modular::validate(*child);
    receipt.parent_digest_before = parent_status == Status::kOk
        ? modular::canonical_digest(*parent)
        : 0U;
    receipt.child_digest_before = child_status == Status::kOk
        ? modular::canonical_digest(*child)
        : 0U;
    receipt.child_digest_after = receipt.child_digest_before;
    if (parent_status != Status::kOk) {
        receipt.native_status = code(parent_status);
        *receipt_out = receipt;
        return code(Status::kOk);
    }
    if (child_status != Status::kOk) {
        receipt.native_status = code(child_status);
        *receipt_out = receipt;
        return code(Status::kOk);
    }

    float draw0 = 0.0F;
    float draw1 = 0.0F;
    if (!decode_unit_float64_bits(draw0_float64_bits, &draw0)
        || !decode_unit_float64_bits(draw1_float64_bits, &draw1)) {
        receipt.transaction_status = EIEL_MODULAR_RECURRENT_CORE_BIRTH_INVALID_DRAW_V1;
        receipt.native_status = code(Status::kInvalidNumeric);
        receipt.disabled_reason_code =
            EIEL_MODULAR_RECURRENT_CORE_DISABLED_REASON_NONE_V1;
        *receipt_out = receipt;
        return code(Status::kOk);
    }
    const uint8_t operation_index = static_cast<uint8_t>(draw0 * 4.0F);
    receipt.operation = static_cast<int32_t>(
        operation_index < 4U ? operation_index : 3U);

    if (!valid_owner_transition(
            parent_agent_id,
            parent_slot,
            parent_slot_token,
            child_agent_id,
            child_slot,
            expected_child_slot_token,
            child_publication_token)) {
        receipt.transaction_status = EIEL_MODULAR_RECURRENT_CORE_BIRTH_OWNER_MISMATCH_V1;
        receipt.disabled_reason_code =
            EIEL_MODULAR_RECURRENT_CORE_DISABLED_REASON_NONE_V1;
        *receipt_out = receipt;
        return code(Status::kOk);
    }

    if (capacity_available == 0U) {
        receipt.transaction_status =
            EIEL_MODULAR_RECURRENT_CORE_BIRTH_CAPACITY_UNAVAILABLE_V1;
        receipt.disabled_reason_code =
            EIEL_MODULAR_RECURRENT_CORE_DISABLED_REASON_NONE_V1;
        *receipt_out = receipt;
        return code(Status::kOk);
    }

    receipt.transaction_status =
        EIEL_MODULAR_RECURRENT_CORE_BIRTH_MUTATION_DISABLED_V1;
    *receipt_out = receipt;
    return code(Status::kOk);
}
