#include "modular_recurrent_v1.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

namespace eiel::recurrent::modular_v1 {
namespace {

constexpr float kInputBound = 1.0F;
constexpr float kActionResidualBound = 0.25F;
constexpr float kPreactivationBound = 16.0F;
constexpr std::uint16_t kInvalidSlot = std::numeric_limits<std::uint16_t>::max();

[[nodiscard]] bool finite(const float value) noexcept { return std::isfinite(value); }

[[nodiscard]] float clipped(const float value, const float low, const float high) noexcept {
    return std::min(high, std::max(low, value));
}

[[nodiscard]] float sigmoid(const float value) noexcept {
    const float x = clipped(value, -kPreactivationBound, kPreactivationBound);
    if (x >= 0.0F) {
        const float term = std::exp(-x);
        return 1.0F / (1.0F + term);
    }
    const float term = std::exp(x);
    return term / (1.0F + term);
}

[[nodiscard]] bool valid_unit_draw(const float value) noexcept {
    return finite(value) && value >= 0.0F && value < 1.0F;
}

[[nodiscard]] bool valid_input_semantic(const InputSemantic value) noexcept {
    return value == InputSemantic::kSelfEnergy || value == InputSemantic::kSelfHealth;
}

[[nodiscard]] bool valid_output_semantic(const OutputSemantic value) noexcept {
    return value == OutputSemantic::kMoveNorth || value == OutputSemantic::kMoveSouth
        || value == OutputSemantic::kMoveEast || value == OutputSemantic::kMoveWest
        || value == OutputSemantic::kRest;
}

[[nodiscard]] std::size_t action_index(const OutputSemantic value) noexcept {
    switch (value) {
        case OutputSemantic::kMoveNorth: return 0U;
        case OutputSemantic::kMoveSouth: return 1U;
        case OutputSemantic::kMoveEast: return 2U;
        case OutputSemantic::kMoveWest: return 3U;
        case OutputSemantic::kRest: return 5U;
        case OutputSemantic::kInvalid: break;
    }
    return kActionWidth;
}

[[nodiscard]] bool stable_less(const ModuleRecord& left, const ModuleRecord& right) noexcept {
    if (left.homology_id != right.homology_id) {
        return left.homology_id < right.homology_id;
    }
    if (left.module_instance_id != right.module_instance_id) {
        return left.module_instance_id < right.module_instance_id;
    }
    return left.local_id < right.local_id;
}

[[nodiscard]] std::uint16_t select_ordinal(const float draw, const std::uint16_t count) noexcept {
    if (count == 0U) {
        return 0U;
    }
    const auto scaled = static_cast<std::uint32_t>(draw * static_cast<float>(count));
    return static_cast<std::uint16_t>(std::min<std::uint32_t>(scaled, count - 1U));
}

[[nodiscard]] bool module_numeric_valid(const ModuleRecord& module) noexcept {
    const std::array<float, 4U> integrator = {
        module.integrator.expression_scale, module.integrator.retention,
        module.integrator.input_gain, module.integrator.output_gain};
    const std::array<float, 10U> gated = {
        module.gated.reset_bias, module.gated.reset_input_gain,
        module.gated.reset_state_gain, module.gated.update_bias,
        module.gated.update_input_gain, module.gated.update_state_gain,
        module.gated.candidate_bias, module.gated.candidate_input_gain,
        module.gated.candidate_state_gain, module.gated.output_gain};
    for (const float value : integrator) {
        if (!finite(value)) return false;
    }
    for (const float value : gated) {
        if (!finite(value)) return false;
    }
    if (module.op == Operator::kIntegrator) {
        return module.integrator.expression_scale >= -2.0F
            && module.integrator.expression_scale <= 2.0F
            && module.integrator.retention >= 0.0F
            && module.integrator.retention <= 1.0F
            && module.integrator.input_gain >= -2.0F && module.integrator.input_gain <= 2.0F
            && module.integrator.output_gain >= -2.0F
            && module.integrator.output_gain <= 2.0F;
    }
    if (module.op == Operator::kGatedScalar) {
        if (!finite(module.integrator.expression_scale)
            || module.integrator.expression_scale < -2.0F
            || module.integrator.expression_scale > 2.0F) {
            return false;
        }
        for (std::size_t index = 0; index + 1U < gated.size(); ++index) {
            const float value = gated[index];
            if (!finite(value) || value < -4.0F || value > 4.0F) {
                return false;
            }
        }
        return module.gated.output_gain >= -2.0F
            && module.gated.output_gain <= 2.0F;
    }
    return false;
}

[[nodiscard]] bool input_binding_zero(const InputBinding& binding) noexcept {
    return binding.semantic == InputSemantic::kSelfEnergy && binding.gain == 0.0F
        && !binding.enabled;
}

[[nodiscard]] bool output_binding_zero(const OutputBinding& binding) noexcept {
    return binding.semantic == OutputSemantic::kMoveNorth && binding.gain == 0.0F
        && !binding.enabled;
}

[[nodiscard]] bool module_record_zero(const ModuleRecord& module) noexcept {
    if (module.module_instance_id != 0U || module.homology_id != 0U
        || module.parent_module_instance_id != 0U || module.topology_key != 0U
        || module.local_id != 0U || module.op != Operator::kNone
        || module.integrator.expression_scale != 0.0F || module.integrator.retention != 0.0F
        || module.integrator.input_gain != 0.0F || module.integrator.output_gain != 0.0F
        || module.gated.reset_bias != 0.0F || module.gated.reset_input_gain != 0.0F
        || module.gated.reset_state_gain != 0.0F || module.gated.update_bias != 0.0F
        || module.gated.update_input_gain != 0.0F || module.gated.update_state_gain != 0.0F
        || module.gated.candidate_bias != 0.0F || module.gated.candidate_input_gain != 0.0F
        || module.gated.candidate_state_gain != 0.0F || module.gated.output_gain != 0.0F
        || module.input_binding_count != 0U || module.output_binding_count != 0U
        || module.prerequisite_count != 0U
        || module.specialization.source_module_instance_id != 0U
        || module.specialization.changed_dimensions != 0U
        || module.specialization.duplicate_descendant) {
        return false;
    }
    for (std::size_t i = 0; i < kBindingsPerModule; ++i) {
        if (!input_binding_zero(module.input_bindings[i])
            || !output_binding_zero(module.output_bindings[i])
            || module.prerequisite_homology_ids[i] != 0U) {
            return false;
        }
    }
    return true;
}

[[nodiscard]] bool connection_record_zero(const ConnectionRecord& connection) noexcept {
    return connection.local_id == 0U && connection.source_module_slot == 0U
        && connection.target_module_slot == 0U && connection.weight == 0.0F;
}

[[nodiscard]] bool hereditary_no_output(
    const FixedEnvelope& envelope, const std::uint16_t slot) noexcept {
    if (envelope.module_dormant[slot]) {
        return true;
    }
    const ModuleRecord& module = envelope.modules[slot];
    if (module.op == Operator::kIntegrator) {
        return module.integrator.expression_scale == 0.0F || module.integrator.output_gain == 0.0F;
    }
    if (module.op == Operator::kGatedScalar) {
        return module.integrator.expression_scale == 0.0F || module.gated.output_gain == 0.0F;
    }
    return false;
}

void clear_module_slot(FixedEnvelope* envelope, const std::size_t slot) noexcept {
    envelope->modules[slot] = ModuleRecord{};
    envelope->module_occupied[slot] = false;
    envelope->module_dormant[slot] = false;
    envelope->module_state_present[slot] = false;
    envelope->module_output_present[slot] = false;
    envelope->module_execution_order[slot] = 0U;
    envelope->development_explicit_present[slot] = false;
    envelope->development_latch[slot] = false;
    envelope->development_effective_gain_present[slot] = false;
    envelope->development_effective_gain[slot] = 0.0F;
    envelope->module_state[slot] = 0.0F;
    envelope->module_previous_output[slot] = 0.0F;
}

void clear_connection_slot(FixedEnvelope* envelope, const std::size_t slot) noexcept {
    envelope->connections[slot] = ConnectionRecord{};
    envelope->connection_occupied[slot] = false;
}

[[nodiscard]] std::uint64_t hash_bytes(
    std::uint64_t hash, const void* data, const std::size_t size) noexcept {
    const auto* bytes = static_cast<const unsigned char*>(data);
    for (std::size_t i = 0; i < size; ++i) {
        hash ^= static_cast<std::uint64_t>(bytes[i]);
        hash *= 1099511628211ULL;
    }
    return hash;
}

[[nodiscard]] std::uint64_t hash_u8(std::uint64_t hash, const std::uint8_t value) noexcept {
    return hash_bytes(hash, &value, 1U);
}

[[nodiscard]] std::uint64_t hash_u16(std::uint64_t hash, const std::uint16_t value) noexcept {
    for (unsigned shift = 0U; shift < 16U; shift += 8U) {
        hash = hash_u8(hash, static_cast<std::uint8_t>((value >> shift) & 0xffU));
    }
    return hash;
}

[[nodiscard]] std::uint64_t hash_u32(std::uint64_t hash, const std::uint32_t value) noexcept {
    for (unsigned shift = 0U; shift < 32U; shift += 8U) {
        hash = hash_u8(hash, static_cast<std::uint8_t>((value >> shift) & 0xffU));
    }
    return hash;
}

[[nodiscard]] std::uint64_t hash_u64(std::uint64_t hash, const std::uint64_t value) noexcept {
    for (unsigned shift = 0U; shift < 64U; shift += 8U) {
        hash = hash_u8(hash, static_cast<std::uint8_t>((value >> shift) & 0xffU));
    }
    return hash;
}

[[nodiscard]] std::uint64_t hash_f32(std::uint64_t hash, const float value) noexcept {
    std::uint32_t bits = 0U;
    static_assert(sizeof(bits) == sizeof(value));
    std::memcpy(&bits, &value, sizeof(bits));
    return hash_u32(hash, bits);
}

}  // namespace

CapacityLimits limits_for(const CapacityProfile profile) noexcept {
    if (profile == CapacityProfile::kLegacy12x32) {
        return {kLegacyLogicalModules, kLegacyLogicalConnections};
    }
    if (profile == CapacityProfile::kExpanded16x48) {
        return {kExpandedLogicalModules, kExpandedLogicalConnections};
    }
    return {0U, 0U};
}

void initialize_empty(FixedEnvelope* envelope, const CapacityProfile profile) noexcept {
    if (envelope == nullptr) {
        return;
    }
    *envelope = FixedEnvelope{};
    envelope->profile = profile;
}

Status validate(const FixedEnvelope& envelope) noexcept {
    const CapacityLimits limits = limits_for(envelope.profile);
    if (limits.module_slots == 0U || limits.connection_slots == 0U) {
        return Status::kInvalidProfile;
    }
    if (envelope.module_count > limits.module_slots
        || envelope.connection_count > limits.connection_slots) {
        return Status::kCapacityExceeded;
    }
    for (std::size_t slot = 0; slot < kPhysicalModuleSlots; ++slot) {
        const bool expected = slot < envelope.module_count;
        if (envelope.module_occupied[slot] != expected) {
            return Status::kInvalidMask;
        }
        if (!expected) {
            if (envelope.module_dormant[slot] || envelope.module_state_present[slot]
                || envelope.module_output_present[slot] || envelope.module_state[slot] != 0.0F
                || envelope.module_previous_output[slot] != 0.0F
                || envelope.module_execution_order[slot] != 0U
                || envelope.development_explicit_present[slot]
                || envelope.development_latch[slot]
                || envelope.development_effective_gain_present[slot]
                || envelope.development_effective_gain[slot] != 0.0F
                || !module_record_zero(envelope.modules[slot])) {
                return Status::kInvalidMask;
            }
            continue;
        }
        const ModuleRecord& module = envelope.modules[slot];
        if (module.module_instance_id == 0U || module.homology_id == 0U
            || module.local_id != slot || !module_numeric_valid(module)
            || module.input_binding_count > kBindingsPerModule
            || module.output_binding_count > kBindingsPerModule
            || module.prerequisite_count > kBindingsPerModule) {
            return Status::kInvalidArgument;
        }
        if ((envelope.module_state_present[slot] != envelope.module_output_present[slot])
            || !finite(envelope.module_state[slot]) || !finite(envelope.module_previous_output[slot])
            || envelope.module_state[slot] < -1.0F || envelope.module_state[slot] > 1.0F
            || envelope.module_previous_output[slot] < -1.0F
            || envelope.module_previous_output[slot] > 1.0F) {
            return Status::kInvalidNumeric;
        }
        if (envelope.module_execution_order[slot] >= envelope.module_count
            || (!envelope.development_explicit_present[slot]
                && envelope.development_latch[slot])
            || !finite(envelope.development_effective_gain[slot])
            || envelope.development_effective_gain[slot] < 0.0F
            || envelope.development_effective_gain[slot] > 1.0F
            || (!envelope.development_effective_gain_present[slot]
                && envelope.development_effective_gain[slot] != 0.0F)) {
            return Status::kInvalidMask;
        }
        for (std::size_t earlier = 0; earlier < slot; ++earlier) {
            if (envelope.module_execution_order[earlier]
                == envelope.module_execution_order[slot]) {
                return Status::kInvalidMask;
            }
        }
        for (std::size_t other = 0; other < slot; ++other) {
            if (envelope.modules[other].module_instance_id == module.module_instance_id
                || envelope.modules[other].homology_id == module.homology_id) {
                return Status::kDuplicateIdentity;
            }
        }
        for (std::size_t i = 0; i < module.input_binding_count; ++i) {
            if (!valid_input_semantic(module.input_bindings[i].semantic)
                || !finite(module.input_bindings[i].gain)
                || module.input_bindings[i].gain < -1.0F || module.input_bindings[i].gain > 1.0F) {
                return Status::kInvalidSemantic;
            }
        }
        for (std::size_t i = 0; i < module.output_binding_count; ++i) {
            if (!valid_output_semantic(module.output_bindings[i].semantic)
                || !finite(module.output_bindings[i].gain)
                || module.output_bindings[i].gain < -1.0F || module.output_bindings[i].gain > 1.0F) {
                return Status::kInvalidSemantic;
            }
        }
        for (std::size_t i = 0; i < module.prerequisite_count; ++i) {
            const std::uint64_t prerequisite = module.prerequisite_homology_ids[i];
            bool found = false;
            for (std::size_t candidate = 0; candidate < envelope.module_count; ++candidate) {
                found = found || envelope.modules[candidate].homology_id == prerequisite;
            }
            if (prerequisite == 0U || prerequisite == module.homology_id || !found) {
                return Status::kDanglingConnection;
            }
        }
    }
    for (std::size_t slot = 0; slot < kPhysicalConnectionSlots; ++slot) {
        const bool expected = slot < envelope.connection_count;
        if (envelope.connection_occupied[slot] != expected) {
            return Status::kInvalidMask;
        }
        if (!expected) {
            if (!connection_record_zero(envelope.connections[slot])) {
                return Status::kInvalidMask;
            }
            continue;
        }
        const ConnectionRecord& connection = envelope.connections[slot];
        if (connection.local_id != slot || connection.source_module_slot >= envelope.module_count
            || connection.target_module_slot >= envelope.module_count) {
            return Status::kDanglingConnection;
        }
        if (!finite(connection.weight)) {
            return Status::kInvalidNumeric;
        }
    }
    return Status::kOk;
}

Status canonicalize_checkpoint(FixedEnvelope* envelope) noexcept {
    if (envelope == nullptr) {
        return Status::kInvalidArgument;
    }
    Status status = validate(*envelope);
    if (status != Status::kOk) {
        return status;
    }
    std::array<std::uint16_t, kPhysicalModuleSlots> order{};
    std::array<std::uint16_t, kPhysicalModuleSlots> old_to_new{};
    for (std::uint16_t i = 0; i < envelope->module_count; ++i) {
        order[i] = i;
    }
    std::sort(order.begin(), order.begin() + envelope->module_count,
        [envelope](const std::uint16_t left, const std::uint16_t right) noexcept {
            return stable_less(envelope->modules[left], envelope->modules[right]);
        });
    FixedEnvelope canonical{};
    canonical.profile = envelope->profile;
    canonical.capacity_epoch_id = envelope->capacity_epoch_id;
    canonical.capacity_epoch_generation = envelope->capacity_epoch_generation;
    canonical.module_count = envelope->module_count;
    canonical.connection_count = envelope->connection_count;
    for (std::uint16_t next = 0; next < envelope->module_count; ++next) {
        const std::uint16_t old = order[next];
        old_to_new[old] = next;
        canonical.modules[next] = envelope->modules[old];
        canonical.modules[next].local_id = next;
        canonical.module_occupied[next] = true;
        canonical.module_dormant[next] = envelope->module_dormant[old];
        canonical.module_state_present[next] = envelope->module_state_present[old];
        canonical.module_output_present[next] = envelope->module_output_present[old];
        canonical.module_execution_order[next] = envelope->module_execution_order[old];
        canonical.development_explicit_present[next]
            = envelope->development_explicit_present[old];
        canonical.development_latch[next] = envelope->development_latch[old];
        canonical.development_effective_gain_present[next]
            = envelope->development_effective_gain_present[old];
        canonical.development_effective_gain[next]
            = envelope->development_effective_gain[old];
        canonical.module_state[next] = envelope->module_state[old];
        canonical.module_previous_output[next] = envelope->module_previous_output[old];
        std::sort(canonical.modules[next].input_bindings.begin(),
                  canonical.modules[next].input_bindings.begin()
                      + canonical.modules[next].input_binding_count,
                  [](const InputBinding& left, const InputBinding& right) noexcept {
                      return static_cast<std::uint8_t>(left.semantic)
                          < static_cast<std::uint8_t>(right.semantic);
                  });
        std::sort(canonical.modules[next].output_bindings.begin(),
                  canonical.modules[next].output_bindings.begin()
                      + canonical.modules[next].output_binding_count,
                  [](const OutputBinding& left, const OutputBinding& right) noexcept {
                      return static_cast<std::uint8_t>(left.semantic)
                          < static_cast<std::uint8_t>(right.semantic);
                  });
        std::sort(canonical.modules[next].prerequisite_homology_ids.begin(),
                  canonical.modules[next].prerequisite_homology_ids.begin()
                      + canonical.modules[next].prerequisite_count);
        for (std::size_t binding = canonical.modules[next].input_binding_count;
             binding < kBindingsPerModule; ++binding) {
            canonical.modules[next].input_bindings[binding] = InputBinding{};
        }
        for (std::size_t binding = canonical.modules[next].output_binding_count;
             binding < kBindingsPerModule; ++binding) {
            canonical.modules[next].output_bindings[binding] = OutputBinding{};
        }
        for (std::size_t dependency = canonical.modules[next].prerequisite_count;
             dependency < kBindingsPerModule; ++dependency) {
            canonical.modules[next].prerequisite_homology_ids[dependency] = 0U;
        }
    }
    std::array<ConnectionRecord, kPhysicalConnectionSlots> connections{};
    for (std::uint16_t i = 0; i < envelope->connection_count; ++i) {
        connections[i] = envelope->connections[i];
        connections[i].source_module_slot = old_to_new[connections[i].source_module_slot];
        connections[i].target_module_slot = old_to_new[connections[i].target_module_slot];
    }
    std::sort(connections.begin(), connections.begin() + envelope->connection_count,
        [](const ConnectionRecord& left, const ConnectionRecord& right) noexcept {
            if (left.source_module_slot != right.source_module_slot) {
                return left.source_module_slot < right.source_module_slot;
            }
            if (left.target_module_slot != right.target_module_slot) {
                return left.target_module_slot < right.target_module_slot;
            }
            if (left.weight != right.weight) {
                return left.weight < right.weight;
            }
            return left.local_id < right.local_id;
        });
    for (std::uint16_t i = 0; i < envelope->connection_count; ++i) {
        canonical.connections[i] = connections[i];
        canonical.connections[i].local_id = i;
        canonical.connection_occupied[i] = true;
    }
    status = validate(canonical);
    if (status != Status::kOk) {
        return status;
    }
    *envelope = canonical;
    return Status::kOk;
}

std::uint64_t canonical_digest(const FixedEnvelope& envelope) noexcept {
    FixedEnvelope canonical = envelope;
    if (canonicalize_checkpoint(&canonical) != Status::kOk) {
        return 0U;
    }
    std::uint64_t hash = 1469598103934665603ULL;
    hash = hash_u8(hash, static_cast<std::uint8_t>(canonical.profile));
    hash = hash_u64(hash, canonical.capacity_epoch_id);
    hash = hash_u64(hash, canonical.capacity_epoch_generation);
    hash = hash_u16(hash, canonical.module_count);
    hash = hash_u16(hash, canonical.connection_count);
    for (std::size_t i = 0; i < canonical.module_count; ++i) {
        const ModuleRecord& module = canonical.modules[i];
        hash = hash_u64(hash, module.module_instance_id);
        hash = hash_u64(hash, module.homology_id);
        hash = hash_u64(hash, module.parent_module_instance_id);
        hash = hash_u64(hash, module.topology_key);
        hash = hash_u16(hash, module.local_id);
        hash = hash_u8(hash, static_cast<std::uint8_t>(module.op));
        hash = hash_f32(hash, module.integrator.expression_scale);
        hash = hash_f32(hash, module.integrator.retention);
        hash = hash_f32(hash, module.integrator.input_gain);
        hash = hash_f32(hash, module.integrator.output_gain);
        const std::array<float, 10U> gated = {module.gated.reset_bias,
            module.gated.reset_input_gain, module.gated.reset_state_gain,
            module.gated.update_bias, module.gated.update_input_gain,
            module.gated.update_state_gain, module.gated.candidate_bias,
            module.gated.candidate_input_gain, module.gated.candidate_state_gain,
            module.gated.output_gain};
        for (const float value : gated) hash = hash_f32(hash, value);
        hash = hash_u8(hash, module.input_binding_count);
        for (std::size_t binding = 0; binding < module.input_binding_count; ++binding) {
            hash = hash_u8(hash, static_cast<std::uint8_t>(module.input_bindings[binding].semantic));
            hash = hash_f32(hash, module.input_bindings[binding].gain);
            hash = hash_u8(hash, module.input_bindings[binding].enabled ? 1U : 0U);
        }
        hash = hash_u8(hash, module.output_binding_count);
        for (std::size_t binding = 0; binding < module.output_binding_count; ++binding) {
            hash = hash_u8(hash, static_cast<std::uint8_t>(module.output_bindings[binding].semantic));
            hash = hash_f32(hash, module.output_bindings[binding].gain);
            hash = hash_u8(hash, module.output_bindings[binding].enabled ? 1U : 0U);
        }
        hash = hash_u8(hash, module.prerequisite_count);
        for (std::size_t dependency = 0; dependency < module.prerequisite_count; ++dependency) {
            hash = hash_u64(hash, module.prerequisite_homology_ids[dependency]);
        }
        hash = hash_u64(hash, module.specialization.source_module_instance_id);
        hash = hash_u32(hash, module.specialization.changed_dimensions);
        hash = hash_u8(hash, module.specialization.duplicate_descendant ? 1U : 0U);
        hash = hash_u8(hash, canonical.module_dormant[i] ? 1U : 0U);
        hash = hash_u16(hash, canonical.module_execution_order[i]);
        hash = hash_u8(hash, canonical.module_state_present[i] ? 1U : 0U);
        hash = hash_u8(hash, canonical.module_output_present[i] ? 1U : 0U);
        hash = hash_f32(hash, canonical.module_state[i]);
        hash = hash_f32(hash, canonical.module_previous_output[i]);
        hash = hash_u8(hash, canonical.development_explicit_present[i] ? 1U : 0U);
        hash = hash_u8(hash, canonical.development_latch[i] ? 1U : 0U);
        hash = hash_u8(hash, canonical.development_effective_gain_present[i] ? 1U : 0U);
        hash = hash_f32(hash, canonical.development_effective_gain[i]);
    }
    for (std::size_t i = 0; i < canonical.connection_count; ++i) {
        const ConnectionRecord& connection = canonical.connections[i];
        hash = hash_u16(hash, connection.local_id);
        hash = hash_u16(hash, connection.source_module_slot);
        hash = hash_u16(hash, connection.target_module_slot);
        hash = hash_f32(hash, connection.weight);
    }
    return hash;
}

Status project_checkpoint(const FixedEnvelope& owner, FixedEnvelope* checkpoint_out) noexcept {
    if (checkpoint_out == nullptr) {
        return Status::kInvalidArgument;
    }
    FixedEnvelope candidate = owner;
    const Status status = canonicalize_checkpoint(&candidate);
    if (status != Status::kOk) {
        return status;
    }
    *checkpoint_out = candidate;
    return Status::kOk;
}

Status restore_checkpoint(FixedEnvelope* fresh_owner, const FixedEnvelope& checkpoint) noexcept {
    if (fresh_owner == nullptr) {
        return Status::kInvalidArgument;
    }
    const Status owner_status = validate(*fresh_owner);
    if (owner_status != Status::kOk) {
        return owner_status;
    }
    if (fresh_owner->module_count != 0U || fresh_owner->connection_count != 0U
        || fresh_owner->capacity_epoch_id != 0U
        || fresh_owner->capacity_epoch_generation != 0U) {
        return Status::kInvalidArgument;
    }
    FixedEnvelope candidate = checkpoint;
    const Status status = canonicalize_checkpoint(&candidate);
    if (status != Status::kOk) {
        return status;
    }
    *fresh_owner = candidate;
    return Status::kOk;
}

Status activate_module(FixedEnvelope* envelope, const ModuleRecord& module, const bool dormant,
                       std::uint16_t* slot_out) noexcept {
    if (envelope == nullptr || slot_out == nullptr) {
        return Status::kInvalidArgument;
    }
    const Status input_status = validate(*envelope);
    if (input_status != Status::kOk) {
        return input_status;
    }
    const CapacityLimits limits = limits_for(envelope->profile);
    if (envelope->module_count >= limits.module_slots) {
        return Status::kCapacityExceeded;
    }
    for (std::uint16_t i = 0; i < envelope->module_count; ++i) {
        if (envelope->modules[i].module_instance_id == module.module_instance_id
            || envelope->modules[i].homology_id == module.homology_id) {
            return Status::kDuplicateIdentity;
        }
    }
    FixedEnvelope candidate = *envelope;
    const std::uint16_t slot = candidate.module_count;
    candidate.modules[slot] = module;
    candidate.modules[slot].local_id = slot;
    candidate.module_occupied[slot] = true;
    candidate.module_dormant[slot] = dormant;
    candidate.module_state_present[slot] = true;
    candidate.module_output_present[slot] = true;
    candidate.module_execution_order[slot] = slot;
    candidate.development_explicit_present[slot] = false;
    candidate.development_latch[slot] = false;
    candidate.development_effective_gain_present[slot] = false;
    candidate.development_effective_gain[slot] = 0.0F;
    candidate.module_state[slot] = 0.0F;
    candidate.module_previous_output[slot] = 0.0F;
    ++candidate.module_count;
    const Status status = validate(candidate);
    if (status != Status::kOk) {
        return status;
    }
    *envelope = candidate;
    *slot_out = slot;
    return Status::kOk;
}

Status neutral_duplicate(FixedEnvelope* envelope, const std::uint16_t source_slot,
                         const std::uint64_t new_instance_id, const std::uint64_t new_homology_id,
                         const std::uint64_t new_topology_key, std::uint16_t* slot_out) noexcept {
    if (envelope == nullptr || source_slot >= envelope->module_count || new_instance_id == 0U
        || new_homology_id == 0U) {
        return Status::kInvalidArgument;
    }
    const Status input_status = validate(*envelope);
    if (input_status != Status::kOk) {
        return input_status;
    }
    ModuleRecord duplicate = envelope->modules[source_slot];
    duplicate.parent_module_instance_id = duplicate.module_instance_id;
    duplicate.module_instance_id = new_instance_id;
    duplicate.homology_id = new_homology_id;
    duplicate.topology_key = new_topology_key;
    duplicate.specialization = {duplicate.parent_module_instance_id, 0U, true};
    if (duplicate.op == Operator::kGatedScalar) {
        duplicate.gated.output_gain = 0.0F;
    } else if (duplicate.op == Operator::kIntegrator) {
        duplicate.integrator.output_gain = 0.0F;
    } else {
        return Status::kInvalidArgument;
    }
    return activate_module(envelope, duplicate, false, slot_out);
}

Status set_dormant(FixedEnvelope* envelope, const std::uint16_t slot, const bool dormant) noexcept {
    if (envelope == nullptr || slot >= envelope->module_count) {
        return Status::kInvalidArgument;
    }
    const Status input_status = validate(*envelope);
    if (input_status != Status::kOk) {
        return input_status;
    }
    FixedEnvelope candidate = *envelope;
    candidate.module_dormant[slot] = dormant;
    const Status status = validate(candidate);
    if (status != Status::kOk) {
        return status;
    }
    *envelope = candidate;
    return Status::kOk;
}

Status set_module_development_state(FixedEnvelope* envelope, const std::uint16_t slot,
                                    const bool explicit_present, const bool latch,
                                    const bool effective_gain_present,
                                    const float effective_gain) noexcept {
    if (envelope == nullptr || slot >= envelope->module_count || !finite(effective_gain)
        || effective_gain < 0.0F || effective_gain > 1.0F
        || (!explicit_present && latch)
        || (!effective_gain_present && effective_gain != 0.0F)) {
        return Status::kInvalidArgument;
    }
    const Status input_status = validate(*envelope);
    if (input_status != Status::kOk) {
        return input_status;
    }
    FixedEnvelope candidate = *envelope;
    candidate.development_explicit_present[slot] = explicit_present;
    candidate.development_latch[slot] = latch;
    candidate.development_effective_gain_present[slot] = effective_gain_present;
    candidate.development_effective_gain[slot] = effective_gain;
    const Status status = validate(candidate);
    if (status != Status::kOk) {
        return status;
    }
    *envelope = candidate;
    return Status::kOk;
}

Status prune_module(FixedEnvelope* envelope, const std::uint16_t slot) noexcept {
    if (envelope == nullptr || slot >= envelope->module_count || envelope->module_count <= 1U) {
        return Status::kInvalidArgument;
    }
    const Status input_status = validate(*envelope);
    if (input_status != Status::kOk) {
        return input_status;
    }
    FixedEnvelope candidate = *envelope;
    FixedEnvelope* target = &candidate;
    if (!hereditary_no_output(*target, slot)) {
        return Status::kNoEligibleModule;
    }
    const std::uint64_t homology_id = target->modules[slot].homology_id;
    for (std::uint16_t other = 0; other < target->module_count; ++other) {
        if (other == slot) {
            continue;
        }
        for (std::size_t dependency = 0;
             dependency < target->modules[other].prerequisite_count; ++dependency) {
            if (target->modules[other].prerequisite_homology_ids[dependency] == homology_id) {
                return Status::kDependencyBlocked;
            }
        }
    }
    std::uint16_t next_connection = 0U;
    for (std::uint16_t i = 0; i < target->connection_count; ++i) {
        ConnectionRecord connection = target->connections[i];
        if (connection.source_module_slot == slot || connection.target_module_slot == slot) {
            continue;
        }
        if (connection.source_module_slot > slot) {
            --connection.source_module_slot;
        }
        if (connection.target_module_slot > slot) {
            --connection.target_module_slot;
        }
        connection.local_id = next_connection;
        target->connections[next_connection] = connection;
        target->connection_occupied[next_connection] = true;
        ++next_connection;
    }
    for (std::size_t i = next_connection; i < kPhysicalConnectionSlots; ++i) {
        clear_connection_slot(target, i);
    }
    target->connection_count = next_connection;
    for (std::uint16_t i = slot; i + 1U < target->module_count; ++i) {
        target->modules[i] = target->modules[i + 1U];
        target->modules[i].local_id = i;
        target->module_occupied[i] = true;
        target->module_dormant[i] = target->module_dormant[i + 1U];
        target->module_state_present[i] = target->module_state_present[i + 1U];
        target->module_output_present[i] = target->module_output_present[i + 1U];
        target->development_explicit_present[i]
            = target->development_explicit_present[i + 1U];
        target->development_latch[i] = target->development_latch[i + 1U];
        target->development_effective_gain_present[i]
            = target->development_effective_gain_present[i + 1U];
        target->development_effective_gain[i] = target->development_effective_gain[i + 1U];
        target->module_state[i] = target->module_state[i + 1U];
        target->module_previous_output[i] = target->module_previous_output[i + 1U];
    }
    --target->module_count;
    clear_module_slot(target, target->module_count);
    for (std::uint16_t i = 0; i < target->module_count; ++i) {
        target->module_execution_order[i] = i;
    }
    const Status status = canonicalize_checkpoint(target);
    if (status != Status::kOk) {
        return status;
    }
    *envelope = candidate;
    return Status::kOk;
}

MutationResult mutate_two_draws(FixedEnvelope* envelope, const float draw0, const float draw1) noexcept {
    MutationResult result{Status::kInvalidArgument, MutationOperation::kToggleDormancy,
                          kInvalidSlot, 0U, 0U, 0U, 0.0F, 0.0F, false};
    if (envelope == nullptr || !valid_unit_draw(draw0) || !valid_unit_draw(draw1)) {
        return result;
    }
    const Status input_status = validate(*envelope);
    if (input_status != Status::kOk) {
        result.status = input_status;
        return result;
    }
    FixedEnvelope candidate = *envelope;
    FixedEnvelope* target = &candidate;
    const std::uint8_t op_index = static_cast<std::uint8_t>(draw0 * 4.0F);
    result.operation = static_cast<MutationOperation>(std::min<std::uint8_t>(op_index, 3U));
    if (target->module_count == 0U) {
        result.status = Status::kNoEligibleModule;
        return result;
    }
    std::array<std::uint16_t, kPhysicalModuleSlots> eligible{};
    std::uint16_t count = 0U;
    for (std::uint16_t i = 0; i < target->module_count; ++i) {
        const bool duplicate = target->modules[i].specialization.duplicate_descendant;
        const bool refinable_operator = target->modules[i].op == Operator::kIntegrator
            || target->modules[i].op == Operator::kGatedScalar;
        const bool admit = result.operation == MutationOperation::kToggleDormancy
            || (result.operation == MutationOperation::kPruneEligible && hereditary_no_output(*target, i))
            || ((result.operation == MutationOperation::kRefineParameter
                 || result.operation == MutationOperation::kRefineRoutingOrExpression)
                && duplicate && refinable_operator);
        if (admit) {
            eligible[count++] = i;
        }
    }
    if (count == 0U) {
        result.status = Status::kNoEligibleModule;
        return result;
    }
    std::sort(eligible.begin(), eligible.begin() + count,
        [target](const std::uint16_t left, const std::uint16_t right) noexcept {
            return stable_less(target->modules[left], target->modules[right]);
        });
    const std::uint16_t ordinal = select_ordinal(draw1, count);
    std::uint16_t slot = eligible[ordinal];
    float fraction = draw1 * static_cast<float>(count) - static_cast<float>(ordinal);
    struct RoutingProposal {
        std::uint16_t module_slot;
        std::uint8_t kind;
        std::uint8_t binding_slot;
    };
    std::array<RoutingProposal, kPhysicalModuleSlots * kBindingsPerModule * 2U> proposals{};
    std::uint16_t proposal_count = 0U;
    if (result.operation == MutationOperation::kRefineRoutingOrExpression) {
        for (std::uint16_t eligible_index = 0; eligible_index < count; ++eligible_index) {
            const std::uint16_t module_slot = eligible[eligible_index];
            const ModuleRecord& candidate_module = target->modules[module_slot];
            for (std::uint8_t binding = 0; binding < candidate_module.input_binding_count; ++binding) {
                proposals[proposal_count++] = {module_slot, 0U, binding};
            }
            for (std::uint8_t binding = 0; binding < candidate_module.output_binding_count; ++binding) {
                proposals[proposal_count++] = {module_slot, 1U, binding};
            }
        }
        if (proposal_count == 0U) {
            for (std::uint16_t eligible_index = 0; eligible_index < count; ++eligible_index) {
                proposals[proposal_count++] = {eligible[eligible_index], 2U, 0U};
            }
        }
        const std::uint16_t proposal = select_ordinal(draw1, proposal_count);
        slot = proposals[proposal].module_slot;
        fraction = draw1 * static_cast<float>(proposal_count) - static_cast<float>(proposal);
    }
    result.selected_slot = slot;
    result.selected_module_instance_id = target->modules[slot].module_instance_id;
    result.selected_homology_id = target->modules[slot].homology_id;
    if (result.operation == MutationOperation::kToggleDormancy) {
        result.value_before = target->module_dormant[slot] ? 1.0F : 0.0F;
        target->module_dormant[slot] = !target->module_dormant[slot];
        result.value_after = target->module_dormant[slot] ? 1.0F : 0.0F;
        result.changed_dimension = 1U;
        result.status = validate(*target);
        result.changed = result.status == Status::kOk;
        if (result.changed) {
            *envelope = candidate;
        }
        return result;
    }
    if (result.operation == MutationOperation::kPruneEligible) {
        result.value_before = 1.0F;
        result.value_after = 0.0F;
        result.status = prune_module(target, slot);
        result.changed = result.status == Status::kOk;
        if (result.changed) {
            *envelope = candidate;
        }
        return result;
    }
    ModuleRecord& module = target->modules[slot];
    float value_before = 0.0F;
    float value_after = 0.0F;
    if (result.operation == MutationOperation::kRefineParameter) {
        if (module.op == Operator::kIntegrator) {
            const std::uint16_t field = select_ordinal(fraction, 4U);
            const float value_fraction = fraction * 4.0F - static_cast<float>(field);
            switch (field) {
                case 0U:
                    value_before = module.integrator.expression_scale;
                    value_after = -2.0F + 4.0F * value_fraction;
                    module.integrator.expression_scale = value_after;
                    break;
                case 1U:
                    value_before = module.integrator.retention;
                    value_after = value_fraction;
                    module.integrator.retention = value_after;
                    break;
                case 2U:
                    value_before = module.integrator.input_gain;
                    value_after = -2.0F + 4.0F * value_fraction;
                    module.integrator.input_gain = value_after;
                    break;
                case 3U:
                    value_before = module.integrator.output_gain;
                    value_after = -2.0F + 4.0F * value_fraction;
                    module.integrator.output_gain = value_after;
                    break;
                default: result.status = Status::kInvalidArgument; return result;
            }
            result.changed_dimension = 1U << (1U + field);
        } else if (module.op == Operator::kGatedScalar) {
            const std::uint16_t field = select_ordinal(fraction, 10U);
            const float value_fraction = fraction * 10.0F - static_cast<float>(field);
            const float core_value = -4.0F + 8.0F * value_fraction;
            switch (field) {
                case 0U: value_before = module.gated.reset_bias; module.gated.reset_bias = core_value; break;
                case 1U: value_before = module.gated.reset_input_gain; module.gated.reset_input_gain = core_value; break;
                case 2U: value_before = module.gated.reset_state_gain; module.gated.reset_state_gain = core_value; break;
                case 3U: value_before = module.gated.update_bias; module.gated.update_bias = core_value; break;
                case 4U: value_before = module.gated.update_input_gain; module.gated.update_input_gain = core_value; break;
                case 5U: value_before = module.gated.update_state_gain; module.gated.update_state_gain = core_value; break;
                case 6U: value_before = module.gated.candidate_bias; module.gated.candidate_bias = core_value; break;
                case 7U: value_before = module.gated.candidate_input_gain; module.gated.candidate_input_gain = core_value; break;
                case 8U: value_before = module.gated.candidate_state_gain; module.gated.candidate_state_gain = core_value; break;
                case 9U:
                    value_before = module.gated.output_gain;
                    module.gated.output_gain = -2.0F + 4.0F * value_fraction;
                    break;
                default: result.status = Status::kInvalidArgument; return result;
            }
            value_after = field == 9U ? module.gated.output_gain : core_value;
            result.changed_dimension = 1U << (8U + field);
        } else {
            result.status = Status::kNoEligibleModule;
            return result;
        }
    } else if (proposals[select_ordinal(draw1, proposal_count)].kind == 0U) {
        const RoutingProposal proposal = proposals[select_ordinal(draw1, proposal_count)];
        value_before = module.input_bindings[proposal.binding_slot].gain;
        module.input_bindings[proposal.binding_slot].gain = -1.0F + 2.0F * fraction;
        value_after = module.input_bindings[proposal.binding_slot].gain;
        result.changed_dimension = 1U << (18U + proposal.binding_slot);
    } else if (proposals[select_ordinal(draw1, proposal_count)].kind == 1U) {
        const RoutingProposal proposal = proposals[select_ordinal(draw1, proposal_count)];
        value_before = module.output_bindings[proposal.binding_slot].gain;
        module.output_bindings[proposal.binding_slot].gain = -1.0F + 2.0F * fraction;
        value_after = module.output_bindings[proposal.binding_slot].gain;
        result.changed_dimension = 1U << (20U + proposal.binding_slot);
    } else {
        value_before = module.integrator.expression_scale;
        module.integrator.expression_scale = -2.0F + 4.0F * fraction;
        value_after = module.integrator.expression_scale;
        result.changed_dimension = 1U << 5U;
    }
    if (value_before == value_after) {
        result.value_before = value_before;
        result.value_after = value_after;
        result.status = Status::kOk;
        result.changed = false;
        return result;
    }
    result.value_before = value_before;
    result.value_after = value_after;
    module.specialization.changed_dimensions |= result.changed_dimension;
    result.status = validate(*target);
    result.changed = result.status == Status::kOk;
    if (result.changed) {
        *envelope = candidate;
    }
    return result;
}

std::uint8_t select_inheritance_donor(const float externally_supplied_draw) noexcept {
    if (!valid_unit_draw(externally_supplied_draw)) {
        return 255U;
    }
    return externally_supplied_draw < 0.5F ? 0U : 1U;
}

Status inherit_module_atomically(FixedEnvelope* child, const FixedEnvelope& parent_a,
                                 const FixedEnvelope& parent_b, const std::uint16_t parent_slot,
                                 const float externally_supplied_draw,
                                 std::uint16_t* child_slot_out) noexcept {
    if (child == nullptr || child_slot_out == nullptr) {
        return Status::kInvalidArgument;
    }
    const Status child_status = validate(*child);
    if (child_status != Status::kOk) return child_status;
    const Status parent_a_status = validate(parent_a);
    if (parent_a_status != Status::kOk) return parent_a_status;
    const Status parent_b_status = validate(parent_b);
    if (parent_b_status != Status::kOk) return parent_b_status;
    if (parent_slot >= parent_a.module_count || parent_slot >= parent_b.module_count) {
        return Status::kInvalidArgument;
    }
    const ModuleRecord& a = parent_a.modules[parent_slot];
    const ModuleRecord& b = parent_b.modules[parent_slot];
    if (a.homology_id != b.homology_id || a.op != b.op) {
        return Status::kInvalidArgument;
    }
    const std::uint8_t donor = select_inheritance_donor(externally_supplied_draw);
    if (donor > 1U) {
        return Status::kInvalidNumeric;
    }
    const FixedEnvelope& source = donor == 0U ? parent_a : parent_b;
    return activate_module(child, source.modules[parent_slot], source.module_dormant[parent_slot],
                           child_slot_out);
}

Status route_input(const ModuleRecord& module, const SemanticInputs& inputs, float* routed_input) noexcept {
    if (routed_input == nullptr || !finite(inputs.self_energy) || !finite(inputs.self_health)
        || inputs.self_energy < 0.0F || inputs.self_energy > 1.0F
        || inputs.self_health < 0.0F || inputs.self_health > 1.0F
        || module.input_binding_count > kBindingsPerModule
        || !module_numeric_valid(module)) {
        return Status::kInvalidArgument;
    }
    float aggregate = 0.0F;
    for (std::size_t i = 0; i < module.input_binding_count; ++i) {
        const InputBinding& binding = module.input_bindings[i];
        if (!valid_input_semantic(binding.semantic) || !finite(binding.gain)
            || binding.gain < -1.0F || binding.gain > 1.0F) {
            return Status::kInvalidSemantic;
        }
        if (!binding.enabled) {
            continue;
        }
        const float value = binding.semantic == InputSemantic::kSelfEnergy
            ? inputs.self_energy : inputs.self_health;
        aggregate = aggregate + binding.gain * value;
    }
    *routed_input = clipped(aggregate, -kInputBound, kInputBound);
    return Status::kOk;
}

Status step_module(FixedEnvelope* envelope, const std::uint16_t slot, const float external_input,
                   const float routed_previous, ModuleStep* result) noexcept {
    if (envelope == nullptr || result == nullptr || slot >= envelope->module_count
        || !finite(external_input) || !finite(routed_previous)) {
        return Status::kInvalidArgument;
    }
    const Status input_status = validate(*envelope);
    if (input_status != Status::kOk) {
        return input_status;
    }
    const float previous = envelope->module_state_present[slot] ? envelope->module_state[slot] : 0.0F;
    ModuleStep candidate{};
    candidate.state_before = previous;
    if (envelope->module_dormant[slot]) {
        candidate.state_after = previous;
        envelope->module_previous_output[slot] = 0.0F;
        envelope->module_output_present[slot] = true;
        *result = candidate;
        return Status::kOk;
    }
    const ModuleRecord& module = envelope->modules[slot];
    const float developmental_gain = envelope->development_effective_gain_present[slot]
        ? envelope->development_effective_gain[slot] : 1.0F;
    const float aggregate = clipped(external_input + routed_previous, -1.0F, 1.0F);
    candidate.routed_input = aggregate;
    float state_after = 0.0F;
    float output = 0.0F;
    if (module.op == Operator::kIntegrator) {
        const float retained = module.integrator.retention * previous;
        const float external_term = module.integrator.input_gain * external_input;
        const float state_sum_1 = retained + external_term;
        state_after = clipped(state_sum_1 + routed_previous, -1.0F, 1.0F);
        const float expression_product = module.integrator.expression_scale
            * module.integrator.output_gain;
        const float base_output = clipped(expression_product * state_after, -1.0F, 1.0F);
        output = developmental_gain == 1.0F ? base_output
            : clipped(developmental_gain * base_output, -1.0F, 1.0F);
    } else if (module.op == Operator::kGatedScalar) {
        const GatedParameters& p = module.gated;
        GatedStep gated{};
        gated.input_aggregate = aggregate;
        const float reset_input_term = p.reset_input_gain * aggregate;
        const float reset_state_term = p.reset_state_gain * previous;
        const float reset_sum_1 = p.reset_bias + reset_input_term;
        gated.reset_pre = clipped(reset_sum_1 + reset_state_term, -16.0F, 16.0F);
        gated.reset_gate = sigmoid(gated.reset_pre);
        const float update_input_term = p.update_input_gain * aggregate;
        const float update_state_term = p.update_state_gain * previous;
        const float update_sum_1 = p.update_bias + update_input_term;
        gated.update_pre = clipped(update_sum_1 + update_state_term, -16.0F, 16.0F);
        gated.update_gate = sigmoid(gated.update_pre);
        const float reset_state = gated.reset_gate * previous;
        const float candidate_input_term = p.candidate_input_gain * aggregate;
        const float candidate_state_term = p.candidate_state_gain * reset_state;
        const float candidate_sum_1 = p.candidate_bias + candidate_input_term;
        gated.candidate_pre = clipped(candidate_sum_1 + candidate_state_term, -16.0F, 16.0F);
        gated.candidate = std::tanh(gated.candidate_pre);
        const float one_minus_update = 1.0F - gated.update_gate;
        const float keep_term = one_minus_update * previous;
        const float write_term = gated.update_gate * gated.candidate;
        state_after = clipped(keep_term + write_term, -1.0F, 1.0F);
        gated.state_after = state_after;
        gated.base_output = clipped(p.output_gain * state_after, -1.0F, 1.0F);
        gated.expressed_output = clipped(module.integrator.expression_scale
            * gated.base_output, -1.0F, 1.0F);
        output = developmental_gain == 1.0F ? gated.expressed_output
            : clipped(developmental_gain * gated.expressed_output, -1.0F, 1.0F);
        gated.effective_output = output;
        candidate.gated = gated;
    } else {
        return Status::kInvalidArgument;
    }
    if (!finite(state_after) || !finite(output)) {
        return Status::kInvalidNumeric;
    }
    candidate.state_after = state_after;
    candidate.output = output;
    envelope->module_state[slot] = state_after;
    envelope->module_state_present[slot] = true;
    envelope->module_previous_output[slot] = output;
    envelope->module_output_present[slot] = true;
    *result = candidate;
    return Status::kOk;
}

Status route_outputs(const FixedEnvelope& envelope, const float legacy_rest_residual,
                     std::array<float, kActionWidth>* residuals) noexcept {
    if (residuals == nullptr || !finite(legacy_rest_residual)
        || legacy_rest_residual < -kActionResidualBound
        || legacy_rest_residual > kActionResidualBound) {
        return Status::kInvalidArgument;
    }
    const Status input_status = validate(envelope);
    if (input_status != Status::kOk) {
        return input_status;
    }
    std::array<float, kActionWidth> candidate{};
    candidate[5U] = legacy_rest_residual;
    for (std::uint16_t slot = 0; slot < envelope.module_count; ++slot) {
        if (envelope.module_dormant[slot]) {
            continue;
        }
        const ModuleRecord& module = envelope.modules[slot];
        const float output = envelope.module_output_present[slot]
            ? envelope.module_previous_output[slot] : 0.0F;
        for (std::size_t i = 0; i < module.output_binding_count; ++i) {
            const OutputBinding& binding = module.output_bindings[i];
            const std::size_t index = action_index(binding.semantic);
            if (index >= kActionWidth || !finite(binding.gain)
                || binding.gain < -1.0F || binding.gain > 1.0F) {
                return Status::kInvalidSemantic;
            }
            if (binding.enabled) {
                const float next = candidate[index] + binding.gain * output;
                if (!finite(next)) {
                    return Status::kInvalidNumeric;
                }
                candidate[index] = next;
            }
        }
    }
    for (float& value : candidate) {
        value = clipped(value, -kActionResidualBound, kActionResidualBound);
    }
    *residuals = candidate;
    return Status::kOk;
}

CostSummary structural_cost(const FixedEnvelope& envelope) noexcept {
    CostSummary result{};
    result.reporting_only = true;
    if (validate(envelope) != Status::kOk) {
        return result;
    }
    result.counts.allocated_modules = envelope.module_count;
    result.counts.evaluated_connections = envelope.connection_count;
    for (std::uint16_t slot = 0; slot < envelope.module_count; ++slot) {
        const ModuleRecord& module = envelope.modules[slot];
        if (envelope.module_dormant[slot]) {
            ++result.counts.dormant_modules;
            continue;
        }
        if (module.op == Operator::kIntegrator) {
            ++result.counts.executing_integrators;
        } else if (module.op == Operator::kGatedScalar) {
            ++result.counts.executing_gated;
        }
        for (std::size_t i = 0; i < module.input_binding_count; ++i) {
            result.counts.active_input_bindings += module.input_bindings[i].enabled ? 1U : 0U;
        }
        for (std::size_t i = 0; i < module.output_binding_count; ++i) {
            result.counts.active_output_bindings += module.output_bindings[i].enabled ? 1U : 0U;
        }
        result.counts.active_dependency_edges += module.prerequisite_count;
    }
    const std::array<std::uint64_t, 8U> counts = {
        result.counts.allocated_modules, result.counts.executing_integrators,
        result.counts.executing_gated, result.counts.evaluated_connections,
        result.counts.active_input_bindings, result.counts.active_output_bindings,
        result.counts.active_dependency_edges, result.counts.dormant_modules};
    constexpr std::array<std::uint64_t, 8U> coefficients = {1U, 4U, 32U, 2U, 1U, 2U, 1U, 0U};
    for (std::size_t i = 0; i < counts.size(); ++i) {
        result.components[i] = counts[i] * coefficients[i];
        result.total += result.components[i];
    }
    return result;
}

CapacityEpochReceipt expand_capacity_epoch(FixedEnvelope* envelope,
                                           const CapacityEpochRequest& request) noexcept {
    CapacityEpochReceipt receipt{Status::kInvalidArgument, request.source_profile,
        request.target_profile, kLegacyLogicalModules, kLegacyLogicalConnections, 0U, 0U, false,
        envelope == nullptr ? 0U : envelope->capacity_epoch_id, 0U, 0U};
    if (envelope == nullptr) {
        return receipt;
    }
    const Status input_status = validate(*envelope);
    if (input_status != Status::kOk) {
        receipt.status = input_status;
        return receipt;
    }
    if (!request.authorized) {
        receipt.status = Status::kUnauthorized;
        return receipt;
    }
    bool sustained_pressure = request.pressure_window_samples == 3U
        && request.cap_contacts_in_window >= 3U
        && request.newly_blocked_legal_mutations_in_window >= 2U
        && request.near_cap_lineage_count >= 2U
        && request.latest_observed_generation == request.authorized_generation;
    for (std::size_t sample = 0; sample < 3U; ++sample) {
        sustained_pressure = sustained_pressure
            && finite(request.active_capacity_fraction[sample])
            && request.active_capacity_fraction[sample] >= 0.8F
            && request.active_capacity_fraction[sample] <= 1.0F
            && request.low_bloat[sample]
            && request.compression_relief_failed[sample];
    }
    if (!sustained_pressure) {
        receipt.status = Status::kUnauthorized;
        return receipt;
    }
    if (!request.rollback_checkpoint_verified || request.rollback_checkpoint_digest == 0U) {
        receipt.status = Status::kRollbackUnverified;
        return receipt;
    }
    if (request.boundary != BoundaryKind::kCompletedGenerationBeforeNextGenerationRngDraw
        || request.authorized_generation == std::numeric_limits<std::uint64_t>::max()
        || request.next_generation != request.authorized_generation + 1U
        || request.authorized_generation < envelope->capacity_epoch_generation) {
        receipt.status = Status::kUnsafeBoundary;
        return receipt;
    }
    if (request.source_profile != CapacityProfile::kLegacy12x32
        || request.target_profile != CapacityProfile::kExpanded16x48
        || envelope->profile != request.source_profile) {
        receipt.status = Status::kInvalidProfile;
        return receipt;
    }
    if (request.rng_digest_before != request.rng_digest_after) {
        receipt.status = Status::kRngChanged;
        return receipt;
    }
    if (request.old_subspace_digest_before == 0U
        || request.old_subspace_digest_before != request.old_subspace_digest_after
        || request.old_subspace_digest_before != canonical_digest(*envelope)) {
        receipt.status = Status::kOldSubspaceMismatch;
        return receipt;
    }
    for (std::size_t i = kLegacyLogicalModules; i < kPhysicalModuleSlots; ++i) {
        if (envelope->module_occupied[i] || envelope->module_dormant[i]
            || envelope->module_state_present[i] || envelope->module_output_present[i]
            || envelope->module_state[i] != 0.0F || envelope->module_previous_output[i] != 0.0F
            || envelope->module_execution_order[i] != 0U
            || envelope->development_explicit_present[i]
            || envelope->development_latch[i]
            || envelope->development_effective_gain_present[i]
            || envelope->development_effective_gain[i] != 0.0F
            || !module_record_zero(envelope->modules[i])) {
            receipt.status = Status::kInvalidMask;
            return receipt;
        }
    }
    for (std::size_t i = kLegacyLogicalConnections; i < kPhysicalConnectionSlots; ++i) {
        if (envelope->connection_occupied[i]) {
            receipt.status = Status::kInvalidMask;
            return receipt;
        }
    }
    FixedEnvelope candidate = *envelope;
    if (candidate.capacity_epoch_id == std::numeric_limits<std::uint64_t>::max()) {
        receipt.status = Status::kInvalidArgument;
        return receipt;
    }
    candidate.profile = CapacityProfile::kExpanded16x48;
    ++candidate.capacity_epoch_id;
    candidate.capacity_epoch_generation = request.next_generation;
    receipt.status = validate(candidate);
    receipt.tail_canonical_empty = receipt.status == Status::kOk;
    if (receipt.status == Status::kOk) {
        *envelope = candidate;
        receipt.target_capacity_epoch_id = candidate.capacity_epoch_id;
        receipt.capacity_epoch_generation = candidate.capacity_epoch_generation;
    }
    return receipt;
}

}  // namespace eiel::recurrent::modular_v1
