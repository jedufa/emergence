#include "stable_gru_v1.hpp"

#include <cfloat>
#include <cmath>
#include <cstdint>

#if defined(__CUDACC__)
#define EIEL_RECURRENT_HOST_DEVICE __host__ __device__
#else
#define EIEL_RECURRENT_HOST_DEVICE
#endif

namespace eiel::native::recurrent_v1 {
namespace {

EIEL_RECURRENT_HOST_DEVICE bool finite_f32(const float value) noexcept {
    return value == value && value <= FLT_MAX && value >= -FLT_MAX;
}

EIEL_RECURRENT_HOST_DEVICE constexpr bool accepted_active_memory_size(
    const std::int64_t value
) noexcept {
    return value == 0 || value == 8 || value == 16 || value == 32 || value == 64;
}

EIEL_RECURRENT_HOST_DEVICE float ordered_add_product(
    const float accumulator,
    const float left,
    const float right
) noexcept {
    const volatile float product = left * right;
    const volatile float sum = accumulator + product;
    return sum;
}

EIEL_RECURRENT_HOST_DEVICE float ordered_add(
    const float left,
    const float right
) noexcept {
    const volatile float result = left + right;
    return result;
}

EIEL_RECURRENT_HOST_DEVICE float ordered_subtract(
    const float left,
    const float right
) noexcept {
    const volatile float result = left - right;
    return result;
}

EIEL_RECURRENT_HOST_DEVICE float ordered_multiply(
    const float left,
    const float right
) noexcept {
    const volatile float result = left * right;
    return result;
}

EIEL_RECURRENT_HOST_DEVICE float ordered_divide(
    const float left,
    const float right
) noexcept {
    const volatile float result = left / right;
    return result;
}

EIEL_RECURRENT_HOST_DEVICE float sigmoid_f32(const float value) noexcept {
    const float negative = ordered_subtract(0.0F, value);
    const float exponential = ::expf(negative);
    const float denominator = ordered_add(1.0F, exponential);
    return ordered_divide(1.0F, denominator);
}

EIEL_RECURRENT_HOST_DEVICE StatusV1 validate_shape(
    const StableGruShapeV1& shape
) noexcept {
    if (shape.contract_version != kRecurrentContractVersionV1) {
        return StatusV1::kContractVersionMismatch;
    }
    if (shape.observation_size <= 0 || shape.hidden_size <= 0
        || shape.action_count <= 0) {
        return StatusV1::kInvalidDimension;
    }
    if (shape.hidden_size != kAcceptedStableGruHiddenV1
        || shape.action_count != kAcceptedActionCountV1) {
        return StatusV1::kUnsupportedStableShape;
    }
    return StatusV1::kSuccess;
}

EIEL_RECURRENT_HOST_DEVICE StatusV1 append_extent(
    std::int64_t& offset,
    const std::int64_t extent,
    std::int64_t& recorded_offset
) noexcept {
    recorded_offset = offset;
    std::int64_t next = 0;
    if (!checked_add_i64(offset, extent, next)) {
        return StatusV1::kIntegerOverflow;
    }
    offset = next;
    return StatusV1::kSuccess;
}

EIEL_RECURRENT_HOST_DEVICE bool can_increment(
    const std::uint64_t current,
    const std::uint64_t delta
) noexcept {
    std::uint64_t ignored = 0;
    return checked_add_u64(current, delta, ignored);
}

EIEL_RECURRENT_HOST_DEVICE ConstRecurrentStateViewV1 const_view(
    const RecurrentStateViewV1& state
) noexcept {
    return ConstRecurrentStateViewV1{
        state.hidden,
        state.hidden_element_count,
        state.tokens,
        state.token_count,
        state.cell_count,
        state.agents_per_cell,
        state.hidden_cell_stride,
        state.hidden_agent_stride,
        state.token_cell_stride,
        state.token_agent_stride,
    };
}

EIEL_RECURRENT_HOST_DEVICE StatusV1 validate_finite_array(
    const float* const values,
    const std::int64_t count
) noexcept {
    if (!values) {
        return StatusV1::kNullPointer;
    }
    if (count < 0) {
        return StatusV1::kInvalidDimension;
    }
    for (std::int64_t index = 0; index < count; ++index) {
        if (!finite_f32(values[index])) {
            return StatusV1::kNonFinite;
        }
    }
    return StatusV1::kSuccess;
}

EIEL_RECURRENT_HOST_DEVICE StatusV1 validate_lifecycle_counter(
    const StableGruCompactCountersV1* const counters,
    const LifecycleOperationV1 operation
) noexcept {
    std::uint64_t current = 0;
    switch (operation) {
        case LifecycleOperationV1::kReset:
            if (counters) current = counters->resets;
            break;
        case LifecycleOperationV1::kBirthInitialize:
            if (counters) current = counters->births;
            break;
        case LifecycleOperationV1::kDeathCleanup:
            if (counters) current = counters->deaths;
            break;
        case LifecycleOperationV1::kTurnoverReset:
            if (counters) current = counters->turnovers;
            break;
        default:
            return StatusV1::kInvalidDimension;
    }
    if (!counters) {
        return StatusV1::kSuccess;
    }
    return can_increment(current, 1) ? StatusV1::kSuccess : StatusV1::kCounterOverflow;
}

EIEL_RECURRENT_HOST_DEVICE void increment_lifecycle_counter(
    StableGruCompactCountersV1* const counters,
    const LifecycleOperationV1 operation
) noexcept {
    if (!counters) {
        return;
    }
    switch (operation) {
        case LifecycleOperationV1::kReset:
            ++counters->resets;
            break;
        case LifecycleOperationV1::kBirthInitialize:
            ++counters->births;
            break;
        case LifecycleOperationV1::kDeathCleanup:
            ++counters->deaths;
            break;
        case LifecycleOperationV1::kTurnoverReset:
            ++counters->turnovers;
            break;
    }
}

}  // namespace

EIEL_RECURRENT_HOST_DEVICE StatusV1 stable_gru_packed_layout(
    const StableGruShapeV1& shape,
    StableGruPackedLayoutV1& layout
) noexcept {
    const StatusV1 shape_status = validate_shape(shape);
    if (!ok(shape_status)) {
        return shape_status;
    }
    std::int64_t hidden_observation = 0;
    std::int64_t hidden_hidden = 0;
    std::int64_t three_hidden = 0;
    std::int64_t three_hidden_hidden = 0;
    std::int64_t action_hidden = 0;
    if (!checked_mul_i64(shape.hidden_size, shape.observation_size, hidden_observation)
        || !checked_mul_i64(shape.hidden_size, shape.hidden_size, hidden_hidden)
        || !checked_mul_i64(3, shape.hidden_size, three_hidden)
        || !checked_mul_i64(3, hidden_hidden, three_hidden_hidden)
        || !checked_mul_i64(shape.action_count, shape.hidden_size, action_hidden)) {
        return StatusV1::kIntegerOverflow;
    }

    StableGruPackedLayoutV1 candidate{};
    std::int64_t offset = 0;
    StatusV1 status = append_extent(offset, hidden_observation, candidate.encoder_weight);
    if (!ok(status)) return status;
    status = append_extent(offset, shape.hidden_size, candidate.encoder_bias);
    if (!ok(status)) return status;
    status = append_extent(offset, three_hidden_hidden, candidate.input_weight);
    if (!ok(status)) return status;
    status = append_extent(offset, three_hidden, candidate.input_bias);
    if (!ok(status)) return status;
    status = append_extent(offset, three_hidden_hidden, candidate.hidden_weight);
    if (!ok(status)) return status;
    status = append_extent(offset, three_hidden, candidate.hidden_bias);
    if (!ok(status)) return status;
    status = append_extent(offset, action_hidden, candidate.head_weight);
    if (!ok(status)) return status;
    status = append_extent(offset, shape.action_count, candidate.head_bias);
    if (!ok(status)) return status;
    candidate.total_elements = offset;
    layout = candidate;
    return StatusV1::kSuccess;
}

EIEL_RECURRENT_HOST_DEVICE StatusV1 validate_recurrent_state_view(
    const ConstRecurrentStateViewV1& state,
    const std::int64_t hidden_size
) noexcept {
    if (!state.hidden || !state.tokens) {
        return StatusV1::kNullPointer;
    }
    if (hidden_size <= 0 || state.cell_count <= 0 || state.agents_per_cell <= 0
        || state.hidden_element_count <= 0 || state.token_count <= 0) {
        return StatusV1::kInvalidDimension;
    }
    if (state.hidden_agent_stride < hidden_size || state.token_agent_stride < 1) {
        return StatusV1::kStrideInvalid;
    }
    std::int64_t minimum_hidden_cell_stride = 0;
    std::int64_t minimum_token_cell_stride = 0;
    if (!checked_mul_i64(state.hidden_agent_stride, state.agents_per_cell, minimum_hidden_cell_stride)
        || !checked_mul_i64(state.token_agent_stride, state.agents_per_cell, minimum_token_cell_stride)) {
        return StatusV1::kIntegerOverflow;
    }
    if (state.hidden_cell_stride < minimum_hidden_cell_stride
        || state.token_cell_stride < minimum_token_cell_stride) {
        return StatusV1::kStrideInvalid;
    }
    std::int64_t last_hidden_cell = 0;
    std::int64_t last_hidden_agent = 0;
    std::int64_t last_hidden_end = 0;
    std::int64_t last_token_cell = 0;
    std::int64_t last_token_agent = 0;
    std::int64_t last_token_end = 0;
    if (!checked_mul_i64(state.cell_count - 1, state.hidden_cell_stride, last_hidden_cell)
        || !checked_mul_i64(state.agents_per_cell - 1, state.hidden_agent_stride, last_hidden_agent)
        || !checked_add_i64(last_hidden_cell, last_hidden_agent, last_hidden_end)
        || !checked_add_i64(last_hidden_end, hidden_size, last_hidden_end)
        || !checked_mul_i64(state.cell_count - 1, state.token_cell_stride, last_token_cell)
        || !checked_mul_i64(state.agents_per_cell - 1, state.token_agent_stride, last_token_agent)
        || !checked_add_i64(last_token_cell, last_token_agent, last_token_end)
        || !checked_add_i64(last_token_end, 1, last_token_end)) {
        return StatusV1::kIntegerOverflow;
    }
    if (last_hidden_end > state.hidden_element_count || last_token_end > state.token_count) {
        return StatusV1::kExtentMismatch;
    }
    return StatusV1::kSuccess;
}

EIEL_RECURRENT_HOST_DEVICE StatusV1 recurrent_row_address(
    const ConstRecurrentStateViewV1& state,
    const std::int64_t hidden_size,
    const std::int64_t cell_index,
    const std::int64_t agent_index,
    const std::uint64_t expected_token,
    RecurrentRowAddressV1& address
) noexcept {
    const StatusV1 state_status = validate_recurrent_state_view(state, hidden_size);
    if (!ok(state_status)) {
        return state_status;
    }
    if (cell_index < 0 || cell_index >= state.cell_count
        || agent_index < 0 || agent_index >= state.agents_per_cell) {
        return StatusV1::kIndexOutOfRange;
    }
    RecurrentRowAddressV1 candidate{};
    std::int64_t hidden_cell = 0;
    std::int64_t hidden_agent = 0;
    std::int64_t token_cell = 0;
    std::int64_t token_agent = 0;
    if (!checked_mul_i64(cell_index, state.hidden_cell_stride, hidden_cell)
        || !checked_mul_i64(agent_index, state.hidden_agent_stride, hidden_agent)
        || !checked_add_i64(hidden_cell, hidden_agent, candidate.hidden_offset)
        || !checked_mul_i64(cell_index, state.token_cell_stride, token_cell)
        || !checked_mul_i64(agent_index, state.token_agent_stride, token_agent)
        || !checked_add_i64(token_cell, token_agent, candidate.token_offset)) {
        return StatusV1::kIntegerOverflow;
    }
    if (state.tokens[candidate.token_offset] != expected_token) {
        return StatusV1::kTokenMismatch;
    }
    address = candidate;
    return StatusV1::kSuccess;
}

EIEL_RECURRENT_HOST_DEVICE StatusV1 stable_gru_forward_one(
    const StableGruStepInputV1& input,
    StableGruStepScratchV1& scratch,
    StableGruStepOutputV1& output,
    StableGruCompactCountersV1* const counters
) noexcept {
    StableGruPackedLayoutV1 layout{};
    StatusV1 status = stable_gru_packed_layout(input.shape, layout);
    if (!ok(status)) return status;
    const std::int64_t hidden = input.shape.hidden_size;
    const std::int64_t actions = input.shape.action_count;
    const std::int64_t observations = input.shape.observation_size;
    if (!input.observation || !input.hidden_before || !input.packed_parameters
        || !input.drives || !scratch.encoder || !scratch.input_projection
        || !scratch.hidden_projection || !scratch.hidden_after || !scratch.logits
        || !output.hidden_after || !output.logits || !output.action) {
        return StatusV1::kNullPointer;
    }
    if (input.observation_count != observations || input.hidden_before_count != hidden
        || input.packed_parameter_count != layout.total_elements || input.drive_count != 4
        || output.hidden_after_count < hidden || output.logits_count < actions) {
        return StatusV1::kExtentMismatch;
    }
    if (!accepted_active_memory_size(input.active_memory_size)) {
        return StatusV1::kInvalidDimension;
    }
    if (scratch.encoder_count < hidden || scratch.input_projection_count < 3 * hidden
        || scratch.hidden_projection_count < 3 * hidden
        || scratch.hidden_after_count < hidden || scratch.logits_count < actions) {
        return StatusV1::kScratchTooSmall;
    }
    const float scalars[] = {
        input.aggression, input.temperature, input.gain,
        input.hunger_gather_weight, input.pain_rest_weight,
        input.fatigue_rest_weight, input.pain_attack_weight,
        input.fear_attack_weight,
    };
    status = validate_finite_array(scalars, 8);
    if (!ok(status)) return status;
    status = validate_finite_array(input.observation, observations);
    if (!ok(status)) return status;
    status = validate_finite_array(input.hidden_before, hidden);
    if (!ok(status)) return status;
    status = validate_finite_array(input.packed_parameters, layout.total_elements);
    if (!ok(status)) return status;
    status = validate_finite_array(input.drives, 4);
    if (!ok(status)) return status;
    if (counters
        && (!can_increment(counters->committed_steps, 1)
            || !can_increment(counters->hidden_elements_updated, static_cast<std::uint64_t>(hidden)))) {
        return StatusV1::kCounterOverflow;
    }

    const float* const pack = input.packed_parameters;
    for (std::int64_t row = 0; row < hidden; ++row) {
        float value = pack[layout.encoder_bias + row];
        for (std::int64_t column = 0; column < observations; ++column) {
            value = ordered_add_product(
                value,
                pack[layout.encoder_weight + row * observations + column],
                input.observation[column]
            );
        }
        scratch.encoder[row] = ::tanhf(value);
    }
    for (std::int64_t row = 0; row < 3 * hidden; ++row) {
        float input_value = pack[layout.input_bias + row];
        float hidden_value = pack[layout.hidden_bias + row];
        for (std::int64_t column = 0; column < hidden; ++column) {
            const float previous = column < input.active_memory_size
                ? input.hidden_before[column]
                : 0.0F;
            input_value = ordered_add_product(
                input_value,
                pack[layout.input_weight + row * hidden + column],
                scratch.encoder[column]
            );
            hidden_value = ordered_add_product(
                hidden_value,
                pack[layout.hidden_weight + row * hidden + column],
                previous
            );
        }
        scratch.input_projection[row] = input_value;
        scratch.hidden_projection[row] = hidden_value;
    }
    for (std::int64_t index = 0; index < hidden; ++index) {
        const float previous = index < input.active_memory_size
            ? input.hidden_before[index]
            : 0.0F;
        const float reset_sum = ordered_add(
            scratch.input_projection[index], scratch.hidden_projection[index]
        );
        const float update_sum = ordered_add(
            scratch.input_projection[hidden + index],
            scratch.hidden_projection[hidden + index]
        );
        const float reset = sigmoid_f32(reset_sum);
        const float update = sigmoid_f32(update_sum);
        const float reset_hidden = ordered_multiply(
            reset, scratch.hidden_projection[2 * hidden + index]
        );
        const float candidate_sum = ordered_add(
            scratch.input_projection[2 * hidden + index], reset_hidden
        );
        const float candidate = ::tanhf(candidate_sum);
        const float one_minus_update = ordered_subtract(1.0F, update);
        const float left = ordered_multiply(one_minus_update, candidate);
        const float right = ordered_multiply(update, previous);
        // The accepted runner computes the policy head from the complete raw
        // recurrent result.  The effective-width mask is a settlement rule,
        // not a policy-head input rule.
        scratch.hidden_after[index] = ordered_add(left, right);
    }
    for (std::int64_t action = 0; action < actions; ++action) {
        float value = pack[layout.head_bias + action];
        for (std::int64_t index = 0; index < hidden; ++index) {
            value = ordered_add_product(
                value,
                pack[layout.head_weight + action * hidden + index],
                scratch.hidden_after[index]
            );
        }
        scratch.logits[action] = value;
    }
    scratch.logits[6] = ordered_add(scratch.logits[6], input.aggression);
    const float temperature = input.temperature < 0.05F ? 0.05F : input.temperature;
    for (std::int64_t action = 0; action < actions; ++action) {
        scratch.logits[action] = ordered_divide(scratch.logits[action], temperature);
    }
    const float hunger_term = ordered_multiply(
        input.hunger_gather_weight, input.drives[0]
    );
    scratch.logits[4] = ordered_add(
        scratch.logits[4], ordered_multiply(input.gain, hunger_term)
    );
    const float pain_rest_term = ordered_multiply(
        input.pain_rest_weight, input.drives[1]
    );
    const float fatigue_rest_term = ordered_multiply(
        input.fatigue_rest_weight, input.drives[2]
    );
    scratch.logits[5] = ordered_add(
        scratch.logits[5],
        ordered_multiply(input.gain, ordered_add(pain_rest_term, fatigue_rest_term))
    );
    const float pain_attack_term = ordered_multiply(
        input.pain_attack_weight, input.drives[1]
    );
    const float fear_attack_term = ordered_multiply(
        input.fear_attack_weight, input.drives[3]
    );
    scratch.logits[6] = ordered_subtract(
        scratch.logits[6],
        ordered_multiply(input.gain, ordered_add(pain_attack_term, fear_attack_term))
    );
    status = validate_finite_array(scratch.hidden_after, hidden);
    if (!ok(status)) return status;
    status = validate_finite_array(scratch.logits, actions);
    if (!ok(status)) return status;

    std::int32_t best = 0;
    float best_value = scratch.logits[0];
    for (std::int64_t action = 1; action < actions; ++action) {
        if (scratch.logits[action] > best_value) {
            best_value = scratch.logits[action];
            best = static_cast<std::int32_t>(action);
        }
    }
    for (std::int64_t index = 0; index < hidden; ++index) {
        output.hidden_after[index] = index < input.active_memory_size
            ? scratch.hidden_after[index]
            : 0.0F;
    }
    for (std::int64_t action = 0; action < actions; ++action) {
        output.logits[action] = scratch.logits[action];
    }
    *output.action = best;
    if (counters) {
        ++counters->committed_steps;
        counters->hidden_elements_updated += static_cast<std::uint64_t>(hidden);
    }
    return StatusV1::kSuccess;
}

EIEL_RECURRENT_HOST_DEVICE StatusV1 apply_recurrent_lifecycle(
    RecurrentStateViewV1& state,
    const std::int64_t hidden_size,
    const std::int64_t cell_index,
    const std::int64_t agent_index,
    const std::uint64_t expected_token,
    const std::uint64_t publication_token,
    const LifecycleOperationV1 operation,
    StableGruCompactCountersV1* const counters
) noexcept {
    if (hidden_size != kAcceptedStableGruHiddenV1) {
        return StatusV1::kUnsupportedStableShape;
    }
    if ((operation == LifecycleOperationV1::kReset && publication_token != expected_token)
        || (operation == LifecycleOperationV1::kBirthInitialize
            && (expected_token != 0 || publication_token == 0))
        || (operation == LifecycleOperationV1::kDeathCleanup
            && (expected_token == 0 || publication_token != 0))
        || (operation == LifecycleOperationV1::kTurnoverReset
            && (expected_token == 0 || publication_token == 0
                || publication_token == expected_token))) {
        return StatusV1::kTokenInvalid;
    }
    const StatusV1 counter_status = validate_lifecycle_counter(counters, operation);
    if (!ok(counter_status)) return counter_status;
    RecurrentRowAddressV1 address{};
    const StatusV1 address_status = recurrent_row_address(
        const_view(state), hidden_size, cell_index, agent_index, expected_token, address
    );
    if (!ok(address_status)) return address_status;
    if (operation != LifecycleOperationV1::kDeathCleanup) {
        for (std::int64_t index = 0; index < hidden_size; ++index) {
            state.hidden[address.hidden_offset + index] = 0.0F;
        }
    }
    // Death only makes a row unreachable.  Its stale hidden bytes are retained
    // until reset at slot reuse, matching the accepted live runner.  The token
    // is the validity/publication marker and is always written last.
    state.tokens[address.token_offset] = publication_token;
    increment_lifecycle_counter(counters, operation);
    return StatusV1::kSuccess;
}

EIEL_RECURRENT_HOST_DEVICE StatusV1 reset_recurrent_row(
    RecurrentStateViewV1& state,
    const std::int64_t hidden_size,
    const std::int64_t cell_index,
    const std::int64_t agent_index,
    const std::uint64_t token,
    StableGruCompactCountersV1* const counters
) noexcept {
    if (token == 0) return StatusV1::kTokenInvalid;
    return apply_recurrent_lifecycle(
        state, hidden_size, cell_index, agent_index, token, token,
        LifecycleOperationV1::kReset, counters
    );
}

EIEL_RECURRENT_HOST_DEVICE StatusV1 initialize_birth_row(
    RecurrentStateViewV1& state,
    const std::int64_t hidden_size,
    const std::int64_t cell_index,
    const std::int64_t agent_index,
    const std::uint64_t newborn_token,
    StableGruCompactCountersV1* const counters
) noexcept {
    return apply_recurrent_lifecycle(
        state, hidden_size, cell_index, agent_index, 0, newborn_token,
        LifecycleOperationV1::kBirthInitialize, counters
    );
}

EIEL_RECURRENT_HOST_DEVICE StatusV1 cleanup_death_row(
    RecurrentStateViewV1& state,
    const std::int64_t hidden_size,
    const std::int64_t cell_index,
    const std::int64_t agent_index,
    const std::uint64_t deceased_token,
    StableGruCompactCountersV1* const counters
) noexcept {
    return apply_recurrent_lifecycle(
        state, hidden_size, cell_index, agent_index, deceased_token, 0,
        LifecycleOperationV1::kDeathCleanup, counters
    );
}

EIEL_RECURRENT_HOST_DEVICE StatusV1 reset_turnover_row(
    RecurrentStateViewV1& state,
    const std::int64_t hidden_size,
    const std::int64_t cell_index,
    const std::int64_t agent_index,
    const std::uint64_t replaced_token,
    const std::uint64_t newborn_token,
    StableGruCompactCountersV1* const counters
) noexcept {
    return apply_recurrent_lifecycle(
        state, hidden_size, cell_index, agent_index, replaced_token, newborn_token,
        LifecycleOperationV1::kTurnoverReset, counters
    );
}

EIEL_RECURRENT_HOST_DEVICE StatusV1 inherit_and_mutate_parameters(
    const float* const parent,
    const std::int64_t parameter_count,
    const MutationDrawTapeV1& tape,
    float* const scratch,
    const std::int64_t scratch_count,
    float* const child,
    const std::int64_t child_count,
    StableGruCompactCountersV1* const counters
) noexcept {
    if (!parent || !tape.additive_draws || !tape.mutation_mask || !scratch || !child) {
        return StatusV1::kNullPointer;
    }
    if (tape.contract_version != kRecurrentContractVersionV1) {
        return StatusV1::kContractVersionMismatch;
    }
    if (parameter_count <= 0 || tape.draw_count != parameter_count
        || tape.mask_count != parameter_count || scratch_count < parameter_count
        || child_count < parameter_count || !finite_f32(tape.scale) || tape.scale < 0.0F) {
        return StatusV1::kMutationTapeInvalid;
    }
    StatusV1 status = validate_finite_array(parent, parameter_count);
    if (!ok(status)) return status;
    status = validate_finite_array(tape.additive_draws, parameter_count);
    if (!ok(status)) return status;
    if (counters
        && !can_increment(
            counters->mutation_draws_consumed,
            static_cast<std::uint64_t>(parameter_count)
        )) {
        return StatusV1::kCounterOverflow;
    }
    for (std::int64_t index = 0; index < parameter_count; ++index) {
        if (tape.mutation_mask[index] > 1) {
            return StatusV1::kMutationTapeInvalid;
        }
        const float delta = tape.scale * tape.additive_draws[index];
        scratch[index] = tape.mutation_mask[index] != 0
            ? parent[index] + delta
            : parent[index];
        if (!finite_f32(scratch[index])) {
            return StatusV1::kNonFinite;
        }
    }
    for (std::int64_t index = 0; index < parameter_count; ++index) {
        child[index] = scratch[index];
    }
    if (counters) {
        counters->mutation_draws_consumed += static_cast<std::uint64_t>(parameter_count);
    }
    return StatusV1::kSuccess;
}

EIEL_RECURRENT_HOST_DEVICE StatusV1 validate_checkpoint_row(
    const ConstCheckpointRowV1& checkpoint
) noexcept {
    if (checkpoint.contract_version != kRecurrentContractVersionV1) {
        return StatusV1::kContractVersionMismatch;
    }
    if (!checkpoint.hidden) return StatusV1::kNullPointer;
    if (checkpoint.cell_index < 0 || checkpoint.agent_index < 0
        || checkpoint.hidden_size != kAcceptedStableGruHiddenV1
        || checkpoint.hidden_count != checkpoint.hidden_size || checkpoint.token == 0) {
        return StatusV1::kCheckpointInvalid;
    }
    return validate_finite_array(checkpoint.hidden, checkpoint.hidden_count);
}

EIEL_RECURRENT_HOST_DEVICE StatusV1 project_checkpoint_row(
    const ConstRecurrentStateViewV1& state,
    const std::int64_t hidden_size,
    const std::int64_t cell_index,
    const std::int64_t agent_index,
    const std::uint64_t expected_token,
    CheckpointRowV1& destination,
    StableGruCompactCountersV1* const counters
) noexcept {
    if (!destination.hidden) return StatusV1::kNullPointer;
    if (destination.hidden_count < hidden_size) return StatusV1::kExtentMismatch;
    if (hidden_size != kAcceptedStableGruHiddenV1 || expected_token == 0) {
        return StatusV1::kCheckpointInvalid;
    }
    RecurrentRowAddressV1 address{};
    StatusV1 status = recurrent_row_address(
        state, hidden_size, cell_index, agent_index, expected_token, address
    );
    if (!ok(status)) return status;
    status = validate_finite_array(state.hidden + address.hidden_offset, hidden_size);
    if (!ok(status)) return status;
    if (counters && !can_increment(counters->checkpoint_rows_projected, 1)) {
        return StatusV1::kCounterOverflow;
    }
    destination.contract_version = 0;
    destination.cell_index = cell_index;
    destination.agent_index = agent_index;
    destination.hidden_size = hidden_size;
    destination.token = expected_token;
    for (std::int64_t index = 0; index < hidden_size; ++index) {
        destination.hidden[index] = state.hidden[address.hidden_offset + index];
    }
    destination.hidden_count = hidden_size;
    destination.contract_version = kRecurrentContractVersionV1;
    if (counters) ++counters->checkpoint_rows_projected;
    return StatusV1::kSuccess;
}

EIEL_RECURRENT_HOST_DEVICE StatusV1 restore_checkpoint_row(
    const ConstCheckpointRowV1& checkpoint,
    RecurrentStateViewV1& destination,
    const std::uint64_t expected_destination_token,
    StableGruCompactCountersV1* const counters
) noexcept {
    StatusV1 status = validate_checkpoint_row(checkpoint);
    if (!ok(status)) return status;
    RecurrentRowAddressV1 address{};
    status = recurrent_row_address(
        const_view(destination), checkpoint.hidden_size,
        checkpoint.cell_index, checkpoint.agent_index,
        expected_destination_token, address
    );
    if (!ok(status)) return status;
    if (counters && !can_increment(counters->checkpoint_rows_restored, 1)) {
        return StatusV1::kCounterOverflow;
    }
    for (std::int64_t index = 0; index < checkpoint.hidden_size; ++index) {
        destination.hidden[address.hidden_offset + index] = checkpoint.hidden[index];
    }
    destination.tokens[address.token_offset] = checkpoint.token;
    if (counters) ++counters->checkpoint_rows_restored;
    return StatusV1::kSuccess;
}

EIEL_RECURRENT_HOST_DEVICE StatusV1 stable_gru_cost(
    const StableGruShapeV1& shape,
    const std::int64_t active_memory_size,
    const StableGruCostConfigV1& config,
    StableGruCostResultV1& result
) noexcept {
    StableGruPackedLayoutV1 layout{};
    StatusV1 status = stable_gru_packed_layout(shape, layout);
    if (!ok(status)) return status;
    if (config.contract_version != kRecurrentContractVersionV1) {
        return StatusV1::kContractVersionMismatch;
    }
    if (!accepted_active_memory_size(active_memory_size)
        || !finite_f32(config.base_cost) || config.base_cost < 0.0F
        || !finite_f32(config.per_active_hidden_cost)
        || config.per_active_hidden_cost < 0.0F) {
        return StatusV1::kInvalidDimension;
    }
    const float hidden_cost = config.per_active_hidden_cost
        * static_cast<float>(active_memory_size);
    const float unbounded_total = config.base_cost + hidden_cost;
    const float total = unbounded_total < 0.0F ? 0.0F : unbounded_total;
    if (!finite_f32(total)) return StatusV1::kNonFinite;
    result = StableGruCostResultV1{
        total,
        static_cast<std::uint64_t>(active_memory_size),
        static_cast<std::uint64_t>(layout.total_elements),
    };
    return StatusV1::kSuccess;
}

}  // namespace eiel::native::recurrent_v1

#undef EIEL_RECURRENT_HOST_DEVICE
