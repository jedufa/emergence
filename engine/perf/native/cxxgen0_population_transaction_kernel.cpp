#include <cstdint>

extern "C" int cxxgen0_apply_population_transactions_v1(
    const int32_t* in_agent_ids,
    const int64_t* in_slot_tokens,
    const uint8_t* in_alive_mask,
    const float* in_energy,
    const float* in_health,
    const float* in_hmem,
    const int32_t* in_action_cache_cursor,
    int64_t agents,
    int64_t hmem_cols,
    int64_t slot_epoch,
    int64_t event_cursor,
    const int32_t* transaction_type,
    const int32_t* parent_slot,
    const int32_t* child_slot,
    const int32_t* transaction_agent_id,
    const int64_t* slot_token_after,
    const int64_t* event_cursor_after,
    int64_t transactions,
    int32_t* out_agent_ids,
    int64_t* out_slot_tokens,
    uint8_t* out_alive_mask,
    float* out_energy,
    float* out_health,
    float* out_hmem,
    int32_t* out_action_cache_cursor,
    int64_t* out_slot_epoch,
    int64_t* out_event_cursor
) {
    if (!in_agent_ids || !in_slot_tokens || !in_alive_mask || !in_energy || !in_health || !in_hmem ||
        !in_action_cache_cursor || !transaction_type || !parent_slot || !child_slot || !transaction_agent_id ||
        !slot_token_after || !event_cursor_after || !out_agent_ids || !out_slot_tokens || !out_alive_mask ||
        !out_energy || !out_health || !out_hmem || !out_action_cache_cursor || !out_slot_epoch || !out_event_cursor) {
        return 1;
    }
    if (agents < 0 || hmem_cols < 0 || transactions < 0) {
        return 2;
    }

    for (int64_t i = 0; i < agents; ++i) {
        out_agent_ids[i] = in_agent_ids[i];
        out_slot_tokens[i] = in_slot_tokens[i];
        out_alive_mask[i] = in_alive_mask[i];
        out_energy[i] = in_energy[i];
        out_health[i] = in_health[i];
        out_action_cache_cursor[i] = in_action_cache_cursor[i];
        for (int64_t j = 0; j < hmem_cols; ++j) {
            out_hmem[i * hmem_cols + j] = in_hmem[i * hmem_cols + j];
        }
    }

    int64_t cursor = event_cursor;
    for (int64_t i = 0; i < transactions; ++i) {
        const int32_t kind = transaction_type[i];
        if (kind == 1) {
            const int32_t slot = child_slot[i];
            if (slot < 0 || static_cast<int64_t>(slot) >= agents) {
                return 3;
            }
            out_agent_ids[slot] = transaction_agent_id[i];
            out_slot_tokens[slot] = slot_token_after[i];
            out_alive_mask[slot] = 1;
            if (out_energy[slot] < 1.0f) {
                out_energy[slot] = 1.0f;
            }
            if (out_health[slot] < 1.0f) {
                out_health[slot] = 1.0f;
            }
            out_action_cache_cursor[slot] = 0;
            for (int64_t j = 0; j < hmem_cols; ++j) {
                out_hmem[static_cast<int64_t>(slot) * hmem_cols + j] = 0.0f;
            }
        } else if (kind == 2 || kind == 3) {
            const int32_t slot = parent_slot[i];
            if (slot < 0 || static_cast<int64_t>(slot) >= agents) {
                return 4;
            }
            out_slot_tokens[slot] = slot_token_after[i];
            out_alive_mask[slot] = 0;
            out_energy[slot] = 0.0f;
            out_health[slot] = 0.0f;
            out_action_cache_cursor[slot] = 0;
            if (kind == 2) {
                for (int64_t j = 0; j < hmem_cols; ++j) {
                    out_hmem[static_cast<int64_t>(slot) * hmem_cols + j] = 0.0f;
                }
            }
        } else {
            return 5;
        }
        cursor = event_cursor_after[i];
    }

    out_slot_epoch[0] = slot_epoch + transactions;
    out_event_cursor[0] = cursor;
    return 0;
}
