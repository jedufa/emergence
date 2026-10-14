#include <cstdint>

static inline std::uint64_t bb1_splitmix64_next(std::uint64_t& state) {
    state += 0x9E3779B97F4A7C15ULL;
    std::uint64_t z = state;
    z = (z ^ (z >> 30U)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27U)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31U);
}

extern "C" int cxxbackbone1_rng_schedule_v1(
    const std::uint64_t* initial_stream_state,
    const std::uint64_t* initial_draw_index,
    const std::int32_t* stream_id_by_draw,
    const std::int32_t* draws_per_event,
    const std::int32_t* placeholder_draws_per_event,
    const std::int64_t stream_count,
    const std::int64_t event_count,
    std::uint64_t* out_stream_state,
    std::uint64_t* out_draw_index,
    std::uint64_t* out_draw_words,
    std::int64_t* out_draw_offsets,
    std::int64_t* out_total_words
) {
    if (
        !initial_stream_state || !initial_draw_index || !stream_id_by_draw || !draws_per_event
        || !placeholder_draws_per_event || !out_stream_state || !out_draw_index || !out_draw_words
        || !out_draw_offsets || !out_total_words
    ) {
        return 1;
    }
    if (stream_count <= 0 || event_count < 0) {
        return 2;
    }
    for (std::int64_t s = 0; s < stream_count; ++s) {
        out_stream_state[s] = initial_stream_state[s];
        out_draw_index[s] = initial_draw_index[s];
    }
    std::int64_t cursor = 0;
    for (std::int64_t event = 0; event < event_count; ++event) {
        const std::int32_t stream = stream_id_by_draw[event];
        if (stream < 0 || static_cast<std::int64_t>(stream) >= stream_count) {
            return 3;
        }
        const std::int32_t draws = draws_per_event[event];
        const std::int32_t placeholders = placeholder_draws_per_event[event];
        if (draws < 0 || placeholders < 0) {
            return 4;
        }
        out_draw_offsets[event] = cursor;
        for (std::int32_t i = 0; i < draws; ++i) {
            out_draw_words[cursor] = bb1_splitmix64_next(out_stream_state[stream]);
            ++cursor;
        }
        for (std::int32_t i = 0; i < placeholders; ++i) {
            (void)bb1_splitmix64_next(out_stream_state[stream]);
        }
        out_draw_index[stream] += static_cast<std::uint64_t>(draws + placeholders);
    }
    out_draw_offsets[event_count] = cursor;
    *out_total_words = cursor;
    return 0;
}
