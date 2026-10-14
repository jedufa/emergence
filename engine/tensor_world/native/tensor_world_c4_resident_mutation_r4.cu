#include "tensor_world_c4_resident_birth_r2.cu"

#include "cuda_architecture_b_persistent_birth_scalar_genome.cuh"
#include "cuda_architecture_b_persistent_mate_birth_plan_v1.cuh"
#if defined(EIEL_C4_R11_RNGV2)
#include "tensor_world_c4_r11_rngv2_mutation.cuh"
#endif

#include <climits>
#include <cstdint>

namespace r4 {

#ifdef EIEL_C4_PHASE_B_SURFACE_TIMING_HOOK
// Test-only device timestamps.  The production ABI does not contain this
// pointer; the marker rows are caller-owned and only compiled into the
// attribution extension.  Each world writes its own ordered start/end pair.
constexpr int32_t kTimingGaussianStart = 0;
constexpr int32_t kTimingGaussianEnd = 1;
constexpr int32_t kTimingR4CapacityStart = 4;
constexpr int32_t kTimingR4CapacityEnd = 5;
constexpr int32_t kTimingR4PlannerStart = 6;
constexpr int32_t kTimingR4PlannerEnd = 7;
constexpr int32_t kTimingR4MutationStart = 12;
constexpr int32_t kTimingR4MutationEnd = 13;
struct PhaseTiming {
    int64_t* values;
};

__device__ __forceinline__ void phase_stamp(
    PhaseTiming timing, int64_t world, int32_t edge) {
    if (timing.values != nullptr && threadIdx.x == 0)
        timing.values[world * 16 + edge] = static_cast<int64_t>(clock64());
}
#endif

namespace mate = eiel::cuda_architecture_b::persistent_mate_birth_plan_v1;
namespace scalar = eiel::cuda_architecture_b::persistent_birth_scalar_genome;

constexpr int32_t kInvalidConfig = 30;
constexpr int32_t kInvalidPcg64 = 31;
constexpr int32_t kInvalidResident = 32;
constexpr int32_t kCapacity = 33;
constexpr int32_t kUnsupportedGru = 34;
constexpr int32_t kInvalidPython = 35;
constexpr int32_t kGaussTape = 36;
constexpr int32_t kMutationEvent = 2;

struct Genome {
    int32_t* hidden;
    int32_t* memory;
    int64_t* policy_memory;
    bool* use_gru;
    float* exploration;
    float* aggression;
    float* mutation_rate;
    float* reproduction_threshold;
    float* action_temperature;
    float* drive_inertia;
    float* drive_noise;
    float* drive_logit_gain;
    float* pred_lr;
    float* drive_setpoints;
    float* exp_strength;
    float* exp_noise;
    float* drive_birth_noise;
};

struct RandomState {
    uint32_t* words;
    int64_t* index;
    bool* has_gauss;
    double* gauss_next;
    const double* primary_tape;
    const double* secondary_tape;
    int64_t* raw_draws;
    int64_t* semantic_draws;
    int64_t tape_capacity;
#if defined(EIEL_C5_GAUSSIAN_WINDOW_V1)
    int64_t* window_base_raw_draws;
    int64_t* window_valid_raw_draw_ends;
    int64_t* window_refill_generations;
    uint8_t* window_provenance_sha256;
    int32_t* window_status;
    int64_t* window_exhaustion_raw_draws;
#endif
#if defined(EIEL_C4_R11_RNGV2)
    const int64_t* absolute_tick;
#endif
};

struct Replay {
    int64_t* state_words;
    int64_t* buffer_words;
    int64_t* cursors;
    int64_t* semantic_draws;
    int64_t* event_sequence;
    int64_t streams;
    int64_t environment_stream;
    int64_t runner_stream;
#if defined(EIEL_C4_R11_BOUNDED_RNG_V1)
    int64_t* raw64_transitions;
    int64_t* output32_returns;
#endif
};

struct R4Stage {
    Stage base;
    int32_t* mate_slots;
    int64_t* decision_mate_slots;
    int64_t* runner_draws;
    int64_t* runner_semantic;
    int64_t* pcg_state;
    int64_t* pcg_buffer;
    int64_t* pcg_cursors;
    int64_t* pcg_semantic;
    uint32_t* python_words;
    int64_t* python_index;
    bool* python_has_gauss;
    double* python_gauss_next;
    int64_t* python_raw;
    int64_t* python_semantic;
    float* scalar_values;
    bool* mutation_occurred;
    float* mutation_delta;
    int32_t* shadow_y;
    int32_t* shadow_x;
    double* shadow_energy;
    double* planned_energy;
    double* parent_energy_after;
    double* mate_energy_after;
    int32_t* spawn_positions;
    int32_t* decision_spawn_counts;
    int64_t* reason_counts;
    int64_t* free_head;
    int64_t* plan_parent;
    int64_t* plan_mate;
    int64_t* plan_child;
    double* plan_child_energy;
#if defined(EIEL_C4_R11_BOUNDED_RNG_V1)
    int64_t* post_raw64_transitions;
    int64_t* post_output32_returns;
#endif
#ifdef EIEL_C4_PHASE_B_SURFACE_TIMING_HOOK
    PhaseTiming timing;
#endif
};

struct R4State {
    State base;
    Genome genome;
    RandomState python;
    Replay replay;
    float* injury;
};

__device__ __forceinline__ double midpoint(const float* values, int64_t a, int64_t b) {
    return 0.5 * (
        scalar::canon_genome_double(static_cast<double>(values[a]))
        + scalar::canon_genome_double(static_cast<double>(values[b]))
    );
}

__device__ __forceinline__ uint32_t raw32(
    scalar::DeviceCpythonRandom& rng) {
    if (rng.index >= scalar::kCpythonRandomWordCount) {
        rng.twist();
        // c4_genome.py is authoritative and uses word 397 in its terminal
        // twist row; the shared scalar helper uses word 396.
        rng.state[623] ^= rng.state[396] ^ rng.state[397];
    }
    uint32_t value = rng.state[rng.index++];
    value ^= value >> 11U;
    value ^= (value << 7U) & 0x9d2c5680U;
    value ^= (value << 15U) & 0xefc60000U;
    value ^= value >> 18U;
    ++rng.raw_draws;
    return value;
}

__device__ __forceinline__ double random_double(
    scalar::DeviceCpythonRandom& rng) {
    const uint32_t a = raw32(rng) >> 5U;
    const uint32_t b = raw32(rng) >> 6U;
    return (static_cast<double>(a) * 67108864.0 + static_cast<double>(b))
        * (1.0 / 9007199254740992.0);
}

__device__ __forceinline__ bool consume_gauss(
    scalar::DeviceCpythonRandom& rng, double sigma, double* value) {
    if (rng.has_gauss_next) {
        *value = rng.gauss_next * sigma;
        rng.has_gauss_next = false;
        rng.gauss_next = 0.0;
        return true;
    }
    const int64_t tape_index = rng.raw_draws;
    if (tape_index < 0 || tape_index >= rng.gauss_tape_capacity
        || rng.gauss_primary_tape == nullptr || rng.gauss_secondary_tape == nullptr) {
        return false;
    }
    (void)random_double(rng);
    (void)random_double(rng);
    *value = rng.gauss_primary_tape[tape_index] * sigma;
    rng.gauss_next = rng.gauss_secondary_tape[tape_index];
    rng.has_gauss_next = true;
    return true;
}

__device__ __forceinline__ uint64_t getrandbits(
    scalar::DeviceCpythonRandom& rng, int64_t bit_count) {
    if (bit_count == 0) return 0;
    if (bit_count <= 32) return static_cast<uint64_t>(raw32(rng) >> (32 - bit_count));
    uint64_t value = 0;
    int64_t filled = 0;
    while (bit_count > 0) {
        const int64_t take = bit_count < 32 ? bit_count : 32;
        uint32_t word = raw32(rng);
        if (take < 32) word >>= 32 - take;
        value |= static_cast<uint64_t>(word) << filled;
        filled += take;
        bit_count -= take;
    }
    return value;
}

__device__ __forceinline__ uint64_t randbelow(
    scalar::DeviceCpythonRandom& rng, uint64_t upper) {
    int64_t bit_count = 0;
    for (uint64_t value = upper; value != 0; value >>= 1U) ++bit_count;
#if defined(EIEL_C4_R11_BOUNDED_RNG_V1)
    for (int32_t attempt = 0; attempt < EIEL_C4_R11_MT_MAX_RANDBELOW_ATTEMPTS;
         ++attempt) {
        const uint64_t value = getrandbits(rng, bit_count);
        if (value < upper) return value;
    }
    return UINT64_MAX;
#else
    uint64_t value = getrandbits(rng, bit_count);
    while (value >= upper) value = getrandbits(rng, bit_count);
    return value;
#endif
}

__device__ __forceinline__ double mutate_gene(
    scalar::DeviceCpythonRandom& rng, double value, double rate, double sigma,
    double lower, double upper, int64_t& semantic, bool& occurred, double& delta_sum) {
    ++semantic;
    if (random_double(rng) < rate) {
        double delta = 0.0;
        if (!consume_gauss(rng, sigma, &delta)) return NAN;
        ++semantic;
        occurred = true;
        delta_sum += fabs(delta);
        return scalar::clamp_double(value + delta, lower, upper);
    }
    return scalar::clamp_double(value, lower, upper);
}

__device__ bool stage_scalar_children(
    R4State state, R4Stage stage, int64_t world, int64_t capacity) {
    const int64_t offset = world * capacity;
#if defined(EIEL_C4_R11_RNGV2)
    const int64_t planned = stage.base.planned_count[world];
    if (planned < 0 || planned > 64 || state.python.absolute_tick == nullptr) {
        stage.base.status[world] = kInvalidPython;
        return false;
    }
    const int64_t absolute_tick = state.python.absolute_tick[world];
    if (absolute_tick < 0) {
        stage.base.status[world] = kInvalidPython;
        return false;
    }
    for (int64_t birth = 0; birth < planned; ++birth) {
        const int64_t parent = stage.plan_parent[offset + birth];
        const int64_t mate_slot = stage.plan_mate[offset + birth];
        if (state.genome.use_gru[offset + parent]
            || state.genome.use_gru[offset + mate_slot]) {
            stage.base.status[world] = kUnsupportedGru;
            return false;
        }
        const int64_t parent_index = offset + parent;
        const int64_t mate_index = offset + mate_slot;
        const int64_t subject = rngv2::birth_subject(
            state.base.agent_id, stage.plan_parent, stage.plan_mate,
            offset, planned, birth
        );
        if (subject < 0) {
            stage.base.status[world] = kInvalidPython;
            return false;
        }
        const float* fields[8] = {
            state.genome.exploration, state.genome.aggression,
            state.genome.reproduction_threshold, state.genome.action_temperature,
            state.genome.drive_inertia, state.genome.drive_noise,
            state.genome.drive_logit_gain, state.genome.pred_lr,
        };
        double parent_genes[8];
        double mate_genes[8];
        for (int32_t field = 0; field < 8; ++field) {
            parent_genes[field] = static_cast<double>(fields[field][parent_index]);
            mate_genes[field] = static_cast<double>(fields[field][mate_index]);
        }
        double child_genes[8];
        double child_rate = 0.0;
        const double parent_rate = static_cast<double>(state.genome.mutation_rate[parent_index]);
        const double mate_rate = static_cast<double>(state.genome.mutation_rate[mate_index]);
        if (!rngv2::mutate_planned_birth_v2(
                static_cast<uint32_t>(world),
                static_cast<uint64_t>(absolute_tick),
                static_cast<uint64_t>(subject),
                parent_genes, mate_genes, parent_rate, mate_rate,
                child_genes, &child_rate)) {
            stage.base.status[world] = kGaussTape;
            return false;
        }
        float* output = stage.scalar_values + (offset + birth) * 16;
        bool occurred = false;
        double delta_sum = 0.0;
        for (int32_t field = 0; field < 8; ++field) {
            if (child_genes[field] != 0.5 * (parent_genes[field] + mate_genes[field])) {
                occurred = true;
                delta_sum += fabs(child_genes[field] - 0.5 * (parent_genes[field] + mate_genes[field]));
            }
            output[field] = static_cast<float>(child_genes[field]);
        }
        output[8] = static_cast<float>(child_rate);
        for (int32_t drive = 0; drive < 4; ++drive) {
            const double value = 0.5 * (
                static_cast<double>(state.genome.drive_setpoints[parent_index * 4 + drive])
                + static_cast<double>(state.genome.drive_setpoints[mate_index * 4 + drive])
            );
            output[9 + drive] = static_cast<float>(scalar::clamp_double(value, 0.0, 1.0));
        }
        output[13] = scalar::canon_genome_float(scalar::clamp_double(
            midpoint(state.genome.exp_strength, parent_index, mate_index), 0.0, 1.0));
        output[14] = scalar::canon_genome_float(scalar::clamp_double(
            midpoint(state.genome.exp_noise, parent_index, mate_index), 0.0, 0.5));
        output[15] = scalar::canon_genome_float(scalar::clamp_double(
            midpoint(state.genome.drive_birth_noise, parent_index, mate_index), 0.0, 0.2));
        stage.mutation_occurred[offset + birth] = occurred;
        stage.mutation_delta[offset + birth] = static_cast<float>(delta_sum);
    }
    stage.python_index[world] = state.python.index[world];
    stage.python_has_gauss[world] = state.python.has_gauss[world];
    stage.python_gauss_next[world] = state.python.gauss_next[world];
    stage.python_raw[world] = state.python.raw_draws[world];
    stage.python_semantic[world] = state.python.semantic_draws[world];
    return true;
#else
    const int64_t word_offset = world * scalar::kCpythonRandomWordCount;
    if (state.python.index[world] < 0
        || state.python.index[world] > scalar::kCpythonRandomWordCount
        || state.python.raw_draws[world] < 0
        || state.python.semantic_draws[world] < 0
        || !isfinite(state.python.gauss_next[world])) {
        stage.base.status[world] = kInvalidPython;
        return false;
    }
#if !defined(EIEL_C4_R11_PARALLEL_GAUSSIAN_TAPE_VALIDATION)
#ifdef EIEL_C4_PHASE_B_SURFACE_TIMING_HOOK
    phase_stamp(stage.timing, world, kTimingGaussianStart);
#endif
    for (int64_t row = 0; row < state.python.tape_capacity; ++row) {
        if (!isfinite(state.python.primary_tape[world * state.python.tape_capacity + row])
            || !isfinite(state.python.secondary_tape[world * state.python.tape_capacity + row])) {
            stage.base.status[world] = kInvalidPython;
            return false;
        }
    }
#ifdef EIEL_C4_PHASE_B_SURFACE_TIMING_HOOK
    phase_stamp(stage.timing, world, kTimingGaussianEnd);
#endif
#endif
    for (int64_t word = 0; word < scalar::kCpythonRandomWordCount; ++word) {
        stage.python_words[word_offset + word] = state.python.words[word_offset + word];
    }
    int64_t window_base = 0;
    int64_t window_capacity = state.python.tape_capacity;
#if defined(EIEL_C5_GAUSSIAN_WINDOW_V1)
    if (state.python.window_base_raw_draws != nullptr) {
        window_base = state.python.window_base_raw_draws[world];
        const int64_t window_end = state.python.window_valid_raw_draw_ends[world];
        if (state.python.window_status[world] != 0
            || window_base < 0 || window_end < window_base
            || window_end - window_base > state.python.tape_capacity
            || state.python.raw_draws[world] < window_base
            || state.python.raw_draws[world] > window_end) {
            stage.base.status[world] = kInvalidPython;
            return false;
        }
        window_capacity = window_end - window_base;
    }
#endif
    scalar::DeviceCpythonRandom rng{
        stage.python_words + word_offset,
        state.python.index[world],
        state.python.has_gauss[world],
        state.python.gauss_next[world],
        state.python.primary_tape + world * state.python.tape_capacity,
        state.python.secondary_tape + world * state.python.tape_capacity,
        window_capacity,
        state.python.raw_draws[world] - window_base,
    };
    int64_t semantic = state.python.semantic_draws[world];
    for (int64_t birth = 0; birth < stage.base.planned_count[world]; ++birth) {
        const int64_t parent = stage.plan_parent[offset + birth];
        const int64_t mate_slot = stage.plan_mate[offset + birth];
        if (state.genome.use_gru[offset + parent]
            || state.genome.use_gru[offset + mate_slot]) {
            stage.base.status[world] = kUnsupportedGru;
            return false;
        }
        // CPU C4 consumes this decision but inherits primary hidden size.
        ++semantic;
        (void)random_double(rng);
        const int64_t parent_index = offset + parent;
        const int64_t mate_index = offset + mate_slot;
        const double base_rate = midpoint(
            state.genome.mutation_rate, parent_index, mate_index
        );
        const double scalar_rate = scalar::clamp_double(base_rate, 0.0, 1.0);
        bool occurred = false;
        double delta_sum = 0.0;
        float* output = stage.scalar_values + (offset + birth) * 16;
        const double specs[8][3] = {
            {0.10, -1.0, 1.0}, {0.15, -2.0, 2.0}, {2.0, 2.0, 90.0},
            {0.10, 0.25, 2.5}, {0.05, 0.0, 0.999}, {0.01, 0.0, 0.10},
            {0.20, 0.0, 3.0}, {0.03, 0.0, 1.0},
        };
        const float* fields[8] = {
            state.genome.exploration, state.genome.aggression,
            state.genome.reproduction_threshold, state.genome.action_temperature,
            state.genome.drive_inertia, state.genome.drive_noise,
            state.genome.drive_logit_gain, state.genome.pred_lr,
        };
        for (int32_t field = 0; field < 8; ++field) {
            const double value = mutate_gene(
                rng, midpoint(fields[field], parent_index, mate_index), scalar_rate,
                specs[field][0], specs[field][1], specs[field][2], semantic,
                occurred, delta_sum
            );
            if (!isfinite(value)) {
                stage.base.status[world] = kGaussTape;
                return false;
            }
            // The CPU oracle assigns the post-mutation double directly into
            // the authoritative float32 tensor.  Re-canonicalizing here was
            // invisible for the historical constant 0.01 tape but changes
            // legitimate CPython Gaussian results by one ULP.
            output[field] = static_cast<float>(value);
        }
        double child_rate = base_rate;
        ++semantic;
        if (random_double(rng) < 0.2) {
            double delta = 0.0;
            if (!consume_gauss(rng, 0.005, &delta)) {
                stage.base.status[world] = kGaussTape;
                return false;
            }
            child_rate += delta;
            ++semantic;
        }
        child_rate = scalar::clamp_double(child_rate, 0.001, 0.20);
        output[8] = static_cast<float>(child_rate);
        const double memory_probability = scalar::clamp_double(
            0.25 * child_rate, 0.002, 0.05
        );
        ++semantic;
        if (random_double(rng) < memory_probability) {
            const uint64_t memory_choice = randbelow(rng, 2);
#if defined(EIEL_C4_R11_BOUNDED_RNG_V1)
            if (memory_choice == UINT64_MAX) {
                stage.base.status[world] = kInvalidPython;
                return false;
            }
#else
            (void)memory_choice;
#endif
            ++semantic;
        }
        for (int32_t drive = 0; drive < 4; ++drive) {
            const double value = 0.5 * (
                static_cast<double>(state.genome.drive_setpoints[parent_index * 4 + drive])
                + static_cast<double>(state.genome.drive_setpoints[mate_index * 4 + drive])
            );
            output[9 + drive] = static_cast<float>(scalar::clamp_double(value, 0.0, 1.0));
        }
        output[13] = scalar::canon_genome_float(scalar::clamp_double(
            midpoint(state.genome.exp_strength, parent_index, mate_index), 0.0, 1.0));
        output[14] = scalar::canon_genome_float(scalar::clamp_double(
            midpoint(state.genome.exp_noise, parent_index, mate_index), 0.0, 0.5));
        output[15] = scalar::canon_genome_float(scalar::clamp_double(
            midpoint(state.genome.drive_birth_noise, parent_index, mate_index), 0.0, 0.2));
        stage.mutation_occurred[offset + birth] = occurred;
        stage.mutation_delta[offset + birth] = static_cast<float>(delta_sum);
    }
    stage.python_index[world] = rng.index;
    stage.python_has_gauss[world] = rng.has_gauss_next;
    stage.python_gauss_next[world] = rng.gauss_next;
    stage.python_raw[world] = rng.raw_draws + window_base;
    stage.python_semantic[world] = semantic;
    return true;
#endif
}

__global__ void preflight_kernel(
    R4State state, R4Stage stage, Config config) {
    if (threadIdx.x != 0) return;
    const int64_t world = blockIdx.x;
    if (world >= state.base.batch) return;
    const int64_t capacity = state.base.capacity;
    const int64_t offset = world * capacity;
#ifdef EIEL_C4_R11_PARALLEL_GAUSSIAN_TAPE_VALIDATION
    if (stage.base.status[world] != kOk) return;
#endif
    stage.base.status[world] = kOk;
    if (state.replay.streams < 2
        || state.replay.environment_stream < 0
        || state.replay.runner_stream < 0
        || state.replay.environment_stream >= state.replay.streams
        || state.replay.runner_stream >= state.replay.streams
        || state.replay.environment_stream == state.replay.runner_stream) {
        stage.base.status[world] = kInvalidConfig;
        return;
    }
    for (int64_t stream = 0; stream < state.replay.streams; ++stream) {
        const int64_t stream_index = world * state.replay.streams + stream;
        const int64_t* input_state = state.replay.state_words + stream_index * 4;
        const int64_t* input_buffer = state.replay.buffer_words + stream_index * 2;
        if ((input_state[2] & 1LL) == 0
            || input_buffer[0] < 0 || input_buffer[0] > 1
            || input_buffer[1] < 0
            || static_cast<uint64_t>(input_buffer[1]) > 0xffffffffULL
            || state.replay.cursors[stream_index] < 0
            || state.replay.semantic_draws[stream_index] < 0) {
            stage.base.status[world] = kInvalidPcg64;
            return;
        }
        for (int32_t word = 0; word < 4; ++word) {
            stage.pcg_state[stream_index * 4 + word] =
                state.replay.state_words[stream_index * 4 + word];
        }
        for (int32_t word = 0; word < 2; ++word) {
            stage.pcg_buffer[stream_index * 2 + word] =
                state.replay.buffer_words[stream_index * 2 + word];
        }
        stage.pcg_cursors[stream_index] = state.replay.cursors[stream_index];
        stage.pcg_semantic[stream_index] = state.replay.semantic_draws[stream_index];
    }
    for (int64_t slot = 0; slot < capacity; ++slot) {
#ifdef EIEL_C4_PHASE_B_SURFACE_TIMING_HOOK
        if (slot == 0) phase_stamp(stage.timing, world, kTimingR4CapacityStart);
#endif
        const int64_t y = state.base.pos_y[offset + slot];
        const int64_t x = state.base.pos_x[offset + slot];
        if (y < INT32_MIN || y > INT32_MAX || x < INT32_MIN || x > INT32_MAX) {
            stage.base.status[world] = kInvalidResident;
            return;
        }
        stage.shadow_y[offset + slot] = static_cast<int32_t>(y);
        stage.shadow_x[offset + slot] = static_cast<int32_t>(x);
        double energy = state.base.policy_energy_observation[offset + slot];
        if (!isfinite(energy)) energy = state.base.energy[offset + slot];
        stage.shadow_energy[offset + slot] = energy;
        stage.base.planned_repro_cooldown[offset + slot] =
            state.base.repro_cooldown[offset + slot];
        stage.mutation_occurred[offset + slot] = false;
        stage.mutation_delta[offset + slot] = 0.0f;
        stage.base.parent_slots[offset + slot] = -1;
        stage.mate_slots[offset + slot] = -1;
        stage.base.child_slots[offset + slot] = -1;
        stage.base.child_y[offset + slot] = -1;
        stage.base.child_x[offset + slot] = -1;
        stage.base.child_energy[offset + slot] = 0.0f;
        stage.base.child_agent_ids[offset + slot] = -1;
    }
#ifdef EIEL_C4_PHASE_B_SURFACE_TIMING_HOOK
    phase_stamp(stage.timing, world, kTimingR4CapacityEnd);
    phase_stamp(stage.timing, world, kTimingR4PlannerStart);
#endif
    stage.free_head[world] = 0;
    const int64_t env = world * state.replay.streams + state.replay.environment_stream;
    const int64_t runner = world * state.replay.streams + state.replay.runner_stream;
    mate::FlatMateBirthPlanStateV1 plan_state{
        state.base.agent_id + offset, state.base.alive + offset,
        stage.shadow_y + offset, stage.shadow_x + offset,
        state.base.patch_agent + world * state.base.grid_size * state.base.grid_size,
        stage.shadow_energy + offset, state.base.health + offset,
        state.injury + offset, state.base.scar + offset,
        state.base.agent_survival_steps + offset,
        state.base.health + offset, state.base.alive + offset,
        stage.base.planned_repro_cooldown + offset,
        state.base.repro_threshold + offset, state.base.free_queue + offset,
        state.base.free_count + world, stage.free_head + world,
        reinterpret_cast<uint64_t*>(stage.pcg_state + env * 4),
        stage.pcg_buffer + env * 2, stage.pcg_semantic + env,
        reinterpret_cast<uint64_t*>(stage.pcg_state + runner * 4),
        stage.pcg_buffer + runner * 2, stage.pcg_semantic + runner,
#if defined(EIEL_C4_R11_BOUNDED_RNG_V1)
        stage.post_raw64_transitions + env,
        stage.post_output32_returns + env,
        stage.post_raw64_transitions + runner,
        stage.post_output32_returns + runner,
#endif
        state.base.agent_id_to_slot + offset,
        state.base.active_agent_ids + offset,
        state.base.active_agent_slots + offset,
        state.base.active_count + world,
    };
    mate::FlatMateBirthPlanBuffersV1 buffers{
        stage.base.planned_repro_cooldown + offset, stage.planned_energy + offset,
        stage.plan_parent + offset, stage.plan_mate + offset, stage.plan_child + offset,
        stage.spawn_positions + offset * 2, stage.plan_child_energy + offset,
        stage.parent_energy_after + offset, stage.mate_energy_after + offset,
        stage.base.decision_parent_slots + offset, stage.decision_mate_slots + offset,
        stage.base.decision_reason_codes + offset,
        stage.base.decision_adjacent_counts + offset,
        stage.decision_spawn_counts + offset, stage.reason_counts + world * 12,
        stage.base.planned_count + world, stage.base.decision_count + world,
        stage.base.spawn_draws + world, stage.base.spawn_semantic_draws + world,
        stage.runner_draws + world, stage.runner_semantic + world,
        stage.base.status + world,
    };
    mate::MateBirthPlanConfigV1 plan_config{
        capacity, state.base.grid_size, state.base.grid_size, capacity,
        config.max_births, config.parent_energy_share, config.minimum_energy_buffer,
        config.cooldown_steps, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        true, true, false, false, false, config.life_history_mode,
        config.maturity_start_ticks, config.maturity_rise_ticks,
        config.fertility_decline_start_ticks, config.fertility_decline_ticks,
        static_cast<float>(config.elder_fertility_floor),
    };
    mate::plan_births_serial_v1(plan_state, buffers, plan_config);
#ifdef EIEL_C4_PHASE_B_SURFACE_TIMING_HOOK
    phase_stamp(stage.timing, world, kTimingR4PlannerEnd);
#endif
    if (stage.base.status[world] != mate::kStatusOk) {
        stage.base.status[world] = stage.base.status[world] == mate::kStatusInvalidPcg64State
            ? kInvalidPcg64 : kInvalidResident;
        return;
    }
    if (stage.base.spawn_draws[world] > INT64_MAX - stage.pcg_cursors[env]
        || stage.runner_draws[world] > INT64_MAX - stage.pcg_cursors[runner]) {
        stage.base.status[world] = kInvalidPcg64;
        return;
    }
    stage.pcg_cursors[env] += stage.base.spawn_draws[world];
    stage.pcg_cursors[runner] += stage.runner_draws[world];
    const int64_t planned = stage.base.planned_count[world];
    const int64_t next_id = state.base.next_agent_id[world];
    int64_t max_live_id = -1;
    for (int64_t slot = 0; slot < capacity; ++slot) {
        if (state.base.alive[offset + slot])
            max_live_id = max(max_live_id, state.base.agent_id[offset + slot]);
    }
    const bool identity_invalid = config.identity_mode == kIdentitySortedMonotonic
        ? (next_id < 0 || next_id <= max_live_id || planned > INT64_MAX - next_id)
        : (next_id < 0 || planned > capacity - next_id);
    if (identity_invalid) {
        stage.base.status[world] = kCapacity;
        return;
    }
    for (int64_t birth = 0; birth < planned; ++birth) {
        const int64_t parent = stage.plan_parent[offset + birth];
        const int64_t mate_slot = stage.plan_mate[offset + birth];
        // The CPU commit requires accepted parent/mate cooldowns to have been
        // zero before planning; fail closed on its cooldown=1 edge ambiguity.
        if (state.base.repro_cooldown[offset + parent] != 0
            || state.base.repro_cooldown[offset + mate_slot] != 0) {
            stage.base.status[world] = kInvalidResident;
            return;
        }
        stage.base.parent_slots[offset + birth] = static_cast<int32_t>(parent);
        stage.mate_slots[offset + birth] = static_cast<int32_t>(mate_slot);
        stage.base.child_slots[offset + birth] =
            static_cast<int32_t>(stage.plan_child[offset + birth]);
        stage.base.child_y[offset + birth] = stage.spawn_positions[(offset + birth) * 2];
        stage.base.child_x[offset + birth] = stage.spawn_positions[(offset + birth) * 2 + 1];
        stage.base.child_energy[offset + birth] =
            static_cast<float>(stage.plan_child_energy[offset + birth]);
        stage.base.child_agent_ids[offset + birth] = next_id + birth;
    }
    stage.base.queue_cursor[world] = stage.free_head[world];
#ifdef EIEL_C4_PHASE_B_SURFACE_TIMING_HOOK
    phase_stamp(stage.timing, world, kTimingR4MutationStart);
#endif
    if (!stage_scalar_children(state, stage, world, capacity)) return;
#ifdef EIEL_C4_PHASE_B_SURFACE_TIMING_HOOK
    phase_stamp(stage.timing, world, kTimingR4MutationEnd);
#endif
    int64_t events = 0;
    int64_t accepted = 0;
    for (int64_t decision = 0; decision < stage.base.decision_count[world]; ++decision) {
        const int32_t reason = stage.base.decision_reason_codes[offset + decision];
        if (reason == 0) {
            events += stage.mutation_occurred[offset + accepted] ? 2 : 1;
            ++accepted;
        } else if ((reason >= 1 && reason <= 3) || (reason >= 5 && reason <= 6)) {
            ++events;
        }
    }
    if (state.replay.event_sequence[world] != state.base.event_active_count[world]
        || state.base.event_status[world] == 1 || state.base.event_status[world] == 2
        || events > state.base.event_capacity - state.base.event_active_count[world]) {
        stage.base.status[world] = kCapacity;
        return;
    }
}

__device__ void install_scalar(
    R4State state, R4Stage stage, int64_t world, int64_t birth) {
    const int64_t offset = world * state.base.capacity;
    const int64_t child = stage.base.child_slots[offset + birth];
    const int64_t index = offset + child;
    const float* value = stage.scalar_values + (offset + birth) * 16;
    state.genome.memory[index] = 0;
    state.genome.policy_memory[index] = 0;
    state.genome.exploration[index] = value[0];
    state.genome.aggression[index] = value[1];
    state.genome.reproduction_threshold[index] = value[2];
    state.genome.action_temperature[index] = value[3];
    state.genome.drive_inertia[index] = value[4];
    state.genome.drive_noise[index] = value[5];
    state.genome.drive_logit_gain[index] = value[6];
    state.genome.pred_lr[index] = value[7];
    state.genome.mutation_rate[index] = value[8];
    for (int32_t drive = 0; drive < 4; ++drive) {
        state.genome.drive_setpoints[index * 4 + drive] = value[9 + drive];
    }
    state.genome.exp_strength[index] = value[13];
    state.genome.exp_noise[index] = value[14];
    state.genome.drive_birth_noise[index] = value[15];
}

// The serial planner establishes the write-set used here: accepted child
// slots and spawn cells are unique, and a parent/mate slot participates in at
// most one accepted row.  Keep parent debits and ordered event/RNG publication
// in commit_kernel's thread-0 tail; this helper contains only independent
// newborn-row writes.
__device__ void install_newborn_row(
    R4State state, R4Stage stage, Config config, int64_t world,
    int64_t birth, int64_t grid_offset) {
    const int64_t capacity = state.base.capacity;
    const int64_t offset = world * capacity;
    const int64_t parent = stage.base.parent_slots[offset + birth];
    const int64_t child = stage.base.child_slots[offset + birth];
    const int64_t parent_id = state.base.agent_id[offset + parent];
    const int64_t child_id = stage.base.child_agent_ids[offset + birth];
    const float child_energy = stage.base.child_energy[offset + birth];
    state.base.birth_parent_ids[offset + birth] = parent_id;
    state.base.birth_child_ids[offset + birth] = child_id;
    state.base.mutation_occurred[offset + birth] = stage.mutation_occurred[offset + birth];
    state.base.mutation_delta[offset + birth] = stage.mutation_delta[offset + birth];
    ++state.base.slot_tokens[offset + child];
    state.base.alive[offset + child] = true;
    state.base.agent_id[offset + child] = child_id;
    state.base.pos_y[offset + child] = stage.base.child_y[offset + birth];
    state.base.pos_x[offset + child] = stage.base.child_x[offset + birth];
    const int64_t cell = grid_offset + stage.base.child_y[offset + birth] * state.base.grid_size
        + stage.base.child_x[offset + birth];
    state.base.occupancy_agent[cell] = child_id;
    state.base.occupancy_slot[cell] = child;
    state.base.patch_agent[cell] = static_cast<int32_t>(child_id);
    state.base.energy[offset + child] = child_energy;
    state.base.health[offset + child] = config.start_health;
    state.base.age[offset + child] = 0;
    state.base.survival_steps[offset + child] = 0;
    state.base.energy_collected[offset + child] = 0.0;
    if (child_id >= 0 && child_id < capacity) {
        state.base.agent_id_to_slot[offset + child_id] = static_cast<int32_t>(child);
    }
    state.base.agent_age[offset + child] = 0;
    state.base.agent_survival_steps[offset + child] = 0;
    state.base.policy_energy_observation[offset + child] = child_energy;
    state.base.repro_threshold[offset + child] = state.genome.reproduction_threshold[offset + child];
    state.base.drive_inertia[offset + child] = state.genome.drive_inertia[offset + child];
    state.base.drive_noise[offset + child] = state.genome.drive_noise[offset + child];
    const double energy_n = newborn_energy_n_cpu_oracle(
        child_energy, config.max_energy);
    state.base.drives[(offset + child) * 4] = newborn_hunger_drive_cpu_oracle(
        state.genome.drive_setpoints[(offset + child) * 4], energy_n);
    for (int32_t drive = 1; drive < 4; ++drive) {
        state.base.drives[(offset + child) * 4 + drive] = fmaxf(0.0f, fminf(1.0f,
            state.genome.drive_setpoints[(offset + child) * 4 + drive]));
    }
    for (int32_t drive = 0; drive < 4; ++drive) {
        state.base.policy_drives[(offset + child) * 4 + drive] =
            state.base.drives[(offset + child) * 4 + drive];
    }
    state.base.policy_attack_bias[offset + child] = state.genome.aggression[offset + child];
    state.base.policy_action_temperature[offset + child] = state.genome.action_temperature[offset + child];
    state.base.policy_drive_logit_gain[offset + child] = state.genome.drive_logit_gain[offset + child];
    float* chemistry = state.base.chemistry + (offset + child) * 6;
    chemistry[0] = newborn_energy_store_cpu_oracle(energy_n); chemistry[1] = 0.55f;
    chemistry[2] = 0.0f; chemistry[3] = 0.0f; chemistry[4] = 0.50f; chemistry[5] = 0.25f;
    state.base.pre_energy[offset + child] = child_energy;
    state.base.pre_health[offset + child] = config.start_health;
    state.base.pre_injury[offset + child] = 0.0f;
    state.base.pre_survival_steps[offset + child] = 0;
    const int64_t founder = state.base.lineage_founder[offset + parent];
    state.base.lineage_founder[offset + child] = founder < 0 ? parent_id : founder;
    state.base.lineage_depth[offset + child] = state.base.lineage_depth[offset + parent] + 1;
    state.base.lineage_birth_generation[offset + child] =
        config_generation(config, world);
    state.base.lineage_birth_tick[offset + child] = config_tick(config, world);
    state.base.lineage_parent[offset + child] = parent_id;
    const int32_t patch = state.base.patch_lineage[offset + parent];
    state.base.patch_lineage[offset + child] = patch == 0 ? static_cast<int32_t>(parent_id) : patch;
    state.base.scar[offset + child] = 0.0f;
    state.base.repro_cooldown[offset + child] = 0;
    state.base.offspring_count[offset + child] = 0;
    state.base.first_repro_step[offset + child] = -1;
}

__global__ void commit_kernel(R4State state, R4Stage stage, Config config) {
    if (!all_ok(stage.base, state.base.batch)) return;
    const int64_t world = blockIdx.x;
    if (world >= state.base.batch) return;
    const int64_t capacity = state.base.capacity;
    const int64_t offset = world * capacity;
    const int64_t planned = stage.base.planned_count[world];
    const int64_t grid_offset = world * state.base.grid_size * state.base.grid_size;
    for (int64_t row = threadIdx.x; row < capacity; row += blockDim.x) {
        state.base.parent_slots[offset + row] = stage.base.parent_slots[offset + row];
        state.base.mate_slots[offset + row] = stage.mate_slots[offset + row];
        state.base.child_slots[offset + row] = stage.base.child_slots[offset + row];
        state.base.child_y[offset + row] = stage.base.child_y[offset + row];
        state.base.child_x[offset + row] = stage.base.child_x[offset + row];
        state.base.child_energy[offset + row] = stage.base.child_energy[offset + row];
        state.base.decision_parent_slots[offset + row] = stage.base.decision_parent_slots[offset + row];
        state.base.decision_mate_slots[offset + row] = stage.decision_mate_slots[offset + row];
        state.base.decision_adjacent_counts[offset + row] = stage.base.decision_adjacent_counts[offset + row];
        state.base.decision_reason_codes[offset + row] = stage.base.decision_reason_codes[offset + row];
        state.base.planned_repro_cooldown[offset + row] = stage.base.planned_repro_cooldown[offset + row];
        state.base.birth_parent_ids[offset + row] = -1;
        state.base.birth_child_ids[offset + row] = -1;
        state.base.mutation_occurred[offset + row] = false;
        state.base.mutation_delta[offset + row] = 0.0f;
        if (row < planned) install_scalar(state, stage, world, row);
        if (row < planned)
            install_newborn_row(state, stage, config, world, row, grid_offset);
    }
    __syncthreads();
    if (threadIdx.x != 0) return;
    state.base.planned_count[world] = planned;
    state.base.plan_status[world] = 0;
    state.base.decision_count[world] = stage.base.decision_count[world];
    state.base.birth_count[world] = planned;
    state.base.spawn_draws[world] = stage.base.spawn_draws[world];
    state.base.spawn_semantic_draws[world] = stage.base.spawn_semantic_draws[world];
    state.base.runner_draws[world] = stage.runner_draws[world];
    state.base.runner_semantic_draws[world] = stage.runner_semantic[world];
    for (int64_t birth = 0; birth < planned; ++birth) {
        const int64_t parent = stage.base.parent_slots[offset + birth];
        const int64_t mate_slot = stage.mate_slots[offset + birth];
        const int64_t parent_id = state.base.agent_id[offset + parent];
        double parent_energy = state.base.policy_energy_observation[offset + parent];
        if (!isfinite(parent_energy)) parent_energy = state.base.energy[offset + parent];
        double mate_energy = state.base.policy_energy_observation[offset + mate_slot];
        if (!isfinite(mate_energy)) mate_energy = state.base.energy[offset + mate_slot];
        const double parent_debit = parent_energy * config.parent_energy_share;
        const double mate_debit = mate_energy * config.parent_energy_share;
        state.base.energy[offset + parent] = static_cast<float>(parent_energy - parent_debit);
        state.base.policy_energy_observation[offset + parent] = parent_energy - parent_debit;
        state.base.energy[offset + mate_slot] = static_cast<float>(mate_energy - mate_debit);
        state.base.policy_energy_observation[offset + mate_slot] = mate_energy - mate_debit;
        for (int32_t role = 0; role < 2; ++role) {
            const int64_t slot = role == 0 ? parent : mate_slot;
            state.base.repro_cooldown[offset + slot] = config.cooldown_steps;
            ++state.base.offspring_count[offset + slot];
            if (state.base.first_repro_step[offset + slot] < 0) {
                state.base.first_repro_step[offset + slot] =
                    static_cast<int32_t>(state.base.survival_steps[offset + slot]);
            }
        }
    }
    if (planned > 0) {
        int64_t remaining = 0;
        const int64_t old_count = state.base.free_count[world];
        for (int64_t row = stage.base.queue_cursor[world]; row < old_count; ++row) {
            const int64_t slot = state.base.free_queue[offset + row];
            if (slot >= 0 && slot < capacity && !state.base.alive[offset + slot]
                && state.base.agent_id[offset + slot] < 0) {
                state.base.free_queue[offset + remaining++] = slot;
            }
        }
        for (int64_t row = remaining; row < capacity; ++row) state.base.free_queue[offset + row] = -1;
        state.base.free_count[world] = remaining;
        state.base.next_agent_id[world] = stage.base.child_agent_ids[offset + planned - 1] + 1;
        for (int64_t row = 0; row < capacity; ++row) {
            state.base.active_agent_ids[offset + row] = -1;
            state.base.active_agent_slots[offset + row] = -1;
            state.base.survivor_agent_ids[offset + row] = -1;
        }
#ifndef EIEL_C4_R11_PARALLEL_ACTIVE_REBUILD
        int64_t active = 0, previous_id = INT64_MIN, previous_slot = -1;
        while (true) {
            int64_t selected_id = INT64_MAX, selected_slot = -1;
            for (int64_t slot = 0; slot < capacity; ++slot) {
                if (!state.base.alive[offset + slot] || state.base.agent_id[offset + slot] < 0) continue;
                const int64_t id = state.base.agent_id[offset + slot];
                const bool follows = id > previous_id || (id == previous_id && slot > previous_slot);
                if (follows && (id < selected_id || (id == selected_id && slot < selected_slot))) {
                    selected_id = id; selected_slot = slot;
                }
            }
            if (selected_slot < 0) break;
            state.base.active_agent_ids[offset + active] = selected_id;
            state.base.active_agent_slots[offset + active] = static_cast<int32_t>(selected_slot);
            state.base.survivor_agent_ids[offset + active] = selected_id;
            ++active; previous_id = selected_id; previous_slot = selected_slot;
        }
        state.base.active_count[world] = active;
#endif
    }
    int64_t output = state.base.event_active_count[world], success = 0;
    for (int64_t decision = 0; decision < stage.base.decision_count[world]; ++decision) {
        const int64_t parent_slot = stage.base.decision_parent_slots[offset + decision];
        const int64_t parent_id = state.base.agent_id[offset + parent_slot];
        const int32_t reason = stage.base.decision_reason_codes[offset + decision];
        if ((reason >= 1 && reason <= 3) || (reason >= 5 && reason <= 6)) {
            int64_t founder = state.base.lineage_founder[offset + parent_slot];
            if (founder < 0) founder = parent_id;
            set_event(state.base, world, output++, kBirthFailedEvent, parent_id, kSentinel,
                parent_slot, kSentinel, kSentinel, kSentinel, founder,
                state.base.lineage_depth[offset + parent_slot], reason, 0,
                static_cast<float>(reason), config);
        } else if (reason == 0) {
            const int64_t child_slot = stage.base.child_slots[offset + success];
            const int64_t child_id = stage.base.child_agent_ids[offset + success];
            if (stage.mutation_occurred[offset + success]) {
                set_event(state.base, world, output++, kMutationEvent, child_id, child_id,
                    child_slot, child_slot, parent_id, child_id,
                    state.base.lineage_founder[offset + child_slot],
                    state.base.lineage_depth[offset + child_slot], -1, 1,
                    stage.mutation_delta[offset + success], config);
            }
            set_event(state.base, world, output++, kBirthCommittedEvent, parent_id, child_id,
                parent_slot, child_slot, parent_id, child_id,
                state.base.lineage_founder[offset + child_slot],
                state.base.lineage_depth[offset + child_slot], -1, 1,
                stage.base.child_energy[offset + success], config);
            ++success;
        }
    }
    state.base.event_active_count[world] = output;
    state.base.event_status[world] = 0;
    stage.base.event_sequence[world] = output;
    state.replay.event_sequence[world] = output;
    for (int64_t stream = 0; stream < state.replay.streams; ++stream) {
        const int64_t stream_index = world * state.replay.streams + stream;
        for (int32_t word = 0; word < 4; ++word) state.replay.state_words[stream_index * 4 + word] = stage.pcg_state[stream_index * 4 + word];
        for (int32_t word = 0; word < 2; ++word) state.replay.buffer_words[stream_index * 2 + word] = stage.pcg_buffer[stream_index * 2 + word];
        state.replay.cursors[stream_index] = stage.pcg_cursors[stream_index];
        state.replay.semantic_draws[stream_index] = stage.pcg_semantic[stream_index];
    }
    const int64_t word_offset = world * scalar::kCpythonRandomWordCount;
    for (int64_t word = 0; word < scalar::kCpythonRandomWordCount; ++word) state.python.words[word_offset + word] = stage.python_words[word_offset + word];
    state.python.index[world] = stage.python_index[world];
    state.python.has_gauss[world] = stage.python_has_gauss[world];
    state.python.gauss_next[world] = stage.python_gauss_next[world];
    state.python.raw_draws[world] = stage.python_raw[world];
    state.python.semantic_draws[world] = stage.python_semantic[world];
}

R4Stage make_stage(const std::vector<torch::Tensor>& s) {
    Stage base{
        s[0].data_ptr<double>(), s[1].data_ptr<int32_t>(), s[2].data_ptr<int32_t>(),
        s[3].data_ptr<int32_t>(), s[4].data_ptr<int32_t>(), s[5].data_ptr<float>(),
        s[6].data_ptr<int64_t>(), s[7].data_ptr<int64_t>(), s[8].data_ptr<int32_t>(),
        s[9].data_ptr<int32_t>(), s[10].data_ptr<int64_t>(), s[11].data_ptr<int32_t>(),
        s[12].data_ptr<int64_t>(), s[13].data_ptr<int64_t>(), s[14].data_ptr<int64_t>(),
        s[15].data_ptr<int64_t>(), s[16].data_ptr<int64_t>(), s[17].data_ptr<int32_t>()};
    return R4Stage{
        base, s[18].data_ptr<int32_t>(), s[19].data_ptr<int64_t>(),
        s[20].data_ptr<int64_t>(), s[21].data_ptr<int64_t>(),
        s[22].data_ptr<int64_t>(), s[23].data_ptr<int64_t>(),
        s[24].data_ptr<int64_t>(), s[25].data_ptr<int64_t>(),
        s[26].data_ptr<uint32_t>(), s[27].data_ptr<int64_t>(),
        s[28].data_ptr<bool>(), s[29].data_ptr<double>(),
        s[30].data_ptr<int64_t>(), s[31].data_ptr<int64_t>(),
        s[32].data_ptr<float>(), s[33].data_ptr<bool>(), s[34].data_ptr<float>(),
        s[35].data_ptr<int32_t>(), s[36].data_ptr<int32_t>(),
        s[37].data_ptr<double>(), s[38].data_ptr<double>(),
        s[39].data_ptr<double>(), s[40].data_ptr<double>(),
        s[41].data_ptr<int32_t>(), s[42].data_ptr<int32_t>(),
        s[43].data_ptr<int64_t>(), s[44].data_ptr<int64_t>(),
        s[45].data_ptr<int64_t>(), s[46].data_ptr<int64_t>(),
        s[47].data_ptr<int64_t>(), s[48].data_ptr<double>()
#if defined(EIEL_C4_R11_BOUNDED_RNG_V1)
        , nullptr, nullptr
#endif
#ifdef EIEL_C4_PHASE_B_SURFACE_TIMING_HOOK
        , PhaseTiming{nullptr}
#endif
    };
}

}  // namespace r4

void tensor_world_c4_resident_mutation_r4_cuda(
    std::vector<torch::Tensor> primary, std::vector<torch::Tensor> inherit,
    std::vector<torch::Tensor> zero, std::vector<torch::Tensor> minus_one,
    std::vector<torch::Tensor> half, std::vector<torch::Tensor> one,
    std::vector<torch::Tensor> scalar_rng, std::vector<torch::Tensor> replay,
    std::vector<torch::Tensor> staging, int64_t max_births,
    double parent_energy_share, double minimum_energy_buffer, int64_t cooldown_steps,
    double start_health, double max_energy, int64_t life_history_mode,
    int64_t maturity_start_ticks, int64_t maturity_rise_ticks,
    int64_t fertility_decline_start_ticks, int64_t fertility_decline_ticks,
    double elder_fertility_floor, int64_t generation, int64_t tick,
    int64_t environment_stream, int64_t runner_stream) {
    TORCH_CHECK(primary.size() == 101 && inherit.size() == 29,
        "C4 R4 authoritative ABI is incomplete");
    TORCH_CHECK(scalar_rng.size() == 8 && replay.size() == 6 && staging.size() == 49,
        "C4 R4 continuation/staging ABI is incomplete");
    const int64_t batch = primary[0].size(0), capacity = primary[0].size(1);
    TORCH_CHECK(batch == 1 && capacity > 0, "C4 R4 is bounded to positive B1");
    const int64_t grid_size = primary[9].size(1), event_capacity = primary[77].size(1);
    const int64_t streams = replay[0].size(1), tape_capacity = scalar_rng[4].size(1);
    const auto device = primary[0].device();
    TORCH_CHECK(device.is_cuda(), "C4 R4 requires CUDA tensors");
    c10::cuda::CUDAGuard guard(device);
    for (const auto& group : {primary, inherit, zero, minus_one, half, one,
                              scalar_rng, replay, staging}) {
        for (const auto& tensor : group) TORCH_CHECK(
            tensor.is_cuda() && tensor.device() == device && tensor.is_contiguous(),
            "C4 R4 tensors must be contiguous on one CUDA device");
    }
    State base = make_state(primary, batch, capacity, grid_size, event_capacity);
    base.policy_use_gru = inherit[11].data_ptr<bool>();
    r4::Genome genome{
        inherit[2].data_ptr<int32_t>(), inherit[3].data_ptr<int32_t>(),
        inherit[10].data_ptr<int64_t>(), inherit[11].data_ptr<bool>(),
        inherit[12].data_ptr<float>(), inherit[13].data_ptr<float>(),
        inherit[14].data_ptr<float>(), inherit[15].data_ptr<float>(),
        inherit[16].data_ptr<float>(), inherit[17].data_ptr<float>(),
        inherit[18].data_ptr<float>(), inherit[19].data_ptr<float>(),
        inherit[20].data_ptr<float>(), inherit[21].data_ptr<float>(),
        inherit[26].data_ptr<float>(), inherit[27].data_ptr<float>(),
        inherit[28].data_ptr<float>()};
    r4::RandomState python{
        scalar_rng[0].data_ptr<uint32_t>(), scalar_rng[1].data_ptr<int64_t>(),
        scalar_rng[2].data_ptr<bool>(), scalar_rng[3].data_ptr<double>(),
        scalar_rng[4].data_ptr<double>(), scalar_rng[5].data_ptr<double>(),
        scalar_rng[6].data_ptr<int64_t>(), scalar_rng[7].data_ptr<int64_t>(),
        tape_capacity};
    r4::Replay replay_state{
        replay[0].data_ptr<int64_t>(), replay[1].data_ptr<int64_t>(),
        replay[2].data_ptr<int64_t>(), replay[3].data_ptr<int64_t>(),
        replay[4].data_ptr<int64_t>(), streams, environment_stream, runner_stream};
    r4::R4State state{base, genome, python, replay_state, zero[15].data_ptr<float>()};
    r4::R4Stage stage = r4::make_stage(staging);
    Config config{
        max_births, parent_energy_share, minimum_energy_buffer,
        static_cast<int32_t>(cooldown_steps), static_cast<float>(start_health),
        static_cast<float>(max_energy), static_cast<int32_t>(life_history_mode),
        static_cast<int32_t>(maturity_start_ticks), static_cast<int32_t>(maturity_rise_ticks),
        static_cast<int32_t>(fertility_decline_start_ticks),
        static_cast<int32_t>(fertility_decline_ticks), elder_fertility_floor,
        generation, tick};
    const auto stream = at::cuda::getCurrentCUDAStream(device.index());
    r4::preflight_kernel<<<static_cast<unsigned int>(batch), 1, 0, stream>>>(state, stage, config);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    for (auto& tensor : inherit) dispatch_inherit(tensor, stage.base, batch, capacity, stream);
    for (auto& tensor : zero) dispatch_fill(tensor, 0.0, stage.base, batch, capacity, stream);
    for (auto& tensor : minus_one) dispatch_fill(tensor, -1.0, stage.base, batch, capacity, stream);
    for (auto& tensor : half) dispatch_fill(tensor, 0.5, stage.base, batch, capacity, stream);
    for (auto& tensor : one) dispatch_fill(tensor, 1.0, stage.base, batch, capacity, stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    r4::commit_kernel<<<static_cast<unsigned int>(batch), 1, 0, stream>>>(state, stage, config);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
