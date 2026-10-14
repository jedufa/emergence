#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

#include "cuda_architecture_b_jungle_primitives.cuh"
#include "cuda_architecture_b_persistent_patch_p2.cuh"
#include "cuda_architecture_b_persistent_wdl_v1.cuh"

#include <cstdint>
#include <vector>

namespace {

namespace patch = eiel::cuda_architecture_b::persistent_patch_p2;
namespace wdl = eiel::cuda_architecture_b::persistent_wdl_v1;
namespace jungle = eiel::cuda_architecture_b::jungle_primitives;

constexpr int64_t kObservationWidth = 83;
constexpr int64_t kHiddenWidth = 64;
constexpr int64_t kActions = 7;
constexpr int64_t kPolicyPackedWidth = 9991;
constexpr float kGumbelEps = 1.0e-6f;

// State tensor ABI. Keep synchronized with c2_backend._native_state_v1.
enum T : int64_t {
    ALIVE, AGENT_ID, POS_Y, POS_X, ENERGY, HEALTH, AGENT_AGE,
    SURVIVAL, ENERGY_COLLECTED, ID_GRID_I64, SLOT_GRID_I64, ACTIONS,
    MOVED, GATHERED, FOOD_GAINED, MOVES_ATTEMPTED, MOVES_COMMITTED,
    GATHERS_COMMITTED, ACTION_UNIFORMS, ACTION_CURSOR, COVER, SHELTER,
    MOVEMENT, FOOD, SCENT, WASTE, DENSITY, FERTILITY, SCENT_PENDING,
    WASTE_PENDING, RECOVERY, VISIBILITY, AGENT_GRID_I32, FOOD_GRID_I32,
    MULTIPLIER_SCRATCH, WASTE_PENALTY_SCRATCH, DENSITY_PENALTY_SCRATCH,
    SORTED_MULTIPLIER_SCRATCH, RECOVERY_BOOST_SCRATCH,
    SORTED_RECOVERY_BOOST_SCRATCH, WDL_CUE, WDL_TIMER, WDL_ACTIVE,
    WDL_INTENSITY, WDL_AGE, WDL_EVENT_ID, WDL_PRIOR_CUE, WDL_BASE_SEED,
    WDL_NEXT_STEP, EED_A_AMOUNT, EED_B_AMOUNT, EED_A_CAPACITY,
    EED_B_CAPACITY, EED_A_RENEWAL, EED_B_RENEWAL, EED_A_PENDING,
    EED_B_PENDING, EED_A_LOAD, EED_B_LOAD, EED_CLASS, EED_ACTIVE_Y,
    EED_ACTIVE_X, EED_ACTIVE_CLASS, EED_PROGRESS, EED_AXIS, C2_PACK,
    JUNGLE_UNIFORMS, EFFECT_PRE, EFFECT_POST, HAZARD_FLAGS, EVENT_SCENT,
    EVENT_WASTE, EVENT_GATHER_AMOUNT, EVENT_GATHER_CLASS, OBSERVATIONS,
    LOGITS, PATCH_STATS, RECOVERY_STATS, JUNGLE_INTEGERS, JUNGLE_DOUBLES,
    WDL_INTEGERS, WDL_DELAY_SUM, WDL_ACTION_INTEGERS, EED_RECEIPT,
    DRIVES, ATTACK_BIAS, ACTION_TEMPERATURE, DRIVE_LOGIT_GAIN,
    TENSOR_COUNT
};

// int config: width, capacity, WDL delay/cue/active/period/radius,
// Patch radius/include-self/normalized/fertility/cap/waste/density/recovery,
// Jungle enabled/juvenile/terrain, EED applied/frozen/equalized.
enum CI : int64_t {
    I_WIDTH, I_CAPACITY, I_WDL_DELAY, I_WDL_CUE, I_WDL_ACTIVE,
    I_WDL_PERIOD, I_WDL_RADIUS, I_PATCH_RADIUS, I_PATCH_INCLUDE_SELF,
    I_PATCH_NORMALIZE, I_PATCH_USE_FERTILITY, I_PATCH_CAP_FERTILITY,
    I_PATCH_WASTE_ENABLE, I_PATCH_WASTE_PENALTY, I_PATCH_DENSITY_PENALTY,
    I_PATCH_RECOVERY, I_PATCH_RECOVERY_BOOST, I_JUNGLE_ENABLED,
    I_JUNGLE_JUVENILE, I_JUNGLE_TERRAIN, I_EED_IMPACT_APPLIED,
    I_EED_ECOLOGY_FROZEN, I_EED_EQUALIZED, I_CONFIG_COUNT
};

enum CF : int64_t {
    F_MAX_ENERGY, F_START_HEALTH, F_SCENT_DECAY, F_WASTE_DECAY,
    F_SCENT_PRESENCE, F_WASTE_PRESENCE, F_WASTE_DENSITY_EMIT,
    F_SCENT_MAX, F_WASTE_MAX, F_FOOD_MAX, F_FOOD_REGROW,
    F_FOOD_THRESHOLD, F_WASTE_PENALTY_SCALE, F_WASTE_PENALTY_MAX,
    F_DENSITY_PENALTY_SCALE, F_DENSITY_PENALTY_MAX, F_REGROWTH_MIN,
    F_RECOVERY_MIN, F_RECOVERY_MAX, F_FERTILITY_MIN, F_FERTILITY_MAX,
    F_RECOVERY_DECAY, F_RECOVERY_BASE, F_RECOVERY_COVER,
    F_RECOVERY_FERTILITY, F_RECOVERY_WASTE, F_RECOVERY_DENSITY,
    F_RECOVERY_SCENT, F_RECOVERY_BOOST_SCALE, F_RECOVERY_BOOST_MAX,
    F_JUNGLE_BASE_RISK, F_JUNGLE_SCENT, F_JUNGLE_WASTE,
    F_JUNGLE_DENSITY, F_JUNGLE_VISIBILITY, F_JUNGLE_COVER,
    F_JUNGLE_SHELTER, F_JUNGLE_RISK_MIN, F_JUNGLE_RISK_CAP,
    F_JUNGLE_DAMAGE, F_JUNGLE_JUVENILE_AGE, F_JUNGLE_JUVENILE_MULT,
    F_JUNGLE_COVER_DAMAGE, F_JUNGLE_SHELTER_DAMAGE, F_JUNGLE_TERRAIN_MIN,
    F_WDL_INTENSITY, F_WDL_DROPOUT, F_EED_DECAY, F_EED_MAX_LOAD,
    F_EED_IMPACT_PER_SUCCESS, F_EED_GATHER, F_EED_HANDLE_A,
    F_EED_HANDLE_B, F_EED_RETURN_A, F_EED_RETURN_B,
    F_MOVE_COST, F_GATHER_COST, F_REST_COST, F_ATTACK_COST,
    F_ACTION_MULT, F_REST_MULT, F_MOVEMENT_SCALE,
    F_SCENT_MOVE_EMIT, F_WASTE_MOVE_EMIT, F_SCENT_GATHER_EMIT,
    F_WASTE_GATHER_EMIT, F_W_HUNGER_GATHER, F_W_PAIN_REST,
    F_W_FATIGUE_REST, F_W_PAIN_ATTACK, F_W_FEAR_ATTACK, F_CONFIG_COUNT
};

template <typename P>
P ptr(const std::vector<torch::Tensor>& t, T index) {
    using Value = std::remove_cv_t<std::remove_pointer_t<P>>;
    return t[index].data_ptr<Value>();
}

void validate_call(
    const std::vector<torch::Tensor>& t,
    const torch::Tensor& ci,
    const torch::Tensor& cf
) {
    TORCH_CHECK(t.size() == TENSOR_COUNT, "C2 native state tensor ABI mismatch");
    TORCH_CHECK(ci.is_cuda() && ci.is_contiguous() && ci.scalar_type() == torch::kInt64
        && ci.numel() == I_CONFIG_COUNT, "C2 integer config ABI mismatch");
    TORCH_CHECK(cf.is_cuda() && cf.is_contiguous() && cf.scalar_type() == torch::kFloat64
        && cf.numel() == F_CONFIG_COUNT, "C2 floating config ABI mismatch");
    for (const auto& tensor : t) {
        TORCH_CHECK(tensor.is_cuda() && tensor.is_contiguous(),
            "all C2 native state tensors must be contiguous CUDA tensors");
    }
}

__device__ __forceinline__ float safe_norm(float value, float maximum) {
    if (maximum <= 0.0f) return 0.0f;
    return fminf(1.0f, fmaxf(0.0f, value / maximum));
}

__global__ void pre_tick_kernel(
    const bool* alive, const int64_t* pos_y, const int64_t* pos_x,
    float* health, const int64_t* agent_age, const int32_t* agent_grid,
    int32_t* food_grid, float* food, const float* fertility,
    const float* cover, const float* shelter, float* scent, float* waste,
    float* density, float* scent_pending, float* waste_pending,
    float* recovery_activity, float* visibility, float* multiplier,
    float* waste_penalty, float* density_penalty, float* sorted_multiplier,
    float* recovery_boost, float* sorted_recovery_boost,
    float* wdl_cue, int32_t* wdl_timer, float* wdl_active,
    float* wdl_intensity, int32_t* wdl_age, int64_t* wdl_event_id,
    int8_t* wdl_prior_cue, const int64_t* wdl_base_seed,
    int64_t* wdl_next_step, float* eed_a, float* eed_b,
    const float* eed_a_cap, const float* eed_b_cap,
    const float* eed_a_renew, const float* eed_b_renew,
    float* eed_a_pending, float* eed_b_pending, float* eed_a_load,
    float* eed_b_load, const float* eed_class, const double* jungle_uniforms,
    float* effect_pre, float* effect_post, uint8_t* hazard_flags,
    double* patch_stats, double* recovery_stats, int64_t* jungle_ints,
    double* jungle_doubles, int64_t* wdl_ints, double* wdl_delay,
    double* eed_receipt, const int64_t* ci, const double* cf,
    int64_t /* absolute_step */, int32_t phase
) {
    if (threadIdx.x != 0) return;
    const int64_t world = blockIdx.x;
    const int64_t width = ci[I_WIDTH], capacity = ci[I_CAPACITY];
    const int64_t cells = width * width;
    const int64_t grid_offset = world * cells;
    const int64_t slot_offset = world * capacity;

    if (phase == 0) {
    for (int64_t slot = 0; slot < capacity; ++slot) {
        effect_pre[slot_offset + slot] = 0.0f;
        effect_post[slot_offset + slot] = 0.0f;
        hazard_flags[slot_offset + slot] = 0;
    }

    wdl::Config wc{
        wdl::kSchemaVersion, static_cast<int32_t>(width), static_cast<int32_t>(width),
        static_cast<int32_t>(ci[I_WDL_DELAY]), static_cast<int32_t>(ci[I_WDL_CUE]),
        static_cast<int32_t>(ci[I_WDL_ACTIVE]), static_cast<int32_t>(ci[I_WDL_PERIOD]),
        static_cast<int32_t>(ci[I_WDL_RADIUS]), static_cast<float>(cf[F_WDL_INTENSITY]),
        static_cast<float>(cf[F_WDL_DROPOUT]), 1, 1, 1, 1, 1
    };
    wdl::StateView ws{
        wdl_cue + grid_offset, wdl_timer + grid_offset, wdl_active + grid_offset,
        wdl_intensity + grid_offset, wdl_age + grid_offset,
        wdl_event_id + grid_offset, wdl_prior_cue + grid_offset, cells
    };
    wdl::ScheduleState schedule{wdl_base_seed[world], wdl_next_step[world]};
    wdl::TickMetrics wm{};
    wdl::ScheduledEventRow we{};
    const int64_t current_step = wdl_next_step[world];
    const auto status = wdl::advance_tick_exact(
        wc, ws, &schedule, current_step, false, &wm, &we
    );
    if (status == wdl::Status::kOk) wdl_next_step[world] = schedule.next_abs_step;
    int64_t* wi = wdl_ints + world * 7;
    wi[0] = static_cast<int64_t>(status);
    wi[1] = wm.cue_cell_count; wi[2] = wm.active_cell_count;
    wi[3] = wm.event_count; wi[4] = wm.cue_without_active_count;
    wi[5] = wm.active_without_prior_cue_count;
    wi[6] = wm.cue_to_active_delay_count;
    wdl_delay[world] = wm.cue_to_active_delay_sum;
    if (status != wdl::Status::kOk) return;
    return;
    }

    if (phase == 1) {
    double renewal_a_total = 0.0, renewal_b_total = 0.0;
    double load_a_total = 0.0, load_b_total = 0.0;
    if (!ci[I_EED_ECOLOGY_FROZEN]) {
        const float retained = 1.0f - static_cast<float>(cf[F_EED_DECAY]);
        for (int64_t local = 0; local < cells; ++local) {
            const int64_t i = grid_offset + local;
            if (ci[I_EED_IMPACT_APPLIED]) {
                eed_a_load[i] = fminf(static_cast<float>(cf[F_EED_MAX_LOAD]),
                    fmaxf(0.0f, eed_a_load[i] * retained + eed_a_pending[i]));
                eed_b_load[i] = fminf(static_cast<float>(cf[F_EED_MAX_LOAD]),
                    fmaxf(0.0f, eed_b_load[i] * retained + eed_b_pending[i]));
            } else {
                eed_a_load[i] = 0.0f; eed_b_load[i] = 0.0f;
            }
            eed_a_pending[i] = 0.0f; eed_b_pending[i] = 0.0f;
            const float ra = eed_a_renew[i] * (1.0f - fminf(1.0f, fmaxf(0.0f, eed_a_load[i])));
            const float rb = eed_b_renew[i] * (1.0f - fminf(1.0f, fmaxf(0.0f, eed_b_load[i])));
            const float before_a = eed_a[i], before_b = eed_b[i];
            eed_a[i] = fminf(eed_a_cap[i], fmaxf(0.0f, eed_a[i] + ra));
            eed_b[i] = fminf(eed_b_cap[i], fmaxf(0.0f, eed_b[i] + rb));
            renewal_a_total += static_cast<double>(eed_a[i] - before_a);
            renewal_b_total += static_cast<double>(eed_b[i] - before_b);
            load_a_total += eed_a_load[i]; load_b_total += eed_b_load[i];
            const int32_t klass = static_cast<int32_t>(eed_class[i]);
            const float projected = klass == 1 ? eed_a[i] : (klass == 2 ? eed_b[i] : 0.0f);
            food[i] = projected;
            food_grid[i] = projected > cf[F_FOOD_THRESHOLD] ? patch::kFood : patch::kEmpty;
        }
    }
    double* er = eed_receipt + world * 4;
    er[0] = renewal_a_total; er[1] = renewal_b_total;
    er[2] = load_a_total; er[3] = load_b_total;
    return;
    }

    if (phase == 2) {
    patch::PatchP2Config pc{
        width, width, static_cast<float>(cf[F_SCENT_DECAY]),
        static_cast<float>(cf[F_WASTE_DECAY]), cf[F_SCENT_PRESENCE],
        cf[F_WASTE_PRESENCE], static_cast<float>(cf[F_WASTE_DENSITY_EMIT]),
        static_cast<float>(cf[F_SCENT_MAX]), static_cast<float>(cf[F_WASTE_MAX]),
        static_cast<float>(cf[F_FOOD_MAX]), 0.0f,
        static_cast<bool>(ci[I_PATCH_USE_FERTILITY]),
        static_cast<bool>(ci[I_PATCH_CAP_FERTILITY]),
        static_cast<float>(cf[F_FOOD_THRESHOLD]), ci[I_PATCH_RADIUS],
        static_cast<bool>(ci[I_PATCH_INCLUDE_SELF]),
        static_cast<bool>(ci[I_PATCH_NORMALIZE]),
        static_cast<bool>(ci[I_PATCH_WASTE_ENABLE]),
        static_cast<bool>(ci[I_PATCH_WASTE_PENALTY]),
        static_cast<float>(cf[F_WASTE_PENALTY_SCALE]),
        static_cast<float>(cf[F_WASTE_PENALTY_MAX]),
        static_cast<bool>(ci[I_PATCH_DENSITY_PENALTY]),
        static_cast<float>(cf[F_DENSITY_PENALTY_SCALE]),
        static_cast<float>(cf[F_DENSITY_PENALTY_MAX]),
        static_cast<float>(cf[F_REGROWTH_MIN])
    };
    patch::PatchP2RecoveryContext rc{};
    if (ci[I_PATCH_RECOVERY]) {
        rc = {true, recovery_activity + grid_offset, cover + grid_offset,
            recovery_boost + grid_offset, sorted_recovery_boost + grid_offset,
            recovery_stats + world * 7, static_cast<float>(cf[F_RECOVERY_MIN]),
            static_cast<float>(cf[F_RECOVERY_MAX]), static_cast<float>(cf[F_FERTILITY_MIN]),
            static_cast<float>(cf[F_FERTILITY_MAX]), static_cast<float>(cf[F_RECOVERY_DECAY]),
            static_cast<float>(cf[F_RECOVERY_BASE]), static_cast<float>(cf[F_RECOVERY_COVER]),
            static_cast<float>(cf[F_RECOVERY_FERTILITY]), static_cast<float>(cf[F_RECOVERY_WASTE]),
            static_cast<float>(cf[F_RECOVERY_DENSITY]), static_cast<float>(cf[F_RECOVERY_SCENT]),
            static_cast<bool>(ci[I_PATCH_RECOVERY_BOOST]),
            static_cast<float>(cf[F_RECOVERY_BOOST_SCALE]),
            static_cast<float>(cf[F_RECOVERY_BOOST_MAX])};
    }
    patch::patch_p2_single_tick_exact(
        {food + grid_offset, fertility + grid_offset, scent + grid_offset,
         waste + grid_offset, scent_pending + grid_offset, waste_pending + grid_offset,
         agent_grid + grid_offset, density + grid_offset, food_grid + grid_offset},
        {multiplier + grid_offset, waste_penalty + grid_offset,
         density_penalty + grid_offset, sorted_multiplier + grid_offset},
        {patch_stats + world * 25}, pc, rc
    );
    return;
    }

    if (phase != 3) return;

    int64_t* ji = jungle_ints + world * 8;
    double* jd = jungle_doubles + world * 6;
    for (int i = 0; i < 8; ++i) ji[i] = 0;
    for (int i = 0; i < 6; ++i) jd[i] = 0.0;
    if (!ci[I_JUNGLE_ENABLED]) return;
    float density_denom = static_cast<float>((2 * ci[I_PATCH_RADIUS] + 1) *
        (2 * ci[I_PATCH_RADIUS] + 1) - (ci[I_PATCH_INCLUDE_SELF] ? 0 : 1));
    if (density_denom <= 0.0f) density_denom = 1.0f;
    for (int64_t local = 0; local < cells; ++local) {
        const int64_t i = grid_offset + local;
        visibility[i] = fminf(1.0f, fmaxf(0.0f, 1.0f - cover[i]));
    }
    for (int64_t slot = 0; slot < capacity; ++slot) {
        const int64_t s = slot_offset + slot;
        if (!alive[s]) continue;
        ++ji[0];
        const int64_t g = grid_offset + pos_y[s] * width + pos_x[s];
        const float risk = jungle::predator_risk_exact({
            scent[g], waste[g], density[g], visibility[g], cover[g], shelter[g],
            static_cast<float>(cf[F_SCENT_MAX]), static_cast<float>(cf[F_WASTE_MAX]),
            density_denom, static_cast<bool>(ci[I_PATCH_NORMALIZE]),
            static_cast<float>(cf[F_JUNGLE_BASE_RISK]),
            static_cast<float>(cf[F_JUNGLE_SCENT]), static_cast<float>(cf[F_JUNGLE_WASTE]),
            static_cast<float>(cf[F_JUNGLE_DENSITY]), static_cast<float>(cf[F_JUNGLE_VISIBILITY]),
            static_cast<float>(cf[F_JUNGLE_COVER]), static_cast<float>(cf[F_JUNGLE_SHELTER]),
            static_cast<float>(cf[F_JUNGLE_RISK_MIN]), static_cast<float>(cf[F_JUNGLE_RISK_CAP])});
        if (risk <= 0.0f) continue;
        ++ji[1];
        if (jungle_uniforms[s] >= static_cast<double>(risk)) continue;
        const auto damage = jungle::predator_damage_exact({
            cf[F_JUNGLE_DAMAGE], static_cast<int32_t>(agent_age[s]),
            static_cast<int32_t>(cf[F_JUNGLE_JUVENILE_AGE]), cf[F_JUNGLE_JUVENILE_MULT],
            static_cast<bool>(ci[I_JUNGLE_JUVENILE]), static_cast<bool>(ci[I_JUNGLE_TERRAIN]),
            cover[g], shelter[g], static_cast<float>(cf[F_JUNGLE_COVER_DAMAGE]),
            static_cast<float>(cf[F_JUNGLE_SHELTER_DAMAGE]),
            static_cast<float>(cf[F_JUNGLE_TERRAIN_MIN]), health[s], 0.0f});
        effect_pre[s] = damage.health_after - health[s];
        health[s] += effect_pre[s];
        hazard_flags[s] |= 1U;
        ++ji[2]; ji[3] += damage.juvenile; ji[4] += damage.lethal;
        jd[0] += damage.raw_damage; jd[1] += damage.effective_damage;
        jd[2] += damage.prevented_damage;
    }
}

// Target-route exact Patch update for the 32x32 C2 shape. Four row-major cells
// are owned by each lane; scalar reductions retain source row order, while the
// two value-preserving sorts use a block-wide bitonic network.
__global__ void patch_parallel_1024_kernel(
    const int32_t* agent_grid, int32_t* food_grid, float* food,
    const float* fertility, const float* cover, float* scent, float* waste,
    float* density, float* scent_pending, float* waste_pending,
    float* recovery_activity, float* multiplier, float* waste_penalty,
    float* density_penalty, float* sorted_multiplier, float* recovery_boost,
    float* sorted_recovery_boost, double* stats, double* recovery_stats,
    const int64_t* ci, const double* cf
) {
    using patch::detail::clamp01;
    using patch::detail::clamp_nonnegative;
    using patch::detail::f32_add;
    using patch::detail::f32_div;
    using patch::detail::f32_mul;
    using patch::detail::f32_sub;
    using patch::detail::max_zero;
    constexpr int64_t cells = 1024;
    const int64_t lane = threadIdx.x;
    const int64_t world = blockIdx.x;
    const int64_t offset = world * cells;
    float density_denom = static_cast<float>((2 * ci[I_PATCH_RADIUS] + 1)
        * (2 * ci[I_PATCH_RADIUS] + 1) - (ci[I_PATCH_INCLUDE_SELF] ? 0 : 1));
    if (density_denom <= 0.0f) density_denom = 1.0f;

    for (int64_t local = lane; local < cells; local += blockDim.x) {
        const int64_t i = offset + local;
        const float scent_before = scent[i], waste_before = waste[i];
        const float scent_after = max_zero(f32_mul(scent_before, static_cast<float>(cf[F_SCENT_DECAY])));
        const float waste_after = max_zero(f32_mul(waste_before, static_cast<float>(cf[F_WASTE_DECAY])));
        multiplier[i] = f32_sub(scent_before, scent_after);
        waste_penalty[i] = f32_sub(waste_before, waste_after);
        density_penalty[i] = scent_pending[i];
        sorted_multiplier[i] = waste_pending[i];
        scent[i] = f32_add(scent_after, scent_pending[i]);
        waste[i] = f32_add(waste_after, waste_pending[i]);
        scent_pending[i] = 0.0f; waste_pending[i] = 0.0f;
    }
    __syncthreads();
    double scent_emitted = 0.0, scent_decayed = 0.0;
    double waste_emitted = 0.0, waste_decayed = 0.0;
    if (lane == 0) {
        for (int64_t local = 0; local < cells; ++local) {
            const int64_t i = offset + local;
            scent_decayed += static_cast<double>(multiplier[i]);
            waste_decayed += static_cast<double>(waste_penalty[i]);
            scent_emitted += static_cast<double>(density_penalty[i]);
            waste_emitted += static_cast<double>(sorted_multiplier[i]);
        }
    }
    __syncthreads();

    const int64_t width = ci[I_WIDTH];
    for (int64_t local = lane; local < cells; local += blockDim.x) {
        const int64_t y = local / width, x = local - y * width;
        const int64_t y0 = max(int64_t{0}, y - ci[I_PATCH_RADIUS]);
        const int64_t y1 = min(width, y + ci[I_PATCH_RADIUS] + 1);
        const int64_t x0 = max(int64_t{0}, x - ci[I_PATCH_RADIUS]);
        const int64_t x1 = min(width, x + ci[I_PATCH_RADIUS] + 1);
        float raw = 0.0f;
        for (int64_t yy = y0; yy < y1; ++yy)
            for (int64_t xx = x0; xx < x1; ++xx)
                if (agent_grid[offset + yy * width + xx] != -1) raw = f32_add(raw, 1.0f);
        if (!ci[I_PATCH_INCLUDE_SELF] && agent_grid[offset + local] != -1) raw = f32_sub(raw, 1.0f);
        density[offset + local] = raw;
    }
    __syncthreads();
    for (int64_t local = lane; local < cells; local += blockDim.x) {
        const int64_t i = offset + local; const float raw = density[i];
        if (ci[I_PATCH_NORMALIZE]) density[i] = clamp01(f32_div(raw, density_denom));
        const bool occupied = agent_grid[i] != -1;
        if (occupied) {
            scent[i] = f32_add(scent[i], static_cast<float>(cf[F_SCENT_PRESENCE]));
            waste[i] = f32_add(waste[i], static_cast<float>(cf[F_WASTE_PRESENCE]));
        }
        float density_emit = 0.0f;
        if (cf[F_WASTE_DENSITY_EMIT] > 0.0) {
            const float value = ci[I_PATCH_NORMALIZE] ? clamp01(f32_div(raw, density_denom)) : raw;
            density_emit = f32_mul(value, static_cast<float>(cf[F_WASTE_DENSITY_EMIT]));
            waste[i] = f32_add(waste[i], density_emit);
        }
        scent[i] = clamp_nonnegative(scent[i], static_cast<float>(cf[F_SCENT_MAX]));
        waste[i] = clamp_nonnegative(waste[i], static_cast<float>(cf[F_WASTE_MAX]));
        multiplier[i] = density_emit;
        waste_penalty[i] = occupied ? 1.0f : 0.0f;
        const float value = ci[I_PATCH_NORMALIZE] ? clamp01(f32_div(raw, density_denom)) : raw;
        density_penalty[i] = value > 1.0f ? 1.0f : 0.0f;
    }
    __syncthreads();
    int64_t occupied_count = 0, density_hotspot_count = 0;
    if (lane == 0) {
        for (int64_t local = 0; local < cells; ++local) {
            const int64_t i = offset + local;
            waste_emitted += static_cast<double>(multiplier[i]);
            occupied_count += waste_penalty[i] != 0.0f;
            density_hotspot_count += density_penalty[i] != 0.0f;
        }
        scent_emitted += cf[F_SCENT_PRESENCE] * static_cast<double>(occupied_count);
        waste_emitted += cf[F_WASTE_PRESENCE] * static_cast<double>(occupied_count);
    }
    __syncthreads();

    double recovery_positive = 0.0, recovery_negative = 0.0, recovery_decay = 0.0;
    double recovery_boost_total = 0.0; float recovery_boost_max = 0.0f;
    if (ci[I_PATCH_RECOVERY]) {
        for (int64_t local = lane; local < cells; local += blockDim.x) {
            const int64_t i = offset + local;
            const float fertility_span = f32_sub(static_cast<float>(cf[F_FERTILITY_MAX]), static_cast<float>(cf[F_FERTILITY_MIN]));
            const float fertility_norm = fertility_span > 0.0f ? clamp01(f32_div(f32_sub(fertility[i], static_cast<float>(cf[F_FERTILITY_MIN])), fertility_span)) : 0.0f;
            const float waste_norm = cf[F_WASTE_MAX] > 0 ? clamp01(f32_div(waste[i], static_cast<float>(cf[F_WASTE_MAX]))) : 0.0f;
            const float scent_norm = cf[F_SCENT_MAX] > 0 ? clamp01(f32_div(scent[i], static_cast<float>(cf[F_SCENT_MAX]))) : 0.0f;
            const float density_norm = clamp01(f32_div(density[i], density_denom));
            const float positive = f32_add(f32_add(static_cast<float>(cf[F_RECOVERY_BASE]), f32_mul(clamp01(cover[i]), static_cast<float>(cf[F_RECOVERY_COVER]))), f32_mul(fertility_norm, static_cast<float>(cf[F_RECOVERY_FERTILITY])));
            const float negative = f32_add(f32_add(f32_mul(waste_norm, static_cast<float>(cf[F_RECOVERY_WASTE])), f32_mul(density_norm, static_cast<float>(cf[F_RECOVERY_DENSITY]))), f32_mul(scent_norm, static_cast<float>(cf[F_RECOVERY_SCENT])));
            const float pre_decay = f32_sub(f32_add(recovery_activity[i], positive), negative);
            const float post_decay = f32_mul(pre_decay, static_cast<float>(cf[F_RECOVERY_DECAY]));
            recovery_activity[i] = max(static_cast<float>(cf[F_RECOVERY_MIN]), min(static_cast<float>(cf[F_RECOVERY_MAX]), post_decay));
            float boost = 0.0f;
            if (ci[I_PATCH_RECOVERY_BOOST] && cf[F_RECOVERY_MAX] > 0.0) boost = min(static_cast<float>(cf[F_RECOVERY_BOOST_MAX]), f32_mul(clamp01(f32_div(recovery_activity[i], static_cast<float>(cf[F_RECOVERY_MAX]))), static_cast<float>(cf[F_RECOVERY_BOOST_SCALE])));
            multiplier[i] = positive; waste_penalty[i] = negative;
            density_penalty[i] = f32_sub(pre_decay, post_decay);
            recovery_boost[i] = boost; sorted_recovery_boost[i] = boost;
        }
        __syncthreads();
        if (lane == 0) for (int64_t local = 0; local < cells; ++local) {
            const int64_t i = offset + local;
            recovery_positive += static_cast<double>(multiplier[i]);
            recovery_negative += static_cast<double>(waste_penalty[i]);
            recovery_decay += static_cast<double>(density_penalty[i]);
            recovery_boost_total += static_cast<double>(recovery_boost[i]);
            recovery_boost_max = max(recovery_boost_max, recovery_boost[i]);
        }
        for (int64_t span = 2; span <= cells; span <<= 1) for (int64_t stride = span >> 1; stride > 0; stride >>= 1) {
            for (int64_t left = lane; left < cells; left += blockDim.x) {
                const int64_t right = left ^ stride; if (right <= left) continue;
                const bool ascending = (left & span) == 0;
                const float lv = sorted_recovery_boost[offset + left], rv = sorted_recovery_boost[offset + right];
                if ((ascending && lv > rv) || (!ascending && lv < rv)) { sorted_recovery_boost[offset + left] = rv; sorted_recovery_boost[offset + right] = lv; }
            }
            __syncthreads();
        }
        if (lane == 0) {
            double* rs = recovery_stats + world * 7;
            rs[0]=recovery_positive;rs[1]=recovery_negative;rs[2]=max(0.0,recovery_decay);
            rs[3]=recovery_boost_total;rs[4]=recovery_boost_total/static_cast<double>(cells);
            rs[5]=recovery_boost_max;rs[6]=patch::detail::percentile_sorted_f32(sorted_recovery_boost+offset,cells,0.90f);
        }
    }
    __syncthreads();

    for (int64_t local = lane; local < cells; local += blockDim.x) {
        const int64_t i = offset + local; float m=1.0f, wp=0.0f, dp=0.0f;
        if (ci[I_PATCH_WASTE_ENABLE]) {
            const float wn = cf[F_WASTE_MAX] > 0 ? clamp01(f32_div(waste[i],static_cast<float>(cf[F_WASTE_MAX]))) : 0.0f;
            const float raw = ci[I_PATCH_NORMALIZE] ? f32_mul(density[i],density_denom) : density[i];
            const float dn = clamp01(f32_div(raw,density_denom));
            if (ci[I_PATCH_WASTE_PENALTY]) wp=min(static_cast<float>(cf[F_WASTE_PENALTY_MAX]),f32_mul(wn,static_cast<float>(cf[F_WASTE_PENALTY_SCALE])));
            if (ci[I_PATCH_DENSITY_PENALTY]) dp=min(static_cast<float>(cf[F_DENSITY_PENALTY_MAX]),f32_mul(dn,static_cast<float>(cf[F_DENSITY_PENALTY_SCALE])));
            m=max(static_cast<float>(cf[F_REGROWTH_MIN]),min(1.0f,f32_sub(1.0f,clamp01(f32_add(wp,dp)))));
        }
        multiplier[i]=m;waste_penalty[i]=wp;density_penalty[i]=dp;sorted_multiplier[i]=m;
    }
    __syncthreads();
    float multiplier_min=1.0f,waste_max_seen=0.0f,density_max_seen=0.0f;
    int64_t high_waste=0,high_density=0;
    double multiplier_mean=0.0,waste_mean=0.0,density_mean=0.0;
    if(lane==0){for(int64_t local=0;local<cells;++local){const int64_t i=offset+local;
        multiplier_min=min(multiplier_min,multiplier[i]);waste_max_seen=max(waste_max_seen,waste_penalty[i]);density_max_seen=max(density_max_seen,density_penalty[i]);high_waste+=waste_penalty[i]>0;high_density+=density_penalty[i]>0;}
        const float count=static_cast<float>(cells);multiplier_mean=f32_div(patch::detail::numpy_pairwise_sum_f32(multiplier+offset,cells),count);
        waste_mean=f32_div(patch::detail::numpy_pairwise_sum_f32(waste_penalty+offset,cells),count);density_mean=f32_div(patch::detail::numpy_pairwise_sum_f32(density_penalty+offset,cells),count);}
    for(int64_t span=2;span<=cells;span<<=1)for(int64_t stride=span>>1;stride>0;stride>>=1){for(int64_t left=lane;left<cells;left+=blockDim.x){const int64_t right=left^stride;if(right<=left)continue;const bool ascending=(left&span)==0;const float lv=sorted_multiplier[offset+left],rv=sorted_multiplier[offset+right];if((ascending&&lv>rv)||(!ascending&&lv<rv)){sorted_multiplier[offset+left]=rv;sorted_multiplier[offset+right]=lv;}}__syncthreads();}
    if(lane==0){int64_t depleted=0,full=0;for(int64_t local=0;local<cells;++local){const int64_t i=offset+local;const float cap=ci[I_PATCH_CAP_FERTILITY]?f32_mul(static_cast<float>(cf[F_FOOD_MAX]),max_zero(fertility[i])):static_cast<float>(cf[F_FOOD_MAX]);food_grid[i]=food[i]>cf[F_FOOD_THRESHOLD]?patch::kFood:patch::kEmpty;depleted+=food[i]<=cf[F_FOOD_THRESHOLD];full+=food[i]>=f32_sub(cap,1.0e-6f);}
        double* ps=stats+world*25;ps[0]=scent_emitted;ps[1]=max(0.0,scent_decayed);ps[2]=waste_emitted;ps[3]=max(0.0,waste_decayed);
        for(int i=4;i<=7;++i)ps[i]=0.0;ps[8]=multiplier_mean;ps[9]=multiplier_min;ps[10]=patch::detail::percentile_sorted_f32(sorted_multiplier+offset,cells,0.10f);ps[11]=patch::detail::percentile_sorted_f32(sorted_multiplier+offset,cells,0.90f);
        ps[12]=waste_mean;ps[13]=waste_max_seen;ps[14]=density_mean;ps[15]=density_max_seen;ps[16]=static_cast<double>(high_waste)/cells;ps[17]=static_cast<double>(high_density)/cells;
        ps[18]=ps[19]=ps[20]=0.0;ps[21]=static_cast<double>(depleted)/cells;ps[22]=static_cast<double>(full)/cells;ps[23]=static_cast<double>(density_hotspot_count)/cells;ps[24]=1.0;}
}

__global__ void observe_kernel(
    float* out, const bool* alive, const int64_t* pos_y, const int64_t* pos_x,
    const int32_t* occupancy, const float* energy, const float* health,
    const float* food, const float* cover, const float* shelter,
    const float* scent, const float* waste, const float* density,
    const float* movement, const float* wdl_cue, const int64_t* wdl_seed,
    const int64_t* ci, const double* cf, int64_t batch
) {
    const int64_t capacity = ci[I_CAPACITY], width = ci[I_WIDTH];
    const int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= batch * capacity) return;
    float* o = out + row * kObservationWidth;
    if (!alive[row]) { for (int i = 0; i < kObservationWidth; ++i) o[i] = 0.0f; return; }
    const int64_t world = row / capacity, grid_offset = world * width * width;
    const int64_t cy = pos_y[row], cx = pos_x[row];
    float density_denom = static_cast<float>((2 * ci[I_PATCH_RADIUS] + 1) *
        (2 * ci[I_PATCH_RADIUS] + 1) - (ci[I_PATCH_INCLUDE_SELF] ? 0 : 1));
    if (density_denom <= 0.0f) density_denom = 1.0f;
    for (int64_t dy = -1, t = 0; dy <= 1; ++dy) {
        for (int64_t dx = -1; dx <= 1; ++dx, ++t) {
            const int64_t y = cy + dy, x = cx + dx;
            float legacy=0, fn=0, c=0, sh=0, sn=0, wn=0, dn=0, mn=0, cue=0;
            if (y >= 0 && y < width && x >= 0 && x < width) {
                const int64_t g = grid_offset + y * width + x;
                if (occupancy[g] != -1) legacy = -1.0f;
                else if (food[g] > cf[F_FOOD_THRESHOLD]) legacy = 1.0f;
                fn=safe_norm(food[g], cf[F_FOOD_MAX]); c=fminf(1,fmaxf(0,cover[g]));
                sh=fminf(1,fmaxf(0,shelter[g])); sn=safe_norm(scent[g],cf[F_SCENT_MAX]);
                wn=safe_norm(waste[g],cf[F_WASTE_MAX]);
                dn=ci[I_PATCH_NORMALIZE] ? fminf(1,fmaxf(0,density[g])) : safe_norm(density[g],density_denom);
                const float span=4.0f; mn=fminf(1,fmaxf(0,movement[g]/span));
                cue=wdl::cue_observation_value_exact(wdl_cue[g],static_cast<float>(cf[F_WDL_DROPOUT]),wdl_seed[world],y,x);
            }
            o[t]=legacy; o[9+t]=fn; o[18+t]=c; o[27+t]=sh; o[36+t]=sn;
            o[45+t]=wn; o[54+t]=dn; o[63+t]=mn; o[72+t]=cue;
        }
    }
    o[81]=energy[row]/cf[F_MAX_ENERGY]; o[82]=health[row]/cf[F_START_HEALTH];
}

__device__ __forceinline__ void policy_exact(
    const float* pack, const float* observation, const float* drives,
    float attack_bias, float temperature, float gain, float* logits,
    const double* cf
) {
    float h1[kHiddenWidth], h2[kHiddenWidth]; int64_t offset=0;
    for (int64_t h=0;h<kHiddenWidth;++h){float v=pack[kHiddenWidth*kObservationWidth+h];
        const float* w=pack+h*kObservationWidth; for(int64_t f=0;f<kObservationWidth;++f)v+=w[f]*observation[f]; h1[h]=tanhf(v);}
    offset+=kHiddenWidth*kObservationWidth+kHiddenWidth;
    for(int64_t h=0;h<kHiddenWidth;++h){float v=pack[offset+kHiddenWidth*kHiddenWidth+h];
        const float* w=pack+offset+h*kHiddenWidth;for(int64_t i=0;i<kHiddenWidth;++i)v+=w[i]*h1[i];h2[h]=tanhf(v);}
    offset+=kHiddenWidth*kHiddenWidth+kHiddenWidth; temperature=fmaxf(.05f,temperature); gain=fmaxf(0,gain);
    for(int64_t a=0;a<kActions;++a){const float* w=pack+offset+a*kHiddenWidth;float s=pack[offset+kActions*kHiddenWidth+a];
        for(int64_t h=0;h<kHiddenWidth;++h)s+=w[h]*h2[h];if(a==6)s+=attack_bias;s/=temperature;
        if(a==4)s+=gain*cf[F_W_HUNGER_GATHER]*drives[0];else if(a==5)s+=gain*(cf[F_W_PAIN_REST]*drives[1]+cf[F_W_FATIGUE_REST]*drives[2]);
        else if(a==6)s-=gain*(cf[F_W_PAIN_ATTACK]*drives[1]+cf[F_W_FEAR_ATTACK]*drives[3]);logits[a]=s;}
}

__global__ void policy_kernel(float* logits,const float* pack,const float* obs,
    const float* drives,const float* attack,const float* temp,const float* gain,
    const bool* alive,const int64_t* ci,const double* cf,int64_t batch){
    const int64_t row=blockIdx.x;if(threadIdx.x||row>=batch*ci[I_CAPACITY])return;
    if(!alive[row]){for(int a=0;a<7;++a)logits[row*7+a]=0;return;}
    policy_exact(pack+row*kPolicyPackedWidth,obs+row*kObservationWidth,drives+row*4,
        attack[row],temp[row],gain[row],logits+row*7,cf);
}

__global__ void select_kernel(int64_t* actions,const float* logits,const double* uniforms,
    const bool* alive,int64_t* cursors,const int64_t* ci,int64_t batch){
    const int64_t capacity=ci[I_CAPACITY],row=static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(row>=batch*capacity)return;if(!alive[row]){actions[row]=-1;return;}int64_t best=0;
    float u=fminf(1-kGumbelEps,fmaxf(kGumbelEps,static_cast<float>(uniforms[row*7])));
    float score=logits[row*7]-logf(-logf(u));for(int a=1;a<7;++a){u=fminf(1-kGumbelEps,fmaxf(kGumbelEps,static_cast<float>(uniforms[row*7+a])));
        float next=logits[row*7+a]-logf(-logf(u));if(next>score){score=next;best=a;}}actions[row]=best;if(row%capacity==0)++cursors[row/capacity];}

__device__ __forceinline__ float action_cost(int64_t a,const double* cf){
    double c=a<=3?cf[F_MOVE_COST]:(a==4?cf[F_GATHER_COST]:(a==5?cf[F_REST_COST]:cf[F_ATTACK_COST]));
    if(a==5)c*=cf[F_REST_MULT];return static_cast<float>(c*cf[F_ACTION_MULT]);}

__global__ void settle_kernel(
    const bool* alive,const int64_t* agent_id,int64_t* pos_y,int64_t* pos_x,
    int64_t* id_grid,int64_t* slot_grid,int32_t* agent_grid,int32_t* food_grid,
    const int64_t* actions,float* energy,float* health,double* collected,
    const float* movement,float* food,float* scent_pending,float* waste_pending,
    float* eed_a,float* eed_b,float* eed_a_pending,float* eed_b_pending,
    const float* eed_class,int32_t* active_y,int32_t* active_x,int32_t* active_class,
    float* progress,const float* axis,const float* wdl_cue,const float* wdl_active,
    const float* wdl_intensity,bool* moved,bool* gathered,float* food_gained,
    int64_t* attempted,int64_t* committed,int64_t* gathers,float* effect_post,
    uint8_t* flags,float* event_scent,float* event_waste,float* event_gather,
    int32_t* event_class,int64_t* wdl_action,const int64_t* ci,const double* cf
) {
    if(threadIdx.x)return;const int64_t world=blockIdx.x,width=ci[I_WIDTH],capacity=ci[I_CAPACITY];
    const int64_t so=world*capacity,go=world*width*width;attempted[world]=committed[world]=gathers[world]=0;
    for(int64_t g=0;g<width*width;++g){event_scent[go+g]=0;event_waste[go+g]=0;}
    for(int64_t s=0;s<capacity;++s){const int64_t i=so+s;moved[i]=gathered[i]=false;food_gained[i]=0;effect_post[i]=0;event_gather[i]=0;event_class[i]=0;}
    int64_t* wa=wdl_action+world*5;for(int i=0;i<5;++i)wa[i]=0;
    for(int64_t s=0;s<capacity;++s){const int64_t i=so+s;if(!alive[i])continue;const int64_t a=actions[i];
        const int64_t here=go+pos_y[i]*width+pos_x[i];if(wdl_cue[here]>0)++wa[0];if(wdl_active[here]>0)++wa[1];
        if(a<0||a>3)continue;++attempted[world];int64_t ny=pos_y[i],nx=pos_x[i];if(a==0)--ny;else if(a==1)++ny;else if(a==2)++nx;else --nx;
        if(ny<0||ny>=width||nx<0||nx>=width)continue;const int64_t dst=go+ny*width+nx;if(id_grid[dst]!=-1)continue;
        const int64_t src=here;if(id_grid[src]==agent_id[i]){id_grid[src]=-1;slot_grid[src]=-1;agent_grid[src]=-1;}
        id_grid[dst]=agent_id[i];slot_grid[dst]=s;agent_grid[dst]=static_cast<int32_t>(agent_id[i]);pos_y[i]=ny;pos_x[i]=nx;moved[i]=true;++committed[world];
        const float se=static_cast<float>(cf[F_SCENT_MOVE_EMIT]),we=static_cast<float>(cf[F_WASTE_MOVE_EMIT]);
        scent_pending[dst]+=se;waste_pending[dst]+=we;event_scent[dst]+=se;event_waste[dst]+=we;}
    for(int64_t s=0;s<capacity;++s){const int64_t i=so+s;if(!alive[i])continue;if(actions[i]!=4){active_y[i]=-1;active_x[i]=-1;active_class[i]=0;progress[i]=0;continue;}
        const int64_t g=go+pos_y[i]*width+pos_x[i];const int32_t klass=static_cast<int32_t>(eed_class[g]);if(klass!=1&&klass!=2)continue;
        float* amount=klass==1?eed_a:eed_b;const float available=amount[g];if(available<=cf[F_FOOD_THRESHOLD])continue;
        float ax=ci[I_EED_EQUALIZED]?0.0f:fminf(1.0f,fmaxf(-1.0f,axis[i]));const float capacity_handle=klass==1?1.0f+ax:1.0f-ax;
        const float threshold=klass==1?cf[F_EED_HANDLE_A]:cf[F_EED_HANDLE_B];if(active_y[i]!=pos_y[i]||active_x[i]!=pos_x[i]||active_class[i]!=klass){active_y[i]=pos_y[i];active_x[i]=pos_x[i];active_class[i]=klass;progress[i]=0;}
        progress[i]+=capacity_handle;if(progress[i]<threshold)continue;const float harvest=fminf(available,static_cast<float>(cf[F_EED_GATHER]));
        const float post=fmaxf(0,available-harvest);if(!ci[I_EED_ECOLOGY_FROZEN]){amount[g]=post;food[g]=post;food_grid[g]=post>cf[F_FOOD_THRESHOLD]?1:0;}
        active_y[i]=-1;active_x[i]=-1;active_class[i]=0;progress[i]=0;food_gained[i]=klass==1?cf[F_EED_RETURN_A]:cf[F_EED_RETURN_B];
        if(ci[I_EED_IMPACT_APPLIED]&&!ci[I_EED_ECOLOGY_FROZEN]){const float signed_axis=klass==1?ax:-ax;float* pending=klass==1?eed_a_pending:eed_b_pending;pending[g]+=cf[F_EED_IMPACT_PER_SUCCESS]*fmaxf(0,signed_axis);}
        gathered[i]=true;++gathers[world];event_gather[i]=harvest;event_class[i]=klass;const float se=cf[F_SCENT_GATHER_EMIT],we=cf[F_WASTE_GATHER_EMIT];
        scent_pending[g]+=se;waste_pending[g]+=we;event_scent[g]+=se;event_waste[g]+=we;}
    for(int64_t s=0;s<capacity;++s){const int64_t i=so+s;if(!alive[i])continue;float cost=action_cost(actions[i],cf);
        if(moved[i]){const int64_t g=go+pos_y[i]*width+pos_x[i];cost=fmaxf(0,cost*fmaxf(0,1+cf[F_MOVEMENT_SCALE]*(movement[g]-1)));}
        energy[i]=fminf(cf[F_MAX_ENERGY],fmaxf(0,energy[i]-cost+food_gained[i]));if(food_gained[i]>0)collected[i]+=food_gained[i];
        const int64_t g=go+pos_y[i]*width+pos_x[i];if(wdl_active[g]>0){const float damage=fmaxf(0,wdl_intensity[g]);effect_post[i]=-fminf(health[i],damage);health[i]+=effect_post[i];flags[i]|=2U;++wa[2];if(health[i]<=0)++wa[3];}}
}

} // namespace

void c2_pre_phase(std::vector<torch::Tensor> t,torch::Tensor ci,torch::Tensor cf,int64_t absolute_step,int32_t phase){
    validate_call(t,ci,cf);const auto batch=t[ALIVE].size(0);pre_tick_kernel<<<batch,1,0,at::cuda::getCurrentCUDAStream()>>>(
        ptr<const bool*>(t,ALIVE),ptr<const int64_t*>(t,POS_Y),ptr<const int64_t*>(t,POS_X),ptr<float*>(t,HEALTH),ptr<const int64_t*>(t,AGENT_AGE),
        ptr<const int32_t*>(t,AGENT_GRID_I32),ptr<int32_t*>(t,FOOD_GRID_I32),ptr<float*>(t,FOOD),ptr<const float*>(t,FERTILITY),ptr<const float*>(t,COVER),ptr<const float*>(t,SHELTER),
        ptr<float*>(t,SCENT),ptr<float*>(t,WASTE),ptr<float*>(t,DENSITY),ptr<float*>(t,SCENT_PENDING),ptr<float*>(t,WASTE_PENDING),ptr<float*>(t,RECOVERY),ptr<float*>(t,VISIBILITY),
        ptr<float*>(t,MULTIPLIER_SCRATCH),ptr<float*>(t,WASTE_PENALTY_SCRATCH),ptr<float*>(t,DENSITY_PENALTY_SCRATCH),ptr<float*>(t,SORTED_MULTIPLIER_SCRATCH),ptr<float*>(t,RECOVERY_BOOST_SCRATCH),ptr<float*>(t,SORTED_RECOVERY_BOOST_SCRATCH),
        ptr<float*>(t,WDL_CUE),ptr<int32_t*>(t,WDL_TIMER),ptr<float*>(t,WDL_ACTIVE),ptr<float*>(t,WDL_INTENSITY),ptr<int32_t*>(t,WDL_AGE),ptr<int64_t*>(t,WDL_EVENT_ID),ptr<int8_t*>(t,WDL_PRIOR_CUE),ptr<const int64_t*>(t,WDL_BASE_SEED),ptr<int64_t*>(t,WDL_NEXT_STEP),
        ptr<float*>(t,EED_A_AMOUNT),ptr<float*>(t,EED_B_AMOUNT),ptr<const float*>(t,EED_A_CAPACITY),ptr<const float*>(t,EED_B_CAPACITY),ptr<const float*>(t,EED_A_RENEWAL),ptr<const float*>(t,EED_B_RENEWAL),ptr<float*>(t,EED_A_PENDING),ptr<float*>(t,EED_B_PENDING),ptr<float*>(t,EED_A_LOAD),ptr<float*>(t,EED_B_LOAD),ptr<const float*>(t,EED_CLASS),
        ptr<const double*>(t,JUNGLE_UNIFORMS),ptr<float*>(t,EFFECT_PRE),ptr<float*>(t,EFFECT_POST),ptr<uint8_t*>(t,HAZARD_FLAGS),ptr<double*>(t,PATCH_STATS),ptr<double*>(t,RECOVERY_STATS),ptr<int64_t*>(t,JUNGLE_INTEGERS),ptr<double*>(t,JUNGLE_DOUBLES),ptr<int64_t*>(t,WDL_INTEGERS),ptr<double*>(t,WDL_DELAY_SUM),ptr<double*>(t,EED_RECEIPT),ci.data_ptr<int64_t>(),cf.data_ptr<double>(),absolute_step,phase);C10_CUDA_KERNEL_LAUNCH_CHECK();}

void c2_wdl_step(std::vector<torch::Tensor> t,torch::Tensor ci,torch::Tensor cf){c2_pre_phase(t,ci,cf,0,0);}
void c2_eed_step(std::vector<torch::Tensor> t,torch::Tensor ci,torch::Tensor cf){c2_pre_phase(t,ci,cf,0,1);}
void c2_patch_step(std::vector<torch::Tensor> t,torch::Tensor ci,torch::Tensor cf){
    validate_call(t,ci,cf);
    if (t[FOOD].size(1) != 32 || t[FOOD].size(2) != 32) {
        c2_pre_phase(t,ci,cf,0,2);
        return;
    }
    patch_parallel_1024_kernel<<<t[ALIVE].size(0),256,0,at::cuda::getCurrentCUDAStream()>>>(
        ptr<const int32_t*>(t,AGENT_GRID_I32),ptr<int32_t*>(t,FOOD_GRID_I32),
        ptr<float*>(t,FOOD),ptr<const float*>(t,FERTILITY),ptr<const float*>(t,COVER),
        ptr<float*>(t,SCENT),ptr<float*>(t,WASTE),ptr<float*>(t,DENSITY),
        ptr<float*>(t,SCENT_PENDING),ptr<float*>(t,WASTE_PENDING),ptr<float*>(t,RECOVERY),
        ptr<float*>(t,MULTIPLIER_SCRATCH),ptr<float*>(t,WASTE_PENALTY_SCRATCH),
        ptr<float*>(t,DENSITY_PENALTY_SCRATCH),ptr<float*>(t,SORTED_MULTIPLIER_SCRATCH),
        ptr<float*>(t,RECOVERY_BOOST_SCRATCH),ptr<float*>(t,SORTED_RECOVERY_BOOST_SCRATCH),
        ptr<double*>(t,PATCH_STATS),ptr<double*>(t,RECOVERY_STATS),
        ci.data_ptr<int64_t>(),cf.data_ptr<double>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void c2_jungle_step(std::vector<torch::Tensor> t,torch::Tensor ci,torch::Tensor cf){c2_pre_phase(t,ci,cf,0,3);}

void c2_pre_tick(std::vector<torch::Tensor> t,torch::Tensor ci,torch::Tensor cf,int64_t) {
    c2_pre_phase(t,ci,cf,0,0); c2_pre_phase(t,ci,cf,0,1);
    c2_pre_phase(t,ci,cf,0,2); c2_pre_phase(t,ci,cf,0,3);
}

void c2_observe(std::vector<torch::Tensor> t,torch::Tensor ci,torch::Tensor cf){validate_call(t,ci,cf);const auto batch=t[ALIVE].size(0),capacity=t[ALIVE].size(1);const int blocks=(batch*capacity+127)/128;
    observe_kernel<<<blocks,128,0,at::cuda::getCurrentCUDAStream()>>>(ptr<float*>(t,OBSERVATIONS),ptr<const bool*>(t,ALIVE),ptr<const int64_t*>(t,POS_Y),ptr<const int64_t*>(t,POS_X),ptr<const int32_t*>(t,AGENT_GRID_I32),ptr<const float*>(t,ENERGY),ptr<const float*>(t,HEALTH),ptr<const float*>(t,FOOD),ptr<const float*>(t,COVER),ptr<const float*>(t,SHELTER),ptr<const float*>(t,SCENT),ptr<const float*>(t,WASTE),ptr<const float*>(t,DENSITY),ptr<const float*>(t,MOVEMENT),ptr<const float*>(t,WDL_CUE),ptr<const int64_t*>(t,WDL_BASE_SEED),ci.data_ptr<int64_t>(),cf.data_ptr<double>(),batch);C10_CUDA_KERNEL_LAUNCH_CHECK();}

void c2_policy(std::vector<torch::Tensor> t,torch::Tensor ci,torch::Tensor cf){validate_call(t,ci,cf);const auto rows=t[ALIVE].numel();policy_kernel<<<rows,1,0,at::cuda::getCurrentCUDAStream()>>>(ptr<float*>(t,LOGITS),ptr<const float*>(t,C2_PACK),ptr<const float*>(t,OBSERVATIONS),ptr<const float*>(t,DRIVES),ptr<const float*>(t,ATTACK_BIAS),ptr<const float*>(t,ACTION_TEMPERATURE),ptr<const float*>(t,DRIVE_LOGIT_GAIN),ptr<const bool*>(t,ALIVE),ci.data_ptr<int64_t>(),cf.data_ptr<double>(),t[ALIVE].size(0));C10_CUDA_KERNEL_LAUNCH_CHECK();}

void c2_select(std::vector<torch::Tensor> t,torch::Tensor ci,torch::Tensor cf){validate_call(t,ci,cf);const auto rows=t[ALIVE].numel();const int blocks=(rows+127)/128;select_kernel<<<blocks,128,0,at::cuda::getCurrentCUDAStream()>>>(ptr<int64_t*>(t,ACTIONS),ptr<const float*>(t,LOGITS),ptr<const double*>(t,ACTION_UNIFORMS),ptr<const bool*>(t,ALIVE),ptr<int64_t*>(t,ACTION_CURSOR),ci.data_ptr<int64_t>(),t[ALIVE].size(0));C10_CUDA_KERNEL_LAUNCH_CHECK();}

void c2_settle(std::vector<torch::Tensor> t,torch::Tensor ci,torch::Tensor cf){validate_call(t,ci,cf);const auto batch=t[ALIVE].size(0);settle_kernel<<<batch,1,0,at::cuda::getCurrentCUDAStream()>>>(
    ptr<const bool*>(t,ALIVE),ptr<const int64_t*>(t,AGENT_ID),ptr<int64_t*>(t,POS_Y),ptr<int64_t*>(t,POS_X),ptr<int64_t*>(t,ID_GRID_I64),ptr<int64_t*>(t,SLOT_GRID_I64),ptr<int32_t*>(t,AGENT_GRID_I32),ptr<int32_t*>(t,FOOD_GRID_I32),ptr<const int64_t*>(t,ACTIONS),ptr<float*>(t,ENERGY),ptr<float*>(t,HEALTH),ptr<double*>(t,ENERGY_COLLECTED),ptr<const float*>(t,MOVEMENT),ptr<float*>(t,FOOD),ptr<float*>(t,SCENT_PENDING),ptr<float*>(t,WASTE_PENDING),ptr<float*>(t,EED_A_AMOUNT),ptr<float*>(t,EED_B_AMOUNT),ptr<float*>(t,EED_A_PENDING),ptr<float*>(t,EED_B_PENDING),ptr<const float*>(t,EED_CLASS),ptr<int32_t*>(t,EED_ACTIVE_Y),ptr<int32_t*>(t,EED_ACTIVE_X),ptr<int32_t*>(t,EED_ACTIVE_CLASS),ptr<float*>(t,EED_PROGRESS),ptr<const float*>(t,EED_AXIS),ptr<const float*>(t,WDL_CUE),ptr<const float*>(t,WDL_ACTIVE),ptr<const float*>(t,WDL_INTENSITY),ptr<bool*>(t,MOVED),ptr<bool*>(t,GATHERED),ptr<float*>(t,FOOD_GAINED),ptr<int64_t*>(t,MOVES_ATTEMPTED),ptr<int64_t*>(t,MOVES_COMMITTED),ptr<int64_t*>(t,GATHERS_COMMITTED),ptr<float*>(t,EFFECT_POST),ptr<uint8_t*>(t,HAZARD_FLAGS),ptr<float*>(t,EVENT_SCENT),ptr<float*>(t,EVENT_WASTE),ptr<float*>(t,EVENT_GATHER_AMOUNT),ptr<int32_t*>(t,EVENT_GATHER_CLASS),ptr<int64_t*>(t,WDL_ACTION_INTEGERS),ci.data_ptr<int64_t>(),cf.data_ptr<double>());C10_CUDA_KERNEL_LAUNCH_CHECK();}

void c2_oracle_tick(std::vector<torch::Tensor> t,torch::Tensor ci,torch::Tensor cf) {
    c2_pre_tick(t, ci, cf, 0);
    c2_observe(t, ci, cf);
    c2_policy(t, ci, cf);
    c2_select(t, ci, cf);
    c2_settle(t, ci, cf);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("pre_tick", &c2_pre_tick);
    module.def("wdl_step", &c2_wdl_step);
    module.def("eed_step", &c2_eed_step);
    module.def("patch_step", &c2_patch_step);
    module.def("jungle_step", &c2_jungle_step);
    module.def("observe", &c2_observe);
    module.def("policy_logits", &c2_policy);
    module.def("select_actions", &c2_select);
    module.def("settle", &c2_settle);
    module.def("oracle_tick", &c2_oracle_tick);
}
