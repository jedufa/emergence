#define EIEL_C4_R11_BOUNDED_RNG_V1 1
#define EIEL_C4_R11_PARALLEL_ACTIVE_REBUILD 1
#define EIEL_C4_R11_PCG_MAX_RAW32_ATTEMPTS 64
#define EIEL_C4_R11_MT_MAX_RANDBELOW_ATTEMPTS 64
#define EIEL_C4_R11_PARALLEL_GAUSSIAN_TAPE_VALIDATION 1
#define EIEL_C4_PHASE_C_SINGLE_SHADOW 1
#define EIEL_C5_GAUSSIAN_WINDOW_V1 1
#include "tensor_world_c4_resident_population_r9.cu"

#include <type_traits>

namespace r11 {

#ifdef EIEL_C4_PHASE_B_SURFACE_TIMING_HOOK
__global__ void phase_timing_marker(
    int64_t* values, int64_t batch, int32_t edge) {
    if (threadIdx.x != 0) return;
    const int64_t world = static_cast<int64_t>(blockIdx.x);
    if (world < batch && values != nullptr)
        values[world * 16 + edge] = static_cast<int64_t>(clock64());
}
#endif

constexpr int32_t kRingCursor = 45;
constexpr int32_t kRingCapacity = 46;
constexpr int32_t kSlotTokenMismatch = 65;
constexpr int32_t kSlotTokenOverflow = 66;
constexpr int32_t kPatchProjectionOverflow = 67;
constexpr int32_t kRngInvalidAbi = 68;
constexpr int32_t kRngPcgCapacity = 69;
constexpr int32_t kRngMtCapacity = 70;
constexpr int32_t kRngCounterOverflow = 71;
constexpr int32_t kGaussianWindowAuthoritative = 0;
constexpr int32_t kGaussianWindowExhausted = 1;
constexpr int64_t kRngWindowTicks = 5;
constexpr int64_t kRngMaxBirthsPerTick = 64;

constexpr int32_t kNewbornFanoutFieldCount = 97;
constexpr int32_t kNewbornFanoutLegacyLaunchCount = 97;
constexpr int32_t kNewbornFanoutFusedLaunchCount = 1;
constexpr int32_t kNewbornFanoutThreads = 256;

enum NewbornFanoutOperation : int32_t {
    kFanoutInherit = 0,
    kFanoutZero = 1,
    kFanoutMinusOne = 2,
    kFanoutHalf = 3,
    kFanoutOne = 4,
};

enum NewbornFanoutDtype : int32_t {
    kFanoutBool = 0,
    kFanoutUInt8 = 1,
    kFanoutInt32 = 2,
    kFanoutInt64 = 3,
    kFanoutFloat32 = 4,
    kFanoutFloat64 = 5,
};

struct NewbornFanoutField {
    void* values;
    int64_t row_width;
    int64_t byte_count;
    int32_t operation;
    int32_t dtype;
};

struct NewbornFanoutFields {
    NewbornFanoutField values[kNewbornFanoutFieldCount];
    int32_t count;
};

static_assert(std::is_trivially_copyable_v<NewbornFanoutField>);
static_assert(std::is_trivially_copyable_v<NewbornFanoutFields>);

template <typename T>
__device__ __forceinline__ void apply_newborn_fanout_field(
    const NewbornFanoutField field,
    const int64_t child_index,
    const int64_t parent_index
) {
    T* values = static_cast<T*>(field.values);
    if (field.operation == kFanoutInherit) {
        values[child_index] = values[parent_index];
    } else if (field.operation == kFanoutZero) {
        values[child_index] = static_cast<T>(0);
    } else if (field.operation == kFanoutMinusOne) {
        values[child_index] = static_cast<T>(-1);
    } else if (field.operation == kFanoutHalf) {
        values[child_index] = static_cast<T>(0.5);
    } else {
        values[child_index] = static_cast<T>(1);
    }
}

__device__ __forceinline__ bool persistent_window_all_ok(
    const int32_t* status, const int64_t batch) {
    for (int64_t world = 0; world < batch; ++world) {
        if (status[world] != 0) return false;
    }
    return true;
}

#ifdef EIEL_C4_R11_TEST_ONLY_ALIAS_GUARD_HOOK
struct AliasGuardDiagnostic {
    int64_t* control;
    int32_t* local_status;
    int64_t* planned_births;
    int32_t* writer_reached;
    int32_t* writer_skipped;
    int32_t* outer_status;
    int32_t* injected;
};

__global__ void alias_guard_test_begin(
    AliasGuardDiagnostic diagnostic, int32_t* status, int64_t batch) {
    if (blockIdx.x != 0 || threadIdx.x != 0
        || diagnostic.control == nullptr) return;
    // The graph invokes this one-thread kernel in a single stream before the
    // corresponding transaction, so an ordinary device scalar increment is
    // deterministic and avoids requiring an int64 atomicAdd overload on older
    // CUDA targets.
    const int64_t step = diagnostic.control[0];
    diagnostic.control[0] = step + 1;
    diagnostic.control[2] = step;
    if (step < 0 || step >= 5) return;
    diagnostic.local_status[step] = -1;
    diagnostic.planned_births[step] = 0;
    diagnostic.writer_reached[step] = 0;
    diagnostic.writer_skipped[step] = 0;
    diagnostic.outer_status[step] = 0;
    diagnostic.injected[step] = 0;
    if (diagnostic.control[1] == step && status[0] == 0 && batch > 0) {
        // Test-only recoverable local failure.  The ordinary merge kernels
        // convert this into the existing batch-global peer-failure policy;
        // later captured steps still enter with a freshly reset local status.
        status[0] = 72;
        diagnostic.injected[step] = 1;
    }
}
#endif

__global__ void fused_newborn_fanout_kernel(
    NewbornFanoutFields fields,
    Stage stage,
    int64_t batch,
    int64_t capacity,
    const int32_t* persistent_outer_status,
    const int32_t* local_status
#ifdef EIEL_C4_R11_TEST_ONLY_ALIAS_GUARD_HOOK
    , AliasGuardDiagnostic diagnostic
#endif
) {
    __shared__ bool batch_ok;
    if (threadIdx.x == 0) {
        // Shared packed-policy rows have no shadow copy.  Their writer is
        // therefore guarded by both transaction-local success and the
        // persistent outer-window success predicate; a later locally valid
        // step cannot write after an earlier window failure.
        batch_ok = all_ok(stage, batch)
            && persistent_window_all_ok(persistent_outer_status, batch);
#ifdef EIEL_C4_R11_TEST_ONLY_ALIAS_GUARD_HOOK
        if (blockIdx.x == 0 && diagnostic.control != nullptr) {
            const int64_t step = diagnostic.control[2];
            if (step >= 0 && step < 5) {
                int64_t planned = 0;
                for (int64_t world = 0; world < batch; ++world)
                    planned += stage.planned_count[world];
                diagnostic.local_status[step] = local_status[0];
                diagnostic.planned_births[step] = planned;
                diagnostic.outer_status[step] = persistent_outer_status[0];
                diagnostic.writer_reached[step] = planned > 0 ? 1 : 0;
                diagnostic.writer_skipped[step] =
                    (planned > 0 && !persistent_window_all_ok(
                        persistent_outer_status, batch)) ? 1 : 0;
            }
        }
#endif
    }
    __syncthreads();
    if (!batch_ok) return;

    const int64_t world = static_cast<int64_t>(blockIdx.x);
    if (world >= batch) return;
    const int64_t offset = world * capacity;
    for (int64_t birth = 0; birth < stage.planned_count[world]; ++birth) {
        const int64_t parent = stage.parent_slots[offset + birth];
        const int64_t child = stage.child_slots[offset + birth];
        for (int32_t descriptor = 0; descriptor < fields.count; ++descriptor) {
            const NewbornFanoutField field = fields.values[descriptor];
            for (int64_t element = threadIdx.x; element < field.row_width;
                 element += blockDim.x) {
                const int64_t child_index =
                    (offset + child) * field.row_width + element;
                const int64_t parent_index =
                    (offset + parent) * field.row_width + element;
                switch (field.dtype) {
                    case kFanoutBool:
                        apply_newborn_fanout_field<bool>(
                            field, child_index, parent_index);
                        break;
                    case kFanoutUInt8:
                        apply_newborn_fanout_field<uint8_t>(
                            field, child_index, parent_index);
                        break;
                    case kFanoutInt32:
                        apply_newborn_fanout_field<int32_t>(
                            field, child_index, parent_index);
                        break;
                    case kFanoutInt64:
                        apply_newborn_fanout_field<int64_t>(
                            field, child_index, parent_index);
                        break;
                    case kFanoutFloat32:
                        apply_newborn_fanout_field<float>(
                            field, child_index, parent_index);
                        break;
                    default:
                        apply_newborn_fanout_field<double>(
                            field, child_index, parent_index);
                        break;
                }
            }
        }
    }
}

NewbornFanoutField make_newborn_fanout_field(
    const torch::Tensor& tensor,
    const int32_t operation,
    const int64_t batch,
    const int64_t capacity
) {
    const int64_t slot_rows = batch * capacity;
    TORCH_CHECK(slot_rows > 0 && tensor.numel() > 0
            && tensor.numel() % slot_rows == 0,
        "C4 R11 newborn fanout tensor does not have complete slot rows");
    NewbornFanoutField field{
        nullptr,
        tensor.numel() / slot_rows,
        tensor.numel() * static_cast<int64_t>(tensor.element_size()),
        operation,
        -1};
    switch (tensor.scalar_type()) {
        case torch::kBool:
            field.values = tensor.data_ptr<bool>();
            field.dtype = kFanoutBool;
            break;
        case torch::kUInt8:
            field.values = tensor.data_ptr<uint8_t>();
            field.dtype = kFanoutUInt8;
            break;
        case torch::kInt32:
            field.values = tensor.data_ptr<int32_t>();
            field.dtype = kFanoutInt32;
            break;
        case torch::kInt64:
            field.values = tensor.data_ptr<int64_t>();
            field.dtype = kFanoutInt64;
            break;
        case torch::kFloat32:
            field.values = tensor.data_ptr<float>();
            field.dtype = kFanoutFloat32;
            break;
        case torch::kFloat64:
            field.values = tensor.data_ptr<double>();
            field.dtype = kFanoutFloat64;
            break;
        default:
            TORCH_CHECK(false, "C4 R11 newborn fanout dtype is unsupported");
    }
    return field;
}

NewbornFanoutFields make_newborn_fanout_fields(
    const std::vector<torch::Tensor>& inherit,
    const std::vector<torch::Tensor>& zero,
    const std::vector<torch::Tensor>& minus_one,
    const std::vector<torch::Tensor>& half,
    const std::vector<torch::Tensor>& one,
    const int64_t batch,
    const int64_t capacity
) {
    NewbornFanoutFields fields{};
    const auto append = [&](const std::vector<torch::Tensor>& tensors,
                            const int32_t operation) {
        for (const auto& tensor : tensors) {
            TORCH_CHECK(fields.count < kNewbornFanoutFieldCount,
                "C4 R11 newborn fanout descriptor capacity exceeded");
            const NewbornFanoutField candidate = make_newborn_fanout_field(
                tensor, operation, batch, capacity);
            const uintptr_t candidate_begin = reinterpret_cast<uintptr_t>(
                candidate.values);
            const uintptr_t candidate_end = candidate_begin
                + static_cast<uintptr_t>(candidate.byte_count);
            for (int32_t prior = 0; prior < fields.count; ++prior) {
                const uintptr_t prior_begin = reinterpret_cast<uintptr_t>(
                    fields.values[prior].values);
                const uintptr_t prior_end = prior_begin
                    + static_cast<uintptr_t>(fields.values[prior].byte_count);
                TORCH_CHECK(candidate_end <= prior_begin
                        || prior_end <= candidate_begin,
                    "C4 R11 newborn fanout tensors must not alias");
            }
            fields.values[fields.count++] = candidate;
        }
    };
    append(inherit, kFanoutInherit);
    append(zero, kFanoutZero);
    append(minus_one, kFanoutMinusOne);
    append(half, kFanoutHalf);
    append(one, kFanoutOne);
    TORCH_CHECK(fields.count == kNewbornFanoutFieldCount,
        "C4 R11 newborn fanout descriptor list is incomplete");
    return fields;
}

struct IdentityStage {
    int64_t* death_tokens;
    int64_t* parent_tokens;
    int64_t* mate_tokens;
    int64_t* child_tokens;
    int64_t* next_before;
    int32_t* status;
};

// The standalone R5/R4 commit kernels retain their historical one-thread
// active-list rebuild for their independent adapters.  The composed R11 graph
// uses this deterministic block-parallel equivalent after death and birth
// commits.  Each live slot computes its stable rank by (agent_id, slot), so
// writes are collision-free and the externally visible order is unchanged.
__global__ void rebuild_active_parallel(
    State state, const int32_t* status, int64_t batch
#ifdef EIEL_C4_PHASE_B_SURFACE_TIMING_HOOK
    , r4::PhaseTiming timing
#endif
    ) {
    const int64_t world = static_cast<int64_t>(blockIdx.x);
    if (world >= batch || status[world] != 0) return;
#ifdef EIEL_C4_PHASE_B_SURFACE_TIMING_HOOK
    r4::phase_stamp(timing, world, 14);
#endif
    const int64_t slot = static_cast<int64_t>(threadIdx.x);
    if (slot >= state.capacity) return;
    const int64_t offset = world * state.capacity;
    state.active_agent_ids[offset + slot] = -1;
    state.active_agent_slots[offset + slot] = -1;
    state.survivor_agent_ids[offset + slot] = -1;
    __syncthreads();
    if (slot == 0) {
        int64_t count = 0;
        for (int64_t other = 0; other < state.capacity; ++other) {
            count += state.alive[offset + other]
                && state.agent_id[offset + other] >= 0;
        }
        state.active_count[world] = count;
    }
#ifdef EIEL_C4_PHASE_B_SURFACE_TIMING_HOOK
    if (state.alive[offset + slot] && state.agent_id[offset + slot] >= 0) {
#else
    if (!state.alive[offset + slot] || state.agent_id[offset + slot] < 0) return;
    {
#endif
        const int64_t id = state.agent_id[offset + slot];
        int64_t rank = 0;
        for (int64_t other = 0; other < state.capacity; ++other) {
            if (!state.alive[offset + other] || state.agent_id[offset + other] < 0) continue;
            const int64_t other_id = state.agent_id[offset + other];
            rank += other_id < id || (other_id == id && other < slot);
        }
        state.active_agent_ids[offset + rank] = id;
        state.active_agent_slots[offset + rank] = static_cast<int32_t>(slot);
        state.survivor_agent_ids[offset + rank] = id;
    }
#ifdef EIEL_C4_PHASE_B_SURFACE_TIMING_HOOK
    __syncthreads();
    r4::phase_stamp(timing, world, 15);
#endif
}

inline IdentityStage make_identity(const std::vector<torch::Tensor>& values) {
    return IdentityStage{
        values[0].data_ptr<int64_t>(), values[1].data_ptr<int64_t>(),
        values[2].data_ptr<int64_t>(), values[3].data_ptr<int64_t>(),
        values[4].data_ptr<int64_t>(), values[5].data_ptr<int32_t>()};
}

__global__ void capture_identity_tokens(
    State live, r4::R4Stage stage, r5::TurnStage turn, IdentityStage identity) {
    if (threadIdx.x != 0) return;
    const int64_t world = blockIdx.x;
    if (world >= live.batch) return;
    const int64_t offset = world * live.capacity;
    identity.status[world] = 0;
    identity.next_before[world] = live.next_agent_id[world];
    for (int64_t row = 0; row < live.capacity; ++row) {
        identity.death_tokens[offset + row] = -1;
        identity.parent_tokens[offset + row] = -1;
        identity.mate_tokens[offset + row] = -1;
        identity.child_tokens[offset + row] = -1;
    }
    for (int64_t row = 0; row < turn.death_count[world]; ++row) {
        const int64_t slot = turn.death_slots[offset + row];
        if (slot >= 0 && slot < live.capacity)
            identity.death_tokens[offset + row] = live.slot_tokens[offset + slot];
    }
    for (int64_t row = 0; row < stage.base.planned_count[world]; ++row) {
        const int64_t parent = stage.base.parent_slots[offset + row];
        const int64_t mate = stage.mate_slots[offset + row];
        const int64_t child = stage.base.child_slots[offset + row];
        if (parent >= 0 && parent < live.capacity)
            identity.parent_tokens[offset + row] = live.slot_tokens[offset + parent];
        if (mate >= 0 && mate < live.capacity)
            identity.mate_tokens[offset + row] = live.slot_tokens[offset + mate];
        if (child >= 0 && child < live.capacity)
            identity.child_tokens[offset + row] = live.slot_tokens[offset + child];
    }
}

__global__ void validate_identity_tokens(
    State live, r4::R4Stage stage, r5::TurnStage turn, IdentityStage identity) {
    if (threadIdx.x != 0) return;
    const int64_t world = blockIdx.x;
    if (world >= live.batch || identity.status[world] != 0) return;
    const int64_t offset = world * live.capacity;
    if (identity.next_before[world] != live.next_agent_id[world]) {
        identity.status[world] = kSlotTokenMismatch;
        return;
    }
    for (int64_t row = 0; row < turn.death_count[world]; ++row) {
        const int64_t slot = turn.death_slots[offset + row];
        bool death_mismatch = slot < 0 || slot >= live.capacity
            || live.slot_tokens[offset + slot] != identity.death_tokens[offset + row];
#ifdef EIEL_C4_PHASE_C_SINGLE_SHADOW
        // C3 validates the original biological ID during prepare, then clears
        // the disposable row before this ABA check. The token and post-death
        // sentinels are therefore the correct identity boundary here.
        death_mismatch = death_mismatch
            || live.alive[offset + slot]
            || live.agent_id[offset + slot] != -1;
#else
        death_mismatch = death_mismatch
            || live.agent_id[offset + slot] != turn.death_ids[offset + row];
#endif
        if (death_mismatch) {
            identity.status[world] = kSlotTokenMismatch;
            return;
        }
    }
    for (int64_t row = 0; row < stage.base.planned_count[world]; ++row) {
        const int64_t parent = stage.base.parent_slots[offset + row];
        const int64_t mate = stage.mate_slots[offset + row];
        const int64_t child = stage.base.child_slots[offset + row];
        const int64_t child_id = stage.base.child_agent_ids[offset + row];
        if (parent < 0 || parent >= live.capacity || mate < 0 || mate >= live.capacity
            || child < 0 || child >= live.capacity
            || live.slot_tokens[offset + parent] != identity.parent_tokens[offset + row]
            || live.slot_tokens[offset + mate] != identity.mate_tokens[offset + row]
            || live.slot_tokens[offset + child] != identity.child_tokens[offset + row]
            || !live.alive[offset + parent] || !live.alive[offset + mate]
            || turn.alive[offset + child]) {
            identity.status[world] = kSlotTokenMismatch;
            return;
        }
        if (identity.child_tokens[offset + row] == INT64_MAX) {
            identity.status[world] = kSlotTokenOverflow;
            return;
        }
        if (child_id > INT32_MAX) {
            identity.status[world] = kPatchProjectionOverflow;
            return;
        }
    }
}

__global__ void merge_identity_status(
    r4::R4Stage stage, r5::TurnStage turn, IdentityStage identity,
    int32_t* status, int64_t batch) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    bool failed = false;
    for (int64_t world = 0; world < batch; ++world)
        failed |= identity.status[world] != 0;
    if (!failed) return;
    for (int64_t world = 0; world < batch; ++world) {
        const int32_t code = identity.status[world];
        if (status[world] == 0) status[world] = code == 0 ? r6::kPeerFailed : code;
        if (turn.status[world] == 0) turn.status[world] = r5::kPeerFailed;
        if (stage.base.status[world] == 0) stage.base.status[world] = r4::kInvalidResident;
    }
}

__global__ void publish_first_status(
    r4::R4Stage stage, int32_t* status, int64_t batch) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    for (int64_t world = 0; world < batch; ++world)
        if (status[world] == 0) status[world] = stage.base.status[world];
}

struct RingTape {
    int32_t* event_type;
    int64_t* generation;
    int64_t* tick;
    int64_t* actor;
    int64_t* target;
    float* value;
    int32_t* typed_event_type;
    int64_t* typed_generation;
    int64_t* typed_tick;
    int64_t* typed_sequence;
    int64_t* typed_actor;
    int64_t* typed_target;
    int64_t* typed_actor_slot;
    int64_t* typed_target_slot;
    int64_t* typed_parent;
    int64_t* typed_child;
    int64_t* typed_founder;
    int32_t* typed_depth;
    int32_t* typed_subtype;
    int32_t* typed_reason;
    int8_t* typed_success;
    float* typed_value;
    int64_t* active_count;
    int32_t* status;
};

struct RingState {
    int64_t* bank_counts;
    int64_t* bank_start_sequences;
    int64_t* producer_cursor;
    int64_t* consumer_cursor;
    int32_t* active_bank;
    RingTape banks[2];
    int64_t batch;
    int64_t capacity;
};

inline RingTape make_ring_tape(const std::vector<torch::Tensor>& ring, int64_t offset) {
    return RingTape{
        ring[offset + 0].data_ptr<int32_t>(), ring[offset + 1].data_ptr<int64_t>(),
        ring[offset + 2].data_ptr<int64_t>(), ring[offset + 3].data_ptr<int64_t>(),
        ring[offset + 4].data_ptr<int64_t>(), ring[offset + 5].data_ptr<float>(),
        ring[offset + 6].data_ptr<int32_t>(), ring[offset + 7].data_ptr<int64_t>(),
        ring[offset + 8].data_ptr<int64_t>(), ring[offset + 9].data_ptr<int64_t>(),
        ring[offset + 10].data_ptr<int64_t>(), ring[offset + 11].data_ptr<int64_t>(),
        ring[offset + 12].data_ptr<int64_t>(), ring[offset + 13].data_ptr<int64_t>(),
        ring[offset + 14].data_ptr<int64_t>(), ring[offset + 15].data_ptr<int64_t>(),
        ring[offset + 16].data_ptr<int64_t>(), ring[offset + 17].data_ptr<int32_t>(),
        ring[offset + 18].data_ptr<int32_t>(), ring[offset + 19].data_ptr<int32_t>(),
        ring[offset + 20].data_ptr<int8_t>(), ring[offset + 21].data_ptr<float>(),
        ring[offset + 22].data_ptr<int64_t>(), ring[offset + 23].data_ptr<int32_t>(),
    };
}

inline RingState make_ring(const std::vector<torch::Tensor>& ring, int64_t batch,
                           int64_t capacity) {
    return RingState{
        ring[0].data_ptr<int64_t>(), ring[1].data_ptr<int64_t>(),
        ring[2].data_ptr<int64_t>(), ring[3].data_ptr<int64_t>(),
        ring[4].data_ptr<int32_t>(), make_ring_tape(ring, 5),
        make_ring_tape(ring, 29), batch, capacity};
}

__global__ void reset_local_event_cursor(
    State state, r4::Replay replay, RingState ring, int32_t* status) {
    if (threadIdx.x != 0) return;
    const int64_t world = static_cast<int64_t>(blockIdx.x);
    if (world >= ring.batch) return;
    // Graph capture deliberately poisons this surface after capture.  Every
    // replay starts a new transaction and must clear that capture sentinel
    // before the first semantic preflight reads it.
    status[world] = 0;
    // The bundle event tape is a per-transaction scratch surface in R11;
    // the ring producer is the durable continuation sequence.  A mismatch
    // means the checkpoint/restored owners are not the same trajectory.
    if (replay.event_sequence[world] != ring.producer_cursor[world]) {
        status[world] = kRingCursor;
        return;
    }
    state.event_active_count[world] = 0;
    state.event_status[world] = 0;
    replay.event_sequence[world] = 0;
}

#ifdef EIEL_C4_R11_TEST_ONLY_ACTIVE_PREFLIGHT_HOOK
struct ActivePreflightDiagnosticStage {
    int64_t* active_count;
    int64_t* alive_sum;
    int32_t* tail_valid;
    int32_t* order_valid;
    int64_t* device_pointers;
    int64_t* checksum;
};

__global__ void capture_active_preflight_diagnostic(
    State live, ActivePreflightDiagnosticStage diagnostic) {
    if (threadIdx.x != 0) return;
    const int64_t world = static_cast<int64_t>(blockIdx.x);
    if (world >= live.batch) return;
    const int64_t offset = world * live.capacity;
    const int64_t count = live.active_count[world];
    int64_t alive = 0;
    bool tail_ok = count >= 0 && count <= live.capacity;
    bool order_ok = tail_ok;
    int64_t previous = -1;
    for (int64_t slot = 0; slot < live.capacity; ++slot)
        alive += live.alive[offset + slot] ? 1 : 0;
    for (int64_t row = 0; row < live.capacity; ++row) {
        const int64_t id = live.active_agent_ids[offset + row];
        const int64_t slot = live.active_agent_slots[offset + row];
        if (row >= count) {
            tail_ok &= id == -1 && slot == -1;
            continue;
        }
        order_ok &= id >= 0 && (row == 0 || id > previous)
            && slot >= 0 && slot < live.capacity
            && live.alive[offset + slot]
            && live.agent_id[offset + slot] == id;
        previous = id;
    }
    diagnostic.active_count[world] = count;
    diagnostic.alive_sum[world] = alive;
    diagnostic.tail_valid[world] = tail_ok ? 1 : 0;
    diagnostic.order_valid[world] = order_ok ? 1 : 0;
    if (world == 0) {
        diagnostic.device_pointers[0] = static_cast<int64_t>(
            reinterpret_cast<uintptr_t>(live.alive));
        diagnostic.device_pointers[1] = static_cast<int64_t>(
            reinterpret_cast<uintptr_t>(live.active_count));
        diagnostic.device_pointers[2] = static_cast<int64_t>(
            reinterpret_cast<uintptr_t>(live.active_agent_ids));
        diagnostic.device_pointers[3] = static_cast<int64_t>(
            reinterpret_cast<uintptr_t>(live.active_agent_slots));
        diagnostic.device_pointers[4] = static_cast<int64_t>(
            reinterpret_cast<uintptr_t>(live.agent_id));
    }
}

__global__ void active_preflight_noop() {}

__global__ void active_preflight_system_fence() {
    if (blockIdx.x == 0 && threadIdx.x == 0) __threadfence_system();
}

__global__ void capture_active_preflight_checksum(
    State live, ActivePreflightDiagnosticStage diagnostic) {
    if (threadIdx.x != 0) return;
    const int64_t world = static_cast<int64_t>(blockIdx.x);
    if (world >= live.batch) return;
    const int64_t offset = world * live.capacity;
    volatile const bool* alive = live.alive;
    volatile const int64_t* active_count = live.active_count;
    volatile const int64_t* active_ids = live.active_agent_ids;
    volatile const int32_t* active_slots = live.active_agent_slots;
    volatile const int64_t* agent_ids = live.agent_id;
    uint64_t value = 1469598103934665603ULL;
    value = (value ^ static_cast<uint64_t>(active_count[world]))
        * 1099511628211ULL;
    for (int64_t row = 0; row < live.capacity; ++row) {
        value = (value ^ static_cast<uint64_t>(alive[offset + row]))
            * 1099511628211ULL;
        value = (value ^ static_cast<uint64_t>(active_ids[offset + row]))
            * 1099511628211ULL;
        value = (value ^ static_cast<uint64_t>(active_slots[offset + row]))
            * 1099511628211ULL;
        value = (value ^ static_cast<uint64_t>(agent_ids[offset + row]))
            * 1099511628211ULL;
    }
    diagnostic.checksum[world] = static_cast<int64_t>(value);
}
#endif

__device__ __forceinline__ bool rng_counter_overflows(
    int64_t value, int64_t addition) {
    return value < 0 || addition < 0 || value > INT64_MAX - addition;
}

__device__ __forceinline__ int64_t pcg_raw64_for_output32(
    int64_t output32, bool cached32) {
    const int64_t cached = cached32 ? 1 : 0;
    return max(static_cast<int64_t>(0), (output32 - cached + 1) / 2);
}

__device__ void mt_window_bounds(
    int64_t births, bool initial_has_gauss,
    int64_t* raw32, int64_t* tape_headroom) {
    int64_t raw = 0;
    int64_t last_tape = -1;
    bool has_gauss = initial_has_gauss;
    for (int64_t birth = 0; birth < births; ++birth) {
        raw += 2;
        for (int32_t gene = 0; gene < 8; ++gene) {
            raw += 2;
            if (has_gauss) {
                has_gauss = false;
            } else {
                last_tape = raw;
                raw += 4;
                has_gauss = true;
            }
        }
        raw += 2;
        if (has_gauss) {
            has_gauss = false;
        } else {
            last_tape = raw;
            raw += 4;
            has_gauss = true;
        }
        raw += 2 + EIEL_C4_R11_MT_MAX_RANDBELOW_ATTEMPTS;
    }
    *raw32 = raw;
    *tape_headroom = last_tape + 1;
}

__global__ void preflight_rng_window(
    State live, r4::Replay replay, r4::RandomState python,
    const int64_t* requested_ticks, const int64_t* completed_ticks,
    int32_t* status, int64_t batch, int64_t max_births
#ifdef EIEL_C4_PHASE_B_SURFACE_TIMING_HOOK
    , r4::PhaseTiming timing
#endif
    ) {
    if (threadIdx.x != 0) return;
    const int64_t world = static_cast<int64_t>(blockIdx.x);
    if (world >= batch || status[world] != 0) return;
#ifdef EIEL_C4_PHASE_B_SURFACE_TIMING_HOOK
    r4::phase_stamp(timing, world, 2);
#endif
    const int64_t requested = requested_ticks[world];
    const int64_t completed = completed_ticks[world];
    if (requested < 1 || requested > kRngWindowTicks
        || completed < 0 || completed >= requested
        || max_births < 0 || max_births > kRngMaxBirthsPerTick
        || max_births > live.capacity) {
        status[world] = kRngInvalidAbi;
        return;
    }
    const int64_t remaining = requested - completed;
    const int64_t active = live.active_count[world];
    if (active < 0 || active > live.capacity) {
        status[world] = kRngInvalidAbi;
        return;
    }
    const int64_t decisions = max_births == 0 ? 0 : active;
    const int64_t env = world * replay.streams + replay.environment_stream;
    const int64_t runner = world * replay.streams + replay.runner_stream;
    if (replay.raw64_transitions == nullptr || replay.output32_returns == nullptr) {
        status[world] = kRngInvalidAbi;
        return;
    }
    for (int64_t stream = 0; stream < replay.streams; ++stream) {
        const int64_t index = world * replay.streams + stream;
        if (replay.raw64_transitions[index] < 0
            || replay.output32_returns[index] < 0
            || replay.cursors[index] < 0 || replay.semantic_draws[index] < 0) {
            status[world] = kRngInvalidAbi;
            return;
        }
    }
    const int64_t env_output32 = remaining * decisions * 259;
    const int64_t runner_output32 = remaining * decisions * 66;
    const int64_t env_raw64 = pcg_raw64_for_output32(
        env_output32, replay.buffer_words[env * 2] == 1);
    const int64_t runner_raw64 = remaining * decisions * 4 + pcg_raw64_for_output32(
        runner_output32, replay.buffer_words[runner * 2] == 1);
    if (rng_counter_overflows(replay.raw64_transitions[env], env_raw64)
        || rng_counter_overflows(replay.output32_returns[env], env_output32)
        || rng_counter_overflows(replay.cursors[env], remaining * decisions * 7)
        || rng_counter_overflows(replay.semantic_draws[env], remaining * decisions * 8)
        || rng_counter_overflows(replay.raw64_transitions[runner], runner_raw64)
        || rng_counter_overflows(replay.output32_returns[runner], runner_output32)
        || rng_counter_overflows(replay.cursors[runner], remaining * decisions * 3)
        || rng_counter_overflows(replay.semantic_draws[runner], remaining * decisions * 4)) {
        status[world] = kRngCounterOverflow;
        return;
    }
#if !defined(EIEL_C4_R11_RNGV2)
    int64_t mt_raw32 = 0, mt_tape = 0;
    mt_window_bounds(remaining * max_births,
        python.has_gauss[world], &mt_raw32, &mt_tape);
    int64_t window_base = 0;
    int64_t window_end = python.tape_capacity;
    if (python.window_base_raw_draws != nullptr) {
        window_base = python.window_base_raw_draws[world];
        window_end = python.window_valid_raw_draw_ends[world];
        bool provenance_present = false;
        for (int32_t byte = 0; byte < 32; ++byte) {
            provenance_present = provenance_present
                || python.window_provenance_sha256[world * 32 + byte] != 0;
        }
        if (python.window_status[world] != kGaussianWindowAuthoritative
            || python.window_refill_generations[world] < 0
            || python.window_exhaustion_raw_draws[world] != -1
            || window_base < 0 || window_end < window_base
            || window_end - window_base > python.tape_capacity
            || !provenance_present) {
            status[world] = kRngInvalidAbi;
            return;
        }
    }
    if (python.raw_draws[world] < window_base
        || python.raw_draws[world] > window_end
        || mt_tape > window_end - python.raw_draws[world]) {
        if (python.window_status != nullptr) {
            python.window_status[world] = kGaussianWindowExhausted;
            python.window_exhaustion_raw_draws[world] = python.raw_draws[world];
        }
        status[world] = kRngMtCapacity;
        return;
    }
    if (rng_counter_overflows(python.raw_draws[world], mt_raw32)
        || rng_counter_overflows(
            python.semantic_draws[world], remaining * max_births * 21)) {
        status[world] = kRngCounterOverflow;
    }
#else
    if (python.tape_capacity != 0) {
        status[world] = kRngInvalidAbi;
        return;
    }
    (void)remaining;
    (void)max_births;
#endif
#ifdef EIEL_C4_PHASE_B_SURFACE_TIMING_HOOK
    r4::phase_stamp(timing, world, 3);
#endif
}

// The tape is immutable trajectory input for a resident graph window.  The
// historical R4 path checked both 65,536-entry tapes once per world from the
// planner's single thread.  This exact finite/NaN validation is independent
// per entry, so perform it in a deterministic fixed 256-thread block while
// leaving the accepted MT/Gaussian draw order untouched in R4.
__global__ void validate_gaussian_tape_parallel(
    r4::RandomState python, int32_t* stage_status, int64_t batch
#ifdef EIEL_C4_PHASE_B_SURFACE_TIMING_HOOK
    , r4::PhaseTiming timing
#endif
    ) {
    const int64_t world = static_cast<int64_t>(blockIdx.x);
    if (world >= batch) return;
    __shared__ int32_t invalid;
    if (threadIdx.x == 0) {
        invalid = 0;
        stage_status[world] = 0;
#ifdef EIEL_C4_PHASE_B_SURFACE_TIMING_HOOK
        r4::phase_stamp(timing, world, r4::kTimingGaussianStart);
#endif
    }
    __syncthreads();
    const int64_t base = world * python.tape_capacity;
    for (int64_t row = static_cast<int64_t>(threadIdx.x);
         row < python.tape_capacity; row += blockDim.x) {
        if (!isfinite(python.primary_tape[base + row])
            || !isfinite(python.secondary_tape[base + row])) {
            atomicExch(&invalid, 1);
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        if (invalid != 0) stage_status[world] = r4::kInvalidPython;
#ifdef EIEL_C4_PHASE_B_SURFACE_TIMING_HOOK
        r4::phase_stamp(timing, world, r4::kTimingGaussianEnd);
#endif
    }
}

__global__ void prepare_rng_counter_shadow(
    r4::Replay replay, r4::R4Stage stage, int64_t batch) {
    if (threadIdx.x != 0) return;
    const int64_t world = static_cast<int64_t>(blockIdx.x);
    if (world >= batch) return;
    for (int64_t stream = 0; stream < replay.streams; ++stream) {
        const int64_t index = world * replay.streams + stream;
        stage.post_raw64_transitions[index] = replay.raw64_transitions[index];
        stage.post_output32_returns[index] = replay.output32_returns[index];
    }
}

__global__ void commit_rng_counter_shadow(
    r4::Replay replay, r4::R4Stage stage, int64_t batch) {
    if (threadIdx.x != 0 || !::all_ok(stage.base, batch)) return;
    const int64_t world = static_cast<int64_t>(blockIdx.x);
    if (world >= batch) return;
    for (int64_t stream = 0; stream < replay.streams; ++stream) {
        const int64_t index = world * replay.streams + stream;
        replay.raw64_transitions[index] = stage.post_raw64_transitions[index];
        replay.output32_returns[index] = stage.post_output32_returns[index];
    }
}

__global__ void sync_stage_event_cursor(
    r4::R4Stage stage, r5::TurnStage turn, int64_t batch) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    for (int64_t world = 0; world < batch; ++world)
        stage.base.event_sequence[world] = turn.replay_event_sequence[world];
}

__device__ __forceinline__ bool ring_can_append(
    RingState ring, int64_t world, int64_t rows) {
    if (rows < 0) return false;
    int64_t producer = ring.producer_cursor[world];
    const int64_t consumer = ring.consumer_cursor[world];
    int32_t active = ring.active_bank[world];
    if (producer < consumer || active < 0 || active > 1) return false;
    int64_t counts[2] = {
        ring.bank_counts[world * 2], ring.bank_counts[world * 2 + 1]};
    int64_t starts[2] = {
        ring.bank_start_sequences[world * 2],
        ring.bank_start_sequences[world * 2 + 1]};
    for (int bank = 0; bank < 2; ++bank) {
        if (counts[bank] < 0 || counts[bank] > ring.capacity) return false;
        if (counts[bank] == 0) continue;
        if (starts[bank] < 0 || starts[bank] > producer
            || counts[bank] > producer - starts[bank]) return false;
    }
    for (int64_t row = 0; row < rows; ++row) {
        if (counts[active] >= ring.capacity) {
            const int32_t other = 1 - active;
            const int64_t other_end = starts[other] + counts[other];
            if (counts[other] != 0 && other_end > consumer) return false;
            counts[other] = 0;
            starts[other] = producer;
            active = other;
        }
        if (counts[active] == 0) starts[active] = producer;
        ++counts[active];
        ++producer;
        if (producer < 0) return false;
    }
    return true;
}

__global__ void preflight_event_ring(
    r4::R4Stage stage, r5::TurnStage turn, RingState ring, int32_t* status,
    int64_t capacity) {
    if (threadIdx.x != 0) return;
    const int64_t world = static_cast<int64_t>(blockIdx.x);
    if (world >= ring.batch || stage.base.status[world] != 0) return;
    const int64_t offset = world * capacity;
    int64_t events = turn.death_count[world];
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
    if (!ring_can_append(ring, world, events)) {
        status[world] = kRingCapacity;
        stage.base.status[world] = kRingCapacity;
        return;
    }
    // The local event tape is reset at transaction start.  This staged end
    // cursor is consumed by ring commit and never becomes the durable C5
    // sequence; ring commit publishes the absolute producer cursor.
    stage.base.event_sequence[world] = events;
}

struct WindowControlState {
    int32_t* preflight_status;
    int32_t* global_status;
    int32_t* c4_status;
    int32_t* commit_status;
    int32_t* stop_requested;
    int32_t* first_error_stage;
    int32_t* first_error_world;
    int32_t* first_error_code;
    int64_t* first_error_tick;
    int64_t* tick;
    int64_t* completed_ticks;
    int64_t* current_tick;
    int64_t* receipt_write_index;
    int64_t batch;
};

// This replaces the captured torch.where/arange/min/gather fanout with one
// device control barrier.  The reduction and first-error arbitration remain
// exactly the source order; per-world copies/counters are independent after
// the shared decision is made.
__global__ void merge_window_control_kernel(WindowControlState control) {
    __shared__ int32_t prior_ok;
    __shared__ int32_t local_ok;
    __shared__ int64_t failed_world;
    __shared__ int32_t first_code;
    if (threadIdx.x == 0) {
        prior_ok = 1;
        local_ok = 1;
        int64_t primary = control.batch;
        int64_t any = control.batch;
        for (int64_t world = 0; world < control.batch; ++world) {
            if (control.global_status[world] != 0) prior_ok = 0;
            const int32_t status = control.preflight_status[world];
            if (status == 0) continue;
            local_ok = 0;
            if (any == control.batch) any = world;
            if (status != r6::kPeerFailed && primary == control.batch)
                primary = world;
        }
        failed_world = primary < control.batch ? primary : any;
        const int64_t safe_world = failed_world < control.batch
            ? failed_world : control.batch - 1;
        first_code = control.preflight_status[safe_world];
    }
    __syncthreads();
    const bool ok = prior_ok != 0 && local_ok != 0;
    const bool record = prior_ok != 0 && local_ok == 0;
    const int64_t world = static_cast<int64_t>(threadIdx.x);
    if (world >= control.batch) return;
    const int32_t status = control.preflight_status[world];
    control.c4_status[world] = status;
    control.commit_status[world] = status == 0 ? 0 : status;
    control.global_status[world] = ok ? control.global_status[world] : 1;
    control.stop_requested[world] = ok ? control.stop_requested[world] : 1;
    if (record) {
        control.first_error_world[world] = static_cast<int32_t>(failed_world);
        control.first_error_code[world] = first_code;
        control.first_error_stage[world] = 4;
        control.first_error_tick[world] = control.tick[world];
    }
    if (ok) {
        control.completed_ticks[world] += 1;
        control.receipt_write_index[world] += 1;
        control.current_tick[world] = control.tick[world];
    }
}

struct WindowClockState {
    int32_t* global_status;
    int64_t* tick;
    int64_t* generation;
    int64_t batch;
};

__global__ void advance_window_clock_kernel(WindowClockState clock) {
    const int64_t world = static_cast<int64_t>(blockIdx.x) * blockDim.x
        + threadIdx.x;
    if (world >= clock.batch || clock.global_status[world] != 0) return;
    clock.tick[world] += 1;
    clock.generation[world] += 1;
}

__device__ void copy_ring_row(
    State state, RingTape tape, int64_t source, int64_t destination,
    int64_t sequence) {
    tape.event_type[destination] = state.event_type[source];
    tape.generation[destination] = state.event_generation[source];
    tape.tick[destination] = state.event_tick[source];
    tape.actor[destination] = state.event_actor[source];
    tape.target[destination] = state.event_target[source];
    tape.value[destination] = state.event_value[source];
    tape.typed_event_type[destination] = state.typed_event_type[source];
    tape.typed_generation[destination] = state.typed_generation[source];
    tape.typed_tick[destination] = state.typed_tick[source];
    tape.typed_sequence[destination] = sequence;
    tape.typed_actor[destination] = state.typed_actor[source];
    tape.typed_target[destination] = state.typed_target[source];
    tape.typed_actor_slot[destination] = state.typed_actor_slot[source];
    tape.typed_target_slot[destination] = state.typed_target_slot[source];
    tape.typed_parent[destination] = state.typed_parent[source];
    tape.typed_child[destination] = state.typed_child[source];
    tape.typed_founder[destination] = state.typed_founder[source];
    tape.typed_depth[destination] = state.typed_depth[source];
    tape.typed_subtype[destination] = state.typed_subtype[source];
    tape.typed_reason[destination] = state.typed_reason[source];
    tape.typed_success[destination] = state.typed_success[source];
    tape.typed_value[destination] = state.typed_value[source];
}

__global__ void commit_event_ring(
    State state, r4::Replay replay, RingState ring, r4::R4Stage stage) {
    if (threadIdx.x != 0 || !::all_ok(stage.base, ring.batch)) return;
    const int64_t world = static_cast<int64_t>(blockIdx.x);
    if (world >= ring.batch) return;
    int64_t producer = ring.producer_cursor[world];
    const int64_t consumer = ring.consumer_cursor[world];
    int32_t active = ring.active_bank[world];
    int64_t counts[2] = {
        ring.bank_counts[world * 2], ring.bank_counts[world * 2 + 1]};
    int64_t starts[2] = {
        ring.bank_start_sequences[world * 2],
        ring.bank_start_sequences[world * 2 + 1]};
    const int64_t rows = state.event_active_count[world];
    const int64_t event_offset = world * state.event_capacity;
    for (int64_t row = 0; row < rows; ++row) {
        if (counts[active] >= ring.capacity) {
            const int32_t other = 1 - active;
            const int64_t other_end = starts[other] + counts[other];
            if (counts[other] != 0 && other_end > consumer) return;
            counts[other] = 0;
            starts[other] = producer;
            active = other;
        }
        if (counts[active] == 0) starts[active] = producer;
        const int64_t destination = world * ring.capacity + counts[active];
        copy_ring_row(state, ring.banks[active], event_offset + row, destination,
            producer);
        ++counts[active];
        ++producer;
    }
    ring.bank_counts[world * 2] = counts[0];
    ring.bank_counts[world * 2 + 1] = counts[1];
    ring.bank_start_sequences[world * 2] = starts[0];
    ring.bank_start_sequences[world * 2 + 1] = starts[1];
    ring.active_bank[world] = active;
    ring.producer_cursor[world] = producer;
    ring.banks[0].active_count[world] = counts[0];
    ring.banks[1].active_count[world] = counts[1];
    ring.banks[0].status[world] = 0;
    ring.banks[1].status[world] = 0;
    state.event_active_count[world] = 0;
    state.event_status[world] = 0;
    replay.event_sequence[world] = producer;
}

}  // namespace r11

#ifdef EIEL_C4_R11_TEST_ONLY_ACTIVE_PREFLIGHT_HOOK
namespace {

std::vector<torch::Tensor> test_only_active_preflight_diagnostic;
std::vector<int64_t> test_only_active_preflight_enqueue_metadata;
int64_t test_only_active_preflight_mode = 0;

}  // namespace
#endif

#ifdef EIEL_C4_PHASE_B_SURFACE_TIMING_HOOK
namespace {

std::vector<torch::Tensor> test_only_phase_timing;

}  // namespace
#endif

#ifdef EIEL_C4_R11_TEST_ONLY_ALIAS_GUARD_HOOK
namespace {

r11::AliasGuardDiagnostic test_only_alias_guard_diagnostic{};
std::vector<torch::Tensor> test_only_alias_guard_tensors;

}  // namespace
#endif

void tensor_world_c4_resident_population_r11_graph_cuda(
    std::vector<torch::Tensor> primary, std::vector<torch::Tensor> death_tensors,
    std::vector<torch::Tensor> inherit, std::vector<torch::Tensor> zero,
    std::vector<torch::Tensor> minus_one, std::vector<torch::Tensor> half,
    std::vector<torch::Tensor> one, std::vector<torch::Tensor> scalar_rng,
    std::vector<torch::Tensor> replay, std::vector<torch::Tensor> staging,
    std::vector<torch::Tensor> turnover, std::vector<torch::Tensor> identity,
    std::vector<torch::Tensor> rng_staging,
    std::vector<torch::Tensor> event_ring,
    torch::Tensor status,
    torch::Tensor persistent_outer_status,
    torch::Tensor generation, torch::Tensor tick, torch::Tensor requested_ticks,
    torch::Tensor completed_ticks, int64_t max_births,
    double parent_energy_share, double minimum_energy_buffer,
    int64_t cooldown_steps, double start_health, double max_energy,
    int64_t life_history_mode, int64_t maturity_start_ticks,
    int64_t maturity_rise_ticks, int64_t fertility_decline_start_ticks,
    int64_t fertility_decline_ticks, double elder_fertility_floor,
    int64_t environment_stream, int64_t runner_stream) {
    TORCH_CHECK(primary.size() == 101 && death_tensors.size() == 13,
        "C4 R11 authoritative ABI is incomplete");
    TORCH_CHECK(inherit.size() == 29 && zero.size() == 55 && minus_one.size() == 11
        && half.size() == 1 && one.size() == 1,
        "C4 R11 reset ABI is incomplete");
    TORCH_CHECK((scalar_rng.size() == 8 || scalar_rng.size() == 14)
        && replay.size() == 8
        && staging.size() == 49 && turnover.size() == 22 && identity.size() == 6,
        "C4 R11 continuation/staging ABI is incomplete");
    TORCH_CHECK(rng_staging.size() == 2,
        "C4 R11 RNG counter staging ABI is incomplete");
    TORCH_CHECK(event_ring.size() == 53,
        "C4 R11 reusable event-ring ABI is incomplete");
    const int64_t batch = primary[0].size(0), capacity = primary[0].size(1);
    TORCH_CHECK(batch == 48 && capacity > 0, "C4 R11 is bounded to positive B48");
    const int64_t grid_size = primary[9].size(1);
    const int64_t event_capacity = primary[77].size(1);
    const int64_t streams = replay[0].size(1);
    const int64_t tape_capacity = scalar_rng[4].size(1);
    const int64_t ring_capacity = event_ring[5].size(1);
    const auto device = primary[0].device();
    TORCH_CHECK(device.is_cuda(), "C4 R11 requires CUDA state");
    for (const auto& value : {
             generation, tick, requested_ticks, completed_ticks, status,
             persistent_outer_status}) {
        TORCH_CHECK(value.device() == device && value.is_cuda() && value.is_contiguous(),
            "C4 R11 device control tensors must share the state device");
        TORCH_CHECK(value.scalar_type() == torch::kInt64 || value.scalar_type() == torch::kInt32,
            "C4 R11 device control tensor has an invalid dtype");
        TORCH_CHECK(value.dim() == 1 && value.size(0) == batch,
            "C4 R11 device control tensor has an invalid shape");
    }
    TORCH_CHECK(generation.scalar_type() == torch::kInt64
        && tick.scalar_type() == torch::kInt64
        && requested_ticks.scalar_type() == torch::kInt64
        && completed_ticks.scalar_type() == torch::kInt64
        && status.scalar_type() == torch::kInt32,
        "C4 R11 generation/tick/status dtypes are fixed");
    TORCH_CHECK(persistent_outer_status.scalar_type() == torch::kInt32,
        "C4 R11 persistent outer status dtype is fixed");
    c10::cuda::CUDAGuard guard(device);
    const std::vector<torch::Tensor> legacy_replay(replay.begin(), replay.begin() + 6);
    const std::vector<torch::Tensor> legacy_scalar_rng(
        scalar_rng.begin(), scalar_rng.begin() + 8);
    r5::validate_abi(primary, death_tensors, inherit, zero, minus_one, half, one,
        legacy_scalar_rng, legacy_replay, staging, turnover, device, batch, capacity, grid_size,
        event_capacity, streams, tape_capacity);
    r5::exact(replay[6], device, torch::kInt64, {batch, streams},
        "R11 raw64 transition counters");
    r5::exact(replay[7], device, torch::kInt64, {batch, streams},
        "R11 output32 return counters");
    r5::exact(rng_staging[0], device, torch::kInt64, {batch, streams},
        "R11 staged raw64 transition counters");
    r5::exact(rng_staging[1], device, torch::kInt64, {batch, streams},
        "R11 staged output32 return counters");
    if (scalar_rng.size() == 14) {
        r5::exact(scalar_rng[8], device, torch::kInt64, {batch},
            "R11 Gaussian window base");
        r5::exact(scalar_rng[9], device, torch::kInt64, {batch},
            "R11 Gaussian window valid end");
        r5::exact(scalar_rng[10], device, torch::kInt64, {batch},
            "R11 Gaussian window refill generation");
        r5::exact(scalar_rng[11], device, torch::kUInt8, {batch, 32},
            "R11 Gaussian window provenance");
        r5::exact(scalar_rng[12], device, torch::kInt32, {batch},
            "R11 Gaussian window status");
        r5::exact(scalar_rng[13], device, torch::kInt64, {batch},
            "R11 Gaussian window exhaustion cursor");
    }
    r5::exact(status, device, torch::kInt32, {batch}, "R11 status");
    for (int index = 0; index < 4; ++index)
        r5::exact(identity[index], device, torch::kInt64, {batch, capacity},
            "R11 identity slot-token staging");
    r5::exact(identity[4], device, torch::kInt64, {batch}, "R11 identity next-ID staging");
    r5::exact(identity[5], device, torch::kInt32, {batch}, "R11 identity status");
    r5::exact(event_ring[0], device, torch::kInt64, {batch, 2}, "R11 ring counts");
    r5::exact(event_ring[1], device, torch::kInt64, {batch, 2}, "R11 ring starts");
    r5::exact(event_ring[2], device, torch::kInt64, {batch}, "R11 ring producer");
    r5::exact(event_ring[3], device, torch::kInt64, {batch}, "R11 ring consumer");
    r5::exact(event_ring[4], device, torch::kInt32, {batch}, "R11 ring active bank");
    TORCH_CHECK(ring_capacity > 0, "C4 R11 event-ring bank capacity must be positive");
    const torch::ScalarType ring_types[24] = {
        torch::kInt32, torch::kInt64, torch::kInt64, torch::kInt64, torch::kInt64,
        torch::kFloat32, torch::kInt32, torch::kInt64, torch::kInt64, torch::kInt64,
        torch::kInt64, torch::kInt64, torch::kInt64, torch::kInt64, torch::kInt64,
        torch::kInt64, torch::kInt64, torch::kInt32, torch::kInt32, torch::kInt32,
        torch::kInt8, torch::kFloat32, torch::kInt64, torch::kInt32};
    const std::vector<int64_t> ring_shape{batch, ring_capacity};
    for (int bank = 0; bank < 2; ++bank) {
        const int64_t base = 5 + bank * 24;
        for (int field = 0; field < 24; ++field) {
            const bool scalar_field = field >= 22;
            TORCH_CHECK(event_ring[base + field].is_cuda()
                && event_ring[base + field].device() == device
                && event_ring[base + field].is_contiguous()
                && event_ring[base + field].scalar_type() == ring_types[field]
                && event_ring[base + field].sizes().vec()
                    == (scalar_field ? std::vector<int64_t>{batch} : ring_shape),
                "C4 R11 ring bank field ABI mismatch at index ", base + field,
                " shape=", event_ring[base + field].sizes());
        }
    }
    const r11::NewbornFanoutFields newborn_fanout =
        r11::make_newborn_fanout_fields(
            inherit, zero, minus_one, half, one, batch, capacity);

    State live = make_state(primary, batch, capacity, grid_size, event_capacity);
    live.policy_use_gru = inherit[11].data_ptr<bool>();
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
        tape_capacity,
        scalar_rng.size() == 14 ? scalar_rng[8].data_ptr<int64_t>() : nullptr,
        scalar_rng.size() == 14 ? scalar_rng[9].data_ptr<int64_t>() : nullptr,
        scalar_rng.size() == 14 ? scalar_rng[10].data_ptr<int64_t>() : nullptr,
        scalar_rng.size() == 14 ? scalar_rng[11].data_ptr<uint8_t>() : nullptr,
        scalar_rng.size() == 14 ? scalar_rng[12].data_ptr<int32_t>() : nullptr,
        scalar_rng.size() == 14 ? scalar_rng[13].data_ptr<int64_t>() : nullptr
#if defined(EIEL_C4_R11_RNGV2)
        , tick.data_ptr<int64_t>()
#endif
        };
    r4::Replay replay_state{
        replay[0].data_ptr<int64_t>(), replay[1].data_ptr<int64_t>(),
        replay[2].data_ptr<int64_t>(), replay[3].data_ptr<int64_t>(),
        replay[4].data_ptr<int64_t>(), streams, environment_stream, runner_stream,
        replay[6].data_ptr<int64_t>(), replay[7].data_ptr<int64_t>()};
    r4::R4State live_r4{live, genome, python, replay_state, zero[15].data_ptr<float>()};
    r4::R4Stage stage = r4::make_stage(staging);
#ifdef EIEL_C4_PHASE_B_SURFACE_TIMING_HOOK
    r4::PhaseTiming phase_timing{nullptr};
    if (!test_only_phase_timing.empty()) {
        TORCH_CHECK(test_only_phase_timing.size() == 1,
            "C4 Phase-B surface timing diagnostic is incomplete");
        r5::exact(test_only_phase_timing[0], device, torch::kInt64,
            {batch, 16}, "R11 Phase-B surface timing markers");
        phase_timing = r4::PhaseTiming{
            test_only_phase_timing[0].data_ptr<int64_t>()};
    }
    stage.timing = phase_timing;
#endif
    stage.post_raw64_transitions = rng_staging[0].data_ptr<int64_t>();
    stage.post_output32_returns = rng_staging[1].data_ptr<int64_t>();
    r5::DeathState death{
        death_tensors[0].data_ptr<bool>(), death_tensors[1].data_ptr<int32_t>(),
        death_tensors[2].data_ptr<int64_t>(), death_tensors[3].data_ptr<int64_t>(),
        death_tensors[4].data_ptr<int64_t>(), death_tensors[5].data_ptr<int64_t>(),
        death_tensors[6].data_ptr<int64_t>(), death_tensors[7].data_ptr<int32_t>(),
        death_tensors[8].data_ptr<int64_t>(), death_tensors[9].data_ptr<int32_t>(),
        death_tensors[10].data_ptr<int64_t>(), death_tensors[11].data_ptr<int32_t>(),
        death_tensors[12].data_ptr<int32_t>()};
    r5::TurnStage turn = r5::make_turn(turnover);
#ifdef EIEL_C4_PHASE_C_SINGLE_SHADOW
    // Phase C reuses the disposable outer R13 shadow as the post-death
    // projection. TurnStage still owns the ordered death/event ledger, but
    // all capacity/grid state consumed by R4 and identity preflight points at
    // the same outer shadow instead of a second copied shadow.
    turn.alive = live.alive;
    turn.agent_id = live.agent_id;
    turn.pos_y = live.pos_y;
    turn.pos_x = live.pos_x;
    turn.energy = live.energy;
    turn.health = live.health;
    turn.occupancy_agent = live.occupancy_agent;
    turn.occupancy_slot = live.occupancy_slot;
    turn.patch_agent = live.patch_agent;
    turn.id_to_slot = live.agent_id_to_slot;
    turn.repro_cooldown = live.repro_cooldown;
    turn.policy_energy = live.policy_energy_observation;
    turn.virtual_queue = live.free_queue;
    turn.virtual_count = live.free_count;
#endif
#ifdef EIEL_C4_PHASE_B_SURFACE_TIMING_HOOK
    turn.timing = phase_timing;
#endif
    r11::IdentityStage identity_stage = r11::make_identity(identity);
    Config config{max_births, parent_energy_share, minimum_energy_buffer,
        static_cast<int32_t>(cooldown_steps), static_cast<float>(start_health),
        static_cast<float>(max_energy), static_cast<int32_t>(life_history_mode),
        static_cast<int32_t>(maturity_start_ticks),
        static_cast<int32_t>(maturity_rise_ticks),
        static_cast<int32_t>(fertility_decline_start_ticks),
        static_cast<int32_t>(fertility_decline_ticks), elder_fertility_floor,
        0, 0, generation.data_ptr<int64_t>(), tick.data_ptr<int64_t>(),
        kIdentitySortedMonotonic};
    r11::RingState ring = r11::make_ring(event_ring, batch, ring_capacity);
    const auto stream = at::cuda::getCurrentCUDAStream(device.index());

#ifdef EIEL_C4_R11_TEST_ONLY_ALIAS_GUARD_HOOK
    r11::AliasGuardDiagnostic alias_guard_diagnostic{};
    if (!test_only_alias_guard_tensors.empty()) {
        TORCH_CHECK(test_only_alias_guard_tensors.size() == 7,
            "C4 R11 test-only alias guard diagnostic is incomplete");
        r5::exact(test_only_alias_guard_tensors[0], device, torch::kInt64,
            {3}, "R11 test-only alias guard control");
        r5::exact(test_only_alias_guard_tensors[1], device, torch::kInt32,
            {5}, "R11 test-only alias guard local status");
        r5::exact(test_only_alias_guard_tensors[2], device, torch::kInt64,
            {5}, "R11 test-only alias guard planned births");
        r5::exact(test_only_alias_guard_tensors[3], device, torch::kInt32,
            {5}, "R11 test-only alias guard writer reached");
        r5::exact(test_only_alias_guard_tensors[4], device, torch::kInt32,
            {5}, "R11 test-only alias guard writer skipped");
        r5::exact(test_only_alias_guard_tensors[5], device, torch::kInt32,
            {5}, "R11 test-only alias guard outer status");
        r5::exact(test_only_alias_guard_tensors[6], device, torch::kInt32,
            {5}, "R11 test-only alias guard injected");
        alias_guard_diagnostic = r11::AliasGuardDiagnostic{
            test_only_alias_guard_tensors[0].data_ptr<int64_t>(),
            test_only_alias_guard_tensors[1].data_ptr<int32_t>(),
            test_only_alias_guard_tensors[2].data_ptr<int64_t>(),
            test_only_alias_guard_tensors[3].data_ptr<int32_t>(),
            test_only_alias_guard_tensors[4].data_ptr<int32_t>(),
            test_only_alias_guard_tensors[5].data_ptr<int32_t>(),
            test_only_alias_guard_tensors[6].data_ptr<int32_t>()};
    }
#endif

#ifdef EIEL_C4_R11_TEST_ONLY_ACTIVE_PREFLIGHT_HOOK
    if (!test_only_active_preflight_diagnostic.empty()) {
        cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
        C10_CUDA_CHECK(cudaStreamIsCapturing(stream.stream(), &capture_status));
        test_only_active_preflight_enqueue_metadata = {
            static_cast<int64_t>(reinterpret_cast<uintptr_t>(live.alive)),
            static_cast<int64_t>(reinterpret_cast<uintptr_t>(live.active_count)),
            static_cast<int64_t>(reinterpret_cast<uintptr_t>(live.active_agent_ids)),
            static_cast<int64_t>(reinterpret_cast<uintptr_t>(live.active_agent_slots)),
            static_cast<int64_t>(reinterpret_cast<uintptr_t>(live.agent_id)),
            static_cast<int64_t>(reinterpret_cast<uintptr_t>(stream.stream())),
            static_cast<int64_t>(capture_status),
            test_only_active_preflight_mode};
    }
#endif

    r11::reset_local_event_cursor<<<batch, 1, 0, stream>>>(
        live, replay_state, ring, status.data_ptr<int32_t>());
    r11::preflight_rng_window<<<batch, 1, 0, stream>>>(
        live, replay_state, python, requested_ticks.data_ptr<int64_t>(),
        completed_ticks.data_ptr<int64_t>(), status.data_ptr<int32_t>(), batch,
        max_births
#ifdef EIEL_C4_PHASE_B_SURFACE_TIMING_HOOK
        , phase_timing
#endif
    );
#if !defined(EIEL_C4_R11_RNGV2)
    r11::validate_gaussian_tape_parallel<<<batch, 256, 0, stream>>>(
        python, stage.base.status, batch
#ifdef EIEL_C4_PHASE_B_SURFACE_TIMING_HOOK
        , phase_timing
#endif
    );
#endif
#ifdef EIEL_C4_R11_TEST_ONLY_ACTIVE_PREFLIGHT_HOOK
    if (!test_only_active_preflight_diagnostic.empty()) {
        TORCH_CHECK(test_only_active_preflight_diagnostic.size() == 6,
            "C4 R11 test-only active preflight diagnostic is incomplete");
        r5::exact(test_only_active_preflight_diagnostic[0], device, torch::kInt64,
            {batch}, "R11 test-only observed active count");
        r5::exact(test_only_active_preflight_diagnostic[1], device, torch::kInt64,
            {batch}, "R11 test-only observed alive sum");
        r5::exact(test_only_active_preflight_diagnostic[2], device, torch::kInt32,
            {batch}, "R11 test-only active tail validity");
        r5::exact(test_only_active_preflight_diagnostic[3], device, torch::kInt32,
            {batch}, "R11 test-only active order validity");
        r5::exact(test_only_active_preflight_diagnostic[4], device, torch::kInt64,
            {5}, "R11 test-only preflight device pointers");
        r5::exact(test_only_active_preflight_diagnostic[5], device, torch::kInt64,
            {batch}, "R11 test-only volatile checksum");
        r11::ActivePreflightDiagnosticStage diagnostic{
            test_only_active_preflight_diagnostic[0].data_ptr<int64_t>(),
            test_only_active_preflight_diagnostic[1].data_ptr<int64_t>(),
            test_only_active_preflight_diagnostic[2].data_ptr<int32_t>(),
            test_only_active_preflight_diagnostic[3].data_ptr<int32_t>(),
            test_only_active_preflight_diagnostic[4].data_ptr<int64_t>(),
            test_only_active_preflight_diagnostic[5].data_ptr<int64_t>()};
        if (test_only_active_preflight_mode == 0) {
            r11::capture_active_preflight_diagnostic<<<batch, 1, 0, stream>>>(
                live, diagnostic);
        } else if (test_only_active_preflight_mode == 1) {
            r11::active_preflight_noop<<<1, 1, 0, stream>>>();
        } else if (test_only_active_preflight_mode == 2) {
            r11::active_preflight_system_fence<<<1, 1, 0, stream>>>();
        } else {
            r11::capture_active_preflight_checksum<<<batch, 1, 0, stream>>>(
                live, diagnostic);
        }
    }
#endif
    r6::all_live_semantic_preflight<<<batch, 1, 0, stream>>>(
        live, status.data_ptr<int32_t>(), config);
#ifdef EIEL_C4_PHASE_C_SINGLE_SHADOW
    r5::prepare_post_death_outer<<<batch, 1, 0, stream>>>(
        live, death, replay_state, turn, config);
#else
    r5::prepare_post_death_shadow<<<batch, 1, 0, stream>>>(
        live, death, replay_state, turn, config);
#endif
    State shadow = live;
    shadow.alive = turn.alive;
    shadow.agent_id = turn.agent_id;
    shadow.pos_y = turn.pos_y;
    shadow.pos_x = turn.pos_x;
    shadow.energy = turn.energy;
    shadow.health = turn.health;
    shadow.occupancy_agent = turn.occupancy_agent;
    shadow.occupancy_slot = turn.occupancy_slot;
    shadow.patch_agent = turn.patch_agent;
    shadow.agent_id_to_slot = turn.id_to_slot;
    shadow.free_queue = turn.virtual_queue;
    shadow.free_count = turn.virtual_count;
    shadow.repro_cooldown = turn.repro_cooldown;
    shadow.policy_energy_observation = turn.policy_energy;
    shadow.event_active_count = turn.event_count;
    shadow.event_status = turn.event_status;
    r4::Replay shadow_replay = replay_state;
    shadow_replay.event_sequence = turn.replay_event_sequence;
    r4::R4State shadow_r4{
        shadow, genome, python, shadow_replay, zero[15].data_ptr<float>()};
    r11::sync_stage_event_cursor<<<1, 1, 0, stream>>>(stage, turn, batch);
    r11::prepare_rng_counter_shadow<<<batch, 1, 0, stream>>>(
        replay_state, stage, batch);
    r4::preflight_kernel<<<batch, 1, 0, stream>>>(shadow_r4, stage, config);
    r11::preflight_event_ring<<<batch, 1, 0, stream>>>(
        stage, turn, ring, status.data_ptr<int32_t>(), capacity);
    r11::capture_identity_tokens<<<batch, 1, 0, stream>>>(
        live, stage, turn, identity_stage);
    r11::validate_identity_tokens<<<batch, 1, 0, stream>>>(
        live, stage, turn, identity_stage);
    r11::merge_identity_status<<<1, 1, 0, stream>>>(
        stage, turn, identity_stage, status.data_ptr<int32_t>(), batch);
#ifdef EIEL_C4_R11_TEST_ONLY_ALIAS_GUARD_HOOK
    if (alias_guard_diagnostic.control != nullptr) {
        r11::alias_guard_test_begin<<<1, 1, 0, stream>>>(
            alias_guard_diagnostic, status.data_ptr<int32_t>(), batch);
    }
#endif
    r6::merge_preflight_status<<<1, 1, 0, stream>>>(
        stage, turn, live, status.data_ptr<int32_t>());
    r5::merge_all_status<<<1, 1, 0, stream>>>(stage, turn, live, config);
    r11::publish_first_status<<<1, 1, 0, stream>>>(
        stage, status.data_ptr<int32_t>(), batch);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

#ifdef EIEL_C4_PHASE_B_SURFACE_TIMING_HOOK
    if (phase_timing.values != nullptr)
        r11::phase_timing_marker<<<batch, 1, 0, stream>>>(
            phase_timing.values, batch, 10);
#endif

    r11::fused_newborn_fanout_kernel<<<
        batch, r11::kNewbornFanoutThreads, 0, stream>>>(
            newborn_fanout, stage.base, batch, capacity,
            persistent_outer_status.data_ptr<int32_t>(),
            status.data_ptr<int32_t>()
#ifdef EIEL_C4_R11_TEST_ONLY_ALIAS_GUARD_HOOK
            , alias_guard_diagnostic
#endif
        );
#ifdef EIEL_C4_PHASE_C_SINGLE_SHADOW
    r5::commit_deaths_outer<<<batch, 1, 0, stream>>>(
        live, death, stage, turn, config);
#else
    r5::commit_deaths<<<batch, 1, 0, stream>>>(live, death, stage, turn, config);
#endif
    // C1 parallelizes only the independent accepted-row installation.  The
    // kernel's thread-0 tail retains ordered parent effects, queue compaction,
    // events, and RNG publication; 256 threads cover capacity rows in fixed
    // strides without changing the graph's semantic barrier.
    r4::commit_kernel<<<batch, 256, 0, stream>>>(live_r4, stage, config);
    r11::rebuild_active_parallel<<<
        batch, static_cast<unsigned int>(capacity), 0, stream>>>(
            live, status.data_ptr<int32_t>(), batch
#ifdef EIEL_C4_PHASE_B_SURFACE_TIMING_HOOK
            , phase_timing
#endif
        );
    r11::commit_rng_counter_shadow<<<batch, 1, 0, stream>>>(
        replay_state, stage, batch);
    r11::commit_event_ring<<<batch, 1, 0, stream>>>(
        live, replay_state, ring, stage);
#ifdef EIEL_C4_PHASE_B_SURFACE_TIMING_HOOK
    if (phase_timing.values != nullptr)
        r11::phase_timing_marker<<<batch, 1, 0, stream>>>(
            phase_timing.values, batch, 11);
#endif
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void tensor_world_c4_resident_r11_merge_window_control_cuda(
    torch::Tensor preflight_status, torch::Tensor global_status,
    torch::Tensor c4_status, torch::Tensor commit_status,
    torch::Tensor stop_requested, torch::Tensor first_error_stage,
    torch::Tensor first_error_world, torch::Tensor first_error_code,
    torch::Tensor first_error_tick, torch::Tensor tick,
    torch::Tensor completed_ticks, torch::Tensor current_tick,
    torch::Tensor receipt_write_index) {
    TORCH_CHECK(preflight_status.is_cuda() && preflight_status.is_contiguous()
        && preflight_status.scalar_type() == torch::kInt32
        && preflight_status.dim() == 1,
        "R11 window control preflight ABI mismatch");
    const auto device = preflight_status.device();
    const int64_t batch = preflight_status.size(0);
    const auto exact = [&](const torch::Tensor& value, torch::ScalarType type,
                           const char* name) {
        r5::exact(value, device, type, {batch}, name);
    };
    exact(global_status, torch::kInt32, "R11 window control global status");
    exact(c4_status, torch::kInt32, "R11 window control C4 status");
    exact(commit_status, torch::kInt32, "R11 window control commit status");
    exact(stop_requested, torch::kInt32, "R11 window control stop status");
    exact(first_error_stage, torch::kInt32, "R11 window control error stage");
    exact(first_error_world, torch::kInt32, "R11 window control error world");
    exact(first_error_code, torch::kInt32, "R11 window control error code");
    exact(first_error_tick, torch::kInt64, "R11 window control error tick");
    exact(tick, torch::kInt64, "R11 window control tick");
    exact(completed_ticks, torch::kInt64, "R11 window control completed ticks");
    exact(current_tick, torch::kInt64, "R11 window control current tick");
    exact(receipt_write_index, torch::kInt64, "R11 window control receipt index");
    c10::cuda::CUDAGuard guard(device);
    r11::WindowControlState control{
        preflight_status.data_ptr<int32_t>(), global_status.data_ptr<int32_t>(),
        c4_status.data_ptr<int32_t>(), commit_status.data_ptr<int32_t>(),
        stop_requested.data_ptr<int32_t>(), first_error_stage.data_ptr<int32_t>(),
        first_error_world.data_ptr<int32_t>(), first_error_code.data_ptr<int32_t>(),
        first_error_tick.data_ptr<int64_t>(), tick.data_ptr<int64_t>(),
        completed_ticks.data_ptr<int64_t>(), current_tick.data_ptr<int64_t>(),
        receipt_write_index.data_ptr<int64_t>(), batch};
    const auto stream = at::cuda::getCurrentCUDAStream(device.index());
    r11::merge_window_control_kernel<<<1, 256, 0, stream>>>(control);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

#if defined(EIEL_C4_R11_RNGV2)
struct V2WindowReplayPrep {
    int32_t* global_status;
    int64_t* completed_ticks;
    int64_t* receipt_write_index;
    int64_t* generation;
    const int64_t* generation_base;
    int64_t* live_consumer;
    const int64_t* live_producer;
    int64_t* shadow_consumer;
    const int64_t* shadow_producer;
    int64_t batch;
};

__global__ void prepare_v2_window_replay_kernel(V2WindowReplayPrep prep) {
    const int64_t world = static_cast<int64_t>(blockIdx.x) * blockDim.x
        + threadIdx.x;
    if (world >= prep.batch) return;
    if (prep.global_status[world] != 0) return;
    prep.completed_ticks[world] = 0;
    prep.receipt_write_index[world] = 0;
    prep.generation[world] = prep.generation_base[world];
    if (prep.live_consumer != nullptr && prep.live_producer != nullptr) {
        prep.live_consumer[world] = prep.live_producer[world];
    }
    if (prep.shadow_consumer != nullptr && prep.shadow_producer != nullptr) {
        prep.shadow_consumer[world] = prep.shadow_producer[world];
    }
}
#endif

void tensor_world_c4_resident_r11_prepare_v2_window_replay_cuda(
    torch::Tensor global_status, torch::Tensor completed_ticks,
    torch::Tensor receipt_write_index, torch::Tensor generation,
    torch::Tensor generation_base, torch::Tensor live_consumer,
    torch::Tensor live_producer, torch::Tensor shadow_consumer,
    torch::Tensor shadow_producer) {
#if !defined(EIEL_C4_R11_RNGV2)
    TORCH_CHECK(false, "R11 V2 window replay prep requires EIEL_C4_R11_RNGV2");
#else
    TORCH_CHECK(global_status.is_cuda() && global_status.is_contiguous()
        && global_status.scalar_type() == torch::kInt32
        && global_status.dim() == 1,
        "R11 V2 replay prep global status ABI mismatch");
    const auto device = global_status.device();
    const int64_t batch = global_status.size(0);
    r5::exact(completed_ticks, device, torch::kInt64, {batch},
        "R11 V2 replay prep completed ticks");
    r5::exact(receipt_write_index, device, torch::kInt64, {batch},
        "R11 V2 replay prep receipt index");
    r5::exact(generation, device, torch::kInt64, {batch},
        "R11 V2 replay prep generation");
    r5::exact(generation_base, device, torch::kInt64, {batch},
        "R11 V2 replay prep generation base");
    r5::exact(live_consumer, device, torch::kInt64, {batch},
        "R11 V2 replay prep live consumer");
    r5::exact(live_producer, device, torch::kInt64, {batch},
        "R11 V2 replay prep live producer");
    r5::exact(shadow_consumer, device, torch::kInt64, {batch},
        "R11 V2 replay prep shadow consumer");
    r5::exact(shadow_producer, device, torch::kInt64, {batch},
        "R11 V2 replay prep shadow producer");
    c10::cuda::CUDAGuard guard(device);
    V2WindowReplayPrep prep{
        global_status.data_ptr<int32_t>(),
        completed_ticks.data_ptr<int64_t>(),
        receipt_write_index.data_ptr<int64_t>(),
        generation.data_ptr<int64_t>(),
        generation_base.data_ptr<int64_t>(),
        live_consumer.data_ptr<int64_t>(),
        live_producer.data_ptr<int64_t>(),
        shadow_consumer.data_ptr<int64_t>(),
        shadow_producer.data_ptr<int64_t>(),
        batch};
    const auto stream = at::cuda::getCurrentCUDAStream(device.index());
    const unsigned int blocks = static_cast<unsigned int>((batch + 255) / 256);
    prepare_v2_window_replay_kernel<<<blocks, 256, 0, stream>>>(prep);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
#endif
}

void tensor_world_c4_resident_r11_advance_window_clock_cuda(
    torch::Tensor global_status, torch::Tensor tick, torch::Tensor generation) {
    TORCH_CHECK(global_status.is_cuda() && global_status.is_contiguous()
        && global_status.scalar_type() == torch::kInt32
        && global_status.dim() == 1,
        "R11 window clock global status ABI mismatch");
    const auto device = global_status.device();
    const int64_t batch = global_status.size(0);
    r5::exact(tick, device, torch::kInt64, {batch}, "R11 window clock tick");
    r5::exact(generation, device, torch::kInt64, {batch},
        "R11 window clock generation");
    c10::cuda::CUDAGuard guard(device);
    r11::WindowClockState clock{
        global_status.data_ptr<int32_t>(), tick.data_ptr<int64_t>(),
        generation.data_ptr<int64_t>(), batch};
    const auto stream = at::cuda::getCurrentCUDAStream(device.index());
    const unsigned int blocks = static_cast<unsigned int>((batch + 255) / 256);
    r11::advance_window_clock_kernel<<<blocks, 256, 0, stream>>>(clock);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

std::vector<int64_t> tensor_world_c4_resident_r11_newborn_fanout_contract() {
    return {
        r11::kNewbornFanoutFieldCount,
        r11::kNewbornFanoutLegacyLaunchCount,
        r11::kNewbornFanoutFusedLaunchCount,
        r11::kNewbornFanoutLegacyLaunchCount
            - r11::kNewbornFanoutFusedLaunchCount,
        r11::kNewbornFanoutThreads,
    };
}

int64_t tensor_world_c4_resident_r11_graph_node_count(
    const int64_t raw_graph
) {
    TORCH_CHECK(raw_graph != 0, "C4 R11 raw CUDA graph handle is null");
    size_t count = 0;
    cudaGraph_t graph = reinterpret_cast<cudaGraph_t>(
        static_cast<uintptr_t>(raw_graph));
    C10_CUDA_CHECK(cudaGraphGetNodes(graph, nullptr, &count));
    return static_cast<int64_t>(count);
}

#ifdef EIEL_C4_PHASE_B_SURFACE_TIMING_HOOK
void tensor_world_c4_resident_r11_test_only_configure_phase_timing_cuda(
    torch::Tensor markers) {
    TORCH_CHECK(markers.is_cuda() && markers.is_contiguous()
        && markers.scalar_type() == torch::kInt64 && markers.dim() == 2
        && markers.size(1) == 16,
        "C4 Phase-B surface timing marker ABI mismatch");
    test_only_phase_timing = {markers};
}

void tensor_world_c4_resident_r11_test_only_clear_phase_timing_cuda() {
    test_only_phase_timing.clear();
}
#endif

#ifdef EIEL_C4_R11_TEST_ONLY_ACTIVE_PREFLIGHT_HOOK
void tensor_world_c4_resident_r11_test_only_configure_active_preflight_cuda(
    torch::Tensor active_count, torch::Tensor alive_sum,
    torch::Tensor tail_valid, torch::Tensor order_valid,
    torch::Tensor device_pointers, torch::Tensor checksum, int64_t mode) {
    TORCH_CHECK(active_count.is_cuda() && active_count.is_contiguous()
        && active_count.scalar_type() == torch::kInt64 && active_count.dim() == 1,
        "C4 R11 test-only active count diagnostic ABI mismatch");
    const auto device = active_count.device();
    const int64_t batch = active_count.size(0);
    r5::exact(alive_sum, device, torch::kInt64, {batch},
        "R11 test-only observed alive sum");
    r5::exact(tail_valid, device, torch::kInt32, {batch},
        "R11 test-only active tail validity");
    r5::exact(order_valid, device, torch::kInt32, {batch},
        "R11 test-only active order validity");
    r5::exact(device_pointers, device, torch::kInt64, {5},
        "R11 test-only preflight device pointers");
    r5::exact(checksum, device, torch::kInt64, {batch},
        "R11 test-only volatile checksum");
    TORCH_CHECK(mode >= 0 && mode <= 3,
        "C4 R11 test-only active preflight mode must be in [0, 3]");
    test_only_active_preflight_diagnostic = {
        active_count, alive_sum, tail_valid, order_valid, device_pointers, checksum};
    test_only_active_preflight_mode = mode;
    test_only_active_preflight_enqueue_metadata.clear();
}

std::vector<int64_t>
tensor_world_c4_resident_r11_test_only_active_preflight_enqueue_metadata() {
    TORCH_CHECK(test_only_active_preflight_enqueue_metadata.size() == 8,
        "C4 R11 test-only active preflight enqueue metadata is unavailable");
    return test_only_active_preflight_enqueue_metadata;
}

void tensor_world_c4_resident_r11_test_only_clear_active_preflight_cuda() {
    test_only_active_preflight_diagnostic.clear();
    test_only_active_preflight_enqueue_metadata.clear();
    test_only_active_preflight_mode = 0;
}
#endif

#ifdef EIEL_C4_R11_TEST_ONLY_ALIAS_GUARD_HOOK
void tensor_world_c4_resident_r11_test_only_configure_alias_guard_cuda(
    torch::Tensor control, torch::Tensor local_status,
    torch::Tensor planned_births, torch::Tensor writer_reached,
    torch::Tensor writer_skipped, torch::Tensor outer_status,
    torch::Tensor injected) {
    TORCH_CHECK(control.is_cuda() && control.is_contiguous()
        && control.scalar_type() == torch::kInt64 && control.sizes().vec()
            == std::vector<int64_t>{3},
        "C4 R11 test-only alias guard control ABI mismatch");
    const auto device = control.device();
    r5::exact(local_status, device, torch::kInt32, {5},
        "R11 test-only alias guard local status");
    r5::exact(planned_births, device, torch::kInt64, {5},
        "R11 test-only alias guard planned births");
    r5::exact(writer_reached, device, torch::kInt32, {5},
        "R11 test-only alias guard writer reached");
    r5::exact(writer_skipped, device, torch::kInt32, {5},
        "R11 test-only alias guard writer skipped");
    r5::exact(outer_status, device, torch::kInt32, {5},
        "R11 test-only alias guard outer status");
    r5::exact(injected, device, torch::kInt32, {5},
        "R11 test-only alias guard injected");
    test_only_alias_guard_tensors = {
        control, local_status, planned_births, writer_reached,
        writer_skipped, outer_status, injected};
    test_only_alias_guard_diagnostic = r11::AliasGuardDiagnostic{
        control.data_ptr<int64_t>(), local_status.data_ptr<int32_t>(),
        planned_births.data_ptr<int64_t>(), writer_reached.data_ptr<int32_t>(),
        writer_skipped.data_ptr<int32_t>(), outer_status.data_ptr<int32_t>(),
        injected.data_ptr<int32_t>()};
}

void tensor_world_c4_resident_r11_test_only_clear_alias_guard_cuda() {
    test_only_alias_guard_tensors.clear();
    test_only_alias_guard_diagnostic = r11::AliasGuardDiagnostic{};
}
#endif

#ifdef EIEL_C4_R11_TEST_ONLY_IDENTITY_HOOK
namespace {

void validate_r11_identity_probe_abi(
    const std::vector<torch::Tensor>& primary,
    const std::vector<torch::Tensor>& death_tensors,
    const std::vector<torch::Tensor>& inherit,
    const std::vector<torch::Tensor>& zero,
    const std::vector<torch::Tensor>& minus_one,
    const std::vector<torch::Tensor>& half,
    const std::vector<torch::Tensor>& one,
    const std::vector<torch::Tensor>& scalar_rng,
    const std::vector<torch::Tensor>& replay,
    const std::vector<torch::Tensor>& staging,
    const std::vector<torch::Tensor>& turnover,
    const std::vector<torch::Tensor>& identity,
    const torch::Tensor& status) {
    TORCH_CHECK(primary.size() == 101 && death_tensors.size() == 13,
        "C4 R11 identity probe authoritative ABI is incomplete");
    TORCH_CHECK(inherit.size() == 29 && zero.size() == 55
        && minus_one.size() == 11 && half.size() == 1 && one.size() == 1,
        "C4 R11 identity probe reset ABI is incomplete");
    TORCH_CHECK(scalar_rng.size() == 8 && replay.size() == 8
        && staging.size() == 49 && turnover.size() == 22 && identity.size() == 6,
        "C4 R11 identity probe continuation/staging ABI is incomplete");
    const int64_t batch = primary[0].size(0);
    const int64_t capacity = primary[0].size(1);
    TORCH_CHECK(batch == 48 && capacity > 0,
        "C4 R11 identity probe is bounded to positive B48");
    const int64_t grid_size = primary[9].size(1);
    const int64_t event_capacity = primary[77].size(1);
    const int64_t streams = replay[0].size(1);
    const int64_t tape_capacity = scalar_rng[4].size(1);
    const auto device = primary[0].device();
    TORCH_CHECK(device.is_cuda(), "C4 R11 identity probe requires CUDA state");
    const std::vector<torch::Tensor> legacy_replay(replay.begin(), replay.begin() + 6);
    r5::validate_abi(primary, death_tensors, inherit, zero, minus_one, half, one,
        scalar_rng, legacy_replay, staging, turnover, device, batch, capacity,
        grid_size, event_capacity, streams, tape_capacity);
    r5::exact(replay[6], device, torch::kInt64, {batch, streams},
        "R11 identity probe raw64 transition counters");
    r5::exact(replay[7], device, torch::kInt64, {batch, streams},
        "R11 identity probe output32 return counters");
    r5::exact(status, device, torch::kInt32, {batch},
        "R11 identity probe global status");
    for (int index = 0; index < 4; ++index)
        r5::exact(identity[index], device, torch::kInt64, {batch, capacity},
            "R11 identity probe slot-token staging");
    r5::exact(identity[4], device, torch::kInt64, {batch},
        "R11 identity probe next-ID staging");
    r5::exact(identity[5], device, torch::kInt32, {batch},
        "R11 identity probe status");
}

}  // namespace

void tensor_world_c4_resident_r11_test_only_identity_capture_cuda(
    std::vector<torch::Tensor> primary, std::vector<torch::Tensor> death_tensors,
    std::vector<torch::Tensor> inherit, std::vector<torch::Tensor> zero,
    std::vector<torch::Tensor> minus_one, std::vector<torch::Tensor> half,
    std::vector<torch::Tensor> one, std::vector<torch::Tensor> scalar_rng,
    std::vector<torch::Tensor> replay, std::vector<torch::Tensor> staging,
    std::vector<torch::Tensor> turnover, std::vector<torch::Tensor> identity,
    torch::Tensor status) {
    validate_r11_identity_probe_abi(primary, death_tensors, inherit, zero,
        minus_one, half, one, scalar_rng, replay, staging, turnover, identity,
        status);
    const int64_t batch = primary[0].size(0);
    const int64_t capacity = primary[0].size(1);
    const int64_t grid_size = primary[9].size(1);
    const int64_t event_capacity = primary[77].size(1);
    const auto device = primary[0].device();
    c10::cuda::CUDAGuard guard(device);
    State live = make_state(primary, batch, capacity, grid_size, event_capacity);
    r4::R4Stage stage = r4::make_stage(staging);
    r5::TurnStage turn = r5::make_turn(turnover);
    r11::IdentityStage identity_stage = r11::make_identity(identity);
    const auto stream = at::cuda::getCurrentCUDAStream(device.index());
    r11::capture_identity_tokens<<<batch, 1, 0, stream>>>(
        live, stage, turn, identity_stage);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void tensor_world_c4_resident_r11_test_only_identity_validate_cuda(
    std::vector<torch::Tensor> primary, std::vector<torch::Tensor> death_tensors,
    std::vector<torch::Tensor> inherit, std::vector<torch::Tensor> zero,
    std::vector<torch::Tensor> minus_one, std::vector<torch::Tensor> half,
    std::vector<torch::Tensor> one, std::vector<torch::Tensor> scalar_rng,
    std::vector<torch::Tensor> replay, std::vector<torch::Tensor> staging,
    std::vector<torch::Tensor> turnover, std::vector<torch::Tensor> identity,
    torch::Tensor status) {
    validate_r11_identity_probe_abi(primary, death_tensors, inherit, zero,
        minus_one, half, one, scalar_rng, replay, staging, turnover, identity,
        status);
    const int64_t batch = primary[0].size(0);
    const int64_t capacity = primary[0].size(1);
    const int64_t grid_size = primary[9].size(1);
    const int64_t event_capacity = primary[77].size(1);
    const auto device = primary[0].device();
    c10::cuda::CUDAGuard guard(device);
    State live = make_state(primary, batch, capacity, grid_size, event_capacity);
    r4::R4Stage stage = r4::make_stage(staging);
    r5::TurnStage turn = r5::make_turn(turnover);
    r11::IdentityStage identity_stage = r11::make_identity(identity);
    const auto stream = at::cuda::getCurrentCUDAStream(device.index());
    r11::validate_identity_tokens<<<batch, 1, 0, stream>>>(
        live, stage, turn, identity_stage);
    r11::merge_identity_status<<<1, 1, 0, stream>>>(
        stage, turn, identity_stage, status.data_ptr<int32_t>(), batch);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
#endif
