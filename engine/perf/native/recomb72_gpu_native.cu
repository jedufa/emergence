#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <vector>

namespace {

constexpr int kPopulation = 64;
constexpr int kSurvivors = 32;
constexpr int kLoci = 6;
constexpr int kFounders = 64;

__device__ __forceinline__ uint64_t mix64(uint64_t value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31U);
}

__device__ __forceinline__ uint64_t word(
    const int64_t seed,
    const int64_t generation,
    const int64_t ordinal,
    const int64_t channel
) {
    uint64_t value = static_cast<uint64_t>(seed);
    value ^= mix64(static_cast<uint64_t>(generation) + 0x100000001b3ULL);
    value ^= mix64(static_cast<uint64_t>(ordinal) + 0x517cc1b727220a95ULL);
    value ^= mix64(static_cast<uint64_t>(channel) + 0x6eed0e9da4d94a4fULL);
    return mix64(value);
}

__device__ __forceinline__ float uniform01(const uint64_t value) {
    return static_cast<float>((value >> 40U) & 0xffffffULL)
        * (1.0f / 16777216.0f);
}

__device__ __forceinline__ float lower_bound(const int locus) {
    const float values[kLoci] = {-1.0f, -2.0f, 0.25f, -1.0f, 0.0f, 0.0f};
    return values[locus];
}

__device__ __forceinline__ float upper_bound(const int locus) {
    const float values[kLoci] = {1.0f, 2.0f, 2.5f, 1.0f, 0.999f, 1.0f};
    return values[locus];
}

__device__ __forceinline__ float neutral_value(const int locus) {
    const float values[kLoci] = {0.0f, 0.0f, 1.0f, 0.0f, 0.85f, 0.08f};
    return values[locus];
}

__device__ __forceinline__ float founder_value(
    const int pattern,
    const int group,
    const int locus
) {
    if (pattern == 0) {
        return locus == 0 ? (group == 0 ? -1.0f : 1.0f) : neutral_value(locus);
    }
    if (pattern == 1) {
        return locus == 3 ? (group == 0 ? -1.0f : 1.0f) : neutral_value(locus);
    }
    return group == 0 ? lower_bound(locus) : upper_bound(locus);
}

__device__ void summarize_cell(
    const int cell,
    const int capacity,
    const int output_row,
    const float* loci,
    const bool* alive,
    const float* ancestry,
    const int32_t pattern,
    float* variance,
    float* extreme,
    float* intact,
    float* effective,
    float* max_founder
) {
    const int cell_slot = cell * capacity;
    float mean[kLoci] = {};
    int count = 0;
    for (int slot = 0; slot < capacity; ++slot) {
        if (!alive[cell_slot + slot]) continue;
        ++count;
        for (int locus = 0; locus < kLoci; ++locus) {
            mean[locus] += loci[(cell_slot + slot) * kLoci + locus];
        }
    }
    for (int locus = 0; locus < kLoci; ++locus) {
        mean[locus] /= static_cast<float>(count);
    }
    float founder_mass[kFounders] = {};
    int intact_count = 0;
    for (int slot = 0; slot < capacity; ++slot) {
        if (!alive[cell_slot + slot]) continue;
        bool package_a = true;
        bool package_b = true;
        for (int locus = 0; locus < kLoci; ++locus) {
            const float value = loci[(cell_slot + slot) * kLoci + locus];
            const float delta = value - mean[locus];
            variance[(output_row * kLoci) + locus] += delta * delta;
            const float span = upper_bound(locus) - lower_bound(locus);
            if (
                value <= lower_bound(locus) + 0.1f * span
                || value >= upper_bound(locus) - 0.1f * span
            ) {
                extreme[(output_row * kLoci) + locus] += 1.0f;
            }
            package_a = package_a
                && fabsf(value - founder_value(pattern, 0, locus)) <= 1.0e-6f;
            package_b = package_b
                && fabsf(value - founder_value(pattern, 1, locus)) <= 1.0e-6f;
        }
        intact_count += package_a || package_b ? 1 : 0;
        for (int founder = 0; founder < kFounders; ++founder) {
            founder_mass[founder] +=
                ancestry[(cell_slot + slot) * kFounders + founder];
        }
    }
    float concentration = 0.0f;
    float maximum = 0.0f;
    for (int founder = 0; founder < kFounders; ++founder) {
        const float mass = founder_mass[founder] / static_cast<float>(count);
        concentration += mass * mass;
        maximum = mass > maximum ? mass : maximum;
    }
    for (int locus = 0; locus < kLoci; ++locus) {
        variance[(output_row * kLoci) + locus] /= static_cast<float>(count);
        extreme[(output_row * kLoci) + locus] /= static_cast<float>(count);
    }
    intact[output_row] = static_cast<float>(intact_count) / static_cast<float>(count);
    effective[output_row] = concentration > 0.0f ? 1.0f / concentration : 0.0f;
    max_founder[output_row] = maximum;
}

__global__ void initialize_kernel(
    float* loci,
    bool* alive,
    int64_t* agent_ids,
    float* ancestry,
    int64_t* next_id,
    int64_t* generation,
    const int64_t* seeds,
    const int32_t* modes,
    const int32_t* patterns,
    int32_t* status,
    const int batch,
    const int capacity
) {
    const int cell = blockIdx.x;
    if (cell >= batch || threadIdx.x != 0) return;
    if (capacity < kPopulation || modes[cell] < 0 || modes[cell] > 3
        || patterns[cell] < 0 || patterns[cell] > 2) {
        status[cell] = 1;
        return;
    }
    const int cell_slot = cell * capacity;
    for (int slot = 0; slot < capacity; ++slot) {
        alive[cell_slot + slot] = slot < kPopulation;
        agent_ids[cell_slot + slot] = slot < kPopulation ? slot : -1;
        for (int locus = 0; locus < kLoci; ++locus) {
            loci[(cell_slot + slot) * kLoci + locus] = slot < kPopulation
                ? founder_value(patterns[cell], slot < 32 ? 0 : 1, locus)
                : neutral_value(locus);
        }
        for (int founder = 0; founder < kFounders; ++founder) {
            ancestry[(cell_slot + slot) * kFounders + founder] =
                slot < kPopulation && slot == founder ? 1.0f : 0.0f;
        }
    }
    next_id[cell] = kPopulation;
    generation[cell] = 0;
    status[cell] = 0;
    (void)seeds;
}

__global__ void advance_kernel(
    float* loci,
    bool* alive,
    int64_t* agent_ids,
    float* ancestry,
    int64_t* next_id,
    int64_t* generation,
    const int64_t* seeds,
    const int32_t* modes,
    const int32_t* patterns,
    int64_t* parent_ids,
    int64_t* mate_ids,
    int64_t* child_ids,
    float* parent_loci,
    float* mate_loci,
    float* child_loci,
    int8_t* donors,
    int32_t* draw_counts,
    float* variance,
    float* extreme,
    float* intact,
    float* effective,
    float* max_founder,
    int32_t* mutation_events,
    int32_t* status,
    const int batch,
    const int capacity,
    const int steps
) {
    const int cell = blockIdx.x;
    if (cell >= batch || threadIdx.x != 0 || status[cell] != 0) return;
    const int cell_slot = cell * capacity;
    for (int step = 0; step < steps; ++step) {
        const int64_t next_generation = generation[cell] + 1;
        int slots[kPopulation];
        uint64_t scores[kPopulation];
        int live = 0;
        for (int slot = 0; slot < capacity; ++slot) {
            if (alive[cell_slot + slot]) {
                if (live >= kPopulation) {
                    status[cell] = 2;
                    return;
                }
                slots[live] = slot;
                scores[live] = word(
                    seeds[cell], next_generation, agent_ids[cell_slot + slot], 0
                );
                ++live;
            }
        }
        if (live != kPopulation) {
            status[cell] = 3;
            return;
        }
        for (int i = 1; i < kPopulation; ++i) {
            const int slot = slots[i];
            const uint64_t score = scores[i];
            int j = i - 1;
            while (
                j >= 0
                && (scores[j] > score
                    || (scores[j] == score
                        && agent_ids[cell_slot + slots[j]]
                            > agent_ids[cell_slot + slot]))
            ) {
                slots[j + 1] = slots[j];
                scores[j + 1] = scores[j];
                --j;
            }
            slots[j + 1] = slot;
            scores[j + 1] = score;
        }
        for (int child = 0; child < kSurvivors; ++child) {
            const int child_slot = slots[kSurvivors + child];
            const int parent_index = static_cast<int>(
                word(seeds[cell], next_generation, child, 1) % kSurvivors
            );
            int mate_index = static_cast<int>(
                word(seeds[cell], next_generation, child, 2) % (kSurvivors - 1)
            );
            if (mate_index >= parent_index) ++mate_index;
            const int parent_slot = slots[parent_index];
            const int mate_slot = slots[mate_index];
            const int event = (cell * steps + step) * kSurvivors + child;
            parent_ids[event] = agent_ids[cell_slot + parent_slot];
            mate_ids[event] = agent_ids[cell_slot + mate_slot];
            int package_donor = -1;
            if (modes[cell] == 3) {
                package_donor = uniform01(
                    word(seeds[cell], next_generation, child, 10)
                ) < 0.5f ? 0 : 1;
            }
            for (int locus = 0; locus < kLoci; ++locus) {
                const float a = loci[(cell_slot + parent_slot) * kLoci + locus];
                const float b = loci[(cell_slot + mate_slot) * kLoci + locus];
                parent_loci[event * kLoci + locus] = a;
                mate_loci[event * kLoci + locus] = b;
                float value = 0.5f * (a + b);
                int8_t donor = -1;
                if (modes[cell] == 1) {
                    const float draw = uniform01(
                        word(seeds[cell], next_generation, child, 10 + locus)
                    );
                    donor = draw < 0.5f ? 0 : 1;
                    value = donor == 0 ? a : b;
                } else if (modes[cell] == 2) {
                    const float draw = uniform01(
                        word(seeds[cell], next_generation, child, 10 + locus)
                    );
                    value = a + draw * (b - a);
                } else if (modes[cell] == 3) {
                    donor = static_cast<int8_t>(package_donor);
                    value = donor == 0 ? a : b;
                }
                const float low = lower_bound(locus);
                const float high = upper_bound(locus);
                value = value < low ? low : (value > high ? high : value);
                loci[(cell_slot + child_slot) * kLoci + locus] = value;
                child_loci[event * kLoci + locus] = value;
                donors[event * kLoci + locus] = donor;
            }
            for (int founder = 0; founder < kFounders; ++founder) {
                ancestry[(cell_slot + child_slot) * kFounders + founder] =
                    0.5f * (
                        ancestry[(cell_slot + parent_slot) * kFounders + founder]
                        + ancestry[(cell_slot + mate_slot) * kFounders + founder]
                    );
            }
            alive[cell_slot + child_slot] = true;
            const int64_t child_id = next_id[cell]++;
            agent_ids[cell_slot + child_slot] = child_id;
            child_ids[event] = child_id;
            draw_counts[event] = modes[cell] == 0
                ? 0
                : (modes[cell] == 3 ? 1 : kLoci);
            mutation_events[event] = 0;
        }
        generation[cell] = next_generation;
        const int output_row = cell * steps + step;
        summarize_cell(
            cell, capacity, output_row, loci, alive, ancestry, patterns[cell],
            variance, extreme, intact, effective, max_founder
        );
    }
}

__global__ void summarize_kernel(
    const float* loci,
    const bool* alive,
    const float* ancestry,
    const int32_t* patterns,
    float* variance,
    float* extreme,
    float* intact,
    float* effective,
    float* max_founder,
    const int batch,
    const int capacity
) {
    const int cell = blockIdx.x;
    if (cell >= batch || threadIdx.x != 0) return;
    summarize_cell(
        cell, capacity, cell, loci, alive, ancestry, patterns[cell],
        variance, extreme, intact, effective, max_founder
    );
}

void check_state(
    const torch::Tensor& loci,
    const torch::Tensor& alive,
    const torch::Tensor& agent_ids,
    const torch::Tensor& ancestry
) {
    TORCH_CHECK(loci.is_cuda() && alive.is_cuda() && agent_ids.is_cuda()
        && ancestry.is_cuda(), "RECOMB72_GPU_REQUIRES_CUDA");
    TORCH_CHECK(loci.is_contiguous() && alive.is_contiguous()
        && agent_ids.is_contiguous() && ancestry.is_contiguous(),
        "RECOMB72_GPU_REQUIRES_CONTIGUOUS_STATE");
    TORCH_CHECK(loci.scalar_type() == torch::kFloat32
        && alive.scalar_type() == torch::kBool
        && agent_ids.scalar_type() == torch::kInt64
        && ancestry.scalar_type() == torch::kFloat32,
        "RECOMB72_GPU_STATE_DTYPE_MISMATCH");
}

}  // namespace

void recomb72_gpu_initialize_v1(
    torch::Tensor loci, torch::Tensor alive, torch::Tensor agent_ids,
    torch::Tensor ancestry, torch::Tensor next_id, torch::Tensor generation,
    torch::Tensor seeds, torch::Tensor modes, torch::Tensor patterns,
    torch::Tensor status
) {
    check_state(loci, alive, agent_ids, ancestry);
    const int batch = static_cast<int>(loci.size(0));
    const int capacity = static_cast<int>(loci.size(1));
    TORCH_CHECK(loci.size(2) == kLoci && ancestry.size(2) == kFounders,
        "RECOMB72_GPU_STATE_SHAPE_MISMATCH");
    c10::cuda::CUDAGuard guard(loci.device());
    initialize_kernel<<<batch, 1, 0, at::cuda::getCurrentCUDAStream()>>>(
        loci.data_ptr<float>(), alive.data_ptr<bool>(), agent_ids.data_ptr<int64_t>(),
        ancestry.data_ptr<float>(), next_id.data_ptr<int64_t>(),
        generation.data_ptr<int64_t>(), seeds.data_ptr<int64_t>(),
        modes.data_ptr<int32_t>(), patterns.data_ptr<int32_t>(),
        status.data_ptr<int32_t>(), batch, capacity
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void recomb72_gpu_advance_v1(
    torch::Tensor loci, torch::Tensor alive, torch::Tensor agent_ids,
    torch::Tensor ancestry, torch::Tensor next_id, torch::Tensor generation,
    torch::Tensor seeds, torch::Tensor modes, torch::Tensor patterns,
    torch::Tensor parent_ids, torch::Tensor mate_ids, torch::Tensor child_ids,
    torch::Tensor parent_loci, torch::Tensor mate_loci, torch::Tensor child_loci,
    torch::Tensor donors, torch::Tensor draw_counts, torch::Tensor variance,
    torch::Tensor extreme, torch::Tensor intact, torch::Tensor effective,
    torch::Tensor max_founder, torch::Tensor mutation_events,
    torch::Tensor status, int64_t steps
) {
    check_state(loci, alive, agent_ids, ancestry);
    TORCH_CHECK(steps > 0 && steps <= 30, "RECOMB72_GPU_STEPS_OUT_OF_RANGE");
    const int batch = static_cast<int>(loci.size(0));
    const int capacity = static_cast<int>(loci.size(1));
    c10::cuda::CUDAGuard guard(loci.device());
    advance_kernel<<<batch, 1, 0, at::cuda::getCurrentCUDAStream()>>>(
        loci.data_ptr<float>(), alive.data_ptr<bool>(), agent_ids.data_ptr<int64_t>(),
        ancestry.data_ptr<float>(), next_id.data_ptr<int64_t>(),
        generation.data_ptr<int64_t>(), seeds.data_ptr<int64_t>(),
        modes.data_ptr<int32_t>(), patterns.data_ptr<int32_t>(),
        parent_ids.data_ptr<int64_t>(), mate_ids.data_ptr<int64_t>(),
        child_ids.data_ptr<int64_t>(), parent_loci.data_ptr<float>(),
        mate_loci.data_ptr<float>(), child_loci.data_ptr<float>(),
        donors.data_ptr<int8_t>(), draw_counts.data_ptr<int32_t>(),
        variance.data_ptr<float>(), extreme.data_ptr<float>(),
        intact.data_ptr<float>(), effective.data_ptr<float>(),
        max_founder.data_ptr<float>(), mutation_events.data_ptr<int32_t>(),
        status.data_ptr<int32_t>(), batch, capacity, static_cast<int>(steps)
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void recomb72_gpu_summarize_v1(
    torch::Tensor loci, torch::Tensor alive, torch::Tensor ancestry,
    torch::Tensor patterns, torch::Tensor variance, torch::Tensor extreme,
    torch::Tensor intact, torch::Tensor effective, torch::Tensor max_founder
) {
    TORCH_CHECK(loci.is_cuda() && alive.is_cuda() && ancestry.is_cuda(),
        "RECOMB72_GPU_REQUIRES_CUDA");
    const int batch = static_cast<int>(loci.size(0));
    const int capacity = static_cast<int>(loci.size(1));
    c10::cuda::CUDAGuard guard(loci.device());
    summarize_kernel<<<batch, 1, 0, at::cuda::getCurrentCUDAStream()>>>(
        loci.data_ptr<float>(), alive.data_ptr<bool>(), ancestry.data_ptr<float>(),
        patterns.data_ptr<int32_t>(), variance.data_ptr<float>(),
        extreme.data_ptr<float>(), intact.data_ptr<float>(),
        effective.data_ptr<float>(), max_founder.data_ptr<float>(),
        batch, capacity
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
